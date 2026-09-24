"""Optional Discord notifications, with every HTTP operation replaced locally."""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import yaml

from experiments_wo_stress.config import load_config
from experiments_wo_stress.notifications import (
    _MAX_PROGRESS_BYTES,
    ExperimentNotifier,
    _NoRedirect,
    _webhook_url,
    validate_notifications,
)

_ENV = "EWS_TEST_DISCORD_WEBHOOK"
_URL = "https://discord.com/api/webhooks/123456789012345678/fictional_test_secret"


class _Response:
    def __init__(self, headers=None):
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _TrackedBody(io.BytesIO):
    def __init__(self, body):
        super().__init__(body)
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


class _Opener:
    def __init__(self):
        self.calls = []
        self.effect = None
        self.headers = {}
        self.lock = threading.Lock()

    def open(self, request, *, timeout):
        with self.lock:
            self.calls.append((request, timeout, threading.current_thread().name))
        if self.effect is not None:
            self.effect(request)
        return _Response(self.headers)

    def contents(self):
        with self.lock:
            return [json.loads(request.data)["content"] for request, _, _ in self.calls]


class _NotificationTest(unittest.TestCase):
    def setUp(self):
        self.opener = _Opener()
        patcher = mock.patch(
            "experiments_wo_stress.notifications.urllib.request.build_opener",
            return_value=self.opener,
        )
        self.build_opener = patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {_ENV: _URL})
        environment.start()
        self.addCleanup(environment.stop)

    def notifier(self, **options):
        settings = {"discord": {"webhook_env": _ENV, "interval_seconds": 1, **options}}
        return ExperimentNotifier(settings, "example @everyone", 3)


class NotificationConfigurationTests(_NotificationTest):
    def test_absent_or_explicitly_disabled_notifications_do_no_network_work(self):
        for settings in ({}, {"discord": {"enabled": False, "webhook_env": "UNSET_EWS_TEST"}}):
            notifier = ExperimentNotifier(settings, "study", 1)
            notifier.start()
            notifier.run_started("run", Path("unused"), 5)
            notifier.run_finished(("run", "completed", None))
            notifier.finish({"completed": 1})
            self.assertIsNone(notifier._thread)
        self.build_opener.assert_not_called()
        self.assertEqual(self.opener.calls, [])

    def test_background_thread_start_failure_does_not_abort_execution(self):
        notifier = self.notifier()
        with mock.patch.object(threading.Thread, "start", side_effect=RuntimeError(_URL)):
            with self.assertLogs(
                "experiments_wo_stress.notifications", level="WARNING"
            ) as captured:
                notifier.start()
        notifier.finish({"completed": 3})
        self.assertFalse(notifier.enabled)
        self.assertIsNone(notifier._thread)
        self.assertEqual(self.opener.calls, [])
        self.assertNotIn(_URL, " ".join(captured.output))

    def test_strict_configuration_validation(self):
        invalid = [
            None,
            [],
            {"email": {}},
            {"discord": None},
            {"discord": {"webhook_url": _URL}},
            {"discord": {"enabled": "true"}},
            {"discord": {"webhook_env": _URL}},
            {"discord": {"webhook_env": "INVALID NAME"}},
            {"discord": {"interval_seconds": 0.5}},
            {"discord": {"interval_seconds": True}},
            {"discord": {"interval_seconds": float("nan")}},
            {"discord": {"timeout_seconds": 0}},
            {"discord": {"timeout_seconds": 31}},
            {"discord": {"timeout_seconds": float("inf")}},
        ]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError) as caught:
                validate_notifications(settings)
            self.assertNotIn(_URL, str(caught.exception))
        defaults = validate_notifications({"discord": {}})["discord"]
        self.assertEqual(defaults["interval_seconds"], 300)
        self.assertEqual(defaults["timeout_seconds"], 5)
        self.assertTrue(defaults["enabled"])

    def test_missing_or_invalid_secret_has_a_sanitized_error(self):
        invalid_urls = [
            "http://discord.com/api/webhooks/123/token",
            "https://discord.com.evil.invalid/api/webhooks/123/token",
            "https://example.com/api/webhooks/123/token",
            "https://user:password@discord.com/api/webhooks/123/token",
            _URL + "#fragment",
            _URL + "?unexpected=secret",
            _URL + "?thread_id=not-a-number",
            _URL.replace("discord.com", "discord.com:1234"),
            _URL.replace("discord.com", "discord.com:invalid"),
        ]
        with mock.patch.dict(os.environ, {_ENV: ""}), self.assertRaisesRegex(ValueError, "missing"):
            self.notifier()
        for value in invalid_urls:
            with self.subTest(value=value), mock.patch.dict(os.environ, {_ENV: value}):
                with self.assertRaises(ValueError) as caught:
                    self.notifier()
                self.assertNotIn(value, str(caught.exception))
                self.assertNotIn("fictional_test_secret", str(caught.exception))
        self.build_opener.assert_not_called()

    def test_webhook_wait_confirmation_and_thread_query(self):
        self.assertEqual(_webhook_url(_URL), _URL + "?wait=true")
        url = _webhook_url(_URL + "?wait=false&thread_id=123456")
        self.assertEqual(url, _URL + "?wait=true&thread_id=123456")
        self.assertIsNone(
            _NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://example.com")
        )

    def test_configuration_serializes_only_environment_name_and_preserves_science(self):
        document = {
            "name": "study",
            "runs": [
                {
                    "name": "main",
                    "protocol": {"type": "online", "params": {"horizon": 3}},
                    "data": {"type": "gaussian_bandit", "params": {"means": [0.2, 0.8]}},
                    "algorithms": [{"name": "learner", "type": "tests.test_settings:_Learner"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.yml"
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            original = load_config(path)
            document["notifications"] = {"discord": {"webhook_env": _ENV}}
            path.write_text(yaml.safe_dump(document), encoding="utf-8")
            configured = load_config(path)
            persisted = yaml.safe_dump(configured.to_dict())
        self.assertIn(_ENV, persisted)
        self.assertNotIn(_URL, persisted)
        self.assertNotIn("fictional_test_secret", persisted)
        self.assertEqual(original.simulation_dict(), configured.simulation_dict())


class NotificationTransportTests(_NotificationTest):
    def test_payload_suppresses_mentions_and_uses_post_with_a_timeout(self):
        notifier = self.notifier(timeout_seconds=0.25)
        content = notifier._message("started")
        notifier._send(content)
        request, timeout, _ = self.opener.calls[0]
        payload = json.loads(request.data)
        self.assertEqual(request.method, "POST")
        self.assertTrue(request.full_url.endswith("?wait=true"))
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(payload["allowed_mentions"], {"parse": []})
        self.assertFalse(payload["tts"])
        self.assertEqual(payload["content"], content)
        self.assertEqual(timeout, 0.25)
        self.assertLessEqual(len(content), 2000)

    def test_429_respects_fractional_header_and_body_cooldown(self):
        notifier = self.notifier()
        body = _TrackedBody(b'{"retry_after": 2.75}')
        error = urllib.error.HTTPError(_URL, 429, "limited", {"Retry-After": "1.5"}, body)
        self.opener.effect = mock.Mock(side_effect=[error, None])
        with mock.patch(
            "experiments_wo_stress.notifications.time.monotonic", return_value=100
        ) as clock:
            with self.assertLogs("experiments_wo_stress.notifications", level="WARNING"):
                notifier._send("first")
            self.assertEqual(notifier._next_allowed, 102.75)
            clock.return_value = 102.5
            notifier._send("too soon")
            self.assertEqual(len(self.opener.calls), 1)
            clock.return_value = 102.75
            notifier._send("retry")
            self.assertEqual(len(self.opener.calls), 2)
        self.assertEqual(body.read_sizes, [65536])
        self.assertTrue(body.closed)

    def test_success_headers_delay_subsequent_notifications(self):
        notifier = self.notifier()
        self.opener.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "0.75"}
        with mock.patch(
            "experiments_wo_stress.notifications.time.monotonic", return_value=100
        ) as clock:
            notifier._send("first")
            self.assertEqual(notifier._next_allowed, 100.75)
            clock.return_value = 100.5
            notifier._send("blocked")
            self.assertEqual(len(self.opener.calls), 1)
            clock.return_value = 100.75
            notifier._send("allowed")
            self.assertEqual(len(self.opener.calls), 2)

    def test_error_body_read_is_bounded_and_invalid_delay_uses_interval(self):
        notifier = self.notifier()
        body = _TrackedBody(b"{" + b" " * 100000)
        self.opener.effect = mock.Mock(
            side_effect=urllib.error.HTTPError(_URL, 429, _URL, {"Retry-After": "NaN"}, body)
        )
        with mock.patch("experiments_wo_stress.notifications.time.monotonic", return_value=100):
            with self.assertLogs(
                "experiments_wo_stress.notifications", level="WARNING"
            ) as captured:
                notifier._send("status")
        self.assertEqual(body.read_sizes, [65536])
        self.assertEqual(notifier._next_allowed, 101)
        self.assertNotIn(_URL, " ".join(captured.output))

    def test_invalid_or_deleted_webhooks_disable_delivery(self):
        for status in (401, 403, 404):
            notifier = self.notifier()
            body = _TrackedBody(_URL.encode())
            self.opener.effect = mock.Mock(
                side_effect=urllib.error.HTTPError(_URL, status, _URL, {}, body)
            )
            before = len(self.opener.calls)
            with (
                self.subTest(status=status),
                self.assertLogs("experiments_wo_stress.notifications", level="WARNING") as captured,
            ):
                notifier._send("first")
                notifier._send("second")
            self.assertFalse(notifier.enabled)
            self.assertEqual(len(self.opener.calls), before + 1)
            self.assertTrue(body.closed)
            self.assertNotIn(_URL, " ".join(captured.output))

    def test_transient_network_and_http_errors_never_log_secret_exception_text(self):
        errors = [
            urllib.error.URLError(_URL),
            TimeoutError(_URL),
            urllib.error.HTTPError(_URL, 503, _URL, {}, io.BytesIO(_URL.encode())),
        ]
        for error in errors:
            notifier = self.notifier()
            self.opener.effect = mock.Mock(side_effect=error)
            with (
                self.subTest(error=type(error).__name__),
                self.assertLogs("experiments_wo_stress.notifications", level="WARNING") as captured,
            ):
                notifier._send("status")
            self.assertTrue(notifier.enabled)
            self.assertNotIn(_URL, " ".join(captured.output))
            self.assertNotIn("fictional_test_secret", " ".join(captured.output))


class NotificationProgressTests(_NotificationTest):
    def test_durable_progress_reads_are_bounded_and_tolerate_invalid_files(self):
        directory = Path("unused")
        values = [
            b'{"step": 12}',
            b'{"step": -1}',
            b'{"step": true}',
            b'{"step": 1.5}',
            b"[]",
            b"{incomplete",
        ]
        for index, value in enumerate(values):
            body = _TrackedBody(value)
            with self.subTest(value=value), mock.patch.object(Path, "open", return_value=body):
                self.assertEqual(
                    ExperimentNotifier._durable_step(directory), 12 if index == 0 else None
                )
            self.assertEqual(body.read_sizes, [_MAX_PROGRESS_BYTES + 1])
        body = _TrackedBody(b" " * (_MAX_PROGRESS_BYTES + 2))
        with (
            mock.patch.object(Path, "open", return_value=body),
            mock.patch("experiments_wo_stress.notifications.json.loads") as parse,
        ):
            self.assertIsNone(ExperimentNotifier._durable_step(directory))
            parse.assert_not_called()
        with mock.patch.object(Path, "open", side_effect=FileNotFoundError):
            self.assertIsNone(ExperimentNotifier._durable_step(directory))

    def test_only_four_active_checkpoints_are_read_and_all_runs_are_counted(self):
        notifier = self.notifier()
        notifier.total = 10
        for index in range(8):
            notifier.run_started(f"run-{index}", Path(f"private-{index}"), 100)
        with mock.patch.object(notifier, "_durable_step", return_value=25) as read:
            message = notifier._message("progress")
        self.assertEqual(read.call_count, 4)
        self.assertIn("Active runs: 8", message)
        self.assertIn("Showing 4 of 8", message)
        self.assertIn("pending 2", message)
        self.assertIn("step 25/100", message)
        self.assertNotIn("private-", message)

    def test_periodic_updates_continue_without_run_callbacks_and_finish_reports_failures(self):
        notifier = self.notifier(timeout_seconds=0.1)
        notifier.interval = 0.01
        progress_sent = threading.Event()

        def capture(request):
            if "Run stage: progress" in json.loads(request.data)["content"]:
                progress_sent.set()

        self.opener.effect = capture
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "progress.json").write_text('{"step": 7}', encoding="utf-8")
            notifier.run_started("first", path, 20)
            notifier.start()
            try:
                self.assertTrue(progress_sent.wait(1), "No periodic summary during an active run")
                notifier.run_finished(("first", "completed", None))
                notifier.run_started("second", path, 20)
                notifier.run_finished(("second", "failed", "private traceback " + _URL))
            finally:
                notifier.finish({"completed": 1, "failed": 1, "pending": 1})
        self.assertFalse(notifier._thread.is_alive())
        messages = self.opener.contents()
        self.assertIn("Run stage: started", messages[0])
        self.assertTrue(any("step 7/20" in value for value in messages))
        self.assertIn("Run stage: finished with failures", messages[-1])
        self.assertIn("completed 1", messages[-1])
        self.assertIn("failed 1", messages[-1])
        self.assertIn("pending 1", messages[-1])
        self.assertNotIn(_URL, " ".join(messages))
        self.assertTrue(all(name == "ews-discord" for _, _, name in self.opener.calls))

    def test_blocked_delivery_does_not_block_callbacks_and_finish_has_a_deadline(self):
        notifier = self.notifier(timeout_seconds=0.01)
        entered, release = threading.Event(), threading.Event()

        def blocked(_request):
            entered.set()
            release.wait(3)

        self.opener.effect = blocked
        notifier.start()
        try:
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            notifier.run_started("first", Path("unused"), 20)
            notifier.run_finished(("first", "completed", None))
            self.assertLess(time.monotonic() - started, 0.2)
            started = time.monotonic()
            with self.assertLogs("experiments_wo_stress.notifications", level="WARNING"):
                notifier.finish({"completed": 1, "pending": 2})
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertTrue(notifier._cancelled.is_set())
        finally:
            release.set()
            notifier._thread.join(1)
        self.assertFalse(notifier._thread.is_alive())
        self.assertEqual(len(self.opener.calls), 1)

    def test_finish_during_a_retry_window_omits_delivery_without_waiting(self):
        notifier = self.notifier(timeout_seconds=0.1)
        self.opener.headers = {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset-After": "300"}
        delivered = threading.Event()
        self.opener.effect = lambda _: delivered.set()
        notifier.start()
        self.assertTrue(delivered.wait(1))
        # The send callback runs just before response headers are processed.
        # Holding this test's loop under the notifier lock would change behavior;
        # completion itself joins the sender and establishes the final result.
        with self.assertLogs("experiments_wo_stress.notifications", level="WARNING") as captured:
            notifier.finish({"completed": 3})
        self.assertFalse(notifier._thread.is_alive())
        self.assertEqual(len(self.opener.calls), 1)
        self.assertTrue(any("retry window" in line for line in captured.output))

    def test_final_status_distinguishes_paused_completed_and_aborted(self):
        notifier = self.notifier()
        for final, expected in (
            ({"completed": 3}, "completed"),
            ({"paused": 1, "pending": 2}, "paused"),
            ({"aborted": True, "pending": 3}, "aborted"),
        ):
            notifier._final = {key: final.get(key, 0) for key in (*notifier._counts, "pending")}
            notifier._final["aborted"] = final.get("aborted", False)
            with self.subTest(expected=expected):
                self.assertIn(f"Run stage: {expected};", notifier._message("finished"))


if __name__ == "__main__":
    unittest.main()
