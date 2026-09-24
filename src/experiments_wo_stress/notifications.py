"""Optional coordinator-only Discord summaries, isolated from scientific execution."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)
_DEFAULTS = {
    "enabled": True,
    "webhook_env": "EWS_DISCORD_WEBHOOK_URL",
    "interval_seconds": 300.0,
    "timeout_seconds": 5.0,
}
_MAX_PROGRESS_BYTES = 1024 * 1024
_MAX_ACTIVE_DETAILS = 4


def validate_notifications(value: Any) -> dict[str, Any]:
    """Normalize optional settings without resolving or persisting a webhook secret."""
    if not isinstance(value, dict) or set(value) - {"discord"}:
        raise ValueError("notifications must be a mapping containing only discord")
    if "discord" not in value:
        return {}
    options = value["discord"]
    if not isinstance(options, dict) or set(options) - set(_DEFAULTS):
        raise ValueError(
            "notifications.discord accepts enabled, webhook_env, interval_seconds, and timeout_seconds"
        )
    options = {**_DEFAULTS, **options}
    if not isinstance(options["enabled"], bool):
        raise ValueError("notifications.discord.enabled must be a boolean")
    name = options["webhook_env"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("notifications.discord.webhook_env must be an environment variable name")
    for key in ("interval_seconds", "timeout_seconds"):
        number = options[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"notifications.discord.{key} must be finite and positive")
    if options["interval_seconds"] < 1:
        raise ValueError("notifications.discord.interval_seconds must be at least 1")
    if options["timeout_seconds"] > 30:
        raise ValueError("notifications.discord.timeout_seconds must not exceed 30")
    return {"discord": options}


def _webhook_url(value: str) -> str:
    """Accept Discord HTTPS endpoints, without permitting redirects or arbitrary hosts."""
    try:
        parsed = urllib.parse.urlsplit(value)
        # Python 3.10 treats an empty string as a malformed strict query.
        query = urllib.parse.parse_qs(parsed.query, strict_parsing=True) if parsed.query else {}
        valid = (
            parsed.scheme == "https"
            and parsed.hostname
            in {"discord.com", "discordapp.com", "canary.discord.com", "ptb.discord.com"}
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
            and re.fullmatch(r"/api/(?:v[0-9]+/)?webhooks/[0-9]+/[A-Za-z0-9._-]+", parsed.path)
            and not set(query) - {"wait", "thread_id"}
            and (
                "thread_id" not in query
                or (len(query["thread_id"]) == 1 and query["thread_id"][0].isdigit())
            )
        )
        if not valid:
            raise ValueError
    except ValueError:
        raise ValueError(
            "Discord webhook environment variable must contain a valid Discord HTTPS webhook URL"
        ) from None
    parameters = {"wait": "true"}
    if "thread_id" in query:
        parameters["thread_id"] = query["thread_id"][0]
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(parameters), "")
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _seconds(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


class ExperimentNotifier:
    """Send best-effort run-stage summaries from one bounded background sender.

    Only the coordinator owns this object. Run callbacks copy small bookkeeping
    values; no worker or protocol step performs notification IO. HTTP failures never
    fail a simulation. Explicit configuration errors are rejected before execution.
    """

    def __init__(self, settings: Mapping[str, Any], name: str, total: int):
        options = validate_notifications(dict(settings)).get("discord", {})
        self.enabled = bool(options and options["enabled"])
        self.name = name
        self.total = total
        self.interval = options.get("interval_seconds", 300.0)
        self.timeout = options.get("timeout_seconds", 5.0)
        self._url = ""
        if self.enabled:
            value = os.environ.get(options["webhook_env"])
            if not value:
                raise ValueError(
                    "Discord notifications are enabled but the configured webhook environment variable is missing"
                )
            self._url = _webhook_url(value)
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._cancelled = threading.Event()
        self._thread: threading.Thread | None = None
        self._active: dict[str, tuple[Path, int | None]] = {}
        self._counts = dict.fromkeys(("completed", "skipped", "paused", "failed"), 0)
        self._final: dict[str, Any] | None = None
        self._next_allowed = 0.0
        self._last_warning = ""
        self._started_at = time.monotonic()
        self._opener = urllib.request.build_opener(_NoRedirect()) if self.enabled else None

    def start(self) -> None:
        """Start delivery after the executor has acquired its lock and prepared runs."""
        if not self.enabled:
            return
        self._started_at = time.monotonic()
        initial = self._message("started")
        self._thread = threading.Thread(
            target=self._loop, args=(initial,), name="ews-discord", daemon=True
        )
        try:
            self._thread.start()
        except RuntimeError:
            self._thread = None
            self.enabled = False
            self._warn("background sender could not start; delivery disabled")

    def run_started(self, run_id: str, directory: Path, budget: int | None) -> None:
        if self.enabled:
            with self._lock:
                self._active[run_id] = (directory, budget)

    def run_finished(self, result: tuple[str, str, str | None]) -> None:
        if self.enabled:
            run_id, status, _ = result
            with self._lock:
                self._active.pop(run_id, None)
                self._counts[status] += 1

    def finish(self, report: Mapping[str, Any], *, aborted: bool = False) -> None:
        """Attempt a final summary, with bounded waiting and no required delivery."""
        if self._thread is None:
            return
        with self._lock:
            self._final = {key: int(report.get(key, 0)) for key in (*self._counts, "pending")}
            self._final["aborted"] = aborted
            if aborted:
                self._final["pending"] = max(
                    self._final["pending"],
                    self.total - sum(self._final[key] for key in self._counts),
                )
        self._finished.set()
        # urllib timeouts do not bound DNS resolution or a whole request. The
        # daemon sender cannot delay process termination beyond this join budget.
        self._thread.join(timeout=min(2 * self.timeout + 0.25, 60.0))
        if self._thread.is_alive():
            self._cancelled.set()
            self._warn("final summary was not delivered before shutdown")

    def _warn(self, message: str) -> None:
        if message != self._last_warning:
            _LOGGER.warning("Discord notification: %s", message)
            self._last_warning = message

    def _message(self, event: str) -> str:
        with self._lock:
            counts = dict(self._counts)
            active_count = len(self._active)
            active = list(self._active.items())[:_MAX_ACTIVE_DETAILS]
            final = dict(self._final) if self._final is not None else None
        if final is not None:
            counts = {key: final[key] for key in counts}
            pending = final["pending"]
            event = (
                "aborted"
                if final["aborted"]
                else (
                    "finished with failures"
                    if counts["failed"]
                    else "paused"
                    if counts["paused"] or pending
                    else "completed"
                )
            )
        else:
            pending = max(0, self.total - sum(counts.values()) - active_count)
        # Only the experiment label, counts, elapsed time and checkpoint counters
        # leave this process. No paths, parameters, records, exceptions or secrets.
        name = " ".join(str(self.name).split())[:160]
        elapsed = max(0, int(time.monotonic() - self._started_at))
        lines = [
            f"Experiments W/O Stress — {name}",
            f"Run stage: {event}; elapsed {elapsed}s",
            f"Runs: {self.total}; completed {counts['completed']}; reused {counts['skipped']}; "
            f"paused {counts['paused']}; failed {counts['failed']}; pending {pending}",
        ]
        if final is None:
            lines.append(f"Active runs: {active_count}")
            for run_id, (directory, budget) in active:
                step = self._durable_step(directory)
                description = (
                    "no checkpoint available"
                    if step is None
                    else f"step {step}" + (f"/{budget}" if budget is not None else "")
                )
                lines.append(f"Last durable checkpoint {run_id[:12]}: {description}")
            if active_count > len(active):
                lines.append(f"Showing {len(active)} of {active_count} active checkpoints.")
        return "\n".join(lines)[:1900]

    @staticmethod
    def _durable_step(directory: Path) -> int | None:
        try:
            with (directory / "progress.json").open("rb") as stream:
                raw = stream.read(_MAX_PROGRESS_BYTES + 1)
            if len(raw) > _MAX_PROGRESS_BYTES:
                return None
            value = json.loads(raw).get("step")
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                else None
            )
        except (OSError, ValueError, AttributeError):
            return None

    def _cooldown(self, seconds: float) -> None:
        self._next_allowed = max(self._next_allowed, time.monotonic() + seconds)

    def _send(self, content: str) -> None:
        if not self.enabled or self._cancelled.is_set():
            return
        if time.monotonic() < self._next_allowed:
            return
        payload = json.dumps(
            {"content": content, "allowed_mentions": {"parse": []}, "tts": False}
        ).encode()
        request = urllib.request.Request(
            self._url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "experiments-wo-stress/0.1"},
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                if response.headers.get("X-RateLimit-Remaining") == "0":
                    self._cooldown(_seconds(response.headers.get("X-RateLimit-Reset-After")))
                self._last_warning = ""
        except urllib.error.HTTPError as exc:
            try:
                if exc.code == 429:
                    delay = _seconds((exc.headers or {}).get("Retry-After"))
                    try:
                        body = json.loads(exc.read(65536))
                        delay = max(delay, _seconds(body.get("retry_after")))
                    except (OSError, ValueError, AttributeError):
                        pass
                    self._cooldown(delay or self.interval)
                    self._warn("rate limited; updates will wait for the retry window")
                elif exc.code in {401, 403, 404}:
                    self.enabled = False
                    self._warn("webhook unavailable or unauthorized; delivery disabled")
                else:
                    self._cooldown(self.interval)
                    self._warn(f"HTTP {exc.code}; a later update will retry")
            finally:
                exc.close()
        except Exception:
            # Exception text may include the secret webhook URL. Never log it.
            self._cooldown(self.interval)
            self._warn("delivery failed; a later update will retry")

    def _loop(self, initial: str) -> None:
        try:
            self._send(initial)
            while self.enabled and not self._cancelled.is_set():
                delay = max(self.interval, self._next_allowed - time.monotonic())
                if self._finished.wait(min(delay, threading.TIMEOUT_MAX)):
                    if time.monotonic() >= self._next_allowed:
                        self._send(self._message("finished"))
                    else:
                        self._warn("final summary omitted during the retry window")
                    return
                self._send(self._message("progress"))
        except Exception:
            self._warn("delivery disabled after an internal notification error")
