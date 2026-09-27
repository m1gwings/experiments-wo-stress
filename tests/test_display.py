"""Display settings and retention policy stay outside scientific identity."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import timezone
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import yaml

from experiments_wo_stress.execution.provenance import prepare_request
from experiments_wo_stress.study.config import load_config, resolve_display_timezone
from experiments_wo_stress.study.planning import plan_runs
from experiments_wo_stress.study.rng import make_rngs


class DisplayConfigurationTests(unittest.TestCase):
    """Resolve explicit display settings without host timezone or scientific effects."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "study.yml"
        self.document = {
            "name": "display",
            "runs": [
                {
                    "protocol": {"type": "offline"},
                    "data": {"type": "normal", "params": {"size": 2}},
                    "algorithms": [{"name": "example", "type": "null_algorithm"}],
                }
            ],
        }

    def load(self):
        self.path.write_text(yaml.safe_dump(self.document), encoding="utf-8")
        return load_config(self.path)

    def test_utc_default_and_cli_override_take_precedence_over_yaml(self):
        default = self.load()
        self.assertEqual(default.display, {"timezone": "UTC"})
        self.assertEqual(resolve_display_timezone(default.display).key, "UTC")
        self.document["display"] = {"timezone": "Europe/Rome"}
        config = self.load()
        self.assertEqual(config.to_dict()["display"], {"timezone": "Europe/Rome"})
        self.assertEqual(resolve_display_timezone(config.display).key, "Europe/Rome")
        self.assertEqual(resolve_display_timezone(config.display, "Asia/Tokyo").key, "Asia/Tokyo")

    def test_invalid_display_names_and_shapes_fail_clearly(self):
        for settings in (
            None,
            [],
            {"unknown": "UTC"},
            {"timezone": None},
            {"timezone": 7},
            {"timezone": ""},
            {"timezone": "NoSuchPlace/Somewhere"},
            {"timezone": "/etc/localtime"},
        ):
            with self.subTest(settings=settings):
                self.document["display"] = settings
                with self.assertRaisesRegex(ValueError, "display"):
                    self.load()
        with self.assertRaisesRegex(ValueError, "Invalid display.timezone"):
            resolve_display_timezone({"timezone": "UTC"}, "Invalid/Override")

    def test_display_and_retention_preserve_run_streams_and_variant_identity(self):
        original = self.load()
        original_runs = plan_runs(original)
        provenance = {"implementation": {}, "environment": {}}
        original_request = prepare_request(original, original_runs, provenance)
        self.document["display"] = {"timezone": "Europe/Rome"}
        self.document["recording"] = {"retention": "until_analyzed"}
        changed = self.load()
        changed_runs = plan_runs(changed)
        self.assertEqual(changed.simulation_dict(), original.simulation_dict())
        self.assertEqual(changed_runs, original_runs)
        changed_request = prepare_request(changed, changed_runs, provenance)
        self.assertEqual(changed_request.locations, original_request.locations)
        self.assertEqual(changed_request.request_id, original_request.request_id)
        for name in ("algorithm", "data", "protocol", "instance"):
            np.testing.assert_array_equal(
                make_rngs(original_runs[0])[name].normal(size=10),
                make_rngs(changed_runs[0])[name].normal(size=10),
            )

    def test_default_utc_works_without_an_installed_timezone_database(self):
        with patch(
            "experiments_wo_stress.study.config.ZoneInfo", side_effect=ZoneInfoNotFoundError
        ):
            self.assertIs(resolve_display_timezone({}), timezone.utc)
            with self.assertRaisesRegex(ValueError, "installed IANA timezone"):
                resolve_display_timezone({}, "Europe/Rome")

    def test_retention_defaults_to_keep_and_rejects_unknown_modes(self):
        self.assertEqual(self.load().recording["retention"], "keep")
        for value in ("keep", "until_analyzed"):
            self.document["recording"] = {"retention": value}
            self.assertEqual(self.load().recording["retention"], value)
        for value in (None, [], {}, 1, True, "delete", "KEEP"):
            with self.subTest(value=value):
                self.document["recording"] = {"retention": value}
                with self.assertRaisesRegex(ValueError, "recording.retention"):
                    self.load()

    def test_cli_passes_timezone_override_to_the_terminal_observer(self):
        from experiments_wo_stress.cli import main
        from experiments_wo_stress.execution.coordinator import RunReport

        self.document["display"] = {"timezone": "Europe/Rome"}
        self.load()
        for arguments, expected in (
            ([], "Europe/Rome"),
            (["--timezone", "Asia/Tokyo"], "Asia/Tokyo"),
        ):
            with (
                self.subTest(arguments=arguments),
                patch("experiments_wo_stress.cli.Invocation") as invocation,
                patch("experiments_wo_stress.cli.TerminalProgress") as terminal,
                patch("experiments_wo_stress.cli.run_experiment", return_value=RunReport()) as run,
                redirect_stdout(StringIO()),
            ):
                invocation.return_value.display_summary.return_value = None
                self.assertEqual(
                    main(["run", str(self.path), "-o", str(self.path.parent / "out"), *arguments]),
                    0,
                )
                self.assertEqual(terminal.call_args.kwargs["timezone"].key, expected)
                self.assertEqual(run.call_args.args[0].display, {"timezone": "Europe/Rome"})

    def test_invalid_cli_timezone_fails_before_any_simulation_is_scheduled(self):
        from experiments_wo_stress.cli import main

        self.load()
        error = StringIO()
        with (
            patch("experiments_wo_stress.cli.Invocation"),
            patch("experiments_wo_stress.cli.run_experiment") as run,
            redirect_stderr(error),
        ):
            result = main(
                [
                    "run",
                    str(self.path),
                    "-o",
                    str(self.path.parent / "out"),
                    "--timezone",
                    "Invalid/Zone",
                ]
            )
        self.assertEqual(result, 1)
        run.assert_not_called()
        self.assertIn("Invalid display.timezone", error.getvalue())
