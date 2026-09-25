"""Verify public commands, complete study workflows, and safe interruption."""

from __future__ import annotations

import io
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments_wo_stress.cli import main
from experiments_wo_stress.execution.coordinator import RunReport
from experiments_wo_stress.storage import iter_completed_runs

COMPONENTS = '''"""Slow, stateful components for an actual interruption test."""
import time
from experiments_wo_stress import Feedback

class Environment:
    """Accumulate a random position slowly enough for the test to send SIGTERM."""

    def __init__(self, *, rng, shift=0.0, fail=False):
        self.rng = rng
        self.position = 0.0
        self.shift = shift
        self.fail = fail

    def generate(self, action):
        if self.fail:
            raise RuntimeError("intentional simulation failure")
        time.sleep(0.01)
        self.position += float(self.rng.normal()) + action + self.shift
        return Feedback(self.position, {"position": self.position})

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]

class Learner:
    """Use saved reward history and injected randomness when choosing each action."""

    def __init__(self, *, rng):
        self.rng = rng
        self.total = 0.0

    def act(self, context=None):
        return int(self.rng.integers(2)) + int(self.total < 0)

    def observe(self, action, feedback):
        self.total += feedback

    def state_dict(self):
        return {"total": self.total}

    def load_state_dict(self, state):
        self.total = state["total"]
'''


class CommandLineTests(unittest.TestCase):
    """Exercise workflows through the real CLI, with a report-boundary pending check."""

    def setUp(self) -> None:
        """Create a standalone study with slow, stateful paper components."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "signal_components.py").write_text(COMPONENTS, encoding="utf-8")
        self.config = self.root / "experiment.yml"
        self.settings = {
            "name": "signal_recovery",
            "seed": 91,
            "runs": [
                {
                    "name": "main",
                    "repetitions": 2,
                    "protocol": {"type": "online", "params": {"horizon": 50}},
                    "data": {"type": "signal_components:Environment"},
                    "algorithms": [{"name": "learner", "type": "signal_components:Learner"}],
                }
            ],
            "execution": {"checkpoint_steps": 2},
            "recording": {"buffer_bytes": 64},
        }
        self.write_configuration()

    def write_configuration(self) -> None:
        """Publish the selected study through the public YAML interface."""
        self.config.write_text(yaml.safe_dump(self.settings), encoding="utf-8")

    def analysis_settings(self, *, figures: bool = False) -> dict:
        """Describe a numerical metric and optional dependency-free TikZ figure."""
        settings = {
            "metrics": [{"name": "position", "type": "field", "params": {"field": "position"}}],
            "aggregator": {"group_by": ["algorithm.name"], "uncertainty": "none"},
        }
        if figures:
            settings["figures"] = [{"name": "position", "metric": "position", "formats": ["tikz"]}]
        return settings

    def short_study(self, *, analysis: dict | None = None, fail: bool = False) -> None:
        """Keep command workflow checks small while retaining actual checkpoint writes."""
        group = self.settings["runs"][0]
        group["protocol"]["params"]["horizon"] = 4
        if fail:
            group["data"]["params"] = {"fail": True}
        if analysis is not None:
            self.settings["analysis"] = analysis
        self.write_configuration()

    def cache_files(self, output: Path) -> dict[str, tuple[int, bytes]]:
        """Observe immutable cache artifacts without relying on their identity format."""
        directory = output / "analysis" / "cache"
        return {
            str(path.relative_to(directory)): (path.stat().st_mtime_ns, path.read_bytes())
            for path in directory.rglob("*")
            if path.is_file()
        }

    def command(self, *args: str) -> list[str]:
        """Invoke the installed package with the same interpreter running these tests."""
        return [sys.executable, "-m", "experiments_wo_stress", *args]

    def invoke(self, *args: str) -> dict:
        """Run a successful CLI command and decode its machine-readable report."""
        result = subprocess.run(self.command(*args), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def test_help_exposes_exactly_six_commands_and_rejects_removed_aliases(self) -> None:
        result = subprocess.run(self.command("--help"), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        choices = re.search(r"\{([^}]+)\}", result.stdout)
        self.assertIsNotNone(choices, result.stdout)
        self.assertEqual(
            set(choices.group(1).split(",")),
            {"count-runs", "run", "analyze", "plot", "inspect", "clean"},
        )
        for removed in ("build", "plan"):
            with self.subTest(command=removed):
                result = subprocess.run(
                    self.command(removed, "--help"), capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("invalid choice", result.stderr)

    def test_count_runs_reports_grid_times_algorithms_times_repetitions_without_execution(
        self,
    ) -> None:
        group = self.settings["runs"][0]
        group["repetitions"] = 3
        group["grid"] = {"data.params.shift": [0.0, 1.0, 2.0]}
        group["algorithms"].append({"name": "other", "type": "signal_components:Learner"})
        self.write_configuration()
        # Built-in planning needs descriptions, not importable simulation classes.
        (self.root / "signal_components.py").unlink()
        result = self.invoke("count-runs", str(self.config))
        self.assertEqual(result, {"name": "signal_recovery", "runs": 18})
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "analysis").exists())

    def test_count_runs_rejects_invalid_configuration(self) -> None:
        self.settings["runs"][0]["repetitions"] = 0
        self.write_configuration()
        result = subprocess.run(
            self.command("count-runs", str(self.config)),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("repetitions", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_run_without_analysis_executes_directly_and_reuses_completed_results(self) -> None:
        self.short_study()
        output = self.root / "execution"
        first = self.invoke("run", str(self.config), "--output", str(output))
        second = self.invoke("run", str(self.config), "--output", str(output))
        self.assertEqual((first["completed"], first["failed"]), (2, 0))
        self.assertEqual((second["skipped"], second["failed"]), (2, 0))
        self.assertNotIn("groups", first)
        self.assertNotIn("figures", first)
        self.assertFalse((output / "analysis").exists())

    def test_run_with_metrics_creates_summaries_without_a_separate_analysis_command(self) -> None:
        self.short_study(analysis=self.analysis_settings())
        output = self.root / "metrics"
        report = self.invoke("run", str(self.config), "--output", str(output))
        self.assertEqual((report["completed"], report["failed"], report["groups"]), (2, 0, 1))
        self.assertTrue((output / "analysis" / "position.csv").is_file())
        self.assertFalse((output / "analysis" / "figures").exists())

    def test_run_with_figures_creates_outputs_and_reuses_simulation_and_analysis_caches(
        self,
    ) -> None:
        self.short_study(analysis=self.analysis_settings(figures=True))
        output = self.root / "figures"
        first = self.invoke("run", str(self.config), "--output", str(output), "--workers", "2")
        self.assertEqual((first["completed"], first["failed"]), (2, 0))
        self.assertEqual(len(first["figures"]), 1)
        figure = Path(first["figures"][0])
        self.assertIn("\\begin{tikzpicture}", figure.read_text(encoding="utf-8"))
        self.assertTrue((output / "analysis" / "position.csv").is_file())
        cached = self.cache_files(output)
        self.assertTrue(cached)
        second = self.invoke("run", str(self.config), "--output", str(output))
        self.assertEqual((second["skipped"], second["failed"]), (2, 0))
        self.assertEqual(second["figures"], first["figures"])
        self.assertEqual(self.cache_files(output), cached)

    def test_failed_or_explicitly_paused_runs_do_not_start_configured_analysis(self) -> None:
        for fail in (False, True):
            with self.subTest(fail=fail):
                self.short_study(analysis=self.analysis_settings(figures=True), fail=fail)
                output = self.root / str(fail)
                arguments = ["run", str(self.config), "--output", str(output)]
                if not fail:
                    arguments += ["--max-steps", "1"]
                result = subprocess.run(
                    self.command(*arguments), capture_output=True, text=True, timeout=30
                )
                self.assertEqual(result.returncode, 1 if fail else 0, result.stderr + result.stdout)
                report = json.loads(result.stdout)
                self.assertEqual(report["failed" if fail else "paused"], 2)
                self.assertNotIn("groups", report)
                self.assertNotIn("figures", report)
                self.assertFalse((output / "analysis").exists())

    def test_pending_only_report_returns_interruption_without_starting_analysis(self) -> None:
        """Exercise the pending-only exit condition without a timing-dependent signal race."""
        self.short_study(analysis=self.analysis_settings(figures=True))
        stdout = io.StringIO()
        with (
            patch("experiments_wo_stress.cli.run_experiment", return_value=RunReport(pending=2)),
            patch("experiments_wo_stress.analysis.figures.plot") as plot,
            redirect_stdout(stdout),
        ):
            status = main(["run", str(self.config), "--output", str(self.root / "pending")])
        self.assertEqual(status, 130)
        self.assertEqual(json.loads(stdout.getvalue())["pending"], 2)
        plot.assert_not_called()

    def test_analyze_and_plot_use_saved_results_after_simulation_code_is_removed(self) -> None:
        self.short_study()
        output = self.root / "saved"
        self.assertEqual(
            self.invoke("run", str(self.config), "--output", str(output))["completed"], 2
        )
        self.settings["analysis"] = self.analysis_settings(figures=True)
        self.write_configuration()
        (self.root / "signal_components.py").unlink()
        report = self.invoke("analyze", str(self.config), "--output", str(output))
        self.assertEqual(report["groups"], 1)
        self.assertTrue((output / "analysis" / "position.csv").is_file())
        report = self.invoke("plot", str(self.config), "--output", str(output))
        self.assertEqual(len(report["figures"]), 1)
        self.assertTrue(Path(report["figures"][0]).is_file())

    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGTERM semantics")
    def test_sigterm_checkpoints_and_resumes_with_identical_results(self) -> None:
        """Interrupt after a durable checkpoint, resume with two workers, and compare every row."""
        self.settings["analysis"] = self.analysis_settings(figures=True)
        self.write_configuration()
        interrupted = self.root / "interrupted"
        process = subprocess.Popen(
            self.command("run", str(self.config), "--output", str(interrupted)),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Wait for a committed step so SIGTERM exercises recovery from real work.
            deadline = time.monotonic() + 15
            checkpoint_found = False
            while time.monotonic() < deadline:
                for progress in interrupted.glob("runs/*/progress.json"):
                    snapshot = json.loads(progress.read_text(encoding="utf-8"))
                    if snapshot.get("checkpoints") and snapshot.get("status") == "running":
                        checkpoint_found = True
                        break
                if checkpoint_found or process.poll() is not None:
                    break
                time.sleep(0.02)
            self.assertTrue(checkpoint_found, "process did not commit a checkpoint before timeout")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            self.assertIn(process.returncode, (0, 130), stderr + stdout)
            report = json.loads(stdout)
            self.assertEqual(report["failed"], 0, report["errors"])
            self.assertGreaterEqual(report["paused"], 1)
            self.assertEqual(report["completed"] + report["paused"] + report["pending"], 2)
            self.assertNotIn("figures", report)
            self.assertFalse((interrupted / "analysis").exists())
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=15)

        # A resumed subprocess must finish all runs despite a changed worker count.
        resumed = self.invoke(
            "run", str(self.config), "--output", str(interrupted), "--workers", "2"
        )
        self.assertEqual(resumed["completed"] + resumed["skipped"], 2)
        self.assertEqual(resumed["failed"], 0)
        # The uninterrupted execution is the reference for every value and step.
        reference = self.root / "reference"
        self.assertEqual(
            self.invoke("run", str(self.config), "--output", str(reference))["completed"], 2
        )
        expected = {spec.run_id: result for spec, result in iter_completed_runs(reference)}
        actual = {spec.run_id: result for spec, result in iter_completed_runs(interrupted)}
        self.assertEqual(expected.keys(), actual.keys())
        for run_id, arrays in expected.items():
            for field, values in arrays.items():
                with self.subTest(run=run_id, field=field):
                    np.testing.assert_array_equal(values, actual[run_id][field])
            np.testing.assert_array_equal(actual[run_id]["step"], np.arange(1, 51))
        inspection = self.invoke("inspect", str(interrupted))
        self.assertEqual(inspection["counts"]["completed"], 2)


if __name__ == "__main__":
    unittest.main()
