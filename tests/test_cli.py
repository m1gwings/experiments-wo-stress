"""Verify the installed command and real process interruption at safe boundaries."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import yaml

from experiments_wo_stress.storage import iter_completed_runs

COMPONENTS = '''"""Slow, stateful components for an actual interruption test."""
import time
from experiments_wo_stress import Feedback

class Environment:
    """Accumulate a random position slowly enough for the test to send SIGTERM."""

    def __init__(self, *, rng):
        self.rng = rng
        self.position = 0.0

    def generate(self, action):
        time.sleep(0.01)
        self.position += float(self.rng.normal()) + action
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
    """Exercise the real CLI in subprocesses, including POSIX signal recovery."""

    def setUp(self) -> None:
        """Create a standalone study with slow, stateful paper components."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "signal_components.py").write_text(COMPONENTS, encoding="utf-8")
        self.config = self.root / "experiment.yml"
        settings = {
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
        self.config.write_text(yaml.safe_dump(settings), encoding="utf-8")

    def command(self, *args: str) -> list[str]:
        """Invoke the installed package with the same interpreter running these tests."""
        return [sys.executable, "-m", "experiments_wo_stress", *args]

    def invoke(self, *args: str) -> dict:
        """Run a successful CLI command and decode its machine-readable report."""
        result = subprocess.run(self.command(*args), capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def test_module_command_plans_without_running(self) -> None:
        result = self.invoke("plan", str(self.config))
        self.assertEqual(result["runs"], 2)
        self.assertEqual(len(result["plan"]), 2)
        self.assertFalse((self.root / "runs").exists())

    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGTERM semantics")
    def test_sigterm_checkpoints_and_resumes_with_identical_results(self) -> None:
        """Interrupt after a durable checkpoint, resume with two workers, and compare every row."""
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
