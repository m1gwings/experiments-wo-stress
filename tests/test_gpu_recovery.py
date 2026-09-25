"""GPU worker interruption, abrupt exits, and clean reuse without CUDA hardware."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

from experiments_wo_stress.storage import iter_completed_runs

COMPONENTS = '''"""Slow deterministic science with an optional one-time process crash."""
import os
import time
from pathlib import Path
from experiments_wo_stress import Feedback

class Environment:
    """Keep random position and step state across interruption and recovery."""

    def __init__(self, *, rng, crash_marker=None, release_marker=None):
        self.rng = rng
        self.crash_marker = Path(crash_marker) if crash_marker else None
        self.release_marker = Path(release_marker) if release_marker else None
        self.position = 0.0
        self.step = 0

    def generate(self, action):
        time.sleep(0.01)
        self.step += 1
        self.position += float(self.rng.normal()) + action
        if self.crash_marker is not None and self.step == 4:
            try:
                self.crash_marker.touch(exist_ok=False)
            except FileExistsError:
                pass
            else:
                if self.release_marker is not None and os.fork() == 0:
                    # Hold inherited worker pipes open until the coordinator
                    # detects its assigned worker's death and returns control.
                    deadline = time.monotonic() + 60
                    while not self.release_marker.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    os._exit(0)
                os._exit(23)
        return Feedback(self.position, {"position": self.position})

    def state_dict(self):
        return {"position": self.position, "step": self.step}

    def load_state_dict(self, state):
        self.position = state["position"]
        self.step = state["step"]

class Learner:
    """Choose actions from an injected RNG and the saved feedback history."""

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

RETRY_DRIVER = '''"""Bound crash recovery checks by the enclosing subprocess timeout."""
import json
import multiprocessing
import os
import sys
from pathlib import Path
from experiments_wo_stress import run_experiment

if __name__ == "__main__":
    config, output, release_marker = sys.argv[1:]
    environment = dict(os.environ)
    reports = []
    for attempt in range(2):
        report = run_experiment(config, output)
        Path(release_marker).touch()
        metadata = json.loads((Path(output) / "metadata.json").read_text())
        reports.append({
            "report": report.to_dict(),
            "environment_unchanged": dict(os.environ) == environment,
            "children": [child.pid for child in multiprocessing.active_children()],
            "variants": metadata["run_ids"],
        })
    print(json.dumps(reports))
'''


class GPURecoveryTests(unittest.TestCase):
    """Recover numerical trajectories and release worker ownership on all exits."""

    def setUp(self) -> None:
        """Create a standalone study whose four runs exceed its two GPU slots."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "gpu_recovery_components.py").write_text(COMPONENTS, encoding="utf-8")
        self.config = self.root / "experiment.yml"
        self.settings = {
            "name": "gpu_recovery",
            "seed": 91,
            "runs": [
                {
                    "name": "main",
                    "repetitions": 4,
                    "budget": {"steps": 100},
                    "protocol": {"type": "online"},
                    "data": {"type": "gpu_recovery_components:Environment"},
                    "algorithms": [{"name": "learner", "type": "gpu_recovery_components:Learner"}],
                }
            ],
            "execution": {"workers": 2, "gpu_ids": [2, 7], "checkpoint_steps": 2},
            "recording": {"buffer_bytes": 64},
        }
        self.write_configuration()
        self.environment = dict(os.environ)
        self.environment.pop("CUDA_VISIBLE_DEVICES", None)

    def write_configuration(self) -> None:
        """Publish the same scientific description for every compared execution."""
        self.config.write_text(yaml.safe_dump(self.settings), encoding="utf-8")

    @contextmanager
    def process(self, command: list[str]) -> Iterator[subprocess.Popen]:
        """Reap the coordinator and its process group even if a regression hangs."""
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.environment,
            start_new_session=os.name == "posix",
        )
        try:
            yield process
        finally:
            if os.name == "posix":
                # A failed coordinator can leave workers holding these pipes
                # even after it exits, so clean the entire session in that case.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.communicate(timeout=15)

    def run_command(self, output: Path) -> list[str]:
        """Use the current interpreter for the actual public CLI."""
        return [
            sys.executable,
            "-m",
            "experiments_wo_stress",
            "run",
            str(self.config),
            "--output",
            str(output),
        ]

    def complete(self, output: Path) -> dict:
        """Run the configured GPU study to completion within a bounded wait."""
        with self.process(self.run_command(output)) as process:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr + stdout)
        return json.loads(stdout)

    def assert_results_equal(self, expected: Path, actual: Path) -> None:
        """Compare every scientific observation and reject duplicated resumed steps."""
        expected_runs = {spec.run_id: result for spec, result in iter_completed_runs(expected)}
        actual_runs = {spec.run_id: result for spec, result in iter_completed_runs(actual)}
        self.assertEqual(len(expected_runs), 4)
        self.assertEqual(expected_runs.keys(), actual_runs.keys())
        for run_id, result in expected_runs.items():
            self.assertEqual(result.keys(), actual_runs[run_id].keys())
            for field, values in result.items():
                with self.subTest(run=run_id, field=field):
                    np.testing.assert_array_equal(values, actual_runs[run_id][field])
            np.testing.assert_array_equal(actual_runs[run_id]["step"], np.arange(1, 101))

    @unittest.skipUnless(os.name == "posix", "requires POSIX SIGTERM semantics")
    def test_sigterm_pauses_active_gpu_runs_and_keeps_unscheduled_work_pending(self) -> None:
        """A real signal checkpoints two active runs and resumes the same retained variants."""
        interrupted = self.root / "interrupted"
        with self.process(self.run_command(interrupted)) as process:
            deadline = time.monotonic() + 20
            durable = []
            while time.monotonic() < deadline and process.poll() is None:
                durable = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in interrupted.glob("runs/*/progress.json")
                ]
                if len(durable) == 2 and all(
                    item.get("checkpoints") and item.get("status") == "running" for item in durable
                ):
                    break
                time.sleep(0.01)
            self.assertEqual(len(durable), 2, "two GPU runs did not reach durable work")
            self.assertTrue(all(item.get("checkpoints") for item in durable))
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=20)
            self.assertEqual(process.returncode, 130, stderr + stdout)
            report = json.loads(stdout)
        self.assertEqual(report["failed"], 0, report["errors"])
        self.assertEqual((report["completed"], report["paused"], report["pending"]), (0, 2, 2))
        self.assertEqual(len(list(interrupted.glob("runs/*/progress.json"))), 2)
        variants = json.loads((interrupted / "metadata.json").read_text())["run_ids"]

        self.settings["execution"]["gpu_ids"] = [9, 4]
        self.write_configuration()
        self.assertEqual(self.complete(interrupted)["completed"], 4)
        resumed_metadata = json.loads((interrupted / "metadata.json").read_text())
        self.assertEqual(variants, resumed_metadata["run_ids"])
        self.assertEqual(resumed_metadata["provenance"]["execution"]["gpu_ids"], [9, 4])
        self.assertEqual(len(list((interrupted / "runs").iterdir())), 4)
        reference = self.root / "reference"
        self.assertEqual(self.complete(reference)["completed"], 4)
        self.assert_results_equal(reference, interrupted)

    def test_abrupt_worker_exit_releases_resources_and_allows_a_clean_retry(self) -> None:
        """Detect worker death despite inherited pipes, then retry with fresh ownership."""
        marker = self.root / "crashed_once"
        release_marker = self.root / "release_descendant"
        parameters = {"crash_marker": str(marker)}
        if os.name == "posix":
            # The forked descendant outlives the failed worker and retains both
            # its command pipe and multiprocessing's process-sentinel pipe.
            parameters["release_marker"] = str(release_marker)
        self.settings["runs"][0]["data"]["params"] = parameters
        self.write_configuration()
        driver = self.root / "retry_driver.py"
        driver.write_text(RETRY_DRIVER, encoding="utf-8")
        recovered = self.root / "recovered"
        with self.process(
            [sys.executable, str(driver), str(self.config), str(recovered), str(release_marker)]
        ) as process:
            stdout, stderr = process.communicate(timeout=30)
            self.assertEqual(process.returncode, 0, stderr + stdout)
        first, second = json.loads(stdout)
        self.assertEqual(first["report"]["failed"], 1, first["report"])
        self.assertEqual(first["report"]["pending"], 2, first["report"])
        self.assertEqual(first["report"]["paused"], 1, first["report"])
        self.assertIn("exited without a result", next(iter(first["report"]["errors"].values())))
        self.assertEqual(second["report"]["failed"], 0, second["report"])
        self.assertEqual(second["report"]["completed"], 4, second["report"])
        self.assertEqual(first["variants"], second["variants"])
        for attempt in (first, second):
            self.assertTrue(attempt["environment_unchanged"])
            self.assertEqual(attempt["children"], [])
        self.assertTrue(marker.exists())
        self.assertTrue(release_marker.exists())
        reference = self.root / "reference"
        self.assertEqual(self.complete(reference)["completed"], 4)
        self.assert_results_equal(reference, recovered)
