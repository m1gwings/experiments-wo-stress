"""Verify GPU allocation through process-visible state, without a CUDA runtime.

The fake learner records import and construction environments. GPU IDs are only
labels here; actual results still use the injected NumPy stream.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from rich.console import Console

from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.execution.compute_environment import supported
from experiments_wo_stress.execution.progress import TerminalProgress
from experiments_wo_stress.storage import iter_completed_runs

COMPONENTS = '''"""Inspect process visibility without importing a GPU framework."""
import os
import time
from pathlib import Path
from experiments_wo_stress import GridPlanner, StateMixin

IMPORT_GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")

class Learner(StateMixin):
    """Record worker identity, enforce exclusive use, and sample a scientific RNG."""

    def __init__(self, *, rng, audit, peers=1):
        self.rng = rng
        self.audit = Path(audit)
        self.audit.mkdir(exist_ok=True)
        self.peers = peers
        self.gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")
        self.claim = self.audit / ("active_" + self.gpu)
        self.claim.touch(exist_ok=False)

    def fit(self, dataset):
        (self.audit / ("seen_" + self.gpu)).touch()
        deadline = time.monotonic() + 10
        while len(list(self.audit.glob("seen_*"))) < self.peers:
            if time.monotonic() > deadline:
                raise RuntimeError("peer worker did not start")
            time.sleep(0.01)
        time.sleep(0.05)
        return {"pid": os.getpid(), "gpu": int(self.gpu),
                "import_gpu": int(IMPORT_GPU), "sample": self.rng.random()}

    def close(self):
        self.claim.unlink()

class BoundPlanner(GridPlanner):
    """Fail if planning runs in a process without its GPU assignment."""

    def plan(self, group, seed):
        assert IMPORT_GPU != "-1", "planning must own a GPU"
        yield from super().plan(group, seed)

class FailOnce(Learner):
    """Fail one fit to verify component cleanup and continued worker ownership."""

    def fit(self, dataset):
        marker = self.audit / "failed_once"
        if not marker.exists():
            marker.touch()
            raise RuntimeError("injected scientific failure")
        return super().fit(dataset)
'''


class GPUExecutionTests(unittest.TestCase):
    """Exercise persistent assignments and CPU compatibility through the public API."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module = f"gpu_components_{self._testMethodName}"
        (self.root / f"{self.module}.py").write_text(COMPONENTS, encoding="utf-8")
        self.environment = patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def configuration(
        self, gpu_ids=None, *, repetitions=1, peers=1, planner="grid", learner="Learner"
    ):
        """Load a small offline study, retaining its scientific inputs across modes."""
        settings = {
            "name": "gpu_visibility",
            "seed": 37,
            "runs": [
                {
                    "repetitions": repetitions,
                    "planner": planner,
                    "protocol": {"type": "offline"},
                    "data": {"type": "null"},
                    "algorithms": [
                        {
                            "name": "learner",
                            "type": f"{self.module}:{learner}",
                            "params": {"audit": str(self.root / "audit"), "peers": peers},
                        }
                    ],
                }
            ],
        }
        if gpu_ids is not None:
            settings["execution"] = {"workers": len(gpu_ids), "gpu_ids": gpu_ids}
        path = self.root / "experiment.yml"
        path.write_text(yaml.safe_dump(settings), encoding="utf-8")
        return load_config(path)

    def results(self, output):
        """Read recorded worker observations from validated completed results."""
        return {spec.run_id: result for spec, result in iter_completed_runs(output)}

    def attempts(self, output):
        """Read terminal observations while excluding their immutable start records."""
        return [
            json.loads(path.read_text())
            for path in (output / "compute" / "attempts").glob("*.json")
            if not path.name.endswith(".start.json")
        ]

    def test_cpu_single_worker_retains_coordinator_and_inherited_visibility(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "12"
        config = self.configuration()
        output = self.root / "cpu"
        report = run_experiment(config, output)
        self.assertEqual((report.completed, report.failed), (1, 0), report.errors)
        result = next(iter(self.results(output).values()))
        self.assertEqual(result["pid"][0], os.getpid())
        self.assertEqual(result["gpu"][0], 12)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "12")
        if supported():
            (attempt,) = self.attempts(output)
            self.assertEqual(attempt["status"], "completed")
            self.assertEqual(attempt["allocated_gpu_count"], 0)
            self.assertEqual(attempt["gpu_ids"], [])
            self.assertEqual(attempt["gpu_seconds"], 0)
            summary = json.loads((output / "compute" / "summary.json").read_text())
            self.assertEqual(summary["totals"]["gpu_seconds"], 0)

    def test_one_gpu_spawns_before_import_and_logs_its_assignment(self):
        config = self.configuration([7], planner=f"{self.module}:BoundPlanner")
        output = self.root / "gpu"
        report = run_experiment(config, output)
        self.assertEqual((report.completed, report.failed), (1, 0), report.errors)
        result = next(iter(self.results(output).values()))
        self.assertNotEqual(result["pid"][0], os.getpid())
        self.assertEqual(result["gpu"][0], 7)
        self.assertEqual(result["import_gpu"][0], 7)
        self.assertNotIn(self.module, sys.modules, "coordinator imported study code")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
        metadata = json.loads((output / "metadata.json").read_text())
        execution = metadata["provenance"]["execution"]
        self.assertEqual(execution["gpu_ids"], [7])
        self.assertEqual(
            execution["worker_assignments"], [{"gpu_id": 7, "pid": int(result["pid"][0])}]
        )
        resolved = yaml.safe_load((output / "config.resolved.yml").read_text())
        self.assertEqual(resolved["execution"]["gpu_ids"], [7])
        log = next(output.glob("runs/*/run.log")).read_text()
        self.assertIn("CUDA_VISIBLE_DEVICES=7 local_device=cuda:0", log)
        if supported():
            (attempt,) = self.attempts(output)
            self.assertEqual(attempt["status"], "completed")
            self.assertEqual(attempt["allocated_gpu_count"], 1)
            self.assertEqual(attempt["gpu_ids"], [7])
            self.assertGreaterEqual(attempt["wall_seconds"], 0)
            self.assertEqual(attempt["gpu_seconds"], attempt["wall_seconds"])
            summary = json.loads((output / "compute" / "summary.json").read_text())
            self.assertEqual(summary["totals"]["gpu_seconds"], attempt["wall_seconds"])
            self.assertEqual(summary["totals"]["gpu_hours"], attempt["wall_seconds"] / 3600)

    def test_progress_observes_gpu_workers_without_importing_study_in_parent(self):
        """The same progress queue observes real assigned processes and keeps bounded rows."""
        config = self.configuration([3, 7], repetitions=4, peers=2)
        output = self.root / "progress"
        stream = io.StringIO()
        with TerminalProgress(config.name, 2, console=Console(file=stream)) as progress:
            report = run_experiment(config, output, progress=progress)
            progress.finish(report.to_dict())
        self.assertEqual((report.completed, report.failed), (4, 0), report.errors)
        self.assertEqual(len(progress.rows), 2)
        self.assertTrue(all(row.fraction == 1 for row in progress.rows.values()))
        self.assertNotIn(self.module, sys.modules)
        self.assertNotIn("\x1b", stream.getvalue())
        self.assertFalse(multiprocessing.active_children())

    def test_workers_keep_distinct_assignments_across_runs(self):
        config = self.configuration([3, 8], repetitions=8, peers=2)
        output = self.root / "gpu"
        report = run_experiment(config, output)
        self.assertEqual((report.completed, report.failed), (8, 0), report.errors)
        by_process = {}
        for result in self.results(output).values():
            pid, gpu = int(result["pid"][0]), int(result["gpu"][0])
            self.assertEqual(result["import_gpu"][0], gpu)
            by_process.setdefault(pid, []).append(gpu)
        self.assertEqual(len(by_process), 2)
        self.assertEqual({values[0] for values in by_process.values()}, {3, 8})
        for values in by_process.values():
            self.assertEqual(len(set(values)), 1)
            self.assertGreater(len(values), 1)
        self.assertFalse(list((self.root / "audit").glob("active_*")))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_device_changes_reuse_variants_and_preserve_random_results(self):
        cpu = self.configuration(repetitions=3)
        gpu = replace(cpu, execution={**cpu.execution, "gpu_ids": [4]})
        first, second = self.root / "cpu", self.root / "gpu"
        self.assertEqual(run_experiment(cpu, first).completed, 3)
        self.assertEqual(run_experiment(gpu, second).completed, 3)
        expected, actual = self.results(first), self.results(second)
        self.assertEqual(expected.keys(), actual.keys())
        for run_id in expected:
            np.testing.assert_array_equal(expected[run_id]["sample"], actual[run_id]["sample"])
        self.assertEqual(run_experiment(gpu, first).skipped, 3)
        moved = replace(gpu, execution={**gpu.execution, "gpu_ids": [9]})
        self.assertEqual(run_experiment(moved, first).skipped, 3)
        self.assertEqual(len(list((first / "runs").iterdir())), 3)

    def test_inherited_masks_and_worker_overrides_fail_before_execution(self):
        config = self.configuration([2])
        output = self.root / "gpu"
        for mask in ("", "2", "2,3", "-1"):
            with self.subTest(mask=mask), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=mask):
                with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES"):
                    run_experiment(config, output)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)
        with self.assertRaisesRegex(ValueError, "one worker per GPU"):
            run_experiment(config, output, workers=2)
        self.assertFalse(output.exists())

    def test_preflight_failure_releases_processes_and_allows_retry(self):
        config = self.configuration([2, 5], repetitions=2)
        invalid = replace(config, runs=[{**config.runs[0], "planner": "missing_gpu_code:Planner"}])
        children = {child.pid for child in multiprocessing.active_children()}
        with self.assertRaisesRegex(RuntimeError, "missing_gpu_code"):
            run_experiment(invalid, self.root / "failed")
        self.assertEqual({child.pid for child in multiprocessing.active_children()}, children)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
        report = run_experiment(config, self.root / "retry")
        self.assertEqual((report.completed, report.failed), (2, 0), report.errors)

    def test_component_failure_cleans_up_and_worker_continues_with_same_gpu(self):
        config = self.configuration([4], repetitions=3, learner="FailOnce")
        output = self.root / "gpu"
        report = run_experiment(config, output)
        self.assertEqual((report.failed, report.completed, report.pending), (1, 2, 0))
        self.assertIn("injected scientific failure", next(iter(report.errors.values())))
        results = list(self.results(output).values())
        self.assertEqual(len({int(result["pid"][0]) for result in results}), 1)
        self.assertTrue(all(result["gpu"][0] == 4 for result in results))
        self.assertFalse(list((self.root / "audit").glob("active_*")))
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
        report = run_experiment(config, output)
        self.assertEqual((report.completed, report.skipped, report.failed), (1, 2, 0))

    def test_spawn_main_import_already_sees_assigned_visibility(self):
        """Model a framework imported at launcher module scope, before a pool initializer."""
        self.configuration([6])
        script = self.root / "launch.py"
        script.write_text(
            "import os\nfrom pathlib import Path\n"
            f"import {self.module} as science\n"
            "from experiments_wo_stress import run_experiment\n"
            "if __name__ == '__mp_main__':\n"
            "    assert science.IMPORT_GPU == '6', 'GPU mask arrived after main import'\n"
            "if __name__ == '__main__':\n"
            "    root = Path(__file__).parent\n"
            "    report = run_experiment(root / 'experiment.yml', root / 'output')\n"
            "    assert report.completed == 1, report.to_dict()\n",
            encoding="utf-8",
        )
        process = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=30
        )
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        result = next(iter(self.results(self.root / "output").values()))
        self.assertEqual(result["import_gpu"][0], 6)
