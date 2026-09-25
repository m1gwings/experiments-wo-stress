"""Compute observation preserves lifecycle, scientific reuse, and CLI artifacts."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments_wo_stress import load_config, plan_runs, run_experiment
from experiments_wo_stress.cli import main
from experiments_wo_stress.execution import compute, provenance
from experiments_wo_stress.execution.compatibility import (
    _COMPUTE_ONLY_REVISIONS,
    implementation_digest,
)
from experiments_wo_stress.storage import iter_completed_runs
from experiments_wo_stress.storage.files import atomic_json, read_json
from experiments_wo_stress.study.rng import make_rngs


class AttemptTimingTests(unittest.TestCase):
    """Use boundary clocks to distinguish elapsed, process CPU, and allocated GPU time."""

    def test_attempt_preserves_outcome_and_records_cpu_deltas_and_gpu_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {"run_id": "science", "group": "main", "algorithm_name": "learner"}

            @compute.observe_attempt
            def execute(*args, **kwargs):
                return "science", "paused", None

            with (
                patch.object(compute, "monotonic", side_effect=[10.0, 3610.0]),
                patch.object(compute, "_cpu_times", side_effect=[(15.0, 2.0), (75.0, 5.0)]),
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "7"}),
            ):
                result = execute(
                    spec,
                    directory,
                    {"_compute_invocation_id": "invocation", "gpu_ids": [7]},
                    {},
                    None,
                    "variant",
                )
            self.assertEqual(result, ("science", "paused", None))
            final = next(
                path
                for path in (root / "compute/attempts").glob("*.json")
                if not path.name.endswith(".start.json")
            )
            record = read_json(final)
            self.assertEqual(record["wall_seconds"], 3600)
            self.assertEqual(record["cpu_user_seconds"], 60)
            self.assertEqual(record["cpu_system_seconds"], 3)
            self.assertEqual(record["gpu_seconds"] / 3600, 1)
            self.assertEqual(record["gpu_ids"], [7])
            self.assertEqual(record["status"], "paused")
            self.assertIsNone(read_json(final.with_suffix(".start.json"))["wall_seconds"])

    def test_skip_records_zero_compute_and_exception_retains_failed_time(self):
        for status in ("skipped", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:

                @compute.observe_attempt
                def execute(*args, **kwargs):
                    if status == "failed":
                        raise RuntimeError("science failed")
                    return "science", status, None

                with (
                    patch.object(compute, "monotonic", side_effect=[0.0, 9.0]),
                    patch.object(compute, "_cpu_times", return_value=None),
                ):
                    arguments = (
                        {"run_id": "science", "group": "g", "algorithm_name": "a"},
                        directory,
                        {"_compute_invocation_id": "invocation"},
                        {},
                        None,
                        "v",
                    )
                    if status == "failed":
                        with self.assertRaisesRegex(RuntimeError, "science failed"):
                            execute(*arguments)
                    else:
                        execute(*arguments)
                record = next(
                    read_json(path)
                    for path in Path(directory).glob("compute/attempts/*.json")
                    if not path.name.endswith(".start.json")
                )
                self.assertEqual(record["wall_seconds"], 0 if status == "skipped" else 9)
                self.assertEqual(record["status"], status)
                self.assertEqual(record["gpu_seconds"], 0)
                self.assertEqual(record["allocated_gpu_count"], 0)


@unittest.skipUnless(os.name == "posix", "report publication uses POSIX locking")
class ComputeLifecycleTests(unittest.TestCase):
    """Exercise automatic reports using a small resumable scientific study."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.path = self.root / "study.yml"
        settings = {
            "name": "compute_accounting",
            "seed": 12,
            "runs": [
                {
                    "name": "main",
                    "budget": {"steps": 4},
                    "protocol": {"type": "online"},
                    "data": {"type": "gaussian_bandit", "params": {"means": [0.0, 1.0]}},
                    "algorithms": [
                        {"name": "learner", "type": "tests.sample_components:RecordingLearner"}
                    ],
                }
            ],
            "analysis": {
                "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
                "aggregator": {"group_by": ["algorithm.name"], "uncertainty": "none"},
                "figures": [{"name": "reward", "metric": "reward", "formats": ["tikz"]}],
            },
        }
        self.path.write_text(yaml.safe_dump(settings))
        self.config = load_config(self.path)
        # Keep hardware out of lifecycle assertions; unsupported mode is tested below.
        for name, value in (("supported", True), ("collect_machine", {"platform": "test"})):
            patcher = patch.object(compute, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_resume_appends_history_and_reuse_adds_no_attempt_compute(self):
        self.assertEqual(run_experiment(self.config, self.output, max_steps=2).paused, 1)
        first = next((self.output / "compute/invocations").glob("*.json"))
        original = first.read_bytes()
        self.assertEqual(run_experiment(self.config, self.output).completed, 1)
        before = read_json(self.output / "compute/summary.json")
        self.assertEqual(before["totals"]["attempt_count"], 2)
        self.assertGreater(before["totals"]["paused_worker_seconds"], 0)
        self.assertEqual(run_experiment(self.config, self.output).skipped, 1)
        after = read_json(self.output / "compute/summary.json")
        self.assertEqual(after["totals"]["worker_seconds"], before["totals"]["worker_seconds"])
        self.assertEqual(after["totals"]["attempt_count"], 2)
        self.assertEqual(after["totals"]["gpu_seconds"], 0)
        self.assertEqual(len(list((self.output / "compute/invocations").glob("*.json"))), 3)
        self.assertEqual(first.read_bytes(), original)
        self.assertEqual(after["latest_invocation"]["counts"]["skipped"], 1)

    def test_reporting_failures_preserve_completed_artifacts_and_success_exit(self):
        for boundary in (
            "collect_machine",
            "publish_attempt",
            "publish_invocation",
            "artifact_size",
        ):
            with self.subTest(boundary=boundary):
                output = self.root / boundary
                stderr = io.StringIO()
                with (
                    patch.object(compute, boundary, side_effect=OSError("report unavailable")),
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(stderr),
                ):
                    code = main(["run", str(self.path), "--output", str(output)])
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertEqual(len(list(iter_completed_runs(output))), 1)
                self.assertEqual(run_experiment(self.config, output).skipped, 1)

    def test_presentation_failure_cannot_change_scientific_success(self):
        with (
            patch.object(
                compute.Invocation, "_display_summary", side_effect=ValueError("bad display")
            ),
            redirect_stderr(io.StringIO()),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["run", str(self.path), "--output", str(self.output)]), 0)
        self.assertEqual(len(list(iter_completed_runs(self.output))), 1)

    def test_windows_runs_normally_without_reporting_or_warning(self):
        with (
            patch.object(compute, "supported", return_value=False),
            patch.object(compute, "collect_machine") as machine,
            self.assertNoLogs("experiments_wo_stress.execution.compute", level="WARNING"),
        ):
            self.assertEqual(run_experiment(self.config, self.output).completed, 1)
        machine.assert_not_called()
        self.assertFalse((self.output / "compute").exists())

    def test_cli_separates_stage_durations_keeps_json_and_exposes_inspect_pointer(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(["run", str(self.path), "--output", str(self.output)])
        self.assertEqual(code, 0, stderr.getvalue())
        report = json.loads(stdout.getvalue())
        self.assertTrue(Path(report["compute"]["report"]).is_file())
        self.assertIn("worker-hours", stderr.getvalue())
        saved = read_json(self.output / "compute/summary.json")["latest_invocation"]
        for stage in ("execution", "analysis", "plotting"):
            self.assertGreaterEqual(saved["stages"][stage], 0)
        self.assertGreaterEqual(saved["wall_seconds"], sum(saved["stages"].values()))
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["inspect", str(self.output)]), 0)
        self.assertEqual(
            json.loads(stdout.getvalue())["compute_report"], report["compute"]["report"]
        )

    def test_analysis_failure_still_publishes_invocation_and_completed_simulation(self):
        with (
            patch(
                "experiments_wo_stress.analysis.pipeline.analyze",
                side_effect=ValueError("bad metric"),
            ),
            redirect_stderr(io.StringIO()),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["run", str(self.path), "--output", str(self.output)]), 1)
        saved = read_json(self.output / "compute/summary.json")["latest_invocation"]
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["counts"]["completed"], 1)
        self.assertIsNotNone(saved["stages"]["analysis"])
        self.assertIsNone(saved["stages"]["plotting"])
        self.assertEqual(len(list(iter_completed_runs(self.output))), 1)

    def test_compute_changes_preserve_legacy_variants_rng_and_analysis_cache(self):
        original_collect = provenance.collect_provenance

        def legacy_provenance(*args):
            result = original_collect(*args)
            for name, (_, previous) in _COMPUTE_ONLY_REVISIONS.items():
                if name in result["implementation"]:
                    result["implementation"][name] = previous
            return result

        with patch.object(provenance, "collect_provenance", side_effect=legacy_provenance):
            # Coordinator imports the function directly; emulate the old source
            # inventory there as well to produce a real compatible saved variant.
            with patch(
                "experiments_wo_stress.execution.coordinator.collect_provenance", legacy_provenance
            ):
                self.assertEqual(run_experiment(self.config, self.output).completed, 1)
        spec = plan_runs(self.config)[0]
        expected_rng = make_rngs(spec)["algorithm"].random(5)
        from experiments_wo_stress.analysis.figures import plot

        with patch(
            "experiments_wo_stress.analysis.figures.implementation_digest",
            return_value=_COMPUTE_ONLY_REVISIONS["analysis/figures.py"][1],
        ):
            plot(self.config, self.output)
        cache = {
            path: path.read_bytes()
            for path in (self.output / "analysis/cache").rglob("*")
            if path.is_file()
        }
        with patch.object(
            compute, "collect_machine", return_value={"platform": "different hardware"}
        ):
            self.assertEqual(run_experiment(self.config, self.output, workers=1).skipped, 1)
        atomic_json(self.output / "compute/unrelated.json", {"new": "operational metadata"})
        plot(self.config, self.output)
        self.assertEqual(
            cache,
            {
                path: path.read_bytes()
                for path in (self.output / "analysis/cache").rglob("*")
                if path.is_file()
            },
        )
        np.testing.assert_array_equal(
            make_rngs(plan_runs(self.config)[0])["algorithm"].random(5), expected_rng
        )
        self.assertEqual(len(list((self.output / "runs").iterdir())), 1)

    def test_reviewed_hooks_preserve_old_digests_but_future_edits_do_not(self):
        package = Path(provenance.__file__).resolve().parents[1]
        for name, (reviewed, previous) in _COMPUTE_ONLY_REVISIONS.items():
            with self.subTest(module=name):
                self.assertEqual(provenance.digest_file(package / name), reviewed)
                self.assertEqual(implementation_digest(name, reviewed), previous)
                self.assertEqual(
                    implementation_digest(name, "changed implementation"), "changed implementation"
                )


if __name__ == "__main__":
    unittest.main()
