"""Pipeline reuse and disposable trajectories through the public execution API.

Tiny real studies exercise durable dependency boundaries, invalidation, and fresh
rematerialization. Scheduler policy and graph traversal have separate unit tests.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments_wo_stress import run_experiment
from experiments_wo_stress.analysis import analyze
from experiments_wo_stress.analysis.graph import execute_derivation
from experiments_wo_stress.execution.worker import RunSession
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage import StorageError, atomic_json, fingerprint, read_json
from experiments_wo_stress.storage.experiment import inspect_experiment
from experiments_wo_stress.storage.trajectories import (
    pruning_history,
    pruning_receipt,
    rematerialize_trajectory,
    result_reference,
)
from experiments_wo_stress.study.specs import RunSpec
from tests.test_execution import _ExecutionStudyTestCase


class _PipelineCase(_ExecutionStudyTestCase):
    """Provide two repetitions with inexpensive field metrics and a TikZ figure."""

    def pipeline_settings(self, *, retention: str = "until_analyzed") -> dict:
        settings = self.single_run_settings()
        settings["runs"][0]["repetitions"] = 2
        settings["runs"][0]["protocol"]["params"]["horizon"] = 6
        settings["recording"]["retention"] = retention
        settings["analysis"] = {
            "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
            "aggregator": {"group_by": ["algorithm.name"], "uncertainty": "standard_error"},
            "figures": [{"type": "line", "metric": "reward", "formats": ["tikz"]}],
        }
        return settings

    def variants(self, output: Path) -> list[Path]:
        return [output / "runs" / key for key in read_json(output / "metadata.json")["run_ids"]]

    def assert_pruned(self, output: Path) -> None:
        for directory in self.variants(output):
            self.assertTrue((directory / "trajectory.json").is_file())
            self.assertFalse(list((directory / "results").glob("*.npz")))
            self.assertTrue(list((directory / "checkpoints").iterdir()))
            metadata = read_json(directory / "metadata.json")
            self.assertTrue((output / "instances" / metadata["instance_id"]).is_dir())

    def run_with_task_counts(self, config, output: Path):
        kinds = Counter()

        def execute(task):
            kinds[task.kind] += 1
            return execute_derivation(task)

        with patch("experiments_wo_stress.analysis.graph.execute_derivation", side_effect=execute):
            report = run_experiment(config, output)
        self.assertEqual((report.failed, report.task_failures), (0, 0), report.errors)
        return report, kinds

    def assert_summaries_equal(self, config, first: Path, second: Path) -> None:
        expected = analyze(config, first)
        actual = analyze(config, second)
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertEqual(
                (left.metric, left.labels, left.count), (right.metric, right.labels, right.count)
            )
            np.testing.assert_array_equal(left.x, right.x)
            np.testing.assert_array_equal(left.mean, right.mean)
            np.testing.assert_array_equal(left.uncertainty, right.uncertainty)


class RetentionBoundaryTests(_PipelineCase):
    """Pruning is authorized only by every configured metric's durable generation."""

    def test_failed_second_metric_keeps_raw_until_all_metrics_are_durable(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 1
        settings["analysis"]["metrics"] = [
            {"name": "a_reward", "type": "field", "params": {"field": "reward"}},
            {"name": "z_action", "type": "field", "params": {"field": "action"}},
        ]
        settings["analysis"]["figures"] = []
        config = self.load_study_config(settings)
        output = self.root / "output"
        observations = []

        def fail_last_metric(task):
            if task.kind == "metric" and task.payload["name"] == "z_action":
                directory = self.variants(output)[0]
                observations.append(list((directory / "results").glob("*.npz")))
                self.assertFalse((directory / "trajectory.json").exists())
                raise RuntimeError("injected final metric failure")
            return execute_derivation(task)

        with patch(
            "experiments_wo_stress.analysis.graph.execute_derivation", side_effect=fail_last_metric
        ):
            report = run_experiment(config, output)
        self.assertEqual((report.completed, report.task_failures), (1, 1))
        self.assertTrue(observations[0])
        self.assertTrue(list((self.variants(output)[0] / "results").glob("*.npz")))

        def observe_aggregate(task):
            if task.kind == "aggregate" and task.subject.startswith("z_action"):
                self.assert_pruned(output)
            return execute_derivation(task)

        with patch(
            "experiments_wo_stress.analysis.graph.execute_derivation", side_effect=observe_aggregate
        ):
            report = run_experiment(config, output)
        self.assertEqual((report.skipped, report.task_failures), (1, 0), report.errors)
        self.assert_pruned(output)

    def test_keep_default_preserves_raw_and_no_metrics_does_not_authorize_pruning(self) -> None:
        settings = self.pipeline_settings()
        settings["recording"].pop("retention")
        config = self.load_study_config(settings)
        output = self.root / "keep"
        self.assertEqual(run_experiment(config, output).completed, 2)
        for directory in self.variants(output):
            self.assertFalse((directory / "trajectory.json").exists())
            self.assertTrue(list((directory / "results").glob("*.npz")))
        settings["recording"]["retention"] = "until_analyzed"
        settings.pop("analysis")
        output = self.root / "no-metrics"
        self.assertEqual(run_experiment(self.load_study_config(settings), output).completed, 2)
        for directory in self.variants(output):
            self.assertFalse((directory / "trajectory.json").exists())
            self.assertTrue(list((directory / "results").glob("*.npz")))

    def test_missing_trajectory_is_corruption_and_not_intentional_pruning(self) -> None:
        config = self.load_study_config(self.pipeline_settings(retention="keep"))
        output = self.root / "output"
        run_experiment(config, output)
        directory = self.variants(output)[0]
        next((directory / "results").glob("*.npz")).unlink()
        self.assertEqual(inspect_experiment(output)["counts"]["corrupt"], 1)
        self.assertFalse((directory / "trajectory.json").exists())
        report = run_experiment(config, output)
        self.assertEqual(report.failed, 1)
        self.assertIn("Missing or corrupt", next(iter(report.errors.values())))

    def test_modified_pruning_proof_is_rejected(self) -> None:
        config = self.load_study_config(self.pipeline_settings())
        output = self.root / "output"
        run_experiment(config, output)
        marker = self.variants(output)[0] / "trajectory.json"
        marker.write_text(marker.read_text().replace('"state": "pruned"', '"state": "missing"'))
        self.assertEqual(inspect_experiment(output)["counts"]["corrupt"], 1)
        with self.assertRaisesRegex(StorageError, "Invalid trajectory pruning receipt"):
            run_experiment(config, output)

    def test_malformed_current_and_historical_proofs_fail_with_storage_errors(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 1
        output = self.root / "output"
        run_experiment(self.load_study_config(settings), output)
        directory = self.variants(output)[0]
        active = read_json(directory / "trajectory.json")
        rematerialize_trajectory(directory)
        historical = read_json(directory / "trajectory-history.json")
        reference = active["references"]["6"]
        changes = [
            {"schema_version": True},
            {"references": []},
            {"references": {}},
            {"references": {"06": reference}},
            {"references": {"6": {**reference, "completed_steps": True}}},
            {"references": {"6": {**reference, "completed_steps": -1}}},
            {"references": {"6": {**reference, "completed_steps": 6.0}}},
            {"references": {"6": {**reference, "revision": "invalid"}}},
            {"references": {"6": []}},
            {"latest": "999"},
            {"latest": []},
        ]
        for filename, valid, reader in (
            ("trajectory.json", active, pruning_receipt),
            ("trajectory-history.json", historical, pruning_history),
        ):
            candidates = [[], *({**valid, **change} for change in changes)]
            if reader is pruning_receipt:
                candidates.extend(({**valid, "metrics": []}, {**valid, "metrics": {}}))
            for candidate in candidates:
                with self.subTest(filename=filename, candidate=candidate):
                    if isinstance(candidate, dict):
                        candidate["sha256"] = fingerprint(
                            {key: value for key, value in candidate.items() if key != "sha256"}
                        )
                    atomic_json(directory / filename, candidate)
                    with self.assertRaisesRegex(StorageError, "Invalid trajectory pruning receipt"):
                        reader(directory)
            atomic_json(directory / filename, valid)
        spec = RunSpec.from_dict(
            {**read_json(directory / "metadata.json")["spec"], "run_id": "f" * 32}
        )
        with self.assertRaisesRegex(StorageError, "Run identity mismatch"):
            result_reference(output, directory.name, spec)


class DependencyReuseTests(_PipelineCase):
    """Only missing requested descendants and necessary raw ancestors are executed."""

    def test_exact_rerun_after_pruning_executes_no_simulations_or_derivations(self) -> None:
        config = self.load_study_config(self.pipeline_settings())
        output = self.root / "output"
        report, kinds = self.run_with_task_counts(config, output)
        self.assertEqual(report.completed, 2)
        self.assertEqual(kinds, {"metric": 2, "aggregate": 1, "figure": 1})
        self.assert_pruned(output)
        with patch(
            "experiments_wo_stress.execution.coordinator.execute_run",
            side_effect=AssertionError("unexpected simulation"),
        ):
            report, kinds = self.run_with_task_counts(config, output)
        self.assertEqual((report.completed, report.skipped), (0, 2))
        self.assertFalse(kinds)
        self.assertEqual(inspect_experiment(output)["counts"]["completed"], 2)

    def test_figure_and_aggregation_changes_stop_at_their_retained_inputs(self) -> None:
        settings = self.pipeline_settings()
        output = self.root / "output"
        run_experiment(self.load_study_config(settings), output)
        settings["analysis"]["figures"][0]["title"] = "A new figure title"
        report, kinds = self.run_with_task_counts(self.load_study_config(settings), output)
        self.assertEqual(report.skipped, 2)
        self.assertEqual(kinds, {"figure": 1})
        settings["analysis"]["aggregator"]["uncertainty"] = "none"
        report, kinds = self.run_with_task_counts(self.load_study_config(settings), output)
        self.assertEqual(report.skipped, 2)
        self.assertEqual(kinds, {"aggregate": 1, "figure": 1})
        self.assert_pruned(output)

    def test_changed_metric_recomputes_simulations_only_when_raw_was_pruned(self) -> None:
        for retention in ("keep", "until_analyzed"):
            with self.subTest(retention=retention):
                settings = self.pipeline_settings(retention=retention)
                output = self.root / retention
                config = self.load_study_config(settings)
                run_experiment(config, output)
                identities = [directory.name for directory in self.variants(output)]
                settings["analysis"]["metrics"][0]["type"] = "cumulative_sum"
                changed = self.load_study_config(settings)
                report, kinds = self.run_with_task_counts(changed, output)
                self.assertEqual(report.completed, 2 if retention == "until_analyzed" else 0)
                self.assertEqual(report.skipped, 2 if retention == "keep" else 0)
                self.assertEqual(kinds, {"metric": 2, "aggregate": 1, "figure": 1})
                self.assertEqual(
                    [directory.name for directory in self.variants(output)], identities
                )
                direct = self.root / f"direct-{retention}"
                run_experiment(changed, direct)
                self.assert_summaries_equal(changed, direct, output)

    def test_standalone_commands_diagnose_required_pruned_raw_without_simulating(self) -> None:
        settings = self.pipeline_settings()
        output = self.root / "output"
        run_experiment(self.load_study_config(settings), output)
        settings["analysis"]["metrics"][0]["type"] = "cumulative_sum"
        changed = self.load_study_config(settings)
        with patch(
            "experiments_wo_stress.execution.coordinator.execute_run",
            side_effect=AssertionError("standalone simulation"),
        ):
            for operation in (analyze, plot):
                with self.subTest(operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, "ews run.*rematerialize"):
                        operation(changed, output)
        self.assert_pruned(output)


class RematerializationTests(_PipelineCase):
    """Fresh replay and paused replay match direct computation after raw deletion."""

    def test_earlier_budget_reuses_its_derivations_after_a_larger_budget_was_pruned(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 1
        original = self.load_study_config(settings)
        output = self.root / "output"
        self.assertEqual(run_experiment(original, output).completed, 1)
        settings["runs"][0]["protocol"]["params"]["horizon"] = 9
        longer = self.load_study_config(settings)
        self.assertEqual(run_experiment(longer, output).completed, 1)
        directory = self.variants(output)[0]
        marker_before = (directory / "trajectory.json").read_bytes()
        report, kinds = self.run_with_task_counts(original, output)
        self.assertEqual((report.completed, report.skipped), (0, 1))
        self.assertFalse(kinds)
        self.assertEqual((directory / "trajectory.json").read_bytes(), marker_before)
        self.assert_pruned(output)

    def test_earlier_budget_reuse_preserves_a_paused_larger_budget_replay(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 1
        original = self.load_study_config(settings)
        output = self.root / "output"
        run_experiment(original, output)
        settings["runs"][0]["protocol"]["params"]["horizon"] = 9
        longer = self.load_study_config(settings)
        self.assertEqual(run_experiment(longer, output, max_steps=2).paused, 1)
        directory = self.variants(output)[0]
        progress_before = (directory / "progress.json").read_bytes()
        report, kinds = self.run_with_task_counts(original, output)
        self.assertEqual((report.completed, report.skipped), (0, 1))
        self.assertFalse(kinds)
        self.assertEqual((directory / "progress.json").read_bytes(), progress_before)
        self.assertFalse((directory / "trajectory.json").exists())
        self.assertEqual(run_experiment(longer, output).completed, 1)
        direct = self.root / "direct"
        run_experiment(longer, direct)
        self.assert_summaries_equal(longer, direct, output)

    def test_changed_earlier_budget_metric_preserves_larger_budget_derivations(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 1
        output = self.root / "output"
        run_experiment(self.load_study_config(settings), output)
        settings["runs"][0]["protocol"]["params"]["horizon"] = 9
        longer = self.load_study_config(settings)
        run_experiment(longer, output)
        settings["runs"][0]["protocol"]["params"]["horizon"] = 6
        settings["analysis"]["metrics"][0]["type"] = "cumulative_sum"
        changed = self.load_study_config(settings)
        report, kinds = self.run_with_task_counts(changed, output)
        self.assertEqual(report.completed, 1)
        self.assertEqual(kinds, {"metric": 1, "aggregate": 1, "figure": 1})
        report, kinds = self.run_with_task_counts(longer, output)
        self.assertEqual((report.completed, report.skipped), (0, 1))
        self.assertFalse(kinds)
        self.assert_pruned(output)

    def test_interrupted_pruning_and_reset_keep_a_recoverable_durable_boundary(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 1
        output = self.root / "output"
        original = self.load_study_config(settings)
        unlink = Path.unlink

        def interrupt_pruning(path, *args, **kwargs):
            if path.parent.name == "results" and path.suffix == ".npz":
                raise OSError("interrupted trajectory deletion")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", interrupt_pruning):
            with self.assertRaisesRegex(OSError, "interrupted trajectory deletion"):
                run_experiment(original, output)
        directory = self.variants(output)[0]
        self.assertTrue((directory / "trajectory.json").exists())
        self.assertTrue(list((directory / "results").glob("*.npz")))
        self.assertEqual(run_experiment(original, output).skipped, 1)
        self.assert_pruned(output)

        settings["analysis"]["metrics"][0]["type"] = "cumulative_sum"
        changed = self.load_study_config(settings)

        def interrupt_reset(path, *args, **kwargs):
            if path.name == "trajectory.json":
                raise OSError("interrupted trajectory reset")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", interrupt_reset):
            report = run_experiment(changed, output)
        self.assertEqual(report.failed, 1)
        self.assertIn("interrupted trajectory reset", next(iter(report.errors.values())))
        progress = read_json(directory / "progress.json")
        self.assertEqual(
            (progress["status"], progress["step"], progress["checkpoints"]), ("pending", 0, [])
        )
        self.assertTrue((directory / "trajectory.json").exists())
        self.assertEqual(run_experiment(changed, output).completed, 1)
        self.assert_pruned(output)
        direct = self.root / "direct"
        run_experiment(changed, direct)
        self.assert_summaries_equal(changed, direct, output)

    def test_metric_rematerialization_can_pause_and_resume_without_duplicate_records(self) -> None:
        settings = self.pipeline_settings()
        output = self.root / "output"
        run_experiment(self.load_study_config(settings), output)
        settings["analysis"]["metrics"][0]["type"] = "cumulative_sum"
        config = self.load_study_config(settings)
        report = run_experiment(config, output, max_steps=2)
        self.assertEqual((report.paused, report.failed), (2, 0), report.errors)
        for directory in self.variants(output):
            self.assertEqual(read_json(directory / "progress.json")["step"], 2)
            self.assertFalse((directory / "trajectory.json").exists())
        self.assertEqual(run_experiment(config, output, workers=2).completed, 2)
        direct = self.root / "direct"
        self.assertEqual(run_experiment(config, direct).completed, 2)
        self.assert_summaries_equal(config, direct, output)
        self.assert_pruned(output)

    def test_pruned_budget_extension_starts_fresh_and_matches_direct_longer_run(self) -> None:
        settings = self.pipeline_settings()
        output = self.root / "extended"
        run_experiment(self.load_study_config(settings), output)
        old_ids = [directory.name for directory in self.variants(output)]
        settings["runs"][0]["protocol"]["params"]["horizon"] = 9
        config = self.load_study_config(settings)
        initial_steps = []
        initialize = RunSession.restore_or_initialize

        def observe_initial_step(session):
            initialize(session)
            initial_steps.append(session.protocol.step)

        with patch.object(RunSession, "restore_or_initialize", observe_initial_step):
            report = run_experiment(config, output, max_steps=2)
        self.assertEqual((report.paused, report.failed), (2, 0), report.errors)
        self.assertEqual(initial_steps, [0, 0])
        self.assertEqual([directory.name for directory in self.variants(output)], old_ids)
        run_experiment(config, output)
        direct = self.root / "direct"
        run_experiment(config, direct)
        self.assert_summaries_equal(config, direct, output)

    def test_one_and_multiple_workers_produce_identical_pruned_pipeline_outputs(self) -> None:
        settings = self.pipeline_settings()
        settings["runs"][0]["repetitions"] = 3
        config = self.load_study_config(deepcopy(settings))
        sequential, parallel = self.root / "sequential", self.root / "parallel"
        self.assertEqual(run_experiment(config, sequential).completed, 3)
        self.assertEqual(run_experiment(config, parallel, workers=2).completed, 3)
        self.assert_summaries_equal(config, sequential, parallel)
        self.assert_pruned(sequential)
        self.assert_pruned(parallel)
        self.assertEqual(
            sorted(path.read_bytes() for path in (sequential / "analysis/figures").glob("*.tikz")),
            sorted(path.read_bytes() for path in (parallel / "analysis/figures").glob("*.tikz")),
        )
