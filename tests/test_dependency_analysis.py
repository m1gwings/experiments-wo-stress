"""Nearest retained ancestors, partial-group readiness, and durable analysis dependencies."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from experiments_wo_stress.analysis import analyze
from experiments_wo_stress.analysis import graph as analysis_graph
from experiments_wo_stress.analysis.cache import source_identity
from experiments_wo_stress.analysis.graph import AnalysisGraph, execute_derivation
from experiments_wo_stress.analysis.metrics import FieldMetric
from experiments_wo_stress.execution.provenance import collect_provenance
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage.experiment import save_instance
from experiments_wo_stress.storage.files import atomic_json
from experiments_wo_stress.storage.models import Instance
from experiments_wo_stress.storage.run import Recorder, RunStore
from experiments_wo_stress.storage.trajectories import prune_trajectory, result_reference
from experiments_wo_stress.study.config import ExperimentConfig
from experiments_wo_stress.study.planning import plan_runs


def _dispatch_without_uncertainty(task):
    """Model an output-affecting graph implementation edit without changing its YAML."""
    if task.kind == "aggregate":
        task = replace(task, payload={**task.payload, "uncertainty": "none"})
    execute_derivation(task)


class DependencyAnalysisTests(unittest.TestCase):
    """Exercise saved-only derivations without importing any simulation components."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = ExperimentConfig(
            name="dependencies",
            seed=12,
            runs=[
                {
                    "name": "study",
                    "planner": "grid",
                    "repetitions": 2,
                    "budget": {"steps": 3},
                    "protocol": {"type": "unavailable:Protocol"},
                    "data": {"type": "unavailable:Data", "params": {"size": 2}},
                    "algorithms": [{"name": "algorithm", "type": "unavailable:Algorithm"}],
                }
            ],
            analysis={
                "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
                "aggregator": {"group_by": ["data.params.size"], "uncertainty": "standard_error"},
                "figures": [{"name": "curve", "metric": "reward", "formats": ["tikz"]}],
            },
        )
        self.specs = plan_runs(self.config)
        self.publish_request()

    def publish_request(self):
        """Publish an active schema-2 request independently of the simulation executor."""
        self.locations = {spec.run_id: spec.run_id for spec in self.specs}
        atomic_json(
            self.root / "metadata.json",
            {
                "schema_version": 2,
                "name": self.config.name,
                "run_ids": list(self.locations),
                "requests": {spec.run_id: spec.to_dict() for spec in self.specs},
            },
        )

    def save_run(self, spec):
        """Materialize a valid completed trajectory and its immutable instance."""
        directory = self.root / "runs" / spec.run_id
        instance_id = save_instance(self.root, Instance())
        atomic_json(
            directory / "metadata.json",
            {
                "spec": spec.to_dict(),
                "identity": {"run_id": spec.run_id},
                "instance_id": instance_id,
            },
        )
        store = RunStore(directory)
        recorder = Recorder(store, {"buffer_bytes": 1024}, {"chunks": [], "schema": {}})
        for step in range(1, 4):
            recorder.record(
                step, {"reward": float(step * (spec.repetition + 1)), "action": step % 2}
            )
        recorder.flush()
        store.finish(recorder.manifest, 3)

    def graph(self, config=None, **kwargs):
        """Construct the same request with optional derived settings or target selection."""
        return AnalysisGraph(config or self.config, self.root, self.specs, self.locations, **kwargs)

    def materialize(self):
        """Finish every requested derivation and return its graph."""
        for spec in self.specs:
            self.save_run(spec)
        graph = self.graph()
        self.drain(graph)
        return graph

    def drain(self, graph):
        """Execute ready tasks deterministically while checking every publication."""
        while not graph.complete:
            tasks = graph.ready_tasks()
            self.assertTrue(tasks, graph.required_simulations)
            for task in tasks:
                execute_derivation(task)
                graph.finish(task)
        return graph.finalize()

    def prune(self, graph):
        """Use all configured metric proofs as the sole authority to remove trajectories."""
        for spec in self.specs:
            proofs = graph.metric_proofs(spec.run_id)
            self.assertIsNotNone(proofs)
            prune_trajectory(self.root, spec.run_id, spec, proofs)
            self.assertEqual(result_reference(self.root, spec.run_id, spec)["trajectory"], "pruned")
            self.assertFalse(list((self.root / "runs" / spec.run_id / "results").glob("*.npz")))

    def changed(self, section, key, value):
        """Replace one derived setting without altering the requested run plan."""
        settings = deepcopy(self.config.analysis)
        target = settings[section] if section == "aggregator" else settings[section][0]
        target[key] = value
        return replace(self.config, analysis=settings)

    def test_exact_reuse_after_pruning_needs_no_raw_or_metric_reads(self):
        graph = self.materialize()
        expected = graph.finalize()[0][0].mean.copy()
        self.prune(graph)
        with (
            mock.patch(
                "experiments_wo_stress.storage.trajectories.load_run_result",
                side_effect=AssertionError("raw ancestor opened"),
            ),
            mock.patch.object(
                FieldMetric, "compute", side_effect=AssertionError("metric recomputed")
            ),
        ):
            fresh = self.graph()
            self.assertTrue(fresh.complete)
            self.assertEqual(fresh.ready_tasks(), [])
            self.assertEqual(fresh.required_simulations, set())
            np.testing.assert_array_equal(analyze(self.config, self.root)[0].mean, expected)
            self.assertTrue(plot(self.config, self.root)[0].is_file())

    def test_figure_change_stops_at_valid_aggregate_even_without_metric_files(self):
        graph = self.materialize()
        self.prune(graph)
        shutil.rmtree(self.root / "analysis" / "cache" / "metrics")
        config = self.changed("figures", "ylabel", "Changed presentation")
        fresh = self.graph(config)
        self.assertEqual([task.kind for task in fresh.ready_tasks()], ["figure"])
        self.assertEqual(fresh.required_simulations, set())
        self.drain(fresh)
        self.assertIn("Changed presentation", plot(config, self.root)[0].read_text())

    def test_retained_figure_does_not_require_deleted_aggregate_ancestors(self):
        graph = self.materialize()
        self.prune(graph)
        shutil.rmtree(self.root / "analysis" / "cache" / "metrics")
        shutil.rmtree(self.root / "analysis" / "cache" / "aggregates")
        fresh = self.graph(figures_only=True)
        self.assertTrue(fresh.complete)
        self.assertEqual(fresh.task_counts["pending"], 0)
        self.assertEqual(fresh.required_simulations, set())
        self.assertTrue(plot(self.config, self.root)[0].is_file())

    def test_aggregate_change_uses_retained_full_resolution_metrics(self):
        graph = self.materialize()
        self.prune(graph)
        config = self.changed("aggregator", "uncertainty", "std")
        fresh = self.graph(config)
        self.assertEqual([task.kind for task in fresh.ready_tasks()], ["aggregate"])
        self.assertEqual(fresh.required_simulations, set())
        summaries, _ = self.drain(fresh)
        np.testing.assert_allclose(summaries[0].uncertainty, np.arange(1, 4) / np.sqrt(2))

    def test_metric_change_requires_only_missing_raw_ancestors(self):
        graph = self.materialize()
        first = self.specs[0]
        prune_trajectory(self.root, first.run_id, first, graph.metric_proofs(first.run_id))
        config = self.changed("metrics", "params", {"field": "action"})
        fresh = self.graph(config)
        self.assertEqual(fresh.required_simulations, {first.run_id})
        self.assertEqual([task.run_ids for task in fresh.ready_tasks()], [(self.specs[1].run_id,)])
        for function in (analyze, plot):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "ews run.*never launch simulations"):
                    function(config, self.root)

    def test_metric_change_reuses_retained_raw_data(self):
        self.materialize()
        config = self.changed("metrics", "params", {"field": "action"})
        fresh = self.graph(config)
        self.assertEqual(fresh.required_simulations, set())
        self.assertEqual([task.kind for task in fresh.ready_tasks()], ["metric", "metric"])
        summaries, _ = self.drain(fresh)
        np.testing.assert_array_equal(summaries[0].mean, [1, 0, 1])

    def test_pruning_proof_waits_for_every_configured_metric(self):
        settings = deepcopy(self.config.analysis)
        settings["metrics"].append(
            {"name": "action", "type": "field", "params": {"field": "action"}}
        )
        for spec in self.specs:
            self.save_run(spec)
        graph = self.graph(replace(self.config, analysis=settings))
        first = self.specs[0].run_id
        tasks = [task for task in graph.ready_tasks() if task.run_ids == (first,)]
        self.assertIsNone(graph.metric_proofs(first))
        execute_derivation(tasks[0])
        graph.finish(tasks[0])
        self.assertIsNone(graph.metric_proofs(first))
        execute_derivation(tasks[1])
        graph.finish(tasks[1])
        self.assertEqual(set(graph.metric_proofs(first)), {"action", "reward"})

    def test_one_aggregate_becomes_ready_before_unrelated_simulations(self):
        runs = deepcopy(self.config.runs)
        runs[0]["grid"] = {"data.params.size": [2, 4]}
        self.config = replace(self.config, runs=runs)
        self.specs = plan_runs(self.config)
        self.publish_request()
        first_group = [spec for spec in self.specs if spec.data.params["size"] == 2]
        for spec in first_group:
            self.save_run(spec)
        graph = self.graph()
        self.assertEqual(graph.task_counts["pending"], 7)
        self.assertEqual(len(graph.required_simulations), 2)
        self.assertEqual(
            graph.analysis_unit(first_group[0].run_id), graph.analysis_unit(first_group[1].run_id)
        )
        for task in graph.ready_tasks():
            execute_derivation(task)
            graph.finish(task)
        tasks = graph.ready_tasks()
        self.assertEqual([task.kind for task in tasks], ["aggregate"])
        execute_derivation(tasks[0])
        graph.finish(tasks[0])
        self.assertEqual(graph.ready_tasks(), [])
        self.assertFalse(graph.complete)

    def test_canonical_reduction_does_not_depend_on_completion_order(self):
        graph = self.materialize()
        expected = graph.finalize()[0][0]
        shutil.rmtree(self.root / "analysis")
        fresh = AnalysisGraph(self.config, self.root, list(reversed(self.specs)), self.locations)
        for task in reversed(fresh.ready_tasks()):
            execute_derivation(task)
            fresh.finish(task)
        actual = self.drain(fresh)[0][0]
        np.testing.assert_array_equal(actual.mean, expected.mean)
        np.testing.assert_array_equal(actual.uncertainty, expected.uncertainty)

    def test_building_large_pending_graph_publishes_no_per_run_files(self):
        runs = deepcopy(self.config.runs)
        runs[0]["repetitions"] = 2000
        config = replace(self.config, runs=runs)
        specs = plan_runs(config)
        with mock.patch(
            "experiments_wo_stress.analysis.graph.source_identity", wraps=source_identity
        ) as identify:
            graph = AnalysisGraph(
                config, self.root, specs, {spec.run_id: spec.run_id for spec in specs}
            )
        self.assertEqual(len(graph.required_simulations), 2000)
        self.assertEqual(identify.call_count, 1)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "analysis").exists())

    def test_completed_run_updates_only_its_dependency_chain(self):
        """A large pending group does not repeatedly rebuild every metric identity."""
        runs = deepcopy(self.config.runs)
        runs[0]["repetitions"] = 2000
        config = replace(self.config, runs=runs)
        specs = plan_runs(config)
        graph = AnalysisGraph(
            config, self.root, specs, {spec.run_id: spec.run_id for spec in specs}
        )
        self.save_run(specs[0])
        with mock.patch.object(graph, "_node_identity", wraps=graph._node_identity) as identify:
            graph.refresh(specs[0].run_id)
            self.assertEqual(identify.call_count, 1)
        tasks = graph.ready_tasks()
        self.assertEqual(len(tasks), 1)
        with mock.patch.object(graph, "_node_identity", wraps=graph._node_identity) as identify:
            execute_derivation(tasks[0])
            graph.finish(tasks[0])
            self.assertEqual(identify.call_count, 0)
        self.assertEqual(len(graph.required_simulations), 1999)

    def test_incremental_completions_release_metric_aggregate_and_figure(self):
        """Newly published raw receipts advance the same graph through all task kinds."""
        graph = self.graph()
        self.assertEqual(graph.task_counts["pending"], 4)
        observed = []
        for spec in reversed(self.specs):
            self.save_run(spec)
            graph.refresh(spec.run_id)
            while tasks := graph.ready_tasks():
                task = tasks[0]
                observed.append(task.kind)
                execute_derivation(task)
                graph.finish(task)
        self.assertEqual(observed, ["metric", "metric", "aggregate", "figure"])
        self.assertTrue(graph.complete)
        self.assertEqual(graph.task_counts["pending"], 0)
        self.assertTrue((self.root / "analysis" / "figures" / "curve.tikz").is_file())

    def test_changed_output_dispatch_versions_artifacts_without_changing_simulations(self):
        """Changed graph transformations cannot silently reuse previous numerical outputs."""
        original = self.materialize()
        original_keys = {node_id: node.key for node_id, node in original.nodes.items()}
        self.assertTrue(np.all(original.finalize()[0][0].uncertainty > 0))
        references = dict(original.references)
        simulation_implementation = collect_provenance(self.config, {})["implementation"]
        with (
            mock.patch.dict(
                analysis_graph._DEPENDENCY_BOUNDARIES,
                {"execute_derivation": _dispatch_without_uncertainty},
            ),
            mock.patch.object(analysis_graph, "execute_derivation", _dispatch_without_uncertainty),
        ):
            changed = self.graph()
            self.assertFalse(changed.complete)
            self.assertEqual(changed.required_simulations, set())
            self.assertEqual(changed.references, references)
            self.assertTrue(
                all(node.key != original_keys[node_id] for node_id, node in changed.nodes.items())
            )
            while not changed.complete:
                for task in changed.ready_tasks():
                    analysis_graph.execute_derivation(task)
                    changed.finish(task)
            np.testing.assert_array_equal(changed.finalize()[0][0].uncertainty, [0, 0, 0])
            self.assertEqual(
                collect_provenance(self.config, {})["implementation"], simulation_implementation
            )
        self.assertTrue(
            self.graph().complete, "the original retained implementation remains reusable"
        )

    def test_reporting_only_changes_preserve_pruned_artifact_reuse(self):
        """Progress counters cannot change metric, aggregate, or figure identities."""
        original = self.materialize()
        original_keys = {node_id: node.key for node_id, node in original.nodes.items()}
        self.prune(original)
        with mock.patch.object(AnalysisGraph, "task_counts", property(lambda _: {"observed": 42})):
            changed = self.graph()
            self.assertEqual(changed.task_counts, {"observed": 42})
            self.assertTrue(changed.complete)
            self.assertEqual(changed.ready_tasks(), [])
            self.assertEqual(changed.required_simulations, set())
            self.assertEqual(
                {node_id: node.key for node_id, node in changed.nodes.items()}, original_keys
            )
