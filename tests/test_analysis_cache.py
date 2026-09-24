"""Saved-instance metrics, explicit capabilities, and independent analysis revisions."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments_wo_stress.analysis import analyze
from experiments_wo_stress.artifacts import Instance, RunResult
from experiments_wo_stress.jobs import ComponentSpec, RunSpec
from experiments_wo_stress.metrics import (
    CumulativeSumMetric,
    FieldMetric,
    PseudoRegretMetric,
    RealizedRegretMetric,
    validate_requirements,
)
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage import atomic_json


def result(*, means=None, records=None, completed_steps=3, revision="first"):
    spec = RunSpec(
        "a" * 20,
        "study",
        0,
        "algorithm",
        ComponentSpec("missing_sim:Algorithm"),
        ComponentSpec("missing_sim:Data"),
        ComponentSpec("missing_sim:Protocol"),
        12,
    )
    return RunResult(
        records=records
        if records is not None
        else {
            "step": np.array([1, 2, 3]),
            "action": np.array([0, 1, 0]),
            "reward": np.array([1.0, 2.0, 3.0]),
        },
        instance=Instance(arrays={"means": np.array([0.2, 0.8]) if means is None else means}),
        spec=spec,
        completed_steps=completed_steps,
        revision=revision,
    )


class SavedInstanceMetricTests(unittest.TestCase):
    def test_stationary_pseudo_regret_uses_means_not_realized_rewards(self):
        saved = result()
        computed = PseudoRegretMetric().compute(saved)
        np.testing.assert_allclose(computed.values, [0.6, 0.6, 1.2])
        changed = replace(saved, records={**saved.records, "reward": np.array([-100, 50, 8])})
        np.testing.assert_array_equal(PseudoRegretMetric().compute(changed).values, computed.values)

    def test_nonstationary_dynamic_and_fixed_benchmarks(self):
        saved = result(
            means=np.array([[1, 0], [0, 2], [3, 1]]),
            records={
                "step": np.array([1, 2, 3]),
                "action": np.array([0, 0, 1]),
            },
        )
        np.testing.assert_allclose(PseudoRegretMetric().compute(saved).values, [0, 2, 4])
        np.testing.assert_allclose(
            PseudoRegretMetric(comparator="best_fixed").compute(saved).values, [0, 1, 2]
        )

    def test_small_gaps_are_accumulated_before_large_reward_sums_cancel(self):
        steps = 10000
        means = np.array([1e12, 1e12 + 0.01])
        saved = result(
            means=means,
            completed_steps=steps,
            records={
                "step": np.arange(1, steps + 1),
                "action": np.zeros(steps, dtype=int),
            },
        )
        expected = np.cumsum(np.full(steps, means[1] - means[0]))
        for comparator in ("dynamic", "best_fixed"):
            np.testing.assert_array_equal(
                PseudoRegretMetric(comparator=comparator).compute(saved).values, expected
            )

    def test_sparse_truncated_and_missing_measurements_fail_explicitly(self):
        for steps, completed in (([1, 3], 3), ([1, 2], 3), ([2, 3], 3)):
            saved = result(
                records={"step": np.array(steps), "reward": np.ones(len(steps))},
                completed_steps=completed,
            )
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "complete trajectory"):
                    CumulativeSumMetric("reward").compute(saved)
        with self.assertRaisesRegex(ValueError, "missing recorded fields: absent"):
            FieldMetric("absent").compute(result())
        with self.assertRaisesRegex(ValueError, "missing instance arrays: means"):
            PseudoRegretMetric().compute(replace(result(), instance=Instance()))

    def test_realized_regret_requires_consistent_counterfactual_rewards(self):
        with self.assertRaisesRegex(ValueError, "counterfactual_rewards"):
            RealizedRegretMetric().compute(result())
        matrix = np.array([[1, 0], [0, 2], [3, 1]])
        saved = replace(
            result(),
            instance=Instance(arrays={"counterfactual_rewards": matrix}),
            records={
                "step": np.array([1, 2, 3]),
                "action": np.array([0, 0, 1]),
                "reward": np.array([1, 0, 1]),
            },
        )
        np.testing.assert_allclose(RealizedRegretMetric().compute(saved).values, [0, 1, 2])
        inconsistent = replace(saved, records={**saved.records, "reward": np.array([9, 0, 1])})
        with self.assertRaisesRegex(ValueError, "disagree"):
            RealizedRegretMetric().compute(inconsistent)

    def test_capability_declarations_apply_to_custom_metrics(self):
        metric = SimpleNamespace(required_fields=("reward",), required_instance_fields=("context",))
        with self.assertRaisesRegex(ValueError, "instance arrays: context"):
            validate_requirements(metric, result())


class AnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.saved = result()
        atomic_json(self.root / "metadata.json", {"schema_version": 1, "run_ids": []})
        self.loader = mock.patch(
            "experiments_wo_stress.storage.iter_completed_runs",
            side_effect=lambda _: iter([(self.saved.spec, self.saved)]),
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.settings = {
            "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
            "aggregator": {"group_by": ["algorithm.name"], "uncertainty": "none"},
            "figures": [{"name": "curve", "metric": "reward", "formats": ["tikz"]}],
        }

    def config(self, settings=None):
        return SimpleNamespace(analysis=self.settings if settings is None else settings)

    def cache_directories(self, kind):
        return sorted((self.root / "analysis" / "cache" / kind).glob("[!.]*"))

    def test_metric_reused_for_aggregation_and_plot_changes(self):
        original = FieldMetric.compute
        with mock.patch.object(
            FieldMetric,
            "compute",
            autospec=True,
            side_effect=lambda metric, saved: original(metric, saved),
        ) as compute:
            first = analyze(self.config(), self.root)
            analyze(self.config(), self.root)
            settings = deepcopy(self.settings)
            settings["aggregator"]["uncertainty"] = "std"
            analyze(self.config(settings), self.root)
            plot(self.config(settings), self.root)
            settings["figures"][0]["ylabel"] = "Revised display label"
            plot(self.config(settings), self.root)
            self.assertEqual(compute.call_count, 1)
        self.assertEqual(len(self.cache_directories("metrics")), 1)
        self.assertEqual(len(self.cache_directories("aggregates")), 2)
        self.assertEqual(len(self.cache_directories("plots")), 2)
        np.testing.assert_array_equal(first[0].mean, [1, 2, 3])

    def test_metric_parameters_and_result_revision_invalidate_only_their_outputs(self):
        analyze(self.config(), self.root)
        settings = deepcopy(self.settings)
        settings["metrics"][0]["params"]["field"] = "action"
        summary = analyze(self.config(settings), self.root)[0]
        np.testing.assert_array_equal(summary.mean, [0, 1, 0])
        self.saved = replace(self.saved, revision="longer-result")
        analyze(self.config(settings), self.root)
        self.assertEqual(len(self.cache_directories("metrics")), 3)

    def test_corrupt_metric_cache_is_reported(self):
        analyze(self.config(), self.root)
        (self.cache_directories("metrics")[0] / "result.npz").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "Corrupt analysis cache"):
            analyze(self.config(), self.root)

    def test_cached_exports_restore_modified_convenience_outputs(self):
        path = plot(self.config(), self.root)[0]
        expected = path.read_bytes()
        path.write_text("changed export")
        with mock.patch(
            "experiments_wo_stress.plotting._tikz", side_effect=AssertionError("rendered")
        ):
            self.assertEqual(plot(self.config(), self.root)[0].read_bytes(), expected)

    def test_metric_source_and_declared_dependency_versions_are_separate(self):
        module_name = "temporary_analysis_metric"
        source = self.root / f"{module_name}.py"
        dependency = self.root / "scale.txt"
        dependency.write_text("2")
        source.write_text(
            "from pathlib import Path\n"
            "from experiments_wo_stress.metrics import MetricResult\n"
            "class Scaled:\n"
            "    dependency_files = ('scale.txt',)\n"
            "    def compute(self, result):\n"
            "        scale = float(Path(__file__).with_name('scale.txt').read_text())\n"
            "        return MetricResult(result['step'], result['reward'] * scale)\n"
        )
        sys.path.insert(0, str(self.root))
        self.addCleanup(sys.path.remove, str(self.root))
        self.addCleanup(sys.modules.pop, module_name, None)
        settings = deepcopy(self.settings)
        settings["metrics"][0] = {"name": "scaled", "type": f"{module_name}:Scaled"}
        np.testing.assert_array_equal(analyze(self.config(settings), self.root)[0].mean, [2, 4, 6])
        dependency.write_text("3")
        np.testing.assert_array_equal(analyze(self.config(settings), self.root)[0].mean, [3, 6, 9])
        # Fresh CLI invocations import the edited code. Remove the module here to
        # model that lifecycle without starting a subprocess for this unit check.
        source.write_text(source.read_text().replace("* scale)", "* scale + 10)"))
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
        np.testing.assert_array_equal(
            analyze(self.config(settings), self.root)[0].mean, [13, 16, 19]
        )
        self.assertEqual(len(self.cache_directories("metrics")), 3)

    def test_cache_manifest_tracks_upstream_metric_identity(self):
        analyze(self.config(), self.root)
        manifest = json.loads((self.cache_directories("metrics")[0] / "cache.json").read_text())
        self.assertEqual(manifest["identity"]["result_revision"], "first")
        self.assertEqual(set(manifest["files"]), {"result.npz"})


if __name__ == "__main__":
    unittest.main()
