"""Independent metric, aggregate, and plot cache identities derived from saved artifacts."""

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
from experiments_wo_stress.metrics import FieldMetric
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage import atomic_json
from tests.sample_results import make_saved_result


class AnalysisCacheTests(unittest.TestCase):
    """Verify cache reuse, invalidation, provenance, and recovery of exported copies."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.saved = make_saved_result()
        atomic_json(self.root / "metadata.json", {"schema_version": 1, "run_ids": []})
        # Storage validation is covered separately; each analysis request sees this revision.
        self.loader = mock.patch(
            "experiments_wo_stress.storage.experiment.iter_completed_runs",
            side_effect=lambda _: iter([(self.saved.spec, self.saved)]),
        )
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.settings = {
            "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
            "aggregator": {"group_by": ["algorithm.name"], "uncertainty": "none"},
            "figures": [{"name": "curve", "metric": "reward", "formats": ["tikz"]}],
        }

    def analysis_config(self, settings=None):
        """Expose an analysis-only configuration for the selected cache request."""
        return SimpleNamespace(analysis=self.settings if settings is None else settings)

    def cache_directories(self, kind):
        """List retained cache entries, excluding temporary staging directories."""
        return sorted((self.root / "analysis" / "cache" / kind).glob("[!.]*"))

    def test_metric_reused_for_aggregation_and_plot_changes(self):
        """Changing aggregation or figure settings reuses the computed per-run metric."""
        original = FieldMetric.compute
        with mock.patch.object(
            FieldMetric,
            "compute",
            autospec=True,
            side_effect=lambda metric, saved: original(metric, saved),
        ) as compute:
            first = analyze(self.analysis_config(), self.root)
            analyze(self.analysis_config(), self.root)
            settings = deepcopy(self.settings)
            settings["aggregator"]["uncertainty"] = "std"
            analyze(self.analysis_config(settings), self.root)
            plot(self.analysis_config(settings), self.root)
            settings["figures"][0]["ylabel"] = "Revised display label"
            plot(self.analysis_config(settings), self.root)
            self.assertEqual(compute.call_count, 1)
        self.assertEqual(len(self.cache_directories("metrics")), 1)
        self.assertEqual(len(self.cache_directories("aggregates")), 2)
        self.assertEqual(len(self.cache_directories("plots")), 2)
        np.testing.assert_array_equal(first[0].mean, [1, 2, 3])

    def test_metric_parameters_and_result_revision_invalidate_only_their_outputs(self):
        """Metric parameters and saved-result revisions each create a new metric cache entry."""
        analyze(self.analysis_config(), self.root)
        settings = deepcopy(self.settings)
        settings["metrics"][0]["params"]["field"] = "action"
        summary = analyze(self.analysis_config(settings), self.root)[0]
        np.testing.assert_array_equal(summary.mean, [0, 1, 0])
        self.saved = replace(self.saved, revision="longer-result")
        analyze(self.analysis_config(settings), self.root)
        self.assertEqual(len(self.cache_directories("metrics")), 3)

    def test_point_limit_versions_exports_and_figures_without_recomputing_metrics(self):
        """Changing resolution reuses exact metric curves and refreshes downstream output."""
        first = plot(self.analysis_config(), self.root)[0].read_text()
        settings = deepcopy(self.settings)
        settings["points"] = 2
        with mock.patch.object(FieldMetric, "compute", side_effect=AssertionError("recomputed")):
            reduced = plot(self.analysis_config(settings), self.root)[0].read_text()
            self.assertNotEqual(first, reduced)
            settings["points"] = None
            self.assertEqual(plot(self.analysis_config(settings), self.root)[0].read_text(), first)
        self.assertEqual(len(self.cache_directories("metrics")), 1)
        self.assertEqual(len(self.cache_directories("aggregates")), 3)
        self.assertEqual(len(self.cache_directories("plots")), 3)

    def test_corrupt_metric_cache_is_reported(self):
        """Damaged cached numerical results raise an explicit corruption error."""
        analyze(self.analysis_config(), self.root)
        (self.cache_directories("metrics")[0] / "result.npz").write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError, "Corrupt analysis cache"):
            analyze(self.analysis_config(), self.root)

    def test_cached_exports_restore_modified_convenience_outputs(self):
        """A valid cached figure repairs a changed export without invoking the renderer."""
        path = plot(self.analysis_config(), self.root)[0]
        expected = path.read_bytes()
        path.write_text("changed export")
        with mock.patch(
            "experiments_wo_stress.analysis.figures._tikz", side_effect=AssertionError("rendered")
        ):
            self.assertEqual(plot(self.analysis_config(), self.root)[0].read_bytes(), expected)

    def test_metric_source_and_declared_dependency_versions_are_separate(self):
        """Edits to a custom metric or its tracked input file independently invalidate cache."""
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
        np.testing.assert_array_equal(
            analyze(self.analysis_config(settings), self.root)[0].mean, [2, 4, 6]
        )
        dependency.write_text("3")
        np.testing.assert_array_equal(
            analyze(self.analysis_config(settings), self.root)[0].mean, [3, 6, 9]
        )
        # Fresh CLI invocations import the edited code. Remove the module here to
        # model that lifecycle without starting a subprocess for this unit check.
        source.write_text(source.read_text().replace("* scale)", "* scale + 10)"))
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
        np.testing.assert_array_equal(
            analyze(self.analysis_config(settings), self.root)[0].mean, [13, 16, 19]
        )
        self.assertEqual(len(self.cache_directories("metrics")), 3)

    def test_cache_manifest_tracks_upstream_metric_identity(self):
        """The metric manifest records the source result revision and saved output files."""
        analyze(self.analysis_config(), self.root)
        manifest = json.loads((self.cache_directories("metrics")[0] / "cache.json").read_text())
        self.assertEqual(manifest["identity"]["result_revision"], "first")
        self.assertEqual(set(manifest["files"]), {"result.npz"})

    def test_bandit_metric_alias_tracks_supplied_implementation(self):
        """The regret alias fingerprints the implementation in builtins for cache provenance."""
        settings = deepcopy(self.settings)
        settings["metrics"][0] = {"name": "regret", "type": "pseudo_regret"}
        np.testing.assert_allclose(
            analyze(self.analysis_config(settings), self.root)[0].mean, [0.6, 0.6, 1.2]
        )
        manifest = json.loads((self.cache_directories("metrics")[0] / "cache.json").read_text())
        sources = manifest["identity"]["metric"]["implementation"]["sources"]
        self.assertIn("experiments_wo_stress.builtins.metrics", sources)
        settings["points"] = 2
        reduced = analyze(self.analysis_config(settings), self.root)[0]
        np.testing.assert_array_equal(reduced.x, [1, 3])
        np.testing.assert_allclose(reduced.mean, [0.6, 1.2])
