"""Statistical checks and figure regeneration from persisted numerical artifacts."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments_wo_stress.analysis import analyze
from experiments_wo_stress.jobs import ComponentSpec, RunSpec
from experiments_wo_stress.metrics import CumulativeSumMetric, MetricResult
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage import Recorder, RunStore, atomic_json


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_ids: list[str] = []
        # These modules deliberately do not exist: analysis must not import them.
        self.base = RunSpec(
            "first",
            "study",
            0,
            "algorithm_50%",
            ComponentSpec("unavailable_paper:Algorithm", {"rate": 0.5}),
            ComponentSpec("unavailable_paper:Generator", {"size": 10}),
            ComponentSpec("unavailable_paper:Protocol", {"horizon": 3}),
            1234,
        )
        self.settings = {
            "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
            "aggregator": {
                "group_by": ["algorithm.name", "data.params.size"],
                "uncertainty": "standard_error",
            },
            "figures": [
                {
                    "type": "line",
                    "name": "reward",
                    "metric": "reward",
                    "x": "step",
                    "color": "algorithm.name",
                    "panel": "data.params.size",
                    "formats": ["tikz"],
                }
            ],
        }

    def config(self, settings=None):
        return SimpleNamespace(analysis=settings or self.settings)

    def save_run(self, spec: RunSpec, values, steps=(1, 2, 3), completed=True) -> None:
        directory = self.root / "runs" / spec.run_id
        store = RunStore(directory)
        recorder = Recorder(store, {"buffer_bytes": 32}, {"chunks": [], "schema": {}})
        for step, value in zip(steps, values):
            recorder.record(step, {"reward": float(value)})
        recorder.flush()
        atomic_json(directory / "metadata.json", {"spec": spec.to_dict()})
        if completed:
            store.finish(recorder.manifest, steps[-1])
        else:
            atomic_json(directory / "progress.json", store.progress)
        self.run_ids.append(spec.run_id)
        atomic_json(self.root / "metadata.json", {"schema_version": 1, "run_ids": self.run_ids})

    def test_mean_and_sample_standard_error_across_independent_runs(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        self.save_run(replace(self.base, run_id="second", repetition=1), [3, 6, 9])
        self.save_run(
            replace(self.base, run_id="pending", repetition=2), [100, 100, 100], completed=False
        )
        summaries = analyze(self.config(), self.root)
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.count, 2)
        np.testing.assert_array_equal(summary.x, [1, 2, 3])
        np.testing.assert_allclose(summary.mean, [2, 4, 6])
        np.testing.assert_allclose(summary.uncertainty, [1, 2, 3])
        with np.load(self.root / "analysis" / "reward.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["group_0_mean"], summary.mean)
        manifest = json.loads((self.root / "analysis" / "metadata.json").read_text())
        self.assertEqual(manifest["metrics"]["reward"][0]["count"], 2)

    def test_single_repetition_has_undefined_sample_uncertainty(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        summary = analyze(self.config(), self.root)[0]
        self.assertTrue(np.isnan(summary.uncertainty).all())
        settings = deepcopy(self.settings)
        settings["aggregator"]["uncertainty"] = "none"
        np.testing.assert_array_equal(analyze(self.config(settings), self.root)[0].uncertainty, 0)

    def test_rejects_pooling_different_parameters(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        other = replace(
            self.base,
            run_id="other",
            repetition=1,
            algorithm=ComponentSpec(self.base.algorithm.type, {"rate": 0.9}),
        )
        self.save_run(other, [3, 4, 5])
        with self.assertRaisesRegex(ValueError, "different scientific configurations"):
            analyze(self.config(), self.root)
        settings = deepcopy(self.settings)
        settings["aggregator"]["group_by"].append("algorithm.params.rate")
        self.assertEqual(len(analyze(self.config(settings), self.root)), 2)

    def test_rejects_duplicate_repetitions_and_misaligned_curves(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        self.save_run(replace(self.base, run_id="second"), [3, 4, 5])
        with self.assertRaisesRegex(ValueError, "Duplicate repetition"):
            analyze(self.config(), self.root)
        metadata = self.root / "runs" / "second" / "metadata.json"
        atomic_json(metadata, {"spec": replace(self.base, run_id="second", repetition=1).to_dict()})
        # Change the x axis using a custom metric, preserving valid storage itself.
        settings = deepcopy(self.settings)
        settings["metrics"][0]["params"]["x"] = "reward"
        with self.assertRaisesRegex(ValueError, "mismatched x"):
            analyze(self.config(settings), self.root)

    def test_tikz_regeneration_needs_no_simulation_modules(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        self.save_run(replace(self.base, run_id="second", repetition=1), [3, 4, 5])
        paths = plot(self.config(), self.root)
        self.assertEqual([path.suffix for path in paths], [".tikz"])
        source = paths[0].read_text(encoding="utf-8")
        self.assertIn(r"\documentclass", source)
        self.assertIn(r"algorithm\_50\%", source)
        self.assertIn("fill between", source)
        self.assertIn("standard error", source)
        self.assertNotIn("nan", source)
        first_source = source
        self.assertEqual(plot(self.config(), self.root)[0].read_text(), first_source)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is optional")
    def test_pdf_and_jpg_regeneration(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        settings = deepcopy(self.settings)
        settings["figures"][0]["formats"] = ["pdf", "jpg"]
        paths = plot(self.config(settings), self.root)
        self.assertEqual({path.suffix for path in paths}, {".pdf", ".jpg"})
        for path in paths:
            self.assertGreater(path.stat().st_size, 100)
        self.assertTrue(paths[0].read_bytes().startswith(b"%PDF"))
        self.assertTrue(paths[1].read_bytes().startswith(b"\xff\xd8"))

    def test_log_axes_reject_nonpositive_coordinates(self) -> None:
        self.save_run(self.base, [0, 1, 2])
        settings = deepcopy(self.settings)
        settings["figures"][0]["yscale"] = "log"
        with self.assertRaisesRegex(ValueError, "positive metric means"):
            plot(self.config(settings), self.root)

    def test_invalid_metric_name_cannot_escape_output_directory(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        settings = deepcopy(self.settings)
        settings["metrics"][0]["name"] = "../../escaped"
        with self.assertRaisesRegex(ValueError, "Metric name"):
            analyze(self.config(settings), self.root)

    def test_rejects_unknown_analysis_options(self) -> None:
        self.save_run(self.base, [1, 2, 3])
        for section, key in (
            ("metrics", "param"),
            ("aggregator", "uncertanty"),
            ("figures", "formts"),
        ):
            with self.subTest(section=section):
                settings = deepcopy(self.settings)
                target = settings[section] if section == "aggregator" else settings[section][0]
                target[key] = "typo"
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    plot(self.config(settings), self.root)

    def test_scalar_figure_uses_visible_marker_and_error_bar(self) -> None:
        self.save_run(self.base, [1], steps=(1,))
        self.save_run(replace(self.base, run_id="second", repetition=1), [3], steps=(1,))
        source = plot(self.config(), self.root)[0].read_text()
        self.assertIn("only marks,mark=*", source)
        self.assertIn("error bars/.cd,y dir=both,y explicit", source)
        self.assertIn("(1,2) +- (0,1)", source)


class MetricTests(unittest.TestCase):
    def test_scalar_and_cumulative_metrics(self) -> None:
        scalar = MetricResult(np.asarray(1), np.asarray(4.5))
        self.assertEqual(scalar.values.shape, (1,))
        result = CumulativeSumMetric("increment").compute(
            {
                "step": np.array([1, 2, 3]),
                "increment": np.array([1.0, 0.5, 2.0]),
            }
        )
        np.testing.assert_allclose(result.values, [1, 1.5, 3.5])

    def test_rejects_nonnumerical_misaligned_and_nonfinite_results(self) -> None:
        with self.assertRaises(TypeError):
            MetricResult(np.array([1]), np.array(["bad"]))
        with self.assertRaises(ValueError):
            MetricResult(np.array([1, 2]), np.array([1]))
        with self.assertRaises(ValueError):
            MetricResult(np.array([1]), np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
