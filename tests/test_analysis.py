"""Statistical aggregation and figure export from saved results with no simulation imports."""

from __future__ import annotations

import csv
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
from experiments_wo_stress.analysis.pipeline import Summary, _subsample_summary
from experiments_wo_stress.jobs import ComponentSpec, RunSpec
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage import Recorder, RunStore, atomic_json


class _PersistedAnalysisCase(unittest.TestCase):
    """Write completed or partial numerical artifacts without running scientific components."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_ids: list[str] = []
        # These modules deliberately do not exist: analysis must not import them.
        self.base_spec = RunSpec(
            "aaaaaaaaaaaaaaaaaaaa",
            "study",
            0,
            "algorithm_50%",
            ComponentSpec("unavailable_components:Algorithm", {"rate": 0.5}),
            ComponentSpec("unavailable_components:Generator", {"size": 10}),
            ComponentSpec("unavailable_components:Protocol", {"horizon": 3}),
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

    def analysis_config(self, settings=None):
        """Expose only the analysis settings consumed by analyze() and plot()."""
        return SimpleNamespace(analysis=settings or self.settings)

    def save_run(self, spec: RunSpec, values, steps=(1, 2, 3), completed=True) -> None:
        """Persist reward records and mark the run complete only when requested."""
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


class AggregationTests(_PersistedAnalysisCase):
    """Combine independent repetitions while rejecting incompatible scientific curves."""

    def test_mean_and_sample_standard_error_across_independent_runs(self) -> None:
        """Only completed repetitions contribute to the saved mean and sample standard error."""
        self.save_run(self.base_spec, [1, 2, 3])
        self.save_run(
            replace(self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb", repetition=1), [3, 6, 9]
        )
        self.save_run(
            replace(self.base_spec, run_id="cccccccccccccccccccc", repetition=2),
            [100, 100, 100],
            completed=False,
        )
        summaries = analyze(self.analysis_config(), self.root)
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.count, 2)
        np.testing.assert_array_equal(summary.x, [1, 2, 3])
        np.testing.assert_allclose(summary.mean, [2, 4, 6])
        # With two repetitions, sample standard error is half their pointwise difference.
        np.testing.assert_allclose(summary.uncertainty, [1, 2, 3])
        with np.load(self.root / "analysis" / "reward.npz", allow_pickle=False) as saved:
            np.testing.assert_allclose(saved["group_0_mean"], summary.mean)
        manifest = json.loads((self.root / "analysis" / "metadata.json").read_text())
        self.assertEqual(manifest["metrics"]["reward"][0]["count"], 2)

    def test_single_repetition_has_undefined_sample_uncertainty(self) -> None:
        """One repetition has undefined sample uncertainty unless uncertainty is disabled."""
        self.save_run(self.base_spec, [1, 2, 3])
        summary = analyze(self.analysis_config(), self.root)[0]
        self.assertTrue(np.isnan(summary.uncertainty).all())
        settings = deepcopy(self.settings)
        settings["aggregator"]["uncertainty"] = "none"
        np.testing.assert_array_equal(
            analyze(self.analysis_config(settings), self.root)[0].uncertainty, 0
        )

    def test_rejects_pooling_different_parameters(self) -> None:
        """Different scientific parameters require separate aggregation groups."""
        self.save_run(self.base_spec, [1, 2, 3])
        other = replace(
            self.base_spec,
            run_id="dddddddddddddddddddd",
            repetition=1,
            algorithm=ComponentSpec(self.base_spec.algorithm.type, {"rate": 0.9}),
        )
        self.save_run(other, [3, 4, 5])
        with self.assertRaisesRegex(ValueError, "different scientific configurations"):
            analyze(self.analysis_config(), self.root)
        settings = deepcopy(self.settings)
        settings["aggregator"]["group_by"].append("algorithm.params.rate")
        self.assertEqual(len(analyze(self.analysis_config(settings), self.root)), 2)

    def test_rejects_duplicate_repetitions_and_misaligned_curves(self) -> None:
        """An aggregate requires distinct repetitions with matching curve coordinates."""
        self.save_run(self.base_spec, [1, 2, 3])
        self.save_run(replace(self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb"), [3, 4, 5])
        with self.assertRaisesRegex(ValueError, "Duplicate repetition"):
            analyze(self.analysis_config(), self.root)
        metadata = self.root / "runs" / "bbbbbbbbbbbbbbbbbbbb" / "metadata.json"
        atomic_json(
            metadata,
            {
                "spec": replace(
                    self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb", repetition=1
                ).to_dict()
            },
        )
        # Change the x axis using a custom metric, preserving valid storage itself.
        settings = deepcopy(self.settings)
        settings["metrics"][0]["params"]["x"] = "reward"
        with self.assertRaisesRegex(ValueError, "mismatched x"):
            analyze(self.analysis_config(settings), self.root)


class AnalysisResolutionTests(_PersistedAnalysisCase):
    """Keep exports compact without changing metric calculations or alignment checks."""

    def test_selection_is_bounded_evenly_spaced_and_deterministic(self) -> None:
        """Retain short/scalar curves and unique endpoints, including on very long curves."""
        cases = [(length, 100) for length in (1, 99, 100, 101, 1_000_000)]
        cases.extend([(11, 4), (101, 2), (101, None)])
        for length, points in cases:
            with self.subTest(length=length, points=points):
                x = 5 + 3 * np.arange(length)
                original = Summary("reward", {}, x, x, x, 1, "none")
                selected = _subsample_summary(original, points)
                expected_size = length if points is None else min(length, points)
                self.assertEqual(len(selected.x), expected_size)
                np.testing.assert_array_equal(selected.x[[0, -1]], x[[0, -1]])
                self.assertEqual(len(np.unique(selected.x)), expected_size)
                np.testing.assert_array_equal(selected.mean, selected.x)
                np.testing.assert_array_equal(selected.uncertainty, selected.x)
                np.testing.assert_array_equal(_subsample_summary(original, points).x, selected.x)
                if length == expected_size:
                    self.assertIs(selected, original)
                else:
                    gaps = np.diff(selected.x) // 3
                    self.assertLessEqual(gaps.max() - gaps.min(), 1)

    def test_cumulative_exports_preserve_exact_values_and_uncertainty(self) -> None:
        """Compute all increments before reducing tables and their binary summary copies."""
        steps = range(1, 302)
        rewards = np.asarray(steps, dtype=float) ** 2
        self.save_run(self.base_spec, rewards, steps=steps)
        self.save_run(
            replace(self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb", repetition=1),
            3 * rewards,
            steps=steps,
        )
        settings = deepcopy(self.settings)
        settings["metrics"][0]["type"] = "cumulative_sum"
        summary = analyze(self.analysis_config(settings), self.root)[0]
        self.assertEqual(len(summary.x), 100)
        cumulative = np.cumsum(rewards)
        np.testing.assert_array_equal(summary.mean, 2 * cumulative[summary.x - 1])
        np.testing.assert_allclose(summary.uncertainty, cumulative[summary.x - 1])
        table = self.root / "analysis" / "reward.csv"
        with table.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 100)
        np.testing.assert_array_equal([int(row["x"]) for row in rows], summary.x)
        np.testing.assert_allclose([float(row["mean"]) for row in rows], summary.mean)
        with np.load(table.with_suffix(".npz"), allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["group_0_x"], summary.x)
            np.testing.assert_array_equal(saved["group_0_mean"], summary.mean)
        for path in (self.root / "analysis" / "cache" / "metrics").glob("*/result.npz"):
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["x"], steps)

        settings["points"] = None
        full = analyze(self.analysis_config(settings), self.root)[0]
        np.testing.assert_array_equal(full.x, steps)
        np.testing.assert_array_equal(full.mean, 2 * cumulative)
        with table.open(newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), len(steps))

    def test_subsampling_cannot_hide_misaligned_coordinates(self) -> None:
        """A mismatch at an omitted interior coordinate must still reject aggregation."""
        self.save_run(self.base_spec, [1, 2, 3])
        self.save_run(
            replace(self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb", repetition=1), [1, 9, 3]
        )
        settings = deepcopy(self.settings)
        settings["points"] = 2
        settings["metrics"][0]["params"]["x"] = "reward"
        with self.assertRaisesRegex(ValueError, "mismatched x"):
            analyze(self.analysis_config(settings), self.root)


class FigureExportTests(_PersistedAnalysisCase):
    """Regenerate correctly formatted figures directly from persisted numerical results."""

    def test_tikz_regeneration_needs_no_simulation_modules(self) -> None:
        """Saved results produce repeatable, escaped TikZ without importing simulation code."""
        self.save_run(self.base_spec, [1, 2, 3])
        self.save_run(
            replace(self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb", repetition=1), [3, 4, 5]
        )
        paths = plot(self.analysis_config(), self.root)
        self.assertEqual([path.suffix for path in paths], [".tikz"])
        source = paths[0].read_text(encoding="utf-8")
        self.assertIn(r"\documentclass", source)
        self.assertIn(r"algorithm\_50\%", source)
        self.assertIn("fill between", source)
        self.assertIn("standard error", source)
        self.assertNotIn("nan", source)
        first_source = source
        self.assertEqual(plot(self.analysis_config(), self.root)[0].read_text(), first_source)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is optional")
    def test_pdf_and_jpg_regeneration(self) -> None:
        """Optional Matplotlib exports contain valid PDF and JPEG file headers."""
        self.save_run(self.base_spec, [1, 2, 3])
        settings = deepcopy(self.settings)
        settings["figures"][0]["formats"] = ["pdf", "jpg"]
        paths = plot(self.analysis_config(settings), self.root)
        self.assertEqual({path.suffix for path in paths}, {".pdf", ".jpg"})
        for path in paths:
            self.assertGreater(path.stat().st_size, 100)
        self.assertTrue(paths[0].read_bytes().startswith(b"%PDF"))
        self.assertTrue(paths[1].read_bytes().startswith(b"\xff\xd8"))

    def test_log_axes_reject_nonpositive_coordinates(self) -> None:
        """Logarithmic figures reject nonpositive metric means with an explicit error."""
        self.save_run(self.base_spec, [0, 1, 2])
        settings = deepcopy(self.settings)
        settings["figures"][0]["yscale"] = "log"
        with self.assertRaisesRegex(ValueError, "positive metric means"):
            plot(self.analysis_config(settings), self.root)

    def test_scalar_figure_uses_visible_marker_and_error_bar(self) -> None:
        """Single-point summaries use a visible marker and their sample uncertainty bar."""
        self.save_run(self.base_spec, [1], steps=(1,))
        self.save_run(
            replace(self.base_spec, run_id="bbbbbbbbbbbbbbbbbbbb", repetition=1), [3], steps=(1,)
        )
        source = plot(self.analysis_config(), self.root)[0].read_text()
        self.assertIn("only marks,mark=*", source)
        self.assertIn("error bars/.cd,y dir=both,y explicit", source)
        self.assertIn("(1,2) +- (0,1)", source)


class AnalysisValidationTests(_PersistedAnalysisCase):
    """Reject invalid analysis settings before producing misleading or misplaced outputs."""

    def test_invalid_metric_name_cannot_escape_output_directory(self) -> None:
        """Metric names cannot use parent paths to write outside the analysis directory."""
        self.save_run(self.base_spec, [1, 2, 3])
        settings = deepcopy(self.settings)
        settings["metrics"][0]["name"] = "../../escaped"
        with self.assertRaisesRegex(ValueError, "Metric name"):
            analyze(self.analysis_config(settings), self.root)

    def test_rejects_unknown_analysis_options(self) -> None:
        """Misspelled options are rejected in metric, aggregation, and figure settings."""
        self.save_run(self.base_spec, [1, 2, 3])
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
                    plot(self.analysis_config(settings), self.root)
