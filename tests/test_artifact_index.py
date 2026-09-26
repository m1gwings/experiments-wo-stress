"""Semantic discovery is deterministic and independent of scientific artifact state."""

import json
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from experiments_wo_stress import run_experiment
from experiments_wo_stress.analysis import analyze
from experiments_wo_stress.execution.compute_report import regenerate_summary
from experiments_wo_stress.plotting import plot
from experiments_wo_stress.storage.artifact_index import publish_artifact_index
from experiments_wo_stress.storage.experiment import inspect_experiment
from tests.test_analysis import _PersistedAnalysisCase
from tests.test_execution import _ExecutionStudyTestCase


def read_index(root: Path) -> dict:
    """Read the public wire format, independently of the publisher's constants."""
    return json.loads((root / "artifacts.json").read_text())


class ArtifactCatalogTests(unittest.TestCase):
    """Validate the portable contract without inventorying any result subtrees."""

    def test_schema_optional_locations_and_concurrent_deterministic_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch.object(Path, "rglob", side_effect=AssertionError("unexpected scan")),
                patch.object(Path, "iterdir", side_effect=AssertionError("unexpected scan")),
            ):
                publish_artifact_index(root)
                before = (root / "artifacts.json").read_bytes()
                with ThreadPoolExecutor(max_workers=4) as workers:
                    list(workers.map(publish_artifact_index, [root] * 12))
                self.assertEqual(before, (root / "artifacts.json").read_bytes())
            index = read_index(root)
            self.assertEqual(index["schema"], "experiments-wo-stress/artifacts")
            self.assertEqual(type(index["schema_version"]), int)
            self.assertEqual(index["schema_version"], 1)
            self.assertEqual(
                set(index["artifacts"]),
                {
                    "figures",
                    "analysis",
                    "compute_report",
                    "compute",
                    "runs",
                    "instances",
                    "requests",
                },
            )
            for entry in index["artifacts"].values():
                path = PurePosixPath(entry["path"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertIn(entry["kind"], {"directory", "file"})
                self.assertIs(entry["optional"], True)
                self.assertFalse((root / path).exists())
            self.assertEqual([p.name for p in root.iterdir()], ["artifacts.json"])


class ExecutionDiscoveryTests(_ExecutionStudyTestCase):
    """Run and compute publication expose metadata without requiring it to inspect."""

    def test_run_catalog_compute_report_and_legacy_inspection(self):
        output = self.root / "output"
        config = self.load_study_config(self.single_run_settings())
        self.assertEqual(run_experiment(config, output).completed, 1)
        index = read_index(output)["artifacts"]
        self.assertTrue((output / index["runs"]["path"]).is_dir())
        self.assertTrue((output / index["compute_report"]["path"]).is_file())
        self.assertFalse((output / index["figures"]["path"]).exists())
        (output / "artifacts.json").unlink()
        self.assertEqual(inspect_experiment(output)["counts"]["completed"], 1)
        regenerate_summary(output)
        self.assertEqual(read_index(output)["artifacts"], index)
        self.assertEqual(run_experiment(config, output).skipped, 1)


class AnalysisDiscoveryTests(_PersistedAnalysisCase):
    """Legacy saved results gain discovery through independent analyze/plot workflows."""

    def test_analysis_regeneration_and_absent_present_figures(self):
        self.save_run(self.base_spec, [1, 2, 3])
        config = self.analysis_config()
        analyze(config, self.root)
        index = read_index(self.root)
        figures = self.root / index["artifacts"]["figures"]["path"]
        self.assertFalse(figures.exists())
        paths = plot(config, self.root)
        self.assertTrue(paths)
        self.assertTrue(all(path.is_relative_to(figures) and path.is_file() for path in paths))
        self.assertEqual(read_index(self.root), index)
        shutil.rmtree(self.root / index["artifacts"]["analysis"]["path"])
        (self.root / "artifacts.json").unlink()
        analyze(config, self.root)
        self.assertEqual(read_index(self.root), index)
        self.assertFalse(figures.exists())
        plot(config, self.root)
        self.assertTrue(all(path.is_file() for path in paths))
