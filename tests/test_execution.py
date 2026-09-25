"""Exercise persistence and replay through the public API with real components."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.execution import provenance
from experiments_wo_stress.storage import iter_completed_runs

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root / "study"
        self.study.mkdir()
        self.package_name = f"experiment_code_{self._testMethodName}"
        shutil.copytree(
            EXAMPLES / "sequential_study" / "experiment_code",
            self.study / self.package_name,
        )
        self.settings = {
            "name": "integration",
            "seed": 718,
            "runs": [
                {
                    "name": "main",
                    "planner": "grid",
                    "repetitions": 2,
                    "protocol": {"type": "online", "params": {"horizon": 13}},
                    "data": {
                        "type": f"{self.package_name}.data:GaussianBandit",
                        "params": {"n_arms": 4},
                    },
                    "algorithms": [
                        {
                            "name": "ucb",
                            "type": f"{self.package_name}.algorithms:UCB",
                            "params": {"n_arms": 4},
                        },
                        {
                            "name": "epsilon",
                            "type": f"{self.package_name}.algorithms:EpsilonGreedy",
                            "params": {"n_arms": 4, "epsilon": 0.25},
                        },
                    ],
                    "grid": {"data.params.noise_std": [0.1, 0.3]},
                }
            ],
            "execution": {"checkpoint_steps": 3, "keep_checkpoints": 2},
            "recording": {"buffer_bytes": 128, "every_steps": 1},
        }

    def config(self, settings: dict | None = None):
        path = self.study / "experiment.yml"
        path.write_text(yaml.safe_dump(settings or self.settings), encoding="utf-8")
        return load_config(path)

    def one_run(self) -> dict:
        settings = deepcopy(self.settings)
        group = settings["runs"][0]
        group["repetitions"] = 1
        group["algorithms"] = group["algorithms"][:1]
        group.pop("grid")
        return settings

    def results(self, output: Path) -> dict[str, dict[str, np.ndarray]]:
        return {spec.run_id: results for spec, results in iter_completed_runs(output)}

    def assert_same_results(self, first: Path, second: Path) -> None:
        expected, actual = self.results(first), self.results(second)
        self.assertEqual(expected.keys(), actual.keys())
        self.assertTrue(expected)
        for run_id, arrays in expected.items():
            self.assertEqual(arrays.keys(), actual[run_id].keys())
            for field, values in arrays.items():
                with self.subTest(run=run_id, field=field):
                    np.testing.assert_array_equal(values, actual[run_id][field])
            steps = actual[run_id]["step"]
            self.assertTrue(np.all(np.diff(steps) > 0), "replay duplicated records")

    def test_resume_with_different_worker_count_matches_uninterrupted(self) -> None:
        config = self.config()
        uninterrupted = self.root / "uninterrupted"
        resumed = self.root / "resumed"
        report = run_experiment(config, uninterrupted, workers=1)
        self.assertEqual((report.completed, report.failed), (8, 0), report.errors)
        paused = run_experiment(config, resumed, workers=1, max_steps=5)
        self.assertEqual((paused.paused, paused.completed, paused.failed), (8, 0, 0))
        report = run_experiment(config, resumed, workers=2)
        self.assertEqual((report.completed, report.failed), (8, 0), report.errors)
        self.assert_same_results(uninterrupted, resumed)
        report = run_experiment(config, resumed)
        self.assertEqual((report.skipped, report.completed), (8, 0))

    def test_sparse_recording_preserves_scheduled_steps_and_observations(self) -> None:
        settings = self.one_run()
        dense = self.root / "dense"
        sparse = self.root / "sparse"
        run_experiment(self.config(settings), dense)
        settings["recording"]["every_steps"] = 4
        config = self.config(settings)
        self.assertEqual(run_experiment(config, sparse, max_steps=6).paused, 1)
        self.assertEqual(run_experiment(config, sparse).completed, 1)
        dense_result = next(iter(self.results(dense).values()))
        sparse_result = next(iter(self.results(sparse).values()))
        np.testing.assert_array_equal(sparse_result["step"], [4, 8, 12])
        positions = np.searchsorted(dense_result["step"], sparse_result["step"])
        np.testing.assert_array_equal(sparse_result["reward"], dense_result["reward"][positions])

    def test_changed_science_retains_and_reuses_previous_variant(self) -> None:
        settings = self.one_run()
        output = self.root / "output"
        original = self.config(settings)
        self.assertEqual(run_experiment(original, output).completed, 1)
        settings["runs"][0]["data"]["params"]["noise_std"] = 0.8
        self.assertEqual(run_experiment(self.config(settings), output).completed, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 2)
        self.assertEqual(run_experiment(original, output).skipped, 1)

    def test_extension_matches_direct_run_across_worker_counts(self) -> None:
        settings = deepcopy(self.settings)
        initial = self.config(settings)
        extended, direct = self.root / "extended", self.root / "direct"
        self.assertEqual(run_experiment(initial, extended, workers=2).completed, 8)
        settings["runs"][0]["protocol"]["params"]["horizon"] = 27
        longer = self.config(settings)
        self.assertEqual(run_experiment(longer, extended, max_steps=3).paused, 8)
        # Earlier completed prefixes remain readable while continuation is paused.
        self.assertEqual(run_experiment(initial, extended).skipped, 8)
        self.assertTrue(
            all(len(result["step"]) == 13 for result in self.results(extended).values())
        )
        self.assertEqual(run_experiment(longer, extended, workers=1).completed, 8)
        self.assertEqual(run_experiment(longer, direct, workers=2).completed, 8)
        self.assert_same_results(direct, extended)
        self.assertEqual(len(list((extended / "runs").iterdir())), 8)
        self.assertEqual(len(list((extended / "instances").iterdir())), 4)

    def test_component_source_change_selects_new_variant(self) -> None:
        config = self.config(self.one_run())
        output = self.root / "output"
        self.assertEqual(run_experiment(config, output, max_steps=4).paused, 1)
        source = self.study / self.package_name / "algorithms.py"
        original_source = source.read_text(encoding="utf-8")
        source.write_text(
            original_source + "\n# A changed scientific implementation.\n",
            encoding="utf-8",
        )
        self.assertEqual(run_experiment(config, output).completed, 1)
        source.write_text(original_source, encoding="utf-8")
        self.assertEqual(run_experiment(config, output).completed, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 2)

    def test_moved_implementation_changes_select_new_retained_variants(self) -> None:
        config = self.config(self.one_run())
        output = self.root / "output"
        self.assertEqual(run_experiment(config, output).completed, 1)
        original_digest = provenance.digest_file
        implementation_files = (
            "execution/worker.py",
            "components/contracts.py",
            "storage/run.py",
            "study/rng.py",
        )
        for relative_path in implementation_files:
            with self.subTest(implementation=relative_path):

                def changed_digest(path: Path) -> str:
                    if path.as_posix().endswith("/" + relative_path):
                        return "changed implementation"
                    return original_digest(path)

                with patch.object(provenance, "digest_file", side_effect=changed_digest):
                    self.assertEqual(run_experiment(config, output).completed, 1)
                self.assertEqual(run_experiment(config, output).skipped, 1)
        self.assertEqual(len(list((output / "runs").iterdir())), 1 + len(implementation_files))

    def test_changed_csv_contents_select_new_variant(self) -> None:
        source = self.study / "input.csv"
        source.write_text("1.0\n2.0\n3.0\n", encoding="utf-8")
        shutil.copytree(
            EXAMPLES / "offline_csv" / "experiment_code",
            self.study / self.package_name,
            dirs_exist_ok=True,
        )
        settings = self.one_run()
        group = settings["runs"][0]
        group["protocol"] = {"type": "offline"}
        group["data"] = {"type": "csv", "params": {"path": "input.csv"}}
        group["algorithms"] = [
            {"name": "mean", "type": f"{self.package_name}.algorithms:MeanEstimator"}
        ]
        config = self.config(settings)
        output = self.root / "output"
        self.assertEqual(run_experiment(config, output).completed, 1)
        source.write_text("1.0\n2.0\n9.0\n", encoding="utf-8")
        self.assertEqual(run_experiment(config, output).completed, 1)
        result = next(iter(self.results(output).values()))
        np.testing.assert_allclose(result["mean"], [4.0])
        self.assertEqual(len(list((output / "runs").iterdir())), 2)

    def test_max_steps_requires_a_nonnegative_integer(self) -> None:
        config = self.config(self.one_run())
        for invalid in (-1, True, 1.5, "3"):
            with self.subTest(max_steps=invalid), self.assertRaises(ValueError):
                run_experiment(config, self.root / "output", max_steps=invalid)

    def test_execution_and_analysis_changes_can_resume(self) -> None:
        settings = self.one_run()
        output = self.root / "output"
        self.assertEqual(run_experiment(self.config(settings), output, max_steps=4).paused, 1)
        settings["execution"]["checkpoint_steps"] = 5
        settings["analysis"] = {
            "metrics": [{"name": "regret", "type": "field", "params": {"field": "reward"}}]
        }
        self.assertEqual(run_experiment(self.config(settings), output).completed, 1)

    def test_corrupt_latest_checkpoint_falls_back_without_duplicate_results(self) -> None:
        config = self.config(self.one_run())
        reference = self.root / "reference"
        recovered = self.root / "recovered"
        self.assertEqual(run_experiment(config, reference).completed, 1)
        self.assertEqual(run_experiment(config, recovered, max_steps=7).paused, 1)
        run_dir = next((recovered / "runs").iterdir())
        progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(len(progress["checkpoints"]), 2)
        generation = progress["checkpoints"][0]["generation"]
        (run_dir / "checkpoints" / generation / "arrays.npz").write_bytes(b"interrupted write")
        report = run_experiment(config, recovered)
        self.assertEqual((report.completed, report.failed), (1, 0), report.errors)
        self.assert_same_results(reference, recovered)

    def test_corrupt_completed_output_is_reported(self) -> None:
        config = self.config(self.one_run())
        output = self.root / "output"
        self.assertEqual(run_experiment(config, output).completed, 1)
        chunk = next((output / "runs").glob("*/results/*.npz"))
        chunk.write_bytes(b"damaged result")
        report = run_experiment(config, output)
        self.assertEqual((report.failed, report.skipped), (1, 0))
        self.assertTrue(report.errors)

    def test_failure_retries_from_last_safe_checkpoint(self) -> None:
        marker = self.root / "failure-happened"
        module = self.study / "failure_component.py"
        module.write_text(
            "from pathlib import Path\n"
            f"from {self.package_name}.algorithms import UCB\n\n"
            "class FailOnce(UCB):\n"
            "    def __init__(self, *, rng, marker, **params):\n"
            "        super().__init__(rng=rng, **params)\n"
            "        self.marker = Path(marker)\n\n"
            "    def observe(self, action, feedback):\n"
            "        if self.counts.sum() == 5 and not self.marker.exists():\n"
            "            self.marker.touch()\n"
            "            raise RuntimeError('injected failure after generating feedback')\n"
            "        super().observe(action, feedback)\n",
            encoding="utf-8",
        )
        settings = self.one_run()
        algorithm = settings["runs"][0]["algorithms"][0]
        algorithm["type"] = "failure_component:FailOnce"
        algorithm["params"]["marker"] = str(marker)
        config = self.config(settings)
        reference, recovered = self.root / "reference", self.root / "recovered"
        marker.touch()
        self.assertEqual(run_experiment(config, reference).completed, 1)
        marker.unlink()
        report = run_experiment(config, recovered)
        self.assertEqual(report.failed, 1)
        self.assertIn("injected failure", next(iter(report.errors.values())))
        self.assertEqual(run_experiment(config, recovered).completed, 1)
        self.assert_same_results(reference, recovered)

    def test_offline_csv_example(self) -> None:
        config = load_config(EXAMPLES / "offline_csv" / "experiment.yml")
        output = self.root / "offline"
        report = run_experiment(config, output)
        self.assertEqual((report.completed, report.failed), (1, 0), report.errors)
        results = next(iter(self.results(output).values()))
        raw = np.loadtxt(EXAMPLES / "offline_csv" / "observations.csv", skiprows=1)
        np.testing.assert_allclose(results["mean"], [raw.mean()])
        np.testing.assert_allclose(results["variance"], [raw.var(ddof=1)])
        np.testing.assert_array_equal(results["samples"], [len(raw)])


if __name__ == "__main__":
    unittest.main()
