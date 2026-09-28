"""Settled checkpoint retirement, old-output cleanup, and recovery selection."""

from __future__ import annotations

import io
import json
import shutil
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from experiments_wo_stress import clean_experiment, run_experiment
from experiments_wo_stress.analysis.cache import AnalysisCache
from experiments_wo_stress.cli import main
from experiments_wo_stress.storage import (
    StorageError,
    atomic_json,
    create_snapshot,
    digest_file,
    read_json,
    validate_snapshot,
)
from experiments_wo_stress.storage.trajectories import prune_trajectory
from experiments_wo_stress.study.specs import RunSpec
from tests.test_execution import _ExecutionStudyTestCase


class SettledCheckpointTests(_ExecutionStudyTestCase):
    """Run disposable local studies and inspect publication boundaries."""

    def setUp(self) -> None:
        super().setUp()
        settings = self.single_run_settings()
        self.config = self.load_study_config(settings)
        self.output = self.root / "output"
        self.assertEqual(run_experiment(self.config, self.output).completed, 1)
        self.run = self.output / "runs" / read_json(self.output / "metadata.json")["run_ids"][0]
        self.spec = RunSpec.from_dict(read_json(self.run / "metadata.json")["spec"])
        self.metrics = {"metric": {"source": "settled-test"}}
        AnalysisCache(self.output).publish(
            "metrics", self.metrics["metric"], lambda path: (path / "values").write_text("1")
        )

    def prune(self) -> None:
        prune_trajectory(self.output, self.run.name, self.spec, self.metrics)

    def test_active_and_completed_unpruned_runs_keep_fallback_checkpoints(self) -> None:
        completed = read_json(self.run / "progress.json")
        self.assertEqual(len(completed["checkpoints"]), 2)
        self.assertEqual(len(list((self.run / "checkpoints").iterdir())), 2)
        active_output = self.root / "paused"
        self.assertEqual(run_experiment(self.config, active_output, max_steps=5).paused, 1)
        active_run = (
            active_output / "runs" / read_json(active_output / "metadata.json")["run_ids"][0]
        )
        active = read_json(active_run / "progress.json")
        self.assertEqual(active["status"], "paused")
        self.assertEqual(len(active["checkpoints"]), 2)
        self.assertEqual(len(list((active_run / "checkpoints").iterdir())), 2)

    def test_pruning_publishes_receipt_before_clearing_all_references(self) -> None:
        before = read_json(self.run / "progress.json")
        self.assertTrue(before["checkpoints"])
        self.assertTrue(before["completed_budgets"][str(before["step"])]["checkpoint"])
        original = atomic_json
        observed = []

        def observe(path, payload):
            if path == self.run / "progress.json" and not payload["checkpoints"]:
                observed.append(
                    (
                        (self.run / "trajectory.json").is_file(),
                        bool(list((self.run / "checkpoints").iterdir())),
                    )
                )
            return original(path, payload)

        with patch("experiments_wo_stress.storage.trajectories.atomic_json", side_effect=observe):
            self.prune()
        self.assertEqual(observed, [(True, True)])
        after = read_json(self.run / "progress.json")
        self.assertEqual(after["checkpoints"], [])
        self.assertTrue(
            all(item["checkpoint"] is None for item in after["completed_budgets"].values())
        )
        self.assertEqual(after["results"], before["results"])
        self.assertEqual(after["checkpoint_count"], before["checkpoint_count"])
        self.assertFalse(list((self.run / "checkpoints").iterdir()))

    def test_interrupted_deletion_is_safe_and_repeated_pruning_finishes_it(self) -> None:
        with patch(
            "experiments_wo_stress.storage.trajectories.shutil.rmtree",
            side_effect=OSError("interrupted checkpoint deletion"),
        ):
            with self.assertRaisesRegex(OSError, "interrupted checkpoint deletion"):
                self.prune()
        progress = read_json(self.run / "progress.json")
        self.assertEqual(progress["checkpoints"], [])
        self.assertTrue(
            all(item["checkpoint"] is None for item in progress["completed_budgets"].values())
        )
        self.assertTrue(list((self.run / "checkpoints").iterdir()))
        self.prune()
        self.prune()
        self.assertFalse(list((self.run / "checkpoints").iterdir()))

    def test_settled_cleanup_previews_and_retires_old_stale_checkpoints(self) -> None:
        old_progress = deepcopy(read_json(self.run / "progress.json"))
        old_checkpoints = self.root / "old-checkpoints"
        shutil.copytree(self.run / "checkpoints", old_checkpoints)
        self.prune()
        atomic_json(self.run / "progress.json", old_progress)
        shutil.rmtree(self.run / "checkpoints")
        shutil.copytree(old_checkpoints, self.run / "checkpoints")
        preview = clean_experiment(self.output, scope="settled")
        self.assertTrue(preview["dry_run"])
        self.assertEqual(preview["settled_runs"], 1)
        self.assertGreater(preview["checkpoint_generations"], 0)
        self.assertGreater(preview["checkpoint_files"], 0)
        self.assertGreater(preview["bytes"], 0)
        self.assertEqual(read_json(self.run / "progress.json"), old_progress)
        self.assertEqual(
            clean_experiment(self.output, scope="settled", yes=True)["bytes"], preview["bytes"]
        )
        self.assertEqual(read_json(self.run / "progress.json")["checkpoints"], [])
        self.assertFalse(list((self.run / "checkpoints").iterdir()))
        self.assertEqual(clean_experiment(self.output, scope="settled", yes=True)["bytes"], 0)

    def test_corrupt_receipt_refuses_settled_cleanup(self) -> None:
        self.prune()
        receipt = read_json(self.run / "trajectory.json")
        receipt["latest"] = "wrong"
        atomic_json(self.run / "trajectory.json", receipt)
        with self.assertRaisesRegex(StorageError, "Invalid trajectory pruning receipt"):
            clean_experiment(self.output, scope="settled", yes=True)

    def test_settled_scope_preserves_unpruned_runs_and_cli_defaults_to_preview(self) -> None:
        settings = self.single_run_settings()
        settings["runs"][0]["repetitions"] = 2
        self.assertEqual(run_experiment(self.load_study_config(settings), self.output).completed, 1)
        other_id = next(
            run_id
            for run_id in read_json(self.output / "metadata.json")["run_ids"]
            if run_id != self.run.name
        )
        other = self.output / "runs" / other_id
        self.prune()
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["clean", str(self.output), "--scope", "settled"]), 0)
        self.assertTrue(json.loads(output.getvalue())["dry_run"])
        clean_experiment(self.output, scope="settled", yes=True)
        self.assertTrue(list((other / "checkpoints").iterdir()))
        self.assertTrue(read_json(other / "progress.json")["checkpoints"])

    def test_recovery_omits_stale_pruned_payloads_without_validating_them(self) -> None:
        old_progress = deepcopy(read_json(self.run / "progress.json"))
        old_checkpoints = self.root / "old-checkpoints"
        shutil.copytree(self.run / "checkpoints", old_checkpoints)
        self.prune()
        atomic_json(self.run / "progress.json", old_progress)
        shutil.rmtree(self.run / "checkpoints")
        shutil.copytree(old_checkpoints, self.run / "checkpoints")
        original_iterdir = Path.iterdir

        def visit(path):
            if path == self.run / "checkpoints":
                raise AssertionError("pruned checkpoint directory was traversed")
            return original_iterdir(path)

        def hash_selected(path):
            if "/checkpoints/" in path.as_posix():
                raise AssertionError("pruned checkpoint payload was hashed")
            return digest_file(path)

        with (
            patch.object(Path, "iterdir", visit),
            patch(
                "experiments_wo_stress.storage.recovery._checkpoint",
                side_effect=AssertionError("pruned checkpoint was validated"),
            ),
            patch("experiments_wo_stress.storage.recovery.digest_file", side_effect=hash_selected),
        ):
            create_snapshot(self.output, self.root / "snapshot")
        manifest = validate_snapshot(self.root / "snapshot")
        self.assertFalse(any("/checkpoints/" in name for name in manifest["files"]))
        self.assertEqual(manifest["pruned_runs"], [self.run.name])
        self.assertTrue(list((self.run / "checkpoints").iterdir()))
