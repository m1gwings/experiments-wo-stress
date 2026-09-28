"""Cloud snapshots preserve committed recovery boundaries without interpreting science."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments_wo_stress import run_experiment
from experiments_wo_stress.analysis.cache import AnalysisCache
from experiments_wo_stress.storage import (
    ExperimentStore,
    Recorder,
    RunStore,
    StorageError,
    atomic_json,
    create_snapshot,
    digest_file,
    fingerprint,
    read_json,
    restore_snapshot,
    save_instance,
    validate_snapshot,
)
from experiments_wo_stress.storage.models import Instance
from experiments_wo_stress.storage.trajectories import (
    prune_trajectory,
    pruning_history,
    pruning_receipt,
    rematerialize_trajectory,
    result_reference,
)
from experiments_wo_stress.study.specs import RunSpec
from tests.test_execution import _ExecutionStudyTestCase


class _RecoveryCase(unittest.TestCase):
    """Use small committed checkpoints to test inventories, fallback, and publication."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        atomic_json(self.output / "metadata.json", {"schema_version": 2, "run_ids": ["a" * 20]})
        self.run = self.output / "runs" / ("a" * 20)
        self.store = RunStore(self.run)
        atomic_json(
            self.run / "metadata.json",
            {
                "identity": {"scientific": "test"},
                "instance_id": save_instance(self.output, Instance()),
            },
        )
        self.recorder = Recorder(self.store, {"buffer_bytes": 16}, {"schema": {}, "chunks": []})
        self.recorder.record(1, {"value": 11})
        self.store.checkpoint({"value": 11}, self.recorder.manifest, 1, status="paused")
        self.snapshot = self.root / "snapshot"


class RecoveryBoundaryTests(_RecoveryCase):
    """Only published checkpoints and derived generations become snapshot data."""

    def test_snapshot_excludes_uncommitted_objects_and_preserves_recovery(self) -> None:
        self.recorder.record(2, {"value": 22})
        uncommitted = self.run / "checkpoints" / ("b" * 32)
        uncommitted.mkdir()
        (uncommitted / "state.json").write_text("not committed")
        (self.output / ".metadata.json.interrupted.tmp").write_text("partial")
        pending = self.output / "analysis/cache/metrics/.pending-test"
        pending.mkdir(parents=True)
        (pending / "metric.npz").write_text("partial")
        with patch(
            "experiments_wo_stress.storage.run.create_checkpoint_backend",
            side_effect=AssertionError("snapshots must not decode checkpoints"),
        ):
            create_snapshot(self.output, self.snapshot)
        manifest = validate_snapshot(self.snapshot)
        self.assertEqual(manifest["schema"], "experiments-wo-stress/recovery")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertFalse(any(".pending" in name or ".tmp" in name for name in manifest["files"]))
        self.assertFalse(any("b" * 32 in name for name in manifest["files"]))
        self.assertNotIn(f"runs/{self.run.name}/results/000001.npz", manifest["files"])
        self.assertNotIn(".lock", manifest["files"])
        restored = restore_snapshot(self.snapshot, self.root / "restored")
        state, _, step = RunStore(restored / "runs" / self.run.name).restore()
        self.assertEqual((state, step), ({"value": 11}, 1))

    def test_empty_checkpoint_payload_directories_are_restored(self) -> None:
        generation = self.store.progress["checkpoints"][0]["generation"]
        empty = self.run / "checkpoints" / generation / "payload/empty"
        empty.mkdir(parents=True)
        create_snapshot(self.output, self.snapshot)
        relative = empty.relative_to(self.output).as_posix()
        self.assertIn(relative, validate_snapshot(self.snapshot)["directories"])
        restored = restore_snapshot(self.snapshot, self.root / "restored")
        self.assertTrue((restored / relative).is_dir())
        self.assertEqual(list((restored / relative).iterdir()), [])
        self.assertEqual(RunStore(restored / "runs" / self.run.name).restore()[2], 1)

    def test_corrupt_latest_generation_keeps_the_valid_predecessor(self) -> None:
        self.recorder.record(2, {"value": 22})
        self.store.checkpoint({"value": 22}, self.recorder.manifest, 2, status="paused")
        latest = self.store.progress["checkpoints"][0]["generation"]
        (self.run / "checkpoints" / latest / "arrays.npz").write_bytes(b"damaged")
        create_snapshot(self.output, self.snapshot)
        manifest = validate_snapshot(self.snapshot)
        self.assertEqual([item["generation"] for item in manifest["omitted_checkpoints"]], [latest])
        self.assertFalse(any(latest in name for name in manifest["files"]))
        restored = restore_snapshot(self.snapshot, self.root / "restored")
        store = RunStore(restored / "runs" / self.run.name)
        self.assertEqual(store.restore()[2], 1)
        self.assertEqual(store.progress["step"], 2)  # Normal recovery owns progress updates.

    def test_malformed_newest_checkpoint_entry_preserves_valid_fallback(self) -> None:
        self.store.progress["checkpoints"].insert(0, "invalid entry")
        atomic_json(self.run / "progress.json", self.store.progress)
        create_snapshot(self.output, self.snapshot)
        self.assertEqual(len(validate_snapshot(self.snapshot)["omitted_checkpoints"]), 1)
        restored = restore_snapshot(self.snapshot, self.root / "restored")
        self.assertEqual(RunStore(restored / "runs" / self.run.name).restore()[2], 1)

    def test_unknown_checkpoint_envelope_is_not_discarded_as_missing_state(self) -> None:
        entry = self.store.progress["checkpoints"][0]
        envelope = self.run / "checkpoints" / entry["generation"] / "checkpoint.json"
        contents = read_json(envelope)
        contents["schema_version"] = 2
        atomic_json(envelope, contents)
        entry["checkpoint_sha256"] = digest_file(envelope)
        atomic_json(self.run / "progress.json", self.store.progress)
        with self.assertRaisesRegex(StorageError, "Unsupported checkpoint envelope"):
            create_snapshot(self.output, self.snapshot)
        self.assertFalse(self.snapshot.exists())
        self.assertTrue(envelope.exists())

    def test_completed_corruption_is_reported_without_publishing_a_snapshot(self) -> None:
        self.store.finish(self.recorder.manifest, 1)
        (self.run / "results/000000.npz").unlink()
        with self.assertRaisesRegex(StorageError, "Missing or corrupt result chunk"):
            create_snapshot(self.output, self.snapshot)
        self.assertFalse(self.snapshot.exists())
        self.assertFalse(list(self.root.glob(".recovery-*")))

    def test_missing_referenced_instance_is_not_certified_as_a_valid_snapshot(self) -> None:
        instance_id = read_json(self.run / "metadata.json")["instance_id"]
        shutil.rmtree(self.output / "instances" / instance_id)
        with self.assertRaisesRegex(StorageError, "Cannot read"):
            create_snapshot(self.output, self.snapshot)
        self.assertFalse(self.snapshot.exists())

    def test_active_writer_and_existing_destinations_are_rejected(self) -> None:
        with ExperimentStore(self.output).lock():
            with self.assertRaisesRegex(StorageError, "Another executor"):
                create_snapshot(self.output, self.snapshot)
        create_snapshot(self.output, self.snapshot)
        with self.assertRaisesRegex(StorageError, "must not exist"):
            create_snapshot(self.output, self.snapshot)
        with self.assertRaisesRegex(StorageError, "must not exist"):
            restore_snapshot(self.snapshot, self.output)
        with self.assertRaisesRegex(StorageError, "must not contain"):
            create_snapshot(self.output, self.output / "snapshot")

    def test_failed_copy_never_publishes_a_partial_snapshot(self) -> None:
        with patch(
            "experiments_wo_stress.storage.recovery.shutil.copyfile",
            side_effect=OSError("injected copy failure"),
        ):
            with self.assertRaisesRegex(OSError, "copy failure"):
                create_snapshot(self.output, self.snapshot)
        self.assertFalse(self.snapshot.exists())
        self.assertFalse(list(self.root.glob(".recovery-*")))
        self.assertEqual(RunStore(self.run).restore()[2], 1)

    def test_committed_analysis_generations_are_validated(self) -> None:
        cache = AnalysisCache(self.output)
        identity = {"kind": "metric", "source": "fixture"}
        directory = cache.publish(
            "metrics", identity, lambda path: (path / "values").write_text("1")
        )
        create_snapshot(self.output, self.snapshot)
        self.assertIn(
            f"analysis/cache/metrics/{fingerprint(identity)}/values",
            validate_snapshot(self.snapshot)["files"],
        )
        (directory / "values").write_text("damaged")
        with self.assertRaisesRegex(StorageError, "Corrupt analysis cache"):
            create_snapshot(self.output, self.root / "damaged")


class RecoveryValidationTests(_RecoveryCase):
    """Reject incompatible or modified snapshots before writing restored output."""

    def test_future_version_fails_before_restore_writes_output(self) -> None:
        create_snapshot(self.output, self.snapshot)
        manifest = read_json(self.snapshot / "recovery.json")
        manifest["schema_version"] = 2
        atomic_json(self.snapshot / "recovery.json", manifest)
        destination = self.root / "restored"
        with self.assertRaisesRegex(StorageError, "Unsupported EWS recovery contract"):
            restore_snapshot(self.snapshot, destination)
        self.assertFalse(destination.exists())

    def test_malformed_contract_fails_before_restore_writes_output(self) -> None:
        create_snapshot(self.output, self.snapshot)
        original = read_json(self.snapshot / "recovery.json")
        malformed = [None, [], "invalid", {**original, "schema_version": True}]
        for index, manifest in enumerate(malformed):
            with self.subTest(manifest=manifest):
                atomic_json(self.snapshot / "recovery.json", manifest)
                destination = self.root / f"restored-{index}"
                with self.assertRaisesRegex(StorageError, "Unsupported EWS recovery contract"):
                    restore_snapshot(self.snapshot, destination)
                self.assertFalse(destination.exists())

    def test_altered_missing_and_extra_files_are_not_valid_snapshots(self) -> None:
        for alteration in ("altered", "missing", "extra"):
            with self.subTest(alteration=alteration):
                snapshot = self.root / alteration
                create_snapshot(self.output, snapshot)
                metadata = snapshot / "output/metadata.json"
                if alteration == "altered":
                    metadata.write_text("wrong")
                elif alteration == "missing":
                    metadata.unlink()
                else:
                    (snapshot / "output/extra").write_text("unexpected")
                with self.assertRaises(StorageError):
                    restore_snapshot(snapshot, self.root / f"restored-{alteration}")

    def test_unsafe_paths_fail_even_with_a_matching_manifest_digest(self) -> None:
        create_snapshot(self.output, self.snapshot)
        original = read_json(self.snapshot / "recovery.json")
        for unsafe in ("../escape", "/absolute", "a//b", "a\\b", "a:b", "a/./b"):
            with self.subTest(path=unsafe):
                manifest = {**original, "files": {**original["files"], unsafe: {}}}
                manifest["snapshot_id"] = fingerprint(
                    {key: value for key, value in manifest.items() if key != "snapshot_id"}
                )
                atomic_json(self.snapshot / "recovery.json", manifest)
                with self.assertRaisesRegex(StorageError, "Unsafe recovery path"):
                    validate_snapshot(self.snapshot)

    def test_links_cannot_escape_the_snapshot(self) -> None:
        create_snapshot(self.output, self.snapshot)
        metadata = self.snapshot / "output/metadata.json"
        metadata.unlink()
        metadata.symlink_to(self.output / "metadata.json")
        with self.assertRaisesRegex(StorageError, "regular files"):
            validate_snapshot(self.snapshot)


class StudyRecoveryTests(_ExecutionStudyTestCase):
    """Restored output reuses normal EWS compatibility and intentional-pruning rules."""

    def test_resumed_snapshot_matches_uninterrupted_study(self) -> None:
        config = self.load_study_config(self.single_run_settings())
        reference = self.root / "reference"
        interrupted = self.root / "interrupted"
        run_experiment(config, reference)
        self.assertEqual(run_experiment(config, interrupted, max_steps=5).paused, 1)
        snapshot = create_snapshot(interrupted, self.root / "snapshot")
        restored = restore_snapshot(snapshot, self.root / "restored")
        self.assertEqual(run_experiment(config, restored).completed, 1)
        self.assert_results_equal(reference, restored)

    def test_pruned_receipt_omits_checkpoints_without_recreating_chunks(self) -> None:
        config = self.load_study_config(self.single_run_settings())
        output = self.root / "output"
        run_experiment(config, output)
        storage_id = read_json(output / "metadata.json")["run_ids"][0]
        directory = output / "runs" / storage_id
        spec = RunSpec.from_dict(read_json(directory / "metadata.json")["spec"])
        identity = {"kind": "metric", "source": "fixture"}
        AnalysisCache(output).publish(
            "metrics", identity, lambda path: (path / "values").write_text("1")
        )
        prune_trajectory(output, storage_id, spec, {"metric": identity})
        snapshot = create_snapshot(output, self.root / "snapshot")
        manifest = validate_snapshot(snapshot)
        self.assertEqual(manifest["pruned_runs"], [storage_id])
        self.assertEqual(manifest["omitted_checkpoints"], [])
        self.assertFalse(any("/checkpoints/" in name for name in manifest["files"]))
        self.assertFalse(any("/results/" in name for name in manifest["files"]))
        restored = restore_snapshot(snapshot, self.root / "restored")
        self.assertEqual(
            pruning_receipt(restored / "runs" / storage_id), pruning_receipt(directory)
        )
        self.assertFalse(list((restored / "runs" / storage_id).glob("results/*.npz")))

    def test_pending_replay_retains_valid_old_budget_history_in_the_snapshot(self) -> None:
        config = self.load_study_config(self.single_run_settings())
        output = self.root / "output"
        run_experiment(config, output)
        storage_id = read_json(output / "metadata.json")["run_ids"][0]
        directory = output / "runs" / storage_id
        spec = RunSpec.from_dict(read_json(directory / "metadata.json")["spec"])
        identity = {"kind": "metric", "source": "fixture"}
        AnalysisCache(output).publish(
            "metrics", identity, lambda path: (path / "values").write_text("1")
        )
        prune_trajectory(output, storage_id, spec, {"metric": identity})
        rematerialize_trajectory(directory)
        snapshot = create_snapshot(output, self.root / "snapshot")
        restored = restore_snapshot(snapshot, self.root / "restored")
        self.assertFalse((restored / "runs" / storage_id / "trajectory.json").exists())
        self.assertEqual(
            pruning_history(restored / "runs" / storage_id), pruning_history(directory)
        )
        self.assertEqual(result_reference(restored, storage_id, spec)["trajectory"], "pruned")
        history = read_json(directory / "trajectory-history.json")
        history["latest"] = "999"
        atomic_json(directory / "trajectory-history.json", history)
        with self.assertRaisesRegex(StorageError, "Invalid trajectory pruning receipt"):
            create_snapshot(output, self.root / "invalid-snapshot")
