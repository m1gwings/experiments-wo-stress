"""Test storage contracts directly, without coordinating a whole experiment.

The groups cover state encoding, checkpoint commit/recovery, result recording,
and immutable instance persistence. Execution-level replay is in
``test_execution.py``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments_wo_stress import Instance
from experiments_wo_stress.storage import (
    Recorder,
    RunStore,
    StorageError,
    atomic_json,
    load_instance,
    read_json,
    save_instance,
)


class _StorageTestCase(unittest.TestCase):
    """Give each storage scenario an empty, disposable artifact directory."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.empty_manifest = {"schema": {}, "chunks": []}


class CheckpointStateTests(_StorageTestCase):
    """Explicit checkpoint encoding preserves numerical state and RNG replay."""

    def test_explicit_state_roundtrip_preserves_arrays_tuples_and_large_integers(self) -> None:
        """Restored state preserves user containers and the RNG's next sequence of draws."""
        rng = np.random.Generator(np.random.PCG64(42))
        rng.random(7)
        state = {
            "rng": rng.bit_generator.state,
            "values": np.arange(9).reshape(3, 3),
            "nested": [None, (True, 2**100, {"kind": "user data"})],
            "infinity": float("inf"),
        }
        store = RunStore(self.path)
        store.checkpoint(state, self.empty_manifest, 0)
        restored, _, _ = RunStore(self.path).restore()
        np.testing.assert_array_equal(restored["values"], state["values"])
        self.assertEqual(restored["nested"], state["nested"])
        self.assertEqual(restored["infinity"], float("inf"))
        replay = np.random.Generator(np.random.PCG64(0))
        replay.bit_generator.state = restored["rng"]
        np.testing.assert_array_equal(rng.random(10), replay.random(10))

    def test_numpy_complex_and_extended_scalars_roundtrip(self) -> None:
        """Numerical scalars without a lossless Python scalar retain their NumPy representation."""
        values = [
            np.complex64(1 + 2j),
            np.complex128(3 - 4j),
            np.longdouble("1.234567890123456789"),
        ]
        store = RunStore(self.path)
        store.checkpoint({"values": values}, self.empty_manifest, 0)
        restored, _, _ = RunStore(self.path).restore()
        for expected, actual in zip(values, restored["values"]):
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(np.asarray(actual).dtype, expected.dtype)


class CheckpointCommitTests(_StorageTestCase):
    """Publication and retention keep a previous valid checkpoint recoverable."""

    def test_failed_publication_preserves_previous_commit(self) -> None:
        """Writing checkpoint files is insufficient until progress.json publishes them."""
        store = RunStore(self.path)
        store.checkpoint({"value": 1}, self.empty_manifest, 1)

        def interrupted(path, value):
            if path.name == "progress.json":
                raise OSError("injected publication failure")
            atomic_json(path, value)

        with patch("experiments_wo_stress.storage.run.atomic_json", side_effect=interrupted):
            with self.assertRaises(OSError):
                store.checkpoint({"value": 2}, self.empty_manifest, 2)
        state, _, step = RunStore(self.path).restore()
        self.assertEqual((state, step), ({"value": 1}, 1))

    def test_fallback_retains_a_valid_previous_generation(self) -> None:
        """After fallback, even a second damaged checkpoint leaves the old valid state."""
        store = RunStore(self.path)
        store.checkpoint({"value": 1}, self.empty_manifest, 1)
        store.checkpoint({"value": 2}, self.empty_manifest, 2)
        corrupt = store.progress["checkpoints"][0]["generation"]
        (self.path / "checkpoints" / corrupt / "arrays.npz").write_bytes(b"broken")
        restored = RunStore(self.path)
        self.assertEqual(restored.restore()[2], 1)

        # Retention must keep the valid fallback when a new generation is written.
        restored.checkpoint({"value": 3}, self.empty_manifest, 3)
        corrupt = restored.progress["checkpoints"][0]["generation"]
        (self.path / "checkpoints" / corrupt / "arrays.npz").write_bytes(b"broken again")
        self.assertEqual(RunStore(self.path).restore()[2], 1)

    def test_checkpoint_retention_is_bounded(self) -> None:
        store = RunStore(self.path)
        for step in range(5):
            store.checkpoint({"step": step}, self.empty_manifest, step)
        self.assertEqual(len(list((self.path / "checkpoints").iterdir())), 2)
        self.assertEqual(len(read_json(self.path / "progress.json")["checkpoints"]), 2)


class ResultRecordingTests(_StorageTestCase):
    """Recorded arrays obey their schema and the last committed result boundary."""

    def test_uncommitted_result_tail_is_removed_before_replay(self) -> None:
        """Recovery removes flushed rows beyond the checkpoint's committed manifest."""
        store = RunStore(self.path)
        recorder = Recorder(store, {"buffer_bytes": 16}, self.empty_manifest)
        recorder.record(1, {"value": 1.0})
        store.checkpoint({"value": 1}, recorder.manifest, 1)
        # The tiny buffer flushes step two, but no checkpoint commits that row.
        recorder.record(2, {"value": 2.0})
        self.assertTrue((self.path / "results" / "000001.npz").exists())
        RunStore(self.path).restore()
        self.assertFalse((self.path / "results" / "000001.npz").exists())
        self.assertTrue((self.path / "results" / "000000.npz").exists())

    def test_recording_rejects_changed_schema_and_arbitrary_objects(self) -> None:
        recorder = Recorder(RunStore(self.path), {}, self.empty_manifest)
        recorder.record(1, {"value": 1.0})
        with self.assertRaises(ValueError):
            recorder.record(2, {"value": np.ones(2)})
        with self.assertRaises(TypeError):
            recorder.record(2, {"value": object()})
        self.assertFalse((self.path / "progress.json").exists())

    def test_npz_keyword_names_roundtrip_as_measurements_and_instance_arrays(self) -> None:
        """Array names are scientific field names, not arguments to NumPy's exporter."""
        values = {"file": np.array([1, 2], dtype=np.int16), "allow_pickle": np.array(3.5)}
        for compression in (False, True):
            with self.subTest(compression=compression):
                root = self.path / str(compression)
                store = RunStore(root / "run", compression=compression)
                recorder = Recorder(store, {}, self.empty_manifest)
                recorder.record(1, values)
                recorder.flush()
                with np.load(store.directory / "results/000000.npz", allow_pickle=False) as arrays:
                    self.assertEqual(set(arrays.files), {*values, "step"})
                    for name, expected in values.items():
                        np.testing.assert_array_equal(arrays[name], expected[None])
                        self.assertEqual(arrays[name].dtype, expected.dtype)
                identifier = save_instance(root, Instance(arrays=values), compression=compression)
                restored = load_instance(root, identifier)
                for name, expected in values.items():
                    np.testing.assert_array_equal(restored.arrays[name], expected)


class InstancePersistenceTests(_StorageTestCase):
    """Saved instances deduplicate by content and reject tampered metadata."""

    def test_instance_deduplicates_and_loads_without_simulator(self) -> None:
        instance = Instance(metadata={"labels": ["a", "b"]}, arrays={"means": np.array([0.1, 0.9])})
        key = save_instance(self.path, instance, compression=True)
        self.assertEqual(save_instance(self.path, instance), key)
        restored = load_instance(self.path, key)
        np.testing.assert_array_equal(restored.arrays["means"], instance.arrays["means"])
        self.assertFalse(restored.arrays["means"].flags.writeable)
        self.assertEqual(len(list((self.path / "instances").iterdir())), 1)

    def test_instance_metadata_corruption_is_rejected(self) -> None:
        key = save_instance(
            self.path, Instance(metadata={"truth": 1}, arrays={"empty": np.empty((0, 2))})
        )
        path = self.path / "instances" / key / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["metadata"]["truth"] = 2
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(StorageError, "identity"):
            load_instance(self.path, key)
