"""Failure-oriented tests of checkpoint publication and explicit state encoding."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments_wo_stress.storage import Recorder, RunStore, atomic_json, read_json


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.empty = {"schema": {}, "chunks": []}

    def test_explicit_state_roundtrip_preserves_arrays_tuples_and_large_integers(self) -> None:
        rng = np.random.Generator(np.random.PCG64(42))
        rng.random(7)
        state = {
            "rng": rng.bit_generator.state,
            "values": np.arange(9).reshape(3, 3),
            "nested": [None, (True, 2**100, {"kind": "user data"})],
            "infinity": float("inf"),
        }
        store = RunStore(self.path)
        store.checkpoint(state, self.empty, 0)
        restored, _, _ = RunStore(self.path).restore()
        np.testing.assert_array_equal(restored["values"], state["values"])
        self.assertEqual(restored["nested"], state["nested"])
        self.assertEqual(restored["infinity"], float("inf"))
        replay = np.random.Generator(np.random.PCG64(0))
        replay.bit_generator.state = restored["rng"]
        np.testing.assert_array_equal(rng.random(10), replay.random(10))

    def test_failed_publication_preserves_previous_commit(self) -> None:
        store = RunStore(self.path)
        store.checkpoint({"value": 1}, self.empty, 1)

        def interrupted(path, value):
            if path.name == "progress.json":
                raise OSError("injected publication failure")
            atomic_json(path, value)

        with patch("experiments_wo_stress.storage.run.atomic_json", side_effect=interrupted):
            with self.assertRaises(OSError):
                store.checkpoint({"value": 2}, self.empty, 2)
        state, _, step = RunStore(self.path).restore()
        self.assertEqual((state, step), ({"value": 1}, 1))

    def test_fallback_retains_a_valid_previous_generation(self) -> None:
        store = RunStore(self.path)
        store.checkpoint({"value": 1}, self.empty, 1)
        store.checkpoint({"value": 2}, self.empty, 2)
        corrupt = store.progress["checkpoints"][0]["generation"]
        (self.path / "checkpoints" / corrupt / "arrays.npz").write_bytes(b"broken")
        restored = RunStore(self.path)
        self.assertEqual(restored.restore()[2], 1)
        restored.checkpoint({"value": 3}, self.empty, 3)
        corrupt = restored.progress["checkpoints"][0]["generation"]
        (self.path / "checkpoints" / corrupt / "arrays.npz").write_bytes(b"broken again")
        self.assertEqual(RunStore(self.path).restore()[2], 1)

    def test_uncommitted_result_tail_is_removed_before_replay(self) -> None:
        store = RunStore(self.path)
        recorder = Recorder(store, {"buffer_bytes": 16}, self.empty)
        recorder.record(1, {"value": 1.0})
        store.checkpoint({"value": 1}, recorder.manifest, 1)
        recorder.record(2, {"value": 2.0})
        self.assertTrue((self.path / "results" / "000001.npz").exists())
        RunStore(self.path).restore()
        self.assertFalse((self.path / "results" / "000001.npz").exists())
        self.assertTrue((self.path / "results" / "000000.npz").exists())

    def test_recording_rejects_changed_schema_and_arbitrary_objects(self) -> None:
        recorder = Recorder(RunStore(self.path), {}, self.empty)
        recorder.record(1, {"value": 1.0})
        with self.assertRaises(ValueError):
            recorder.record(2, {"value": np.ones(2)})
        with self.assertRaises(TypeError):
            recorder.record(2, {"value": object()})
        self.assertFalse((self.path / "progress.json").exists())

    def test_checkpoint_retention_is_bounded(self) -> None:
        store = RunStore(self.path)
        for step in range(5):
            store.checkpoint({"step": step}, self.empty, step)
        self.assertEqual(len(list((self.path / "checkpoints").iterdir())), 2)
        self.assertEqual(len(read_json(self.path / "progress.json")["checkpoints"]), 2)
