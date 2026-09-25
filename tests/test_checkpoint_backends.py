"""Exercise serializer independence, generation integrity, and native-state replay.

Backend fixtures are importable so these tests also exercise normal class-path
loading in spawned workers. No optional framework or GPU runtime is required.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments_wo_stress import load_config, run_experiment
from experiments_wo_stress.storage import (
    Recorder,
    RunStore,
    StorageError,
    atomic_json,
    digest_file,
    iter_completed_runs,
    read_json,
)
from experiments_wo_stress.storage.files import SCHEMA_VERSION, pack_state

from .checkpoint_backends import NativeState


def backend(name: str = "ShardedBackend", **params: object) -> dict:
    """Describe an importable test serializer using the public configuration shape."""
    return {"type": f"tests.checkpoint_backends:{name}", "params": params}


class _BackendTestCase(unittest.TestCase):
    """Give each generation scenario its own filesystem and empty result manifest."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {"schema": {}, "chunks": []}

    def generation(self, store: RunStore) -> Path:
        """Locate the latest generation committed by this store."""
        return store.directory / "checkpoints" / store.progress["checkpoints"][0]["generation"]

    def file_contents(self, directory: Path) -> dict[str, bytes]:
        """Snapshot files to detect any destructive recovery before a valid load."""
        return {
            str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*")
            if path.is_file()
        }

    def legacy_checkpoint(self, directory: Path, state: dict, step: int) -> None:
        """Build the original two-file generation without using the new writer."""
        generation = uuid.uuid4().hex
        checkpoint = directory / "checkpoints" / generation
        checkpoint.mkdir(parents=True)
        (directory / "results").mkdir()
        arrays = {}
        packed = pack_state(state, arrays)
        np.savez(checkpoint / "arrays.npz", **arrays)
        atomic_json(
            checkpoint / "state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state": packed,
                "results": self.manifest,
                "step": step,
            },
        )
        atomic_json(
            directory / "progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "paused",
                "step": step,
                "checkpoints": [
                    {
                        "generation": generation,
                        "state_sha256": digest_file(checkpoint / "state.json"),
                        "arrays_sha256": digest_file(checkpoint / "arrays.npz"),
                    }
                ],
            },
        )


class BackendRoundtripTests(_BackendTestCase):
    """Saved descriptors choose decoders independently of current write settings."""

    def test_native_values_nested_arrays_and_rng_states_roundtrip(self) -> None:
        rng = np.random.default_rng(17)
        rng.random(7)
        state = {
            "native": NativeState(b"\x00opaque\xff"),
            "nested": [NativeState(b"second"), (np.arange(6).reshape(2, 3), 2**100)],
            "rng": rng.bit_generator.state,
        }
        store = RunStore(self.root, checkpoint_backend=backend(label="saved"))
        store.checkpoint(state, self.manifest, 7)
        restored, manifest, step = RunStore(
            self.root, checkpoint_backend=backend(label="different current writer")
        ).restore()
        self.assertEqual((step, manifest), (7, self.manifest))
        self.assertEqual(restored["native"], state["native"])
        self.assertEqual(restored["nested"][0], state["nested"][0])
        self.assertIsInstance(restored["nested"][1], tuple)
        np.testing.assert_array_equal(restored["nested"][1][0], state["nested"][1][0])
        self.assertEqual(restored["nested"][1][1], 2**100)
        replay = np.random.default_rng()
        replay.bit_generator.state = restored["rng"]
        np.testing.assert_array_equal(rng.random(10), replay.random(10))
        envelope = read_json(self.generation(store) / "checkpoint.json")
        self.assertEqual(envelope["backend"]["params"], {"label": "saved"})
        self.assertIn("implementation", envelope["backend"])
        self.assertEqual(
            set(envelope["files"]),
            {
                "payload/blobs/0000.bin",
                "payload/blobs/0001.bin",
                "payload/numpy/state.json",
                "payload/numpy/arrays.npz",
            },
        )

    def test_mixed_generations_load_their_own_backend_and_parameters(self) -> None:
        store = RunStore(self.root, checkpoint_backend=backend(label="first"))
        store.checkpoint({"native": NativeState(b"first")}, self.manifest, 1)
        store = RunStore(self.root, checkpoint_backend=backend(label="second"))
        store.checkpoint({"native": NativeState(b"second")}, self.manifest, 2)
        (self.generation(store) / "payload/blobs/0000.bin").write_bytes(b"damaged")
        reader = RunStore(self.root, checkpoint_backend={"type": "numpy", "params": {}})
        state, _, step = reader.restore()
        self.assertEqual((state, step), ({"native": NativeState(b"first")}, 1))
        reader.checkpoint({"numerical": np.arange(4)}, self.manifest, 3)
        state, _, step = RunStore(self.root, checkpoint_backend=backend(label="unused")).restore()
        self.assertEqual(step, 3)
        np.testing.assert_array_equal(state["numerical"], np.arange(4))

    def test_legacy_generation_restores_and_survives_new_backend_fallback(self) -> None:
        legacy = {"value": np.arange(3), "tuple": (True, 2**70)}
        self.legacy_checkpoint(self.root, legacy, 5)
        store = RunStore(self.root, checkpoint_backend=backend())
        state, _, step = store.restore()
        self.assertEqual((step, state["tuple"]), (5, legacy["tuple"]))
        np.testing.assert_array_equal(state["value"], legacy["value"])
        store.checkpoint({"native": NativeState(b"new")}, self.manifest, 6)
        (self.generation(store) / "payload/numpy/arrays.npz").write_bytes(b"damaged")
        state, _, step = RunStore(self.root).restore()
        self.assertEqual(step, 5)
        np.testing.assert_array_equal(state["value"], legacy["value"])

    def test_default_numpy_backend_retains_execution_compression_behavior(self) -> None:
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                store = RunStore(self.root / str(compressed), compression=compressed)
                store.checkpoint({"weights": np.arange(20)}, self.manifest, 1)
                with zipfile.ZipFile(self.generation(store) / "arrays.npz") as archive:
                    expected = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
                    self.assertTrue(archive.infolist())
                    self.assertTrue(
                        all(item.compress_type == expected for item in archive.infolist())
                    )
                state, _, _ = RunStore(store.directory).restore()
                np.testing.assert_array_equal(state["weights"], np.arange(20))

    def test_numpy_compression_override_does_not_change_result_chunk_compression(self) -> None:
        store = RunStore(
            self.root,
            compression=True,
            checkpoint_backend={"type": "numpy", "params": {"compression": False}},
        )
        recorder = Recorder(store, {"buffer_bytes": 16}, self.manifest)
        recorder.record(1, {"value": 1.0})
        store.checkpoint({"weights": np.arange(20)}, recorder.manifest, 1)
        files = {
            self.generation(store) / "arrays.npz": zipfile.ZIP_STORED,
            self.root / "results" / "000000.npz": zipfile.ZIP_DEFLATED,
        }
        for path, compression in files.items():
            with self.subTest(file=path.name), zipfile.ZipFile(path) as archive:
                self.assertTrue(archive.infolist())
                self.assertTrue(
                    all(item.compress_type == compression for item in archive.infolist())
                )


class BackendIntegrityTests(_BackendTestCase):
    """File inventories, publication, and fallback remain owned by the run store."""

    def test_corrupting_any_nested_payload_file_or_envelope_falls_back(self) -> None:
        paths = (
            "payload/blobs/0000.bin",
            "payload/numpy/state.json",
            "payload/numpy/arrays.npz",
            "checkpoint.json",
        )
        for filename in paths:
            with self.subTest(file=filename):
                store = RunStore(
                    self.root / filename.replace("/", "_"), checkpoint_backend=backend()
                )
                store.checkpoint({"native": NativeState(b"one")}, self.manifest, 1)
                store.checkpoint({"native": NativeState(b"two")}, self.manifest, 2)
                (self.generation(store) / filename).write_bytes(b"corrupt")
                restored = RunStore(store.directory)
                state, _, step = restored.restore()
                self.assertEqual((state, step), ({"native": NativeState(b"one")}, 1))
                restored.checkpoint({"value": 3}, self.manifest, 3)
                (self.generation(restored) / "arrays.npz").write_bytes(b"corrupt again")
                self.assertEqual(RunStore(store.directory).restore()[2], 1)

    def test_missing_or_extra_payload_files_reject_the_latest_generation(self) -> None:
        for change in ("missing", "extra"):
            with self.subTest(change=change):
                store = RunStore(self.root / change, checkpoint_backend=backend())
                store.checkpoint({"native": NativeState(b"one")}, self.manifest, 1)
                store.checkpoint({"native": NativeState(b"two")}, self.manifest, 2)
                generation = self.generation(store)
                if change == "missing":
                    (generation / "payload/blobs/0000.bin").unlink()
                else:
                    (generation / "payload/blobs/uncommitted.bin").write_bytes(b"extra")
                state, _, step = RunStore(store.directory).restore()
                self.assertEqual((state, step), ({"native": NativeState(b"one")}, 1))

    def test_failed_save_and_failed_publication_preserve_last_commit(self) -> None:
        store = RunStore(self.root)
        store.checkpoint({"value": 1}, self.manifest, 1)
        before = (self.root / "progress.json").read_bytes()
        failing = RunStore(self.root, checkpoint_backend=backend("FailedSaveBackend"))
        with self.assertRaisesRegex(Exception, "injected backend save failure"):
            failing.checkpoint({"native": NativeState(b"partial")}, self.manifest, 2)
        self.assertEqual((self.root / "progress.json").read_bytes(), before)
        self.assertEqual(RunStore(self.root).restore()[0], {"value": 1})

        def fail_publication(path, value):
            if path.name == "progress.json":
                raise OSError("injected publication failure")
            atomic_json(path, value)

        with patch("experiments_wo_stress.storage.run.atomic_json", side_effect=fail_publication):
            with self.assertRaisesRegex(OSError, "publication failure"):
                RunStore(self.root, checkpoint_backend=backend()).checkpoint(
                    {"native": NativeState(b"uncommitted")}, self.manifest, 2
                )
        self.assertEqual((self.root / "progress.json").read_bytes(), before)
        self.assertEqual(RunStore(self.root).restore()[0], {"value": 1})

    def test_recursive_retention_keeps_only_committed_generations(self) -> None:
        store = RunStore(self.root, checkpoint_backend=backend(), keep_checkpoints=2)
        for step in range(5):
            store.checkpoint({"native": NativeState(bytes([step]))}, self.manifest, step)
        generations = list((self.root / "checkpoints").iterdir())
        self.assertEqual(len(generations), 2)
        self.assertEqual(RunStore(self.root).restore()[2], 4)
        self.assertEqual(
            {item.name for item in generations},
            {item["generation"] for item in store.progress["checkpoints"]},
        )

    def test_reserved_envelope_and_non_json_metadata_are_not_published(self) -> None:
        for name in ("ReservedEnvelopeBackend", "InvalidMetadataBackend"):
            with self.subTest(backend=name):
                directory = self.root / name
                store = RunStore(directory, checkpoint_backend=backend(name))
                with self.assertRaises((StorageError, ValueError, TypeError)):
                    store.checkpoint({"value": 1}, self.manifest, 1)
                self.assertFalse((directory / "progress.json").exists())

    @unittest.skipUnless(os.name == "posix", "requires POSIX symlinks and sockets")
    def test_symlinks_and_nonregular_files_are_not_published(self) -> None:
        for name in ("SymlinkBackend", "NonregularFileBackend"):
            with self.subTest(backend=name):
                directory = self.root / name
                store = RunStore(directory, checkpoint_backend=backend(name))
                with self.assertRaises((StorageError, ValueError, OSError)):
                    store.checkpoint({"value": 1}, self.manifest, 1)
                self.assertFalse((directory / "progress.json").exists())

    def test_decoder_failure_falls_back_to_an_older_backend(self) -> None:
        RunStore(self.root).checkpoint({"value": 1}, self.manifest, 1)
        RunStore(self.root, checkpoint_backend=backend("FailedLoadBackend")).checkpoint(
            {"native": NativeState(b"undecodable")}, self.manifest, 2
        )
        state, _, step = RunStore(self.root).restore()
        self.assertEqual((state, step), ({"value": 1}, 1))

    def test_unavailable_or_failed_decoder_preserves_saved_results_without_fallback(self) -> None:
        for unavailable in (False, True):
            with self.subTest(unavailable=unavailable):
                directory = self.root / str(unavailable)
                selected = backend() if unavailable else backend("FailedLoadBackend")
                store = RunStore(directory, checkpoint_backend=selected)
                recorder = Recorder(store, {"buffer_bytes": 16}, self.manifest)
                recorder.record(1, {"value": 1.0})
                store.checkpoint({"value": 1}, recorder.manifest, 1)
                recorder.record(2, {"value": 2.0})
                if unavailable:
                    checkpoint = self.generation(store) / "checkpoint.json"
                    envelope = read_json(checkpoint)
                    envelope["backend"]["type"] = "missing_checkpoint_backend:Unavailable"
                    atomic_json(checkpoint, envelope)
                    progress = read_json(directory / "progress.json")
                    progress["checkpoints"][0]["checkpoint_sha256"] = digest_file(checkpoint)
                    atomic_json(directory / "progress.json", progress)
                before = self.file_contents(directory)
                with self.assertRaises(StorageError):
                    RunStore(directory).restore()
                self.assertEqual(self.file_contents(directory), before)


class NativeCheckpointExecutionTests(_BackendTestCase):
    """Public execution passes native component state to backends and restores RNGs."""

    def configuration(self):
        """Write an ordinary online study whose learner owns unsupported native state."""
        path = self.root / "experiment.yml"
        path.write_text(
            yaml.safe_dump(
                {
                    "name": "native_checkpoint_replay",
                    "seed": 719,
                    "runs": [
                        {
                            "repetitions": 2,
                            "budget": {"steps": 11},
                            "protocol": {"type": "online"},
                            "data": {"type": "tests.checkpoint_backends:ReplayEnvironment"},
                            "algorithms": [{"type": "tests.checkpoint_backends:NativeLearner"}],
                        }
                    ],
                    "execution": {"checkpoint_steps": 2, "checkpoint_backend": backend()},
                    "recording": {"buffer_bytes": 32},
                }
            ),
            encoding="utf-8",
        )
        return load_config(path)

    def test_native_state_pause_and_spawned_resume_match_uninterrupted_results(self) -> None:
        config = self.configuration()
        reference, resumed = self.root / "reference", self.root / "resumed"
        report = run_experiment(config, reference)
        self.assertEqual((report.completed, report.failed), (2, 0), report.errors)
        report = run_experiment(config, resumed, max_steps=5)
        self.assertEqual((report.paused, report.failed), (2, 0), report.errors)
        retained = read_json(resumed / "metadata.json")["run_ids"]
        resumed_config = replace(
            config,
            execution={**config.execution, "checkpoint_backend": backend(label="new writer")},
        )
        report = run_experiment(resumed_config, resumed, workers=2)
        self.assertEqual((report.completed, report.failed), (2, 0), report.errors)
        metadata = read_json(resumed / "metadata.json")
        self.assertEqual(metadata["run_ids"], retained)
        self.assertEqual(len(list((resumed / "runs").iterdir())), 2)
        self.assertEqual(
            metadata["provenance"]["checkpoint_backend"]["params"], {"label": "new writer"}
        )
        expected = {spec.run_id: result for spec, result in iter_completed_runs(reference)}
        actual = {spec.run_id: result for spec, result in iter_completed_runs(resumed)}
        self.assertEqual(expected.keys(), actual.keys())
        for run_id, result in expected.items():
            self.assertEqual(result.keys(), actual[run_id].keys())
            for name, values in result.items():
                np.testing.assert_array_equal(values, actual[run_id][name])
            np.testing.assert_array_equal(actual[run_id]["step"], np.arange(1, 12))
        self.assertEqual(run_experiment(resumed_config, resumed).skipped, 2)

    def test_completed_analysis_does_not_resolve_the_saved_checkpoint_backend(self) -> None:
        config = self.configuration()
        output = self.root / "output"
        self.assertEqual(run_experiment(config, output).completed, 2)
        for directory in (output / "runs").iterdir():
            progress = read_json(directory / "progress.json")
            for entry in progress["checkpoints"]:
                checkpoint = directory / "checkpoints" / entry["generation"] / "checkpoint.json"
                envelope = read_json(checkpoint)
                envelope["backend"]["type"] = "missing_checkpoint_backend:Unavailable"
                atomic_json(checkpoint, envelope)
                entry["checkpoint_sha256"] = digest_file(checkpoint)
            atomic_json(directory / "progress.json", progress)
        self.assertEqual(len(list(iter_completed_runs(output))), 2)
        self.assertEqual(run_experiment(config, output).skipped, 2)
