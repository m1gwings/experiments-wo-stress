"""Atomic local artifacts, explicit checkpoint encoding, and buffered numerical records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 2


class StorageError(RuntimeError):
    """An artifact is incompatible, incomplete, or corrupt."""


def digest_file(path: Path) -> str:
    """Hash without loading an entire artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_text(path: Path, value: str) -> None:
    """Publish text only after the replacement file is complete and synced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete, human-readable JSON artifact."""
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise StorageError(f"Cannot read {path}: {exc}") from exc


def write_arrays(path: Path, arrays: Mapping[str, np.ndarray], compression: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    for name, array in arrays.items():
        if np.asarray(array).dtype.kind not in "biufc":
            raise TypeError(f"Array {name!r} must have a numerical dtype, not {array.dtype}")
    try:
        with temporary.open("wb") as stream:
            writer = np.savez_compressed if compression else np.savez
            writer(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _pack(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(value, np.ndarray):
        name = f"a{len(arrays)}"
        arrays[name] = value
        return {"kind": "array", "name": name}
    if isinstance(value, np.generic):
        return _pack(value.item(), arrays)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Checkpoint dictionaries require string keys")
        return {
            "kind": "dict",
            "items": [[key, _pack(item, arrays)] for key, item in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [_pack(item, arrays) for item in value],
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"kind": "float", "value": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _unpack(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "array":
        return arrays[value["name"]]
    if kind == "dict":
        return {key: _unpack(item, arrays) for key, item in value["items"]}
    if kind in ("list", "tuple"):
        items = [_unpack(item, arrays) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "float":
        return float(value["value"])
    raise StorageError(f"Unknown checkpoint encoding: {kind!r}")


def _chunk_path(directory: Path, filename: str) -> Path:
    if not re.fullmatch(r"[0-9]{6,}\.npz", filename):
        raise StorageError(f"Invalid result chunk name: {filename!r}")
    return directory / "results" / filename


def _validated_chunks(directory: Path, manifest: dict[str, Any]) -> Iterator[dict[str, np.ndarray]]:
    """Read each numerical chunk once while validating its integrity and schema."""
    previous_step = 0
    schema = manifest["schema"]
    for index, entry in enumerate(manifest["chunks"]):
        if entry["file"] != f"{index:06d}.npz":
            raise StorageError("Result chunk sequence is incomplete")
        path = _chunk_path(directory, entry["file"])
        if not path.is_file() or digest_file(path) != entry["sha256"]:
            raise StorageError(f"Missing or corrupt result chunk: {path}")
        try:
            with np.load(path, allow_pickle=False) as arrays:
                if set(arrays.files) != set(schema):
                    raise StorageError(f"Result fields do not match schema: {path}")
                columns = {key: arrays[key] for key in schema}
                for key, description in schema.items():
                    array = columns[key]
                    expected = (entry["rows"], *description["shape"])
                    if array.shape != expected or array.dtype.str != description["dtype"]:
                        raise StorageError(f"Invalid result shape/dtype for {key!r}: {path}")
                steps = columns["step"]
                if len(steps) == 0 or steps[0] <= previous_step or np.any(np.diff(steps) <= 0):
                    raise StorageError(f"Invalid result step ordering: {path}")
                previous_step = int(steps[-1])
                yield columns
        except (OSError, ValueError, KeyError) as exc:
            raise StorageError(f"Cannot validate result chunk {path}: {exc}") from exc


def validate_results(directory: Path, manifest: dict[str, Any]) -> None:
    """Validate committed chunks, their fixed schema, and strictly increasing steps."""
    for _ in _validated_chunks(directory, manifest):
        pass


class RunStore:
    """One writer's view of a run directory and its committed generations."""

    def __init__(self, directory: Path, *, compression: bool = False, keep_checkpoints: int = 2):
        self.directory = Path(directory)
        self.compression = compression
        self.keep_checkpoints = keep_checkpoints
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "results").mkdir(exist_ok=True)
        (self.directory / "checkpoints").mkdir(exist_ok=True)
        self.progress_path = self.directory / "progress.json"
        self.progress = (
            read_json(self.progress_path)
            if self.progress_path.exists()
            else {
                "schema_version": SCHEMA_VERSION,
                "status": "pending",
                "checkpoints": [],
                "step": 0,
            }
        )
        if self.progress.get("schema_version") != SCHEMA_VERSION:
            raise StorageError(f"Unsupported run schema in {self.progress_path}")

    def completion(
        self, target: int | None = None, *, validate: bool = True
    ) -> dict[str, Any] | None:
        """Return a committed result covering the requested budget, even during extension."""
        candidates = list(self.progress.get("completed_budgets", {}).values())
        if self.progress.get("status") == "completed":
            candidates.append({"step": self.progress["step"], "results": self.progress["results"]})
        candidates = [item for item in candidates if target is None or item["step"] >= target]
        if not candidates:
            return None
        result = (
            min(candidates, key=lambda item: item["step"])
            if target is not None
            else max(candidates, key=lambda item: item["step"])
        )
        if validate:
            validate_results(self.directory, result["results"])
        return result

    def completed(self) -> bool:
        if self.progress["status"] != "completed":
            return False
        validate_results(self.directory, self.progress["results"])
        return True

    def restore(self) -> tuple[dict[str, Any] | None, dict[str, Any], int]:
        """Restore a valid committed generation; abandon uncommitted result tails."""
        state = None
        manifest: dict[str, Any] = {"chunks": [], "schema": {}}
        step = 0
        retained = []
        entries = list(self.progress.get("checkpoints", []))
        endpoints = sorted(
            self.progress.get("completed_budgets", {}).values(),
            key=lambda item: item["step"],
            reverse=True,
        )
        entries.extend(item["checkpoint"] for item in endpoints if item.get("checkpoint"))
        for index, entry in enumerate(entries):
            generation = entry.get("generation", "")
            if not re.fullmatch(r"[a-f0-9]{32}", generation):
                raise StorageError("Invalid checkpoint generation")
            directory = self.directory / "checkpoints" / generation
            try:
                if (
                    digest_file(directory / "state.json") != entry["state_sha256"]
                    or digest_file(directory / "arrays.npz") != entry["arrays_sha256"]
                ):
                    continue
                snapshot = read_json(directory / "state.json")
                if snapshot["schema_version"] != SCHEMA_VERSION:
                    continue
                validate_results(self.directory, snapshot["results"])
                with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
                    state = _unpack(snapshot["state"], arrays)
                manifest, step = snapshot["results"], snapshot["step"]
                retained = entries[index : index + self.keep_checkpoints]
                break
            except (StorageError, OSError, ValueError, KeyError):
                continue
        if endpoints and step < endpoints[0]["step"]:
            raise StorageError(
                "No valid checkpoint covers the completed prefix; saved results were preserved"
            )
        self.progress["checkpoints"] = retained
        keep = {entry["file"] for entry in manifest["chunks"]}
        for path in (self.directory / "results").iterdir():
            if path.is_file() and path.name not in keep:
                path.unlink()
        return state, manifest, step

    def checkpoint(
        self, state: dict[str, Any], manifest: dict[str, Any], step: int, *, status: str = "running"
    ) -> None:
        started = time.monotonic()
        generation = uuid.uuid4().hex
        directory = self.directory / "checkpoints" / generation
        directory.mkdir()
        arrays: dict[str, np.ndarray] = {}
        packed = _pack(state, arrays)
        write_arrays(directory / "arrays.npz", arrays, self.compression)
        atomic_json(
            directory / "state.json",
            {
                "schema_version": SCHEMA_VERSION,
                "state": packed,
                "results": manifest,
                "step": step,
            },
        )
        _sync_directory(directory.parent)
        entry = {
            "generation": generation,
            "state_sha256": digest_file(directory / "state.json"),
            "arrays_sha256": digest_file(directory / "arrays.npz"),
        }
        checkpoints = [entry, *self.progress.get("checkpoints", [])][: self.keep_checkpoints]
        size = sum(path.stat().st_size for path in directory.iterdir())
        progress = {
            **self.progress,
            "status": status,
            "step": step,
            "checkpoints": checkpoints,
            "checkpoint_seconds": time.monotonic() - started,
            "checkpoint_bytes": size,
            "checkpoint_count": self.progress.get("checkpoint_count", 0) + 1,
        }
        atomic_json(self.progress_path, progress)
        self.progress = progress
        retained = {item["generation"] for item in checkpoints}
        retained.update(
            item["checkpoint"]["generation"]
            for item in self.progress.get("completed_budgets", {}).values()
            if item.get("checkpoint")
        )
        for obsolete in (self.directory / "checkpoints").iterdir():
            if obsolete.is_dir() and obsolete.name not in retained:
                for path in obsolete.iterdir():
                    path.unlink()
                obsolete.rmdir()

    def finish(self, manifest: dict[str, Any], step: int) -> None:
        completed = dict(self.progress.get("completed_budgets", {}))
        completed[str(step)] = {
            "step": step,
            "results": manifest,
            "checkpoint": next(iter(self.progress.get("checkpoints", [])), None),
        }
        progress = {
            **self.progress,
            "status": "completed",
            "results": manifest,
            "step": step,
            "completed_budgets": completed,
        }
        atomic_json(self.progress_path, progress)
        self.progress = progress
        (self.directory / "failure.json").unlink(missing_ok=True)

    def fail(self, error: str, traceback: str) -> None:
        atomic_json(self.directory / "failure.json", {"error": error, "traceback": traceback})
        self.progress["status"] = "failed"
        atomic_json(self.progress_path, self.progress)


class Recorder:
    """Buffer fixed-schema numerical columns with a configurable memory target."""

    def __init__(self, store: RunStore, options: Mapping[str, Any], manifest: dict[str, Any]):
        self.store = store
        self.every_steps = options.get("every_steps", 1)
        self.fields = options.get("fields")
        self.budget = options.get("buffer_bytes", 16 * 1024 * 1024)
        self.schema = manifest["schema"].copy()
        self.chunks = list(manifest["chunks"])
        self.buffers: dict[str, np.ndarray] = {}
        self.count = 0
        self.capacity = 0
        self.maximum_rows = 0

    def record(self, step: int, observations: Mapping[str, Any], *, final: bool = False) -> None:
        if step % self.every_steps and not final:
            return
        if "step" in observations:
            raise ValueError("'step' is reserved for the protocol's completed-step counter")
        selected = sorted(observations) if self.fields is None else self.fields
        values = {key: np.asarray(observations[key]) for key in selected}
        values["step"] = np.asarray(step, dtype=np.int64)
        for key, value in values.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid measurement field name: {key!r}")
            if value.dtype.kind not in "biufc":
                raise TypeError(f"Measurement {key!r} must be numerical")
        schema = {
            key: {"dtype": value.dtype.str, "shape": list(value.shape)}
            for key, value in values.items()
        }
        if self.schema and schema != self.schema:
            raise ValueError("Measurement fields, shapes, or dtypes changed during the run")
        self.schema = schema
        if not self.buffers:
            row_bytes = sum(value.nbytes for value in values.values())
            self.maximum_rows = max(1, self.budget // max(1, row_bytes))
            self.capacity = min(1024, self.maximum_rows)
            self.buffers = {
                key: np.empty((self.capacity, *value.shape), dtype=value.dtype)
                for key, value in values.items()
            }
        if self.count == self.capacity:
            new_capacity = min(self.capacity * 2, self.maximum_rows)
            for key, buffer in self.buffers.items():
                grown = np.empty((new_capacity, *buffer.shape[1:]), dtype=buffer.dtype)
                grown[: self.count] = buffer[: self.count]
                self.buffers[key] = grown
            self.capacity = new_capacity
        for key, value in values.items():
            self.buffers[key][self.count] = value
        self.count += 1
        if self.count == self.maximum_rows:
            self.flush()

    def flush(self) -> None:
        if not self.count:
            return
        filename = f"{len(self.chunks):06d}.npz"
        path = self.store.directory / "results" / filename
        write_arrays(
            path,
            {key: value[: self.count] for key, value in self.buffers.items()},
            self.store.compression,
        )
        self.chunks.append({"file": filename, "sha256": digest_file(path), "rows": self.count})
        self.count = 0

    @property
    def manifest(self) -> dict[str, Any]:
        return {"chunks": list(self.chunks), "schema": self.schema.copy()}


def _instance_fingerprint(descriptor: dict[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(fingerprint(descriptor).encode())
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        digest.update(
            fingerprint({"name": name, "shape": array.shape, "dtype": array.dtype.str}).encode()
        )
        raw = memoryview(array.reshape(-1)).cast("B")
        for offset in range(0, len(raw), 1024 * 1024):
            digest.update(raw[offset : offset + 1024 * 1024])
    return digest.hexdigest()


def save_instance(root: Path, instance: Any, *, compression: bool = False) -> str:
    """Deduplicate immutable scientific instances by metadata and numerical contents."""
    descriptor = instance.to_dict()
    arrays = descriptor.pop("arrays")
    instance_id = _instance_fingerprint(descriptor, arrays)
    directory = root / "instances" / instance_id
    if directory.exists():
        load_instance(root, instance_id)
        return instance_id
    temporary = root / "instances" / f".{instance_id}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        write_arrays(temporary / "arrays.npz", arrays, compression)
        atomic_json(
            temporary / "metadata.json",
            {
                **descriptor,
                "id": instance_id,
                "arrays_sha256": digest_file(temporary / "arrays.npz"),
            },
        )
        try:
            os.rename(temporary, directory)
            _sync_directory(directory.parent)
        except OSError:
            if not directory.is_dir():
                raise
            load_instance(root, instance_id)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return instance_id


def load_instance(root: Path, instance_id: str) -> Any:
    """Load persisted scientific data without importing the simulator that created it."""
    from .artifacts import Instance

    if not re.fullmatch(r"[a-f0-9]{64}", instance_id):
        raise StorageError("Invalid instance identifier")
    directory = root / "instances" / instance_id
    descriptor = read_json(directory / "metadata.json")
    if descriptor.pop("id") != instance_id or digest_file(
        directory / "arrays.npz"
    ) != descriptor.pop("arrays_sha256"):
        raise StorageError(f"Corrupt instance {instance_id}")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        descriptor["arrays"] = {name: arrays[name] for name in arrays.files}
    if (
        _instance_fingerprint(
            {key: value for key, value in descriptor.items() if key != "arrays"},
            descriptor["arrays"],
        )
        != instance_id
    ):
        raise StorageError(f"Instance contents do not match identity {instance_id}")
    return Instance.from_dict(descriptor)


def iter_completed_runs(output_dir: str | Path) -> Iterator[tuple[Any, Any]]:
    """Yield requested completed prefixes as RunResults, loading one run at a time."""
    from .artifacts import Instance, RunResult
    from .jobs import RunSpec

    root = Path(output_dir)
    metadata = read_json(root / "metadata.json")
    version = metadata.get("schema_version")
    if version not in (SCHEMA_VERSION, EXPERIMENT_SCHEMA_VERSION):
        raise StorageError("Unsupported experiment artifact schema")
    for run_id in sorted(metadata["run_ids"]):
        if not re.fullmatch(r"[a-f0-9]{20,64}", run_id):
            raise StorageError("Invalid run identifier")
        directory = root / "runs" / run_id
        if not (directory / "progress.json").exists():
            continue
        run_metadata = read_json(directory / "metadata.json")
        spec = RunSpec.from_dict(metadata.get("requests", {}).get(run_id, run_metadata["spec"]))
        if version == SCHEMA_VERSION:
            if spec.run_id != run_id:
                raise StorageError(f"Run identity mismatch: {directory}")
            for key in ("simulation_fingerprint", "implementation_fingerprint"):
                if key in metadata and run_metadata.get(key) != metadata[key]:
                    raise StorageError(f"Run provenance mismatch: {directory}")
        elif run_metadata["spec"]["run_id"] != spec.run_id:
            raise StorageError(f"Run identity mismatch: {directory}")
        store = RunStore(directory)
        target = getattr(spec, "budget_steps", None)
        completion = store.completion(target, validate=False)
        if completion is None:
            continue
        manifest = completion["results"]
        completed_steps = completion["step"] if target is None else target
        total = sum(entry["rows"] for entry in manifest["chunks"])
        # Prefix requests allocate only their upper bound, not the entire longer run.
        total = min(total, completed_steps)
        results = {
            key: np.empty((total, *description["shape"]), dtype=description["dtype"])
            for key, description in manifest["schema"].items()
        }
        offset = 0
        prefix_chunks = []
        for entry, arrays in zip(manifest["chunks"], _validated_chunks(directory, manifest)):
            count = int(np.searchsorted(arrays["step"], completed_steps, side="right"))
            if count:
                prefix_chunks.append({**entry, "used_rows": count})
            for key, array in arrays.items():
                results[key][offset : offset + count] = array[:count]
            offset += count
            if count < len(arrays["step"]) or offset == total:
                break
        results = {key: values[:offset] for key, values in results.items()}
        instance_id = run_metadata.get("instance_id")
        instance = load_instance(root, instance_id) if instance_id else Instance()
        revision = fingerprint(
            {
                "chunks": prefix_chunks,
                "instance": instance_id,
                "steps": completed_steps,
                "schema": manifest["schema"],
            }
        )
        final_outputs = (
            {key: values[-1] for key, values in results.items() if key != "step"}
            if completed_steps == 1
            else {}
        )
        yield (
            spec,
            RunResult(
                records=results,
                instance=instance,
                spec=spec,
                completed_steps=completed_steps,
                revision=revision,
                final_outputs=final_outputs,
            ),
        )
