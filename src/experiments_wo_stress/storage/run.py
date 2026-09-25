"""Run checkpoints, validated result chunks, and bounded numerical recording."""

from __future__ import annotations

import re
import shutil
import time
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoints import (
    create_checkpoint_backend,
    describe_checkpoint_backend,
    json_metadata,
)
from .files import (
    SCHEMA_VERSION,
    StorageError,
    atomic_json,
    digest_file,
    read_json,
    seal_checkpoint_files,
    sync_directory,
    validate_checkpoint_files,
    write_arrays,
)

_CHECKPOINT_ENVELOPE_VERSION = 1


class _CheckpointDecodeError(StorageError):
    """A valid payload needs a backend or representation that could not be loaded."""


def _chunk_path(directory: Path, filename: str) -> Path:
    if not re.fullmatch(r"[0-9]{6,}\.npz", filename):
        raise StorageError(f"Invalid result chunk name: {filename!r}")
    return directory / "results" / filename


def iter_validated_chunks(
    directory: Path, manifest: dict[str, Any]
) -> Iterator[dict[str, np.ndarray]]:
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
    for _ in iter_validated_chunks(directory, manifest):
        pass


def read_result_prefix(
    directory: Path, manifest: dict[str, Any], completed_steps: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Load a completed prefix and identify exactly which committed rows it uses."""
    maximum_rows = sum(entry["rows"] for entry in manifest["chunks"])
    # Prefix requests allocate only their upper bound, not the entire longer run.
    maximum_rows = min(maximum_rows, completed_steps)
    records = {
        key: np.empty((maximum_rows, *description["shape"]), dtype=description["dtype"])
        for key, description in manifest["schema"].items()
    }
    offset = 0
    prefix_chunks = []
    for entry, arrays in zip(manifest["chunks"], iter_validated_chunks(directory, manifest)):
        used_rows = int(np.searchsorted(arrays["step"], completed_steps, side="right"))
        if used_rows:
            prefix_chunks.append({**entry, "used_rows": used_rows})
        for key, array in arrays.items():
            records[key][offset : offset + used_rows] = array[:used_rows]
        offset += used_rows
        if used_rows < len(arrays["step"]) or offset == maximum_rows:
            break
    records = {key: values[:offset] for key, values in records.items()}
    return records, prefix_chunks


class RunStore:
    """Own one retained run variant's progress, checkpoints, and result manifests.

    A Recorder writes numerical chunks; a RunSession supplies matching component
    and RNG state. ``checkpoint`` writes a generation, then atomically publishes
    progress pointing to it. ``restore`` finds a valid retained generation and
    discards uncommitted result tails. Completed prefixes stay pinned so extension
    and recovery cannot silently remove results already published as complete.

    The configured backend is created lazily for new writes. Saved generation
    descriptors choose readers independently; neither backend controls integrity,
    result boundaries, publication, or retention. Reading completed results does
    not construct a backend.
    """

    def __init__(
        self,
        directory: Path,
        *,
        compression: bool = False,
        keep_checkpoints: int = 2,
        checkpoint_backend: Mapping[str, Any] | None = None,
    ) -> None:
        self.directory = Path(directory)
        self.compression = compression
        self.keep_checkpoints = keep_checkpoints
        # Selection controls future writes only. Each saved generation supplies
        # its own reader, so changing representation never reinterprets old bytes.
        self.checkpoint_backend = checkpoint_backend
        self._writer = None
        self._writer_description = None
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
        """Whether the latest state is completed and all of its records are valid."""
        if self.progress["status"] != "completed":
            return False
        validate_results(self.directory, self.progress["results"])
        return True

    def restore(self) -> tuple[Mapping[str, Any] | None, dict[str, Any], int]:
        """Restore a valid committed generation; abandon uncommitted result tails."""
        state = None
        manifest: dict[str, Any] = {"chunks": [], "schema": {}}
        step = 0
        retained = []
        decode_errors = []
        entries = list(self.progress.get("checkpoints", []))
        completed_prefixes = sorted(
            self.progress.get("completed_budgets", {}).values(),
            key=lambda item: item["step"],
            reverse=True,
        )
        entries.extend(item["checkpoint"] for item in completed_prefixes if item.get("checkpoint"))
        # Try the current checkpoint first, then retained predecessors. Both state
        # files and their referenced result chunks must validate before resuming.
        for index, entry in enumerate(entries):
            try:
                state, manifest, step = self._load_checkpoint(entry)
                retained = entries[index : index + self.keep_checkpoints]
                break
            except _CheckpointDecodeError as exc:
                decode_errors.append(str(exc))
            except (StorageError, OSError, ValueError, KeyError, TypeError):
                continue
        # Unavailable/custom decoders are not evidence of an empty run. Preserve
        # artifacts rather than silently restarting and deleting their records.
        if state is None and decode_errors:
            raise StorageError(
                "No checkpoint backend could restore state: " + "; ".join(decode_errors)
            )
        # Falling back cannot discard observations already published as completed.
        if completed_prefixes and step < completed_prefixes[0]["step"]:
            raise StorageError(
                "No valid checkpoint covers the completed prefix; saved results were preserved"
            )
        self.progress["checkpoints"] = retained
        keep = {entry["file"] for entry in manifest["chunks"]}
        for path in (self.directory / "results").iterdir():
            if path.is_file() and path.name not in keep:
                path.unlink()
        return state, manifest, step

    def _load_checkpoint(
        self, entry: dict[str, Any]
    ) -> tuple[Mapping[str, Any], dict[str, Any], int]:
        """Validate a complete state/result boundary before invoking its saved decoder."""
        if not isinstance(entry, dict):
            raise StorageError("Invalid checkpoint entry")
        generation = entry.get("generation", "")
        if not isinstance(generation, str) or not re.fullmatch(r"[a-f0-9]{32}", generation):
            raise StorageError("Invalid checkpoint generation")
        directory = self.directory / "checkpoints" / generation
        if directory.is_symlink():
            raise StorageError("Checkpoint generations must not be symbolic links")
        if "checkpoint_sha256" in entry:
            envelope = directory / "checkpoint.json"
            if (
                envelope.is_symlink()
                or not envelope.is_file()
                or digest_file(envelope) != entry["checkpoint_sha256"]
            ):
                raise StorageError(f"Corrupt checkpoint envelope: {envelope}")
            snapshot = read_json(envelope)
            if snapshot["schema_version"] != _CHECKPOINT_ENVELOPE_VERSION:
                raise _CheckpointDecodeError(f"Unsupported checkpoint envelope version: {envelope}")
            validate_checkpoint_files(directory, snapshot["files"])
            description = snapshot["backend"]
            metadata = snapshot["backend_metadata"]
        else:
            # Legacy progress entries commit these two exact files. Their JSON
            # combines state with the result boundary; the encoding is unchanged.
            for name, key in (("state.json", "state_sha256"), ("arrays.npz", "arrays_sha256")):
                path = directory / name
                if path.is_symlink() or not path.is_file() or digest_file(path) != entry[key]:
                    raise StorageError(f"Corrupt legacy checkpoint payload: {path}")
            snapshot = read_json(directory / "state.json")
            if snapshot["schema_version"] != SCHEMA_VERSION:
                raise _CheckpointDecodeError("Unsupported legacy checkpoint state schema")
            description = {"type": "numpy", "params": {}}
            metadata = {"format_version": 1}
        if (
            not isinstance(description, dict)
            or not isinstance(description.get("type"), str)
            or not isinstance(description.get("params"), dict)
            or not isinstance(metadata, dict)
        ):
            raise StorageError("Invalid checkpoint backend descriptor or metadata")
        step = snapshot["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise StorageError("Checkpoint step must be a nonnegative integer")
        validate_results(self.directory, snapshot["results"])
        try:
            backend = create_checkpoint_backend(description)
            state = backend.load(directory, metadata)
            if not isinstance(state, Mapping):
                raise TypeError("Checkpoint backend must restore a logical run state mapping")
        except Exception as exc:
            raise _CheckpointDecodeError(
                f"Generation {generation}, backend {description.get('type')!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return state, snapshot["results"], step

    def _write_checkpoint(
        self, state: Mapping[str, Any], manifest: dict[str, Any], step: int
    ) -> tuple[dict[str, Any], int]:
        """Seal backend files and EWS metadata, leaving publication to checkpoint()."""
        if self._writer is None:
            self._writer_description = describe_checkpoint_backend(
                self.checkpoint_backend, compression=self.compression
            )
            self._writer = create_checkpoint_backend(self._writer_description)
        generation = uuid.uuid4().hex
        directory = self.directory / "checkpoints" / generation
        directory.mkdir()
        try:
            metadata = json_metadata(self._writer.save(state, directory))
            envelope = directory / "checkpoint.json"
            if envelope.exists() or envelope.is_symlink():
                raise StorageError("Checkpoint backend must not create reserved checkpoint.json")
            files = seal_checkpoint_files(directory)
            atomic_json(
                envelope,
                {
                    "schema_version": _CHECKPOINT_ENVELOPE_VERSION,
                    "backend": self._writer_description,
                    "backend_metadata": metadata,
                    "files": files,
                    "results": manifest,
                    "step": step,
                },
            )
            sync_directory(directory.parent)
            entry = {"generation": generation, "checkpoint_sha256": digest_file(envelope)}
            size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
            return entry, size
        except BaseException:
            # No progress publication has been attempted, so this incomplete
            # generation cannot be referenced by any durable checkpoint boundary.
            if directory.is_symlink():
                directory.unlink()
            else:
                shutil.rmtree(directory)
            raise

    def checkpoint(
        self,
        state: Mapping[str, Any],
        manifest: dict[str, Any],
        step: int,
        *,
        status: str = "running",
    ) -> None:
        """Publish state and records together before pruning old generations."""
        started = time.monotonic()
        entry, size = self._write_checkpoint(state, manifest, step)
        checkpoints = [entry, *self.progress.get("checkpoints", [])][: self.keep_checkpoints]
        progress = {
            **self.progress,
            "status": status,
            "step": step,
            "checkpoints": checkpoints,
            "checkpoint_seconds": time.monotonic() - started,
            "checkpoint_bytes": size,
            "checkpoint_count": self.progress.get("checkpoint_count", 0) + 1,
        }
        # This replacement is the commit point: all referenced files are complete.
        atomic_json(self.progress_path, progress)
        self.progress = progress
        # Publication comes first: failure here may leave extra generations, never
        # remove the only checkpoint referenced by durable progress metadata.
        retained = {item["generation"] for item in checkpoints}
        retained.update(
            item["checkpoint"]["generation"]
            for item in self.progress.get("completed_budgets", {}).values()
            if item.get("checkpoint")
        )
        for obsolete in (self.directory / "checkpoints").iterdir():
            if obsolete.is_dir() and obsolete.name not in retained:
                if obsolete.is_symlink():
                    obsolete.unlink()
                else:
                    shutil.rmtree(obsolete)

    def finish(self, manifest: dict[str, Any], step: int) -> None:
        """Pin a completed prefix and its checkpoint for later reads or extension."""
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
        """Record a failure while retaining every previously committed checkpoint."""
        atomic_json(self.directory / "failure.json", {"error": error, "traceback": traceback})
        self.progress["status"] = "failed"
        atomic_json(self.progress_path, self.progress)


class Recorder:
    """Select observations and buffer numerical columns for one active run.

    The first recorded row fixes field names, shapes, and dtypes. Later rows must
    match that schema, including after restoration. Buffers grow up to the memory
    target (or one oversized row), then flush to numbered NumPy chunks. A flush
    makes a chunk available; a RunStore checkpoint commits the resulting manifest
    together with the matching execution state.
    """

    def __init__(
        self, store: RunStore, options: Mapping[str, Any], manifest: dict[str, Any]
    ) -> None:
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
        """Append selected observations at the requested interval or final step."""
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
        """Write the buffered rows as the next immutable numerical chunk."""
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
        """Describe flushed chunks and their schema, excluding buffered observations."""
        return {"chunks": list(self.chunks), "schema": self.schema.copy()}
