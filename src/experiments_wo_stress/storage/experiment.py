"""Experiment requests, retained variants, and reusable scientific artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..study.specs import RunSpec
from .files import (
    EXPERIMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    StorageError,
    atomic_json,
    atomic_text,
    digest_file,
    fingerprint,
    read_json,
    sync_directory,
    write_arrays,
)
from .models import Instance, RunResult
from .run import RunStore, read_result_prefix


class ExperimentStore:
    """Own durable requests and retained variants beneath one experiment root.

    Execution supplies already prepared identities and provenance. This store
    validates and publishes their metadata without making scientific decisions.

    The active request selects which retained run variants to execute or analyze;
    older variants remain in ``runs`` for later reuse. The coordinator holds the
    experiment lock while publishing and executing a request. Per-run checkpoint
    mechanics belong to RunStore, while instance files are shared by content.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.metadata_path = self.root / "metadata.json"
        self.runs_path = self.root / "runs"
        self.requests_path = self.root / "requests"

    def metadata(self) -> dict[str, Any]:
        """Read recognized experiment metadata, including legacy analysis artifacts."""
        metadata = read_json(self.metadata_path)
        if metadata.get("schema_version") not in (SCHEMA_VERSION, EXPERIMENT_SCHEMA_VERSION):
            raise StorageError("Unsupported experiment artifact schema")
        return metadata

    def validate_execution_root(self) -> None:
        """Check that execution can safely publish a schema-2 request here."""
        if self.metadata_path.exists():
            previous = read_json(self.metadata_path)
            if previous.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
                raise ValueError(
                    "Legacy output is readable for analysis; use a new directory for schema-2 execution"
                )
        elif any(path.name != ".lock" for path in self.root.iterdir()):
            raise ValueError("Output directory is not empty and has no experiment metadata")

    def retained_variants(self) -> dict[str, Path]:
        """Return all run directories, including variants outside the active request."""
        return {path.name: path for path in self.runs_path.iterdir() if path.is_dir()}

    def publish_variant(self, storage_id: str, metadata: dict[str, Any]) -> None:
        """Create a variant, or verify that its stored identity still matches."""
        path = self.runs_path / storage_id / "metadata.json"
        if not path.exists():
            atomic_json(path, metadata)
        elif read_json(path).get("identity") != metadata["identity"]:
            raise StorageError(f"Stored identity mismatch for {storage_id}")

    def publish_request(
        self, request_id: str, request: dict[str, Any], resolved_config: str
    ) -> None:
        """Publish request history before selecting it as the active experiment state."""
        atomic_json(self.requests_path / f"{request_id}.json", request)
        atomic_json(self.metadata_path, request)
        atomic_text(self.root / "config.resolved.yml", resolved_config)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """An OS advisory lock is automatically released even after process termination."""
        root = self.root
        path = root / ".lock"
        with path.open("a+b") as stream:
            if os.name == "posix":
                import fcntl

                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise StorageError(f"Another executor is using {root}") from exc
            else:
                import msvcrt

                stream.seek(0)
                if not stream.read(1):
                    stream.write(b" ")
                    stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise StorageError(f"Another executor is using {root}") from exc
            try:
                yield
            finally:
                if os.name == "posix":
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                else:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)

    def inspect(self) -> dict[str, Any]:
        """Inspect stored status without importing or constructing scientific components."""
        root = self.root
        metadata = self.metadata()
        counts = dict.fromkeys(
            ("pending", "running", "paused", "completed", "failed", "corrupt"), 0
        )
        runs = []
        for run_id in metadata["run_ids"]:
            path = root / "runs" / run_id / "progress.json"
            try:
                progress = read_json(path) if path.exists() else {"status": "pending", "step": 0}
                status = progress["status"]
                if path.exists():
                    target = metadata.get("requests", {}).get(run_id, {}).get("budget_steps")
                    completion = RunStore(path.parent).completion(target)
                    if completion is not None:
                        progress = {
                            **progress,
                            "stored_status": status,
                            "stored_step": progress["step"],
                            "status": "completed",
                            "step": target or completion["step"],
                        }
                        status = "completed"
                counts[status] += 1
                runs.append({"run_id": run_id, **progress})
            except (StorageError, KeyError, ValueError) as exc:
                counts["corrupt"] += 1
                runs.append({"run_id": run_id, "status": "corrupt", "error": str(exc)})
        variants = len(self.retained_variants())
        return {
            "name": metadata["name"],
            "counts": counts,
            "runs": runs,
            "retained_variants": variants,
            "active_variants": len(metadata["run_ids"]),
        }


def inspect_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Inspect saved execution status without importing scientific components."""
    return ExperimentStore(output_dir).inspect()


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


def save_instance(root: Path, instance: Instance, *, compression: bool = False) -> str:
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
            sync_directory(directory.parent)
        except OSError:
            if not directory.is_dir():
                raise
            load_instance(root, instance_id)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return instance_id


def load_instance(root: Path, instance_id: str) -> Instance:
    """Load persisted scientific data without importing the simulator that created it."""
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


def iter_completed_runs(output_dir: str | Path) -> Iterator[tuple[RunSpec, RunResult]]:
    """Yield requested completed prefixes as RunResults, loading one run at a time."""
    experiment_store = ExperimentStore(output_dir)
    root = experiment_store.root
    metadata = experiment_store.metadata()
    version = metadata["schema_version"]
    for run_id in sorted(metadata["run_ids"]):
        if not re.fullmatch(r"[a-f0-9]{20,64}", run_id):
            raise StorageError("Invalid run identifier")
        directory = root / "runs" / run_id
        if not (directory / "progress.json").exists():
            continue
        run_metadata = read_json(directory / "metadata.json")
        run_spec = RunSpec.from_dict(metadata.get("requests", {}).get(run_id, run_metadata["spec"]))
        if version == SCHEMA_VERSION:
            if run_spec.run_id != run_id:
                raise StorageError(f"Run identity mismatch: {directory}")
            for key in ("simulation_fingerprint", "implementation_fingerprint"):
                if key in metadata and run_metadata.get(key) != metadata[key]:
                    raise StorageError(f"Run provenance mismatch: {directory}")
        elif run_metadata["spec"]["run_id"] != run_spec.run_id:
            raise StorageError(f"Run identity mismatch: {directory}")
        run_store = RunStore(directory)
        requested_steps = getattr(run_spec, "budget_steps", None)
        completion = run_store.completion(requested_steps, validate=False)
        if completion is None:
            continue
        manifest = completion["results"]
        completed_steps = completion["step"] if requested_steps is None else requested_steps
        records, prefix_chunks = read_result_prefix(directory, manifest, completed_steps)
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
            {key: values[-1] for key, values in records.items() if key != "step"}
            if completed_steps == 1
            else {}
        )
        yield (
            run_spec,
            RunResult(
                records=records,
                instance=instance,
                spec=run_spec,
                completed_steps=completed_steps,
                revision=revision,
                final_outputs=final_outputs,
            ),
        )
