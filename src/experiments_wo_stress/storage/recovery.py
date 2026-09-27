"""Versioned, sealed recovery snapshots for external persistence consumers.

Snapshots capture committed artifacts under the experiment lock. Cloud wrappers
transfer the resulting immutable inventory instead of interpreting private run,
checkpoint, or analysis layouts. Restoring bytes never bypasses EWS compatibility
checks or invokes a study's checkpoint decoder.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .experiment import ExperimentStore, load_instance
from .files import (
    SCHEMA_VERSION,
    StorageError,
    atomic_json,
    digest_file,
    fingerprint,
    read_json,
    sync_directory,
    validate_checkpoint_files,
)
from .run import validate_results
from .trajectories import pruning_history, pruning_receipt

RECOVERY_SCHEMA = "experiments-wo-stress/recovery"
RECOVERY_VERSION = 1
RECOVERY_MANIFEST = "recovery.json"


class _UnsupportedCheckpoint(StorageError):
    """A committed representation cannot safely be reduced to missing/corrupt bytes."""


def _relative_path(value: Any) -> str:
    """Accept only canonical portable paths contained by their inventory root."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or any(not character.isprintable() for character in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise StorageError(f"Unsafe recovery path: {value!r}")
    return value


def _files(
    directory: Path,
    *,
    skip_temporary: bool = False,
    directories: set[str] | None = None,
) -> dict[str, Path]:
    """Inventory regular files without following links or accepting special files."""
    result = {}

    def visit(parent: Path) -> None:
        if parent.is_symlink() or not parent.is_dir():
            raise StorageError(f"Recovery requires a real directory: {parent}")
        for path in sorted(parent.iterdir()):
            if skip_temporary and path.name.startswith("."):
                continue
            name = _relative_path(path.relative_to(directory).as_posix())
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                if directories is not None:
                    directories.add(name)
                visit(path)
            elif stat.S_ISREG(mode):
                result[name] = path
            else:
                raise StorageError(f"Recovery requires regular files: {path}")

    visit(directory)
    return result


def _checkpoint(directory: Path, entry: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate a published checkpoint envelope and payload without decoding state."""
    if not isinstance(entry, dict):
        raise StorageError("Invalid committed checkpoint entry")
    generation = entry.get("generation")
    if not isinstance(generation, str) or not re.fullmatch(r"[a-f0-9]{32}", generation):
        raise StorageError("Invalid committed checkpoint generation")
    checkpoint = directory / "checkpoints" / generation
    if checkpoint.is_symlink():
        raise StorageError(f"Checkpoint generation is a symbolic link: {checkpoint}")
    if "checkpoint_sha256" in entry:
        envelope = checkpoint / "checkpoint.json"
        if envelope.is_symlink() or digest_file(envelope) != entry["checkpoint_sha256"]:
            raise StorageError(f"Corrupt committed checkpoint envelope: {envelope}")
        snapshot = read_json(envelope)
        if snapshot.get("schema_version") != 1:
            raise _UnsupportedCheckpoint("Unsupported checkpoint envelope version")
        validate_checkpoint_files(checkpoint, snapshot["files"])
    else:
        for filename, key in (("state.json", "state_sha256"), ("arrays.npz", "arrays_sha256")):
            path = checkpoint / filename
            if path.is_symlink() or digest_file(path) != entry[key]:
                raise StorageError(f"Corrupt legacy checkpoint: {path}")
        snapshot = read_json(checkpoint / "state.json")
        if snapshot.get("schema_version") != SCHEMA_VERSION:
            raise _UnsupportedCheckpoint("Unsupported legacy checkpoint version")
    return checkpoint, snapshot


class _CommittedOutput:
    """Select committed file boundaries while preserving EWS's recovery decisions.

    Temporary objects and unreferenced checkpoint/result tails are excluded.
    Invalid checkpoints are recorded as omissions; unchanged progress metadata
    lets normal EWS recovery try the remaining committed predecessors. Completed
    results and derived caches must validate, so damage cannot become a new proof.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.directories: set[str] = set()
        self.files = _files(root, skip_temporary=True, directories=self.directories)
        self.omitted: list[dict[str, str]] = []
        self.pruned_runs: list[str] = []
        self.validated_instances: set[str] = set()
        self.schema_version = ExperimentStore(root).metadata()["schema_version"]

    def _exclude(self, directory: Path) -> None:
        if directory.exists():
            directories: set[str] = set()
            for path in _files(directory, skip_temporary=True, directories=directories).values():
                self.files.pop(path.relative_to(self.root).as_posix(), None)
            self.directories.discard(directory.relative_to(self.root).as_posix())
            for name in directories:
                self.directories.discard((directory / name).relative_to(self.root).as_posix())

    def _include(self, directory: Path) -> None:
        directories: set[str] = set()
        for path in _files(directory, directories=directories).values():
            self.files[path.relative_to(self.root).as_posix()] = path
        self.directories.add(directory.relative_to(self.root).as_posix())
        for name in directories:
            self.directories.add((directory / name).relative_to(self.root).as_posix())

    def select(self) -> dict[str, Path]:
        """Select committed generations, instances, trajectories, and derived artifacts."""
        runs = self.root / "runs"
        if runs.exists():
            for directory in sorted(runs.iterdir()):
                if not directory.name.startswith("."):
                    self._run(directory)
        instances = self.root / "instances"
        if instances.exists():
            for directory in sorted(instances.iterdir()):
                if not directory.name.startswith("."):
                    self._instance(directory.name)
        cache = self.root / "analysis" / "cache"
        if cache.exists():
            for collection in sorted(cache.iterdir()):
                if collection.name.startswith("."):
                    continue
                for directory in sorted(collection.iterdir()):
                    if not directory.name.startswith("."):
                        self._cache(directory)
        for name in [*self.files, *self.directories]:
            self.directories.update(
                parent.as_posix() for parent in PurePosixPath(name).parents if parent.parts
            )
        return self.files

    def _instance(self, instance_id: str) -> None:
        if instance_id not in self.validated_instances:
            load_instance(self.root, instance_id)
            self.validated_instances.add(instance_id)

    def _run(self, directory: Path) -> None:
        self._exclude(directory / "results")
        self._exclude(directory / "checkpoints")
        progress_path = directory / "progress.json"
        if not progress_path.exists():
            return
        progress = read_json(progress_path)
        if progress.get("schema_version") != SCHEMA_VERSION:
            raise StorageError(f"Unsupported run progress version: {progress_path}")
        metadata = read_json(directory / "metadata.json")
        instance_id = metadata.get("instance_id")
        if instance_id:
            self._instance(instance_id)
        elif self.schema_version != SCHEMA_VERSION and (
            progress.get("checkpoints") or progress.get("status") == "completed"
        ):
            raise StorageError(f"Run is missing its scientific instance reference: {directory}")
        pruning_history(directory)
        pruned = pruning_receipt(directory) is not None
        if pruned:
            self.pruned_runs.append(directory.name)
        completed = list(progress.get("completed_budgets", {}).values())
        if progress.get("status") == "completed":
            completed.append({"results": progress["results"]})
        manifests = [item["results"] for item in completed]
        entries = list(progress.get("checkpoints", []))
        entries.extend(item["checkpoint"] for item in completed if item.get("checkpoint"))
        visited = set()
        for entry in entries:
            generation = entry.get("generation") if isinstance(entry, dict) else None
            identifier = str(generation)
            if identifier in visited:
                continue
            visited.add(identifier)
            try:
                checkpoint, snapshot = _checkpoint(directory, entry)
                if not pruned:
                    validate_results(directory, snapshot["results"])
            except _UnsupportedCheckpoint:
                # Unknown representations remain evidence of saved work. Removing
                # them could incorrectly turn a decoder error into a fresh run.
                raise
            except (StorageError, OSError, KeyError, TypeError, ValueError) as exc:
                self.omitted.append(
                    {
                        "run": directory.name,
                        "generation": str(generation),
                        "reason": f"{type(exc).__name__}: checkpoint integrity validation failed",
                    }
                )
                continue
            self._include(checkpoint)
            manifests.append(snapshot["results"])
        if pruned:
            return
        for manifest in manifests:
            validate_results(directory, manifest)
            for chunk in manifest["chunks"]:
                path = directory / "results" / chunk["file"]
                self.files[path.relative_to(self.root).as_posix()] = path

    def _cache(self, directory: Path) -> None:
        manifest_path = directory / "cache.json"
        if not manifest_path.exists():
            self._exclude(directory)
            return
        manifest = read_json(manifest_path)
        if directory.name != fingerprint(manifest["identity"]):
            raise StorageError(f"Analysis cache identity mismatch: {directory}")
        files = _files(directory)
        files.pop("cache.json")
        if files.keys() != manifest["files"].keys():
            raise StorageError(f"Analysis cache inventory mismatch: {directory}")
        for name, path in files.items():
            if digest_file(path) != manifest["files"][name]:
                raise StorageError(f"Corrupt analysis cache: {path}")
        self._include(directory)


def _copy_files(
    files: dict[str, Path], destination: Path, directories: set[str]
) -> dict[str, dict[str, Any]]:
    """Copy and synchronize exact bytes, detecting a writer that ignored the lock."""
    inventory = {}
    destination.mkdir()
    for name in sorted(directories):
        (destination / name).mkdir(parents=True, exist_ok=True)
    for name, source in sorted(files.items()):
        expected = digest_file(source)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        if digest_file(target) != expected:
            raise StorageError(f"Artifact changed while creating recovery snapshot: {source}")
        inventory[name] = {"sha256": expected, "size": target.stat().st_size}
    copied_directories = [path for path in destination.rglob("*") if path.is_dir()]
    for directory in sorted(copied_directories, key=lambda path: len(path.parts), reverse=True):
        sync_directory(directory)
    sync_directory(destination)
    return inventory


def _destination(source: Path, destination: str | Path) -> Path:
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise StorageError(f"Recovery destination must not exist: {target}")
    resolved = target.resolve()
    if resolved.is_relative_to(source.resolve()) or source.resolve().is_relative_to(resolved):
        raise StorageError("Recovery source and destination must not contain one another")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def create_snapshot(output_dir: str | Path, destination: str | Path) -> Path:
    """Seal a recovery snapshot while holding the output's exclusive writer lock.

    ``destination`` must not exist and must be outside the output tree. An active
    writer causes StorageError; callers should gracefully pause execution first.
    The returned directory contains ``output/`` and a versioned ``recovery.json``
    checksum inventory. Publish that manifest last when transferring to object
    storage. No live-writer snapshot, checkpoint decoding, or environment
    compatibility is implied. Incomplete unpublished objects are omitted.
    """
    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise StorageError(f"Recovery requires an existing output directory: {root}")
    target = _destination(root, destination)
    temporary = Path(tempfile.mkdtemp(prefix=".recovery-", dir=target.parent))
    try:
        with ExperimentStore(root).lock():
            committed = _CommittedOutput(root)
            files = _copy_files(committed.select(), temporary / "output", committed.directories)
            manifest = {
                "schema": RECOVERY_SCHEMA,
                "schema_version": RECOVERY_VERSION,
                "files": files,
                "directories": sorted(committed.directories),
                "omitted_checkpoints": committed.omitted,
                "pruned_runs": committed.pruned_runs,
            }
            atomic_json(
                temporary / RECOVERY_MANIFEST, {**manifest, "snapshot_id": fingerprint(manifest)}
            )
            os.rename(temporary, target)
            sync_directory(target.parent)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate_snapshot(snapshot_dir: str | Path) -> dict[str, Any]:
    """Validate the version, safe paths, exact inventory, and checksums of a snapshot.

    Unknown versions fail closed. The returned manifest describes bytes safe to
    replicate, not proof that a particular study can resume in a new environment.
    Callers must treat a published snapshot as immutable while reading it.
    """
    root = Path(snapshot_dir)
    if root.is_symlink() or not root.is_dir():
        raise StorageError(f"Recovery snapshot must be a real directory: {root}")
    manifest = read_json(root / RECOVERY_MANIFEST)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != RECOVERY_SCHEMA
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != RECOVERY_VERSION
    ):
        raise StorageError("Unsupported EWS recovery contract; use a compatible EWS reader")
    if manifest.get("snapshot_id") != fingerprint(
        {key: value for key, value in manifest.items() if key != "snapshot_id"}
    ):
        raise StorageError("Recovery manifest identity mismatch")
    entries = manifest.get("files")
    if not isinstance(entries, dict) or "metadata.json" not in entries:
        raise StorageError("Recovery manifest is missing its output inventory")
    directory_names = manifest.get("directories")
    if (
        not isinstance(directory_names, list)
        or any(not isinstance(name, str) for name in directory_names)
        or len(set(directory_names)) != len(directory_names)
    ):
        raise StorageError("Invalid recovery directory inventory")
    for name in [*entries, *directory_names]:
        _relative_path(name)
    actual_directories: set[str] = set()
    actual = _files(root, directories=actual_directories)
    expected = {RECOVERY_MANIFEST, *(f"output/{name}" for name in entries)}
    expected_directories = {"output", *(f"output/{name}" for name in directory_names)}
    if actual.keys() != expected or actual_directories != expected_directories:
        raise StorageError("Recovery snapshot inventory mismatch")
    for name, descriptor in entries.items():
        path = actual[f"output/{name}"]
        if (
            not isinstance(descriptor, dict)
            or type(descriptor.get("size")) is not int
            or path.stat().st_size != descriptor["size"]
            or digest_file(path) != descriptor.get("sha256")
        ):
            raise StorageError(f"Missing or corrupt recovery artifact: {name}")
    return manifest


def restore_snapshot(snapshot_dir: str | Path, output_dir: str | Path) -> Path:
    """Atomically restore a verified snapshot into a new, nonexistent output tree.

    Existing outputs are never merged or overwritten, preventing stale objects
    from reviving intentionally pruned trajectories. The next EWS invocation owns
    normal variant, checkpoint, dependency, and environment validation.
    """
    root = Path(snapshot_dir)
    manifest = validate_snapshot(root)
    target = _destination(root, output_dir)
    temporary = Path(tempfile.mkdtemp(prefix=".restore-", dir=target.parent))
    try:
        inventory = _copy_files(
            {name: root / "output" / name for name in manifest["files"]},
            temporary / "output",
            set(manifest["directories"]),
        )
        if inventory != manifest["files"]:
            raise StorageError("Recovery snapshot changed during restore")
        os.rename(temporary / "output", target)
        sync_directory(target.parent)
        return target
    finally:
        shutil.rmtree(temporary)
