"""Durable file publication and explicit JSON/NumPy checkpoint encoding."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping
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
    """Hash a canonical JSON value, rejecting nonfinite numbers."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def sync_directory(path: Path) -> None:
    """Persist directory entry changes on platforms that support directory fsync."""
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
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    """Publish a complete, human-readable JSON artifact."""
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    """Read JSON with a storage error that identifies unreadable or corrupt files."""
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise StorageError(f"Cannot read {path}: {exc}") from exc


def write_arrays(path: Path, arrays: Mapping[str, np.ndarray], compression: bool = False) -> None:
    """Atomically publish numerical arrays after syncing the complete NPZ file."""
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
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def pack_state(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Encode state explicitly, collecting numerical arrays for NPZ storage."""
    if isinstance(value, np.ndarray):
        name = f"a{len(arrays)}"
        arrays[name] = value
        return {"kind": "array", "name": name}
    if isinstance(value, np.generic):
        return pack_state(value.item(), arrays)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Checkpoint dictionaries require string keys")
        return {
            "kind": "dict",
            "items": [[key, pack_state(item, arrays)] for key, item in value.items()],
        }
    if isinstance(value, (list, tuple)):
        return {
            "kind": "tuple" if isinstance(value, tuple) else "list",
            "items": [pack_state(item, arrays) for item in value],
        }
    if isinstance(value, float) and not np.isfinite(value):
        return {"kind": "float", "value": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def unpack_state(value: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    """Restore the supported state values without loading executable pickle data."""
    if not isinstance(value, dict):
        return value
    kind = value["kind"]
    if kind == "array":
        return arrays[value["name"]]
    if kind == "dict":
        return {key: unpack_state(item, arrays) for key, item in value["items"]}
    if kind in ("list", "tuple"):
        items = [unpack_state(item, arrays) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "float":
        return float(value["value"])
    raise StorageError(f"Unknown checkpoint encoding: {kind!r}")
