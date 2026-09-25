"""Checkpoint representations, independent of generation publication and recovery.

Backends see logical run state directly. RunStore owns the surrounding transaction:
file inventory, durability, integrity, result boundaries, and the commit point.
"""

from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .files import (
    SCHEMA_VERSION,
    StorageError,
    atomic_json,
    digest_file,
    pack_state,
    read_json,
    unpack_state,
    write_arrays,
)


class CheckpointBackend(Protocol):
    """Encode a complete logical run state without owning its checkpoint transaction.

    Constructors receive only configured ``params``. ``save`` must finish all
    writes inside the supplied generation and flush and close its handles before
    returning finite JSON metadata. Nested regular files are supported; links,
    external payloads, and the reserved ``checkpoint.json`` filename are not.
    ``load`` is read-only and must reconstruct equivalent state using its saved
    representation metadata.
    Neither method may mutate the logical state or manage EWS progress/retention.

    State includes the algorithm, generator, protocol, and injected RNG snapshots.
    Values are passed through unchanged, allowing implementations to handle native
    tensors or other objects without first converting them to NumPy.
    """

    def save(self, state: Mapping[str, Any], directory: Path) -> dict[str, Any]:
        """Write this generation's complete payload and describe its representation."""
        ...

    def load(self, directory: Path, metadata: dict[str, Any]) -> Mapping[str, Any]:
        """Decode the saved payload using this generation's representation metadata."""
        ...


class NumPyCheckpointBackend:
    """Store explicit scalar/container state in JSON and numerical arrays in NPZ.

    This is EWS's original non-pickle encoding. ``compression`` controls only this
    backend's NPZ payload; RunStore separately controls numerical result chunks.
    Instances own no changing scientific state and can write many generations.
    """

    def __init__(self, *, compression: bool = False) -> None:
        if not isinstance(compression, bool):
            raise ValueError("NumPy checkpoint compression must be a boolean")
        self.compression = compression

    def save(self, state: Mapping[str, Any], directory: Path) -> dict[str, Any]:
        """Encode the original supported state values without generic object pickling."""
        arrays: dict[str, np.ndarray] = {}
        packed = pack_state(state, arrays)
        write_arrays(directory / "arrays.npz", arrays, self.compression)
        atomic_json(directory / "state.json", {"schema_version": SCHEMA_VERSION, "state": packed})
        return {"format_version": 1}

    def load(self, directory: Path, metadata: dict[str, Any]) -> Mapping[str, Any]:
        """Read both current payloads and the state portion of legacy snapshots."""
        if metadata.get("format_version") != 1:
            raise StorageError("Unsupported NumPy checkpoint representation version")
        snapshot = read_json(directory / "state.json")
        if snapshot["schema_version"] != SCHEMA_VERSION:
            raise StorageError("Unsupported NumPy checkpoint state schema")
        with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
            return unpack_state(snapshot["state"], arrays)


def json_metadata(value: Any) -> dict[str, Any]:
    """Copy backend metadata into finite JSON values, rejecting ambiguous keys."""

    def check(item: Any) -> None:
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("Checkpoint backend metadata requires string dictionary keys")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    if not isinstance(value, dict):
        raise TypeError("Checkpoint backend metadata must be a JSON mapping")
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _backend_class(type_name: str) -> type:
    """Resolve a backend only when checkpoint writing or restoration requires it."""
    if type_name == "numpy":
        return NumPyCheckpointBackend
    # Import lazily: component loading itself uses storage models. Importing it
    # while the storage package initializes would create a module cycle.
    from ..components.loading import resolve_type

    if not isinstance(type_name, str) or ":" not in type_name:
        raise ValueError("Checkpoint backend type must be 'numpy' or a module:Class path")
    backend_class = resolve_type(type_name)
    if not isinstance(backend_class, type):
        raise TypeError(f"Checkpoint backend {type_name!r} must identify a class")
    for method in ("save", "load"):
        if not callable(getattr(backend_class, method, None)):
            raise TypeError(f"Checkpoint backend {type_name!r} must implement {method}()")
    return backend_class


def describe_checkpoint_backend(
    options: Mapping[str, Any] | None = None, *, compression: bool = False
) -> dict[str, Any]:
    """Describe the effective writer and its source files for operational provenance.

    These fingerprints are diagnostic, not scientific compatibility keys. A
    backend's loader owns compatibility with its saved representation versions;
    simulation source and tracked-input checks remain independent and unchanged.
    """
    options = {"type": "numpy", "params": {}} if options is None else options
    type_name = options["type"]
    backend_class = _backend_class(type_name)
    params = copy.deepcopy(options.get("params", {}))
    if backend_class is NumPyCheckpointBackend:
        params.setdefault("compression", compression)
    # Check constructor parameters without creating backend resources in preflight.
    inspect.signature(backend_class).bind(**params)
    implementation = {}
    for base in backend_class.__mro__:
        if base is object:
            continue
        try:
            source = inspect.getsourcefile(base)
        except TypeError:
            source = None
        if source:
            implementation[f"module:{base.__module__}"] = digest_file(Path(source))
            for dependency in getattr(base, "dependency_files", ()):
                path = (Path(source).parent / dependency).resolve()
                implementation[f"dependency:{path}"] = digest_file(path)
    return {"type": type_name, "params": params, "implementation": implementation}


def create_checkpoint_backend(description: Mapping[str, Any]) -> CheckpointBackend:
    """Construct a writer or saved-generation reader with its own recorded params."""
    return _backend_class(description["type"])(**copy.deepcopy(description["params"]))
