"""Generic scientific instances and the results of executing them."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .jobs import RunSpec


def _readonly_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError("Artifact arrays must have a numerical dtype, without Python objects")
    # A bytes backing store cannot be made writable through setflags().
    return np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)


def _freeze(value: Any, *, allow_arrays: bool = False) -> Any:
    if isinstance(value, np.ndarray):
        if not allow_arrays:
            raise TypeError("Put numerical arrays in Instance.arrays, not its metadata")
        return _readonly_array(value)
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Artifact mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze(item, allow_arrays=allow_arrays) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, allow_arrays=allow_arrays) for item in value)
    raise TypeError("Artifact metadata must contain finite JSON values")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Instance:
    """Immutable problem definition shared by simulation and later analysis.

    Metadata contains plain structured values. Arrays hold numerical scientific
    inputs such as arm means or a time-dependent reward schedule. Constructors
    copy array inputs once; later consumers can safely share the frozen arrays.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict)
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)
    kind: str = "generic"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Mapping) or not isinstance(self.arrays, Mapping):
            raise TypeError("Instance metadata and arrays must be mappings")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("Instance kind must be a nonempty string")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("Instance schema_version must be a positive integer")
        if not all(isinstance(key, str) and key for key in self.arrays):
            raise ValueError("Instance array names must be nonempty strings")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        object.__setattr__(
            self,
            "arrays",
            MappingProxyType({name: _readonly_array(value) for name, value in self.arrays.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return serializable metadata and immutable array references."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "metadata": _thaw(self.metadata),
            "arrays": dict(self.arrays),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Instance:
        return cls(**dict(value))

    def __reduce__(self):
        return (type(self).from_dict, (self.to_dict(),))


@dataclass(frozen=True)
class RunResult(Mapping[str, np.ndarray]):
    """Saved records plus the instance and provenance required to interpret them.

    Mapping access remains compatible with metrics written for a dictionary of
    arrays. Scientific metrics can also inspect ``instance`` and ``final_outputs``.
    Record arrays are exposed as read-only views without duplicating loaded data.
    """

    records: Mapping[str, np.ndarray]
    instance: Instance
    spec: RunSpec
    completed_steps: int
    revision: str | int = 0
    final_outputs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.completed_steps, bool)
            or not isinstance(self.completed_steps, int)
            or self.completed_steps < 0
        ):
            raise ValueError("completed_steps must be a nonnegative integer")
        if not isinstance(self.instance, Instance) or not isinstance(self.spec, RunSpec):
            raise TypeError("RunResult requires an Instance and a RunSpec")
        arrays = {}
        for name, value in self.records.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Record names must be nonempty strings")
            array = np.asarray(value).view()
            if array.dtype.kind not in "biufc":
                raise TypeError("Record arrays must be numerical")
            array.flags.writeable = False
            arrays[name] = array
        object.__setattr__(self, "records", MappingProxyType(arrays))
        object.__setattr__(self, "final_outputs", _freeze(self.final_outputs, allow_arrays=True))

    def __getitem__(self, name: str) -> np.ndarray:
        return self.records[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)
