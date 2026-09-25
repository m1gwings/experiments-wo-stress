"""Importable checkpoint adapters and native-state science for backend tests.

These adapters use only the standard library and NumPy. ``NativeState`` stands
in for a framework value that the default NumPy encoder cannot serialize.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments_wo_stress import Feedback
from experiments_wo_stress.storage import NumPyCheckpointBackend


@dataclass(frozen=True)
class NativeState:
    """Hold opaque state bytes without offering an implicit NumPy conversion."""

    payload: bytes


class ShardedBackend:
    """Store native values as nested binary files alongside NumPy-encoded state.

    A constructor label is persisted and checked during loading, exposing any
    attempt to decode an old generation with the current writer's parameters.
    Each save completes its writes before returning its small JSON descriptor.
    """

    def __init__(self, *, label: str = "original") -> None:
        self.label = label

    def save(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        """Pass supported values to NumPy and preserve native values in binary shards."""
        counter = 0

        def encode(value: Any) -> Any:
            nonlocal counter
            if isinstance(value, NativeState):
                relative = f"payload/blobs/{counter:04d}.bin"
                counter += 1
                path = directory / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value.payload)
                return {"native_file": relative}
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(encode(item) for item in value)
            return value

        values = encode(state)
        numpy_directory = directory / "payload" / "numpy"
        numpy_directory.mkdir(parents=True, exist_ok=True)
        metadata = NumPyCheckpointBackend().save(values, numpy_directory)
        return {"label": self.label, "numpy": metadata}

    def load(self, directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        """Restore native objects using this generation's saved constructor label."""
        if metadata["label"] != self.label:
            raise ValueError("checkpoint was loaded with another backend's parameters")

        def decode(value: Any) -> Any:
            if isinstance(value, dict):
                if set(value) == {"native_file"}:
                    return NativeState((directory / value["native_file"]).read_bytes())
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(decode(item) for item in value)
            return value

        values = NumPyCheckpointBackend().load(directory / "payload" / "numpy", metadata["numpy"])
        return decode(values)


class FailedSaveBackend(ShardedBackend):
    """Leave incomplete nested files and fail before any checkpoint publication."""

    def save(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        super().save(state, directory)
        (directory / "partial.bin").write_bytes(b"unfinished")
        raise RuntimeError("injected backend save failure")


class FailedLoadBackend(ShardedBackend):
    """Produce intact files whose decoder reports an unsupported saved state."""

    def load(self, directory: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("injected backend load failure")


class InvalidMetadataBackend(ShardedBackend):
    """Return a native object where the backend contract requires JSON metadata."""

    def save(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        super().save(state, directory)
        return {"invalid": NativeState(b"not JSON")}


class ReservedEnvelopeBackend(ShardedBackend):
    """Attempt to claim the filename reserved for EWS's commit envelope."""

    def save(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        metadata = super().save(state, directory)
        (directory / "checkpoint.json").write_text("{}", encoding="utf-8")
        return metadata


class SymlinkBackend(ShardedBackend):
    """Create a symbolic link that must not enter a retained payload inventory."""

    def save(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        metadata = super().save(state, directory)
        (directory / "linked-state").symlink_to("payload/numpy/state.json")
        return metadata


class NonregularFileBackend(ShardedBackend):
    """Create a socket entry to verify rejection without blocking on a FIFO."""

    def save(self, state: dict[str, Any], directory: Path) -> dict[str, Any]:
        metadata = super().save(state, directory)
        with socket.socket(socket.AF_UNIX) as endpoint:
            endpoint.bind(str(directory / "socket"))
        return metadata


class NativeLearner:
    """Keep learned state in an opaque value and consume the injected RNG."""

    supports_extension = True

    def __init__(self, *, rng: Any) -> None:
        self.rng = rng
        self.total = 0.0
        self.steps = 0
        self.rng.random(2)

    def act(self, context: Any = None) -> float:
        """Choose a numerical action from learned state and an independent draw."""
        return float(self.rng.normal()) + 0.05 * self.total

    def observe(self, action: float, feedback: float) -> None:
        """Accumulate learned state after a complete environment interaction."""
        self.total += feedback
        self.steps += 1

    def state_dict(self) -> dict[str, NativeState]:
        """Expose framework-like native state without converting it to NumPy."""
        return {"native": NativeState(struct.pack("!dI", self.total, self.steps))}

    def load_state_dict(self, state: dict[str, NativeState]) -> None:
        """Require the serializer to recreate the actual native value type."""
        value = state["native"]
        if not isinstance(value, NativeState):
            raise TypeError("backend did not restore NativeState")
        self.total, self.steps = struct.unpack("!dI", value.payload)


class ReplayEnvironment:
    """Maintain scalar scientific state separately from the learner's native value."""

    supports_extension = True

    def __init__(self, *, rng: Any) -> None:
        self.rng = rng
        self.position = 0.0

    def generate(self, action: float) -> Feedback:
        """Record the random trajectory whose replay is checked after resumption."""
        self.position += action + float(self.rng.normal())
        return Feedback(self.position, {"position": self.position})

    def state_dict(self) -> dict[str, float]:
        """Return the environment's evolving scalar position."""
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, float]) -> None:
        """Restore its complete scientific state before RNG restoration."""
        self.position = state["position"]
