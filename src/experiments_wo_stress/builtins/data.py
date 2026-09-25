"""Small data sources illustrating the shared generation contract."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..components.contracts import StateMixin
from ..storage.models import Instance


class NullDataGenerator(StateMixin):
    """Supply an empty data source for trials that generate their own inputs.

    ``generate`` returns None, and the checkpoint contains no scientific state.
    The injected RNG remains available to a trial through this object.
    """

    supports_extension = True

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, request: Any = None) -> None:
        """Return no dataset; the trial function supplies its own inputs."""
        return None


class NormalDataGenerator(StateMixin):
    """Draw normally distributed offline data with a configured shape and scale.

    The immutable instance describes the distribution. Each ``generate`` call
    draws fresh samples from the data RNG, using the configured shape unless the
    request supplies another shape. No evolving state beyond that RNG is needed,
    so the executor's RNG snapshot is sufficient for continuation.
    """

    @classmethod
    def create_instance(
        cls,
        *,
        rng: np.random.Generator,
        size: int | list[int] | tuple[int, ...] = 100,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> Instance:
        """Describe the distribution; samples are drawn by the data RNG at execution."""
        generator = cls(rng=rng, size=size, loc=loc, scale=scale)
        return Instance(
            kind="normal_distribution",
            metadata={
                "size": list(generator.size),
                "loc": float(generator.loc),
                "scale": float(generator.scale),
            },
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        size: int | list[int] | tuple[int, ...] = 100,
        loc: float = 0.0,
        scale: float = 1.0,
        instance: Instance | None = None,
    ) -> None:
        self.rng = rng
        self.instance = instance
        if instance is not None:
            if instance.kind != "normal_distribution":
                raise ValueError("NormalDataGenerator requires a normal_distribution Instance")
            size, loc, scale = (
                instance.metadata["size"],
                instance.metadata["loc"],
                instance.metadata["scale"],
            )
        shape = (size,) if isinstance(size, int) else tuple(size)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape):
            raise ValueError("Normal data size must contain positive integer dimensions")
        if not np.isfinite(loc) or not np.isfinite(scale) or scale < 0:
            raise ValueError("Normal loc must be finite and scale must be finite and nonnegative")
        self.size = shape
        self.loc = loc
        self.scale = scale

    def generate(self, request: Any = None) -> np.ndarray:
        """Draw one dataset using the requested shape or the configured default."""
        size = self.size if request is None else request
        return self.rng.normal(loc=self.loc, scale=self.scale, size=size)


class CSVDataGenerator:
    """Read numeric CSV data once; optionally consume successive row batches.

    ``generate(None)`` returns the complete read-only matrix. A positive integer
    request consumes that many rows from the current cursor. Checkpoints retain
    only the cursor; library-managed runs read the persisted immutable dataset.
    """

    @classmethod
    def create_instance(
        cls,
        *,
        rng: np.random.Generator,
        path: str | Path,
        delimiter: str = ",",
        skip_header: int = 0,
        dtype: str = "float64",
    ) -> Instance:
        """Read and fingerprint the same input bytes in one pass."""
        if isinstance(skip_header, bool) or not isinstance(skip_header, int) or skip_header < 0:
            raise ValueError("CSV skip_header must be a nonnegative integer")
        numeric_dtype = np.dtype(dtype)
        if numeric_dtype.kind not in "biufc":
            raise ValueError("CSV dtype must be numerical")
        path = Path(path).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:

            def lines():
                for line in stream:
                    digest.update(line)
                    yield line

            values = np.loadtxt(
                lines(), delimiter=delimiter, skiprows=skip_header, dtype=numeric_dtype, ndmin=2
            )
        return Instance(
            kind="dataset",
            metadata={
                "format": "csv",
                "path": str(path),
                "sha256": digest.hexdigest(),
                "delimiter": delimiter,
                "skip_header": skip_header,
                "dtype": numeric_dtype.str,
            },
            arrays={"data": values},
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        path: str | Path | None = None,
        delimiter: str = ",",
        skip_header: int = 0,
        dtype: str = "float64",
        instance: Instance | None = None,
    ) -> None:
        self.rng = rng
        if instance is None:
            if path is None:
                raise ValueError("CSVDataGenerator requires path or a persisted Instance")
            instance = self.create_instance(
                rng=rng, path=path, delimiter=delimiter, skip_header=skip_header, dtype=dtype
            )
        if (
            instance.kind != "dataset"
            or instance.metadata.get("format") != "csv"
            or "data" not in instance.arrays
        ):
            raise ValueError("CSVDataGenerator requires a CSV dataset Instance")
        self.instance = instance
        self.path = Path(instance.metadata["path"])
        self.values = instance.arrays["data"]
        if self.values.ndim != 2:
            raise ValueError("CSV instance data must be a two-dimensional matrix")
        self.cursor = 0

    def generate(self, request: Any = None) -> np.ndarray:
        """Return all rows, or consume up to the requested number from the cursor."""
        if request is None:
            return self.values
        if isinstance(request, bool) or not isinstance(request, int) or request < 1:
            raise ValueError("CSV request must be None or a positive number of rows")
        stop = min(self.cursor + request, len(self.values))
        result = self.values[self.cursor : stop]
        self.cursor = stop
        return result

    def state_dict(self) -> dict[str, int]:
        """Save the next unread row; the immutable dataset is persisted separately."""
        return {"cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a cursor that lies within the persisted dataset."""
        cursor = state.get("cursor")
        if (
            set(state) != {"cursor"}
            or isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor <= len(self.values)
        ):
            raise ValueError("CSV checkpoint contains an invalid cursor")
        self.cursor = cursor
