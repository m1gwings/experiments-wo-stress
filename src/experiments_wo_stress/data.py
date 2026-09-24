"""Small data sources illustrating the shared generation contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .components import StateMixin


class NullDataGenerator(StateMixin):
    """A stateless placeholder for self-contained trial functions."""

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, request: Any = None) -> None:
        return None


class NormalDataGenerator(StateMixin):
    """Draw an offline dataset from a normal distribution."""

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        size: int | list[int] | tuple[int, ...] = 100,
        loc: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.rng = rng
        shape = (size,) if isinstance(size, int) else tuple(size)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape):
            raise ValueError("Normal data size must contain positive integer dimensions")
        if not np.isfinite(loc) or not np.isfinite(scale) or scale < 0:
            raise ValueError("Normal loc must be finite and scale must be finite and nonnegative")
        self.size = shape
        self.loc = loc
        self.scale = scale

    def generate(self, request: Any = None) -> np.ndarray:
        size = self.size if request is None else request
        return self.rng.normal(loc=self.loc, scale=self.scale, size=size)


class CSVDataGenerator:
    """Read numeric CSV data once; optionally consume successive row batches.

    ``generate(None)`` returns the complete read-only matrix. A positive integer
    request consumes that many rows from the current cursor. Checkpoints retain
    only the cursor; the source file must remain unchanged for resumption.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        path: str | Path,
        delimiter: str = ",",
        skip_header: int = 0,
        dtype: str = "float64",
    ) -> None:
        self.rng = rng
        self.path = Path(path)
        if isinstance(skip_header, bool) or not isinstance(skip_header, int) or skip_header < 0:
            raise ValueError("CSV skip_header must be a nonnegative integer")
        numeric_dtype = np.dtype(dtype)
        if numeric_dtype.kind not in "biufc":
            raise ValueError("CSV dtype must be numerical")
        self.values = np.loadtxt(
            self.path, delimiter=delimiter, skiprows=skip_header, dtype=numeric_dtype, ndmin=2
        )
        self.values.flags.writeable = False
        self.cursor = 0

    def generate(self, request: Any = None) -> np.ndarray:
        if request is None:
            return self.values
        if isinstance(request, bool) or not isinstance(request, int) or request < 1:
            raise ValueError("CSV request must be None or a positive number of rows")
        stop = min(self.cursor + request, len(self.values))
        result = self.values[self.cursor : stop]
        self.cursor = stop
        return result

    def state_dict(self) -> dict[str, int]:
        return {"cursor": self.cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        cursor = state.get("cursor")
        if (
            set(state) != {"cursor"}
            or isinstance(cursor, bool)
            or not isinstance(cursor, int)
            or not 0 <= cursor <= len(self.values)
        ):
            raise ValueError("CSV checkpoint contains an invalid cursor")
        self.cursor = cursor
