"""Metrics computed from saved run results, independently of simulation code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    """A scalar (represented by one point) or a one-dimensional numerical curve."""

    x: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        x = np.atleast_1d(np.asarray(self.x))
        values = np.atleast_1d(np.asarray(self.values))
        if x.ndim != 1 or values.ndim != 1 or x.shape != values.shape:
            raise ValueError("Metric x and values must be matching one-dimensional arrays")
        if not x.size:
            raise ValueError("A metric must return at least one point")
        for name, array in (("x", x), ("values", values)):
            if array.dtype.kind not in "biuf":
                raise TypeError(f"Metric {name} must contain real numerical values")
            if not np.isfinite(array).all():
                raise ValueError(f"Metric {name} contains non-finite values")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "values", values.astype(np.float64, copy=False))


class Metric(Protocol):
    """Custom metrics implement this interface; no inheritance is necessary."""

    def compute(self, results: Mapping[str, np.ndarray]) -> MetricResult:
        """Return a curve or scalar from a single run's recorded observations."""
        ...


class FieldMetric:
    """Select a recorded numerical field without changing its coordinates."""

    def __init__(self, field: str, x: str = "step") -> None:
        self.field = field
        self.x = x

    def compute(self, results: Mapping[str, np.ndarray]) -> MetricResult:
        try:
            return MetricResult(results[self.x], results[self.field])
        except KeyError as exc:
            raise ValueError(f"Metric requires missing recorded field {exc.args[0]!r}") from exc


class CumulativeSumMetric(FieldMetric):
    """Sum a field over recorded observations, which must include every increment.

    Sparse recording cannot reconstruct increments that were never saved. For
    sparse curves, accumulate during simulation and use :class:`FieldMetric`.
    """

    def compute(self, results: Mapping[str, np.ndarray]) -> MetricResult:
        result = super().compute(results)
        return MetricResult(result.x, np.cumsum(result.values, dtype=np.float64))


BUILTIN_METRICS = {"field": FieldMetric, "cumulative_sum": CumulativeSumMetric}
