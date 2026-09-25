"""Metrics computed from saved run results, independently of simulation code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from ..storage.models import RunResult


@dataclass(frozen=True)
class MetricResult:
    """Represent one run's metric as paired coordinates and numerical values.

    Scalars become a one-point curve; curves must have matching, nonempty,
    one-dimensional real arrays with finite values. Values are normalized to
    float64 for aggregation. Repetitions can be averaged only when their ``x``
    coordinates match, so metrics should choose coordinates consistently.
    """

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
    """Compute scientific measurements from one saved run without simulation objects.

    A custom class only needs ``compute``; inheritance is optional. The same
    metric instance can process several runs, so each result should depend on the
    supplied RunResult and configured metric parameters. Optional requirement
    attributes declare needed record fields, instance arrays, or a full trajectory
    before computation or cache reuse.
    """

    def compute(self, results: RunResult) -> MetricResult:
        """Return a curve or scalar from a single run's recorded observations."""
        ...


def validate_requirements(metric: Any, results: Mapping[str, np.ndarray]) -> None:
    """Reject insufficient saved data before computing or reusing a metric."""
    for attribute, available, description in (
        ("required_fields", results, "recorded fields"),
        (
            "required_instance_fields",
            getattr(getattr(results, "instance", None), "arrays", {}),
            "instance arrays",
        ),
    ):
        required = getattr(metric, attribute, ())
        if isinstance(required, str) or any(not isinstance(name, str) for name in required):
            raise TypeError(f"Metric {attribute} must be a sequence of field names")
        missing = sorted(set(required) - set(available))
        if missing:
            raise ValueError(f"Metric requires missing {description}: {', '.join(missing)}")
    if getattr(metric, "requires_complete_trajectory", False):
        if "step" not in results:
            raise ValueError("Metric requires a complete trajectory with a recorded step field")
        steps = np.asarray(results["step"])
        completed = getattr(results, "completed_steps", int(steps[-1]) if steps.size else 0)
        if (
            steps.ndim != 1
            or steps.dtype.kind not in "iu"
            or not steps.size
            or len(steps) != completed
            or steps[0] != 1
            or np.any(np.diff(steps) != 1)
        ):
            raise ValueError(
                "Metric requires a complete trajectory: record every step from 1 through "
                "completed_steps; sparse or truncated observations cannot reconstruct it"
            )


class FieldMetric:
    """Expose one saved field as a metric curve using another field as its x axis.

    Both fields must satisfy MetricResult's one-dimensional shape contract.
    Sparse recording is valid here: the curve describes exactly the observations
    that were saved, with ``step`` providing the default coordinates.
    """

    def __init__(self, field: str, x: str = "step") -> None:
        self.field = field
        self.x = x
        self.required_fields = (x, field)

    def compute(self, results: Mapping[str, np.ndarray]) -> MetricResult:
        """Validate the requested fields and return their coordinate/value pairs."""
        validate_requirements(self, results)
        try:
            return MetricResult(results[self.x], results[self.field])
        except KeyError as exc:
            raise ValueError(f"Metric requires missing recorded field {exc.args[0]!r}") from exc


class CumulativeSumMetric(FieldMetric):
    """Sum a field over recorded observations, which must include every increment.

    Sparse recording cannot reconstruct increments that were never saved. For
    sparse curves, accumulate during simulation and use :class:`FieldMetric`.
    """

    requires_complete_trajectory = True

    def compute(self, results: Mapping[str, np.ndarray]) -> MetricResult:
        """Accumulate a complete sequence of increments along the recorded x axis."""
        result = super().compute(results)
        return MetricResult(result.x, np.cumsum(result.values, dtype=np.float64))
