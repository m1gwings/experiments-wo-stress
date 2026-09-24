"""Metrics computed from saved run results, independently of simulation code."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from .artifacts import RunResult


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
    """Select a recorded numerical field without changing its coordinates."""

    def __init__(self, field: str, x: str = "step") -> None:
        self.field = field
        self.x = x
        self.required_fields = (x, field)

    def compute(self, results: Mapping[str, np.ndarray]) -> MetricResult:
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
        result = super().compute(results)
        return MetricResult(result.x, np.cumsum(result.values, dtype=np.float64))


class PseudoRegretMetric:
    """Cumulative regret in expected rewards, recovered from an instance and actions.

    ``dynamic`` compares with the best arm at each round. ``best_fixed`` compares
    with the best fixed arm over each observed prefix. Means may have shape K
    (stationary) or T x K (nonstationary); recorded steps are one-based.
    """

    requires_complete_trajectory = True

    def __init__(self, means: str = "means", action: str = "action", comparator: str = "dynamic"):
        if comparator not in {"dynamic", "best_fixed"}:
            raise ValueError("Regret comparator must be dynamic or best_fixed")
        self.means = means
        self.action = action
        self.comparator = comparator
        self.required_fields = ("step", action)
        self.required_instance_fields = (means,)

    def _rewards(self, results: RunResult) -> tuple[np.ndarray, np.ndarray]:
        validate_requirements(self, results)
        means = np.asarray(results.instance.arrays[self.means])
        steps = np.asarray(results["step"])
        actions = np.asarray(results[self.action])
        if means.ndim not in (1, 2) or not means.size or means.dtype.kind not in "biuf":
            raise ValueError("Bandit instance rewards must be real arrays of shape K or T x K")
        if not np.isfinite(means).all():
            raise ValueError("Bandit instance rewards contain non-finite values")
        if actions.shape != steps.shape or actions.dtype.kind not in "iu":
            raise ValueError(
                "Bandit actions must be one-dimensional zero-based integer arm indices"
            )
        if np.any(actions < 0) or np.any(actions >= means.shape[-1]):
            raise ValueError("Bandit action index is outside the instance's arm count")
        if means.ndim == 1:
            means = np.broadcast_to(means, (len(steps), len(means)))
        else:
            if steps[-1] > means.shape[0]:
                raise ValueError("Nonstationary instance has fewer reward rows than recorded steps")
            means = means[: len(steps)]
        return means, actions

    def _regret(self, rewards: np.ndarray, observed: np.ndarray) -> np.ndarray:
        if self.comparator == "dynamic":
            return np.cumsum(np.max(rewards, axis=1) - observed)
        # Accumulate differences to avoid cancellation between large reward sums.
        differences = rewards - observed[:, None]
        np.cumsum(differences, axis=0, out=differences)
        return np.max(differences, axis=1)

    def compute(self, results: RunResult) -> MetricResult:
        means, actions = self._rewards(results)
        expected = means[np.arange(len(actions)), actions]
        original = np.asarray(results.instance.arrays[self.means])
        if original.ndim == 1:
            # Both benchmarks coincide in a stationary instance. Keep working
            # memory O(T + K), instead of materializing a T x K cumulative array.
            return MetricResult(results["step"], np.cumsum(np.max(original) - expected))
        return MetricResult(results["step"], self._regret(means, expected))


class RealizedRegretMetric(PseudoRegretMetric):
    """Realized regret requires the saved counterfactual reward for every arm.

    The default benchmark is the best fixed arm at each prefix. This quantity
    differs from pseudo-regret; means alone cannot identify unobserved rewards.
    """

    def __init__(
        self,
        counterfactual_rewards: str = "counterfactual_rewards",
        action: str = "action",
        reward: str = "reward",
        comparator: str = "best_fixed",
    ):
        super().__init__(means=counterfactual_rewards, action=action, comparator=comparator)
        self.reward = reward
        self.required_fields = ("step", action, reward)

    def compute(self, results: RunResult) -> MetricResult:
        rewards, actions = self._rewards(results)
        original = np.asarray(results.instance.arrays[self.means])
        if original.ndim != 2:
            raise ValueError("Realized regret requires a T x K counterfactual reward matrix")
        observed = np.asarray(results[self.reward])
        if observed.shape != actions.shape or observed.dtype.kind not in "biuf":
            raise ValueError("Recorded rewards must be a one-dimensional real numerical field")
        if not np.allclose(
            observed, rewards[np.arange(len(actions)), actions], rtol=1e-10, atol=1e-12
        ):
            raise ValueError(
                "Observed rewards disagree with the saved counterfactual reward matrix"
            )
        return MetricResult(results["step"], self._regret(rewards, observed))


BUILTIN_METRICS = {
    "field": FieldMetric,
    "cumulative_sum": CumulativeSumMetric,
    "pseudo_regret": PseudoRegretMetric,
    "realized_regret": RealizedRegretMetric,
}
