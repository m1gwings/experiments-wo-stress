"""Bandit metrics computed from saved results without simulation components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..analysis.metrics import MetricResult, validate_requirements

if TYPE_CHECKING:
    from ..storage.models import RunResult


class PseudoRegretMetric:
    """Cumulative regret in expected rewards, recovered from an instance and actions.

    ``dynamic`` compares with the best arm at each round. ``best_fixed`` compares
    with the best fixed arm over each observed prefix. Means may have shape K
    (stationary) or T x K (nonstationary); recorded steps are one-based.

    The saved instance supplies arm means and the result records supply chosen
    arms. Every step is required to reconstruct the cumulative curve. This is a
    bandit-specific implementation of the general analysis metric contract.
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
        # Promote before subtraction: saved booleans cannot be subtracted, and
        # integer gaps can wrap before cumsum or MetricResult converts them.
        if self.comparator == "dynamic":
            return np.cumsum(np.subtract(np.max(rewards, axis=1), observed, dtype=np.float64))
        # Accumulate differences to avoid cancellation between large reward sums.
        differences = np.subtract(rewards, observed[:, None], dtype=np.float64)
        np.cumsum(differences, axis=0, out=differences)
        return np.max(differences, axis=1)

    def compute(self, results: RunResult) -> MetricResult:
        """Compare chosen-arm expected rewards with the configured prefix benchmark."""
        means, actions = self._rewards(results)
        expected = means[np.arange(len(actions)), actions]
        original = np.asarray(results.instance.arrays[self.means])
        if original.ndim == 1:
            # Both benchmarks coincide in a stationary instance. Keep working
            # memory O(T + K), instead of materializing a T x K cumulative array.
            gaps = np.subtract(np.max(original), expected, dtype=np.float64)
            return MetricResult(results["step"], np.cumsum(gaps))
        return MetricResult(results["step"], self._regret(means, expected))


class RealizedRegretMetric(PseudoRegretMetric):
    """Realized regret requires the saved counterfactual reward for every arm.

    The default benchmark is the best fixed arm at each prefix. This quantity
    differs from pseudo-regret; means alone cannot identify unobserved rewards.

    The instance must contain the full T x K counterfactual reward matrix, and
    recorded rewards must agree with its chosen-arm entries. The supplied bandit
    generators record chosen rewards only, so this metric requires a data source
    that explicitly constructs and persists the counterfactual matrix.
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
        """Validate chosen rewards against the saved matrix and compute prefix regret."""
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
