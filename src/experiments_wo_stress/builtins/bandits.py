"""Reusable bandit environments with explicit instances and checkpoint state."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..components.contracts import Feedback
from ..storage.models import Instance


def _reward_options(distribution: str, noise_std: float) -> None:
    if distribution not in {"bernoulli", "gaussian"}:
        raise ValueError("Bandit distribution must be 'bernoulli' or 'gaussian'")
    if not np.isscalar(noise_std) or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and nonnegative")


def _means(value: Any, distribution: str, dimensions: int) -> np.ndarray:
    means = np.asarray(value, dtype=float)
    if (
        means.ndim != dimensions
        or any(size == 0 for size in means.shape)
        or not np.all(np.isfinite(means))
    ):
        raise ValueError(f"Bandit means must be a nonempty finite {dimensions}-dimensional array")
    if distribution == "bernoulli" and (np.any(means < 0) or np.any(means > 1)):
        raise ValueError("Bernoulli means must lie in [0, 1]")
    return means


class StationaryBandit:
    """Independent arm rewards with an immutable vector of expected values.

    The generator emits observations only. Regret and other scientific metrics
    are computed afterward from the persisted means, actions, and rewards.

    Instance creation fixes the arm means using the instance RNG when needed.
    Each call to ``generate`` samples only the chosen arm using the data RNG and
    advances the environment's step counter. Checkpoints retain that counter;
    the executor stores RNG state and the immutable instance separately.
    """

    supports_extension = True

    @classmethod
    def create_instance(
        cls,
        *,
        rng: np.random.Generator,
        means: list[float] | np.ndarray | None = None,
        n_arms: int | None = None,
        distribution: str = "bernoulli",
        noise_std: float = 1.0,
    ) -> Instance:
        _reward_options(distribution, noise_std)
        if n_arms is not None and (
            isinstance(n_arms, bool) or not isinstance(n_arms, int) or n_arms < 1
        ):
            raise ValueError("n_arms must be a positive integer")
        if means is None:
            means = rng.uniform(0.0, 1.0, size=n_arms or 10)
        means = _means(means, distribution, 1)
        if n_arms is not None and len(means) != n_arms:
            raise ValueError("n_arms does not match the supplied means")
        return Instance(
            kind="stationary_bandit",
            metadata={
                "distribution": distribution,
                "noise_std": float(noise_std),
                "n_arms": len(means),
            },
            arrays={"means": means},
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        instance: Instance | None = None,
        means: list[float] | np.ndarray | None = None,
        n_arms: int | None = None,
        distribution: str = "bernoulli",
        noise_std: float = 1.0,
    ) -> None:
        self.rng = rng
        self.instance = (
            instance
            if instance is not None
            else StationaryBandit.create_instance(
                rng=rng, means=means, n_arms=n_arms, distribution=distribution, noise_std=noise_std
            )
        )
        if self.instance.kind != "stationary_bandit":
            raise ValueError("StationaryBandit requires a stationary_bandit Instance")
        self.distribution = self.instance.metadata["distribution"]
        self.noise_std = self.instance.metadata["noise_std"]
        _reward_options(self.distribution, self.noise_std)
        _means(self.instance.arrays["means"], self.distribution, 1)
        self.means = self.instance.arrays["means"]
        self.step = 0

    def context(self) -> dict[str, int]:
        """Tell the learner how many arms are available without exposing their means."""
        return {"n_arms": self.means.shape[-1]}

    def _current_means(self) -> np.ndarray:
        return self.means

    def generate(self, request: Any) -> Feedback:
        """Sample the selected arm and expose its reward for learning and recording."""
        if (
            isinstance(request, bool)
            or not isinstance(request, (int, np.integer))
            or not 0 <= request < self.means.shape[-1]
        ):
            raise ValueError("Bandit action must be a valid integer arm index")
        mean = float(self._current_means()[request])
        reward = (
            float(self.rng.binomial(1, mean))
            if self.distribution == "bernoulli"
            else float(self.rng.normal(mean, self.noise_std))
        )
        self.step += 1
        return Feedback(reward, {"action": int(request), "reward": reward})

    def state_dict(self) -> dict[str, int]:
        """Save elapsed interaction steps, independently of immutable arm means."""
        return {"step": self.step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        step = state.get("step")
        if (
            set(state) != {"step"}
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            raise ValueError("Bandit checkpoint requires a nonnegative integer step")
        self.step = step


class GaussianBandit(StationaryBandit):
    """Specialize StationaryBandit to Gaussian rewards with configurable noise.

    Instance creation fixes the distribution to Gaussian; interaction and step
    checkpointing use the stationary implementation unchanged.
    """

    supports_extension = True

    @classmethod
    def create_instance(
        cls,
        *,
        rng: np.random.Generator,
        means: list[float] | np.ndarray | None = None,
        n_arms: int | None = None,
        noise_std: float = 1.0,
    ) -> Instance:
        return StationaryBandit.create_instance(
            rng=rng, means=means, n_arms=n_arms, distribution="gaussian", noise_std=noise_std
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        instance: Instance | None = None,
        means: list[float] | np.ndarray | None = None,
        n_arms: int | None = None,
        noise_std: float = 1.0,
    ) -> None:
        super().__init__(
            rng=rng,
            instance=instance,
            means=means,
            n_arms=n_arms,
            distribution="gaussian",
            noise_std=noise_std,
        )


class NonstationaryBandit(StationaryBandit):
    """A bandit whose immutable means array has shape ``(steps, arms)``.

    Budgets can extend within the supplied schedule. Exhausting it raises an
    error instead of inventing unconfigured rewards beyond its final row.

    ``step`` indexes the next schedule row. The inherited generation method
    samples the chosen arm from that row, then advances the counter; restoring
    the counter and data RNG resumes the same point in the saved schedule.
    """

    supports_extension = True

    @classmethod
    def create_instance(
        cls,
        *,
        rng: np.random.Generator,
        schedule: list[list[float]] | np.ndarray,
        distribution: str = "gaussian",
        noise_std: float = 1.0,
    ) -> Instance:
        _reward_options(distribution, noise_std)
        means = _means(schedule, distribution, 2)
        return Instance(
            kind="nonstationary_bandit",
            metadata={
                "distribution": distribution,
                "noise_std": float(noise_std),
                "n_arms": means.shape[1],
                "schedule_steps": len(means),
            },
            arrays={"means": means},
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        schedule: list[list[float]] | np.ndarray,
        instance: Instance | None = None,
        distribution: str = "gaussian",
        noise_std: float = 1.0,
    ) -> None:
        self.rng = rng
        self.instance = (
            instance
            if instance is not None
            else self.create_instance(
                rng=rng, schedule=schedule, distribution=distribution, noise_std=noise_std
            )
        )
        if self.instance.kind != "nonstationary_bandit":
            raise ValueError("NonstationaryBandit requires a nonstationary_bandit Instance")
        self.distribution = self.instance.metadata["distribution"]
        self.noise_std = self.instance.metadata["noise_std"]
        _reward_options(self.distribution, self.noise_std)
        _means(self.instance.arrays["means"], self.distribution, 2)
        self.means = self.instance.arrays["means"]
        self.step = 0

    def _current_means(self) -> np.ndarray:
        if self.step >= len(self.means):
            raise ValueError("The nonstationary reward schedule is exhausted")
        return self.means[self.step]

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        if self.step > len(self.means):
            raise ValueError("Bandit checkpoint exceeds the configured schedule")
