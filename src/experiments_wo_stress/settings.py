"""Reusable scientific settings with explicit instances and faithful checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from .artifacts import Instance
from .components import Feedback, resolve_type, validate_component


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
        return {"n_arms": self.means.shape[-1]}

    def _current_means(self) -> np.ndarray:
        return self.means

    def generate(self, request: Any) -> Feedback:
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
    """Stationary bandit with Gaussian rewards and configurable noise scale."""

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


class EnvironmentStateAdapter(Protocol):
    """Explicit serialization of an environment, its wrappers, and RNG states."""

    def snapshot(self, environment: Any) -> Mapping[str, Any]: ...

    def restore(self, environment: Any, state: Mapping[str, Any]) -> None: ...


class GymnasiumAdapter:
    """Adapt the Gymnasium reset/step API with a user-supplied state adapter.

    Gymnasium has no universal checkpoint API. ``state_adapter`` identifies a
    zero-argument class implementing snapshot(env) and restore(env, state). The
    adapter must cover the entire wrapper stack, environment randomness, and any
    external state required to continue faithfully; no implicit pickling occurs.

    ``factory`` defaults to ``gymnasium:make`` and receives ``env_params``. That
    optional package is imported only when selected. Other factories implementing
    the same public environment API also work.
    """

    supports_extension = True

    @classmethod
    def create_instance(
        cls,
        *,
        rng: np.random.Generator,
        state_adapter: str,
        factory: str = "gymnasium:make",
        env_params: Mapping[str, Any] | None = None,
        reset_options: Mapping[str, Any] | None = None,
    ) -> Instance:
        return Instance(
            kind="gymnasium",
            metadata={
                "factory": factory,
                "state_adapter": state_adapter,
                "env_params": dict(env_params or {}),
                "reset_options": dict(reset_options or {}),
                "initial_seed": int(rng.integers(0, 2**32, dtype=np.uint64)),
            },
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        state_adapter: str,
        factory: str = "gymnasium:make",
        env_params: Mapping[str, Any] | None = None,
        reset_options: Mapping[str, Any] | None = None,
        instance: Instance | None = None,
    ) -> None:
        self.rng = rng
        self.instance = (
            instance
            if instance is not None
            else self.create_instance(
                rng=rng,
                state_adapter=state_adapter,
                factory=factory,
                env_params=env_params,
                reset_options=reset_options,
            )
        )
        if self.instance.kind != "gymnasium":
            raise ValueError("GymnasiumAdapter requires a gymnasium Instance")
        metadata = self.instance.to_dict()["metadata"]
        self.env = resolve_type(metadata["factory"])(**metadata["env_params"])
        validate_component(self.env, ("reset", "step"))
        self.adapter = resolve_type(metadata["state_adapter"])()
        validate_component(self.adapter, ("snapshot", "restore"))
        self.initial_seed = metadata["initial_seed"]
        self.reset_options = metadata["reset_options"]
        self.initialized = False
        self.needs_reset = True

    def reset(self) -> tuple[Any, Mapping[str, Any]]:
        seed = None if self.initialized else self.initial_seed
        observation, info = self.env.reset(seed=seed, options=copy.deepcopy(self.reset_options))
        if not isinstance(info, Mapping):
            raise TypeError("Gymnasium reset must return (observation, info mapping)")
        self.initialized = True
        self.needs_reset = False
        return observation, info

    def generate(self, request: Any) -> Feedback:
        if not self.initialized or self.needs_reset:
            raise RuntimeError("Reset the environment before taking an action")
        observation, reward, terminated, truncated, info = self.env.step(request)
        if not isinstance(info, Mapping):
            raise TypeError("Gymnasium step info must be a mapping")
        self.needs_reset = bool(terminated or truncated)
        value = {
            "observation": observation,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info),
        }
        return Feedback(
            value,
            {"reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)},
        )

    def state_dict(self) -> dict[str, Any]:
        environment = self.adapter.snapshot(self.env)
        if not isinstance(environment, Mapping):
            raise TypeError("Environment snapshot must be a mapping")
        return {
            "environment": copy.deepcopy(dict(environment)),
            "initialized": self.initialized,
            "needs_reset": self.needs_reset,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            set(state) != {"environment", "initialized", "needs_reset"}
            or not isinstance(state["initialized"], bool)
            or not isinstance(state["needs_reset"], bool)
            or not isinstance(state["environment"], Mapping)
        ):
            raise ValueError("Invalid Gymnasium adapter checkpoint")
        self.adapter.restore(self.env, copy.deepcopy(state["environment"]))
        self.initialized = state["initialized"]
        self.needs_reset = state["needs_reset"]

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
