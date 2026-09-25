"""Adapt Gymnasium environments through explicit checkpoint serialization."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..components.contracts import Feedback
from ..components.loading import resolve_type, validate_component
from ..storage.models import Instance


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
