"""Interaction loops with a checkpoint boundary after every complete step."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..components.contracts import Feedback, StateMixin
from ..components.loading import resolve_type, validate_component


class NullAlgorithm(StateMixin):
    """Supply an RNG and an empty checkpoint for a self-contained trial function.

    This placeholder has no policy or fitting methods; the trial function owns
    the computation and can use the injected RNG when it needs randomness.
    """

    supports_extension = True

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng


class _SteppedProtocol:
    """Share completed-step bookkeeping across the supplied interaction loops.

    Subclasses define how a step runs and when execution ends. This base saves
    only progress; protocols with additional state, such as RL observations and
    episode boundaries, extend the checkpoint mapping themselves.
    """

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng
        self.step = 0

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
            raise ValueError("Protocol checkpoint must contain a nonnegative integer step")
        limit = getattr(self, "horizon", 1)
        if step > limit and not getattr(self, "supports_extension", False):
            raise ValueError(f"Checkpoint step {step} exceeds protocol horizon {limit}")
        self.step = step


class OnlineProtocol(_SteppedProtocol):
    """Complete one action/feedback/update cycle per step until the requested budget.

    Each round obtains optional context from the data generator, asks the learner
    for an action, generates feedback, and updates the learner with its observable
    value. Evaluation measurements are returned to the recorder. Only ``step``
    is checkpointed here; algorithm and environment state have their own owners.
    """

    supports_extension = True

    def __init__(self, *, rng: np.random.Generator, horizon: int | None = None) -> None:
        super().__init__(rng=rng)
        self.horizon = 0
        if horizon is not None:
            self.set_budget(horizon)

    def set_budget(self, steps: int) -> None:
        """Set the execution target without changing the scientific policy."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Online budget must be a positive integer")
        self.horizon = steps

    def initialize(self, algorithm: Any, data: Any) -> None:
        """Check the budget and component methods before starting interaction."""
        if self.horizon < 1:
            raise ValueError("Call set_budget(steps) before initializing an online protocol")
        validate_component(algorithm, ("act", "observe", "state_dict", "load_state_dict"))
        validate_component(data, ("generate", "state_dict", "load_state_dict"))

    def is_finished(self) -> bool:
        """Report whether the completed-step counter has reached the requested budget."""
        return self.step >= self.horizon

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
        """Complete one learner update and return its numerical observations."""
        if self.is_finished():
            raise RuntimeError("Cannot advance a completed online protocol")
        context_method = getattr(data, "context", None)
        context = context_method() if callable(context_method) else None
        action = algorithm.act(context=context)
        feedback = data.generate(action)
        if not isinstance(feedback, Feedback):
            raise TypeError(
                "Online data.generate(action) must return Feedback(value, measurements)"
            )
        if not isinstance(feedback.measurements, Mapping):
            raise TypeError("Feedback.measurements must be a mapping of numerical observations")
        # Evaluation-only measurements stay outside the learner's information.
        algorithm.observe(action, feedback.value)
        observations = dict(feedback.measurements)
        try:
            action_array = np.asarray(action)
        except (TypeError, ValueError):
            action_array = None
        if action_array is not None and action_array.dtype.kind in "biufc":
            observations.setdefault("action", action)
        self.step += 1
        return observations


def _numeric_fields(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        fields = {}
        for name, child in value.items():
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError(
                    "Recorded observation dictionary keys must contain letters, digits, or underscores"
                )
            nested = _numeric_fields(f"{prefix}_{name}", child)
            if fields.keys() & nested.keys():
                raise ValueError("Observation dictionary keys collide after flattening")
            fields.update(nested)
        return fields
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return {}
    return {prefix: array} if array.dtype.kind in "biufc" else {}


class RLProtocol(OnlineProtocol):
    """Single-environment RL with explicit episode and transition semantics.

    The learner uses ``act(context=observation)`` and ``observe(action, transition)``.
    A transition is a mapping with observation, next_observation, reward,
    terminated, truncated, and info. Termination and truncation are kept separate
    so learners can make the appropriate bootstrapping decision.

    This protocol owns the current observation, episode index, and pending reset.
    The environment owns its internal dynamics. A terminal transition is fully
    recorded before the next call resets the environment, so checkpoints preserve
    either side of an episode boundary without repeating a transition.
    """

    supports_extension = True

    def __init__(self, *, rng: np.random.Generator, horizon: int | None = None) -> None:
        super().__init__(rng=rng, horizon=horizon)
        self.observation: Any = None
        self.episode = 0
        self.needs_reset = True

    def initialize(self, algorithm: Any, data: Any) -> None:
        """Validate the components and obtain the first episode's observation."""
        super().initialize(algorithm, data)
        validate_component(data, ("reset",))
        self.observation, _ = data.reset()
        self.observation = copy.deepcopy(self.observation)
        self.needs_reset = False

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
        """Reset if necessary, complete one transition, and update the learner."""
        if self.is_finished():
            raise RuntimeError("Cannot advance a completed RL protocol")
        if self.needs_reset:
            self.observation, _ = data.reset()
            self.observation = copy.deepcopy(self.observation)
            self.episode += 1
            self.needs_reset = False
        # The learner and environment may mutate their inputs; preserve the
        # transition's observations independently of those component objects.
        observation = copy.deepcopy(self.observation)
        action = algorithm.act(context=copy.deepcopy(observation))
        feedback = data.generate(action)
        if not isinstance(feedback, Feedback) or not isinstance(feedback.value, Mapping):
            raise TypeError("RL data must return Feedback with a transition mapping")
        value = feedback.value
        required = {"observation", "reward", "terminated", "truncated"}
        if not required.issubset(value):
            raise ValueError("RL feedback requires observation, reward, terminated, and truncated")
        transition = {
            "observation": observation,
            "next_observation": copy.deepcopy(value["observation"]),
            "reward": float(value["reward"]),
            "terminated": bool(value["terminated"]),
            "truncated": bool(value["truncated"]),
            "info": copy.deepcopy(value.get("info", {})),
        }
        algorithm.observe(action, copy.deepcopy(transition))
        self.observation = transition["next_observation"]
        self.needs_reset = transition["terminated"] or transition["truncated"]
        self.step += 1
        observations = dict(feedback.measurements)
        observations.update(
            {
                "reward": transition["reward"],
                "terminated": transition["terminated"],
                "truncated": transition["truncated"],
                "episode": self.episode,
            }
        )
        observations.update(_numeric_fields("action", action))
        observations.update(_numeric_fields("observation", observation))
        observations.update(_numeric_fields("next_observation", self.observation))
        return observations

    def state_dict(self) -> dict[str, Any]:
        """Snapshot progress and the observation/reset state at a transition boundary."""
        return {
            "step": self.step,
            "observation": copy.deepcopy(self.observation),
            "episode": self.episode,
            "needs_reset": self.needs_reset,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore protocol progress without resetting the restored environment."""
        if set(state) != {"step", "observation", "episode", "needs_reset"}:
            raise ValueError("Invalid RL protocol checkpoint fields")
        super().load_state_dict({"step": state["step"]})
        episode = state["episode"]
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode < 0
            or not isinstance(state["needs_reset"], bool)
        ):
            raise ValueError("Invalid RL protocol episode state")
        self.observation = copy.deepcopy(state["observation"])
        self.episode = episode
        self.needs_reset = state["needs_reset"]


def _observations(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(
            "A fit/trial must return numerical output or a mapping of numerical observations"
        )
    return {"output": array}


class OfflineProtocol(_SteppedProtocol):
    """Generate a dataset and fit an algorithm as one indivisible protocol step.

    ``request`` is passed to the data generator and the resulting dataset to
    ``algorithm.fit``. The fit's numerical output becomes the observations.
    Checkpointing can preserve the finished fit, but cannot resume inside it.
    """

    def __init__(self, *, rng: np.random.Generator, request: Any = None) -> None:
        super().__init__(rng=rng)
        self.request = request

    def initialize(self, algorithm: Any, data: Any) -> None:
        """Check fitting, generation, and checkpoint methods before the trial starts."""
        validate_component(algorithm, ("fit", "state_dict", "load_state_dict"))
        validate_component(data, ("generate", "state_dict", "load_state_dict"))

    def is_finished(self) -> bool:
        return self.step >= 1

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
        """Generate the dataset, finish fitting, and expose the fit's observations."""
        if self.is_finished():
            raise RuntimeError("Cannot advance a completed offline protocol")
        result = _observations(algorithm.fit(data.generate(self.request)))
        self.step += 1
        return result


class TrialProtocol(_SteppedProtocol):
    """Call a paper's function as one atomic trial.

    The function receives ``algorithm``, ``data``, ``rng``, and the values in
    ``params``. It must return numerical observations. Arbitrary execution inside
    the function cannot be resumed; use a stepped protocol for that requirement.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        function: str,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(rng=rng)
        self.function = resolve_type(function)
        self.params = dict(params or {})
        reserved = {"algorithm", "data", "rng"}.intersection(self.params)
        if reserved:
            raise ValueError(f"Trial parameters use reserved names: {', '.join(sorted(reserved))}")

    def initialize(self, algorithm: Any, data: Any) -> None:
        """Check that both objects passed to the trial can participate in a checkpoint."""
        for component in (algorithm, data):
            validate_component(component, ("state_dict", "load_state_dict"))

    def is_finished(self) -> bool:
        return self.step >= 1

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
        """Execute the configured function once and normalize its numerical output."""
        if self.is_finished():
            raise RuntimeError("Cannot advance a completed trial protocol")
        result = _observations(
            self.function(algorithm=algorithm, data=data, rng=self.rng, **self.params)
        )
        self.step += 1
        return result
