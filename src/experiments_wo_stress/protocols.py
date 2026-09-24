"""Interaction loops with a checkpoint boundary after every complete step."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .components import Feedback, resolve_type, validate_component


class _SteppedProtocol:
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
        if step > limit:
            raise ValueError(f"Checkpoint step {step} exceeds protocol horizon {limit}")
        self.step = step


class OnlineProtocol(_SteppedProtocol):
    """Act, generate feedback, and observe it for a fixed number of rounds."""

    def __init__(self, *, rng: np.random.Generator, horizon: int) -> None:
        super().__init__(rng=rng)
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Online horizon must be a positive integer")
        self.horizon = horizon

    def initialize(self, algorithm: Any, data: Any) -> None:
        validate_component(algorithm, ("act", "observe", "state_dict", "load_state_dict"))
        validate_component(data, ("generate", "state_dict", "load_state_dict"))

    def is_finished(self) -> bool:
        return self.step >= self.horizon

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
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
    """Generate a dataset and fit an algorithm in one atomic step."""

    def __init__(self, *, rng: np.random.Generator, request: Any = None) -> None:
        super().__init__(rng=rng)
        self.request = request

    def initialize(self, algorithm: Any, data: Any) -> None:
        validate_component(algorithm, ("fit", "state_dict", "load_state_dict"))
        validate_component(data, ("generate", "state_dict", "load_state_dict"))

    def is_finished(self) -> bool:
        return self.step >= 1

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
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
        for component in (algorithm, data):
            validate_component(component, ("state_dict", "load_state_dict"))

    def is_finished(self) -> bool:
        return self.step >= 1

    def advance(self, algorithm: Any, data: Any) -> dict[str, Any]:
        if self.is_finished():
            raise RuntimeError("Cannot advance a completed trial protocol")
        result = _observations(
            self.function(algorithm=algorithm, data=data, rng=self.rng, **self.params)
        )
        self.step += 1
        return result
