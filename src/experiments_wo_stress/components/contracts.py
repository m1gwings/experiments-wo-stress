"""Structural contracts implemented by external experiment components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Feedback:
    """Separate observable feedback from evaluator-only measurements."""

    value: Any
    measurements: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Checkpointable(Protocol):
    """State that can be restored at a complete protocol-step boundary."""

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class DataGenerator(Checkpointable, Protocol):
    """An environment or data source responding to protocol requests."""

    def generate(self, request: Any) -> Any: ...


class OnlineAlgorithm(Checkpointable, Protocol):
    """A policy updated after each action and its observable feedback."""

    def act(self, context: Any = None) -> Any: ...

    def observe(self, action: Any, feedback: Any) -> None: ...


class OfflineAlgorithm(Checkpointable, Protocol):
    """An estimator fitted once to the dataset supplied by the protocol."""

    def fit(self, dataset: Any) -> Any: ...


class InteractionProtocol(Checkpointable, Protocol):
    """Interaction order and progress for one run of fresh components."""

    step: int

    def initialize(self, algorithm: Any, data: DataGenerator) -> None: ...

    def advance(self, algorithm: Any, data: DataGenerator) -> Mapping[str, Any]: ...

    def is_finished(self) -> bool: ...


class EnvironmentStateAdapter(Protocol):
    """Explicit serialization of an environment, its wrappers, and RNG states."""

    def snapshot(self, environment: Any) -> Mapping[str, Any]: ...

    def restore(self, environment: Any, state: Mapping[str, Any]) -> None: ...


class StateMixin:
    """Checkpoint implementation for components with no mutable scientific state.

    Stateful subclasses must override both methods. RNG state is handled by the
    executor and does not belong in these mappings.
    """

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state:
            raise ValueError(f"{type(self).__name__} expects empty checkpoint state")
