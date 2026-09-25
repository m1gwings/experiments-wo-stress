"""Structural contracts implemented by external experiment components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Feedback:
    """Carry one data response to both the learner and the recorder.

    ``value`` is the information the algorithm may observe. ``measurements``
    contains numerical observations for later analysis; the online protocol
    records them without passing them to the learner. A generator can therefore
    expose evaluation data without revealing it to the algorithm.
    """

    value: Any
    measurements: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Checkpointable(Protocol):
    """Describe the mutable component state needed to continue an interrupted run.

    The executor snapshots every component at the same complete protocol step.
    Implementations must include all evolving scientific state. The selected
    checkpoint backend determines which values it can encode; the default handles
    plain values and numerical arrays. The executor gathers injected RNGs separately.
    """

    def state_dict(self) -> Mapping[str, Any]:
        """Return the state needed to resume from the current step boundary."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a snapshot into a freshly constructed component."""
        ...


class DataGenerator(Checkpointable, Protocol):
    """Own the environment or data source for one run.

    The protocol chooses what ``request`` means: an online action, an offline
    dataset request, or a custom interaction. Generated problem definitions can
    be persisted as an Instance; evolving environment state belongs in checkpoints.
    Scientific metrics are computed later from the saved instance and records.
    """

    def generate(self, request: Any) -> Any:
        """Respond to the protocol's request, advancing data state if needed."""
        ...


class OnlineAlgorithm(Checkpointable, Protocol):
    """Own a learner's policy and history for sequential interaction.

    A protocol calls ``act`` with the available context, obtains feedback from the
    data generator, and passes the observable value to ``observe``. The algorithm
    checkpoints its own learned state, independently of environment progress.
    """

    def act(self, context: Any = None) -> Any:
        """Choose the next action from the supplied context and learned state."""
        ...

    def observe(self, action: Any, feedback: Any) -> None:
        """Update the learner after its action receives observable feedback."""
        ...


class OfflineAlgorithm(Checkpointable, Protocol):
    """Own an estimator fitted to the dataset supplied by the offline protocol.

    ``fit`` returns numerical output or a mapping of numerical observations for
    recording. With the built-in offline protocol, the entire fit is one step;
    interruption inside fitting cannot be resumed from a partial fit.
    """

    def fit(self, dataset: Any) -> Any:
        """Fit the estimator and return observations for the completed fit."""
        ...


class InteractionProtocol(Checkpointable, Protocol):
    """Own interaction order and the completed-step counter for one run.

    The executor calls ``initialize`` once for a new run, then ``advance`` until
    ``is_finished``. Each successful advance must increment ``step`` exactly once
    and return numerical observations. Checkpoints occur between those calls;
    restoration loads state instead of calling ``initialize`` again.
    """

    step: int

    def initialize(self, algorithm: Any, data: DataGenerator) -> None:
        """Validate and initialize fresh components before the first step."""
        ...

    def advance(self, algorithm: Any, data: DataGenerator) -> Mapping[str, Any]:
        """Finish one interaction step and return its recordable observations."""
        ...

    def is_finished(self) -> bool:
        """Report whether the requested execution is complete at this boundary."""
        ...


class EnvironmentStateAdapter(Protocol):
    """Supply environment-specific checkpointing for the Gymnasium adapter.

    A snapshot must cover the environment, its wrapper stack, and internal RNGs.
    Restoring into a fresh environment must reproduce the next transition; the
    library cannot infer that state from the generic reset/step interface.
    """

    def snapshot(self, environment: Any) -> Mapping[str, Any]:
        """Describe all state needed to reproduce the environment's next step."""
        ...

    def restore(self, environment: Any, state: Mapping[str, Any]) -> None:
        """Apply a snapshot to an environment and its wrappers."""
        ...


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
