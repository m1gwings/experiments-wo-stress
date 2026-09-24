"""Small structural interfaces and explicit resolution of project components."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .jobs import ComponentSpec


@dataclass(frozen=True)
class Feedback:
    """Separate observable feedback from evaluator-only measurements."""

    value: Any
    measurements: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Checkpointable(Protocol):
    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class DataGenerator(Checkpointable, Protocol):
    def generate(self, request: Any) -> Any: ...


class OnlineAlgorithm(Checkpointable, Protocol):
    def act(self, context: Any = None) -> Any: ...

    def observe(self, action: Any, feedback: Any) -> None: ...


class OfflineAlgorithm(Checkpointable, Protocol):
    def fit(self, dataset: Any) -> Any: ...


class InteractionProtocol(Checkpointable, Protocol):
    step: int

    def initialize(self, algorithm: Any, data: DataGenerator) -> None: ...

    def advance(self, algorithm: Any, data: DataGenerator) -> Mapping[str, Any]: ...

    def is_finished(self) -> bool: ...


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


class NullAlgorithm(StateMixin):
    """Placeholder for trial functions that do not need an algorithm object."""

    def __init__(self, *, rng: np.random.Generator) -> None:
        self.rng = rng


_ALIASES = {
    "online": "experiments_wo_stress.protocols:OnlineProtocol",
    "offline": "experiments_wo_stress.protocols:OfflineProtocol",
    "trial": "experiments_wo_stress.protocols:TrialProtocol",
    "csv": "experiments_wo_stress.data:CSVDataGenerator",
    "normal": "experiments_wo_stress.data:NormalDataGenerator",
    "null": "experiments_wo_stress.data:NullDataGenerator",
    "null_algorithm": "experiments_wo_stress.components:NullAlgorithm",
}


def resolve_type(path_or_alias: str) -> Any:
    """Resolve a documented built-in alias or an explicit ``module:Name`` path."""
    path = _ALIASES.get(path_or_alias, path_or_alias)
    if ":" not in path:
        raise ValueError(
            f"Unknown component {path_or_alias!r}; use module:Class or an alias "
            f"({', '.join(sorted(_ALIASES))})"
        )
    module_name, attribute = path.split(":", 1)
    if not module_name or not attribute:
        raise ValueError(f"Invalid import path {path!r}; expected module:Name")
    try:
        value = importlib.import_module(module_name)
        for name in attribute.split("."):
            value = getattr(value, name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot resolve component {path_or_alias!r}: {exc}") from exc
    if not callable(value):
        raise TypeError(f"Component {path_or_alias!r} is not callable")
    return value


def construct(spec: ComponentSpec, rng: np.random.Generator) -> Any:
    """Construct one fresh component with its own explicitly injected generator."""
    factory = resolve_type(spec.type)
    if "rng" in spec.params:
        raise ValueError(f"{spec.type}: rng is injected; configure seed instead")
    parameters = {"rng": rng, **spec.params}
    try:
        inspect.signature(factory).bind(**parameters)
    except ValueError:
        pass  # Some extension types do not expose a Python signature.
    except TypeError as exc:
        raise TypeError(f"Invalid parameters for {spec.type}: {exc}") from exc
    return factory(**parameters)


def validate_component(instance: Any, methods: Iterable[str]) -> None:
    """Fail before interaction when a component lacks required capabilities."""
    missing = [name for name in methods if not callable(getattr(instance, name, None))]
    if missing:
        raise TypeError(
            f"{type(instance).__name__} is missing required methods: {', '.join(missing)}"
        )
