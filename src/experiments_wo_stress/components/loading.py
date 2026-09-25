"""Resolve configured components and inject their execution capabilities."""

from __future__ import annotations

import copy
import importlib
import inspect
import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..storage.models import Instance

if TYPE_CHECKING:
    from ..study.specs import ComponentSpec


_ALIASES = {
    "online": "experiments_wo_stress.protocols:OnlineProtocol",
    "offline": "experiments_wo_stress.protocols:OfflineProtocol",
    "trial": "experiments_wo_stress.protocols:TrialProtocol",
    "csv": "experiments_wo_stress.data:CSVDataGenerator",
    "normal": "experiments_wo_stress.data:NormalDataGenerator",
    "null": "experiments_wo_stress.data:NullDataGenerator",
    "null_algorithm": "experiments_wo_stress.components:NullAlgorithm",
    "stationary_bandit": "experiments_wo_stress.settings:StationaryBandit",
    "gaussian_bandit": "experiments_wo_stress.settings:GaussianBandit",
    "nonstationary_bandit": "experiments_wo_stress.settings:NonstationaryBandit",
    "gymnasium": "experiments_wo_stress.settings:GymnasiumAdapter",
    "rl": "experiments_wo_stress.protocols:RLProtocol",
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


def constructor_kwargs(
    spec: ComponentSpec,
    rng: np.random.Generator,
    *,
    instance: Instance | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> dict[str, Any]:
    """Build constructor arguments, injecting supported optional capabilities."""
    factory = resolve_type(spec.type)
    reserved = {"rng", "instance", "logger"}.intersection(spec.params)
    if reserved:
        raise ValueError(f"{spec.type}: {', '.join(sorted(reserved))} is injected by the library")
    parameters = {"rng": rng, **copy.deepcopy(spec.params)}
    try:
        signature = inspect.signature(factory)
    except ValueError:
        return parameters  # Some extension types do not expose a Python signature.
    accepts_kwargs = any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in signature.parameters.values()
    )
    for name, value in (("instance", instance), ("logger", logger)):
        # Forwarding **kwargs wrappers must not unexpectedly receive a logger or
        # a None instance that their wrapped, older constructor cannot accept.
        if name in signature.parameters or (
            name == "instance" and value is not None and accepts_kwargs
        ):
            parameters[name] = value
    try:
        signature.bind(**parameters)
    except TypeError as exc:
        raise TypeError(f"Invalid parameters for {spec.type}: {exc}") from exc
    return parameters


def construct(
    spec: ComponentSpec,
    rng: np.random.Generator,
    *,
    instance: Instance | None = None,
    logger: logging.Logger | logging.LoggerAdapter | None = None,
) -> Any:
    """Construct a fresh component, retaining compatibility with older classes."""
    component_logger = (
        logger if logger is not None else logging.getLogger(f"experiments_wo_stress.{spec.type}")
    )
    component = resolve_type(spec.type)(
        **constructor_kwargs(spec, rng, instance=instance, logger=component_logger)
    )
    try:
        component.logger = component_logger
    except (AttributeError, TypeError):
        pass  # Slotted or frozen user components can opt into constructor injection.
    return component


def create_instance(spec: ComponentSpec, rng: np.random.Generator) -> Instance:
    """Create an immutable scientific instance or a generic component descriptor."""
    factory = resolve_type(spec.type)
    hook = getattr(factory, "create_instance", None)
    if hook is None:
        return Instance(metadata={"data": spec.to_dict()})
    if not callable(hook):
        raise TypeError(f"{spec.type}.create_instance must be callable")
    instance = hook(rng=rng, **copy.deepcopy(spec.params))
    if not isinstance(instance, Instance):
        raise TypeError(f"{spec.type}.create_instance must return an Instance")
    return instance


def validate_component(instance: Any, methods: Iterable[str]) -> None:
    """Fail before interaction when a component lacks required capabilities."""
    missing = [name for name in methods if not callable(getattr(instance, name, None))]
    if missing:
        raise TypeError(
            f"{type(instance).__name__} is missing required methods: {', '.join(missing)}"
        )


def load_class(type_name: str, builtins: Mapping[str, type] | None = None) -> type:
    """Load an analysis extension class, optionally using a local alias table."""
    if not isinstance(type_name, str):
        raise ValueError("Component type must be a string")
    if builtins and type_name in builtins:
        return builtins[type_name]
    if ":" not in type_name:
        raise ValueError(f"Unknown component {type_name!r}; use a built-in alias or module:Class")
    module_name, class_name = type_name.split(":", 1)
    try:
        component_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import component {type_name!r}: {exc}") from exc
    if not isinstance(component_class, type):
        raise TypeError(f"Component {type_name!r} must identify a class")
    return component_class
