"""Serializable descriptions of independent scientific runs."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import ExperimentConfig

_BUDGET_PROTOCOLS = {
    "online",
    "rl",
    "experiments_wo_stress.protocols:OnlineProtocol",
    "experiments_wo_stress.protocols:RLProtocol",
}


def canonical_json(value: Any) -> str:
    """Serialize configuration deterministically, rejecting nonfinite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ComponentSpec:
    """An importable component and its constructor parameters."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    dependencies: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {"type": self.type, "params": copy.deepcopy(self.params), "seed": self.seed}
        if self.dependencies:
            value["dependencies"] = list(self.dependencies)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ComponentSpec:
        return cls(
            value["type"],
            copy.deepcopy(value.get("params", {})),
            value.get("seed"),
            tuple(value.get("dependencies", ())),
        )


@dataclass(frozen=True)
class RunSpec:
    """The complete scientific inputs for one repetition of one algorithm."""

    run_id: str
    group: str
    repetition: int
    algorithm_name: str
    algorithm: ComponentSpec
    data: ComponentSpec
    protocol: ComponentSpec
    seed: int
    budget_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "group": self.group,
            "repetition": self.repetition,
            "algorithm_name": self.algorithm_name,
            "algorithm": self.algorithm.to_dict(),
            "data": self.data.to_dict(),
            "protocol": self.protocol.to_dict(),
            "seed": self.seed,
            "budget_steps": self.budget_steps,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunSpec:
        fields = dict(value)
        for name in ("algorithm", "data", "protocol"):
            fields[name] = ComponentSpec.from_dict(fields[name])
        return cls(**fields)

    @property
    def labels(self) -> dict[str, Any]:
        """Flat parameter labels used to select and group saved results."""
        labels: dict[str, Any] = {
            "group": self.group,
            "repetition": self.repetition,
            "algorithm.name": self.algorithm_name,
            "budget.steps": self.budget_steps,
        }

        def flatten(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for name, nested in value.items():
                    flatten(f"{prefix}.{name}", nested)
            else:
                labels[prefix] = copy.deepcopy(value)

        for name in ("algorithm", "data", "protocol"):
            component = getattr(self, name)
            labels[f"{name}.type"] = component.type
            flatten(f"{name}.params", component.params)
        return labels


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Grid path {path!r} traverses non-mapping parameter {part!r}")
        target = child
    target[parts[-1]] = copy.deepcopy(value)


def _plain_values(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _plain_values(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _plain_values(item)
        return
    raise ValueError("Component parameters must contain plain values and string-keyed mappings")


def make_run_spec(
    *,
    group: str,
    repetition: int,
    algorithm_name: str,
    algorithm: ComponentSpec | Mapping[str, Any],
    data: ComponentSpec | Mapping[str, Any],
    protocol: ComponentSpec | Mapping[str, Any],
    seed: int,
    budget_steps: int | None = None,
) -> RunSpec:
    """Build a validated run with the library's stable scientific identity.

    Custom planners should use this helper instead of generating identifiers.
    Component mappings use the same ``type``, ``params``, and ``seed`` fields as
    YAML; an algorithm mapping may also carry its separately supplied name.
    """
    for name, value in (("group", group), ("algorithm_name", algorithm_name)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Run {name} must be a nonempty string")
    for name, value in (("repetition", repetition), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Run {name} must be a nonnegative integer")
    if budget_steps is not None and (
        isinstance(budget_steps, bool) or not isinstance(budget_steps, int) or budget_steps < 1
    ):
        raise ValueError("Run budget_steps must be a positive integer or None")
    descriptors = {}
    for name, component in (("algorithm", algorithm), ("data", data), ("protocol", protocol)):
        if isinstance(component, Mapping):
            component = ComponentSpec.from_dict(component)
        if not isinstance(component, ComponentSpec):
            raise TypeError(f"Run {name} must be a ComponentSpec or component mapping")
        if not isinstance(component.type, str) or not component.type.strip():
            raise ValueError(f"Run {name}.type must be a nonempty string")
        if not isinstance(component.params, dict):
            raise ValueError(f"Run {name}.params must be a mapping")
        _plain_values(component.params)
        if {"rng", "instance", "logger"}.intersection(component.params):
            raise ValueError(f"Run {name} parameters contain a library-injected argument")
        if not isinstance(component.dependencies, (list, tuple)) or any(
            not isinstance(item, str) or not item for item in component.dependencies
        ):
            raise ValueError(f"Run {name}.dependencies must contain file paths")
        if component.seed is not None and (
            isinstance(component.seed, bool)
            or not isinstance(component.seed, int)
            or component.seed < 0
        ):
            raise ValueError(f"Run {name}.seed must be a nonnegative integer or None")
        descriptors[name] = component.to_dict()
    if descriptors["protocol"]["type"] in _BUDGET_PROTOCOLS:
        legacy_horizon = descriptors["protocol"]["params"].pop("horizon", None)
        if legacy_horizon is not None:
            if (
                isinstance(legacy_horizon, bool)
                or not isinstance(legacy_horizon, int)
                or legacy_horizon < 1
            ):
                raise ValueError("Online horizon must be a positive integer")
            if budget_steps is not None and budget_steps != legacy_horizon:
                raise ValueError(
                    "budget.steps conflicts with protocol.params.horizon; use only budget.steps"
                )
            budget_steps = legacy_horizon
        if budget_steps is None:
            raise ValueError(
                "Online and RL runs require budget.steps or legacy protocol.params.horizon"
            )
    identity = {
        "group": group,
        "repetition": repetition,
        "algorithm_name": algorithm_name,
        **descriptors,
        "seed": seed,
    }
    run_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]
    return RunSpec.from_dict({"run_id": run_id, **identity, "budget_steps": budget_steps})


class RunPlanner(Protocol):
    """A deterministic expansion of one configured run group.

    Implementations have a zero-argument constructor and return RunSpecs. Planning
    must depend only on the supplied group and seed, so calling it again during
    resume or analysis produces the same scientific plan.
    """

    def plan(self, group: Mapping[str, Any], seed: int) -> Iterable[RunSpec]: ...


class GridPlanner:
    """Cross grid coordinates with algorithms and independent repetitions."""

    def plan(self, group: Mapping[str, Any], seed: int) -> Iterable[RunSpec]:
        axes = sorted(group.get("grid", {}))
        values = [group["grid"][axis] for axis in axes]
        for point in itertools.product(*values):
            for algorithm in group["algorithms"]:
                components = {
                    "algorithm": copy.deepcopy(algorithm),
                    "data": copy.deepcopy(group["data"]),
                    "protocol": copy.deepcopy(group["protocol"]),
                }
                for axis, value in zip(axes, point):
                    _set_path(components, axis, value)
                for repetition in range(group["repetitions"]):
                    yield make_run_spec(
                        group=group["name"],
                        repetition=repetition,
                        algorithm_name=algorithm["name"],
                        algorithm=components["algorithm"],
                        data=components["data"],
                        protocol=components["protocol"],
                        seed=seed,
                        budget_steps=group.get("budget", {}).get("steps"),
                    )


def plan_runs(config: ExperimentConfig) -> list[RunSpec]:
    """Expand configured planners and validate every resulting scientific run."""
    from .components import resolve_type

    runs: list[RunSpec] = []
    seen: set[str] = set()
    for group in config.runs:
        name = group["planner"]
        planner = GridPlanner() if name == "grid" else resolve_type(name)()
        if not callable(getattr(planner, "plan", None)):
            raise TypeError(f"Planner {name!r} must implement plan(group, seed)")
        for run in planner.plan(copy.deepcopy(group), config.seed):
            if not isinstance(run, RunSpec):
                raise TypeError(f"Planner {name!r} must yield RunSpec objects")
            if run.group != group["name"] or run.seed != config.seed:
                raise ValueError(
                    f"Planner {name!r} must preserve the run group and experiment seed"
                )
            for component in (run.algorithm, run.data, run.protocol):
                if not isinstance(component, ComponentSpec):
                    raise TypeError(
                        f"Planner {name!r} must yield RunSpecs containing ComponentSpecs"
                    )
            values = run.to_dict()
            values.pop("run_id")
            validated = make_run_spec(**values)
            if run.run_id != validated.run_id:
                raise ValueError(
                    f"Planner {name!r} returned an invalid run ID; use make_run_spec()"
                )
            if run.run_id in seen:
                raise ValueError(f"Duplicate run in group {group['name']!r}: {run.run_id}")
            seen.add(run.run_id)
            runs.append(validated)
    return runs
