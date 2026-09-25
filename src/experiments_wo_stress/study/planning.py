"""Expand experiment groups into validated, deterministic run plans."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ..components.loading import resolve_type
from .specs import ComponentSpec, RunSpec, make_run_spec

if TYPE_CHECKING:
    from .config import ExperimentConfig


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        child = target.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Grid path {path!r} traverses non-mapping parameter {part!r}")
        target = child
    target[parts[-1]] = copy.deepcopy(value)


class RunPlanner(Protocol):
    """A deterministic expansion of one configured run group.

    Implementations have a zero-argument constructor and return RunSpecs. Planning
    must depend only on the supplied group and seed, so calling it again during
    resume or analysis produces the same scientific plan.
    """

    def plan(self, group: Mapping[str, Any], seed: int) -> Iterable[RunSpec]: ...


class GridPlanner:
    """Expand one group into a Cartesian grid of independent run descriptions.

    For each grid point, copy the component settings, apply the selected values,
    and yield every algorithm/repetition combination through ``make_run_spec``.
    Sorting axis names makes traversal predictable; run identities and RNG streams
    depend on scientific inputs rather than their positions in that traversal.
    """

    def plan(self, group: Mapping[str, Any], seed: int) -> Iterable[RunSpec]:
        """Yield one validated description per grid point, algorithm, and repetition."""
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
