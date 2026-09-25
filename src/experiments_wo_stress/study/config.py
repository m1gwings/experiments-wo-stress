"""Safe YAML loading and validation of the experiment's public contract."""

from __future__ import annotations

import copy
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .planning import plan_runs
from .specs import ComponentSpec, RunSpec, canonical_json

__all__ = ["ComponentSpec", "ExperimentConfig", "RunSpec", "load_config"]

_EXECUTION = {
    "workers": 1,
    "checkpoint_seconds": 120.0,
    "checkpoint_steps": None,
    "keep_checkpoints": 2,
    "compression": False,
    "logging_level": "INFO",
    "log_max_bytes": 2 * 1024 * 1024,
    "log_backups": 2,
}
_RECORDING = {"every_steps": 1, "fields": None, "buffer_bytes": 16 * 1024 * 1024}
_CSV_TYPES = {"csv", "experiments_wo_stress.data:CSVDataGenerator"}


_NOTIFICATION_DEFAULTS = {
    "enabled": True,
    "webhook_env": "EWS_DISCORD_WEBHOOK_URL",
    "interval_seconds": 300.0,
    "timeout_seconds": 5.0,
}


def validate_notifications(value: Any) -> dict[str, Any]:
    """Normalize optional settings without resolving or persisting a webhook secret."""
    if not isinstance(value, dict) or set(value) - {"discord"}:
        raise ValueError("notifications must be a mapping containing only discord")
    if "discord" not in value:
        return {}
    options = value["discord"]
    if not isinstance(options, dict) or set(options) - set(_NOTIFICATION_DEFAULTS):
        raise ValueError(
            "notifications.discord accepts enabled, webhook_env, interval_seconds, and timeout_seconds"
        )
    options = {**_NOTIFICATION_DEFAULTS, **options}
    if not isinstance(options["enabled"], bool):
        raise ValueError("notifications.discord.enabled must be a boolean")
    name = options["webhook_env"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("notifications.discord.webhook_env must be an environment variable name")
    for key in ("interval_seconds", "timeout_seconds"):
        number = options[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"notifications.discord.{key} must be finite and positive")
    if options["interval_seconds"] < 1:
        raise ValueError("notifications.discord.interval_seconds must be at least 1")
    if options["timeout_seconds"] > 30:
        raise ValueError("notifications.discord.timeout_seconds must not exceed 30")
    return {"discord": options}


@dataclass(frozen=True)
class ExperimentConfig:
    """Effective experiment settings; source_dir is an import/path context."""

    name: str
    seed: int
    runs: list[dict[str, Any]]
    execution: dict[str, Any] = field(default_factory=lambda: dict(_EXECUTION))
    recording: dict[str, Any] = field(default_factory=lambda: dict(_RECORDING))
    analysis: dict[str, Any] = field(default_factory=dict)
    source_dir: Path = field(default_factory=Path.cwd)
    notifications: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "name": self.name,
                "seed": self.seed,
                "runs": self.runs,
                "execution": self.execution,
                "recording": self.recording,
                "analysis": self.analysis,
                "notifications": self.notifications,
            }
        )

    def simulation_dict(self) -> dict[str, Any]:
        """Canonical scientific contract used to reject incompatible resumption.

        Planning order and operational tuning do not affect scientific results.
        Recording selection does: omitted observations cannot be recreated later.
        """
        planned = sorted(plan_runs(self), key=lambda run: run.run_id)
        fields = self.recording["fields"]
        return {
            "name": self.name,
            "runs": [run.to_dict() for run in planned],
            "recording": {
                "every_steps": self.recording["every_steps"],
                "fields": sorted(fields) if fields is not None else None,
            },
        }


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently replacing experiment settings."""


def _mapping_constructor(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, Any]:
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ValueError(f"YAML keys must be strings (line {key_node.start_mark.line + 1})")
        if key in mapping:
            raise ValueError(f"Duplicate YAML key {key!r} (line {key_node.start_mark.line + 1})")
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping_constructor
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be a mapping with string keys")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"Unknown keys in {path}: {', '.join(sorted(extra))}")


def _integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")
    return value


def _json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        _mapping(value, path)
        for name, item in value.items():
            _json_value(item, f"{path}.{name}")
        return
    raise ValueError(
        f"{path} contains unsupported YAML value {type(value).__name__}; use plain values"
    )


def _component(
    value: Any, path: str, source_dir: Path, *, algorithm: bool = False
) -> dict[str, Any]:
    value = copy.deepcopy(_mapping(value, path))
    _keys(
        value,
        {"type", "params", "seed", "dependencies", "name"}
        if algorithm
        else {"type", "params", "seed", "dependencies"},
        path,
    )
    _text(value.get("type"), f"{path}.type")
    value.setdefault("params", {})
    _mapping(value["params"], f"{path}.params")
    if {"rng", "instance", "logger"}.intersection(value["params"]):
        raise ValueError(
            f"{path}.params contains a library-injected argument (rng, instance, or logger)"
        )
    if "dependencies" in value:
        dependencies = value["dependencies"]
        if not isinstance(dependencies, list) or any(
            not isinstance(item, str) or not item for item in dependencies
        ):
            raise ValueError(f"{path}.dependencies must be a list of file paths")
        value["dependencies"] = sorted(
            {str((source_dir / item).resolve()) for item in dependencies}
        )
    value.setdefault("seed", None)
    if value["seed"] is not None:
        _integer(value["seed"], f"{path}.seed")
    if algorithm:
        value.setdefault("name", value["type"].split(":")[-1])
        _text(value["name"], f"{path}.name")
    if value["type"] in _CSV_TYPES and "path" in value["params"]:
        filename = _text(value["params"]["path"], f"{path}.params.path")
        value["params"]["path"] = str((source_dir / filename).resolve())
    return value


def _group(value: Any, index: int, source_dir: Path) -> dict[str, Any]:
    path = f"runs[{index}]"
    value = copy.deepcopy(_mapping(value, path))
    _keys(
        value,
        {"name", "planner", "repetitions", "protocol", "data", "algorithms", "grid", "budget"},
        path,
    )
    if "budget" in value:
        budget = _mapping(value["budget"], f"{path}.budget")
        _keys(budget, {"steps"}, f"{path}.budget")
        _integer(budget.get("steps"), f"{path}.budget.steps", 1)
    value.setdefault("name", f"group_{index}")
    _text(value["name"], f"{path}.name")
    value.setdefault("planner", "grid")
    planner = _text(value["planner"], f"{path}.planner")
    if planner != "grid" and (":" not in planner or not all(planner.split(":", 1))):
        raise ValueError(f"{path}.planner must be 'grid' or a module:Class import path")
    value.setdefault("repetitions", 1)
    _integer(value["repetitions"], f"{path}.repetitions", 1)
    for name in ("protocol", "data"):
        value[name] = _component(value.get(name), f"{path}.{name}", source_dir)
    algorithms = value.get("algorithms")
    if not isinstance(algorithms, list) or not algorithms:
        raise ValueError(f"{path}.algorithms must be a nonempty list")
    value["algorithms"] = [
        _component(algorithm, f"{path}.algorithms[{item}]", source_dir, algorithm=True)
        for item, algorithm in enumerate(algorithms)
    ]
    names = [algorithm["name"] for algorithm in value["algorithms"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate algorithm names")
    value.setdefault("grid", {})
    _mapping(value["grid"], f"{path}.grid")
    for axis, candidates in value["grid"].items():
        parts = axis.split(".")
        if (
            len(parts) < 3
            or parts[0] not in {"algorithm", "data", "protocol"}
            or parts[1] != "params"
            or any(not part for part in parts)
            or parts[2] in {"rng", "instance", "logger"}
        ):
            raise ValueError(
                f"Invalid grid path {axis!r}; use algorithm.params.*, data.params.*, or protocol.params.*"
            )
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{path}.grid.{axis} must be a nonempty list")
        if axis == "data.params.path" and value["data"]["type"] in _CSV_TYPES:
            candidates = [str((source_dir / _text(item, axis)).resolve()) for item in candidates]
            value["grid"][axis] = candidates
        if len({canonical_json(item) for item in candidates}) != len(candidates):
            raise ValueError(f"{path}.grid.{axis} contains duplicate values")
    for axis in value["grid"]:
        if any(other.startswith(axis + ".") for other in value["grid"]):
            raise ValueError(f"Grid axes overlap at {axis!r}")
    return value


def _execution_settings(value: Any) -> dict[str, Any]:
    execution = _mapping(value, "execution")
    _keys(execution, set(_EXECUTION), "execution")
    execution = {**_EXECUTION, **execution}
    for key in ("workers", "keep_checkpoints"):
        _integer(execution[key], f"execution.{key}", 1)
    _integer(execution["log_max_bytes"], "execution.log_max_bytes", 1)
    _integer(execution["log_backups"], "execution.log_backups", 1)
    level = execution["logging_level"]
    if not isinstance(level, str) or level.upper() not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise ValueError("execution.logging_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    execution["logging_level"] = level.upper()
    if execution["checkpoint_steps"] is not None:
        _integer(execution["checkpoint_steps"], "execution.checkpoint_steps", 1)
    seconds = execution["checkpoint_seconds"]
    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0
    ):
        raise ValueError("execution.checkpoint_seconds must be positive or null")
    if not isinstance(execution["compression"], bool):
        raise ValueError("execution.compression must be a boolean")
    return execution


def _recording_settings(value: Any) -> dict[str, Any]:
    recording = _mapping(value, "recording")
    _keys(recording, set(_RECORDING), "recording")
    recording = {**_RECORDING, **recording}
    for key in ("every_steps", "buffer_bytes"):
        _integer(recording[key], f"recording.{key}", 1)
    fields = recording["fields"]
    if fields is not None:
        if not isinstance(fields, list) or any(
            not isinstance(item, str) or not item for item in fields
        ):
            raise ValueError("recording.fields must be a list of nonempty field names or null")
        if len(set(fields)) != len(fields):
            raise ValueError("recording.fields must not contain duplicate names")
    return recording


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML file with defaults, path resolution, and actionable errors."""
    path = Path(path).resolve()
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    raw = _mapping(raw, "experiment")
    _keys(
        raw,
        {"name", "seed", "runs", "execution", "recording", "analysis", "notifications"},
        "experiment",
    )
    _json_value(raw, "experiment")
    name = _text(raw.get("name"), "name")
    seed = _integer(raw.get("seed", 0), "seed")
    if not isinstance(raw.get("runs"), list) or not raw["runs"]:
        raise ValueError("runs must be a nonempty list")
    groups = [_group(group, index, path.parent) for index, group in enumerate(raw["runs"])]
    if len({group["name"] for group in groups}) != len(groups):
        raise ValueError("Run group names must be unique")

    execution = _execution_settings(raw.get("execution", {}))
    recording = _recording_settings(raw.get("recording", {}))
    analysis = _mapping(raw.get("analysis", {}), "analysis")
    _keys(analysis, {"metrics", "aggregator", "figures"}, "analysis")
    notifications = validate_notifications(raw.get("notifications", {}))
    config = ExperimentConfig(
        name, seed, groups, execution, recording, analysis, path.parent, notifications
    )
    source = str(path.parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    return config
