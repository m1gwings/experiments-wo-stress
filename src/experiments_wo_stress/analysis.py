"""Aggregate independent repetitions without retaining every run in memory."""

from __future__ import annotations

import csv
import importlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import BUILTIN_METRICS, Metric, MetricResult


@dataclass(frozen=True)
class Summary:
    """Pointwise estimates for one metric and one group of independent runs."""

    metric: str
    labels: dict[str, Any]
    x: np.ndarray
    mean: np.ndarray
    uncertainty: np.ndarray
    count: int
    uncertainty_kind: str


@dataclass
class _Accumulator:
    x: np.ndarray
    mean: np.ndarray
    squared_deviations: np.ndarray
    identity: str
    repetitions: set[int] = field(default_factory=set)
    count: int = 0

    def add(self, result: MetricResult, repetition: int, identity: str) -> None:
        if self.identity != identity:
            raise ValueError(
                "Aggregation would pool different scientific configurations; "
                "add their differing labels to analysis.aggregator.group_by"
            )
        if repetition in self.repetitions:
            raise ValueError(f"Duplicate repetition {repetition} within an analysis group")
        if not np.array_equal(self.x, result.x):
            raise ValueError("Repetitions in an analysis group have mismatched x coordinates")
        self.repetitions.add(repetition)
        self.count += 1
        delta = result.values - self.mean
        self.mean += delta / self.count
        self.squared_deviations += delta * (result.values - self.mean)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_name(value: Any, description: str = "Name") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{description} must use letters, digits, underscores, dots, or hyphens")
    return value


def _load_class(type_name: str, builtins: Mapping[str, type] | None = None) -> type:
    if not isinstance(type_name, str):
        raise ValueError("Component type must be a string")
    if builtins and type_name in builtins:
        return builtins[type_name]
    if not isinstance(type_name, str) or ":" not in type_name:
        raise ValueError(f"Unknown component {type_name!r}; use a built-in alias or module:Class")
    module_name, class_name = type_name.split(":", 1)
    try:
        component = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import component {type_name!r}: {exc}") from exc
    if not isinstance(component, type):
        raise TypeError(f"Component {type_name!r} must identify a class")
    return component


def _metric_specs(analysis: Mapping[str, Any]) -> list[tuple[str, Metric]]:
    specs = analysis.get("metrics", [])
    if not isinstance(specs, list) or not specs:
        raise ValueError("analysis.metrics must contain at least one metric")
    metrics: list[tuple[str, Metric]] = []
    names: set[str] = set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each analysis metric must be a mapping")
        unknown = set(spec) - {"name", "type", "params"}
        if unknown:
            raise ValueError(f"Unknown metric options: {', '.join(sorted(unknown))}")
        name = _safe_name(spec.get("name"), "Metric name")
        if name in names:
            raise ValueError(f"Duplicate metric name {name!r}")
        names.add(name)
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Parameters for metric {name!r} must be a mapping")
        metric = _load_class(spec.get("type"), BUILTIN_METRICS)(**params)
        if not callable(getattr(metric, "compute", None)):
            raise TypeError(f"Metric {name!r} must implement compute(results)")
        metrics.append((name, metric))
    return metrics


def _aggregation_options(analysis: Mapping[str, Any]) -> tuple[list[str], str]:
    options = analysis.get("aggregator", {})
    if not isinstance(options, dict):
        raise ValueError("analysis.aggregator must be a mapping")
    unknown = set(options) - {"type", "reduce_over", "summary", "group_by", "uncertainty"}
    if unknown:
        raise ValueError(f"Unknown aggregator options: {', '.join(sorted(unknown))}")
    if options.get("type", "repeated_runs") != "repeated_runs":
        raise ValueError("Only the repeated_runs aggregator is supported")
    if options.get("reduce_over", "repetition") != "repetition":
        raise ValueError("The repeated_runs aggregator reduces over repetition")
    if options.get("summary", "mean") != "mean":
        raise ValueError("The repeated_runs aggregator supports summary: mean")
    group_by = options.get("group_by", ["group", "algorithm.name"])
    if (
        not isinstance(group_by, list)
        or any(not isinstance(label, str) for label in group_by)
        or len(set(group_by)) != len(group_by)
        or "repetition" in group_by
    ):
        raise ValueError("group_by must be a list of distinct labels excluding repetition")
    uncertainty = options.get("uncertainty", "standard_error")
    if not isinstance(uncertainty, str) or uncertainty not in {"standard_error", "std", "none"}:
        raise ValueError("uncertainty must be standard_error, std, or none")
    return group_by, uncertainty


def _scientific_identity(spec: Any) -> str:
    identity = dict(spec.to_dict())
    identity.pop("run_id", None)
    identity.pop("repetition", None)
    return _canonical(identity)


def analyze(config: Any, output_dir: str | Path) -> list[Summary]:
    """Read validated completed results, summarize repetitions, and write tables.

    Simulation components are never imported or instantiated here. Only the
    requested custom metric classes must be importable. Each run's observations
    are loaded once, then released after updating the running summaries.
    """
    from .storage import iter_completed_runs

    metrics = _metric_specs(config.analysis)
    group_by, uncertainty_kind = _aggregation_options(config.analysis)
    groups: dict[tuple[str, str], tuple[dict[str, Any], _Accumulator]] = {}
    completed = 0
    for spec, results in iter_completed_runs(Path(output_dir)):
        completed += 1
        try:
            labels = {label: spec.labels[label] for label in group_by}
        except KeyError as exc:
            raise ValueError(f"Unknown aggregation label {exc.args[0]!r}") from exc
        key = _canonical(labels)
        identity = _scientific_identity(spec)
        for name, metric in metrics:
            result = metric.compute(results)
            if not isinstance(result, MetricResult):
                raise TypeError(f"Metric {name!r} must return MetricResult")
            group_key = (name, key)
            if group_key not in groups:
                groups[group_key] = (
                    labels,
                    _Accumulator(
                        x=result.x.copy(),
                        mean=np.zeros_like(result.values),
                        squared_deviations=np.zeros_like(result.values),
                        identity=identity,
                    ),
                )
            groups[group_key][1].add(result, spec.repetition, identity)
    if not completed:
        raise ValueError("No validated completed runs are available for analysis")

    summaries: list[Summary] = []
    for (metric, _), (labels, accumulator) in sorted(groups.items()):
        uncertainty = np.zeros_like(accumulator.mean)
        if uncertainty_kind != "none":
            uncertainty.fill(np.nan)
            if accumulator.count > 1:
                variance = accumulator.squared_deviations / (accumulator.count - 1)
                uncertainty = np.sqrt(np.maximum(variance, 0))
                if uncertainty_kind == "standard_error":
                    uncertainty /= np.sqrt(accumulator.count)
        summaries.append(
            Summary(
                metric,
                labels,
                accumulator.x,
                accumulator.mean,
                uncertainty,
                accumulator.count,
                uncertainty_kind,
            )
        )
    _write_tables(Path(output_dir) / "analysis", summaries, group_by)
    return summaries


def _write_tables(directory: Path, summaries: list[Summary], group_by: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "metrics": {}}
    for metric in sorted({summary.metric for summary in summaries}):
        selected = [summary for summary in summaries if summary.metric == metric]
        table_path = directory / f"{metric}.csv"
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*group_by, "x", "mean", "uncertainty", "count", "uncertainty_kind"])
            for summary in selected:
                for x, mean, uncertainty in zip(summary.x, summary.mean, summary.uncertainty):
                    writer.writerow(
                        [
                            *[summary.labels[label] for label in group_by],
                            x,
                            mean,
                            uncertainty,
                            summary.count,
                            summary.uncertainty_kind,
                        ]
                    )
        arrays = {}
        entries = []
        for index, summary in enumerate(selected):
            prefix = f"group_{index}"
            arrays[f"{prefix}_x"] = summary.x
            arrays[f"{prefix}_mean"] = summary.mean
            arrays[f"{prefix}_uncertainty"] = summary.uncertainty
            entries.append(
                {
                    "prefix": prefix,
                    "labels": summary.labels,
                    "count": summary.count,
                    "uncertainty_kind": summary.uncertainty_kind,
                }
            )
        np.savez(directory / f"{metric}.npz", **arrays)
        manifest["metrics"][metric] = entries
    (directory / "metadata.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
