"""Compute saved-run metrics and aggregate independent repetitions."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..builtins.metrics import PseudoRegretMetric, RealizedRegretMetric
from ..components.loading import load_class
from ..storage import experiment as experiment_storage
from ..storage.files import atomic_json, digest_file, fingerprint, read_json, write_arrays
from ..study.planning import plan_runs
from ..study.specs import RunSpec, canonical_json
from . import metrics as metric_definitions
from .cache import AnalysisCache, cache_identity, safe_name, source_identity
from .metrics import CumulativeSumMetric, FieldMetric, Metric, MetricResult, validate_requirements

BUILTIN_METRICS = {
    "field": FieldMetric,
    "cumulative_sum": CumulativeSumMetric,
    "pseudo_regret": PseudoRegretMetric,
    "realized_regret": RealizedRegretMetric,
}

if TYPE_CHECKING:
    from ..storage.models import RunResult
    from ..study.config import ExperimentConfig


@dataclass(frozen=True)
class Summary:
    """Hold an aggregate curve for one metric and one group of independent runs.

    ``labels`` identifies the configured group. ``mean`` and ``uncertainty`` share
    coordinates ``x``; ``count`` is the number of distinct repetitions contributing
    at every point. ``uncertainty_kind`` names standard deviation, standard error,
    or no uncertainty, so table and figure consumers can interpret the values.
    """

    metric: str
    labels: dict[str, Any]
    x: np.ndarray
    mean: np.ndarray
    uncertainty: np.ndarray
    count: int
    uncertainty_kind: str


@dataclass
class _RepetitionAccumulator:
    """Maintain a pointwise running mean and variance numerator for one group.

    Each incoming curve must share the group's scientific identity and exact x
    coordinates, and contribute a previously unseen repetition. Welford's update
    keeps only the mean and squared deviations, avoiding storage of every curve.
    """

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
        # Welford's update accumulates the variance numerator without subtracting
        # two potentially large sums at the end of aggregation.
        delta = result.values - self.mean
        self.mean += delta / self.count
        self.squared_deviations += delta * (result.values - self.mean)


@dataclass(frozen=True)
class _CachedMetric:
    """Reference a saved per-run metric curve without keeping its arrays in memory.

    Group labels and scientific identity govern pooling; the cache key identifies
    the metric's exact inputs and implementation. Aggregation loads the curve from
    ``directory`` only when needed and uses ``identity`` to version its output.
    """

    metric: str
    labels: dict[str, Any]
    scientific_identity: str
    repetition: int
    run_id: str
    cache_key: str
    directory: Path

    def identity(self) -> dict[str, Any]:
        # Cache locations are incidental; scientific inputs and upstream revisions
        # determine whether an aggregate can be reused.
        return {
            "metric": self.metric,
            "labels": self.labels,
            "scientific_identity": self.scientific_identity,
            "repetition": self.repetition,
            "run_id": self.run_id,
            "cache_key": self.cache_key,
        }


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
        name = safe_name(spec.get("name"), "Metric name")
        if name in names:
            raise ValueError(f"Duplicate metric name {name!r}")
        names.add(name)
        params = spec.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"Parameters for metric {name!r} must be a mapping")
        metric = load_class(spec.get("type"), BUILTIN_METRICS)(**params)
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


def _scientific_identity(run_spec: RunSpec) -> str:
    identity = dict(run_spec.to_dict())
    identity.pop("run_id", None)
    identity.pop("repetition", None)
    return canonical_json(identity)


def _result_revision(run_spec: RunSpec, results: Mapping[str, np.ndarray]) -> str:
    """Support legacy results while new RunResults supply their durable revision."""
    revision = getattr(results, "revision", None)
    if revision is not None:
        return str(revision)
    digest = hashlib.sha256(canonical_json(run_spec.to_dict()).encode())
    for name in sorted(results):
        array = np.ascontiguousarray(results[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(canonical_json(list(array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return fingerprint({"legacy": digest.hexdigest()})


def _read_metric(directory: Path) -> MetricResult:
    with np.load(directory / "result.npz", allow_pickle=False) as values:
        return MetricResult(values["x"], values["values"])


def _cached_metric(
    cache: AnalysisCache,
    run_spec: RunSpec,
    results: RunResult,
    metric: Metric,
    descriptor: dict[str, Any],
) -> tuple[Path, str]:
    validate_requirements(metric, results)
    identity = cache_identity(
        "metric",
        implementation={
            "pipeline": digest_file(Path(__file__)),
            "metrics": digest_file(Path(metric_definitions.__file__)),
        },
        result_revision=_result_revision(run_spec, results),
        spec=run_spec.to_dict(),
        metric=descriptor,
    )
    key = fingerprint(identity)
    directory = cache.find("metrics", identity)
    if directory is None:

        def write(target: Path) -> None:
            computed = metric.compute(results)
            if not isinstance(computed, MetricResult):
                raise TypeError("Metric.compute() must return MetricResult")
            write_arrays(target / "result.npz", {"x": computed.x, "values": computed.values})

        directory = cache.publish("metrics", identity, write)
    return directory, key


def _validate_analysis_selection(config: ExperimentConfig, output_dir: Path) -> None:
    metadata = read_json(output_dir / "metadata.json")
    if metadata.get("schema_version") == 1:
        return
    requests = metadata.get("requests", {})
    expected = sorted(canonical_json(spec.to_dict()) for spec in plan_runs(config))
    actual = sorted(canonical_json(spec) for spec in requests.values())
    if expected != actual:
        raise ValueError(
            "Analysis configuration does not match the stored requested run plan/budget; "
            "run the requested configuration first, or analyze its saved resolved configuration"
        )


def _read_summaries(directory: Path) -> list[Summary]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    summaries = []
    for metric, entries in sorted(metadata["metrics"].items()):
        with np.load(directory / f"{metric}.npz", allow_pickle=False) as arrays:
            for entry in entries:
                prefix = entry["prefix"]
                summaries.append(
                    Summary(
                        metric,
                        entry["labels"],
                        arrays[f"{prefix}_x"],
                        arrays[f"{prefix}_mean"],
                        arrays[f"{prefix}_uncertainty"],
                        entry["count"],
                        entry["uncertainty_kind"],
                    )
                )
    return summaries


def analyze(config: ExperimentConfig, output_dir: str | Path) -> list[Summary]:
    """Compute versioned per-run metrics, aggregate repetitions, and export tables.

    Every cache identity includes upstream revisions. Metric changes never alter
    simulation artifacts. One complete RunResult is loaded at a time; aggregation
    reads cached curves rather than retaining every run's observations.
    """
    output_dir = Path(output_dir)
    _validate_analysis_selection(config, output_dir)
    metrics = _metric_specs(config.analysis)
    group_by, uncertainty_kind = _aggregation_options(config.analysis)
    descriptors = {}
    for (name, metric), declaration in zip(metrics, config.analysis["metrics"]):
        descriptors[name] = {
            "type": declaration["type"],
            "params": declaration.get("params", {}),
            "implementation": source_identity(type(metric)),
        }
    cache = AnalysisCache(output_dir)
    entries: list[_CachedMetric] = []
    completed = 0
    # Compute or reuse each metric while only one run's raw observations are
    # loaded. Retain lightweight references for the subsequent aggregation pass.
    for run_spec, results in experiment_storage.iter_completed_runs(output_dir):
        completed += 1
        try:
            labels = {label: run_spec.labels[label] for label in group_by}
        except KeyError as exc:
            raise ValueError(f"Unknown aggregation label {exc.args[0]!r}") from exc
        for name, metric in metrics:
            directory, key = _cached_metric(cache, run_spec, results, metric, descriptors[name])
            entries.append(
                _CachedMetric(
                    metric=name,
                    labels=labels,
                    scientific_identity=_scientific_identity(run_spec),
                    repetition=run_spec.repetition,
                    run_id=run_spec.run_id,
                    cache_key=key,
                    directory=directory,
                )
            )
    if not completed:
        raise ValueError("No validated completed runs are available for analysis")
    # Canonical order gives an aggregate the same identity regardless of the order
    # in which run artifacts happened to be produced or discovered.
    entries.sort(key=lambda entry: (entry.metric, canonical_json(entry.labels), entry.run_id))
    aggregate_identity = cache_identity(
        "aggregate",
        implementation=digest_file(Path(__file__)),
        options={"group_by": group_by, "uncertainty": uncertainty_kind},
        metrics=[entry.identity() for entry in entries],
    )

    def write_aggregate(target: Path) -> None:
        summaries = _aggregate_metrics(entries, uncertainty_kind)
        _write_tables(target, summaries, group_by)

    directory = cache.publish("aggregates", aggregate_identity, write_aggregate)
    summaries = _read_summaries(directory)
    filenames = ["metadata.json"]
    for metric in sorted({summary.metric for summary in summaries}):
        filenames.extend((f"{metric}.csv", f"{metric}.npz"))
    cache.export(directory, filenames)
    atomic_json(
        cache.current_path,
        {"aggregate_key": fingerprint(aggregate_identity), "completed_runs": completed},
    )
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


def _aggregate_metrics(entries: list[_CachedMetric], uncertainty_kind: str) -> list[Summary]:
    """Reduce cached curves without retaining all run observations in memory."""
    groups: dict[tuple[str, str], tuple[dict[str, Any], _RepetitionAccumulator]] = {}
    for entry in entries:
        result = _read_metric(entry.directory)
        group_key = (entry.metric, canonical_json(entry.labels))
        if group_key not in groups:
            groups[group_key] = (
                entry.labels,
                _RepetitionAccumulator(
                    x=result.x.copy(),
                    mean=np.zeros_like(result.values),
                    squared_deviations=np.zeros_like(result.values),
                    identity=entry.scientific_identity,
                ),
            )
        groups[group_key][1].add(result, entry.repetition, entry.scientific_identity)
    summaries = []
    for (name, _), (labels, accumulator) in sorted(groups.items()):
        uncertainty = np.zeros_like(accumulator.mean)
        if uncertainty_kind != "none":
            uncertainty.fill(np.nan)
            if accumulator.count > 1:
                # The sample variance uses independent repetitions, not time
                # points. One repetition leaves uncertainty undefined (NaN).
                variance = accumulator.squared_deviations / (accumulator.count - 1)
                uncertainty = np.sqrt(np.maximum(variance, 0))
                if uncertainty_kind == "standard_error":
                    uncertainty /= np.sqrt(accumulator.count)
        summaries.append(
            Summary(
                name,
                labels,
                accumulator.x,
                accumulator.mean,
                uncertainty,
                accumulator.count,
                uncertainty_kind,
            )
        )
    return summaries
