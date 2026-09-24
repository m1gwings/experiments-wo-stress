"""Aggregate independent repetitions without retaining every run in memory."""

from __future__ import annotations

import csv
import importlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import BUILTIN_METRICS, Metric, MetricResult, validate_requirements


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


def _source_identity(component: type) -> dict[str, Any]:
    """Fingerprint analysis code and explicitly declared external dependencies."""
    from .storage import digest_file

    sources = {}
    dependencies = {}
    for base in component.__mro__:
        if base is object:
            continue
        source = inspect.getsourcefile(base)
        if source and Path(source).is_file():
            sources[base.__module__] = digest_file(Path(source))
        declared = base.__dict__.get("dependency_files", ())
        if isinstance(declared, (str, Path)):
            raise TypeError("dependency_files must be a sequence of paths")
        for value in declared:
            path = Path(value)
            if not path.is_absolute():
                if not source:
                    raise ValueError("Relative metric dependencies require a source file")
                path = Path(source).resolve().parent / path
            if not path.is_file():
                raise ValueError(f"Missing analysis dependency: {path}")
            dependencies[str(path.resolve())] = digest_file(path)
    if not sources:
        raise ValueError("Cached analysis components must have inspectable Python source")
    return {"sources": sources, "dependencies": dependencies}


def _cache_identity(kind: str, **values: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "numpy": np.__version__,
        "python": list(sys.version_info[:3]),
        **values,
    }


def _validate_cache(directory: Path, identity: Mapping[str, Any]) -> bool:
    from .storage import digest_file, read_json

    if not directory.exists():
        return False
    manifest_path = directory / "cache.json"
    if not manifest_path.is_file():
        raise ValueError(f"Incomplete analysis cache: {directory}")
    manifest = read_json(manifest_path)
    if manifest.get("identity") != identity or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"Analysis cache identity mismatch: {directory}")
    actual_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != set(manifest["files"]):
        raise ValueError(f"Analysis cache file manifest mismatch: {directory}")
    for filename, expected in manifest["files"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid analysis cache path: {filename}")
        path = directory / relative
        if not path.is_file() or digest_file(path) != expected:
            raise ValueError(f"Corrupt analysis cache artifact: {path}")
    return True


def _publish_cache(parent: Path, identity: dict[str, Any], writer: Any) -> Path:
    """Publish immutable, checksum-validated files through a directory rename."""
    from .storage import atomic_json, digest_file, fingerprint

    parent = parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / fingerprint(identity)
    if _validate_cache(destination, identity):
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        writer(temporary)
        checksums = {
            path.relative_to(temporary).as_posix(): digest_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        atomic_json(temporary / "cache.json", {"identity": identity, "files": checksums})
        try:
            os.rename(temporary, destination)
        except OSError:
            if not _validate_cache(destination, identity):
                raise
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _copy_exports(source: Path, destination: Path, filenames: list[str]) -> None:
    """Refresh convenience paths while preserving all immutable cache generations."""
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".export-", dir=target.parent)
        os.close(descriptor)
        try:
            shutil.copyfile(source / filename, temporary)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)


def _result_revision(spec: Any, results: Any) -> str:
    """Support legacy results while new RunResults supply their durable revision."""
    from .storage import fingerprint

    revision = getattr(results, "revision", None)
    if revision is not None:
        return str(revision)
    import hashlib

    digest = hashlib.sha256(_canonical(spec.to_dict()).encode())
    for name in sorted(results):
        array = np.ascontiguousarray(results[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(_canonical(list(array.shape)).encode())
        digest.update(memoryview(array).cast("B"))
    return fingerprint({"legacy": digest.hexdigest()})


def _read_metric(directory: Path) -> MetricResult:
    with np.load(directory / "result.npz", allow_pickle=False) as values:
        return MetricResult(values["x"], values["values"])


def _cached_metric(
    root: Path, spec: Any, results: Any, metric: Metric, descriptor: dict[str, Any]
) -> tuple[Path, str]:
    from .storage import fingerprint, write_arrays

    validate_requirements(metric, results)
    identity = _cache_identity(
        "metric",
        result_revision=_result_revision(spec, results),
        spec=spec.to_dict(),
        metric=descriptor,
    )
    key = fingerprint(identity)
    directory = root / "cache" / "metrics" / key
    if not _validate_cache(directory, identity):

        def write(target: Path) -> None:
            computed = metric.compute(results)
            if not isinstance(computed, MetricResult):
                raise TypeError("Metric.compute() must return MetricResult")
            write_arrays(target / "result.npz", {"x": computed.x, "values": computed.values})

        directory = _publish_cache(directory.parent, identity, write)
    return directory, key


def _validate_analysis_selection(config: Any, output_dir: Path) -> None:
    from .jobs import plan_runs
    from .storage import read_json

    metadata = read_json(output_dir / "metadata.json")
    if metadata.get("schema_version") == 1:
        return
    requests = metadata.get("requests", {})
    expected = sorted(_canonical(spec.to_dict()) for spec in plan_runs(config))
    actual = sorted(_canonical(spec) for spec in requests.values())
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


def analyze(config: Any, output_dir: str | Path) -> list[Summary]:
    """Compute versioned per-run metrics, aggregate repetitions, and export tables.

    Every cache identity includes upstream revisions. Metric changes never alter
    simulation artifacts. One complete RunResult is loaded at a time; aggregation
    reads cached curves rather than retaining every run's observations.
    """
    from .storage import atomic_json, digest_file, fingerprint, iter_completed_runs

    output_dir = Path(output_dir)
    _validate_analysis_selection(config, output_dir)
    metrics = _metric_specs(config.analysis)
    group_by, uncertainty_kind = _aggregation_options(config.analysis)
    descriptors = {}
    for (name, metric), declaration in zip(metrics, config.analysis["metrics"]):
        descriptors[name] = {
            "type": declaration["type"],
            "params": declaration.get("params", {}),
            "implementation": _source_identity(type(metric)),
        }
    analysis_dir = output_dir / "analysis"
    entries = []
    completed = 0
    for spec, results in iter_completed_runs(output_dir):
        completed += 1
        try:
            labels = {label: spec.labels[label] for label in group_by}
        except KeyError as exc:
            raise ValueError(f"Unknown aggregation label {exc.args[0]!r}") from exc
        for name, metric in metrics:
            directory, key = _cached_metric(analysis_dir, spec, results, metric, descriptors[name])
            entries.append(
                {
                    "metric": name,
                    "labels": labels,
                    "scientific_identity": _scientific_identity(spec),
                    "repetition": spec.repetition,
                    "run_id": spec.run_id,
                    "cache_key": key,
                    "directory": str(directory),
                }
            )
    if not completed:
        raise ValueError("No validated completed runs are available for analysis")
    entries.sort(key=lambda item: (item["metric"], _canonical(item["labels"]), item["run_id"]))
    aggregate_identity = _cache_identity(
        "aggregate",
        implementation=digest_file(Path(__file__)),
        options={"group_by": group_by, "uncertainty": uncertainty_kind},
        metrics=[
            {key: value for key, value in entry.items() if key != "directory"} for entry in entries
        ],
    )

    def aggregate(target: Path) -> None:
        groups: dict[tuple[str, str], tuple[dict[str, Any], _Accumulator]] = {}
        for entry in entries:
            result = _read_metric(Path(entry["directory"]))
            group_key = (entry["metric"], _canonical(entry["labels"]))
            if group_key not in groups:
                groups[group_key] = (
                    entry["labels"],
                    _Accumulator(
                        x=result.x.copy(),
                        mean=np.zeros_like(result.values),
                        squared_deviations=np.zeros_like(result.values),
                        identity=entry["scientific_identity"],
                    ),
                )
            groups[group_key][1].add(result, entry["repetition"], entry["scientific_identity"])
        summaries = []
        for (name, _), (labels, accumulator) in sorted(groups.items()):
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
                    name,
                    labels,
                    accumulator.x,
                    accumulator.mean,
                    uncertainty,
                    accumulator.count,
                    uncertainty_kind,
                )
            )
        _write_tables(target, summaries, group_by)

    directory = _publish_cache(analysis_dir / "cache" / "aggregates", aggregate_identity, aggregate)
    summaries = _read_summaries(directory)
    filenames = ["metadata.json"]
    for metric in sorted({summary.metric for summary in summaries}):
        filenames.extend((f"{metric}.csv", f"{metric}.npz"))
    _copy_exports(directory, analysis_dir, filenames)
    atomic_json(
        analysis_dir / "current.json",
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
