"""Dependency planning and worker tasks for durable, independently reusable analysis."""

from __future__ import annotations

import inspect
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..storage.artifact_index import publish_artifact_index
from ..storage.experiment import ExperimentStore
from ..storage.files import atomic_json, digest_file, fingerprint, read_json
from ..study.config import analysis_points
from ..study.specs import RunSpec, canonical_json
from . import metric_tasks, pipeline
from . import metrics as metric_definitions
from .cache import AnalysisCache, cache_identity, safe_name, source_identity

if TYPE_CHECKING:
    from ..study.config import ExperimentConfig


@dataclass(frozen=True)
class DerivationTask:
    """Describe one immutable artifact publication sent to an available worker.

    Payloads contain plain descriptors and cache references, never live simulation
    objects. ``run_ids`` identifies the scientific ancestors for progress and
    scheduling; it does not imply that their trajectories remain necessary.
    """

    id: str
    kind: str
    subject: str
    run_ids: tuple[str, ...]
    output_dir: str
    identity: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class _Node:
    """Track one derivation's dependencies and validated publication in this request."""

    id: str
    kind: str
    subject: str
    run_ids: tuple[str, ...]
    dependencies: tuple[str, ...]
    payload: dict[str, Any]
    identity: dict[str, Any] | None = None
    key: str | None = None
    directory: Path | None = None
    checked_key: str | None = None
    inputs: tuple[Any, ...] | None = None


_COLLECTIONS = {"metric": "metrics", "aggregate": "aggregates", "figure": "plots"}


class AnalysisGraph:
    """Own an invocation's analysis DAG, resolving only necessary retained ancestors.

    The coordinator supplies selected run variants and advances this graph after
    durable worker publications. Metric identities use saved result receipts;
    aggregate and figure identities can consequently be proved without opening
    deleted trajectories or requiring retained upstream cache files. Traversal
    stops at the first valid generation. Scheduling and worker ownership remain
    entirely in execution infrastructure.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str | Path,
        specs: list[RunSpec],
        locations: dict[str, str],
        *,
        figures: bool = True,
        figures_only: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.cache = AnalysisCache(output_dir)
        self.specs = {spec.run_id: spec for spec in specs}
        self.locations = locations
        self.group_by, self.uncertainty = pipeline._aggregation_options(config.analysis)
        self.points = analysis_points(config.analysis)
        self.nodes: dict[str, _Node] = {}
        self.references: dict[str, dict[str, Any] | None] = {}
        self.metric_nodes: dict[str, list[str]] = defaultdict(list)
        self.units: dict[str, str] = {}
        self._base_identity = cache_identity(
            "dependency", dependency_implementation=_dependency_implementation()
        )
        self._metric_implementation = {
            "computation": digest_file(Path(metric_tasks.__file__)),
            "contract": digest_file(Path(metric_definitions.__file__)),
        }
        self._aggregate_implementation = digest_file(Path(pipeline.__file__))
        self._missing_raw: dict[str, set[str]] = defaultdict(set)
        self._ready: set[str] = set()
        self._targets: list[str] = []
        self._finished: set[str] = set()
        self._needed_by: dict[str, set[str]] = defaultdict(set)
        self._expanded: set[str] = set()
        self._build(config, figures, figures_only)
        self._order = {node_id: index for index, node_id in enumerate(self.nodes)}
        self._dependents: dict[str, set[str]] = defaultdict(set)
        self._unresolved = {node.id: set(node.dependencies) for node in self.nodes.values()}
        for node in self.nodes.values():
            for dependency in node.dependencies:
                self._dependents[dependency].add(node.id)
        self.refresh()

    def _identity(self, kind: str, **values: Any) -> dict[str, Any]:
        return {**self._base_identity, "kind": kind, **values}

    def _build(self, config: ExperimentConfig, figures: bool, figures_only: bool) -> None:
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        metrics = pipeline._metric_specs(config.analysis)
        descriptors = {
            name: {
                "type": declaration["type"],
                "params": declaration.get("params", {}),
                "implementation": source_identity(type(metric)),
            }
            for (name, metric), declaration in zip(metrics, config.analysis["metrics"])
        }
        for spec in sorted(self.specs.values(), key=lambda item: item.run_id):
            try:
                labels = {label: spec.labels[label] for label in self.group_by}
            except KeyError as exc:
                raise ValueError(f"Unknown aggregation label {exc.args[0]!r}") from exc
            unit = canonical_json(labels)
            self.units[spec.run_id] = unit
            for name, _ in metrics:
                node_id = f"metric:{name}:{spec.run_id}"
                self.nodes[node_id] = _Node(
                    node_id,
                    "metric",
                    f"{name} / {spec.algorithm_name} r{spec.repetition:03d}",
                    (spec.run_id,),
                    (),
                    {
                        "name": name,
                        "spec": spec.to_dict(),
                        "storage_id": self.locations[spec.run_id],
                        "metric": descriptors[name],
                        "labels": labels,
                        "scientific_identity": pipeline._scientific_identity(spec),
                    },
                )
                self.metric_nodes[spec.run_id].append(node_id)
                grouped[name, unit].append(node_id)
        aggregate_nodes: dict[str, list[str]] = defaultdict(list)
        for (name, unit), dependencies in sorted(grouped.items()):
            node_id = f"aggregate:{name}:{fingerprint(unit)[:20]}"
            self.nodes[node_id] = _Node(
                node_id,
                "aggregate",
                f"{name} / {unit}",
                tuple(self.nodes[node].run_ids[0] for node in dependencies),
                tuple(dependencies),
                {
                    "group_by": self.group_by,
                    "uncertainty": self.uncertainty,
                    "points": self.points,
                },
            )
            aggregate_nodes[name].append(node_id)
            if not figures_only:
                self._targets.append(node_id)
        if not figures:
            return
        from .figures import _figure_implementation

        declarations = config.analysis.get("figures", [])
        if not isinstance(declarations, list):
            raise ValueError("analysis.figures must be a list")
        names: set[str] = set()
        for index, declaration in enumerate(declarations):
            if not isinstance(declaration, dict):
                raise ValueError("Each figure must be a mapping")
            metric = declaration.get("metric")
            if metric not in aggregate_nodes:
                raise ValueError(f"Figure refers to unknown metric {metric!r}")
            name = safe_name(declaration.get("name", f"{metric}-{index + 1}"), "Figure name")
            if name in names:
                raise ValueError(f"Duplicate figure name {name!r}")
            names.add(name)
            implementation, _ = _figure_implementation(declaration)
            dependencies = tuple(aggregate_nodes[metric])
            node_id = f"figure:{name}"
            self.nodes[node_id] = _Node(
                node_id,
                "figure",
                name,
                tuple(run_id for dep in dependencies for run_id in self.nodes[dep].run_ids),
                dependencies,
                {"figure": declaration, "name": name, "implementation": implementation},
            )
            self._targets.append(node_id)

    def refresh(self, run_id: str | None = None) -> None:
        """Read newly committed result references and resolve requested dependencies.

        Supply ``run_id`` after one simulation finishes to avoid re-reading every
        retained completion. Calls without it discover all current references.
        """
        from ..storage.trajectories import result_reference

        selected = self.specs if run_id is None else (run_id,)
        for selected_id in selected:
            self.references[selected_id] = result_reference(
                self.output_dir, self.locations[selected_id], self.specs[selected_id]
            )
        if run_id is None:
            for node in self.nodes.values():
                self._update_identity(node)
            for target in self._targets:
                self._need(target, "request", True)
            # A repeated full refresh also observes changed existing references.
            for node_id in reversed(self.nodes):
                self._reconcile(node_id)
        else:
            affected = set(self.metric_nodes[run_id])
            pending = affected.copy()
            while pending:
                node_id = min(pending, key=self._order.__getitem__)
                pending.remove(node_id)
                if self._update_identity(self.nodes[node_id]):
                    pending.update(self._dependents[node_id])
                    affected.update(self._dependents[node_id])
            # Resolve from requested outputs towards their inputs, so a retained
            # descendant can suppress demand before any upstream files are read.
            for node_id in sorted(affected, key=self._order.__getitem__, reverse=True):
                self._reconcile(node_id)

    def _update_identity(self, node: _Node) -> bool:
        """Update identities only when an input changes or the last input is known."""
        if node.kind != "metric" and self._unresolved[node.id]:
            inputs = None
        elif node.kind == "metric":
            reference = self.references.get(node.run_ids[0])
            inputs = (reference["revision"] if reference else None,)
        else:
            inputs = tuple(self.nodes[dep].key for dep in node.dependencies)
        if inputs == node.inputs:
            return False
        node.inputs = inputs
        identity = self._node_identity(node) if inputs is not None else None
        key = fingerprint(identity) if identity is not None else None
        if key == node.key:
            return False
        node.identity, node.key = identity, key
        node.directory = None
        node.checked_key = None
        for parent in self._dependents[node.id]:
            if key is None:
                self._unresolved[parent].add(node.id)
            else:
                self._unresolved[parent].discard(node.id)
        return True

    def _need(self, node_id: str, parent: str, needed: bool) -> None:
        """Count unsatisfied downstream consumers without traversing unrelated groups."""
        consumers = self._needed_by[node_id]
        was_needed = bool(consumers)
        if needed:
            consumers.add(parent)
        else:
            consumers.discard(parent)
        if bool(consumers) != was_needed:
            self._reconcile(node_id)

    def _reconcile(self, node_id: str) -> None:
        """Stop demand at retained outputs and expose runnable missing derivations."""
        node = self.nodes[node_id]
        self._ready.discard(node_id)
        if node.kind == "metric":
            missing = self._missing_raw[node.run_ids[0]]
            missing.discard(node_id)
        needed = bool(self._needed_by[node_id]) and not self._available(node)
        expand = needed and node.kind != "metric"
        if expand != (node_id in self._expanded):
            if expand:
                self._expanded.add(node_id)
            else:
                self._expanded.discard(node_id)
            for dependency in node.dependencies:
                self._need(dependency, node_id, expand)
        if not needed:
            return
        if node.kind == "metric":
            reference = self.references[node.run_ids[0]]
            if reference is None or reference["trajectory"] != "retained":
                missing.add(node_id)
            else:
                self._ready.add(node_id)
        elif all(self._available(self.nodes[dep]) for dep in node.dependencies):
            self._ready.add(node_id)

    def _node_identity(self, node: _Node) -> dict[str, Any] | None:
        if node.kind == "metric":
            reference = self.references.get(node.run_ids[0])
            if reference is None:
                return None
            return self._identity(
                "metric",
                implementation=self._metric_implementation,
                result_revision=reference["revision"],
                spec=node.payload["spec"],
                metric=node.payload["metric"],
            )
        if any(self.nodes[dependency].key is None for dependency in node.dependencies):
            return None
        if node.kind == "aggregate":
            entries = [self._metric_entry(self.nodes[dep]).identity() for dep in node.dependencies]
            return self._identity(
                "aggregate",
                implementation=self._aggregate_implementation,
                options=node.payload,
                metrics=entries,
            )
        return self._identity(
            "plot",
            aggregates=[self.nodes[dep].key for dep in node.dependencies],
            **node.payload,
        )

    def _available(self, node: _Node) -> bool:
        if node.identity is None:
            return False
        if node.checked_key != node.key:
            node.directory = self.cache.find(_COLLECTIONS[node.kind], node.identity)
            node.checked_key = node.key
        return node.directory is not None

    def _metric_entry(self, node: _Node) -> pipeline._CachedMetric:
        spec = self.specs[node.run_ids[0]]
        return pipeline._CachedMetric(
            node.payload["name"],
            node.payload["labels"],
            node.payload["scientific_identity"],
            spec.repetition,
            spec.run_id,
            node.key or "",
            self.cache.root / "cache" / "metrics" / (node.key or ""),
        )

    @property
    def required_simulations(self) -> set[str]:
        """Return raw ancestors that requested derivations cannot currently reach."""
        return {run_id for run_id, metrics in self._missing_raw.items() if metrics}

    @property
    def complete(self) -> bool:
        """Whether every requested output has a validated retained generation."""
        return all(self._available(self.nodes[target]) for target in self._targets)

    @property
    def task_counts(self) -> dict[str, int]:
        """Expose observational totals without making them part of artifact identity."""
        return {
            "total": len(self.nodes),
            "completed": len(self._finished),
            "reused": sum(node.directory is not None for node in self.nodes.values())
            - len(self._finished),
            "pending": sum(
                bool(self._needed_by[node.id]) and not self._available(node)
                for node in self.nodes.values()
            ),
        }

    def analysis_unit(self, run_id: str) -> str:
        """Return the actual repeated-run aggregation group of a simulation."""
        return self.units[run_id]

    def ready_tasks(self) -> list[DerivationTask]:
        """Return runnable derivations, prioritizing raw-consuming metrics."""
        tasks = []
        for node_id in self._ready:
            node = self.nodes[node_id]
            assert node.identity is not None
            tasks.append(
                DerivationTask(
                    node.id,
                    node.kind,
                    node.subject,
                    node.run_ids,
                    str(self.output_dir),
                    node.identity,
                    self._task_payload(node),
                )
            )
        priority = {"metric": 0, "aggregate": 1, "figure": 2}
        return sorted(tasks, key=lambda task: (priority[task.kind], task.id))

    def _task_payload(self, node: _Node) -> dict[str, Any]:
        """Describe a worker's exact persisted inputs independently of its display label."""
        payload = dict(node.payload)
        if node.kind == "aggregate":
            payload["entries"] = [self._metric_entry(self.nodes[dep]) for dep in node.dependencies]
        elif node.kind == "figure":
            payload["directories"] = [str(self.nodes[dep].directory) for dep in node.dependencies]
        return payload

    def finish(self, task: DerivationTask) -> None:
        """Validate a worker's committed publication before releasing dependents."""
        node = self.nodes[task.id]
        node.checked_key = None
        if not self._available(node):
            raise ValueError(f"Task did not publish its expected artifact: {task.id}")
        self._finished.add(task.id)
        self._reconcile(task.id)
        for parent in self._dependents[task.id]:
            self._reconcile(parent)

    def metric_proofs(self, run_id: str) -> dict[str, str] | None:
        """Prove every configured per-run metric is durably retained before pruning."""
        proofs = {}
        for node_id in self.metric_nodes[run_id]:
            node = self.nodes[node_id]
            if node.identity is None:
                return None
            directory = self.cache.find("metrics", node.identity)
            if directory is None:
                return None
            proofs[node.payload["name"]] = str(directory.relative_to(self.output_dir))
        return proofs or None

    def finalize(self) -> tuple[list[pipeline.Summary], list[Path]]:
        """Export available requested summaries and figures from immutable generations."""
        summaries = []
        aggregate_keys = []
        for node in self.nodes.values():
            if node.kind == "aggregate" and node.directory is not None:
                assert node.directory is not None
                summaries.extend(pipeline._read_summaries(node.directory))
                aggregate_keys.append(node.key)
        summaries.sort(key=lambda summary: (summary.metric, canonical_json(summary.labels)))
        bundle_identity = self._identity("tables", aggregates=aggregate_keys)
        if summaries:
            directory = self.cache.publish(
                "tables",
                bundle_identity,
                lambda target: pipeline._write_tables(target, summaries, self.group_by),
            )
            filenames = ["metadata.json"]
            for name in sorted({summary.metric for summary in summaries}):
                filenames.extend((f"{name}.csv", f"{name}.npz"))
            self.cache.export(directory, filenames)
        paths = []
        plot_keys = []
        for node in self.nodes.values():
            if node.kind == "figure" and self._available(node):
                assert node.directory is not None
                filenames = read_json(node.directory / "figures.json")["files"]
                paths.extend(self.cache.export(node.directory, filenames, figures=True))
                plot_keys.append(node.key)
        atomic_json(
            self.cache.current_path,
            {
                "aggregate_key": fingerprint(bundle_identity),
                "aggregate_keys": aggregate_keys,
                "completed_runs": sum(ref is not None for ref in self.references.values()),
                "plot_keys": plot_keys,
            },
        )
        publish_artifact_index(self.output_dir)
        return summaries, paths


def _dependency_implementation() -> str:
    """Version output and dependency transformations independently of progress observations."""
    return fingerprint(
        {name: inspect.getsource(function) for name, function in _DEPENDENCY_BOUNDARIES.items()}
    )


def execute_derivation(task: DerivationTask) -> None:
    """Compute one task using persisted inputs, then atomically publish its outputs."""
    cache = AnalysisCache(task.output_dir)

    def write(target: Path) -> None:
        payload = task.payload
        if task.kind == "metric":
            metric_tasks.write_metric(
                target, Path(task.output_dir), payload, task.identity["result_revision"]
            )
        elif task.kind == "aggregate":
            summaries = [
                pipeline._subsample_summary(summary, payload["points"])
                for summary in pipeline._aggregate_metrics(
                    payload["entries"], payload["uncertainty"]
                )
            ]
            pipeline._write_tables(target, summaries, payload["group_by"])
        elif task.kind == "figure":
            from .figures import _figure_implementation, _prepare_figures, _render_figure

            summaries = [
                summary
                for directory in payload["directories"]
                for summary in pipeline._read_summaries(Path(directory))
            ]
            figure = {**payload["figure"], "name": payload["name"]}
            _prepare_figures([figure], summaries)
            _, plotter = _figure_implementation(figure)
            _render_figure(
                target,
                summaries=summaries,
                figure=figure,
                name=payload["name"],
                plotter=plotter,
            )
        else:
            raise ValueError(f"Unknown derivation task kind {task.kind!r}")

    directory = cache.publish(_COLLECTIONS[task.kind], task.identity, write)
    if task.kind == "figure":
        # A ready figure is useful immediately, even while unrelated runs remain
        # active. The immutable cache remains the authority for this export.
        cache.export(directory, read_json(directory / "figures.json")["files"], figures=True)


# Capture shipped definitions, so wrapping dispatch for observation does not
# change provenance or require the wrapper itself to have inspectable source.
_DEPENDENCY_BOUNDARIES = {
    name: getattr(AnalysisGraph, name)
    for name in ("_identity", "_node_identity", "_metric_entry", "_task_payload", "finalize")
}
_DEPENDENCY_BOUNDARIES["execute_derivation"] = execute_derivation


def analyze_saved(
    config: ExperimentConfig,
    output_dir: str | Path,
    *,
    figures: bool,
) -> tuple[list[pipeline.Summary], list[Path]]:
    """Resolve saved-only derivations, diagnosing missing ancestors without simulating."""
    output_dir = Path(output_dir)
    with ExperimentStore(output_dir).lock():
        return _analyze_saved(config, output_dir, figures=figures)


def _analyze_saved(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    figures: bool,
) -> tuple[list[pipeline.Summary], list[Path]]:
    """Run saved-only tasks while the public entry point holds output ownership."""
    pipeline._validate_analysis_selection(config, output_dir)
    metadata = read_json(output_dir / "metadata.json")
    specs = [RunSpec.from_dict(value) for value in metadata["requests"].values()]
    locations = {
        RunSpec.from_dict(value).run_id: key for key, value in metadata["requests"].items()
    }
    graph = AnalysisGraph(
        config,
        output_dir,
        specs,
        locations,
        figures=figures,
        figures_only=figures and bool(config.analysis.get("figures")),
    )
    if not specs:
        raise ValueError("No validated completed runs are available for analysis")
    while not graph.complete:
        if graph.required_simulations:
            count = len(graph.required_simulations)
            raise ValueError(
                f"Analysis requires {count} unavailable raw trajector{'y' if count == 1 else 'ies'} "
                "(possibly intentionally pruned). Use `ews run` to rematerialize the required "
                "ancestors; analyze and plot never launch simulations."
            )
        ready = graph.ready_tasks()
        if not ready:
            raise ValueError("Analysis dependencies cannot be satisfied from retained artifacts")
        for task in ready:
            execute_derivation(task)
            graph.finish(task)
    return graph.finalize()
