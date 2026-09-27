"""Dependency readiness and fair priority for a bounded local worker pool."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.experiment import load_instance
from ..storage.files import StorageError, read_json
from ..storage.trajectories import prune_trajectory, pruning_receipt, rematerialize_trajectory
from ..study.specs import RunSpec


@dataclass(frozen=True)
class ScheduledTask:
    """Describe worker work without holding scientific component instances."""

    id: str
    kind: str
    subject: str
    run_ids: tuple[str, ...]
    payload: Any


class TaskScheduler:
    """Own pending/running task state and close dependency groups without starvation.

    Derived work has priority, with a simulation admitted after eight consecutive
    derivations when any remain. Simulations prefer groups with completed or active
    members; stable plan order breaks ties. Aging derived kinds lets ready
    aggregates and figures close promptly even when raw metrics keep arriving.
    Readiness comes only from durable artifacts, never worker progress observations.
    """

    def __init__(
        self, config: Any, root: Path, specs: list[RunSpec], locations: dict, *, graph: Any = None
    ) -> None:
        self.config = config
        self.root = root
        self.specs = {spec.run_id: spec for spec in specs}
        self.locations = locations
        self.pending = dict(self.specs)
        self.active: set[str] = set()
        self.finished: set[str] = set()
        self.failed_tasks: set[str] = set()
        self.group_progress: Counter = Counter()
        self.derived_streak = 0
        self._validated_instances: set[str] = set()
        self.graph = graph
        self.waiting_kinds: Counter = Counter()
        if self.graph is None and (
            config.analysis.get("metrics") or config.analysis.get("figures")
        ):
            from ..analysis.graph import AnalysisGraph

            self.graph = AnalysisGraph(config, root, specs, locations)
        self.units = {
            spec.run_id: self.graph.analysis_unit(spec.run_id) if self.graph else spec.run_id
            for spec in specs
        }
        if self.graph:
            self.group_progress.update(
                self.units[run_id]
                for run_id, reference in self.graph.references.items()
                if reference
            )

    def next_task(self) -> ScheduledTask | None:
        """Take ready work; bounded submission prevents queued tasks hiding readiness."""
        ready = self.graph.ready_tasks() if self.graph else []
        ready = [task for task in ready if task.id not in self.active | self.failed_tasks]
        if ready and (self.derived_streak < 8 or not self.pending):
            priority = {"metric": 0, "aggregate": 1, "figure": 2}
            kinds = {item.kind for item in ready}
            self.waiting_kinds.update(kinds)
            aged = [kind for kind in kinds if self.waiting_kinds[kind] >= 4]
            selected_kind = (
                max(aged, key=lambda kind: (self.waiting_kinds[kind], -priority[kind]))
                if aged
                else min(kinds, key=priority.__getitem__)
            )
            task = min(
                (item for item in ready if item.kind == selected_kind), key=lambda item: item.id
            )
            self.waiting_kinds[selected_kind] = 0
            self.active.add(task.id)
            self.derived_streak += 1
            return ScheduledTask(task.id, task.kind, task.subject, tuple(task.run_ids), task)
        if not self.pending:
            return None
        run_id = max(self.pending, key=lambda key: self.group_progress[self.units[key]])
        spec = self.pending.pop(run_id)
        self.active.add(run_id)
        self.group_progress[self.units[run_id]] += 1
        self.derived_streak = 0
        return ScheduledTask(
            run_id, "simulation", f"{spec.algorithm_name} r{spec.repetition}", (run_id,), spec
        )

    def needs_rematerialization(self, spec: RunSpec) -> bool:
        """A pruned trajectory is needed only if no retained derivation satisfies work."""
        if self.config.recording.get("retention", "keep") == "keep" or not self.graph:
            return True
        return spec.run_id in self.graph.required_simulations

    def prepare_simulation(self, spec: RunSpec) -> bool:
        """Return false for reusable pruned work; otherwise prepare fresh raw ancestors."""
        directory = self.root / "runs" / self.locations[spec.run_id]
        active_pruning = (directory / "trajectory.json").exists()
        reference = self.graph.references.get(spec.run_id) if self.graph else None
        historical_pruning = reference is not None and reference["trajectory"] == "pruned"
        if active_pruning or historical_pruning:
            if active_pruning:
                pruning_receipt(directory)
            instance_id = read_json(directory / "metadata.json").get("instance_id")
            if not instance_id:
                raise StorageError("Completed run is missing its scientific instance reference")
            if instance_id not in self._validated_instances:
                load_instance(self.root, instance_id)
                self._validated_instances.add(instance_id)
            if not self.needs_rematerialization(spec):
                return False
            if active_pruning:
                rematerialize_trajectory(directory)
        return True

    def finish(self, task: ScheduledTask, status: str) -> None:
        """Observe an authoritative worker outcome and release newly ready dependencies."""
        self.active.discard(task.id)
        if status not in {"completed", "skipped"}:
            self.failed_tasks.add(task.id)
            return
        self.finished.add(task.id)
        if self.graph:
            if task.kind == "simulation":
                self.graph.refresh(task.run_ids[0])
            else:
                self.graph.finish(task.payload)
            if self.config.recording.get("retention") == "until_analyzed" and task.kind in {
                "simulation",
                "metric",
            }:
                for run_id in task.run_ids:
                    proofs = self.graph.metric_proofs(run_id)
                    if proofs:
                        prune_trajectory(
                            self.root, self.locations[run_id], self.specs[run_id], proofs
                        )
                # All current metric dependencies are already satisfied. Pruning
                # changes availability for future requests, never this revision.


def execute_derivation_task(
    task: ScheduledTask, execution: dict, root: str, progress_queue: Any = None
) -> None:
    """Execute a derived task in a worker, with independent operational observation."""
    from ..analysis.graph import execute_derivation
    from . import worker
    from .compute import observe_derivation
    from .progress import ProgressReporter

    queue = worker._WORKER_PROGRESS_QUEUE if progress_queue is None else progress_queue
    reporter = (
        ProgressReporter(
            queue,
            task.id,
            None,
            kind={"metric": "METRIC", "aggregate": "AGG", "figure": "FIGURE"}[task.kind],
            subject=task.subject,
        )
        if queue is not None
        else None
    )
    if reporter:
        reporter.advance(0, force=True)
    status = "failed"
    try:
        with observe_derivation(task, execution, root):
            execute_derivation(task.payload)
        status = "completed"
    finally:
        if reporter:
            reporter.finish(status)
