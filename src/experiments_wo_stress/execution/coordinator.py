"""One execution request: preparation, bounded scheduling, and interruption."""

from __future__ import annotations

import multiprocessing
import signal
import sys
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..storage.experiment import ExperimentStore
from ..study.config import ExperimentConfig, load_config, validate_gpu_workers
from ..study.planning import plan_runs
from ..study.specs import RunSpec
from .compute import observe_execution
from .notifications import ExperimentNotifier
from .progress import TerminalProgress
from .provenance import PreparedRequest, collect_provenance, preflight, prepare_request
from .resources import GPUWorker, gpu_workers, validate_gpu_environment, wait_gpu_workers
from .scheduler import ScheduledTask, TaskScheduler, execute_derivation_task
from .worker import execute_run, initialize_worker


@dataclass
class RunReport:
    """Summarize the outcome of one invocation, including per-run failure messages.

    Workers return completed, skipped, paused, or failed outcomes. The coordinator
    counts unsubmitted work as pending after interruption; skipped means a valid
    saved result already covered the requested budget.
    """

    completed: int = 0
    skipped: int = 0
    paused: int = 0
    failed: int = 0
    pending: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    task_failures: int = 0
    pending_tasks: int = 0
    interrupted: bool = False
    groups: int | None = None
    figures: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return plain counts and errors for CLI output and coordinator summaries."""
        result = asdict(self)
        for key in ("groups", "figures"):
            if result[key] is None:
                result.pop(key)
        for key in ("task_failures", "pending_tasks", "interrupted"):
            if not result[key]:
                result.pop(key)
        return result

    def add(self, result: tuple[str, str, str | None]) -> None:
        """Accumulate one worker's run ID, terminal status, and optional error."""
        run_id, status, error = result
        setattr(self, status, getattr(self, status) + 1)
        if error:
            self.errors[run_id] = error


class ExecutionCoordinator:
    """Own scheduling, signals, and notifications for one experiment invocation.

    Only plain run descriptions and execution options cross the process boundary.
    Scientific components and their mutable state are owned by worker RunSessions.

    ``run`` plans and validates the request, acquires the experiment lock, selects
    retained variants, and schedules workers. A shared stop event lets active runs
    checkpoint at step boundaries while leaving unscheduled work pending. The
    coordinator combines worker outcomes into a RunReport and closes notifications.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str | Path,
        *,
        workers: int | None = None,
        max_steps: int | None = None,
        progress: TerminalProgress | None = None,
    ) -> None:
        if max_steps is not None and (
            isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0
        ):
            raise ValueError("max_steps must be a nonnegative integer")
        workers = config.execution.get("workers", 1) if workers is None else workers
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.config = config
        self.workers = workers
        self.max_steps = max_steps
        self.progress = progress
        self.execution = dict(config.execution)
        self.gpu_ids = validate_gpu_workers(self.execution, workers)
        if self.gpu_ids is not None:
            validate_gpu_environment()
        self.store = ExperimentStore(Path(output_dir).resolve())
        self.report = RunReport()
        self.context = multiprocessing.get_context("spawn")
        self.stop_event = (
            self.context.Event() if workers > 1 or self.gpu_ids is not None else threading.Event()
        )
        self.plan: list[RunSpec] = []
        self.locations: dict[str, str] = {}
        self.variants: dict[str, dict] = {}
        self.scheduler: TaskScheduler | None = None
        self.notifier: ExperimentNotifier | None = None

    @observe_execution
    def run(self) -> RunReport:
        """Prepare a durable request, execute it, and close invocation resources."""
        source_dir = str(self.config.source_dir)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        if self.gpu_ids is not None:
            return self._run_with_gpus()
        self.plan = plan_runs(self.config)
        if self.progress:
            self.progress.configure(len(self.plan))
        self.notifier = ExperimentNotifier(
            self.config.notifications, self.config.name, len(self.plan)
        )
        # Check component contracts before publishing a durable execution request.
        provenance = collect_provenance(self.config, preflight(self.plan))
        self.store.root.mkdir(parents=True, exist_ok=True)
        with self._signal_handlers(), self.store.lock():
            self._publish_request(provenance)
            self.notifier.start()
            aborted = True
            try:
                if self.workers == 1:
                    self._run_sequentially()
                else:
                    self._run_parallel()
                self._finalize_pipeline()
                aborted = False
            finally:
                self.notifier.finish(self.report.to_dict(), aborted=aborted)
        return self.report

    def _publish_request(self, provenance: dict[str, Any]) -> None:
        self.store.validate_execution_root()
        prepared = prepare_request(self.config, self.plan, provenance)
        self._publish_prepared_request(prepared)

    def _publish_prepared_request(self, prepared: PreparedRequest, graph: Any = None) -> None:
        """Publish a request and install its optionally worker-prepared analysis graph."""
        self.variants = prepared.variants
        self.store.publish_request(
            prepared.request_id,
            prepared.metadata,
            yaml.safe_dump(self.config.to_dict(), sort_keys=False),
        )
        self.locations = prepared.locations
        self.scheduler = TaskScheduler(
            self.config, self.store.root, self.plan, self.locations, graph=graph
        )
        if self.progress and self.scheduler.graph:
            totals = Counter(self._task_label(node) for node in self.scheduler.graph.nodes.values())
            self.progress.configure_tasks(dict(totals))

    def _run_with_gpus(self) -> RunReport:
        """Prepare and run science only in processes with lifetime GPU assignments."""
        with (
            self._signal_handlers(),
            gpu_workers(
                self.gpu_ids,
                self.context,
                self.stop_event,
                self.progress.queue if self.progress else None,
            ) as workers,
        ):
            # Planning, preflight, and variant selection can all import study code.
            # Do them in an assigned worker before constructing any run components.
            self.plan, provenance, prepared = workers[0].prepare(self.config)
            if self.progress:
                self.progress.configure(len(self.plan))
            provenance["execution"] = {
                "workers": self.workers,
                "gpu_ids": list(self.gpu_ids),
                "worker_assignments": [
                    {"pid": worker.pid, "gpu_id": worker.gpu_id} for worker in workers
                ],
            }
            # This operational record is excluded from prepared scientific identities.
            prepared.metadata["provenance"] = provenance
            self.notifier = ExperimentNotifier(
                self.config.notifications, self.config.name, len(self.plan)
            )
            self.store.root.mkdir(parents=True, exist_ok=True)
            with self.store.lock():
                self.store.validate_execution_root()
                # Metric and plotter constructors may import GPU frameworks too.
                # Only their plain dependency descriptions cross back to us.
                graph = workers[0].prepare_analysis(
                    self.config, self.store.root, self.plan, prepared.locations
                )
                self._publish_prepared_request(prepared, graph)
                self.notifier.start()
                aborted = True
                try:
                    self._schedule_gpu_runs(workers)
                    self._finalize_pipeline()
                    aborted = False
                finally:
                    self.notifier.finish(self.report.to_dict(), aborted=aborted)
        return self.report

    def _schedule_gpu_runs(self, workers: list[GPUWorker]) -> None:
        """Share fixed GPU worker lifetimes across simulation and derived tasks."""
        active: dict[GPUWorker, ScheduledTask] = {}

        def submit_next(worker: GPUWorker) -> None:
            task = self._next_task()
            if task is None:
                return
            try:
                if task.kind == "simulation":
                    worker.submit(self._arguments(task.payload))
                else:
                    worker.submit_task(task, self.execution, str(self.store.root))
            except Exception as exc:
                self._task_finished(task, error=exc)
                self.stop_event.set()
            else:
                active[worker] = task

        for worker in workers:
            submit_next(worker)
        while active:
            for worker in wait_gpu_workers(list(active)):
                task = active.pop(worker)
                try:
                    result = worker.result()
                except Exception as exc:
                    self._task_finished(task, error=exc)
                    self.stop_event.set()
                else:
                    self._task_finished(task, result)
            for worker in workers:
                if worker not in active:
                    submit_next(worker)
        self.report.pending = len(self.scheduler.pending)

    @contextmanager
    def _signal_handlers(self) -> Iterator[None]:
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: self.stop_event.set())
        try:
            yield
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)

    def _arguments(self, spec: RunSpec) -> tuple:
        return (
            spec.to_dict(),
            str(self.store.root),
            self.execution,
            dict(self.config.recording),
            self.max_steps,
            self.locations[spec.run_id],
        )

    def _run_started(self, spec: RunSpec) -> None:
        if self.progress:
            self.progress.run_started(spec)
        self.notifier.run_started(
            spec.run_id, self.store.runs_path / self.locations[spec.run_id], spec.budget_steps
        )

    def _run_finished(self, result: tuple[str, str, str | None]) -> None:
        self.report.add(result)
        self.notifier.run_finished(result)
        if self.progress:
            self.progress.run_finished(result)

    def _next_task(self) -> ScheduledTask | None:
        """Select work and publish a variant only immediately before it is used."""
        while not self.stop_event.is_set():
            task = self.scheduler.next_task()
            if task is None:
                return None
            if task.kind == "simulation":
                spec = task.payload
                storage_id = self.locations[spec.run_id]
                self.store.publish_variant(storage_id, self.variants[storage_id])
                self._run_started(spec)
                try:
                    execute = self.scheduler.prepare_simulation(spec)
                except Exception as exc:
                    self._task_finished(task, error=exc)
                    continue
                if not execute:
                    self._task_finished(task, (spec.run_id, "skipped", None))
                    continue
            elif self.progress:
                self.progress.task_started(task.id, self._task_label(task), task.subject)
            return task
        return None

    @staticmethod
    def _task_label(task: ScheduledTask) -> str:
        return {"metric": "METRIC", "aggregate": "AGG", "figure": "FIGURE"}[task.kind]

    def _task_finished(
        self, task: ScheduledTask, result: Any = None, *, error: Exception | None = None
    ) -> None:
        """Release dependencies only after workers return from durable publication."""
        if task.kind == "simulation":
            if error is not None:
                result = (task.id, "failed", f"Worker failure: {error}")
            self._run_finished(result)
            status = result[1]
        else:
            status = "failed" if error else "completed"
            if error:
                self.report.task_failures += 1
                self.report.errors[task.id] = f"{type(error).__name__}: {error}"
            if self.progress:
                self.progress.task_finished(task.id, status, str(error) if error else None)
        self.scheduler.finish(task, status)

    def _finalize_pipeline(self) -> None:
        self.report.interrupted = self.stop_event.is_set()
        if self.scheduler.graph:
            self.report.pending_tasks = self.scheduler.graph.task_counts["pending"]
        if (
            self.scheduler.graph
            and self.scheduler.graph.complete
            and not (
                self.report.failed
                or self.report.paused
                or self.report.pending
                or self.report.task_failures
            )
        ):
            summaries, figures = self.scheduler.graph.finalize()
            if self.progress:
                reused = Counter(
                    self._task_label(node)
                    for node in self.scheduler.graph.nodes.values()
                    if node.id not in self.scheduler.finished
                )
                self.progress.reuse_tasks(dict(reused))
            if self.config.analysis.get("figures"):
                self.report.figures = [str(path) for path in figures]
            else:
                self.report.groups = len(summaries)

    def _run_sequentially(self) -> None:
        while (task := self._next_task()) is not None:
            try:
                if task.kind == "simulation":
                    result = execute_run(
                        *self._arguments(task.payload),
                        stop_event=self.stop_event,
                        progress_queue=self.progress.queue if self.progress else None,
                    )
                else:
                    execute_derivation_task(
                        task,
                        self.execution,
                        str(self.store.root),
                        self.progress.queue if self.progress else None,
                    )
                    result = None
            except Exception as exc:
                self._task_finished(task, error=exc)
            else:
                self._task_finished(task, result)
        self.report.pending = len(self.scheduler.pending)

    def _run_parallel(self) -> None:
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=self.context,
            initializer=initialize_worker,
            initargs=(self.stop_event, self.progress.queue if self.progress else None),
        ) as pool:
            active: dict[Future, ScheduledTask] = {}
            while True:
                while len(active) < self.workers:
                    task = self._next_task()
                    if task is None:
                        break
                    if task.kind == "simulation":
                        future = pool.submit(execute_run, *self._arguments(task.payload))
                    else:
                        future = pool.submit(
                            execute_derivation_task, task, self.execution, str(self.store.root)
                        )
                    active[future] = task
                if not active:
                    break
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    task = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._task_finished(task, error=exc)
                    else:
                        self._task_finished(task, result)
            self.report.pending = len(self.scheduler.pending)


def run_experiment(
    config: ExperimentConfig | str | Path,
    output_dir: str | Path,
    *,
    workers: int | None = None,
    max_steps: int | None = None,
    progress: TerminalProgress | None = None,
) -> RunReport:
    """Execute the requested dependency graph, reusing retained valid artifacts.

    ``max_steps`` pauses each run after new steps. Python calls are silent by
    default; the CLI supplies a parent-owned terminal observer.
    """
    if not isinstance(config, ExperimentConfig):
        config = load_config(config)
    return ExecutionCoordinator(
        config, output_dir, workers=workers, max_steps=max_steps, progress=progress
    ).run()
