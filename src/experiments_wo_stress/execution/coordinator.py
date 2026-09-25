"""One execution request: preparation, bounded scheduling, and interruption."""

from __future__ import annotations

import multiprocessing
import signal
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..storage.experiment import ExperimentStore
from ..study.config import ExperimentConfig, load_config
from ..study.planning import plan_runs
from ..study.specs import RunSpec
from .notifications import ExperimentNotifier
from .provenance import collect_provenance, preflight, prepare_request
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

    def to_dict(self) -> dict[str, Any]:
        """Return plain counts and errors for CLI output and coordinator summaries."""
        return asdict(self)

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
        self.execution = dict(config.execution)
        self.store = ExperimentStore(Path(output_dir).resolve())
        self.report = RunReport()
        self.context = multiprocessing.get_context("spawn")
        self.stop_event = self.context.Event() if workers > 1 else threading.Event()
        self.plan: list[RunSpec] = []
        self.locations: dict[str, str] = {}
        self.notifier: ExperimentNotifier | None = None

    def run(self) -> RunReport:
        """Prepare a durable request, execute it, and close invocation resources."""
        source_dir = str(self.config.source_dir)
        if source_dir not in sys.path:
            sys.path.insert(0, source_dir)
        self.plan = plan_runs(self.config)
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
                aborted = False
            finally:
                self.notifier.finish(self.report.to_dict(), aborted=aborted)
        return self.report

    def _publish_request(self, provenance: dict[str, Any]) -> None:
        self.store.validate_execution_root()
        prepared = prepare_request(self.config, self.plan, provenance)
        for storage_id, metadata in prepared.variants.items():
            self.store.publish_variant(storage_id, metadata)
        self.store.publish_request(
            prepared.request_id,
            prepared.metadata,
            yaml.safe_dump(self.config.to_dict(), sort_keys=False),
        )
        self.locations = prepared.locations

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
        self.notifier.run_started(
            spec.run_id, self.store.runs_path / self.locations[spec.run_id], spec.budget_steps
        )

    def _run_finished(self, result: tuple[str, str, str | None]) -> None:
        self.report.add(result)
        self.notifier.run_finished(result)

    def _run_sequentially(self) -> None:
        for index, spec in enumerate(self.plan):
            if self.stop_event.is_set():
                self.report.pending = len(self.plan) - index
                break
            self._run_started(spec)
            result = execute_run(*self._arguments(spec), stop_event=self.stop_event)
            self._run_finished(result)

    def _run_parallel(self) -> None:
        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=self.context,
            initializer=initialize_worker,
            initargs=(self.stop_event,),
        ) as pool:
            remaining = iter(self.plan)
            active: dict[Future, str] = {}

            def submit_next() -> None:
                spec = next(remaining, None)
                if spec is None:
                    return
                future = pool.submit(execute_run, *self._arguments(spec))
                active[future] = spec.run_id
                self._run_started(spec)

            # Keep only one submitted task per worker so an interruption leaves
            # unscheduled work pending instead of filling the process-pool queue.
            for _ in range(min(self.workers, len(self.plan))):
                submit_next()
            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    run_id = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = (run_id, "failed", f"Worker failure: {exc}")
                        self.stop_event.set()
                    self._run_finished(result)
                    if not self.stop_event.is_set():
                        submit_next()
            self.report.pending = sum(1 for _ in remaining)


def run_experiment(
    config: ExperimentConfig | str | Path,
    output_dir: str | Path,
    *,
    workers: int | None = None,
    max_steps: int | None = None,
) -> RunReport:
    """Run or resume an experiment. ``max_steps`` pauses each run after new steps."""
    if not isinstance(config, ExperimentConfig):
        config = load_config(config)
    return ExecutionCoordinator(config, output_dir, workers=workers, max_steps=max_steps).run()
