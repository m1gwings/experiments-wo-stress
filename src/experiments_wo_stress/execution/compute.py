"""Best-effort observation around invocation and worker lifecycle boundaries.

Elapsed invocation time includes orchestration; cumulative worker time sums
actual attempts, never configured workers times invocation duration. Process CPU
time counts all threads in the executing process, excluding descendants. GPU
time is elapsed attempt time times the existing exclusive assignment count,
not sampled utilization. None of these observations enter scientific metadata.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from time import monotonic
from typing import Any

from ..storage.files import read_json
from .compute_environment import artifact_size, collect_machine, supported
from .compute_report import publish_attempt, publish_invocation

_LOG = logging.getLogger(__name__)
_CURRENT: ContextVar[Invocation | None] = ContextVar("ews_compute_invocation", default=None)
_INVOCATION_KEY = "_compute_invocation_id"
_COUNTS = ("completed", "skipped", "paused", "failed", "pending")


def _observe(operation: str, function: Callable, *args: Any, **kwargs: Any) -> Any:
    """Contain reporting errors without catching failures of scientific work."""
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        _LOG.warning("Compute reporting could not %s (%s: %s)", operation, type(exc).__name__, exc)
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_times() -> tuple[float, float] | None:
    """Read process user/system CPU seconds; unavailable platforms return None."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime, usage.ru_stime
    except (ImportError, OSError, ValueError):
        return None


def _saved_step(root: Path, storage_id: str) -> int | None:
    """Read only the last committed boundary, never load numerical artifacts."""
    try:
        path = root / "runs" / storage_id / "progress.json"
        return read_json(path).get("step", 0) if path.exists() else 0
    except Exception:
        return None


class Invocation:
    """Own one operational clock, stage durations, and eventual immutable report.

    CLI composition keeps this context open through analysis and plotting. The
    coordinator reuses that context or opens its own for the execution-only API.
    Construction does not create output files: the experiment must first publish
    its scientific request. Finalization errors are logged and never propagated.
    """

    def __init__(
        self, root: str | Path, workers: int, gpu_ids: list[int] | None, *, command: str = "run"
    ) -> None:
        self.root = Path(root).resolve()
        self.enabled = bool(_observe("check platform support", supported))
        self.summary: dict[str, Any] | None = None
        self.record: dict[str, Any] = {}
        self._started = 0.0
        self._token = None
        self._workers = workers
        self._gpu_ids = gpu_ids
        self._command = command

    def __enter__(self) -> Invocation:
        if self.enabled:
            _observe("start invocation observation", self._start)
        self._token = _CURRENT.set(self)
        return self

    def _start(self) -> None:
        self._started = monotonic()
        self.record = {
            "schema_version": 1,
            "invocation_id": uuid.uuid4().hex,
            "command": self._command,
            "started_at": _utc_now(),
            "workers": self._workers,
            "counts": dict.fromkeys(_COUNTS, 0),
            "stages": dict.fromkeys(("execution", "analysis", "plotting")),
            "machine": {},
        }
        self.record["machine"] = (
            _observe("inspect machine", collect_machine, self.root, self._workers, self._gpu_ids)
            or {}
        )

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        _CURRENT.reset(self._token)
        if self.enabled and self.record:
            _observe("publish invocation and summary", self._finish, exc_type is not None)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure an invoked stage, including elapsed work before a stage failure."""
        start = _observe("start stage clock", monotonic) if self.enabled else None
        try:
            yield
        finally:
            if start is not None and self.record:
                finish = _observe("finish stage clock", monotonic)
                if finish is not None:
                    self.record["stages"][name] = max(0.0, finish - start)

    def outcome(self, report: Any, planned: int) -> None:
        """Snapshot terminal counts, accounting for work abandoned by an exception."""
        if not self.record:
            return
        counts = {key: getattr(report, key) for key in _COUNTS}
        counts["pending"] += max(0, planned - sum(counts.values()))
        self.record["counts"] = counts

    def _finish(self, failed: bool) -> None:
        # Do not populate an unowned directory or break first-request validation
        # after a configuration/preflight failure on an otherwise empty root.
        if not (self.root / "metadata.json").is_file():
            return
        counts = self.record["counts"]
        size = _observe("measure output size", artifact_size, self.root)
        self.record.update(
            finished_at=_utc_now(),
            wall_seconds=max(0.0, monotonic() - self._started),
            status=(
                "failed"
                if failed or counts["failed"]
                else "incomplete"
                if counts["paused"] or counts["pending"]
                else "completed"
            ),
            artifact_bytes=size,
        )
        self.summary = publish_invocation(self.root, self.record)

    def display_summary(self, stream: Any) -> dict[str, Any] | None:
        """Best-effort CLI presentation, preserving the scientific command's outcome."""
        return _observe("display summary", self._display_summary, stream)

    def _display_summary(self, stream: Any) -> dict[str, Any] | None:
        if self.summary is None:
            return None
        totals = self.summary["totals"]
        result = {
            "report": str(self.root / "compute" / "summary.md"),
            "invocation_id": self.record["invocation_id"],
            "wall_seconds": self.record["wall_seconds"],
            "observed_worker_hours": totals["worker_hours"],
            "observed_gpu_hours": totals["gpu_hours"],
        }
        machine = self.record["machine"]
        cpu = machine.get("cpu_model") or "CPU model unavailable"
        cores = machine.get("physical_cpu_cores")
        ram = machine.get("ram_bytes")
        details = f"{cpu}" + (f", {cores} physical cores" if cores else "")
        if ram is not None:
            details += f", {ram / 1024**3:.1f} GiB RAM"
        print(
            f"Invocation: {result['wall_seconds']:.1f}s. Observed compute in this output: "
            f"{result['observed_worker_hours']:.3f} worker-hours, "
            f"{result['observed_gpu_hours']:.3f} allocated GPU-hours.\n"
            f"Machine: {details}; workers: {self._workers}; "
            f"allocated GPUs: {len(self._gpu_ids or [])}.\n"
            f"Compute report: {result['report']}",
            file=stream,
        )
        return result


def observe_execution(function: Callable) -> Callable:
    """Wrap the coordinator boundary while leaving scientific scheduling unchanged."""

    @wraps(function)
    def measured(coordinator: Any) -> Any:
        current = _CURRENT.get()
        if current is not None and current.root == coordinator.store.root:
            return _execute_observed(function, coordinator, current)
        with Invocation(
            coordinator.store.root,
            coordinator.workers,
            coordinator.gpu_ids,
            command="run_experiment",
        ) as invocation:
            return _execute_observed(function, coordinator, invocation)

    return measured


def _execute_observed(function: Callable, coordinator: Any, invocation: Invocation) -> Any:
    if invocation.record:
        coordinator.execution[_INVOCATION_KEY] = invocation.record["invocation_id"]
    try:
        with invocation.stage("execution"):
            return function(coordinator)
    finally:
        _observe("retain run counts", invocation.outcome, coordinator.report, len(coordinator.plan))


class Attempt:
    """Observe one worker task through cleanup, persisting separate start/end records.

    Starts survive abrupt worker death with unknown duration. Successful reuse
    gets an explicit skipped endpoint with zero new compute. CPU deltas use the
    same process at both boundaries, including its threads but no child processes
    or prior tasks. In embedded single-worker use, unrelated threads can contribute.
    """

    def __init__(self, spec: dict, root: str, execution: dict, storage_id: str) -> None:
        self.root = Path(root)
        self.start = monotonic()
        self.cpu = _cpu_times()
        # resources.py established this lifetime lease before importing the study.
        # An inherited mask in CPU mode is not an EWS allocation.
        gpu_ids = [int(os.environ["CUDA_VISIBLE_DEVICES"])] if "gpu_ids" in execution else []
        self.record = {
            "schema_version": 1,
            "invocation_id": execution[_INVOCATION_KEY],
            "attempt_id": uuid.uuid4().hex,
            "run_id": spec["run_id"],
            "storage_id": storage_id,
            "group": spec["group"],
            "algorithm": spec["algorithm_name"],
            "budget_steps": spec.get("budget_steps"),
            "started_at": _utc_now(),
            "finished_at": None,
            "status": "unresolved",
            "wall_seconds": None,
            "cpu_user_seconds": None,
            "cpu_system_seconds": None,
            "allocated_gpu_count": len(gpu_ids),
            "gpu_ids": gpu_ids,
            "gpu_seconds": None,
            "start_step": _saved_step(self.root, storage_id),
            "finish_step": None,
        }
        _observe("publish attempt start", publish_attempt, self.root, self.record, started=True)

    def finish(self, status: str) -> None:
        """Publish duration and outcome without rewriting the start or older attempts."""
        duration = max(0.0, monotonic() - self.start)
        cpu = _cpu_times()
        skipped = status == "skipped"
        self.record.update(
            status=status,
            finished_at=_utc_now(),
            wall_seconds=0.0 if skipped else duration,
            cpu_user_seconds=(
                0.0 if skipped else max(0.0, cpu[0] - self.cpu[0]) if cpu and self.cpu else None
            ),
            cpu_system_seconds=(
                0.0 if skipped else max(0.0, cpu[1] - self.cpu[1]) if cpu and self.cpu else None
            ),
            gpu_seconds=0.0 if skipped else duration * self.record["allocated_gpu_count"],
            finish_step=_saved_step(self.root, self.record["storage_id"]),
        )
        publish_attempt(self.root, self.record)


def observe_attempt(function: Callable) -> Callable:
    """Decorate the spawn-safe worker entry point without changing its result tuple."""

    @wraps(function)
    def measured(spec, root, execution, recording, max_steps, storage_id, **kwargs):
        attempt = None
        if _INVOCATION_KEY in execution:
            attempt = _observe(
                "start attempt observation", Attempt, spec, root, execution, storage_id
            )
        status = "failed"
        try:
            result = function(spec, root, execution, recording, max_steps, storage_id, **kwargs)
            status = result[1]
            return result
        finally:
            if attempt is not None:
                _observe("publish attempt outcome", attempt.finish, status)

    return measured
