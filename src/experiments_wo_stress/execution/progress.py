"""Ephemeral worker progress and a parent-owned terminal display, separate from science."""

from __future__ import annotations

import multiprocessing
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, tzinfo
from datetime import timezone as datetime_timezone
from queue import Empty, Full
from statistics import median
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..study.config import resolve_display_timezone

if TYPE_CHECKING:
    from ..study.specs import RunSpec

_INTERVAL = 0.25
_ACTIVE = {"starting", "running"}
_DERIVED_KINDS = ("METRIC", "AGG", "FIGURE")
_STYLES = {
    "starting": "yellow",
    "running": "cyan",
    "completed": "green",
    "skipped": "green",
    "failed": "red",
    "paused": "yellow",
}


@dataclass
class WorkerProgress:
    """Snapshot one process's current task using the shared monotonic clock.

    The initial completed count is taken after restoration, so ETA measures only
    this attempt's work. Terminal snapshots freeze elapsed time. The parent keeps
    one snapshot per process, replacing it when that worker starts another task.
    """

    pid: int
    run_id: str
    started: float
    updated: float
    status: str = "starting"
    completed: int = 0
    total: int | None = None
    initial: int | None = None
    baseline_at: float = 0.0
    kind: str = "SIM"
    subject: str = ""

    @property
    def fraction(self) -> float | None:
        """Return requested-prefix progress, or None when total work is unknown."""
        if self.status in {"completed", "skipped"}:
            return 1.0
        if self.total is None or self.total <= 0:
            return None
        return min(1.0, max(0.0, self.completed / self.total))

    @property
    def attempt_fraction(self) -> float:
        """Measure this invocation's fraction, excluding any restored prefix."""
        if self.total is None or self.initial is None or self.total <= self.initial:
            return 0.0
        return min(1.0, max(0.0, (self.completed - self.initial) / (self.total - self.initial)))

    def elapsed(self, now: float) -> float:
        """Return wall time since this worker began the task, including setup."""
        return max(0.0, (now if self.status in _ACTIVE else self.updated) - self.started)

    def eta(self, now: float) -> float | None:
        """Estimate remaining work after at least two seconds and one percent of new work."""
        if self.status in {"completed", "skipped"}:
            return 0.0
        fraction = self.attempt_fraction
        elapsed = now - self.baseline_at
        if self.status != "running" or fraction < 0.01 or elapsed < 2:
            return None
        return elapsed * (1 - fraction) / fraction


class ProgressReporter:
    """Send small, throttled snapshots from a worker; never render or block on the UI."""

    def __init__(
        self,
        queue: Any,
        run_id: str,
        budget: int | None,
        *,
        kind: str = "SIM",
        subject: str = "",
    ) -> None:
        self.queue = queue
        now = time.monotonic()
        self.state = WorkerProgress(
            os.getpid(), run_id, now, now, total=budget, kind=kind, subject=subject
        )
        self.next_update = now
        self._send()

    def _send(self) -> None:
        if self.queue is not None:
            try:
                self.queue.put_nowait(replace(self.state))
            except (Full, OSError, ValueError):
                # Monitoring must not backpressure execution. The authoritative
                # worker result separately guarantees the parent's final status.
                pass

    def update(self, protocol: Any, *, force: bool = False) -> None:
        """Observe a complete step, resolving total work only when sending an update."""
        now = time.monotonic()
        if not force and now < self.next_update:
            return
        self.next_update = now + _INTERVAL
        self.state.completed = protocol.step
        if self.state.initial is None:
            self.state.initial = protocol.step
            self.state.baseline_at = now
            if self.state.total is None:
                from ..builtins.protocols import OfflineProtocol, TrialProtocol

                # Built-in fits/trials are one indivisible operation. Custom
                # protocols may expose a total in the existing protocol-step unit.
                try:
                    total = getattr(protocol, "total_steps", None)
                except Exception:
                    total = None
                if total is None and isinstance(protocol, (OfflineProtocol, TrialProtocol)):
                    total = 1
                if isinstance(total, int) and not isinstance(total, bool) and total > 0:
                    self.state.total = total
        self.state.status = "running"
        self.state.updated = now
        self._send()

    def finish(self, status: str) -> None:
        """Always send a terminal snapshot, even inside the throttle interval."""
        self.state.status = status
        self.state.updated = time.monotonic()
        self._send()

    def advance(self, completed: int, total: int | None = None, *, force: bool = False) -> None:
        """Observe meaningful task units; atomic derivations may remain indeterminate."""
        now = time.monotonic()
        if not force and now < self.next_update:
            return
        self.next_update = now + _INTERVAL
        self.state.completed = completed
        if total is not None:
            self.state.total = total
        if self.state.initial is None:
            self.state.initial = 0
            self.state.baseline_at = self.state.started
        self.state.status = "running"
        self.state.updated = now
        self._send()


def _label(value: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value).strip()


def _duration(seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def _estimate(seconds: float | None) -> str:
    if seconds is None:
        return "estimating…"
    if 0 < seconds < 5:
        return "<5s"
    if seconds < 60:
        return f"~{max(5, round(seconds / 5) * 5)}s" if seconds else "0s"
    minutes = round(seconds / 60)
    return f"~{minutes}m" if minutes < 60 else f"~{minutes // 60}h {minutes % 60}m"


class TerminalProgress:
    """Observe scheduling and worker snapshots in the parent process only.

    A bounded queue carries expendable progress, while coordinator callbacks own
    authoritative counters and failures. One display thread drains that queue and
    refreshes Rich at 4 Hz (plain logs every 30 seconds). The context owns its
    queue, thread, and live display; exit restores the terminal and prints a final
    summary. No monitoring state is persisted or used to schedule scientific work.
    """

    def __init__(
        self,
        name: str,
        workers: int,
        *,
        quiet: bool = False,
        console: Console | None = None,
        timezone: str | tzinfo = "UTC",
    ) -> None:
        self.name = _label(name)
        self.workers = workers
        self.quiet = quiet
        self.timezone = (
            timezone
            if isinstance(timezone, tzinfo)
            else resolve_display_timezone({}, override=timezone)
        )
        self.console = console or Console(stderr=True, highlight=False)
        self.interactive = bool(self.console.file.isatty() and not self.console.is_dumb_terminal)
        if not self.interactive:
            self.console = Console(
                file=self.console.file,
                width=self.console.width,
                force_terminal=False,
                color_system=None,
                highlight=False,
            )
        self.queue = (
            None if quiet else multiprocessing.get_context("spawn").Queue(max(16, workers * 8))
        )
        self.total = 0
        self.active: dict[str, RunSpec] = {}
        self.tasks: dict[str, tuple[str, str]] = {}
        self.task_totals = dict.fromkeys(_DERIVED_KINDS, 0)
        self.task_counts = {
            kind: dict.fromkeys(("completed", "skipped", "failed", "paused"), 0)
            for kind in _DERIVED_KINDS
        }
        self.rows: dict[int, WorkerProgress] = {}
        self.names: dict[int, str] = {}
        self.counts = dict.fromkeys(("completed", "skipped", "failed", "paused"), 0)
        self.errors: dict[str, str] = {}
        self.durations: deque[float] = deque(maxlen=32)
        self.started = time.monotonic()
        self.next_log = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._report: dict[str, Any] | None = None

    def __enter__(self) -> TerminalProgress:
        if not self.quiet:
            if self.interactive:
                self._live = Live(
                    self.render(time.monotonic()),
                    console=self.console,
                    auto_refresh=False,
                    redirect_stdout=False,
                )
                self._live.start(refresh=True)
            self._thread = threading.Thread(target=self._loop, name="ews-terminal", daemon=True)
            self._thread.start()
        return self

    def configure(self, total: int) -> None:
        """Receive the prepared plan size without importing or planning study code."""
        with self._lock:
            self.total = total

    def run_started(self, spec: RunSpec) -> None:
        """Register submitted work; the worker's first snapshot identifies its process."""
        with self._lock:
            self.active[spec.run_id] = spec

    def configure_tasks(self, totals: dict[str, int]) -> None:
        """Observe the derived-task plan without interpreting its dependencies."""
        with self._lock:
            for kind, count in totals.items():
                if kind not in _DERIVED_KINDS or count < 0:
                    raise ValueError(f"Invalid progress task total: {kind}={count}")
                self.task_totals[kind] = count

    def reuse_tasks(self, counts: dict[str, int]) -> None:
        """Count tasks satisfied by retained artifacts without fabricating worker activity."""
        with self._lock:
            for kind, count in counts.items():
                if kind not in _DERIVED_KINDS or count < 0:
                    raise ValueError(f"Invalid reused task count: {kind}={count}")
                self.task_counts[kind]["skipped"] += count

    def task_started(self, task_id: str, kind: str, subject: str) -> None:
        """Register a derived task before its worker emits optional snapshots."""
        if kind not in _DERIVED_KINDS:
            raise ValueError(f"Unknown progress task kind: {kind}")
        with self._lock:
            self.tasks[task_id] = (kind, _label(subject))

    def task_finished(
        self, task_id: str, status: str = "completed", error: str | None = None
    ) -> None:
        """Record an authoritative derived outcome independently of simulation counts."""
        with self._lock:
            self._drain()
            kind, subject = self.tasks.pop(task_id)
            self.task_counts[kind][status] += 1
            now = time.monotonic()
            for row in self.rows.values():
                if row.run_id == task_id and row.kind == kind:
                    row.status, row.updated = status, now
            if error:
                self.errors[task_id] = error
                self.console.print(
                    Text(f"[ews] Failed {kind} {subject}: {_label(error)}", style="red")
                )
                if self._live:
                    self._live.update(self.render(now), refresh=True)

    def _drain(self) -> None:
        if self.queue is None:
            return
        for _ in range(max(16, self.workers * 8)):
            try:
                row = self.queue.get_nowait()
            except (Empty, OSError, ValueError):
                break
            task = self.tasks.get(row.run_id)
            if task is not None:
                self.rows[row.pid] = row
                row.kind, self.names[row.pid] = task
                continue
            spec = self.active.get(row.run_id)
            if spec is None:
                continue  # A delayed snapshot must not resurrect finished work.
            self.rows[row.pid] = row
            self.names[row.pid] = (
                f"{_label(spec.algorithm_name)[:12]} r{spec.repetition:03} "
                f"{spec.run_id[:6]} /{_label(spec.group)[:24]}"
            )

    def run_finished(self, result: tuple[str, str, str | None]) -> None:
        """Apply an authoritative result and immediately surface a concise failure."""
        run_id, status, error = result
        with self._lock:
            self._drain()
            self.active.pop(run_id, None)
            self.counts[status] += 1
            now = time.monotonic()
            for row in self.rows.values():
                if row.run_id == run_id:
                    row.status, row.updated = status, now
                    if status == "completed":
                        self.durations.append(row.elapsed(now))
            if error:
                self.errors[run_id] = error
                # Text treats study-controlled strings literally, never as markup.
                self.console.print(Text(f"[ews] Failed {run_id}: {_label(error)}", style="red"))
                if self._live:
                    self._live.update(self.render(now), refresh=True)

    @property
    def queued(self) -> int:
        """Count unsubmitted runs, excluding every terminal outcome and active task."""
        return max(0, self.total - sum(self.counts.values()) - len(self.active))

    def _derived_pending(self) -> bool:
        return bool(self.tasks) or any(
            self.task_totals[kind] > sum(self.task_counts[kind].values()) for kind in _DERIVED_KINDS
        )

    def _simulation_remaining(self, now: float) -> float | None:
        """Estimate the simulation backlog without treating analysis as simulation work."""
        if not self.active and not self.queued:
            return 0.0 if self.total else None
        if not self.durations or now - self.started < 5:
            return None
        # Reuse and failures do not train throughput. Account for queued jobs and
        # the unfinished part of each active attempt, including restored prefixes.
        fractions = {
            row.run_id: row.attempt_fraction
            for row in self.rows.values()
            if row.run_id in self.active
        }
        work = self.queued + sum(1 - fractions.get(run_id, 0) for run_id in self.active)
        concurrency = min(self.workers, self.queued + len(self.active))
        return median(self.durations) * work / max(1, concurrency)

    def remaining(self, now: float) -> float | None:
        """Estimate full-study time only when no unmeasured derived work remains."""
        if self._derived_pending():
            return None
        return self._simulation_remaining(now)

    def _estimate_text(self, now: float, *, finish: bool) -> str:
        if self._derived_pending() and (self.active or self.queued):
            remaining = self._simulation_remaining(now)
            label = "SIM remaining"
            suffix = " | analysis pending"
            finish_label = "SIM finish"
        else:
            remaining = self.remaining(now)
            label = "Remaining" if finish else "remaining"
            suffix = ""
            finish_label = "Finish"
        estimate = f"{label} {_estimate(remaining)}"
        if finish:
            estimate += f" | {finish_label} {self._finish_estimate(remaining)}"
        return estimate + suffix

    def _summary(self, now: float) -> str:
        if not self.total:
            return f"preparing study | elapsed {_duration(now - self.started)}"
        done = self.counts["completed"] + self.counts["skipped"]
        summary = (
            f"{done}/{self.total} completed | {len(self.active) + len(self.tasks)} running | "
            f"{self.queued} queued | {self.counts['failed']} failed | "
            f"elapsed {_duration(now - self.started)} | {self._estimate_text(now, finish=False)}"
        )
        return summary + self._task_summary()

    def _task_summary(self) -> str:
        return "".join(
            f" | {kind} {self.task_counts[kind]['completed'] + self.task_counts[kind]['skipped']}"
            f"/{total}"
            + (
                f" ({self.task_counts[kind]['failed']} failed)"
                if self.task_counts[kind]["failed"]
                else ""
            )
            for kind, total in self.task_totals.items()
            if total
        )

    def _finish_estimate(self, remaining: float | None) -> str:
        if remaining is None:
            return "estimating…"
        finish = datetime.now(datetime_timezone.utc) + timedelta(seconds=remaining)
        return finish.astimezone(self.timezone).strftime("~%H:%M %Z")

    def render(self, now: float) -> Group:
        """Build a width-aware view of worker rows and study counters."""
        heading = Panel(Text(self.name), title="Experiments W/O Stress", border_style="blue")
        if not self.total:
            return Group(heading, Text("Preparing study…", style="dim"))
        narrow = self.console.width < 60
        table = Table(expand=True, box=None, padding=(0, 0 if narrow else 1))
        columns = [
            ("W" if narrow else "Worker", 1 if narrow else 6),
            ("Task", 6),
            ("Subject", None),
            ("%" if narrow else "Progress", 16 if self.console.width >= 90 else 4 if narrow else 8),
            ("Time" if narrow else "Elapsed", 5 if narrow else 8),
            ("Status", 9),
        ]
        if not narrow:
            columns.insert(-1, ("ETA", 7))
        for title, width in columns:
            table.add_column(
                title,
                width=width,
                no_wrap=True,
                overflow="ellipsis",
                ratio=1 if title == "Subject" else None,
            )
        for index, (pid, row) in enumerate(self.rows.items()):
            fraction = row.fraction
            percent = "—" if fraction is None else f"{fraction:.0%}"
            progress: Any = Text(percent, style=_STYLES[row.status])
            if self.console.width >= 90:
                progress = Table.grid(padding=(0, 1))
                progress.add_row(
                    ProgressBar(
                        total=100,
                        completed=100 * (fraction or 0),
                        pulse=fraction is None,
                        width=10,
                        complete_style=_STYLES[row.status],
                        finished_style=_STYLES[row.status],
                    ),
                    Text(percent),
                )
            cells = [
                str(index),
                row.kind,
                Text(row.run_id[:8] if narrow else self.names[pid]),
                progress,
                _duration(row.elapsed(now)),
                Text(row.status, style=_STYLES[row.status]),
            ]
            if not narrow:
                cells.insert(-1, "—" if row.eta(now) is None else _estimate(row.eta(now)))
            table.add_row(*cells)
        return Group(
            heading,
            table,
            Text(
                f"Runs  {self.counts['completed'] + self.counts['skipped']}/{self.total} completed | "
                f"{len(self.active) + len(self.tasks)} running | {self.queued} queued | "
                f"{self.counts['failed']} failed" + self._task_summary()
            ),
            Text(
                f"Elapsed {_duration(now - self.started)} | "
                f"{self._estimate_text(now, finish=True)}",
                style="dim",
            ),
        )

    def _refresh(self, now: float) -> None:
        if self._live:
            self._live.update(self.render(now), refresh=True)
        elif now >= self.next_log:
            self.console.print(Text(f"[ews] {self.name}: {self._summary(now)}"), soft_wrap=True)
            self.next_log = now + 30

    def _loop(self) -> None:
        while not self._stop.wait(_INTERVAL):
            with self._lock:
                self._drain()
                self._refresh(time.monotonic())

    def finish(self, report: dict[str, Any]) -> None:
        """Retain final authoritative counts, including work left pending on interruption."""
        self._report = report

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        try:
            if self._thread:
                self._thread.join()
            with self._lock:
                self._drain()
                if self._report:
                    self.counts.update({key: self._report[key] for key in self.counts})
                if self._live:
                    self._live.update(self.render(time.monotonic()))
        finally:
            if self._live:
                self._live.stop()
            if self.queue is not None:
                self.queue.close()
                self.queue.cancel_join_thread()
        report = self._report or {}
        pending = report.get("pending", self.queued + len(self.active))
        pending_tasks = report.get("pending_tasks", 0)
        status = (
            "Interrupted"
            if exc_type is KeyboardInterrupt
            else "Aborted"
            if exc_type
            else "Failed"
            if self.counts["failed"] or any(c["failed"] for c in self.task_counts.values())
            else (
                "Paused/interrupted"
                if pending or pending_tasks or self.counts["paused"] or report.get("interrupted")
                else "Completed"
            )
        )
        text = (
            f"[ews] {status}: {self.counts['completed']} completed, "
            f"{self.counts['skipped']} reused, {self.counts['failed']} failed, "
            f"{self.counts['paused']} paused, {pending} pending | "
            f"elapsed {_duration(time.monotonic() - self.started)}" + self._task_summary()
        )
        if pending_tasks:
            text += f" | {pending_tasks} derived tasks pending"
        self.console.print(
            Text(text, style="green" if status == "Completed" else "yellow"), soft_wrap=True
        )
        for run_id, error in self.errors.items():
            self.console.print(
                Text(f"[ews] Failed {run_id}: {_label(error)}", style="red"), soft_wrap=True
            )
