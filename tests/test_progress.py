"""Ephemeral progress, attempt-aware estimates, bounded worker rows, and terminal fallback."""

from __future__ import annotations

import io
import unittest
from dataclasses import replace
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from experiments_wo_stress.builtins.protocols import OfflineProtocol, TrialProtocol
from experiments_wo_stress.execution.progress import (
    ProgressReporter,
    TerminalProgress,
    WorkerProgress,
)
from tests.sample_results import make_saved_result


class WorkerProgressTests(unittest.TestCase):
    """Measure only completed steps, without assigning progress to indivisible work."""

    def test_progress_eta_and_restored_prefix(self):
        row = WorkerProgress(1, "run", 0, 0, "running", 0, 100, 0, 0)
        self.assertEqual(row.fraction, 0)
        self.assertIsNone(row.eta(0))
        row.completed = 50
        self.assertIsNone(row.eta(1))
        self.assertEqual(row.eta(10), 10)
        row.initial, row.completed, row.total = 800, 900, 1000
        self.assertEqual(row.fraction, 0.9)
        self.assertEqual(row.eta(10), 10)
        row.total = None
        self.assertIsNone(row.fraction)
        self.assertIsNone(row.eta(10))
        row.status, row.updated = "completed", 12
        self.assertEqual(row.fraction, 1)
        self.assertEqual(row.eta(20), 0)
        self.assertEqual(row.elapsed(20), 12)

    def test_messages_are_time_throttled_but_completion_is_always_sent(self):
        queue = Queue()
        protocol = SimpleNamespace(step=0, total_steps=100)
        with patch(
            "experiments_wo_stress.execution.progress.time.monotonic", return_value=0
        ) as clock:
            reporter = ProgressReporter(queue, "run", None)
            reporter.update(protocol, force=True)
            for step in range(1, 100):
                protocol.step = step
                reporter.update(protocol)
            self.assertEqual(queue.qsize(), 2)  # starting, restored/initialized boundary
            clock.return_value = 0.25
            reporter.update(protocol)
            self.assertEqual(queue.qsize(), 3)
            protocol.step = 100
            reporter.update(protocol, force=True)
            reporter.finish("completed")
        events = list(queue.queue)
        self.assertEqual(events[-1].status, "completed")
        self.assertEqual((events[-1].completed, events[-1].fraction), (100, 1))
        self.assertEqual(events[1].completed, 0)  # snapshots cannot mutate after enqueue

    def test_unknown_and_atomic_protocols_and_a_full_queue(self):
        for protocol, total in (
            (SimpleNamespace(step=0), None),
            (OfflineProtocol(rng=None), 1),
            (object.__new__(TrialProtocol), 1),
        ):
            with self.subTest(protocol=type(protocol).__name__):
                protocol.step = 0
                queue = Queue(maxsize=2)
                reporter = ProgressReporter(queue, "run", None)
                reporter.update(protocol, force=True)
                self.assertEqual(reporter.state.total, total)
                reporter.finish("completed")  # a full monitoring queue never blocks execution
                self.assertEqual(queue.qsize(), 2)


class TerminalProgressTests(unittest.TestCase):
    """Inspect state and readable text, leaving ANSI redraw mechanics to Rich."""

    def setUp(self):
        self.stream = io.StringIO()
        self.console = Console(file=self.stream, width=100, force_terminal=False)
        self.display = TerminalProgress("study", 2, quiet=True, console=self.console)
        self.display.queue = Queue()  # synchronous transport for deterministic unit checks
        self.display.started = 0
        self.spec = make_saved_result().spec

    def send(self, spec, pid, completed=0, total=100, started=0):
        self.display.run_started(spec)
        self.display.queue.put(
            WorkerProgress(pid, spec.run_id, started, 10, "running", completed, total, 0, started)
        )
        self.display._drain()

    def test_counters_worker_reuse_and_global_eta_include_queued_work(self):
        display = self.display
        display.configure(10)
        self.send(self.spec, 10)
        second = replace(self.spec, run_id="second", repetition=1)
        self.send(second, 11, completed=50)
        self.assertEqual((len(display.active), display.queued), (2, 8))
        self.assertIsNone(display.remaining(0))
        with patch("experiments_wo_stress.execution.progress.time.monotonic", return_value=10):
            display.run_finished((self.spec.run_id, "completed", None))
        self.assertEqual(display.counts["completed"], 1)
        self.assertEqual(display.remaining(10), 42.5)  # (8 queued + half a run) / 2 workers
        self.assertEqual(display.rows[10].fraction, 1)
        third = replace(self.spec, run_id="third", repetition=2)
        self.send(third, 10, started=10)
        self.assertEqual(len(display.rows), 2)
        self.assertEqual(display.rows[10].run_id, "third")
        self.assertEqual((len(display.active), display.queued), (2, 7))
        display.run_finished(("third", "failed", "failure\nwith detail"))
        self.assertEqual((display.counts["failed"], display.rows[10].status), (1, "failed"))
        self.assertIn("Failed third: failure with detail", self.stream.getvalue())
        display.run_finished(("second", "skipped", None))
        self.assertEqual(len(display.durations), 1)  # failure and reuse never train throughput
        self.assertEqual((len(display.active), display.queued), (0, 7))
        self.assertEqual(display.remaining(10), 35)

    def test_late_messages_cannot_resurrect_finished_work(self):
        self.display.configure(1)
        self.send(self.spec, 10)
        self.display.run_finished((self.spec.run_id, "completed", None))
        self.display.queue.put(WorkerProgress(10, self.spec.run_id, 0, 1))
        self.display._drain()
        self.assertEqual(self.display.rows[10].status, "completed")
        self.assertEqual(self.display.remaining(20), 0)

    def test_plain_updates_are_throttled_and_contain_no_escape_sequences(self):
        self.display.configure(1)
        self.send(self.spec, 10)
        for now in (0, 1, 2, 29, 30):
            self.display._refresh(now)
        output = self.stream.getvalue()
        self.assertEqual(output.count("[ews]"), 2)
        self.assertIn("0/1 completed | 1 running | 0 queued", output)
        self.assertNotIn("\x1b", output)

    def test_long_names_fit_narrow_terminals_and_are_literal(self):
        spec = replace(self.spec, group="group" * 100, algorithm_name="[red]" * 100)
        self.display.configure(1)
        self.send(spec, 10)
        for width in (40, 60, 80, 120):
            with self.subTest(width=width):
                stream = io.StringIO()
                self.display.console = Console(file=stream, width=width, force_terminal=False)
                self.display.console.print(self.display.render(10))
                self.assertLessEqual(max(map(len, stream.getvalue().splitlines())), width)
                self.assertEqual(self.display.rows[10].run_id, spec.run_id)

    def test_quiet_and_exception_exit_still_print_final_summary(self):
        for aborted in (False, True):
            with self.subTest(aborted=aborted):
                self.stream.seek(0)
                self.stream.truncate()
                try:
                    with TerminalProgress("study", 1, quiet=True, console=self.console) as display:
                        display.configure(1)
                        if aborted:
                            raise ValueError("setup failed")
                        display.run_finished(("run", "completed", None))
                        display.finish(
                            {"completed": 1, "skipped": 0, "failed": 0, "paused": 0, "pending": 0}
                        )
                except ValueError:
                    pass
                text = self.stream.getvalue()
                self.assertEqual(text.count("[ews]"), 1)
                self.assertIn("Aborted" if aborted else "Completed", text)
                self.assertNotIn("\x1b", text)
