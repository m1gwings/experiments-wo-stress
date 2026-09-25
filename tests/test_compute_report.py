"""Compute history is immutable and totals distinguish observed work from unknown history.

These scenarios publish deterministic records without inspecting local hardware,
running scientific components, or depending on elapsed test execution time.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from experiments_wo_stress.execution.compute_report import (
    publish_attempt,
    publish_invocation,
    publish_record,
    regenerate_summary,
)
from experiments_wo_stress.storage.files import atomic_json, fingerprint, read_json


@unittest.skipUnless(os.name == "posix", "compute reports are POSIX-only")
class ComputeHistoryTests(unittest.TestCase):
    """Retain attempts and invocations independently from disposable aggregate reports."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def attempt(
        self,
        identifier: str,
        *,
        invocation: str = "invocation1",
        status: str = "completed",
        wall: float | None = 10,
        storage_id: str = "variant1",
        run_id: str = "science1",
        gpu_ids: tuple[int, ...] = (),
        group: str = "main",
    ) -> dict:
        """Describe one independently clocked attempt or an unresolved start."""
        return {
            "schema_version": 1,
            "attempt_id": identifier,
            "invocation_id": invocation,
            "run_id": run_id,
            "storage_id": storage_id,
            "group": group,
            "algorithm": "learner",
            "started_at": "2026-01-01T00:00:00+00:00",
            "finished_at": "2026-01-01T00:00:10+00:00" if wall is not None else None,
            "status": status,
            "wall_seconds": wall,
            "cpu_user_seconds": wall / 2 if wall is not None else None,
            "cpu_system_seconds": wall / 10 if wall is not None else None,
            "allocated_gpu_count": len(gpu_ids),
            "gpu_ids": list(gpu_ids),
            "gpu_seconds": wall * len(gpu_ids) if wall is not None else None,
        }

    def invocation(self, identifier: str = "invocation1", *, day: int = 1) -> dict:
        """Describe finalized wall time independently of concurrent worker time."""
        return {
            "schema_version": 1,
            "invocation_id": identifier,
            "command": "run",
            "started_at": f"2026-01-{day:02d}T00:00:00+00:00",
            "finished_at": f"2026-01-{day:02d}T00:00:20+00:00",
            "wall_seconds": 20,
            "stages": {"execution": 15, "analysis": 3, "plotting": None},
            "workers": 4,
            "machine": {
                "platform": "Linux",
                "architecture": "x86_64",
                "cpu_model": "Example CPU",
                "physical_cpu_cores": 16,
                "logical_cpu_count": 32,
                "ram_bytes": 64 * 1024**3,
                "workers": 4,
                "allocated_gpu_count": 0,
                "gpu_ids": [],
                "gpus": [],
                "filesystem_total_bytes": 100 * 1024**3,
                "filesystem_available_bytes": 80 * 1024**3,
            },
            "counts": {"completed": 1, "skipped": 0, "paused": 0, "failed": 0, "pending": 0},
            "status": "completed",
            "artifact_bytes": 1024**3,
        }

    def retained_completion(self, storage_id: str, run_id: str, *, active: bool = True) -> None:
        """Inventory a stored completion without creating any numerical artifacts."""
        directory = self.root / "runs" / storage_id
        atomic_json(directory / "progress.json", {"status": "completed", "step": 5})
        atomic_json(directory / "metadata.json", {"spec": {"run_id": run_id}})
        if active:
            path = self.root / "metadata.json"
            metadata = read_json(path) if path.exists() else {"run_ids": []}
            metadata["run_ids"].append(storage_id)
            atomic_json(path, metadata)

    def test_publication_is_checksummed_and_never_overwrites_history(self) -> None:
        record = self.invocation()
        path = self.root / "compute" / "invocations" / "invocation1.json"
        publish_record(path, record)
        original = path.read_bytes()
        stored = read_json(path)
        self.assertEqual(stored.pop("sha256"), fingerprint(record))
        self.assertEqual(stored, record)
        with self.assertRaises(FileExistsError):
            publish_record(path, {**record, "wall_seconds": 999})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(path.parent.glob(".*.pending")), [])

    def test_resume_and_reuse_preserve_attempts_without_counting_compute_twice(self) -> None:
        paused = self.attempt("attempt1", status="paused", wall=4)
        resumed = self.attempt("attempt2", invocation="invocation2", wall=6)
        skipped = self.attempt("attempt3", invocation="invocation3", status="skipped", wall=8)
        self.retained_completion("variant1", "science1")
        for record in (paused, resumed, skipped):
            publish_attempt(
                self.root, {**record, "status": "unresolved", "wall_seconds": None}, started=True
            )
            publish_attempt(self.root, record)
        for number in (1, 2, 3):
            summary = publish_invocation(
                self.root, self.invocation(f"invocation{number}", day=number)
            )
        self.assertEqual(len(list((self.root / "compute" / "attempts").glob("*.json"))), 6)
        totals = summary["totals"]
        self.assertEqual(totals["attempt_count"], 2)
        self.assertEqual(totals["skipped_count"], 1)
        self.assertEqual(totals["worker_seconds"], 10)
        self.assertEqual(totals["completed_worker_seconds"], 6)
        self.assertEqual(totals["paused_worker_seconds"], 4)
        self.assertEqual(summary["coverage"]["completed_scientific_runs_with_timing"], 1)
        self.assertEqual(summary["invocations"][2]["attempt_totals"]["worker_seconds"], 0)
        self.assertEqual(summary["attempt_timing"]["median_seconds"], 5)

    def test_multiple_invocations_sum_wall_worker_cpu_and_allocated_gpu_time_separately(
        self,
    ) -> None:
        records = [
            self.attempt("attempt1", wall=20, gpu_ids=(2, 5)),
            self.attempt("attempt2", wall=12, gpu_ids=(3,)),
            self.attempt("attempt3", invocation="invocation2", status="failed", wall=8),
        ]
        for record in records:
            publish_attempt(self.root, record)
        publish_invocation(self.root, self.invocation())
        summary = publish_invocation(self.root, self.invocation("invocation2", day=2))
        totals = summary["totals"]
        self.assertEqual(totals["invocation_wall_seconds"], 40)
        self.assertEqual(totals["execution_wall_seconds"], 30)
        self.assertEqual(totals["analysis_wall_seconds"], 6)
        self.assertEqual(totals["plotting_wall_seconds"], 0)
        self.assertEqual(totals["worker_seconds"], 40)
        self.assertEqual(totals["gpu_seconds"], 52)
        self.assertEqual(totals["gpu_hours"], 52 / 3600)
        self.assertEqual(totals["worker_hours"], 40 / 3600)
        self.assertEqual(totals["cpu_user_seconds"], 20)
        self.assertEqual(totals["cpu_system_seconds"], 4)
        self.assertEqual(totals["failed_paused_worker_seconds"], 8)
        self.assertEqual(summary["invocations"][0]["attempt_totals"]["worker_seconds"], 32)

    def test_cpu_only_attempts_have_zero_gpu_usage_and_partial_cpu_coverage_is_explicit(
        self,
    ) -> None:
        first = self.attempt("first")
        second = self.attempt("second")
        second["cpu_user_seconds"] = second["cpu_system_seconds"] = None
        publish_attempt(self.root, first)
        publish_attempt(self.root, second)
        summary = publish_invocation(self.root, self.invocation())
        self.assertEqual(summary["totals"]["gpu_seconds"], 0)
        self.assertEqual(summary["totals"]["cpu_user_seconds"], 5)
        self.assertEqual(summary["totals"]["cpu_user_timed_attempts"], 1)
        self.assertEqual(summary["totals"]["timed_attempt_count"], 2)

    def test_unresolved_start_survives_a_missing_invocation_without_inventing_time(self) -> None:
        start = self.attempt("killed", wall=None, status="unresolved", gpu_ids=(0,))
        publish_attempt(self.root, start, started=True)
        summary = regenerate_summary(self.root)
        totals = summary["totals"]
        self.assertEqual(totals["attempt_count"], 1)
        self.assertEqual(totals["unresolved_attempt_count"], 1)
        self.assertEqual(totals["attempts_without_wall_timing"], 1)
        self.assertEqual(totals["gpu_attempts_without_wall_timing"], 1)
        self.assertEqual(totals["invocations_without_final_record"], 1)
        self.assertEqual(totals["worker_seconds"], 0)
        self.assertIsNone(totals["cpu_user_seconds"])
        self.assertIsNone(summary["latest_invocation"])
        self.assertFalse(summary["invocations"][0]["invocation_record_available"])
        self.assertIsNone(summary["invocations"][0]["wall_seconds"])

    def test_old_completed_variants_without_history_are_unknown(self) -> None:
        self.retained_completion("old", "science_old")
        self.retained_completion("timed", "science_timed")
        self.retained_completion("historical", "science_other", active=False)
        publish_attempt(self.root, self.attempt("new", storage_id="timed", run_id="science_timed"))
        publish_attempt(
            self.root,
            self.attempt("older", storage_id="historical", run_id="science_other", wall=5),
        )
        summary = publish_invocation(self.root, self.invocation())
        coverage = summary["coverage"]
        self.assertEqual(coverage["retained_completed_variants"], 3)
        self.assertEqual(coverage["retained_completed_variants_without_timing"], 1)
        self.assertEqual(coverage["retained_completed_scientific_runs_without_timing"], 1)
        self.assertEqual(coverage["variants_without_timing"], ["old"])
        self.assertEqual(coverage["active_completed_attempt_worker_seconds"], 10)
        self.assertEqual(coverage["other_completed_attempt_worker_seconds"], 5)
        self.assertEqual(summary["totals"]["worker_seconds"], 15)

    def test_corrupt_finish_is_excluded_and_start_remains_unresolved(self) -> None:
        record = self.attempt("attempt1")
        publish_attempt(self.root, {**record, "wall_seconds": None}, started=True)
        publish_attempt(self.root, record)
        path = self.root / "compute" / "attempts" / "attempt1.json"
        damaged = read_json(path)
        damaged["wall_seconds"] = 1000000
        atomic_json(path, damaged)
        with self.assertLogs("experiments_wo_stress.execution.compute_report", level="WARNING"):
            summary = publish_invocation(self.root, self.invocation())
        self.assertEqual(summary["totals"]["worker_seconds"], 0)
        self.assertEqual(summary["totals"]["unresolved_attempt_count"], 1)
        self.assertEqual(len(summary["warnings"]), 1)
        self.assertIn("checksum", summary["warnings"][0])

    def test_legacy_resume_and_unresolved_attempts_expose_partial_completion_history(self) -> None:
        """A timed completion cannot supply timing for an older saved prefix or lost attempt."""
        self.retained_completion("legacy", "science_legacy")
        legacy_tail = self.attempt("legacy_tail", storage_id="legacy", run_id="science_legacy")
        publish_attempt(self.root, {**legacy_tail, "start_step": 5, "finish_step": 10})

        self.retained_completion("recovered", "science_recovered")
        killed = self.attempt(
            "killed",
            storage_id="recovered",
            run_id="science_recovered",
            status="unresolved",
            wall=None,
        )
        publish_attempt(self.root, {**killed, "start_step": 0}, started=True)
        recovered = self.attempt("recovered", storage_id="recovered", run_id="science_recovered")
        publish_attempt(
            self.root,
            {**recovered, "started_at": "2026-01-02T00:00:00+00:00", "start_step": 5},
        )
        summary = publish_invocation(self.root, self.invocation())
        coverage = summary["coverage"]
        self.assertEqual(coverage["retained_completed_variants_without_timing"], 0)
        self.assertEqual(coverage["completed_scientific_runs_with_timing"], 2)
        self.assertEqual(coverage["retained_completed_variants_with_partial_timing_evidence"], 2)
        self.assertEqual(coverage["variants_with_partial_timing_evidence"], ["legacy", "recovered"])
        self.assertEqual(coverage["variants_with_unrecorded_prefix"], ["legacy"])
        self.assertEqual(coverage["variants_with_unresolved_attempts"], ["recovered"])
        markdown = (self.root / "compute" / "summary.md").read_text()
        self.assertIn("known missing/partial timing evidence: 2", markdown)
        self.assertIn("does not prove complete timing history", markdown)

    def test_report_failure_leaves_invocation_and_scientific_files_intact(self) -> None:
        self.retained_completion("old", "science_old")
        scientific = {path: path.read_bytes() for path in (self.root / "runs").rglob("*.json")}
        with patch(
            "experiments_wo_stress.execution.compute_report.atomic_text",
            side_effect=OSError("report disk error"),
        ):
            with self.assertRaisesRegex(OSError, "report disk error"):
                publish_invocation(self.root, self.invocation())
        for path, content in scientific.items():
            self.assertEqual(path.read_bytes(), content)
        invocation_path = self.root / "compute" / "invocations" / "invocation1.json"
        original = invocation_path.read_bytes()
        summary = regenerate_summary(self.root)
        self.assertEqual(invocation_path.read_bytes(), original)
        self.assertEqual(summary["totals"]["invocation_count"], 1)
        self.assertTrue((self.root / "compute" / "summary.md").exists())

    def test_markdown_bounds_condition_rows_and_explains_semantics_without_hostnames(self) -> None:
        for index in range(25):
            publish_attempt(
                self.root, self.attempt(f"attempt{index}", group=f"condition_{index:02d}")
            )
        record = self.invocation()
        record["machine"]["hostname"] = "private-hostname"
        record["machine"]["apple_accelerator"] = [{"model": "Apple M4", "cores": 10}]
        summary = publish_invocation(self.root, record)
        markdown = (self.root / "compute" / "summary.md").read_text()
        self.assertEqual(len(summary["timing_by_group"]), 25)
        self.assertEqual(markdown.count("| condition_"), 20)
        self.assertIn("Apple M4", markdown)
        self.assertIn("64 GiB", markdown)
        self.assertIn("worker-hours", markdown)
        self.assertIn("not GPU utilization", markdown)
        self.assertIn("not the cost of the entire research project", markdown)
        self.assertIn("not whole-run durations", markdown)
        self.assertNotIn("private-hostname", markdown)
        self.assertEqual(json.loads((self.root / "compute" / "summary.json").read_text()), summary)

    def test_invalid_record_filename_is_rejected_before_writing(self) -> None:
        with self.assertRaisesRegex(ValueError, "identifier"):
            publish_attempt(self.root, self.attempt("../outside"))
        self.assertFalse((self.root / "compute").exists())

    def test_concurrent_invocations_preserve_all_history_in_the_final_summary(self) -> None:
        records = [self.invocation(f"concurrent{number}", day=number) for number in range(1, 5)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda record: publish_invocation(self.root, record), records))
        summary = read_json(self.root / "compute" / "summary.json")
        self.assertEqual(summary["totals"]["invocation_count"], 4)
        self.assertEqual(summary["totals"]["invocation_wall_seconds"], 80)
        self.assertEqual(
            {item["invocation_id"] for item in summary["invocations"]},
            {record["invocation_id"] for record in records},
        )

    def test_corrupt_metadata_warning_omits_the_output_directory_path(self) -> None:
        path = self.root / "compute" / "invocations" / "corrupt.json"
        path.parent.mkdir(parents=True)
        path.write_text("{invalid json")
        with self.assertLogs("experiments_wo_stress.execution.compute_report", level="WARNING"):
            summary = regenerate_summary(self.root)
        self.assertEqual(len(summary["warnings"]), 1)
        self.assertNotIn(str(self.root), summary["warnings"][0])
        self.assertIn("<output>", summary["warnings"][0])
