"""Publish operational compute history and summarize only the records retained here.

Invocation and attempt records are immutable, checksummed JSON. The two summaries
are disposable views of that history, independent of numerical results and their
identities. A start record without a finish is an unresolved attempt: neither its
duration nor the missing invocation's end time can be reconstructed honestly.
"""

from __future__ import annotations

import logging
import math
import os
import re
import statistics
import uuid
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..storage.files import (
    StorageError,
    atomic_json,
    atomic_text,
    fingerprint,
    read_json,
    sync_directory,
)

_LOGGER = logging.getLogger(__name__)
_STATUSES = {"completed", "paused", "failed", "skipped", "unresolved"}
_SCOPE = (
    "This report covers only compute records retained in this output directory. "
    "It is not the cost of the entire research project: other directories, deleted "
    "compute records, experiments outside EWS, and unrecorded work on other machines "
    "are unknown. Researchers remain responsible for disclosing that additional compute."
)


def publish_record(path: Path, record: Mapping[str, Any]) -> None:
    """Atomically create a checksummed record, refusing to replace existing history.

    UUID-named worker records need no shared lock. A hard link publishes the fully
    synced temporary file only if the destination does not yet exist; unlike a
    check followed by replacement, concurrent writers cannot overwrite a record.
    """
    path = Path(path)
    payload = {key: value for key, value in record.items() if key != "sha256"}
    sealed = {**payload, "sha256": fingerprint(payload)}
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        atomic_json(temporary, sealed)
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_attempt(root: Path, record: Mapping[str, Any], *, started: bool = False) -> None:
    """Retain an attempt start or finish without replacing either immutable record."""
    identifier = _identifier(record["attempt_id"])
    suffix = ".start.json" if started else ".json"
    publish_record(Path(root) / "compute" / "attempts" / f"{identifier}{suffix}", record)


def publish_invocation(root: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Publish a finished invocation and refresh summaries under their own lock.

    This lock is independent of execution ownership because CLI analysis and
    figure export finish after the execution coordinator releases its root lock.
    Reporting errors propagate to the best-effort caller, never to run artifacts.
    """
    root = Path(root)
    identifier = _identifier(record["invocation_id"])
    with _summary_lock(root):
        publish_record(root / "compute" / "invocations" / f"{identifier}.json", record)
        return _regenerate_summary(root)


def regenerate_summary(root: Path) -> dict[str, Any]:
    """Rebuild aggregate JSON and Markdown from immutable records and run metadata."""
    root = Path(root)
    with _summary_lock(root):
        return _regenerate_summary(root)


@contextmanager
def _summary_lock(root: Path) -> Iterator[None]:
    """Serialize summary writers on supported POSIX systems without locking workers."""
    import fcntl

    directory = root / "compute"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Invalid compute record identifier")
    return value


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _read_record(path: Path, *, attempt: bool) -> dict[str, Any]:
    record = read_json(path)
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        raise StorageError("Unsupported compute record schema")
    payload = {key: value for key, value in record.items() if key != "sha256"}
    if fingerprint(payload) != record.get("sha256"):
        raise StorageError("Compute record checksum mismatch")
    identifier = "attempt_id" if attempt else "invocation_id"
    expected = path.name.removesuffix(".json").removesuffix(".start")
    if _identifier(record.get(identifier)) != expected:
        raise StorageError("Compute record identifier does not match its filename")
    _identifier(record.get("invocation_id"))
    for key in ("wall_seconds", "cpu_user_seconds", "cpu_system_seconds", "gpu_seconds"):
        if record.get(key) is not None and not _number(record[key]):
            raise StorageError(f"Invalid compute duration: {key}")
    if attempt:
        if record.get("status") not in _STATUSES:
            raise StorageError("Invalid compute attempt status")
        for key in ("run_id", "storage_id"):
            _identifier(record.get(key))
        allocated = record.get("allocated_gpu_count")
        if isinstance(allocated, bool) or not isinstance(allocated, int) or allocated < 0:
            raise StorageError("Invalid allocated GPU count")
        for key in ("group", "algorithm"):
            if record.get(key) is not None and not isinstance(record[key], str):
                raise StorageError(f"Invalid attempt label: {key}")
        for key in ("start_step", "finish_step"):
            step = record.get(key)
            if step is not None and (
                isinstance(step, bool) or not isinstance(step, int) or step < 0
            ):
                raise StorageError(f"Invalid attempt boundary: {key}")
    else:
        for key in ("stages", "counts", "machine"):
            if not isinstance(record.get(key), dict):
                raise StorageError(f"Invalid invocation field: {key}")
        for value in record["stages"].values():
            if value is not None and not _number(value):
                raise StorageError("Invalid invocation stage duration")
    return payload


def _records(directory: Path, *, attempt: bool, warnings: list[str]) -> list[dict[str, Any]]:
    records = {}
    for path in sorted(directory.glob("*.json")):
        try:
            record = _read_record(path, attempt=attempt)
        except (StorageError, OSError, ValueError, TypeError, KeyError) as exc:
            detail = str(exc).replace(str(directory.parent.parent), "<output>")
            message = f"Excluded damaged compute record {directory.name}/{path.name}: {detail}"
            warnings.append(message)
            _LOGGER.warning(message)
            continue
        key = record["attempt_id" if attempt else "invocation_id"]
        # The start survives forever for crash accounting, but a valid finish is
        # authoritative. Files sort finish-before-start; never count them twice.
        if path.name.endswith(".start.json"):
            if key in records:
                continue
            record = {
                **record,
                "status": "unresolved",
                "wall_seconds": None,
                "cpu_user_seconds": None,
                "cpu_system_seconds": None,
                "gpu_seconds": None,
            }
        records[key] = record
    return list(records.values())


def _attempt_totals(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    attempts = [record for record in records if record["status"] != "skipped"]
    timed = [record for record in attempts if record.get("wall_seconds") is not None]
    cpu_user = [
        record["cpu_user_seconds"]
        for record in attempts
        if record.get("cpu_user_seconds") is not None
    ]
    cpu_system = [
        record["cpu_system_seconds"]
        for record in attempts
        if record.get("cpu_system_seconds") is not None
    ]
    by_status = {
        status: sum(record["wall_seconds"] for record in timed if record["status"] == status)
        for status in ("completed", "failed", "paused", "unresolved")
    }
    worker_seconds = sum(record["wall_seconds"] for record in timed)
    # Elapsed worker time is not CPU-core time. GPU time is elapsed attempt time
    # multiplied by the worker's exclusive allocation, not measured utilization.
    gpu_seconds = sum(record["wall_seconds"] * record["allocated_gpu_count"] for record in timed)
    return {
        "attempt_count": len(attempts),
        "timed_attempt_count": len(timed),
        "unresolved_attempt_count": sum(record["status"] == "unresolved" for record in attempts),
        "attempts_without_wall_timing": len(attempts) - len(timed),
        "skipped_count": len(records) - len(attempts),
        "worker_seconds": worker_seconds,
        "worker_hours": worker_seconds / 3600,
        "gpu_seconds": gpu_seconds,
        "gpu_hours": gpu_seconds / 3600,
        "gpu_attempts_without_wall_timing": sum(
            record["allocated_gpu_count"] > 0 and record.get("wall_seconds") is None
            for record in attempts
        ),
        "cpu_user_seconds": sum(cpu_user) if cpu_user else None,
        "cpu_system_seconds": sum(cpu_system) if cpu_system else None,
        "cpu_user_timed_attempts": len(cpu_user),
        "cpu_system_timed_attempts": len(cpu_system),
        **{f"{status}_worker_seconds": seconds for status, seconds in by_status.items()},
        "failed_paused_worker_seconds": by_status["failed"] + by_status["paused"],
    }


def _timing_stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "total_seconds": sum(values),
        "median_seconds": statistics.median(values) if values else None,
        "mean_seconds": statistics.mean(values) if values else None,
        "min_seconds": min(values) if values else None,
        "max_seconds": max(values) if values else None,
    }


def _group_timings(attempts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for attempt in attempts:
        if attempt["status"] != "skipped":
            groups[
                (attempt.get("group") or "unknown", attempt.get("algorithm") or "unknown")
            ].append(attempt)
    return [
        {
            "group": group,
            "algorithm": algorithm,
            "attempt_totals": _attempt_totals(records),
            "all_attempts": _timing_stats(
                [
                    record["wall_seconds"]
                    for record in records
                    if record.get("wall_seconds") is not None
                ]
            ),
            "completed_attempts": _timing_stats(
                [
                    record["wall_seconds"]
                    for record in records
                    if record["status"] == "completed" and record.get("wall_seconds") is not None
                ]
            ),
        }
        for (group, algorithm), records in sorted(groups.items())
    ]


def _coverage(
    root: Path, attempts: Sequence[dict[str, Any]], warnings: list[str]
) -> dict[str, Any]:
    """Inventory completion metadata without reading or revalidating numerical arrays."""
    active = set()
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        try:
            active = set(read_json(metadata_path)["run_ids"])
        except (StorageError, TypeError, KeyError) as exc:
            detail = str(exc).replace(str(root), "<output>")
            warnings.append(f"Active variant inventory unavailable: {detail}")
    completed = {}
    for progress_path in sorted((root / "runs").glob("*/progress.json")):
        try:
            progress = read_json(progress_path)
            if progress.get("status") != "completed" and not progress.get("completed_budgets"):
                continue
            metadata = read_json(progress_path.parent / "metadata.json")
            completed[progress_path.parent.name] = metadata["spec"]["run_id"]
        except (StorageError, TypeError, KeyError, AttributeError) as exc:
            detail = str(exc).replace(str(root), "<output>")
            warnings.append(
                f"Completion inventory unavailable for {progress_path.parent.name}: {detail}"
            )
    timed_completions = [
        attempt
        for attempt in attempts
        if attempt["status"] == "completed" and attempt.get("wall_seconds") is not None
    ]
    timed_variants = {attempt["storage_id"] for attempt in timed_completions}
    unknown_variants = sorted(set(completed) - timed_variants)
    earliest_attempts = {}
    unresolved_variants = set()
    for attempt in sorted(
        attempts, key=lambda record: (record.get("started_at") or "", record["attempt_id"])
    ):
        storage_id = attempt["storage_id"]
        if storage_id not in completed or attempt["status"] == "skipped":
            continue
        earliest_attempts.setdefault(storage_id, attempt)
        if attempt["status"] == "unresolved":
            unresolved_variants.add(storage_id)
    unrecorded_prefixes = {
        storage_id
        for storage_id, attempt in earliest_attempts.items()
        if (attempt.get("start_step") or 0) > 0
    }
    # These are positive signs of missing history, not a completeness proof for
    # the remaining variants. Earlier records may have been deleted even when
    # the first retained attempt begins at step zero.
    partial_variants = unrecorded_prefixes | unresolved_variants
    return {
        "completed_scientific_runs_with_timing": len(
            {attempt["run_id"] for attempt in timed_completions}
        ),
        "retained_completed_scientific_runs_with_timing": len(
            {run_id for storage_id, run_id in completed.items() if storage_id in timed_variants}
        ),
        "retained_completed_variants": len(completed),
        "retained_completed_variants_without_timing": len(unknown_variants),
        "retained_completed_scientific_runs_without_timing": len(
            {completed[storage_id] for storage_id in unknown_variants}
            - {completed[storage_id] for storage_id in set(completed) & timed_variants}
        ),
        "variants_without_timing": unknown_variants,
        "retained_completed_variants_with_partial_timing_evidence": len(partial_variants),
        "variants_with_partial_timing_evidence": sorted(partial_variants),
        "variants_with_unrecorded_prefix": sorted(unrecorded_prefixes),
        "variants_with_unresolved_attempts": sorted(unresolved_variants),
        "active_variant_count": len(active),
        "active_completed_variant_count": len(active & set(completed)),
        "active_completed_attempt_worker_seconds": sum(
            attempt["wall_seconds"]
            for attempt in timed_completions
            if attempt["storage_id"] in active
        ),
        "other_completed_attempt_worker_seconds": sum(
            attempt["wall_seconds"]
            for attempt in timed_completions
            if attempt["storage_id"] not in active
        ),
        "inventory_semantics": "Completion metadata only; numerical artifact integrity is checked by ews inspect, not compute reporting.",
        "timing_semantics": (
            "Known timing means at least one completed attempt was recorded, not that the "
            "entire run history is known. Resumed attempts cover only the work in that "
            "attempt; older, deleted, or unresolved attempts may be missing. An earliest "
            "recorded attempt starting above step zero or an unresolved attempt is known "
            "missing/partial timing evidence. The absence of these signs does not prove "
            "complete timing history."
        ),
    }


def _regenerate_summary(root: Path) -> dict[str, Any]:
    warnings: list[str] = []
    directory = root / "compute"
    invocations = _records(directory / "invocations", attempt=False, warnings=warnings)
    invocations.sort(key=lambda record: (record.get("started_at") or "", record["invocation_id"]))
    attempts = _records(directory / "attempts", attempt=True, warnings=warnings)
    attempts_by_invocation = defaultdict(list)
    for attempt in attempts:
        attempts_by_invocation[attempt["invocation_id"]].append(attempt)
    invocation_summaries = []
    for invocation in invocations:
        invocation_summaries.append(
            {
                **invocation,
                "invocation_record_available": True,
                "attempt_totals": _attempt_totals(
                    attempts_by_invocation.pop(invocation["invocation_id"], [])
                ),
            }
        )
    for identifier, records in sorted(attempts_by_invocation.items()):
        invocation_summaries.append(
            {
                "invocation_id": identifier,
                "invocation_record_available": False,
                "started_at": None,
                "finished_at": None,
                "wall_seconds": None,
                "status": "unresolved",
                "stages": {},
                "counts": {},
                "machine": {},
                "artifact_bytes": None,
                "attempt_totals": _attempt_totals(records),
            }
        )
    totals = {
        **_attempt_totals(attempts),
        "invocation_count": len(invocations),
        "invocations_without_final_record": len(attempts_by_invocation),
        "invocation_wall_seconds": sum(record.get("wall_seconds") or 0 for record in invocations),
        **{
            f"{stage}_wall_seconds": sum(record["stages"].get(stage) or 0 for record in invocations)
            for stage in ("execution", "analysis", "plotting")
        },
    }
    summary = {
        "schema_version": 1,
        "scope": _SCOPE,
        "latest_invocation": invocations[-1] if invocations else None,
        "totals": totals,
        "coverage": _coverage(root, attempts, warnings),
        "invocations": invocation_summaries,
        "attempt_timing": _timing_stats(
            [
                attempt["wall_seconds"]
                for attempt in attempts
                if attempt["status"] != "skipped" and attempt.get("wall_seconds") is not None
            ]
        ),
        "timing_by_group": _group_timings(attempts),
        "warnings": warnings,
    }
    atomic_json(directory / "summary.json", summary)
    atomic_text(directory / "summary.md", _markdown(summary))
    return summary


def _seconds(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 60:
        return f"{value:.3g} s"
    if value < 3600:
        return f"{value / 60:.3g} min"
    return f"{value / 3600:.3g} h"


def _bytes(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 1024**3:.3g} GiB"


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _machine_lines(machine: dict[str, Any]) -> list[str]:
    platform = (
        " ".join(str(machine[key]) for key in ("platform", "architecture") if machine.get(key))
        or "unavailable"
    )
    lines = [
        f"- Platform: {_cell(platform)}; "
        f"OS {_cell(machine.get('os_version') or 'unavailable')}; "
        f"kernel {_cell(machine.get('kernel_version') or 'unavailable')}.",
        f"- CPU: {_cell(machine.get('cpu_model') or 'unavailable')}.",
        f"- CPU cores: {machine.get('physical_cpu_cores') or 'unavailable'} physical / "
        f"{machine.get('logical_cpu_count') or 'unavailable'} logical.",
        f"- System RAM: {_bytes(machine.get('ram_bytes'))}.",
        f"- Configured workers: {machine.get('workers', 'unavailable')}.",
        f"- Allocated GPUs: {machine.get('allocated_gpu_count', 'unavailable')}.",
    ]
    for gpu in machine.get("gpus", []):
        lines.append(
            f"- GPU: {_cell(gpu.get('model') or 'unavailable')}, "
            f"{_bytes(gpu.get('vram_bytes'))} VRAM; "
            f"driver {_cell(gpu.get('driver_version') or 'unavailable')}."
        )
    if machine.get("gpu_inventory_scope") and (
        machine.get("allocated_gpu_count") or machine.get("gpus")
    ):
        lines.append(f"- GPU inventory scope: {_cell(machine['gpu_inventory_scope'])}.")
    for chip in machine.get("apple_accelerator", []):
        lines.append(
            f"- Apple accelerator/chip: {_cell(chip.get('model') or 'unavailable')}; "
            f"cores: {chip.get('cores') or 'unavailable'} (detected, not an EWS GPU allocation)."
        )
    lines.append(
        f"- Output filesystem: {_bytes(machine.get('filesystem_total_bytes'))} capacity; "
        f"{_bytes(machine.get('filesystem_available_bytes'))} available at invocation start."
    )
    return lines


def _markdown(summary: dict[str, Any]) -> str:
    latest = summary["latest_invocation"] or {}
    totals, coverage = summary["totals"], summary["coverage"]
    lines = [
        "# Experimental compute report",
        "",
        "## Latest invocation environment",
        "",
        *_machine_lines(latest.get("machine", {})),
        "",
        "Hardware and worker assignments for every recorded invocation are retained in JSON; "
        "this environment describes only the latest invocation.",
        "",
        "## Observed experiment compute",
        "",
        "- Completed scientific runs with recorded attempt timing: "
        f"{coverage['completed_scientific_runs_with_timing']}.",
        "- Retained completed variants without recorded completion timing: "
        f"{coverage['retained_completed_variants_without_timing']} "
        "(historical compute unavailable).",
        f"- Recorded execution attempts: {totals['attempt_count']}; "
        f"timed: {totals['timed_attempt_count']}; "
        f"unresolved: {totals['unresolved_attempt_count']}.",
        f"- End-to-end wall time, summed across {totals['invocation_count']} "
        f"finalized invocations: {_seconds(totals['invocation_wall_seconds'])}.",
        f"- Execution / analysis / plotting wall time: "
        f"{_seconds(totals['execution_wall_seconds'])} / "
        f"{_seconds(totals['analysis_wall_seconds'])} / "
        f"{_seconds(totals['plotting_wall_seconds'])}.",
        f"- Cumulative worker time: {totals['worker_hours']:.6g} worker-hours "
        f"({_seconds(totals['worker_seconds'])}).",
        f"- Cumulative allocated GPU time: {totals['gpu_hours']:.6g} GPU-hours "
        f"({_seconds(totals['gpu_seconds'])}).",
        f"- Process CPU user / system time: {_seconds(totals['cpu_user_seconds'])} / "
        f"{_seconds(totals['cpu_system_seconds'])}; measured for "
        f"{totals['cpu_user_timed_attempts']} / {totals['cpu_system_timed_attempts']} attempts.",
        f"- Completed-attempt worker time: {_seconds(totals['completed_worker_seconds'])}; "
        "currently selected variants: "
        f"{_seconds(coverage['active_completed_attempt_worker_seconds'])}; "
        "other historical variants: "
        f"{_seconds(coverage['other_completed_attempt_worker_seconds'])}.",
        f"- Failed / paused attempt worker time: {_seconds(totals['failed_worker_seconds'])} / "
        f"{_seconds(totals['paused_worker_seconds'])}.",
        f"- Output artifacts after the latest invocation: {_bytes(latest.get('artifact_bytes'))}.",
        "",
        "Wall time measures elapsed duration. Worker time sums actual execution attempts, "
        "excluding reused results; it is not CPU-core-hours. Process CPU time measures the "
        "executing process across its threads and excludes child processes. GPU time is "
        "attempt wall time × allocated GPUs, not GPU utilization. Machine RAM is capacity, "
        "not measured run memory.",
        "",
        "Artifact size sums regular file lengths after execution and analysis, immediately "
        "before final report publication. It excludes symbolic links and is logical file "
        "size, not allocated filesystem blocks or peak disk usage.",
        "",
        "Completed-attempt time and failed/paused time are shown separately. A resumed "
        "completion may depend on earlier paused work; these categories do not identify "
        "the full cost of final results or whether a historical condition was exploratory.",
    ]
    if latest:
        counts = latest.get("counts", {})
        lines += [
            "",
            "## Latest invocation",
            "",
            f"UTC: {_cell(latest.get('started_at') or 'unavailable')} → "
            f"{_cell(latest.get('finished_at') or 'unavailable')}. "
            f"End-to-end wall time: {_seconds(latest.get('wall_seconds'))}.",
            "",
            ", ".join(
                f"{key}: {counts.get(key, 0)}"
                for key in ("completed", "skipped", "paused", "failed", "pending")
            )
            + ". Skipped results add no new simulation compute.",
        ]
    lines += [
        "",
        "## Per-attempt timing",
        "",
        "Statistics are elapsed seconds for recorded execution attempts, including failed "
        "and paused attempts. Resumed attempts are partial work, not whole-run durations. "
        "JSON also retains statistics for completed attempts separately.",
        "",
        "| Group | Algorithm | Count | Median (s) | Mean (s) | Min (s) | Max (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["timing_by_group"][:20]:
        stats = group["all_attempts"]
        values = [
            "unavailable" if stats[key] is None else f"{stats[key]:.6g}"
            for key in ("median_seconds", "mean_seconds", "min_seconds", "max_seconds")
        ]
        lines.append(
            f"| {_cell(group['group'])} | {_cell(group['algorithm'])} | {stats['count']} | "
            + " | ".join(values)
            + " |"
        )
    if not summary["timing_by_group"]:
        lines.append("| — | — | 0 | unavailable | unavailable | unavailable | unavailable |")
    if len(summary["timing_by_group"]) > 20:
        lines += [
            "",
            "The first 20 conditions are shown; all conditions are retained in summary.json.",
        ]
    lines += [
        "",
        "## Coverage and limitations",
        "",
        coverage["timing_semantics"],
        "",
        "Retained completed variants with known missing/partial timing evidence: "
        f"{coverage['retained_completed_variants_with_partial_timing_evidence']} "
        f"(unrecorded prefix: {len(coverage['variants_with_unrecorded_prefix'])}; "
        f"unresolved attempts: {len(coverage['variants_with_unresolved_attempts'])}). "
        "Exact variant identifiers are retained in summary.json.",
        "",
        f"Attempts with unknown duration: {totals['attempts_without_wall_timing']}; "
        "GPU attempts with unknown duration: "
        f"{totals['gpu_attempts_without_wall_timing']}; invocations lacking a final record: "
        f"{totals['invocations_without_final_record']}. Unknown durations are excluded from "
        "totals, never estimated as zero consumption.",
        "",
        coverage["inventory_semantics"],
        "",
        _SCOPE,
    ]
    if summary["warnings"]:
        lines += [
            "",
            f"Reporting encountered {len(summary['warnings'])} damaged or unavailable "
            "metadata records. See summary.json warnings; totals may be incomplete.",
        ]
    return "\n".join(lines) + "\n"
