"""Completed-result references and deliberate removal of consumed trajectories.

Receipts preserve the exact input revision of derived artifacts. Publication of
a checksummed pruning marker precedes removal, so interruption cannot turn an
intentional deletion into apparent corruption. Settled trajectories discard their
unusable checkpoints after removing durable references to them.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..study.specs import RunSpec
from .experiment import load_instance
from .files import SCHEMA_VERSION, StorageError, atomic_json, fingerprint, read_json, sync_directory
from .models import RunResult
from .run import RunStore, read_result_prefix


def _pruning_proof(directory: Path, filename: str, state: str) -> dict[str, Any] | None:
    """Validate pruning evidence against the retained variant's scientific identity."""
    path = directory / filename
    if not path.exists():
        return None
    receipt = read_json(path)
    metadata = read_json(directory / "metadata.json")
    if (
        not isinstance(receipt, dict)
        or not isinstance(metadata, dict)
        or not isinstance(metadata.get("identity"), dict)
    ):
        raise StorageError(f"Invalid trajectory pruning receipt: {path}")
    payload = {key: value for key, value in receipt.items() if key != "sha256"}
    try:
        valid_checksums = fingerprint(payload) == receipt.get("sha256") and receipt.get(
            "identity"
        ) == fingerprint(metadata["identity"])
    except (TypeError, ValueError):
        valid_checksums = False
    references = receipt.get("references")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("state") != state
        or not valid_checksums
        or not isinstance(references, dict)
        or not references
        or any(not _valid_reference(key, value) for key, value in references.items())
        or not isinstance(receipt.get("latest"), str)
        or receipt["latest"] not in references
        or (
            state == "pruned"
            and (not isinstance(receipt.get("metrics"), dict) or not receipt["metrics"])
        )
    ):
        raise StorageError(f"Invalid trajectory pruning receipt: {path}")
    return receipt


def _valid_reference(key: Any, reference: Any) -> bool:
    """Recognize a proof of one exact completed budget and its input revision."""
    if not isinstance(reference, dict):
        return False
    steps = reference.get("completed_steps")
    revision = reference.get("revision")
    return (
        type(steps) is int
        and steps >= 0
        and key == str(steps)
        and isinstance(revision, str)
        and re.fullmatch(r"[a-f0-9]{64}", revision) is not None
    )


def pruning_receipt(directory: Path) -> dict[str, Any] | None:
    """Read a valid marker for the currently pruned trajectory, rejecting damaged proof."""
    return _pruning_proof(directory, "trajectory.json", "pruned")


def settle_pruned_checkpoints(directory: Path) -> None:
    """Discard checkpoint references, then payloads, for a proven pruned run.

    The receipt must already be durable. Replaying after either publication
    boundary is safe: progress never names a generation removed by this method.
    Historical checkpoint counters remain useful diagnostics.
    """
    if pruning_receipt(directory) is None:
        raise StorageError(f"Trajectory is not proven pruned: {directory}")
    progress_path = directory / "progress.json"
    progress = read_json(progress_path)
    completed = progress.get("completed_budgets", {})
    if progress.get("checkpoints") or any(item.get("checkpoint") for item in completed.values()):
        progress["checkpoints"] = []
        for item in completed.values():
            if "checkpoint" in item:
                item["checkpoint"] = None
        atomic_json(progress_path, progress)
    checkpoints = directory / "checkpoints"
    if checkpoints.exists():
        for generation in checkpoints.iterdir():
            if generation.is_symlink() or generation.is_file():
                generation.unlink()
            else:
                shutil.rmtree(generation)
        sync_directory(checkpoints)


def pruning_history(directory: Path) -> dict[str, Any] | None:
    """Read exact completed-budget references preserved across fresh rematerialization.

    History proves inputs of retained derived artifacts, never the availability
    of a current raw trajectory or a resumable checkpoint. A later execution may
    be in progress or have completed at another budget while these proofs remain
    valid for their original derivations.
    """
    return _pruning_proof(directory, "trajectory-history.json", "pruned_history")


def _pruned_reference(receipt: dict | None, spec: RunSpec) -> dict[str, Any] | None:
    if receipt is None:
        return None
    key = str(spec.budget_steps) if spec.budget_steps is not None else receipt["latest"]
    reference = receipt["references"].get(key)
    return {**reference, "trajectory": "pruned"} if reference else None


def _run_metadata(directory: Path, spec: RunSpec) -> dict:
    metadata = read_json(directory / "metadata.json")
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("spec"), dict)
        or metadata["spec"].get("run_id") != spec.run_id
    ):
        raise StorageError(f"Run identity mismatch: {directory}")
    return metadata


def _completion(
    root: Path, storage_id: str, spec: RunSpec, metadata: dict | None = None
) -> tuple[dict, dict] | None:
    directory = root / "runs" / storage_id
    if not (directory / "progress.json").is_file():
        return None
    if metadata is None:
        metadata = _run_metadata(directory, spec)
    completion = RunStore(directory).completion(spec.budget_steps, validate=False)
    return (metadata, completion) if completion else None


def _revision(manifest: dict, chunks: list[dict], instance_id: str, steps: int) -> str:
    return fingerprint(
        {"chunks": chunks, "instance": instance_id, "steps": steps, "schema": manifest["schema"]}
    )


def result_reference(root: Path, storage_id: str, spec: RunSpec) -> dict[str, Any] | None:
    """Describe a completed input revision without requiring intentionally deleted data.

    A full completed boundary is described entirely by committed metadata. Reading
    a shorter, previously unrequested prefix requires its retained raw ancestor.
    Consumers validate a retained cache at their nearest dependency boundary;
    loading raw results separately validates every consumed chunk and instance.
    """
    root = Path(root)
    directory = root / "runs" / storage_id
    if not (directory / "progress.json").is_file():
        return None
    metadata = _run_metadata(directory, spec)
    receipt = pruning_receipt(directory)
    if receipt:
        return _pruned_reference(receipt, spec) or _pruned_reference(
            pruning_history(directory), spec
        )
    selected = _completion(root, storage_id, spec, metadata)
    if selected is None:
        return _pruned_reference(pruning_history(directory), spec)
    metadata, completion = selected
    instance_id = metadata.get("instance_id")
    if not instance_id:
        raise StorageError("Completed run is missing its scientific instance reference")
    steps = spec.budget_steps if spec.budget_steps is not None else completion["step"]
    manifest = completion["results"]
    if steps == completion["step"]:
        chunks = [{**entry, "used_rows": entry["rows"]} for entry in manifest["chunks"]]
    else:
        _, chunks = read_result_prefix(directory, manifest, steps)
    return {
        "revision": _revision(manifest, chunks, instance_id, steps),
        "completed_steps": steps,
        "trajectory": "retained",
    }


def load_run_result(root: Path, storage_id: str, spec: RunSpec) -> RunResult:
    """Load and validate one requested trajectory, never construct simulation objects."""
    root = Path(root)
    directory = root / "runs" / storage_id
    if pruning_receipt(directory):
        raise StorageError(
            f"Trajectory for {spec.run_id} was intentionally pruned; use ews run with "
            "recording.retention: keep to rematerialize raw observations"
        )
    selected = _completion(root, storage_id, spec)
    if selected is None:
        raise StorageError(f"Run {spec.run_id} is not completed; use ews run first")
    metadata, completion = selected
    instance_id = metadata.get("instance_id")
    if not instance_id:
        raise StorageError("Completed run is missing its scientific instance reference")
    steps = spec.budget_steps if spec.budget_steps is not None else completion["step"]
    manifest = completion["results"]
    records, chunks = read_result_prefix(directory, manifest, steps)
    instance = load_instance(root, instance_id)
    return RunResult(
        records=records,
        instance=instance,
        spec=spec,
        completed_steps=steps,
        revision=_revision(manifest, chunks, instance_id, steps),
        final_outputs={key: values[-1] for key, values in records.items() if key != "step"}
        if steps == 1
        else {},
    )


def prune_trajectory(root: Path, storage_id: str, spec: RunSpec, metrics: dict) -> None:
    """Commit metric and trajectory proofs, then discard obsolete raw state.

    Callers hold experiment ownership and supply validated metric cache identities.
    A running extension is never pruned. Repeating this operation completes a
    deletion interrupted after its durable marker was published.
    """
    directory = Path(root) / "runs" / storage_id
    receipt = pruning_receipt(directory)
    if receipt is None:
        store = RunStore(directory)
        if store.progress["status"] != "completed" or not metrics:
            return
        # A prefix-only request must not consume unanalysed observations beyond it.
        if spec.budget_steps is not None and spec.budget_steps != store.progress["step"]:
            return
        load_run_result(root, storage_id, spec)
        metadata = read_json(directory / "metadata.json")
        history = pruning_history(directory)
        references = dict(history["references"]) if history else {}
        for step in {*store.progress.get("completed_budgets", {}), str(store.progress["step"])}:
            reference = result_reference(root, storage_id, replace(spec, budget_steps=int(step)))
            references[step] = {
                key: value for key, value in reference.items() if key != "trajectory"
            }
        receipt = {
            "schema_version": 1,
            "state": "pruned",
            "identity": fingerprint(metadata["identity"]),
            "latest": str(store.progress["step"]),
            "references": references,
            "metrics": metrics,
        }
        atomic_json(directory / "trajectory.json", {**receipt, "sha256": fingerprint(receipt)})
    results = directory / "results"
    # A sealed recovery snapshot omits the empty results directory for pruned runs.
    # Replaying an already published pruning receipt still has no chunks to delete.
    if results.exists():
        for path in results.glob("*.npz"):
            path.unlink()
        sync_directory(results)
    settle_pruned_checkpoints(directory)


def rematerialize_trajectory(directory: Path) -> None:
    """Reset a deliberately pruned run to fresh execution without borrowing its endpoint.

    Archive exact completed-budget references before replacing current progress.
    Checkpoint payloads are no longer resume candidates: their trajectory prefixes
    are unavailable. Publish the reset before clearing
    the marker; a crash at either boundary remains safely restartable, and older
    derived artifacts retain their input identities throughout the replay.
    """
    receipt = pruning_receipt(directory)
    if receipt is None:
        return
    previous = pruning_history(directory)
    history = {
        "schema_version": 1,
        "state": "pruned_history",
        "identity": receipt["identity"],
        "latest": receipt["latest"],
        "references": {
            **(previous["references"] if previous else {}),
            **receipt["references"],
        },
    }
    atomic_json(directory / "trajectory-history.json", {**history, "sha256": fingerprint(history)})
    atomic_json(
        directory / "progress.json",
        {"schema_version": SCHEMA_VERSION, "status": "pending", "checkpoints": [], "step": 0},
    )
    (directory / "trajectory.json").unlink()
    sync_directory(directory)
