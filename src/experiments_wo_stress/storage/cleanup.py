"""Explicit, previewable removal of owned experiment artifacts."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .experiment import ExperimentStore
from .files import StorageError, atomic_json, read_json
from .trajectories import pruning_receipt, settle_pruned_checkpoints


def clean_experiment(
    output_dir: str | Path,
    *,
    scope: str = "inactive",
    run_ids: list[str] | None = None,
    yes: bool = False,
) -> dict[str, Any]:
    """Preview cleanup by default; ``yes=True`` removes precisely the selected data.

    ``inactive`` removes variants outside the latest execution request. Clearing
    checkpoints preserves numerical results but forfeits resume/extension state.
    ``settled`` removes only checkpoints of runs with valid pruning receipts.
    ``all`` empties this experiment directory without removing the directory itself.
    """
    if scope not in {"analysis", "checkpoints", "inactive", "runs", "settled", "all"}:
        raise ValueError("Unknown cleanup scope")
    if run_ids and scope not in {"runs", "checkpoints"}:
        raise ValueError("run_ids is supported only for runs or checkpoints cleanup")
    root = Path(output_dir).absolute()
    if root.is_symlink():
        raise StorageError("Cleanup does not follow a symlinked experiment root")
    experiment_store = ExperimentStore(root)
    metadata = experiment_store.metadata()
    with experiment_store.lock():
        # Execution may have published a new request while this process waited.
        metadata = experiment_store.metadata()
        # Refuse linked subtrees before computing or applying a deletion plan.
        if any(path.is_symlink() for path in root.rglob("*")):
            raise StorageError("Cleanup does not follow symlinks inside experiment artifacts")
        variants = experiment_store.retained_variants()
        if run_ids:
            for run_id in run_ids:
                if not re.fullmatch(r"[a-f0-9]{20,64}", run_id) or run_id not in variants:
                    raise ValueError(f"Unknown stored run variant: {run_id}")
            selected = {key: variants[key] for key in run_ids}
        else:
            selected = variants
        if scope == "settled":
            settled = []
            affected = []
            generations = 0
            files = 0
            size = 0
            for directory in sorted(selected.values()):
                if pruning_receipt(directory) is None:
                    continue
                settled.append(directory)
                progress = read_json(directory / "progress.json")
                has_references = bool(progress.get("checkpoints")) or any(
                    item.get("checkpoint")
                    for item in progress.get("completed_budgets", {}).values()
                )
                checkpoints = directory / "checkpoints"
                has_payloads = False
                if checkpoints.exists():
                    for generation in checkpoints.iterdir():
                        has_payloads = True
                        generations += 1
                        for path in generation.rglob("*") if generation.is_dir() else [generation]:
                            if path.is_file():
                                files += 1
                                size += path.stat().st_size
                if has_references or has_payloads:
                    affected.append(directory)
            result = {
                "scope": scope,
                "dry_run": not yes,
                "runs_examined": len(selected),
                "settled_runs": len(settled),
                "runs_with_obsolete_checkpoints": len(affected),
                "checkpoint_generations": generations,
                "checkpoint_files": files,
                "bytes": size,
                "paths": [str(path.relative_to(root) / "checkpoints") for path in affected[:10]],
                "paths_omitted": max(0, len(affected) - 10),
            }
            if yes:
                for directory in settled:
                    settle_pruned_checkpoints(directory)
            return result
        if scope == "all":
            paths = [path for path in root.iterdir() if path.name != ".lock"]
        elif scope == "analysis":
            paths = [root / "analysis"]
        elif scope == "checkpoints":
            paths = [path / "checkpoints" for path in selected.values()]
        else:
            if scope == "inactive":
                selected = {
                    key: path for key, path in variants.items() if key not in metadata["run_ids"]
                }
            paths = list(selected.values())
            referenced = set()
            for key, path in variants.items():
                if key not in selected:
                    referenced.add(read_json(path / "metadata.json").get("instance_id"))
            instances = root / "instances"
            if instances.exists():
                paths.extend(path for path in instances.iterdir() if path.name not in referenced)
            requests = root / "requests"
            if requests.exists():
                paths.extend(
                    path
                    for path in requests.glob("*.json")
                    if set(read_json(path)["run_ids"]) & selected.keys()
                )
        paths = sorted((path for path in paths if path.exists()), key=str)
        size = sum(
            path.stat().st_size
            if path.is_file()
            else sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
            for path in paths
        )
        result = {
            "scope": scope,
            "dry_run": not yes,
            "bytes": size,
            "paths": [str(path.relative_to(root)) for path in paths],
        }
        if not yes:
            return result
        # Update pointers first: an interrupted cleanup must not retain invalid references.
        if scope == "checkpoints":
            for directory in selected.values():
                progress_path = directory / "progress.json"
                if progress_path.exists():
                    progress = read_json(progress_path)
                    progress["checkpoints"] = []
                    for completion in progress.get("completed_budgets", {}).values():
                        completion["checkpoint"] = None
                    atomic_json(progress_path, progress)
        elif scope in {"runs", "inactive"}:
            metadata["run_ids"] = [key for key in metadata["run_ids"] if key not in selected]
            if "requests" in metadata:
                metadata["requests"] = {
                    key: value for key, value in metadata["requests"].items() if key not in selected
                }
            atomic_json(root / "metadata.json", metadata)
        for path in paths:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return result
