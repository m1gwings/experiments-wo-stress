"""Explicit, previewable removal of owned experiment artifacts."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from .storage import EXPERIMENT_SCHEMA_VERSION, StorageError, atomic_json, read_json


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
    ``all`` empties this experiment directory without removing the directory itself.
    """
    from .runner import _experiment_lock

    if scope not in {"analysis", "checkpoints", "inactive", "runs", "all"}:
        raise ValueError("Unknown cleanup scope")
    if run_ids and scope not in {"runs", "checkpoints"}:
        raise ValueError("run_ids is supported only for runs or checkpoints cleanup")
    root = Path(output_dir).absolute()
    if root.is_symlink():
        raise StorageError("Cleanup does not follow a symlinked experiment root")
    metadata = read_json(root / "metadata.json")
    if metadata.get("schema_version") not in (1, EXPERIMENT_SCHEMA_VERSION):
        raise StorageError("Cleanup requires recognized experiment metadata")
    with _experiment_lock(root):
        # Execution may have published a new request while this process waited.
        metadata = read_json(root / "metadata.json")
        if metadata.get("schema_version") not in (1, EXPERIMENT_SCHEMA_VERSION):
            raise StorageError("Cleanup requires recognized experiment metadata")
        # Refuse linked subtrees before computing or applying a deletion plan.
        if any(path.is_symlink() for path in root.rglob("*")):
            raise StorageError("Cleanup does not follow symlinks inside experiment artifacts")
        variants = {path.name: path for path in (root / "runs").iterdir() if path.is_dir()}
        if run_ids:
            for run_id in run_ids:
                if not re.fullmatch(r"[a-f0-9]{20,64}", run_id) or run_id not in variants:
                    raise ValueError(f"Unknown stored run variant: {run_id}")
            selected = {key: variants[key] for key in run_ids}
        else:
            selected = variants
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
