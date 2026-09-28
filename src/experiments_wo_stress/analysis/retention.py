"""Discard per-run metric generations after their aggregate is durable.

The validated aggregate is the retained proof of its metric inputs. Renaming a
metric generation before removal ensures an interrupted deletion never exposes a
partly removed generation as a usable cache entry. Recovery snapshots ignore the
hidden deletion directory; a later invocation completes its removal.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path

from ..storage.files import StorageError, sync_directory

_PENDING = re.compile(r"\.pruning-[0-9a-f]{64}-[0-9a-f]{32}")
_KEY = re.compile(r"[0-9a-f]{64}")


def prune_metric_generations(output_dir: Path, keys: list[str]) -> None:
    """Remove exact metric generations whose downstream aggregate was validated.

    The caller holds the experiment lock and has validated the aggregate's
    immutable cache identity. Other aggregates and unrelated metric revisions are
    untouched. Repeating the operation also removes hidden partial deletions.
    """
    parent = output_dir / "analysis" / "cache" / "metrics"
    if not parent.exists():
        return
    if parent.is_symlink() or not parent.is_dir():
        raise StorageError(f"Metric cache must be a real directory: {parent}")
    for path in parent.iterdir():
        if _PENDING.fullmatch(path.name):
            if path.is_symlink() or not path.is_dir():
                raise StorageError(f"Invalid interrupted metric deletion: {path}")
            shutil.rmtree(path)
    for key in keys:
        if not _KEY.fullmatch(key):
            raise StorageError(f"Invalid metric cache key: {key!r}")
        directory = parent / key
        if not directory.exists() and not directory.is_symlink():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise StorageError(f"Metric cache must be a real directory: {directory}")
        pending = parent / f".pruning-{key}-{uuid.uuid4().hex}"
        os.rename(directory, pending)
        sync_directory(parent)
        shutil.rmtree(pending)
    sync_directory(parent)
