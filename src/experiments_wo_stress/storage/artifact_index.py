"""Publish the semantic catalog for consumers outside EWS's storage implementation."""

from __future__ import annotations

from pathlib import Path

from .files import atomic_json

ARTIFACT_INDEX_FILENAME = "artifacts.json"
ARTIFACT_INDEX_SCHEMA = "experiments-wo-stress/artifacts"
ARTIFACT_INDEX_VERSION = 1


def publish_artifact_index(output_dir: str | Path) -> None:
    """Atomically publish deterministic role locations relative to the output root.

    This is a catalog, not an inventory or completion record. Optional locations
    may be absent or empty; consumers check their filesystem or object listing.
    No presence flags, timestamps, user configuration, or tree scans are needed.
    Concurrent workflows publish identical bytes, without read/modify/write races.
    Keep these locations aligned with their EWS owners when changing the layout.
    """
    locations = {
        "figures": ("analysis/figures", "directory"),
        "analysis": ("analysis", "directory"),
        "compute_report": ("compute/summary.md", "file"),
        "compute": ("compute", "directory"),
        "runs": ("runs", "directory"),
        "instances": ("instances", "directory"),
        "requests": ("requests", "directory"),
    }
    atomic_json(
        Path(output_dir) / ARTIFACT_INDEX_FILENAME,
        {
            "schema": ARTIFACT_INDEX_SCHEMA,
            "schema_version": ARTIFACT_INDEX_VERSION,
            "artifacts": {
                role: {"path": path, "kind": kind, "optional": True}
                for role, (path, kind) in locations.items()
            },
        },
    )
