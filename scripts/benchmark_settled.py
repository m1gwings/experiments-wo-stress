"""Measure checkpoint object reduction on thousands of disposable settled runs.

This fixture models the old two-generation NumPy checkpoint inventory. It uses
temporary local files only and is deliberately outside the normal test suite.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from experiments_wo_stress.storage.cleanup import clean_experiment
from experiments_wo_stress.storage.files import fingerprint


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(root: Path, count: int) -> None:
    _json(root / "metadata.json", {"schema_version": 2, "run_ids": []})
    (root / ".lock").touch()
    for number in range(count):
        run = root / "runs" / f"{number:032x}"
        identity = {"synthetic_run": number}
        _json(run / "metadata.json", {"identity": identity})
        entries = []
        for generation_number in range(2):
            generation = f"{number * 2 + generation_number:032x}"
            directory = run / "checkpoints" / generation
            directory.mkdir(parents=True)
            for name in ("checkpoint.json", "state.json", "arrays.npz"):
                (directory / name).write_bytes(b"synthetic checkpoint payload")
            entries.append({"generation": generation})
        _json(
            run / "progress.json",
            {
                "schema_version": 1,
                "status": "completed",
                "step": 1,
                "checkpoints": entries,
                "completed_budgets": {
                    "1": {
                        "step": 1,
                        "results": {"schema": {}, "chunks": []},
                        "checkpoint": entries[0],
                    }
                },
                "checkpoint_count": 2,
            },
        )
        receipt = {
            "schema_version": 1,
            "state": "pruned",
            "identity": fingerprint(identity),
            "latest": "1",
            "references": {"1": {"completed_steps": 1, "revision": "0" * 64}},
            "metrics": {"metric": {"synthetic": True}},
        }
        _json(run / "trajectory.json", {**receipt, "sha256": fingerprint(receipt)})


def main() -> None:
    """Print before/after file counts and the explicit cleanup summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2000)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    with tempfile.TemporaryDirectory(prefix="ews-settled-benchmark-") as temporary:
        root = Path(temporary)
        _fixture(root, args.runs)
        before = sum(path.is_file() for path in root.rglob("*"))
        start = time.monotonic()
        preview = clean_experiment(root, scope="settled")
        clean_experiment(root, scope="settled", yes=True)
        elapsed = time.monotonic() - start
        after = sum(path.is_file() for path in root.rglob("*"))
        print(
            json.dumps(
                {
                    "runs": args.runs,
                    "files_before": before,
                    "files_after": after,
                    "files_removed": before - after,
                    "preview": preview,
                    "cleanup_seconds": round(elapsed, 3),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
