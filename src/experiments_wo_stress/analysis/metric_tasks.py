"""Per-run metric computation from a validated persisted numerical trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..components.loading import load_class
from ..storage.files import write_arrays
from ..storage.trajectories import load_run_result
from ..study.specs import RunSpec
from .metrics import MetricResult, validate_requirements
from .pipeline import BUILTIN_METRICS


def write_metric(
    target: Path, output_dir: Path, payload: dict[str, Any], result_revision: str
) -> None:
    """Validate the exact raw ancestor and write one full-resolution metric curve.

    This module's digest belongs to metric identities. Keeping numerical input
    handling here separates metric science from scheduling and aggregation.
    Changes to computation, validation, or the graph's dependency transformations
    invalidate derived artifacts conservatively.
    """
    descriptor = payload["metric"]
    metric = load_class(descriptor["type"], BUILTIN_METRICS)(**descriptor["params"])
    result = load_run_result(output_dir, payload["storage_id"], RunSpec.from_dict(payload["spec"]))
    if result.revision != result_revision:
        raise ValueError("Saved result changed after metric task preparation")
    validate_requirements(metric, result)
    computed = metric.compute(result)
    if not isinstance(computed, MetricResult):
        raise TypeError("Metric.compute() must return MetricResult")
    write_arrays(target / "result.npz", {"x": computed.x, "values": computed.values})
