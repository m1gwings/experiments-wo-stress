"""A saved trajectory for tests that must analyze without loading simulation code."""

from __future__ import annotations

import numpy as np

from experiments_wo_stress.artifacts import Instance, RunResult
from experiments_wo_stress.jobs import ComponentSpec, RunSpec


def make_saved_result(*, means=None, records=None, completed_steps=3, revision="first"):
    """Build a completed bandit trajectory whose simulation modules cannot be imported."""
    spec = RunSpec(
        "a" * 20,
        "study",
        0,
        "algorithm",
        ComponentSpec("missing_sim:Algorithm"),
        ComponentSpec("missing_sim:Data"),
        ComponentSpec("missing_sim:Protocol"),
        12,
    )
    return RunResult(
        records=records
        if records is not None
        else {
            "step": np.array([1, 2, 3]),
            "action": np.array([0, 1, 0]),
            "reward": np.array([1.0, 2.0, 3.0]),
        },
        instance=Instance(arrays={"means": np.array([0.2, 0.8]) if means is None else means}),
        spec=spec,
        completed_steps=completed_steps,
        revision=revision,
    )
