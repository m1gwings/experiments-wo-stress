"""A minimal study-specific estimator used with the library's CSV generator."""

from __future__ import annotations

from typing import Any

import numpy as np


class MeanEstimator:
    """Estimate the sample mean and unbiased sample variance of one CSV column."""

    def __init__(self, *, rng: np.random.Generator, column: int = 0) -> None:
        self.rng = rng
        self.column = column
        self.result: dict[str, float | int] = {}

    def fit(self, dataset: np.ndarray) -> dict[str, float | int]:
        values = np.asarray(dataset)[:, self.column]
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("at least two finite observations are required")
        self.result = {
            "mean": float(values.mean()),
            "variance": float(values.var(ddof=1)),
            "samples": len(values),
        }
        return dict(self.result)

    def state_dict(self) -> dict[str, Any]:
        return {"result": self.result}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.result = dict(state["result"])
