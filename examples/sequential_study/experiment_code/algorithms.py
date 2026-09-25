"""Learning rules for the sequential Gaussian bandit study."""

from __future__ import annotations

from typing import Any

import numpy as np


class _SampleMeans:
    """Shared bookkeeping for this paper's two learning rules."""

    def __init__(self, *, rng: np.random.Generator, n_arms: int | None = None) -> None:
        if n_arms is not None and n_arms < 2:
            raise ValueError("n_arms must be at least two")
        self.rng = rng
        self.counts = np.zeros(n_arms or 0, dtype=np.int64)
        self.sums = np.zeros(n_arms or 0, dtype=np.float64)

    def _prepare(self, context: dict[str, int] | None) -> None:
        if context is not None:
            n_arms = context["n_arms"]
            if not len(self.counts):
                self.counts = np.zeros(n_arms, dtype=np.int64)
                self.sums = np.zeros(n_arms, dtype=np.float64)
            elif len(self.counts) != n_arms:
                raise ValueError("algorithm and environment disagree about n_arms")
        if not len(self.counts):
            raise ValueError("n_arms must be provided in parameters or context")

    def observe(self, action: int, feedback: float) -> None:
        self.counts[action] += 1
        self.sums[action] += float(feedback)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"counts": self.counts, "sums": self.sums}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.counts = np.array(state["counts"], copy=True)
        self.sums = np.array(state["sums"], copy=True)


class UCB(_SampleMeans):
    """An illustrative upper-confidence algorithm for the example study.

    The exploration constant is a study parameter; this implementation does not
    claim a confidence guarantee for every possible reward distribution.
    """

    supports_extension = True

    def __init__(
        self, *, rng: np.random.Generator, n_arms: int | None = None, exploration: float = 0.1
    ) -> None:
        super().__init__(rng=rng, n_arms=n_arms)
        if not np.isfinite(exploration) or exploration < 0:
            raise ValueError("exploration must be finite and nonnegative")
        self.exploration = exploration

    def act(self, context: dict[str, int] | None = None) -> int:
        self._prepare(context)
        unplayed = np.flatnonzero(self.counts == 0)
        if len(unplayed):
            return int(unplayed[0])
        bonus = np.sqrt(self.exploration * np.log(self.counts.sum() + 1) / self.counts)
        return int(np.argmax(self.sums / self.counts + bonus))


class EpsilonGreedy(_SampleMeans):
    """Explore with fixed probability; otherwise choose the largest sample mean."""

    supports_extension = True

    def __init__(
        self, *, rng: np.random.Generator, n_arms: int | None = None, epsilon: float = 0.1
    ) -> None:
        super().__init__(rng=rng, n_arms=n_arms)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one")
        self.epsilon = epsilon

    def act(self, context: dict[str, int] | None = None) -> int:
        self._prepare(context)
        unplayed = np.flatnonzero(self.counts == 0)
        if len(unplayed):
            return int(unplayed[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.counts)))
        return int(np.argmax(self.sums / self.counts))
