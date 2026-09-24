"""Paper-owned algorithms and generators for a Gaussian bandit comparison.

All randomness comes from the injected NumPy generator. The library supplies the
interaction loop, independent repetitions, persistence, and analysis.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from experiments_wo_stress import Feedback
from experiments_wo_stress.artifacts import Instance


class GaussianBandit:
    """An environment with fixed arm means and independent Gaussian rewards.

    The generated arm means are saved once as an immutable instance. Each run
    records actions and rewards; a separate metric computes pseudo-regret using
    those observations and the saved means.
    """

    supports_extension = True
    logger = logging.getLogger(__name__)

    @classmethod
    def create_instance(
        cls, *, rng: np.random.Generator, n_arms: int = 10, noise_std: float = 0.1
    ) -> Instance:
        if n_arms < 2:
            raise ValueError("n_arms must be at least two")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        return Instance(
            kind="stationary_bandit",
            metadata={"n_arms": n_arms, "noise_std": noise_std},
            arrays={"means": rng.uniform(0.0, 1.0, n_arms)},
        )

    def __init__(
        self,
        *,
        rng: np.random.Generator,
        instance: Instance,
        n_arms: int = 10,
        noise_std: float = 0.1,
    ) -> None:
        if n_arms < 2:
            raise ValueError("n_arms must be at least two")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        self.rng = rng
        self.means = instance.arrays["means"]
        if len(self.means) != n_arms:
            raise ValueError("saved instance does not match the configured arm count")
        self.noise_std = noise_std
        self.round = 0

    def context(self) -> dict[str, int]:
        """Expose the action-space size without exposing the unknown arm means."""
        return {"n_arms": len(self.means)}

    def generate(self, action: int) -> Feedback:
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(self.means):
            raise ValueError("action must be a valid arm index")
        reward = float(self.rng.normal(self.means[action], self.noise_std))
        self.round += 1
        self.logger.debug("round=%d action=%d reward=%.8g", self.round, action, reward)
        return Feedback(reward, {"action": action, "reward": reward})

    def state_dict(self) -> dict[str, Any]:
        return {"round": self.round}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.round = int(state["round"])


class ClippedFeedbackBandit(GaussianBandit):
    """An example feedback rule: clip the observation delivered to the algorithm.

    Underlying Gaussian rewards and arm means retain their original meaning.
    This changes the learning problem and is deliberately opt-in.
    """

    def generate(self, action: int) -> Feedback:
        feedback = super().generate(action)
        observation = float(np.clip(feedback.value, 0.0, 1.0))
        measurements = dict(feedback.measurements)
        measurements["raw_reward"] = measurements["reward"]
        measurements["reward"] = observation
        return Feedback(observation, measurements)


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
