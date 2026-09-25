"""Data generators for the sequential Gaussian bandit study.

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
    those observations and the saved means. ``context`` reveals only the number
    of arms. ``generate`` samples the chosen arm's reward and advances the round
    counter, which is the component state saved in a checkpoint.
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
    ``generate`` first obtains the parent's reward, then clips the feedback and
    saves the original value as ``raw_reward`` for later evaluation. This changes
    the learning problem and is deliberately opt-in.
    """

    def generate(self, action: int) -> Feedback:
        feedback = super().generate(action)
        observation = float(np.clip(feedback.value, 0.0, 1.0))
        measurements = dict(feedback.measurements)
        measurements["raw_reward"] = measurements["reward"]
        measurements["reward"] = observation
        return Feedback(observation, measurements)
