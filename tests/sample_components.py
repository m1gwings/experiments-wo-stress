"""Small importable scientific components used by bandit and environment tests."""

from __future__ import annotations

import copy


class RecordingLearner:
    """Choose arm/action 1 and save observed feedback for exact continuation checks."""

    supports_extension = True

    def __init__(self, *, rng=None):
        self.rng = rng
        self.observed = []

    def act(self, context=None):
        return 1

    def observe(self, action, feedback):
        self.observed.append(copy.deepcopy(feedback))

    def state_dict(self):
        return {"observed": copy.deepcopy(self.observed)}

    def load_state_dict(self, state):
        self.observed = copy.deepcopy(state["observed"])
