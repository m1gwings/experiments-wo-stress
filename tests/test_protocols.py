"""How built-in protocols hand data to algorithms and report complete steps."""

from __future__ import annotations

import unittest

import numpy as np

from experiments_wo_stress.components import Feedback, StateMixin
from experiments_wo_stress.data import NormalDataGenerator, NullDataGenerator
from experiments_wo_stress.protocols import OfflineProtocol, OnlineProtocol, TrialProtocol


def trial_function(*, algorithm, data, rng, count):
    """Return a sample using only the RNG explicitly injected into the trial."""
    return {"sample": rng.normal(size=count)}


class ObservingAlgorithm(StateMixin):
    """Record feedback received by the algorithm to expose the observation boundary."""

    def __init__(self):
        self.observed = []

    def act(self, context=None):
        return context["action"] if context else 2

    def observe(self, action, feedback):
        self.observed.append((action, feedback))


class MeasuredData(StateMixin):
    """Return learner feedback plus a hidden measurement intended only for recording."""

    def context(self):
        return {"action": 3}

    def generate(self, action):
        return Feedback(action + 0.5, {"reward": action + 0.5, "hidden": 100.0})


class DatasetSummaryAlgorithm(StateMixin):
    """Summarize the dataset passed by the offline protocol."""

    def fit(self, dataset):
        return {"mean": np.mean(dataset), "size": np.size(dataset)}


class ProtocolContractTests(unittest.TestCase):
    """Check online, offline, and trial interaction order and progress restoration."""

    def test_online_observation_boundary_and_restore(self):
        """Only feedback reaches the learner, while records and restored progress stay complete."""
        algorithm, data = ObservingAlgorithm(), MeasuredData()
        protocol = OnlineProtocol(rng=np.random.default_rng(0), horizon=3)
        protocol.initialize(algorithm, data)
        result = protocol.advance(algorithm, data)
        self.assertEqual(algorithm.observed, [(3, 3.5)])
        self.assertEqual(result, {"reward": 3.5, "hidden": 100.0, "action": 3})
        restored = OnlineProtocol(rng=np.random.default_rng(0), horizon=3)
        restored.load_state_dict(protocol.state_dict())
        self.assertEqual(restored.step, 1)
        self.assertFalse(restored.is_finished())
        restored.advance(algorithm, data)
        restored.advance(algorithm, data)
        self.assertTrue(restored.is_finished())
        with self.assertRaises(RuntimeError):
            restored.advance(algorithm, data)

    def test_offline_protocol_uses_shared_data_contract(self):
        """Offline fitting receives the generator dataset and finishes after one step."""
        data = NormalDataGenerator(rng=np.random.default_rng(3), size=10, loc=2, scale=0)
        algorithm = DatasetSummaryAlgorithm()
        protocol = OfflineProtocol(rng=np.random.default_rng(4))
        protocol.initialize(algorithm, data)
        self.assertEqual(protocol.advance(algorithm, data), {"mean": 2, "size": 10})
        self.assertTrue(protocol.is_finished())

    def test_trial_function_and_reserved_parameters(self):
        """Trials receive their injected RNG and reject parameters that replace injected objects."""
        protocol = TrialProtocol(
            rng=np.random.default_rng(7), function=f"{__name__}:trial_function", params={"count": 3}
        )
        empty = NullDataGenerator(rng=np.random.default_rng(8))
        protocol.initialize(empty, empty)
        result = protocol.advance(empty, empty)
        np.testing.assert_array_equal(result["sample"], np.random.default_rng(7).normal(size=3))
        with self.assertRaisesRegex(ValueError, "reserved names"):
            TrialProtocol(
                rng=np.random.default_rng(0),
                function=f"{__name__}:trial_function",
                params={"rng": 1},
            )
