"""Stationary and nonstationary bandit instances, feedback, and exact budget continuation."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from experiments_wo_stress.components import construct, create_instance
from experiments_wo_stress.jobs import ComponentSpec, make_run_spec
from experiments_wo_stress.protocols import OnlineProtocol
from experiments_wo_stress.rng import make_rngs
from experiments_wo_stress.settings import (
    GaussianBandit,
    NonstationaryBandit,
    StationaryBandit,
)


class BanditGeneratorTests(unittest.TestCase):
    """Keep bandit feedback, immutable instances, and restored random prefixes consistent."""

    def test_gaussian_instance_is_authoritative_and_does_not_compute_regret(self):
        """Gaussian feedback uses the saved means and records only action and reward."""
        instance = GaussianBandit.create_instance(
            rng=np.random.default_rng(0), means=[0.2, 0.8], noise_std=0
        )
        generator = GaussianBandit(rng=np.random.default_rng(1), instance=instance)
        self.assertEqual(generator.context(), {"n_arms": 2})
        feedback = generator.generate(1)
        self.assertEqual(feedback.value, 0.8)
        self.assertEqual(feedback.measurements, {"action": 1, "reward": 0.8})
        self.assertEqual(generator.state_dict(), {"step": 1})
        np.testing.assert_array_equal(instance.arrays["means"], [0.2, 0.8])

    def test_stationary_bernoulli_and_invalid_actions(self):
        """Bernoulli endpoints give exact rewards and reject invalid arm selections."""
        generator = StationaryBandit(rng=np.random.default_rng(3), means=[0, 1])
        self.assertEqual(generator.generate(0).value, 0)
        self.assertEqual(generator.generate(1).value, 1)
        for action in (-1, 2, True, 0.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                generator.generate(action)

    def test_nonstationary_schedule_and_restored_cursor(self):
        """Restored bandits continue at the next scheduled mean and report exhaustion."""
        schedule = [[0, 1], [2, 3], [4, 5]]
        generator = NonstationaryBandit(
            rng=np.random.default_rng(0), schedule=schedule, noise_std=0
        )
        self.assertEqual(generator.generate(1).value, 1)
        restored = NonstationaryBandit(
            rng=np.random.default_rng(0), schedule=schedule, instance=generator.instance
        )
        restored.load_state_dict(generator.state_dict())
        self.assertEqual(restored.generate(0).value, 2)
        self.assertEqual(restored.generate(1).value, 5)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            restored.generate(0)

    def test_extending_budget_preserves_exact_random_prefix(self):
        """A checkpointed short run extended to eight steps matches eight uninterrupted steps."""

        def make_bandit_run(budget):
            """Vary only the requested budget; every scientific parameter stays fixed."""
            return make_run_spec(
                group="study",
                repetition=0,
                algorithm_name="one",
                algorithm=ComponentSpec("tests.sample_components:RecordingLearner"),
                data=ComponentSpec("gaussian_bandit", {"means": [0.1, 0.9]}),
                protocol=ComponentSpec("online"),
                seed=93,
                budget_steps=budget,
            )

        def initialize_run(run):
            """Construct fresh components with the run's four independent streams."""
            streams = make_rngs(run)
            instance = create_instance(run.data, streams["instance"])
            data = construct(run.data, streams["data"], instance=instance)
            learner = construct(run.algorithm, streams["algorithm"])
            protocol = OnlineProtocol(rng=streams["protocol"])
            protocol.set_budget(run.budget_steps)
            protocol.initialize(learner, data)
            return streams, learner, data, protocol

        # Establish the uninterrupted reference, then stop a fresh run at step three.
        _, full_learner, full_data, full_protocol = initialize_run(make_bandit_run(8))
        expected = [full_protocol.advance(full_learner, full_data)["reward"] for _ in range(8)]
        streams, learner, data, protocol = initialize_run(make_bandit_run(3))
        prefix = [protocol.advance(learner, data)["reward"] for _ in range(3)]
        # Capture component and RNG state at the same complete protocol step.
        state = (learner.state_dict(), data.state_dict(), protocol.state_dict())
        rng_state = {name: copy.deepcopy(rng.bit_generator.state) for name, rng in streams.items()}
        # Restore into fresh components with the larger budget before continuing.
        streams, learner, data, protocol = initialize_run(make_bandit_run(8))
        for component, saved in zip((learner, data, protocol), state):
            component.load_state_dict(saved)
        for name, rng in streams.items():
            rng.bit_generator.state = rng_state[name]
        suffix = [protocol.advance(learner, data)["reward"] for _ in range(5)]
        np.testing.assert_array_equal(prefix + suffix, expected)
