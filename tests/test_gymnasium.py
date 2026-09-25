"""RL episode semantics and restoration of synthetic and optional real Gymnasium environments."""

from __future__ import annotations

import copy
import importlib.util
import unittest

import numpy as np

from experiments_wo_stress.protocols import RLProtocol
from experiments_wo_stress.settings import GymnasiumAdapter
from tests.sample_components import RecordingLearner


class _Environment:
    """Minimal stochastic Gymnasium-style environment with explicit episode endings."""

    def __init__(self, episode_steps=3, truncate=False):
        self.episode_steps = episode_steps
        self.truncate = truncate
        self.np_random = np.random.Generator(np.random.PCG64(0))
        self.value = 0.0
        self.elapsed = 0
        self.closed = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.Generator(np.random.PCG64(seed))
        self.value = float(self.np_random.normal())
        self.elapsed = 0
        return np.array([self.value]), {}

    def step(self, action):
        self.value += action + self.np_random.normal()
        self.elapsed += 1
        ended = self.elapsed >= self.episode_steps
        return (
            np.array([self.value]),
            self.value,
            ended and not self.truncate,
            ended and self.truncate,
            {},
        )

    def close(self):
        self.closed = True


class _EnvironmentState:
    """Snapshot both evolving environment values and its private random generator."""

    def snapshot(self, environment):
        return {
            "value": environment.value,
            "elapsed": environment.elapsed,
            "rng": environment.np_random.bit_generator.state,
        }

    def restore(self, environment, state):
        environment.value = state["value"]
        environment.elapsed = state["elapsed"]
        environment.np_random.bit_generator.state = state["rng"]


def _environment_factory(**params):
    """Construct an importable test environment without the optional Gymnasium package."""
    return _Environment(**params)


class _CartPoleState:
    """Explicit test adapter for the default unrendered CartPole wrapper stack."""

    _WRAPPER_FIELDS = {
        "TimeLimit": ("_elapsed_steps",),
        "OrderEnforcing": ("_has_reset",),
        "PassiveEnvChecker": ("checked_reset", "checked_step", "checked_render", "close_called"),
    }

    def snapshot(self, environment):
        base = environment.unwrapped
        wrappers = []
        current = environment
        while current is not base:
            fields = self._WRAPPER_FIELDS[type(current).__name__]
            wrappers.append({name: copy.deepcopy(getattr(current, name)) for name in fields})
            current = current.env
        return {
            "state": np.array(base.state, copy=True),
            "steps_beyond_terminated": base.steps_beyond_terminated,
            "rng": copy.deepcopy(base.np_random.bit_generator.state),
            "rng_seed": base.np_random_seed,
            "wrappers": wrappers,
        }

    def restore(self, environment, state):
        base = environment.unwrapped
        base.state = np.array(state["state"], copy=True)
        base.steps_beyond_terminated = state["steps_beyond_terminated"]
        base._np_random = np.random.Generator(np.random.PCG64())
        base._np_random.bit_generator.state = state["rng"]
        base._np_random_seed = state["rng_seed"]
        current = environment
        for wrapper in state["wrappers"]:
            for name, value in wrapper.items():
                setattr(current, name, value)
            current = current.env


class GymnasiumAdapterTests(unittest.TestCase):
    """Check episode boundaries and exact restoration through explicit environment adapters."""

    def build_rl_run(self, *, horizon=8, truncate=False, instance=None):
        """Create fresh learner, environment, and protocol with stable independent seeds."""
        options = {
            "factory": f"{__name__}:_environment_factory",
            "state_adapter": f"{__name__}:_EnvironmentState",
            "env_params": {"truncate": truncate},
        }
        instance = instance or GymnasiumAdapter.create_instance(
            rng=np.random.default_rng(7), **options
        )
        data = GymnasiumAdapter(rng=np.random.default_rng(8), instance=instance, **options)
        learner = RecordingLearner()
        protocol = RLProtocol(rng=np.random.default_rng(9), horizon=horizon)
        return learner, data, protocol

    def test_rl_distinguishes_termination_and_truncation_and_resets(self):
        """Episode endings preserve their cause and reset observations before the next step."""
        for truncate in (False, True):
            learner, data, protocol = self.build_rl_run(truncate=truncate)
            protocol.initialize(learner, data)
            rows = [protocol.advance(learner, data) for _ in range(4)]
            self.assertEqual(rows[2]["terminated"], not truncate)
            self.assertEqual(rows[2]["truncated"], truncate)
            self.assertEqual(rows[3]["episode"], 1)
            self.assertIn("next_observation", learner.observed[0])
            self.assertNotEqual(rows[2]["next_observation"][0], rows[3]["observation"][0])
            data.close()
            self.assertTrue(data.env.closed)

    def test_rl_environment_and_protocol_restore_exactly_across_episodes(self):
        """Restoring learner, environment, and protocol reproduces every later recorded field."""
        learner, data, protocol = self.build_rl_run()
        protocol.initialize(learner, data)
        for _ in range(4):
            protocol.advance(learner, data)
        state = (learner.state_dict(), data.state_dict(), protocol.state_dict())
        expected = [protocol.advance(learner, data) for _ in range(4)]
        restored_learner, restored_data, restored_protocol = self.build_rl_run(
            instance=data.instance
        )
        for component, saved in zip((restored_learner, restored_data, restored_protocol), state):
            component.load_state_dict(saved)
        actual = [restored_protocol.advance(restored_learner, restored_data) for _ in range(4)]
        for left, right in zip(expected, actual):
            self.assertEqual(set(left), set(right))
            for field in left:
                np.testing.assert_array_equal(left[field], right[field])

    def test_adapter_requires_reset_and_explicit_snapshot_capability(self):
        """The adapter requires reset before stepping and an explicit state adapter."""
        learner, data, protocol = self.build_rl_run()
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            data.generate(0)
        with self.assertRaises(TypeError):
            GymnasiumAdapter(
                rng=np.random.default_rng(0), factory=f"{__name__}:_environment_factory"
            )

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional Gymnasium dependency")
    def test_real_cartpole_resumes_across_time_limit_and_rng_resets(self):
        """An actual CartPole wrapper stack resumes exactly across episode boundaries."""
        options = {
            "state_adapter": f"{__name__}:_CartPoleState",
            "env_params": {"id": "CartPole-v1", "max_episode_steps": 7},
        }
        instance = GymnasiumAdapter.create_instance(rng=np.random.default_rng(27), **options)
        data = GymnasiumAdapter(rng=np.random.default_rng(28), instance=instance, **options)
        self.addCleanup(data.close)
        learner = RecordingLearner()
        protocol = RLProtocol(rng=np.random.default_rng(29), horizon=25)
        protocol.initialize(learner, data)
        for _ in range(8):
            protocol.advance(learner, data)
        saved = (learner.state_dict(), data.state_dict(), protocol.state_dict())
        expected = [protocol.advance(learner, data) for _ in range(17)]

        restored_data = GymnasiumAdapter(
            rng=np.random.default_rng(28), instance=instance, **options
        )
        self.addCleanup(restored_data.close)
        restored_learner = RecordingLearner()
        restored_protocol = RLProtocol(rng=np.random.default_rng(29), horizon=25)
        for component, state in zip((restored_learner, restored_data, restored_protocol), saved):
            component.load_state_dict(state)
        actual = [restored_protocol.advance(restored_learner, restored_data) for _ in range(17)]
        self.assertTrue(any(row["truncated"] for row in expected))
        self.assertGreater(actual[-1]["episode"], 1)
        for left, right in zip(expected, actual):
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
