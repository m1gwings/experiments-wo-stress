"""Immutable instances, budget prefixes, and explicit environment restoration."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import logging
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments_wo_stress.artifacts import Instance, RunResult
from experiments_wo_stress.components import StateMixin, construct, create_instance
from experiments_wo_stress.data import CSVDataGenerator, NormalDataGenerator
from experiments_wo_stress.jobs import ComponentSpec, make_run_spec
from experiments_wo_stress.protocols import OnlineProtocol, RLProtocol
from experiments_wo_stress.rng import make_rngs
from experiments_wo_stress.settings import (
    GaussianBandit,
    GymnasiumAdapter,
    NonstationaryBandit,
    StationaryBandit,
)


class _Learner:
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


class _AwareComponent(StateMixin):
    def __init__(self, *, rng, instance, logger):
        self.rng, self.instance, self.logger = rng, instance, logger


class _ForwardingComponent(_Learner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _Environment:
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


class ArtifactTests(unittest.TestCase):
    def test_instance_copies_inputs_and_is_deeply_immutable(self):
        source = np.array([0.2, 0.8])
        metadata = {"nested": {"labels": ["a", "b"]}}
        instance = Instance(metadata=metadata, arrays={"means": source}, kind="bandit")
        source[0] = 9
        metadata["nested"]["labels"][0] = "changed"
        self.assertEqual(instance.metadata["nested"]["labels"], ("a", "b"))
        np.testing.assert_array_equal(instance.arrays["means"], [0.2, 0.8])
        with self.assertRaises(TypeError):
            instance.metadata["nested"]["other"] = 1
        with self.assertRaises(ValueError):
            instance.arrays["means"][0] = 1
        with self.assertRaises(ValueError):
            instance.arrays["means"].flags.writeable = True
        restored = pickle.loads(pickle.dumps(instance))
        np.testing.assert_array_equal(restored.arrays["means"], instance.arrays["means"])
        self.assertEqual(restored.to_dict()["metadata"], instance.to_dict()["metadata"])

    def test_instance_rejects_object_arrays_and_unstructured_metadata(self):
        for arguments in (
            {"arrays": {"bad": np.array([object()], dtype=object)}},
            {"metadata": {"bad": float("nan")}},
            {"metadata": {"bad": np.arange(3)}},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(TypeError):
                Instance(**arguments)

    def test_run_result_retains_mapping_compatibility_and_instance(self):
        spec = make_run_spec(
            group="test",
            repetition=0,
            algorithm_name="learner",
            algorithm=ComponentSpec("null_algorithm"),
            data=ComponentSpec("gaussian_bandit", {"means": [0.2, 0.8]}),
            protocol=ComponentSpec("online"),
            seed=4,
            budget_steps=2,
        )
        instance = create_instance(spec.data, make_rngs(spec)["instance"])
        result = RunResult(
            {"step": np.array([1, 2]), "action": np.array([0, 1])},
            instance,
            spec,
            2,
            "revision",
            {"estimate": np.array([0.1, 0.7])},
        )
        np.testing.assert_array_equal(result["action"], [0, 1])
        self.assertEqual(set(result), {"step", "action"})
        self.assertIs(result.instance, instance)
        self.assertEqual(result.revision, "revision")
        self.assertFalse(result["action"].flags.writeable)

    def test_optional_injection_preserves_legacy_forwarding_constructors(self):
        logger = logging.getLogger("test.component")
        rng = np.random.default_rng(1)
        instance = Instance(metadata={"value": 7})
        aware = construct(
            ComponentSpec(f"{__name__}:_AwareComponent"), rng, instance=instance, logger=logger
        )
        self.assertIs(aware.instance, instance)
        self.assertIs(aware.logger, logger)
        legacy = construct(ComponentSpec(f"{__name__}:_ForwardingComponent"), rng, logger=logger)
        self.assertIs(legacy.rng, rng)
        self.assertIs(legacy.logger, logger)
        descriptor = create_instance(ComponentSpec("null"), rng)
        self.assertEqual(descriptor.metadata["data"]["type"], "null")

    def test_csv_instance_retains_source_data_and_digest_without_rereading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            payload = b"x,y\n1,2\n3,4\n"
            path.write_bytes(payload)
            instance = CSVDataGenerator.create_instance(
                rng=np.random.default_rng(0), path=path, skip_header=1
            )
            self.assertEqual(instance.kind, "dataset")
            self.assertEqual(instance.metadata["sha256"], hashlib.sha256(payload).hexdigest())
            path.unlink()
            source = CSVDataGenerator(rng=np.random.default_rng(0), instance=instance)
            np.testing.assert_array_equal(source.generate(None), [[1, 2], [3, 4]])
            self.assertIs(source.values, instance.arrays["data"])

    def test_normal_instance_describes_distribution_and_keeps_sampling_separate(self):
        instance_rng = np.random.default_rng(0)
        before = copy.deepcopy(instance_rng.bit_generator.state)
        instance = NormalDataGenerator.create_instance(
            rng=instance_rng, size=[2, 3], loc=4, scale=0
        )
        self.assertEqual(instance_rng.bit_generator.state, before)
        self.assertEqual(instance.kind, "normal_distribution")
        self.assertEqual(instance.metadata["size"], (2, 3))
        generator = NormalDataGenerator(rng=np.random.default_rng(1), instance=instance)
        np.testing.assert_array_equal(generator.generate(None), np.full((2, 3), 4))


class BanditTests(unittest.TestCase):
    def test_gaussian_instance_is_authoritative_and_does_not_compute_regret(self):
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
        generator = StationaryBandit(rng=np.random.default_rng(3), means=[0, 1])
        self.assertEqual(generator.generate(0).value, 0)
        self.assertEqual(generator.generate(1).value, 1)
        for action in (-1, 2, True, 0.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                generator.generate(action)

    def test_nonstationary_schedule_and_restored_cursor(self):
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
        def spec(budget):
            return make_run_spec(
                group="study",
                repetition=0,
                algorithm_name="one",
                algorithm=ComponentSpec(f"{__name__}:_Learner"),
                data=ComponentSpec("gaussian_bandit", {"means": [0.1, 0.9]}),
                protocol=ComponentSpec("online"),
                seed=93,
                budget_steps=budget,
            )

        def build(run):
            streams = make_rngs(run)
            instance = create_instance(run.data, streams["instance"])
            data = construct(run.data, streams["data"], instance=instance)
            learner = construct(run.algorithm, streams["algorithm"])
            protocol = OnlineProtocol(rng=streams["protocol"])
            protocol.set_budget(run.budget_steps)
            protocol.initialize(learner, data)
            return streams, learner, data, protocol

        full_streams, full_learner, full_data, full_protocol = build(spec(8))
        expected = [full_protocol.advance(full_learner, full_data)["reward"] for _ in range(8)]
        streams, learner, data, protocol = build(spec(3))
        prefix = [protocol.advance(learner, data)["reward"] for _ in range(3)]
        state = (learner.state_dict(), data.state_dict(), protocol.state_dict())
        rng_state = {name: copy.deepcopy(rng.bit_generator.state) for name, rng in streams.items()}
        streams, learner, data, protocol = build(spec(8))
        for component, saved in zip((learner, data, protocol), state):
            component.load_state_dict(saved)
        for name, rng in streams.items():
            rng.bit_generator.state = rng_state[name]
        suffix = [protocol.advance(learner, data)["reward"] for _ in range(5)]
        np.testing.assert_array_equal(prefix + suffix, expected)


class GymnasiumTests(unittest.TestCase):
    def build(self, *, horizon=8, truncate=False, instance=None):
        options = {
            "factory": f"{__name__}:_environment_factory",
            "state_adapter": f"{__name__}:_EnvironmentState",
            "env_params": {"truncate": truncate},
        }
        instance = instance or GymnasiumAdapter.create_instance(
            rng=np.random.default_rng(7), **options
        )
        data = GymnasiumAdapter(rng=np.random.default_rng(8), instance=instance, **options)
        learner = _Learner()
        protocol = RLProtocol(rng=np.random.default_rng(9), horizon=horizon)
        return learner, data, protocol

    def test_rl_distinguishes_termination_and_truncation_and_resets(self):
        for truncate in (False, True):
            learner, data, protocol = self.build(truncate=truncate)
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
        learner, data, protocol = self.build()
        protocol.initialize(learner, data)
        for _ in range(4):
            protocol.advance(learner, data)
        state = (learner.state_dict(), data.state_dict(), protocol.state_dict())
        expected = [protocol.advance(learner, data) for _ in range(4)]
        restored_learner, restored_data, restored_protocol = self.build(instance=data.instance)
        for component, saved in zip((restored_learner, restored_data, restored_protocol), state):
            component.load_state_dict(saved)
        actual = [restored_protocol.advance(restored_learner, restored_data) for _ in range(4)]
        for left, right in zip(expected, actual):
            self.assertEqual(set(left), set(right))
            for field in left:
                np.testing.assert_array_equal(left[field], right[field])

    def test_adapter_requires_reset_and_explicit_snapshot_capability(self):
        learner, data, protocol = self.build()
        with self.assertRaisesRegex(RuntimeError, "Reset"):
            data.generate(0)
        with self.assertRaises(TypeError):
            GymnasiumAdapter(
                rng=np.random.default_rng(0), factory=f"{__name__}:_environment_factory"
            )

    @unittest.skipUnless(importlib.util.find_spec("gymnasium"), "optional Gymnasium dependency")
    def test_real_cartpole_resumes_across_time_limit_and_rng_resets(self):
        options = {
            "state_adapter": f"{__name__}:_CartPoleState",
            "env_params": {"id": "CartPole-v1", "max_episode_steps": 7},
        }
        instance = GymnasiumAdapter.create_instance(rng=np.random.default_rng(27), **options)
        data = GymnasiumAdapter(rng=np.random.default_rng(28), instance=instance, **options)
        self.addCleanup(data.close)
        learner = _Learner()
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
        restored_learner = _Learner()
        restored_protocol = RLProtocol(rng=np.random.default_rng(29), horizon=25)
        for component, state in zip((restored_learner, restored_data, restored_protocol), saved):
            component.load_state_dict(state)
        actual = [restored_protocol.advance(restored_learner, restored_data) for _ in range(17)]
        self.assertTrue(any(row["truncated"] for row in expected))
        self.assertGreater(actual[-1]["episode"], 1)
        for left, right in zip(expected, actual):
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])


if __name__ == "__main__":
    unittest.main()
