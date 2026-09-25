"""Saved scientific instances and completed results are immutable numerical artifacts."""

from __future__ import annotations

import pickle
import unittest

import numpy as np

from experiments_wo_stress.artifacts import Instance, RunResult
from experiments_wo_stress.components import create_instance
from experiments_wo_stress.jobs import ComponentSpec, make_run_spec
from experiments_wo_stress.rng import make_rngs


class ArtifactModelTests(unittest.TestCase):
    """Keep instance inputs immutable and preserve the completed-result mapping contract."""

    def test_instance_copies_inputs_and_is_deeply_immutable(self):
        """Saved instances copy mutable inputs and retain immutable values through pickling."""
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
        """Instances reject Python objects, nonfinite metadata, and arrays in metadata."""
        for arguments in (
            {"arrays": {"bad": np.array([object()], dtype=object)}},
            {"metadata": {"bad": float("nan")}},
            {"metadata": {"bad": np.arange(3)}},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(TypeError):
                Instance(**arguments)

    def test_run_result_retains_mapping_compatibility_and_instance(self):
        """A completed result exposes immutable records alongside its instance and revision."""
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
