"""General metric result shapes, cumulative trajectories, and explicit input requirements."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from experiments_wo_stress.metrics import (
    CumulativeSumMetric,
    FieldMetric,
    MetricResult,
    validate_requirements,
)
from tests.sample_results import make_saved_result


class MetricContractTests(unittest.TestCase):
    """Validate numerical metric results and their declared requirements on saved inputs."""

    def test_scalar_and_cumulative_metrics(self) -> None:
        """Scalar outputs normalize to one point and cumulative metrics sum each increment."""
        scalar = MetricResult(np.asarray(1), np.asarray(4.5))
        self.assertEqual(scalar.values.shape, (1,))
        result = CumulativeSumMetric("increment").compute(
            {
                "step": np.array([1, 2, 3]),
                "increment": np.array([1.0, 0.5, 2.0]),
            }
        )
        np.testing.assert_allclose(result.values, [1, 1.5, 3.5])

    def test_rejects_nonnumerical_misaligned_and_nonfinite_results(self) -> None:
        """Metric outputs must contain aligned, finite numerical coordinates and values."""
        with self.assertRaises(TypeError):
            MetricResult(np.array([1]), np.array(["bad"]))
        with self.assertRaises(ValueError):
            MetricResult(np.array([1, 2]), np.array([1]))
        with self.assertRaises(ValueError):
            MetricResult(np.array([1]), np.array([np.nan]))

    def test_sparse_truncated_and_missing_measurements_fail_explicitly(self):
        """Metrics report incomplete trajectories and missing recorded fields."""
        for steps, completed in (([1, 3], 3), ([1, 2], 3), ([2, 3], 3)):
            saved = make_saved_result(
                records={"step": np.array(steps), "reward": np.ones(len(steps))},
                completed_steps=completed,
            )
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(ValueError, "complete trajectory"):
                    CumulativeSumMetric("reward").compute(saved)
        with self.assertRaisesRegex(ValueError, "missing recorded fields: absent"):
            FieldMetric("absent").compute(make_saved_result())

    def test_capability_declarations_apply_to_custom_metrics(self):
        """Custom metric requirements are checked against saved instance arrays too."""
        metric = SimpleNamespace(required_fields=("reward",), required_instance_fields=("context",))
        with self.assertRaisesRegex(ValueError, "instance arrays: context"):
            validate_requirements(metric, make_saved_result())
