"""Supplied bandit metrics use saved scientific instances and preserve regret semantics."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from experiments_wo_stress.artifacts import Instance
from experiments_wo_stress.builtins.metrics import PseudoRegretMetric, RealizedRegretMetric
from tests.sample_results import make_saved_result


class BanditMetricTests(unittest.TestCase):
    """Compute regret from saved means or counterfactual rewards with explicit comparators."""

    def test_pseudo_regret_requires_saved_means(self):
        """Pseudo-regret reports an explicit error when saved arm means are absent."""
        with self.assertRaisesRegex(ValueError, "missing instance arrays: means"):
            PseudoRegretMetric().compute(replace(make_saved_result(), instance=Instance()))

    def test_stationary_pseudo_regret_uses_means_not_realized_rewards(self):
        """Stationary pseudo-regret depends on the saved means even when rewards change."""
        saved = make_saved_result()
        computed = PseudoRegretMetric().compute(saved)
        np.testing.assert_allclose(computed.values, [0.6, 0.6, 1.2])
        changed = replace(saved, records={**saved.records, "reward": np.array([-100, 50, 8])})
        np.testing.assert_array_equal(PseudoRegretMetric().compute(changed).values, computed.values)

    def test_nonstationary_dynamic_and_fixed_benchmarks(self):
        """Dynamic and best-fixed comparators produce their distinct cumulative regret curves."""
        saved = make_saved_result(
            means=np.array([[1, 0], [0, 2], [3, 1]]),
            records={
                "step": np.array([1, 2, 3]),
                "action": np.array([0, 0, 1]),
            },
        )
        # The dynamic comparator chooses the best arm separately at each step.
        np.testing.assert_allclose(PseudoRegretMetric().compute(saved).values, [0, 2, 4])
        # The fixed comparator chooses the best single arm over each observed prefix.
        np.testing.assert_allclose(
            PseudoRegretMetric(comparator="best_fixed").compute(saved).values, [0, 1, 2]
        )

    def test_small_gaps_are_accumulated_before_large_reward_sums_cancel(self):
        """Accumulating per-step gaps preserves small differences between large means."""
        steps = 10000
        means = np.array([1e12, 1e12 + 0.01])
        saved = make_saved_result(
            means=means,
            completed_steps=steps,
            records={
                "step": np.arange(1, steps + 1),
                "action": np.zeros(steps, dtype=int),
            },
        )
        expected = np.cumsum(np.full(steps, means[1] - means[0]))
        for comparator in ("dynamic", "best_fixed"):
            np.testing.assert_array_equal(
                PseudoRegretMetric(comparator=comparator).compute(saved).values, expected
            )

    def test_regret_arithmetic_does_not_use_the_storage_dtype(self):
        """Boolean and small integer rewards retain signed gaps and full cumulative sums."""
        for dtype in (np.bool_, np.uint8, np.uint64, np.int8, np.float32):
            for stationary in (False, True):
                with self.subTest(dtype=dtype, stationary=stationary):
                    matrix = np.array([[1, 0], [0, 1], [1, 0]], dtype=dtype)
                    means = matrix[0] if stationary else matrix
                    actions = np.array([0, 1, 0])
                    saved = make_saved_result(
                        means=means,
                        records={"step": np.arange(1, 4), "action": actions},
                    )
                    for comparator, expected in (
                        ("dynamic", [0, 1, 1] if stationary else [0, 0, 0]),
                        ("best_fixed", [0, 1, 1] if stationary else [0, -1, -1]),
                    ):
                        np.testing.assert_array_equal(
                            PseudoRegretMetric(comparator=comparator).compute(saved).values,
                            expected,
                        )
                    realized = replace(
                        saved,
                        instance=Instance(arrays={"counterfactual_rewards": matrix}),
                        records={**saved.records, "reward": matrix[np.arange(3), actions]},
                    )
                    np.testing.assert_array_equal(
                        RealizedRegretMetric().compute(realized).values, [0, -1, -1]
                    )
        # Even signed integer subtraction can overflow before cumsum promotes it.
        saved = make_saved_result(
            means=np.array([[-100, 100]] * 3, dtype=np.int8),
            records={"step": np.arange(1, 4), "action": np.zeros(3, dtype=int)},
        )
        for comparator in ("dynamic", "best_fixed"):
            np.testing.assert_array_equal(
                PseudoRegretMetric(comparator=comparator).compute(saved).values, [200, 400, 600]
            )

    def test_realized_regret_requires_consistent_counterfactual_rewards(self):
        """Realized regret requires counterfactual rewards consistent with observed rewards."""
        with self.assertRaisesRegex(ValueError, "counterfactual_rewards"):
            RealizedRegretMetric().compute(make_saved_result())
        matrix = np.array([[1, 0], [0, 2], [3, 1]])
        saved = replace(
            make_saved_result(),
            instance=Instance(arrays={"counterfactual_rewards": matrix}),
            records={
                "step": np.array([1, 2, 3]),
                "action": np.array([0, 0, 1]),
                "reward": np.array([1, 0, 1]),
            },
        )
        np.testing.assert_allclose(RealizedRegretMetric().compute(saved).values, [0, 1, 2])
        inconsistent = replace(saved, records={**saved.records, "reward": np.array([9, 0, 1])})
        with self.assertRaisesRegex(ValueError, "disagree"):
            RealizedRegretMetric().compute(inconsistent)
