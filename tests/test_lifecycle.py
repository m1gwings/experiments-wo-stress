"""Check what happens when a completed experiment is reused, extended, or cleaned.

The local learner records enough state and log output to expose these lifecycle
boundaries. Notification integration also lives here; its transport is always
mocked, with detailed delivery tests in ``test_notifications.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments_wo_stress import clean_experiment, load_config, plan_runs, run_experiment
from experiments_wo_stress.storage import (
    RunStore,
    StorageError,
    iter_completed_runs,
)

COMPONENTS = '''"""Minimal paper code for lifecycle integration scenarios."""

class Learner:
    """Choose random arms and persist the observation count used by assertions."""

    supports_extension = True

    def __init__(self, *, rng, logger):
        self.rng = rng
        self.logger = logger
        self.n = 0

    def act(self, context=None):
        return int(self.rng.integers(context["n_arms"]))

    def observe(self, action, feedback):
        self.n += 1
        self.logger.info("observed %s", self.n)

    def state_dict(self):
        return {"n": self.n}

    def load_state_dict(self, state):
        self.n = state["n"]


class FixedLearner(Learner):
    """Use the same behavior while explicitly refusing completed-run extension."""

    supports_extension = False
'''


class _LifecycleStudyTestCase(unittest.TestCase):
    """Build one short study whose budget, dependencies, and settings can change."""

    def setUp(self) -> None:
        """Create private paper code and a seven-step study for each scenario."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module = f"lifecycle_{self._testMethodName}"
        (self.root / f"{self.module}.py").write_text(COMPONENTS, encoding="utf-8")
        self.output = self.root / "output"
        self.settings = {
            "name": "lifecycle",
            "seed": 10,
            "runs": [
                {
                    "name": "main",
                    "repetitions": 1,
                    "budget": {"steps": 7},
                    "protocol": {"type": "online"},
                    "data": {"type": "gaussian_bandit", "params": {"means": [0.2, 0.8]}},
                    "algorithms": [{"name": "learner", "type": f"{self.module}:Learner"}],
                }
            ],
            "execution": {"checkpoint_steps": 2, "log_max_bytes": 256, "log_backups": 1},
        }

    def load_study_config(self):
        """Reload the current settings so each mutation takes effect through YAML."""
        path = self.root / "experiment.yml"
        path.write_text(yaml.safe_dump(self.settings), encoding="utf-8")
        return load_config(path)

    def run_successfully(self):
        """Execute or reuse the configured study and expose any worker failure."""
        report = run_experiment(self.load_study_config(), self.output)
        self.assertFalse(report.failed, report.errors)
        return report


class NotificationExecutionTests(_LifecycleStudyTestCase):
    """Coordinator notifications preserve scientific results and keep credentials private."""

    def test_notification_changes_preserve_results_and_reuse(self):
        """Delivery stays in the coordinator and cannot affect values or persisted secrets."""
        self.settings["runs"][0]["repetitions"] = 2
        original = self.load_study_config()
        self.assertEqual(run_experiment(original, self.output).completed, 2)
        expected = {spec.run_id: result for spec, result in iter_completed_runs(self.output)}
        self.settings["notifications"] = {"discord": {"webhook_env": "EWS_TEST_DISCORD"}}
        enabled = self.load_study_config()
        secret = "https://discord.com/api/webhooks/123/test-secret-not-real"
        observed = []

        def sent(notifier, message):
            observed.append((os.getpid(), message))

        with (
            patch.dict(os.environ, {"EWS_TEST_DISCORD": secret}),
            patch("experiments_wo_stress.execution.notifications.ExperimentNotifier._send", sent),
        ):
            # The new operational setting reuses both existing scientific runs.
            reused = run_experiment(enabled, self.output, workers=2)
            self.assertEqual((reused.skipped, reused.failed), (2, 0), reused.errors)
            self.assertEqual(len(list((self.output / "runs").iterdir())), 2)
            fresh = self.root / "notified-fresh"
            report = run_experiment(enabled, fresh, workers=2)
            self.assertEqual((report.completed, report.failed), (2, 0), report.errors)
        actual = {spec.run_id: result for spec, result in iter_completed_runs(fresh)}
        # Compare a fresh notified execution with the original unnotified results.
        for run_id, result in expected.items():
            for field in result:
                np.testing.assert_array_equal(result[field], actual[run_id][field])
        self.assertEqual({pid for pid, _ in observed}, {os.getpid()})
        self.assertTrue(any("reused 2" in text for _, text in observed))
        self.assertTrue(any("completed 2" in text for _, text in observed))
        for path in fresh.rglob("*"):
            if path.suffix in {".json", ".yml", ".log"}:
                self.assertNotIn(secret, path.read_text())
        self.assertEqual(original.simulation_dict(), enabled.simulation_dict())

    def test_notification_secret_is_only_required_for_enabled_execution(self):
        self.settings["notifications"] = {"discord": {"webhook_env": "EWS_MISSING_TEST_SECRET"}}
        config = self.load_study_config()
        self.assertEqual(len(plan_runs(config)), 1)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "environment variable is missing"):
                run_experiment(config, self.output)
        self.assertFalse(self.output.exists())
        self.settings["notifications"]["discord"]["enabled"] = False
        self.assertEqual(self.run_successfully().completed, 1)


class ExperimentCleanupTests(_LifecycleStudyTestCase):
    """Cleanup requires ownership and preserves the artifacts promised by each scope."""

    def test_cleanup_preview_and_reclaim_inactive_variants(self):
        """Preview is read-only; applying it removes only the inactive scientific variant."""
        self.run_successfully()
        self.settings["runs"][0]["data"]["params"]["means"] = [0.4, 0.6]
        self.run_successfully()
        preview = clean_experiment(self.output)
        self.assertTrue(preview["dry_run"])
        self.assertGreater(preview["bytes"], 0)
        self.assertEqual(len(list((self.output / "runs").iterdir())), 2)
        applied = clean_experiment(self.output, yes=True)
        self.assertEqual(preview["paths"], applied["paths"])
        self.assertEqual(len(list((self.output / "runs").iterdir())), 1)
        self.assertEqual(len(list(iter_completed_runs(self.output))), 1)
        self.assertEqual(self.run_successfully().skipped, 1)

    def test_cleanup_allows_full_reset_but_never_unowned_directory(self):
        with self.assertRaises(StorageError):
            clean_experiment(self.root, scope="all", yes=True)
        self.run_successfully()
        clean_experiment(self.output, scope="all", yes=True)
        self.assertEqual(self.run_successfully().completed, 1)

    def test_clearing_checkpoints_keeps_analysis_and_reports_unavailable_extension(self):
        """Recorded results survive cleanup, while extending without final state fails."""
        self.run_successfully()
        clean_experiment(self.output, scope="checkpoints", yes=True)
        self.assertEqual(len(list(iter_completed_runs(self.output))), 1)
        self.assertEqual(self.run_successfully().skipped, 1)
        self.settings["runs"][0]["budget"]["steps"] = 10
        report = run_experiment(self.load_study_config(), self.output)
        self.assertEqual(report.failed, 1)
        self.assertIn("completed prefix", next(iter(report.errors.values())))
        self.settings["runs"][0]["budget"]["steps"] = 7
        self.assertEqual(self.run_successfully().skipped, 1)


class RunContinuationTests(_LifecycleStudyTestCase):
    """Budget extension respects component support and the original recording schedule."""

    def test_nonextendable_component_selects_distinct_budget_variant(self):
        self.settings["runs"][0]["algorithms"][0]["type"] = f"{self.module}:FixedLearner"
        self.run_successfully()
        self.settings["runs"][0]["budget"]["steps"] = 10
        self.assertEqual(self.run_successfully().completed, 1)
        self.assertEqual(len(list((self.output / "runs").iterdir())), 2)

    def test_sparse_extension_matches_uninterrupted_schedule(self):
        """Extension continues the original step schedule across the old budget boundary."""
        self.settings["recording"] = {"every_steps": 3}
        self.run_successfully()
        self.settings["runs"][0]["budget"]["steps"] = 14
        self.run_successfully()
        extended = next(iter_completed_runs(self.output))[1]
        report = run_experiment(self.load_study_config(), self.root / "direct")
        self.assertEqual(report.completed, 1, report.errors)
        direct = next(iter_completed_runs(self.root / "direct"))[1]
        for key in direct:
            np.testing.assert_array_equal(extended[key], direct[key])
        np.testing.assert_array_equal(extended["step"], [3, 6, 9, 12])


class ExecutionArtifactTests(_LifecycleStudyTestCase):
    """Completed runs retain final state, instance references, and bounded component logs."""

    def test_final_state_and_bounded_component_logs_are_saved(self):
        self.run_successfully()
        directory = next((self.output / "runs").iterdir())
        state, _, step = RunStore(directory).restore()
        self.assertEqual((step, state["algorithm"]["n"]), (7, 7))
        logs = list(directory.glob("run.log*"))
        self.assertLessEqual(len(logs), 2)
        text = "".join(path.read_text() for path in logs)
        self.assertIn("observed", text)
        self.assertIn(".algorithm", text)
        metadata = json.loads((directory / "metadata.json").read_text())
        self.assertTrue(metadata["instance_id"])


class TrackedSourceTests(_LifecycleStudyTestCase):
    """Only declared simulation dependencies affect scientific reuse."""

    def test_separate_metric_source_edits_do_not_invalidate_simulation(self):
        """Metric-only edits matter only after the file is declared a scientific dependency."""
        metric = self.root / "metrics_only.py"
        metric.write_text("# first metric implementation\n")
        self.run_successfully()
        metric.write_text("# revised metric implementation\n")
        self.assertEqual(self.run_successfully().skipped, 1)
        self.settings["runs"][0]["algorithms"][0]["dependencies"] = ["metrics_only.py"]
        self.assertEqual(self.run_successfully().completed, 1)
        metric.write_text("# explicitly tracked scientific dependency\n")
        self.assertEqual(self.run_successfully().completed, 1)
