"""Check retained variants, storage reclamation, and extension failure boundaries."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments_wo_stress import Instance, clean_experiment, load_config, run_experiment
from experiments_wo_stress.storage import (
    RunStore,
    StorageError,
    iter_completed_runs,
    load_instance,
    save_instance,
)

COMPONENTS = """
import numpy as np
class Learner:
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
    supports_extension = False
"""


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module = f"lifecycle_{self._testMethodName}"
        (self.root / f"{self.module}.py").write_text(COMPONENTS)
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

    def config(self):
        path = self.root / "experiment.yml"
        path.write_text(yaml.safe_dump(self.settings))
        return load_config(path)

    def run_ok(self):
        report = run_experiment(self.config(), self.output)
        self.assertFalse(report.failed, report.errors)
        return report

    def test_notification_changes_preserve_results_and_reuse(self):
        self.settings["runs"][0]["repetitions"] = 2
        original = self.config()
        self.assertEqual(run_experiment(original, self.output).completed, 2)
        expected = {spec.run_id: result for spec, result in iter_completed_runs(self.output)}
        self.settings["notifications"] = {"discord": {"webhook_env": "EWS_TEST_DISCORD"}}
        enabled = self.config()
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
        config = self.config()
        from experiments_wo_stress import plan_runs

        self.assertEqual(len(plan_runs(config)), 1)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "environment variable is missing"):
                run_experiment(config, self.output)
        self.assertFalse(self.output.exists())
        self.settings["notifications"]["discord"]["enabled"] = False
        self.assertEqual(self.run_ok().completed, 1)

    def test_instance_deduplicates_and_loads_without_simulator(self):
        instance = Instance(metadata={"labels": ["a", "b"]}, arrays={"means": np.array([0.1, 0.9])})
        key = save_instance(self.root, instance, compression=True)
        self.assertEqual(save_instance(self.root, instance), key)
        restored = load_instance(self.root, key)
        np.testing.assert_array_equal(restored.arrays["means"], instance.arrays["means"])
        self.assertFalse(restored.arrays["means"].flags.writeable)
        self.assertEqual(len(list((self.root / "instances").iterdir())), 1)

    def test_instance_metadata_corruption_is_rejected(self):
        key = save_instance(
            self.root, Instance(metadata={"truth": 1}, arrays={"empty": np.empty((0, 2))})
        )
        path = self.root / "instances" / key / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["metadata"]["truth"] = 2
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(StorageError, "identity"):
            load_instance(self.root, key)

    def test_cleanup_preview_and_reclaim_inactive_variants(self):
        self.run_ok()
        self.settings["runs"][0]["data"]["params"]["means"] = [0.4, 0.6]
        self.run_ok()
        preview = clean_experiment(self.output)
        self.assertTrue(preview["dry_run"])
        self.assertGreater(preview["bytes"], 0)
        self.assertEqual(len(list((self.output / "runs").iterdir())), 2)
        applied = clean_experiment(self.output, yes=True)
        self.assertEqual(preview["paths"], applied["paths"])
        self.assertEqual(len(list((self.output / "runs").iterdir())), 1)
        self.assertEqual(len(list(iter_completed_runs(self.output))), 1)
        self.assertEqual(self.run_ok().skipped, 1)

    def test_cleanup_allows_full_reset_but_never_unowned_directory(self):
        with self.assertRaises(StorageError):
            clean_experiment(self.root, scope="all", yes=True)
        self.run_ok()
        clean_experiment(self.output, scope="all", yes=True)
        self.assertEqual(self.run_ok().completed, 1)

    def test_clearing_checkpoints_keeps_analysis_and_reports_unavailable_extension(self):
        self.run_ok()
        clean_experiment(self.output, scope="checkpoints", yes=True)
        self.assertEqual(len(list(iter_completed_runs(self.output))), 1)
        self.assertEqual(self.run_ok().skipped, 1)
        self.settings["runs"][0]["budget"]["steps"] = 10
        report = run_experiment(self.config(), self.output)
        self.assertEqual(report.failed, 1)
        self.assertIn("completed prefix", next(iter(report.errors.values())))
        self.settings["runs"][0]["budget"]["steps"] = 7
        self.assertEqual(self.run_ok().skipped, 1)

    def test_nonextendable_component_selects_distinct_budget_variant(self):
        self.settings["runs"][0]["algorithms"][0]["type"] = f"{self.module}:FixedLearner"
        self.run_ok()
        self.settings["runs"][0]["budget"]["steps"] = 10
        self.assertEqual(self.run_ok().completed, 1)
        self.assertEqual(len(list((self.output / "runs").iterdir())), 2)

    def test_final_state_and_bounded_component_logs_are_saved(self):
        self.run_ok()
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

    def test_separate_metric_source_edits_do_not_invalidate_simulation(self):
        metric = self.root / "metrics_only.py"
        metric.write_text("# first metric implementation\n")
        self.run_ok()
        metric.write_text("# revised metric implementation\n")
        self.assertEqual(self.run_ok().skipped, 1)
        self.settings["runs"][0]["algorithms"][0]["dependencies"] = ["metrics_only.py"]
        self.assertEqual(self.run_ok().completed, 1)
        metric.write_text("# explicitly tracked scientific dependency\n")
        self.assertEqual(self.run_ok().completed, 1)

    def test_sparse_extension_matches_uninterrupted_schedule(self):
        self.settings["recording"] = {"every_steps": 3}
        self.run_ok()
        self.settings["runs"][0]["budget"]["steps"] = 14
        self.run_ok()
        extended = next(iter_completed_runs(self.output))[1]
        report = run_experiment(self.config(), self.root / "direct")
        self.assertEqual(report.completed, 1, report.errors)
        direct = next(iter_completed_runs(self.root / "direct"))[1]
        for key in direct:
            np.testing.assert_array_equal(extended[key], direct[key])
        np.testing.assert_array_equal(extended["step"], [3, 6, 9, 12])
