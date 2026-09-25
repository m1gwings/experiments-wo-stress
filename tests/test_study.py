"""YAML loading, run planning, scientific identity, and independent random streams."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from experiments_wo_stress.config import load_config
from experiments_wo_stress.jobs import GridPlanner, RunSpec, make_run_spec, plan_runs
from experiments_wo_stress.rng import make_rngs


class SubsetPlanner:
    """Example extension: retain the smallest data size and first repetition."""

    def plan(self, group, seed):
        for run in GridPlanner().plan(group, seed):
            if run.data.params["size"] == 2 and run.repetition == 0:
                yield run


class InvalidPlanner:
    """Emit one selected contract violation to exercise planner validation."""

    def plan(self, group, seed):
        run = next(iter(GridPlanner().plan(group, seed)))
        mode = group["data"]["params"]["invalid_mode"]
        if mode == "duplicate":
            yield run
            yield run
        elif mode == "identity":
            yield replace(run, run_id="invented")
        elif mode == "repetition":
            yield replace(run, repetition=-1)
        elif mode == "seed":
            yield replace(run, seed=seed + 1)
        elif mode == "group":
            yield replace(run, group="different_group")
        elif mode == "type":
            yield {"run_id": run.run_id}
        elif mode == "component":
            yield replace(run, data={"type": "null"})


class _StudyConfigurationCase(unittest.TestCase):
    """Provide a temporary YAML study; subclasses organize configuration and identity checks."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "study.yml"
        self.document = {
            "name": "study",
            "seed": 42,
            "runs": [
                {
                    "name": "main",
                    "repetitions": 3,
                    "protocol": {"type": "online", "params": {"horizon": 5}},
                    "data": {"type": "normal", "params": {"size": 2}},
                    "algorithms": [
                        {"name": "first", "type": "null_algorithm", "params": {"alpha": 1}},
                        {"name": "second", "type": "null_algorithm", "params": {"alpha": 2}},
                    ],
                    "grid": {"data.params.size": [2, 4]},
                }
            ],
        }

    def load_configuration(self, document=None):
        """Write and load a study relative to its temporary configuration directory."""
        self.path.write_text(yaml.safe_dump(document or self.document), encoding="utf-8")
        return load_config(self.path)


class ConfigurationLoadingTests(_StudyConfigurationCase):
    """Validate and normalize YAML without constructing scientific components."""

    def test_loading_custom_planner_does_not_import_it(self):
        """Configuration parsing can describe a planner whose Python module is unavailable."""
        self.document["runs"][0]["planner"] = "unavailable_components.planners:SubsetPlanner"
        config = self.load_configuration()
        self.assertEqual(config.runs[0]["planner"], "unavailable_components.planners:SubsetPlanner")

    def test_dependency_paths_and_logging_settings(self):
        """Relative tracked files resolve beside YAML and logging settings are validated."""
        self.document["runs"][0]["data"]["dependencies"] = ["helpers.py", "inputs/data.csv"]
        self.document["execution"] = {
            "logging_level": "debug",
            "log_max_bytes": 4096,
            "log_backups": 3,
        }
        config = self.load_configuration()
        dependencies = plan_runs(config)[0].data.dependencies
        self.assertEqual(
            dependencies,
            (str(self.path.parent / "helpers.py"), str(self.path.parent / "inputs/data.csv")),
        )
        self.assertEqual(config.execution["logging_level"], "DEBUG")
        self.assertEqual(config.execution["log_max_bytes"], 4096)
        for key, value in (("logging_level", "verbose"), ("log_max_bytes", 0), ("log_backups", -1)):
            self.document["execution"] = {key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load_configuration()

    def test_csv_path_resolution_in_parameters_and_grid(self):
        """CSV paths in direct parameters and grids resolve relative to the configuration."""
        self.document["runs"][0]["data"] = {"type": "csv", "params": {"path": "data.csv"}}
        self.document["runs"][0]["grid"] = {"data.params.path": ["a.csv", "b.csv"]}
        config = self.load_configuration()
        self.assertEqual(
            config.runs[0]["data"]["params"]["path"], str(self.path.parent / "data.csv")
        )
        self.assertEqual(plan_runs(config)[0].data.params["path"], str(self.path.parent / "a.csv"))

    def test_invalid_settings_are_rejected(self):
        """Invalid values and ambiguous grid definitions fail during configuration loading."""
        invalid_cases = [
            ("unknown study option", lambda value: value.update({"unknown": True})),
            ("boolean seed", lambda value: value.update({"seed": True})),
            ("zero workers", lambda value: value.update({"execution": {"workers": 0}})),
            (
                "negative checkpoint interval",
                lambda value: value.update({"execution": {"checkpoint_seconds": -1}}),
            ),
            (
                "zero recording interval",
                lambda value: value.update({"recording": {"every_steps": 0}}),
            ),
            (
                "duplicate recording fields",
                lambda value: value.update({"recording": {"fields": ["reward", "reward"]}}),
            ),
            ("zero repetitions", lambda value: value["runs"][0].update({"repetitions": 0})),
            (
                "duplicate grid values",
                lambda value: value["runs"][0].update({"grid": {"data.params.size": [2, 2]}}),
            ),
            (
                "grid path outside parameters",
                lambda value: value["runs"][0].update({"grid": {"data.size": [2]}}),
            ),
            (
                "overlapping grid paths",
                lambda value: value["runs"][0].update(
                    {"grid": {"data.params.x": [1], "data.params.x.y": [2]}}
                ),
            ),
            (
                "reserved injected parameter",
                lambda value: value["runs"][0]["data"].update({"params": {"rng": 123}}),
            ),
            (
                "nonfinite parameter",
                lambda value: value["runs"][0]["data"].update({"params": {"scale": float("nan")}}),
            ),
        ]
        for description, mutate in invalid_cases:
            document = copy.deepcopy(self.document)
            mutate(document)
            with self.subTest(case=description), self.assertRaises(ValueError):
                self.load_configuration(document)

    def test_duplicate_yaml_keys_are_rejected(self):
        """Duplicate YAML keys raise an error rather than silently replacing earlier values."""
        self.path.write_text("name: first\nname: second\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
            load_config(self.path)


class RunPlanningTests(_StudyConfigurationCase):
    """Expand or customize the run grid while retaining valid run descriptions."""

    def test_defaults_and_cartesian_grid(self):
        """Defaults and the grid produce one serializable run per parameter/repetition pair."""
        config = self.load_configuration()
        planned = plan_runs(config)
        self.assertEqual(len(planned), 12)
        self.assertEqual(len({run.run_id for run in planned}), 12)
        self.assertEqual(config.execution["checkpoint_seconds"], 120)
        self.assertEqual(config.recording["buffer_bytes"], 16 * 1024 * 1024)
        self.assertEqual(planned[0].labels["data.params.size"], 2)
        self.assertEqual(planned[0].labels["algorithm.name"], "first")
        self.assertEqual(RunSpec.from_dict(planned[0].to_dict()), planned[0])

    def test_custom_planner_selects_subset_with_unchanged_identities(self):
        """A custom planner may select runs without changing their scientific identities."""
        grid_runs = plan_runs(self.load_configuration())
        self.document["runs"][0]["planner"] = f"{__name__}:SubsetPlanner"
        config = self.load_configuration()
        selected = plan_runs(config)
        expected = [
            run for run in grid_runs if run.data.params["size"] == 2 and run.repetition == 0
        ]
        self.assertEqual(selected, expected)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            config.simulation_dict()["runs"],
            sorted([run.to_dict() for run in expected], key=lambda item: item["run_id"]),
        )

    def test_custom_planner_output_is_validated(self):
        """Malformed, duplicate, or inconsistent custom runs fail before execution."""
        self.document["runs"][0]["planner"] = f"{__name__}:InvalidPlanner"
        for mode in ("duplicate", "identity", "repetition", "seed", "group", "type", "component"):
            self.document["runs"][0]["data"]["params"]["invalid_mode"] = mode
            with self.subTest(mode=mode), self.assertRaises((TypeError, ValueError)):
                plan_runs(self.load_configuration())

    def test_run_construction_helper_preserves_grid_identity(self):
        """Manual run construction matches the grid planner and copies mutable parameters."""
        run = plan_runs(self.load_configuration())[0]
        rebuilt = make_run_spec(
            group=run.group,
            repetition=run.repetition,
            algorithm_name=run.algorithm_name,
            algorithm=run.algorithm,
            data=run.data.to_dict(),
            protocol=run.protocol,
            seed=run.seed,
            budget_steps=run.budget_steps,
        )
        self.assertEqual(rebuilt, run)
        self.assertIsNot(rebuilt.data.params, run.data.params)

    def test_serialization_does_not_expose_mutable_parameters(self):
        """Mutating a serialized description cannot alter the original run."""
        run = plan_runs(self.load_configuration())[0]
        saved = run.to_dict()
        saved["data"]["params"]["size"] = 999
        self.assertEqual(run.data.params["size"], 2)


class ScientificIdentityTests(_StudyConfigurationCase):
    """Keep scientific identities and random streams independent of execution choices."""

    def test_grid_order_workers_and_analysis_do_not_change_identity(self):
        """Scheduling, display options, and grid order leave scientific identities unchanged."""
        original = self.load_configuration()
        self.document["runs"][0]["grid"]["data.params.size"].reverse()
        self.document["runs"][0]["algorithms"].reverse()
        self.document["execution"] = {"workers": 4, "checkpoint_seconds": 10}
        self.document["recording"] = {"buffer_bytes": 1024}
        self.document["analysis"] = {"figures": []}
        changed = self.load_configuration()
        self.assertEqual(original.simulation_dict(), changed.simulation_dict())
        first_ids = {run.run_id for run in plan_runs(original)}
        self.assertEqual(first_ids, {run.run_id for run in plan_runs(changed)})

    def test_recording_contract_changes_without_changing_run_rng_identity(self):
        """Recording belongs to the execution request while run identity stays fixed."""
        original = self.load_configuration()
        self.document["recording"] = {"every_steps": 10}
        changed = self.load_configuration()
        self.assertNotEqual(original.simulation_dict(), changed.simulation_dict())
        self.assertEqual(plan_runs(original), plan_runs(changed))

    def test_execution_budget_changes_preserve_scientific_identity_and_rngs(self):
        """Execution budgets preserve streams; an algorithm design parameter changes identity."""
        original = plan_runs(self.load_configuration())[0]
        self.document["runs"][0]["protocol"]["params"].pop("horizon")
        self.document["runs"][0]["budget"] = {"steps": 20}
        extended = plan_runs(self.load_configuration())[0]
        self.assertEqual(original.budget_steps, 5)
        self.assertEqual(extended.budget_steps, 20)
        self.assertEqual(original.run_id, extended.run_id)
        self.assertNotIn("horizon", original.protocol.params)
        for stream in ("algorithm", "data", "protocol", "instance"):
            np.testing.assert_array_equal(
                make_rngs(original)[stream].normal(size=20),
                make_rngs(extended)[stream].normal(size=20),
            )
        self.document["runs"][0]["algorithms"][0]["params"]["design_horizon"] = 20
        self.assertNotEqual(original.run_id, plan_runs(self.load_configuration())[0].run_id)

    def test_component_seed_overrides(self):
        """A component seed overrides the study seed only for that component stream."""
        run = plan_runs(self.load_configuration())[0]
        run = replace(run, data=replace(run.data, seed=73))
        changed = replace(run, seed=123456)
        np.testing.assert_array_equal(
            make_rngs(run)["data"].normal(size=20), make_rngs(changed)["data"].normal(size=20)
        )
        self.assertNotEqual(
            make_rngs(run)["algorithm"].random(), make_rngs(changed)["algorithm"].random()
        )

    def test_data_is_paired_and_streams_are_separate(self):
        """Algorithms share repetition data but have independent random streams."""
        runs = plan_runs(self.load_configuration())
        first = next(run for run in runs if run.repetition == 0 and run.algorithm_name == "first")
        second = next(run for run in runs if run.repetition == 0 and run.algorithm_name == "second")
        first_streams, second_streams = make_rngs(first), make_rngs(second)
        first_streams["algorithm"].random(1000)
        np.testing.assert_array_equal(
            first_streams["data"].normal(size=50), second_streams["data"].normal(size=50)
        )
        self.assertNotEqual(
            make_rngs(first)["algorithm"].random(), make_rngs(second)["algorithm"].random()
        )
        other_repetition = replace(first, repetition=1)
        self.assertNotEqual(
            make_rngs(first)["data"].random(), make_rngs(other_repetition)["data"].random()
        )


class RandomStateTests(unittest.TestCase):
    """Check the generator-state assumption used by checkpoint restoration."""

    def test_rng_state_continues_exactly(self):
        """Restoring a NumPy generator state reproduces the next draws exactly."""
        generator = np.random.Generator(np.random.PCG64(77))
        generator.normal(size=9)
        state = generator.bit_generator.state
        expected = generator.normal(size=100)
        generator.bit_generator.state = state
        np.testing.assert_array_equal(generator.normal(size=100), expected)
