"""Scientific identity, random streams, and component contracts."""

from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from experiments_wo_stress.components import (
    Feedback,
    StateMixin,
    construct,
    resolve_type,
    validate_component,
)
from experiments_wo_stress.config import load_config
from experiments_wo_stress.data import CSVDataGenerator, NormalDataGenerator, NullDataGenerator
from experiments_wo_stress.jobs import ComponentSpec, GridPlanner, RunSpec, make_run_spec, plan_runs
from experiments_wo_stress.protocols import OfflineProtocol, OnlineProtocol, TrialProtocol
from experiments_wo_stress.rng import make_rngs


def trial_function(*, algorithm, data, rng, count):
    return {"sample": rng.normal(size=count)}


class _Algorithm(StateMixin):
    def __init__(self):
        self.observed = []

    def act(self, context=None):
        return context["action"] if context else 2

    def observe(self, action, feedback):
        self.observed.append((action, feedback))


class _Data(StateMixin):
    def context(self):
        return {"action": 3}

    def generate(self, action):
        return Feedback(action + 0.5, {"reward": action + 0.5, "hidden": 100.0})


class _OfflineAlgorithm(StateMixin):
    def fit(self, dataset):
        return {"mean": np.mean(dataset), "size": np.size(dataset)}


class SubsetPlanner:
    """Example extension: retain the smallest data size and first repetition."""

    def plan(self, group, seed):
        for run in GridPlanner().plan(group, seed):
            if run.data.params["size"] == 2 and run.repetition == 0:
                yield run


class InvalidPlanner:
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


class ConfigurationTests(unittest.TestCase):
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

    def load(self, document=None):
        self.path.write_text(yaml.safe_dump(document or self.document), encoding="utf-8")
        return load_config(self.path)

    def test_defaults_and_cartesian_grid(self):
        config = self.load()
        planned = plan_runs(config)
        self.assertEqual(len(planned), 12)
        self.assertEqual(len({run.run_id for run in planned}), 12)
        self.assertEqual(config.execution["checkpoint_seconds"], 120)
        self.assertEqual(config.recording["buffer_bytes"], 16 * 1024 * 1024)
        self.assertEqual(planned[0].labels["data.params.size"], 2)
        self.assertEqual(planned[0].labels["algorithm.name"], "first")
        self.assertEqual(RunSpec.from_dict(planned[0].to_dict()), planned[0])

    def test_custom_planner_selects_subset_with_unchanged_identities(self):
        grid_runs = plan_runs(self.load())
        self.document["runs"][0]["planner"] = f"{__name__}:SubsetPlanner"
        config = self.load()
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
        self.document["runs"][0]["planner"] = f"{__name__}:InvalidPlanner"
        for mode in ("duplicate", "identity", "repetition", "seed", "group", "type", "component"):
            self.document["runs"][0]["data"]["params"]["invalid_mode"] = mode
            with self.subTest(mode=mode), self.assertRaises((TypeError, ValueError)):
                plan_runs(self.load())

    def test_run_construction_helper_preserves_grid_identity(self):
        run = plan_runs(self.load())[0]
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

    def test_loading_custom_planner_does_not_import_it(self):
        self.document["runs"][0]["planner"] = "unavailable_components.planners:SubsetPlanner"
        config = self.load()
        self.assertEqual(config.runs[0]["planner"], "unavailable_components.planners:SubsetPlanner")

    def test_grid_order_workers_and_analysis_do_not_change_identity(self):
        original = self.load()
        self.document["runs"][0]["grid"]["data.params.size"].reverse()
        self.document["runs"][0]["algorithms"].reverse()
        self.document["execution"] = {"workers": 4, "checkpoint_seconds": 10}
        self.document["recording"] = {"buffer_bytes": 1024}
        self.document["analysis"] = {"figures": []}
        changed = self.load()
        self.assertEqual(original.simulation_dict(), changed.simulation_dict())
        first_ids = {run.run_id for run in plan_runs(original)}
        self.assertEqual(first_ids, {run.run_id for run in plan_runs(changed)})

    def test_recording_contract_changes_without_changing_run_rng_identity(self):
        original = self.load()
        self.document["recording"] = {"every_steps": 10}
        changed = self.load()
        self.assertNotEqual(original.simulation_dict(), changed.simulation_dict())
        self.assertEqual(plan_runs(original), plan_runs(changed))

    def test_execution_budget_changes_preserve_scientific_identity_and_rngs(self):
        original = plan_runs(self.load())[0]
        self.document["runs"][0]["protocol"]["params"].pop("horizon")
        self.document["runs"][0]["budget"] = {"steps": 20}
        extended = plan_runs(self.load())[0]
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
        self.assertNotEqual(original.run_id, plan_runs(self.load())[0].run_id)

    def test_dependency_paths_and_logging_settings(self):
        self.document["runs"][0]["data"]["dependencies"] = ["helpers.py", "inputs/data.csv"]
        self.document["execution"] = {
            "logging_level": "debug",
            "log_max_bytes": 4096,
            "log_backups": 3,
        }
        config = self.load()
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
                self.load()

    def test_component_seed_overrides(self):
        run = plan_runs(self.load())[0]
        run = replace(run, data=replace(run.data, seed=73))
        changed = replace(run, seed=123456)
        np.testing.assert_array_equal(
            make_rngs(run)["data"].normal(size=20), make_rngs(changed)["data"].normal(size=20)
        )
        self.assertNotEqual(
            make_rngs(run)["algorithm"].random(), make_rngs(changed)["algorithm"].random()
        )

    def test_data_is_paired_and_streams_are_separate(self):
        runs = plan_runs(self.load())
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

    def test_csv_path_resolution_in_parameters_and_grid(self):
        self.document["runs"][0]["data"] = {"type": "csv", "params": {"path": "data.csv"}}
        self.document["runs"][0]["grid"] = {"data.params.path": ["a.csv", "b.csv"]}
        config = self.load()
        self.assertEqual(
            config.runs[0]["data"]["params"]["path"], str(self.path.parent / "data.csv")
        )
        self.assertEqual(plan_runs(config)[0].data.params["path"], str(self.path.parent / "a.csv"))

    def test_invalid_settings_are_rejected(self):
        mutations = [
            lambda value: value.update({"unknown": True}),
            lambda value: value.update({"seed": True}),
            lambda value: value.update({"execution": {"workers": 0}}),
            lambda value: value.update({"execution": {"checkpoint_seconds": -1}}),
            lambda value: value.update({"recording": {"every_steps": 0}}),
            lambda value: value.update({"recording": {"fields": ["reward", "reward"]}}),
            lambda value: value["runs"][0].update({"repetitions": 0}),
            lambda value: value["runs"][0].update({"grid": {"data.params.size": [2, 2]}}),
            lambda value: value["runs"][0].update({"grid": {"data.size": [2]}}),
            lambda value: value["runs"][0].update(
                {"grid": {"data.params.x": [1], "data.params.x.y": [2]}}
            ),
            lambda value: value["runs"][0]["data"].update({"params": {"rng": 123}}),
            lambda value: value["runs"][0]["data"].update({"params": {"scale": float("nan")}}),
        ]
        for mutate in mutations:
            document = copy.deepcopy(self.document)
            mutate(document)
            with self.subTest(document=document), self.assertRaises(ValueError):
                self.load(document)

    def test_duplicate_yaml_keys_are_rejected(self):
        self.path.write_text("name: first\nname: second\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML key"):
            load_config(self.path)

    def test_serialization_does_not_expose_mutable_parameters(self):
        run = plan_runs(self.load())[0]
        saved = run.to_dict()
        saved["data"]["params"]["size"] = 999
        self.assertEqual(run.data.params["size"], 2)


class ComponentTests(unittest.TestCase):
    def test_online_observation_boundary_and_restore(self):
        algorithm, data = _Algorithm(), _Data()
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
        data = NormalDataGenerator(rng=np.random.default_rng(3), size=10, loc=2, scale=0)
        algorithm = _OfflineAlgorithm()
        protocol = OfflineProtocol(rng=np.random.default_rng(4))
        protocol.initialize(algorithm, data)
        self.assertEqual(protocol.advance(algorithm, data), {"mean": 2, "size": 10})
        self.assertTrue(protocol.is_finished())

    def test_trial_function_and_reserved_parameters(self):
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

    def test_csv_batches_resume_without_storing_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            np.savetxt(path, np.arange(12).reshape(6, 2), delimiter=",")
            source = CSVDataGenerator(rng=np.random.default_rng(0), path=path)
            np.testing.assert_array_equal(source.generate(2), [[0, 1], [2, 3]])
            restored = CSVDataGenerator(rng=np.random.default_rng(0), path=path)
            restored.load_state_dict(source.state_dict())
            np.testing.assert_array_equal(restored.generate(2), [[4, 5], [6, 7]])
            self.assertFalse(source.generate(None).flags.writeable)
            self.assertEqual(source.state_dict(), {"cursor": 2})
            with self.assertRaises(ValueError):
                restored.load_state_dict({"cursor": 10})

    def test_component_resolution_and_parameters(self):
        self.assertIs(resolve_type("normal"), NormalDataGenerator)
        instance = construct(ComponentSpec("normal", {"size": 5}), np.random.default_rng(1))
        self.assertEqual(instance.generate(None).shape, (5,))
        with self.assertRaisesRegex(TypeError, "Invalid parameters"):
            construct(ComponentSpec("normal", {"typo": 5}), np.random.default_rng(1))
        with self.assertRaisesRegex(ValueError, "Unknown component"):
            resolve_type("missing")
        with self.assertRaisesRegex(TypeError, "missing required methods"):
            validate_component(instance, ["act", "observe"])

    def test_rng_state_continues_exactly(self):
        generator = np.random.Generator(np.random.PCG64(77))
        generator.normal(size=9)
        state = generator.bit_generator.state
        expected = generator.normal(size=100)
        generator.bit_generator.state = state
        np.testing.assert_array_equal(generator.normal(size=100), expected)


if __name__ == "__main__":
    unittest.main()
