"""Lazy durable preparation and dependency-oriented worker admission policies."""

from collections import Counter
from unittest.mock import patch

import yaml

from experiments_wo_stress import load_config, plan_runs
from experiments_wo_stress.execution import coordinator, provenance
from experiments_wo_stress.execution.scheduler import TaskScheduler
from experiments_wo_stress.storage.files import read_json


def study(tmp_path, repetitions=3, analysis=True):
    """Return a short two-unit study with real persisted metric inputs."""
    path = tmp_path / "study.yml"
    settings = {
        "name": "scheduler",
        "runs": [
            {
                "name": "main",
                "repetitions": repetitions,
                "budget": {"steps": 3},
                "protocol": {"type": "online"},
                "data": {"type": "gaussian_bandit", "params": {"means": [0, 1]}},
                "algorithms": [{"name": "a", "type": "tests.sample_components:RecordingLearner"}],
                "grid": {"data.params.noise_std": [0.1, 0.2]},
            }
        ],
        "recording": {"retention": "until_analyzed"},
    }
    if analysis:
        settings["analysis"] = {
            "metrics": [{"name": "reward", "type": "field", "params": {"field": "reward"}}],
            "aggregator": {"group_by": ["data.params.noise_std"], "uncertainty": "none"},
        }
    path.write_text(yaml.safe_dump(settings))
    return load_config(path)


def test_large_request_publishes_one_variant_before_first_worker(tmp_path):
    """Thousands of specs require one request, not thousands of synchronous commits."""
    config = study(tmp_path, repetitions=1000, analysis=False)
    owner = coordinator.ExecutionCoordinator(config, tmp_path / "output")
    observed = []

    def first_worker(spec, root, *args, **kwargs):
        request = read_json(owner.store.metadata_path)
        observed.append((len(request["run_ids"]), len(list(owner.store.runs_path.iterdir()))))
        owner.stop_event.set()
        return spec["run_id"], "paused", None

    with patch.object(coordinator, "execute_run", side_effect=first_worker):
        report = owner.run()
    assert observed == [(2000, 1)]
    assert report.paused == 1 and report.pending == 1999
    assert len(list(owner.store.requests_path.glob("*.json"))) == 1


def test_repeated_component_files_are_fingerprinted_once_during_preparation(tmp_path):
    config = study(tmp_path, repetitions=10, analysis=False)
    specs = plan_runs(config)
    provenance_record = provenance.collect_provenance(config, provenance.preflight(specs))
    original = provenance.digest_file
    paths = Counter()

    def count(path):
        paths[str(path)] += 1
        return original(path)

    with patch.object(provenance, "digest_file", side_effect=count):
        provenance.prepare_request(config, specs, provenance_record)
    assert paths and max(paths.values()) == 1


def test_metrics_release_raw_data_and_aggregation_precedes_unrelated_simulations(tmp_path):
    """A group closes even when a planner interleaves its repetitions with another group."""
    config = study(tmp_path)
    specs = plan_runs(config)
    interleaved = [spec for pair in zip(specs[:3], specs[3:]) for spec in pair]
    events = []
    original_run = coordinator.execute_run
    original_derive = coordinator.execute_derivation_task

    def simulate(spec, *args, **kwargs):
        events.append(("simulation", spec["data"]["params"]["noise_std"]))
        return original_run(spec, *args, **kwargs)

    def derive(task, *args, **kwargs):
        events.append((task.kind, task.subject))
        return original_derive(task, *args, **kwargs)

    with (
        patch.object(coordinator, "plan_runs", return_value=interleaved),
        patch.object(coordinator, "execute_run", side_effect=simulate),
        patch.object(coordinator, "execute_derivation_task", side_effect=derive),
    ):
        report = coordinator.run_experiment(config, tmp_path / "output")
    assert not report.errors
    assert [value for kind, value in events if kind == "simulation"] == [0.1] * 3 + [0.2] * 3
    assert [kind for kind, _ in events[:7]] == ["simulation", "metric"] * 3 + ["aggregate"]
    assert not list((tmp_path / "output" / "runs").glob("*/results/*.npz"))


def test_ready_derivations_cannot_starve_simulations(tmp_path):
    """Even an always-ready derived queue must admit pending simulation work."""
    from types import SimpleNamespace

    config = study(tmp_path, analysis=False)
    specs = plan_runs(config)
    scheduler = TaskScheduler(
        config, tmp_path / "output", specs, {s.run_id: s.run_id for s in specs}
    )
    scheduler.graph = SimpleNamespace(
        ready_tasks=lambda: [
            SimpleNamespace(
                id=f"metric-{len(scheduler.active)}",
                kind="metric",
                subject="reward",
                run_ids=(),
                payload=None,
            )
        ]
    )
    assert [scheduler.next_task().kind for _ in range(9)] == ["metric"] * 8 + ["simulation"]


def test_interruption_after_final_simulation_retains_pending_derivations(tmp_path):
    """A drained simulation queue cannot disguise unfinished analysis as success."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from experiments_wo_stress.cli import main

    config = study(tmp_path, repetitions=1)
    owner = coordinator.ExecutionCoordinator(config, tmp_path / "output")
    original = coordinator.execute_run
    completed = 0

    def simulate(*args, **kwargs):
        nonlocal completed
        result = original(*args, **kwargs)
        completed += 1
        if completed == 2:
            owner.stop_event.set()
        return result

    with patch.object(coordinator, "execute_run", side_effect=simulate):
        report = owner.run()
    assert report.completed == 2 and report.pending == 0
    assert report.interrupted and report.pending_tasks > 0
    assert report.groups is None
    saved = read_json(tmp_path / "output/compute/summary.json")
    assert saved["latest_invocation"]["status"] == "incomplete"
    with (
        patch("experiments_wo_stress.cli.run_experiment", return_value=report),
        redirect_stdout(io.StringIO()),
        redirect_stderr(io.StringIO()),
    ):
        assert (
            main(["run", str(tmp_path / "study.yml"), "--output", str(tmp_path / "output")]) == 130
        )
    resumed = coordinator.run_experiment(config, tmp_path / "output")
    assert resumed.skipped == 2 and not resumed.pending_tasks
    assert resumed.groups == 2


def test_pruned_reuse_still_validates_its_retained_scientific_instance(tmp_path):
    """Deliberate trajectory disposal never authorizes missing scientific inputs."""
    config = study(tmp_path, repetitions=1)
    output = tmp_path / "output"
    assert coordinator.run_experiment(config, output).completed == 2
    for path in (output / "instances").glob("*/arrays.npz"):
        path.unlink()
    report = coordinator.run_experiment(config, output)
    assert report.failed == 2 and report.skipped == 0
    assert all("instance" in message.lower() for message in report.errors.values())


def test_ready_aggregates_and_figures_are_not_starved_by_metric_backlog(tmp_path):
    """A persistent metric queue cannot hide an already useful downstream output."""
    from types import SimpleNamespace

    config = study(tmp_path, analysis=False)
    specs = plan_runs(config)
    scheduler = TaskScheduler(
        config, tmp_path / "output", specs, {s.run_id: s.run_id for s in specs}
    )
    scheduler.graph = SimpleNamespace(
        ready_tasks=lambda: [
            SimpleNamespace(
                id=f"{kind}-{len(scheduler.active)}", kind=kind, subject=kind, run_ids=()
            )
            for kind in ("metric", "aggregate", "figure")
        ]
    )
    admitted = [scheduler.next_task().kind for _ in range(6)]
    assert admitted[0] == "metric"
    assert "aggregate" in admitted and "figure" in admitted


def test_retained_repetition_prefers_its_unfinished_analysis_group(tmp_path):
    """Prior invocations influence admission without entering identities or RNG seeds."""
    from types import SimpleNamespace

    config = study(tmp_path)
    specs = plan_runs(config)
    existing = specs[3].run_id
    graph = SimpleNamespace(
        references={
            spec.run_id: {"revision": "retained"} if spec.run_id == existing else None
            for spec in specs
        },
        analysis_unit=lambda run_id: next(
            spec.data.params["noise_std"] for spec in specs if spec.run_id == run_id
        ),
        ready_tasks=lambda: [],
    )
    scheduler = TaskScheduler(
        config, tmp_path / "output", specs, {s.run_id: s.run_id for s in specs}, graph=graph
    )
    assert [scheduler.next_task().payload.data.params["noise_std"] for _ in range(3)] == [0.2] * 3
