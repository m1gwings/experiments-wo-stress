"""Thin command-line entry points for running studies and working with saved results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .execution.coordinator import run_experiment
from .storage.experiment import inspect_experiment
from .study.config import ExperimentConfig, load_config
from .study.planning import plan_runs


def _run_study(
    config: ExperimentConfig,
    output: Path,
    *,
    workers: int | None,
    max_steps: int | None,
) -> tuple[dict[str, Any], int]:
    """Execute the request, then derive its configured outputs only after completion."""
    report = run_experiment(config, output, workers=workers, max_steps=max_steps)
    result = report.to_dict()
    # A completed subset must not stand in for the whole requested study. An
    # intentional --max-steps pause succeeds, but still defers analysis and figures.
    if not (report.failed or report.paused or report.pending):
        if config.analysis.get("figures"):
            from .analysis.figures import plot

            # Plotting already computes or reuses the analysis it needs.
            result["figures"] = [str(path) for path in plot(config, output)]
        elif config.analysis.get("metrics"):
            from .analysis.pipeline import analyze

            result["groups"] = len(analyze(config, output))
    interrupted = report.pending or (report.paused and max_steps is None)
    exit_code = 1 if report.failed else (130 if interrupted else 0)
    return result, exit_code


def main(argv: list[str] | None = None) -> int:
    """Dispatch one public command and print its JSON report or a concise error."""
    parser = argparse.ArgumentParser(prog="ews", description="Reproducible numerical experiments.")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text, description in (
        (
            "count-runs",
            "Optionally validate and count independent runs before execution.",
            "Validate the study and count its concrete independent runs to help estimate "
            "computational cost. Each study group combines parameter-grid values, "
            "algorithms, and independent repetitions. count-runs is optional. "
            "You do not need to invoke it before run.",
        ),
        (
            "run",
            "Run the study and produce its configured analysis and figures.",
            "Execute, resume, extend, or reuse simulation runs, then produce or reuse "
            "configured analysis and figures when all requested runs complete successfully. "
            "Failed, paused, or pending work defers analysis and figures. "
            "This is the normal complete workflow; count-runs is optional and "
            "does not need to be invoked first.",
        ),
        (
            "analyze",
            "Compute or reuse analysis from saved completed runs.",
            "Change or regenerate metrics and aggregations from compatible completed "
            "results without rerunning expensive simulations.",
        ),
        (
            "plot",
            "Produce or regenerate figures from saved results and analysis.",
            "Update figure formatting without rerunning expensive simulations. "
            "Compute or reuse the needed analysis from saved completed results.",
        ),
    ):
        child = commands.add_parser(command, help=help_text, description=description)
        child.add_argument("config", type=Path, help="Experiment YAML file.")
        if command != "count-runs":
            child.add_argument(
                "--output", "-o", type=Path, required=True, help="Artifact directory."
            )
        if command == "run":
            child.add_argument("--workers", type=int, help="Override the local worker count.")
            child.add_argument(
                "--max-steps", type=int, help="Pause each run after this many new steps."
            )
    inspection = commands.add_parser(
        "inspect",
        help="Inspect saved progress, artifacts, and validation information.",
        description="Inspect saved experiment state, progress, retained runs and artifacts, "
        "and validation information without executing anything.",
    )
    inspection.add_argument("output", type=Path)
    cleanup = commands.add_parser(
        "clean",
        help="Preview or remove selected saved artifacts.",
        description="Preview selected artifacts for cleanup. Files are removed only with --yes.",
    )
    cleanup.add_argument("output", type=Path)
    cleanup.add_argument(
        "--scope",
        choices=("analysis", "checkpoints", "inactive", "runs", "all"),
        default="inactive",
    )
    cleanup.add_argument(
        "--run-id", action="append", help="Stored variant ID; repeat to select several."
    )
    cleanup.add_argument(
        "--yes", action="store_true", help="Apply cleanup; otherwise only preview."
    )
    args = parser.parse_args(argv)
    try:
        exit_code = 0
        if args.command == "clean":
            from .storage.cleanup import clean_experiment

            result = clean_experiment(
                args.output, scope=args.scope, run_ids=args.run_id, yes=args.yes
            )
        elif args.command == "inspect":
            result = inspect_experiment(args.output)
        else:
            config = load_config(args.config)
            if args.command == "count-runs":
                result = {"name": config.name, "runs": len(plan_runs(config))}
            elif args.command == "run":
                result, exit_code = _run_study(
                    config, args.output, workers=args.workers, max_steps=args.max_steps
                )
            elif args.command == "analyze":
                from .analysis.pipeline import analyze

                summaries = analyze(config, args.output)
                result = {"groups": len(summaries), "output": str(args.output / "analysis")}
            else:
                from .analysis.figures import plot

                result = {"figures": [str(path) for path in plot(config, args.output)]}
        print(json.dumps(result, indent=2))
        return exit_code
    except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"ews: {exc}", file=sys.stderr)
        return 1
