"""Thin command-line entry points for planning, execution, and saved-data analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .execution.coordinator import run_experiment
from .storage.experiment import inspect_experiment
from .study.config import load_config
from .study.planning import plan_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ews", description="Reproducible numerical experiments.")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("plan", "Validate configuration and list stable runs."),
        ("run", "Execute pending runs or resume their latest checkpoints."),
        ("build", "Reuse or extend runs, then update analysis and figures."),
        ("analyze", "Aggregate saved numerical results."),
        ("plot", "Regenerate figures from saved results."),
    ):
        child = commands.add_parser(command, help=help_text)
        child.add_argument("config", type=Path, help="Experiment YAML file.")
        if command != "plan":
            child.add_argument(
                "--output", "-o", type=Path, required=True, help="Artifact directory."
            )
        if command in {"run", "build"}:
            child.add_argument("--workers", type=int, help="Override the local worker count.")
            child.add_argument(
                "--max-steps", type=int, help="Pause each run after this many new steps."
            )
    inspection = commands.add_parser(
        "inspect", help="Inspect stored progress and validate completed runs."
    )
    inspection.add_argument("output", type=Path)
    cleanup = commands.add_parser("clean", help="Preview or remove selected saved artifacts.")
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
        if args.command == "clean":
            from .storage.cleanup import clean_experiment

            result = clean_experiment(
                args.output, scope=args.scope, run_ids=args.run_id, yes=args.yes
            )
        elif args.command == "inspect":
            result = inspect_experiment(args.output)
        else:
            config = load_config(args.config)
            if args.command == "plan":
                specs = plan_runs(config)
                result = {
                    "name": config.name,
                    "runs": len(specs),
                    "plan": [spec.to_dict() for spec in specs],
                }
            elif args.command in {"run", "build"}:
                report = run_experiment(
                    config, args.output, workers=args.workers, max_steps=args.max_steps
                )
                result = report.to_dict()
                if args.command == "build" and not (
                    report.failed or report.paused or report.pending
                ):
                    if config.analysis.get("figures"):
                        from .analysis.figures import plot

                        result["figures"] = [str(path) for path in plot(config, args.output)]
                    elif config.analysis.get("metrics"):
                        from .analysis.pipeline import analyze

                        result["groups"] = len(analyze(config, args.output))
                print(json.dumps(result, indent=2))
                interrupted = report.pending or (report.paused and args.max_steps is None)
                return 1 if report.failed else (130 if interrupted else 0)
            elif args.command == "analyze":
                from .analysis.pipeline import analyze

                summaries = analyze(config, args.output)
                result = {"groups": len(summaries), "output": str(args.output / "analysis")}
            else:
                from .analysis.figures import plot

                result = {"figures": [str(path) for path in plot(config, args.output)]}
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"ews: {exc}", file=sys.stderr)
        return 1
