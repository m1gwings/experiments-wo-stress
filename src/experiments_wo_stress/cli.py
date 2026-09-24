"""Thin command-line entry points for planning, execution, and saved-data analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .jobs import plan_runs
from .runner import inspect_experiment, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ews", description="Reproducible numerical experiments.")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("plan", "Validate configuration and list stable runs."),
        ("run", "Execute pending runs or resume their latest checkpoints."),
        ("analyze", "Aggregate saved numerical results."),
        ("plot", "Regenerate figures from saved results."),
    ):
        child = commands.add_parser(command, help=help_text)
        child.add_argument("config", type=Path, help="Experiment YAML file.")
        if command != "plan":
            child.add_argument(
                "--output", "-o", type=Path, required=True, help="Artifact directory."
            )
        if command == "run":
            child.add_argument("--workers", type=int, help="Override the local worker count.")
            child.add_argument(
                "--max-steps", type=int, help="Pause each run after this many new steps."
            )
    inspection = commands.add_parser(
        "inspect", help="Inspect stored progress and validate completed runs."
    )
    inspection.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
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
            elif args.command == "run":
                report = run_experiment(
                    config, args.output, workers=args.workers, max_steps=args.max_steps
                )
                print(json.dumps(report.to_dict(), indent=2))
                interrupted = report.pending or (report.paused and args.max_steps is None)
                return 1 if report.failed else (130 if interrupted else 0)
            elif args.command == "analyze":
                from .analysis import analyze

                summaries = analyze(config, args.output)
                result = {"groups": len(summaries), "output": str(args.output / "analysis")}
            else:
                from .plotting import plot

                result = {"figures": [str(path) for path in plot(config, args.output)]}
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"ews: {exc}", file=sys.stderr)
        return 1
