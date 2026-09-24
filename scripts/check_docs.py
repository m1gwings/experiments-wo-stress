#!/usr/bin/env python3
"""Check local Markdown links, fences, repository paths, and optional live contracts.

This is a mechanical consistency check, not a verifier of scientific or semantic
claims. Run it after editing a file or a tightly related batch of changes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
REPO_PATH = re.compile(r"(?<![\w./])(?:src|docs|examples|tests|scripts|\.github)/[\w./*-]+")
COMMAND = re.compile(r"\bews\s+([a-z][a-z-]*)\b")


def markdown_errors(path: Path, root: Path) -> list[str]:
    """Return actionable errors for one Markdown file without executing examples."""
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    opened: tuple[str, int, int] | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = FENCE.match(line)
        if marker:
            fence = marker.group(1)
            if opened is None:
                opened = fence[0], len(fence), line_number
            elif fence[0] == opened[0] and len(fence) >= opened[1]:
                opened = None
            continue
        if opened is not None:
            continue
        for match in LINK.finditer(line):
            target = match.group(1).strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = path.parent / unquote(parsed.path)
            if not resolved.exists():
                errors.append(
                    f"{path.relative_to(root)}:{line_number}: missing link target {target}"
                )
    if opened is not None:
        errors.append(f"{path.relative_to(root)}:{opened[2]}: unclosed Markdown fence")
    for target in sorted(set(REPO_PATH.findall(text))):
        target = target.rstrip(".")
        if "*" in target:
            exists = any(root.glob(target))
        else:
            exists = (root / target).exists()
        if not exists:
            errors.append(f"{path.relative_to(root)}: missing repository path {target}")
    return errors


def documentation_paths(root: Path) -> list[Path]:
    paths = list(root.glob("*.md"))
    for directory in ("docs", "examples"):
        paths.extend((root / directory).rglob("*.md"))
    return sorted(paths)


def check_examples(root: Path) -> list[str]:
    from experiments_wo_stress import load_config, plan_runs

    errors = []
    for path in sorted((root / "examples").rglob("*.yml")):
        try:
            config = load_config(path)
            if not plan_runs(config):
                raise ValueError("example produces an empty run plan")
        except (ValueError, TypeError, OSError, ImportError) as exc:
            errors.append(f"{path.relative_to(root)}: {exc}")
    return errors


def check_commands(paths: list[Path], root: Path) -> list[str]:
    commands = {command for path in paths for command in COMMAND.findall(path.read_text())}
    errors = []
    for command in sorted(commands):
        result = subprocess.run(
            [sys.executable, "-m", "experiments_wo_stress", command, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            errors.append(f"documented CLI command ews {command}: {result.stderr.strip()}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--examples", action="store_true", help="Load and plan example YAML files")
    parser.add_argument("--cli", action="store_true", help="Check help for documented CLI commands")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    paths = documentation_paths(root)
    errors = [error for path in paths for error in markdown_errors(path, root)]
    if args.examples:
        errors.extend(check_examples(root))
    if args.cli:
        errors.extend(check_commands(paths, root))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    checks = ["links", "fences", "repository paths"]
    if args.examples:
        checks.append("example configurations")
    if args.cli:
        checks.append("CLI commands")
    print(f"Documentation checks passed for {len(paths)} files: {', '.join(checks)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
