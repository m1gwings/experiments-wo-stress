#!/usr/bin/env python3
"""Build the standalone-study Docker example and verify durable pause/resume/reuse.

Requires Docker, pip-tools, Python 3.12, and a non-root Linux account. Downloads
the supplied accessible Git revision and dependencies; provisions no cloud resources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path

CONFIG = """name: container_smoke
seed: 2026
runs:
  - name: main
    repetitions: 2
    budget: {steps: 20}
    protocol: {type: online}
    data: {type: 'experiment_code.data:GaussianBandit', params: {n_arms: 3}}
    algorithms:
      - {name: ucb, type: 'experiment_code.algorithms:UCB'}
execution: {workers: 2}
recording: {every_steps: 1, fields: [action, reward]}
analysis:
  metrics:
    - {name: regret, type: pseudo_regret}
  aggregator:
    group_by: [algorithm.name]
    uncertainty: standard_error
  figures:
    - {type: line, metric: regret, color: algorithm.name, formats: [tikz]}
"""


def run(command: list[str], *, cwd: Path | None = None, report: bool = False):
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if report else None,
        check=True,
        timeout=600,
    )
    if report:
        print(result.stdout, flush=True)
        return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", required=True, help="Accessible full 40-character library SHA"
    )
    parser.add_argument(
        "--github-token-env", help="Environment variable holding a repository-read GitHub token"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full 40-character lowercase Git commit hash")
    if os.name != "posix" or os.getuid() == 0 or shutil.which("docker") is None:
        parser.error("run as a non-root Linux account with Docker available")
    build_secrets = []
    if args.github_token_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.github_token_env):
            parser.error("--github-token-env must name an environment variable")
        if not os.environ.get(args.github_token_env):
            parser.error("the environment variable named by --github-token-env is missing or empty")
        build_secrets = ["--secret", f"id=github_token,env={args.github_token_env}"]
    root = Path(__file__).resolve().parents[1]
    name = "ews-smoke-" + uuid.uuid4().hex[:12]
    image = name + ":test"
    with tempfile.TemporaryDirectory(prefix="ews-container-") as temporary:
        study = Path(temporary) / "study"
        output = Path(temporary) / "output"
        study.mkdir()
        output.mkdir()
        for source, target in (
            ("deploy/Dockerfile.example", "Dockerfile"),
            ("deploy/.dockerignore.example", ".dockerignore"),
        ):
            shutil.copyfile(root / source, study / target)
        shutil.copytree(
            root / "examples/sequential_study/experiment_code", study / "experiment_code"
        )
        (study / "experiment.yml").write_text(CONFIG, encoding="utf-8")
        (study / "requirements.in").write_text(
            "numpy>=1.24\nPyYAML>=6.0\nsetuptools>=68\nwheel\n", encoding="utf-8"
        )
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "piptools",
                    "compile",
                    "--generate-hashes",
                    "--allow-unsafe",
                    "--output-file",
                    "requirements.lock",
                    "requirements.in",
                ],
                cwd=study,
            )
            run(
                [
                    "docker",
                    "build",
                    *build_secrets,
                    "--build-arg",
                    f"EWS_REV={args.revision}",
                    "--build-arg",
                    f"STUDY_REV={args.revision}",
                    "--tag",
                    image,
                    ".",
                ],
                cwd=study,
            )
            container = [
                "docker",
                "run",
                "--rm",
                "--name",
                name,
                "--init",
                "--stop-timeout",
                "120",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,src={output},dst=/output",
                image,
            ]
            invocation = [
                "/workspace/experiment.yml",
                "--output",
                "/output/study",
                "--workers",
                "2",
            ]
            paused = run([*container, "run", *invocation, "--max-steps", "5"], report=True)
            assert paused["paused"] == 2 and paused["failed"] == 0, paused
            resumed = run([*container, "run", *invocation], report=True)
            assert resumed["completed"] == 2 and resumed["failed"] == 0, resumed
            inspected = run([*container, "inspect", "/output/study"], report=True)
            assert inspected["counts"]["completed"] == 2, inspected
            assert all(item["step"] == 20 for item in inspected["runs"]), inspected
            for _ in range(2):
                built = run([*container, "build", *invocation], report=True)
                assert built["skipped"] == 2 and built["failed"] == 0, built
                assert built["figures"], built
            figures = list((output / "study" / "analysis" / "figures").glob("*.tikz"))
            assert figures and all("\\begin{tikzpicture}" in path.read_text() for path in figures)
            print(
                "Container smoke passed: two runs paused, resumed, inspected, plotted, and reused."
            )
        finally:
            for command in (["docker", "rm", "--force", name], ["docker", "image", "rm", image]):
                with suppress(OSError, subprocess.SubprocessError):
                    subprocess.run(
                        command,
                        check=False,
                        timeout=30,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
