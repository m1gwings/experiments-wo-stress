"""Run the standalone guide's copyable study in a separate research directory."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class StandaloneGuideTests(unittest.TestCase):
    """Treat the guide's marked code blocks as a complete external research project."""

    def test_copyable_study_executes_resumes_analyzes_and_reuses(self) -> None:
        """Copy only documented files, run their checks, and build/reuse the stated outputs."""
        guide = Path(__file__).resolve().parents[1] / "docs" / "LLM_GUIDE.md"
        # File markers are the contract for what a reader must copy from the guide.
        snippets = dict(
            re.findall(
                r"<!-- file: ([\w./]+) -->\s*```(?:python|yaml)\n(.*?)\n```",
                guide.read_text(encoding="utf-8"),
                flags=re.DOTALL,
            )
        )
        self.assertEqual(
            set(snippets),
            {
                "experiment_code/algorithms.py",
                "experiment_code/metrics.py",
                "experiment.yml",
                "verify.py",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            # The temporary directory provides no implicit files from this repository.
            project = Path(directory)
            (project / "experiment_code").mkdir()
            (project / "experiment_code" / "__init__.py").write_text("", encoding="utf-8")
            for name, content in snippets.items():
                (project / name).write_text(content + "\n", encoding="utf-8")
            verification = subprocess.run(
                [sys.executable, "verify.py"],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(verification.returncode, 0, verification.stdout + verification.stderr)
            self.assertIn("Direct and resumed results match", verification.stdout)
            # The first build executes all runs; the second must reuse them and reanalyze.
            command = [
                sys.executable,
                "-m",
                "experiments_wo_stress",
                "build",
                "experiment.yml",
                "--output",
                "outputs/study",
                "--workers",
                "2",
            ]
            for expected_status in ("completed", "skipped"):
                with self.subTest(status=expected_status):
                    result = subprocess.run(
                        command, cwd=project, capture_output=True, text=True, timeout=60
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    report = json.loads(result.stdout)
                    self.assertEqual(report[expected_status], 12)
                    self.assertEqual(report["failed"], 0)
                    self.assertEqual(len(report["figures"]), 1)
            # Verify both the figure artifact and its separately stored numerical summary.
            figure = project / "outputs" / "study" / "analysis" / "figures" / "regret.tikz"
            self.assertIn("\\begin{tikzpicture}", figure.read_text(encoding="utf-8"))
            summary = project / "outputs" / "study" / "analysis" / "mean_reward.csv"
            self.assertTrue(summary.is_file())


if __name__ == "__main__":
    unittest.main()
