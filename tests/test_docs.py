"""Focused tests for the mechanical documentation checker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_docs import markdown_errors


class DocumentationCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "README.md"

    def check_text(self, text: str) -> list[str]:
        self.path.write_text(text, encoding="utf-8")
        return markdown_errors(self.path, self.root)

    def test_local_links_are_checked_and_external_links_are_not_fetched(self) -> None:
        errors = self.check_text("[external](https://example.com/x)\n[missing](absent.md)\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("missing link target absent.md", errors[0])

    def test_existing_relative_links_and_balanced_fences_pass(self) -> None:
        (self.root / "docs").mkdir()
        (self.root / "docs" / "page.md").touch()
        self.assertEqual(
            self.check_text("[page](docs/page.md#section)\n```python\nx = 1\n```\n"), []
        )

    def test_unclosed_fence_is_reported(self) -> None:
        self.assertIn("unclosed Markdown fence", self.check_text("```python\nx = 1\n")[0])

    def test_stale_repository_path_is_reported(self) -> None:
        self.assertIn("missing repository path", self.check_text("See `src/removed.py`.\n")[0])


if __name__ == "__main__":
    unittest.main()
