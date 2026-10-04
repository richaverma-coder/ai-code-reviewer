"""Tests for Bandit, Black, and Pytest report helpers."""

import unittest
from pathlib import Path
from unittest.mock import patch

from app.quality import ToolResult, run_quality_tools


class QualityToolsTests(unittest.TestCase):
    def test_runs_bandit_black_and_pytest(self):
        fake = ToolResult("Bandit", True, True, "ok", "")
        with patch("app.quality._run_module", return_value=fake) as mocked:
            results = run_quality_tools("sample_code")
        self.assertEqual(len(results), 3)
        names = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(names, ["Bandit", "Black", "Pytest"])

    def test_black_is_check_only(self):
        with patch("app.quality._run_module") as mocked:
            mocked.return_value = ToolResult("Black", True, True, "ok", "")
            run_quality_tools(Path("sample_code"))
        black_args = mocked.call_args_list[1].args[1]
        self.assertIn("--check", black_args)
        self.assertNotIn("--write", black_args)


if __name__ == "__main__":
    unittest.main()
