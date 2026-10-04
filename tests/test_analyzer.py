"""Tests for local Python analysis."""

from pathlib import Path
import unittest

from app.analyzer import analyze_file, analyze_path


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_code"


class AnalyzerTests(unittest.TestCase):
    def test_finds_issues_in_sample_math_file(self):
        issues = analyze_file(SAMPLE_DIR / "buggy_math.py")
        rules = {issue.rule for issue in issues}

        self.assertIn("unused-import", rules)
        self.assertIn("mutable-default", rules)
        self.assertIn("bare-except", rules)
        self.assertIn("none-comparison", rules)
        self.assertIn("dangerous-call", rules)
        self.assertIn("unused-variable", rules)

    def test_analyze_path_scans_folder(self):
        issues = analyze_path(SAMPLE_DIR)
        files = {issue.file_path for issue in issues}
        self.assertTrue(any("buggy_math.py" in path for path in files))
        self.assertTrue(any("unused_helpers.py" in path for path in files))

    def test_syntax_error_is_reported(self):
        bad_file = SAMPLE_DIR / "syntax_error.py"
        bad_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
        try:
            issues = analyze_file(bad_file)
            self.assertEqual(issues[0].rule, "syntax-error")
        finally:
            bad_file.unlink()


if __name__ == "__main__":
    unittest.main()
