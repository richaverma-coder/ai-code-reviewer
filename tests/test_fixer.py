"""Tests for safe unused-import fixes."""

from pathlib import Path
import tempfile
import unittest

from app.analyzer import Issue, SEVERITY_INFO, analyze_file
from app.fixer import apply_fixes, plan_fixes


class FixerTests(unittest.TestCase):
    def test_plans_unused_import_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            path.write_text("import os\n\nvalue = 1\nprint(value)\n", encoding="utf-8")
            issues = analyze_file(path)
            planned = plan_fixes(issues)
            self.assertEqual(len(planned), 1)
            self.assertEqual(planned[0].rule, "unused-import")

    def test_apply_fixes_removes_unused_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            original = "import os\n\nvalue = 1\nprint(value)\n"
            path.write_text(original, encoding="utf-8")
            issues = analyze_file(path)
            reports_dir = Path(tmp) / "reports"
            results = apply_fixes(
                issues,
                write_files=True,
                target=path,
                reports_dir=reports_dir,
            )
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertTrue(results)
            self.assertTrue(results[0].accepted)
            self.assertTrue(results[0].fixed_file)
            fixed_text = Path(results[0].fixed_file).read_text(encoding="utf-8")
            self.assertNotIn("import os", fixed_text)
            self.assertIn("value = 1", fixed_text)
            self.assertTrue(any(reports_dir.joinpath("fixes").rglob("*.diff")))

    def test_rejects_non_import_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            original = "value = 1\nprint(value)\n"
            path.write_text(original, encoding="utf-8")
            fake = Issue(
                file_path=str(path),
                line=1,
                column=0,
                severity=SEVERITY_INFO,
                rule="unused-import",
                message="Imported name `os` is never used.",
                suggestion="Remove the unused import `os`.",
            )
            results = apply_fixes(
                [fake],
                write_files=True,
                target=path,
                reports_dir=Path(tmp) / "reports",
            )
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(results[0].status, "rejected")
            self.assertFalse(results[0].accepted)
            self.assertIn("not an import", results[0].detail)

    def test_does_not_drop_shared_import_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.py"
            original = "import os, sys\nprint(sys.version)\n"
            path.write_text(original, encoding="utf-8")
            issues = analyze_file(path)
            unused = [issue for issue in issues if issue.rule == "unused-import"]
            results = apply_fixes(
                unused,
                write_files=True,
                target=path,
                reports_dir=Path(tmp) / "reports",
            )
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertTrue(results)
            self.assertFalse(results[0].accepted)


if __name__ == "__main__":
    unittest.main()
