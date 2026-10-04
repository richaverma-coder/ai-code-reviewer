"""Tests for Ollama AI fixes on a temporary working copy."""

from pathlib import Path
import json
import tempfile
import unittest

from app.analyzer import analyze_file
from app.ai_fixer import apply_ai_fixes, parse_ai_fix_response
from app.ollama_reviewer import AIFixProposal, OllamaError, parse_fix_response


NONE_COMPARE_SOURCE = (
    "def ready(value):\n"
    "    if value == None:\n"
    "        return False\n"
    "    return True\n"
)


def _none_compare_file(folder: Path) -> Path:
    path = folder / "demo.py"
    path.write_text(NONE_COMPARE_SOURCE, encoding="utf-8")
    return path


class AIFixParsingTests(unittest.TestCase):
    def test_parses_required_json_fields(self):
        payload = {
            "explanation": "Compare None with is.",
            "original_code": "if value == None:",
            "corrected_code": "if value is None:",
            "reason": "PEP 8.",
        }
        proposal = parse_fix_response(json.dumps(payload))
        self.assertEqual(proposal.original_code, "if value == None:")
        self.assertEqual(proposal.corrected_code, "if value is None:")
        self.assertEqual(proposal.explanation, "Compare None with is.")
        self.assertEqual(proposal.reason, "PEP 8.")

    def test_parses_fenced_json(self):
        raw = (
            "Here is the fix:\n"
            "```json\n"
            '{"explanation": "e", "original_code": "a = 1", '
            '"corrected_code": "a = 2", "reason": "r"}\n'
            "```\n"
        )
        proposal = parse_ai_fix_response(raw)
        self.assertEqual(proposal.original_code, "a = 1")
        self.assertEqual(proposal.corrected_code, "a = 2")

    def test_missing_fields_are_rejected(self):
        with self.assertRaises(OllamaError):
            parse_fix_response('{"explanation": "only one field"}')


BARE_EXCEPT_SOURCE = (
    "def safe_divide(left, right):\n"
    "    try:\n"
    "        return left / right\n"
    "    except:\n"
    "        return None\n"
)


def _write_source(folder: Path, name: str, source: str) -> Path:
    path = folder / name
    path.write_text(source, encoding="utf-8")
    return path


def _bare_except_proposal(issue, source_text) -> AIFixProposal:
    del issue, source_text
    return AIFixProposal(
        explanation="Catch a specific expected exception.",
        original_code="except:",
        corrected_code="except ZeroDivisionError:",
        reason="A bare except catches every exception.",
    )


class AIFixWorkflowTests(unittest.TestCase):

    def test_original_files_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _none_compare_file(Path(tmp))
            original = path.read_text(encoding="utf-8")
            issues = analyze_file(path)

            apply_ai_fixes(
                issues,
                target=path,
                reports_dir=Path(tmp) / "reports",
                proposer=_good_none_proposal,
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                original,
            )

    def test_accepted_fix_when_validation_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _none_compare_file(Path(tmp))
            original = path.read_text(encoding="utf-8")

            issues = [
                item
                for item in analyze_file(path)
                if item.rule == "none-comparison"
            ]

            results = apply_ai_fixes(
                issues,
                target=path,
                reports_dir=Path(tmp) / "reports",
                proposer=_good_none_proposal,
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                original,
            )
            self.assertTrue(results)
            self.assertTrue(results[0].accepted)
            self.assertEqual(results[0].status, "accepted")
            self.assertTrue(results[0].fixed_file)

            fixed = Path(results[0].fixed_file).read_text(
                encoding="utf-8"
            )

            self.assertIn("if value is None:", fixed)
            self.assertNotIn("== None", fixed)
            self.assertIn(
                "ai_fixes",
                results[0].fixed_file.replace("\\", "/"),
            )

    def test_rejected_fix_when_validation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _none_compare_file(Path(tmp))
            original = path.read_text(encoding="utf-8")

            issues = [
                item
                for item in analyze_file(path)
                if item.rule == "none-comparison"
            ]

            results = apply_ai_fixes(
                issues,
                target=path,
                reports_dir=Path(tmp) / "reports",
                proposer=_syntax_error_proposal,
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                original,
            )
            self.assertTrue(results)
            self.assertFalse(results[0].accepted)
            self.assertEqual(results[0].status, "rejected")
            self.assertIn(
                "syntax",
                results[0].validation_result.lower(),
            )
            self.assertFalse(results[0].fixed_file)

    def test_diff_generation_for_accepted_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _none_compare_file(Path(tmp))

            issues = [
                item
                for item in analyze_file(path)
                if item.rule == "none-comparison"
            ]

            results = apply_ai_fixes(
                issues,
                target=path,
                reports_dir=Path(tmp) / "reports",
                proposer=_good_none_proposal,
            )

            self.assertTrue(results[0].accepted)

            diff_path = Path(results[0].diff_path)
            self.assertTrue(diff_path.is_file())

            diff_text = diff_path.read_text(
                encoding="utf-8"
            )

            self.assertIn("== None", diff_text)
            self.assertIn("is None", diff_text)
            self.assertTrue(
                diff_text.startswith("---")
                or "--- " in diff_text
            )

    def test_bare_except_fix_matches_local_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_source(
                Path(tmp),
                "demo.py",
                BARE_EXCEPT_SOURCE,
            )

            issues = analyze_file(path)

            bare_issue = next(
                issue
                for issue in issues
                if issue.rule == "bare-except"
            )

            result = apply_ai_fixes(
                [bare_issue],
                target=path,
                reports_dir=Path(tmp) / "reports",
                proposer=_bare_except_proposal,
            )

            assert result
            assert path.read_text(encoding="utf-8") == BARE_EXCEPT_SOURCE

    def test_surrounding_context_proposal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _none_compare_file(Path(tmp))

            issues = [
                item
                for item in analyze_file(path)
                if item.rule == "none-comparison"
            ]

            self.assertTrue(issues)

            def surrounding_context_proposal(issue, source_text):
                del issue, source_text
                return AIFixProposal(
                    explanation="Use identity comparison for None.",
                    original_code=(
                        "def ready(value):\n"
                        "    if value == None:\n"
                        "        return False"
                    ),
                    corrected_code=(
                        "def ready(value):\n"
                        "    if value is None:\n"
                        "        return False"
                    ),
                    reason="Use `is None` for None comparison.",
                )

            results = apply_ai_fixes(
                issues,
                target=path,
                reports_dir=Path(tmp) / "reports",
                proposer=surrounding_context_proposal,
            )

            self.assertTrue(results)
            self.assertTrue(results[0].accepted)

            fixed = Path(results[0].fixed_file).read_text(
                encoding="utf-8"
            )

            self.assertIn("if value is None:", fixed)
            self.assertIn("return True", fixed)
            self.assertNotIn("if value == None:", fixed)


def _good_none_proposal(issue, source_text) -> AIFixProposal:
    del issue, source_text
    return AIFixProposal(
        explanation="Use identity comparison for None.",
        original_code="if value == None:",
        corrected_code="if value is None:",
        reason="PEP 8 recommends `is None`.",
    )


def _syntax_error_proposal(issue, source_text) -> AIFixProposal:
    del issue, source_text
    return AIFixProposal(
        explanation="Broken on purpose for the rejection test.",
        original_code="if value == None:",
        corrected_code="if value is None",
        reason="This replacement is missing a colon.",
    )


if __name__ == "__main__":
    unittest.main()
