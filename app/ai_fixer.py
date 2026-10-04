"""Ask local Ollama for minimal issue fixes on a temporary copy.

Original source files are never rewritten. Each accepted fix is saved under
reports/fixes/<timestamp>/ai_fixes/ after syntax, Bandit, Black, and Pytest
checks succeed.
"""

from __future__ import annotations

import difflib
import os
import tempfile
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.analyzer import Issue
from app.fixer import (
    _compare_validation,
    _copied_path,
    _copy_project,
    _infer_target,
    _relative_name,
)
from app.ollama_reviewer import (
    AIFixProposal,
    OllamaError,
    parse_fix_response,
    propose_fix_with_ollama,
)
from app.quality import NESTED_FIX_VALIDATION_ENV, run_validation_suite

# Deterministic local --fix still owns unused-import. AI handles the rest of
# the beginner-friendly analyzer rules that need a small rewrite.
AI_FIX_RULES = {
    "mutable-default",
    "bare-except",
    "none-comparison",


}

ProposeFn = Callable[[Issue, str], AIFixProposal]


@dataclass
class AIFixResult:
    """One Ollama-proposed fix, accepted or rejected after validation."""

    file_path: str
    line: int
    rule: str
    issue: str
    original_code: str
    proposed_fix: str
    explanation: str
    reason: str
    validation_result: str
    accepted: bool
    status: str
    original_file: str
    fixed_file: str
    diff_path: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def plan_ai_fixes(issues: list[Issue]) -> list[Issue]:
    """Return analyzer issues this version will send to Ollama."""
    return [issue for issue in issues if issue.rule in AI_FIX_RULES]


def apply_ai_fixes(
    issues: list[Issue],
    *,
    target: str | Path | None = None,
    reports_dir: str | Path = "reports",
    model: str | None = None,
    proposer: ProposeFn | None = None,
) -> list[AIFixResult]:
    """Generate and validate Ollama fixes on a temp copy of the analyzed code.

    Pass ``proposer`` in tests to avoid calling a live Ollama server.
    """
    planned = plan_ai_fixes(issues)
    if not planned:
        return []
    if os.environ.get(NESTED_FIX_VALIDATION_ENV) == "1":
        return [
            _result_from_issue(
                issue,
                status="rejected",
                detail="Skipped nested --ai-fix while validation tests were running.",
                validation="Skipped nested AI fix.",
            )
            for issue in planned
        ]

    propose = proposer
    if propose is None:
        try:
            from app.ollama_reviewer import (
                DEFAULT_OLLAMA_HOST,
                DEFAULT_OLLAMA_MODEL,
                ensure_ollama_ready,
            )

            host = (os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
            ollama_model = (
                model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
            ).strip()
            ensure_ollama_ready(host, ollama_model)

            def propose(issue: Issue, source_text: str) -> AIFixProposal:
                snippet = _snippet_around(source_text, issue.line)
                return propose_fix_with_ollama(
                    issue,
                    source_text,
                    model=ollama_model,
                    snippet=snippet,
                )

        except OllamaError as exc:
            return [
                _result_from_issue(
                    issue,
                    status="rejected",
                    detail=str(exc),
                    validation="Ollama was not available, so the AI fix was not applied.",
                )
                for issue in planned
            ]

    source_root = Path(target).resolve() if target is not None else _infer_target(planned)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_root = Path(reports_dir) / "fixes" / stamp / "ai_fixes"

    with tempfile.TemporaryDirectory(prefix="ai-code-reviewer-ai-fix-") as tmp:
        work_root = Path(tmp) / "work"
        _copy_project(source_root, work_root)
        validation_target = (
            work_root
            if source_root.is_dir()
            else _copied_path(source_root, source_root, work_root)
        )
        baseline = run_validation_suite(validation_target)

        results: list[AIFixResult] = []
        for issue in planned:
            copied = _copied_path(Path(issue.file_path), source_root, work_root)
            original_text = copied.read_text(encoding="utf-8")
            results.append(
                _fix_one_issue(
                    issue=issue,
                    copied=copied,
                    original_text=original_text,
                    source_root=source_root,
                    validation_target=validation_target,
                    baseline=baseline,
                    output_root=output_root,
                    propose=propose,
                )
            )
            # Always restore the copy so the next issue starts from the original.
            copied.write_text(original_text, encoding="utf-8")
        return results


def _fix_one_issue(
    *,
    issue: Issue,
    copied: Path,
    original_text: str,
    source_root: Path,
    validation_target: Path,
    baseline,
    output_root: Path,
    propose: ProposeFn,
) -> AIFixResult:
    try:
        proposal = propose(issue, original_text)
    except (OllamaError, ValueError) as exc:
        return _result_from_issue(
            issue,
            status="rejected",
            detail=str(exc),
            validation="The AI fix was not applied because the proposal was invalid.",
        )

    ok, message, new_text = _apply_proposal(original_text, issue, proposal)
    if not ok:
        return _result_from_issue(
            issue,
            status="rejected",
            detail=message,
            validation=message,
            original_code=proposal.original_code,
            proposed_fix=proposal.corrected_code,
            explanation=proposal.explanation,
            reason=proposal.reason,
        )

    copied.write_text(new_text, encoding="utf-8")
    after = run_validation_suite(validation_target)
    valid, validation_summary = _compare_validation(baseline, after)
    if not valid:
        copied.write_text(original_text, encoding="utf-8")
        return _result_from_issue(
            issue,
            status="rejected",
            detail=validation_summary,
            validation=validation_summary,
            original_code=proposal.original_code,
            proposed_fix=proposal.corrected_code,
            explanation=proposal.explanation,
            reason=proposal.reason,
        )

    relative = _relative_name(Path(issue.file_path), source_root)
    stem = f"{relative.stem}__line{issue.line}_{issue.rule}"
    dest_dir = output_root / relative.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = dest_dir / f"{stem}{relative.suffix or '.py'}"
    diff_path = dest_dir / f"{stem}.diff"
    fixed_path.write_text(new_text, encoding="utf-8")
    diff_text = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(Path(issue.file_path)),
            tofile=f"{issue.file_path} (ai fix copy)",
        )
    )
    diff_path.write_text(diff_text, encoding="utf-8")

    return AIFixResult(
        original_file=issue.file_path,
        file_path=issue.file_path,
        line=issue.line,
        rule=issue.rule,
        issue=issue.message,
        original_code=proposal.original_code,
        proposed_fix=proposal.corrected_code,
        explanation=proposal.explanation,
        reason=proposal.reason,
        validation_result=validation_summary,
        accepted=True,
        status="accepted",
        fixed_file=str(fixed_path),
        diff_path=str(diff_path),
    )
        
       

def _apply_proposal(
    source: str,
    issue: Issue,
    proposal: AIFixProposal,
) -> tuple[bool, str, str]:
    """Replace original_code with corrected_code when the change looks related."""
    original_code = proposal.original_code.replace("\r\n", "\n")
    corrected_code = proposal.corrected_code.replace("\r\n", "\n")

    if original_code.strip() == corrected_code.strip():
        return False, "Corrected code is the same as the original code.", source

    replaced = _replace_near_line(
        source,
        original_code,
        corrected_code,
        issue.line,
    )

    if replaced is None:
        dedented = textwrap.dedent(original_code)

        if dedented != original_code:
            replaced = _replace_near_line(
                source,
                dedented,
                textwrap.dedent(corrected_code),
                issue.line,
            )

    if replaced is None:
        replaced = _replace_issue_line_from_context(
            source,
            original_code,
            corrected_code,
            issue.line,
        )

    if replaced is None:
        return (
            False,
            "original_code from the AI was not found in the source file.",
            source,
        )

    if not _covers_issue_line(
        source,
        issue.line,
        original_code,
    ):
        return (
            False,
            "The AI original_code does not include the reported issue line.",
            source,
        )

    if _looks_unrelated(
        source,
        replaced,
        original_code,
        corrected_code,
    ):
        return (
            False,
            "Rejected: the AI proposed unrelated changes beyond this issue.",
            source,
        )

    return True, "Applied the AI snippet on the working copy.", replaced  

    return True, "Applied the AI snippet on the working copy.", replaced


def _replace_issue_line_from_context(
    source: str,
    old: str,
    new: str,
    line: int,
) -> str | None:
    """Replace the issue line when AI returned surrounding context."""
    source_lines = source.splitlines(keepends=True)
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    if not old_lines or not new_lines:
        return None

    issue_index = line - 1

    if issue_index < 0 or issue_index >= len(source_lines):
        return None

    issue_text = source_lines[issue_index].strip()

    candidates = []

    for index, old_line in enumerate(old_lines):
        normalized_old = old_line.strip()

        if not normalized_old:
            continue

        if normalized_old == issue_text:
            candidates.append(index)
        elif (
            normalized_old.replace(" ", "")
            == issue_text.replace(" ", "")
        ):
            candidates.append(index)

    if len(candidates) != 1:
        return None

    old_index = candidates[0]

    if old_index >= len(new_lines):
        return None

    corrected_line = new_lines[old_index]

    if not corrected_line.strip():
        return None

    original_indent = source_lines[issue_index][
        :len(source_lines[issue_index])
        - len(source_lines[issue_index].lstrip())
    ]

    corrected_line = original_indent + corrected_line.lstrip()

    if (
        source_lines[issue_index].endswith("\n")
        and not corrected_line.endswith("\n")
    ):
        corrected_line += "\n"

    source_lines[issue_index] = corrected_line

    return "".join(source_lines)



def _replace_near_line(source: str, old: str, new: str, line: int) -> str | None:
    """Replace an exact AI snippet near the reported analyzer line."""
    candidates = [old]

    if old.endswith("\n"):
        candidates.append(old.rstrip("\n"))
    else:
        candidates.append(old + "\n")

    stripped = old.strip("\n")
    if stripped not in candidates:
        candidates.append(stripped)

    matches: list[tuple[int, str]] = []

    for needle in candidates:
        if not needle:
            continue

        start = 0
        while True:
            idx = source.find(needle, start)

            if idx == -1:
                break

            matches.append((idx, needle))
            start = idx + 1

    if not matches:
        return None

    def line_of(index: int) -> int:
        return source.count("\n", 0, index) + 1

    # Keep the match closest to the reported issue line.
    best_distance = min(
        abs(line_of(index) - line)
        for index, _ in matches
    )

    nearest = [
        (index, needle)
        for index, needle in matches
        if abs(line_of(index) - line) == best_distance
    ]

    # If multiple equally-near matches exist, reject the proposal
    # instead of guessing which occurrence the AI intended.
    if len(nearest) != 1:
        return None

    best_idx, best_needle = nearest[0]

    return (
        source[:best_idx]
        + new
        + source[best_idx + len(best_needle):]
    )

def _covers_issue_line(source: str, line: int, original_code: str) -> bool:
    lines = source.splitlines()
    if line < 1 or line > len(lines):
        return False
    issue_text = lines[line - 1].strip()
    if not issue_text:
        return True
    if issue_text in original_code:
        return True
    # Allow the model to quote the line without identical spacing.
    return issue_text.replace(" ", "") in original_code.replace(" ", "")


def _looks_unrelated(
    before: str,
    after: str,
    original_snippet: str,
    corrected_snippet: str,
) -> bool:
    if after == before:
        return True
    snippet_lines = max(len(original_snippet.splitlines()), 1)
    corrected_lines = max(len(corrected_snippet.splitlines()), 1)
    if corrected_lines > max(20, snippet_lines * 4):
        return True

    changed = 0
    for line in difflib.unified_diff(
        before.splitlines(), after.splitlines(), lineterm=""
    ):
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed += 1
    if changed > max(16, snippet_lines * 6):
        return True

    ratio = difflib.SequenceMatcher(None, before, after).ratio()
    if len(before) > 80 and ratio < 0.82:
        return True
    return False


def _snippet_around(source: str, line: int, radius: int = 12) -> str:
    lines = source.splitlines(keepends=True)
    if not lines:
        return source
    start = max(0, line - 1 - radius)
    end = min(len(lines), line + radius)
    return "".join(lines[start:end])


def _result_from_issue(
    issue: Issue,
    *,
    status: str,
    detail: str,
    validation: str,
    original_code: str = "",
    proposed_fix: str = "",
    explanation: str = "",
    reason: str = "",
    fixed_file: str = "",
    diff_path: str = "",
) -> AIFixResult:
    return AIFixResult(
        original_file=issue.file_path,
        file_path=issue.file_path,
        line=issue.line,
        rule=issue.rule,
        issue=issue.message,
        original_code=original_code,
        proposed_fix=proposed_fix,
        explanation=explanation,
        reason=reason,
        validation_result=validation,
        accepted=False,
        status=status,
        fixed_file=fixed_file,
        diff_path=diff_path,
    )

# Re-export for tests that parse JSON without importing ollama_reviewer.
parse_ai_fix_response = parse_fix_response
