"""Suggest and optionally apply simple, safe local fixes.

This version only removes unused import lines, and only on a temporary copy
of the analyzed project. Original source files are never rewritten.
"""

from __future__ import annotations

import ast
import difflib
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.analyzer import Issue
from app.quality import (
    NESTED_FIX_VALIDATION_ENV,
    ToolResult,
    count_bandit_findings,
    run_validation_suite,
)

SAFE_FIX_RULES = {"unused-import", "unused-variable"}
_UNUSED_IMPORT_NAME = re.compile(r"Imported name `([^`]+)`")
_SKIP_COPY_NAMES = {
    ".git",
    "venv",
    ".venv",
    "__pycache__",
    "site-packages",
    ".pytest_cache",
    "reports",
}


@dataclass
class FixResult:
    """What happened when we tried to fix one issue."""

    file_path: str
    line: int
    rule: str
    applied: bool
    detail: str
    original_file: str = ""
    fixed_file: str = ""
    change_made: str = ""
    validation_result: str = ""
    accepted: bool = False
    status: str = "planned"

    def to_dict(self) -> dict:
        return asdict(self)


def plan_fixes(issues: list[Issue]) -> list[Issue]:
    """Return issues that this version knows how to fix automatically."""
    return [issue for issue in issues if issue.rule in SAFE_FIX_RULES]


def apply_fixes(
    issues: list[Issue],
    write_files: bool = False,
    *,
    target: str | Path | None = None,
    reports_dir: str | Path = "reports",
) -> list[FixResult]:
    """Plan or apply safe deterministic fixes one at a time.

    Original files are never modified. Each fix is applied to a temporary
    working copy and validated independently.
    """
    planned = plan_fixes(issues)

    if not write_files:
        return [_planned_result(issue) for issue in planned]

    if not planned:
        return []

    if os.environ.get(NESTED_FIX_VALIDATION_ENV) == "1":
        return [
            _planned_result(
                issue,
                detail="Skipped nested --fix validation.",
            )
            for issue in planned
        ]

    source_root = (
        Path(target).resolve()
        if target is not None
        else _infer_target(planned)
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_root = Path(reports_dir) / "fixes" / stamp

    with tempfile.TemporaryDirectory(
        prefix="ai-code-reviewer-fix-"
    ) as tmp:
        work_root = Path(tmp) / "work"
        _copy_project(source_root, work_root)

        validation_target = (
            work_root
            if source_root.is_dir()
            else _copied_path(
                source_root,
                source_root,
                work_root,
            )
        )

        before = run_validation_suite(validation_target)

        per_issue: list[_LineFix] = []
        originals: dict[str, str] = {}

        grouped: dict[str, list[Issue]] = {}
        for issue in planned:
            grouped.setdefault(issue.file_path, []).append(issue)

        for file_path, file_issues in grouped.items():
            original_path = Path(file_path)
            copied = _copied_path(
                original_path,
                source_root,
                work_root,
            )

            originals[file_path] = copied.read_text(
                encoding="utf-8"
            )

            # Apply unused-variable before unused-import so
            # original line numbers stay stable.
            ordered_issues = sorted(
                file_issues,
                key=lambda item: (
                    0 if item.rule == "unused-variable" else 1,
                    item.line,
                ),
            )

            for issue in ordered_issues:
                current_source = copied.read_text(
                    encoding="utf-8"
                )

                if issue.rule == "unused-variable":
                    result = _apply_unused_variables(
                        copied,
                        [issue],
                    )
                elif issue.rule == "unused-import":
                    result = _apply_unused_imports(
                        copied,
                        [issue],
                    )
                else:
                    result = [
                        _LineFix(
                            issue=issue,
                            verified=False,
                            change_made="",
                            reason=(
                                f"Unsupported safe-fix rule: "
                                f"{issue.rule}"
                            ),
                        )
                    ]

                item = result[0]

                if not item.verified:
                    per_issue.append(item)
                    copied.write_text(
                        current_source,
                        encoding="utf-8",
                    )
                    continue

                after = run_validation_suite(
                    validation_target
                )

                ok, validation_summary = _compare_validation(
                    before,
                    after,
                )

                if ok:
                    per_issue.append(item)
                    before = after
                else:
                    per_issue.append(
                        _LineFix(
                            issue=issue,
                            verified=True,
                            change_made=item.change_made,
                            reason=validation_summary,
                        )
                    )

                    # Reject this fix and restore the previous state.
                    copied.write_text(
                        current_source,
                        encoding="utf-8",
                    )

        after = run_validation_suite(validation_target)

        verified_any = any(
            item.verified
            and not item.reason.startswith("Rejected:")
            for item in per_issue
        )

        validation_summary = (
            "Individual fix validation completed. "
            "Accepted fixes were kept; rejected fixes were restored."
        )

        return _save_outputs(
            per_issue=per_issue,
            originals=originals,
            source_root=source_root,
            work_root=work_root,
            output_root=output_root,
            batch_accepted=verified_any,
            validation_summary=validation_summary,
            before=before,
            after=after,
        )


def _planned_result(issue: Issue, detail: str | None = None) -> FixResult:
    text = detail or "Would remove unused import line. Re-run with --fix to apply."
    return FixResult(
        file_path=issue.file_path,
        line=issue.line,
        rule=issue.rule,
        applied=False,
        detail=text,
        original_file=issue.file_path,
        fixed_file="",
        change_made="Would remove unused import line.",
        validation_result="Not applied. Validation runs only when --fix is used.",
        accepted=False,
        status="planned",
    )


@dataclass
class _LineFix:
    issue: Issue
    verified: bool
    change_made: str
    reason: str


def _apply_unused_imports(path: Path, file_issues: list[Issue]) -> list[_LineFix]:
    original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    unused_by_line: dict[int, set[str]] = {}
    for issue in file_issues:
        name = _unused_name(issue)
        if name:
            unused_by_line.setdefault(issue.line, set()).add(name)

    results: list[_LineFix] = []
    lines_to_drop: set[int] = set()
    for issue in file_issues:
        ok, change, reason = _verify_unused_import_line(
            original_lines, issue, unused_by_line.get(issue.line, set())
        )
        if ok:
            lines_to_drop.add(issue.line)
        results.append(
            _LineFix(issue=issue, verified=ok, change_made=change, reason=reason)
        )

    if lines_to_drop:
        new_lines = [
            line
            for index, line in enumerate(original_lines, start=1)
            if index not in lines_to_drop
        ]
        while new_lines and new_lines[0].strip() == "":
            new_lines.pop(0)
        path.write_text("".join(new_lines), encoding="utf-8")
    return results


def _verify_unused_import_line(
    lines: list[str],
    issue: Issue,
    unused_names_on_line: set[str],
) -> tuple[bool, str, str]:
    if issue.line < 1 or issue.line > len(lines):
        return False, "", f"Line {issue.line} is outside the file."

    raw = lines[issue.line - 1]
    stripped = raw.strip()
    is_import, names = _import_names(stripped)
    if not is_import:
        return (
            False,
            "",
            f"Line {issue.line} is not an import statement; left unchanged.",
        )

    expected = _unused_name(issue)
    if expected and expected not in names:
        return (
            False,
            "",
            f"Line {issue.line} does not import `{expected}`; left unchanged.",
        )

    leftover = names - unused_names_on_line
    if leftover:
        leftover_txt = ", ".join(sorted(leftover))
        return (
            False,
            "",
            f"Line {issue.line} also imports {leftover_txt}; not removing the whole line.",
        )

    return True, f"Remove unused import line: {stripped}", ""


def _import_names(line: str) -> tuple[bool, set[str]]:
    if not line or line.startswith("#"):
        return False, set()
    try:
        tree = ast.parse(line)
    except SyntaxError:
        return False, set()
    names: set[str] = set()
    found = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            found = True
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            found = True
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
    return found, names


def _unused_name(issue: Issue) -> str | None:
    match = _UNUSED_IMPORT_NAME.search(issue.message)
    return match.group(1) if match else None


def _copy_project(source_root: Path, work_root: Path) -> None:
    if source_root.is_file():
        work_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root, work_root / source_root.name)
        return
    shutil.copytree(source_root, work_root, ignore=_copy_ignore, dirs_exist_ok=True)


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    del directory
    return {name for name in names if name in _SKIP_COPY_NAMES}


def _copied_path(original: Path, source_root: Path, work_root: Path) -> Path:
    orig = original.resolve()
    root = source_root.resolve()
    if root.is_file():
        return work_root / root.name
    try:
        return work_root / orig.relative_to(root)
    except ValueError:
        return work_root / orig.name


def _infer_target(issues: list[Issue]) -> Path:
    paths = [Path(issue.file_path).resolve() for issue in issues]
    if len(paths) == 1:
        return paths[0]
    return Path(os.path.commonpath([str(path) for path in paths]))


def _compare_validation(
    before: list[ToolResult], after: list[ToolResult]
) -> tuple[bool, str]:
    by_before = {item.name: item for item in before}
    by_after = {item.name: item for item in after}
    notes: list[str] = []

    syntax = by_after.get("Syntax")
    if syntax is None or not syntax.ok:
        detail = syntax.summary if syntax else "Syntax validation did not run."
        return False, f"Rejected: {detail}"

    bandit_before = by_before.get("Bandit")
    bandit_after = by_after.get("Bandit")
    if bandit_after and not bandit_after.available:
        notes.append("Bandit was not available after the fix.")
    elif bandit_before and bandit_after:
        before_count = count_bandit_findings(bandit_before)
        after_count = count_bandit_findings(bandit_after)
        if (
            after_count is not None
            and before_count is not None
            and after_count > before_count
        ):
            return (
                False,
                f"Rejected: Bandit findings increased from {before_count} to {after_count}.",
            )
        if bandit_before.ok and not bandit_after.ok and after_count is None:
            return False, "Rejected: Bandit passed before the fix but failed after."
        if after_count is not None:
            notes.append(f"Bandit findings: {before_count} before, {after_count} after.")

    for name in ("UndefinedNames", "Black", "Pytest"):
        prior = by_before.get(name)
        later = by_after.get(name)
        if prior and later and prior.ok and not later.ok:
            return False, f"Rejected: {name} passed before the fix but failed after."
        if later:
            notes.append(later.summary)

    notes.insert(0, "Python syntax validation passed.")
    return True, " ".join(notes)


def _save_outputs(
    *,
    per_issue: list[_LineFix],
    originals: dict[str, str],
    source_root: Path,
    work_root: Path,
    output_root: Path,
    batch_accepted: bool,
    validation_summary: str,
    before: list[ToolResult],
    after: list[ToolResult],
) -> list[FixResult]:
    del before, after
    results: list[FixResult] = []
    files_dir = output_root / ("files" if batch_accepted else "rejected")
    patches_dir = output_root / "patches"

    for item in per_issue:
        issue = item.issue
        copied = _copied_path(Path(issue.file_path), source_root, work_root)
        original_text = originals.get(issue.file_path, "")
        new_text = copied.read_text(encoding="utf-8") if copied.exists() else original_text
        relative = _relative_name(Path(issue.file_path), source_root)

        accepted = batch_accepted and item.verified
        saved_file = ""
        if item.verified and original_text != new_text:
            files_dir.mkdir(parents=True, exist_ok=True)
            patches_dir.mkdir(parents=True, exist_ok=True)
            dest = files_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(new_text, encoding="utf-8")
            patch_path = patches_dir / f"{relative}.diff"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(
                "".join(
                    difflib.unified_diff(
                        original_text.splitlines(keepends=True),
                        new_text.splitlines(keepends=True),
                        fromfile=str(Path(issue.file_path)),
                        tofile=f"{issue.file_path} (fixed copy)",
                    )
                ),
                encoding="utf-8",
            )
            saved_file = str(dest)

        if accepted:
            status = "accepted"
            detail = item.change_made
        elif not item.verified:
            status = "rejected"
            detail = item.reason or "Import line verification failed."
        else:
            status = "rejected"
            detail = validation_summary

        results.append(
            FixResult(
                file_path=issue.file_path,
                line=issue.line,
                rule=issue.rule,
                applied=accepted,
                detail=detail,
                original_file=issue.file_path,
                fixed_file=saved_file,
                change_made=item.change_made or detail,
                validation_result=validation_summary,
                accepted=accepted,
                status=status,
            )
        )
    return results


def _relative_name(original: Path, source_root: Path) -> Path:
    orig = original.resolve()
    root = source_root.resolve()
    if root.is_file():
        return Path(root.name)
    try:
        return orig.relative_to(root)
    except ValueError:
        return Path(orig.name)

def _apply_unused_variables(path: Path, file_issues: list[Issue]) -> list[_LineFix]:
    """Remove only the exact unused-variable assignment line."""
    original_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    results: list[_LineFix] = []
    lines_to_drop: set[int] = set()

    for issue in file_issues:
        if issue.line < 1 or issue.line > len(original_lines):
            results.append(
                _LineFix(
                    issue=issue,
                    verified=False,
                    change_made="",
                    reason=f"Line {issue.line} is outside the file.",
                )
            )
            continue

        raw = original_lines[issue.line - 1]
        stripped = raw.strip()

        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            results.append(
                _LineFix(
                    issue=issue,
                    verified=False,
                    change_made="",
                    reason=f"Line {issue.line} is not a valid Python statement.",
                )
            )
            continue

        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
            results.append(
                _LineFix(
                    issue=issue,
                    verified=False,
                    change_made="",
                    reason=f"Line {issue.line} is not a simple assignment; left unchanged.",
                )
            )
            continue

        assignment = tree.body[0]
        if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
            results.append(
                _LineFix(
                    issue=issue,
                    verified=False,
                    change_made="",
                    reason=f"Line {issue.line} is not a simple variable assignment; left unchanged.",
                )
            )
            continue

        lines_to_drop.add(issue.line)
        results.append(
            _LineFix(
                issue=issue,
                verified=True,
                change_made=f"Remove unused variable assignment: {stripped}",
                reason="The analyzer identified this assignment as unused.",
            )
        )

    if lines_to_drop:
        new_lines = [
            line
            for index, line in enumerate(original_lines, start=1)
            if index not in lines_to_drop
        ]
        path.write_text("".join(new_lines), encoding="utf-8")

    return results

