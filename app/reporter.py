"""Save analysis and review results as Markdown and JSON files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.ai_fixer import AIFixResult
from app.ai_reviewer import ReviewComment
from app.analyzer import Issue
from app.fixer import FixResult
from app.quality import ToolResult


def write_reports(
    issues: list[Issue],
    comments: list[ReviewComment],
    fixes: list[FixResult],
    output_dir: str | Path = "reports",
    *,
    quality_results: list[ToolResult] | None = None,
    ai_error: str | None = None,
    provider: str = "ollama",
    ai_fixes: list[AIFixResult] | None = None,
    ai_fix_error: str | None = None,
) -> Path:
    """Write a timestamped report folder and return the Markdown path."""
    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    markdown_path = reports_dir / f"review-{stamp}.md"
    json_path = reports_dir / f"review-{stamp}.json"

    markdown_path.write_text(
        _render_markdown(
            issues,
            comments,
            fixes,
            quality_results or [],
            ai_error,
            provider,
            ai_fixes,
            ai_fix_error,
        ),
        encoding="utf-8",
    )
    payload = {
        "generated_at": stamp,
        "provider": provider,
        "ai_error": ai_error,
        "issues": [issue.to_dict() for issue in issues],
        "comments": [comment.to_dict() for comment in comments],
        "fixes": [fix.to_dict() for fix in fixes],
        "quality": [item.to_dict() for item in (quality_results or [])],
    }
    # Only add AI-fix fields when --ai-fix ran, so default reports stay the same.
    if ai_fixes is not None:
        payload["ai_fixes"] = [item.to_dict() for item in ai_fixes]
        payload["ai_fix_error"] = ai_fix_error
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return markdown_path


def _render_markdown(
    issues: list[Issue],
    comments: list[ReviewComment],
    fixes: list[FixResult],
    quality_results: list[ToolResult],
    ai_error: str | None,
    provider: str,
    ai_fixes: list[AIFixResult] | None = None,
    ai_fix_error: str | None = None,
) -> str:
    lines = [
        "# Code Review Report",
        "",
        f"- Issues found: **{len(issues)}**",
        f"- Review comments: **{len(comments)}**",
        f"- Fix actions: **{len(fixes)}**",
        "",
        "## Findings",
        "",
    ]

    if not issues:
        lines.append("No issues found.")
        lines.append("")
    else:
        for issue in issues:
            lines.extend(
                [
                    f"### `{issue.file_path}:{issue.line}` ({issue.severity})",
                    "",
                    f"- Rule: `{issue.rule}`",
                    f"- {issue.message}",
                    f"- Suggestion: {issue.suggestion}",
                    "",
                ]
            )

    lines.extend(["## Review comments", ""])
    if not comments:
        lines.append("No review comments.")
        lines.append("")
    else:
        for comment in comments:
            lines.extend(
                [
                    f"### {comment.title} (`{comment.file_path}:{comment.line}`)",
                    "",
                    comment.body,
                    "",
                ]
            )

    lines.extend(["## Bandit, Black, and Pytest", ""])
    if not quality_results:
        lines.append("Quality tools were not run.")
        lines.append("")
    else:
        for tool in quality_results:
            status = "ok" if tool.ok else "needs attention"
            lines.extend(
                [
                    f"### {tool.name} ({status})",
                    "",
                    tool.summary,
                    "",
                ]
            )
            if tool.output:
                lines.extend(["```", tool.output[:4000], "```", ""])

    lines.extend([f"## AI review ({provider})", ""])
    if ai_error:
        lines.extend(
            [
                "The AI reviewer did not run.",
                "",
                "```",
                ai_error,
                "```",
                "",
            ]
        )
    elif comments:
        lines.append("The AI review is included in **Review comments** above.")
        lines.append("")
    else:
        lines.append("No AI review comments.")
        lines.append("")

    lines.extend(["## Fixes", ""])
    if not fixes:
        lines.append("No automatic fixes were planned.")
        lines.append("")
    else:
        for fix in fixes:
            status = fix.status or ("applied" if fix.applied else "planned")
            original = fix.original_file or fix.file_path
            lines.extend(
                [
                    f"### `{original}:{fix.line}` `{fix.rule}` ({status})",
                    "",
                    f"- Original file: `{original}`",
                    f"- Fixed file: `{fix.fixed_file or '(not saved)'}`",
                    f"- Rule: `{fix.rule}`",
                    f"- Line: {fix.line}",
                    f"- Change made: {fix.change_made or fix.detail}",
                    f"- Validation result: {fix.validation_result or 'n/a'}",
                    f"- Status: {status}",
                    "",
                ]
            )

    if ai_fixes is not None:
        lines.extend(["## AI-generated fixes", ""])
        if ai_fix_error:
            lines.extend(["```", ai_fix_error, "```", ""])
        if not ai_fixes:
            lines.append("No AI-generated fixes were proposed.")
            lines.append("")
        else:
            for item in ai_fixes:
                lines.extend(
                    [
                        f"### `{item.file_path}:{item.line}` `{item.rule}` ({item.status})",
                        "",
                        f"- Issue: {item.issue}",
                        f"- Original file: `{item.original_file}`",
                        f"- Fixed file: `{item.fixed_file or '(not saved)'}`",
                        f"- Explanation: {item.explanation or item.detail}",
                        f"- Reason: {item.reason or 'n/a'}",
                        f"- Validation result: {item.validation_result or 'n/a'}",
                        f"- Status: {item.status}",
                        "",
                        "Original code:",
                        "",
                        "```python",
                        item.original_code or "(not provided)",
                        "```",
                        "",
                        "Proposed fix:",
                        "",
                        "```python",
                        item.proposed_fix or "(not provided)",
                        "```",
                        "",
                    ]
                )

    return "\n".join(lines)
