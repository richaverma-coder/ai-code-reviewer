"""Run a local Python code review from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.ai_fixer import apply_ai_fixes
from app.ai_reviewer import ReviewError, review_issues
from app.analyzer import analyze_path
from app.fixer import apply_fixes
from app.quality import run_quality_tools
from app.reporter import write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local Python code and write a review report."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="sample_code",
        help="File or folder to analyze (default: sample_code)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe unused-import fixes to a working copy (never writes originals).",
    )
    parser.add_argument(
        "--ai-fix",
        action="store_true",
        help=(
            "Ask local Ollama for minimal fixes on a working copy. "
            "Never writes original files. Requires --provider ollama."
        ),
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Folder where Markdown and JSON reports are saved.",
    )
    parser.add_argument(
        "--provider",
        choices=("ollama", "openai", "none"),
        default="ollama",
        help="AI reviewer to use (default: ollama, no paid API key).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model name (default: llama3.2 or OLLAMA_MODEL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = Path(args.path)

    issues = analyze_path(target)
    quality_results = run_quality_tools(target)

    ai_error = None
    try:
        comments = review_issues(issues, provider=args.provider, model=args.model)
    except ReviewError as exc:
        ai_error = str(exc)
        comments = review_issues(issues, provider="none")
        print(f"AI review skipped: {exc}")

    # Original source files are never rewritten. --fix writes a validated copy
    # under reports/fixes/ only when checks do not regress.
    fixes = apply_fixes(
        issues,
        write_files=args.fix,
        target=target,
        reports_dir=args.reports_dir,
    )

    # --ai-fix is optional and local-only. Skip it unless the user asked.
    ai_fixes = None
    ai_fix_error = None
    if args.ai_fix:
        if args.provider != "ollama":
            ai_fix_error = (
                "--ai-fix only works with the local Ollama provider "
                "(--provider ollama). No OpenAI API is used."
            )
            print(ai_fix_error)
            ai_fixes = []
        else:
            ai_fixes = apply_ai_fixes(
                issues,
                target=target,
                reports_dir=args.reports_dir,
                model=args.model,
            )

    report_path = write_reports(
        issues,
        comments,
        fixes,
        args.reports_dir,
        quality_results=quality_results,
        ai_error=ai_error,
        provider=args.provider,
        ai_fixes=ai_fixes,
        ai_fix_error=ai_fix_error,
    )

    print(f"Analyzed: {target.resolve()}")
    print(f"Issues found: {len(issues)}")
    print(f"Review comments: {len(comments)}")
    print(f"Fix actions: {len(fixes)}")
    if ai_fixes is not None:
        print(f"AI fix actions: {len(ai_fixes)}")
    print(f"Report: {report_path}")

    if issues:
        print("\nTop findings:")
        for issue in issues[:10]:
            print(f"- {issue.file_path}:{issue.line} [{issue.rule}] {issue.message}")


if __name__ == "__main__":
    main()
