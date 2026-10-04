"""Turn analyzer findings into a readable code review.

The default path uses a local Ollama model and does not need OPENAI_API_KEY.
OpenAI remains available as an optional provider. If no source files can be
loaded, comments are built from local static-analysis findings so existing
tests keep working.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
    RateLimitError,
)
from pydantic import BaseModel, Field

from app.analyzer import Issue

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
_ALLOWED_SEVERITIES = ("low", "medium", "high", "critical")
_ALLOWED_TYPES = (
    "bug",
    "security",
    "code_smell",
    "performance",
    "maintainability",
)
_DEFAULT_MODEL = "gpt-4o-mini"

Severity = Literal["low", "medium", "high", "critical"]
IssueType = Literal[
    "bug",
    "security",
    "code_smell",
    "performance",
    "maintainability",
]


class ReviewError(RuntimeError):
    """Raised when the AI review cannot be completed."""


class ReviewedIssue(BaseModel):
    file: str
    line: int = Field(ge=1)
    type: IssueType
    severity: Severity
    description: str
    evidence: str
    recommendation: str


class CodeReview(BaseModel):
    summary: str
    issues: list[ReviewedIssue]


@dataclass
class ReviewComment:
    """A review note that a human (or later, an LLM) could post."""

    file_path: str
    line: int
    severity: str
    title: str
    body: str
    type: str = ""
    description: str = ""
    evidence: str = ""
    recommendation: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def review_issues(
    issues: list[Issue],
    source_code: str | dict[str, str] | None = None,
    *,
    provider: str = "ollama",
    model: str | None = None,
) -> list[ReviewComment]:
    """Build review comments from static analysis issues.

    ``provider`` is ``ollama`` by default (no OpenAI billing). Use ``openai``
    only when you explicitly want the cloud API. Use ``none`` for local
    comments built from analyzer findings.
    """
    sources = _resolve_sources(issues, source_code)
    if not sources or provider == "none":
        return [_local_comment(issue) for issue in issues]

    if provider == "openai":
        review = review_python_code(sources, issues)
    elif provider == "ollama":
        from app.ollama_reviewer import review_with_ollama

        review = review_with_ollama(sources, issues, model=model)
    else:
        raise ReviewError(
            f"Unknown review provider '{provider}'. Use ollama, openai, or none."
        )
    return _review_to_comments(review)


def review_python_code(
    source_code: str | dict[str, str],
    findings: Iterable[Issue | dict[str, Any]] | None = None,
    *,
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Review Python source plus static-analysis findings.

    Returns structured JSON-compatible data with ``summary`` and ``issues``.
    Never prints or returns the API key.
    """
    sources = _normalize_sources(source_code)
    if not sources:
        raise ReviewError("No Python source code was provided for review.")

    payload = {
        "static_analysis_findings": _findings_payload(findings or []),
        "source_files": {
            path: _with_line_numbers(text) for path, text in sources.items()
        },
    }

    openai_client = client or _create_client()
    model = os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL

    try:
        response = openai_client.responses.parse(
            model=model,
            instructions=_SYSTEM_INSTRUCTIONS,
            input=json.dumps(payload, indent=2),
            text_format=CodeReview,
        )
    except AuthenticationError as exc:
        raise ReviewError(
            "OpenAI authentication failed. Check that OPENAI_API_KEY is set "
            "correctly in the project .env file."
        ) from _without_secret_context(exc)
    except RateLimitError as exc:
        raise ReviewError(
            "OpenAI rate limit exceeded. Wait and retry the review."
        ) from _without_secret_context(exc)
    except APITimeoutError as exc:
        raise ReviewError("The OpenAI request timed out. Retry the review.") from _without_secret_context(
            exc
        )
    except APIConnectionError as exc:
        raise ReviewError(
            "Could not connect to the OpenAI API. Check the network connection."
        ) from _without_secret_context(exc)
    except APIStatusError as exc:
        status = getattr(exc, "status_code", "unknown")
        raise ReviewError(
            f"OpenAI API returned HTTP {status}. {_safe_error_text(exc)}"
        ) from _without_secret_context(exc)
    except OpenAIError as exc:
        raise ReviewError(
            f"OpenAI request failed. {_safe_error_text(exc)}"
        ) from _without_secret_context(exc)
    except Exception as exc:  # noqa: BLE001 - surface unexpected SDK failures clearly
        raise ReviewError(
            f"Unexpected error while requesting an AI review. {_safe_error_text(exc)}"
        ) from _without_secret_context(exc)

    parsed = response.output_parsed
    if parsed is None:
        parsed = _parse_output_text(response.output_text)

    review = parsed.model_dump()
    return _filter_unsupported_issues(review, sources)


def _create_client() -> OpenAI:
    """Load `.env` and build a client from OPENAI_API_KEY in the environment."""
    load_dotenv(_ENV_PATH)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not str(api_key).strip():
        raise ReviewError(
            "OPENAI_API_KEY is missing or empty. Set it in the project .env file."
        )
    # OpenAI() reads OPENAI_API_KEY from the environment; do not log the value.
    return OpenAI()


def _resolve_sources(
    issues: list[Issue],
    source_code: str | dict[str, str] | None,
) -> dict[str, str]:
    if source_code:
        return _normalize_sources(source_code)

    sources: dict[str, str] = {}
    for issue in issues:
        path = Path(issue.file_path)
        if not path.is_file():
            continue
        key = str(path)
        if key not in sources:
            sources[key] = path.read_text(encoding="utf-8")
    return sources


def _normalize_sources(source_code: str | dict[str, str]) -> dict[str, str]:
    if isinstance(source_code, dict):
        return {str(path): text for path, text in source_code.items() if text}
    if isinstance(source_code, str) and source_code.strip():
        return {"input.py": source_code}
    return {}


def _findings_payload(findings: Iterable[Issue | dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in findings:
        if isinstance(item, Issue):
            payload.append(
                {
                    "file": item.file_path,
                    "line": item.line,
                    "rule": item.rule,
                    "severity": item.severity,
                    "message": item.message,
                    "suggestion": item.suggestion,
                }
            )
        elif isinstance(item, dict):
            payload.append(item)
    return payload


def _with_line_numbers(source: str) -> str:
    lines = source.splitlines()
    width = len(str(max(len(lines), 1)))
    return "\n".join(f"{index:>{width}}| {line}" for index, line in enumerate(lines, start=1))


def _parse_output_text(output_text: str) -> CodeReview:
    try:
        return CodeReview.model_validate_json(output_text)
    except Exception as exc:  # noqa: BLE001
        raise ReviewError(
            "OpenAI returned a response that was not valid review JSON."
        ) from _without_secret_context(exc)


def _filter_unsupported_issues(
    review: dict[str, Any],
    sources: dict[str, str],
) -> dict[str, Any]:
    known_files = set(sources)
    known_basenames = {Path(path).name for path in sources}
    source_text = "\n".join(sources.values())

    kept: list[dict[str, Any]] = []
    for issue in review.get("issues", []):
        file_name = str(issue.get("file") or "")
        description = str(issue.get("description") or "").strip()
        evidence = str(issue.get("evidence") or "").strip()
        if not description or not evidence:
            continue
        if not _file_is_known(file_name, known_files, known_basenames):
            continue
        if not _evidence_supported(evidence, source_text):
            continue
        issue["severity"] = _normalize_severity(issue.get("severity"))
        issue["type"] = _normalize_type(issue.get("type"))
        issue["line"] = max(int(issue.get("line") or 1), 1)
        issue["description"] = description
        issue["evidence"] = evidence
        issue["recommendation"] = str(issue.get("recommendation") or "").strip()
        kept.append(issue)

    return {
        "summary": str(review.get("summary") or "").strip(),
        "issues": kept,
    }


def _file_is_known(file_name: str, known_files: set[str], known_basenames: set[str]) -> bool:
    if not file_name:
        return False
    if file_name in known_files:
        return True
    return Path(file_name).name in known_basenames


def _evidence_supported(evidence: str, source_text: str) -> bool:
    snippet = evidence.strip()
    if not snippet:
        return False
    if snippet in source_text:
        return True
    # Allow short quoted fragments or line references when the full sentence
    # is commentary around real code tokens already present in the source.
    tokens = [part.strip("`'\"") for part in snippet.split() if len(part.strip("`'\" ")) >= 4]
    if not tokens:
        return False
    matches = sum(1 for token in tokens if token in source_text)
    return matches >= max(1, len(tokens) // 3)


def _normalize_severity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in _ALLOWED_SEVERITIES:
        return text
    return "medium"


def _normalize_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "bugs": "bug",
        "security_vulnerability": "security",
        "security_vulnerabilities": "security",
        "code_quality": "code_smell",
        "quality": "code_smell",
        "performance_problem": "performance",
        "performance_problems": "performance",
        "maintainability_problem": "maintainability",
        "maintainability_problems": "maintainability",
    }
    text = aliases.get(text, text)
    if text in _ALLOWED_TYPES:
        return text
    return "code_smell"


def _review_to_comments(review: dict[str, Any]) -> list[ReviewComment]:
    summary = str(review.get("summary") or "").strip()
    comments: list[ReviewComment] = []
    for index, issue in enumerate(review.get("issues", [])):
        body_parts = [
            issue.get("description") or "",
            f"**Evidence:** {issue.get('evidence') or ''}",
            f"**Recommendation:** {issue.get('recommendation') or ''}",
        ]
        if index == 0 and summary:
            body_parts.insert(0, f"**Summary:** {summary}")
        comments.append(
            ReviewComment(
                file_path=str(issue.get("file") or ""),
                line=int(issue.get("line") or 1),
                severity=str(issue.get("severity") or "medium"),
                title=str(issue.get("type") or "review").replace("_", " ").title(),
                body="\n\n".join(part for part in body_parts if part),
                type=str(issue.get("type") or ""),
                description=str(issue.get("description") or ""),
                evidence=str(issue.get("evidence") or ""),
                recommendation=str(issue.get("recommendation") or ""),
                summary=summary,
            )
        )
    return comments


def _local_comment(issue: Issue) -> ReviewComment:
    return ReviewComment(
        file_path=issue.file_path,
        line=issue.line,
        severity=issue.severity,
        title=_title_for_rule(issue.rule),
        body=_comment_body(issue),
        type=issue.rule,
        description=issue.message,
        recommendation=issue.suggestion,
    )


def _title_for_rule(rule: str) -> str:
    titles = {
        "syntax-error": "Syntax error",
        "bare-except": "Bare except clause",
        "mutable-default": "Mutable default argument",
        "none-comparison": "Compare with None using `is`",
        "dangerous-call": "Risky dynamic execution",
        "unused-import": "Unused import",
        "unused-variable": "Unused variable",
    }
    return titles.get(rule, rule.replace("-", " ").title())


def _comment_body(issue: Issue) -> str:
    return (
        f"{issue.message}\n\n"
        f"**Suggestion:** {issue.suggestion}\n\n"
        f"_Rule: `{issue.rule}`_"
    )


def _safe_error_text(exc: BaseException) -> str:
    text = str(exc)
    secret = os.environ.get("OPENAI_API_KEY")
    if secret:
        text = text.replace(str(secret), "[redacted]")
    return text.strip() or exc.__class__.__name__


def _without_secret_context(exc: BaseException) -> None:
    """Avoid chaining SDK exceptions that might include request secrets."""
    del exc
    return None


_SYSTEM_INSTRUCTIONS = """
You are a senior Python code reviewer.
Review only the supplied Python source code. Static-analysis findings are hints,
not extra facts: ignore any finding you cannot confirm in the source.

Look for:
- bugs
- security vulnerabilities
- code quality issues
- performance issues
- maintainability issues

Rules:
- Do not invent issues that are not supported by the supplied code.
- Every issue must cite evidence copied from or clearly present in the source.
- Use only these type values: bug, security, code_smell, performance, maintainability.
- Use only these severity values: low, medium, high, critical.
- If there are no supported issues, return an empty issues list and a short summary.
""".strip()
