"""Review Python code with a local Ollama model. No OpenAI API key is required."""

from __future__ import annotations

import json
import re
import os
import urllib.error
import urllib.request
from typing import Any, Iterable

from dataclasses import dataclass

from app.ai_reviewer import (
    CodeReview,
    ReviewError,
    _filter_unsupported_issues,
    _findings_payload,
    _normalize_sources,
    _SYSTEM_INSTRUCTIONS,
    _with_line_numbers,
)
from app.analyzer import Issue

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
_INSTALL_HINT = (
    "Install Ollama from https://ollama.com, start the Ollama app, "
    "then download a model with: ollama pull llama3.2"
)


_FIX_SYSTEM_INSTRUCTIONS = """
You generate a minimal Python fix for ONE static-analysis issue.
Return JSON only with exactly these keys:
explanation, original_code, corrected_code, reason.

Rules:
- original_code must be copied exactly from the provided source (the text to replace).
- Never include line numbers, line-number prefixes, or source annotations such as "1|", "2|", etc. in original_code or corrected_code.
- corrected_code must change only what is needed for this one issue.
- Do not rename functions, add features, reformat the whole file, or rewrite unrelated code.
- Do not invent files, APIs, or issues that are not in the source.
- Do not wrap the JSON in markdown fences.
- For dangerous-call, replace only the unsafe call itself. Do not rewrite surrounding try/except blocks.
- Do not introduce a new import unless the replacement can be safely applied with the provided code context.
- Do not change or reintroduce any other analyzer issue in the surrounding code.
""".strip()

_FIX_KEYS = ("explanation", "original_code", "corrected_code", "reason")


class OllamaError(ReviewError):
    """Raised when the local Ollama reviewer cannot run."""


@dataclass
class AIFixProposal:
    """Structured fix returned by a local Ollama model."""

    explanation: str
    original_code: str
    corrected_code: str
    reason: str


def review_with_ollama(
    source_code: str | dict[str, str],
    findings: Iterable[Issue | dict[str, Any]] | None = None,
    *,
    host: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Send Python source and static-analysis findings to a local Ollama model."""
    sources = _normalize_sources(source_code)
    if not sources:
        raise OllamaError("No Python source code was provided for review.")

    ollama_host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    ollama_model = (model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    ensure_ollama_ready(ollama_host, ollama_model)

    payload = {
        "static_analysis_findings": _findings_payload(findings or []),
        "source_files": {
            path: _with_line_numbers(text) for path, text in sources.items()
        },
    }
    prompt = (
        "Review this Python code. Return JSON only, with keys summary and issues. "
        "Each issue must include file, line, type, severity, description, evidence, "
        "and recommendation.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    raw_text = _chat(ollama_host, ollama_model, prompt)
    review = _parse_review_json(raw_text)
    return _filter_unsupported_issues(review, sources)


def propose_fix_with_ollama(
    issue: Issue,
    source_text: str,
    *,
    host: str | None = None,
    model: str | None = None,
    snippet: str | None = None,
) -> AIFixProposal:
    """Ask local Ollama for a minimal patch for one analyzer finding."""
    if not source_text.strip():
        raise OllamaError("No Python source code was provided for an AI fix.")

    ollama_host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
    ollama_model = (model or os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    ensure_ollama_ready(ollama_host, ollama_model)

    relevant = snippet if snippet is not None else source_text
    payload = {
        "issue": {
            "file": issue.file_path,
            "line": issue.line,
            "rule": issue.rule,
            "severity": issue.severity,
            "message": issue.message,
            "suggestion": issue.suggestion,
        },
        "relevant_code": _with_line_numbers(relevant),
    }    
    prompt = (
        "Fix only the issue described below. "
        "Return JSON with keys explanation, original_code, corrected_code, and reason. "
        "original_code must match the source exactly. "
        "corrected_code must be a minimal replacement for that original_code.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )
    raw_text = _chat(
        ollama_host,
        ollama_model,
        prompt,
        system=_FIX_SYSTEM_INSTRUCTIONS,
        empty_message="Ollama returned an empty fix. Try another local model.",
    )
    try:
        return parse_fix_response(raw_text)
    except OllamaError as exc:
        raise OllamaError(
            f"{exc} Raw Ollama response: {raw_text[:1000]}"
        ) from exc


def _strip_line_number_prefixes(code: str) -> str:
    """Remove line-number prefixes such as '1| ' from AI-generated code."""
    lines = code.splitlines()
    cleaned = []

    for line in lines:
        cleaned.append(re.sub(r"^\s*\d+\|\s?", "", line))

    return "\n".join(cleaned)


def parse_fix_response(raw_text: str) -> AIFixProposal:
    """Parse an Ollama JSON fix. Used by tests without calling the network."""
    parsed = _extract_json_object(raw_text)
    if parsed is None:
        raise OllamaError("Ollama fix response was not valid JSON.")

    # Some models wrap the four fields in a nested object.
    if not all(key in parsed for key in _FIX_KEYS) and isinstance(parsed.get("fix"), dict):
        parsed = parsed["fix"]

    missing = [key for key in _FIX_KEYS if not str(parsed.get(key) or "").strip()]
    if missing:
        raise OllamaError(
            "Ollama fix JSON is missing required fields: " + ", ".join(missing)
        )

    return AIFixProposal(
    explanation=str(parsed["explanation"]).strip(),
    original_code=_strip_line_number_prefixes(
        _strip_code_fence(str(parsed["original_code"]))
    ),
    corrected_code=_strip_line_number_prefixes(
        _strip_code_fence(str(parsed["corrected_code"]))
    ),
    reason=str(parsed["reason"]).strip(),
)


def ensure_ollama_ready(host: str, model: str) -> None:
    """Fail with a beginner-friendly message if Ollama or the model is missing."""
    try:
        data = _request_json("GET", f"{host}/api/tags", timeout=5)
    except OllamaError:
        raise
    except urllib.error.URLError as exc:
        raise OllamaError(
            "Could not connect to Ollama at "
            f"{host}. {_INSTALL_HINT} "
            f"Details: {exc.reason or exc}"
        ) from None
    except TimeoutError:
        raise OllamaError(
            f"Timed out while contacting Ollama at {host}. Is the Ollama app running?"
        ) from None

    names = [str(item.get("name") or "") for item in data.get("models") or []]
    if _model_is_installed(model, names):
        return

    available = ", ".join(names) if names else "(none)"
    raise OllamaError(
        f"Ollama is running, but model '{model}' is not installed. "
        f"Available models: {available}. Install it with: ollama pull {model}"
    )


def _model_is_installed(model: str, names: list[str]) -> bool:
    wanted = model.strip()
    for name in names:
        if name == wanted or name.startswith(f"{wanted}:") or wanted.startswith(f"{name}"):
            return True
        base = name.split(":", 1)[0]
        if base == wanted:
            return True
    return False


def _chat(
    host: str,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    empty_message: str | None = None,
) -> str:
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": system or _SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        data = _request_json("POST", f"{host}/api/chat", payload=body, timeout=180)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        if exc.code == 404:
            raise OllamaError(
                f"Ollama could not use model '{model}'. "
                f"Install it with: ollama pull {model}"
            ) from None
        raise OllamaError(
            f"Ollama returned HTTP {exc.code}. {detail.strip() or exc.reason}"
        ) from None
    except urllib.error.URLError as exc:
        raise OllamaError(
            "Lost the connection to Ollama while generating the review. "
            f"Is the Ollama app still running? Details: {exc.reason or exc}"
        ) from None

    message = data.get("message") or {}
    text = str(message.get("content") or data.get("response") or "").strip()
    if not text:
        raise OllamaError(
            empty_message or "Ollama returned an empty review. Try another local model."
        )
    return text


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise OllamaError(
            f"Ollama returned HTTP {exc.code} for {url}. {detail.strip() or exc.reason}"
        ) from None

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise OllamaError(
            "Ollama returned a response that was not valid JSON."
        ) from exc
    if not isinstance(parsed, dict):
        raise OllamaError("Ollama returned JSON that was not an object.")
    return parsed


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of model text (plain or fenced)."""
    candidates = [raw_text.strip()]
    if "```" in raw_text:
        for block in raw_text.split("```"):
            cleaned = block.removeprefix("json").strip()
            if cleaned.startswith("{"):
                candidates.append(cleaned)

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.replace("\r\n", "\n")


def _parse_review_json(raw_text: str) -> dict[str, Any]:
    parsed = _extract_json_object(raw_text)
    if parsed is not None:
        try:
            return CodeReview.model_validate(parsed).model_dump()
        except Exception:
            if "issues" in parsed:
                return {
                    "summary": str(parsed.get("summary") or "").strip(),
                    "issues": parsed.get("issues") or [],
                }

    return {
        "summary": raw_text.strip()[:2000],
        "issues": [],
    }
