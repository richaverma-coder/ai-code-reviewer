"""Run Bandit, Black, and Pytest without changing the reviewed source files."""

from __future__ import annotations

import ast
import builtins
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import TimeoutExpired, run

from app.analyzer import _iter_python_files

NESTED_FIX_VALIDATION_ENV = "AI_CODE_REVIEWER_FIX_VALIDATION"


@dataclass
class ToolResult:
    """One quality-tool run captured for the Markdown report."""

    name: str
    available: bool
    ok: bool
    summary: str
    output: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_quality_tools(target: str | Path) -> list[ToolResult]:
    """Run Bandit, Black (check only), and Pytest. Never rewrite source files."""
    path = Path(target)
    project_root = Path(__file__).resolve().parent.parent
    return [
        _run_module(
            "Bandit",
            ["bandit", "-q", "-r", str(path), "-x", "venv,.venv"],
            missing="Bandit is not installed. Run: pip install bandit",
            cwd=project_root,
        ),
        _run_module(
            "Black",
            ["black", "--check", "--diff", str(path)],
            missing="Black is not installed. Run: pip install black",
            cwd=project_root,
        ),
        _run_module(
            "Pytest",
            ["pytest", str(project_root / "tests"), "-q", "--tb=short"],
            missing="Pytest is not installed. Run: pip install pytest",
            cwd=project_root,
        ),
    ]


def run_syntax_validation(target: str | Path) -> ToolResult:
    """Parse every Python file under `target`. Never writes files."""
    path = Path(target)
    failures: list[str] = []
    files = list(_iter_python_files(path))
    if not files:
        return ToolResult(
            "Syntax",
            True,
            False,
            "Python syntax validation found no Python files.",
            "",
        )

    for file_path in files:
        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(f"{file_path}: {exc}")
            continue
        try:
            ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            failures.append(
                f"{file_path}:{exc.lineno or 1}: {exc.msg}"
            )

    if failures:
        output = "\n".join(failures)
        return ToolResult(
            "Syntax",
            True,
            False,
            f"Python syntax validation failed ({len(failures)} file(s)).",
            output,
        )
    return ToolResult(
        "Syntax",
        True,
        True,
        "Python syntax validation passed.",
        "",
    )

def run_undefined_name_validation(target: str | Path) -> ToolResult:
    """Detect names used in Python files without a local/imported definition."""
    path = Path(target)
    failures: list[str] = []

    for file_path in _iter_python_files(path):
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
        except (OSError, SyntaxError):
            continue

        defined: set[str] = set()
        imported: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.asname or alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        imported.add(alias.asname or alias.name)

            elif isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            ):
                defined.add(node.name)

            elif isinstance(node, ast.Name) and isinstance(
                node.ctx, ast.Store
            ):
                defined.add(node.id)

            elif isinstance(node, ast.arg):
                defined.add(node.arg)

        builtin_names = set(dir(builtins))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Name):
                continue

            if not isinstance(node.ctx, ast.Load):
                continue

            if node.id in defined:
                continue

            if node.id in imported:
                continue

            if node.id in builtin_names:
                continue

            failures.append(
                f"{file_path}:{node.lineno}: undefined name '{node.id}'"
            )

    if failures:
        output = "\n".join(failures)
        return ToolResult(
            "UndefinedNames",
            True,
            False,
            f"Undefined-name validation failed ({len(failures)} finding(s)).",
            output,
        )

    return ToolResult(
        "UndefinedNames",
        True,
        True,
        "Undefined-name validation passed.",
        "",
    )


def run_validation_suite(target: str | Path) -> list[ToolResult]:
    """Syntax, Bandit, Black --check, and Pytest for a tree. Never rewrite files."""
    path = Path(target)
    project_root = Path(__file__).resolve().parent.parent
    pytest_args = ["pytest", str(path), "-q", "--tb=short"]
    extra_env = {NESTED_FIX_VALIDATION_ENV: "1"}
    return [
        run_syntax_validation(path),
        run_undefined_name_validation(path),
        _run_module(
            "Bandit",
            ["bandit", "-q", "-r", str(path), "-x", "venv,.venv", "-f", "json"],
            missing="Bandit is not installed. Run: pip install bandit",
            cwd=project_root,
            extra_env=extra_env,
        ),
        _run_module(
            "Black",
            ["black", "--check", "--diff", str(path)],
            missing="Black is not installed. Run: pip install black",
            cwd=project_root,
            extra_env=extra_env,
        ),
        _run_module(
            "Pytest",
            pytest_args,
            missing="Pytest is not installed. Run: pip install pytest",
            cwd=project_root,
            extra_env=extra_env,
            treat_no_tests_as_ok=True,
        ),
    ]


def count_bandit_findings(result: ToolResult) -> int | None:
    """Return Bandit issue count from JSON output, or None if unknown."""
    if not result.available:
        return None
    payload = _extract_json(result.output)
    if payload is None:
        return 0 if result.ok else None
    findings = payload.get("results")
    if isinstance(findings, list):
        return len(findings)
    return 0 if result.ok else None


def _extract_json(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _run_module(
    name: str,
    module_args: list[str],
    *,
    missing: str,
    cwd: Path,
    extra_env: dict[str, str] | None = None,
    treat_no_tests_as_ok: bool = False,
) -> ToolResult:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        completed = run(
            [sys.executable, "-m", *module_args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=90,
            env=env,
        )
    except FileNotFoundError:
        return ToolResult(name, False, False, missing, "")
    except TimeoutExpired:
        return ToolResult(
            name,
            True,
            False,
            f"{name} timed out after 90 seconds.",
            "",
        )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode == 0:
        summary = f"{name} completed successfully."
        return ToolResult(name, True, True, summary, output)

    if treat_no_tests_as_ok and completed.returncode == 5:
        summary = f"{name} collected no tests (treated as OK)."
        return ToolResult(name, True, True, summary, output)

    if _module_missing(output):
        return ToolResult(name, False, False, missing, output)

    summary = f"{name} reported issues (exit code {completed.returncode})."
    return ToolResult(name, True, False, summary, output)


def _module_missing(output: str) -> bool:
    lowered = output.lower()
    return "no module named" in lowered or "modulenotfounderror" in lowered
