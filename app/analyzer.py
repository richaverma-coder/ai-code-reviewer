"""Scan local Python files and collect simple code issues.

This version uses only the Python standard library (`ast` and `pathlib`).
It does not call GitHub or any AI API.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


@dataclass
class Issue:
    """One finding from static analysis."""

    file_path: str
    line: int
    column: int
    severity: str
    rule: str
    message: str
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


def analyze_path(target: str | Path) -> list[Issue]:
    """Analyze a Python file or every `.py` file in a folder."""
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    issues: list[Issue] = []
    for file_path in _iter_python_files(path):
        issues.extend(analyze_file(file_path))
    return issues


def analyze_file(file_path: Path) -> list[Issue]:
    """Analyze a single Python file."""
    source = file_path.read_text(encoding="utf-8")
    rel_path = str(file_path)

    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError as exc:
        return [
            Issue(
                file_path=rel_path,
                line=exc.lineno or 1,
                column=(exc.offset or 1) - 1,
                severity=SEVERITY_ERROR,
                rule="syntax-error",
                message=exc.msg,
                suggestion="Fix the syntax error so the file can be parsed.",
            )
        ]

    visitor = _IssueVisitor(rel_path, source.splitlines())
    visitor.visit(tree)
    return visitor.issues


def _iter_python_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix == ".py":
            yield path
        return

    for py_file in sorted(path.rglob("*.py")):
        if _should_skip(py_file):
            continue
        yield py_file


def _should_skip(path: Path) -> bool:
    skipped_parts = {".git", "venv", ".venv", "__pycache__", "site-packages"}
    return any(part in skipped_parts for part in path.parts)


class _IssueVisitor(ast.NodeVisitor):
    """Walk the AST and record beginner-friendly issues."""

    def __init__(self, file_path: str, lines: list[str]) -> None:
        self.file_path = file_path
        self.lines = lines
        self.issues: list[Issue] = []
        self._imported_names: dict[str, ast.AST] = {}
        self._used_names: set[str] = set()
        self._assigned_names: dict[str, ast.AST] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            self._imported_names[name] = node
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            self._imported_names[name] = node
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self._used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store) and node.id not in self._assigned_names:
            self._assigned_names[node.id] = node
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._add(
                node,
                SEVERITY_WARNING,
                "bare-except",
                "Bare `except:` catches every error, including KeyboardInterrupt.",
                "Catch a specific exception, for example `except ValueError:`.",
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_mutable_defaults(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_mutable_defaults(node)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for operator, comparator in zip(node.ops, node.comparators):
            uses_equality = isinstance(operator, (ast.Eq, ast.NotEq))
            is_none = isinstance(comparator, ast.Constant) and comparator.value is None
            if uses_equality and is_none:
                self._add(
                    node,
                    SEVERITY_INFO,
                    "none-comparison",
                    "Compared a value with `None` using `==` or `!=`.",
                    "Use `is None` or `is not None`.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
            self._add(
                node,
                SEVERITY_WARNING,
                "dangerous-call",
                f"`{func.id}()` can run untrusted code.",
                "Avoid `eval`/`exec`. Parse data with `json` or `ast.literal_eval` instead.",
            )
        self.generic_visit(node)

    def visit_Module(self, node: ast.Module) -> None:
        self.generic_visit(node)
        self._report_unused_imports()
        self._report_unused_variables()

    def _check_mutable_defaults(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self._add(
                    node,
                    SEVERITY_WARNING,
                    "mutable-default",
                    f"Function `{node.name}` uses a mutable default argument.",
                    "Use `None` as the default and create the list/dict inside the function.",
                )

    def _report_unused_imports(self) -> None:
        for name, node in self._imported_names.items():
            if name.startswith("_") or name in self._used_names:
                continue
            self._add(
                node,
                SEVERITY_INFO,
                "unused-import",
                f"Imported name `{name}` is never used.",
                f"Remove the unused import `{name}`.",
            )

    def _report_unused_variables(self) -> None:
        skip = set(self._imported_names)
        for name, node in self._assigned_names.items():
            if name.startswith("_") or name in skip or name in self._used_names:
                continue
            if name.isupper():
                continue
            self._add(
                node,
                SEVERITY_INFO,
                "unused-variable",
                f"Variable `{name}` is assigned but never used.",
                f"Use `{name}` or remove it.",
            )

    def _add(
        self,
        node: ast.AST,
        severity: str,
        rule: str,
        message: str,
        suggestion: str,
    ) -> None:
        self.issues.append(
            Issue(
                file_path=self.file_path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0),
                severity=severity,
                rule=rule,
                message=message,
                suggestion=suggestion,
            )
        )
