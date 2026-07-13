"""
ast_utils.py — signature-level outlines of Python source.

Both editor nodes import get_code_outline() to build cross-file context: the
file being edited goes into the prompt in full, every *other* project file
goes in as an outline. Feeding whole files for context blows through the
editor's 4096-token num_ctx after two or three modules.

This module was imported throughout the codebase but never actually existed.
async_editor_node had a try/except fallback that silently degraded to a
500-char head; editor_node imported it unconditionally and crashed on import.
"""
from __future__ import annotations

import ast

_FALLBACK_CHARS = 500


def get_code_outline(content: str) -> str:
    """
    Returns a compact outline of `content`: imports, module constants, class
    and function signatures. Bodies and docstrings are dropped.

    Falls back to a truncated head of the source if it will not parse — the
    outline is prompt context, so unparseable input degrades rather than
    raising into the middle of a sprint.
    """
    if not content or not content.strip():
        return ""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _truncate(content)

    lines: list[str] = []
    for node in tree.body:
        lines.extend(_describe(node, indent=0))

    return "\n".join(lines) if lines else _truncate(content)


def _truncate(content: str) -> str:
    stripped = content.strip()
    if len(stripped) <= _FALLBACK_CHARS:
        return stripped
    return stripped[:_FALLBACK_CHARS] + "\n# [outline truncated]"


def _describe(node: ast.AST, indent: int) -> list[str]:
    pad = "    " * indent

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [pad + _unparse(node)]

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [pad + _signature(node)]

    if isinstance(node, ast.ClassDef):
        bases = ", ".join(_unparse(b) for b in node.bases)
        header = f"{pad}class {node.name}({bases}):" if bases else f"{pad}class {node.name}:"
        lines = [header]
        members = [
            line
            for child in node.body
            for line in _describe(child, indent + 1)
        ]
        lines.extend(members or [pad + "    ..."])
        return lines

    # Module- or class-level constants: name only, values can be huge.
    if isinstance(node, ast.Assign):
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        return [f"{pad}{name} = ..." for name in targets]

    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        annotation = _unparse(node.annotation)
        return [f"{pad}{node.target.id}: {annotation} = ..."]

    return []


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    keyword = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = _unparse(node.args)
    returns = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    return f"{keyword} {node.name}({args}){returns}: ..."


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."
