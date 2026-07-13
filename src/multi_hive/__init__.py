"""
multi_hive — Sentinel Prime: an async, self-healing, multi-file code
generation hive built on LangGraph + local Ollama models.

Nothing heavy is imported here on purpose. `import multi_hive` must stay cheap
enough for tooling (and tests) to touch it without pulling in langgraph,
langchain, or rich.

pyproject.toml is the single source of the version — never hardcode it here.
"""
from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def _resolve_version() -> str:
    """
    In a source checkout, read pyproject.toml directly.

    An editable install writes its metadata once, at install time. Bumping the
    version in pyproject.toml does not refresh it, so importlib.metadata reports
    the version the tree had when it was last installed — the app printed 4.2.0
    in its own banner immediately after being released as 4.3.0. When the
    pyproject that governs this source tree is right there next to it, it is the
    truth; metadata is the fallback for a real (non-editable) install, where
    there is no pyproject to read.
    """
    try:
        if _PYPROJECT.is_file():
            match = _VERSION_RE.search(_PYPROJECT.read_text(encoding="utf-8"))
            if match:
                return match.group(1)
    except OSError:
        pass

    try:
        return version("multi-hive")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _resolve_version()

__all__ = ["__version__"]
