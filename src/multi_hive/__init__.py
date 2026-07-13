"""
multi_hive — Sentinel Prime: an async, self-healing, multi-file code
generation hive built on LangGraph + local Ollama models.

Nothing heavy is imported here on purpose. `import multi_hive` must stay cheap
enough for tooling (and tests) to touch it without pulling in langgraph,
langchain, or rich.

The version is read from installed package metadata rather than hardcoded, so
pyproject.toml is the single source of truth. Declaring it in both places is
how the two drift apart.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("multi-hive")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
