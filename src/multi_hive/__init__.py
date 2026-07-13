"""
multi_hive — Sentinel Prime: an async, self-healing, multi-file code
generation hive built on LangGraph + a local Ollama model.

Nothing heavy is imported here on purpose. `import multi_hive` must stay
cheap enough for tooling (and tests) to touch it without pulling in
langgraph, langchain, or rich.
"""

__version__ = "4.2.0"
