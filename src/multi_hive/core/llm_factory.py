"""
llm_factory.py — LLM instance cache.

Two caches, one purpose-key schema:
  _sync_cache  — ChatOllama for sync nodes (sprint_planner, ticket_writer).
  _async_cache — ChatOllama for async nodes (async_editor_node,
                 semantic_reviewer_node).

They are kept separate so an invalidate_llm("editor") call from inside an
async node cannot evict a sync planner instance that is in flight on another
thread. Both share _PURPOSE_KWARGS, so per-purpose tuning lives in one place.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama

from multi_hive.config import MODEL_NAME

_sync_cache: dict[str, ChatOllama] = {}
_async_cache: dict[str, ChatOllama] = {}

_PURPOSE_KWARGS: dict[str, dict] = {
    "planner": dict(
        model=MODEL_NAME,
        temperature=0.1,
    ),
    "ticket": dict(
        model=MODEL_NAME,
        temperature=0.1,
        num_predict=512,
        num_ctx=2048,
    ),
    "editor": dict(
        model=MODEL_NAME,
        temperature=0.1,
        num_predict=2048,
        num_ctx=4096,
        keep_alive=-1,
    ),
    # Semantic reviewer: the verdict is one line, so num_predict stays low;
    # num_ctx must fit the whole generated file being reviewed; temperature 0
    # for a deterministic PASS/FAIL classification.
    "reviewer": dict(
        model=MODEL_NAME,
        temperature=0.0,
        num_predict=128,
        num_ctx=4096,
        keep_alive=-1,
    ),
}


def _validated_kwargs(purpose: str) -> dict:
    if purpose not in _PURPOSE_KWARGS:
        raise ValueError(
            f"Unknown LLM purpose {purpose!r}. Valid purposes: {sorted(_PURPOSE_KWARGS)}"
        )
    return _PURPOSE_KWARGS[purpose]


def get_llm(purpose: str) -> ChatOllama:
    """Cached sync ChatOllama for `purpose`. Raises ValueError if unknown."""
    if purpose not in _sync_cache:
        _sync_cache[purpose] = ChatOllama(**_validated_kwargs(purpose))
    return _sync_cache[purpose]


def get_async_llm(purpose: str) -> ChatOllama:
    """
    Cached ChatOllama for async use (ainvoke).

    ChatOllama exposes both .invoke() and .ainvoke() on one instance; the
    separate cache exists purely so invalidation in one execution mode cannot
    yank an instance out from under the other.
    """
    if purpose not in _async_cache:
        _async_cache[purpose] = ChatOllama(**_validated_kwargs(purpose))
    return _async_cache[purpose]


def invalidate_llm(purpose: str) -> None:
    """
    Drops both cached instances for `purpose`.

    Call from an exception handler when a connection-level failure suggests
    Ollama crashed, restarted, or went unreachable. The next get_llm() /
    get_async_llm() rebuilds.
    """
    _sync_cache.pop(purpose, None)
    _async_cache.pop(purpose, None)


def cache_size() -> int:
    """Total live instances across both caches — reported in the sprint footer."""
    return len(_sync_cache) + len(_async_cache)
