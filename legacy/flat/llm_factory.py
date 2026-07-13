"""
llm_factory.py — LLM instance cache for Sentinel Prime v4.1.

Two caches, same purpose-key schema:
  _sync_cache  — ChatOllama instances for sync nodes (sprint_planner,
                 ticket_writer, reviewer_node subprocess path).
  _async_cache — ChatOllama instances for async nodes (async_editor_node).
                 Kept separate so an invalidate_llm("editor") call from
                 inside the async node doesn't accidentally evict the sync
                 planner instance that may be in-flight on another thread.

Both caches share _PURPOSE_KWARGS so config lives in one place.
"""
from langchain_ollama import ChatOllama
from hive_config import MODEL_NAME

_sync_cache:  dict = {}
_async_cache: dict = {}

_PURPOSE_KWARGS: dict = {
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
    # Phase 3: semantic reviewer — low num_predict (verdict is one line),
    # larger num_ctx to fit the full generated code being reviewed,
    # temperature=0 for deterministic PASS/FAIL classification.
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
            f"Unknown LLM purpose {purpose!r}. "
            f"Valid purposes: {sorted(_PURPOSE_KWARGS)}"
        )
    return _PURPOSE_KWARGS[purpose]


def get_llm(purpose: str) -> ChatOllama:
    """
    Returns a cached sync ChatOllama for the given purpose.
    Instantiates on first call; raises ValueError for unknown purposes.
    """
    if purpose not in _sync_cache:
        _sync_cache[purpose] = ChatOllama(**_validated_kwargs(purpose))
    return _sync_cache[purpose]


def get_async_llm(purpose: str) -> ChatOllama:
    """
    Returns a cached ChatOllama for async use (ainvoke).

    ChatOllama supports both .invoke() and .ainvoke() on the same
    instance, but keeping a separate async cache prevents an
    invalidate_llm() call in one node from evicting an instance
    that a concurrent async node still holds a reference to.
    """
    if purpose not in _async_cache:
        _async_cache[purpose] = ChatOllama(**_validated_kwargs(purpose))
    return _async_cache[purpose]


def invalidate_llm(purpose: str) -> None:
    """
    Drops both cached instances for the given purpose.

    Call from an exception handler when a connection-level error is
    suspected (Ollama crashed / restarted / unreachable). The next
    get_llm() / get_async_llm() call will rebuild the instance.
    """
    _sync_cache.pop(purpose, None)
    _async_cache.pop(purpose, None)
