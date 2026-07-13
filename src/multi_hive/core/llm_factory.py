"""
llm_factory.py — model instance cache, keyed by (purpose, tier).

*Purpose* decides the sampling parameters — a planner wants room to think, a
reviewer emits one line and must be deterministic. *Tier* decides which model
runs them (see core/model_router.py).

The two are orthogonal, which is why the cache key is the pair: the reviewer
prompt against the fast model and the reviewer prompt against the strong model
are two different clients, and neither should evict the other.

Sync and async caches are also kept apart, so an invalidate_llm() call from
inside an async node cannot yank an instance out from under a sync node that
is mid-flight on another thread.

On keep_alive
-------------
Instances are NOT pinned in VRAM (`keep_alive` is left at Ollama's default).
An earlier version pinned the editor with keep_alive=-1, which is right for a
single-model setup and wrong for a tiered one: the fast and strong models do not
fit in 8 GB together, so pinning one guarantees a fight with the other. Ollama's
own LRU eviction handles this better than we can from here.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama

from multi_hive.config import MODELS

_sync_cache: dict[tuple[str, str], ChatOllama] = {}
_async_cache: dict[tuple[str, str], ChatOllama] = {}

# Sampling parameters per purpose. The model is chosen separately, by tier.
_PURPOSE_KWARGS: dict[str, dict] = {
    "planner": dict(
        temperature=0.1,
    ),
    "ticket": dict(
        temperature=0.1,
        num_predict=512,
        num_ctx=2048,
    ),
    "editor": dict(
        temperature=0.1,
        num_predict=2048,
        num_ctx=4096,
    ),
    # The verdict is one line, so num_predict stays low; num_ctx must fit the
    # whole file under review; temperature 0 for a deterministic PASS/FAIL.
    "reviewer": dict(
        temperature=0.0,
        num_predict=128,
        num_ctx=4096,
    ),
}

DEFAULT_TIER = "fast"


def _resolve(purpose: str, tier: str) -> dict:
    if purpose not in _PURPOSE_KWARGS:
        raise ValueError(
            f"Unknown LLM purpose {purpose!r}. Valid purposes: {sorted(_PURPOSE_KWARGS)}"
        )
    if tier not in MODELS:
        raise ValueError(f"Unknown model tier {tier!r}. Valid tiers: {sorted(MODELS)}")

    return {"model": MODELS[tier], **_PURPOSE_KWARGS[purpose]}


def get_llm(purpose: str, tier: str = DEFAULT_TIER) -> ChatOllama:
    """Cached sync client for (purpose, tier)."""
    key = (purpose, tier)
    if key not in _sync_cache:
        _sync_cache[key] = ChatOllama(**_resolve(purpose, tier))
    return _sync_cache[key]


def get_async_llm(purpose: str, tier: str = DEFAULT_TIER) -> ChatOllama:
    """Cached async client for (purpose, tier)."""
    key = (purpose, tier)
    if key not in _async_cache:
        _async_cache[key] = ChatOllama(**_resolve(purpose, tier))
    return _async_cache[key]


def invalidate_llm(purpose: str, tier: str | None = None) -> None:
    """
    Drops cached clients for `purpose` — for one tier, or all tiers if None.

    Call from an exception handler when a connection-level failure suggests
    Ollama crashed, restarted, or went unreachable. The next get_llm() rebuilds.
    """
    for cache in (_sync_cache, _async_cache):
        for key in [k for k in cache if k[0] == purpose and (tier is None or k[1] == tier)]:
            cache.pop(key, None)


def model_for(tier: str) -> str:
    """The model name backing a tier — for logging and the sprint footer."""
    return MODELS.get(tier, "?")


def cache_size() -> int:
    """Total live clients across both caches."""
    return len(_sync_cache) + len(_async_cache)
