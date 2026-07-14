"""
llm_factory.py — model instance cache, keyed by (purpose, tier).

*Purpose* decides the sampling parameters — a planner wants room to think, a
reviewer emits one line and must be deterministic. *Tier* decides which model
runs them (see core/model_router.py). *Provider* decides where that model lives.

The first two are orthogonal, which is why the cache key is the pair: the
reviewer prompt against the fast model and the reviewer prompt against the strong
model are two different clients, and neither should evict the other.

The provider is not part of the key, because it is fixed for the life of the
process — it is read once from HIVE_PROVIDER at import.

This is the ONLY module in the system that knows a provider exists. Every node
calls get_llm()/get_async_llm() and is handed something with .invoke()/.ainvoke().
Swapping local Ollama for the Claude API is a change here and nowhere else, which
is the entire reason this indirection was worth having.

Sync and async caches are also kept apart, so an invalidate_llm() call from
inside an async node cannot yank an instance out from under a sync node that
is mid-flight on another thread.

Why two parameter tables and not one
------------------------------------
Ollama speaks num_predict and num_ctx; Anthropic speaks max_tokens and has no
context knob at all (the window is fixed and enormous). A single "neutral" table
mapped onto both would be a leaky abstraction pretending not to be one — and the
mapping would silently change Ollama's tuning, which is measured. So each
provider gets its own table, in its own vocabulary. The duplication is four
lines, and it is the honest kind.

On keep_alive
-------------
Ollama instances are NOT pinned in VRAM (`keep_alive` is left at its default).
An earlier version pinned the editor with keep_alive=-1, which is right for a
single-model setup and wrong for a tiered one: the fast and strong models do not
fit in 8 GB together, so pinning one guarantees a fight with the other. Ollama's
own LRU eviction handles this better than we can from here.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from multi_hive.config import MODELS, PROVIDER
from multi_hive.core import governor

_sync_cache: dict[tuple[str, str], Any] = {}

# The async cache stores (loop, client). An async client's connection pool is
# bound to the event loop that created it, so a client outliving its loop is a
# landmine: every call raises `Event loop is closed`.
#
# It does not fail cleanly either. async_editor_node catches the error and
# retries; the retry raises the identical error; the repeat-error fingerprint
# matches, the circuit breaker fires, and the task is escalated to a human having
# never once reached a model — and scored as a model failure. It cost a
# benchmark run: `merge_intervals` recorded as `✗ FAIL (no code) [human gate]` on
# a task the 7B passes in 80 seconds.
#
# Keying on the loop makes that impossible rather than merely discouraged. The
# strong reference is to the *current* loop only, and it is dropped the moment a
# different one shows up — so this neither leaks loops nor trusts id() not to be
# recycled.
_async_cache: dict[tuple[str, str], tuple[asyncio.AbstractEventLoop, Any]] = {}

# Ollama. num_predict caps the response; num_ctx is the window the model sees.
# These numbers are tuned against the local 7B/30B pair — changing them changes
# the benchmark, so change them deliberately.
_OLLAMA_KWARGS: dict[str, dict] = {
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

# Anthropic. No context parameter — the window is fixed and far larger than
# anything the hive sends, so the num_ctx ceilings above simply have no analogue.
# max_tokens is required by the API, including for the planner, which Ollama was
# happy to leave unbounded.
_ANTHROPIC_KWARGS: dict[str, dict] = {
    "planner": dict(
        temperature=0.1,
        max_tokens=1024,
    ),
    "ticket": dict(
        temperature=0.1,
        max_tokens=512,
    ),
    # Roomier than the Ollama editor's 2048: a frontier model writes longer,
    # better-structured files, and truncating one mid-function would look to the
    # reviewer like a syntax error the model kept "failing" to fix.
    "editor": dict(
        temperature=0.1,
        max_tokens=8192,
    ),
    "reviewer": dict(
        temperature=0.0,
        max_tokens=128,
    ),
}

_PURPOSE_KWARGS: dict[str, dict[str, dict]] = {
    "ollama": _OLLAMA_KWARGS,
    "anthropic": _ANTHROPIC_KWARGS,
}

DEFAULT_TIER = "fast"

# Sampling parameters are REJECTED with a 400 by the Claude 5 family and Opus 4.7+:
# `temperature`, `top_p`, and `top_k` were removed from those models, which are
# steered with prompting and `output_config.effort` instead. haiku-4-5 (the fast
# tier) is a 4.5 model and still accepts them — which is why the fast tier ran fine
# and only fable-5 escalations died with `temperature is deprecated for this model`.
# So this is model-specific: strip for the models that refuse the params, and leave
# the fast tier — and all of the measured local Ollama tuning — untouched.
_SAMPLING_PARAMS = ("temperature", "top_p", "top_k")
_REJECTS_SAMPLING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
)


def _rejects_sampling_params(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _REJECTS_SAMPLING_PREFIXES)


def _resolve(purpose: str, tier: str) -> dict:
    table = _PURPOSE_KWARGS[PROVIDER]

    if purpose not in table:
        raise ValueError(
            f"Unknown LLM purpose {purpose!r}. Valid purposes: {sorted(table)}"
        )
    if tier not in MODELS:
        raise ValueError(f"Unknown model tier {tier!r}. Valid tiers: {sorted(MODELS)}")

    return {"model": MODELS[tier], **table[purpose]}


def _build(purpose: str, tier: str) -> Any:
    """
    Constructs a client for the configured provider.

    Both imports are deferred. langchain_anthropic is an optional dependency, and
    an Ollama-only user must not be forced to install it — nor should an import
    error from a package they never asked for be the first thing they see.

    Every client is built with the governor's meter attached. This module is the
    only one allowed to construct a client, which makes it the only place the
    meter *can* be attached such that no call escapes it — a node that built its
    own client would not just ignore HIVE_PROVIDER (which is why the rule exists),
    it would now also spend money nothing is counting.
    """
    kwargs = _resolve(purpose, tier)
    kwargs["callbacks"] = [governor.meter(kwargs["model"])]

    if PROVIDER == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "HIVE_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Export your key, or unset HIVE_PROVIDER to run locally on Ollama."
            )
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise RuntimeError(
                "HIVE_PROVIDER=anthropic needs langchain-anthropic. "
                'Install it with:  pip install -e ".[anthropic]"'
            ) from e

        # The strong tier (fable-5) rejects temperature with a 400; the fast tier
        # (haiku-4-5) accepts it. Drop the sampling params only for the models that
        # refuse them — see _rejects_sampling_params.
        if _rejects_sampling_params(kwargs["model"]):
            for key in _SAMPLING_PARAMS:
                kwargs.pop(key, None)

        return ChatAnthropic(**kwargs)

    from langchain_ollama import ChatOllama

    return ChatOllama(**kwargs)


def get_llm(purpose: str, tier: str = DEFAULT_TIER) -> Any:
    """Cached sync client for (purpose, tier)."""
    key = (purpose, tier)
    if key not in _sync_cache:
        _sync_cache[key] = _build(purpose, tier)
    return _sync_cache[key]


def get_async_llm(purpose: str, tier: str = DEFAULT_TIER) -> Any:
    """
    Cached async client for (purpose, tier), valid for the *running* loop.

    A client built on a loop that has since closed is rebuilt rather than handed
    back. See the note on _async_cache: reusing one does not raise where you can
    see it — it raises inside a node, gets retried into the circuit breaker, and
    surfaces as a model failure that never happened.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Called outside a loop. There is no loop to bind to and therefore nothing
        # safe to cache against, so hand back a fresh client and cache nothing.
        return _build(purpose, tier)

    key = (purpose, tier)

    entry = _async_cache.get(key)
    if entry is None or entry[0] is not loop or entry[0].is_closed():
        _async_cache[key] = (loop, _build(purpose, tier))

    return _async_cache[key][1]


def invalidate_llm(purpose: str, tier: str | None = None) -> None:
    """
    Drops cached clients for `purpose` — for one tier, or all tiers if None.

    Call from an exception handler when a connection-level failure suggests the
    backend crashed, restarted, or went unreachable. The next get_llm() rebuilds.
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
