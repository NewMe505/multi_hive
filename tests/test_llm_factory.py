"""
llm_factory is the only module that knows a provider exists.

That claim is the whole justification for the indirection, so it gets a test. If
a node ever imports ChatOllama or ChatAnthropic directly, switching providers
silently stops working for that node — and it would work fine in every test that
does not happen to exercise it.
"""
from __future__ import annotations

import importlib

import pytest

from multi_hive.core import llm_factory


def test_only_llm_factory_knows_the_provider():
    from pathlib import Path

    src = Path(llm_factory.__file__).resolve().parents[1]

    offenders = []
    for path in src.rglob("*.py"):
        if path.name in ("llm_factory.py", "config.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if "ChatOllama" in text or "ChatAnthropic" in text:
            offenders.append(path.relative_to(src).as_posix())

    assert not offenders, (
        f"these modules construct a model client directly: {offenders}. "
        f"Every client must come from llm_factory.get_llm()/get_async_llm(), or "
        f"HIVE_PROVIDER silently stops working for them."
    )


def test_purpose_tables_cover_the_same_purposes_on_every_provider():
    # A purpose defined for one provider and missing on another is a crash that
    # only fires on the provider nobody ran locally.
    tables = list(llm_factory._PURPOSE_KWARGS.values())
    first = set(tables[0])

    for table in tables[1:]:
        assert set(table) == first

    # Every node's purpose string must actually exist.
    assert first == {"planner", "ticket", "editor", "reviewer"}


def test_unknown_purpose_and_tier_are_rejected():
    with pytest.raises(ValueError, match="Unknown LLM purpose"):
        llm_factory._resolve("archaeologist", "fast")

    with pytest.raises(ValueError, match="Unknown model tier"):
        llm_factory._resolve("editor", "colossal")


def test_default_provider_is_local():
    from multi_hive import config

    # The default has to stay Ollama. An accidental flip to a paid API would be
    # discovered by a bill, not by a test — so it is discovered by this test.
    assert config.PROVIDER == "ollama"
    assert config.MODELS["fast"] == "qwen2.5-coder:7b"


def test_an_unknown_provider_fails_at_import_not_at_the_first_llm_call():
    # Failing late means failing *after* the planner has already run and the user
    # is watching a progress spinner.
    import multi_hive.config as config

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HIVE_PROVIDER", "openai-but-imaginary")

        with pytest.raises(ValueError, match="is not a provider"):
            importlib.reload(config)

    importlib.reload(config)  # restore the real module for everyone else


def test_anthropic_tiers_resolve_to_claude_models(monkeypatch):
    import multi_hive.config as config

    monkeypatch.setenv("HIVE_PROVIDER", "anthropic")
    importlib.reload(config)

    try:
        assert config.MODELS["strong"] == "claude-fable-5"
        assert config.MODELS["fast"].startswith("claude-")

        # The tier names are provider-independent. That is what lets the router,
        # the editor's escalation, and the sprint footer stay untouched.
        assert set(config.MODELS) == {"fast", "strong"}
    finally:
        monkeypatch.delenv("HIVE_PROVIDER")
        importlib.reload(config)


def test_an_async_client_is_never_reused_across_event_loops(monkeypatch):
    """
    The bug this pins cost a benchmark run and looked exactly like a model failure.

    An async client's connection pool is bound to the loop that built it. The
    bench called asyncio.run() once per task, so task 2 got task 1's client and
    every call raised `Event loop is closed`. It never surfaced as an infra error:
    the editor caught it, retried, got the identical error, the repeat-error
    circuit breaker fired, and the task went to the human gate having never
    reached a model — scored as `✗ FAIL (no code)` on a task the 7B passes in 80
    seconds. A benchmark that invents failures is worse than none, because it is
    believed.
    """
    built: list[object] = []

    class FakeClient:
        pass

    def fake_build(purpose, tier):
        client = FakeClient()
        built.append(client)
        return client

    monkeypatch.setattr(llm_factory, "_build", fake_build)
    llm_factory.invalidate_llm("editor")

    async def get_one():
        return llm_factory.get_async_llm("editor", "fast")

    # Same loop: cached, so the client is reused.
    async def twice_in_one_loop():
        return llm_factory.get_async_llm("editor", "fast"), llm_factory.get_async_llm(
            "editor", "fast"
        )

    import asyncio as aio

    first, second = aio.run(twice_in_one_loop())
    assert first is second, "a client must be cached within a single loop"

    # New loop: the cached client belongs to a closed one and must NOT come back.
    third = aio.run(get_one())
    assert third is not first, (
        "get_async_llm handed back a client bound to a closed event loop — "
        "every call on it raises `Event loop is closed`, which the editor will "
        "retry into the circuit breaker and report as a model failure"
    )

    llm_factory.invalidate_llm("editor")


def test_sampling_params_are_stripped_only_for_models_that_reject_them():
    """
    fable-5 (the strong tier) and the rest of the Claude 5 / Opus 4.7+ family return
    a 400 on temperature/top_p/top_k — they were removed from those models. haiku-4-5
    (the fast tier) is a 4.5 model and still accepts them. This is what let the fast
    tier run clean while every escalation to fable-5 died with `temperature is
    deprecated for this model`, scoring capability failures that were really config.
    """
    assert llm_factory._rejects_sampling_params("claude-fable-5")
    assert llm_factory._rejects_sampling_params("claude-sonnet-5")
    assert llm_factory._rejects_sampling_params("claude-opus-4-8")

    # The fast tier must keep its temperature — it accepts it, and the measured
    # numbers were taken with it set.
    assert not llm_factory._rejects_sampling_params("claude-haiku-4-5-20251001")
    assert not llm_factory._rejects_sampling_params("qwen2.5-coder:7b")


def test_build_drops_temperature_for_fable_but_not_for_haiku(monkeypatch):
    """The strip happens where the client is actually built, so pin it there."""
    captured: dict[str, dict] = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import sys
    import types

    fake_mod = types.ModuleType("langchain_anthropic")
    fake_mod.ChatAnthropic = FakeChatAnthropic

    monkeypatch.setattr(llm_factory, "PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-real")
    monkeypatch.setitem(sys.modules, "langchain_anthropic", fake_mod)
    # The editor kwargs table carries temperature=0.1; MODELS[tier] supplies the model.
    monkeypatch.setattr(llm_factory, "MODELS", {"fast": "claude-haiku-4-5-20251001", "strong": "claude-fable-5"})

    llm_factory._build("editor", "strong")
    assert "temperature" not in captured, "fable-5 rejects temperature — it must be stripped"

    captured.clear()
    llm_factory._build("editor", "fast")
    assert captured.get("temperature") == 0.1, "haiku accepts temperature — it must survive"


def test_anthropic_without_a_key_says_so_instead_of_failing_obscurely(monkeypatch):
    monkeypatch.setattr(llm_factory, "PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    llm_factory.invalidate_llm("editor")

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        llm_factory.get_llm("editor", "fast")
