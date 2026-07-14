"""
The governor is the only thing standing between an unattended loop and an
unbounded bill, so these tests are mostly about the two properties that make it
real rather than decorative:

1. The ceiling is checked BEFORE a call, not after.
2. An exception raised inside the callback actually escapes LangChain.

Everything else is arithmetic.
"""
from __future__ import annotations

import pytest

from multi_hive.core import governor
from multi_hive.core.governor import BudgetExhausted, Governor


@pytest.fixture(autouse=True)
def _fresh_governor():
    """Every test gets its own process governor and leaves the next one clean."""
    yield governor.reset()
    governor.reset()


@pytest.fixture
def anthropic(monkeypatch):
    """price() short-circuits to 0.0 unless the provider actually bills."""
    monkeypatch.setattr(governor, "PROVIDER", "anthropic")


# ── Pricing ───────────────────────────────────────────────────────────────────


def test_ollama_is_free():
    # Not a rounding error — it is the reason the local provider exists.
    assert governor.price("qwen2.5-coder:7b", 1_000_000, 1_000_000) == 0.0


def test_haiku_and_fable_rates(anthropic):
    # 1M in + 1M out. Haiku: $1 + $5. Fable: $10 + $50.
    assert governor.price("claude-haiku-4-5", 1_000_000, 1_000_000) == pytest.approx(6.0)
    assert governor.price("claude-fable-5", 1_000_000, 1_000_000) == pytest.approx(60.0)


def test_dated_pin_resolves_to_its_family(anthropic):
    # config.py pins the dated name; the pricing table keys the family.
    assert governor.price("claude-haiku-4-5-20251001", 1_000_000, 0) == pytest.approx(1.0)


def test_unknown_model_is_priced_at_the_most_expensive_known_rate(anthropic):
    """
    A budget guard that fails open is not a budget guard. An unrecognised model
    must cost AT LEAST as much as the priciest one we know about, so the cap
    trips early rather than never.
    """
    unknown = governor.price("claude-something-unreleased", 1_000_000, 1_000_000)
    priciest = max(
        governor.price(m, 1_000_000, 1_000_000) for m in governor._PRICING_USD_PER_MTOK
    )
    assert unknown >= priciest


# ── Ceilings ──────────────────────────────────────────────────────────────────


def test_no_ceilings_means_no_breach():
    g = Governor(max_usd=0, max_tokens=0, max_wall_sec=0, max_sprints=0)
    g.record("claude-fable-5", 10_000_000, 10_000_000)
    assert g.breach() is None
    g.check()  # does not raise


def test_usd_ceiling_trips(anthropic):
    g = Governor(max_usd=1.00)
    g.record("claude-fable-5", 100_000, 0)  # $1.00 exactly
    with pytest.raises(BudgetExhausted, match="HIVE_MAX_USD"):
        g.check()


def test_token_ceiling_trips_on_a_free_provider():
    """The cap that matters on Ollama: free tokens still buy an infinite loop."""
    g = Governor(max_tokens=1000)
    g.record("qwen2.5-coder:7b", 600, 500)
    with pytest.raises(BudgetExhausted, match="HIVE_MAX_TOKENS"):
        g.check()


def test_sprint_ceiling_trips():
    g = Governor(max_sprints=2)
    g.record_sprint()
    g.check()
    g.record_sprint()
    with pytest.raises(BudgetExhausted, match="HIVE_MAX_SPRINTS"):
        g.check()


def test_wall_ceiling_trips(monkeypatch):
    g = Governor(max_wall_sec=60)
    g.check()
    monkeypatch.setattr(g.spend, "started_at", g.spend.started_at - 61)
    with pytest.raises(BudgetExhausted, match="HIVE_MAX_WALL_SEC"):
        g.check()


def test_spend_accumulates_across_calls(anthropic):
    g = Governor()
    g.record("claude-haiku-4-5", 1000, 2000)
    g.record("claude-haiku-4-5", 3000, 4000)
    assert g.spend.calls == 2
    assert g.spend.input_tokens == 4000
    assert g.spend.output_tokens == 6000
    assert g.spend.total_tokens == 10_000


# ── The meter ─────────────────────────────────────────────────────────────────


class _FakeMessage:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


class _FakeResult:
    def __init__(self, message):
        self.generations = [[type("G", (), {"message": message})()]]


def test_reads_anthropic_usage_metadata():
    result = _FakeResult(_FakeMessage(usage_metadata={"input_tokens": 11, "output_tokens": 22}))
    assert governor._tokens_from(result) == (11, 22)


def test_reads_ollama_eval_counts():
    # The same fields bench/runner.py already reads.
    result = _FakeResult(
        _FakeMessage(response_metadata={"prompt_eval_count": 33, "eval_count": 44})
    )
    assert governor._tokens_from(result) == (33, 44)


def test_unmeterable_response_does_not_crash_the_sprint():
    assert governor._tokens_from(_FakeResult(_FakeMessage())) == (0, 0)
    assert governor._tokens_from(object()) == (0, 0)


def test_meter_records_on_end():
    g = governor.current()
    m = governor.meter("claude-haiku-4-5")
    m.on_llm_end(_FakeResult(_FakeMessage(usage_metadata={"input_tokens": 5, "output_tokens": 7})))
    assert g.spend.input_tokens == 5
    assert g.spend.output_tokens == 7
    assert g.spend.calls == 1


def test_meter_refuses_the_call_before_it_is_made(anthropic):
    """
    The load-bearing ordering. on_llm_start must raise, so the ceiling is enforced
    INSTEAD of a call rather than after one. Enforcing it only in on_llm_end would
    make the governor an audit log: it would faithfully record every dollar it
    failed to prevent.
    """
    g = governor.reset(max_usd=0.01)
    g.record("claude-fable-5", 100_000, 0)  # $1.00 — well past the ceiling

    m = governor.meter("claude-fable-5")
    with pytest.raises(BudgetExhausted):
        m.on_chat_model_start({}, [])
    with pytest.raises(BudgetExhausted):
        m.on_llm_start({}, [])

    # And no call was counted for the attempt that never happened.
    assert g.spend.calls == 1


def test_budget_exhausted_survives_a_node_swallowing_exceptions():
    """
    The single most important line in governor.py: BudgetExhausted is a
    BaseException, so `except Exception` cannot see it.

    Every node wraps its model call in `except Exception` and converts what it
    catches into an editor_error — which the graph RETRIES. If BudgetExhausted
    were an ordinary Exception, the retry would call the editor again, the meter
    would refuse again, the node would catch it again, and the repeat-error
    breaker would eventually escalate to the human gate reporting a model failure
    that never happened. The sprint would spin against a dead budget and then lie
    about why it stopped.

    llm_factory carries a scar from exactly this shape. Do not let it recur.
    """
    assert not issubclass(BudgetExhausted, Exception)

    def a_node_doing_what_every_node_here_does():
        try:
            governor.current().check()
            return "reached the model"
        except Exception as e:  # noqa: BLE001 — this is the point of the test
            return f"swallowed and will retry: {e}"

    g = governor.reset(max_tokens=1)
    g.record("qwen2.5-coder:7b", 1, 0)

    with pytest.raises(BudgetExhausted):
        a_node_doing_what_every_node_here_does()


def test_spend_since_measures_one_sprint_not_the_whole_process(anthropic):
    g = governor.reset()
    g.record("claude-haiku-4-5", 1000, 1000)  # a previous sprint

    before = g.snapshot()
    g.record("claude-haiku-4-5", 300, 700)  # this sprint

    delta = governor.spend_since(before)
    assert delta["input_tokens"] == 300
    assert delta["output_tokens"] == 700
    assert delta["calls"] == 1
    # ...while the governor's own total — what the ceiling is enforced against —
    # still counts both.
    assert g.spend.calls == 2


def test_meter_lets_exceptions_escape_langchain():
    """
    LangChain swallows exceptions raised inside a callback handler unless the
    handler sets raise_error. Without it, BudgetExhausted would be logged and the
    call would proceed — every other test here would still pass, and the governor
    would do nothing at all in production.
    """
    assert governor.meter("claude-haiku-4-5").raise_error is True
