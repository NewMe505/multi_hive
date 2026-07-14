"""
governor.py — the thing that can say stop.

The hive has good *per-sprint* backstops: MAX_RETRIES, RECURSION_LIMIT, and the
repeat-error fingerprint that escalates the moment the model starts fixing
symptoms instead of converging. What it did not have was a *cost* ceiling, which
was fine for exactly as long as inference was free and a human was watching.

It is neither, once HIVE_PROVIDER=anthropic meets a scheduler. And the escalation
ladder is what makes that sharp: the ladder climbs to the expensive tier
*precisely when a task is failing* — the situation where the loop spends the most
and produces the least.

Two properties are load-bearing:

**The check happens before the call, not after.** `on_llm_start` raises;
`on_llm_end` records. A ceiling that is only enforced once the tokens are already
spent is an audit log wearing a cap's clothing. This ordering means the governor
can overshoot by at most one call, never by an unbounded number.

**An unpriced model is priced at the most expensive rate we know.** A budget guard
that fails open is not a budget guard. If a new model name shows up and we cannot
price it, we assume the worst and let the cap trip early — a false stop is an
annoyance, a false pass is a bill.

The meter hangs off `core/llm_factory.py`, which is the only module in the system
permitted to construct a client (`tests/test_llm_factory.py` enforces this). That
makes it the one chokepoint every call must pass through, so no node can opt out
of being metered — not even by accident, which is the only way it would happen.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from multi_hive.config import (
    MAX_SPRINTS,
    MAX_TOKENS,
    MAX_USD,
    MAX_WALL_SEC,
    PROVIDER,
    SPEND_FILE,
)


class BudgetExhausted(BaseException):
    """
    A ceiling was reached. Raised from the pre-flight check, so it surfaces
    *instead of* a model call rather than after one.

    **BaseException, not Exception, and that is the whole design.**

    Every node in this system wraps its model call in `except Exception` and turns
    what it catches into an `editor_error` — which the graph then *retries*. That
    is correct for a generation failure and catastrophic for a budget stop: the
    retry calls the editor again, the meter refuses again, the node catches it
    again, and the repeat-error fingerprint eventually escalates to the human gate
    reporting a model failure that never happened. The sprint would spin against a
    dead budget and then lie about why it stopped.

    This is not hypothetical. `llm_factory` carries a scar from the identical
    shape: a stale event-loop client raised inside `ainvoke`, the editor caught it,
    retried into the circuit breaker, and a task the 7B passes in 80 seconds was
    scored `✗ FAIL (no code) [human gate]` having never once reached a model.

    A budget stop is a control-flow signal — unwind, do not handle — which puts it
    in the same family as `KeyboardInterrupt` and `SystemExit`, and that is exactly
    where it belongs. `except Exception` cannot see it. Only code that means to
    stop the loop catches it, and only `cli.py` and `supervisor.py` do.

    `tests/test_governor.py` pins this. Do not "fix" the base class.
    """


# ── Pricing ───────────────────────────────────────────────────────────────────
#
# USD per million tokens, (input, output). Matched by longest prefix, so the
# dated pin `claude-haiku-4-5-20251001` resolves against `claude-haiku-4-5`.
#
# Ollama is free. That is not a rounding error, it is the whole reason the local
# provider exists, and the governor still meters tokens there — a free loop can
# still spin forever, and HIVE_MAX_TOKENS / HIVE_MAX_WALL_SEC are what stop it.

_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5": (10.00, 50.00),
}

# The fail-safe rate for a model we cannot price. See the module docstring: a
# budget guard that fails open is not a budget guard.
_UNKNOWN_MODEL_RATE = max(_PRICING_USD_PER_MTOK.values(), key=lambda r: r[1])


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of one call. Zero on a free provider."""
    if PROVIDER != "anthropic":
        return 0.0

    matches = [k for k in _PRICING_USD_PER_MTOK if model.startswith(k)]
    rate = _PRICING_USD_PER_MTOK[max(matches, key=len)] if matches else _UNKNOWN_MODEL_RATE

    return (input_tokens * rate[0] + output_tokens * rate[1]) / 1_000_000


# ── Spend ─────────────────────────────────────────────────────────────────────


@dataclass
class Spend:
    """What has been consumed so far. Cumulative for the life of the process."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    sprints: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def wall_sec(self) -> float:
        return time.monotonic() - self.started_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usd": round(self.usd, 6),
            "calls": self.calls,
            "sprints": self.sprints,
            "wall_sec": round(self.wall_sec, 1),
        }


class Governor:
    """
    Meters spend and enforces the ceilings. One per process.

    Thread-safe because LangChain fires callbacks from whatever thread ran the
    call, and `async_editor_node` is async: two nodes can land in `record()` at
    once. An unsynchronised `+=` would drop calls, and a budget guard that
    undercounts is the failure mode that matters.
    """

    def __init__(
        self,
        max_usd: float = MAX_USD,
        max_tokens: int = MAX_TOKENS,
        max_wall_sec: float = MAX_WALL_SEC,
        max_sprints: int = MAX_SPRINTS,
    ) -> None:
        self.max_usd = max_usd
        self.max_tokens = max_tokens
        self.max_wall_sec = max_wall_sec
        self.max_sprints = max_sprints
        self.spend = Spend()
        self._lock = threading.Lock()

    # ── Enforcement ───────────────────────────────────────────────────────────

    def breach(self) -> str | None:
        """The ceiling that has been reached, as a human sentence, or None."""
        s = self.spend

        if self.max_usd and s.usd >= self.max_usd:
            return f"spend ${s.usd:.4f} reached the ${self.max_usd:.2f} ceiling (HIVE_MAX_USD)"
        if self.max_tokens and s.total_tokens >= self.max_tokens:
            return (
                f"{s.total_tokens} tokens reached the {self.max_tokens} ceiling "
                f"(HIVE_MAX_TOKENS)"
            )
        if self.max_wall_sec and s.wall_sec >= self.max_wall_sec:
            return (
                f"wall time {s.wall_sec:.0f}s reached the {self.max_wall_sec:.0f}s "
                f"ceiling (HIVE_MAX_WALL_SEC)"
            )
        if self.max_sprints and s.sprints >= self.max_sprints:
            return f"{s.sprints} sprints reached the {self.max_sprints} ceiling (HIVE_MAX_SPRINTS)"

        return None

    def check(self) -> None:
        """Raises BudgetExhausted if any ceiling has been reached."""
        reason = self.breach()
        if reason is not None:
            raise BudgetExhausted(reason)

    # ── Metering ──────────────────────────────────────────────────────────────

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.spend.input_tokens += input_tokens
            self.spend.output_tokens += output_tokens
            self.spend.usd += price(model, input_tokens, output_tokens)
            self.spend.calls += 1

    def record_sprint(self) -> None:
        """Called by the supervisor once per completed sprint."""
        with self._lock:
            self.spend.sprints += 1

    def snapshot(self) -> Spend:
        """A frozen copy of the running total, for measuring a delta across it."""
        with self._lock:
            return copy.copy(self.spend)

    # ── Audit ─────────────────────────────────────────────────────────────────

    def flush(self, note: str = "") -> None:
        """
        Appends the running total to spend.jsonl.

        Best-effort: a governor that crashed the sprint because it could not write
        its own bookkeeping would be a bad trade. The ceiling is enforced from
        memory, not from this file.
        """
        try:
            SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
            entry = {"timestamp": time.time(), "provider": PROVIDER, **self.spend.as_dict()}
            if note:
                entry["note"] = note
            with SPEND_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass


# ── Process-wide instance ─────────────────────────────────────────────────────
#
# A module-level singleton, because the ceiling is a property of the *run*, not
# of any one sprint. The supervisor runs many sprints in one process and the
# whole point is that their spend adds up.

_governor = Governor()


def current() -> Governor:
    return _governor


def reset(**kwargs: Any) -> Governor:
    """Replaces the process governor. For tests, and for the supervisor's setup."""
    global _governor
    _governor = Governor(**kwargs)
    return _governor


def spend_since(before: Spend) -> dict[str, Any]:
    """
    What has been spent since `before` was snapshotted — i.e. one sprint's cost.

    The governor's own total is cumulative for the process, which is what the
    *ceiling* has to be enforced against (the supervisor runs many sprints and the
    whole point is that they add up). But the journal wants per-sprint cost, so it
    brackets each sprint with a snapshot and diffs across it.
    """
    now = current().spend
    return {
        "input_tokens": now.input_tokens - before.input_tokens,
        "output_tokens": now.output_tokens - before.output_tokens,
        "total_tokens": now.total_tokens - before.total_tokens,
        "usd": round(now.usd - before.usd, 6),
        "calls": now.calls - before.calls,
    }


# ── The meter ─────────────────────────────────────────────────────────────────


def _tokens_from(response: Any) -> tuple[int, int]:
    """
    Pulls (input, output) token counts out of a LangChain LLMResult.

    Two shapes, because two providers. `usage_metadata` is the modern
    provider-neutral field, and the Anthropic client populates it. The Ollama
    client reports `prompt_eval_count` / `eval_count` in response_metadata — the
    same fields bench/runner.py already reads — and older langchain-ollama does
    not fill usage_metadata at all, hence the fallback rather than a single path.

    (The provider client classes are deliberately not named here: the guard in
    tests/test_llm_factory.py is a substring scan, which is what makes it
    impossible to fool, and keeping it that way is worth more than the phrasing.)

    Returns (0, 0) rather than raising if neither is present. An un-metered call
    is a bug, but crashing a sprint over a bookkeeping miss is a worse one — and
    the wall-clock ceiling still bounds the run either way.
    """
    try:
        generation = response.generations[0][0]
    except (AttributeError, IndexError):
        return 0, 0

    message = getattr(generation, "message", None)

    usage = getattr(message, "usage_metadata", None)
    if usage:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    meta = getattr(message, "response_metadata", None) or {}
    if "eval_count" in meta or "prompt_eval_count" in meta:
        return int(meta.get("prompt_eval_count", 0)), int(meta.get("eval_count", 0))

    return 0, 0


def _handler_base() -> Any:
    from langchain_core.callbacks import BaseCallbackHandler

    return BaseCallbackHandler


def meter(model: str) -> Any:
    """
    A LangChain callback that checks the ceiling before a call and records the
    spend after it.

    `model` is bound at construction rather than read off the response, because
    llm_factory builds one client per (purpose, tier) and therefore already knows
    exactly which model this handler is attached to. Reading it back out of the
    response would be a second source of truth for no gain.
    """
    base = _handler_base()

    class _Meter(base):  # type: ignore[misc, valid-type]
        # LangChain swallows exceptions raised inside a callback unless the
        # handler opts in. Without this the BudgetExhausted would be logged and
        # the call would proceed — the governor would be decorative.
        raise_error = True

        def __init__(self, model_name: str) -> None:
            super().__init__()
            self.model_name = model_name

        # Chat models fire on_chat_model_start; plain LLMs fire on_llm_start.
        # Both providers here are chat models, but binding only one of them
        # would make the guard silently provider-specific.
        def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
            current().check()

        def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
            current().check()

        def on_llm_end(self, response: Any, **kwargs: Any) -> None:
            input_tokens, output_tokens = _tokens_from(response)
            current().record(self.model_name, input_tokens, output_tokens)

    return _Meter(model)
