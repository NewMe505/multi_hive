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

Three properties are load-bearing, and they all say the same thing: **a budget
guard that fails open is not a budget guard.**

**The check happens before the call, not after.** `on_llm_start` raises;
`on_llm_end` records. A ceiling that is only enforced once the tokens are already
spent is an audit log wearing a cap's clothing. This ordering means the governor
can overshoot by at most one call, never by an unbounded number.

**An unpriced model is priced at the most expensive rate we know.** If a new model
name shows up and we cannot price it, we assume the worst and let the cap trip
early — a false stop is an annoyance, a false pass is a bill.

**A meter that cannot read a response stops the run rather than guessing zero.**
This was the hole, and it was a real one. `_tokens_from` could not parse every
response shape, and when it could not it returned `(0, 0)` — which `record()` added
as zero tokens and $0.00, indistinguishable from a genuinely free call. So one
change to a provider's usage field would make the meter read zero *forever*:
`HIVE_MAX_USD` could never trip, and an overnight loop would bill the entire night
while reporting that it had spent nothing. The old docstring even excused it —
"the wall-clock ceiling still bounds the run either way" — and `HIVE_MAX_WALL_SEC`
defaults to **0, i.e. off**. The named backstop did not exist.

Now an unreadable response is counted as *unreadable*, not as free, and once
`HIVE_MAX_UNMETERED` of them pile up the run stops — but only when a ceiling
actually DEPENDS on the meter (`HIVE_MAX_USD` / `HIVE_MAX_TOKENS`). Stopping then
is not "the budget is spent". It is "I can no longer tell", and continuing to spend
money you cannot count is precisely the failure this module exists to prevent.

A free local run sets no spend ceiling, so it is never stopped over bookkeeping it
does not need. See `Governor.meter_is_load_bearing`.

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
    MAX_UNMETERED,
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
#
# The maximum on BOTH axes independently — not the entry with the highest output
# price. Those happen to be the same row today, so the old `max(..., key=r[1])` was
# correct by luck. Add one model priced (20.00, 2.00) and an unknown model would
# have been billed at (10, 50), under-counting its input by 2x, inside the one
# function whose entire declared job is to fail closed.
_UNKNOWN_MODEL_RATE = (
    max(r[0] for r in _PRICING_USD_PER_MTOK.values()),
    max(r[1] for r in _PRICING_USD_PER_MTOK.values()),
)


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
    # Calls that RETURNED CONTENT but reported no usage. See Governor.breach.
    #
    # This is the fail-open hole. _tokens_from() returns (0, 0) for any response
    # shape it does not recognise, and record() then adds 0 tokens and $0.00 — so a
    # single change in a provider's usage field makes the meter read zero forever,
    # HIVE_MAX_USD never trips, and an overnight loop bills the whole night. A budget
    # guard that fails open is not a budget guard; this counter is what closes it.
    unmetered: int = 0
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
            "unmetered": self.unmetered,
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
        max_unmetered: int = MAX_UNMETERED,
    ) -> None:
        self.max_usd = max_usd
        self.max_tokens = max_tokens
        self.max_wall_sec = max_wall_sec
        self.max_sprints = max_sprints
        self.max_unmetered = max_unmetered
        self.spend = Spend()
        self._lock = threading.Lock()

    # ── Enforcement ───────────────────────────────────────────────────────────

    @property
    def meter_is_load_bearing(self) -> bool:
        """
        True when a ceiling can only be enforced if the meter actually works.

        HIVE_MAX_USD and HIVE_MAX_TOKENS are computed FROM the meter. If the meter
        silently reads zero, neither of them can ever trip.

        HIVE_MAX_WALL_SEC and HIVE_MAX_SPRINTS are not — they are counted by the
        clock and by the supervisor, and they hold no matter what the model reports.
        So when only those are set, an unmeterable response costs us bookkeeping
        accuracy and nothing else, and stopping the run over it would be a worse
        trade than continuing.
        """
        return bool(self.max_usd or self.max_tokens)

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

        # ── The meter itself has failed ───────────────────────────────────────
        #
        # This is the fail-open hole, closed.
        #
        # _tokens_from() cannot read every possible response shape, and when it
        # cannot it returns (0, 0) — so record() adds zero tokens and $0.00, the
        # spend total never grows, HIVE_MAX_USD never trips, and an overnight loop
        # bills the entire night while reporting that it spent nothing. One change
        # to a provider's usage field is all it takes. The module docstring above
        # says "a budget guard that fails open is not a budget guard", and until now
        # that is exactly what this was.
        #
        # So: if a ceiling DEPENDS on the meter and the meter has failed this many
        # times, stop. Not because the budget is spent, but because we can no longer
        # tell — and continuing to spend money you cannot count is the failure the
        # governor exists to prevent.
        #
        # Guarded by meter_is_load_bearing so a free local run, which sets no
        # spend ceiling, is never stopped over bookkeeping it does not need.
        if (
            self.meter_is_load_bearing
            and self.max_unmetered
            and s.unmetered >= self.max_unmetered
        ):
            return (
                f"{s.unmetered} model calls returned content but reported no token "
                f"usage — the meter is broken, so HIVE_MAX_USD/HIVE_MAX_TOKENS cannot "
                f"be enforced. Refusing to keep spending money that cannot be counted "
                f"(HIVE_MAX_UNMETERED)"
            )

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

    def record_unmetered(self) -> None:
        """A call that returned content and reported no usage. See breach()."""
        with self._lock:
            self.spend.unmetered += 1
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
        "unmetered": now.unmetered - before.unmetered,
    }


# ── The meter ─────────────────────────────────────────────────────────────────


def _tokens_from(response: Any) -> tuple[int, int] | None:
    """
    (input, output) token counts from a LangChain LLMResult, or None if the usage
    could not be read at all.

    **None and (0, 0) are different answers, and conflating them was the bug.**

    This used to return (0, 0) for a response it could not parse. record() then
    added zero tokens and $0.00 — indistinguishable from a genuinely free call. So
    a single change in a provider's usage field would make the meter read zero
    forever: HIVE_MAX_USD would never trip, and an overnight loop would bill the
    whole night while reporting that it had spent nothing.

    The docstring even excused it — "the wall-clock ceiling still bounds the run
    either way" — and HIVE_MAX_WALL_SEC defaults to **0, i.e. off**. The named
    backstop did not exist.

    None now means "I could not meter this", the meter counts it, and the governor
    stops the run once a ceiling that DEPENDS on the meter can no longer be trusted.
    See Governor.breach.

    Two shapes, because two providers. `usage_metadata` is the modern
    provider-neutral field, and the Anthropic client populates it. The Ollama client
    reports `prompt_eval_count` / `eval_count` in response_metadata — the same
    fields bench/runner.py already reads — and older langchain-ollama does not fill
    usage_metadata at all, hence the fallback rather than a single path.

    (The provider client classes are deliberately not named here: the guard in
    tests/test_llm_factory.py is a substring scan, which is what makes it impossible
    to fool, and keeping it that way is worth more than the phrasing.)
    """
    try:
        generation = response.generations[0][0]
    except (AttributeError, IndexError):
        return None

    message = getattr(generation, "message", None)

    usage = getattr(message, "usage_metadata", None)
    if usage:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    meta = getattr(message, "response_metadata", None) or {}
    if "eval_count" in meta or "prompt_eval_count" in meta:
        return int(meta.get("prompt_eval_count", 0)), int(meta.get("eval_count", 0))

    return None


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
        #
        # THIS HANDLER MUST STAY SYNCHRONOUS. langchain_core's callback manager
        # documents that an ASYNC handler driven through the SYNC handle_event path
        # has its exceptions "always logged and swallowed, regardless of the
        # handler's raise_error setting". Half this system's nodes are sync
        # (ticket_writer, sprint_planner, reviewer_node), so promoting _Meter to an
        # AsyncCallbackHandler — an obvious-looking modernisation — would silently
        # disable the budget check for every one of them, and no test would fail.
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
            usage = _tokens_from(response)
            if usage is None:
                # The call happened and we cannot say what it cost. Counting it as
                # $0.00 is what let the meter fail open; count it as *unreadable*
                # instead, and let the governor decide whether that is survivable.
                current().record_unmetered()
                return
            current().record(self.model_name, *usage)

    return _Meter(model)
