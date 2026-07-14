"""
An autonomous loop that cannot prove it stops is not a feature, it is an incident.

Every test in this file is a termination proof. The sprint itself is stubbed out —
`tests/test_loop_terminates.py` already pins the graph. What is under test here is
the loop *around* the graph, and specifically the three independent things that
bound it: the governor, the attempt cap, and the progress check.

The one that actually bites is `test_a_crashing_sprint_cannot_spin_forever`.
"""
from __future__ import annotations

import asyncio

import pytest

from multi_hive import discovery, supervisor
from multi_hive.cli import SprintOutcome
from multi_hive.config import JOURNAL_FILE
from multi_hive.core import governor, journal


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    if JOURNAL_FILE.exists():
        JOURNAL_FILE.unlink()
    governor.reset()
    # A real StdinBroker spawns a daemon thread parked in readline(). The
    # supervisor never feeds it, so it is pure overhead in a test.
    monkeypatch.setattr(supervisor, "StdinBroker", _NullBroker)
    yield
    governor.reset()
    if JOURNAL_FILE.exists():
        JOURNAL_FILE.unlink()


class _NullBroker:
    def start(self): ...
    async def close(self): ...


def _sprints_run(monkeypatch, outcome_status=journal.ESCALATED, crash=False):
    """Stub run_sprint. Returns the list it appends one entry to per call."""
    calls: list[dict] = []

    async def fake_run_sprint(objective, broker, *, source, attempt, tier_floor):
        calls.append({"objective": objective, "attempt": attempt, "tier_floor": tier_floor})
        if crash:
            raise RuntimeError("the sandbox exploded")
        journal.record_sprint(objective, outcome_status, source=source, attempt=attempt)
        return SprintOutcome(key=journal.key_for(objective), status=outcome_status)

    monkeypatch.setattr(supervisor, "run_sprint", fake_run_sprint)
    return calls


# ── Termination ───────────────────────────────────────────────────────────────


def test_an_empty_backlog_stops_immediately(monkeypatch):
    calls = _sprints_run(monkeypatch)
    completed = asyncio.run(supervisor.run())
    assert completed == 0
    assert calls == []


def test_the_attempt_cap_bounds_the_backlog(monkeypatch):
    """
    Each sprint either resolves a work item or burns one of its attempts, so the
    backlog is strictly finite. With a cap of 2 and one escalated objective
    already on the books, the loop gets exactly one more go at it and stops.
    """
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)
    calls = _sprints_run(monkeypatch, outcome_status=journal.ESCALATED)

    journal.record_sprint("wrap it", journal.ESCALATED)  # attempt 1, by a human

    completed = asyncio.run(supervisor.run())
    assert completed == 1
    assert len(calls) == 1
    assert calls[0]["attempt"] == 2
    # ...and it escalated again, so it is now parked rather than retried.
    assert discovery.discover() == []
    assert len(discovery.parked()) == 1


def test_a_crashing_sprint_cannot_spin_forever(monkeypatch):
    """
    The bug this module was written around.

    A crashing sprint writes no journal record. No record means attempts_for()
    never advances. Which means discover() hands back the SAME work item on the
    next pass — forever, in a tight loop that spends no tokens and therefore never
    trips the governor. Free, silent, and infinite.

    The supervisor journals the crash itself, explicitly, so the attempt counter
    advances no matter what. Without that, this test hangs.
    """
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)
    calls = _sprints_run(monkeypatch, crash=True)

    journal.record_sprint("wrap it", journal.ESCALATED)

    completed = asyncio.run(asyncio.wait_for(supervisor.run(), timeout=10))

    assert completed == 1
    assert len(calls) == 1  # tried once, crashed, did NOT try again
    assert journal.attempts_for(journal.key_for("wrap it")) == 2  # the crash was recorded


def test_a_resolved_item_leaves_the_backlog(monkeypatch):
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 5)
    _sprints_run(monkeypatch, outcome_status=journal.CLEAN)

    journal.record_sprint("wrap it", journal.ESCALATED)

    completed = asyncio.run(supervisor.run())
    assert completed == 1  # it passed on the retry, so there is nothing left
    assert discovery.discover() == []


def test_a_journal_that_cannot_be_written_does_not_spin_the_loop(monkeypatch):
    """
    Termination must not depend on a disk write landing.

    Journal writes are best-effort — they swallow OSError, because a sprint that
    did real work and then died over its own bookkeeping would be a bad trade. But
    a full disk would freeze the attempt counter, discovery would hand back the
    same item forever, and the loop would spin for free without ever troubling the
    governor.

    So the supervisor keeps its own in-process memory of what it has run. Without
    it, this test hangs.
    """
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 99)
    calls = _sprints_run(monkeypatch)

    journal.record_sprint("wrap it", journal.ESCALATED)
    monkeypatch.setattr(journal, "record_sprint", lambda *a, **k: {})  # the disk is full

    completed = asyncio.run(asyncio.wait_for(supervisor.run(), timeout=10))

    assert completed == 1
    assert len(calls) == 1  # it noticed the counter was stuck and stopped


# ── The governor ──────────────────────────────────────────────────────────────


def test_a_spent_budget_stops_the_loop_before_it_starts(monkeypatch):
    calls = _sprints_run(monkeypatch)
    journal.record_sprint("wrap it", journal.ESCALATED)

    g = governor.reset(max_tokens=1)
    g.record("qwen2.5-coder:7b", 1, 0)  # already over

    completed = asyncio.run(supervisor.run())
    assert completed == 0
    assert calls == []  # the sprint was never even attempted


def test_the_sprint_ceiling_stops_the_loop(monkeypatch):
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 99)
    calls = _sprints_run(monkeypatch, outcome_status=journal.ESCALATED)

    governor.reset(max_sprints=3)
    journal.record_sprint("wrap it", journal.ESCALATED)

    completed = asyncio.run(asyncio.wait_for(supervisor.run(), timeout=10))
    assert completed == 3
    assert len(calls) == 3


# ── The retry is a real retry ─────────────────────────────────────────────────


def test_the_supervisor_replays_on_the_tier_that_has_not_failed_yet(monkeypatch):
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)
    calls = _sprints_run(monkeypatch)

    journal.record_sprint("wrap it", journal.ESCALATED)
    asyncio.run(supervisor.run())

    assert calls[0]["tier_floor"] == "strong"
    assert calls[0]["objective"] == "wrap it"  # replayed byte for byte
