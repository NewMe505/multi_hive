"""
Discovery turns the hive's own escalations into its next backlog.

Two tests here carry real weight:

- `test_a_replay_does_not_repeat_the_failing_run` — without the tier floor,
  discovery is a nodding loop that re-runs known-broken work on the model that
  broke it.
- `test_a_task_that_beats_the_ladder_is_parked_not_retried_forever` — the open
  door for human review, held open by a counter instead of by good intentions.
"""
from __future__ import annotations

import pytest

from multi_hive import discovery
from multi_hive.config import JOURNAL_FILE
from multi_hive.core import journal
from multi_hive.core.model_router import STRONG


@pytest.fixture(autouse=True)
def _fresh_journal():
    if JOURNAL_FILE.exists():
        JOURNAL_FILE.unlink()
    yield
    if JOURNAL_FILE.exists():
        JOURNAL_FILE.unlink()


ESCALATION = [{"file": "outputs/wrap.py", "task": "wrap text", "retries": 3}]


# ── Finding work ──────────────────────────────────────────────────────────────


def test_an_empty_journal_finds_nothing():
    """
    The normal answer, most of the time. A discovery source that always finds
    work is not a discovery source, it is a treadmill.
    """
    assert discovery.discover() == []


def test_an_escalated_sprint_becomes_a_work_item():
    journal.record_sprint("wrap it", journal.ESCALATED, escalations=ESCALATION)

    items = discovery.discover()
    assert len(items) == 1
    assert items[0].objective == "wrap it"
    assert items[0].attempt == 2
    assert "outputs/wrap.py" in items[0].reason


def test_a_replay_does_not_repeat_the_failing_run():
    """
    The load-bearing one.

    agent_router_node seeds every fresh task with select_tier(editor_retries=0),
    which returns "fast". So an objective replayed verbatim would run on the exact
    model that already failed it and reproduce the identical failure — the loop
    re-doing known-broken work at machine speed and calling it progress.

    The tier floor is what makes the retry a retry.
    """
    journal.record_sprint("wrap it", journal.ESCALATED, escalations=ESCALATION)
    assert discovery.discover()[0].tier_floor == STRONG


def test_the_objective_is_replayed_byte_for_byte():
    """The contract must survive the round trip, or it stops being ground truth."""
    objective = (
        "Implement a word wrapper. Save it to outputs/wrap.py\n\n"
        "ACCEPTANCE outputs/wrap.py\n"
        'assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]\n'
    )
    journal.record_sprint(objective, journal.ESCALATED)
    assert discovery.discover()[0].objective == objective


# ── Not finding work ──────────────────────────────────────────────────────────


def test_a_clean_sprint_is_not_work():
    journal.record_sprint("wrap it", journal.CLEAN)
    assert discovery.discover() == []


def test_a_task_that_has_ever_passed_is_left_alone():
    """
    A human re-ran it and escalated it again — that is a human at work, not a
    backlog item. The loop barging in behind them is exactly the unhelpful
    autonomy this system exists to avoid.
    """
    journal.record_sprint("wrap it", journal.CLEAN)
    journal.record_sprint("wrap it", journal.ESCALATED)
    assert discovery.discover() == []


def test_a_budget_stopped_sprint_is_not_retried():
    """
    A sprint the governor halted did not fail on its merits, and retrying it
    changes nothing — the budget is still gone. Auto-retrying into a spent budget
    is how you build a loop that spends its whole night discovering it has no
    money. A human raises the cap.
    """
    journal.record_sprint("wrap it", journal.FAILED)
    assert discovery.discover() == []


def test_one_work_item_per_objective_not_one_per_escalation(monkeypatch):
    """Discovery emits work, not history. Three escalations are still one task."""
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 10)

    journal.record_sprint("wrap it", journal.ESCALATED)
    journal.record_sprint("wrap it", journal.ESCALATED)
    journal.record_sprint("wrap it", journal.ESCALATED)

    items = discovery.discover()
    assert len(items) == 1
    assert items[0].attempt == 4  # three burned, this is the fourth


# ── The open door ─────────────────────────────────────────────────────────────


def test_a_task_that_beats_the_ladder_is_parked_not_retried_forever(monkeypatch):
    """
    MAX_DISCOVERY_ATTEMPTS is what keeps the open door for human review from
    quietly closing. Once the fast model AND the strong model have both escalated
    a task, the ladder is out of rungs — a third automated attempt is not a retry,
    it is a nodding loop with extra steps.
    """
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)

    journal.record_sprint("wrap it", journal.ESCALATED)  # attempt 1, fast
    assert len(discovery.discover()) == 1  # -> retry at strong

    journal.record_sprint("wrap it", journal.ESCALATED, source=journal.SOURCE_ESCALATION)
    assert discovery.discover() == []  # attempt 2 burned; no attempt 3


def test_parked_work_is_surfaced_rather_than_silently_dropped(monkeypatch):
    """
    A loop that silently stops trying looks exactly like a loop with nothing to
    do, and the difference matters enormously to whoever reads the digest.
    """
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)

    journal.record_sprint("wrap it", journal.ESCALATED, escalations=ESCALATION)
    journal.record_sprint("wrap it", journal.ESCALATED, escalations=ESCALATION)

    assert discovery.discover() == []
    stuck = discovery.parked()
    assert len(stuck) == 1
    assert "Needs a human" in stuck[0].reason


# ── The seam discovery depends on ─────────────────────────────────────────────


def test_the_router_honours_the_tier_floor():
    """
    Discovery's whole retry story rests on agent_router_node actually respecting
    the floor. If it does not, `tier_floor=STRONG` is a comment, the replay runs
    on the fast model that already failed, and the loop nods along.

    "wrap text" classifies trivial, so the router would seed FAST on its own.
    """
    from multi_hive.core.model_router import FAST, classify_complexity
    from multi_hive.nodes.execution.agent_router_node import agent_router_node

    task = "wrap text at a given width"
    assert classify_complexity(task) == "trivial"

    without = agent_router_node({"current_task": task, "tier_floor": None})
    assert without["model_tier"] == FAST

    with_floor = agent_router_node({"current_task": task, "tier_floor": STRONG})
    assert with_floor["model_tier"] == STRONG


def test_the_operators_pin_still_beats_the_floor(monkeypatch):
    """
    HIVE_FORCE_TIER exists so a benchmark can pin a tier, and a benchmark that is
    silently un-pinned by a routing rule is not a benchmark. The floor is a
    routing rule. It does not get to outrank the operator.
    """
    from multi_hive.core import model_router
    from multi_hive.core.model_router import FAST, select_tier

    monkeypatch.setattr(model_router, "FORCE_TIER", FAST)
    assert select_tier("trivial", tier_floor=STRONG) == FAST


def test_parked_work_that_later_passes_stops_being_parked(monkeypatch):
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)

    journal.record_sprint("wrap it", journal.ESCALATED)
    journal.record_sprint("wrap it", journal.ESCALATED)
    assert len(discovery.parked()) == 1

    journal.record_sprint("wrap it", journal.CLEAN)  # a human fixed it
    assert discovery.parked() == []


# ── Work that FAILED before the ladder engaged ───────────────────────────────


def test_a_sprint_that_failed_before_the_ladder_is_not_lost(monkeypatch):
    """
    _unfinished only looks at ESCALATED, and BOTH discover() and parked() were built
    on it. So a work item whose only record was FAILED — the ticket writer emitted
    nothing writable, the graph crashed, the budget ran out — appeared in NEITHER.

    Never retried. Never handed back. Never mentioned. It simply stopped existing,
    which is the one outcome an autonomous loop is never allowed to produce.
    """
    journal.record_sprint(
        "wrap it",
        journal.FAILED,
        failure="PATH ERROR: every ticket named a file outside workspace",
    )

    # Not auto-retried: a FAILED sprint did not lose an argument with the reviewers,
    # something underneath it broke. Retrying that automatically is how you build a
    # loop that spends its night rediscovering that Ollama is down.
    assert discovery.discover() == []

    stuck = discovery.parked()
    assert len(stuck) == 1
    assert "before the retry ladder even engaged" in stuck[0].reason
    assert "PATH ERROR" in stuck[0].reason  # and it says WHY
    assert stuck[0].tier_floor is None  # nothing was learned; do not presume a tier


def test_the_two_kinds_of_stuck_are_not_reported_as_the_same_thing(monkeypatch):
    """
    Reporting "failed before the ladder engaged" as "the ladder is out of rungs"
    would be a lie — and it is the lie a human reads at 9am.
    """
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 2)

    journal.record_sprint("beat the ladder", journal.ESCALATED)
    journal.record_sprint("beat the ladder", journal.ESCALATED)
    journal.record_sprint("never reached it", journal.FAILED, failure="boom")

    reasons = {i.objective: i.reason for i in discovery.parked()}
    assert "out of rungs" in reasons["beat the ladder"]
    assert "before the retry ladder even engaged" in reasons["never reached it"]


def test_a_failed_item_that_also_escalated_is_ordinary_backlog(monkeypatch):
    """An item with an ESCALATED record is already owned by _unfinished; no double."""
    monkeypatch.setattr(discovery, "MAX_DISCOVERY_ATTEMPTS", 5)

    journal.record_sprint("wrap it", journal.ESCALATED)
    journal.record_sprint("wrap it", journal.FAILED, failure="ollama was down")

    assert len(discovery.discover()) == 1  # still retryable — the crash cost it nothing
    assert discovery.parked() == []
