"""
The journal exists to answer one question: what did yesterday's run already try?

The test that matters most is the last one — that an escalation recorded by
human_gate_node actually survives the clear_ledger() that would otherwise delete
it. Everything the discovery layer does rests on that record still being there.
"""
from __future__ import annotations

import json

import pytest

from multi_hive.config import JOURNAL_FILE
from multi_hive.core import journal, memory


@pytest.fixture(autouse=True)
def _fresh_journal():
    if JOURNAL_FILE.exists():
        JOURNAL_FILE.unlink()
    yield
    if JOURNAL_FILE.exists():
        JOURNAL_FILE.unlink()


# ── Identity ──────────────────────────────────────────────────────────────────


def test_same_objective_is_the_same_work():
    assert journal.key_for("write a wrapper") == journal.key_for("write a wrapper")


def test_an_edited_objective_is_new_work():
    """
    A human who rewrites the task has changed the ask, and the retry counter
    SHOULD reset. This is a feature, not a hash collision hazard.
    """
    assert journal.key_for("write a wrapper") != journal.key_for("write a wrapper.")


# ── Persistence ───────────────────────────────────────────────────────────────


def test_a_sprint_survives_being_written():
    journal.record_sprint("build a thing", journal.CLEAN, tier="fast")
    sprints = journal.read_sprints()
    assert len(sprints) == 1
    assert sprints[0]["status"] == journal.CLEAN
    assert sprints[0]["objective"] == "build a thing"


def test_the_objective_is_stored_in_full_so_it_can_be_replayed():
    """
    The objective is the replay payload, contracts included. Half an ACCEPTANCE
    block is worse than none: it would either corrupt what the human asserted or
    fail to compile, and the contract is the one part of the input that is meant
    to be exactly, literally true.
    """
    objective = (
        "Implement a word wrapper. Save it to outputs/wrap.py\n\n"
        "ACCEPTANCE outputs/wrap.py\n"
        'assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]\n'
    )
    journal.record_sprint(objective, journal.ESCALATED)
    assert journal.read_sprints()[0]["objective"] == objective


def test_a_torn_line_does_not_destroy_the_rest_of_the_file():
    journal.record_sprint("first", journal.CLEAN)
    with JOURNAL_FILE.open("a", encoding="utf-8") as f:
        f.write('{"type": "sprint", "key": "trunc\n')  # a half-flushed write
    journal.record_sprint("second", journal.CLEAN)

    objectives = [s["objective"] for s in journal.read_sprints()]
    assert objectives == ["first", "second"]


def test_attempts_accumulate_per_work_item():
    journal.record_sprint("task A", journal.ESCALATED)
    journal.record_sprint("task B", journal.CLEAN)
    journal.record_sprint("task A", journal.ESCALATED)

    assert journal.attempts_for(journal.key_for("task A")) == 2
    assert journal.attempts_for(journal.key_for("task B")) == 1


def test_a_work_item_that_ever_passed_is_resolved():
    key = journal.key_for("task A")
    journal.record_sprint("task A", journal.ESCALATED)
    assert not journal.is_resolved(key)

    journal.record_sprint("task A", journal.CLEAN)
    assert journal.is_resolved(key)


def test_resolution_is_ever_not_most_recently():
    """
    A task that passed and was then re-run by a human who escalated it again is
    not unfinished business for the loop to pick up. It is a human actively
    working on something, and the loop barging in behind them to retry it is
    exactly the kind of unhelpful autonomy this system exists to avoid.
    """
    key = journal.key_for("task A")
    journal.record_sprint("task A", journal.CLEAN)
    journal.record_sprint("task A", journal.ESCALATED)
    assert journal.is_resolved(key)


# ── The record that was being thrown away ─────────────────────────────────────


def test_an_escalation_survives_the_ledger_wipe():
    """
    The whole reason this module exists.

    human_gate_node writes a structured ESCALATION into the rejection ledger — and
    clear_ledger() deletes it at the start of the next sprint. So the hive knew how
    to say "I got stuck here, on this, for this reason" and then threw it away
    before anything could read it. That is the amnesiac loop.

    Harvest it at the end of the sprint, put it in the journal, and it outlives the
    wipe.
    """
    memory.clear_ledger()
    memory.log_rejection(
        "human_gate_node",
        json.dumps(
            {
                "type": "ESCALATION",
                "task": "implement word_wrap",
                "file": "outputs/wrap.py",
                "retries": 3,
                "repeat_hash": "a1b2c3d4",
                "error_preview": "AssertionError",
            }
        ),
    )

    escalations = memory.get_escalations()
    assert len(escalations) == 1
    journal.record_sprint("wrap it", journal.ESCALATED, escalations=escalations)

    memory.clear_ledger()  # the next sprint starts, and the ledger is gone

    assert memory.get_escalations() == []  # ...as it should be
    survived = journal.read_sprints()[0]["escalations"]  # ...but the journal kept it
    assert survived[0]["task"] == "implement word_wrap"
    assert survived[0]["repeat_hash"] == "a1b2c3d4"


def test_unstructured_gate_lines_are_not_mistaken_for_escalations():
    """
    human_gate_node logs BOTH a structured JSON escalation and a plain-text gate
    timeout under the same node name. Only the structured ones are replayable.
    """
    memory.clear_ledger()
    memory.log_rejection("human_gate_node", "GATE TIMEOUT: no acknowledgement after 120s.")
    assert memory.get_escalations() == []


# ── A torn write must not brick the loop ─────────────────────────────────────


def test_invalid_utf8_does_not_crash_the_reader():
    """
    The try/except in read_sprints guards json.loads — but decoding happens in
    `for line in f`, OUTSIDE it. Invalid UTF-8 raised UnicodeDecodeError from the
    iterator and nothing caught it.

    JOURNAL_FILE is never cleared, and discover() / digest() / --digest all read it,
    so one torn byte bricked the whole autonomous loop until a human hand-edited the
    file.
    """
    journal.record_sprint("first", journal.CLEAN)
    with JOURNAL_FILE.open("ab") as f:
        f.write(b'{"type": "sprint", "objective": "\xff\xfe torn"}\n')
    journal.record_sprint("second", journal.CLEAN)

    objectives = [s["objective"] for s in journal.read_sprints()]
    assert "first" in objectives and "second" in objectives


# ── A crash must not permanently retire real work ────────────────────────────


def test_a_crash_does_not_count_as_an_attempt():
    """
    attempts_for used to count records "to any outcome". discovery parks an item at
    2 attempts. So:

        human runs it       -> ESCALATED (attempt 1)
        --loop retries it, Ollama is down -> FAILED (attempt 2)
        -> parked forever, having spent zero tokens

    The strong model — the entire point of the tier_floor replay — never ran, and
    parked() would tell a human "the ladder is out of rungs" when it was never
    climbed. One HIVE_MAX_USD misconfiguration could retire a whole backlog.
    """
    key = journal.key_for("wrap it")

    journal.record_sprint("wrap it", journal.ESCALATED)  # a real attempt
    assert journal.attempts_for(key) == 1

    journal.record_sprint("wrap it", journal.FAILED)  # a crash / spent budget
    assert journal.attempts_for(key) == 1, "a crash was counted as an attempt"

    journal.record_sprint("wrap it", journal.ESCALATED)  # a second real attempt
    assert journal.attempts_for(key) == 2
