"""
journal.py — what survives the sprint.

The hive had persistence, but none of it persisted anything worth having:

- `rejection_ledger.jsonl` is wiped by `clear_ledger()` at the start of every
  sprint. That is correct for what it is — the editor's memory of the mistakes it
  made on *this* task — but it means the structured escalations `human_gate_node`
  so carefully records were deleted before anything could read them.
- `LOOP.md` is overwritten each sprint.
- `metrics.jsonl` is append-only, and records *timings*. Not what was learned.

So the hive already knew how to say "I got stuck here, on this, for this reason",
and then threw it away. Every run started from nothing. That is the amnesiac loop:
it re-does the same work, and re-discovers the same wall, forever.

The journal is the fix, and it is deliberately dumb: one append-only JSONL file,
one record per sprint, never cleared. It is not a database and it should not
become one. Its whole job is to let tomorrow's run know what yesterday's run
already tried.

The consumer is `discovery.py`, which reads escalated-and-unresolved sprints back
out and turns them into the next run's work queue — which is only possible because
this file outlives the sprint that wrote it.

## The identity of a work item

An objective is keyed by the SHA-1 of its **raw text, contracts included**. That
is the exact string `run_sprint` takes, so a discovered work item can be replayed
verbatim rather than reconstructed. Reconstructing it would mean re-deriving the
ACCEPTANCE block, and a contract that does not survive a round-trip is a contract
that silently stops being the ground truth.

Two objectives with the same text are the same work. Change a character and it is
new work — which is right: an edited objective genuinely is a different ask, and
the retry counter *should* reset when a human rewrites the task.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from multi_hive.config import JOURNAL_FILE

# Sprint outcomes. CLEAN is the only one that retires a work item.
CLEAN = "CLEAN"
ESCALATED = "ESCALATED"
FAILED = "FAILED"

# Where a sprint's objective came from. Kept so the digest can answer the question
# that actually matters at 9am: how much of last night was the machine's own idea?
SOURCE_HUMAN = "human"
SOURCE_ESCALATION = "discovery:escalation"


def key_for(objective: str) -> str:
    """The stable identity of a work item: SHA-1 of its raw text, contracts and all."""
    return hashlib.sha1(objective.encode("utf-8")).hexdigest()[:12]


def record_sprint(
    objective: str,
    status: str,
    *,
    tier: str | None = None,
    escalations: list[dict[str, Any]] | None = None,
    spend: dict[str, Any] | None = None,
    source: str = SOURCE_HUMAN,
    attempt: int = 1,
    wall_time_sec: float = 0.0,
    failure: str = "",
) -> dict[str, Any]:
    """
    Appends one sprint to the journal and returns the record.

    `objective` is stored in full and never truncated. It is the replay payload,
    and half an ACCEPTANCE block is worse than none — it would either corrupt what
    the human asserted or fail to compile, and the contract is the one part of the
    input that is meant to be exactly, literally true.

    Best-effort on write: a sprint that did real work and then died because it
    could not journal it would be a bad trade. A failed write costs us tomorrow's
    memory of today; a raised exception costs us today.
    """
    record: dict[str, Any] = {
        "type": "sprint",
        "key": key_for(objective),
        "timestamp": time.time(),
        "status": status,
        "objective": objective,
        "tier": tier,
        "source": source,
        "attempt": attempt,
        "wall_time_sec": round(wall_time_sec, 1),
        # WHY it failed, for the human reading the digest. Without this, parked()
        # can only say "something underneath it broke" — which is true, useless, and
        # exactly the kind of unactionable report that trains people to stop reading
        # the digest at all.
        "failure": (failure or "")[:300],
        "escalations": escalations or [],
        "spend": spend or {},
    }

    try:
        JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
        with JOURNAL_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass

    return record


def read_sprints() -> list[dict[str, Any]]:
    """
    Every sprint ever journalled, oldest first. Malformed lines are skipped.

    `errors="replace"` is load-bearing, and leaving it out was a real bug.

    The try/except below guards `json.loads` — but the decoding happens in
    `for line in f`, OUTSIDE it. A torn write leaves invalid UTF-8, the iterator
    raises UnicodeDecodeError (a ValueError, not a JSONDecodeError), and nothing
    catches it. The comment promising "the rest of the file is still good" was false
    for the exact failure it named.

    And a torn write is reachable. A record stores the objective in full and never
    truncates it, so records routinely exceed the 8 KiB buffer and land as several
    write() calls. Two appenders — a cron `--loop` overlapping a REPL, which is the
    deployment the governor was built for — can split a multi-byte character.

    The blast radius was the entire loop: `discover()` inside supervisor's
    `while True`, `digest()`, and `--digest` all read this file, and JOURNAL_FILE is
    never cleared. One torn byte would brick the autonomous loop until a human
    hand-edited the file.
    """
    if not JOURNAL_FILE.exists():
        return []

    sprints: list[dict[str, Any]] = []
    with JOURNAL_FILE.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn write; the rest of the file really is still good
            if isinstance(record, dict) and record.get("type") == "sprint":
                sprints.append(record)

    return sprints


# A sprint that ran on its own merits. A FAILED record is neither of these: it means
# the sprint never got a fair attempt.
_MERIT = (CLEAN, ESCALATED)


def attempts_for(key: str, sprints: list[dict[str, Any]] | None = None) -> int:
    """
    How many times this work item has actually been TRIED on its merits.

    Deliberately not "to any outcome", which is what this used to count and which
    silently retired real work.

    `discovery` parks an item once attempts >= MAX_DISCOVERY_ATTEMPTS (default 2).
    A FAILED record is written when the sprint never got a fair attempt at all:
    the supervisor journals a crash as FAILED so the counter advances and the loop
    cannot spin, and a budget-exhausted sprint is FAILED too.

    Counting those as attempts meant:

        human runs objective X            -> ESCALATED   (attempt 1)
        `--loop` picks it up, Ollama is down -> FAILED    (attempt 2)
        -> attempts == 2 -> PARKED FOREVER

    Zero tokens were spent. The strong model — the entire point of the
    `tier_floor=STRONG` replay — never ran. And `parked()` then hands it to a human
    saying "escalated 2x — the ladder is out of rungs", when the ladder was never
    climbed. The one artefact a human reads would be asserting something false.

    Same shape, worse: one `HIVE_MAX_USD` misconfiguration during an overnight run
    could permanently retire every item in the backlog, and raising the cap would
    not bring them back.

    Termination does not depend on this count. The supervisor keeps its own
    in-process `attempted` set, which is what actually stops a journal that is not
    advancing (see supervisor.run).
    """
    records = read_sprints() if sprints is None else sprints
    return sum(1 for s in records if s.get("key") == key and s.get("status") in _MERIT)


def is_resolved(key: str, sprints: list[dict[str, Any]] | None = None) -> bool:
    """
    True once this work item has *ever* landed a CLEAN sprint.

    Deliberately "ever" and not "most recently". A work item that passed and was
    then re-run by a human who escalated it again is not unfinished business for
    the loop to pick up on its own — it is a human actively working on something,
    and the loop barging in to retry it behind them is exactly the kind of
    unhelpful autonomy this system is supposed to avoid.
    """
    records = read_sprints() if sprints is None else sprints
    return any(s.get("key") == key and s.get("status") == CLEAN for s in records)
