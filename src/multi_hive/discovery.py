"""
discovery.py — the hive finds its own work.

Every objective this system has ever run was typed by a human into a REPL. The
hive is an extremely good executor of a queue that a person has to hand-curate,
which means the person is still the bottleneck, and the bottleneck is the boring
part. That is the blind loop.

The first source of work is the most obvious one, and the hive was already
producing it and throwing it away: **its own escalations.** When a task defeats
the retry ladder, `human_gate_node` records exactly what it got stuck on. That
record is now in the journal, and a record of unfinished work is a backlog.

## What makes a retry a retry

A rediscovered objective is replayed **verbatim** — same text, same ACCEPTANCE
contract, byte for byte. The only thing that changes is the tier floor.

That is not a detail, it is the entire design. `agent_router_node` seeds every
fresh task with `select_tier(..., editor_retries=0)`, which returns *fast* — so an
objective replayed as-is would run on the exact model that already failed it and
reproduce the identical failure. Discovery would be re-doing known-broken work at
machine speed and reporting it as progress. That is the nodding loop, just slow
enough to look like diligence.

So a rediscovered item carries `tier_floor=STRONG`: the fast model has
demonstrably failed this task, and another attempt from it is the same bet. This
is the escalation ladder's own logic, carried across sprints rather than thrown
away at the end of each one.

## What is deliberately NOT retried

- **CLEAN sprints.** Obviously.
- **Anything that has ever passed.** See `journal.is_resolved` — a task a human
  re-ran and escalated again is a human at work, not a backlog item, and the loop
  barging in behind them is exactly the sort of unhelpful autonomy this system
  exists to avoid.
- **FAILED sprints, which includes budget-exhausted ones.** A sprint the governor
  halted did not fail on its merits and retrying it changes nothing — the budget
  is still gone. A human raises the cap. Auto-retrying into a spent budget is how
  you build a loop that spends its whole night discovering it has no money.
- **Anything past `MAX_DISCOVERY_ATTEMPTS`.** It gets **parked**, and parking is
  the point: it is the open door for human review, held open by a counter rather
  than by anyone's good intentions at 3am.

## What is NOT done here, on purpose

The escalation record carries an `error_preview`. Feeding it back into the
editor's prompt on the retry is the obvious next idea, and it is not done, because
the editor's failure feeds are split three ways — generation / runtime / semantic
— precisely because "each implies a different fix, and merging them produces
incoherent retries where the model does not know what kind of wrong it was". The
escalation record does not say which kind it was. Mis-filing it would degrade the
retry while looking like an improvement.

That change needs `bench.py sprint` to justify it, not a plausible argument. This
codebase's whole culture is measure-it-or-don't-ship-it, and this module is not
where that ends.
"""
from __future__ import annotations

from dataclasses import dataclass

from multi_hive.config import MAX_DISCOVERY_ATTEMPTS
from multi_hive.core import journal
from multi_hive.core.model_router import STRONG


@dataclass(frozen=True)
class WorkItem:
    """One unit of work the hive found for itself."""

    objective: str
    """The raw objective text, contracts included. Replayed verbatim."""

    key: str
    """Stable identity — journal.key_for(objective). Survives across sprints."""

    attempt: int
    """Which attempt this will be. 1 was the original run."""

    tier_floor: str | None
    """A tier the router may not go below. This is what makes the retry differ."""

    reason: str
    """Why the hive picked this up. Goes on the console, and into the digest."""


def _unfinished(sprints: list[dict]) -> dict[str, dict]:
    """
    The newest ESCALATED sprint per work item, for items with no CLEAN run ever.

    Keyed by work item, so an objective that escalated three times yields one
    entry, not three. Discovery emits work, not history.
    """
    unfinished: dict[str, dict] = {}
    for sprint in sprints:
        key = sprint.get("key")
        if not key or sprint.get("status") != journal.ESCALATED:
            continue
        if journal.is_resolved(key, sprints):
            continue
        unfinished[key] = sprint  # later sprints overwrite earlier ones
    return unfinished


def discover(sprints: list[dict] | None = None) -> list[WorkItem]:
    """
    The next work items, oldest escalation first.

    Returns an empty list when there is nothing to do, which is the normal and
    correct answer most of the time. A discovery source that always finds work is
    not a discovery source, it is a treadmill.
    """
    records = journal.read_sprints() if sprints is None else sprints

    items: list[WorkItem] = []
    for key, sprint in _unfinished(records).items():
        attempts = journal.attempts_for(key, records)
        if attempts >= MAX_DISCOVERY_ATTEMPTS:
            continue  # parked — see parked() below

        stuck_on = _stuck_on(sprint)
        items.append(
            WorkItem(
                objective=sprint["objective"],
                key=key,
                attempt=attempts + 1,
                # The whole reason a replay is not a no-op.
                tier_floor=STRONG,
                reason=f"escalated on attempt {attempts}{stuck_on}",
            )
        )

    items.sort(key=lambda i: i.key)  # stable order; the journal decides fairness
    return items


def parked(sprints: list[dict] | None = None) -> list[WorkItem]:
    """
    Work the loop has given up on and is handing back to a human.

    This is the open door. `MAX_DISCOVERY_ATTEMPTS` closes discovery's hands, and
    this is what makes the result *visible* rather than merely dropped — a loop
    that silently stops trying looks exactly like a loop with nothing to do, and
    the difference matters enormously to the person reading the digest.
    """
    records = journal.read_sprints() if sprints is None else sprints

    items: list[WorkItem] = []
    for key, sprint in _unfinished(records).items():
        attempts = journal.attempts_for(key, records)
        if attempts < MAX_DISCOVERY_ATTEMPTS:
            continue

        stuck_on = _stuck_on(sprint)
        items.append(
            WorkItem(
                objective=sprint["objective"],
                key=key,
                attempt=attempts,
                tier_floor=STRONG,
                reason=(
                    f"escalated {attempts}x — the ladder is out of rungs"
                    f"{stuck_on}. Needs a human."
                ),
            )
        )

    items.sort(key=lambda i: i.key)
    return items


def _stuck_on(sprint: dict) -> str:
    """The file the sprint escalated on, if it recorded one."""
    escalations = sprint.get("escalations") or []
    if not escalations:
        return ""
    target = escalations[-1].get("file")
    return f" on {target}" if target else ""
