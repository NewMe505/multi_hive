"""
supervisor.py — the loop that runs itself.

    multi-hive --loop            discover, work, repeat, until there is nothing
                                 left or the budget is gone
    multi-hive --digest          what the loop did while you were asleep

Everything below the loop already existed. The sprint runs, the reviewers reject,
the ladder escalates, the gate escalates to a human — and then a human had to
press Enter to make any of it happen again. This module is the part that does not
need the human, and it is deliberately the smallest module in the system, because
the interesting engineering is not here. It is in the thing that can say no.

## Why this terminates

An autonomous loop that cannot prove it stops is not a feature, it is an incident.
Three independent things bound this one, and they are independent on purpose:

1. **The governor.** Checked before every sprint and before every model call
   inside it. `HIVE_MAX_USD` / `HIVE_MAX_TOKENS` / `HIVE_MAX_WALL_SEC` /
   `HIVE_MAX_SPRINTS`.
2. **The attempt cap.** `discovery` stops handing back a work item after
   `MAX_DISCOVERY_ATTEMPTS`, so the backlog is strictly finite. Each sprint either
   resolves an item or burns one of its attempts.
3. **The progress check.** If a full discovery round completes zero sprints, the
   loop stops rather than spinning. This is the backstop for the case the other
   two do not cover — see below, it is the one that actually bites.

## The bug this module was written around

A crashing sprint writes no journal record. No journal record means
`attempts_for()` does not advance. Which means `discover()` returns the *same
work item* on the next pass, forever, at whatever rate the crash happens — a
tight, free, infinite loop that never spends a token and therefore never trips the
governor.

So a crash is journalled as FAILED **by the supervisor**, explicitly, precisely so
the attempt counter advances. The loop's termination proof rests on every sprint
producing exactly one journal record, whatever happens to it, and that guarantee
lives here rather than in the happy path where it would be easy to lose.

## Headless

`StdinBroker` is started but nothing is typed into it, so it hits EOF, and
`_acknowledge_on_input` treats EOF as "there is no human and no keypress is ever
coming" and auto-acknowledges every gate. That path already existed for CI. The
supervisor is simply its first real user.
"""
from __future__ import annotations

import asyncio
import random
import time
import traceback

from rich.panel import Panel
from rich.table import Table

from multi_hive import discovery
from multi_hive.cli import StdinBroker, run_sprint
from multi_hive.config import ensure_workspace
from multi_hive.core import governor, journal
from multi_hive.core.console import console


async def run(watch_sec: float = 0.0) -> int:
    """
    Discover, work, repeat. Returns the number of sprints completed.

    `watch_sec` > 0 keeps the loop alive when there is no work, re-checking on
    that interval — for the cron-less case where the process just stays up. The
    default (0) exits as soon as the backlog is empty, which is what you want from
    a scheduled invocation: do the work, write it down, get out.
    """
    ensure_workspace()

    broker = StdinBroker()
    broker.start()  # never fed; EOF is what makes the human gate auto-acknowledge

    gov = governor.current()
    completed = 0

    # Every (work item, attempt number) this process has actually run.
    #
    # The attempt counter lives in the journal, and journal writes are best-effort
    # — they swallow OSError, because a sprint that did real work and then died
    # because it could not write its own bookkeeping would be a bad trade. But it
    # means termination cannot be allowed to *depend* on that write landing: a full
    # disk would leave the counter frozen, discovery would hand back the same item
    # forever, and the loop would spin for free without ever troubling the
    # governor.
    #
    # So the process keeps its own memory. If discovery offers work this run has
    # already done, the journal is not advancing and the loop stops, whatever the
    # disk thinks.
    attempted: set[tuple[str, int]] = set()

    console.print(
        Panel.fit(
            "[bold yellow]🐝 HIVE SUPERVISOR[/]\n"
            "[dim]discovering its own work — no human in the execution path[/]\n"
            f"[dim]{_ceilings()}[/]",
            border_style="yellow",
        )
    )

    try:
        while True:
            breach = gov.breach()
            if breach:
                console.print(f"[bold yellow]💸 Stopping: {breach}[/]")
                break

            items = discovery.discover()
            if not items:
                if watch_sec <= 0:
                    console.print("[dim]Nothing to discover. The backlog is empty.[/]")
                    break
                console.print(f"[dim]Nothing to discover — waiting {watch_sec:.0f}s.[/]")
                await asyncio.sleep(watch_sec)
                continue

            repeats = [i for i in items if (i.key, i.attempt) in attempted]
            if repeats:
                console.print(
                    "[bold red]⚠  Discovery is handing back work this run already did "
                    "— the journal is not advancing (disk full?). Stopping rather than "
                    "spinning.[/]"
                )
                break

            console.print(f"\n[bold]🔎 Discovered {len(items)} work item(s).[/]")

            progressed = 0
            for item in items:
                if gov.breach():
                    break

                console.print(
                    f"\n[bold cyan]▶ attempt {item.attempt}[/] "
                    f"[dim]({item.reason})[/]\n"
                    f"[dim]  tier floor: {item.tier_floor} — the model that already "
                    f"failed this does not get another turn[/]"
                )

                attempted.add((item.key, item.attempt))
                progressed += await _work(item, broker)
                gov.record_sprint()
                completed += 1

            # The backstop. If a full round completed nothing, the loop is not
            # making progress and another pass will not change that. Spinning here
            # is free, which is exactly why the governor would never notice it.
            if progressed == 0:
                console.print("[bold red]⚠  A discovery round completed no work. Stopping.[/]")
                break
    finally:
        gov.flush(note="supervisor run")
        await broker.close()

    digest()
    return completed


async def _work(item: discovery.WorkItem, broker: StdinBroker) -> int:
    """
    One sprint. Returns 1 if it produced a journal record, 0 if it did not.

    The `except` here is not defensive clutter — it is the loop's termination
    proof. A sprint that crashes writes no journal record; no record means
    `attempts_for()` never advances; and an unadvancing counter means `discover()`
    hands back the same item on the next pass, forever, in a tight free loop that
    spends no tokens and so never trips the governor.

    Journalling the crash is what makes the attempt count monotonic. Do not remove
    it because "run_sprint handles its own errors" — it handles the errors it
    knows about, and this is here for the ones it does not.
    """
    try:
        outcome = await run_sprint(
            item.objective,
            broker,
            source=journal.SOURCE_ESCALATION,
            attempt=item.attempt,
            tier_floor=item.tier_floor,
        )
        return 0 if outcome.budget_exhausted else 1
    except Exception as e:  # noqa: BLE001 — see the docstring; this is load-bearing
        console.print(f"[bold red]❌ Sprint crashed: {e}[/]")
        console.print(f"[dim]{traceback.format_exc()}[/]")
        journal.record_sprint(
            objective=item.objective,
            status=journal.FAILED,
            source=journal.SOURCE_ESCALATION,
            attempt=item.attempt,
        )
        return 1  # it advanced the counter, which is the only progress required


def _ceilings() -> str:
    g = governor.current()
    parts = []
    if g.max_usd:
        parts.append(f"${g.max_usd:.2f}")
    if g.max_tokens:
        parts.append(f"{g.max_tokens} tokens")
    if g.max_wall_sec:
        parts.append(f"{g.max_wall_sec:.0f}s")
    if g.max_sprints:
        parts.append(f"{g.max_sprints} sprints")
    return "ceilings: " + (", ".join(parts) if parts else "NONE SET — nothing will stop this")


# ── The digest ────────────────────────────────────────────────────────────────


def digest(limit: int = 10) -> None:
    """
    What the loop did, for the person who was not watching.

    This is the defense against the two silent costs no amount of engineering
    fixes — comprehension rot and cognitive surrender. The codebase grows, the
    mental map does not, and eventually the operator stops having an opinion about
    the code and just reads the green checkmark.

    There is no clever mechanism for that. The only defense is to read the
    machine's output, on a schedule, and be able to explain it. The digest exists
    to make that cheap, NOT to make it optional — which is why it ends by naming
    one specific file and asking you to go read it, rather than printing a
    reassuring total and letting you get on with your day.
    """
    sprints = journal.read_sprints()
    if not sprints:
        console.print("[dim]Nothing in the journal yet.[/]")
        return

    recent = sprints[-limit:]

    table = Table(title=f"Last {len(recent)} sprints", title_justify="left")
    table.add_column("when", style="dim")
    table.add_column("status")
    table.add_column("by")
    table.add_column("tier")
    table.add_column("cost", justify="right")
    table.add_column("objective")

    for s in recent:
        status = s.get("status", "?")
        colour = {
            journal.CLEAN: "green",
            journal.ESCALATED: "red",
            journal.FAILED: "yellow",
        }.get(status, "white")
        by = "hive" if s.get("source") == journal.SOURCE_ESCALATION else "human"
        usd = (s.get("spend") or {}).get("usd", 0.0)
        first_line = (s.get("objective") or "").strip().splitlines()[:1]

        table.add_row(
            time.strftime("%H:%M", time.localtime(s.get("timestamp", 0))),
            f"[{colour}]{status}[/]",
            by,
            s.get("tier") or "—",
            f"${usd:.4f}" if usd else "—",
            (first_line[0] if first_line else "")[:48],
        )

    console.print()
    console.print(table)

    total_usd = sum((s.get("spend") or {}).get("usd", 0.0) for s in sprints)
    machine = [s for s in sprints if s.get("source") == journal.SOURCE_ESCALATION]
    console.print(
        f"[dim]{len(sprints)} sprints all-time, {len(machine)} of them the hive's own "
        f"idea. Total spend ${total_usd:.4f}.[/]"
    )

    # ── Work handed back ──────────────────────────────────────────────────────
    stuck = discovery.parked()
    if stuck:
        console.print(
            Panel(
                "\n".join(
                    f"[yellow]•[/] {i.objective.strip().splitlines()[0][:60]}\n"
                    f"  [dim]{i.reason}[/]"
                    for i in stuck
                ),
                title="🚪 Parked — the loop gave up and is handing these back",
                border_style="yellow",
            )
        )

    # ── The nudge ─────────────────────────────────────────────────────────────
    #
    # Deliberately a specific file, not a suggestion to "review the output". The
    # whole failure mode is that reviewing in general is easy to skip and reading
    # one named thing is not.
    landed = [s for s in machine if s.get("status") == journal.CLEAN]
    if landed:
        pick = random.choice(landed)
        objective = (pick.get("objective") or "").strip().splitlines()[0][:60]
        console.print(
            Panel(
                f"[bold]Go read this one.[/]\n\n"
                f"[cyan]{objective}[/]\n"
                f"[dim]The hive wrote it, the hive approved it, and nobody has "
                f"looked at it.[/]\n\n"
                f"[dim]If you cannot explain what it does, the loop is now writing "
                f"code that nobody in the world understands — and it is doing that "
                f"at machine speed.[/]",
                border_style="cyan",
            )
        )
