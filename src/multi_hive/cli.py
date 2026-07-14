"""
cli.py — the REPL entrypoint.

Run with `multi-hive`, or `python -m multi_hive`.

    [USER_OBJECTIVE] > Implement a word wrapper. Save it to outputs/wrap.py
    [USER_OBJECTIVE] > @tasks/wrap.md          <- load the objective from a file
    multi-hive --objective tasks/wrap.md       <- one sprint, no REPL

Objectives from a file
----------------------
An objective may carry a human-written ACCEPTANCE contract (see contract.py),
and a contract is several lines of Python. The REPL reads one line, so the two
do not fit together: hence `@path` and `--objective path`, which are the only
sane way to hand the hive a contract. The one-shot form is also what CI wants.

stdin ownership
---------------
Two things want to read the keyboard: the REPL (waiting for the next
objective) and human_gate_node (waiting for an escalation acknowledgement).
They never want it at the same time — during a sprint the REPL is idle, and
between sprints the gate cannot fire — so a single reader thread pumps lines
into a queue and whoever is waiting consumes them.

One daemon thread, started once, pumps lines into an asyncio.Queue. It is a
daemon thread and not a ThreadPoolExecutor worker for a specific reason: a
blocking readline() cannot be cancelled, so the reader is always parked inside
one. concurrent.futures registers an atexit handler that JOINS every live
non-daemon worker, so a clean `exit`/`quit` hung the interpreter inside that
join until the user pressed Enter one more time to release the parked read. A
daemon thread is reaped at interpreter exit instead of pinning it open. It is
started once and never restarted — an earlier per-sprint listener leaked a
parked thread per sprint until the pool was exhausted and the REPL went deaf.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from rich.panel import Panel

from multi_hive import __version__
from multi_hive.config import (
    MAX_INPUT_CHARS,
    RECURSION_LIMIT,
    WORKSPACE_DIR,
    ensure_workspace,
)
from multi_hive.contract import ContractError, assert_count, parse_objective
from multi_hive.core import governor, journal, llm_factory
from multi_hive.core.console import console
from multi_hive.core.governor import BudgetExhausted
from multi_hive.core.memory import clear_ledger, get_escalations
from multi_hive.core.metrics import SprintMetrics
from multi_hive.orchestrator import hive_app
from multi_hive.state import HiveState, default_loop_health

_EOF = object()


class StdinBroker:
    """Single stdin reader. Both the REPL and the human gate consume from it."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._pump, name="hive-stdin", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        # Runs in the daemon thread. asyncio.Queue is not thread-safe, so items
        # are handed to the loop thread via call_soon_threadsafe rather than put
        # on the queue directly.
        while True:
            line = sys.stdin.readline()
            item = _EOF if line == "" else line.rstrip("\n")  # "" is EOF (Ctrl-D/pipe)
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
            if item is _EOF:
                return

    async def readline(self) -> str | None:
        """Next line, or None at EOF."""
        item = await self._queue.get()
        if item is _EOF:
            # EOF is sticky. There are two consumers — the REPL and the gate's
            # acknowledgement task — and whichever reaches the marker first would
            # otherwise swallow it, leaving the other blocked on a queue that no
            # producer will ever fill again. Putting it back makes EOF idempotent.
            self._queue.put_nowait(_EOF)
            return None
        return item

    async def close(self) -> None:
        # Nothing to join: the pump is a daemon thread. If it is parked in a
        # blocking readline() it is reaped at interpreter exit — it cannot hold
        # the process open, and a blocking read could not be cancelled anyway.
        pass


async def _acknowledge_on_input(broker: StdinBroker, gate_event: asyncio.Event) -> None:
    """
    Sets gate_event when the operator presses Enter during a sprint.

    At EOF it sets the event too. EOF means stdin is a pipe or a closed terminal —
    there is no human, and no keypress is ever coming. Waiting out the full
    GATE_TIMEOUT_SEC in that situation stalls every escalation for two minutes to
    poll a person who does not exist. The timeout is for the case where a human
    *could* answer and did not; EOF is the case where they could not.

    Safe to cancel: it waits on an asyncio.Queue, not on a blocking read.
    """
    while True:
        line = await broker.readline()
        gate_event.set()
        if line is None:  # EOF — headless. Auto-acknowledge every future gate.
            return


@dataclass
class SprintOutcome:
    """
    What one sprint did, for whoever is driving it.

    `run_sprint` used to return None, which was fine when the only caller was a
    human at a REPL who could read the panel. The supervisor cannot read a panel.
    """

    key: str
    status: str  # journal.CLEAN | journal.ESCALATED | journal.FAILED
    error: str | None = None
    tier: str | None = None
    budget_exhausted: bool = False
    wall_time: float = 0.0
    escalations: list[dict[str, Any]] = field(default_factory=list)
    spend: dict[str, Any] = field(default_factory=dict)


async def run_sprint(
    user_input: str,
    broker: StdinBroker,
    *,
    source: str = journal.SOURCE_HUMAN,
    attempt: int = 1,
    tier_floor: str | None = None,
) -> SprintOutcome:
    clear_ledger()

    # The objective handed to the planner is the one with the ACCEPTANCE blocks
    # removed. A planner shown a contract plans work to satisfy it — "Step 3:
    # write the tests" — which is the exact job the contract exists to take away
    # from the model. The editor gets the contract separately, per file.
    #
    # ContractError propagates: a contract that does not compile is a mistake by
    # the human at the keyboard, and it should be reported to them now, not
    # discovered forty seconds into a doomed sprint.
    objective, contracts = parse_objective(user_input)

    # Cap the prose, never the contract.
    #
    # A multi-KB objective silently overflows the ticket model's num_ctx and
    # produces garbled planning with no visible error, so it is capped. But this
    # cap must not be allowed anywhere near a contract: truncating one mid-assert
    # would either corrupt what the human asserted or, at best, fail to compile —
    # and the whole point of the contract is that it is the one part of the input
    # that is exactly, literally true. Trimming the prose is lossy. Trimming the
    # contract is a correctness bug.
    if len(objective) > MAX_INPUT_CHARS:
        console.print(
            f"[yellow]⚠️  Objective truncated from {len(objective)} to "
            f"{MAX_INPUT_CHARS} chars to stay within the planner context window.[/]"
        )
        objective = objective[:MAX_INPUT_CHARS]

    for target, body in contracts.items():
        console.print(
            f"📜 [bold]acceptance contract[/] for [cyan]{target}[/] "
            f"([dim]{assert_count(body)} asserts — the model writes none[/])"
        )

    gate_event = asyncio.Event()

    initial_state: HiveState = {
        "messages": [HumanMessage(content=objective)],
        "project_files": {},
        "active_file": "outputs/main.py",
        "task_queue": [],
        "current_task": None,
        "editor_error": None,
        "editor_retries": 0,
        "sprint_plan": "",
        "specialist_context": "",
        "is_ui_task": False,
        "loop_health": default_loop_health(),
        "semantic_verdict": None,
        "task_complexity": None,
        "model_tier": None,
        "contracts": contracts,
        "contract_satisfied": None,
        "sprint_started_at": time.monotonic(),
        "human_gate_event": gate_event,
        "tier_floor": tier_floor,
    }

    metrics = SprintMetrics()
    metrics.start()

    final_error: str | None = None
    final_loop_health: dict = {}
    final_semantic: str | None = None
    final_tier: str | None = None
    budget_exhausted = False

    # The governor's own total is cumulative for the process — that is what the
    # ceiling must be enforced against, since the supervisor runs many sprints and
    # the point is that they add up. The journal wants *this* sprint's cost, so
    # bracket the run and diff across it.
    spend_before = governor.current().snapshot()

    ack_task = asyncio.create_task(_acknowledge_on_input(broker, gate_event))

    try:
        stream = hive_app.astream(
            initial_state,
            config={"recursion_limit": RECURSION_LIMIT},
        )
        async for output in stream:
            for node_name, delta in output.items():
                # LangGraph yields a None delta for a node that wrote no state
                # (and for its own internal channels). Indexing into it kills the
                # sprint after the work is already done — the same Optional trap
                # the nodes themselves guard against with `or {}`.
                state_delta = delta or {}

                if "editor_error" in state_delta:
                    final_error = state_delta["editor_error"]
                if "loop_health" in state_delta:
                    final_loop_health = state_delta["loop_health"] or {}
                if "semantic_verdict" in state_delta:
                    final_semantic = state_delta["semantic_verdict"]
                if state_delta.get("model_tier"):
                    tier = state_delta["model_tier"]
                    if tier != final_tier:
                        console.print(
                            f"🧠 [bold magenta]tier[/] {final_tier or '—'} → "
                            f"[bold]{tier}[/] ([dim]{llm_factory.model_for(tier)}[/])"
                        )
                    final_tier = tier
                metrics.record_node(node_name)
                console.print(f"🔄 [bold cyan]{node_name}[/] executed.")
    except BudgetExhausted as e:
        # The governor stopped the run. This is the ONE place a budget stop is
        # caught, and it is caught here rather than in a node on purpose: a node's
        # `except Exception` would have turned it into an editor_error and retried
        # it, spinning against a dead budget and then blaming the model. See the
        # BudgetExhausted docstring — it is a BaseException so that cannot happen.
        budget_exhausted = True
        final_error = f"BUDGET EXHAUSTED: {e}"
    finally:
        ack_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ack_task

    metrics.stop(llm_cache_size=llm_factory.cache_size())

    # ── Post-sprint panel ─────────────────────────────────────────────────────
    escalated = final_loop_health.get("escalated", False)

    if budget_exhausted:
        console.print(
            Panel(
                f"[bold yellow]💸 Sprint Halted — Budget Exhausted[/]\n"
                f"[dim]{final_error}[/]\n\n"
                f"[dim]Nothing was retried: the ceiling is enforced before a call, so no\n"
                f"tokens were spent on the attempt that was refused. Raise the cap\n"
                f"(HIVE_MAX_USD / HIVE_MAX_TOKENS) or start a fresh process.[/]",
                border_style="yellow",
            )
        )
    elif escalated:
        console.print(
            Panel(
                f"[bold red]🚨 Sprint Escalated in {metrics.wall_time:.1f}s[/]\n"
                f"[dim]Task was not completable within MAX_RETRIES. "
                f"See the rejection ledger for details.[/]",
                border_style="red",
            )
        )
    elif final_error:
        console.print(
            Panel(
                f"[bold red]⚠️  Sprint Ended With Unresolved Error in {metrics.wall_time:.1f}s[/]\n"
                f"[dim]{str(final_error)[:300]}[/]",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]✅ Sprint Complete in {metrics.wall_time:.1f}s[/]",
                border_style="green",
            )
        )

    # ── Journal ───────────────────────────────────────────────────────────────
    #
    # Harvest the escalations BEFORE the next sprint's clear_ledger() deletes
    # them. human_gate_node has always written this record; nothing has ever read
    # it, because the ledger is wiped at the start of every sprint. This is the
    # read, and the journal is where it goes to survive.
    escalations = get_escalations()
    spend = governor.spend_since(spend_before)

    status = (
        journal.ESCALATED if escalated else journal.FAILED if final_error else journal.CLEAN
    )

    journal.record_sprint(
        objective=user_input,  # raw, contracts included — this is the replay payload
        status=status,
        tier=final_tier,
        escalations=escalations,
        spend=spend,
        source=source,
        attempt=attempt,
        wall_time_sec=metrics.wall_time,
    )

    console.print(
        f"[dim]nodes={metrics.node_count}  "
        f"peak_rss_mb={metrics.peak_rss_mb:.1f}  "
        f"llm_cache={metrics.llm_cache_size}  "
        f"attempts={final_loop_health.get('attempt_count', 0)}  "
        f"escalated={escalated}  "
        f"tier={final_tier or '—'}  "
        f"semantic={final_semantic or '—'}  "
        f"tokens={spend['total_tokens']}  "
        f"cost=${spend['usd']:.4f}[/]"
    )

    return SprintOutcome(
        key=journal.key_for(user_input),
        status=status,
        error=final_error,
        tier=final_tier,
        budget_exhausted=budget_exhausted,
        wall_time=metrics.wall_time,
        escalations=escalations,
        spend=spend,
    )


def _expand(user_input: str) -> str:
    """
    Resolves the `@path` form to the contents of that file.

    A contract is multi-line and the REPL is not, so an objective carrying one
    has to arrive from somewhere other than a single typed line. `@path` is that
    somewhere. The path is typed by the human, so it is read from the working
    directory as given — this is not the model-authored path boundary, and
    safe_path() has no business here.
    """
    if not user_input.startswith("@"):
        return user_input

    path = Path(user_input[1:].strip().strip("\"'"))
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise ContractError(f"cannot read objective file {str(path)!r}: {e}") from e


async def _sprint_once(user_input: str, broker: StdinBroker) -> bool:
    """One sprint. Returns False if the REPL should stop."""
    try:
        await run_sprint(user_input, broker)
    except ContractError as e:
        # The human's own input is malformed. Report it and stay alive — this is
        # a typo, not a crash, and killing the REPL over a typo is obnoxious.
        console.print(f"[bold red]✗ {e}[/]")
    except Exception as e:
        import traceback

        console.print(f"[bold red]❌ FATAL: {e}[/]")
        console.print(f"[dim]{traceback.format_exc()}[/]")
        console.print(f"[dim]Check {WORKSPACE_DIR} for partially written files.[/]")
        return False
    return True


async def main(objective_file: str | None = None) -> None:
    ensure_workspace()

    console.print(
        Panel.fit(
            f"[bold yellow]🐝 HIVE ARCHITECTURE v{__version__}[/]\n"
            f"[dim]Sentinel Prime — Async Self-Healing Multi-File Engine[/]\n"
            f"[dim]workspace: {WORKSPACE_DIR}[/]",
            border_style="yellow",
        )
    )

    broker = StdinBroker()
    broker.start()

    try:
        # One-shot: an objective file, one sprint, exit. The broker still runs —
        # a human at a terminal can still acknowledge an escalation, and at EOF
        # the gate auto-acknowledges, which is what CI needs.
        if objective_file:
            try:
                await _sprint_once(_expand(f"@{objective_file}"), broker)
            except ContractError as e:
                console.print(f"[bold red]✗ {e}[/]")
            return

        while True:
            console.print("\n[bold green][USER_OBJECTIVE] >[/] ", end="")

            user_input = await broker.readline()
            if user_input is None:  # EOF
                console.print("\n[dim]Shutting down.[/]")
                break

            user_input = user_input.strip()
            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Shutting down.[/]")
                break
            if not user_input:
                continue

            try:
                expanded = _expand(user_input)
            except ContractError as e:
                console.print(f"[bold red]✗ {e}[/]")
                continue

            if not await _sprint_once(expanded, broker):
                break
    finally:
        await broker.close()


def run() -> None:
    """Console-script entrypoint."""
    parser = argparse.ArgumentParser(
        prog="multi-hive",
        description="Async self-healing multi-file code generation.",
    )
    parser.add_argument(
        "--objective",
        metavar="PATH",
        help=(
            "run a single sprint from an objective file, then exit. "
            "The file may carry ACCEPTANCE contract blocks; see contract.py."
        ),
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "run unattended: discover work from the hive's own escalations, work "
            "it, repeat, until the backlog is empty or the budget is spent. "
            "Set HIVE_MAX_USD / HIVE_MAX_TOKENS before pointing this at a paid "
            "provider. See supervisor.py."
        ),
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SEC",
        help=(
            "with --loop: when the backlog is empty, wait SEC and look again "
            "instead of exiting."
        ),
    )
    parser.add_argument(
        "--digest",
        action="store_true",
        help="print what the loop has been doing, then exit. Read it.",
    )
    args = parser.parse_args()

    # Imported here, not at module scope: supervisor imports run_sprint from this
    # module, and a top-level import would be a cycle.
    from multi_hive import supervisor

    try:
        if args.digest:
            supervisor.digest()
        elif args.loop:
            asyncio.run(supervisor.run(watch_sec=args.watch))
        else:
            asyncio.run(main(args.objective))
    except KeyboardInterrupt:
        console.print("\n[dim]^C — Shutting down.[/]")


if __name__ == "__main__":
    run()
