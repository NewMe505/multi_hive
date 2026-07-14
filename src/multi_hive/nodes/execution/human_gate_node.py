"""
human_gate_node — the escalation interrupt.

When the circuit breaker fires (retry cap reached, or a repeat-error
fingerprint detected), the graph routes here instead of straight to the
retrospector. This node writes a structured escalation to the ledger, prints
an alert, and waits for the operator to acknowledge — with a timeout, so a
headless or CI run terminates cleanly instead of hanging forever.

The failure mode this exists to prevent: a loop stuck retrying while the human
who could fix it is never told.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from rich.panel import Panel

from multi_hive.config import GATE_TIMEOUT_SEC, MAX_RETRIES
from multi_hive.core.console import console
from multi_hive.core.memory import log_rejection


async def human_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    editor_error = state.get("editor_error") or "unknown error"
    current_task = state.get("current_task") or "unknown task"
    active_file = state.get("active_file") or "unknown file"
    retries = state.get("editor_retries", 0)
    loop_health = dict(state.get("loop_health") or {})
    gate_event = state.get("human_gate_event")

    # ── Structured ledger entry ───────────────────────────────────────────────
    log_rejection(
        "human_gate_node",
        json.dumps(
            {
                "type": "ESCALATION",
                "task": current_task,
                "file": active_file,
                "retries": retries,
                "repeat_hash": loop_health.get("repeat_error_hash"),
                "error_preview": str(editor_error)[:400],
                "timestamp": time.time(),
            }
        ),
    )

    # ── Operator alert ────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold red]🚨 HUMAN GATE — Sprint requires intervention[/]\n\n"
            f"[yellow]Task:[/]         {current_task}\n"
            f"[yellow]File:[/]         {active_file}\n"
            f"[yellow]Retries:[/]      {retries} / {MAX_RETRIES}\n"
            f"[yellow]Error hash:[/]   {loop_health.get('repeat_error_hash', 'n/a')}\n\n"
            f"[dim]{str(editor_error)[:300]}[/]\n\n"
            f"[bold]Press Enter to acknowledge and skip this task "
            f"(auto-continues in {GATE_TIMEOUT_SEC}s)[/]",
            border_style="red",
            title="⚠  ESCALATION",
        )
    )

    # ── Await acknowledgement ─────────────────────────────────────────────────
    if isinstance(gate_event, asyncio.Event):
        try:
            await asyncio.wait_for(gate_event.wait(), timeout=float(GATE_TIMEOUT_SEC))
            console.print("[dim]Gate acknowledged — continuing to retrospector.[/]")
            gate_event.clear()  # Reset so a second gate in the same sprint works.
        except asyncio.TimeoutError:
            console.print(
                f"[yellow]⚠  Gate timeout after {GATE_TIMEOUT_SEC}s — "
                f"continuing automatically.[/]"
            )
            log_rejection(
                "human_gate_node",
                f"GATE TIMEOUT: no acknowledgement after {GATE_TIMEOUT_SEC}s. "
                f"Skipped task: {current_task!r}",
            )
    else:
        console.print("[dim]Headless mode — no gate_event, routing to retrospector.[/]")

    # ── Skip the failed task. Keep the rest of the queue. ─────────────────────
    #
    # This used to return `task_queue: []`, throwing the whole queue away.
    #
    # So an escalation on ticket 1 of 3 silently cancelled tickets 2 and 3 — they
    # were never attempted, and nothing said so. If the file being graded belonged
    # to ticket 2, the sprint scored "no code": a verdict indistinguishable from
    # "the model cannot write this", for work the model was never asked to do. One
    # hard file poisoned every easy file queued behind it.
    #
    # An escalation is a statement about ONE task — this one beat the retry ladder
    # and a human should look at it. It says nothing about the tasks behind it, and
    # it has no business speaking for them. In a project whose whole premise is
    # MULTI-file generation, abandoning the other files on the first hard one is a
    # failure at the headline use case.
    #
    # `sprint_escalated` is the sticky flag, and it is why this is safe. It is set
    # here and NEVER reset — unlike loop_health, which agent_router_node zeroes at
    # the start of each task (correctly: a stale repeat_error_hash would otherwise
    # trip an escalation on the first retry of an unrelated task). Without a sticky
    # flag the sprint would go on to finish the remaining files and then report
    # itself CLEAN, hiding the very escalation this gate exists to announce. That
    # silence is the exact failure the gate was built to prevent.
    #
    # Termination: the queue strictly shrinks. Each visit here pops one task, so
    # the worst case is one escalation per task and then the retrospector.
    queue = list(state.get("task_queue") or [])

    loop_health["escalated"] = True
    loop_health["last_node"] = "human_gate_node"

    if not queue:
        return {
            "current_task": None,
            "task_queue": [],
            "editor_error": None,
            "loop_health": loop_health,
            "sprint_escalated": True,
        }

    nxt = queue.pop(0)
    console.print(
        f"[dim]Skipping the escalated task — {len(queue) + 1} left. "
        f"Continuing with [cyan]{nxt.get('file', '?')}[/].[/]"
    )

    return {
        "current_task": nxt.get("task"),
        "active_file": nxt.get("file"),
        "task_queue": queue,
        "editor_error": None,
        "editor_retries": 0,
        "loop_health": loop_health,
        "sprint_escalated": True,
    }
