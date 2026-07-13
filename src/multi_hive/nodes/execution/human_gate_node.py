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

    # ── Clear blocking state ──────────────────────────────────────────────────
    # current_task=None + editor_error=None + empty queue makes reviewer_logic
    # return "retrospector_node" on the next evaluation, ending the sprint
    # cleanly instead of re-entering the editor loop.
    loop_health["escalated"] = True
    loop_health["last_node"] = "human_gate_node"

    return {
        "current_task": None,
        "task_queue": [],
        "editor_error": None,
        "loop_health": loop_health,
    }
