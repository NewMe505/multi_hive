"""
human_gate_node.py — Async human escalation interrupt (v4.1).

Loop-engineering primitive: "Human Gate" — when the circuit breaker fires
(MAX_RETRIES exhausted OR repeat-error fingerprint detected), the graph
routes here instead of directly to retrospector_node. This node:

  1. Writes a structured escalation entry to the rejection ledger
     (the loop-engineering STATE.md "High Priority — waiting on human"
     equivalent for a terminal-based system).

  2. Prints a Rich-formatted alert panel to the terminal with full
     context: task, file, retry count, error fingerprint, error preview.

  3. Awaits human_gate_event (asyncio.Event injected by run_hive_async.py)
     with GATE_TIMEOUT_SEC timeout. The event is set by a background
     stdin-listener task when the operator presses Enter.

  4. If the timeout expires without acknowledgement, logs the escalation
     failure and continues to retrospector_node anyway — the sprint
     terminates cleanly rather than hanging forever.

Failure mode covered: "Escalation Failure" (S2) from loop-engineering —
loop stuck retrying, human never notified.
"""
import asyncio
import json
import time
from typing import Dict, Any

from rich.console import Console
from rich.panel import Panel

from hive_config import MAX_RETRIES, GATE_TIMEOUT_SEC
from hive_memory import log_rejection

console = Console()


async def human_gate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    editor_error = state.get("editor_error") or "unknown error"
    current_task = state.get("current_task") or "unknown task"
    active_file  = state.get("active_file")  or "unknown file"
    retries      = state.get("editor_retries", 0)
    loop_health  = dict(state.get("loop_health") or {})
    gate_event   = state.get("human_gate_event")

    # ── 1. Structured ledger entry ──────────────────────────────────────────
    escalation_entry = {
        "type":         "ESCALATION",
        "task":         current_task,
        "file":         active_file,
        "retries":      retries,
        "repeat_hash":  loop_health.get("repeat_error_hash"),
        "error_preview": str(editor_error)[:400],
        "timestamp":    time.time(),
    }
    log_rejection("human_gate_node", json.dumps(escalation_entry))

    # ── 2. Rich terminal alert ──────────────────────────────────────────────
    repeat_hash = loop_health.get("repeat_error_hash", "n/a")
    console.print(Panel(
        f"[bold red]🚨 HUMAN GATE — Sprint requires intervention[/]\n\n"
        f"[yellow]Task:[/]         {current_task}\n"
        f"[yellow]File:[/]         {active_file}\n"
        f"[yellow]Retries:[/]      {retries} / {MAX_RETRIES}\n"
        f"[yellow]Error hash:[/]   {repeat_hash}\n\n"
        f"[dim]{str(editor_error)[:300]}[/]\n\n"
        f"[bold]Press Enter to acknowledge and skip this task "
        f"(auto-continues in {GATE_TIMEOUT_SEC}s)[/]",
        border_style="red",
        title="⚠  ESCALATION",
    ))

    # ── 3. Await acknowledgement with timeout ───────────────────────────────
    if gate_event and isinstance(gate_event, asyncio.Event):
        try:
            await asyncio.wait_for(gate_event.wait(), timeout=float(GATE_TIMEOUT_SEC))
            console.print("[dim]Gate acknowledged — continuing to retrospector.[/]")
            gate_event.clear()   # Reset so subsequent gates in the same sprint work
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
        # No gate event — headless / CI mode, continue without waiting.
        console.print("[dim]Headless mode — no gate_event, routing to retrospector.[/]")

    # ── 4. Clear blocking state so reviewer_logic exits to retrospector ──────
    # Setting current_task=None + editor_error=None + task_queue=[] causes
    # reviewer_logic to return "retrospector_node" on the next evaluation,
    # ending the sprint cleanly without re-entering the editor loop.
    loop_health["escalated"]  = True
    loop_health["last_node"]  = "human_gate_node"

    return {
        "current_task":  None,
        "task_queue":    [],
        "editor_error":  None,
        "loop_health":   loop_health,
    }
