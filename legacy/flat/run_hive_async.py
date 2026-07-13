"""
run_hive_async.py — Async entry point for Sentinel Prime v4.1.

Changes vs run_hive.py (v4.0 sync)
------------------------------------
1. asyncio.run() wraps the entire main loop.
2. human_gate_event (asyncio.Event) is created fresh per sprint and
   injected into initial_state so human_gate_node can await it without
   knowing anything about the REPL or stdin.
3. A background asyncio.Task (_stdin_gate_listener) reads stdin
   non-blockingly via run_in_executor and sets the event when the
   operator presses Enter.
4. hive_app.astream() replaces hive_app.stream() so every await inside
   async nodes (ainvoke, asyncio.wait_for) actually yields the loop.
5. loop_health is included in initial_state and its final value is
   reported in the post-sprint summary panel.
6. ThreadPoolExecutor(max_workers=2) is used explicitly for the stdin
   executor to avoid the default executor queue stalling on a CPU-
   saturated machine running local Ollama inference.
"""
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import llm_factory
from hive_memory import clear_ledger, log_rejection
from hive_orchestrator import hive_app
from hive_state import HiveState, default_loop_health
from hive_utils import flush_file
from langchain_core.messages import HumanMessage
from metrics import SprintMetrics
from rich.console import Console
from rich.panel import Panel

console = Console()

MAX_INPUT_CHARS = 4000

# Bounded executor: 2 threads max — one for stdin, one spare.
# Prevents the default ThreadPoolExecutor from queuing behind Ollama
# inference threads on a 16 GB / CPU-only machine.
_executor = ThreadPoolExecutor(max_workers=2)


# ── Background gate listener ──────────────────────────────────────────────────

async def _stdin_gate_listener(gate_event: asyncio.Event) -> None:
    """
    Reads a single Enter keypress from stdin and sets gate_event.
    Runs concurrently with the graph stream via the bounded executor
    so it never blocks the asyncio event loop.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, sys.stdin.readline)
    gate_event.set()


# ── Sprint runner ─────────────────────────────────────────────────────────────

async def run_sprint(user_input: str) -> None:
    clear_ledger()

    gate_event = asyncio.Event()

    initial_state: HiveState = {
        "messages":           [HumanMessage(content=user_input)],
        "project_files":      {},
        "active_file":        "outputs/main.py",
        "task_queue":         [],
        "current_task":       None,
        "editor_error":       None,
        "editor_retries":     0,
        "sprint_plan":        "",
        "specialist_context": "",
        "is_ui_task":         False,
        "loop_health":        default_loop_health(),
        "semantic_verdict":   None,
        "human_gate_event":   gate_event,
    }

    metrics = SprintMetrics()
    metrics.start()

    rescued_files:      dict          = {}
    final_error:        Optional[str] = None
    final_loop_health:  dict          = {}
    final_semantic:     Optional[str] = None

    # Start the background gate listener — cancelled cleanly after stream ends.
    listener_task = asyncio.create_task(_stdin_gate_listener(gate_event))

    try:
        async for output in hive_app.astream(initial_state):
            for node_name, state_delta in output.items():
                rescued_files.update(state_delta.get("project_files", {}))
                if "editor_error" in state_delta:
                    final_error = state_delta["editor_error"]
                if "loop_health" in state_delta:
                    final_loop_health = state_delta["loop_health"] or {}
                if "semantic_verdict" in state_delta:
                    final_semantic = state_delta["semantic_verdict"]
                metrics.record_node(node_name)
                console.print(f"🔄 [bold cyan]{node_name}[/] executed.")
    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

    metrics.stop(llm_cache_size=len(llm_factory._sync_cache) + len(llm_factory._async_cache))

    # ── Post-sprint panel ─────────────────────────────────────────────────
    escalated = final_loop_health.get("escalated", False)

    if escalated:
        console.print(Panel(
            f"[bold red]🚨 Sprint Escalated in {metrics.wall_time:.1f}s[/]\n"
            f"[dim]Task was not completable within MAX_RETRIES. "
            f"See rejection ledger for details.[/]",
            border_style="red",
        ))
    elif final_error:
        console.print(Panel(
            f"[bold red]⚠️  Sprint Ended With Unresolved Error in {metrics.wall_time:.1f}s[/]\n"
            f"[dim]{str(final_error)[:300]}[/]",
            border_style="red",
        ))
    else:
        console.print(Panel(
            f"[bold green]✅ Sprint Complete in {metrics.wall_time:.1f}s[/]",
            border_style="green",
        ))

    # Metrics footer — wall time, memory, graph steps, loop convergence
    console.print(
        f"[dim]nodes={metrics.node_count}  "
        f"peak_rss_mb={metrics.peak_rss_mb:.1f}  "
        f"llm_cache={metrics.llm_cache_size}  "
        f"attempts={final_loop_health.get('attempt_count', 0)}  "
        f"escalated={escalated}  "
        f"semantic={final_semantic or '—'}[/]"
    )


# ── Main REPL ─────────────────────────────────────────────────────────────────

async def main() -> None:
    os.makedirs("src",     exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    console.print(Panel.fit(
        "[bold yellow]🐝 HIVE ARCHITECTURE v4.2[/]\n"
        "[dim]Sentinel Prime — Async Self-Healing Multi-File Engine[/]",
        border_style="yellow",
    ))

    loop = asyncio.get_event_loop()

    while True:
        try:
            # Read input without blocking the event loop
            raw = await loop.run_in_executor(
                _executor,
                lambda: console.input("\n[bold green][USER_OBJECTIVE] >[/] "),
            )
            user_input = raw.strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Shutting down.[/]")
            break

        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Shutting down.[/]")
            break
        if not user_input:
            continue

        # SEC-L2: Hard cap before the input reaches any LLM context window
        if len(user_input) > MAX_INPUT_CHARS:
            console.print(
                f"[yellow]⚠️  Input truncated from {len(user_input)} "
                f"to {MAX_INPUT_CHARS} chars to stay within planner context window.[/]"
            )
            user_input = user_input[:MAX_INPUT_CHARS]

        try:
            await run_sprint(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim]^C — Shutting down.[/]")
            break
        except Exception as e:
            import traceback as _tb
            console.print(f"[bold red]❌ FATAL: {e}[/]")
            console.print(f"[dim]{_tb.format_exc()}[/]")
            console.print("[dim]Check outputs/ for any partially written files.[/]")
            break

    try:
        _executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
