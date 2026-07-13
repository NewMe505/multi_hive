"""
cli.py — the REPL entrypoint.

Run with `multi-hive`, or `python -m multi_hive`.

stdin ownership
---------------
Two things want to read the keyboard: the REPL (waiting for the next
objective) and human_gate_node (waiting for an escalation acknowledgement).
They never want it at the same time — during a sprint the REPL is idle, and
between sprints the gate cannot fire — so a single reader thread pumps lines
into a queue and whoever is waiting consumes them.

This replaces a per-sprint listener task that called sys.stdin.readline() in
an executor. Cancelling that task did not unblock the thread it left sitting
in readline(), so the threads accumulated: after two sprints the bounded
2-worker pool was exhausted and the REPL could no longer read input at all.
A blocking read in a thread cannot be cancelled — so it must never be started
more than once.
"""
from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from langchain_core.messages import HumanMessage
from rich.panel import Panel

from multi_hive import __version__
from multi_hive.config import MAX_INPUT_CHARS, WORKSPACE_DIR, ensure_workspace
from multi_hive.core import llm_factory
from multi_hive.core.console import console
from multi_hive.core.memory import clear_ledger
from multi_hive.core.metrics import SprintMetrics
from multi_hive.orchestrator import hive_app
from multi_hive.state import HiveState, default_loop_health

# One thread for the stdin pump, one spare. Unbounded pools queue behind the
# CPU-saturating Ollama inference threads on a machine running the model locally.
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hive-io")

_EOF = object()


class StdinBroker:
    """Single stdin reader. Both the REPL and the human gate consume from it."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pump_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(_executor, sys.stdin.readline)
            if line == "":  # EOF — Ctrl-D, or a closed pipe.
                await self._queue.put(_EOF)
                return
            await self._queue.put(line.rstrip("\n"))

    async def readline(self) -> Optional[str]:
        """Next line, or None at EOF."""
        item = await self._queue.get()
        return None if item is _EOF else item

    async def close(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass


async def _acknowledge_on_input(broker: StdinBroker, gate_event: asyncio.Event) -> None:
    """
    Sets gate_event when the operator presses Enter during a sprint.

    Safe to cancel: it waits on an asyncio.Queue, not on a blocking read.
    """
    while True:
        line = await broker.readline()
        if line is None:
            return
        gate_event.set()


async def run_sprint(user_input: str, broker: StdinBroker) -> None:
    clear_ledger()

    gate_event = asyncio.Event()

    initial_state: HiveState = {
        "messages": [HumanMessage(content=user_input)],
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
        "human_gate_event": gate_event,
    }

    metrics = SprintMetrics()
    metrics.start()

    final_error: Optional[str] = None
    final_loop_health: dict = {}
    final_semantic: Optional[str] = None

    ack_task = asyncio.create_task(_acknowledge_on_input(broker, gate_event))

    try:
        async for output in hive_app.astream(initial_state):
            for node_name, state_delta in output.items():
                if "editor_error" in state_delta:
                    final_error = state_delta["editor_error"]
                if "loop_health" in state_delta:
                    final_loop_health = state_delta["loop_health"] or {}
                if "semantic_verdict" in state_delta:
                    final_semantic = state_delta["semantic_verdict"]
                metrics.record_node(node_name)
                console.print(f"🔄 [bold cyan]{node_name}[/] executed.")
    finally:
        ack_task.cancel()
        try:
            await ack_task
        except asyncio.CancelledError:
            pass

    metrics.stop(llm_cache_size=llm_factory.cache_size())

    # ── Post-sprint panel ─────────────────────────────────────────────────────
    escalated = final_loop_health.get("escalated", False)

    if escalated:
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

    console.print(
        f"[dim]nodes={metrics.node_count}  "
        f"peak_rss_mb={metrics.peak_rss_mb:.1f}  "
        f"llm_cache={metrics.llm_cache_size}  "
        f"attempts={final_loop_health.get('attempt_count', 0)}  "
        f"escalated={escalated}  "
        f"semantic={final_semantic or '—'}[/]"
    )


async def main() -> None:
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

            # Cap raw input before it reaches any LLM context window. A multi-KB
            # paste silently overflows the ticket model's 2048-token num_ctx and
            # produces garbled planning with no visible error.
            if len(user_input) > MAX_INPUT_CHARS:
                console.print(
                    f"[yellow]⚠️  Input truncated from {len(user_input)} to "
                    f"{MAX_INPUT_CHARS} chars to stay within the planner context window.[/]"
                )
                user_input = user_input[:MAX_INPUT_CHARS]

            try:
                await run_sprint(user_input, broker)
            except Exception as e:
                import traceback

                console.print(f"[bold red]❌ FATAL: {e}[/]")
                console.print(f"[dim]{traceback.format_exc()}[/]")
                console.print(f"[dim]Check {WORKSPACE_DIR} for partially written files.[/]")
                break
    finally:
        await broker.close()
        _executor.shutdown(wait=False, cancel_futures=True)


def run() -> None:
    """Console-script entrypoint."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[dim]^C — Shutting down.[/]")


if __name__ == "__main__":
    run()
