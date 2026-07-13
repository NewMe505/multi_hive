import os, sys, time, resource
from rich.console import Console
from rich.panel import Panel
from hive_orchestrator import hive_app
from langchain_core.messages import HumanMessage
from hive_memory import clear_ledger, log_rejection
from hive_utils import flush_file, safe_path
import llm_factory
from metrics import SprintMetrics

console = Console()

# SEC-L2: Cap raw user input before it ever reaches a HumanMessage / LLM prompt.
# A multi-KB paste silently overflows the ticket LLM's 2048-token num_ctx and
# produces garbled planning output with no visible error.
MAX_INPUT_CHARS = 4000


def main():
    os.makedirs("src", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # P2-2: Banner was stuck on "v3.2 (Patched)" — stale version signal on a
    # system whose whole premise is iterative, trackable correctness.
    console.print(Panel.fit(
        "[bold yellow]🐝 HIVE ARCHITECTURE v4.0[/]\n[dim]Sentinel Prime — Deterministic Multi-File Engine[/]",
        border_style="yellow"
    ))

    while True:
        try:
            user_input = console.input("\n[bold green][USER_OBJECTIVE] >[/] ").strip()
            if user_input.lower() in ['exit', 'quit']:
                console.print("[dim]Shutting down...[/]")
                break
            if not user_input:
                continue

            # SEC-L2
            if len(user_input) > MAX_INPUT_CHARS:
                console.print(
                    f"[yellow]⚠️ Input truncated from {len(user_input)} to "
                    f"{MAX_INPUT_CHARS} chars to stay within planner context window.[/]"
                )
                user_input = user_input[:MAX_INPUT_CHARS]

            clear_ledger()

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "project_files": {},
                "active_file": "outputs/main.py",
                "task_queue": [],
                "current_task": None,
                "editor_error": None,
                "editor_retries": 0,
                "sprint_plan": "",
                "specialist_context": "",
                "is_ui_task": False
            }

            # P3-1: Lightweight, non-invasive metrics — no behavior change to the
            # graph itself, just measurement wrapped around the existing stream loop.
            metrics = SprintMetrics()
            metrics.start()

            rescued_files = {}  # Cumulative tracker for emergency crash dumps
            final_editor_error = None

            for output in hive_app.stream(initial_state):
                for node_name, state_delta in output.items():
                    # NEW-OPT2 idiom: .get() instead of an explicit key-check that
                    # always misses on 5 of 6 nodes.
                    rescued_files.update(state_delta.get("project_files", {}))
                    if "editor_error" in state_delta:
                        final_editor_error = state_delta["editor_error"]
                    metrics.record_node(node_name)
                    console.print(f"🔄 [bold cyan]{node_name}[/] executed.")

            metrics.stop(llm_cache_size=len(llm_factory._sync_cache) + len(llm_factory._async_cache))

            # P2-4 (SEC-L1): Sprint completion panel now reflects whether the
            # sprint actually ended clean, instead of always printing ✅.
            if final_editor_error:
                console.print(Panel(
                    f"[bold red]⚠️ Sprint Ended With Unresolved Error in {metrics.wall_time:.1f}s[/]\n"
                    f"[dim]{str(final_editor_error)[:300]}[/]",
                    border_style="red"
                ))
            else:
                console.print(Panel(
                    f"[bold green]✅ Sprint Complete in {metrics.wall_time:.1f}s[/]",
                    border_style="green"
                ))

            console.print(
                f"[dim]nodes={metrics.node_count}  "
                f"peak_rss_mb={metrics.peak_rss_mb:.1f}  "
                f"llm_cache_size={metrics.llm_cache_size}[/]"
            )

        except KeyboardInterrupt:
            console.print("\n[dim]^C — Shutting down.[/]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[bold red]❌ FATAL EXCEPTION: {e}[/]")

            if 'rescued_files' in locals() and rescued_files:
                # P2-3: Crash-rescue path previously used raw os.path.abspath(fp)
                # + open(...) directly, bypassing safe_path() entirely — the
                # exact SEC-C1 path-traversal vector the rest of the codebase
                # already guards against, just reachable from the one place
                # that was never updated. Now routes through the same
                # safe_path()-gated flush_file() everything else uses.
                rescued_count = 0
                for fp, content in rescued_files.items():
                    try:
                        flush_file(fp, content)
                        rescued_count += 1
                    except Exception as write_err:
                        log_rejection("run_hive_rescue", f"RESCUE WRITE BLOCKED '{fp}': {write_err}")
                console.print(
                    f"[yellow]⚠️ Rescued {rescued_count}/{len(rescued_files)} "
                    f"file(s) to disk before crash exit (validated paths only).[/]"
                )
            break


if __name__ == "__main__":
    main()
