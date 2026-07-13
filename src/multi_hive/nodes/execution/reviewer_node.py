"""
reviewer_node — execution verification.

Writes the generated file to the workspace, compiles it, then runs it in a
sandboxed subprocess. Passing means the code both parses and survives its own
asserts. Whether it is the *right* program is semantic_reviewer_node's problem.

Sandboxing is strongest on POSIX, where core.platform supplies RLIMIT ceilings
via preexec_fn. Windows has no fork and therefore no preexec_fn, so there the
subprocess is bounded by timeout and a minimal environment only — see
core/platform.py. On both platforms the environment is stripped of host
secrets and the process cannot write outside the workspace.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

from multi_hive.config import SANDBOX_TIMEOUT_SEC, WORKSPACE_DIR, sandbox_env
from multi_hive.core.memory import log_rejection
from multi_hive.core.platform import confine, release, sandbox_preexec
from multi_hive.core.utils import flush_file, safe_path

_MAX_OUTPUT_CHARS = 65536
_MAX_TRACEBACK_CHARS = 1500


def _executes(loop_health: Any) -> dict[str, Any]:
    """
    The code ran. That is ALL this node is entitled to say.

    It deliberately does not advance the task queue, clear current_task, or
    reset editor_retries. Declaring the task finished here is what broke the
    loop: semantic_reviewer_node runs *after* this node, so a task marked
    complete on execution alone could still be rejected on intent — and by then
    current_task was None (so the editor regenerated nothing) and editor_retries
    was 0 (so MAX_RETRIES was unreachable). The sprint then cycled forever,
    re-validating identical code. Observed in the wild: 992 identical semantic
    rejections, zero escalations.

    A task is finished when BOTH reviewers pass, and only semantic_reviewer_node
    is in a position to know that. Advancement lives there now.
    """
    return {"editor_error": None, "loop_health": loop_health}


def _fail(state: dict[str, Any], loop_health: Any, error_msg: str) -> dict[str, Any]:
    log_rejection("reviewer_node", error_msg)
    return {
        "editor_error": error_msg,
        "editor_retries": state.get("editor_retries", 0) + 1,
        "loop_health": loop_health,
    }


def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    active_file = state.get("active_file")
    if not active_file:
        return {}

    current_code = state.get("project_files", {}).get(active_file, "")
    if not current_code:
        return {}

    # Pass-through only — reviewer_node never modifies loop_health.
    loop_health = state.get("loop_health")

    # ── Write through the validated path ─────────────────────────────────────
    try:
        impl_path = flush_file(safe_path(active_file), current_code)
    except Exception as e:
        return _fail(state, loop_health, f"FILE SYSTEM ERROR: {e}")

    # ── Syntax check ──────────────────────────────────────────────────────────
    syntax_check = subprocess.run(
        [sys.executable, "-m", "py_compile", str(impl_path)],
        capture_output=True,
        text=True,
    )
    if syntax_check.returncode != 0:
        return _fail(state, loop_health, "SYNTAX ERROR:\n" + syntax_check.stderr)

    # ── UI tasks skip execution: the window would block forever ──────────────
    if state.get("is_ui_task"):
        return _executes(loop_health)

    # ── Sandboxed execution ───────────────────────────────────────────────────
    # Two halves of one sandbox: preexec_fn applies RLIMITs between fork and exec
    # on POSIX; confine() assigns the child to a Job Object on Windows, which has
    # no fork. See core/platform.py for what each actually enforces.
    proc = subprocess.Popen(
        [sys.executable, str(impl_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sandbox_env(),
        cwd=str(WORKSPACE_DIR),
        preexec_fn=sandbox_preexec(),
    )
    job = confine(proc.pid)

    try:
        out_bytes, _ = proc.communicate(timeout=SANDBOX_TIMEOUT_SEC)
        output = out_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        out_bytes, _ = proc.communicate()
        output = (
            out_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
            + f"\nTIMEOUT: Execution exceeded {SANDBOX_TIMEOUT_SEC}s."
        )
        passed = False
    finally:
        # The job is KILL_ON_JOB_CLOSE, so this must not run before communicate()
        # returns — closing it early would kill a perfectly healthy sandbox run.
        release(job)

    if passed:
        return _executes(loop_health)

    return _fail(state, loop_health, "TRACEBACK:\n" + output[-_MAX_TRACEBACK_CHARS:])
