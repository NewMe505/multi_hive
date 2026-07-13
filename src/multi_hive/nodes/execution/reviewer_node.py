"""
reviewer_node — execution verification.

Writes the generated file to the workspace, compiles it, then runs it in a
sandboxed subprocess.

What "passing" means depends on who wrote the test.

  Without a contract, the node runs the model's own script, and passing means
  the code survives the asserts the model wrote about itself. That is the weakest
  useful check there is, and it is weak in a specific, expensive way: the model
  is judge and defendant. It has rejected its own correct code for failing an
  assert that was impossible to satisfy — see contract.py.

  With a human-supplied acceptance contract, the node IMPORTS the module and
  executes the human's asserts against it. Passing then means the code does what
  the person who asked for it said it must do. The model's own test code, if it
  wrote any despite being told not to, never runs: an imported module's __name__
  is not "__main__".

Sandboxing is identical either way, and is strongest on POSIX, where
core.platform supplies RLIMIT ceilings via preexec_fn. Windows has no fork and
therefore no preexec_fn, so there the subprocess is bounded by a Job Object and
a timeout — see core/platform.py. On both platforms the environment is stripped
of host secrets and the child's working directory is inside the workspace.

What the sandbox does NOT do is confine *where* the code may write. The ceilings
are on memory, process count, and (POSIX only) per-file size — not on filesystem
paths. There is no chroot, mount namespace, or AppContainer, so untrusted code
that opens an absolute path outside the workspace is not blocked. This is a
resource sandbox, not a filesystem jail; treat the models it runs as
semi-trusted, and do not run it against code you would not run yourself.
"""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from multi_hive.config import SANDBOX_TIMEOUT_SEC, WORKSPACE_DIR, sandbox_env
from multi_hive.contract import contract_for, pass_marker, render_harness
from multi_hive.core.memory import log_rejection
from multi_hive.core.platform import IS_WINDOWS, confine, release, sandbox_preexec
from multi_hive.core.utils import flush_file, safe_path

_MAX_OUTPUT_CHARS = 65536
_MAX_TRACEBACK_CHARS = 1500

_TIMEOUT_RC = 124  # conventional; distinct from any exit code the harness uses

# After we SIGKILL a timed-out sandbox, we still try to drain its output — but
# with a bound, because a grandchild holding the stdout pipe can make that drain
# block forever. We already have the timeout verdict; the drain is a courtesy.
_DRAIN_TIMEOUT_SEC = 5

# The contract harness lives in the workspace, not the source tree: it is
# generated, it is rewritten on every review, and it must be somewhere the
# sandbox is allowed to read.
_HARNESS_FILE = "outputs/.acceptance_harness.py"

# Exit codes from contract.render_harness — each one tells the editor a different
# thing, so each gets its own message rather than a generic "it failed".
_CONTRACT_FAILURES: dict[int, str] = {
    2: (
        "ACCEPTANCE CONTRACT — YOUR MODULE FAILED TO IMPORT.\n"
        "The contract never ran. Fix the code so the module imports cleanly:"
    ),
    3: (
        "ACCEPTANCE CONTRACT VIOLATED — your code ran but produced the wrong "
        "result. This assert was written by a human and is correct by definition; "
        "the code is what is wrong. Fix the logic:"
    ),
    4: (
        "ACCEPTANCE CONTRACT — MISSING NAME. The contract calls something your "
        "module does not define. Check the exact spelling of every public "
        "function and class:"
    ),
    5: (
        "ACCEPTANCE CONTRACT — UNEXPECTED EXCEPTION while checking your code "
        "(not an AssertionError). Fix the crash:"
    ),
    _TIMEOUT_RC: "ACCEPTANCE CONTRACT — TIMED OUT. The code did not terminate:",
}


def _executes(loop_health: Any, contract_satisfied: bool | None = None) -> dict[str, Any]:
    """
    The code passed whatever check applied to it. That is ALL this node may say.

    It deliberately does not advance the task queue, clear current_task, or
    reset editor_retries. Declaring the task finished here is what broke the
    loop: semantic_reviewer_node runs *after* this node, so a task marked
    complete on execution alone could still be rejected on intent — and by then
    current_task was None (so the editor regenerated nothing) and editor_retries
    was 0 (so MAX_RETRIES was unreachable). The sprint then cycled forever,
    re-validating identical code. Observed in the wild: 992 identical semantic
    rejections, zero escalations.

    A task is finished when every gate has passed, and only the last gate is in a
    position to know that. Advancement lives in semantic_reviewer_node.
    """
    return {
        "editor_error": None,
        "loop_health": loop_health,
        "contract_satisfied": contract_satisfied,
    }


def _fail(
    state: dict[str, Any],
    loop_health: Any,
    error_msg: str,
    contract_satisfied: bool | None = None,
) -> dict[str, Any]:
    log_rejection("reviewer_node", error_msg)
    return {
        "editor_error": error_msg,
        "editor_retries": state.get("editor_retries", 0) + 1,
        "loop_health": loop_health,
        "contract_satisfied": contract_satisfied,
    }


def _bounded(text: str) -> str:
    """
    Cap output length while keeping BOTH ends.

    The pass sentinel and the real assertion both sit at the END of the output,
    so the old head-only truncation could drop them: a passing run whose module
    printed >64 KB would lose its sentinel and read as a failure, and a genuine
    traceback could vanish entirely. Keeping the tail preserves both.
    """
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    half = _MAX_OUTPUT_CHARS // 2
    return text[:half] + "\n...[output truncated]...\n" + text[-half:]


def _kill_tree(proc: subprocess.Popen) -> None:
    """
    SIGKILL the sandboxed process and any descendants it spawned.

    proc.kill() reaps only the direct child. A grandchild that inherited the
    stdout pipe keeps the pipe's write end open, so a plain communicate() after
    it blocks forever waiting for an EOF that never arrives — the advertised hard
    timeout, silently defeated. On POSIX the child is its own session leader
    (start_new_session in _run_sandboxed), so the whole group is signalled. On
    Windows the Job Object reaps the tree when release() closes its handle;
    proc.kill() here just ends the direct child promptly.
    """
    if IS_WINDOWS:
        proc.kill()
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


def _run_sandboxed(script: Path) -> tuple[int, str]:
    """
    Run `script` in the sandbox. Returns (returncode, combined output).

    Two halves of one sandbox: preexec_fn applies RLIMITs between fork and exec
    on POSIX; confine() assigns the child to a Job Object on Windows, which has
    no fork. See core/platform.py for what each actually enforces.
    """
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sandbox_env(),
        cwd=str(WORKSPACE_DIR),
        preexec_fn=sandbox_preexec(),
        # POSIX: give the child its own session so a timeout can SIGKILL the whole
        # process group, not just the direct child. Ignored on Windows, where the
        # Job Object plays that role.
        start_new_session=not IS_WINDOWS,
    )
    job = confine(proc.pid)

    try:
        out_bytes, _ = proc.communicate(timeout=SANDBOX_TIMEOUT_SEC)
        return proc.returncode, _bounded(out_bytes.decode("utf-8", errors="replace"))
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out_bytes, _ = proc.communicate(timeout=_DRAIN_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            # A descendant is still holding the stdout pipe open. We already have
            # the timeout verdict, so do not block the sprint to drain it; on
            # Windows the Job Object reaps the tree when release() runs below.
            out_bytes = b""
        output = (
            _bounded(out_bytes.decode("utf-8", errors="replace"))
            + f"\nTIMEOUT: Execution exceeded {SANDBOX_TIMEOUT_SEC}s."
        )
        return _TIMEOUT_RC, output
    finally:
        # The job is KILL_ON_JOB_CLOSE, so this must not run before communicate()
        # returns — closing it early would kill a perfectly healthy sandbox run.
        release(job)


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

    contract = contract_for(state.get("contracts") or {}, active_file)

    # ── UI tasks skip execution: the window would block forever ──────────────
    #
    # A contract still runs, because importing a module does not open a window —
    # that is what the Controller/View split in the editor prompt is for. Only
    # the *script* is unsafe to execute here.
    if state.get("is_ui_task") and not contract:
        return _executes(loop_health)

    # ── Contract mode: the human's asserts, against an imported module ────────
    if contract:
        # A per-run nonce printed on the pass line. The generated module never
        # sees the harness, let alone this token, so it cannot forge the pass —
        # not by printing "CONTRACT_PASS" itself, and not by calling os._exit(0)
        # at import (which exits 0 but never reaches the sentinel). The exit code
        # alone is never trusted; see contract.py.
        pass_token = secrets.token_hex(8)
        try:
            harness_path = flush_file(
                _HARNESS_FILE, render_harness(str(impl_path), contract, pass_token)
            )
        except Exception as e:
            return _fail(state, loop_health, f"FILE SYSTEM ERROR (contract harness): {e}")

        code, output = _run_sandboxed(harness_path)

        if code == 0 and pass_marker(pass_token) in output:
            return _executes(loop_health, contract_satisfied=True)

        if code == 0:
            # Exit 0 without the sentinel: the module ended the process before the
            # contract confirmed a pass — a top-level exit()/quit()/os._exit() at
            # import time. That is a failure, not a satisfied contract.
            headline = (
                "ACCEPTANCE CONTRACT — your module ended the process before the "
                "contract could run (a top-level exit(), quit(), or os._exit() at "
                "import time?). Remove it: the contract is the only thing allowed "
                "to decide a pass:"
            )
        else:
            headline = _CONTRACT_FAILURES.get(
                code, "ACCEPTANCE CONTRACT — the check exited unexpectedly:"
            )
        return _fail(
            state,
            loop_health,
            f"{headline}\n{output[-_MAX_TRACEBACK_CHARS:]}",
            contract_satisfied=False,
        )

    # ── No contract: run the model's own script and trust its own asserts ─────
    code, output = _run_sandboxed(impl_path)

    if code == 0:
        return _executes(loop_health)

    return _fail(state, loop_health, "TRACEBACK:\n" + output[-_MAX_TRACEBACK_CHARS:])
