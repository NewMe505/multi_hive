import os
import resource
import subprocess
import sys
from typing import Dict, Any

from hive_config import SAFE_ENV
from hive_memory import log_rejection
from hive_utils import flush_file, safe_path


def _limits() -> None:
    """SEC-H3: Hard resource ceilings applied to the sandbox subprocess."""
    try:
        # 2 GB address space — large enough for OpenBLAS/SciPy to initialise.
        resource.setrlimit(resource.RLIMIT_AS,    (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass


def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    active_file  = state.get("active_file")
    loop_health  = state.get("loop_health")   # pass-through only — never modified here

    if not active_file:
        return {}

    current_code = state.get("project_files", {}).get(active_file, "")
    if not current_code:
        return {}

    # ── Write file to disk through the validated path ──────────────────────
    try:
        impl_path = safe_path(active_file)
        flush_file(impl_path, current_code)
    except Exception as e:
        log_rejection("reviewer_node", f"FILE SYSTEM ERROR: {e}")
        return {
            "editor_error":  str(e),
            "editor_retries": state.get("editor_retries", 0) + 1,
            "loop_health":   loop_health,
        }

    # ── Syntax check ───────────────────────────────────────────────────────
    syntax_check = subprocess.run(
        [sys.executable, "-m", "py_compile", impl_path],
        capture_output=True, text=True,
    )
    if syntax_check.returncode != 0:
        error_msg = "SYNTAX ERROR:\n" + syntax_check.stderr
        log_rejection("reviewer_node", error_msg)
        return {
            "editor_error":  error_msg,
            "editor_retries": state.get("editor_retries", 0) + 1,
            "loop_health":   loop_health,
        }

    # ── UI tasks skip sandbox execution (window would block forever) ───────
    if state.get("is_ui_task"):
        task_queue = list(state.get("task_queue", []))
        if task_queue:
            nxt = task_queue.pop(0)
            return {
                "editor_error":  None,
                "editor_retries": 0,
                "task_queue":    task_queue,
                "current_task":  nxt["task"],
                "active_file":   nxt["file"],
                "loop_health":   loop_health,
            }
        return {
            "editor_error":  None,
            "editor_retries": 0,
            "task_queue":    [],
            "current_task":  None,
            "loop_health":   loop_health,
        }

    # ── Sandbox execution ──────────────────────────────────────────────────
    cwd_path    = os.path.abspath(".")
    sandbox_env = SAFE_ENV.copy()
    sandbox_env["PYTHONPATH"]          = f"{cwd_path}:{sandbox_env.get('PYTHONPATH', '')}"
    sandbox_env["OPENBLAS_NUM_THREADS"] = "1"
    sandbox_env["OMP_NUM_THREADS"]      = "1"
    sandbox_env["MKL_NUM_THREADS"]      = "1"

    proc = subprocess.Popen(
        [sys.executable, impl_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sandbox_env,
        cwd=cwd_path,
        preexec_fn=_limits if os.name != "nt" else None,
    )

    try:
        out_bytes, _ = proc.communicate(timeout=10)
        output = out_bytes.decode("utf-8", errors="replace")[:65536]
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        out_bytes, _ = proc.communicate()
        output = out_bytes.decode("utf-8", errors="replace")[:65536] + "\nTIMEOUT: Execution exceeded 10s."
        passed = False

    retries = state.get("editor_retries", 0)

    if passed:
        task_queue = list(state.get("task_queue", []))
        if task_queue:
            nxt = task_queue.pop(0)
            return {
                "editor_error":  None,
                "editor_retries": 0,
                "task_queue":    task_queue,
                "current_task":  nxt["task"],
                "active_file":   nxt["file"],
                "loop_health":   loop_health,
            }
        return {
            "editor_error":  None,
            "editor_retries": 0,
            "task_queue":    [],
            "current_task":  None,
            "loop_health":   loop_health,
        }
    else:
        error_msg = "TRACEBACK:\n" + output[-1500:]
        log_rejection("reviewer_node", error_msg)
        return {
            "editor_error":  error_msg,
            "editor_retries": retries + 1,
            "loop_health":   loop_health,
        }
