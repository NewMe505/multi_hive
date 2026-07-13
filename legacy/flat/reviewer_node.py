import subprocess, os, sys, resource
from typing import Dict, Any
from hive_memory import log_rejection
from hive_utils import safe_path, flush_file
from hive_config import SAFE_ENV

def _limits():
    """SEC-H3: Enforce strict sandbox resource limitations."""
    try:
        # Increased to 2GB to allow OpenBLAS/Scipy to initialize
        resource.setrlimit(resource.RLIMIT_AS, (2048*1024*1024, 2048*1024*1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10*1024*1024, 10*1024*1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass

def reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    active_file = state.get("active_file")
    if not active_file: return {}
    
    current_code = state.get("project_files", {}).get(active_file, "")
    if not current_code: return {}
        
    try:
        impl_path = safe_path(active_file)
        flush_file(impl_path, current_code)
    except Exception as e:
        # FIX-LOG1: log under "reviewer_node" not "editor_node" — the editor reads
        # get_recent_rejections("editor_node") on retry; mixing reviewer filesystem
        # errors into that feed corrupts the model's correction strategy.
        log_rejection("reviewer_node", f"FILE SYSTEM ERROR: {e}")
        return {"editor_error": str(e), "editor_retries": state.get("editor_retries", 0) + 1}
        
    syntax_check = subprocess.run([sys.executable, "-m", "py_compile", impl_path], capture_output=True, text=True)
    if syntax_check.returncode != 0:
        error_msg = "SYNTAX ERROR: \n" + syntax_check.stderr
        # FIX-LOG1: same — syntax errors are reviewer findings, not editor generation errors
        log_rejection("reviewer_node", error_msg)
        return {"editor_error": error_msg, "editor_retries": state.get("editor_retries", 0) + 1}

    if state.get("is_ui_task"):
        q = list(state.get("task_queue", []))
        if q:
            return {"editor_error": None, "task_queue": q[1:], "current_task": q[0]["task"], "active_file": q[0]["file"], "editor_retries": 0}
        return {"editor_error": None, "task_queue": [], "current_task": None, "editor_retries": 0}
        
    cwd_path = os.path.abspath('.')
    
    sandbox_env = SAFE_ENV.copy()
    sandbox_env["PYTHONPATH"] = f"{cwd_path}:{sandbox_env.get('PYTHONPATH', '')}"
    sandbox_env["OPENBLAS_NUM_THREADS"] = "1"
    sandbox_env["OMP_NUM_THREADS"] = "1"
    sandbox_env["MKL_NUM_THREADS"] = "1"
    
    proc = subprocess.Popen(
        [sys.executable, impl_path], 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sandbox_env,
        cwd=cwd_path,
        preexec_fn=_limits if os.name != 'nt' else None
    )
    
    try:
        out_bytes, _ = proc.communicate(timeout=10)
        output = out_bytes.decode('utf-8', errors='replace')[:65536]
        passed = proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        out_bytes, _ = proc.communicate() 
        output = out_bytes.decode('utf-8', errors='replace')[:65536] + "\nTIMEOUT: Execution exceeded 10s."
        passed = False

    retries = state.get("editor_retries", 0)

    if passed:
        task_queue = list(state.get("task_queue", []))
        if task_queue:
            next_item = task_queue.pop(0)
            return {"editor_error": None, "editor_retries": 0, "task_queue": task_queue, "current_task": next_item["task"], "active_file": next_item["file"]}
        return {"editor_error": None, "editor_retries": 0, "task_queue": [], "current_task": None}
    else:
        error_msg = "TRACEBACK:\n" + output[-1500:]
        # FIX-LOG1: runtime sandbox failures are reviewer findings
        log_rejection("reviewer_node", error_msg)
        return {"editor_error": error_msg, "editor_retries": retries + 1}
