import os
from typing import Dict, Any
from hive_utils import flush_file, safe_path
from hive_memory import log_rejection


def retrospector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Final node in the graph.

    P1-2: reviewer_node already writes (flushes) every file incrementally as
    each task passes its syntax/sandbox check. Previously this node called
    flush_files() on the ENTIRE project_files dict unconditionally, which:
      1. Re-wrote every already-on-disk file on every single sprint (wasted I/O
         that scales with project size, not with what actually changed).
      2. Swallowed any write failure inside flush_files()'s own per-file
         try/except, so this node's outer try/except could never fire — dead
         code that looked like error handling but wasn't.

    New behavior: verify-and-backfill. Check which project_files entries are
    actually present on disk (covers the real gap this node exists to close —
    a file whose task never reached reviewer_node, e.g. because the sprint hit
    MAX_RETRIES mid-task). Only write what's missing. Log any path that fails
    validation so it's visible in the ledger, not just a swallowed warning.
    """
    project_files = state.get("project_files", {})
    missing = {}

    for fp, content in project_files.items():
        try:
            abs_path = safe_path(fp)
        except Exception as e:
            log_rejection("retrospector_node", f"INVALID PATH '{fp}': {e}")
            continue
        if not os.path.exists(abs_path):
            missing[fp] = content

    for fp, content in missing.items():
        try:
            flush_file(fp, content)
        except Exception as e:
            log_rejection("retrospector_node", f"BACKFILL WRITE FAILED '{fp}': {e}")

    return {
        "current_task": None,
        "task_queue": []
        # PURPOSELY OMITTING editor_error AND editor_retries
        # This preserves the forensic history for the Orchestrator's final dashboard.
        # See HiveState docstring in hive_orchestrator.py for the full contract.
    }
