import json
import os
import time
from typing import Dict, Any

from hive_memory import log_rejection
from hive_utils import flush_file, safe_path
from loop_audit import write_loop_md
from metrics import METRICS_FILE


def retrospector_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final node in the graph — verify-and-backfill + loop health
    persistence + LOOP.md audit trail.

    File flushing (P1-2 from v4.0)
    reviewer_node already flushes each file incrementally as it passes
    syntax + sandbox checks. This node only writes files that are missing
    from disk — covering the case where a task never reached reviewer_node
    because the sprint hit MAX_RETRIES mid-queue.

    Loop health persistence (v4.1)
    Appends a sprint-end loop_health snapshot to metrics.jsonl.

    LOOP.md audit trail (v4.2 Phase 3)
    Writes outputs/LOOP.md with a human-readable sprint summary:
    plan, task log, semantic verdicts, loop health, and recent metrics.
    """
    project_files    = state.get("project_files", {})
    loop_health      = state.get("loop_health") or {}
    sprint_plan      = state.get("sprint_plan", "")
    editor_error     = state.get("editor_error")
    semantic_verdict = state.get("semantic_verdict")

    # ── Verify-and-backfill ─────────────────────────────────────────────────
    for fp, content in project_files.items():
        try:
            abs_path = safe_path(fp)
        except Exception as e:
            log_rejection("retrospector_node", f"INVALID PATH '{fp}': {e}")
            continue

        if not os.path.exists(abs_path):
            try:
                flush_file(fp, content)
            except Exception as e:
                log_rejection("retrospector_node", f"BACKFILL WRITE FAILED '{fp}': {e}")

    # ── Loop health persistence ─────────────────────────────────────────────
    wall_time_sec = 0.0
    if loop_health:
        try:
            os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
            entry = {
                "type":              "loop_health",
                "timestamp":         time.time(),
                "attempt_count":     loop_health.get("attempt_count", 0),
                "escalated":         loop_health.get("escalated", False),
                "repeat_error_hash": loop_health.get("repeat_error_hash"),
                "last_node":         loop_health.get("last_node"),
            }
            with open(METRICS_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
            # Read wall_time_sec from the most recent perf entry for LOOP.md
            with open(METRICS_FILE) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        if "wall_time_sec" in e:
                            wall_time_sec = e["wall_time_sec"]
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            log_rejection("retrospector_node", f"METRICS WRITE FAILED: {e}")

    # ── LOOP.md audit trail (Phase 3) ───────────────────────────────────────
    try:
        # Build task_queue_log from project_files keys as a best-effort
        # record — the full task log would require state accumulation across
        # nodes, which we avoid to keep retrospector stateless. File list
        # gives the operator enough context to understand what was attempted.
        task_queue_log = [
            {
                "file":            fp,
                "task":            "(see sprint plan)",
                "status":          "PASS" if os.path.exists(safe_path(fp)) else "UNKNOWN",
                "semantic_verdict": semantic_verdict or "—",
            }
            for fp in project_files
            if fp  # skip empty keys
        ]

        semantic_verdicts_log = []
        if semantic_verdict:
            active_file = state.get("active_file", "?")
            semantic_verdicts_log.append({
                "file":    active_file,
                "task":    state.get("current_task") or "(unknown)",
                "verdict": semantic_verdict,
            })

        write_loop_md(
            sprint_plan       = sprint_plan,
            task_queue_log    = task_queue_log,
            loop_health       = loop_health,
            wall_time_sec     = wall_time_sec,
            final_error       = editor_error,
            semantic_verdicts = semantic_verdicts_log,
        )
    except Exception as e:
        log_rejection("retrospector_node", f"LOOP.md WRITE FAILED: {e}")

    return {
        "current_task": None,
        "task_queue":   [],
        # PURPOSELY OMITTING editor_error AND editor_retries
        # Forensic values stay readable in the end-of-sprint panel.
    }
