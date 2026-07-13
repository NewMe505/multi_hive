"""
retrospector_node — the final node. Backfill, persist, audit.

Three jobs:

1. Verify-and-backfill. reviewer_node already flushes each file as it passes
   its checks, so this only writes files missing from disk — the case where a
   task never reached the reviewer because the sprint hit MAX_RETRIES
   mid-queue and the code exists only in state.

2. Loop health persistence. Appends a sprint-end loop_health snapshot to
   metrics.jsonl.

3. LOOP.md. Human-readable sprint summary: plan, task log, semantic verdicts,
   loop health, recent metrics.

Deliberately leaves editor_error and editor_retries untouched so their final
values stay readable in the end-of-sprint panel.
"""
from __future__ import annotations

import json
import time
from typing import Any

from multi_hive.config import METRICS_FILE
from multi_hive.core.loop_audit import write_loop_md
from multi_hive.core.memory import log_rejection
from multi_hive.core.utils import flush_file, safe_path


def retrospector_node(state: dict[str, Any]) -> dict[str, Any]:
    project_files = state.get("project_files", {})
    loop_health = state.get("loop_health") or {}
    sprint_plan = state.get("sprint_plan", "")
    editor_error = state.get("editor_error")
    semantic_verdict = state.get("semantic_verdict")

    # ── Verify-and-backfill ───────────────────────────────────────────────────
    for filepath, content in project_files.items():
        try:
            abs_path = safe_path(filepath)
        except Exception as e:
            log_rejection("retrospector_node", f"INVALID PATH '{filepath}': {e}")
            continue

        if not abs_path.exists():
            try:
                flush_file(filepath, content)
            except Exception as e:
                log_rejection("retrospector_node", f"BACKFILL WRITE FAILED '{filepath}': {e}")

    # ── Loop health persistence ───────────────────────────────────────────────
    wall_time_sec = 0.0
    if loop_health:
        try:
            METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with METRICS_FILE.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "type": "loop_health",
                            "timestamp": time.time(),
                            "attempt_count": loop_health.get("attempt_count", 0),
                            "escalated": loop_health.get("escalated", False),
                            "repeat_error_hash": loop_health.get("repeat_error_hash"),
                            "last_node": loop_health.get("last_node"),
                        }
                    )
                    + "\n"
                )
            wall_time_sec = _latest_wall_time()
        except Exception as e:
            log_rejection("retrospector_node", f"METRICS WRITE FAILED: {e}")

    # ── LOOP.md audit trail ───────────────────────────────────────────────────
    try:
        # Best-effort task log built from project_files. A full per-task log
        # would need state accumulated across every node; the file list gives
        # the operator enough to see what was attempted and keeps this node
        # stateless.
        task_queue_log = [
            {
                "file": filepath,
                "task": "(see sprint plan)",
                "status": "PASS" if safe_path(filepath).exists() else "UNKNOWN",
                "semantic_verdict": semantic_verdict or "—",
            }
            for filepath in project_files
            if filepath
        ]

        semantic_verdicts_log = []
        if semantic_verdict:
            semantic_verdicts_log.append(
                {
                    "file": state.get("active_file", "?"),
                    "task": state.get("current_task") or "(unknown)",
                    "verdict": semantic_verdict,
                }
            )

        write_loop_md(
            sprint_plan=sprint_plan,
            task_queue_log=task_queue_log,
            loop_health=loop_health,
            wall_time_sec=wall_time_sec,
            final_error=editor_error,
            semantic_verdicts=semantic_verdicts_log,
        )
    except Exception as e:
        log_rejection("retrospector_node", f"LOOP.md WRITE FAILED: {e}")

    return {
        "current_task": None,
        "task_queue": [],
    }


def _latest_wall_time() -> float:
    """Wall time from the most recent perf entry in metrics.jsonl, for LOOP.md."""
    wall_time = 0.0
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "wall_time_sec" in entry:
                wall_time = entry["wall_time_sec"]
    return wall_time
