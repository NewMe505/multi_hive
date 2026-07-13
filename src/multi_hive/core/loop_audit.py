"""
loop_audit.py — sprint audit trail writer.

Produces workspace/outputs/LOOP.md after every sprint: a human-readable record
of what the system did, what passed, what failed, and whether it escalated.

Design
------
- LOOP.md is the "current run" view and is overwritten each sprint;
  metrics.jsonl is the append-only history. Use git or a timestamped copy if
  you need LOOP.md history.
- Markdown on purpose. The audience is the operator reading what happened;
  metrics.jsonl is the machine-readable side.
- Called from retrospector_node, so it runs at the end of every sprint —
  clean or escalated.
"""
from __future__ import annotations

import json
import time
from typing import Any

from multi_hive.config import LOOP_MD_FILE, METRICS_FILE


def _severity_icon(escalated: bool, had_error: bool) -> str:
    if escalated:
        return "🚨"
    if had_error:
        return "⚠️"
    return "✅"


def write_loop_md(
    sprint_plan: str,
    task_queue_log: list[dict[str, Any]],
    loop_health: dict[str, Any],
    wall_time_sec: float,
    final_error: str | None,
    semantic_verdicts: list[dict[str, Any]],
) -> None:
    """
    Writes the LOOP.md audit trail for the current sprint.

    sprint_plan       — raw plan text from sprint_planner
    task_queue_log    — [{file, task, status, semantic_verdict}] per attempted task
    loop_health       — final LoopHealth dict from HiveState
    wall_time_sec     — total sprint wall time
    final_error       — final editor_error (None if clean)
    semantic_verdicts — [{file, task, verdict}] from semantic review passes
    """
    LOOP_MD_FILE.parent.mkdir(parents=True, exist_ok=True)

    loop_health = loop_health or {}
    escalated = loop_health.get("escalated", False)
    had_error = final_error is not None
    icon = _severity_icon(escalated, had_error)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    status = "ESCALATED" if escalated else "FAILED" if had_error else "CLEAN"

    lines: list[str] = [
        f"# {icon} Sentinel Prime — Sprint Audit Log",
        "",
        f"**Timestamp:** {timestamp}  ",
        f"**Wall time:** {wall_time_sec:.1f}s  ",
        f"**Status:** {status}  ",
        "",
        "## Sprint Plan",
        "",
        "```",
        sprint_plan.strip() if sprint_plan else "(none)",
        "```",
        "",
        "## Task Execution Log",
        "",
    ]

    if task_queue_log:
        for i, entry in enumerate(task_queue_log, 1):
            task_status = entry.get("status", "unknown")
            verdict = entry.get("semantic_verdict", "—")
            s_icon = "✅" if task_status == "PASS" else "❌" if task_status == "FAIL" else "⏭️"
            lines += [
                f"### Task {i}: `{entry.get('file', '?')}`",
                "",
                f"**Task:** {entry.get('task', '?')}  ",
                f"**Execution:** {s_icon} {task_status}  ",
                f"**Semantic verdict:** {verdict}  ",
                "",
            ]
    else:
        lines += ["*(no tasks recorded)*", ""]

    lines += ["## Semantic Review Summary", ""]
    if semantic_verdicts:
        passed = sum(1 for v in semantic_verdicts if v.get("verdict", "").startswith("PASS"))
        lines += [
            f"- Tasks reviewed: {len(semantic_verdicts)}",
            f"- Passed: {passed}",
            f"- Failed semantic check: {len(semantic_verdicts) - passed}",
            "",
        ]
        for v in semantic_verdicts:
            v_icon = "✅" if v.get("verdict", "").startswith("PASS") else "❌"
            lines.append(f"- {v_icon} `{v.get('file', '?')}`: {v.get('verdict', '?')}")
        lines.append("")
    else:
        lines += ["*(no semantic reviews recorded)*", ""]

    lines += [
        "## Loop Health",
        "",
        f"- **Attempt count:** {loop_health.get('attempt_count', 0)}",
        f"- **Escalated:** {escalated}",
        f"- **Repeat error hash:** {loop_health.get('repeat_error_hash') or '—'}",
        f"- **Last node:** {loop_health.get('last_node') or '—'}",
        "",
    ]

    if final_error:
        lines += [
            "## Unresolved Error",
            "",
            "```",
            str(final_error)[:800],
            "```",
            "",
        ]

    lines += ["## Recent Sprint Metrics", ""]
    lines += _recent_metrics_table()

    LOOP_MD_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recent_metrics_table(limit: int = 5) -> list[str]:
    """Last `limit` perf entries from metrics.jsonl, as a markdown table."""
    try:
        if not METRICS_FILE.exists():
            return ["*(no perf entries yet)*", ""]

        perf_entries: list[dict[str, Any]] = []
        with METRICS_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "wall_time_sec" in entry:
                    perf_entries.append(entry)

        recent = perf_entries[-limit:]
        if not recent:
            return ["*(no perf entries yet)*", ""]

        rows = [
            "| # | Wall (s) | RSS (MB) | Nodes |",
            "|---|----------|----------|-------|",
        ]
        first_index = len(perf_entries) - len(recent) + 1
        for i, entry in enumerate(recent, first_index):
            rows.append(
                f"| {i} | {entry.get('wall_time_sec', '?')} "
                f"| {entry.get('peak_rss_mb', '?')} "
                f"| {entry.get('node_count', '?')} |"
            )
        rows.append("")
        return rows
    except Exception:
        return ["*(metrics read error)*", ""]
