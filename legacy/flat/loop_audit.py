"""
loop_audit.py — Sprint audit trail writer for Sentinel Prime v4.2.

Produces outputs/LOOP.md after every sprint — a human-readable record
of what the system did, what passed, what failed, and whether it
escalated. Equivalent to the loop-engineering STATE.md + run-log
pattern, adapted for a terminal-based autonomous code-generation system.

Design decisions
----------------
- Written to outputs/LOOP.md (single file, overwritten each sprint).
  Use git or a timestamped copy if history is needed — LOOP.md is the
  "current run" view, metrics.jsonl is the append-only history.
- Intentionally human-readable markdown, not JSON. The audience for
  LOOP.md is the operator reviewing what happened; metrics.jsonl is
  for programmatic analysis.
- Called from retrospector_node so it runs unconditionally at the end
  of every sprint, clean or escalated.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

LOOP_MD_FILE = "outputs/LOOP.md"
METRICS_FILE = "outputs/metrics.jsonl"


def _severity_icon(escalated: bool, had_error: bool) -> str:
    if escalated:
        return "🚨"
    if had_error:
        return "⚠️"
    return "✅"


def write_loop_md(
    sprint_plan:      str,
    task_queue_log:   List[Dict[str, Any]],
    loop_health:      Dict[str, Any],
    wall_time_sec:    float,
    final_error:      Optional[str],
    semantic_verdicts: List[Dict[str, Any]],
) -> None:
    """
    Writes the LOOP.md audit trail for the current sprint.

    Parameters
    ----------
    sprint_plan       — raw sprint plan text from sprint_planner node
    task_queue_log    — list of {file, task, status, verdict} dicts,
                        one per task that was attempted this sprint
    loop_health       — final LoopHealth dict from HiveState
    wall_time_sec     — total sprint wall time in seconds
    final_error       — final editor_error value (None if clean)
    semantic_verdicts — list of {file, task, verdict} from semantic
                        reviewer passes this sprint
    """
    os.makedirs(os.path.dirname(LOOP_MD_FILE), exist_ok=True)

    # Guard: callers should pass {} not None, but be defensive
    loop_health = loop_health or {}

    escalated  = loop_health.get("escalated", False)
    had_error  = final_error is not None
    icon       = _severity_icon(escalated, had_error)
    timestamp  = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    lines: List[str] = []

    # ── Header ───────────────────────────────────────────────────────────────
    lines += [
        f"# {icon} Sentinel Prime — Sprint Audit Log",
        f"",
        f"**Timestamp:** {timestamp}  ",
        f"**Wall time:** {wall_time_sec:.1f}s  ",
        f"**Status:** {'ESCALATED' if escalated else 'FAILED' if had_error else 'CLEAN'}  ",
        f"",
    ]

    # ── Sprint plan ───────────────────────────────────────────────────────────
    lines += [
        "## Sprint Plan",
        "",
        "```",
        sprint_plan.strip() if sprint_plan else "(none)",
        "```",
        "",
    ]

    # ── Task log ──────────────────────────────────────────────────────────────
    lines += ["## Task Execution Log", ""]
    if task_queue_log:
        for i, entry in enumerate(task_queue_log, 1):
            status  = entry.get("status", "unknown")
            verdict = entry.get("semantic_verdict", "—")
            s_icon  = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏭️"
            lines += [
                f"### Task {i}: `{entry.get('file', '?')}`",
                f"",
                f"**Task:** {entry.get('task', '?')}  ",
                f"**Execution:** {s_icon} {status}  ",
                f"**Semantic verdict:** {verdict}  ",
                "",
            ]
    else:
        lines += ["*(no tasks recorded)*", ""]

    # ── Semantic review summary ───────────────────────────────────────────────
    lines += ["## Semantic Review Summary", ""]
    if semantic_verdicts:
        pass_count = sum(1 for v in semantic_verdicts if v.get("verdict", "").startswith("PASS"))
        fail_count = len(semantic_verdicts) - pass_count
        lines += [
            f"- Tasks reviewed: {len(semantic_verdicts)}",
            f"- Passed: {pass_count}",
            f"- Failed semantic check: {fail_count}",
            "",
        ]
        for v in semantic_verdicts:
            v_icon = "✅" if v.get("verdict", "").startswith("PASS") else "❌"
            lines += [f"- {v_icon} `{v.get('file', '?')}`: {v.get('verdict', '?')}"]
        lines += [""]
    else:
        lines += ["*(no semantic reviews recorded)*", ""]

    # ── Loop health ───────────────────────────────────────────────────────────
    lines += [
        "## Loop Health",
        "",
        f"- **Attempt count:** {loop_health.get('attempt_count', 0)}",
        f"- **Escalated:** {escalated}",
        f"- **Repeat error hash:** {loop_health.get('repeat_error_hash') or '—'}",
        f"- **Last node:** {loop_health.get('last_node') or '—'}",
        "",
    ]

    # ── Final error (if any) ──────────────────────────────────────────────────
    if final_error:
        lines += [
            "## Unresolved Error",
            "",
            "```",
            str(final_error)[:800],
            "```",
            "",
        ]

    # ── Metrics tail (last 5 sprints from metrics.jsonl) ─────────────────────
    lines += ["## Recent Sprint Metrics", ""]
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE) as f:
                raw_lines = [l.strip() for l in f if l.strip()]
            perf_entries = []
            for l in raw_lines:
                try:
                    e = json.loads(l)
                    if "wall_time_sec" in e:
                        perf_entries.append(e)
                except json.JSONDecodeError:
                    pass
            recent = perf_entries[-5:]
            if recent:
                lines += ["| # | Wall (s) | RSS (MB) | Nodes |"]
                lines += ["|---|----------|----------|-------|"]
                for i, e in enumerate(recent, len(perf_entries) - len(recent) + 1):
                    lines += [
                        f"| {i} | {e.get('wall_time_sec', '?')} "
                        f"| {e.get('peak_rss_mb', '?')} "
                        f"| {e.get('node_count', '?')} |"
                    ]
                lines += [""]
            else:
                lines += ["*(no perf entries yet)*", ""]
    except Exception:
        lines += ["*(metrics read error)*", ""]

    with open(LOOP_MD_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
