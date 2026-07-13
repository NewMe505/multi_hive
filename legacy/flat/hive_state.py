"""
hive_state.py — Single source of truth for LangGraph shared state.

Extracted from hive_orchestrator.py so every node, test, and the
orchestrator itself import the same TypedDict rather than one file
owning it and everyone else depending on the orchestrator just for
a type.

v4.2 additions vs v4.1 HiveState:
  - semantic_verdict: Optional[str]  PASS/FAIL verdict from semantic_reviewer_node
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, TypedDict


class LoopHealth(TypedDict):
    """
    Tracks whether the retry loop is converging or cycling.

    attempt_count     — total editor invocations for the current task.
    repeat_error_hash — 8-char SHA1 fingerprint of the last error.
                        If identical on two consecutive attempts the
                        loop is symptom-fixing, not converging →
                        escalate immediately.
    escalated         — True once human_gate_node has been notified.
                        Prevents double-escalation on the same failure.
    last_node         — name of the last node that wrote to loop_health,
                        useful for post-sprint forensics.
    """
    attempt_count:      int
    repeat_error_hash:  Optional[str]
    escalated:          bool
    last_node:          Optional[str]


def default_loop_health() -> LoopHealth:
    """Returns a zeroed LoopHealth for initial_state construction."""
    return {
        "attempt_count":     0,
        "repeat_error_hash": None,
        "escalated":         False,
        "last_node":         None,
    }


class HiveState(TypedDict):
    """
    Shared LangGraph graph state for Sentinel Prime v4.1.

    Type contract
    -------------
    current_task and editor_error are Optional[str] — every node
    guards them with `or ""` / `or None`. Declaring them as plain
    `str` (the v3.x mistake) caused TRC-T2 AttributeError crashes
    when nodes called .lower() on a None without checking first.

    Error propagation contract (enforced by convention)
    ---------------------------------------------------
    - async_editor_node / reviewer_node: set editor_error on failure,
      bump editor_retries; clear both on success.
    - agent_router_node: never touches editor_error; always resets
      editor_retries to 0 (it only runs at the start of a new task).
    - ticket_writer: may set editor_error once for a JSON-parse
      failure before any task queue exists.
    - human_gate_node: clears editor_error + current_task so
      reviewer_logic routes to retrospector after escalation.
    - retrospector_node: deliberately leaves editor_error and
      editor_retries untouched so the final values are readable
      in the end-of-sprint panel in run_hive_async.py.

    human_gate_event
    ----------------
    asyncio.Event injected fresh per sprint by run_hive_async.py.
    Not persisted in LangGraph checkpoints (not JSON-serialisable).
    Set by a background stdin-listener task when the operator presses
    Enter; human_gate_node awaits it with a configurable timeout.
    """
    messages:           Any
    sprint_plan:        str
    project_files:      Dict[str, str]
    active_file:        str
    task_queue:         List[dict]
    current_task:       Optional[str]
    editor_error:       Optional[str]
    editor_retries:     int
    specialist_context: str
    is_ui_task:         bool
    loop_health:        LoopHealth
    # Phase 3: verdict from semantic_reviewer_node.
    # "PASS", "FAIL: <reason>", or None (not yet evaluated).
    # Reset to None by agent_router_node at the start of each new task.
    semantic_verdict:   Optional[str]
    # asyncio.Event — injected per sprint, not checkpointed
    human_gate_event:   Optional[Any]
