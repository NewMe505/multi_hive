"""
state.py — single source of truth for the LangGraph shared state.

Kept out of orchestrator.py so nodes, tests, and the orchestrator all import
the same TypedDict, rather than every node depending on the orchestrator just
to get a type.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LoopHealth(TypedDict):
    """
    Tracks whether the retry loop is converging or cycling.

    attempt_count     — total editor invocations for the current task.
    repeat_error_hash — 8-char SHA1 fingerprint of the last error. Identical
                        on two consecutive attempts means the loop is
                        symptom-fixing rather than converging → escalate now
                        instead of burning the rest of the retry budget.
    escalated         — True once human_gate_node has been notified. Prevents
                        double-escalation on the same failure.
    last_node         — name of the last node to write loop_health; useful for
                        post-sprint forensics.
    """

    attempt_count: int
    repeat_error_hash: Optional[str]
    escalated: bool
    last_node: Optional[str]


def default_loop_health() -> LoopHealth:
    """A zeroed LoopHealth, for initial_state and per-task resets."""
    return {
        "attempt_count": 0,
        "repeat_error_hash": None,
        "escalated": False,
        "last_node": None,
    }


class HiveState(TypedDict):
    """
    Shared graph state.

    Type contract
    -------------
    current_task and editor_error are Optional[str] — every node guards them
    with `or ""` / `or None`. Declaring them as plain `str` caused
    AttributeError crashes when a node called .lower() on a None.

    Error propagation contract (enforced by convention)
    ---------------------------------------------------
    - async_editor_node / reviewer_node: set editor_error on failure and bump
      editor_retries; clear both on success.
    - agent_router_node: never touches editor_error; always resets
      editor_retries to 0 (it only runs at the start of a new task).
    - ticket_writer: may set editor_error once for a JSON-parse failure,
      before any task queue exists.
    - semantic_reviewer_node: injects its FAIL verdict as editor_error so the
      rejection routes through the existing retry loop.
    - human_gate_node: clears editor_error and current_task so reviewer_logic
      routes to retrospector after escalation.
    - retrospector_node: deliberately leaves editor_error and editor_retries
      untouched, so the final values stay readable in the end-of-sprint panel.

    human_gate_event
    ----------------
    An asyncio.Event injected fresh per sprint by cli.py. Not persisted in
    LangGraph checkpoints (not JSON-serialisable). A background stdin listener
    sets it when the operator presses Enter; human_gate_node awaits it with a
    timeout.
    """

    messages: Any
    sprint_plan: str
    project_files: Dict[str, str]
    active_file: str
    task_queue: List[dict]
    current_task: Optional[str]
    editor_error: Optional[str]
    editor_retries: int
    specialist_context: str
    is_ui_task: bool
    loop_health: LoopHealth
    # "PASS", "FAIL: <reason>", or None (not yet evaluated).
    # Reset to None by agent_router_node at the start of each new task.
    semantic_verdict: Optional[str]
    # asyncio.Event — injected per sprint, not checkpointed.
    human_gate_event: Optional[Any]
