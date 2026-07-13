"""
state.py — single source of truth for the LangGraph shared state.

Kept out of orchestrator.py so nodes, tests, and the orchestrator all import
the same TypedDict, rather than every node depending on the orchestrator just
to get a type.
"""
from __future__ import annotations

from typing import Any, TypedDict


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
    repeat_error_hash: str | None
    escalated: bool
    last_node: str | None


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
    - async_editor_node: sets editor_error on a generation failure. Never
      retires a task.
    - reviewer_node: sets editor_error and bumps editor_retries when the code
      fails to run; clears editor_error when it runs. Does NOT retire the task.
      With a contract in play it runs the human's asserts instead of the model's
      script, and records the outcome in contract_satisfied.
    - semantic_reviewer_node: the last gate, and therefore the ONLY node that
      retires a task. On PASS it clears editor_error, resets editor_retries,
      and pulls the next task off the queue. On FAIL it injects the verdict as
      editor_error and bumps editor_retries, so the rejection routes through
      the existing retry loop. It stands down entirely (auto-PASS, advance) when
      contract_satisfied is True — a human's executable contract outranks a 7B
      model's opinion, and it still retires the task, because something must.
    - agent_router_node: never touches editor_error; always resets
      editor_retries and loop_health (it only runs at the start of a new task).
    - ticket_writer: may set editor_error once for a JSON-parse failure,
      before any task queue exists.
    - human_gate_node: clears editor_error and current_task so reviewer_logic
      routes to retrospector after escalation.
    - retrospector_node: deliberately leaves editor_error and editor_retries
      untouched, so the final values stay readable in the end-of-sprint panel.

    Why only the last gate may retire a task
    ----------------------------------------
    reviewer_node used to retire the task itself, the moment the code executed:
    it popped the queue, cleared current_task, and reset editor_retries. But
    semantic_reviewer_node runs after it. So a semantic rejection landed on a
    task that was already "finished" — current_task was None, so the editor
    regenerated nothing, and editor_retries had just been zeroed, so the retry
    cap was unreachable. The sprint re-validated identical code forever: 992
    identical rejections and zero escalations, in a real run.

    A task is done when BOTH reviewers pass, and only the last one to run knows
    that. Do not move advancement earlier.

    human_gate_event
    ----------------
    An asyncio.Event injected fresh per sprint by cli.py. Not persisted in
    LangGraph checkpoints (not JSON-serialisable). A background stdin listener
    sets it when the operator presses Enter; human_gate_node awaits it with a
    timeout.
    """

    messages: Any
    sprint_plan: str
    project_files: dict[str, str]
    active_file: str
    task_queue: list[dict]
    current_task: str | None
    editor_error: str | None
    editor_retries: int
    specialist_context: str
    is_ui_task: bool
    loop_health: LoopHealth
    # "PASS", "FAIL: <reason>", or None (not yet evaluated).
    # Reset to None by agent_router_node at the start of each new task.
    semantic_verdict: str | None
    # "trivial" | "moderate" | "hard" — a cheap text-only prior on difficulty,
    # set by agent_router_node. Seeds the initial model tier.
    task_complexity: str | None
    # "fast" | "strong" — which model tier this task is running on.
    #
    # Sticky for the duration of a task, and it only ever ratchets upward:
    # agent_router_node seeds it, async_editor_node may escalate it on failure,
    # and semantic_reviewer_node follows whatever the editor used. The two
    # models do not fit in VRAM together, so a tier that could flip back and
    # forth mid-task would thrash the GPU. See core/model_router.py.
    model_tier: str | None
    # Human-written acceptance contracts, keyed by normalised file path (plus a
    # "*" wildcard for the single-file case). Parsed once from the objective by
    # the entrypoint; read-only for every node. See contract.py.
    #
    # A file with a contract is judged by the human's asserts instead of the
    # model's own — which is the whole point, because the model writing both the
    # code and the asserts that judge it is a conflict of interest that has
    # already cost real sprints (it once rejected correct code for failing an
    # assert that was arithmetically impossible to satisfy).
    contracts: dict[str, str]
    # True once reviewer_node has run the contract for the current file and it
    # held; False if it was violated; None when there is no contract, or before
    # the first review. Reset per task by agent_router_node.
    #
    # semantic_reviewer_node reads this and stands down when it is True.
    contract_satisfied: bool | None
    # asyncio.Event — injected per sprint, not checkpointed.
    human_gate_event: Any | None
