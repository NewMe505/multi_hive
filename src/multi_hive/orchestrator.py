"""
orchestrator.py — the LangGraph graph definition.

Graph shape
-----------
sprint_planner → ticket_writer → agent_router_node → async_editor_node
  → reviewer_node → semantic_reviewer_node → [conditional]
                                           → async_editor_node   (retry)
                                           → human_gate_node     (escalate)
                                           → agent_router_node   (next task)
                                           → retrospector_node   (done)
  human_gate_node → [conditional] → agent_router_node   (skip it; more work left)
                                  → retrospector_node   (that was the last task)

The two reviewers verify different things and both must pass before the graph
advances. reviewer_node verifies *execution*: the code runs and its own
asserts hold. semantic_reviewer_node verifies *intent*: the code implements
what the task actually asked for. A model can easily produce a syntactically
valid, cleanly passing script that is semantically the wrong program.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from multi_hive.config import MAX_RETRIES
from multi_hive.nodes.execution.agent_router_node import agent_router_node
from multi_hive.nodes.execution.async_editor_node import async_editor_node
from multi_hive.nodes.execution.human_gate_node import human_gate_node
from multi_hive.nodes.execution.retrospector_node import retrospector_node
from multi_hive.nodes.execution.reviewer_node import reviewer_node
from multi_hive.nodes.execution.semantic_reviewer_node import semantic_reviewer_node
from multi_hive.nodes.execution.sprint_planner import sprint_planner
from multi_hive.nodes.execution.ticket_writer import ticket_writer
from multi_hive.state import HiveState


def reviewer_logic(state: HiveState) -> str:
    """
    Conditional edge router — the circuit breaker. Reads state after both
    reviewer_node (execution) and semantic_reviewer_node (intent) have run.

    Escalation triggers, in priority order:
    1. loop_health.escalated — repeat-error fingerprint detected by
       async_editor_node: the loop is fixing symptoms, not making progress.
    2. editor_retries >= MAX_RETRIES — hard attempt cap. Both execution
       failures and semantic FAIL rejections count toward this cap, since
       semantic_reviewer_node injects its verdict as editor_error and bumps
       editor_retries.
    3. editor_error with retries under the cap — normal retry path.
    4. current_task, no error — advance to the next task.
    5. no current_task, no error — sprint complete, wrap up.
    """
    loop_health = state.get("loop_health") or {}

    if loop_health.get("escalated"):
        return "human_gate_node"

    if state.get("editor_error"):
        if state.get("editor_retries", 0) >= MAX_RETRIES:
            return "human_gate_node"

        # You cannot retry a task that does not exist.
        #
        # Routing an error back to the editor with no current_task was an
        # unkillable loop: the editor's first line is `if not current_task:
        # return {}`, so it no-ops. The reviewers then no-op too (there is no
        # code to check), which means NOTHING bumps editor_retries — so the
        # MAX_RETRIES cap above is never reached, and nothing is even logged.
        #
        # It is reachable in practice: ticket_writer sets editor_error when the
        # model returns unparseable JSON, and at that point no task queue exists
        # yet. Observed live as 10,007 graph steps and an empty rejection ledger.
        if not state.get("current_task"):
            return "human_gate_node"

        return "async_editor_node"

    if state.get("current_task"):
        return "agent_router_node"

    return "retrospector_node"


def gate_logic(state: HiveState) -> str:
    """
    Where the sprint goes after a human gate.

    This used to be an unconditional edge to the retrospector, because
    human_gate_node returned `task_queue: []` — an escalation on ticket 1 of 3
    silently cancelled tickets 2 and 3, and the sprint ended having never attempted
    them.

    The gate now skips the failed task and keeps the queue, so there may be work
    left. If there is, the sprint continues at agent_router_node — which resets
    loop_health, clearing `escalated` so reviewer_logic does not route straight back
    here and spin.

    The escalation is NOT forgotten by that reset: human_gate_node sets
    `sprint_escalated`, which nothing ever clears, and that is what the CLI, the
    retrospector and the bench report. The sprint finishes the work it can and still
    tells the operator a human is needed.
    """
    return "agent_router_node" if state.get("current_task") else "retrospector_node"


def build_graph() -> StateGraph:
    """
    Wires and compiles the graph.

    A function rather than module-level side effects: tests (and any future
    second entrypoint) can build a fresh graph without importing one that was
    already compiled at import time.
    """
    workflow = StateGraph(HiveState)

    workflow.add_node("sprint_planner", sprint_planner)
    workflow.add_node("ticket_writer", ticket_writer)
    workflow.add_node("agent_router_node", agent_router_node)
    workflow.add_node("async_editor_node", async_editor_node)
    workflow.add_node("reviewer_node", reviewer_node)
    workflow.add_node("semantic_reviewer_node", semantic_reviewer_node)
    workflow.add_node("human_gate_node", human_gate_node)
    workflow.add_node("retrospector_node", retrospector_node)

    workflow.set_entry_point("sprint_planner")
    workflow.add_edge("sprint_planner", "ticket_writer")
    workflow.add_edge("ticket_writer", "agent_router_node")
    workflow.add_edge("agent_router_node", "async_editor_node")
    workflow.add_edge("async_editor_node", "reviewer_node")
    workflow.add_edge("reviewer_node", "semantic_reviewer_node")
    workflow.add_conditional_edges("semantic_reviewer_node", reviewer_logic)
    workflow.add_conditional_edges("human_gate_node", gate_logic)
    workflow.add_edge("retrospector_node", END)

    return workflow.compile()


hive_app = build_graph()
