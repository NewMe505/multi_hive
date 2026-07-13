"""
hive_orchestrator.py — LangGraph graph definition for Sentinel Prime v4.2.

Graph shape
-----------
sprint_planner → ticket_writer → agent_router_node → async_editor_node
  → reviewer_node → semantic_reviewer_node → [conditional]
                                           → async_editor_node   (retry)
                                           → human_gate_node     (escalate)
                                           → agent_router_node   (next task)
                                           → retrospector_node   (done)
  human_gate_node → retrospector_node → END

Phase 3 change: semantic_reviewer_node sits between reviewer_node and
the conditional routing edge. reviewer_node verifies execution (runs,
passes its own asserts). semantic_reviewer_node verifies intent (does
the code implement what the task actually asked for?). Both must pass
before the graph advances to the next task.

reviewer_logic routing table (reads state AFTER semantic review)
----------------------------------------------------------------
loop_health.escalated == True          → human_gate_node
editor_error set (semantic FAIL)       → async_editor_node / human_gate_node
  + retries >= MAX_RETRIES             → human_gate_node
  + retries < MAX_RETRIES              → async_editor_node
current_task (no error)               → agent_router_node
no current_task, no error             → retrospector_node
"""
from langgraph.graph import StateGraph, END

from hive_config import MAX_RETRIES
from hive_state import HiveState

from nodes.execution.sprint_planner        import sprint_planner
from nodes.execution.ticket_writer         import ticket_writer
from nodes.execution.agent_router_node     import agent_router_node
from nodes.execution.async_editor_node     import async_editor_node
from nodes.execution.reviewer_node         import reviewer_node
from nodes.execution.semantic_reviewer_node import semantic_reviewer_node
from nodes.execution.human_gate_node       import human_gate_node
from nodes.execution.retrospector_node     import retrospector_node


def reviewer_logic(state: HiveState) -> str:
    """
    Conditional edge router — circuit breaker reading state after both
    reviewer_node (execution) and semantic_reviewer_node (intent) have run.

    Escalation triggers (in priority order):
    1. loop_health.escalated — repeat-error fingerprint detected by
       async_editor_node (symptom-fixing loop, no progress).
    2. editor_retries >= MAX_RETRIES — hard attempt cap reached.
       Both from execution failures AND semantic FAIL rejections
       count toward this cap, since semantic_reviewer_node injects
       its verdict as editor_error + bumps editor_retries.
    3. editor_error + retries < cap — normal retry path.
    4. current_task, no error — advance to next task.
    5. no current_task, no error — sprint complete, wrap up.
    """
    loop_health = state.get("loop_health") or {}

    if loop_health.get("escalated"):
        return "human_gate_node"

    if state.get("editor_error"):
        if state.get("editor_retries", 0) >= MAX_RETRIES:
            return "human_gate_node"
        return "async_editor_node"

    if state.get("current_task"):
        return "agent_router_node"

    return "retrospector_node"


# ── Graph construction ────────────────────────────────────────────────────────

workflow = StateGraph(HiveState)

workflow.add_node("sprint_planner",         sprint_planner)
workflow.add_node("ticket_writer",          ticket_writer)
workflow.add_node("agent_router_node",      agent_router_node)
workflow.add_node("async_editor_node",      async_editor_node)
workflow.add_node("reviewer_node",          reviewer_node)
workflow.add_node("semantic_reviewer_node", semantic_reviewer_node)
workflow.add_node("human_gate_node",        human_gate_node)
workflow.add_node("retrospector_node",      retrospector_node)

workflow.set_entry_point("sprint_planner")
workflow.add_edge("sprint_planner",         "ticket_writer")
workflow.add_edge("ticket_writer",          "agent_router_node")
workflow.add_edge("agent_router_node",      "async_editor_node")
workflow.add_edge("async_editor_node",      "reviewer_node")
workflow.add_edge("reviewer_node",          "semantic_reviewer_node")
workflow.add_conditional_edges("semantic_reviewer_node", reviewer_logic)
workflow.add_edge("human_gate_node",        "retrospector_node")
workflow.add_edge("retrospector_node",      END)

hive_app = workflow.compile()
