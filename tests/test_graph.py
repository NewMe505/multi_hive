"""
The graph must actually build.

This is the test that would have caught the state the project was found in:
orchestrator.py imported `nodes.execution.*`, which did not exist, so every
entrypoint died on ImportError before a single node ran.
"""
import pytest

from multi_hive.state import default_loop_health

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")


def test_graph_compiles():
    from multi_hive.orchestrator import build_graph

    assert build_graph() is not None


def test_reviewer_logic_routing():
    from multi_hive.orchestrator import reviewer_logic

    base = {"loop_health": default_loop_health(), "editor_retries": 0}

    # Escalation beats everything, including a still-pending task.
    escalated = {**base, "loop_health": {**default_loop_health(), "escalated": True}}
    assert reviewer_logic({**escalated, "current_task": "t"}) == "human_gate_node"

    # An error under the retry cap goes back to the editor — but only when there
    # is actually a task to retry. Without one the editor no-ops forever; see
    # tests/test_loop_terminates.py.
    retryable = {**base, "editor_error": "boom", "current_task": "t"}
    assert reviewer_logic(retryable) == "async_editor_node"

    # ...and at the cap it escalates instead of looping forever.
    at_cap = {**retryable, "editor_retries": 3}
    assert reviewer_logic(at_cap) == "human_gate_node"

    # An error with no task behind it cannot be retried — escalate.
    assert reviewer_logic({**base, "editor_error": "boom"}) == "human_gate_node"

    # Clean with work left → next task. Clean with none → wrap up.
    assert reviewer_logic({**base, "current_task": "t"}) == "agent_router_node"
    assert reviewer_logic({**base, "current_task": None}) == "retrospector_node"
