"""
The loop must terminate.

This is the regression suite for a live failure: a sprint ran 992 identical
semantic rejections and escalated zero times. The system's two independent
safety mechanisms — the MAX_RETRIES cap and the repeat-error circuit breaker —
both failed to fire, for the same underlying reason.

reviewer_node used to retire the task the moment the code *executed*: it popped
the queue, cleared current_task, and reset editor_retries to 0. But
semantic_reviewer_node runs afterwards. So on a semantic rejection:

  - editor_retries had just been zeroed, so it could never climb to MAX_RETRIES;
  - current_task was already None, so async_editor_node returned {} without
    regenerating anything;
  - the unchanged code passed execution again, zeroing the counter again.

A task is finished when BOTH reviewers pass. Advancement therefore belongs to
the last gate, not the first.
"""
from unittest.mock import AsyncMock, patch

import pytest

from multi_hive.config import MAX_RETRIES
from multi_hive.nodes.execution.reviewer_node import _executes
from multi_hive.nodes.execution.semantic_reviewer_node import semantic_reviewer_node
from multi_hive.orchestrator import reviewer_logic
from multi_hive.state import default_loop_health


def test_reviewer_node_does_not_retire_the_task():
    """Execution success is not task success — it is one of two gates."""
    delta = _executes(default_loop_health())

    assert delta["editor_error"] is None
    # The three keys whose premature reset caused the infinite loop:
    assert "editor_retries" not in delta
    assert "current_task" not in delta
    assert "task_queue" not in delta


@pytest.mark.asyncio
async def test_semantic_pass_retires_the_task_and_pulls_the_next():
    state = {
        "active_file": "outputs/a.py",
        "project_files": {"outputs/a.py": "print(1)"},
        "task_queue": [{"task": "second task", "file": "outputs/b.py"}],
        "editor_retries": 2,
        "loop_health": default_loop_health(),
    }

    with patch("multi_hive.nodes.execution.semantic_reviewer_node.get_async_llm") as llm:
        llm.return_value.ainvoke = AsyncMock(return_value=type("R", (), {"content": "PASS"})())
        delta = await semantic_reviewer_node(state)

    assert delta["semantic_verdict"] == "PASS"
    assert delta["current_task"] == "second task"
    assert delta["active_file"] == "outputs/b.py"
    assert delta["editor_retries"] == 0  # safe now: the task really is over
    assert delta["editor_error"] is None


def test_an_error_with_no_task_cannot_be_retried():
    """
    The second live loop, and the nastier one: an error with no task behind it.

    ticket_writer sets editor_error when the model returns unparseable JSON — at
    which point there is no task queue and no current_task. Routing that back to
    the editor is unkillable: the editor no-ops (`if not current_task: return {}`),
    the reviewers no-op (there is no code), so NOTHING bumps editor_retries and
    the MAX_RETRIES cap is never reached. Nothing is logged either.

    Observed live as 10,007 graph steps and a completely empty rejection ledger.
    """
    state = {
        "loop_health": default_loop_health(),
        "editor_error": "JSON PARSE ERROR: TicketWriter did not output valid JSON.",
        "editor_retries": 0,  # under the cap, and nothing will ever raise it
        "current_task": None,  # <- nothing to retry
    }

    assert reviewer_logic(state) == "human_gate_node"


@pytest.mark.asyncio
async def test_repeated_semantic_rejection_escalates_instead_of_looping():
    """
    The live failure, reproduced: execution keeps passing, semantic review keeps
    rejecting with the same reason. This must reach the human gate, not spin.
    """
    loop_health = default_loop_health()
    state = {
        "active_file": "outputs/a.py",
        "project_files": {"outputs/a.py": "print(1)"},
        "task_queue": [],
        "current_task": "implement it properly",
        "editor_retries": 0,
        "loop_health": loop_health,
    }

    verdict = type("R", (), {"content": "FAIL: uses OrderedDict, not a linked list"})()

    cycles = 0
    for _ in range(20):
        cycles += 1
        # reviewer_node: the code runs fine, every single time.
        state.update(_executes(state["loop_health"]))

        # semantic_reviewer_node: and it is wrong, every single time.
        with patch("multi_hive.nodes.execution.semantic_reviewer_node.get_async_llm") as llm:
            llm.return_value.ainvoke = AsyncMock(return_value=verdict)
            state.update(await semantic_reviewer_node(state))

        route = reviewer_logic(state)
        if route == "human_gate_node":
            assert state["editor_retries"] >= MAX_RETRIES
            return  # escalated, as it must

        assert route == "async_editor_node", f"unexpected route {route!r}"
        # The editor still has a task to work on — it is not a no-op.
        assert state["current_task"], "task was retired while still being rejected"

    pytest.fail(
        f"ran {cycles} cycles without escalating — "
        f"editor_retries stuck at {state['editor_retries']}"
    )
