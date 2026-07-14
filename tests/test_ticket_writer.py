"""
ticket_writer is where a model-authored path first enters the graph.

Every path in the queue eventually becomes `active_file` — the first one at once,
the rest as semantic_reviewer_node retires tasks and pulls the next. An illegal
one is not caught until write time, by which point a full generation has been paid
for, and the resulting FILE SYSTEM ERROR is then routed back to the editor as
though it were a code failure. It is not: the path lives in state, the editor
cannot change it, and every retry regenerates the same file for the same illegal
path until the budget is gone.

So the queue is legalised here, before any of that.
"""
from __future__ import annotations

import json

import pytest

from multi_hive.core import memory
from multi_hive.nodes.execution import ticket_writer as tw_module
from multi_hive.nodes.execution.ticket_writer import ticket_writer


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def invoke(self, _messages):
        return type("R", (), {"content": self.payload})()


@pytest.fixture(autouse=True)
def _clean_ledger():
    memory.clear_ledger()
    yield
    memory.clear_ledger()


def _run(tasks, monkeypatch):
    payload = json.dumps(tasks) if isinstance(tasks, list) else tasks
    monkeypatch.setattr(tw_module, "get_llm", lambda _p: _FakeLLM(payload))
    return ticket_writer({"sprint_plan": "a plan", "messages": []})


# ── The bug ───────────────────────────────────────────────────────────────────


def test_a_bare_filename_is_fixed_instead_of_burning_the_retry_budget(monkeypatch):
    """
    Observed live, on a two-line add(a, b): the planner emitted a task writing to
    `test_add.py`, safe_path refused it at write time, and the editor was asked to
    "FIX THE CODE SO IT PASSES" — a thing it could not do, because the code was
    never wrong. Four generations and an escalation, for nothing.
    """
    out = _run([{"file": "test_add.py", "task": "write a test"}], monkeypatch)

    assert out["active_file"] == "outputs/test_add.py"
    assert "PATH NORMALISED" in memory.get_recent_rejections("ticket_writer")


def test_every_ticket_is_legalised_not_just_the_first(monkeypatch):
    """
    task_queue entries become active_file later, as tasks retire. Validating only
    tasks[0] would move the bug deeper into the sprint, where it costs more.
    """
    out = _run(
        [
            {"file": "outputs/add.py", "task": "implement"},
            {"file": "test_add.py", "task": "test it"},
        ],
        monkeypatch,
    )

    assert out["active_file"] == "outputs/add.py"
    assert out["task_queue"][0]["file"] == "outputs/test_add.py"


def test_an_unfixable_path_is_dropped_not_guessed_at(monkeypatch):
    out = _run(
        [
            {"file": "outputs/add.py", "task": "implement"},
            {"file": "../../.ssh/authorized_keys", "task": "own the box"},
        ],
        monkeypatch,
    )

    assert out["active_file"] == "outputs/add.py"
    assert out["task_queue"] == []  # the traversal never entered the graph
    assert "DROPPED TICKET" in memory.get_recent_rejections("ticket_writer")


def test_a_plan_with_no_writable_file_fails_loudly_instead_of_spinning(monkeypatch):
    """
    Rather than handing the graph a queue it will spend the whole retry budget
    failing to write — which is exactly what used to happen.
    """
    out = _run([{"file": "/etc/passwd", "task": "nope"}], monkeypatch)

    assert "PATH ERROR" in out["editor_error"]
    assert "task_queue" not in out


def test_legal_plans_are_untouched(monkeypatch):
    out = _run(
        [
            {"file": "outputs/wrap.py", "task": "implement wrap"},
            {"file": "src/util.py", "task": "helper"},
        ],
        monkeypatch,
    )

    assert out["active_file"] == "outputs/wrap.py"
    assert out["task_queue"][0]["file"] == "src/util.py"
    assert memory.get_recent_rejections("ticket_writer") == ""


# ── The category error ────────────────────────────────────────────────────────


def test_path_complaints_never_reach_the_editors_failure_feeds(monkeypatch):
    """
    The editor's prompt splits failures three ways — generation / runtime /
    semantic — because each implies a different fix, and merging them produces
    retries where the model does not know what kind of wrong it was.

    A path complaint implies NONE of them. Handing the editor a routing log and
    telling it to "fix the code structure" is the same category error the
    repeat-error breaker was already taught once. These log under "ticket_writer",
    which no editor feed reads.
    """
    _run([{"file": "test_add.py", "task": "write a test"}], monkeypatch)

    for feed in ("async_editor_node", "reviewer_node", "semantic_reviewer_node"):
        assert memory.get_recent_rejections(feed) == ""
