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
    # (purpose, tier). The node ROUTES its tier now — it used to call get_llm with
    # no tier at all, which silently defaulted to `fast`, so HIVE_FORCE_TIER never
    # reached it and the 7B always decided what the task was.
    monkeypatch.setattr(tw_module, "get_llm", lambda _p, _t=None: _FakeLLM(payload))
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


# ── A parse failure must not kill the whole sprint ───────────────────────────


def test_unparseable_json_retries_on_the_strong_model(monkeypatch):
    """
    The failure the clean baseline caught red-handed.

    A parse failure here does not fail a *task* — it kills the whole SPRINT. No
    queue is built, reviewer_logic routes straight to the human gate, no code is
    written, and the run is scored "no code": indistinguishable from "the model
    cannot code", for a pure infrastructure failure.

    It killed `lru_cache --contract` on THREE RUNS OUT OF THREE — 0/3 on a task the
    same pipeline passes comfortably whenever the JSON happens to parse. The 7B's
    JSON is stochastic, and emitting the task queue is the one job in this system
    where a single bad sample costs everything.
    """
    calls: list[str] = []

    def fake_get_llm(_purpose, tier=None):
        calls.append(tier)
        if len(calls) == 1:
            return _FakeLLM("Sure! Here is the task list: (not JSON)")
        return _FakeLLM(json.dumps([{"file": "outputs/lru.py", "task": "implement it"}]))

    monkeypatch.setattr(tw_module, "get_llm", fake_get_llm)
    out = ticket_writer({"sprint_plan": "a plan", "messages": []})

    assert calls == ["fast", "strong"], calls  # retried, and on the better model
    assert out["active_file"] == "outputs/lru.py"  # the sprint lives
    assert "editor_error" not in out


def test_both_tiers_failing_is_a_real_failure(monkeypatch):
    """One retry, not an infinite one. If strong also fails, the sprint fails."""
    calls: list[str] = []

    def fake_get_llm(_purpose, tier=None):
        calls.append(tier)
        return _FakeLLM("still not JSON")

    monkeypatch.setattr(tw_module, "get_llm", fake_get_llm)
    out = ticket_writer({"sprint_plan": "a plan", "messages": []})

    assert calls == ["fast", "strong"]  # tried twice, gave up
    assert "JSON PARSE ERROR" in out["editor_error"]


# ── One ticket per file ───────────────────────────────────────────────────────


def test_four_tickets_for_one_file_become_one(monkeypatch):
    """
    Measured live on lru_cache, against the real 7B:

        ['outputs/lru.py', 'outputs/lru.py', 'outputs/lru.py', 'outputs/lru.py']

    The planner's "MAXIMUM 4 STEPS" pressured a decomposition out of a single-file
    objective, and the ticket writer turned it into four tickets — four FULL
    rewrites of the same artefact, each one followed by another pass through
    reviewer_node and semantic_reviewer_node.

    The cost is not just 4x the inference. The semantic reviewer is a known source
    of spurious FAILs, and running it four times on one file gives it four chances
    to reject code that was already correct. Worse, on ticket 4 it is handed a
    complete LRU cache and asked whether it implements "Handle eviction" — judging
    a whole program against a quarter of a paraphrase.
    """
    out = _run(
        [
            {"file": "outputs/lru.py", "task": "Define LRUCache class with capacity"},
            {"file": "outputs/lru.py", "task": "Implement get(key)"},
            {"file": "outputs/lru.py", "task": "Implement put(key, value)"},
            {"file": "outputs/lru.py", "task": "Handle eviction"},
        ],
        monkeypatch,
    )

    assert out["task_queue"] == []  # one ticket total, and it is the current one
    assert out["active_file"] == "outputs/lru.py"

    # Every requirement survives the merge. Losing one would be worse than the bug.
    for requirement in ("capacity", "get(key)", "put(key, value)", "eviction"):
        assert requirement in out["current_task"]


def test_collapsing_happens_after_normalisation(monkeypatch):
    """
    Two tickets that both MEANT outputs/lru.py — one of them spelled `lru.py` —
    are the same file, and must collapse. They only look the same after the path
    has been normalised, so the order of the two passes is load-bearing.
    """
    out = _run(
        [
            {"file": "lru.py", "task": "Define the class"},
            {"file": "outputs/lru.py", "task": "Implement eviction"},
        ],
        monkeypatch,
    )

    assert out["task_queue"] == []
    assert out["active_file"] == "outputs/lru.py"
    assert "Define the class" in out["current_task"]
    assert "Implement eviction" in out["current_task"]


def test_genuinely_different_files_are_not_merged(monkeypatch):
    """The fix must not collapse a real multi-file plan into one ticket."""
    out = _run(
        [
            {"file": "outputs/main.py", "task": "entrypoint"},
            {"file": "src/util.py", "task": "helper"},
        ],
        monkeypatch,
    )

    assert out["active_file"] == "outputs/main.py"
    assert len(out["task_queue"]) == 1
    assert out["task_queue"][0]["file"] == "src/util.py"


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
