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


# ── The spec must reach the people judging the work ──────────────────────────


def test_the_semantic_reviewer_is_given_the_objective():
    """
    It used to be handed only the sprint plan and the ticket — a 7B's summary and a
    7B's paraphrase of it. The human's actual words were never given to it, so it was
    asked "does this code do what was asked?" while holding a document that was NOT
    what was asked. Code that correctly implemented the spec but exceeded the lossy
    ticket got FAILed — which becomes editor_error, burns a retry, escalates the tier.

    This test exists because the signature change that fixed it broke its only caller
    and 174 tests still passed. Nothing covered it.
    """
    from multi_hive import prompts

    rendered = prompts.get_semantic_reviewer_prompt(
        "a word longer than width must be hard-split", "a plan", "implement wrapping"
    )
    assert "hard-split" in rendered
    assert "this is the authority" in rendered


def test_the_semantic_reviewer_is_not_asked_to_judge_a_save_path():
    """
    Its user message is ONLY the code — it cannot see active_file and never could.
    The prompt used to hand it a worked example of failing on a save path anyway,
    which is an invitation to hallucinate a violation it has no way to observe.
    """
    from multi_hive import prompts

    rendered = prompts.get_semantic_reviewer_prompt("obj", "plan", "task")
    assert "file saved to src/" not in rendered
    assert "NEVER reject on a filename or save path" in rendered


def test_the_editor_gets_the_requirement_verbatim_and_last():
    """
    The terminal instruction — what a model weights most heavily — used to be
    `EXECUTE THIS SPECIFIC TASK: <ticket>`, and the ticket is a paraphrase of a
    summary. Every trap in this benchmark is exactly the kind of clause a paraphrase
    drops: "ties are broken alphabetically", "touching intervals count as
    overlapping". The model was being graded on requirements it was never shown.
    """
    import asyncio
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    from multi_hive.nodes.execution import async_editor_node as mod

    captured: dict = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured["user"] = messages[-1].content
            return type("R", (), {"content": "```python\nx = 1\n```"})()

    objective = "Ties are broken alphabetically. Save it to outputs/stats.py"

    with patch.object(mod, "get_async_llm", lambda *a, **k: _FakeLLM()):
        asyncio.run(
            mod.async_editor_node(
                {
                    "current_task": "implement top_words",
                    "active_file": "outputs/stats.py",
                    "messages": [HumanMessage(content=objective)],
                    "project_files": {},
                }
            )
        )

    user = captured["user"]
    assert "Ties are broken alphabetically" in user, "the requirement never reached the editor"
    # ...and it is the LAST thing the model reads, not buried above the ticket.
    assert user.rindex("Ties are broken") > user.rindex("EXECUTE THIS SPECIFIC TASK")


def test_the_file_anchor_outranks_the_objective():
    """
    An objective can describe MORE THAN ONE FILE, and moving it last took the file
    selection with it.

    word_stats asks for two modules and names tokens.py first. Handed a ticket for
    stats.py and then told to read a two-file spec as its final word, the editor wrote
    tokens.py INTO stats.py — the file on disk opened with `# outputs/tokens.py` and
    defined tokenize(). Every run. 3/3 -> 0/3.

    The requirement says what correct means. The ticket says which file this call is
    for, and that fact has to survive to the end of the prompt.
    """
    import asyncio
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    from multi_hive.nodes.execution import async_editor_node as mod

    captured: dict = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured["user"] = messages[-1].content
            return type("R", (), {"content": "```python\nx = 1\n```"})()

    # The real two-file objective, in the order that broke it: tokens.py FIRST.
    objective = (
        "Implement a two-module word-frequency tool.\n"
        "outputs/tokens.py defines: tokenize(text) -> list[str]\n"
        "outputs/stats.py defines: top_words(text, n) -> list[tuple[str, int]]\n"
        "stats.py MUST import tokenize from tokens.py."
    )

    with patch.object(mod, "get_async_llm", lambda *a, **k: _FakeLLM()):
        asyncio.run(
            mod.async_editor_node(
                {
                    "current_task": "implement top_words",
                    "active_file": "outputs/stats.py",
                    "messages": [HumanMessage(content=objective)],
                    "project_files": {},
                }
            )
        )

    user = captured["user"]

    # The requirement still reaches the model, and still comes after the ticket.
    assert "MUST import tokenize" in user
    assert user.rindex("MUST import tokenize") > user.rindex("EXECUTE THIS SPECIFIC TASK")

    # But the FILE is the last word — after the objective, not before it.
    assert "YOU ARE WRITING EXACTLY ONE FILE: outputs/stats.py" in user
    assert user.rindex("YOU ARE WRITING EXACTLY ONE FILE") > user.rindex("MUST import tokenize")


def test_the_file_anchor_survives_contract_mode():
    """The anchor is appended after BOTH branches, not just the plain one."""
    import asyncio
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    from multi_hive.nodes.execution import async_editor_node as mod

    captured: dict = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured["user"] = messages[-1].content
            return type("R", (), {"content": "```python\nx = 1\n```"})()

    with patch.object(mod, "get_async_llm", lambda *a, **k: _FakeLLM()):
        asyncio.run(
            mod.async_editor_node(
                {
                    "current_task": "implement top_words",
                    "active_file": "outputs/stats.py",
                    "messages": [HumanMessage(content="outputs/tokens.py defines tokenize")],
                    "project_files": {},
                    "contracts": {"outputs/stats.py": "assert top_words('a', 1) == [('a', 1)]"},
                }
            )
        )

    user = captured["user"]
    assert "YOU ARE WRITING EXACTLY ONE FILE: outputs/stats.py" in user
    # ...and the contract is still the judge in contract mode — the anchor does not
    # reintroduce the authority contradiction that gamed word_wrap.
    assert "Where they disagree, this wins" not in user
    assert user.rindex("YOU ARE WRITING EXACTLY ONE FILE") > user.rindex("for CONTEXT")


def test_the_strong_model_gets_a_blank_page_on_escalation():
    """
    Finding 11. On escalation the strong model used to inherit the fast model's
    BROKEN CODE as `CURRENT FILE CODEBASE`, plus the order "FIX THE CODE SO IT
    PASSES" — asked to patch a bad draft rather than write the program, while
    `bench models` hands the same model a blank page and it scores 8-9/9.

    A tier escalation is a statement that the fast model's attempt was WRONG. Its
    code is the wrongest thing in the context window, and anchoring a better model to
    it is the most expensive mistake in the retry loop.
    """
    import asyncio
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    from multi_hive.nodes.execution import async_editor_node as mod

    captured: dict = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured["user"] = messages[-1].content
            return type("R", (), {"content": "```python\nx = 1\n```"})()

    with patch.object(mod, "get_async_llm", lambda *a, **k: _FakeLLM()):
        asyncio.run(
            mod.async_editor_node(
                {
                    "current_task": "implement it",
                    "active_file": "outputs/x.py",
                    "messages": [HumanMessage(content="the objective")],
                    "project_files": {"outputs/x.py": "THE_WEAK_MODELS_BROKEN_CODE"},
                    "editor_error": "AssertionError: boom",
                    "editor_retries": 1,          # -> escalates to strong
                    "model_tier": "fast",         # ...from fast
                    "loop_health": {"attempt_count": 1, "repeat_error_hash": "deadbeef"},
                }
            )
        )

    user = captured["user"]
    assert "THE_WEAK_MODELS_BROKEN_CODE" not in user, "the strong model inherited the bad draft"
    assert "A WEAKER MODEL ATTEMPTED THIS AND FAILED" in user  # the failure, as a warning
    assert "FIX THE CODE SO IT PASSES" not in user             # ...not as a leash


def test_contract_mode_is_not_given_two_authorities():
    """
    A live contradiction, caught in review, in the ONE mode that scores 9/9.

    _EDITOR_CONTRACT_PREFIX rule 2 tells the model "An ACCEPTANCE CONTRACT is supplied
    below... It is the ONLY thing your code will be judged on." Appending "the
    requirement wins where they disagree" told it something ELSE was the authority —
    and contracts are DELIBERATELY a strict subset of the objective (bench/contracts.py
    rule 1). So the two genuinely differ, and the model was told both were final.

    The requirement still goes last, because the ticket is still a paraphrase. It just
    no longer claims to outrank the contract.
    """
    import asyncio
    from unittest.mock import patch

    from langchain_core.messages import HumanMessage

    from multi_hive.nodes.execution import async_editor_node as mod

    captured: dict = {}

    class _FakeLLM:
        async def ainvoke(self, messages):
            captured["user"] = messages[-1].content
            return type("R", (), {"content": "```python\nx = 1\n```"})()

    def run(contracts):
        with patch.object(mod, "get_async_llm", lambda *a, **k: _FakeLLM()):
            asyncio.run(
                mod.async_editor_node(
                    {
                        "current_task": "implement it",
                        "active_file": "outputs/x.py",
                        "messages": [HumanMessage(content="THE REQUIREMENT")],
                        "project_files": {},
                        "contracts": contracts,
                    }
                )
            )
        return captured["user"]

    # No contract: the requirement IS the authority.
    plain = run({})
    assert "THE REQUIREMENT" in plain
    assert "this wins" in plain

    # With a contract: the requirement is CONTEXT. The contract is the judge.
    with_contract = run({"outputs/x.py": "assert x == 1\n"})
    assert "THE REQUIREMENT" in with_contract, "the spec must still be visible"
    assert "this wins" not in with_contract, "it must not outrank the contract"
    assert "judged on" in with_contract
