"""
spec_writer_node — writes the acceptance criteria, before any code exists.

The editor used to write its own asserts, and a model that grades its own
homework fails in both directions:

  - a WRONG assert rejects correct code. Observed live: the model asserted
    `wrap_text("hello world", 10) == ["hello world"]`. "hello world" is eleven
    characters and the width is ten, so it cannot possibly fit — the assertion
    was simply false. Its own test rejected its own correct implementation, burnt
    the whole retry budget, escalated the model tier, and woke a human.

  - a LAZY assert waves a bug straight through, and nobody notices, because the
    thing that was supposed to catch it was written by the thing that produced it.

Separating authorship fixes both. The spec is derived from the task, before there
is any implementation to rationalise it against, and it does not move between
retries — so the editor is aiming at a fixed target it did not choose.

Runs on the task's tier, before the editor. One extra call per task.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import DEFAULT_TIER, get_llm, invalidate_llm
from multi_hive.core.memory import log_rejection

MAX_LINES = 12
MIN_ASSERTS = 2

# A statement that reaches for the filesystem, the network, or the interpreter is
# not testing behaviour — and it runs inside our sandbox. Model output is
# untrusted input, and dressing a call up as an `assert` does not change that.
_FORBIDDEN = re.compile(
    r"\b(import|open|exec|eval|__import__|compile|input|exit|quit|globals|locals)\s*\(?",
)

# An assert must compare a value to an expectation.
#
# This rejects `assert c.put(1, 1)`, which the spec writer produced on its first
# outing: put() mutates and returns None, so that line fails against a perfectly
# correct implementation. Asserting the truthiness of a mutation is never a
# meaningful test, and it is a very easy mistake for a model to make.
_COMPARISON = re.compile(r"(==|!=|<=|>=|<|>|\bis\b|\bin\b|\bnot\b)")


def _is_setup(statement: str) -> bool:
    """A bare assignment: `c = LRUCache(2)`, or a call like `c.put(1, 1)`."""
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        return False
    return len(tree.body) == 1 and isinstance(tree.body[0], (ast.Assign, ast.Expr))


# Names an assertion may use without touching the implementation.
_BUILTINS = {
    "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "sorted", "sum", "min", "max", "abs", "round", "range", "all", "any",
    "isinstance", "type", "repr", "enumerate", "zip", "reversed", "None",
    "True", "False",
}


def _drop_asserts_that_never_touch_the_code(statements: list[str]) -> list[str]:
    """
    Remove assertions that cannot fail because of the implementation.

    The spec writer produced this for the word-wrap task:

        t = "This is a test string that needs to be wrapped."
        assert len(t.split()) == 7

    That is an assertion about the *test data*, not the code. It calls nothing the
    implementation defines, so no implementation can ever make it pass or fail —
    and it is also just wrong (the string has ten words). An assertion like that
    has exactly one possible effect: rejecting a correct implementation.

    Detecting it is a small taint analysis. A variable is "tainted" when its value
    came from a name the implementation must provide (`c = LRUCache(2)` taints
    `c`, because `LRUCache` is not a builtin and was never assigned here). An
    assertion earns its place only if it uses an implementation name or a tainted
    variable. `assert c.get(1) == 1` qualifies — `c` is tainted. `assert
    len(t.split()) == 7` does not: `t` is a string literal and `len` is a builtin,
    so the whole line is decidable without ever importing the code.
    """
    bound: set[str] = set()
    tainted: set[str] = set()
    kept: list[str] = []

    for statement in statements:
        node = ast.parse(statement).body[0]

        used = {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        from_implementation = (used - bound - _BUILTINS) or (used & tainted)

        if isinstance(node, ast.Assert):
            if not from_implementation:
                continue  # decidable without the code; it can only ever misfire
            kept.append(statement)
            continue

        if isinstance(node, ast.Assign):
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            bound |= targets
            if from_implementation:
                tainted |= targets

        kept.append(statement)

    return kept


def parse_spec(raw: str) -> list[str]:
    """
    Turn the model's response into an executable acceptance script.

    Two kinds of line survive: setup (assignments and calls) and asserts. They run
    in order, in one namespace, so state persists — which is the whole point.

    The first version of this only accepted isolated `assert` lines, and that was
    a design error, not just a parsing one: a stateful contract cannot be
    expressed that way. The model dutifully produced
    `assert LRUCache(2).get(1) == 2` — a brand-new empty cache asked for a key
    nobody ever put in it — because the format left it no other option.

    A spec with no assertions in it is not a spec, so it is refused entirely.
    """
    statements: list[str] = []

    for line in (raw or "").splitlines():
        statement = line.strip().rstrip(";")

        if not statement or statement.startswith("#"):
            continue
        if statement.startswith("```") or _FORBIDDEN.search(statement):
            continue
        if statement in statements:
            continue

        is_assert = statement.startswith("assert ")

        if is_assert and not _COMPARISON.search(statement):
            continue
        if not is_assert and not _is_setup(statement):
            continue  # prose, or something we do not understand — drop it

        # It has to parse on its own, or it explodes inside the harness and gets
        # misread as the implementation crashing.
        try:
            compile(statement, "<acceptance>", "exec")
        except SyntaxError:
            continue

        statements.append(statement)
        if len(statements) >= MAX_LINES:
            break

    statements = _drop_asserts_that_never_touch_the_code(statements)

    # A thin spec is worse than no spec. If fewer than MIN_ASSERTS real checks
    # survive, refuse the whole thing: reviewer_node falls back to "does the module
    # import", which is weak but honest, and the semantic reviewer still guards
    # intent. A single surviving assertion cannot even be adjudicated away if it
    # turns out to be wrong — the last check is never dropped — so a lone bad
    # assertion would guarantee a false failure on correct code.
    if sum(1 for s in statements if s.startswith("assert ")) < MIN_ASSERTS:
        return []

    return statements


def spec_writer_node(state: dict[str, Any]) -> dict[str, Any]:
    current_task = state.get("current_task")
    if not current_task:
        return {}

    # The spec is written once per task. agent_router_node clears it when a new
    # task starts; a retry must aim at the same target it missed, or the goalposts
    # move every attempt and a failure means nothing.
    if state.get("acceptance"):
        return {}

    tier = state.get("model_tier") or DEFAULT_TIER

    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    objective = human_msgs[0].content if human_msgs else ""

    sys_prompt = prompts.get_spec_writer_prompt(objective[:2000], current_task)

    try:
        llm = get_llm("reviewer", tier)  # temperature 0 — a spec should be stable
        response = llm.invoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=current_task)]
        )
    except Exception as e:
        invalidate_llm("reviewer", tier)
        log_rejection("spec_writer_node", f"SPEC GENERATION FAILED: {e}")
        # No spec is not a blocker. reviewer_node falls back to simply importing
        # the module and checking it does not explode — weaker, but the sprint
        # continues, and the failure is in the ledger.
        return {"acceptance": [], "spec_repairs": 0}

    spec = parse_spec(response.content)

    if not spec:
        log_rejection(
            "spec_writer_node",
            f"NO USABLE ACCEPTANCE SPEC parsed from the response:\n{response.content[:400]}",
        )
        return {"acceptance": [], "spec_repairs": 0}

    return {"acceptance": spec, "spec_repairs": 0}
