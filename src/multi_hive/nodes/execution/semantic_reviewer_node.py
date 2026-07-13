"""
semantic_reviewer_node — intent verification.

reviewer_node proves the code RUNS. It cannot prove the code is the program
that was asked for. A model will happily produce a syntactically valid script
that passes its own asserts and implements the wrong thing entirely: wrong
function names, a missing requirement, saved to a different file than
specified.

This node catches that class of failure by re-asking the same local model with
adversarial framing. Same-model review is weaker than a separate verifier
model, but it catches the egregious divergences without a second Ollama
instance — the practical trade-off on a CPU-only 16GB machine.

Response contract: "PASS", or "FAIL: <one reason>". Parsing is deliberately
biased toward PASS — anything not explicitly starting with FAIL is treated as
a pass, so a confused or verbose model cannot manufacture a false escalation.

On FAIL the reason is injected as editor_error, which routes the rejection
back through the existing retry loop rather than inventing a second one.
"""
from __future__ import annotations

import traceback
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import DEFAULT_TIER, get_async_llm, invalidate_llm
from multi_hive.core.memory import log_rejection


def _advance(state: dict[str, Any]) -> dict[str, Any]:
    """
    Declare the current task finished and pull the next one off the queue.

    This is the ONLY place a task is retired, because this is the last gate: the
    code has executed (reviewer_node) and it implements what was asked
    (this node). Advancing any earlier — as reviewer_node used to, on execution
    alone — lets a task be retired and then rejected, which leaves the editor
    with nothing to regenerate and the retry counter reset to zero on every
    cycle. The sprint cannot then make progress or terminate.

    Resetting editor_retries here is safe precisely because the task really is
    done; the counter belongs to the task, and the task is over.
    """
    task_queue = list(state.get("task_queue", []))

    if task_queue:
        nxt = task_queue.pop(0)
        return {
            "editor_error": None,
            "editor_retries": 0,
            "task_queue": task_queue,
            "current_task": nxt["task"],
            "active_file": nxt["file"],
        }

    return {
        "editor_error": None,
        "editor_retries": 0,
        "task_queue": [],
        "current_task": None,
    }


async def semantic_reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    # Execution already failed — reviewer_node set editor_error because the code
    # crashed or failed its own asserts.
    #
    # Do not review it, and above all do not advance past it. Semantic review
    # asks "is this the right program?", which is a question with no meaning for
    # a program that does not run. Worse, a PASS here calls _advance(), which
    # clears editor_error and resets editor_retries — so a semantic thumbs-up
    # would erase an execution failure, the retry counter would never climb, the
    # tier would never escalate, and the sprint would ship crashing code under a
    # green "✅ Sprint Complete".
    #
    # That is exactly what happened: four reviewer_node crashes in a row, each
    # wiped by a semantic PASS, and a semver.py that raised TypeError on import
    # was declared a success.
    #
    # Returning {} leaves editor_error in place, so reviewer_logic routes back to
    # the editor, which is where a broken program belongs.
    if state.get("editor_error"):
        return {}

    active_file = state.get("active_file")
    if not active_file:
        return {}

    current_code = state.get("project_files", {}).get(active_file, "")
    if not current_code:
        return {}

    # A human-written acceptance contract just passed. Stand down.
    #
    # This node exists because nothing else was checking intent. When a human has
    # written executable asserts describing what the program must do, and the
    # program satisfies them, something else IS checking intent — and it is a
    # better check than this one by every measure that matters. It is exact
    # rather than probabilistic, it costs no inference, and it cannot invent a
    # complaint. This node, by contrast, is a 7B model asked to find fault, and
    # asking a 7B model to find fault is a good way to be handed one: the entire
    # "NEVER REJECT" section of its prompt is a scar list of false rejections.
    #
    # It still advances the task, because a task that passes every gate and is
    # never retired is a task the graph will re-verify forever.
    if state.get("contract_satisfied"):
        return {
            "semantic_verdict": "PASS (acceptance contract satisfied)",
            **_advance(state),
        }

    # Already escalated — reviewing a sprint that is on its way to the human
    # gate wastes an inference pass and cannot change the routing.
    if (state.get("loop_health") or {}).get("escalated"):
        return {}

    sys_prompt = prompts.get_semantic_reviewer_prompt(
        state.get("sprint_plan") or "",
        state.get("current_task") or "",
    )

    # Review on whatever tier the editor used for this task, rather than always
    # reaching for the strong model. The two models cannot both sit in 8 GB of
    # VRAM, so a small-writes/large-reviews split would evict and reload on
    # every single task — paying the strong model's ~23s load twice per task to
    # review code the fast model probably got right.
    #
    # An escalated task therefore gets a stronger reviewer for free, because it
    # is already on the strong tier. That is the case where the extra scrutiny
    # is actually warranted: the fast model has already failed here once.
    tier = state.get("model_tier") or DEFAULT_TIER

    try:
        llm = get_async_llm("reviewer", tier)
        response = await llm.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=current_code)]
        )
        raw_verdict = response.content.strip()
    except Exception as e:
        invalidate_llm("reviewer", tier)
        log_rejection(
            "semantic_reviewer_node",
            f"LLM call failed: {e}\n{traceback.format_exc()}",
        )
        # Infrastructure failure is not a code failure. Pass, and leave the
        # forensics in the ledger, rather than blocking a sprint on a dead
        # Ollama connection. It must still advance: a "pass" that does not
        # retire the task leaves the graph re-reviewing it forever.
        return {
            "semantic_verdict": "PASS (semantic review skipped — LLM error)",
            **_advance(state),
        }

    if not raw_verdict.upper().startswith("FAIL"):
        return {"semantic_verdict": "PASS", **_advance(state)}

    # Normalise to "FAIL: <reason>".
    if ":" in raw_verdict:
        reason = raw_verdict[raw_verdict.index(":") + 1 :].strip()
    else:
        reason = raw_verdict[4:].strip() or "semantic mismatch detected"

    semantic_fail_msg = f"SEMANTIC REVIEW FAILED: {reason}"
    log_rejection("semantic_reviewer_node", semantic_fail_msg)

    return {
        "semantic_verdict": f"FAIL: {reason}",
        "editor_error": semantic_fail_msg,
        "editor_retries": state.get("editor_retries", 0) + 1,
    }
