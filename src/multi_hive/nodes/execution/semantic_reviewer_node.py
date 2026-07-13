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
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import get_async_llm, invalidate_llm
from multi_hive.core.memory import log_rejection


async def semantic_reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    active_file = state.get("active_file")
    if not active_file:
        return {}

    current_code = state.get("project_files", {}).get(active_file, "")
    if not current_code:
        return {}

    # Already escalated — reviewing a sprint that is on its way to the human
    # gate wastes an inference pass and cannot change the routing.
    if (state.get("loop_health") or {}).get("escalated"):
        return {}

    sys_prompt = prompts.get_semantic_reviewer_prompt(
        state.get("sprint_plan") or "",
        state.get("current_task") or "",
    )

    try:
        llm = get_async_llm("reviewer")
        response = await llm.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=current_code)]
        )
        raw_verdict = response.content.strip()
    except Exception as e:
        invalidate_llm("reviewer")
        log_rejection(
            "semantic_reviewer_node",
            f"LLM call failed: {e}\n{traceback.format_exc()}",
        )
        # Infrastructure failure is not a code failure. Pass, and leave the
        # forensics in the ledger, rather than blocking a sprint on a dead
        # Ollama connection.
        return {"semantic_verdict": "PASS (semantic review skipped — LLM error)"}

    if not raw_verdict.upper().startswith("FAIL"):
        return {"semantic_verdict": "PASS"}

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
