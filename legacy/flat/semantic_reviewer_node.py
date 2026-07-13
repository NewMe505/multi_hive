"""
semantic_reviewer_node.py — Adversarial semantic verification (v4.2).

Phase 3 addition. Sits between reviewer_node and the conditional routing
decision in the graph:

  async_editor_node → reviewer_node → semantic_reviewer_node → [routing]

Purpose
-------
reviewer_node verifies that code RUNS and passes its own asserts.
It cannot verify that the code implements what was actually asked for.
A model can write a syntactically valid, passing script that is
completely wrong semantically — wrong function signatures, missing
requirements, different file saved than specified.

This node catches that class of failure using the same
qwen2.5-coder:7b model with adversarial role framing:
"You are a hostile reviewer — find reasons to reject."

Same-model adversarial review is not as strong as a separate verifier
model, but it catches the most egregious semantic divergences without
requiring a second Ollama instance or a larger model. On a CPU-only
16GB machine, this is the practical trade-off.

Response contract
-----------------
The model is instructed to respond with exactly:
  PASS                — all requirements met, advance to next task
  FAIL: <one reason>  — specific semantic mismatch, send back to editor

Parsing is defensive: anything that doesn't start with "FAIL" is
treated as PASS to avoid false-positive escalations from model
confusion or verbose responses.

State effects
-------------
On PASS:  sets semantic_verdict="PASS", advances task queue normally
On FAIL:  sets semantic_verdict="FAIL: <reason>", sets editor_error
          to the semantic failure reason so the editor receives it as
          a runtime failure (routed through the existing retry loop)
"""
import asyncio
import traceback
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage

import hive_prompts
from hive_memory import log_rejection
from llm_factory import get_async_llm, invalidate_llm


async def semantic_reviewer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    active_file  = state.get("active_file")
    current_task = state.get("current_task") or ""
    sprint_plan  = state.get("sprint_plan")  or ""
    loop_health  = state.get("loop_health")
    project_files = state.get("project_files", {})

    # Skip semantic review if:
    # - no active file (nothing was generated)
    # - no code in project_files for this file (editor never wrote it)
    # - already escalated (no point reviewing a failed sprint)
    if not active_file:
        return {}
    current_code = project_files.get(active_file, "")
    if not current_code:
        return {}
    if (loop_health or {}).get("escalated"):
        return {}

    # ── Build adversarial review prompt ──────────────────────────────────────
    sys_prompt  = hive_prompts.get_semantic_reviewer_prompt(sprint_plan, current_task)
    user_prompt = current_code

    # ── Async LLM call ────────────────────────────────────────────────────────
    try:
        llm      = get_async_llm("reviewer")
        response = await llm.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        )
        raw_verdict = response.content.strip()
    except Exception as e:
        invalidate_llm("reviewer")
        log_rejection(
            "semantic_reviewer_node",
            f"LLM call failed: {e}\n{traceback.format_exc()}",
        )
        # On LLM failure, treat as PASS to avoid blocking the sprint
        # on infrastructure issues. Log the failure for forensics.
        return {"semantic_verdict": "PASS (semantic review skipped — LLM error)"}

    # ── Parse verdict ─────────────────────────────────────────────────────────
    # Defensive: only treat as FAIL if response explicitly starts with "FAIL"
    # (case-insensitive). Verbose, confused, or empty responses → PASS.
    if raw_verdict.upper().startswith("FAIL"):
        # Normalise to "FAIL: <reason>" format
        if ":" in raw_verdict:
            reason = raw_verdict[raw_verdict.index(":") + 1:].strip()
        else:
            reason = raw_verdict[4:].strip() or "semantic mismatch detected"

        semantic_fail_msg = f"SEMANTIC REVIEW FAILED: {reason}"
        log_rejection("semantic_reviewer_node", semantic_fail_msg)

        # Route back through the editor retry loop by setting editor_error.
        # The editor will receive this in its PAST RUNTIME/ASSERTION FAILURES
        # feed (since semantic_reviewer_node logs under "semantic_reviewer_node"
        # — we inject it as editor_error directly so the retry is triggered,
        # but the label in the prompt will show it as a semantic failure).
        return {
            "semantic_verdict": f"FAIL: {reason}",
            "editor_error":     semantic_fail_msg,
            "editor_retries":   state.get("editor_retries", 0) + 1,
        }

    # PASS — record verdict, leave routing state clean
    return {"semantic_verdict": f"PASS"}
