"""
async_editor_node — the code-generation node.

Async so the 8–45s Ollama inference window yields the event loop: the Rich
console, the stdin gate listener, and any future concurrent task stay
responsive instead of freezing for the duration of every generation.

Repeat-error fingerprinting is the circuit breaker. _error_hash() strips
volatile tokens (line numbers, memory addresses) from a traceback to produce a
stable 8-char fingerprint. If the same logical error arrives twice in a row,
the model is fixing symptoms and will keep doing so — escalate immediately
rather than burning the remaining retry budget to reach the same place.
"""
from __future__ import annotations

import hashlib
import re
import traceback
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.ast_utils import get_code_outline
from multi_hive.core.llm_factory import get_async_llm, invalidate_llm
from multi_hive.core.memory import get_recent_rejections, log_rejection
from multi_hive.state import default_loop_health

_MAX_OBJECTIVE_CHARS = 2000


def _extract_clean_code(raw_text: str) -> str:
    """Longest ```python ... ``` block, or the whole response if unfenced."""
    backticks = chr(96) * 3
    newline = chr(10)
    pattern = backticks + "python" + newline + "(.*?)" + newline + backticks
    matches = re.findall(pattern, raw_text, re.DOTALL)
    return max(matches, key=len).strip() if matches else raw_text.strip()


def _error_hash(error: str) -> str:
    """
    Stable 8-char fingerprint of an error.

    Line numbers and memory addresses shift between retries even when the
    error is logically identical, so they are normalised out before hashing.
    """
    normalised = re.sub(r"line \d+", "line N", error or "")
    normalised = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", normalised)
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:8]


async def async_editor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_task = state.get("current_task")
    if not current_task:
        return {}

    active_file = state.get("active_file") or "outputs/main.py"
    project_files = dict(state.get("project_files", {}))
    current_code = project_files.get(active_file, "")
    editor_error = state.get("editor_error")
    loop_health = dict(state.get("loop_health") or default_loop_health())

    # ── Repeat-error early escalation ────────────────────────────────────────
    if editor_error:
        incoming_hash = _error_hash(editor_error)
        previous_hash = loop_health.get("repeat_error_hash")
        attempts = loop_health.get("attempt_count", 0)

        if previous_hash is not None and previous_hash == incoming_hash and attempts >= 1:
            log_rejection(
                "async_editor_node",
                f"REPEAT ERROR DETECTED (hash={incoming_hash}, attempts={attempts}) "
                f"— escalating without retry. Error: {editor_error[:300]}",
            )
            loop_health["escalated"] = True
            loop_health["last_node"] = "async_editor_node"
            loop_health["repeat_error_hash"] = incoming_hash
            return {
                "editor_error": editor_error,
                "editor_retries": state.get("editor_retries", 0) + 1,
                "loop_health": loop_health,
            }

        loop_health["repeat_error_hash"] = incoming_hash

    loop_health["attempt_count"] = loop_health.get("attempt_count", 0) + 1
    loop_health["last_node"] = "async_editor_node"

    # ── Prompt assembly ───────────────────────────────────────────────────────
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    raw_objective = human_msgs[0].content if human_msgs else ""
    global_objective = (
        raw_objective[:_MAX_OBJECTIVE_CHARS] + "..."
        if len(raw_objective) > _MAX_OBJECTIVE_CHARS
        else raw_objective
    )

    # Three separate failure feeds — the editor must know *which kind* of
    # failure it is fixing. Semantic rejections in particular were invisible
    # here once, and the model reproduced identical code until escalation.
    sys_prompt = prompts.get_editor_prompt(
        global_objective,
        state.get("specialist_context", ""),
        get_recent_rejections("async_editor_node"),
        get_recent_rejections("reviewer_node"),
        get_recent_rejections("semantic_reviewer_node"),
    )

    # Full text for the active file; signature outlines for everything else.
    # Whole files for cross-file context overflow the 4096-token num_ctx after
    # a few modules.
    codebase_context = ""
    for filepath, content in project_files.items():
        if filepath != active_file and content.strip():
            codebase_context += f"\n--- {filepath} (Outline) ---\n{get_code_outline(content)}\n"

    newline = "\n"
    user_prompt = ""
    if codebase_context:
        user_prompt += f"PROJECT ARCHITECTURE OUTLINE:{newline}{codebase_context}{newline}"

    user_prompt += (
        f"CURRENT FILE CODEBASE:{newline}BEGIN_CODEBASE_DATA{newline}"
        f"{current_code}{newline}END_CODEBASE_DATA{newline}{newline}"
        f"EXECUTE THIS SPECIFIC TASK:{newline}{current_task}"
    )

    if editor_error:
        user_prompt += (
            f"{newline}{newline}YOUR LAST ATTEMPT FAILED WITH THIS EXACT TRACEBACK:{newline}"
            f"{editor_error}{newline}FIX THE CODE SO IT PASSES."
        )

    # ── Generation ────────────────────────────────────────────────────────────
    try:
        llm = get_async_llm("editor")
        response = await llm.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        )
        project_files[active_file] = _extract_clean_code(response.content)

        # Clear the fingerprint on success so the next task starts clean.
        loop_health["repeat_error_hash"] = None

        return {
            "project_files": project_files,
            "active_file": active_file,
            "editor_error": None,
            "loop_health": loop_health,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"[{active_file}] Task written.")],
        }

    except Exception as e:
        # A connection-level failure means the cached client may be dead —
        # drop it so the next attempt rebuilds against a restarted Ollama.
        invalidate_llm("editor")
        log_rejection(
            "async_editor_node",
            f"Generation Error: {e}\n{traceback.format_exc()}",
        )
        return {
            "editor_error": str(e),
            "loop_health": loop_health,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"SYSTEM ERROR IN EDITOR: {e}")],
        }
