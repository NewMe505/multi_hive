"""
async_editor_node.py — Async LLM code-generation node (v4.1).

Replaces the sync editor_node.py from v4.0. Changes:

1. llm.ainvoke() instead of llm.invoke()
   Yields the asyncio event loop during the 8-45s Ollama inference
   window so the Rich console, the stdin gate listener, and any future
   concurrent tasks remain responsive.

2. Repeat-error fingerprinting (loop-engineering circuit-breaker)
   _error_hash() produces a stable 8-char fingerprint by stripping
   volatile tokens (line numbers, memory addresses) from tracebacks.
   If the same logical error appears on two consecutive attempts, the
   loop is symptom-fixing — we escalate before burning the remaining
   MAX_RETRIES budget instead of waiting for the counter to hit zero.

3. loop_health tracking
   Every invocation updates loop_health.attempt_count and
   loop_health.last_node so retrospector_node can persist the
   convergence signal to metrics.jsonl for post-sprint analysis.
"""
import hashlib
import re
import traceback
from typing import Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import hive_prompts
from hive_memory import log_rejection, get_recent_rejections
from hive_state import default_loop_health
from llm_factory import get_async_llm, invalidate_llm

try:
    from ast_utils import get_code_outline
except ImportError:
    # ast_utils is a local utility not included in the package; provide
    # a passthrough so the node works even if the module is absent.
    def get_code_outline(content: str) -> str:  # type: ignore
        return content[:500] + "\n# [outline truncated]" if len(content) > 500 else content


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_clean_code(raw_text: str) -> str:
    """
    Extracts the longest ```python ... ``` block from an LLM response.
    Falls back to the full response if no fenced block is found.
    """
    bp = chr(96) * 3
    nl = chr(10)
    pattern = bp + "python" + nl + "(.*?)" + nl + bp
    matches = re.findall(pattern, raw_text, re.DOTALL)
    return max(matches, key=len).strip() if matches else raw_text.strip()


def _error_hash(error: str) -> str:
    """
    8-char stable fingerprint of an error message.

    Strips volatile tokens so the same logical error (e.g.
    NameError: name 'foo' is not defined) produces the same hash even
    if line numbers or memory addresses shift between retries.
    """
    normalised = re.sub(r"line \d+", "line N", error or "")
    normalised = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", normalised)
    return hashlib.sha1(normalised.encode()).hexdigest()[:8]


# ── Node ─────────────────────────────────────────────────────────────────────

async def async_editor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_task  = state.get("current_task")
    active_file   = state.get("active_file") or "outputs/main.py"
    project_files = dict(state.get("project_files", {}))
    current_code  = project_files.get(active_file, "")
    editor_error  = state.get("editor_error")
    loop_health   = dict(state.get("loop_health") or default_loop_health())

    if not current_task:
        return {}

    # ── Repeat-error early-escalation check ──────────────────────────────────
    # If the incoming error fingerprint matches the previous attempt's
    # fingerprint, the model is not making progress — set escalated=True so
    # reviewer_logic routes to human_gate_node on the next evaluation instead
    # of granting another retry that will produce the same failure.
    if editor_error:
        incoming_hash = _error_hash(editor_error)
        prev_hash     = loop_health.get("repeat_error_hash")
        attempts      = loop_health.get("attempt_count", 0)

        if prev_hash is not None and prev_hash == incoming_hash and attempts >= 1:
            log_rejection(
                "async_editor_node",
                f"REPEAT ERROR DETECTED (hash={incoming_hash}, attempts={attempts}) "
                f"— escalating without retry. Error: {editor_error[:300]}",
            )
            loop_health["escalated"]         = True
            loop_health["last_node"]         = "async_editor_node"
            loop_health["repeat_error_hash"] = incoming_hash
            return {
                "editor_error":  editor_error,
                "editor_retries": state.get("editor_retries", 0) + 1,
                "loop_health":   loop_health,
            }

        loop_health["repeat_error_hash"] = incoming_hash

    loop_health["attempt_count"] = loop_health.get("attempt_count", 0) + 1
    loop_health["last_node"]     = "async_editor_node"

    # ── Prompt assembly ───────────────────────────────────────────────────────
    human_msgs     = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    raw_objective  = human_msgs[0].content if human_msgs else ""
    # FIX-OBJ: cap raised to 2000 chars — 200 was truncating filenames and
    # assert specs before they reached the model.
    global_objective = (raw_objective[:2000] + "...") if len(raw_objective) > 2000 else raw_objective

    # FIX-LOG1: read generation failures and runtime/assertion failures
    # separately so the model knows which type of fix to apply on retry.
    # FIX-SEM: also read semantic_reviewer_node failures as a third feed —
    # without this the editor retried with zero context about the semantic
    # rejection and produced identical code three times until escalation.
    past_gen_failures      = get_recent_rejections("async_editor_node")
    past_runtime_failures  = get_recent_rejections("reviewer_node")
    past_semantic_failures = get_recent_rejections("semantic_reviewer_node")
    sys_prompt = hive_prompts.get_editor_prompt(
        global_objective,
        state.get("specialist_context", ""),
        past_gen_failures,
        past_runtime_failures,
        past_semantic_failures,
    )

    codebase_context = ""
    for fp, content in project_files.items():
        if fp != active_file and content.strip():
            codebase_context += f"\n--- {fp} (Outline) ---\n{get_code_outline(content)}\n"

    nl = "\n"
    user_prompt = ""
    if codebase_context:
        user_prompt += f"PROJECT ARCHITECTURE OUTLINE:{nl}{codebase_context}{nl}"

    user_prompt += (
        f"CURRENT FILE CODEBASE:{nl}BEGIN_CODEBASE_DATA{nl}"
        f"{current_code}{nl}END_CODEBASE_DATA{nl}{nl}"
        f"EXECUTE THIS SPECIFIC TASK:{nl}{current_task}"
    )

    if editor_error:
        user_prompt += (
            f"{nl}{nl}YOUR LAST ATTEMPT FAILED WITH THIS EXACT TRACEBACK:{nl}"
            f"{editor_error}{nl}FIX THE CODE SO IT PASSES."
        )

    # ── Async LLM call — yields event loop during Ollama inference ────────────
    try:
        llm      = get_async_llm("editor")
        response = await llm.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        )
        clean_code = _extract_clean_code(response.content)
        project_files[active_file] = clean_code

        # Reset repeat-error tracking on success so the next task starts clean.
        loop_health["repeat_error_hash"] = None

        return {
            "project_files": project_files,
            "active_file":   active_file,
            "editor_error":  None,
            "loop_health":   loop_health,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"[{active_file}] Task written.")
            ],
        }

    except Exception as e:
        invalidate_llm("editor")
        error_msg = f"Generation/AST Error: {e}\n{traceback.format_exc()}"
        log_rejection("async_editor_node", error_msg)
        return {
            "editor_error": str(e),
            "loop_health":  loop_health,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"SYSTEM ERROR IN EDITOR: {e}")
            ],
        }
