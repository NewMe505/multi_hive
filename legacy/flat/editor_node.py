import re, traceback
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import hive_prompts
from llm_factory import get_llm, invalidate_llm
from hive_memory import log_rejection, get_recent_rejections
from ast_utils import get_code_outline

def _extract_clean_code(raw_text: str) -> str:
    bp = chr(96) * 3
    nl = chr(10)
    pattern = bp + "python" + nl + "(.*?)" + nl + bp
    matches = re.findall(pattern, raw_text, re.DOTALL)
    return max(matches, key=len).strip() if matches else raw_text.strip()

def editor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    current_task = state.get("current_task")
    active_file = state.get("active_file") or "outputs/main.py"
    project_files = dict(state.get("project_files", {}))
    
    current_code = project_files.get(active_file, "")
    editor_error = state.get("editor_error")

    if not current_task: return {}

    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    raw_objective = human_msgs[0].content if human_msgs else ""
    # FIX-OBJ: 200-char truncation was cutting off the target filename and assert
    # specifications before they reached the model. The editor num_ctx is 4096 tokens
    # (~16KB text); the objective is at most a few hundred words, so there is no
    # reason to truncate it. Cap raised to 2000 chars to safely fit within the
    # system-prompt + codebase context without overflowing num_ctx.
    global_objective = (raw_objective[:2000] + "...") if len(raw_objective) > 2000 else raw_objective
    
    # FIX-LOG1: read editor generation failures and reviewer runtime failures separately.
    # The model needs to know whether it failed to generate valid code (editor_node key)
    # or whether valid code failed at runtime/assertion (reviewer_node key).
    # Mixing them produces incoherent retry strategies.
    past_gen_failures      = get_recent_rejections("editor_node")
    past_runtime_failures  = get_recent_rejections("reviewer_node")
    sys_prompt = hive_prompts.get_editor_prompt(
        global_objective,
        state.get("specialist_context", ""),
        past_gen_failures,
        past_runtime_failures,
    )

    # OPT-PERF: Inject full text for active file, but AST outlines for cross-file context
    codebase_context = ""
    for fp, content in project_files.items():
        if fp != active_file and content.strip():
            codebase_context += f"\n--- {fp} (Outline) ---\n" + get_code_outline(content) + "\n"

    nl = chr(10)
    user_prompt = ""
    if codebase_context:
        user_prompt += "PROJECT ARCHITECTURE OUTLINE:" + nl + codebase_context + nl

    user_prompt += "CURRENT FILE CODEBASE:" + nl + "BEGIN_CODEBASE_DATA" + nl + current_code + nl + "END_CODEBASE_DATA" + nl + nl
    user_prompt += "EXECUTE THIS SPECIFIC TASK:" + nl + current_task

    if editor_error:
        user_prompt += nl + nl + "YOUR LAST ATTEMPT FAILED WITH THIS EXACT TRACEBACK:" + nl
        user_prompt += editor_error + nl + "FIX THE CODE SO IT PASSES."

    try:
        llm = get_llm("editor")
        response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
        clean_code = _extract_clean_code(response.content)
        
        project_files[active_file] = clean_code
        
        return {
            "project_files": project_files,
            "active_file": active_file,
            "editor_error": None,
            "messages": state.get("messages", []) + [AIMessage(content="[" + active_file + "] Task written.")]
        }
    except Exception as e:
        invalidate_llm("editor")
        error_msg = f"Generation/AST Error: {str(e)}\n{traceback.format_exc()}"
        # FIX-LOG1: generation failures belong under "editor_node" — correct key,
        # now cleanly separated from reviewer_node's sandbox/syntax failures.
        log_rejection("editor_node", error_msg)
        return {"editor_error": str(e), "messages": state.get("messages", []) + [AIMessage(content="SYSTEM ERROR IN EDITOR: " + str(e))]}
