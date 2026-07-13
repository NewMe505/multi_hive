import re
import json
from typing import Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import hive_prompts
from llm_factory import get_llm


# OPT-D2: Private helper
def _extract_task_list(raw_text: str) -> list[dict]:
    """Robust regex-based extraction to prevent greedy bracket failures."""
    match = re.search(r'\[\s*\{.*?\}\s*\]', raw_text, re.DOTALL)
    if match:
        try:
            tasks = json.loads(match.group(0))
            if isinstance(tasks, list):
                return [t for t in tasks if isinstance(t, dict) and isinstance(t.get("file"), str) and isinstance(t.get("task"), str)]
        except json.JSONDecodeError:
            pass
    return []


def ticket_writer(state: Dict[str, Any]) -> Dict[str, Any]:
    # OPT-S1: Replaced falsiness check with rigorous None check
    if state.get("task_queue") is not None and len(state.get("task_queue", [])) > 0: 
        return {}
    if state.get("current_task") is not None: 
        return {}
        
    llm = get_llm("ticket")
    sprint_plan = state.get("sprint_plan") or ""
    system_instruction = hive_prompts.get_ticket_writer_prompt()

    # FIX-PATH: Pass the original user objective alongside the sprint plan so the
    # ticket writer can see any explicit file paths the user specified (e.g.
    # "Save it to outputs/dsp_pipeline.py"). Without this, the model invents
    # filenames from the sprint plan alone and ignores the user's target path.
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    user_objective = human_msgs[0].content if human_msgs else ""
    ticket_input = (
        f"USER OBJECTIVE (use any explicit file paths from here):\n{user_objective}\n\n"
        f"SPRINT PLAN:\n{sprint_plan}"
    )

    response = llm.invoke([SystemMessage(content=system_instruction), HumanMessage(content=ticket_input)])
    tasks = _extract_task_list(response.content)
    
    if not tasks:
        # TRC-L2: Decoupled ticket retries from editor retries
        return {"editor_error": "JSON PARSE ERROR: TicketWriter did not output valid JSON.", "editor_retries": 1}
    
    return {
        "task_queue": tasks[1:], 
        "current_task": tasks[0]["task"], 
        "active_file": tasks[0]["file"], 
        "editor_retries": 0, 
        "messages": state.get("messages", []) + [AIMessage(content="QUEUE GENERATED.")]
    }
