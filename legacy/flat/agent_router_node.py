import re
from typing import Dict, Any

SPECIALIST_RULES = {
    ("dsp", "audio", "delay"): "DOMAIN RULE: Use scipy.signal for DSP math. Avoid raw python loops.",
    ("ui", "tkinter", "gui"): "DOMAIN RULE: Use MVC pattern. Isolate UI from core logic."
}

def agent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # TRC-T2: Fix NoneType lower crash 
    current_task = (state.get("current_task") or "").lower()
    specialist_context = ""
    is_ui_task = False
    
    for keywords, rule in SPECIALIST_RULES.items():
        # SEC-M2: Escaping user keyword to prevent ReDoS
        if any(re.search(rf"\b{re.escape(kw)}\b", current_task) for kw in keywords):
            specialist_context += rule + "\n"
            if "ui" in keywords or "tkinter" in keywords or "gui" in keywords:
                is_ui_task = True
                
    return {
        "specialist_context": specialist_context.strip(), 
        "is_ui_task": is_ui_task,
        "editor_retries": 0
    }
