import re
from typing import Dict, Any
from hive_state import default_loop_health


SPECIALIST_RULES: dict = {
    ("dsp", "audio", "delay"): "DOMAIN RULE: Use scipy.signal for DSP math. Avoid raw python loops.",
    ("ui", "tkinter", "gui"):  "DOMAIN RULE: Use MVC pattern. Isolate UI from core logic.",
}


def agent_router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    # TRC-T2: (state.get("current_task") or "") guards against None.lower() crash.
    current_task      = (state.get("current_task") or "").lower()
    specialist_context = ""
    is_ui_task        = False

    for keywords, rule in SPECIALIST_RULES.items():
        # SEC-M2: re.escape prevents keyword strings from being treated as
        # regex metacharacters, eliminating the ReDoS attack surface.
        if any(re.search(rf"\b{re.escape(kw)}\b", current_task) for kw in keywords):
            specialist_context += rule + "\n"
            if "ui" in keywords or "tkinter" in keywords or "gui" in keywords:
                is_ui_task = True

    # Reset loop_health at the START of each new task so repeat_error_hash
    # from the previous task doesn't falsely trigger an escalation on the
    # first retry of a different task.
    return {
        "specialist_context": specialist_context.strip(),
        "is_ui_task":         is_ui_task,
        "editor_retries":     0,
        "loop_health":        default_loop_health(),
        # Phase 3: reset semantic verdict so the previous task's result
        # doesn't persist into the new task's review cycle.
        "semantic_verdict":   None,
    }
