"""agent_router_node — injects domain rules and resets per-task state."""
from __future__ import annotations

import re
from typing import Any

from multi_hive.state import default_loop_health

SPECIALIST_RULES: dict[tuple[str, ...], str] = {
    ("dsp", "audio", "delay"): "DOMAIN RULE: Use scipy.signal for DSP math. Avoid raw python loops.",
    ("ui", "tkinter", "gui"): "DOMAIN RULE: Use MVC pattern. Isolate UI from core logic.",
}

_UI_KEYWORDS = {"ui", "tkinter", "gui"}


def agent_router_node(state: dict[str, Any]) -> dict[str, Any]:
    # `or ""` guards the None case — current_task is Optional, and .lower() on
    # None was a recurring crash before the type contract was fixed.
    current_task = (state.get("current_task") or "").lower()

    specialist_context = ""
    is_ui_task = False

    for keywords, rule in SPECIALIST_RULES.items():
        # re.escape: keywords are matched as literals, not as regex, which
        # closes the ReDoS surface if this table ever takes user input.
        if any(re.search(rf"\b{re.escape(kw)}\b", current_task) for kw in keywords):
            specialist_context += rule + "\n"
            if _UI_KEYWORDS.intersection(keywords):
                is_ui_task = True

    # loop_health resets at the start of every task: a repeat_error_hash left
    # over from the previous task would otherwise trip an escalation on the
    # first retry of an unrelated one.
    return {
        "specialist_context": specialist_context.strip(),
        "is_ui_task": is_ui_task,
        "editor_retries": 0,
        "loop_health": default_loop_health(),
        "semantic_verdict": None,
    }
