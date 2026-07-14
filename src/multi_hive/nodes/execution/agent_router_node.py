"""agent_router_node — injects domain rules, picks the model tier, resets state."""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from multi_hive.core.model_router import classify_complexity, select_tier
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

    # A fresh task starts on the fast model unless it looks hard up front —
    # retries are what escalate it, and this node only ever runs at the start
    # of a task, so retries are 0 by definition here.
    #
    # ...which is true of this *process* and false of the *work*. tier_floor is how
    # a previous sprint's evidence survives: discovery sets it when replaying an
    # objective that already escalated, because re-running a known failure on the
    # model that produced it is not a retry, it is the same bet. select_tier owns
    # the precedence (FORCE_TIER still wins over everything).
    #
    # Classified from the OBJECTIVE AND THE TICKET, not the ticket alone.
    #
    # It used to read the ticket alone — which is the 7B ticket writer's paraphrase
    # of a 7B planner's summary of what you asked for. So "start a hard task on the
    # strong model" was firing on a summary, and every hard-task signal the human
    # actually wrote ("O(1)", "full semver precedence", "thread-safe") had to
    # survive two lossy rewrites by the very model the rule exists to route away
    # from. It usually did not.
    #
    # The suite's own `complexity="hard"` label on semver and word_wrap was dead
    # metadata for the same reason: nothing downstream ever read it, because the
    # router re-derived complexity from the paraphrase instead.
    #
    # Both are read now. The objective carries the signal, the ticket carries the
    # specifics, and a hard requirement only has to appear in one of them.
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    objective = human_msgs[0].content if human_msgs else ""

    complexity = classify_complexity(f"{objective}\n{state.get('current_task') or ''}")
    tier = select_tier(
        complexity,
        editor_retries=0,
        repeat_error=False,
        tier_floor=state.get("tier_floor"),
    )

    # loop_health resets at the start of every task: a repeat_error_hash left
    # over from the previous task would otherwise trip an escalation on the
    # first retry of an unrelated one.
    #
    # contract_satisfied resets for the same reason, and it matters more: a True
    # left over from the previous file would tell semantic_reviewer_node to stand
    # down for a file whose contract has not been run yet — retiring the task on
    # the strength of a different file's passing grade.
    return {
        "specialist_context": specialist_context.strip(),
        "is_ui_task": is_ui_task,
        "editor_retries": 0,
        "loop_health": default_loop_health(),
        "semantic_verdict": None,
        "contract_satisfied": None,
        "task_complexity": complexity,
        "model_tier": tier,
    }
