"""
sprint_planner — turns the user objective into a short implementation plan.

The tier is routed, not hardcoded. This node used to call get_llm("planner") with
no tier, which silently defaults to `fast` — so HIVE_FORCE_TIER, the operator's
explicit override, never reached it, and neither did the escalation ladder.

"The hive running on the strong model" was therefore never true. The plan was
always drafted by the 7B, the tickets were always written by the 7B, and the 7B
decides what the task IS. Everything downstream — the editor, both reviewers — is
executing and grading its paraphrase. Escalating the *editor* to a 30B while a 7B
still owns the interpretation of the objective is fixing the wrong half.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import get_llm
from multi_hive.core.model_router import select_plan_tier


def sprint_planner(state: dict[str, Any]) -> dict[str, Any]:
    # Filter explicitly for HumanMessage: AIMessage entries injected by later
    # nodes must not bleed into the planner's objective on a re-entry.
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    objective = human_msgs[0].content if human_msgs else ""

    # Routed from the objective — the human's own words, before anything has
    # summarised them. There is no ticket to read yet, which is exactly what makes
    # this the honest place to judge the work. HIVE_PLAN_TIER pins it; see
    # select_plan_tier for why this is a different decision from select_tier.
    tier = select_plan_tier(objective)

    response = get_llm("planner", tier).invoke(
        [
            SystemMessage(content=prompts.get_sprint_planner_prompt()),
            HumanMessage(content=objective),
        ]
    )
    return {"sprint_plan": response.content}
