"""sprint_planner — turns the user objective into a short implementation plan."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import get_llm


def sprint_planner(state: dict[str, Any]) -> dict[str, Any]:
    llm = get_llm("planner")

    # Filter explicitly for HumanMessage: AIMessage entries injected by later
    # nodes must not bleed into the planner's objective on a re-entry.
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    objective = human_msgs[0].content if human_msgs else ""

    response = llm.invoke(
        [
            SystemMessage(content=prompts.get_sprint_planner_prompt()),
            HumanMessage(content=objective),
        ]
    )
    return {"sprint_plan": response.content}
