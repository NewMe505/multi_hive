"""ticket_writer — turns the sprint plan into a JSON task queue."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import get_llm


def _extract_task_list(raw_text: str) -> List[dict]:
    """
    Pulls the JSON task list out of the model's response.

    Non-greedy regex rather than json.loads on the whole response: the model
    routinely wraps the list in prose or a markdown fence, and a greedy bracket
    match swallows trailing junk and fails to parse.
    """
    match = re.search(r"\[\s*\{.*?\}\s*\]", raw_text, re.DOTALL)
    if not match:
        return []

    try:
        tasks = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if not isinstance(tasks, list):
        return []

    return [
        t
        for t in tasks
        if isinstance(t, dict)
        and isinstance(t.get("file"), str)
        and isinstance(t.get("task"), str)
    ]


def ticket_writer(state: Dict[str, Any]) -> Dict[str, Any]:
    # Explicit None checks, not falsiness: an empty-but-present queue means
    # "the queue was built and drained", which is not the same as "no queue".
    if state.get("task_queue") is not None and len(state.get("task_queue", [])) > 0:
        return {}
    if state.get("current_task") is not None:
        return {}

    llm = get_llm("ticket")
    sprint_plan = state.get("sprint_plan") or ""

    # Pass the original user objective alongside the plan: the objective is the
    # only place an explicit target path ("save it to outputs/dsp_pipeline.py")
    # appears. Without it the model invents filenames from the plan alone.
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    user_objective = human_msgs[0].content if human_msgs else ""

    ticket_input = (
        f"USER OBJECTIVE (use any explicit file paths from here):\n{user_objective}\n\n"
        f"SPRINT PLAN:\n{sprint_plan}"
    )

    response = llm.invoke(
        [
            SystemMessage(content=prompts.get_ticket_writer_prompt()),
            HumanMessage(content=ticket_input),
        ]
    )
    tasks = _extract_task_list(response.content)

    if not tasks:
        return {
            "editor_error": "JSON PARSE ERROR: TicketWriter did not output valid JSON.",
            "editor_retries": 1,
        }

    return {
        "task_queue": tasks[1:],
        "current_task": tasks[0]["task"],
        "active_file": tasks[0]["file"],
        "editor_retries": 0,
        "messages": state.get("messages", []) + [AIMessage(content="QUEUE GENERATED.")],
    }
