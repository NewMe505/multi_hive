"""ticket_writer — turns the sprint plan into a JSON task queue."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import get_llm
from multi_hive.core.memory import log_rejection
from multi_hive.core.utils import normalise_model_path


def _extract_task_list(raw_text: str) -> list[dict]:
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


def _legalise_paths(tasks: list[dict]) -> list[dict]:
    """
    Forces every ticket's path inside the workspace, dropping the ones that cannot
    be.

    This is the boundary. Every path in the queue ends up in `active_file` — the
    first one immediately, the rest as semantic_reviewer_node retires tasks and
    pulls the next — so validating only `tasks[0]` would just move the bug later
    into the sprint, where it is more expensive.

    Getting it wrong is not a style question. An illegal path is only caught at
    write time, by which point a full generation has been paid for, and the
    resulting `FILE SYSTEM ERROR` is routed back to the editor as though it were a
    code failure. It is not: the path lives in state and the editor cannot change
    it. Every retry regenerates the same file for the same illegal path, fails
    identically, and burns the whole retry budget before escalating to a human for
    a problem no human was needed for.

    Both outcomes are logged under "ticket_writer" and not under any of the three
    node names the editor's failure feeds read from. The editor must not be handed
    a routing complaint and told to "fix the code structure" — that is a category
    error the repeat-error breaker has already been taught once.
    """
    legal: list[dict] = []

    for task in tasks:
        original = task["file"]
        normalised = normalise_model_path(original)

        if normalised is None:
            log_rejection(
                "ticket_writer",
                f"DROPPED TICKET — {original!r} resolves outside workspace/src and "
                f"workspace/outputs, and cannot be made legal without guessing. "
                f"Task: {task['task'][:120]}",
            )
            continue

        if normalised != original:
            log_rejection(
                "ticket_writer",
                f"PATH NORMALISED {original!r} -> {normalised!r} — a bare filename "
                f"is not a workspace path. The prompt says so and the model emitted "
                f"one anyway, which is why this is enforced in code.",
            )

        legal.append({**task, "file": normalised})

    return legal


def ticket_writer(state: dict[str, Any]) -> dict[str, Any]:
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
        # Log it. This failure produces an editor_error with no task queue behind
        # it, which is a state the router has to handle specially — and when it
        # went unlogged, the resulting loop was invisible: an empty ledger and no
        # clue why the sprint was spinning.
        error = "JSON PARSE ERROR: TicketWriter did not output valid JSON."
        log_rejection("ticket_writer", f"{error}\nRaw response:\n{response.content[:600]}")
        return {
            "editor_error": error,
            "editor_retries": 1,
        }

    tasks = _legalise_paths(tasks)

    if not tasks:
        # Every ticket named a file the hive is not allowed to write. Fail here,
        # loudly, rather than handing the graph a queue it will spend the whole
        # retry budget failing to write — which is precisely what used to happen.
        error = (
            "PATH ERROR: every ticket named a file outside workspace/src and "
            "workspace/outputs. Nothing in this plan is writable."
        )
        log_rejection("ticket_writer", error)
        return {
            "editor_error": error,
            "editor_retries": 1,
        }

    return {
        "task_queue": tasks[1:],
        "current_task": tasks[0]["task"],
        "active_file": tasks[0]["file"],
        "editor_retries": 0,
        "messages": state.get("messages", []) + [AIMessage(content="QUEUE GENERATED.")],
    }
