"""ticket_writer — turns the sprint plan into a JSON task queue."""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.core.llm_factory import get_llm
from multi_hive.core.memory import log_rejection
from multi_hive.core.model_router import STRONG, classify_complexity, select_tier
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


def _collapse_by_file(tasks: list[dict]) -> list[dict]:
    """
    One ticket per file. Every ticket that targets the same file becomes one.

    The planner is told "MAXIMUM 4 STEPS", which on a single-function objective
    pressures it into a decomposition it did not need — and the ticket writer
    dutifully turns that into four tickets. Measured live, on `lru_cache`:

        ['outputs/lru.py', 'outputs/lru.py', 'outputs/lru.py', 'outputs/lru.py']

    Four tickets. One file. And a file is not patched, it is REWRITTEN WHOLE by
    the editor on every ticket — so this is four full regenerations of the same
    artefact, each one followed by a full pass through reviewer_node and
    semantic_reviewer_node.

    That costs four times the inference, which is merely wasteful. What it does to
    *quality* is worse, and it is why this is a correctness fix and not an
    optimisation:

    - **It multiplies every false-rejection source by four.** The semantic
      reviewer is a known source of spurious FAILs — its entire NEVER REJECT block
      is a scar list. Running it four times on one file gives it four chances to
      reject code that was already correct, and any one of them burns a retry,
      escalates the tier, and can wake a human.
    - **It asks the reviewer an unanswerable question.** On ticket 4 the semantic
      reviewer is handed a complete LRU cache and asked whether it implements
      "Handle eviction". It is judging a whole program against one quarter of a
      paraphrase, and it has no way to know that is what it is doing.

    Merging preserves every requirement — the sub-tasks are joined, not dropped —
    and hands the editor one shot with the full picture, which is exactly what the
    `models` suite gives the raw model. First-appearance order is kept, so the
    planner's sequencing survives.

    Enforced here, in code, rather than in the planner's prompt. The prompt is
    fixed too (see prompts.get_sprint_planner_prompt), but a prompt is not a
    guarantee — a lesson this file learned once already, two functions up.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []

    for task in tasks:
        target = task["file"]
        if target not in merged:
            merged[target] = {**task}
            order.append(target)
        else:
            # Newline-joined, not comma-joined: these are separate requirements and
            # the editor reads them as a list. Flattening them into prose is how a
            # requirement gets skimmed past.
            merged[target]["task"] += "\n" + task["task"]

    return [merged[target] for target in order]


def ticket_writer(state: dict[str, Any]) -> dict[str, Any]:
    # Explicit None checks, not falsiness: an empty-but-present queue means
    # "the queue was built and drained", which is not the same as "no queue".
    if state.get("task_queue") is not None and len(state.get("task_queue", [])) > 0:
        return {}
    if state.get("current_task") is not None:
        return {}

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
    messages = [
        SystemMessage(content=prompts.get_ticket_writer_prompt()),
        HumanMessage(content=ticket_input),
    ]

    # The tier is ROUTED, not hardcoded.
    #
    # This node used to call get_llm("ticket") with no tier, which silently
    # defaults to `fast` — so HIVE_FORCE_TIER, the operator's explicit override,
    # did not reach it. "The hive on the strong model" was never true: the plan and
    # the tickets were always written by the 7B, and the 7B decides what the task
    # IS. Everything downstream is executing its paraphrase.
    #
    # Complexity is read from the OBJECTIVE. There is no ticket yet to read it
    # from, which is the point — this is the one node that still has the human's
    # own words, before anything has summarised them.
    tier = select_tier(classify_complexity(user_objective))

    response = get_llm("ticket", tier).invoke(messages)
    tasks = _extract_task_list(response.content)

    # ── One retry, on the strong model ────────────────────────────────────────
    #
    # A parse failure here does not fail a task. It kills the ENTIRE SPRINT: no
    # queue is built, reviewer_logic routes straight to the human gate, no code is
    # written, and the run is scored "no code" — indistinguishable from "the model
    # cannot code". A pure infrastructure failure, recorded as a model failure.
    #
    # This is not theoretical. In the first clean baseline it killed
    # `lru_cache --contract` on THREE RUNS OUT OF THREE — 0/3 on a task the same
    # pipeline passes comfortably when the JSON happens to parse. The 7B's JSON is
    # stochastic; emitting a task queue is the one job in this system where a
    # single bad sample costs everything.
    #
    # So: one retry on the strong model, whose JSON is far more reliable. It is a
    # few seconds of inference against losing the whole sprint, and it only ever
    # runs on the path that was already lost.
    if not tasks and tier != STRONG:
        log_rejection(
            "ticket_writer",
            f"JSON PARSE ERROR on the {tier} model — retrying once on {STRONG}. "
            f"A parse failure here kills the whole sprint, so it is worth the "
            f"inference.\nRaw response:\n{response.content[:400]}",
        )
        response = get_llm("ticket", STRONG).invoke(messages)
        tasks = _extract_task_list(response.content)
        tier = STRONG

    if not tasks:
        # Both tiers failed to emit parseable JSON. Now it is a real failure.
        #
        # Logged, because when it went unlogged the resulting loop was invisible:
        # an empty ledger, and no clue why the sprint was spinning.
        error = "JSON PARSE ERROR: TicketWriter did not output valid JSON."
        log_rejection("ticket_writer", f"{error}\nRaw response:\n{response.content[:600]}")
        return {
            "editor_error": error,
            "editor_retries": 1,
        }

    # Legalise first, collapse second. Collapsing keys on the file path, so two
    # tickets that BOTH meant outputs/lru.py — one spelled "lru.py" — have to be
    # normalised to the same string before they can be recognised as the same file.
    tasks = _collapse_by_file(_legalise_paths(tasks))

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
