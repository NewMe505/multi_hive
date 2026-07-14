"""
async_editor_node — the code-generation node.

Async so the 8–45s Ollama inference window yields the event loop: the Rich
console, the stdin gate listener, and any future concurrent task stay
responsive instead of freezing for the duration of every generation.

Repeat-error fingerprinting is the circuit breaker. _error_hash() strips
volatile tokens (line numbers, memory addresses) from a traceback to produce a
stable 8-char fingerprint. If the same logical error arrives twice in a row,
the model is fixing symptoms and will keep doing so — escalate immediately
rather than burning the remaining retry budget to reach the same place.
"""
from __future__ import annotations

import hashlib
import re
import traceback
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.contract import contract_for
from multi_hive.core.ast_utils import get_code_outline
from multi_hive.core.llm_factory import get_async_llm, invalidate_llm, model_for
from multi_hive.core.memory import get_recent_rejections, log_rejection
from multi_hive.core.model_router import STRONG, classify_complexity, select_tier
from multi_hive.state import default_loop_health

_MAX_OBJECTIVE_CHARS = 2000


def _extract_clean_code(raw_text: str) -> str:
    """Longest ```python ... ``` block, or the whole response if unfenced."""
    backticks = chr(96) * 3
    newline = chr(10)
    pattern = backticks + "python" + newline + "(.*?)" + newline + backticks
    matches = re.findall(pattern, raw_text, re.DOTALL)
    return max(matches, key=len).strip() if matches else raw_text.strip()


def _error_hash(error: str) -> str:
    """
    Stable 8-char fingerprint of an error.

    Line numbers and memory addresses shift between retries even when the
    error is logically identical, so they are normalised out before hashing.
    """
    normalised = re.sub(r"line \d+", "line N", error or "")
    normalised = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", normalised)
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()[:8]


async def async_editor_node(state: dict[str, Any]) -> dict[str, Any]:
    current_task = state.get("current_task")
    if not current_task:
        return {}

    active_file = state.get("active_file") or "outputs/main.py"
    project_files = dict(state.get("project_files", {}))
    current_code = project_files.get(active_file, "")
    editor_error = state.get("editor_error")
    loop_health = dict(state.get("loop_health") or default_loop_health())

    # ── Repeat-error early escalation ────────────────────────────────────────
    if editor_error:
        incoming_hash = _error_hash(editor_error)
        previous_hash = loop_health.get("repeat_error_hash")
        attempts = loop_health.get("attempt_count", 0)

        if previous_hash is not None and previous_hash == incoming_hash and attempts >= 1:
            # Logged under "escalation", not "async_editor_node": this is a routing
            # decision, not a code-generation failure. get_recent_rejections(
            # "async_editor_node") feeds the editor's "PAST GENERATION FAILURES —
            # fix the code structure" section, and telling the model to "fix the
            # code structure" of a router log line is a category error.
            log_rejection(
                "escalation",
                f"REPEAT ERROR DETECTED (hash={incoming_hash}, attempts={attempts}) "
                f"— escalating without retry. Error: {editor_error[:300]}",
            )
            loop_health["escalated"] = True
            loop_health["last_node"] = "async_editor_node"
            loop_health["repeat_error_hash"] = incoming_hash
            return {
                "editor_error": editor_error,
                "editor_retries": state.get("editor_retries", 0) + 1,
                "loop_health": loop_health,
            }

        loop_health["repeat_error_hash"] = incoming_hash

    loop_health["attempt_count"] = loop_health.get("attempt_count", 0) + 1
    loop_health["last_node"] = "async_editor_node"

    # ── Model tier ────────────────────────────────────────────────────────────
    # Escalate to the strong model once the fast one has failed this task.
    # Retrying a failure with the model that just produced it mostly re-buys the
    # failure; the retry budget is only worth what it changes.
    #
    # The tier ratchets upward and never falls back within a task: the two models
    # do not fit in VRAM together, so flip-flopping would evict and reload on
    # every attempt.
    previous_tier = state.get("model_tier")
    complexity = state.get("task_complexity") or classify_complexity(current_task)
    tier = select_tier(
        complexity,
        editor_retries=state.get("editor_retries", 0),
        # NOT repeat_error. A genuine same-error-twice repeat has already returned
        # to the human gate in the block above. By the time control reaches here,
        # repeat_error_hash only means "an error occurred", which is true on the
        # FIRST retry of every failed task — passing that as repeat_error forced
        # STRONG on retry 1 and silently disabled ESCALATE_AFTER_FAILURES > 1.
        # editor_retries >= ESCALATE_AFTER_FAILURES owns escalation from here.
        repeat_error=False,
    )
    if previous_tier == STRONG:
        tier = STRONG

    # ── The strong model gets a CLEAN SHOT ────────────────────────────────────
    #
    # On escalation the strong model used to inherit the fast model's broken code as
    # `CURRENT FILE CODEBASE`, the fast model's traceback, and the order "FIX THE
    # CODE SO IT PASSES". It was being asked to PATCH A BAD DRAFT, not to write the
    # program — while `bench models` hands the same model a blank page and the full
    # spec and it scores 8-9/9.
    #
    # So the benchmark was comparing 30B-as-patcher-of-7B-slop against
    # 30B-as-author, and calling the gap an architectural finding.
    #
    # A tier escalation is a statement that the fast model's ATTEMPT was wrong. Its
    # code is the wrongest thing in the context window, and anchoring a better model
    # to it is the single most expensive mistake in the retry loop: the strong model
    # is now paying VRAM, ~23s of reload, and its whole context budget to inherit a
    # failure.
    #
    # The traceback survives — it says what went wrong, and that is real information —
    # but it is reframed. "A weaker model failed like this, avoid it" is a warning.
    # "Fix this code" is a leash.
    escalating = bool(previous_tier) and previous_tier != tier and tier == STRONG
    if escalating:
        current_code = ""

    if tier != previous_tier:
        # "escalation", not "async_editor_node": a tier change is a routing event,
        # not a generation failure. Logging it under the editor's node name fed it
        # straight back into the editor's "fix the code structure" feed on the next
        # retry — see the REPEAT ERROR note above.
        log_rejection(
            "escalation",
            f"TIER ESCALATION: {previous_tier or 'fast'} -> {tier} "
            f"({model_for(tier)}) after {state.get('editor_retries', 0)} failed "
            f"attempt(s) on a task classified {complexity!r}.",
        )

    # ── Prompt assembly ───────────────────────────────────────────────────────
    human_msgs = [m for m in state.get("messages", []) if isinstance(m, HumanMessage)]
    raw_objective = human_msgs[0].content if human_msgs else ""

    # Truncation is LOUD now. It used to cut mid-character with no warning and no
    # log — half your objective could vanish and nothing said so, while
    # config.MAX_INPUT_CHARS happily accepted an objective twice this long. A
    # silently-halved spec is indistinguishable from a model that ignored half of
    # it, and the second is what you would have gone looking for.
    global_objective = raw_objective
    if len(raw_objective) > _MAX_OBJECTIVE_CHARS:
        global_objective = raw_objective[:_MAX_OBJECTIVE_CHARS] + "..."
        log_rejection(
            # Its own name. It must stay out of the editor's three failure feeds
            # (get_recent_rejections reads async_editor_node / reviewer_node /
            # semantic_reviewer_node), but filing it under "ticket_writer" made the
            # ledger lie: an operator grepping for why the TICKET WRITER misbehaved
            # would find an editor-context event.
            "editor_context",
            f"OBJECTIVE TRUNCATED: {len(raw_objective)} chars cut to "
            f"{_MAX_OBJECTIVE_CHARS} for the editor's context window. The dropped "
            f"tail is: {raw_objective[_MAX_OBJECTIVE_CHARS:][:200]!r}",
        )

    # Three separate failure feeds — the editor must know *which kind* of
    # failure it is fixing. Semantic rejections in particular were invisible
    # here once, and the model reproduced identical code until escalation.
    #
    # The contract, when there is one, is handed to the editor deliberately. It
    # is the spec: hiding it would be asking the model to guess the very thing
    # the human just took the trouble to write down. The risk that it hardcodes
    # against the literals is real, and it is answered with a prompt rule and a
    # benchmark that can actually detect it — not by withholding the spec. See
    # contract.py and bench/contracts.py.
    contract = contract_for(state.get("contracts") or {}, active_file)

    sys_prompt = prompts.get_editor_prompt(
        global_objective,
        state.get("specialist_context", ""),
        get_recent_rejections("async_editor_node"),
        get_recent_rejections("reviewer_node"),
        get_recent_rejections("semantic_reviewer_node"),
        acceptance_contract=contract,
    )

    # Full text for the active file; signature outlines for everything else.
    # Whole files for cross-file context overflow the 4096-token num_ctx after
    # a few modules.
    codebase_context = ""
    for filepath, content in project_files.items():
        if filepath != active_file and content.strip():
            codebase_context += f"\n--- {filepath} (Outline) ---\n{get_code_outline(content)}\n"

    newline = "\n"
    user_prompt = ""
    if codebase_context:
        user_prompt += f"PROJECT ARCHITECTURE OUTLINE:{newline}{codebase_context}{newline}"

    user_prompt += (
        f"CURRENT FILE CODEBASE:{newline}BEGIN_CODEBASE_DATA{newline}"
        f"{current_code}{newline}END_CODEBASE_DATA{newline}{newline}"
        f"EXECUTE THIS SPECIFIC TASK:{newline}{current_task}"
    )

    if editor_error and escalating:
        # A blank page, and the previous failure as a WARNING rather than a patch
        # target. The code that produced this traceback is deliberately not shown:
        # you cannot be anchored to a draft you were never given.
        user_prompt += (
            f"{newline}{newline}A WEAKER MODEL ATTEMPTED THIS AND FAILED WITH:{newline}"
            f"{editor_error}{newline}"
            f"Write the program from scratch. Do NOT try to patch its code — you have "
            f"not been shown it, and it was wrong."
        )
    elif editor_error:
        user_prompt += (
            f"{newline}{newline}YOUR LAST ATTEMPT FAILED WITH THIS EXACT TRACEBACK:{newline}"
            f"{editor_error}{newline}FIX THE CODE SO IT PASSES."
        )

    # The human's own words, verbatim, and LAST.
    #
    # They used to appear only in the system prompt, high up, under a header reading
    # "GLOBAL RULES:" — which looks like boilerplate. The terminal instruction, the
    # thing a model weights most heavily, was `EXECUTE THIS SPECIFIC TASK: <ticket>`,
    # and the ticket is a 7B ticket-writer's paraphrase of a 7B planner's summary of
    # what you actually asked for.
    #
    # Every trap in this benchmark is exactly the kind of clause a paraphrase drops:
    # "touching intervals count as overlapping", "a word longer than width must be
    # hard-split", "ties are broken alphabetically". The model was being graded on
    # requirements it was never shown.
    #
    # So the requirement goes last, verbatim, and it is named as the authority. The
    # ticket stays — it says which FILE and which part of the work — but where the
    # two disagree, the human wins.
    # In CONTRACT mode the requirement is CONTEXT, not the judge — and saying
    # otherwise was a live contradiction.
    #
    # _EDITOR_CONTRACT_PREFIX rule 2 tells the model "An ACCEPTANCE CONTRACT is
    # supplied below... It is the ONLY thing your code will be judged on." Appending
    # "the requirement wins where they disagree" told it something else was the
    # authority — and contracts are DELIBERATELY a strict subset of the objective
    # (bench/contracts.py rule 1: the word_stats contract omits the empty-input and
    # n>vocab cases on purpose). So the model was being handed two authorities that
    # genuinely differ, in the one mode that scores 9/9.
    #
    # The requirement still goes last, because the ticket is still a paraphrase and
    # the model still needs to see what was actually asked. It is just no longer
    # claiming to outrank the contract.
    if contract:
        user_prompt += (
            f"{newline}{newline}THE FULL REQUIREMENT, AS THE HUMAN WROTE IT — for "
            f"CONTEXT. The ACCEPTANCE CONTRACT above is what your code is judged on; "
            f"this is what it is FOR:{newline}{global_objective}"
        )
    else:
        user_prompt += (
            f"{newline}{newline}THE FULL REQUIREMENT, EXACTLY AS THE HUMAN WROTE IT.{newline}"
            f"The task above says which part of the work to do. THIS says what correct "
            f"means. Where they disagree, this wins:{newline}{global_objective}"
        )

    # The FILE ANCHOR — after the objective, and ONLY when the sprint has more than
    # one file.
    #
    # Putting the requirement last is right: it carries the traps a ticket-writer's
    # paraphrase drops. But an objective can describe MORE THAN ONE FILE, and once it
    # became the terminal instruction it took the file selection with it. word_stats
    # asks for two modules and names tokens.py first; handed a ticket for stats.py and
    # then told to read a two-file spec as its final word, the editor wrote tokens.py —
    # into stats.py. The file on disk opened with `# outputs/tokens.py` and defined
    # tokenize(). Every run. 3/3 -> 0/3.
    #
    # So the anchor goes last. But it is a paragraph about file selection, and the last
    # slot is the one the model weights most — an UNCONDITIONAL anchor spent it on a
    # warning about files that do not exist, and semver and word_wrap each lost a run,
    # in BOTH contract and plain mode. Those are precisely the two trap tasks the
    # objective-last change rescued. Displacing the requirement from the terminal slot
    # de-weights it, which is the same mechanism as the original bug, pointed the other
    # way.
    #
    # The anchor exists to disambiguate WHICH file. With one file there is nothing to
    # disambiguate, so it is not worth the slot — the requirement keeps it.
    sprint_files = {active_file}
    sprint_files.update(project_files)
    sprint_files.update(
        t["file"]
        for t in (state.get("task_queue") or [])
        if isinstance(t, dict) and t.get("file")
    )

    if len(sprint_files) > 1:
        user_prompt += (
            f"{newline}{newline}YOU ARE WRITING EXACTLY ONE FILE: {active_file}{newline}"
            f"The requirement above describes several files. The others are written by "
            f"separate calls — do NOT write them here. Output ONLY the full contents of "
            f"{active_file}."
        )

    # ── Generation ────────────────────────────────────────────────────────────
    try:
        llm = get_async_llm("editor", tier)
        response = await llm.ainvoke(
            [SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)]
        )
        extracted = _extract_clean_code(response.content)

        if not extracted:
            # An empty extraction is a generation failure, not a success. Taking
            # the success path here writes "" to the file; reviewer_node then hits
            # `if not current_code: return {}` — no error, no editor_retries bump,
            # nothing logged — and agent_router_node resets the counters every
            # pass, so MAX_RETRIES and the repeat-error breaker can never accrue.
            # The sprint loops to RECURSION_LIMIT and dies FATAL without ever
            # reaching the human gate. Route it through the normal failure ladder.
            msg = "GENERATION PRODUCED NO CODE — the response was empty or an empty code fence."
            log_rejection("async_editor_node", msg)
            return {
                "editor_error": msg,
                "editor_retries": state.get("editor_retries", 0) + 1,
                "loop_health": loop_health,
                "model_tier": tier,
                "task_complexity": complexity,
                "messages": state.get("messages", [])
                + [AIMessage(content=f"[{active_file}] editor produced no code.")],
            }

        project_files[active_file] = extracted

        # The fingerprint is deliberately NOT cleared here.
        #
        # Generating code is not the same as succeeding at the task. Generation
        # almost always "succeeds" — the failure arrives later, from the reviewer
        # or the semantic reviewer. Clearing the fingerprint on generation
        # therefore disarmed the repeat-error circuit breaker for exactly the
        # failures it exists to catch: every cycle wiped the evidence that the
        # last attempt failed the same way, so the same rejection could repeat
        # indefinitely without ever tripping an escalation.
        #
        # agent_router_node resets loop_health at the start of each new task,
        # which is the correct place for a clean slate.

        return {
            "project_files": project_files,
            "active_file": active_file,
            "editor_error": None,
            "loop_health": loop_health,
            "model_tier": tier,
            "task_complexity": complexity,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"[{active_file}] Task written by {model_for(tier)}.")],
        }

    except Exception as e:
        # A connection-level failure means the cached client may be dead —
        # drop it so the next attempt rebuilds against a restarted Ollama.
        invalidate_llm("editor", tier)
        log_rejection(
            "async_editor_node",
            f"Generation Error: {e}\n{traceback.format_exc()}",
        )
        return {
            "editor_error": str(e),
            "loop_health": loop_health,
            "model_tier": tier,
            "task_complexity": complexity,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"SYSTEM ERROR IN EDITOR: {e}")],
        }
