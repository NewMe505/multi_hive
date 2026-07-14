"""
model_router.py — which model tier should this task run on?

Two tiers, chosen by measurement on the target machine (see
scripts/bench_models.py, and docs/ARCHITECTURE.md for the numbers):

    fast    qwen2.5-coder:7b    54.6 tok/s, 100% on GPU
    strong  qwen3-coder:30b     37.0 tok/s, mixture-of-experts

The strong model is bigger and slower per token, but it is the escalation
target: when the fast model has already failed a task, spending the remaining
retry budget on *the same model that just failed* mostly buys the same failure
again. A retry is only worth what it changes.

Why the tier is sticky per task
-------------------------------
The two models cannot both be resident: 4.7 + 6.1 GB against 8 GB of VRAM. Every
switch is an eviction and a reload, and the strong model takes up to ~23s to
load. So the tier is chosen once per task and held for the rest of it — editor
and semantic reviewer alike. A design where the small model writes and the big
model reviews *every* task would ping-pong the two in and out of VRAM and pay
that reload twice per task, which costs more than the review is worth.

Escalation is the rare path by definition, so it can afford the reload.
"""
from __future__ import annotations

import re

from multi_hive.config import ESCALATE_AFTER_FAILURES, FORCE_TIER, PLAN_TIER

FAST = "fast"
STRONG = "strong"

# Signals that a task is more than boilerplate. Deliberately crude — this is a
# prior, not a verdict. Getting it wrong is cheap: a task misjudged as hard runs
# slower, and a task misjudged as easy escalates on its first failure anyway.
_HARD_TASK_PATTERNS = (
    r"\bconcurren\w*|\basync\b|\bthread\b|\brace\b|\block\b",
    r"\balgorithm\b|\boptimi[sz]e\b|\bO\(\s*[1n]\s*\)|\bcomplexity\b",
    r"\brefactor\b|\bmigrat\w+|\barchitect\w+",
    r"\bstate machine\b|\bparser\b|\bcompiler\b|\brecursi\w+",
    r"\bedge case\w*|\binvariant\w*|\bthread-safe\b",
)


def classify_complexity(task: str | None) -> str:
    """
    A cheap prior on task difficulty: "trivial" | "moderate" | "hard".

    Text-only, no model call — spending an inference just to decide which model
    to run the inference on would cost more than it saves.
    """
    text = (task or "").strip().lower()
    if not text:
        return "moderate"

    if any(re.search(pattern, text) for pattern in _HARD_TASK_PATTERNS):
        return "hard"

    # A long, clause-heavy task is usually carrying several requirements.
    if len(text) > 240 or text.count(",") + text.count(";") >= 4:
        return "moderate"

    return "trivial"


def select_tier(
    complexity: str,
    editor_retries: int = 0,
    repeat_error: bool = False,
    tier_floor: str | None = None,
) -> str:
    """
    The routing decision.

    Escalates to the strong model when any of these hold:

    1. FORCE_TIER is set — the operator's override wins over everything,
       including the floor. It exists so a benchmark can pin a tier, and a
       benchmark that is silently un-pinned by a routing rule is not a benchmark.
    2. tier_floor is STRONG — a previous *sprint* already failed this task on the
       fast model. See below.
    3. The task is classified "hard" — start strong rather than pay a failed
       fast attempt plus a reload to arrive there anyway.
    4. editor_retries >= ESCALATE_AFTER_FAILURES — the fast model has now
       demonstrably failed this task. Another attempt from it is the same bet.
    5. repeat_error — the same error fingerprint twice means the model is
       fixing symptoms. A different model is the only thing that changes the
       outcome; more attempts from this one will not.

    Note the ordering relative to the human gate: repeat_error also drives
    escalation to a human. Trying the strong model first is the cheaper move,
    and only if *it* also cycles does the sprint bother the operator.

    On the floor
    ------------
    Rules 4 and 5 escalate within a sprint, from evidence gathered in that sprint.
    The floor is the same idea carried ACROSS sprints, and it is what makes
    discovery's replay a real retry rather than a re-run: `editor_retries` is 0 at
    the start of every sprint, so a rediscovered objective would otherwise be
    routed straight back to the model that already failed it, reproduce the
    identical failure, and be counted as work. The evidence that the fast model
    cannot do this task did not stop being true when the process exited.

    Set only by the entrypoint, from discovery. Never by a node.
    """
    if FORCE_TIER in (FAST, STRONG):
        return FORCE_TIER

    if tier_floor == STRONG:
        return STRONG

    if complexity == "hard":
        return STRONG

    if repeat_error:
        return STRONG

    if editor_retries >= ESCALATE_AFTER_FAILURES:
        return STRONG

    return FAST


def select_plan_tier(objective: str | None) -> str:
    """
    Which tier drafts the plan and writes the tickets.

    A separate decision from select_tier(), because it answers a different
    question. select_tier asks "how hard is this code to WRITE, and has the fast
    model already failed at it?" — a question about the editor, driven by retries.
    This asks "how much do I care about getting the INTERPRETATION of the objective
    right?", and there are no retries here: the plan is drafted once, before any
    code exists, and everything downstream inherits it.

    Precedence, and it matters:

    1. FORCE_TIER — the operator pinned every tier. It outranks everything, because
       a benchmark that is silently un-pinned by a routing rule is not a benchmark.
    2. PLAN_TIER — the operator pinned *this* decision specifically.
    3. The router, classifying the OBJECTIVE. Not a ticket — there is no ticket yet,
       which is precisely what makes this the one node in the system still holding
       the human's own words, before anything has summarised them.

    Note what is NOT here: editor_retries. A ticket writer cannot be "retried into"
    the strong model by a failing editor, because by the time the editor fails, the
    ticket it is failing against has already been written. If the plan is wrong, the
    ladder cannot climb its way out — it will just escalate a bigger model onto the
    wrong task. That asymmetry is the whole argument for HIVE_PLAN_TIER=strong.
    """
    if FORCE_TIER in (FAST, STRONG):
        return FORCE_TIER

    if PLAN_TIER in (FAST, STRONG):
        return PLAN_TIER

    return select_tier(classify_complexity(objective))
