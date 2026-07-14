"""
config.py — every tuneable constant and every path the hive is allowed to
touch, resolved once, in one place.

Workspace model
---------------
Generated code is written to a *workspace*, never into the source tree.
The workspace defaults to ./workspace and holds exactly two directories:

    workspace/src/       generated modules
    workspace/outputs/   generated scripts + the hive's own artefacts
                         (rejection_ledger.jsonl, metrics.jsonl, LOOP.md)

The LLM still emits paths like "outputs/main.py" — those are resolved
relative to WORKSPACE_DIR by core.utils.safe_path(), which also refuses
anything that escapes ALLOWED_DIRS. Keeping the workspace out of src/
matters now that src/ is the package itself: without it, a task writing
to "src/foo.py" would land inside multi_hive's own source.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Provider ──────────────────────────────────────────────────────────────────
#
# Where the models actually run. The graph, the nodes, and the escalation ladder
# do not care — every client comes from core/llm_factory.py, which is the only
# module that knows a provider exists.
#
#   ollama      local, free, offline. The default, and what the whole VRAM story
#               below is about.
#   anthropic   the Claude API. Needs ANTHROPIC_API_KEY and
#               `pip install -e ".[anthropic]"`.
#
# The point of having both is that they can be A/B'd on the same benchmark:
# `HIVE_PROVIDER=anthropic python scripts/bench.py sprint` grades the API models
# against the same hidden test suites, and records under its own subject so the
# local trend is never polluted. See bench/history.py.

PROVIDER = os.environ.get("HIVE_PROVIDER", "ollama").strip().lower()

# ── Model tiers ───────────────────────────────────────────────────────────────
#
# Two tiers, not three. Measured on the target machine with
# `scripts/bench.py models` (RTX 5070 Laptop, 8 GB VRAM):
#
#   qwen2.5-coder:7b    54.6 tok/s   100% on GPU (4.7/4.7 GB)   <- fast
#   qwen2.5-coder:14b   11.6 tok/s    61% on GPU (6.0/10.0 GB)  <- dropped
#   qwen3-coder:30b     37.0 tok/s    32% on GPU (6.1/19.2 GB)  <- strong
#
# The 14B is dropped on purpose: it is the "almost fits" trap. 10 GB into 8 GB
# of VRAM spills 39% of a *dense* model to the CPU, and a dense model touches
# every parameter for every token — 5x slower than the 7B for no measured gain.
#
# The 30B is faster than the 14B despite being twice the size because it is
# mixture-of-experts: ~3B parameters active per token. Bigger model, less
# VRAM-bound, faster.
#
# The two tiers cannot both be resident (4.7 + 6.1 GB > 8 GB), so switching
# tiers evicts and reloads — up to ~23s for the strong model. This is why the
# tier is sticky per task; see core/model_router.py.
#
# NONE of that applies to the anthropic provider: there is no VRAM, no eviction,
# and no reload, so a tier switch there costs nothing but money. The ratchet
# stays anyway — see model_router.select_tier — because "do not downgrade a task
# that has already failed" is good routing regardless of where the model lives.

_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "ollama": {
        "fast": "qwen2.5-coder:7b",
        "strong": "qwen3-coder:30b",
    },
    # Haiku is the cheap first attempt; Fable 5 is the escalation target. The
    # ladder is unchanged in shape — only the axis it climbs. Locally it climbs
    # parameters; here it climbs price.
    "anthropic": {
        "fast": "claude-haiku-4-5-20251001",
        "strong": "claude-fable-5",
    },
}

if PROVIDER not in _DEFAULT_MODELS:
    raise ValueError(
        f"HIVE_PROVIDER={PROVIDER!r} is not a provider. "
        f"Valid: {', '.join(sorted(_DEFAULT_MODELS))}"
    )

MODELS: dict[str, str] = {
    "fast": os.environ.get("HIVE_FAST_MODEL", _DEFAULT_MODELS[PROVIDER]["fast"]),
    "strong": os.environ.get("HIVE_STRONG_MODEL", _DEFAULT_MODELS[PROVIDER]["strong"]),
}

# Retries on the fast model before escalating the task to the strong one.
# 1 means: one failure is enough evidence that another identical attempt is
# not worth its cost.
ESCALATE_AFTER_FAILURES = int(os.environ.get("HIVE_ESCALATE_AFTER", "1"))

# Pin every task to one tier, bypassing the router. "fast" | "strong" | "".
# Useful for benchmarking, and for the case where Ollama is on a machine with
# enough VRAM that none of the above trade-offs apply.
FORCE_TIER = os.environ.get("HIVE_FORCE_TIER", "").strip().lower()

# Which tier drafts the plan and writes the tickets. "fast" | "strong" | "".
# Empty means the normal router decides, which is the current behaviour.
#
# Why you would want this on "strong"
# -----------------------------------
# The plan and the tickets decide what the task IS. Every node after them — the
# editor, both reviewers, the grader — is executing and judging that paraphrase.
# A bad ticket cannot be rescued by escalating the editor, because the editor is
# faithfully building the wrong thing.
#
# It is also the cheapest place in the system to spend the good model. The planner
# emits a few hundred tokens and the ticket writer about 200 of JSON; the editor
# emits thousands, one to three times over. The plan is a rounding error in the
# inference budget and it determines everything downstream of it.
#
# And it removes an entire failure mode. The 7B's JSON is stochastic, and a parse
# failure in the ticket writer does not fail a task — it kills the whole sprint.
# Measured: it killed lru_cache --contract on three runs out of three.
#
# Why it is NOT the default on ollama
# -----------------------------------
# VRAM. The 7B (4.7 GB) and the 30B (6.1 GB) do not fit in 8 GB together, so
# planning on "strong" and writing on "fast" forces an eviction and a reload
# between them: roughly 30-40s added to a sprint whose fast path currently finishes
# in 19-30s. On `brackets` that is a tripling of wall time to improve a plan that
# was already fine.
#
# None of that applies to the anthropic provider — no VRAM, no eviction, no reload,
# and a few hundred tokens of the strong model costs cents. "strong" is very likely
# right there. But it is UNMEASURED, and this project has twice reverted a change
# that was obviously right and turned out not to be: spec_writer (3.11x slower for
# zero quality gain) and deleting the self-asserts (1/4, down from 3/4). So it ships
# as a knob, not as a default, until the benchmark has an opinion.
PLAN_TIER = os.environ.get("HIVE_PLAN_TIER", "").strip().lower()

# Back-compat: the single-model name, still honoured if someone sets HIVE_MODEL.
MODEL_NAME = os.environ.get("HIVE_MODEL", MODELS["fast"])
if os.environ.get("HIVE_MODEL"):
    MODELS["fast"] = MODEL_NAME

# ── Loop tuning ───────────────────────────────────────────────────────────────

MAX_RETRIES = int(os.environ.get("HIVE_MAX_RETRIES", "3"))

# Hard ceiling on graph steps per sprint — a backstop, not a working limit.
#
# A healthy sprint is a handful of nodes per task. LangGraph's own default is
# 25, and it was raised here only because the fallback is what you want to hit
# *fast* when a routing bug produces a cycle no state change can break. Left at
# LangGraph's 10,007-step default, one such loop burned twenty minutes and
# 10,007 LLM-free no-ops before surfacing. Failing at 120 surfaces it in seconds.
RECURSION_LIMIT = int(os.environ.get("HIVE_RECURSION_LIMIT", "120"))

# How long human_gate_node waits for operator acknowledgement before routing to
# retrospector automatically (headless / CI fallback).
GATE_TIMEOUT_SEC = int(os.environ.get("HIVE_GATE_TIMEOUT", "120"))

# How long generated code may run in the reviewer sandbox before it is killed.
SANDBOX_TIMEOUT_SEC = int(os.environ.get("HIVE_SANDBOX_TIMEOUT", "10"))

# Cap on raw user input before it reaches any LLM context window.
MAX_INPUT_CHARS = int(os.environ.get("HIVE_MAX_INPUT_CHARS", "4000"))

# ── Budget governor ───────────────────────────────────────────────────────────
#
# MAX_RETRIES, RECURSION_LIMIT and the repeat-error fingerprint are *per-sprint*
# backstops. None of them is a cost ceiling, and that was fine for exactly as
# long as the hive only ever ran on local Ollama, where inference is free and a
# human is watching it happen.
#
# The anthropic provider changes the arithmetic, and the escalation ladder is
# what makes it sharp: haiku is $1/$5 per Mtok, fable is $10/$50. The ladder
# climbs to the expensive tier precisely when a task is failing — i.e. when the
# loop is spending the most and producing the least. Run that unattended on a
# schedule and there is nothing in the system able to say stop.
#
# So: a real ceiling, on by default where money is at stake. Zero means no cap.
#
#   HIVE_MAX_USD        spend ceiling for the process. Defaults to $5 on the
#                       anthropic provider and to *unlimited* on ollama, where
#                       the tokens cost nothing and a cap would only surprise
#                       people.
#   HIVE_MAX_TOKENS     total (input + output) token ceiling. Off by default —
#                       it is the useful cap when the provider is free but the
#                       loop can still spin, and the supervisor sets one.
#   HIVE_MAX_WALL_SEC   wall-clock ceiling for the process.
#   HIVE_MAX_SPRINTS    how many sprints one supervisor run may execute.
#
# See core/governor.py — the ceilings are checked *before* each model call, not
# after. A check that fires once the tokens are already spent is an audit log,
# not a cap.

_DEFAULT_MAX_USD = "5.00" if PROVIDER == "anthropic" else "0"

MAX_USD = float(os.environ.get("HIVE_MAX_USD", _DEFAULT_MAX_USD))
MAX_TOKENS = int(os.environ.get("HIVE_MAX_TOKENS", "0"))
MAX_WALL_SEC = float(os.environ.get("HIVE_MAX_WALL_SEC", "0"))
MAX_SPRINTS = int(os.environ.get("HIVE_MAX_SPRINTS", "0"))

# ── Discovery ─────────────────────────────────────────────────────────────────
#
# How many times any one work item may be run — by anyone, human or loop — before
# discovery stops picking it up and PARKS it for a human.
#
# 2 is the deliberate default: the original attempt, plus exactly one machine
# retry on the tier that has not yet failed it. If the strong model escalates it
# too, the ladder is out of rungs and a third automated attempt is not a retry,
# it is a nodding loop with extra steps.
#
# The counter is what keeps the open door for human review from quietly closing.
MAX_DISCOVERY_ATTEMPTS = int(os.environ.get("HIVE_MAX_DISCOVERY_ATTEMPTS", "2"))

# ── Workspace layout ──────────────────────────────────────────────────────────

WORKSPACE_DIR = Path(os.environ.get("HIVE_WORKSPACE", "workspace")).resolve()

SRC_DIR = WORKSPACE_DIR / "src"
OUTPUTS_DIR = WORKSPACE_DIR / "outputs"

# SEC-C1: the path-traversal boundary. Nothing outside these two directories
# is writable by any node.
ALLOWED_DIRS: tuple[Path, ...] = (SRC_DIR, OUTPUTS_DIR)

LEDGER_FILE = OUTPUTS_DIR / "rejection_ledger.jsonl"
METRICS_FILE = OUTPUTS_DIR / "metrics.jsonl"
LOOP_MD_FILE = OUTPUTS_DIR / "LOOP.md"

# Append-only spend record. Unlike LEDGER_FILE this is NEVER cleared: it is the
# audit trail for what the loop cost, and a loop that forgets what it spent is
# the one that spends it again.
SPEND_FILE = OUTPUTS_DIR / "spend.jsonl"

# The cross-sprint journal. Also never cleared — that is the entire point of it.
#
# LEDGER_FILE is wiped at the start of every sprint (clear_ledger), which is right
# for what it is: the editor's memory of the mistakes it made on *this* task. But
# it means that until now nothing the hive learned outlived the sprint that
# learned it. The escalations human_gate_node so carefully records were deleted
# before the next run could ever read them.
#
# The journal is what survives. See core/journal.py.
JOURNAL_FILE = OUTPUTS_DIR / "journal.jsonl"


def ensure_workspace() -> None:
    """Creates the workspace skeleton. Idempotent; safe to call every sprint."""
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Sandbox environment ───────────────────────────────────────────────────────


def sandbox_env() -> dict[str, str]:
    """
    SEC-C2: the minimal environment handed to reviewer_node's subprocess.

    Host secrets are not inherited — only what CPython needs to start, plus
    the thread caps that keep OpenBLAS/OMP from saturating every core during
    a sprint that is already running local inference.

    The variable set is platform-dependent because the minimum viable
    environment is: POSIX needs PATH and HOME; Windows additionally needs
    SystemRoot (python.exe cannot load its DLLs without it) and a TEMP it can
    actually write to.
    """
    env: dict[str, str] = {
        "PYTHONPATH": str(WORKSPACE_DIR),
        "PYTHONIOENCODING": "utf-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    if os.name == "nt":
        # Windows upper-cases every key in os.environ, so SYSTEMROOT is the one
        # that is actually present — "SystemRoot" reads as absent and silently
        # falls back to the default.
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        temp_dir = str(OUTPUTS_DIR / ".sandbox_tmp")
        env.update(
            {
                # python.exe resolves its DLLs relative to its own directory,
                # but System32 must still be reachable for the CRT.
                "PATH": os.pathsep.join(
                    [
                        str(Path(sys.executable).parent),
                        str(Path(system_root) / "System32"),
                        system_root,
                    ]
                ),
                "SystemRoot": system_root,
                "TEMP": temp_dir,
                "TMP": temp_dir,
            }
        )
    else:
        env.update(
            {
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "HOME": str(OUTPUTS_DIR / ".sandbox_tmp"),
            }
        )

    return env
