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
