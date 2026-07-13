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

# ── Model / loop tuning ───────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("HIVE_MODEL", "qwen2.5-coder:7b")
MAX_RETRIES = int(os.environ.get("HIVE_MAX_RETRIES", "3"))

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
