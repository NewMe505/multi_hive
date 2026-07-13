"""
reviewer_node — execution verification.

Writes the generated file to the workspace, compiles it, then runs it in a
sandboxed subprocess. Passing means the code both parses and survives its own
asserts. Whether it is the *right* program is semantic_reviewer_node's problem.

Sandboxing is strongest on POSIX, where core.platform supplies RLIMIT ceilings
via preexec_fn. Windows has no fork and therefore no preexec_fn, so there the
subprocess is bounded by timeout and a minimal environment only — see
core/platform.py. On both platforms the environment is stripped of host
secrets and the process cannot write outside the workspace.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from multi_hive import prompts
from multi_hive.config import (
    OUTPUTS_DIR,
    SANDBOX_TIMEOUT_SEC,
    SPEC_REPAIR_LIMIT,
    WORKSPACE_DIR,
    sandbox_env,
)
from multi_hive.core.llm_factory import DEFAULT_TIER, get_llm, invalidate_llm
from multi_hive.core.memory import log_rejection
from multi_hive.core.platform import confine, release, sandbox_preexec
from multi_hive.core.utils import flush_file, safe_path

_MAX_OUTPUT_CHARS = 65536
_MAX_TRACEBACK_CHARS = 1500


def _executes(loop_health: Any) -> dict[str, Any]:
    """
    The code ran. That is ALL this node is entitled to say.

    It deliberately does not advance the task queue, clear current_task, or
    reset editor_retries. Declaring the task finished here is what broke the
    loop: semantic_reviewer_node runs *after* this node, so a task marked
    complete on execution alone could still be rejected on intent — and by then
    current_task was None (so the editor regenerated nothing) and editor_retries
    was 0 (so MAX_RETRIES was unreachable). The sprint then cycled forever,
    re-validating identical code. Observed in the wild: 992 identical semantic
    rejections, zero escalations.

    A task is finished when BOTH reviewers pass, and only semantic_reviewer_node
    is in a position to know that. Advancement lives there now.
    """
    return {"editor_error": None, "loop_health": loop_health}


def _fail(state: dict[str, Any], loop_health: Any, error_msg: str) -> dict[str, Any]:
    log_rejection("reviewer_node", error_msg)
    return {
        "editor_error": error_msg,
        "editor_retries": state.get("editor_retries", 0) + 1,
        "loop_health": loop_health,
    }


# ── Verification ──────────────────────────────────────────────────────────────

_OK = "ok"
_IMPL_BROKEN = "impl_broken"
_ASSERTION_FAILED = "assertion_failed"

# The harness imports the implementation as a module and runs each acceptance
# assertion in isolation, so the failing one can be named exactly.
#
# It IMPORTS rather than executes as a script: __name__ is not "__main__", so any
# test block the editor left behind does not run. The code is judged against the
# spec, and only against the spec.
_HARNESS = '''
import importlib.util, sys, traceback

_spec = importlib.util.spec_from_file_location("_impl", r"{impl}")
_mod = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_mod)
except Exception:
    print("IMPL_BROKEN")
    traceback.print_exc()
    sys.exit(2)

globals().update({{k: v for k, v in vars(_mod).items() if not k.startswith("_")}})

_ASSERTIONS = {assertions!r}

for _i, _src in enumerate(_ASSERTIONS):
    try:
        exec(_src, globals())
    except AssertionError:
        print("ASSERTION_FAILED")
        print(_i)
        print(_src)
        sys.exit(3)
    except Exception as _e:
        print("ASSERTION_FAILED")
        print(_i)
        print(_src)
        print("{{}}: {{}}".format(type(_e).__name__, _e))
        sys.exit(3)

print("OK")
'''


def _verify(impl_path: Path, acceptance: list[str]) -> tuple[str, str, int]:
    """
    Run the implementation against the acceptance criteria, in the sandbox.

    Returns (outcome, detail, failing_index).

    With no acceptance criteria — the spec writer produced nothing usable — this
    falls back to executing the file as a script, which runs the `__main__` block
    the editor is told to write in exactly that case. Weaker, and it is the model
    grading itself, but SOMETHING has to check the code.

    That fallback is not theoretical caution. When the spec was missing and the
    self-asserts had also been taken away, nothing was verifying the code at all,
    and the hive shipped a word-wrap with no hard-split. Measured on the sprint
    benchmark: 3/4 -> 1/4. A flawed check beats no check.
    """
    if acceptance:
        target = OUTPUTS_DIR / ".verify" / f"{impl_path.stem}_check.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            _HARNESS.format(impl=str(impl_path), assertions=acceptance),
            encoding="utf-8",
        )
    else:
        target = impl_path  # run it as a script; its own asserts are all we have

    # Two halves of one sandbox: preexec_fn applies RLIMITs between fork and exec
    # on POSIX; confine() assigns the child to a Job Object on Windows, which has
    # no fork. See core/platform.py for what each actually enforces.
    proc = subprocess.Popen(
        [sys.executable, str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=sandbox_env(),
        cwd=str(WORKSPACE_DIR),
        preexec_fn=sandbox_preexec(),
    )
    job = confine(proc.pid)

    try:
        out_bytes, _ = proc.communicate(timeout=SANDBOX_TIMEOUT_SEC)
        output = out_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
        code = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        out_bytes, _ = proc.communicate()
        output = (
            out_bytes.decode("utf-8", errors="replace")[:_MAX_OUTPUT_CHARS]
            + f"\nTIMEOUT: Execution exceeded {SANDBOX_TIMEOUT_SEC}s."
        )
        code = -1
    finally:
        # KILL_ON_JOB_CLOSE: releasing before communicate() returns would kill a
        # perfectly healthy run.
        release(job)

    if code == 0:
        return _OK, output, -1

    if output.startswith("ASSERTION_FAILED"):
        lines = output.splitlines()
        try:
            index = int(lines[1])
        except (IndexError, ValueError):
            index = -1
        return _ASSERTION_FAILED, output, index

    return _IMPL_BROKEN, output, -1


def _assertion_is_wrong(
    current_task: str,
    assertion: str,
    code: str,
    tier: str,
    can_repair: bool,
) -> bool:
    """
    An acceptance assertion failed. Is the assertion itself wrong?

    Biased hard toward "no". Ruling a valid assertion wrong deletes a real check
    and lets a bug through; ruling a wrong assertion "the code's fault" merely
    reproduces the behaviour we already had. Every ambiguous case therefore
    resolves to the safe answer, which is the status quo.

    Any failure here — a dead Ollama, an unparseable verdict — also resolves to
    "no". A broken adjudicator must never be able to delete the spec.
    """
    if not can_repair:
        return False

    try:
        llm = get_llm("reviewer", tier)  # temperature 0
        verdict = llm.invoke(
            [
                SystemMessage(
                    content=prompts.get_assert_adjudicator_prompt(current_task, assertion, code)
                ),
                HumanMessage(content=assertion),
            ]
        ).content.strip()
    except Exception as e:
        invalidate_llm("reviewer", tier)
        log_rejection("reviewer_node", f"ADJUDICATION FAILED (assuming code is at fault): {e}")
        return False

    return verdict.upper().startswith("ASSERT_WRONG")


def reviewer_node(state: dict[str, Any]) -> dict[str, Any]:
    active_file = state.get("active_file")
    if not active_file:
        return {}

    current_code = state.get("project_files", {}).get(active_file, "")
    if not current_code:
        return {}

    # Pass-through only — reviewer_node never modifies loop_health.
    loop_health = state.get("loop_health")

    # ── Write through the validated path ─────────────────────────────────────
    try:
        impl_path = flush_file(safe_path(active_file), current_code)
    except Exception as e:
        return _fail(state, loop_health, f"FILE SYSTEM ERROR: {e}")

    # ── Syntax check ──────────────────────────────────────────────────────────
    syntax_check = subprocess.run(
        [sys.executable, "-m", "py_compile", str(impl_path)],
        capture_output=True,
        text=True,
    )
    if syntax_check.returncode != 0:
        return _fail(state, loop_health, "SYNTAX ERROR:\n" + syntax_check.stderr)

    # ── UI tasks skip execution: the window would block forever ──────────────
    if state.get("is_ui_task"):
        return _executes(loop_health)

    # ── Verification, against a spec the implementer did not write ────────────
    acceptance = list(state.get("acceptance") or [])
    repairs = state.get("spec_repairs", 0)
    tier = state.get("model_tier") or DEFAULT_TIER

    # A task may be adjudicated at most SPEC_REPAIR_LIMIT times in total, so the
    # loop below is bounded whatever the model says.
    while True:
        outcome, detail, index = _verify(impl_path, acceptance)

        if outcome == _OK:
            delta = _executes(loop_health)
            if acceptance != (state.get("acceptance") or []):
                delta["acceptance"] = acceptance
                delta["spec_repairs"] = repairs
            return delta

        if outcome == _IMPL_BROKEN:
            # The module could not even be imported, or it timed out. That is the
            # code's fault regardless of what the spec says — nothing to adjudicate.
            return _fail(state, loop_health, f"TRACEBACK:\n{detail[-_MAX_TRACEBACK_CHARS:]}")

        # An acceptance assertion failed. Before blaming the code, ask whether the
        # assertion is even right — because the whole reason this spec exists is
        # that a wrong assertion once rejected a correct implementation.
        assertion = acceptance[index] if 0 <= index < len(acceptance) else "?"

        # Only an `assert` may ever be dropped. A setup line is not a claim about
        # the code — if `c = LRUCache(2)` blows up, that is the constructor's
        # fault — and removing one would silently break every assert after it.
        remaining_asserts = sum(1 for s in acceptance if s.startswith("assert "))
        can_repair = (
            assertion.startswith("assert ")
            and repairs < SPEC_REPAIR_LIMIT
            and remaining_asserts > 1  # never drop the last check standing
        )

        if _assertion_is_wrong(current_task=state.get("current_task") or "",
                               assertion=assertion,
                               code=current_code,
                               tier=tier,
                               can_repair=can_repair):
            log_rejection(
                "reviewer_node",
                f"SPEC REPAIR: the acceptance assertion contradicted the task and was "
                f"dropped, NOT the code. Removed: {assertion!r} "
                f"(repair {repairs + 1}/{SPEC_REPAIR_LIMIT})",
            )
            acceptance.pop(index)
            repairs += 1
            continue  # re-verify the same code against the corrected spec

        return _fail(
            state,
            loop_health,
            f"ACCEPTANCE FAILURE — this assertion, written from the task, does not "
            f"hold against your code:\n  {assertion}\n\n{detail[-_MAX_TRACEBACK_CHARS:]}",
        )
