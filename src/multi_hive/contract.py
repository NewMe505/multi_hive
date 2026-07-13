"""
contract.py — the human-supplied acceptance contract.

The problem this exists to solve
--------------------------------
The editor writes the implementation AND the asserts that judge it. That is a
conflict of interest with a measurable cost. Observed live: the model asserted

    assert wrap_text("hello world", 10) == ["hello world"]

11 characters into a width of 10 — impossible by the task's own definition. Its
implementation was correct; its assert was not. So correct code was rejected,
the retry budget burned, the tier escalated to the strong model, and a human was
woken up. Nothing was wrong with the program.

An earlier attempt had a second model write the asserts instead (spec_writer).
It was measured and reverted: 3.11x slower for zero quality gain, and the 7B
still could not produce a usable spec for the task that needed one most. The
model is not bad at *writing* the asserts; it is bad at *knowing what is true*.
No amount of model plumbing fixes that, because the missing information — what
"correct" means — is not in the model.

It is in the human. So the human supplies it.

The format
----------
An objective may carry one or more ACCEPTANCE blocks:

    Implement wrap_text(text, width) which greedily wraps text into lines of at
    most `width` characters. A word longer than `width` is hard-split.
    Save it to outputs/wrap.py

    ACCEPTANCE outputs/wrap.py
    assert wrap_text("one two three", 7) == ["one two", "three"]
    assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]

A block runs from its `ACCEPTANCE <path>` header to the next header or the end
of the text. The path names the file the contract judges; a bare `ACCEPTANCE`
with no path applies to whatever file the task produces, which is the common
single-file case.

What changes when a contract is present
---------------------------------------
1. The editor is told not to write asserts at all — the contract is the test.
2. reviewer_node imports the generated module and executes the contract against
   it, instead of running the model's own script. A violated assert is a real
   failure with a real error message, so the retry loop is finally being fed
   ground truth rather than the model's opinion of itself.
3. semantic_reviewer_node stands down. A human-written executable contract is a
   strictly stronger check than a 7B model asked "is this the right program?",
   and that reviewer is the other source of false rejections. Skipping it also
   removes an LLM call per attempt.

Anti-gaming
-----------
A contract the model can see is a contract the model can hardcode against. The
editor prompt forbids special-casing the contract's literals, but a prompt is
not a guarantee — so the benchmark checks. bench/contracts.py deliberately uses
DIFFERENT literal values than the hidden test suites, which makes the hidden
tests a working gaming detector: code that memorises the contract's inputs
passes the contract and fails the bench. See bench/contracts.py.
"""
from __future__ import annotations

import re

# `ACCEPTANCE outputs/wrap.py`, `acceptance: outputs/wrap.py`, or bare `ACCEPTANCE`.
_HEADER = re.compile(r"^[ \t]*ACCEPTANCE[ \t]*:?[ \t]*(?P<path>\S+)?[ \t]*$", re.IGNORECASE)

# A fenced block inside a contract body is stripped: humans paste ```python.
_FENCE = re.compile(r"^[ \t]*`{3,}[a-zA-Z]*[ \t]*$")

ANY_FILE = "*"


class ContractError(ValueError):
    """A contract was supplied but is not usable. Raised at parse time, on purpose.

    A malformed contract must fail loudly at the prompt, where the human who
    wrote it is still standing there, rather than surfacing forty seconds later
    as an inscrutable sprint failure they will blame on the model.
    """


def normalise_target(path: str) -> str:
    """
    Canonical key for a contract target.

    The header path is typed by a human ("outputs\\wrap.py", "./outputs/wrap.py")
    and active_file is emitted by a model ("outputs/wrap.py"). They have to meet
    somewhere, so both go through this.
    """
    cleaned = (path or "").strip().strip("\"'").replace("\\", "/")
    # Only a leading "./", not a leading "." — ".config/x.py" keeps its dot.
    cleaned = re.sub(r"^(?:\./)+", "", cleaned)
    return cleaned.lower() or ANY_FILE


def parse_objective(text: str) -> tuple[str, dict[str, str]]:
    """
    Split a raw objective into (objective, contracts).

    The returned objective has the ACCEPTANCE blocks removed: the sprint planner
    and the ticket writer must not see them. They plan work, and a contract is
    not work — a planner shown one cheerfully emits "Step 3: write the asserts",
    which is the exact thing being taken away from it.

    The editor *does* see the contract, injected by name for the file it is
    currently writing. See prompts.get_editor_prompt.
    """
    lines = (text or "").splitlines()

    objective_lines: list[str] = []
    contracts: dict[str, list[str]] = {}
    current: list[str] | None = None

    for line in lines:
        header = _HEADER.match(line)
        if header:
            target = normalise_target(header.group("path") or ANY_FILE)
            current = contracts.setdefault(target, [])
            continue

        if current is None:
            objective_lines.append(line)
        elif not _FENCE.match(line):
            current.append(line)

    parsed: dict[str, str] = {}
    for target, body_lines in contracts.items():
        body = "\n".join(body_lines).strip()
        if not body:
            raise ContractError(f"ACCEPTANCE block for {target!r} is empty.")
        try:
            compile(body, "<acceptance>", "exec")
        except SyntaxError as e:
            raise ContractError(
                f"ACCEPTANCE block for {target!r} is not valid Python "
                f"(line {e.lineno}): {e.msg}"
            ) from e
        parsed[target] = body

    return "\n".join(objective_lines).strip(), parsed


def contract_for(contracts: dict[str, str], active_file: str | None) -> str:
    """
    The contract governing `active_file`, or "" if none.

    Falls back to the bare ANY_FILE contract, so a single-file objective does not
    have to repeat a path the human already wrote once in the objective itself.
    """
    if not contracts:
        return ""
    if active_file:
        exact = contracts.get(normalise_target(active_file))
        if exact:
            return exact
    return contracts.get(ANY_FILE, "")


def assert_count(body: str) -> int:
    """How many asserts a contract makes. Cosmetic — for the console line."""
    return sum(1 for line in body.splitlines() if line.strip().startswith("assert"))


# ── The harness ───────────────────────────────────────────────────────────────
#
# The generated module is IMPORTED, not executed as a script. Two consequences,
# both wanted:
#
#   - __name__ is "candidate", so any `if __name__ == "__main__":` block the
#     model wrote anyway does not run. The contract is the only thing that
#     judges the code, even if the editor ignored the prompt and wrote asserts.
#   - the module's public names are lifted into the harness globals, so the human
#     writes `wrap_text(...)` and not `module.wrap_text(...)`. A contract should
#     read like the thing it is asserting about.
#
# Exit codes are distinct because the three failures mean different things to the
# editor: the module did not import (broken code), a name was missing (wrong API),
# or an assert was false (wrong behaviour).
#
# The exit code is NOT the ground truth for a pass — the final sentinel line is.
# A module that calls sys.exit()/os._exit()/exit() at import time terminates the
# harness with returncode 0 *before any assert runs*, and returncode 0 alone would
# read that as a satisfied contract with zero asserts executed. So:
#   - the import is guarded with `except BaseException`, which (unlike `Exception`)
#     catches SystemExit, turning a top-level sys.exit() into a clean IMPORT_FAIL;
#   - the pass is confirmed only by "CONTRACT_PASS <token>", printed on the last
#     line after every assert survived. The token is a per-run nonce the module
#     cannot know, so the pass signal cannot be forged even by a module that prints
#     "CONTRACT_PASS" itself or calls os._exit(0) (which no `except` can catch).

_HARNESS = '''\
import importlib.util
import sys
import traceback

__spec_ = importlib.util.spec_from_file_location("candidate", r"@MODULE@")
__mod_ = importlib.util.module_from_spec(__spec_)
try:
    __spec_.loader.exec_module(__mod_)
except BaseException:
    print("CONTRACT_IMPORT_FAIL")
    traceback.print_exc()
    sys.exit(2)

M = __mod_
globals().update({k: v for k, v in vars(__mod_).items() if not k.startswith("_")})

try:
@BODY@
except AssertionError:
    print("CONTRACT_FAIL")
    traceback.print_exc()
    sys.exit(3)
except NameError as e:
    print("CONTRACT_MISSING")
    print(f"{e}  <- the contract calls a name your module does not define")
    sys.exit(4)
except Exception:
    print("CONTRACT_ERROR")
    traceback.print_exc()
    sys.exit(5)

print("CONTRACT_PASS @TOKEN@")
'''


def pass_marker(token: str) -> str:
    """The exact final line a passing harness prints. The consumer checks for this."""
    return f"CONTRACT_PASS {token}"


def render_harness(module_path: str, body: str, token: str) -> str:
    """
    The runnable harness script for `body` against the module at `module_path`.

    `token` is a per-run nonce printed on the pass line. The caller confirms a
    pass by finding pass_marker(token) in the output — never by the exit code
    alone, which a top-level exit() in the generated code can forge. See the
    _HARNESS comment.
    """
    indented = "\n".join("    " + line for line in body.splitlines())
    return (
        _HARNESS.replace("@MODULE@", str(module_path))
        .replace("@BODY@", indented)
        .replace("@TOKEN@", token)
    )
