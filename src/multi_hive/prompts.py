"""
prompts.py — every system prompt the hive sends, in one file.

Prompts are behaviour. Keeping them out of the nodes means a prompt change is
a one-file diff that can be reviewed on its own, rather than being buried in
control flow.
"""
from __future__ import annotations

_EDITOR_STATIC_PREFIX = (
    "Modify the Python code to fulfill the task.\n"
    "1. Output the FULL updated script inside a single ```python ... ``` block. "
    "No prose before or after the block.\n"
    "2. SAVE PATH RULE: Write the file to the EXACT path specified in the task. "
    "Do not invent a different filename or directory.\n"
    "3. IF CREATING UI (Tkinter/GUI): Separate logic into Controller and View classes. "
    "Logic must be testable without launching the window.\n"
)

# Used when an independent acceptance spec EXISTS. The editor implements against
# a contract it did not write and cannot change.
_AGAINST_SPEC_RULES = (
    "4. Do NOT write assert statements, tests, or an `if __name__ == '__main__':` "
    "test block. The acceptance criteria below were written from the task, not by "
    "you, and are executed against your code independently. Code that grades its "
    "own homework is not verified — write the implementation, nothing else.\n"
    "5. Your module must be safe to IMPORT: no input(), no infinite loops, no "
    "network calls, and no work at module level beyond definitions and constants.\n\n"
)

# Used ONLY when no usable acceptance spec could be produced.
#
# Self-authored asserts are a bad check — a wrong one rejects correct code — but
# they are a check, and the alternative here is none at all. Measured: with the
# spec missing and the asserts also removed, nothing was verifying the code, and
# the hive shipped a word-wrap with no hard-split straight through. A flawed test
# beats no test, so this is the fallback, not the default.
_SELF_TEST_RULES = (
    "4. The script MUST include an `if __name__ == '__main__':` block that asserts "
    "its own behaviour.\n"
    "5. ASSERT RULES — follow these exactly:\n"
    "   a. Only assert what is GUARANTEED by the task spec or by deterministic "
    "arithmetic from it. CHECK YOUR ARITHMETIC: an assertion that is itself wrong "
    "will reject your own correct code. Count the characters. Work the expected "
    "value out by hand.\n"
    "   b. Never assert the truthiness of a mutating call — a method like put() or "
    "append() returns None, so `assert c.put(1, 1)` fails against correct code. "
    "Compare a value to an expectation instead.\n"
    "   c. Never assert floating-point values that depend on phase or accumulated "
    "rounding; assert lengths, dtypes, and types instead.\n"
    "   d. Cover the edge cases the task calls out explicitly.\n\n"
)


def get_sprint_planner_prompt() -> str:
    return (
        "You are a Software Architect.\n"
        "Draft a sequential implementation plan.\n"
        "1. MAXIMUM 4 STEPS.\n"
        "2. PURE SOFTWARE LOGIC.\n"
    )


def get_ticket_writer_prompt() -> str:
    return (
        "You are the Ticket Writer. Translate the sprint plan into a task queue.\n"
        "RULES:\n"
        "1. Output ONLY a valid JSON list. No prose, no markdown, no explanation.\n"
        "2. Each object must have exactly two keys: 'file' and 'task'.\n"
        "3. FILE PATH RULE: If the User Objective specifies an explicit output path "
        "(e.g. 'Save it to outputs/dsp_pipeline.py'), you MUST use that exact path "
        "in the 'file' field. Do not invent a different filename.\n"
        "4. All file paths must start with 'src/' or 'outputs/'.\n"
        'Example: [{"file": "outputs/dsp_pipeline.py", "task": "Implement DSP module"}, '
        '{"file": "src/main.py", "task": "Import and run"}]'
    )


def get_semantic_reviewer_prompt(sprint_plan: str, current_task: str) -> str:
    """
    Adversarial semantic review prompt.

    Adversarial framing catches semantic divergence, but the NEVER REJECT rules
    are load-bearing: without them a 7B model invents spurious complaints about
    correct code — rejecting np.float64 as a "wrong dtype" when it is exactly
    what np.max() returns.
    """
    return (
        "You are a code reviewer checking ONLY whether the code implements "
        "what the task asked for.\n"
        "You are NOT checking style, efficiency, or best practices.\n\n"
        f"SPRINT PLAN CONTEXT:\n{sprint_plan}\n\n"
        f"SPECIFIC TASK THAT WAS ASSIGNED:\n{current_task}\n\n"
        "REVIEW RULES:\n"
        "1. Read the task. Identify every EXPLICIT requirement.\n"
        "2. Check the code implements each explicit requirement.\n"
        "3. Respond with exactly PASS or FAIL: <one specific reason>. Nothing else.\n\n"
        "NEVER REJECT for any of the following — these are always correct:\n"
        "- numpy scalar types (np.float64, np.float32, np.int64) — these are "
        "  the expected return types of numpy operations like np.max(), np.sum().\n"
        "- standard Python numeric types (float, int) returned from math operations.\n"
        "- the presence of extra helper functions beyond what the task specifies.\n"
        "- coding style, variable names, or implementation approach.\n"
        "- assert statements or test code in the execution block.\n\n"
        "ONLY REJECT if the code is MISSING something the task EXPLICITLY requires:\n"
        "- a function that was explicitly named in the task is absent\n"
        "- a file path specified in the task is not used\n"
        "- a required behaviour (e.g. delay effect) is completely absent\n\n"
        "Examples of correct FAIL responses:\n"
        "  FAIL: apply_delay function is missing, only generate_sine_wave is implemented\n"
        "  FAIL: file saved to src/ but task specified outputs/dsp_pipeline.py\n"
        "  FAIL: compute_max_amplitude function is absent from the code\n\n"
        "Examples of WRONG FAIL responses (do NOT do these):\n"
        "  FAIL: Output dtype is incorrect  <- WRONG, np.float64 is correct for np.max()\n"
        "  FAIL: function should use a different algorithm  <- WRONG, style not checked\n"
        "  FAIL: missing docstrings  <- WRONG, not an explicit requirement\n\n"
        "CODE TO REVIEW:\n"
    )


def get_spec_writer_prompt(global_objective: str, current_task: str) -> str:
    """
    Derives the acceptance criteria from the TASK, before any code exists.

    This exists because the editor used to write its own asserts, and a model
    that grades its own homework fails in both directions. A wrong assert
    rejects correct code — observed: the model asserted
    `wrap_text("hello world", 10) == ["hello world"]`, but "hello world" is 11
    characters and cannot fit in a width of 10, so its own test rejected its own
    correct implementation, burned the retry budget, escalated the tier, and
    woke a human. A lazy assert does the reverse and waves broken code through.

    Writing the spec first, from the task alone, removes both. The spec cannot
    be rationalised around an implementation that does not exist yet, and it
    stops moving between retries.
    """
    return (
        "You write ACCEPTANCE TESTS. You do not write implementations.\n\n"
        f"PROJECT OBJECTIVE:\n{global_objective}\n\n"
        f"THE TASK TO BE IMPLEMENTED:\n{current_task}\n\n"
        "Output a short Python test SCRIPT: plain statements, one per line, run in "
        "order, sharing state. Nothing else — no prose, no code fences, no imports, "
        "no function definitions, no comments, no print().\n\n"
        "Only two kinds of line are allowed:\n"
        "  - a variable assignment, to set something up      e.g.  c = LRUCache(2)\n"
        "  - an `assert` comparing a value to an expectation e.g.  assert c.get(1) == 1\n\n"
        "RULES:\n"
        "1. At most 12 lines, of which at least 3 are asserts.\n"
        "2. Use the EXACT function and class names the task specifies.\n"
        "3. Every `assert` must COMPARE something: use ==, !=, <, >, `is`, or `in`. "
        "Never assert a bare call. A method that mutates state (put, add, append) "
        "returns None, so `assert c.put(1, 1)` fails against a perfectly correct "
        "implementation. Call it on its own line instead.\n"
        "4. State persists down the script. Build the object ONCE and then act on "
        "it. Do not rebuild it on every line — `assert LRUCache(2).get(1) == 1` "
        "asks a brand-new empty cache for a key nobody ever put in it.\n"
        "5. Every expectation must follow from the task text alone. If the task does "
        "not determine the answer, do not assert it.\n"
        "6. CHECK YOUR ARITHMETIC on every line. A wrong expectation rejects a "
        "correct implementation. Count the characters. Work the value out by hand. "
        "If you are not certain, leave the line out.\n"
        "7. Test behaviour, not internals: nothing about private attributes or how "
        "the answer is computed.\n\n"
        "Example — task: 'implement an LRUCache class with get and put, evicting the "
        "least recently used entry':\n"
        "c = LRUCache(2)\n"
        "c.put(1, 1)\n"
        "c.put(2, 2)\n"
        "assert c.get(1) == 1\n"
        "c.put(3, 3)\n"
        "assert c.get(2) == -1\n"
        "assert c.get(3) == 3\n"
    )


def get_assert_adjudicator_prompt(current_task: str, assertion: str, code: str) -> str:
    """
    An acceptance test failed. Whose fault is it — the test, or the code?

    Defaults to CODE_WRONG on any doubt, and that bias is deliberate. Calling a
    valid assertion "wrong" deletes a real check and lets a bug through, which is
    the worse error; calling a wrong assertion "code's fault" merely reproduces
    the behaviour we already had. The safe failure mode is the status quo.
    """
    return (
        "An acceptance test failed against an implementation. Decide which of the "
        "two is at fault.\n\n"
        f"THE TASK:\n{current_task}\n\n"
        f"THE FAILING ASSERTION:\n{assertion}\n\n"
        f"THE IMPLEMENTATION:\n{code[:3000]}\n\n"
        "Work out, from the task alone, what the correct expected value actually is. "
        "Then compare it to what the assertion expects.\n\n"
        "Answer with exactly one of:\n"
        "  CODE_WRONG: <one line>    — the assertion is right, the implementation is wrong\n"
        "  ASSERT_WRONG: <one line>  — the assertion contradicts the task; the "
        "expected value in it is simply not what the task requires\n\n"
        "Choose ASSERT_WRONG only when you can state plainly what the assertion "
        "should have expected instead, and why the task demands that. If you are "
        "unsure, or if the assertion is a reasonable reading of the task, answer "
        "CODE_WRONG.\n"
    )


def get_editor_prompt(
    global_objective: str,
    specialist_context: str,
    past_gen_failures: str = "",
    past_runtime_failures: str = "",
    past_semantic_failures: str = "",
    acceptance: list[str] | None = None,
) -> str:
    """
    Builds the editor system prompt.

    Three failure feeds, each labelled distinctly, because each implies a
    different fix:
      past_gen_failures      — code could not be produced or extracted.
                               Fix the structure.
      past_runtime_failures  — code ran but failed an assert or crashed.
                               Fix the logic.
      past_semantic_failures — code ran fine but wasn't what was asked for.
                               Re-read the task.

    Merging them produces incoherent retries: the model doesn't know whether to
    change the structure, the logic, or the intent.
    """
    # Which rules apply depends on whether an independent spec exists. Something
    # must always be checking the code: with a spec, the spec does it; without one,
    # the model's own asserts do, flawed as they are.
    prompt = (
        _EDITOR_STATIC_PREFIX
        + (_AGAINST_SPEC_RULES if acceptance else _SELF_TEST_RULES)
        + f"GLOBAL RULES:\n{global_objective}\n\n"
        + f"{specialist_context}\n"
    )

    if acceptance:
        criteria = "\n".join(acceptance)
        prompt += (
            "\nACCEPTANCE CRITERIA — this exact script will be run against your code. "
            "It was written from the task, not by you, and you cannot change it. "
            "Make it pass:\n"
            f"{criteria}\n"
        )

    if past_gen_failures:
        prompt += (
            "\nPAST GENERATION FAILURES — your code could not be produced or extracted. "
            "Fix the code structure:\n"
            f"{past_gen_failures}\n"
        )
    if past_runtime_failures:
        prompt += (
            "\nPAST RUNTIME/ASSERTION FAILURES — your code ran but produced wrong results "
            "or a failing assert. Fix the logic, not the structure:\n"
            f"{past_runtime_failures}\n"
        )
    if past_semantic_failures:
        prompt += (
            "\nPAST SEMANTIC REVIEW FAILURES — your code did not implement what the task "
            "asked for. Re-read the task carefully and fix the missing or wrong behaviour:\n"
            f"{past_semantic_failures}\n"
        )

    return prompt
