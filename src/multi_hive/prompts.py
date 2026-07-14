"""
prompts.py — every system prompt the hive sends, in one file.

Prompts are behaviour. Keeping them out of the nodes means a prompt change is
a one-file diff that can be reviewed on its own, rather than being buried in
control flow.
"""
from __future__ import annotations

_EDITOR_CONTRACT_PREFIX = (
    "Modify the Python code to fulfill the task.\n"
    "1. Output the FULL updated script inside a single ```python ... ``` block. "
    "No prose before or after the block.\n"
    "2. An ACCEPTANCE CONTRACT is supplied below. A human wrote it. It is the ONLY "
    "thing your code will be judged on.\n"
    "3. DO NOT write asserts, tests, example usage, or an "
    "`if __name__ == '__main__':` block. Your module will be IMPORTED and the "
    "contract executed against it — any test code you write will never run.\n"
    "4. Every name the contract calls must exist at module level, spelled exactly "
    "as the contract spells it.\n"
    "5. DO NOT special-case the contract's literal inputs. Hardcoding the expected "
    "answers is a failure: your code will also be run on inputs that are not in "
    "the contract, and it must be correct on those too.\n"
    "6. SAVE PATH RULE: Write the file to the EXACT path specified in the task. "
    "   Do not invent a different filename or directory.\n"
    "7. IF CREATING UI (Tkinter/GUI): Separate logic into Controller and View classes. "
    "   Logic must be testable without launching the window.\n\n"
)

_EDITOR_STATIC_PREFIX = (
    "Modify the Python code to fulfill the task.\n"
    "1. Output the FULL updated script inside a single ```python ... ``` block. "
    "No prose before or after the block.\n"
    "2. The script MUST include an `if __name__ == '__main__':` execution block.\n"
    "3. ASSERT RULES — follow these exactly, no exceptions:\n"
    "   a. Only assert things that are GUARANTEED TRUE by the laws of the language or "
    "      by deterministic arithmetic from the task spec (e.g. array length from sample "
    "      rate, dtype from numpy constructor). NEVER assert floating-point values that "
    "      depend on a waveform's phase at a specific sample index — these are not "
    "      guaranteed and will cause AssertionError on correct code.\n"
    "   b. Always assert: output array length == exact integer from spec, "
    "      output dtype == expected numpy dtype, return type == expected Python type.\n"
    "   c. NEVER assert: specific float values inside arrays, last-element values, "
    "      min/max of continuous signals, or anything involving np.allclose on a "
    "      phase-dependent sample.\n"
    "4. SAVE PATH RULE: Write the file to the EXACT path specified in the task. "
    "   Do not invent a different filename or directory.\n"
    "5. IF CREATING UI (Tkinter/GUI): Separate logic into Controller and View classes. "
    "   Logic must be testable without launching the window.\n\n"
)


def get_sprint_planner_prompt() -> str:
    """
    "MAXIMUM 4 STEPS" was pressuring a decomposition out of objectives that did not
    have one. Asked for a single LRU cache in a single file, the planner produced
    four steps, the ticket writer produced four tickets, and all four named the
    same file — four full rewrites of one artefact, each re-running both reviewers.

    A cap is not a target. The plan should be as short as the work is, and the
    common case (one function, one file) is one step.

    ticket_writer._collapse_by_file enforces the one-ticket-per-file half of this
    in code, because a prompt is not a guarantee. This is the other half: stop
    asking for the split in the first place.
    """
    return (
        "You are a Software Architect.\n"
        "Draft a sequential implementation plan.\n"
        "1. USE AS FEW STEPS AS THE WORK NEEDS. If the objective produces ONE file, "
        "the plan is ONE step. Do not split a single function into separate steps "
        "for its parts.\n"
        "2. AT MOST 4 STEPS. This is a ceiling, not a target.\n"
        "3. PURE SOFTWARE LOGIC.\n"
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
        "4. Every 'file' MUST start with 'src/' or 'outputs/'. A bare filename "
        "like 'test_add.py' is NOT a valid path — write 'outputs/test_add.py'. "
        "A ticket naming any other location is discarded and its work is lost.\n"
        'Example: [{"file": "outputs/dsp_pipeline.py", "task": "Implement DSP module"}, '
        '{"file": "src/main.py", "task": "Import and run"}]'
    )


def get_semantic_reviewer_prompt(objective: str, sprint_plan: str, current_task: str) -> str:
    """
    Adversarial semantic review prompt.

    Adversarial framing catches semantic divergence, but the NEVER REJECT rules
    are load-bearing: without them a 7B model invents spurious complaints about
    correct code — rejecting np.float64 as a "wrong dtype" when it is exactly
    what np.max() returns.

    `objective` is new, and its absence was a real bug. This reviewer used to be
    handed only the sprint plan and the ticket — a 7B's summary, and a 7B's
    paraphrase of that summary. The human's actual words were never given to it.

    So it was asked "does this code do what was asked?" while holding a document
    that was NOT what was asked. Code that correctly implemented the human's spec
    but exceeded the lossy ticket got FAILed — and that verdict is injected as
    editor_error, burns a retry, escalates the tier, and can overwrite correct code
    with worse code.

    The objective goes first and is named the authority, for the same reason it goes
    LAST in the editor's prompt (see async_editor_node): it is the only text in this
    system that is actually true.
    """
    return (
        "You are a code reviewer checking ONLY whether the code implements "
        "what the task asked for.\n"
        "You are NOT checking style, efficiency, or best practices.\n\n"
        f"THE REQUIREMENT, AS THE HUMAN WROTE IT — this is the authority:\n"
        f"{objective}\n\n"
        f"SPRINT PLAN CONTEXT (a summary; the requirement above outranks it):\n"
        f"{sprint_plan}\n\n"
        f"SPECIFIC TASK ASSIGNED (which PART of the work — not what correct means):\n"
        f"{current_task}\n\n"
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
        # The save-path rule — "a file path specified in the task is not used", and
        # its worked example "FAIL: file saved to src/ but task specified
        # outputs/dsp_pipeline.py" — has been REMOVED, and its absence is
        # load-bearing.
        #
        # This reviewer's user message is ONLY the code. semantic_reviewer_node
        # sends [SystemMessage(prompt), HumanMessage(current_code)] and nothing
        # else — it cannot see active_file and never could. So it was being handed
        # a rule it had no way to evaluate, plus a worked example of how to fail on
        # it, and the only way to comply was to guess. That is not a review, it is
        # an invitation to hallucinate.
        #
        # The path is not its job in any case: normalise_model_path() is the entry
        # boundary and safe_path() is the write boundary. Both are deterministic,
        # both run in code, and neither wants a 7B's opinion. Every false rejection
        # from this node burns a retry, escalates the tier, and can overwrite
        # correct code with worse code — so a rule it cannot evaluate is not a
        # harmless one.
        "ONLY REJECT if the code is MISSING something the task EXPLICITLY requires:\n"
        "- a function that was explicitly named in the task is absent\n"
        "- a required behaviour (e.g. delay effect) is completely absent\n\n"
        "You are shown the CODE ONLY. You cannot see what file it was written to, "
        "so NEVER reject on a filename or save path — you have no way to know, and "
        "guessing is a false rejection.\n\n"
        "Examples of correct FAIL responses:\n"
        "  FAIL: apply_delay function is missing, only generate_sine_wave is implemented\n"
        "  FAIL: compute_max_amplitude function is absent from the code\n\n"
        "Examples of WRONG FAIL responses (do NOT do these):\n"
        "  FAIL: Output dtype is incorrect  <- WRONG, np.float64 is correct for np.max()\n"
        "  FAIL: function should use a different algorithm  <- WRONG, style not checked\n"
        "  FAIL: missing docstrings  <- WRONG, not an explicit requirement\n"
        "  FAIL: file saved to the wrong path  <- WRONG, you cannot see the path\n\n"
        "CODE TO REVIEW:\n"
    )


def get_editor_prompt(
    global_objective: str,
    specialist_context: str,
    past_gen_failures: str = "",
    past_runtime_failures: str = "",
    past_semantic_failures: str = "",
    acceptance_contract: str = "",
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

    acceptance_contract switches the prompt into contract mode: the model is told
    to write no asserts at all, because a human already wrote the ones that count.
    The ASSERT RULES in _EDITOR_STATIC_PREFIX are damage control for the case
    where nobody did — a long list of things the model must not assert about,
    accumulated one false rejection at a time. They help. They do not solve it:
    the model still cannot reliably tell a true statement about its own code from
    a false one. A contract removes the question instead of refining it.
    """
    prefix = _EDITOR_CONTRACT_PREFIX if acceptance_contract else _EDITOR_STATIC_PREFIX

    prompt = prefix + f"GLOBAL RULES:\n{global_objective}\n\n" + f"{specialist_context}\n"

    if acceptance_contract:
        fence = chr(96) * 3
        prompt += (
            "\nACCEPTANCE CONTRACT — your code is judged on this and nothing else. "
            "Every line of it must pass:\n"
            f"{fence}python\n{acceptance_contract}\n{fence}\n"
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
