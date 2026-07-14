"""
runner.py — the two things we can benchmark.

run_model()  prompts a model directly, no graph. Answers "which model should
             back a tier?" Fast, isolated, and blind to everything the hive
             actually does.

run_sprint() drives the real graph end-to-end and grades the file that lands on
             disk. Answers "is the system getting better?" It is slower and
             noisier, and it is the one that counts: it exercises the planner,
             the ticket writer, both reviewers, the retry loop, and the
             escalation ladder — all the machinery a raw model prompt never
             touches. A change that improves the prompts and breaks the router
             looks fine to run_model() and terrible here.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from typing import Any

import requests

from multi_hive.bench.suite import Task, grade
from multi_hive.config import OUTPUTS_DIR, RECURSION_LIMIT, SRC_DIR, ensure_workspace
from multi_hive.contract import normalise_target
from multi_hive.core import governor
from multi_hive.core.memory import clear_ledger
from multi_hive.prompts import get_editor_prompt

OLLAMA = "http://127.0.0.1:11434"


def clean_workspace() -> None:
    """
    A benchmark task starts from an empty workspace and an empty ledger.

    Neither used to be true, and between them they corrupted every sprint number
    this project has ever recorded.

    The ledger
    ----------
    `clear_ledger()` had exactly one caller in the codebase — `cli.py` — so a
    benchmark run never cleared it. And `get_recent_rejections()` filters by NODE
    NAME ONLY: no task scoping, no run scoping. It returns the last three failures
    for that node *from the whole file*.

    So the `word_wrap` editor was handed `semver`'s traceback under the heading
    "PAST RUNTIME/ASSERTION FAILURES — your code ran but produced wrong results.
    Fix the logic" and told to fix a bug in a file it was not writing. The ledger
    was found 221 lines deep, spanning multiple tasks and multiple sessions.

    Three consequences, all of which invalidate the measurement:

    - **The suite was order-dependent.** Task 4 carried tasks 1-3's failures into
      its context. Task 1 carried none. Reordering TASKS changed the score.
    - **The repeats were not independent.** Run 2 began with run 1's failures in
      the editor's prompt — which voids the entire point of `passed == passed
      every run`, a rule this benchmark otherwise defends very carefully.
    - **A run was not reproducible from a clean checkout.** The score depended on
      ledger residue from whatever was run yesterday.

    And `run_model` never touches the ledger, so the whole penalty applied to
    `sprint` and never to `models` — a one-directional bias in precisely the
    comparison the two suites exist to support.

    The workspace
    -------------
    Only the graded artefact was unlinked. Everything else survived. The reviewer
    sandbox runs with `PYTHONPATH=WORKSPACE_DIR` and its cwd inside the workspace,
    so a module left behind by an earlier task could shadow an import and let a
    task pass on *yesterday's code*.

    What is deleted, and what is not
    --------------------------------
    Generated code is `*.py`. The hive's own records are not — `bench_history.jsonl`
    above all, which is the file the benchmark exists to write, and wiping it to
    clean the workspace would be an own goal of a high order. So this removes
    Python and bytecode and leaves every `.jsonl` / `.md` / `.json` alone.
    """
    ensure_workspace()

    for root in (SRC_DIR, OUTPUTS_DIR):
        # Materialise before mutating — rglob is lazy, and deleting out from under
        # it is how you get a walker that skips half the tree.
        for path in list(root.rglob("*.py")):
            path.unlink(missing_ok=True)
        for cache in list(root.rglob("__pycache__")):
            shutil.rmtree(cache, ignore_errors=True)

    clear_ledger()


def _extract_code(raw: str) -> str:
    fence = chr(96) * 3
    matches = re.findall(fence + r"python\n(.*?)\n" + fence, raw, re.DOTALL)
    return max(matches, key=len).strip() if matches else ""


_FILE_HEADER = re.compile(r"^[ \t]*#[ \t]*FILE:[ \t]*(\S+?\.py)[ \t]*$", re.MULTILINE)


def _extract_unfenced(raw: str, task: Task) -> dict[str, str]:
    """
    Split an unfenced, `# FILE:`-labelled response into its files.

    The 30B answers word_stats exactly this way — see _extract_files. Everything
    between one header and the next belongs to that file. A repeated header (it
    emits `# FILE: outputs/stats.py` twice, once for the module and once for a
    demo block) keeps the FIRST occurrence: the later one is a trailing
    `if __name__ == '__main__':` section, and appending it would be inventing code
    the model did not put in the module.
    """
    headers = list(_FILE_HEADER.finditer(raw))
    if not headers:
        return {}

    found: dict[str, str] = {}
    for i, header in enumerate(headers):
        # The FULL labelled path, not just the basename.
        #
        # Taking basename() meant `# FILE: src/tokens.py` was accepted as tokens.py.
        # The sprint suite refuses that path (normalise_model_path rejects src/ for an
        # outputs/ task), so the one-shot baseline was being held to a WEAKER standard
        # than the pipeline — on the exact axis the multi-file comparison exists to
        # test. Four earlier grader bugs favoured the pipeline; this one favoured the
        # baseline, and it is the same disease.
        labelled = header.group(1).lstrip("./")
        name = labelled[len("outputs/") :] if labelled.startswith("outputs/") else labelled
        if name not in task.files or name in found:
            continue
        end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        body = raw[header.end() : end].strip()
        if body:
            found[name] = body

    return found


def _extract_files(raw: str, task: Task) -> dict[str, str]:
    """
    Pull the task's files out of a single model response.

    Single-file tasks keep the old behaviour exactly: the longest fenced block.

    Multi-file tasks need to know which block is which. run_model ASKS for
    `# FILE: outputs/<name>`, but a benchmark must not score a model on whether it
    obeyed a formatting request — it must score whether it wrote the program.

    The first version of this only accepted that exact marker, and the 30B failed
    `word_stats` three times with "no code" while almost certainly having written
    both files, labelled the natural way (`# tokens.py`). That is a HARNESS
    ARTEFACT scored as a model failure, and it biases the multi-file comparison in
    favour of the pipeline — which is the conclusion this benchmark exists to test,
    and therefore the one it must never be allowed to manufacture.

    So the filename is looked for anywhere in the block's opening lines OR in the
    prose immediately before its fence — `# FILE: outputs/stats.py`, `# stats.py`,
    `**outputs/stats.py**`, `Here is stats.py:` all work.

    And it reads UNFENCED output, which is how the 30B actually answers this task.
    Asked for one fenced block per file, it emitted this instead:

        # FILE: outputs/tokens.py
        import re
        ...
        # FILE: outputs/stats.py
        from outputs.tokens import tokenize

    Perfectly labelled, both files, and not a code fence in sight — because the
    multi-file instruction ("one block per file") CONTRADICTS the editor system
    prompt it is given, which says "output the FULL updated script inside a single
    ```python block". The model resolved our own contradiction by dropping fences,
    and the extractor, which only looked inside fences, scored it "no code" three
    times out of three.

    That is the same harness bug as the paragraph above, in a new costume, and it
    failed in the same direction: flattering the pipeline. Twice is a pattern. A
    benchmark whose errors all favour the hypothesis is not a benchmark.

    What is still NOT done is guessing. A block with no filename anywhere near it is
    dropped, not assigned by position: mapping a block to the wrong file would score
    a coherent answer as a failure, and inventing a mapping is the kind of
    helpfulness that turns a benchmark into a story. A model that genuinely cannot
    produce two coherent files scores a real failure ("missing outputs/tokens.py").
    """
    fence = chr(96) * 3
    pattern = re.compile(fence + r"python\n(.*?)\n" + fence, re.DOTALL)

    matches = list(pattern.finditer(raw))

    if not matches:
        # No fences. If the response is labelled, read it anyway.
        return _extract_unfenced(raw, task) if len(task.files) > 1 else {}

    if len(task.files) == 1:
        return {task.filename: max((m.group(1) for m in matches), key=len).strip()}

    found: dict[str, str] = {}
    for match in matches:
        block = match.group(1)

        # The block's first few lines, plus the prose leading up to its fence —
        # models name the file in either place, and which one is not the model's
        # fault.
        preamble = raw[max(0, match.start() - 160) : match.start()]
        head = preamble + "\n".join(block.splitlines()[:4])

        # The filename mentioned FIRST wins. A stats.py block whose body says
        # "from tokens import ..." must not be filed as tokens.py just because the
        # other name appears somewhere in it — the label is the one at the top.
        hits = [(head.find(name), name) for name in task.files if name in head]
        if hits:
            found.setdefault(min(hits)[1], block.strip())

    return found


def gpu_placement(model: str) -> str:
    """How Ollama actually split the model between GPU and CPU, once resident."""
    try:
        for entry in requests.get(f"{OLLAMA}/api/ps", timeout=10).json().get("models", []):
            if model in (entry.get("name"), entry.get("model")):
                total, gpu = entry.get("size", 0), entry.get("size_vram", 0)
                if total:
                    return f"{round(100 * gpu / total)}% GPU ({gpu / 1e9:.1f}/{total / 1e9:.1f} GB)"
    except Exception:
        pass
    return "?"


def run_model(model: str, task: Task, num_ctx: int = 4096, num_predict: int = 2048) -> dict[str, Any]:
    """One task, one model, no graph."""
    targets = ", ".join(f"outputs/{name}" for name in task.files)

    prompt = task.prompt
    if len(task.files) > 1:
        # Give the one-shot baseline a fair, explicit way to emit more than one
        # file. Without it, a model that CAN do the task would still score zero for
        # want of a format, and the multi-file comparison — which exists precisely
        # to ask whether the pipeline beats one prompt — would be rigged.
        prompt += (
            "\n\nOutput ONE ```python code block per file, and begin each block "
            "with a comment naming the file, exactly like:\n"
            "# FILE: outputs/stats.py"
        )

    payload = {
        "model": model,
        "prompt": prompt,
        "system": get_editor_prompt(f"Write {targets}", ""),
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": num_ctx, "num_predict": num_predict},
    }

    started = time.perf_counter()
    chunks: list[str] = []
    final: dict = {}

    try:
        with requests.post(f"{OLLAMA}/api/generate", json=payload, stream=True, timeout=1800) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("response"):
                    chunks.append(event["response"])
                if event.get("done"):
                    final = event
    except Exception as e:
        return {
            "task": task.name,
            "complexity": task.complexity,
            "passed": False,
            "failure": f"{type(e).__name__}: {e}",
            "wall_sec": time.perf_counter() - started,
        }

    wall = time.perf_counter() - started
    tokens = final.get("eval_count", 0)
    prompt_tokens = final.get("prompt_eval_count", 0)
    eval_ns = final.get("eval_duration", 0)

    result = grade(_extract_files("".join(chunks), task), task)

    return {
        "task": task.name,
        "complexity": task.complexity,
        "passed": result.passed,
        "failure": result.failure,
        "wall_sec": round(wall, 2),
        "tok_per_sec": round(tokens / (eval_ns / 1e9), 1) if eval_ns and tokens else 0.0,
        "output_tokens": tokens,
        # Input tokens too, so the sprint suite and the models suite can be compared
        # on COST and not just on score. A pipeline that scores one task higher while
        # burning ten times the tokens has not obviously won, and until now the
        # benchmark had no way to say so.
        "input_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + tokens,
        "attempts": 1,  # one shot, by definition — the baseline for `attempts` below
        "gpu_placement": gpu_placement(model),
    }


async def run_sprint(task: Task, contract: str = "") -> dict[str, Any]:
    """
    One task, through the entire graph, graded on the file that lands on disk.

    Deliberately reads the artefact from the workspace rather than scraping the
    model's response: what the hive *wrote* is the product. If the editor
    generated perfect code and the reviewer failed to flush it, that is a
    failure, and it should be scored as one.

    `contract` supplies a human-written acceptance contract for the task's file,
    which is a different mode of operation and a different question. Without one,
    the bench asks "can the hive guess what I meant?". With one, it asks "when I
    say exactly what I want, does the hive deliver it?" — which is the question
    the feature was built to answer, and the only one a contract can be scored on.

    The two are NOT comparable, and history keeps them apart by subject. Grading
    is against the hidden suite either way, and the hidden suite never sees the
    contract: see bench/contracts.py for why that makes it a gaming detector.
    """
    from langchain_core.messages import HumanMessage

    from multi_hive.orchestrator import hive_app
    from multi_hive.state import default_loop_health

    # Empty workspace, empty ledger. Not a tidy-up — a correctness requirement.
    # See clean_workspace(): without it the suite is order-dependent, the repeats
    # are not independent samples, and the run is not reproducible.
    clean_workspace()

    target = f"outputs/{task.filename}"

    initial = {
        "messages": [HumanMessage(content=task.objective)],
        "project_files": {},
        "active_file": target,
        "task_queue": [],
        "current_task": None,
        "editor_error": None,
        "editor_retries": 0,
        "sprint_plan": "",
        "specialist_context": "",
        "is_ui_task": False,
        "loop_health": default_loop_health(),
        "semantic_verdict": None,
        "task_complexity": None,
        "model_tier": None,
        "contracts": {normalise_target(target): contract} if contract else {},
        "contract_satisfied": None,
        "sprint_started_at": time.monotonic(),
        "human_gate_event": None,  # headless: the gate must not wait for a human
    }

    started = time.perf_counter()
    nodes = 0
    tiers: list[str] = []
    escalated = False
    error: str | None = None
    contract_satisfied: bool | None = None

    # Bracket the sprint so we can say what it COST, not just what it scored.
    spend_before = governor.current().snapshot()

    # ── Every candidate the editor produced, graded ───────────────────────────
    #
    # The benchmark only ever graded the LAST file on disk. So it could not see the
    # single most important thing this pipeline does to itself:
    #
    #     the editor writes correct code -> a reviewer falsely rejects it
    #     -> a retry overwrites it with something worse -> the bench grades the loser
    #
    # That is not a hypothetical. async_editor_node clobbers project_files[active_file]
    # unconditionally and reviewer_node flushes it to disk, so a correct attempt #1 is
    # PHYSICALLY OVERWRITTEN by a worse attempt #2. Every false rejection is therefore
    # not merely wasted time — it is an active downgrade of the artefact, and the
    # score alone cannot distinguish "the model could not do it" from "the model did
    # it and we threw it away".
    #
    # So: grade each distinct version of the graded file as the editor emits it.
    # Cheap (a subprocess, ~0.3s, a handful per sprint) and it turns an invisible
    # failure mode into a number. Measure it BEFORE fixing it — otherwise there is no
    # way to know whether the fix paid.
    candidates: list[bool] = []
    seen: set[int] = set()

    def _grade_candidate(files: dict[str, str]) -> None:
        code = files.get(target, "")
        if not code.strip():
            return

        # A multi-file candidate is only gradeable once its siblings EXIST.
        #
        # The editor writes one file per ticket. So the first stats.py candidate is
        # produced while tokens.py may not be on disk yet, and grading it there
        # returns "missing outputs/tokens.py" — recording passed=False for a file
        # whose quality was never assessed. first_attempt_passed was therefore
        # SYSTEMATICALLY False for word_stats regardless of what the model wrote:
        # the metric built to illuminate the multi-file task was broken for exactly
        # that task.
        #
        # An incomplete system is not a failed attempt. It is not an attempt yet.
        payload = {task.filename: code}
        for name in task.extra_files:
            path = OUTPUTS_DIR / name
            sibling = path.read_text(encoding="utf-8") if path.exists() else ""
            if not sibling.strip():
                return  # not a candidate yet — the system is half-written
            payload[name] = sibling

        fingerprint = hash(tuple(sorted(payload.items())))
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        candidates.append(grade(payload, task).passed)

    try:
        async for output in hive_app.astream(
            initial, config={"recursion_limit": RECURSION_LIMIT}
        ):
            for _node, delta in output.items():
                nodes += 1
                delta = delta or {}
                if delta.get("project_files"):
                    _grade_candidate(delta["project_files"])
                if delta.get("model_tier") and delta["model_tier"] not in tiers:
                    tiers.append(delta["model_tier"])
                if "loop_health" in delta:
                    escalated = escalated or bool((delta["loop_health"] or {}).get("escalated"))
                # Sticky. agent_router_node resets loop_health when a sprint
                # continues past a gate to the next task, so the OR above can miss
                # an escalation that was followed by more work.
                escalated = escalated or bool(delta.get("sprint_escalated"))
                if "editor_error" in delta:
                    error = delta["editor_error"]
                if delta.get("contract_satisfied") is not None:
                    contract_satisfied = delta["contract_satisfied"]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    wall = time.perf_counter() - started

    # EVERY file the task asked for, not just the graded one. A multi-file task is
    # not done until all of them are on disk, and reading only the graded artefact
    # would let a sprint that wrote stats.py and silently skipped tokens.py be
    # graded as though it had produced a working system. It would not even fail
    # cleanly — it would fail at import, and be reported as broken code.
    files = {
        name: (OUTPUTS_DIR / name).read_text(encoding="utf-8")
        if (OUTPUTS_DIR / name).exists()
        else ""
        for name in task.files
    }
    result = grade(files, task)
    spend = governor.spend_since(spend_before)

    # The three numbers the score cannot tell you.
    #
    # attempts               — how many distinct files the editor produced. 1 means it
    #                          got it right (or wrong) first time; 3 means it thrashed.
    # first_attempt_passed   — did the very first thing it wrote already pass? A system
    #                          that passes after three retries is not the same system as
    #                          one that passes immediately, and the score cannot see the
    #                          difference.
    # discarded_a_pass       — THE one. The editor produced code that PASSES THE HIDDEN
    #                          SUITE, and the sprint then shipped something else. That is
    #                          the pipeline destroying a correct answer with its own
    #                          reviewers, and it is invisible to every metric this
    #                          benchmark had.
    passed_ever = any(candidates)

    return {
        "task": task.name,
        "complexity": task.complexity,
        "passed": result.passed,
        "failure": result.failure or (error or ""),
        "wall_sec": round(wall, 1),
        "nodes": nodes,
        "tiers": tiers,           # ["fast"] or ["fast", "strong"] — did it escalate?
        "escalated_to_human": escalated,
        "wrote_artefact": bool(files.get(task.filename)),
        "attempts": len(candidates),
        "first_attempt_passed": bool(candidates and candidates[0]),
        "discarded_a_pass": bool(passed_ever and not result.passed),
        "input_tokens": spend.get("input_tokens", 0),
        "output_tokens": spend.get("output_tokens", 0),
        "total_tokens": spend.get("total_tokens", 0),
        "usd": spend.get("usd", 0.0),
        # Calls the governor could not read. A cost figure computed from a broken
        # meter is worse than no cost figure: plausible, low, and wrong.
        "unmetered": spend.get("unmetered", 0),
        # The gaming detector. In contract mode, contract_satisfied=True with
        # passed=False means the code cleared the human's asserts and failed the
        # hidden ones — which is what memorising the contract's literal inputs
        # looks like from the outside. bench.py flags it loudly.
        "contract_satisfied": contract_satisfied,
    }
