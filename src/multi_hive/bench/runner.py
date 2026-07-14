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


# "# FILE: outputs/stats.py" — how a one-shot response labels which block is which.
_FILE_MARK = re.compile(r"#\s*FILE:\s*(?:outputs/)?([\w.\-]+\.py)", re.IGNORECASE)


def _extract_files(raw: str, task: Task) -> dict[str, str]:
    """
    Pull the task's files out of a single model response.

    Single-file tasks keep the old behaviour exactly: the longest fenced block.

    Multi-file tasks need to know which block is which, so run_model asks the model
    to label each with `# FILE: outputs/<name>`. An unlabelled block is DROPPED
    rather than guessed at: assigning a block to the wrong file would score a
    coherent answer as a failure, and inventing a mapping is the sort of
    helpfulness that turns a benchmark into a story.

    A one-shot model that genuinely cannot produce two coherent files therefore
    scores a real failure ("missing outputs/tokens.py"), not a harness artefact.
    That has to hold for the multi-file comparison to mean anything at all — the
    entire question is whether the pipeline beats one prompt, and the answer is
    worthless if the prompt was handicapped by the grader.
    """
    fence = chr(96) * 3
    blocks = re.findall(fence + r"python\n(.*?)\n" + fence, raw, re.DOTALL)
    if not blocks:
        return {}

    if len(task.files) == 1:
        return {task.filename: max(blocks, key=len).strip()}

    found: dict[str, str] = {}
    for block in blocks:
        mark = _FILE_MARK.search(block[:300])
        if not mark:
            continue
        name = mark.group(1)
        if name in task.files:
            found[name] = block.strip()

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

    try:
        async for output in hive_app.astream(
            initial, config={"recursion_limit": RECURSION_LIMIT}
        ):
            for _node, delta in output.items():
                nodes += 1
                delta = delta or {}
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
        # The gaming detector. In contract mode, contract_satisfied=True with
        # passed=False means the code cleared the human's asserts and failed the
        # hidden ones — which is what memorising the contract's literal inputs
        # looks like from the outside. bench.py flags it loudly.
        "contract_satisfied": contract_satisfied,
    }
