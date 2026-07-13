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
import time
from typing import Any

import requests

from multi_hive.bench.suite import Task, grade
from multi_hive.config import OUTPUTS_DIR, RECURSION_LIMIT, ensure_workspace
from multi_hive.contract import normalise_target
from multi_hive.prompts import get_editor_prompt

OLLAMA = "http://127.0.0.1:11434"


def _extract_code(raw: str) -> str:
    fence = chr(96) * 3
    matches = re.findall(fence + r"python\n(.*?)\n" + fence, raw, re.DOTALL)
    return max(matches, key=len).strip() if matches else ""


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
    payload = {
        "model": model,
        "prompt": task.prompt,
        "system": get_editor_prompt(f"Write outputs/{task.filename}", ""),
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

    result = grade(_extract_code("".join(chunks)), task)

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

    ensure_workspace()

    artefact = OUTPUTS_DIR / task.filename
    if artefact.exists():
        artefact.unlink()  # never grade a previous run's leftovers

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
                if "editor_error" in delta:
                    error = delta["editor_error"]
                if delta.get("contract_satisfied") is not None:
                    contract_satisfied = delta["contract_satisfied"]
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    wall = time.perf_counter() - started

    code = artefact.read_text(encoding="utf-8") if artefact.exists() else ""
    result = grade(code, task)

    return {
        "task": task.name,
        "complexity": task.complexity,
        "passed": result.passed,
        "failure": result.failure or (error or ""),
        "wall_sec": round(wall, 1),
        "nodes": nodes,
        "tiers": tiers,           # ["fast"] or ["fast", "strong"] — did it escalate?
        "escalated_to_human": escalated,
        "wrote_artefact": bool(code),
        # The gaming detector. In contract mode, contract_satisfied=True with
        # passed=False means the code cleared the human's asserts and failed the
        # hidden ones — which is what memorising the contract's literal inputs
        # looks like from the outside. bench.py flags it loudly.
        "contract_satisfied": contract_satisfied,
    }
