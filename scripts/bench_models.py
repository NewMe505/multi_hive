"""
bench_models.py — measure candidate models on THIS machine.

    python scripts/bench_models.py
    python scripts/bench_models.py --models qwen2.5-coder:7b qwen3-coder:30b
    python scripts/bench_models.py --json workspace/outputs/bench.json

Model choice for the escalation ladder cannot be argued from public
leaderboards. It depends on how much of a given model fits in *this* GPU's VRAM,
and on whether its output survives real scrutiny. So we measure both.

How quality is judged
---------------------
Each task ships a **hidden test suite** that the model never sees. The model is
given the task description only; the generated code is then imported and run
against tests written to probe the edge cases the task quietly implies —
capacity-zero caches, touching intervals, prerelease version ordering.

This replaces an earlier version that only checked "does it compile, does it
run, does it contain the required function names". Every model scored full marks
on that, which measured nothing: compiling and running is a floor, not a
ranking. A model that confidently emits clean, running, *subtly wrong* code is
exactly the failure the escalation ladder exists to catch, and symbol-presence
gates are blind to it.

Reported per (model, task):

  speed    tok/s, time to first token, total wall time
  fit      how much of the model Ollama placed on the GPU vs CPU
  quality  extracted -> compiles -> imports -> hidden tests passed (n/total)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Imported for its side effect: reconfigures stdout to UTF-8 before anything
# prints. This script emits ✓/✗, and Windows stdio is cp1252 by default.
import multi_hive.core.console  # noqa: E402, F401
from multi_hive.prompts import get_editor_prompt  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"

DEFAULT_MODELS = ["qwen2.5-coder:7b", "qwen3-coder:30b"]


@dataclass
class Task:
    name: str
    complexity: str
    objective: str
    prompt: str
    tests: str  # hidden from the model; run against its output


TASKS = [
    Task(
        name="lru_cache",
        complexity="moderate",
        objective="Write outputs/lru.py",
        prompt=(
            "Implement an LRUCache class with O(1) get and put, using a dict plus a "
            "doubly linked list. Do NOT use collections.OrderedDict or functools.lru_cache.\n"
            "  __init__(self, capacity: int)\n"
            "  get(self, key) -> int, returning -1 if the key is absent\n"
            "  put(self, key, value) -> None, evicting the least recently used entry "
            "when over capacity\n"
            "Reading a key with get() counts as using it."
        ),
        # The trap: does get() actually refresh recency, and is an overwrite of an
        # existing key treated as a use? Both are easy to get wrong and neither
        # shows up in a naive smoke test.
        tests="""
c = M.LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1                 # 1 is now most-recently used
c.put(3, 3)                          # so 2 must be the victim, not 1
assert c.get(2) == -1, "get() did not refresh recency"
assert c.get(1) == 1 and c.get(3) == 3

c = M.LRUCache(2)
c.put(1, 1); c.put(2, 2); c.put(1, 10)   # overwrite is a use
c.put(3, 3)
assert c.get(2) == -1, "overwriting an existing key did not refresh recency"
assert c.get(1) == 10, "overwrite did not update the value"

assert M.LRUCache(1).get(99) == -1        # empty cache
""",
    ),
    Task(
        name="merge_intervals",
        complexity="moderate",
        objective="Write outputs/intervals.py",
        prompt=(
            "Implement merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]] "
            "which merges all overlapping intervals and returns them sorted by start.\n"
            "The input may be unsorted. Intervals that merely touch (e.g. (1,3) and (3,5)) "
            "count as overlapping and must be merged."
        ),
        # The traps: unsorted input, touching-but-not-overlapping intervals, and
        # a fully-contained interval that must not extend the enclosing one.
        tests="""
assert M.merge_intervals([]) == []
assert M.merge_intervals([(1, 3)]) == [(1, 3)]

got = M.merge_intervals([(1, 3), (2, 6), (8, 10), (15, 18)])
assert got == [(1, 6), (8, 10), (15, 18)], got

got = M.merge_intervals([(8, 10), (1, 3), (2, 6)])          # unsorted
assert got == [(1, 6), (8, 10)], got

got = M.merge_intervals([(1, 3), (3, 5)])                    # touching
assert got == [(1, 5)], f"touching intervals not merged: {got}"

got = M.merge_intervals([(1, 10), (2, 4)])                   # contained
assert got == [(1, 10)], f"contained interval broke the range: {got}"
""",
    ),
    Task(
        name="semver",
        complexity="hard",
        objective="Write outputs/semver.py",
        prompt=(
            "Implement compare_semver(a: str, b: str) -> int returning -1 if a < b, "
            "0 if equal, 1 if a > b, following the Semantic Versioning 2.0.0 precedence "
            "rules.\n"
            "Handle versions of the form MAJOR.MINOR.PATCH with an optional "
            "-prerelease and an optional +build metadata suffix."
        ),
        # The traps, all of them explicitly in the semver spec and all of them
        # commonly missed: build metadata is ignored entirely; a prerelease is
        # LOWER than its release; numeric prerelease identifiers compare
        # numerically, not as strings (so 2 < 10); and more identifiers wins.
        tests="""
assert M.compare_semver("1.0.0", "2.0.0") == -1
assert M.compare_semver("2.0.0", "1.0.0") == 1
assert M.compare_semver("1.2.3", "1.2.3") == 0

assert M.compare_semver("1.0.0+build1", "1.0.0+build2") == 0, "build metadata must be ignored"
assert M.compare_semver("1.0.0-alpha", "1.0.0") == -1, "prerelease must sort below its release"
assert M.compare_semver("1.0.0-alpha", "1.0.0-beta") == -1

assert M.compare_semver("1.0.0-alpha.2", "1.0.0-alpha.10") == -1, \\
    "numeric prerelease identifiers must compare numerically, not as strings"
assert M.compare_semver("1.0.0-alpha", "1.0.0-alpha.1") == -1, \\
    "more prerelease identifiers wins when the prefix is equal"
assert M.compare_semver("1.0.0-alpha.1", "1.0.0-alpha.beta") == -1, \\
    "numeric identifiers sort below alphanumeric ones"
""",
    ),
    Task(
        name="word_wrap",
        complexity="hard",
        objective="Write outputs/wrap.py",
        prompt=(
            "Implement wrap_text(text: str, width: int) -> list[str] which greedily wraps "
            "text into lines of at most `width` characters, breaking only on spaces.\n"
            "A single word longer than `width` must be hard-split across lines. "
            "Runs of multiple spaces collapse. No line may be empty, and no line may "
            "have leading or trailing spaces."
        ),
        # The traps: the oversized word (most models forget the hard split), the
        # exact-width boundary (off-by-one), and collapsing whitespace.
        tests="""
assert M.wrap_text("", 10) == []
assert M.wrap_text("hello world", 20) == ["hello world"]
assert M.wrap_text("hello world", 5) == ["hello", "world"]

got = M.wrap_text("aaa bbb ccc", 7)
assert got == ["aaa bbb", "ccc"], got

got = M.wrap_text("abcdefghij", 4)                  # single oversized word
assert got == ["abcd", "efgh", "ij"], f"oversized word not hard-split: {got}"

got = M.wrap_text("hi     there", 12)               # collapsing whitespace
assert got == ["hi there"], got

for line in M.wrap_text("the quick brown fox jumps over the lazy dog", 9):
    assert line == line.strip() and line, f"bad line {line!r}"
    assert len(line) <= 9, f"line over width: {line!r}"
""",
    ),
]


@dataclass
class Result:
    model: str
    task: str
    complexity: str
    ok: bool = False
    error: str = ""

    tok_per_sec: float = 0.0
    first_token_sec: float = 0.0
    total_sec: float = 0.0
    output_tokens: int = 0
    load_sec: float = 0.0
    gpu_placement: str = "?"

    extracted: bool = False
    compiles: bool = False
    imports: bool = False
    tests_passed: bool = False
    failure: str = ""

    notes: list[str] = field(default_factory=list)


def extract_code(raw: str) -> str:
    fence = chr(96) * 3
    matches = re.findall(fence + r"python\n(.*?)\n" + fence, raw, re.DOTALL)
    return max(matches, key=len).strip() if matches else ""


HARNESS = """
import importlib.util, sys, traceback
spec = importlib.util.spec_from_file_location("candidate", r"{module}")
M = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(M)
except Exception:
    print("IMPORT_FAIL"); traceback.print_exc(); sys.exit(2)
try:
{body}
except AssertionError as e:
    print("TEST_FAIL"); print(str(e)[:200]); sys.exit(3)
except Exception as e:
    print("TEST_ERROR"); print(f"{{type(e).__name__}}: {{str(e)[:160]}}"); sys.exit(4)
print("PASS")
"""


def judge(code: str, task: Task) -> tuple[bool, bool, bool, str]:
    """(compiles, imports, tests_passed, failure) — runs the hidden suite."""
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "candidate.py"
        module.write_text(code, encoding="utf-8")

        compiled = subprocess.run(
            [sys.executable, "-m", "py_compile", str(module)],
            capture_output=True,
            text=True,
        )
        if compiled.returncode != 0:
            return False, False, False, "syntax error"

        body = "\n".join("    " + line for line in task.tests.strip().splitlines())
        harness = Path(tmp) / "harness.py"
        harness.write_text(
            HARNESS.format(module=str(module), body=body), encoding="utf-8"
        )

        try:
            run = subprocess.run(
                [sys.executable, str(harness)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return True, False, False, "timeout"

        out = (run.stdout or "") + (run.stderr or "")

        if run.returncode == 0:
            return True, True, True, ""
        if "IMPORT_FAIL" in out:
            last = [x for x in out.strip().splitlines() if x.strip()]
            return True, False, False, (last[-1][:70] if last else "import failed")

        detail = ""
        lines = [x for x in (run.stdout or "").strip().splitlines() if x.strip()]
        if len(lines) > 1:
            detail = lines[1][:70]
        return True, True, False, detail or "assertion failed"


def run_one(model: str, task: Task, num_ctx: int, num_predict: int) -> Result:
    result = Result(model=model, task=task.name, complexity=task.complexity)

    payload = {
        "model": model,
        "prompt": task.prompt,
        "system": get_editor_prompt(task.objective, ""),
        "stream": True,
        "options": {"temperature": 0.1, "num_ctx": num_ctx, "num_predict": num_predict},
    }

    started = time.perf_counter()
    first_at: float | None = None
    chunks: list[str] = []
    final: dict = {}

    try:
        with requests.post(
            f"{OLLAMA}/api/generate", json=payload, stream=True, timeout=1800
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if event.get("response"):
                    if first_at is None:
                        first_at = time.perf_counter()
                    chunks.append(event["response"])
                if event.get("done"):
                    final = event
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        return result

    result.total_sec = time.perf_counter() - started
    result.first_token_sec = (first_at - started) if first_at else 0.0
    result.output_tokens = final.get("eval_count", 0)
    result.load_sec = final.get("load_duration", 0) / 1e9

    eval_ns = final.get("eval_duration", 0)
    if eval_ns and result.output_tokens:
        result.tok_per_sec = result.output_tokens / (eval_ns / 1e9)

    code = extract_code("".join(chunks))
    result.extracted = bool(code)

    if not code:
        result.failure = "no ```python block"
    else:
        result.compiles, result.imports, result.tests_passed, result.failure = judge(code, task)

    result.ok = True
    return result


def gpu_placement(model: str) -> str:
    try:
        running = requests.get(f"{OLLAMA}/api/ps", timeout=10).json()
        for entry in running.get("models", []):
            if model in (entry.get("name"), entry.get("model")):
                total, gpu = entry.get("size", 0), entry.get("size_vram", 0)
                if total:
                    return f"{round(100 * gpu / total)}% GPU ({gpu / 1e9:.1f}/{total / 1e9:.1f} GB)"
    except Exception:
        pass
    return "?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2048)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    results: list[Result] = []

    for model in args.models:
        print(f"\n=== {model} ===", flush=True)
        for task in TASKS:
            print(f"  {task.name:16} ({task.complexity:8}) ... ", end="", flush=True)
            result = run_one(model, task, args.num_ctx, args.num_predict)
            result.gpu_placement = gpu_placement(model)

            if result.error:
                print(f"ERROR {result.error}")
            else:
                mark = "✓ pass" if result.tests_passed else "✗ FAIL"
                why = f"  ({result.failure})" if result.failure else ""
                print(
                    f"{result.tok_per_sec:5.1f} tok/s  {result.total_sec:6.1f}s  {mark}{why}"
                )
            results.append(result)

    print("\n" + "=" * 84)
    print(f"{'model':22} {'tok/s':>7} {'avg s':>7} {'hidden tests':>14}  {'placement':<22}")
    print("-" * 84)

    for model in args.models:
        rows = [r for r in results if r.model == model and r.ok]
        if not rows:
            print(f"{model:22} {'—':>7} {'—':>7} {'—':>14}  (all runs failed)")
            continue
        speed = sum(r.tok_per_sec for r in rows) / len(rows)
        wall = sum(r.total_sec for r in rows) / len(rows)
        passed = sum(r.tests_passed for r in rows)
        placement = next((r.gpu_placement for r in rows if r.gpu_placement != "?"), "?")
        print(
            f"{model:22} {speed:7.1f} {wall:7.1f} {passed:>7}/{len(rows):<6} {placement:<22}"
        )

    print("\nhidden tests = suites the model never saw, probing the edge cases the task implies")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
