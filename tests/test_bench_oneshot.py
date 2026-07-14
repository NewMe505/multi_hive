"""
The one-shot Claude baseline arm — the half of the cost comparison the `models`
suite used to refuse to measure on a hosted provider.

The load-bearing property is that this arm is METERED: its $/task must come from
the same governor that meters a sprint, or the "5-8x cheaper" comparison is
against a made-up number. These tests drive run_oneshot with a fake client that
records a known spend, and assert the bracket captures it — no API key, no
network, deterministic.
"""
import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from multi_hive.bench import runner
from multi_hive.bench.suite import TASKS
from multi_hive.core import governor

_BRACKETS_OK = """Here is the implementation:

```python
def is_balanced(s: str) -> bool:
    pairs = {')': '(', ']': '[', '}': '{'}
    opening = set(pairs.values())
    stack = []
    for ch in s:
        if ch in opening:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
```
"""


class _FakeLLM:
    """A metered client stand-in: records a known spend, returns fixed content."""

    def __init__(self, content: str, tokens: tuple[int, int] = (1000, 500)) -> None:
        self._content = content
        self._tokens = tokens

    async def ainvoke(self, _messages):
        # Simulate the meter the real llm_factory attaches, so spend_since has
        # something to see. fable-5 is priced (10, 50) per Mtok.
        governor.current().record("claude-fable-5", *self._tokens)
        return type("R", (), {"content": self._content})()


def _task(name: str):
    return next(t for t in TASKS if t.name == name)


def test_run_oneshot_meters_the_call_and_grades_the_code(monkeypatch):
    """
    The whole point of the arm: a real, same-governor cost figure attached to a
    graded one-shot. 1000 input + 500 output on fable-5 is $0.01 + $0.025 = $0.035.
    """
    governor.reset()
    monkeypatch.setattr(governor, "PROVIDER", "anthropic")  # price() bills only when billing
    fake = _FakeLLM(_BRACKETS_OK, tokens=(1000, 500))

    with patch("multi_hive.core.llm_factory.get_async_llm", lambda *a, **k: fake):
        result = asyncio.run(runner.run_oneshot("strong", _task("brackets")))

    assert result["passed"] is True, result["failure"]
    assert result["usd"] == 0.035, "the fable-5 rate must flow through the governor"
    assert result["input_tokens"] == 1000
    assert result["output_tokens"] == 500
    assert result["total_tokens"] == 1500
    assert result["unmetered"] == 0
    # A one-shot has exactly one attempt, and "did the first attempt pass" is then
    # the same question as "did it pass" — so the field is None, not a tautology.
    assert result["attempts"] == 1
    assert result["first_attempt_passed"] is None


def test_run_oneshot_records_spend_even_when_the_code_fails(monkeypatch):
    """
    A wrong answer is a failed task, not a lost measurement: the money was still
    spent, and the meter must still capture it — otherwise a run of failures would
    look free and flatter the baseline's cost.
    """
    governor.reset()
    monkeypatch.setattr(governor, "PROVIDER", "anthropic")
    fake = _FakeLLM("no code here, just prose", tokens=(800, 40))

    with patch("multi_hive.core.llm_factory.get_async_llm", lambda *a, **k: fake):
        result = asyncio.run(runner.run_oneshot("strong", _task("brackets")))

    assert result["passed"] is False
    assert result["failure"]
    assert result["total_tokens"] == 840
    assert result["usd"] > 0


def test_run_oneshot_extracts_code_from_fable_style_block_list():
    """
    fable-5 has thinking always on, so `.content` is a LIST of blocks (a thinking
    block plus the text block), not a string. The arm must flatten it — otherwise
    the fenced code is buried in a repr and every fable one-shot scores zero for a
    reason that is not the model's fault.
    """
    governor.reset()

    class _BlockListLLM:
        async def ainvoke(self, _messages):
            content = [
                {"type": "thinking", "thinking": "", "signature": "abc"},
                {"type": "text", "text": _BRACKETS_OK},
            ]
            return type("R", (), {"content": content})()

    with patch("multi_hive.core.llm_factory.get_async_llm", lambda *a, **k: _BlockListLLM()):
        result = asyncio.run(runner.run_oneshot("strong", _task("brackets")))

    assert result["passed"] is True, result["failure"]


def test_run_oneshot_survives_a_model_error_as_a_failed_task():
    """A transport/model error is scored as a failure, not raised — like run_model."""
    governor.reset()

    class _Boom:
        async def ainvoke(self, _messages):
            raise RuntimeError("overloaded_error")

    with patch("multi_hive.core.llm_factory.get_async_llm", lambda *a, **k: _Boom()):
        result = asyncio.run(runner.run_oneshot("strong", _task("brackets")))

    assert result["passed"] is False
    assert "overloaded_error" in result["failure"]
    assert result["attempts"] == 1


# ── The script-level plumbing (scripts/bench.py) ────────────────────────────────


def _load_bench():
    path = Path(__file__).resolve().parent.parent / "scripts" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_tier_accepts_a_tier_or_the_model_that_backs_it():
    bench = _load_bench()
    strong_model = bench.MODELS["strong"]

    assert bench._resolve_tier("strong") == "strong"
    assert bench._resolve_tier(strong_model) == "strong"

    with pytest.raises(SystemExit):
        bench._resolve_tier("not-a-real-model")


def test_record_oneshot_is_strict_and_subject_tagged():
    bench = _load_bench()

    results = {t.name: [] for t in TASKS}
    # brackets: passed 2 of 3 repeats — under the strict rule that is NOT a pass.
    results["brackets"] = [
        {"passed": True, "failure": "", "wall_sec": 1.0, "usd": 0.03,
         "total_tokens": 100, "input_tokens": 60, "output_tokens": 40},
        {"passed": True, "failure": "", "wall_sec": 1.2, "usd": 0.05,
         "total_tokens": 100, "input_tokens": 60, "output_tokens": 40},
        {"passed": False, "failure": "boom", "wall_sec": 1.1, "usd": 0.04,
         "total_tokens": 100, "input_tokens": 60, "output_tokens": 40},
    ]

    runs: list = []
    bench._record_oneshot(runs, "strong", results, 3)

    assert len(runs) == 1
    run = runs[0]
    assert run.subject == f"1shot:strong@{bench.PROVIDER}"
    assert run.repeat == 3

    bt = next(t for t in run.tasks if t["task"] == "brackets")
    assert bt["passed"] is False, "2 of 3 is not a strict pass"
    assert bt["pass_rate"] == round(2 / 3, 3)
    assert bt["usd"] == 0.04  # median of 0.03, 0.05, 0.04
    assert bt["first_attempt_passed"] is None


def test_record_oneshot_drops_the_interrupted_repeat():
    """
    A budget breach mid-run records only the repeats that completed. Passing
    complete_reps < len(results) must truncate, so tasks are never scored on uneven
    sample counts — the same rule bench_sprint applies.
    """
    bench = _load_bench()

    results = {t.name: [] for t in TASKS}
    results["brackets"] = [
        {"passed": True, "failure": "", "wall_sec": 1.0, "usd": 0.03,
         "total_tokens": 100, "input_tokens": 60, "output_tokens": 40},
        {"passed": True, "failure": "", "wall_sec": 1.2, "usd": 0.03,
         "total_tokens": 100, "input_tokens": 60, "output_tokens": 40},
        {"passed": False, "failure": "cut off", "wall_sec": 0.1, "usd": 0.0,
         "total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
    ]

    runs: list = []
    bench._record_oneshot(runs, "strong", results, 2)  # only 2 repeats finished

    run = runs[0]
    assert run.repeat == 2
    bt = next(t for t in run.tasks if t["task"] == "brackets")
    assert bt["repeats"] == 2, "the interrupted 3rd repeat must be dropped"
    assert bt["passed"] is True, "the two complete repeats both passed"


def test_record_oneshot_records_nothing_on_zero_complete_repeats():
    bench = _load_bench()
    results = {t.name: [] for t in TASKS}
    runs: list = []
    bench._record_oneshot(runs, "strong", results, 0)
    assert runs == []
