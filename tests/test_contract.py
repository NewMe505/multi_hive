"""
The human-supplied acceptance contract.

The bug this feature exists to kill, in one line: the editor asserted

    assert wrap_text("hello world", 10) == ["hello world"]

— eleven characters into a width of ten — and then rejected its own correct code
for failing it, burned the retry budget, escalated to the strong model, and woke
a human. The model is judge and defendant, and it is not good at either job.

These tests hold the line on all four places that could quietly hand the job back
to the model: parsing, the harness, the reviewer, and the editor prompt. Plus one
that keeps the benchmark honest.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from multi_hive.contract import (
    ANY_FILE,
    ContractError,
    assert_count,
    contract_for,
    normalise_target,
    parse_objective,
    render_harness,
)

# ── Parsing ───────────────────────────────────────────────────────────────────


def test_objective_without_a_contract_is_untouched():
    objective, contracts = parse_objective("Build a thing.\nSave it to outputs/x.py")

    assert contracts == {}
    assert objective == "Build a thing.\nSave it to outputs/x.py"


def test_contract_is_lifted_out_of_the_objective():
    objective, contracts = parse_objective(
        "Implement add(a, b).\n"
        "Save it to outputs/add.py\n"
        "\n"
        "ACCEPTANCE outputs/add.py\n"
        "assert add(2, 2) == 4\n"
        "assert add(-1, 1) == 0\n"
    )

    # The planner must never see the contract: shown one, it plans work to
    # satisfy it ("Step 3: write the tests"), which is the exact job the contract
    # exists to take away from the model.
    assert "ACCEPTANCE" not in objective
    assert "assert" not in objective
    assert objective == "Implement add(a, b).\nSave it to outputs/add.py"

    assert contracts == {"outputs/add.py": "assert add(2, 2) == 4\nassert add(-1, 1) == 0"}
    assert assert_count(contracts["outputs/add.py"]) == 2


def test_bare_acceptance_header_applies_to_any_file():
    _, contracts = parse_objective("Do it.\nACCEPTANCE\nassert f(1) == 1\n")

    assert set(contracts) == {ANY_FILE}


def test_multiple_contracts_are_keyed_by_file():
    _, contracts = parse_objective(
        "Do two things.\n"
        "ACCEPTANCE outputs/a.py\n"
        "assert a() == 1\n"
        "ACCEPTANCE outputs/b.py\n"
        "assert b() == 2\n"
    )

    assert contracts == {"outputs/a.py": "assert a() == 1", "outputs/b.py": "assert b() == 2"}


def test_a_fenced_contract_body_is_unwrapped():
    # Humans paste ```python. They should not be punished for it.
    _, contracts = parse_objective(
        "Do it.\nACCEPTANCE outputs/a.py\n```python\nassert a() == 1\n```\n"
    )

    assert contracts == {"outputs/a.py": "assert a() == 1"}


def test_a_contract_that_does_not_compile_is_rejected_at_parse_time():
    # Loudly, at the prompt, while the human who typed it is still standing
    # there — not forty seconds into a doomed sprint.
    with pytest.raises(ContractError, match="not valid Python"):
        parse_objective("Do it.\nACCEPTANCE outputs/a.py\nassert a( == 1\n")


def test_an_empty_contract_is_rejected():
    with pytest.raises(ContractError, match="empty"):
        parse_objective("Do it.\nACCEPTANCE outputs/a.py\n")


# ── Path matching ─────────────────────────────────────────────────────────────


def test_human_paths_and_model_paths_meet_in_the_middle():
    # The header is typed by a human; active_file is emitted by a model.
    assert normalise_target(r"outputs\wrap.py") == "outputs/wrap.py"
    assert normalise_target("./outputs/wrap.py") == "outputs/wrap.py"
    assert normalise_target(" 'Outputs/Wrap.py' ") == "outputs/wrap.py"


def test_contract_lookup_prefers_an_exact_match_then_falls_back_to_wildcard():
    contracts = {"outputs/a.py": "assert a()", ANY_FILE: "assert anything()"}

    assert contract_for(contracts, "outputs/a.py") == "assert a()"
    assert contract_for(contracts, r"outputs\a.py") == "assert a()"
    assert contract_for(contracts, "outputs/other.py") == "assert anything()"
    assert contract_for({}, "outputs/a.py") == ""


# ── The harness ───────────────────────────────────────────────────────────────


_TEST_TOKEN = "tok_test_sentinel_0001"


def _run_contract(module_src: str, contract: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as tmp:
        module = Path(tmp) / "candidate.py"
        module.write_text(module_src, encoding="utf-8")

        harness = Path(tmp) / "harness.py"
        harness.write_text(
            render_harness(str(module), contract, _TEST_TOKEN), encoding="utf-8"
        )

        proc = subprocess.run(
            [sys.executable, str(harness)], capture_output=True, text=True, timeout=60
        )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_correct_code_satisfies_its_contract():
    code, out = _run_contract("def add(a, b):\n    return a + b\n", "assert add(2, 2) == 4")

    assert code == 0
    assert "CONTRACT_PASS" in out


def test_wrong_code_violates_its_contract():
    code, out = _run_contract("def add(a, b):\n    return a * b\n", "assert add(2, 3) == 5")

    assert code == 3
    assert "CONTRACT_FAIL" in out


def test_a_missing_function_is_reported_as_a_missing_name_not_a_crash():
    # The editor needs to know it got the API wrong, not just that "something
    # broke". Different exit code, different message, different fix.
    code, out = _run_contract("def sum_of(a, b):\n    return a + b\n", "assert add(2, 2) == 4")

    assert code == 4
    assert "CONTRACT_MISSING" in out


def test_a_module_that_does_not_import_is_reported_before_any_assert_runs():
    code, out = _run_contract("import nonexistent_module_xyz\n", "assert True")

    assert code == 2
    assert "CONTRACT_IMPORT_FAIL" in out


def test_a_top_level_sys_exit_at_import_is_caught_as_an_import_failure():
    # `except Exception` cannot catch SystemExit. Before the harness caught
    # BaseException, a module running sys.exit(0) at import exited 0 before any
    # assert ran — and exit-0 alone was read as a satisfied contract with zero
    # asserts executed. It must be an import failure instead. (Audit finding #1.)
    code, out = _run_contract("import sys\nsys.exit(0)\n", "assert True")

    assert code == 2
    assert "CONTRACT_IMPORT_FAIL" in out


def test_a_passing_run_carries_the_per_run_nonce():
    # The pass is confirmed by "CONTRACT_PASS <token>", not the exit code. The
    # token is why a module cannot forge a pass by printing the sentinel itself.
    code, out = _run_contract("def add(a, b):\n    return a + b\n", "assert add(2, 2) == 4")

    assert code == 0
    assert _TEST_TOKEN in out


def test_the_models_own_asserts_do_not_run_under_a_contract():
    # The module is imported, not executed, so __name__ is "candidate" and the
    # model's self-authored test block is dead code. This is the whole mechanism:
    # even if the editor ignores the prompt and writes asserts anyway, they
    # cannot fail the build, and they cannot reject correct code.
    module = (
        "def wrap_text(text, width):\n"
        "    return [text[i:i + width] for i in range(0, len(text), width)]\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    # The impossible assert that started all of this: 11 chars into width 10.\n"
        "    assert wrap_text('hello world', 10) == ['hello world']\n"
    )

    code, out = _run_contract(module, "assert wrap_text('hello world', 10) == ['hello worl', 'd']")

    assert code == 0, f"the model's own assert ran and killed a correct module:\n{out}"


# ── reviewer_node ─────────────────────────────────────────────────────────────


def _review(code: str, contract: str) -> dict:
    from multi_hive.nodes.execution.reviewer_node import reviewer_node

    return reviewer_node(
        {
            "active_file": "outputs/contract_probe.py",
            "project_files": {"outputs/contract_probe.py": code},
            "contracts": {"outputs/contract_probe.py": contract},
            "editor_retries": 0,
            "loop_health": None,
        }
    )


def test_reviewer_passes_code_that_satisfies_the_contract():
    result = _review("def add(a, b):\n    return a + b\n", "assert add(2, 2) == 4")

    assert result["editor_error"] is None
    assert result["contract_satisfied"] is True


def test_reviewer_fails_code_that_violates_the_contract_and_says_whose_fault_it_is():
    result = _review("def add(a, b):\n    return a * b\n", "assert add(2, 3) == 5")

    assert result["contract_satisfied"] is False
    assert result["editor_retries"] == 1
    # The message has to tell the editor the assert is right and the code is
    # wrong. Left ambiguous, a 7B model "fixes" the failure by rewriting the test.
    assert "ACCEPTANCE CONTRACT VIOLATED" in result["editor_error"]
    assert "the code is what is wrong" in result["editor_error"]


def test_reviewer_rejects_a_module_that_exits_zero_at_import():
    # os._exit(0) is un-catchable — it exits the process with code 0 before any
    # assert runs and without printing the sentinel. The reviewer confirms a pass
    # by the nonce sentinel, never the exit code, so this is a failure, not the
    # false green the audit flagged as finding #1.
    result = _review("import os\nos._exit(0)\n", "assert add(2, 2) == 4")

    assert result["contract_satisfied"] is False
    assert result["editor_retries"] == 1
    assert "before the contract could run" in result["editor_error"]


def test_reviewer_ignores_a_module_that_prints_the_sentinel_itself():
    # The sentinel carries a per-run nonce the module cannot know, so printing
    # "CONTRACT_PASS" and exiting cannot forge a satisfied contract.
    result = _review(
        "print('CONTRACT_PASS deadbeefdeadbeef')\nimport os\nos._exit(0)\n",
        "assert add(2, 2) == 4",
    )

    assert result["contract_satisfied"] is False


# ── semantic_reviewer_node ────────────────────────────────────────────────────


def test_semantic_reviewer_stands_down_when_a_contract_passed():
    # And still retires the task. A gate that passes without advancing leaves the
    # graph re-verifying the same file forever — that exact bug cost 992 identical
    # rejections once. It also must not call the LLM: this test has no Ollama, so
    # if it reaches the model it fails by connection error, which is the point.
    import asyncio

    from multi_hive.nodes.execution.semantic_reviewer_node import semantic_reviewer_node

    result = asyncio.run(
        semantic_reviewer_node(
            {
                "active_file": "outputs/a.py",
                "project_files": {"outputs/a.py": "def a(): return 1"},
                "contract_satisfied": True,
                "editor_error": None,
                "editor_retries": 2,
                "task_queue": [],
            }
        )
    )

    assert result["semantic_verdict"] == "PASS (acceptance contract satisfied)"
    assert result["current_task"] is None  # retired
    assert result["editor_retries"] == 0


def test_a_failed_contract_still_blocks_the_semantic_reviewer():
    # editor_error is set, so intent review is meaningless and, worse, a PASS
    # would call _advance() and erase the execution failure. That is how the hive
    # once shipped a semver.py that raised TypeError under "✅ Sprint Complete".
    import asyncio

    from multi_hive.nodes.execution.semantic_reviewer_node import semantic_reviewer_node

    result = asyncio.run(
        semantic_reviewer_node(
            {
                "active_file": "outputs/a.py",
                "project_files": {"outputs/a.py": "def a(): return 1"},
                "contract_satisfied": False,
                "editor_error": "ACCEPTANCE CONTRACT VIOLATED: ...",
                "editor_retries": 1,
                "task_queue": [],
            }
        )
    )

    assert result == {}


# ── The editor prompt ─────────────────────────────────────────────────────────


def test_a_contract_takes_the_asserts_away_from_the_editor():
    from multi_hive.prompts import get_editor_prompt

    plain = get_editor_prompt("obj", "")
    contracted = get_editor_prompt("obj", "", acceptance_contract="assert add(2, 2) == 4")

    # Without a contract the model writes its own asserts, so it gets the long
    # list of rules about which asserts are safe. With one, it writes none.
    assert "ASSERT RULES" in plain
    assert "ASSERT RULES" not in contracted
    assert "DO NOT write asserts" in contracted

    # The contract itself is handed over — hiding it would be asking the model to
    # guess the very thing the human took the trouble to write down.
    assert "assert add(2, 2) == 4" in contracted

    # ...along with the rule that makes showing it safe.
    assert "DO NOT special-case the contract's literal inputs" in contracted


# ── The benchmark's honesty ───────────────────────────────────────────────────


def _call_inputs(source: str) -> set:
    """Every string literal passed as an argument to a call — i.e. every input a
    model could memorise. Assert messages and f-strings are not inputs, so they
    are not collected: two suites are allowed to phrase a failure the same way."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and sub.value:
                    found.add(sub.value)
    return found


def _asserts(source: str) -> set:
    """Assert statements, normalised so the hidden suite's `M.` prefix does not
    disguise a copy-paste."""
    return {
        ast.unparse(node).replace("M.", "")
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assert)
    }


def test_bench_contracts_share_no_input_with_the_hidden_tests():
    """
    The load-bearing test of the whole benchmark.

    The editor SEES the contract. So if a bench contract reused the hidden suite's
    inputs, a model that memorised them instead of implementing the requirement
    would pass both, and the benchmark would report a triumph it had itself
    arranged.

    Keeping the inputs disjoint is what makes the hidden suite a gaming detector:
    hardcoded code passes the contract and fails the bench, and the gap between
    the two scores is the amount of cheating. This test fails the moment someone
    copies an assert across, which is the tempting and fatal shortcut — and it is
    why bench/contracts.py splits "supercalifragilistic" at width 6 while the
    hidden suite splits "abcdefghij" at width 4.
    """
    from multi_hive.bench.contracts import CONTRACTS
    from multi_hive.bench.suite import BY_NAME

    for name, contract in CONTRACTS.items():
        task = BY_NAME[name]

        compile(contract, f"<contract:{name}>", "exec")  # it must at least run

        shared_inputs = _call_inputs(contract) & _call_inputs(task.tests)
        assert not shared_inputs, (
            f"{name}: the contract and the hidden suite share the input(s) {shared_inputs!r}. "
            f"The model sees the contract and never sees the suite — a shared input means "
            f"hardcoding it would pass the benchmark. Pick different values."
        )

        shared_asserts = _asserts(contract) & _asserts(task.tests)
        assert not shared_asserts, (
            f"{name}: this assert appears in both the contract and the hidden suite: "
            f"{shared_asserts!r}. Copying an assert across turns the benchmark into an "
            f"open-book exam and the number it prints into a lie."
        )


def test_every_bench_task_has_a_contract():
    # A task with no contract silently falls back to model-written asserts, so a
    # contract-mode run would quietly be half a contract-mode run.
    from multi_hive.bench.contracts import CONTRACTS
    from multi_hive.bench.suite import TASKS

    assert {t.name for t in TASKS} == set(CONTRACTS)


def test_grade_does_not_pass_a_module_that_exits_zero_at_import():
    # The benchmark harness has the same guard as the contract harness. A
    # candidate that calls os._exit(0) at import terminates with returncode 0
    # before a single hidden test runs; scored as a pass, that is a false green on
    # the number that gates every decision. It must be a failure. (Audit #1.)
    from multi_hive.bench.suite import BY_NAME, grade

    result = grade("import os\nos._exit(0)\n", BY_NAME["lru_cache"])

    assert result.passed is False
