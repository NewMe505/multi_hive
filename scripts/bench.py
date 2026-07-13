"""
bench.py — the performance tracker.

    python scripts/bench.py sprint              # the one to track during development
    python scripts/bench.py sprint --contract   # with human-written acceptance contracts
    python scripts/bench.py models              # when choosing or replacing a tier
    python scripts/bench.py models --models qwen2.5-coder:7b qwen3-coder:30b

    python scripts/bench.py sprint --check      # exit 1 on a regression (CI gate)
    python scripts/bench.py history             # the trend, run by run

Every run is recorded against the current git commit, so a regression can be
traced to the change that caused it. Runs on a dirty tree are recorded but never
used as a baseline — a benchmark of uncommitted code cannot be reproduced.

Grading is against hidden test suites the model never sees. See
src/multi_hive/bench/suite.py for what each task is actually probing, and why.

--contract runs the same tasks with a human-written acceptance contract attached
(src/multi_hive/bench/contracts.py). It is recorded under the subject
"hive+contract", which keeps it on its own baseline, because it is not the same
measurement: plain `sprint` asks whether the hive can guess what you meant, and
`--contract` asks whether it delivers what you actually specified. Comparing the
two scores to each other means nothing. Comparing each to its own history means
everything.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Imported for its side effect: UTF-8 stdout before anything prints (Windows
# stdio is cp1252, and this script emits ✓/✗).
import multi_hive.core.console  # noqa: E402, F401
from multi_hive.bench import history, runner  # noqa: E402
from multi_hive.bench.contracts import contract_for_task  # noqa: E402
from multi_hive.bench.suite import TASKS  # noqa: E402
from multi_hive.config import MODELS, PROVIDER  # noqa: E402

DEFAULT_MODELS = ["qwen2.5-coder:7b", "qwen3-coder:30b"]


def _subject(contract: bool) -> str:
    """
    The history key for a sprint run.

    Every axis that changes what is being measured has to be in here, or runs get
    compared that were never comparable. There are two:

      contract   plain asks "can the hive guess what I meant?", contract asks
                 "when I say exactly what I want, does it deliver?"
      provider   a local 7B and Claude are not the same system under test.

    The default (ollama, no contract) keeps the bare subject "hive", so the
    existing history — and the baselines already in it — stay intact.
    """
    subject = "hive+contract" if contract else "hive"
    return subject if PROVIDER == "ollama" else f"{subject}@{PROVIDER}"

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def _mark(passed: bool) -> str:
    return f"{GREEN}✓ pass{RESET}" if passed else f"{RED}✗ FAIL{RESET}"


# ── Suites ────────────────────────────────────────────────────────────────────


def bench_models(models: list[str]) -> list[history.Run]:
    # This suite talks to Ollama's HTTP API directly, on purpose: its whole job is
    # answering "which local model should back a tier?", which is a question about
    # tok/s and GPU placement and has no meaning for a hosted API. To compare
    # providers, use the `sprint` suite — it goes through llm_factory and measures
    # the system, which is the thing you actually care about.
    if PROVIDER != "ollama":
        print(
            f"{RED}the `models` suite is Ollama-only{RESET} (it measures tok/s and GPU "
            f"placement, which mean nothing for {PROVIDER}).\n"
            f"To compare providers end to end:  "
            f"{BOLD}HIVE_PROVIDER={PROVIDER} python scripts/bench.py sprint{RESET}"
        )
        raise SystemExit(2)

    runs = []
    for model in models:
        print(f"\n{BOLD}=== {model} ==={RESET}", flush=True)
        run = history.Run(suite="models", subject=model).stamp()

        for task in TASKS:
            print(f"  {task.name:16} ({task.complexity:8}) ... ", end="", flush=True)
            result = runner.run_model(model, task)
            run.tasks.append(result)

            why = f"  {DIM}({result['failure']}){RESET}" if result["failure"] else ""
            print(
                f"{result.get('tok_per_sec', 0):5.1f} tok/s  "
                f"{result['wall_sec']:6.1f}s  {_mark(result['passed'])}{why}"
            )

        runs.append(run)
    return runs


def bench_sprint(contract: bool = False) -> list[history.Run]:
    label = "full hive + acceptance contracts" if contract else "full hive, end to end"
    print(f"\n{BOLD}=== {label} ==={RESET}", flush=True)
    print(
        f"  {DIM}provider={PROVIDER}  fast={MODELS['fast']}  strong={MODELS['strong']}{RESET}",
        flush=True,
    )

    run = history.Run(suite="sprint", subject=_subject(contract)).stamp()

    gamed: list[str] = []

    async def run_all() -> None:
        """
        Every task, in ONE event loop.

        This used to be `asyncio.run()` per task, and that quietly broke the
        benchmark. `llm_factory` caches the async client, and that client's
        connection pool is bound to the loop that created it — so task 2 inherited
        a client pointing at task 1's closed loop and every call raised
        `Event loop is closed`.

        It does not fail cleanly. The editor catches the error, logs it, and
        retries; the retry raises the identical error; the repeat-error
        fingerprint matches, the circuit breaker fires, and the task is escalated
        to the human gate having never once reached a model. It is scored as a
        model failure. Observed live: `merge_intervals` recorded as `✗ FAIL
        (no code) [human gate]` on a task the 7B passes in 80 seconds.

        A benchmark that invents failures is worse than no benchmark, because it
        is believed. One loop for the process — which is what cli.py always did.
        """
        for task in TASKS:
            print(f"  {task.name:16} ({task.complexity:8}) ... ", end="", flush=True)

            result = await runner.run_sprint(
                task, contract_for_task(task.name) if contract else ""
            )
            run.tasks.append(result)

            tiers = "→".join(result["tiers"]) or "—"
            why = f"  {DIM}({result['failure'][:50]}){RESET}" if result["failure"] else ""
            gate = f" {RED}[human gate]{RESET}" if result["escalated_to_human"] else ""

            # Contract satisfied, hidden suite failed: the code cleared the asserts
            # it was shown and failed the ones it was not. That is the signature of
            # hardcoding against the contract's literals, and it is the one outcome
            # that would invalidate the whole approach — so it gets shouted, not
            # buried in a summary line.
            if contract and result.get("contract_satisfied") and not result["passed"]:
                gamed.append(task.name)
                gate += f" {RED}{BOLD}[CONTRACT GAMED]{RESET}"

            print(
                f"{result['wall_sec']:6.1f}s  {result['nodes']:3d} nodes  "
                f"tier={tiers:12} {_mark(result['passed'])}{gate}{why}"
            )

    asyncio.run(run_all())

    if gamed:
        print(
            f"\n  {RED}{BOLD}CONTRACT GAMED on {', '.join(gamed)}{RESET} — the contract "
            f"passed and the hidden suite did not.\n  {DIM}The model satisfied the "
            f"literals it was shown without implementing the requirement. Tighten the "
            f"anti-hardcoding rule in prompts._EDITOR_CONTRACT_PREFIX.{RESET}"
        )

    return [run]


# ── Reporting ─────────────────────────────────────────────────────────────────


def report(runs: list[history.Run], check: bool) -> int:
    exit_code = 0

    for run in runs:
        print(f"\n{BOLD}{run.subject}{RESET}  {run.passed}/{run.total} passed  "
              f"{run.wall:.0f}s total  {DIM}@ {run.commit}"
              f"{' (dirty)' if run.dirty else ''}{RESET}")

        baseline = history.baseline_for(run)
        if not baseline:
            print(f"  {DIM}no clean baseline yet — this run becomes one once committed{RESET}")
            history.record(run)
            continue

        cmp = history.compare(run, baseline)

        arrow = "→" if cmp.quality_delta == 0 else ("↑" if cmp.quality_delta > 0 else "↓")
        colour = GREEN if cmp.quality_delta > 0 else (RED if cmp.quality_delta < 0 else DIM)
        print(
            f"  vs {baseline.commit}: quality {colour}{baseline.passed}/{baseline.total} "
            f"{arrow} {run.passed}/{run.total}{RESET}   "
            f"speed {cmp.speed_ratio:.2f}x"
        )

        if cmp.fixed_tasks:
            print(f"  {GREEN}fixed:{RESET}     {', '.join(cmp.fixed_tasks)}")
        if cmp.regressed_tasks:
            print(f"  {RED}REGRESSED:{RESET} {', '.join(cmp.regressed_tasks)}")

        if cmp.quality_regression:
            print(f"  {RED}{BOLD}QUALITY REGRESSION{RESET} — code that used to be correct is not")
            exit_code = 1
        if cmp.speed_regression:
            print(
                f"  {RED}{BOLD}SPEED REGRESSION{RESET} — "
                f"{cmp.speed_ratio:.2f}x slower than {baseline.commit}"
            )
            exit_code = 1

        history.record(run)

    if check and exit_code:
        print(f"\n{RED}regression detected — failing{RESET}")
    return exit_code if check else 0


def show_history() -> int:
    runs = history.load()
    if not runs:
        print("no runs recorded yet")
        return 0

    print(f"{'commit':10} {'suite':7} {'subject':20} {'quality':>8} {'wall':>7}  version")
    print("-" * 72)
    for run in runs:
        dirty = "*" if run.dirty else " "
        print(
            f"{run.commit:9}{dirty} {run.suite:7} {run.subject:20} "
            f"{run.passed:>3}/{run.total:<4} {run.wall:6.0f}s  {run.version}"
        )
    print(f"\n{DIM}* = dirty tree; recorded, but never used as a baseline{RESET}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("suite", choices=["models", "sprint", "history"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if this run regressed against its baseline",
    )
    parser.add_argument(
        "--contract",
        action="store_true",
        help=(
            "attach the human-written acceptance contracts (bench/contracts.py). "
            "Recorded as a separate subject; not comparable to a plain sprint run."
        ),
    )
    args = parser.parse_args()

    if args.suite == "history":
        raise SystemExit(show_history())

    runs = (
        bench_models(args.models)
        if args.suite == "models"
        else bench_sprint(contract=args.contract)
    )
    raise SystemExit(report(runs, args.check))


if __name__ == "__main__":
    main()
