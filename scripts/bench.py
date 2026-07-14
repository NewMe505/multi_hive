"""
bench.py — the performance tracker.

    python scripts/bench.py sprint --repeat 3   # the one to track during development
    python scripts/bench.py sprint --contract   # with human-written acceptance contracts
    python scripts/bench.py models              # when choosing or replacing a tier
    python scripts/bench.py models --models qwen2.5-coder:7b qwen3-coder:30b

    python scripts/bench.py sprint --check      # exit 1 on a regression (CI gate)
    python scripts/bench.py history             # the trend, run by run

The cost thesis, measured (both arms on the SAME provider)
----------------------------------------------------------
"2x the tokens on a 10x cheaper model is 5-8x cheaper for the same quality" needs
two numbers, not one. On a hosted provider `models` stops measuring tok/s (which
a hosted API does not have) and instead runs each task as a metered ONE-SHOT — the
"skip the pipeline, just run the good model once" baseline the pipeline must beat:

    HIVE_MAX_USD=20 HIVE_PROVIDER=anthropic python scripts/bench.py sprint --contract --repeat 3
    HIVE_MAX_USD=20 HIVE_PROVIDER=anthropic python scripts/bench.py models --repeat 3

The first records the pipeline's quality and $/task (subject hive+contract@anthropic);
the second records the one-shot's (subject 1shot:strong@anthropic). Both go through
the same governor and tokenizer, so their $/task is finally a subtraction, not a
guess. `--models fast strong` also measures the haiku one-shot. Set HIVE_MAX_USD:
the default $5 will not cover a 30-sprint x3 run.

One run is a sample, not a measurement
--------------------------------------
The models are sampled at temperature 0.1, the retry loop is driven by whatever
they happened to emit, and a single unlucky generation cascades: one bad attempt
escalates the tier, which pays a ~23s model reload, which moves the wall clock by
minutes. Two runs of *identical code* can differ by 40% on time and by a whole
task on quality.

So `--repeat N` runs the suite N times, and **a task counts as passed only if it
passes every run.** A task that passes 2 of 3 has not been fixed; it has been made
likely, and scoring it as a win reports a coin landing heads. Those are printed as
FLAKY and counted as failures. Wall time is reported as the median, because a mean
lets a single reload tell a story about the weather.

Use `--repeat 3` before believing any conclusion. This is not a hypothetical
caution: a single-run comparison on this suite produced a confident "1.45x speed
regression" that was, on investigation, a harness bug plus noise.

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
import statistics
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
from multi_hive.core.governor import BudgetExhausted  # noqa: E402

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


def bench_models(models: list[str], repeat: int = 1) -> list[history.Run]:
    """
    Each model, every task, `repeat` times — under the SAME strict rule as sprint.

    `--repeat` used to be silently ignored here: every models number ever recorded
    is a single sample. And those single samples were being read next to sprint's
    strict pass-every-run aggregate, which is not a comparison, it is a category
    error. The 30B's headline 4/4 is exactly that: at commit 18e4dab the same model
    scored 3/4 with semver failing. Under sprint's own rule the 30B is 3/4.

    A task passes only if it passes EVERY run. Same reason as _aggregate(): a task
    that passes 2 of 3 has not been fixed, it has been made likely, and scoring it
    as a win reports a coin landing heads.
    """
    # This suite talks to Ollama's HTTP API directly, on purpose: its whole job is
    # answering "which local model should back a tier?", which is a question about
    # tok/s and GPU placement and has no meaning for a hosted API. main() routes a
    # hosted provider to bench_oneshot() — the metered one-shot baseline — instead;
    # this assert is the backstop if bench_models is ever called directly off Ollama.
    assert PROVIDER == "ollama", (
        "bench_models is Ollama-only (it measures tok/s and GPU placement); "
        "hosted providers use bench_oneshot()"
    )

    runs = []
    for model in models:
        print(f"\n{BOLD}=== {model} ==={RESET}", flush=True)
        print(f"  {DIM}repeat={repeat}{RESET}", flush=True)

        run = history.Run(suite="models", subject=model).stamp()
        # Part of the run's identity, so history.baseline_for never compares a
        # strict x3 aggregate against an old single-run sample.
        run.repeat = repeat

        results: dict[str, list[dict]] = {task.name: [] for task in TASKS}

        for rep in range(repeat):
            if repeat > 1:
                print(f"\n  {BOLD}run {rep + 1}/{repeat}{RESET}", flush=True)

            for task in TASKS:
                print(f"  {task.name:16} ({task.complexity:8}) ... ", end="", flush=True)
                result = runner.run_model(model, task)
                results[task.name].append(result)

                why = f"  {DIM}({result['failure']}){RESET}" if result["failure"] else ""
                print(
                    f"{result.get('tok_per_sec', 0):5.1f} tok/s  "
                    f"{result['wall_sec']:6.1f}s  {_mark(result['passed'])}{why}"
                )

        flaky: list[str] = []

        for task in TASKS:
            reps = results[task.name]
            if not reps:
                continue

            passes = sum(1 for r in reps if r["passed"])
            failures = [r["failure"] for r in reps if r["failure"]]
            if 0 < passes < len(reps):
                flaky.append(f"{task.name} ({passes}/{len(reps)})")

            run.tasks.append(
                {
                    "task": task.name,
                    "complexity": task.complexity,
                    "passed": passes == len(reps),  # reliably, not luckily
                    "pass_rate": round(passes / len(reps), 3),
                    "repeats": len(reps),
                    "failure": failures[0][:120] if failures else "",
                    "wall_sec": statistics.median([r["wall_sec"] for r in reps]),
                    "tok_per_sec": statistics.median([r.get("tok_per_sec", 0) for r in reps]),
                    "output_tokens": statistics.median([r.get("output_tokens", 0) for r in reps]),
                    # Cost, so the one-shot baseline and the pipeline can be compared
                    # on what they SPENT and not only on what they scored. Without
                    # this the `models` row printed "tokens 0" and the whole
                    # efficiency question was unanswerable.
                    "input_tokens": statistics.median([r.get("input_tokens", 0) for r in reps]),
                    "total_tokens": statistics.median([r.get("total_tokens", 0) for r in reps]),
                    "attempts": 1,
                    # A one-shot model has exactly one attempt, so "did the first
                    # attempt pass" IS "did it pass". Assigning `passes == len(reps)`
                    # made the field carry no information here while _story() printed
                    # it in the same 1st-try column as the hive's genuine first-attempt
                    # rate — inviting a comparison between a real number and a
                    # tautology. It is None: the question does not apply.
                    "first_attempt_passed": None,
                    "usd": 0.0,
                    "gpu_placement": reps[0].get("gpu_placement", "?"),
                }
            )

        if flaky:
            print(
                f"\n  {BOLD}\033[33mFLAKY: {', '.join(flaky)}{RESET}\n"
                f"  {DIM}Counted as NOT passed — `passed` means passed every run.{RESET}"
            )

        runs.append(run)
    return runs


def _resolve_tier(entry: str) -> str:
    """
    Map a --models entry to a tier for a hosted one-shot.

    Accepts a tier name ("fast"/"strong") directly, or the model name that backs a
    tier (so `--models claude-fable-5` works too). Anything else is an error that
    names the valid choices, rather than silently benchmarking nothing.
    """
    if entry in MODELS:
        return entry
    for tier, model in MODELS.items():
        if model == entry:
            return tier
    raise SystemExit(
        f"{RED}'{entry}' is neither a tier nor a model on provider {PROVIDER}.{RESET}\n"
        f"  tiers:  {', '.join(sorted(MODELS))}\n"
        f"  models: {', '.join(MODELS[t] for t in sorted(MODELS))}"
    )


def _record_oneshot(
    runs: list[history.Run], tier: str, results: dict[str, list[dict]], complete_reps: int
) -> None:
    """
    Aggregate one tier's completed repeats into a Run — strict, exactly like the
    other two suites: a task passes only if it passed EVERY recorded repeat.

    Called with `complete_reps` = the number of repeats that ran to completion,
    which is `repeat` normally and fewer if a budget breach cut the run short. Any
    partial repeat's results are dropped so tasks are never scored on uneven sample
    counts — the same rule bench_sprint applies, and for the same reason.
    """
    if complete_reps == 0:
        print(f"  {RED}no complete repeat — nothing recorded for {tier}.{RESET}", flush=True)
        return

    for name in results:
        del results[name][complete_reps:]

    run = history.Run(suite="models", subject=f"1shot:{tier}@{PROVIDER}").stamp()
    run.repeat = complete_reps

    flaky: list[str] = []
    for task in TASKS:
        reps = results[task.name]
        if not reps:
            continue

        passes = sum(1 for r in reps if r["passed"])
        failures = [r["failure"] for r in reps if r["failure"]]
        if 0 < passes < len(reps):
            flaky.append(f"{task.name} ({passes}/{len(reps)})")

        run.tasks.append(
            {
                "task": task.name,
                "complexity": task.complexity,
                "passed": passes == len(reps),  # reliably, not luckily
                "pass_rate": round(passes / len(reps), 3),
                "repeats": len(reps),
                "failure": failures[0][:120] if failures else "",
                "wall_sec": statistics.median([r["wall_sec"] for r in reps]),
                "input_tokens": statistics.median([r.get("input_tokens", 0) for r in reps]),
                "output_tokens": statistics.median([r.get("output_tokens", 0) for r in reps]),
                "total_tokens": statistics.median([r.get("total_tokens", 0) for r in reps]),
                "usd": round(statistics.median([r.get("usd", 0.0) for r in reps]), 6),
                # SUM, not median: one unreadable call anywhere voids the cost figure.
                "unmetered": sum(r.get("unmetered", 0) or 0 for r in reps),
                "attempts": 1,
                "first_attempt_passed": None,  # tautological for a one-shot; see run_oneshot
            }
        )

    if flaky:
        print(
            f"\n  {BOLD}\033[33mFLAKY: {', '.join(flaky)}{RESET}\n"
            f"  {DIM}Counted as NOT passed — `passed` means passed every run.{RESET}"
        )

    runs.append(run)


def bench_oneshot(tiers: list[str], repeat: int = 1) -> list[history.Run]:
    """
    The one-shot baseline on a hosted provider — the arm the `models` suite lacked.

    `models` measures tok/s and GPU placement, which a hosted API does not have, so
    it refuses to run off Ollama — and that left the cost thesis with no baseline:
    the pipeline's $/task was being measured against nothing, and "2x the tokens on
    a 10x cheaper model is 5-8x cheaper" was an extrapolation, not a measurement.

    This runs each task once, straight at the model through the metered llm_factory,
    and grades it against the same hidden suite the pipeline is graded on. The result
    is the missing half: run `sprint --contract` for the pipeline's cost, and this
    for what it must be cheaper THAN — both real, both on the same provider, so the
    ratio between them is finally a subtraction and not a guess.

    History keeps the two on separate subjects (`1shot:<tier>@<provider>` vs
    `hive[+contract]@<provider>`), because a one-shot and a pipeline are not the same
    system under test — so this never auto-compares them; it makes both numbers exist.

    The whole arm runs in ONE event loop (the async client's pool is bound to the
    loop that built it) and against ONE cumulative governor, so `HIVE_MAX_USD` bounds
    the total spend across every tier and repeat. A breach records the repeats that
    completed and skips the rest.
    """
    import asyncio

    from multi_hive.core.llm_factory import model_for

    resolved = [_resolve_tier(t) for t in tiers]

    print(f"\n{BOLD}=== one-shot baseline ({PROVIDER}) ==={RESET}", flush=True)
    print(
        f"  {DIM}tiers={', '.join(resolved)}  repeat={repeat}  "
        f"(HIVE_MAX_USD bounds the total spend){RESET}",
        flush=True,
    )

    runs: list[history.Run] = []

    async def run_all() -> None:
        for tier in resolved:
            model = model_for(tier)
            print(f"\n{BOLD}--- {tier} ({model}) — one shot ---{RESET}", flush=True)

            results: dict[str, list[dict]] = {task.name: [] for task in TASKS}
            completed = 0
            try:
                for rep in range(repeat):
                    if repeat > 1:
                        print(f"\n  {BOLD}run {rep + 1}/{repeat}{RESET}", flush=True)

                    for task in TASKS:
                        print(
                            f"  {task.name:16} ({task.complexity:8}) ... ",
                            end="",
                            flush=True,
                        )
                        result = await runner.run_oneshot(tier, task)
                        results[task.name].append(result)

                        why = (
                            f"  {DIM}({result['failure'][:40]}){RESET}"
                            if result["failure"]
                            else ""
                        )
                        cost = f"${result.get('usd', 0.0):.4f}"
                        print(
                            f"{result['wall_sec']:6.1f}s  "
                            f"{result.get('total_tokens', 0):6,} tok  {cost:>9}  "
                            f"{_mark(result['passed'])}{why}"
                        )
                    completed = rep + 1
            except BudgetExhausted as e:
                # Same contract as bench_sprint: BudgetExhausted is a BaseException
                # that escapes run_oneshot, so record the repeats that finished and
                # stop — the remaining tiers would only breach again immediately.
                print(
                    f"\n  {RED}{BOLD}BUDGET EXHAUSTED{RESET} {DIM}({e}){RESET}  "
                    f"{DIM}recording {completed} complete repeat(s) of {repeat}; "
                    f"remaining tiers skipped.{RESET}",
                    flush=True,
                )
                _record_oneshot(runs, tier, results, completed)
                return

            _record_oneshot(runs, tier, results, repeat)

    asyncio.run(run_all())
    return runs


async def _warmup() -> None:
    """
    Load the fast model before the clock starts on task 1.

    Otherwise the first task of a run pays a cold model load that the other three
    do not, and it looks slower for a reason that has nothing to do with the code
    under test. The strong model's ~23s load is deliberately NOT warmed: it is
    paid only when a task escalates, and that cost is a real property of the
    escalation ladder. Hiding it would flatter the benchmark.
    """
    from langchain_core.messages import HumanMessage

    from multi_hive.core.llm_factory import get_async_llm

    print(f"  {DIM}warming {MODELS['fast']}...{RESET}", end="", flush=True)
    try:
        await get_async_llm("editor", "fast").ainvoke([HumanMessage(content="ok")])
        print(f"\r  {DIM}warmed {MODELS['fast']}{' ' * 20}{RESET}", flush=True)
    except Exception as e:
        print(f"\r  {RED}warmup failed: {e}{RESET}", flush=True)


def bench_sprint(contract: bool = False, repeat: int = 1) -> list[history.Run]:
    label = "full hive + acceptance contracts" if contract else "full hive, end to end"
    print(f"\n{BOLD}=== {label} ==={RESET}", flush=True)
    print(
        f"  {DIM}provider={PROVIDER}  fast={MODELS['fast']}  strong={MODELS['strong']}"
        f"  repeat={repeat}{RESET}",
        flush=True,
    )

    # task name -> one result per repeat
    results: dict[str, list[dict]] = {task.name: [] for task in TASKS}

    # How many repeats ran to completion. A repeat that a budget breach cut off
    # partway is NOT counted: its tasks have uneven sample counts (the ones before
    # the breach ran, the ones after did not), and _aggregate's `passed == passed
    # every run` would then judge different tasks on different numbers of samples —
    # the exact order-dependent, invented-failure class this suite exists to avoid.
    completed_repeats = 0

    async def run_all() -> None:
        nonlocal completed_repeats
        """
        Every task, every repeat, in ONE event loop.

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
        await _warmup()

        for rep in range(repeat):
            if repeat > 1:
                print(f"\n  {BOLD}run {rep + 1}/{repeat}{RESET}", flush=True)

            for task in TASKS:
                print(f"  {task.name:16} ({task.complexity:8}) ... ", end="", flush=True)

                result = await runner.run_sprint(
                    task, contract_for_task(task.name) if contract else ""
                )
                results[task.name].append(result)

                tiers = "→".join(result["tiers"]) or "—"
                why = f"  {DIM}({result['failure'][:44]}){RESET}" if result["failure"] else ""
                gate = f" {RED}[human gate]{RESET}" if result["escalated_to_human"] else ""

                # Contract satisfied, hidden suite failed: the code cleared the
                # asserts it was shown and failed the ones it was not. That is the
                # signature of hardcoding against the contract's literals, and it
                # is the one outcome that would invalidate the whole approach — so
                # it gets shouted, not buried in a summary line.
                if contract and result.get("contract_satisfied") and not result["passed"]:
                    gate += f" {RED}{BOLD}[CONTRACT GAMED]{RESET}"

                print(
                    f"{result['wall_sec']:6.1f}s  {result['nodes']:3d} nodes  "
                    f"tier={tiers:12} {_mark(result['passed'])}{gate}{why}"
                )

            completed_repeats = rep + 1

    # `run_sprint` catches `except Exception`, but `BudgetExhausted` is a
    # BaseException by design (so a budget stop is never mistaken for an editor
    # error and retried). It therefore escapes to here. Without this handler it
    # would unwind straight through `asyncio.run` and kill the process before a
    # single number was recorded — the whole point of a 30-sprint paid run, lost
    # for the last one that happened to trip the cap. Record the repeats that DID
    # finish, and say plainly that the run was cut short.
    try:
        asyncio.run(run_all())
    except BudgetExhausted as e:
        print(
            f"\n  {RED}{BOLD}BUDGET EXHAUSTED{RESET} {DIM}({e}){RESET}\n"
            f"  {DIM}Recording {completed_repeats} complete repeat(s) of {repeat} "
            f"requested; the interrupted repeat is discarded.{RESET}",
            flush=True,
        )
        if completed_repeats == 0:
            print(
                f"  {RED}No complete repeat finished — nothing recorded. "
                f"Raise HIVE_MAX_USD and re-run.{RESET}",
                flush=True,
            )
            return []
        for name in results:
            del results[name][completed_repeats:]
        return [_aggregate(results, contract, completed_repeats)]

    return [_aggregate(results, contract, repeat)]


def _aggregate(results: dict[str, list[dict]], contract: bool, repeat: int) -> history.Run:
    """
    Collapse N repeats into one Run, and say out loud what was thrown away.

    `passed` means passed **every** repeat. That is the strict definition on
    purpose: a task that passes 2 runs in 3 has not been fixed, it has been made
    likely, and a benchmark that scores it as a win will report a "quality
    improvement" that is really a coin landing heads. The pass_rate is kept
    alongside so the flakiness is visible rather than rounded away.

    `wall_sec` is the MEDIAN, not the mean. One task hitting a 23s model reload,
    or the OS deciding to index something mid-run, drags a mean into telling a
    story about the weather.
    """
    run = history.Run(suite="sprint", subject=_subject(contract)).stamp()
    # Part of the run's identity: an x3 aggregate is only ever compared against a
    # prior x3 aggregate, never a single-run sample. See bench/history.py.
    run.repeat = repeat

    flaky: list[str] = []
    gamed: list[str] = []

    for task in TASKS:
        runs = results[task.name]
        if not runs:
            continue

        passes = sum(1 for r in runs if r["passed"])
        walls = sorted(r["wall_sec"] for r in runs)
        failures = [r["failure"] for r in runs if r["failure"]]

        if 0 < passes < len(runs):
            flaky.append(f"{task.name} ({passes}/{len(runs)})")
        if contract and any(r.get("contract_satisfied") and not r["passed"] for r in runs):
            gamed.append(task.name)

        run.tasks.append(
            {
                "task": task.name,
                "complexity": task.complexity,
                "passed": passes == len(runs),  # reliably, not luckily
                "pass_rate": round(passes / len(runs), 3),
                "repeats": len(runs),
                "wall_sec": statistics.median(walls),
                "wall_min": min(walls),
                "wall_max": max(walls),
                "nodes": statistics.median([r["nodes"] for r in runs]),
                "tiers": runs[0]["tiers"],
                "escalated_to_human": any(r["escalated_to_human"] for r in runs),
                # The story metrics. See bench/runner.run_sprint for what each is, and
                # why the score alone cannot tell you.
                "attempts": statistics.median([r.get("attempts", 0) for r in runs]),
                "first_attempt_passed": all(r.get("first_attempt_passed") for r in runs),
                # ANY run that produced a passing file and then shipped something else.
                # `any`, not `all`: destroying a correct answer even once is the finding.
                "discarded_a_pass": any(r.get("discarded_a_pass") for r in runs),
                "total_tokens": statistics.median([r.get("total_tokens", 0) for r in runs]),
                "usd": round(statistics.median([r.get("usd", 0.0) for r in runs]), 6),
                # SUM, not median: one unreadable call anywhere invalidates the cost
                # figure, and a median would hide it behind two clean runs.
                "unmetered": sum(r.get("unmetered", 0) or 0 for r in runs),
                # all(), not any() — it must aggregate the same way `passed` does.
                #
                # With any(), a row could read contract_satisfied=True, passed=False
                # assembled from two DIFFERENT repeats: the contract cleared in run
                # 1, the hidden suite failed in run 3. That is the exact signature of
                # [CONTRACT GAMED] — the one outcome that would invalidate the whole
                # acceptance-contract approach — manufactured out of ordinary
                # flakiness, with no gaming anywhere. A gaming detector that cries
                # wolf gets ignored, and then it is not a detector.
                #
                # The per-run alarm below is unaffected: it correlates
                # contract_satisfied and passed WITHIN a single run, which is the
                # only place the correlation means anything.
                "contract_satisfied": all(r.get("contract_satisfied") for r in runs),
                "failure": failures[0][:120] if failures else "",
            }
        )

    if repeat > 1:
        print(f"\n{BOLD}  aggregate over {repeat} runs{RESET}")
        print(f"  {DIM}{'task':16} {'passed':>8}  {'median':>7}  {'min':>7}  {'max':>7}{RESET}")
        for t in run.tasks:
            colour = GREEN if t["passed"] else (RED if t["pass_rate"] == 0 else "\033[33m")
            count = f"{round(t['pass_rate'] * t['repeats'])}/{t['repeats']}"
            print(
                f"  {t['task']:16} {colour}{count:>8}{RESET}  "
                f"{t['wall_sec']:6.1f}s  {t['wall_min']:6.1f}s  {t['wall_max']:6.1f}s"
            )

    if flaky:
        print(
            f"\n  {BOLD}\033[33mFLAKY: {', '.join(flaky)}{RESET}\n"
            f"  {DIM}These tasks pass sometimes. A single run of any of them proves nothing,\n"
            f"  and scoring one as a win would be reporting a coin landing heads. They are\n"
            f"  counted as NOT passed — `passed` here means passed every run.{RESET}"
        )

    if gamed:
        print(
            f"\n  {RED}{BOLD}CONTRACT GAMED on {', '.join(gamed)}{RESET} — the contract "
            f"passed and the hidden suite did not.\n  {DIM}The model satisfied the "
            f"literals it was shown without implementing the requirement. Tighten the "
            f"anti-hardcoding rule in prompts._EDITOR_CONTRACT_PREFIX.{RESET}"
        )

    return run


# ── Reporting ─────────────────────────────────────────────────────────────────


def _story(run: history.Run) -> None:
    """
    The numbers the score cannot tell you.

    "9/9 passed" was the whole report, and it hides everything that matters about
    HOW. Two systems can both score 9/9 while one gets it right first time for 4k
    tokens and the other thrashes through three retries and 40k. Only one of those
    is good, and this benchmark could not tell them apart.

    Four numbers. Each exists because a real finding was invisible without it:

    COST      Tokens, and dollars on a paid provider. The pipeline can only be said
              to beat a one-shot prompt if you know what it SPENT doing so. A system
              that scores one task higher while burning ten times the tokens has not
              obviously won.

    1st-TRY   How often the first file the editor wrote already passed. A system that
              passes after three retries is not the same system as one that passes
              immediately, and `passed` cannot see the difference.

    THRASH    Median editor attempts. 1.0 means it wrote the answer. 3.0 means it
              argued with itself.

    DISCARDED The one that matters most, and the one that was completely invisible.
              The editor produced code that PASSES THE HIDDEN SUITE, and the sprint
              shipped something else — a reviewer rejected a correct answer and a
              retry overwrote it. The bench only ever graded the last file on disk,
              so the pipeline could destroy its own correct work and score it as "the
              model could not do it". Any number above zero here is a bug with a name
              (best-attempt retention), not a mystery.
    """
    tasks = [t for t in run.tasks if "attempts" in t]
    if not tasks:
        return

    tokens = sum(t.get("total_tokens", 0) or 0 for t in tasks)
    usd = sum(t.get("usd", 0.0) or 0.0 for t in tasks)
    first = sum(1 for t in tasks if t.get("first_attempt_passed") is True)
    thrash = statistics.median([t.get("attempts", 0) or 0 for t in tasks])
    discarded = [t["task"] for t in tasks if t.get("discarded_a_pass")]
    passed = run.passed or 1  # a 0-pass run's tokens-per-pass is meaningless anyway

    cost = f"${usd:.4f}" if usd else "free"

    # `first_attempt_passed` is None for the one-shot suite — the question does not
    # apply to a model that only gets one attempt. Print a dash rather than a number
    # that invites a false comparison.
    applicable = [t for t in tasks if t.get("first_attempt_passed") is not None]
    first_col = f"{first}/{len(applicable)}" if applicable else "n/a"

    print(
        f"  {DIM}tokens {tokens:,} ({tokens // passed:,}/pass, {cost})   "
        f"1st-try {first_col}   thrash {thrash:.1f} attempts{RESET}"
    )

    # A cost figure computed from a BROKEN meter is worse than no cost figure: it is
    # plausible, low, and wrong. record_unmetered() counts the calls the governor
    # could not read, and if HIVE_MAX_USD/HIVE_MAX_TOKENS are unset (the ollama
    # default) the HIVE_MAX_UNMETERED ceiling never fires — so the understated number
    # above is all the operator would ever see. Say so.
    unmetered = sum(t.get("unmetered", 0) or 0 for t in tasks)
    if unmetered:
        print(
            f"  {RED}{BOLD}{unmetered} model call(s) reported NO token usage{RESET} "
            f"{DIM}— the cost above is an undercount, not a measurement.{RESET}"
        )

    if discarded:
        print(
            f"  {RED}{BOLD}DISCARDED A PASSING ANSWER on {', '.join(discarded)}{RESET}\n"
            f"  {DIM}The editor wrote code that passes the hidden suite, and the sprint "
            f"shipped something else.\n  A reviewer rejected a correct answer and a "
            f"retry overwrote it — the pipeline destroying\n  its own work. See the "
            f"audit finding on best-attempt retention.{RESET}"
        )


def report(runs: list[history.Run], check: bool) -> int:
    exit_code = 0

    for run in runs:
        print(f"\n{BOLD}{run.subject}{RESET}  {run.passed}/{run.total} passed  "
              f"{run.wall:.0f}s total  {DIM}@ {run.commit}"
              f"{' (dirty)' if run.dirty else ''}{RESET}")
        _story(run)

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
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "ollama: model names to benchmark (default the tier pair). "
            "hosted: tiers or model names to run as one-shot baselines (default 'strong')."
        ),
    )
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
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help=(
            "run the suite N times and aggregate. A task counts as passed only if "
            "it passes every run. Use N>=3 before believing anything."
        ),
    )
    args = parser.parse_args()

    if args.suite == "history":
        raise SystemExit(show_history())

    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")

    if args.suite == "sprint":
        runs = bench_sprint(contract=args.contract, repeat=args.repeat)
    elif PROVIDER == "ollama":
        # The local arm: raw model prompts over the Ollama HTTP API, for tok/s and
        # GPU placement. Default to the tier pair.
        runs = bench_models(args.models or DEFAULT_MODELS, repeat=args.repeat)
    else:
        # The hosted arm: the metered one-shot baseline. Default to the STRONG tier —
        # "skip the pipeline, just run the good model once" is the alternative the
        # cost thesis is measured against.
        runs = bench_oneshot(args.models or ["strong"], repeat=args.repeat)

    raise SystemExit(report(runs, args.check))


if __name__ == "__main__":
    main()
