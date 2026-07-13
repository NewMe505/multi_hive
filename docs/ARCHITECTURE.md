# Architecture

## What this is

A LangGraph state machine that drives a local LLM through a full development
loop — plan, ticket, route, write, verify, review, escalate, retrospect — and
that treats the model's output as untrusted throughout.

The central assumption is that **a local model will confidently produce wrong
code**, and that the interesting engineering is therefore not in the prompt but
in the loop around it: how failure is detected, what the model is told about the
failure, when retrying stops being worth it, and who gets told when it does.

## The graph

```
sprint_planner → ticket_writer → agent_router_node → async_editor_node
   → reviewer_node → semantic_reviewer_node → ┬→ async_editor_node   (retry)
                                              ├→ human_gate_node     (escalate)
                                              ├→ agent_router_node   (next task)
                                              └→ retrospector_node   (done)

   human_gate_node → retrospector_node → END
```

`reviewer_logic` in `orchestrator.py` is the only branch point. It reads state
*after* both reviewers have run, and decides in this priority order:

| condition | route | why |
|---|---|---|
| `loop_health.escalated` | `human_gate_node` | the loop is cycling, not converging |
| `editor_error` and retries ≥ `MAX_RETRIES` | `human_gate_node` | attempt budget spent |
| `editor_error` | `async_editor_node` | normal retry |
| `current_task` | `agent_router_node` | advance to the next task |
| otherwise | `retrospector_node` | sprint complete |

## Why there are two reviewers

`reviewer_node` runs the generated code in a sandboxed subprocess. It proves the
code **executes** — it parses, it runs, its own asserts hold.

It cannot prove the code is the program that was *asked for*. A model will
happily emit a syntactically valid script that passes its own asserts while
implementing something else entirely: a function under a different name, a file
saved to the wrong path, a requirement quietly dropped. Those asserts are also
written by the same model that wrote the bug.

`semantic_reviewer_node` closes that gap by re-reading the task adversarially
and rejecting code that runs cleanly but answers a different question. Its
verdict is injected as `editor_error`, so a semantic rejection travels back
through the *same* retry loop as a crash — one failure path, not two.

The parse is biased toward PASS: anything that does not explicitly begin with
`FAIL` is treated as a pass. A confused reviewer must not be able to manufacture
an escalation.

### Only the last gate may retire a task

`semantic_reviewer_node` is the **only** node that advances the task queue. This
is load-bearing, and getting it wrong produced a real infinite loop.

`reviewer_node` used to retire the task the moment the code *executed* — popping
the queue, clearing `current_task`, resetting `editor_retries` to 0. But the
semantic reviewer runs *after* it. So a semantic rejection arrived at a task the
graph already considered finished:

- `current_task` was `None`, so `async_editor_node` hit its `if not current_task:
  return {}` guard and **regenerated nothing**;
- `editor_retries` had just been zeroed, so the `MAX_RETRIES` cap was
  **unreachable**;
- the unchanged code passed execution again, zeroing the counter again.

Both independent safety mechanisms failed at once, and the sprint re-validated
identical code forever. Measured in a real run: **992 identical semantic
rejections, zero escalations.**

A task is finished when *both* reviewers pass, and only the last one to run is in
a position to know that. `tests/test_loop_terminates.py` pins this.

## How the loop protects itself

**Retry cap.** `MAX_RETRIES` failures on one task routes to the human gate.

**Repeat-error fingerprinting.** `async_editor_node` hashes each incoming error
with line numbers and memory addresses normalised out. The same fingerprint
twice in a row means the model is fixing symptoms, not converging — so it
escalates *immediately* rather than spending the remaining budget to arrive at
the same place. A plain retry counter cannot see this; it would burn all three
attempts producing the same failure.

**Three failure feeds.** The rejection ledger is read back into the editor's
prompt split into generation failures, runtime failures, and semantic failures —
because each implies a different fix (fix the structure / fix the logic / re-read
the task). Merging them produces incoherent retries where the model does not know
what kind of wrong it was.

**Human gate with a timeout.** Escalation prints an alert and waits for the
operator, then continues anyway after `HIVE_GATE_TIMEOUT`. A headless run
terminates cleanly instead of hanging forever. The failure this prevents is the
worst one: a loop stuck retrying while the human who could fix it is never told.

## The write boundary

Model-authored file paths are untrusted input. Every write in the system goes
through `core.utils.safe_path()`, which resolves the path against the workspace
and refuses anything landing outside `workspace/src` or `workspace/outputs`.
A model that emits `../../.ssh/authorized_keys` gets a `ValueError`, not a write.

Generated code lives in `./workspace`, never in `./src` — `src/` is the package
itself, and a task writing to `src/foo.py` would otherwise land inside
`multi_hive`'s own source.

## The sandbox

`reviewer_node` executes generated code in a subprocess with a stripped
environment (no host secrets), a hard timeout, no write access outside the
workspace, and enforced resource ceilings. All of that holds on both platforms —
but the *mechanism* differs, because Windows has no `fork()`:

- **POSIX** — RLIMITs applied in a `preexec_fn`, between `fork()` and `exec()`.
- **Windows** — the child is assigned to a **Job Object** immediately after
  spawn (`core.platform.confine()`), carrying `ProcessMemoryLimit` and
  `ActiveProcessLimit`. The job is `KILL_ON_JOB_CLOSE`, so the handle must
  outlive `communicate()` — closing it early would kill a healthy run.

Memory (2 GB) and process count (64) are enforced on both. Two things are not,
and are stated rather than faked: Job Objects have no file-size limit, so a
runaway *write* on Windows is bounded only by disk; and there is a
sub-millisecond window between spawn and assignment during which the child is
unconstrained.

### Testing a sandbox honestly

`tests/test_sandbox_limits.py` is worth reading as a cautionary tale. Its first
version ran a 12.8 GB memory bomb and asserted the child died. It passed — while
the sandbox was enforcing **nothing at all**. The child had died because Ollama
was holding 18 GB and the machine genuinely ran out of RAM. A green test that
measured the weather.

The bomb is now 3 GB: over the 2 GB ceiling, far under any real machine's RAM.
It must *survive* unconfined and *die* confined. The unconfined control runs as
its own test, so if the machine is ever too small to hold the bomb, the control
fails loudly instead of letting the confinement test pass for free.

## Model tiering

Models are selected by *purpose* (sampling parameters — a planner needs room to
think, a reviewer emits one deterministic line) **and** by *tier* (which model
runs it). `core/llm_factory.py` caches clients on the `(purpose, tier)` pair;
`core/model_router.py` picks the tier.

Two tiers, chosen by measuring on the target machine — an RTX 5070 Laptop with
8 GB of VRAM — not from leaderboards:

| model | tok/s | GPU placement | hidden tests | |
|---|---|---|---|---|
| qwen2.5-coder:7b | 52.2 | 100% (4.7/4.7 GB) | 2/4 | **fast** |
| qwen2.5-coder:14b | 11.6 | 61% (6.0/10.0 GB) | — | dropped |
| qwen3-coder:30b | 34.2 | 32% (6.1/19.2 GB) | 3/4 | **strong** |

The 14B is the *almost fits* trap: 10 GB into 8 GB of VRAM leaves 39% of a
**dense** model on the CPU, and a dense model touches every parameter for every
token. Five times slower than the 7B, for no measured gain. Dropped.

The 30B is **faster than the 14B despite being twice the size**, because it is
mixture-of-experts — roughly 3B parameters active per token. Bigger model, less
VRAM-bound, faster.

The quality column comes from `scripts/bench_models.py`, which grades against
hidden test suites the model never sees. Both tiers clear the *moderate* tasks,
so routing those to the fast model costs nothing. The strong model wins on
*hard* ones — it implements semver precedence correctly where the 7B silently
ignores build metadata, which is exactly the confident-but-subtly-wrong failure
the ladder exists to catch. Neither model passes `word_wrap`; escalation is not
magic, and that is what the human gate is for.

### Why the tier is sticky per task

The two models cannot both be resident: 4.7 + 6.1 GB against 8 GB of VRAM. Every
tier switch is an eviction and a reload, and the strong model takes up to ~23s to
load.

So the tier is chosen once per task and held — editor and semantic reviewer
alike. The obvious-sounding design, *small model writes, large model reviews
everything*, would ping-pong the two through VRAM and pay that reload twice on
every single task, to scrutinise code the fast model probably got right.

Instead the tier ratchets: a task escalates on failure (or starts strong if it
looks hard), and an escalated task then gets the stronger reviewer **for free**,
because it is already on that tier. That is precisely the task where the extra
scrutiny is warranted — the fast model has already failed it once.

## Module map

| module | role |
|---|---|
| `config.py` | every tuneable, and the workspace paths, resolved once |
| `state.py` | `HiveState` — the shared TypedDict and its contract |
| `prompts.py` | every system prompt, so a prompt change is a one-file diff |
| `orchestrator.py` | graph construction and `reviewer_logic` |
| `cli.py` | REPL, sprint runner, and the single stdin broker |
| `core/platform.py` | the Windows/Linux seam |
| `core/console.py` | shared Rich console + UTF-8 bootstrap |
| `core/memory.py` | the rejection ledger |
| `core/utils.py` | `safe_path` — the write boundary |
| `core/ast_utils.py` | signature outlines for cross-file prompt context |
| `core/llm_factory.py` | per-purpose model cache |
| `core/metrics.py` | per-sprint performance baseline |
| `core/loop_audit.py` | `LOOP.md` writer |

## State contract

`HiveState` is the only thing nodes share. The rules are enforced by convention,
and documented in `state.py` because breaking them silently is easy:

- `async_editor_node` / `reviewer_node` set `editor_error` and bump
  `editor_retries` on failure; clear both on success.
- `agent_router_node` never touches `editor_error`, and always resets
  `editor_retries` — it only runs at the start of a new task.
- `semantic_reviewer_node` injects its FAIL verdict as `editor_error`, so the
  rejection reuses the existing retry path.
- `human_gate_node` clears `editor_error` and `current_task` so `reviewer_logic`
  routes to the retrospector after escalation.
- `retrospector_node` deliberately leaves `editor_error` and `editor_retries`
  alone, so their final values remain readable in the end-of-sprint panel.

`current_task` and `editor_error` are `str | None`. Every node guards them. They
were once typed as plain `str`, and the `.lower()` calls on a `None` that
followed were a recurring crash.
