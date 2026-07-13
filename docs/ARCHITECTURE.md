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

## The acceptance contract

Both reviewers are guesses about intent, and both can guess wrong. Underneath
them sits a deeper problem that no amount of reviewing fixes: **the editor writes
the implementation and the asserts that judge it.**

Asked to wrap text at a width of 10, the model wrote:

```python
assert wrap_text("hello world", 10) == ["hello world"]
```

Eleven characters into a width of ten. The implementation was correct. The assert
was arithmetically impossible. So `reviewer_node` — doing its job perfectly —
reported a failure, correct code was rejected, the retry budget burned, the tier
escalated, and a human was woken. Nothing was wrong with the program.

Two fixes were tried inside the model and both are recorded on
`experiment/acceptance-spec`, unmerged:

1. **A second model writes the spec first** (`spec_writer_node`), before any code
   exists. Measured: **3.11x slower for zero quality gain**, and the 7B still
   could not produce a usable spec for `word_wrap` — the one task that needed it.
   Reverted.
2. **Delete the self-asserts and trust the semantic reviewer.** Measured: **1/4**,
   down from 3/4. Nothing was checking the code any more. *A flawed check beats no
   check* — that is the durable finding, and it is why contract mode does not
   simply switch verification off.

Both failed for the same reason. The missing information — what *correct* means —
is not in the model, so no rearrangement of models can produce it. It is in the
human.

So the human writes it:

```
ACCEPTANCE outputs/wrap.py
assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]
```

`contract.py` parses the block out of the objective (the planner never sees it —
shown a contract, a planner plans "Step 3: write the tests", which is the exact
job being taken away). Then, for any file that has one:

- **The editor writes no asserts.** A different system prompt entirely, and the
  contract is handed to it as the spec — hiding it would mean asking the model to
  guess the thing the human just took the trouble to write down.
- **`reviewer_node` imports the module** and executes the human's asserts against
  it, instead of running the model's script. Under import, `__name__` is
  `"candidate"` — so any test block the model wrote anyway is dead code and
  *cannot* reject correct code. The mechanism does not depend on the model obeying
  the prompt.
- **`semantic_reviewer_node` stands down.** An exact, executable, human-written
  contract outranks a 7B model asked to find fault — and asking a 7B model to find
  fault is a good way to be handed one. The entire `NEVER REJECT` section of its
  prompt is a scar list of false rejections. It still *advances* the task, because
  something has to (see above).

A violated assert is injected as `editor_error` and routes through the existing
retry and escalation ladder. Nothing new was built for failure handling; the loop
is simply, for the first time, being fed ground truth instead of the model's
opinion of itself.

### Gaming, and how the benchmark detects it

The editor sees the contract, so it could hardcode against its literals. The
prompt forbids it — and a prompt is not a guarantee, so the benchmark checks.

Every literal in `bench/contracts.py` differs from the hidden suite's: the
contract hard-splits `"supercalifragilistic"` at width 6, the hidden test splits
`"abcdefghij"` at width 4. Same requirement, different numbers. That makes the
hidden suite a **gaming detector** — memorised code passes the contract and fails
the bench, and `--contract` reports `[CONTRACT GAMED]` on any task where a
satisfied contract meets a failed hidden suite. `tests/test_contract.py` fails the
build if anyone copies an assert across, which is the tempting and fatal shortcut.

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
environment (no host secrets), a hard timeout, a working directory inside the
workspace, and enforced resource ceilings. All of that holds on both platforms —
but the *mechanism* differs, because Windows has no `fork()`:

- **POSIX** — RLIMITs applied in a `preexec_fn`, between `fork()` and `exec()`.
  The child is also its own session leader (`start_new_session`), so a timeout
  SIGKILLs the whole process group, not just the direct child — otherwise a
  grandchild holding the stdout pipe defeats the hard timeout.
- **Windows** — the child is assigned to a **Job Object** immediately after
  spawn (`core.platform.confine()`), carrying `ProcessMemoryLimit` and
  `ActiveProcessLimit`. The job is `KILL_ON_JOB_CLOSE`, so the handle must
  outlive `communicate()` — closing it early would kill a healthy run.

Memory (2 GB) and process count (64) are enforced on both. What is **not**
enforced is stated rather than faked:

- **Write *location* is not confined.** There is no chroot, mount namespace,
  seccomp, or AppContainer — the ceilings bound memory, process count, and
  (POSIX) per-file size, not filesystem paths. `cwd` is set inside the workspace,
  but that only affects relative-path resolution; code that opens an *absolute*
  path outside the workspace runs unimpeded. This is a resource sandbox, not a
  filesystem jail — treat the models it runs as semi-trusted.
- **File size** is unbounded on Windows (Job Objects have no `RLIMIT_FSIZE`
  equivalent), so a runaway *write* there is bounded only by disk.
- **A sub-millisecond window** between spawn and Job-Object assignment on Windows
  leaves the child briefly unconstrained.

The ground-truth verdict does not trust the exit code alone, either: a module
that calls `exit(0)`/`os._exit(0)` at import terminates the harness with
returncode 0 before any assert or hidden test runs. Both the acceptance-contract
harness (`contract.py`) and the benchmark harness (`bench/suite.py`) confirm a
pass by a nonce-tagged sentinel line the generated code cannot forge, never by
the return code.

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

The quality column comes from `scripts/bench.py models`, which grades against
hidden test suites the model never sees. Both tiers clear the *moderate* tasks,
so routing those to the fast model costs nothing. The strong model wins on
*hard* ones — it implements semver precedence correctly where the 7B silently
ignores build metadata, which is exactly the confident-but-subtly-wrong failure
the ladder exists to catch. Neither model passes `word_wrap`; escalation is not
magic, and that is what the human gate is for.

That last sentence held until the acceptance contract. Given one, the **fast**
model passes `word_wrap` on its first attempt, with no escalation and no human
gate — because it was never the model that could not write the wrapper. It was
the model that could not tell whether it had. Escalating to a bigger model is the
right answer to *the code is wrong*; it is an expensive non-answer to *the test is
wrong*, and the ladder cannot tell those apart on its own.

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
| `contract.py` | acceptance contracts: parse, path-match, and the import harness |
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
  rejection reuses the existing retry path. It stands down entirely (auto-PASS,
  advance) when `contract_satisfied` is `True`.
- `contracts` is read-only for every node; the entrypoint parses it once from the
  objective. `contract_satisfied` is written only by `reviewer_node` and reset per
  task by `agent_router_node` — a `True` left over from the previous file would
  retire a task on the strength of a different file's passing grade.
- `human_gate_node` clears `editor_error` and `current_task` so `reviewer_logic`
  routes to the retrospector after escalation.
- `retrospector_node` deliberately leaves `editor_error` and `editor_retries`
  alone, so their final values remain readable in the end-of-sprint panel.

`current_task` and `editor_error` are `str | None`. Every node guards them. They
were once typed as plain `str`, and the `.lower()` calls on a `None` that
followed were a recurring crash.
