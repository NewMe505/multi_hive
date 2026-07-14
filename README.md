# 🐝 multi_hive — Sentinel Prime

An async, self-healing, multi-file code generation hive. A LangGraph state
machine drives a local Ollama model through plan → ticket → route → edit →
review → verify, with a circuit breaker and a human escalation gate when the
retry loop stops converging.

Runs on Windows and Linux.

## Install

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate

pip install -e ".[dev]"
```

Requires a running [Ollama](https://ollama.com) with the model pulled:

```bash
ollama pull qwen2.5-coder:7b
```

## Run

```bash
multi-hive
# or
python -m multi_hive
```

Then type an objective at the prompt:

```
[USER_OBJECTIVE] > Build a DSP pipeline with a sine generator and a delay effect. Save it to outputs/dsp_pipeline.py
```

Or hand it an objective file — which is how you supply an acceptance contract:

```bash
multi-hive --objective examples/wrap_text.md
```
```
[USER_OBJECTIVE] > @examples/wrap_text.md
```

## Acceptance contracts — tell it what "correct" means

By default the editor writes the implementation **and** the asserts that judge
it. That is a conflict of interest, and it has a measured cost. Asked to wrap
text at a width of 10, the model wrote:

```python
assert wrap_text("hello world", 10) == ["hello world"]
```

Eleven characters into a width of ten. Its implementation was correct; its
assert was impossible. So it rejected its own working code, burned the retry
budget, escalated to the strong model, and woke a human. Nothing was wrong with
the program.

The missing information — what *correct* means — is not in the model. It is in
you. So you write it:

```
Implement wrap_text(text, width) which greedily wraps text into lines of at most
`width` characters. A word longer than `width` is hard-split. Runs of spaces
collapse.

Save it to outputs/wrap.py

ACCEPTANCE outputs/wrap.py
assert wrap_text("one two three", 7) == ["one two", "three"]
assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]
assert wrap_text("a  b", 4) == ["a b"]
```

Everything before the `ACCEPTANCE` header is the objective. The block after it is
the contract. When a file has one:

1. The editor is told to write **no asserts at all** — the contract is the test.
2. `reviewer_node` **imports** the module and runs your asserts against it. Any
   test code the model wrote anyway is dead: an imported module's `__name__` is
   not `"__main__"`.
3. `semantic_reviewer_node` stands down. Your executable contract is a stronger
   check than a 7B model asked "is this the right program?", and that reviewer is
   the other source of false rejections.

A violated assert routes through the normal retry and escalation ladder — but now
the loop is being fed ground truth instead of the model's opinion of itself.

A bare `ACCEPTANCE` with no path applies to whichever file the task produces. A
contract that doesn't compile is rejected at the prompt, not forty seconds into a
doomed sprint. Contracts are never truncated by `HIVE_MAX_INPUT_CHARS` — trimming
prose is lossy, trimming a contract is a correctness bug.

**On gaming.** The editor sees the contract, so it could hardcode against it. The
prompt forbids it, and the benchmark checks: `bench/contracts.py` deliberately
uses different literal values than the hidden test suites, so code that memorises
the contract passes the contract and *fails the bench*. `--contract` shouts
`[CONTRACT GAMED]` if that ever happens.

## Layout

```
src/multi_hive/
├── config.py          all tuneables + the workspace paths, resolved once
├── state.py           HiveState — the shared LangGraph TypedDict
├── prompts.py         every system prompt, in one reviewable place
├── contract.py        human-written acceptance contracts: parse, match, harness
├── orchestrator.py    the graph: nodes, edges, and reviewer_logic routing
├── cli.py             REPL, sprint runner, stdin broker
├── core/
│   ├── platform.py    the Windows/Linux seam (RSS, sandbox rlimits)
│   ├── memory.py      the rejection ledger
│   ├── utils.py       safe_path — the write boundary
│   ├── ast_utils.py   code outlines for cross-file prompt context
│   ├── llm_factory.py per-purpose ChatOllama cache
│   ├── metrics.py     per-sprint perf baseline
│   └── loop_audit.py  LOOP.md writer
└── nodes/execution/
    ├── sprint_planner.py         objective  → plan
    ├── ticket_writer.py          plan       → JSON task queue
    ├── agent_router_node.py      injects domain rules, resets per-task state
    ├── async_editor_node.py      generates code; fingerprints repeat errors
    ├── reviewer_node.py          does it RUN? / does it satisfy the contract?
    ├── semantic_reviewer_node.py is it the RIGHT program? (adversarial LLM)
    ├── human_gate_node.py        escalation interrupt
    └── retrospector_node.py      backfill, metrics, LOOP.md
```

## The workspace

Generated code never touches the source tree. Everything the hive produces
goes to `./workspace`:

```
workspace/
├── src/                 generated modules
└── outputs/             generated scripts
    ├── rejection_ledger.jsonl   per-sprint failure memory (cleared each sprint)
    ├── metrics.jsonl            append-only history: wall time, RSS, node count
    └── LOOP.md                  human-readable audit of the last sprint
```

`core.utils.safe_path()` resolves every model-authored path against the
workspace and refuses anything that escapes `workspace/src` or
`workspace/outputs`. Model output is untrusted input; a path like
`../../.ssh/authorized_keys` is a `ValueError`, not a write.

Relocate it with `HIVE_WORKSPACE=/some/path`.

## Configuration

Every tuneable is an environment variable — see `config.py`.

| Variable | Default | Meaning |
|---|---|---|
| `HIVE_PROVIDER` | `ollama` | `ollama` (local) or `anthropic` (Claude API) |
| `HIVE_FAST_MODEL` | per provider | the model backing the *fast* tier |
| `HIVE_STRONG_MODEL` | per provider | the model backing the *strong* tier |
| `HIVE_FORCE_TIER` | — | pin every task to `fast` or `strong`, bypassing the router |
| `HIVE_MODEL` | `qwen2.5-coder:7b` | back-compat alias for the fast Ollama model |
| `HIVE_MAX_RETRIES` | `3` | Retries before escalating to the human gate |
| `HIVE_GATE_TIMEOUT` | `120` | Seconds the gate waits for a human before auto-continuing |
| `HIVE_SANDBOX_TIMEOUT` | `10` | Seconds generated code may run before it is killed |
| `HIVE_WORKSPACE` | `workspace` | Where generated code and artefacts land |
| `HIVE_MAX_INPUT_CHARS` | `4000` | Cap on raw objective text before it hits a context window |

## Running on the Claude API instead of Ollama

The hive is local-first, and the default is unchanged: Ollama, free, offline. But
the models sit behind a single seam — `core/llm_factory.py` — so the backend is
one environment variable:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...

HIVE_PROVIDER=anthropic multi-hive
HIVE_PROVIDER=anthropic python scripts/bench.py sprint    # A/B it on the benchmark
```

| tier | `ollama` (default) | `anthropic` |
|---|---|---|
| fast | `qwen2.5-coder:7b` | `claude-haiku-4-5` |
| strong | `qwen3-coder:30b` | `claude-fable-5` |

Override either with `HIVE_FAST_MODEL` / `HIVE_STRONG_MODEL`.

Nothing else changes. The graph, the nodes, the retry loop, the escalation ladder
and the contracts are all provider-blind — every client is handed to them by
`llm_factory`, and `tests/test_llm_factory.py` fails the build if any module ever
constructs one directly. The ladder still climbs; it just climbs *price* instead
of *parameters*.

Two things worth knowing before you switch:

- **The VRAM story evaporates.** Sticky tiers, the 8 GB ceiling, the dropped 14B,
  the ~23s reload on escalation — all of that is Ollama's problem and none of it
  is Claude's. The tier ratchet stays anyway, because "don't downgrade a task that
  already failed" is good routing wherever the model lives.
- **`bench.py models` is Ollama-only** and will tell you so. It measures tok/s and
  GPU placement, which are meaningless for a hosted API. Compare providers with
  `sprint`, which measures the *system*. Sprint runs are recorded under a
  provider-tagged subject (`hive@anthropic`), so an API run can never contaminate
  the local trend line.

## How the loop protects itself

- **Retry cap.** `MAX_RETRIES` failures on one task routes to the human gate.
- **Repeat-error fingerprinting.** `async_editor_node` hashes each incoming
  error with line numbers and addresses normalised out. The same fingerprint
  twice in a row means the model is fixing symptoms, not converging — escalate
  immediately rather than burn the remaining budget arriving at the same place.
- **Two reviewers.** `reviewer_node` proves the code *runs*. It cannot prove
  the code is the program that was asked for, so `semantic_reviewer_node`
  re-reads the task adversarially and rejects code that runs cleanly while
  implementing the wrong thing. Both are guesses about intent, and both can guess
  wrong — which is what an **acceptance contract** replaces when you supply one.
- **Rejection ledger.** Every failure is logged and fed back into the next
  prompt, split into three feeds — generation, runtime, semantic — because
  each implies a different fix. Merging them produces incoherent retries.
- **Human gate with a timeout.** Escalation prints an alert and waits for
  Enter, then continues anyway after `HIVE_GATE_TIMEOUT`. A headless run
  terminates cleanly instead of hanging forever.

## The sandbox

Generated code is untrusted and it is *executed*. It runs in a subprocess with a
stripped environment (no host secrets), a hard timeout, no write access outside
the workspace, and enforced resource ceilings — on **both** platforms.

Each OS enforces through the only mechanism it has. POSIX applies RLIMITs in a
`preexec_fn` between `fork()` and `exec()`. Windows has no `fork()`, so the
child is assigned to a **Job Object** immediately after spawn.

| ceiling | POSIX | Windows |
|---|---|---|
| memory | 2 GB (`RLIMIT_AS`) | 2 GB (`ProcessMemoryLimit`) |
| process count | 64 (`RLIMIT_NPROC`) | 64 (`ActiveProcessLimit`) |
| file size | 10 MB (`RLIMIT_FSIZE`) | **not enforced** — Job Objects have no equivalent |
| wall clock | `HIVE_SANDBOX_TIMEOUT` | `HIVE_SANDBOX_TIMEOUT` |

Two residual Windows caveats, stated rather than papered over: a runaway *write*
is bounded only by disk, and there is a sub-millisecond window between spawn and
job assignment during which the child is unconstrained (closing it needs
`CREATE_SUSPENDED` on a thread handle `subprocess.Popen` does not expose; in
practice the child spends its first ~50 ms loading CPython).

`tests/test_sandbox_limits.py` proves enforcement by running a 3 GB memory bomb —
sized to *survive* unconfined and *die* confined, so the test cannot pass by
accident on a machine that was merely out of RAM.

## Benchmarks

Two suites, answering different questions. Both grade against **hidden test
suites the model never sees** — probing the edge cases each task implies but
does not spell out, because clean-and-confidently-wrong is the failure that
matters.

```bash
python scripts/bench.py sprint             # the full hive, end to end
python scripts/bench.py sprint --contract  # ...with human-written acceptance contracts
python scripts/bench.py models             # raw models, no graph
python scripts/bench.py history            # the trend, run by run
python scripts/bench.py sprint --check     # exit 1 on a regression
```

**`sprint` is the one to track.** It drives the real graph — planner, tickets,
editor, both reviewers, the retry loop, the escalation ladder — and grades the
file that actually lands on disk. A change that improves the prompts but breaks
the router looks perfect to `models` and terrible here.

`--contract` is recorded under a **separate subject** (`hive+contract`), so it
never shares a baseline with a plain run. The two ask different questions — plain
asks *can the hive guess what I meant*, contract asks *when I say exactly what I
want, does it deliver* — and comparing the two scores to each other means
nothing. Comparing each to its own history means everything.

Every run is recorded against the current git commit in
`workspace/outputs/bench_history.jsonl`, so a regression can be traced to the
change that caused it. Runs on a dirty tree are recorded but never used as a
baseline — a benchmark of uncommitted code cannot be reproduced.

`--check` fails on any **quality** drop (code that used to be correct and is not
any more — no tolerance for that) and on a **speed** drop beyond 25% (local
inference is noisy; a tighter gate would fire on thermal throttling and be
ignored within a week).

## Tests

```bash
pytest
```

`test_graph.py` is the regression guard for how this project was found: the
orchestrator imported `nodes.execution.*`, nothing lived there, and every
entrypoint died on ImportError before a single node ran.
