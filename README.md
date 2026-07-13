# 🐝 multi_hive — Sentinel Prime v4.2

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

## Layout

```
src/multi_hive/
├── config.py          all tuneables + the workspace paths, resolved once
├── state.py           HiveState — the shared LangGraph TypedDict
├── prompts.py         every system prompt, in one reviewable place
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
    ├── reviewer_node.py          does it RUN?      (sandboxed subprocess)
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
| `HIVE_MODEL` | `qwen2.5-coder:7b` | Ollama model tag |
| `HIVE_MAX_RETRIES` | `3` | Retries before escalating to the human gate |
| `HIVE_GATE_TIMEOUT` | `120` | Seconds the gate waits for a human before auto-continuing |
| `HIVE_SANDBOX_TIMEOUT` | `10` | Seconds generated code may run before it is killed |
| `HIVE_WORKSPACE` | `workspace` | Where generated code and artefacts land |
| `HIVE_MAX_INPUT_CHARS` | `4000` | Cap on raw objective text before it hits a context window |

## How the loop protects itself

- **Retry cap.** `MAX_RETRIES` failures on one task routes to the human gate.
- **Repeat-error fingerprinting.** `async_editor_node` hashes each incoming
  error with line numbers and addresses normalised out. The same fingerprint
  twice in a row means the model is fixing symptoms, not converging — escalate
  immediately rather than burn the remaining budget arriving at the same place.
- **Two reviewers.** `reviewer_node` proves the code *runs*. It cannot prove
  the code is the program that was asked for, so `semantic_reviewer_node`
  re-reads the task adversarially and rejects code that runs cleanly while
  implementing the wrong thing.
- **Rejection ledger.** Every failure is logged and fed back into the next
  prompt, split into three feeds — generation, runtime, semantic — because
  each implies a different fix. Merging them produces incoherent retries.
- **Human gate with a timeout.** Escalation prints an alert and waits for
  Enter, then continues anyway after `HIVE_GATE_TIMEOUT`. A headless run
  terminates cleanly instead of hanging forever.

## Sandbox: known platform gap

Generated code runs in a subprocess with a stripped environment, a hard
timeout, and no write access outside the workspace — on both platforms.

On **Linux**, it additionally gets RLIMIT ceilings on address space (2 GB),
file size (10 MB), and process count (64), applied via `preexec_fn`.

On **Windows** those ceilings are **not** applied. `preexec_fn` requires
`fork()`, which Windows does not have; enforcing the equivalent needs a Job
Object through `ctypes`. This is stated plainly rather than faked — if you
intend to run genuinely untrusted code on Windows, that gap is worth closing
first. See `core/platform.py`.

## Tests

```bash
pytest
```

`test_graph.py` is the regression guard for how this project was found: the
orchestrator imported `nodes.execution.*`, nothing lived there, and every
entrypoint died on ImportError before a single node ran.
