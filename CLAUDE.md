# multi_hive — working notes

A LangGraph state machine that drives local Ollama models through
plan → ticket → route → write → verify → review → escalate → retrospect.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing the graph or
the node contracts, and [docs/LOOP_ENGINEERING.md](docs/LOOP_ENGINEERING.md)
before changing anything that runs unattended — it is the layer above the graph,
and it is where the money is. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup,
hooks, and releases.

## Commands

```bash
pytest                                  # fast; run it constantly
ruff check --fix src tests scripts
multi-hive                              # or: python -m multi_hive

multi-hive --loop                       # unattended: the hive finds its own work
multi-hive --digest                     # what the loop did while you were asleep

python scripts/bench.py sprint            # end-to-end; track this during development
python scripts/bench.py sprint --contract # ...with human-written acceptance contracts
python scripts/bench.py sprint --check    # exit 1 on a regression (CI gate)
python scripts/bench.py models            # raw model, no graph — a one-shot baseline on a hosted provider
python scripts/bench.py history           # the trend, run by run

python scripts/release.py patch           # bump, changelog, commit, tag
```

`bench.py sprint` is the number that matters: it drives the real graph and grades
the file that lands on disk, so it catches a change that improves the prompts and
breaks the router. `bench.py models` is blind to all of that. Both grade against
hidden test suites the model never sees.

The venv is at `.venv`. On Windows, `.venv\Scripts\python.exe`.

## Providers

`HIVE_PROVIDER` is `ollama` (default, local) or `anthropic` (Claude API — needs
`ANTHROPIC_API_KEY` and `pip install -e ".[anthropic]"`).

`core/llm_factory.py` is the **only** module that may construct a model client.
Everything else asks it for one by `(purpose, tier)`. Do not `import ChatOllama`
or `ChatAnthropic` anywhere else — `tests/test_llm_factory.py` scans for it and
fails the build, because a node that builds its own client silently ignores
`HIVE_PROVIDER` and nothing would notice until someone switched.

On `anthropic` the tiers are `claude-haiku-4-5` (fast) and `claude-fable-5`
(strong). Two things that only bite on the paid path, both handled in
`llm_factory`: the Claude 5 family **rejects `temperature`/`top_p`/`top_k`** with a
400 (stripped for the models that refuse them; haiku still gets it), and fable-5
returns `.content` as a **list of blocks** (thinking always on), so
`core.utils.flatten_message_text` flattens before any code extraction.

### Running the cost comparison

```bash
HIVE_MAX_USD=20 HIVE_PROVIDER=anthropic python scripts/bench.py sprint --contract --repeat 3  # pipeline
HIVE_MAX_USD=20 HIVE_PROVIDER=anthropic python scripts/bench.py models  --repeat 3            # one-shot baseline
```

`models` on a hosted provider runs `runner.run_oneshot` — each task once, metered,
graded against the same hidden suite. Recorded as `1shot:<tier>@anthropic`, apart
from the pipeline's `hive+contract@anthropic`; both go through one governor and one
tokenizer, so their `$/task` is directly comparable. **Set `HIVE_MAX_USD`** — the
default $5 will not cover a `--repeat 3` run. Measured 2026-07-14: pipeline 9/9 @
$0.135/pass vs one-shot 6/9 @ $0.619/pass — the pipeline is ~4.6× cheaper and
higher quality (the conclusion inverts vs free Ollama). If Anthropic calls fail
with `CERTIFICATE_VERIFY_FAILED`/`APIConnectionError`, a TLS-inspecting AV/proxy is
in the way — `truststore` is the fix, never `verify=False`.

## The loop (`--loop`)

`supervisor.py` → `discovery.py` → `core/journal.py` → `core/governor.py`. Read
[docs/LOOP_ENGINEERING.md](docs/LOOP_ENGINEERING.md) before touching any of them.

**The governor.** `core/governor.py` meters every model call and raises
`BudgetExhausted` at a ceiling (`HIVE_MAX_USD`, `HIVE_MAX_TOKENS`,
`HIVE_MAX_WALL_SEC`, `HIVE_MAX_SPRINTS`). The USD cap defaults to **$5 on
`anthropic`** and to unlimited on Ollama. Three things are load-bearing:

- **The check runs before the call, not after.** `on_llm_start` raises;
  `on_llm_end` records. Move it to the end and the governor becomes an audit log
  that faithfully records every dollar it failed to prevent.
- **`raise_error = True` on the callback handler.** LangChain swallows exceptions
  raised inside a callback unless the handler opts in. Drop that line and every
  test still passes while the governor does nothing at all in production.
- **`BudgetExhausted` is a `BaseException`.** Every node wraps its model call in
  `except Exception` and turns what it catches into an `editor_error` — which the
  graph *retries*. As an ordinary Exception, a budget stop would be caught,
  retried, refused, caught again, and finally escalated to the human gate blaming
  a model failure that never happened. It is a stop signal, not an error.

An unpriced model is charged at the priciest rate in the table, on purpose. A
budget guard that fails open is not a budget guard.

**Discovery.** Escalated-and-unresolved sprints become the next run's queue. A
discovered item is replayed **byte for byte** (the ACCEPTANCE contract must survive
the round trip or it stops being ground truth) with one change: `tier_floor=STRONG`.
That is not a detail. `agent_router_node` seeds every fresh task with
`select_tier(editor_retries=0)` → *fast*, so without the floor a rediscovered
objective runs on the model that already failed it and reproduces the identical
failure — the loop re-doing known-broken work at machine speed and calling it
progress. `HIVE_FORCE_TIER` still outranks the floor.

**Why the loop terminates.** Three independent bounds. The governor; the attempt
cap (`HIVE_MAX_DISCOVERY_ATTEMPTS`, default 2); and a progress check. The third is
the one that bites: a *crashing* sprint writes no journal record, so the attempt
counter never advances, so discovery hands back the same item forever — free,
silent, infinite, and invisible to the governor because it spends nothing. The
supervisor journals crashes itself, and keeps its own in-process memory of what it
ran so termination does not depend on a disk write landing. `tests/test_supervisor.py`
pins all three. **Do not weaken them.**

**The attempt cap is the open door for human review.** Work that beats the ladder
is parked, not retried, and the digest says so loudly — a loop that silently stops
trying looks exactly like a loop with nothing to do.

## Things that will bite you

**This project runs on Windows and Linux.** Both are supported targets, and the
Windows path is the one that breaks:

- `resource` is Unix-only. Anything needing it goes behind `core/platform.py`.
  Do not `import resource` at module scope anywhere else.
- Windows stdio defaults to **cp1252** and the UI is full of emoji. All file I/O
  is explicitly `encoding="utf-8"`, and `core/console.py` reconfigures stdout
  before the Rich `Console` is constructed. Keep it that way — a stray
  `open()` without an encoding will raise `UnicodeEncodeError` on the first
  traceback containing a non-ASCII byte.
- `preexec_fn` needs `fork()` and does not exist on Windows. The sandbox
  therefore has two halves: RLIMITs via `preexec_fn` on POSIX, and a Job Object
  via `core.platform.confine()` on Windows. The job is `KILL_ON_JOB_CLOSE` — its
  handle must outlive `communicate()`, or you kill a healthy sandbox run.
- `os.environ` keys are upper-cased on Windows. `os.environ["SystemRoot"]`
  silently misses; `SYSTEMROOT` is the one that is there.

**Generated code goes to `./workspace`, never `./src`.** `src/` is the package.
Every model-authored path goes through `core.utils.safe_path()`, which refuses
anything outside `workspace/src` and `workspace/outputs`. Model output is
untrusted input — treat it that way.

**A blocking read in a thread cannot be cancelled.** `cli.py` has exactly one
stdin reader for this reason, shared by the REPL and the human gate. An earlier
version started one per sprint; cancelling the task left the thread parked in
`readline()` forever, and after two sprints the bounded pool was exhausted and
input stopped working. Do not add a second reader.

## Conventions

- Conventional Commits (`feat:`, `fix:`, `chore:` …).
- Every user-visible change gets a `CHANGELOG.md` entry under `## [Unreleased]`
  **as you make it**.
- `pyproject.toml` is the single source of the version. `__version__` reads it
  back from installed metadata — never hardcode it in a second place.
- Hooks are in `.githooks/` and enabled with
  `git config core.hooksPath .githooks`. `pre-commit` runs tests and ruff;
  `post-commit` auto-tags any commit that changes the version.

## State

Nodes communicate only through `HiveState` (`state.py`). Its error-propagation
contract is documented there and is enforced by convention, not by types — read
it before adding a node that writes `editor_error` or `editor_retries`.

**Two invariants the whole loop rests on.** Both were broken at some point, and
both broke it badly:

1. **Only the last gate retires a task.** `semantic_reviewer_node` is the sole
   node that advances the queue. When `reviewer_node` did it — on execution
   success, before intent had been checked — a semantic rejection landed on an
   already-finished task and the sprint looped forever.

2. **A semantic PASS must never erase an execution failure.** If `editor_error`
   is set, the semantic reviewer returns immediately without judging or
   advancing. When it did not, a PASS cleared the error and reset the retry
   counter, and the hive shipped crashing code under a green "Sprint Complete".

`tests/test_loop_terminates.py` pins both. Do not weaken them.

## Acceptance contracts

The editor writing both the code and the asserts that judge it is a conflict of
interest that has cost real sprints — it once rejected its own correct code for
failing an assert that was arithmetically impossible. A human-written
`ACCEPTANCE` block (see `contract.py`) replaces the model's self-asserts with the
human's, and `semantic_reviewer_node` stands down when one passes.

Two things here are not optional and are easy to quietly break:

- **Never leave the code unverified.** Deleting the self-asserts *without* a
  contract to replace them was measured at 1/4, down from 3/4. A flawed check
  beats no check. If a file has no contract, the old self-assert path stays.
- **Never copy an assert from `bench/suite.py` into `bench/contracts.py`.** The
  editor sees the contract and never sees the hidden suite, which is precisely
  what makes the suite a gaming detector. Sharing a literal across the two turns
  the benchmark into an open-book exam. `tests/test_contract.py` fails the build
  if you do it.
