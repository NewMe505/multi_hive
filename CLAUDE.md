# multi_hive — working notes

A LangGraph state machine that drives local Ollama models through
plan → ticket → route → write → verify → review → escalate → retrospect.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing the graph or
the node contracts. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup, hooks, and
releases.

## Commands

```bash
pytest                                  # fast; run it constantly
ruff check --fix src tests scripts
multi-hive                              # or: python -m multi_hive

python scripts/bench.py sprint            # end-to-end; track this during development
python scripts/bench.py sprint --contract # ...with human-written acceptance contracts
python scripts/bench.py sprint --check    # exit 1 on a regression (CI gate)
python scripts/bench.py models            # raw models; use when choosing a tier
python scripts/bench.py history           # the trend, run by run

python scripts/release.py patch           # bump, changelog, commit, tag
```

`bench.py sprint` is the number that matters: it drives the real graph and grades
the file that lands on disk, so it catches a change that improves the prompts and
breaks the router. `bench.py models` is blind to all of that. Both grade against
hidden test suites the model never sees.

The venv is at `.venv`. On Windows, `.venv\Scripts\python.exe`.

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
