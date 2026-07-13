# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut with `python scripts/release.py {patch|minor|major}`, which
bumps the version, moves the entries below out of *Unreleased*, commits, and
tags. See [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

### Fixed

- **The hive shipped crashing code under a green "✅ Sprint Complete".** The worst
  failure this system can have: a false success from the thing whose entire job is
  verifying its own output.

  `reviewer_node` runs the code and sets `editor_error` when it crashes. The
  semantic reviewer then ran anyway, judged only *intent*, returned PASS — and a
  PASS advances the queue, which clears `editor_error` and resets
  `editor_retries`. So an opinion about intent **erased an execution failure**.
  The retry counter never climbed, the tier never escalated, and a `semver.py`
  that raised `TypeError` on import was declared a success. Four crashes in a row,
  each laundered away by a semantic thumbs-up.

  The semantic reviewer now returns immediately when `editor_error` is set. You
  cannot approve the intent of a program that does not run.

- **The test suite wrote into the live workspace.** A test mocking a semantic
  rejection put `FAIL: uses OrderedDict, not a linked list` into the real
  rejection ledger, where it sat among genuine entries from an unrelated sprint.
  The same leak would have corrupted `bench_history.jsonl` — the file whose whole
  purpose is to be a trustworthy record. `tests/conftest.py` now points the
  workspace at a temp directory before `multi_hive` is imported.

### Added

- **`scripts/bench.py` — a performance tracker, not just a benchmark.** Replaces
  `bench_models.py`.

  Two suites: `sprint` drives the real graph end-to-end and grades the file that
  lands on disk (this is the one to track — it catches a change that improves the
  prompts and breaks the router, which the raw-model suite is blind to); `models`
  prompts a model directly, for when you are choosing or replacing a tier.

  Every run is recorded against the current git commit, so a regression can be
  traced to the change that caused it. Runs on a dirty tree are recorded but never
  used as a baseline — a benchmark of uncommitted code cannot be reproduced.

  `--check` exits non-zero on any **quality** drop (code that used to be correct
  and is not any more — zero tolerance) or a **speed** drop beyond 25% (local
  inference is noisy; a tighter gate would fire on thermal throttling and be
  ignored within a week).

### Verified live

The full escalation ladder, for the first time, on a semver task the 7B provably
fails:

1. fast tier (`qwen2.5-coder:7b`) wrote code that crashed
2. `TIER ESCALATION: fast → strong (qwen3-coder:30b)` — the retry went to the
   better model, rather than re-buying the same failure
3. the 30B failed too, on the build-metadata rule the benchmark predicted
4. retries hit the cap → **human gate**

No false success. The system tried a better model, and when that was not enough,
it stopped and asked for a person.

## [4.4.0] - 2026-07-12

### Added

- **The Windows sandbox now enforces its ceilings.** Generated code is untrusted
  and it is executed; on Windows it previously ran with no memory or process
  limit at all, because `preexec_fn` requires `fork()`. `core.platform.confine()`
  now assigns the child to a **Job Object** (`ProcessMemoryLimit` 2 GB,
  `ActiveProcessLimit` 64, `KILL_ON_JOB_CLOSE`) immediately after spawn. Memory
  and process count are now capped on both platforms.

  Still not enforced on Windows, and said plainly rather than faked: Job Objects
  have no file-size limit, so a runaway *write* is bounded only by disk; and
  there is a sub-millisecond window between spawn and assignment during which the
  child is unconstrained.

- Hidden-test grading in `scripts/bench_models.py`. The previous gates only
  checked that code compiled, ran, and contained the right function names — every
  model scored full marks, which measured nothing. Grading now runs suites the
  model never sees, probing the edge cases each task implies. It discriminates:

  | task | qwen2.5-coder:7b | qwen3-coder:30b |
  |---|---|---|
  | lru_cache (moderate) | ✓ | ✓ |
  | merge_intervals (moderate) | ✓ | ✓ |
  | semver (hard) | ✗ ignored build metadata | ✓ |
  | word_wrap (hard) | ✗ | ✗ no hard-split |
  | | **2/4** | **3/4** |

  This is the evidence the escalation ladder rested on and previously lacked.

### Fixed

- `release.py` raced its own `post-commit` hook: the hook tags any commit that
  changes the version and fires *during* the release commit, so the script then
  tried to create a tag that already existed and died.
- `__version__` reported a stale version. It read from `importlib.metadata`, but
  an editable install writes that metadata once — the app printed
  `v4.2.0` in its banner immediately after being released as `4.3.0`. In a source
  checkout the adjacent `pyproject.toml` is now the truth.

## [4.3.0] - 2026-07-12

### Added

- **Model tiering / escalation ladder.** Tasks run on a fast model and escalate
  to a strong one when it fails. `core/model_router.py` picks the tier from a
  cheap text-only complexity prior plus the failure history; `core/llm_factory.py`
  now caches clients by `(purpose, tier)`. Chosen from measurements on the target
  machine, not leaderboards — see `scripts/bench_models.py`:

  | model | tok/s | GPU placement | |
  |---|---|---|---|
  | qwen2.5-coder:7b | 54.6 | 100% (4.7/4.7 GB) | **fast tier** |
  | qwen2.5-coder:14b | 11.6 | 61% (6.0/10.0 GB) | dropped |
  | qwen3-coder:30b | 37.0 | 32% (6.1/19.2 GB) | **strong tier** |

  The 14B is the "almost fits" trap: a *dense* model 39% off-GPU, 5× slower than
  the 7B for no measured gain. The 30B is faster despite being twice the size
  because it is mixture-of-experts (~3B active per token).

  The tier is **sticky per task**: the two models do not fit in 8 GB of VRAM
  together, so a design where the small model writes and the big one reviews
  every task would evict and reload on every task, paying a ~23s load twice over.

- `scripts/bench_models.py` — measures tok/s, GPU placement, and code quality
  against **hidden test suites** the model never sees.

### Fixed

- **The sprint loop could not terminate.** Two independent, separately fatal bugs,
  both found by running the thing rather than by reading it:

  1. `reviewer_node` retired a task the moment its code *executed* — popping the
     queue, clearing `current_task`, zeroing `editor_retries`. But
     `semantic_reviewer_node` runs *after* it. A semantic rejection therefore
     landed on an already-finished task: the editor regenerated nothing (`if not
     current_task: return {}`) and the retry cap was unreachable (the counter had
     just been zeroed). Both safety mechanisms failed at once. **Observed live:
     992 identical semantic rejections, zero escalations.** A task is now retired
     only by the last gate to run, which is the only node that knows both
     reviewers passed.

  2. `reviewer_logic` routed an `editor_error` back to the editor even with no
     `current_task` to retry. The editor no-ops, the reviewers no-op, so nothing
     ever bumps `editor_retries` and the cap is never reached — an unkillable,
     completely silent loop. Reachable whenever `ticket_writer` gets unparseable
     JSON back. **Observed live: 10,007 graph steps and an empty ledger.** You
     cannot retry a task that does not exist; it now escalates.

- **The repeat-error circuit breaker was disarmed.** `async_editor_node` cleared
  the error fingerprint whenever *generation* succeeded — but generation nearly
  always succeeds; the failure arrives later, from a reviewer. So the evidence
  that the last attempt failed the same way was wiped every cycle, and the
  breaker never fired for the failures it exists to catch.

- **`ticket_writer` failed silently.** Its JSON-parse failure now reaches the
  rejection ledger; its silence is what made the loop above invisible.

- **A `None` state delta from LangGraph killed the sprint** after all the work
  was already done (`TypeError: argument of type 'NoneType' is not iterable`).

- **The stdin broker deadlocked at EOF.** Two consumers, one EOF marker: whichever
  reached it first swallowed it, leaving the other blocked forever on a queue no
  producer would fill again. EOF is now sticky.

### Changed

- `RECURSION_LIMIT` (default 120) caps graph steps per sprint. LangGraph's 10,007
  default meant a routing cycle burned twenty minutes before surfacing; this fails
  in seconds. A backstop, not a working limit.

### Added

- `scripts/release.py` — bumps the version, dates the changelog, commits, and
  tags. Refuses to run on a dirty tree, on failing tests, or with an empty
  Unreleased section.
- Version-controlled git hooks in `.githooks/`. `pre-commit` runs the tests and
  ruff; `post-commit` auto-tags any commit that changes the version, so no
  released version is ever left unreachable.
- `CONTRIBUTING.md`, and ruff as a dev dependency with lint config.

### Changed

- `__version__` is now read from installed package metadata instead of being
  hardcoded, making `pyproject.toml` the single source of truth. The two
  declarations could previously drift apart.
- Modernised typing throughout (`Dict`/`List`/`Optional` → `dict`/`list`/`| None`).

## [4.2.0] - 2026-07-12

First release as an installable package. The project could not run before this:
`hive_orchestrator.py` imported `nodes.execution.*`, which did not exist, so
every entrypoint died on `ImportError` before a single node executed.

### Added

- `src/multi_hive/` package layout, installable with `pip install -e .`, with a
  `multi-hive` console script and `python -m multi_hive` entrypoint.
- `core/ast_utils.py` — imported by both editor nodes and previously missing
  from the repository entirely.
- `core/platform.py` — the Windows/Linux seam for peak-RSS sampling and sandbox
  resource limits.
- `core/console.py` — shared Rich console with a UTF-8 bootstrap.
- A `./workspace` directory for all generated code and artefacts, kept out of
  the source tree.
- pytest suite (17 tests) covering the path boundary, the outline builder, the
  platform shim, and graph construction.

### Fixed

- **Graph could not be built.** Nodes now live at `multi_hive.nodes.execution.*`,
  which is where the orchestrator always expected them.
- **`resource` is Unix-only.** Three modules imported it at module scope, so the
  project raised `ImportError` on Windows before running. Now behind
  `core/platform.py`.
- **Banner emoji crashed the REPL on Windows.** The console defaults to cp1252,
  which cannot encode `🐝`, so startup died with `UnicodeEncodeError`.
- **stdin listener leaked a blocked thread per sprint.** Cancelling the listener
  task did not unblock the thread parked in `readline()`; two sprints exhausted
  the bounded 2-worker pool and the REPL stopped accepting input. Replaced with
  a single shared reader feeding both the REPL and the human gate.
- **Peak RSS reported 0 MB on Windows.** `GetCurrentProcess()` without an
  explicit `restype` truncates the 64-bit pseudo-handle to `-1`.

### Changed

- Generated code now lands in `./workspace/{src,outputs}` rather than `./src`,
  which is the package itself. A task writing to `src/foo.py` would previously
  have written into the source tree.
- All file I/O is explicitly UTF-8. Windows would otherwise default to cp1252
  and raise on the first non-ASCII byte of a traceback.

### Security

- `safe_path()` resolves every model-authored path against the workspace and
  refuses anything escaping `workspace/src` or `workspace/outputs`. Model output
  is untrusted input.

### Known gaps

- Sandbox RLIMIT ceilings (address space, file size, process count) are applied
  on POSIX only. `preexec_fn` requires `fork()`, which Windows lacks; the
  equivalent needs a Job Object. Generated code is still bounded by the
  subprocess timeout and a stripped environment on both platforms.

[Unreleased]: https://github.com/playa/multi_hive/compare/v4.2.0...HEAD
[4.2.0]: https://github.com/playa/multi_hive/releases/tag/v4.2.0
