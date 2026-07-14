# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are cut with `python scripts/release.py {patch|minor|major}`, which
bumps the version, moves the entries below out of *Unreleased*, commits, and
tags. See [CONTRIBUTING.md](CONTRIBUTING.md).

## [Unreleased]

## [4.8.0] - 2026-07-14

### Added

- **A one-shot baseline arm — the missing half of the cost comparison.** The
  `models` suite measured tok/s and GPU placement and so refused to run off
  Ollama; on a hosted provider it now runs each task as a metered one-shot
  instead. This is the number the whole "5-8x cheaper" thesis is measured
  against, and it did not exist: the pipeline's `$/task` was being compared to a
  baseline nobody had run on the same provider.

  `runner.run_oneshot(tier, task)` prompts the model once through `llm_factory` —
  same system prompt, same token budget, same governor the sprint uses — and
  grades the result against the same hidden suite. Because both arms now go
  through one meter and one tokenizer, `1shot:strong@anthropic` and
  `hive+contract@anthropic` are finally the same kind of number, and the ratio
  between them is a subtraction rather than the extrapolation it had been. The two
  stay on separate history subjects — a one-shot and a pipeline are not the same
  system under test — so the comparison is the reader's to make; the suite's job
  is to make both numbers real.

  `HIVE_PROVIDER=anthropic python scripts/bench.py models` defaults to the strong
  tier (the "just run the good model once" alternative); `--models fast strong`
  measures the haiku one-shot too. The whole arm runs in one event loop against
  one cumulative governor, so `HIVE_MAX_USD` bounds the total and a breach records
  the repeats that completed. `bench.py models` stays Ollama-native for tok/s and
  GPU placement; only the hosted path is the new one-shot.

### Fixed

- **The strong tier was completely non-functional on the anthropic provider —
  two bugs, found by actually running it.** The first paid run surfaced both;
  neither was visible in any test because the strong tier had never made a real
  API call.

  1. **fable-5 rejects `temperature` with a 400.** `_ANTHROPIC_KWARGS` sent
     `temperature=0.1` for every purpose, but the Claude 5 family and Opus 4.7+
     removed sampling parameters — `temperature is deprecated for this model`. So
     every task that escalated haiku → fable died on the request, scoring a
     capability failure (`flatten` 0/3, `word_wrap`/`roman` flaky) that was really
     a config error. haiku-4-5 (the fast tier) is a 4.5 model and still accepts
     `temperature`, which is why only escalations failed. `llm_factory` now strips
     the sampling params for the models that reject them and leaves the fast tier —
     and all the measured Ollama tuning — untouched.

  2. **fable-5's response is a block list, not a string.** fable-5 has thinking
     always on, so langchain returns `.content` as a LIST (a thinking block plus
     the text block). `_extract_clean_code` runs `re.findall` on it — a TypeError
     on a list — and the one-shot arm's `str(content)` buries the fenced code in a
     repr. Either way the code was unreachable, so fable would have failed
     extraction even once the 400 was fixed. `core.utils.flatten_message_text`
     joins the text blocks; both the pipeline editor and the one-shot bench arm
     flatten before extracting.

  The pipeline arm's headline `6/9` from the first paid run is therefore an
  undercount: three of the "failures" were the strong tier never running. Re-run
  after this fix for the real number.

Three findings from a five-agent adversarial audit (2026-07-14), each a way the
system spends real money on the anthropic provider without the guard that is
supposed to stop it.

- **The benchmark discarded a whole paid run when it hit the budget cap.** The
  bench's `run_sprint` (`bench/runner.py`) catches `except Exception`, but
  `BudgetExhausted` is a `BaseException` by design — so a budget stop escaped it,
  unwound through `asyncio.run`, and killed the process before a single number was
  recorded. On a 30-sprint `sprint --contract --repeat 3` against Claude, tripping
  the default `$5` cap on the last sprint threw away the other 29. The bench now
  catches it, records every repeat that ran to completion, and discards only the
  repeat the breach interrupted — never aggregating tasks with uneven sample
  counts, which would reintroduce the order-dependent scoring the suite exists to
  prevent. Zero complete repeats records nothing and says so.

- **A generation exception in the editor never advanced the retry counter.**
  `async_editor_node`'s exception exit set `editor_error` but, unlike its
  empty-extraction sibling, did not bump `editor_retries` — so `MAX_RETRIES` was
  unreachable on that path. The repeat-error fingerprint could not cover for it
  either: it matches on error *text*, and a hosted-API error carries a varying
  request id, so consecutive failures hash differently and never trip it. A
  persistent generation exception (far likelier on `anthropic` than on local
  Ollama) looped silently to `RECURSION_LIMIT`, burning real tokens on every
  cycle and never reaching the human gate. The counter now advances on that exit.

- **The governor's fail-open guard was only half closed.** `_tokens_from`
  returned `None` for an absent usage object — counted as unreadable, correct —
  but a *present* usage object whose token sub-keys were renamed or dropped
  upstream reads `(0, 0)` through `.get(..., 0)` and was metered as a genuine
  free call. langchain builds `UsageMetadata` with `getattr(u, "input_tokens", 0)
  or 0`, so one upstream schema change emits exactly this truthy all-zero dict,
  and from then on `HIVE_MAX_USD` never grows while an overnight loop bills the
  night reporting `$0.00`. A completed call always spends input tokens, so `(0,
  0)` is now treated as unreadable, not free; a genuinely one-sided reading (a
  cache-read prompt: 0 input, real output) is still a real reading.

## [4.7.0] - 2026-07-14

### Added

- **The hive runs itself.** `multi-hive --loop` discovers its own work, does it,
  writes down what happened, and repeats — until the backlog is empty or the
  budget is gone. `multi-hive --digest` shows what it did while you were asleep.

  Everything below the loop already existed: the sprint, the two reviewers, the
  escalation ladder, the human gate. What was missing was everything *above* it.
  Every objective the hive had ever run was typed by a human into a REPL, which
  made the human the bottleneck, and the bottleneck was the boring part.

  Four pieces, and the order they landed in is the whole design:

  1. **The governor** (below) — because you do not build the thing that runs
     unattended until the thing that stops it exists.
  2. **`core/journal.py`** — append-only, never cleared, survives the sprint.
     `human_gate_node` has always recorded *why* the hive got stuck; nothing ever
     read it, because `clear_ledger()` deletes the ledger at the start of the next
     sprint. The hive knew how to say "I got stuck here, on this, for this reason"
     and then threw it away. A record of unfinished work is a backlog.
  3. **`discovery.py`** — that backlog, read back. Escalated-and-unresolved sprints
     become the next run's queue.
  4. **`supervisor.py`** — discover, work, journal, repeat.

  **A replay is not a re-run.** `agent_router_node` seeds every fresh task with
  `select_tier(editor_retries=0)`, which returns *fast* — so a rediscovered
  objective would have run on the exact model that already failed it and
  reproduced the identical failure, at machine speed, reported as progress. A
  discovered item therefore carries `tier_floor=STRONG`: the fast model has
  demonstrably failed this task, and another attempt from it is the same bet. That
  is the escalation ladder's own logic, carried across sprints instead of thrown
  away at the end of each one. `HIVE_FORCE_TIER` still outranks it — a benchmark
  that is silently un-pinned by a routing rule is not a benchmark.

  **It provably stops.** Three independent bounds, and they are independent on
  purpose: the governor; the attempt cap (`HIVE_MAX_DISCOVERY_ATTEMPTS`, default 2
  — the original run plus exactly one retry on the tier that has not yet failed);
  and a progress check. The one that actually bites is the third: a *crashing*
  sprint writes no journal record, so the attempt counter never advances, so
  discovery hands back the same item forever — a tight, free, infinite loop that
  spends no tokens and therefore never troubles the governor. The supervisor
  journals crashes itself so the counter is monotonic, and keeps its own in-process
  memory of what it has run so termination does not depend on a disk write landing.

  **`HIVE_MAX_DISCOVERY_ATTEMPTS` is the open door for human review**, held open by
  a counter rather than by anyone's good intentions at 3am. Work that beats the
  ladder is *parked* and surfaced loudly in the digest — a loop that silently stops
  trying looks exactly like a loop with nothing to do, and the difference matters
  enormously to whoever reads it.

  The digest ends by naming one machine-written file and asking you to go read it.
  That is not decoration. Comprehension rot and cognitive surrender have no clever
  engineering fix; the only defense is to read the machine's output and be able to
  explain it, and the digest exists to make that cheap, not to make it optional.

- **A budget governor — the thing that can say stop.** `core/governor.py` meters
  every model call and raises `BudgetExhausted` at a ceiling: `HIVE_MAX_USD`,
  `HIVE_MAX_TOKENS`, `HIVE_MAX_WALL_SEC`, `HIVE_MAX_SPRINTS`. The USD cap defaults
  to **$5 on the anthropic provider** and to unlimited on Ollama, where tokens are
  free and a cap would only surprise people.

  `MAX_RETRIES`, `RECURSION_LIMIT` and the repeat-error fingerprint are per-sprint
  backstops; none of them is a *cost* ceiling, and until now there was no token
  accounting anywhere in the system. That was fine for as long as inference was
  free and a human was watching. It is neither once `HIVE_PROVIDER=anthropic` meets
  an unattended loop — and the escalation ladder is what makes it sharp: haiku is
  $1/$5 per Mtok, fable is $10/$50, and the ladder climbs to the expensive tier
  *precisely when a task is failing*, which is when the loop spends the most and
  produces the least.

  Two properties are load-bearing. **The ceiling is checked before a call, not
  after** — `on_llm_start` raises, `on_llm_end` records — so the governor can
  overshoot by at most one call rather than by an unbounded number; enforcing it
  only on the way out would produce an audit log that faithfully records every
  dollar it failed to prevent. And **an unpriced model is charged at the most
  expensive rate we know**, because a budget guard that fails open is not a budget
  guard.

  The meter hangs off `llm_factory`, the one module permitted to construct a
  client — which is what makes it the only place it *can* hang such that no call
  escapes it.

- **`docs/LOOP_ENGINEERING.md`** — the layer above the graph: what it takes for
  the hive to run with no human pressing Enter, and what it costs when it does.
  Scores the system honestly, including the finding that its judgment layer was
  already ahead of the framework it is scored against, and that the layer above it
  did not exist.

- **`HIVE_PROVIDER` — run the hive on the Claude API instead of local Ollama.**
  `HIVE_PROVIDER=anthropic` swaps the fast/strong tiers for `claude-haiku-4-5` and
  `claude-fable-5`. Default is unchanged: local, free, offline.

  This is a change to exactly one module. `core/llm_factory.py` was already the
  only place a client is constructed — every node asks it for one by
  `(purpose, tier)` — so the backend was a seam long before anything was plugged
  into it. The graph, the retry loop, the escalation ladder and the acceptance
  contracts are all provider-blind. `tests/test_llm_factory.py` fails the build if
  any module ever constructs a client directly, because that is how a seam quietly
  stops being one.

  Each provider gets its own parameter table rather than a shared "neutral" one:
  Ollama speaks `num_predict`/`num_ctx`, Anthropic speaks `max_tokens` and has no
  context knob at all. Mapping one onto the other would be a leaky abstraction
  pretending not to be one — and would silently retune the local models, whose
  numbers are measured.

  Sprint benchmarks are recorded under a provider-tagged subject
  (`hive@anthropic`), so an API run can never contaminate the local trend line.
  `bench.py models` stays Ollama-only and says so: it measures tok/s and GPU
  placement, which mean nothing for a hosted API.

  Note that most of the interesting local engineering — sticky tiers, the 8 GB
  VRAM ceiling, the dropped 14B, the ~23s reload on escalation — is Ollama's
  problem and evaporates here. The tier ratchet stays regardless, because "do not
  downgrade a task that has already failed" is good routing wherever the model
  lives.

- **`HIVE_PLAN_TIER`** — pins the planner and ticket writer to a tier. The plan decides
  what the task IS, and everything downstream executes that paraphrase; a bad ticket
  cannot be rescued by escalating the editor. Defaults to unset: measured on ollama it
  was **1.80x slower for zero quality gain** (the 7B/30B VRAM swap), and it ships as a
  knob rather than a default because of it. On `anthropic` there is no swap and it is
  very likely right — but that is unmeasured, and this project has twice reverted a
  change that was obviously right.

- **`word_stats`** — the first bench task the hive is actually FOR. Two files, and the
  graded module IMPORTS the other, so the model must hold an interface neither file
  states alone. Every other task is one self-contained function — exactly what a
  one-shot prompt is best at, and exactly where a pipeline has nothing to add.

### Changed

- **`HIVE_SANDBOX_TIMEOUT` default is now 30s (was 10s).** The old 10s was 6x
  stricter than the grader's 60s, so correct code whose demo block was merely slow
  died in the hive and passed in the bench — the system rejecting work its own scorer
  would accept, which is the worst direction for a disagreement. But 60s was the
  wrong correction: `suite.grade()` pays its 60s ONCE, on the final artefact, while
  this sandbox runs on EVERY editor attempt. At 60s a model that emits `while True`
  costs 240s per task instead of 40. 30s clears any legitimate demo block and stays
  bounded.

- **The benchmark reports the story, not just the score.** `sprint` now records
  tokens, USD, editor attempts, whether the FIRST file written already passed, and
  whether the sprint ever produced a passing file and then shipped something else.
  "9/9 passed" hides everything that matters: two systems can both score 9/9 while one
  gets it right first time for 4k tokens and the other thrashes for 40k.

  The last of those — `DISCARDED A PASSING ANSWER` — measured a failure mode that was
  structurally invisible, because the bench only ever graded the LAST file on disk. It
  came back **zero**, which is why best-attempt retention was NOT built. Measuring
  before building saved that day.

### Fixed

- **The 7B decided what every task was, and nothing could override it.**
  `sprint_planner` and `ticket_writer` called `get_llm()` with no tier, which silently
  defaults to `fast`. So `HIVE_FORCE_TIER` never reached them and neither did the
  escalation ladder — "the hive on the strong model" was never true. And a ticket-writer
  JSON parse failure does not fail a task, it kills the whole SPRINT: measured killing
  `lru_cache --contract` on three runs out of three. Both nodes route their tier now,
  and the ticket writer retries once on the strong model.

- **The spec never reached the people judging the work.** The editor's terminal
  instruction was `EXECUTE THIS SPECIFIC TASK: <ticket>` — a paraphrase of a summary —
  and the semantic reviewer was never given the objective at all. Every trap in the
  benchmark is exactly the clause a paraphrase drops ("ties are broken alphabetically",
  "touching intervals count as overlapping"). The requirement now goes to the editor
  last and verbatim, and to the reviewer first, as the authority. **Measured: 6/9 → 7/9,
  with `semver` going from a coin flip to 3/3.**

- **The strong model never got a clean shot.** On escalation it inherited the fast
  model's broken code and the order "FIX THE CODE SO IT PASSES" — asked to patch a bad
  draft, while `bench models` hands the same model a blank page and it scores 8-9/9. It
  gets a blank page now; the traceback survives as a warning, not a leash.

- **An escalation cancelled every file behind it.** `human_gate_node` returned
  `task_queue: []`, so one hard file poisoned every easy file queued behind it — in a
  project whose premise is MULTI-file generation. It skips the failed task and keeps
  the queue, with a sticky `sprint_escalated` flag so a sprint that carries on cannot
  quietly report itself CLEAN.

- **Five grader bugs, every one of them favouring the pipeline.** The multi-file task
  scored the one-shot 30B "no code" three times running. It was not "no code" — the
  extractor demanded exact `# FILE:` labels, then only read fenced output (the 30B
  emits none), the grader flattened the workspace layout so `from outputs.tokens import`
  raised ModuleNotFoundError, and the delegation check asserted a literal module name so
  a correct import was failed for its SPELLING. A benchmark whose errors all flatter the
  conclusion is not a benchmark.

- **`clean_workspace()` could delete the user's source tree.** It recursively unlinks
  every `*.py` under `HIVE_WORKSPACE`, which was unvalidated — and the README says
  "Relocate it with `HIVE_WORKSPACE=/some/path`". `HIVE_WORKSPACE=.` would have deleted
  the entire `multi_hive` package. A workspace that is, or contains, this source tree —
  or is `$HOME`, or a filesystem root — is now refused at import.

- **The governor failed open.** `_tokens_from` returned `(0, 0)` for any response it
  could not parse, which `record()` added as $0.00 — indistinguishable from a free call.
  One change to a provider's usage field and the meter reads zero forever while an
  overnight loop bills the night. Unreadable is now counted as unreadable, and
  `HIVE_MAX_UNMETERED` stops the run when a ceiling that DEPENDS on the meter can no
  longer be trusted.

- **One crashed item abandoned the whole backlog.** A crash does not advance the attempt
  counter, so the item reappeared forever — and the supervisor's repeat-detector
  responded by breaking out of the entire loop. Items B and C never ran because item A
  kept crashing. A stuck item is a fact about that item; it is skipped now, not fatal.

- Objective truncation is logged instead of silently cutting mid-character. The sandbox
  no longer disagrees with its own grader. `semantic_reviewer` is no longer invited to
  reject on a save path it structurally cannot see.



- **The benchmark was contaminated, and every sprint number this project has ever
  recorded was taken through a dirty lens.**

  `clear_ledger()` had exactly one caller in the codebase — `cli.py` — so a bench
  run never cleared it. And `get_recent_rejections()` filters by **node name only**:
  no task scoping, no run scoping. It returns the last three failures for that node
  *from the whole file*.

  So the `word_wrap` editor was handed `semver`'s traceback, under the heading
  "PAST RUNTIME/ASSERTION FAILURES — your code ran but produced wrong results. Fix
  the logic", and told to fix a bug in a file it was not writing. The ledger was
  found **221 lines deep, spanning multiple tasks and multiple sessions.**

  Three consequences, each of which invalidates the measurement:

  - **The suite was order-dependent.** Task 4 carried tasks 1–3's failures into its
    context; task 1 carried none. Reordering `TASKS` changed the score.
  - **The repeats were not independent samples.** Run 2 began with run 1's failures
    already in the editor's prompt — which voids the entire point of
    `passed == passed every run`, the rule this benchmark otherwise defends more
    carefully than anything else in it.
  - **A run was not reproducible from a clean checkout.** The score depended on
    ledger residue from whatever was run yesterday.

  And `run_model` never touches the ledger, so the whole penalty fell on `sprint`
  and never on `models` — a one-directional bias in precisely the comparison the
  two suites exist to support. `bench/runner.clean_workspace()` now empties the
  ledger and removes generated `*.py` and `__pycache__` before every task. The
  hive's own records (`bench_history.jsonl` above all) are deliberately left alone.

- **Only the graded artefact was cleaned between tasks.** Everything else survived,
  and the reviewer sandbox runs with `PYTHONPATH=WORKSPACE_DIR` — so a module left
  behind by an earlier task could shadow an import and let a task pass on
  *yesterday's code*. The whole workspace is now cleaned.

- **`--repeat` was silently ignored by the `models` suite.** Every models number
  ever recorded is a single sample, and those samples were being read next to
  sprint's strict pass-every-run aggregate — which is not a comparison, it is a
  category error. `models` now honours `--repeat` and applies the same rule.
  Expect the 30B to settle at **3/4, not the 4/4 that has been quoted**: at commit
  `18e4dab` it already scored 3/4 with `semver` failing. The 4/4 was a coin landing
  heads.

- **The `[CONTRACT GAMED]` detector could cry wolf.** `_aggregate` stored
  `contract_satisfied` with `any()` while `passed` used `all()`, so a row could
  read `contract_satisfied=True, passed=False` assembled from two *different*
  repeats — the exact gaming signature, manufactured out of ordinary flakiness with
  no gaming anywhere. It aggregates with `all()` now. A detector that cries wolf
  gets ignored, and then it is not a detector.

- **The regression gate fired on task-count changes, not regressions.**
  `quality_regression` was `quality_delta < 0` — a raw count difference. Add a task
  to `TASKS` and every later run looks like an improvement; remove one and every
  later run looks like a regression. It is now driven by `regressed_tasks`, matched
  by name over the shared set, so it survives the suite growing or shrinking — and
  when it fires it names the task that broke.


- **The planner could hand the graph a file it was never allowed to write, and the
  hive would spend its entire retry budget failing to.** Model-authored paths were
  validated at the *write* boundary (`safe_path`) and nowhere else — so an illegal
  path was only caught after a full generation had been paid for against it.

  Worse than the wasted inference was where the error went. `reviewer_node` raised
  `FILE SYSTEM ERROR: Path traversal blocked: 'test_add.py'`, that became an
  `editor_error`, and the editor was handed it with *"FIX THE CODE SO IT PASSES."*
  But the code was never wrong. The **path** was wrong, and the path lives in
  `active_file`, which the editor cannot change. No output the model can produce
  fixes it — so every retry regenerated the same file for the same illegal path,
  failed identically, burned `MAX_RETRIES`, and escalated to a human for a problem
  no human was needed for. Observed live on a two-line `add(a, b)`: four
  generations and an escalation, for a task the 7B solves first try.

  `core/utils.normalise_model_path()` is now the **entry** boundary, enforced in
  `ticket_writer` across the whole queue (not just `tasks[0]` — every path in it
  becomes `active_file` eventually, as `semantic_reviewer_node` retires tasks, so
  validating only the first would have moved the bug deeper into the sprint where
  it costs more).

  It widens exactly one case: a **bare filename** has no directory component, so
  the model's intent is not in question — it meant a workspace file and forgot to
  say where — and `outputs/` is deterministic, not a guess. **Everything else
  `safe_path` would refuse is still refused.** A traversal has a directory
  component, so it is not a bare filename, so it falls straight through to
  `safe_path` and is rejected: normalisation must never launder a path, and
  `tests/test_safe_path.py` pins that. Unfixable tickets are dropped and logged; a
  plan with no writable file at all now fails loudly instead of spinning.

  Both outcomes log under `ticket_writer`, which none of the editor's three failure
  feeds read. Handing the editor a routing complaint and telling it to "fix the
  code structure" is the same category error the repeat-error breaker was already
  taught once.

  The ticket-writer prompt now names the failure explicitly too — but the prompt
  already said "paths must start with 'src/' or 'outputs/'" and the model emitted
  `test_add.py` anyway. A prompt is not a guarantee. That is why this is enforced
  in code.


Findings from a multi-agent adversarial audit of the codebase, most-severe first.

- **The ground-truth verifier trusted a subprocess exit code alone.** Both the
  acceptance-contract harness (`contract.py`) and the benchmark harness
  (`bench/suite.py`) wrapped the module import in `except Exception`, which cannot
  catch `SystemExit`. A module that ran `exit()`/`sys.exit()`/`os._exit()` at
  import terminated the harness with returncode 0 **before any assert or hidden
  test ran**, and returncode 0 alone was recorded as a satisfied contract *and* a
  passed benchmark task — a false green on the number that gates every decision.
  The import now catches `BaseException`, and a pass is confirmed by a nonce-tagged
  sentinel line the generated code cannot forge, never by the exit code.

- **`repeat_error` was `True` on the first retry of every failed task**
  (`async_editor_node`), forcing the strong model one attempt early and silently
  disabling `HIVE_ESCALATE_AFTER > 1`. The genuine same-error-twice case already
  escalates in its own early-return block; the flag is now `False` here so
  `editor_retries >= ESCALATE_AFTER_FAILURES` owns escalation as documented.

- **Typing `exit`/`quit` hung the REPL** until the user pressed Enter again. The
  stdin pump ran in a non-daemon `ThreadPoolExecutor` whose `atexit` handler joins
  every live worker — and the worker was always parked in a blocking `readline()`.
  It is now a daemon thread, reaped at interpreter exit.

- **A sandbox timeout could hang the reviewer.** After `proc.kill()`, the
  post-kill `communicate()` had no timeout, so a grandchild inheriting the stdout
  pipe blocked it forever, defeating the advertised hard timeout. The child is now
  its own session leader (`start_new_session`) so a timeout SIGKILLs the whole
  process group, and the drain is bounded.

- **Empty generated code looped to the recursion limit.** An empty extraction took
  the editor's success path, wrote `""`, and slipped past both safety mechanisms;
  the sprint died `FATAL` at `RECURSION_LIMIT` without escalating. It is now routed
  through the normal failure ladder (an `editor_error` and a bumped retry count).

- **The sandbox docs claimed write confinement it does not have.** The sandbox
  bounds memory, process count, and (POSIX) per-file size, and strips the
  environment — but does **not** restrict *where* code writes. The docs
  (`reviewer_node`, `core/platform.py`, `docs/ARCHITECTURE.md`) now say so.

- **The editor's "generation failures" feed was contaminated with routing
  notices.** `TIER ESCALATION` / `REPEAT ERROR` lines were logged under the
  editor's node name, so `get_recent_rejections("async_editor_node")` fed router
  logs back to the model as "fix the code structure". They now log under
  `"escalation"` — still in the ledger for the operator, out of the editor's feed.

- **LOOP.md's wall time was always the previous sprint's (0.0 on a fresh
  workspace, 0.0 on every benchmark run).** `retrospector_node` read
  `wall_time_sec` from `metrics.jsonl`, which `cli.py` writes only *after* the
  graph drains. The sprint start is now threaded through state and the elapsed
  time computed in the node.

- **`bench.py`'s regression detector compared a strict `--repeat 3` aggregate
  against a lucky single run,** manufacturing a "quality regression" out of a
  fluke. `repeat` is now part of a run's identity, and `baseline_for()` only
  compares runs of the same repeat count.


## [4.6.0] - 2026-07-13

### Added

- **Human-supplied acceptance contracts.** The fix for the self-authored assert
  problem, and the first thing in this system that removes the model from a job
  it was never able to do.

  The editor wrote the implementation *and* the asserts that judged it. Asked to
  wrap text at width 10, it wrote `assert wrap_text("hello world", 10) ==
  ["hello world"]` — eleven characters into a width of ten. The implementation was
  correct; the assert was arithmetically impossible. So it rejected its own
  working code, burned the retry budget, escalated to the strong model, and woke a
  human. Nothing was wrong with the program.

  Two earlier attempts to fix this inside the model failed, and both are recorded
  on `experiment/acceptance-spec`: a second model writing the spec (measured at
  3.11x slower for zero quality gain, reverted), and removing the self-asserts
  with nothing to replace them (1/4 — a flawed check still beats no check). The
  missing information is not *in* the model. It is in the human. So the human now
  supplies it:

  ```
  Save it to outputs/wrap.py

  ACCEPTANCE outputs/wrap.py
  assert wrap_text("supercalifragilistic", 6) == ["superc", "alifra", "gilist", "ic"]
  ```

  When a file has a contract: the editor is told to write **no asserts at all**;
  `reviewer_node` **imports** the module and executes the human's asserts against
  it, so any test code the model wrote anyway is dead (`__name__` is not
  `"__main__"` under import); and `semantic_reviewer_node` stands down, because an
  exact executable contract outranks a 7B model asked to find fault — and that
  reviewer is the other source of false rejections. A violated assert routes
  through the existing retry and escalation ladder, which is now being fed ground
  truth instead of the model's opinion of itself.

  Measured on `word_wrap` — the task **neither tier could pass**, and the task that
  produced the impossible assert: contract satisfied on the **fast** tier, first
  attempt, no escalation, no human gate, and the artefact passes the hidden suite
  with a real hard-split loop. Not a memorised one — see below.

- **`--objective PATH` and `@path`.** A contract is several lines of Python and
  the REPL reads one line, so an objective carrying one has to arrive from a file.
  `multi-hive --objective examples/wrap_text.md` runs a single sprint and exits,
  which is also what CI wants; `@path` does the same from the prompt.

- **`bench.py sprint --contract`**, and `bench/contracts.py` — acceptance contracts
  for all four bench tasks, derived only from what each prompt already states in
  prose.

  The editor *sees* the contract, so it could hardcode against it. The prompt
  forbids that, but a prompt is not a guarantee — so the benchmark detects it. Every
  literal in `bench/contracts.py` differs from the hidden suite's: the contract
  hard-splits `"supercalifragilistic"` at width 6, the hidden test splits
  `"abcdefghij"` at width 4. Same requirement, different numbers. Which makes the
  hidden suite a working gaming detector: memorised code passes the contract and
  **fails the bench**, and `--contract` shouts `[CONTRACT GAMED]` when a satisfied
  contract meets a failed hidden suite. `tests/test_contract.py` fails the build if
  anyone ever copies an assert across, which is the tempting and fatal shortcut.

  Recorded under a separate subject (`hive+contract`) so it never shares a baseline
  with a plain run: plain asks whether the hive can guess what you meant, contract
  asks whether it delivers what you specified, and comparing those two numbers to
  each other means nothing.

### Fixed

- **`HIVE_MAX_INPUT_CHARS` truncated the whole objective, contract included.** Now
  it caps the prose only. Trimming prose is lossy; trimming a contract mid-assert
  corrupts the one part of the input that is exactly, literally true.

## [4.5.0] - 2026-07-12

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

[Unreleased]: https://github.com/NewMe505/multi_hive/compare/v4.8.0...HEAD
[4.8.0]: https://github.com/NewMe505/multi_hive/compare/v4.7.0...v4.8.0
[4.7.0]: https://github.com/NewMe505/multi_hive/compare/v4.6.0...v4.7.0
[4.6.0]: https://github.com/NewMe505/multi_hive/compare/v4.5.0...v4.6.0
[4.5.0]: https://github.com/NewMe505/multi_hive/compare/v4.4.0...v4.5.0
[4.4.0]: https://github.com/NewMe505/multi_hive/compare/v4.3.0...v4.4.0
[4.3.0]: https://github.com/NewMe505/multi_hive/compare/v4.2.0...v4.3.0
[4.2.0]: https://github.com/NewMe505/multi_hive/releases/tag/v4.2.0
