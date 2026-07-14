# Loop Engineering

## What this document is

[ARCHITECTURE.md](ARCHITECTURE.md) describes the graph — how one sprint runs, and
how it protects itself. This document describes the layer *above* that: what it
takes for the hive to run **without a human pressing Enter**, and what it costs
when it does.

It is scored against a specific framework — the four-layer stack (prompt →
context → harness → loop), the five moves of an autonomous loop, and the failure
modes each missing move produces. The framework is not the point. The point is
that multi_hive is unusually strong at one layer and, until recently, had nothing
at all at the layer above it, and that asymmetry is worth being explicit about.

## The four layers

Each layer widens the unit of concern, and with it the blast radius.

| layer | the question it answers | where multi_hive stands |
|---|---|---|
| 4. **Loop** | how does this run itself, over and over, with no human? | the new work — `governor.py`, `journal.py`, `discovery.py`, `supervisor.py` |
| 3. **Harness** | what tools, and what counts as *done* for one run? | `orchestrator.py`, both reviewers, the sandbox, the acceptance contract |
| 2. **Context** | what does the model see right now? | `core/ast_utils.py` signature outlines, the three failure feeds, `num_ctx` ceilings |
| 1. **Prompt** | what exact words? | `prompts.py` |

Layers 1–3 are mature. Layer 4 did not exist.

## The judgment layer — where multi_hive was already ahead

The framework's chapter on evaluation asks for three things. multi_hive had all
three before this document existed, and then went past them.

- **Separate the generator from the evaluator.** `async_editor_node` writes;
  `reviewer_node` and `semantic_reviewer_node` judge. Different nodes, different
  prompts, none of the author's assumptions carried across.
- **Tune a skeptic.** `semantic_reviewer_node` re-reads the task adversarially,
  looking for code that runs cleanly and answers a different question.
- **Act to verify.** `reviewer_node` does not read the code and form an opinion.
  It *executes* it, in a sandboxed subprocess, and reports what happened.

Then the finding that the framework does not contain: **a skeptic still cannot
supply ground truth.** Asked to wrap text at width 10, the model wrote
`assert wrap_text("hello world", 10) == ["hello world"]` — eleven characters into
a width of ten, an assert that is arithmetically impossible to satisfy. The
implementation was correct. Every reviewer in the system did its job perfectly and
rejected correct code, burned the retry budget, escalated the tier, and woke a
human.

No rearrangement of models fixes that, because the missing information — what
*correct* means — is not in any of them. It is in the human. That is what
`contract.py` exists for, and it is the most important thing in this repository.
See [ARCHITECTURE.md](ARCHITECTURE.md) § The acceptance contract.

**Nothing in the loop work below is allowed to weaken this.** Autonomy raises the
value of the evaluator; it does not lower it.

## The five moves, and what was missing

An autonomous loop has five moves. Skip one and you get a named, predictable
failure.

| move | what it means | before | now |
|---|---|---|---|
| **Discovery** | the system finds its own work | ✗ — `cli.py` blocks on a REPL prompt | `discovery.py` |
| **Handoff** | the task is isolated and passed to an executor | ~ — `agent_router_node`; no parallelism, so no collisions | unchanged |
| **Verification** | an independent check that can say *no* | ✓✓ — the strongest part of the system | unchanged |
| **Persistence** | results survive the session | ~ — per-sprint only | `core/journal.py` |
| **Scheduling** | it repeats without being asked | ✗ — nothing | `supervisor.py` |

The rest of this section is the *why* — what each gap actually looked like in the
code, because the fixes only make sense against the failures they answer.

### Blind Loop — no Discovery

`cli.py` is a `while True` around `broker.readline()`. Every objective the hive
has ever worked on was typed by a human. The hive is an extremely good executor
of a queue that a person has to hand-curate — which means the human is still the
bottleneck, and the bottleneck is the boring part.

### Amnesiac Loop — Persistence that does not persist

This one was subtle, and the code already contained the fix's raw material.

`human_gate_node.py` writes a *structured* escalation record — task, file, retry
count, repeat-error fingerprint, timestamp — as JSON, into the rejection ledger.
It is exactly the record a discovery step would want.

And `cli.py` calls `clear_ledger()` at the top of every sprint, which **deletes
it**.

So the hive already knew how to say "I got stuck here, on this, for this reason"
— and then threw that knowledge away before the next run could read it. `LOOP.md`
is overwritten each sprint. `metrics.jsonl` is append-only but records *timings*,
not *what was learned*. Nothing survived a sprint as knowledge.

This is why the journal has to be built **before** discovery: escalation-driven
discovery reads a record that, until now, did not outlive the sprint that wrote
it.

### Manual Loop — no Scheduling

The loop runs when a human runs it, and stops when their attention wanders.

### Tangled Loop — not a live problem

Parallel agents colliding in one working directory is a real failure, and the
usual answer is a git worktree per agent. multi_hive is sequential and
single-agent, so it does not have this problem, and worktrees are **not** being
built for it. A defense against a failure you do not have is just more code to
maintain. If handoff ever goes parallel, revisit this line.

## The four silent costs

A loop that runs all night accrues four debts, and they reinforce each other.

### Token blowout — the one with teeth

The per-sprint backstops are genuinely good: `MAX_RETRIES`, `RECURSION_LIMIT`, and
the repeat-error fingerprint that escalates the moment the model starts fixing
symptoms instead of converging.

None of them is a *cost* ceiling. There was no token accounting anywhere in the
system, and that was fine for exactly as long as the hive ran on local Ollama,
where inference is free and a human is watching.

`HIVE_PROVIDER=anthropic` changed the arithmetic and nobody noticed:

| tier | model | $/Mtok in | $/Mtok out |
|---|---|---|---|
| fast | `claude-haiku-4-5` | $1.00 | $5.00 |
| strong | `claude-fable-5` | $10.00 | $50.00 |

The escalation ladder — the mechanism that is *supposed* to fire on failure — is
a 10× cost multiplier. A task that cannot be solved escalates to the strong tier
and retries there, which is precisely the situation where the loop spends the most
money and produces the least value. Add a scheduler on top of that and an
overnight run is unbounded spend with nothing in the system able to say stop.

**This is why the governor lands before the scheduler.** Not as a matter of taste:
you do not build the thing that runs unattended until the thing that stops it
exists. `core/governor.py` meters every call and raises `BudgetExhausted` at the
ceiling — and it checks *before* a call, not after, because a check that fires
after the tokens are spent is an audit log, not a cap.

**Measured, once the meter existed (2026-07-14, `--repeat 3` strict).** The 10×
escalation multiplier is real per-escalation, but the whole-suite arithmetic is the
opposite of alarming, because most tasks never escalate:

| arm | quality | $/9-task pass | tokens/pass |
|---|---|---|---|
| pipeline (haiku + contract, escalating) | **9/9** | **$0.135** | 3,374 |
| one-shot (fable-5 on every task) | 6/9 | $0.619 | 2,746 |

The pipeline spends **1.84× the tokens** — the whole graph, every node metered —
at **0.22× the cost**, because that token mix is mostly the 10×-cheaper fast model.
Net: **~4.6× cheaper and higher quality** than just running the strong model once.
The escalation ladder pays for itself; running everything on the expensive tier is
the expensive option. Note this only holds on a *paid* provider — on free Ollama
the honest advice is still "skip the pipeline, run the 30B one-shot," so the loop
layer's economic case is provider-dependent, and now it is measured rather than
assumed. The first paid run was also what surfaced that the strong tier had never
actually worked on `anthropic` (two config bugs, `v4.8.0`): the meter earning its
keep before the loop ran unattended is exactly the point.

### Verification debt

Output that nobody checked, piling up in the gap between "the code runs" and "the
code is right." multi_hive's defense here is unusually strong and predates this
work — but the defense is only as good as the *contract*, and contracts are
written by humans. A loop that generates work faster than a human writes
contracts is manufacturing verification debt, quietly. Watch the ratio.

### Comprehension rot & cognitive surrender

The codebase grows; the mental map does not. Eventually the operator stops having
an opinion about the code and just reads the green checkmark.

There is no clever engineering fix for this. The defense is procedural and it has
to be deliberately inconvenient: **read the machine's output on a schedule, at
random, and be able to explain it.** The digest command exists to make that cheap,
not to make it optional. If you find yourself approving sprints you have not read,
the loop is now writing code that nobody in the world understands, and it is doing
it at machine speed.

## Why the loop terminates

An autonomous loop that cannot prove it stops is not a feature, it is an incident.
Three independent bounds, and the independence is the point:

1. **The governor.** Checked before every sprint, and before every model call
   inside it.
2. **The attempt cap.** `MAX_DISCOVERY_ATTEMPTS` (default 2 — the original run,
   plus exactly one retry on the tier that has not yet failed it). Every sprint
   either resolves a work item or burns one of its attempts, so the backlog is
   strictly finite.
3. **The progress check.** If a discovery round completes no work, stop.

The third exists because of a failure the first two cannot see. **A crashing sprint
writes no journal record.** No record means the attempt counter never advances,
which means discovery hands back the same work item on the next pass — forever, in
a tight loop that spends no tokens and therefore never trips the governor. Free,
silent, and infinite.

So the supervisor journals crashes itself, explicitly, and keeps its own
in-process memory of what it has run — because journal writes are best-effort
(they swallow `OSError`, since a sprint that did real work and then died over its
own bookkeeping would be a bad trade), and termination must not depend on a disk
write landing.

`tests/test_supervisor.py` is a file of termination proofs. Two of them hang
outright if the corresponding guard is removed. That is deliberate.

## What the loop is not allowed to decide

The loop executes intent. It does not choose it.

Everything the hive is good at — routing, retrying, escalating, rejecting — is
downstream of a human deciding what *correct* means. The acceptance contract is
that decision, written down. Discovery can find work; the governor can stop it;
the journal can remember it. None of them can want anything.

Build the loop like someone who intends to stay the engineer.
