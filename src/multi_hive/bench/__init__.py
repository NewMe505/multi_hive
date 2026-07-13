"""
The benchmark harness.

Two suites, answering two different questions:

  models   Which model should a tier use? Prompts a model directly, with no
           graph around it, and grades the result. Use when choosing or
           replacing a tier.

  sprint   Is the *system* getting better or worse? Runs the real graph
           end-to-end — planner, tickets, editor, both reviewers, escalation —
           and grades the file that actually lands on disk. This is the one to
           track across development.

Both grade against hidden test suites the model never sees (see suite.py), and
both append to a history keyed by git commit (see history.py), so a regression
can be traced to the change that caused it.
"""
