"""
history.py — an append-only record of every benchmark run, keyed by commit.

A benchmark you run once tells you where you are. A benchmark with history tells
you whether the last change helped, and that is the only question that matters
during development.

Every run is one JSON line in workspace/outputs/bench_history.jsonl, stamped with
the git commit, the branch, and whether the tree was dirty. A dirty run is
recorded but never used as a baseline — you cannot reproduce it, so it cannot be
the thing you compare against.

Regressions
-----------
compare() flags two kinds, and treats them very differently:

  quality   Any drop in tasks passed. There is no acceptable amount of this;
            code that used to be correct and now is not is a regression, full
            stop.

  speed     Wall-clock, with a tolerance. Local inference on a laptop is noisy —
            thermal throttling, another model resident in VRAM, the OS deciding
            to index something. A 5% wobble is weather. The default 25% gate
            fires on real changes without crying about the weather.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multi_hive.config import OUTPUTS_DIR

HISTORY_FILE = OUTPUTS_DIR / "bench_history.jsonl"

SPEED_REGRESSION_TOLERANCE = 0.25  # 25% slower than baseline is a regression


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


@dataclass
class Run:
    """One benchmark run: a suite, against a commit, at a moment."""

    suite: str  # "models" | "sprint"
    subject: str  # model name, or "hive"
    tasks: list[dict[str, Any]] = field(default_factory=list)

    commit: str = ""
    branch: str = ""
    dirty: bool = False
    version: str = ""
    timestamp: float = 0.0

    def stamp(self) -> Run:
        from multi_hive import __version__

        self.commit = _git("rev-parse", "--short", "HEAD") or "unknown"
        self.branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
        self.dirty = bool(_git("status", "--porcelain"))
        self.version = __version__
        self.timestamp = time.time()
        return self

    # ── Aggregates ────────────────────────────────────────────────────────────

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tasks if t.get("passed"))

    @property
    def total(self) -> int:
        return len(self.tasks)

    @property
    def wall(self) -> float:
        return sum(t.get("wall_sec", 0.0) for t in self.tasks)

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "commit": self.commit,
            "branch": self.branch,
            "dirty": self.dirty,
            "version": self.version,
            "suite": self.suite,
            "subject": self.subject,
            "passed": self.passed,
            "total": self.total,
            "wall_sec": round(self.wall, 1),
            "tasks": self.tasks,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> Run:
        run = cls(
            suite=raw.get("suite", "?"),
            subject=raw.get("subject", "?"),
            tasks=raw.get("tasks", []),
        )
        run.commit = raw.get("commit", "?")
        run.branch = raw.get("branch", "?")
        run.dirty = raw.get("dirty", False)
        run.version = raw.get("version", "?")
        run.timestamp = raw.get("timestamp", 0.0)
        return run


def record(run: Run) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run.to_json()) + "\n")


def load(suite: str | None = None, subject: str | None = None) -> list[Run]:
    if not HISTORY_FILE.exists():
        return []

    runs: list[Run] = []
    with HISTORY_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                run = Run.from_json(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            if suite and run.suite != suite:
                continue
            if subject and run.subject != subject:
                continue
            runs.append(run)
    return runs


def baseline_for(run: Run) -> Run | None:
    """
    The most recent clean run of the same suite and subject, before this one.

    Dirty runs are skipped: a benchmark of uncommitted code cannot be reproduced,
    so it is worthless as a point of comparison even though it is worth recording.
    """
    candidates = [
        r
        for r in load(run.suite, run.subject)
        if not r.dirty and r.timestamp < run.timestamp and r.total
    ]
    return candidates[-1] if candidates else None


@dataclass
class Comparison:
    baseline: Run
    current: Run
    quality_delta: int
    speed_ratio: float
    regressed_tasks: list[str]
    fixed_tasks: list[str]

    @property
    def quality_regression(self) -> bool:
        return self.quality_delta < 0

    @property
    def speed_regression(self) -> bool:
        return self.speed_ratio > 1 + SPEED_REGRESSION_TOLERANCE

    @property
    def is_regression(self) -> bool:
        return self.quality_regression or self.speed_regression


def compare(current: Run, baseline: Run) -> Comparison:
    was = {t["task"]: bool(t.get("passed")) for t in baseline.tasks}
    now = {t["task"]: bool(t.get("passed")) for t in current.tasks}

    shared = set(was) & set(now)

    return Comparison(
        baseline=baseline,
        current=current,
        quality_delta=current.passed - baseline.passed,
        speed_ratio=(current.wall / baseline.wall) if baseline.wall else 1.0,
        regressed_tasks=sorted(t for t in shared if was[t] and not now[t]),
        fixed_tasks=sorted(t for t in shared if not was[t] and now[t]),
    )
