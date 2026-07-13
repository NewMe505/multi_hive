"""
P3-1: Sprint-level performance baseline instrumentation.

Deliberately minimal — no new dependencies, no behavior change to the graph.
Wraps the existing hive_app.stream() loop in run_hive.py to capture the
metrics the optimization baseline (Phase 1) calls for:

  - wall-clock time per sprint
  - peak RSS (resource module — already stdlib, no psutil dependency needed
    for single-process CPU-only usage)
  - node execution count (proxy for graph steps / retry-loop length)
  - LLM cache size at sprint end (proxy for cache hit behavior across purposes)

Each sprint is appended as one JSON line to outputs/metrics.jsonl so rounds
are comparable over time without re-deriving anything from logs.
"""
import json
import os
import resource
import time
from typing import Optional

METRICS_FILE = "outputs/metrics.jsonl"


class SprintMetrics:
    def __init__(self):
        self._start_time: Optional[float] = None
        self._start_rss_kb: Optional[int] = None
        self.node_count = 0
        self.wall_time = 0.0
        self.peak_rss_mb = 0.0
        self.llm_cache_size = 0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        # ru_maxrss is KB on Linux, bytes on macOS — this project targets
        # CachyOS Linux per known deployment context, so KB is assumed.
        self._start_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.node_count = 0

    def record_node(self, node_name: str) -> None:
        self.node_count += 1

    def stop(self, llm_cache_size: int = 0) -> None:
        if self._start_time is None:
            raise RuntimeError("SprintMetrics.stop() called before start()")
        self.wall_time = time.perf_counter() - self._start_time
        # ru_maxrss is a high-water mark for the whole process, so this is
        # "peak since process start," not "peak this sprint" — still useful
        # as a comparable upper bound across sprints in the same run.
        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_rss_mb = peak_rss_kb / 1024
        self.llm_cache_size = llm_cache_size
        self._append_to_log()

    def _append_to_log(self) -> None:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        entry = {
            "wall_time_sec": round(self.wall_time, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "node_count": self.node_count,
            "llm_cache_size": self.llm_cache_size,
        }
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
