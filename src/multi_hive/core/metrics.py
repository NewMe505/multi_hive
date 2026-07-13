"""
metrics.py — sprint-level performance baseline.

Deliberately minimal: no new dependencies, no behaviour change to the graph.
Wraps the hive_app stream loop in cli.py to capture

  - wall-clock time per sprint
  - peak RSS (via core.platform, which works on both Windows and POSIX)
  - node execution count (proxy for graph steps / retry-loop length)
  - LLM cache size at sprint end (proxy for cache hit behaviour)

Each sprint is one JSON line in workspace/outputs/metrics.jsonl, so rounds
stay comparable over time without re-deriving anything from logs.
"""
from __future__ import annotations

import json
import time

from multi_hive.config import METRICS_FILE
from multi_hive.core.platform import peak_rss_mb


class SprintMetrics:
    def __init__(self) -> None:
        self._start_time: float | None = None
        self.node_count = 0
        self.wall_time = 0.0
        self.peak_rss_mb = 0.0
        self.llm_cache_size = 0

    def start(self) -> None:
        self._start_time = time.perf_counter()
        self.node_count = 0

    def record_node(self, node_name: str) -> None:
        self.node_count += 1

    def stop(self, llm_cache_size: int = 0) -> None:
        if self._start_time is None:
            raise RuntimeError("SprintMetrics.stop() called before start()")

        self.wall_time = time.perf_counter() - self._start_time
        self.peak_rss_mb = peak_rss_mb()
        self.llm_cache_size = llm_cache_size
        self._append_to_log()

    def _append_to_log(self) -> None:
        METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "wall_time_sec": round(self.wall_time, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "node_count": self.node_count,
            "llm_cache_size": self.llm_cache_size,
        }
        with METRICS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
