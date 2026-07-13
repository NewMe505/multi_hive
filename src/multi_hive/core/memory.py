"""
memory.py — the rejection ledger.

Every node failure is appended here as JSONL. The editor reads its own past
failures back out on retry, which is what stops the model repeating a mistake
it already made this sprint. Cleared at the start of each sprint.

All file I/O is explicitly utf-8: the ledger carries tracebacks and prompt
fragments, and Windows would otherwise default to cp1252 and raise
UnicodeEncodeError on the first non-ASCII byte.
"""
from __future__ import annotations

import json

from multi_hive.config import LEDGER_FILE

_MAX_ERROR_CHARS = 500


def _ensure_ledger_dir() -> None:
    LEDGER_FILE.parent.mkdir(parents=True, exist_ok=True)


def clear_ledger() -> None:
    """Wipes the memory of past failures at the start of a new sprint."""
    _ensure_ledger_dir()
    LEDGER_FILE.write_text("", encoding="utf-8")


def log_rejection(node_name: str, error_msg: str) -> None:
    """Logs a failed attempt as JSONL so the LLM doesn't repeat the same mistake."""
    _ensure_ledger_dir()
    with LEDGER_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"node": node_name, "error": error_msg}) + "\n")


def get_recent_rejections(node_name: str, limit: int = 3) -> str:
    """
    The last `limit` failures logged by `node_name`, newest last, each capped
    at 500 chars so a single 60KB traceback cannot eat the editor's context
    window on its own.
    """
    if not LEDGER_FILE.exists():
        return ""

    rejections: list[str] = []
    with LEDGER_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("node") != node_name:
                continue
            error_text = entry.get("error", "")
            if len(error_text) > _MAX_ERROR_CHARS:
                error_text = error_text[:_MAX_ERROR_CHARS] + "... [TRUNCATED]"
            rejections.append(error_text)

    if not rejections:
        return ""

    return "\n---\n".join(rejections[-limit:])
