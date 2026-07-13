import os
import json

LEDGER_FILE = "outputs/rejection_ledger.jsonl"


def _ensure_ledger_dir() -> None:
    """OPT-DUP2: Single source for the makedirs call that clear_ledger() and
    log_rejection() previously duplicated independently."""
    os.makedirs(os.path.dirname(LEDGER_FILE), exist_ok=True)


def clear_ledger():
    """Wipes the memory of past failures at the start of a new sprint."""
    _ensure_ledger_dir()
    with open(LEDGER_FILE, "w") as f:
        f.write("")


def log_rejection(node_name: str, error_msg: str):
    """Logs a failed attempt as JSONL so the LLM doesn't repeat the same mistake."""
    _ensure_ledger_dir()
    with open(LEDGER_FILE, "a") as f:
        entry = {"node": node_name, "error": error_msg}
        f.write(json.dumps(entry) + "\n")


def get_recent_rejections(node_name: str, limit: int = 3) -> str:
    """Retrieves the last N failures specifically for the requested node, capped at 500 chars each."""
    if not os.path.exists(LEDGER_FILE):
        return ""

    rejections = []
    with open(LEDGER_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if entry.get("node") == node_name:
                        # Truncate to prevent context window bloat
                        error_text = entry.get("error", "")
                        if len(error_text) > 500:
                            error_text = error_text[:500] + "... [TRUNCATED]"
                        rejections.append(error_text)
                except json.JSONDecodeError:
                    pass

    if not rejections:
        return ""

    recent = rejections[-limit:]
    return "\n---\n".join(recent)
