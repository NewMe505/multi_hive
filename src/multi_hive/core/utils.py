"""
utils.py — the write boundary.

Every file the hive writes goes through safe_path(). Model-authored paths are
untrusted input: an LLM that emits "../../.ssh/authorized_keys" must be
refused, not obeyed. safe_path() resolves relative paths against the workspace
and rejects anything landing outside workspace/src or workspace/outputs.
"""
from __future__ import annotations

from pathlib import Path

from multi_hive.config import ALLOWED_DIRS, WORKSPACE_DIR
from multi_hive.core.memory import log_rejection

PathLike = str | Path


def safe_path(p: PathLike) -> Path:
    """
    Resolves `p` inside the workspace and validates it against ALLOWED_DIRS.

    Relative paths (what the LLM emits — "outputs/main.py") resolve against
    WORKSPACE_DIR, so the caller's working directory cannot change where
    generated code lands. Absolute paths are permitted only if they already
    point inside an allowed directory.

    Raises ValueError on traversal; callers treat that as a node failure.
    """
    if not p:
        raise ValueError("Empty path provided")

    candidate = Path(p)
    resolved = (candidate if candidate.is_absolute() else WORKSPACE_DIR / candidate).resolve()

    for allowed in ALLOWED_DIRS:
        if resolved == allowed or allowed in resolved.parents:
            return resolved

    raise ValueError(f"Path traversal blocked: {str(p)!r}")


def flush_file(filepath: PathLike, content: str) -> Path:
    """Canonical write: validate, create parents, write utf-8. Returns the path."""
    abs_path = safe_path(filepath)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")
    return abs_path


def flush_files(files_dict: dict[str, str], source_node: str = "retrospector_node") -> None:
    """
    Writes a batch of files, isolating failures so one bad path doesn't poison
    the rest.

    Failures route through log_rejection() rather than print(): a dropped file
    printed to a Rich console is invisible in the node flow and never reaches
    the ledger, so the model has no way to learn from it on the next sprint.
    `source_node` identifies the caller, since get_recent_rejections() filters
    by node name.
    """
    for filepath, content in files_dict.items():
        try:
            flush_file(filepath, content)
        except Exception as e:
            log_rejection(source_node, f"DROPPED FILE PAYLOAD '{filepath}': {e}")
