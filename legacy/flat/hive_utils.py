import os
from hive_memory import log_rejection

# SEC-C1: Path Traversal boundary
ALLOWED_DIRS = [os.path.abspath('src'), os.path.abspath('outputs')]


def safe_path(p: str) -> str:
    """Validates paths to prevent arbitrary file write and RCE outside project bounds."""
    if not p:
        raise ValueError("Empty path provided")
    abs_p = os.path.abspath(p)
    if not any(abs_p.startswith(d + os.sep) or abs_p == d for d in ALLOWED_DIRS):
        raise ValueError(f"Path traversal blocked: {p!r}")
    return abs_p


# OPT-DUP1: Centralized canonical write logic
def flush_file(filepath: str, content: str):
    abs_path = safe_path(filepath)
    # TRC-L5a: Handle empty dirname
    os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
    with open(abs_path, "w") as f:
        f.write(content)


def flush_files(files_dict: dict, source_node: str = "retrospector_node"):
    """Flushes files iteratively. Isolates failures so one bad path doesn't poison the batch.

    P2-1: Failures previously went to print() only — invisible inside the Rich
    console flow and never reached the rejection ledger, so the LLM had no way
    to learn from a dropped file on the next sprint. Now routed through
    log_rejection() so it surfaces the same way every other node failure does.

    source_node lets callers identify which node's flush call dropped the file,
    since get_recent_rejections() filters by node name.
    """
    for fp, content in files_dict.items():
        try:
            flush_file(fp, content)
        except Exception as e:
            log_rejection(source_node, f"DROPPED FILE PAYLOAD '{fp}': {e}")
