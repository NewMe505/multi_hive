"""
utils.py — the write boundary.

Every file the hive writes goes through safe_path(). Model-authored paths are
untrusted input: an LLM that emits "../../.ssh/authorized_keys" must be
refused, not obeyed. safe_path() resolves relative paths against the workspace
and rejects anything landing outside workspace/src or workspace/outputs.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

from multi_hive.config import ALLOWED_DIRS, WORKSPACE_DIR
from multi_hive.core.memory import log_rejection

PathLike = str | Path

# Where a bare filename lands. The model routinely emits "test_add.py" with no
# directory at all; see normalise_model_path().
_DEFAULT_DIR = "outputs"


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


def normalise_model_path(p: PathLike) -> str | None:
    """
    Coerces a model-authored path into a legal workspace path, or returns None if
    it cannot be made legal without guessing.

    This is the *entry* boundary. safe_path() is the *write* boundary, and by the
    time a path reaches it the hive has already paid for a full generation against
    it — so a path that safe_path() will refuse must never get that far.

    Why this exists
    ---------------
    The ticket writer's prompt already says "All file paths must start with 'src/'
    or 'outputs/'", and the model emits `"test_add.py"` anyway. A prompt is not a
    guarantee, which is a lesson this codebase has learned before and paid for
    again here:

        active_file = "test_add.py"      <- ticket_writer accepts it, unchecked
        editor generates a full file     <- inference paid for
        reviewer calls safe_path()       <- ValueError: Path traversal blocked
        -> editor_error, retry, "FIX THE CODE SO IT PASSES"

    But the code was never wrong. The *path* is wrong, and the path lives in
    `active_file`, which the editor cannot change. No output the model can produce
    fixes this, so every retry regenerates the same file for the same illegal path
    and fails identically — MAX_RETRIES full generations burned on a task the model
    is structurally incapable of solving, then an escalation to a human for a
    problem no human was needed for. Observed live, on a two-line `add(a, b)`.

    What is normalised, and what is refused
    ---------------------------------------
    A **bare filename** — no directory component at all — is unambiguous. The model
    meant a file in the workspace; it just forgot to say where. `outputs/` is the
    only sensible place, and putting it there is deterministic, not a guess.

    **Everything else that safe_path() would refuse is refused here too.** This
    function must never launder a traversal: `../../.ssh/authorized_keys` has a
    directory component, so it is not a bare filename, so it falls straight through
    to safe_path() and is rejected. `outputs/../../etc/passwd` likewise. The
    normalisation widens exactly one case — the one where the model's intent is not
    in question — and nothing else.

    Returning None rather than raising: the caller drops the ticket and logs it.
    One bad ticket should not kill a sprint whose other tickets are fine.
    """
    if p is None or not str(p).strip():
        return None

    # Windows separators, POSIX semantics. The model emits both, and the state
    # dict is keyed by this string, so it has to be canonical either way.
    raw = str(p).strip().replace("\\", "/")
    parts = PurePosixPath(raw).parts

    # The one widened case: a bare filename, and not "." or "..".
    if len(parts) == 1 and parts[0] not in (".", ".."):
        raw = f"{_DEFAULT_DIR}/{parts[0]}"

    try:
        safe_path(raw)
    except ValueError:
        return None

    return raw


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
