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

# DOS device names. Win32 resolves these INSIDE a real directory, so
# workspace/outputs/NUL stats fine, reads empty, and swallows every write — a
# "file" that compiles (empty source), runs (empty program), exits 0, and passes.
# Checked on every platform: the hive is cross-platform, and a workspace written on
# Linux can be read on Windows.
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


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

    raw = str(p).strip().replace("\\", "/")

    # An NTFS alternate data stream. `outputs/wrap.py:evil` writes to a hidden
    # stream and leaves wrap.py itself zero bytes — a file that "exists", compiles
    # (empty source), runs (empty program), and passes.
    if ":" in raw:
        return None

    parts = PurePosixPath(raw).parts

    # The one widened case: a bare filename, and not "." or "..".
    if len(parts) == 1 and parts[0] not in (".", ".."):
        raw = f"{_DEFAULT_DIR}/{parts[0]}"

    # Windows rewrites a component with a trailing dot or space by silently
    # stripping it, so `outputs/...` resolves to the outputs DIRECTORY. safe_path
    # then approves it (it really is inside the workspace), flush_file tries to
    # write_text over a directory, and the PermissionError arrives at the editor as
    # a code failure it structurally cannot fix — the exact unwinnable retry loop
    # this function exists to prevent.
    #
    # And a DOS device name resolves *inside* a real directory on Win32:
    # `workspace/outputs/NUL` stats fine, reads empty, and swallows every write. A
    # file written there is discarded, py_compile sees empty source and passes, the
    # sandbox runs an empty program and exits 0 — and the sprint is journalled CLEAN
    # having produced nothing at all.
    for part in PurePosixPath(raw).parts:
        if part != part.rstrip(". "):
            return None
        if part.split(".")[0].upper() in _WINDOWS_DEVICE_NAMES:
            return None

    try:
        resolved = safe_path(raw)
    except ValueError:
        return None

    # A path that resolves onto an existing directory is not a file the hive can
    # write, and pretending otherwise routes a PermissionError into the code-retry
    # loop.
    if resolved.is_dir():
        return None

    # Return the CANONICAL workspace-relative path, not the string the model typed.
    #
    # This used to return `raw`, and the docstring above claimed it was canonical.
    # It was not. `outputs/lru.py`, `./outputs/lru.py`, `outputs//lru.py` and
    # `outputs/sub/../lru.py` all name the same file on disk and all came back as
    # four different strings — and this string is the KEY for everything:
    #
    # - `_collapse_by_file` keys on it, so two tickets for one file were not merged,
    #   resurrecting the 4x-regeneration bug that function exists to kill;
    # - `project_files` keys on it, so the editor's second ticket read back "" and
    #   rewrote a passing file from scratch;
    # - `contract_for()` matches on it, so an ACCEPTANCE contract keyed under
    #   `outputs/lru.py` was NOT FOUND for `./outputs/lru.py`. reviewer_node then
    #   took the no-contract branch, ran the file, got exit 0, and passed. The
    #   sprint was journalled CLEAN and the human's asserts never ran — on the one
    #   input this codebase calls "exactly, literally true".
    #
    # Deriving it from safe_path's resolved result is what makes it canonical for
    # real: `..` is collapsed, `//` is collapsed, `./` is gone, and it is the path
    # the file will actually be written to.
    return resolved.relative_to(WORKSPACE_DIR).as_posix()


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
