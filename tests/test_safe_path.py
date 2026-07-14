"""
safe_path is the write boundary: model-authored paths are untrusted input.
These tests are the reason it exists.
"""
import pytest

from multi_hive.config import OUTPUTS_DIR, SRC_DIR, WORKSPACE_DIR
from multi_hive.core.utils import (
    flatten_message_text,
    flush_file,
    normalise_model_path,
    safe_path,
)


def test_flatten_message_text_handles_strings_and_block_lists():
    # Ollama / haiku: already a plain string, unchanged.
    assert flatten_message_text("just text") == "just text"

    # fable-5: a list of blocks. Keep text blocks, drop thinking, join with newlines.
    blocks = [
        {"type": "thinking", "thinking": "reasoning", "signature": "sig"},
        {"type": "text", "text": "line one"},
        {"type": "text", "text": "line two"},
    ]
    assert flatten_message_text(blocks) == "line one\nline two"

    # A bare list of strings, and an empty list, both survive.
    assert flatten_message_text(["a", "b"]) == "a\nb"
    assert flatten_message_text([]) == ""


def test_relative_paths_resolve_into_the_workspace():
    assert safe_path("outputs/main.py") == OUTPUTS_DIR / "main.py"
    assert safe_path("src/dsp.py") == SRC_DIR / "dsp.py"


def test_relative_paths_ignore_the_working_directory(monkeypatch, tmp_path):
    # The LLM emits "outputs/main.py" regardless of where the process was
    # launched from. Where that lands must not depend on the caller's cwd.
    monkeypatch.chdir(tmp_path)
    assert safe_path("outputs/main.py") == OUTPUTS_DIR / "main.py"


@pytest.mark.parametrize(
    "hostile",
    [
        "../../.ssh/authorized_keys",
        "outputs/../../etc/passwd",
        "/etc/passwd",
        "notes.txt",  # outside src/ and outputs/, even inside the workspace
        "workspace/outputs/x.py",  # would double-nest the workspace
    ],
)
def test_traversal_is_blocked(hostile):
    with pytest.raises(ValueError):
        safe_path(hostile)


def test_empty_path_is_rejected():
    with pytest.raises(ValueError):
        safe_path("")


def test_flush_file_writes_utf8_and_creates_parents(tmp_path, monkeypatch):
    written = flush_file("outputs/nested/deep/x.py", "# ünïcödé ✅\n")
    assert written.read_text(encoding="utf-8") == "# ünïcödé ✅\n"
    assert written.is_relative_to(WORKSPACE_DIR)
    written.unlink()


# ── normalise_model_path — the ENTRY boundary ────────────────────────────────
#
# safe_path is the write boundary, and by the time a path reaches it the hive has
# already paid for a full generation against it. normalise_model_path stops an
# illegal path getting that far. It is allowed to widen exactly one case, and the
# tests below are mostly about everything it must NOT widen.


def test_a_bare_filename_lands_in_outputs():
    """
    The bug this function exists for. The ticket writer's prompt already forbids
    bare filenames and the model emits them anyway; a prompt is not a guarantee.

    A bare filename is unambiguous — the model meant a workspace file and just
    forgot to say where — so putting it in outputs/ is deterministic, not a guess.
    """
    assert normalise_model_path("test_add.py") == "outputs/test_add.py"


def test_legal_paths_are_left_exactly_alone():
    assert normalise_model_path("outputs/wrap.py") == "outputs/wrap.py"
    assert normalise_model_path("src/utils/semver.py") == "src/utils/semver.py"


def test_windows_separators_are_canonicalised():
    # The model emits both, and project_files is keyed by this string.
    assert normalise_model_path(r"outputs\wrap.py") == "outputs/wrap.py"


def test_normalisation_does_not_launder_a_traversal():
    """
    The one that matters. Widening bare filenames must not become a hole in the
    write boundary — a traversal has a directory component, so it is not a bare
    filename, so it falls straight through to safe_path and is refused.
    """
    assert normalise_model_path("../../.ssh/authorized_keys") is None
    assert normalise_model_path("outputs/../../etc/passwd") is None
    assert normalise_model_path("../secrets.py") is None
    assert normalise_model_path("..") is None
    assert normalise_model_path(".") is None


def test_an_absolute_path_outside_the_workspace_is_refused():
    assert normalise_model_path("/etc/passwd") is None
    assert normalise_model_path(r"C:\Windows\System32\drivers\etc\hosts") is None


def test_a_path_in_a_directory_we_do_not_own_is_refused_not_guessed():
    """
    "tests/test_add.py" could plausibly mean outputs/tests/test_add.py. Plausibly
    is not good enough — guessing where a model meant to write is how a write
    boundary stops being one. Refuse it and drop the ticket.
    """
    assert normalise_model_path("tests/test_add.py") is None
    assert normalise_model_path("lib/foo.py") is None


def test_empty_and_missing_paths_are_refused():
    assert normalise_model_path("") is None
    assert normalise_model_path("   ") is None
    assert normalise_model_path(None) is None


# ── Canonicalisation — the key everything else is keyed on ───────────────────


def test_every_spelling_of_a_file_collapses_to_one_key():
    """
    normalise_model_path used to return the string the MODEL typed. Its docstring
    claimed the result was canonical. It was not, and that string is the key for
    everything downstream:

    - _collapse_by_file keys on it, so two tickets for one file were not merged;
    - project_files keys on it, so a second ticket read back "" and rewrote a
      passing file from scratch;
    - contract_for() matches on it, so an ACCEPTANCE contract keyed under
      `outputs/lru.py` was NOT FOUND for `./outputs/lru.py` — reviewer_node took the
      no-contract branch, ran the file, got exit 0, and passed. The sprint was
      journalled CLEAN and the human's asserts never ran.
    """
    for spelling in (
        "outputs/lru.py",
        "./outputs/lru.py",
        "outputs//lru.py",
        "lru.py",
        r"outputs\lru.py",
    ):
        assert normalise_model_path(spelling) == "outputs/lru.py", spelling


def test_a_windows_device_name_is_refused():
    """
    Win32 resolves a DOS device INSIDE a real directory: workspace/outputs/NUL
    stats fine, reads empty, and swallows every write.

    So the hive would "write" a file that is discarded, py_compile would see empty
    source and pass, the sandbox would run an empty program and exit 0 — and the
    sprint would be journalled CLEAN having produced nothing at all.
    """
    assert normalise_model_path("NUL") is None
    assert normalise_model_path("outputs/NUL") is None
    assert normalise_model_path("outputs/CON.py") is None
    assert normalise_model_path("outputs/COM1") is None


def test_a_trailing_dot_component_is_refused():
    """
    Win32 strips a trailing dot, so `outputs/...` resolves to the outputs DIRECTORY.
    safe_path approves it (it really is inside the workspace), flush_file tries to
    write over a directory, and the PermissionError reaches the editor as a code
    failure it structurally cannot fix — the unwinnable retry loop this function
    exists to prevent.
    """
    assert normalise_model_path("outputs/...") is None
    assert normalise_model_path("outputs/wrap.py.") is None


def test_an_alternate_data_stream_is_refused():
    """`outputs/wrap.py:evil` writes a hidden NTFS stream and leaves wrap.py empty."""
    assert normalise_model_path("outputs/wrap.py:evil") is None
