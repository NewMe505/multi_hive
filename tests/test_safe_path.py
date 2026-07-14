"""
safe_path is the write boundary: model-authored paths are untrusted input.
These tests are the reason it exists.
"""
import pytest

from multi_hive.config import OUTPUTS_DIR, SRC_DIR, WORKSPACE_DIR
from multi_hive.core.utils import flush_file, normalise_model_path, safe_path


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
