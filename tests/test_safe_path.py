"""
safe_path is the write boundary: model-authored paths are untrusted input.
These tests are the reason it exists.
"""
import pytest

from multi_hive.config import OUTPUTS_DIR, SRC_DIR, WORKSPACE_DIR
from multi_hive.core.utils import flush_file, safe_path


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
