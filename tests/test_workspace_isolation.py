"""
The tests must never write to the live workspace — see conftest.py.

Nothing asserted this before. `test_safe_path` checks that written files land
under WORKSPACE_DIR, which passes just as happily when WORKSPACE_DIR *is* the
real workspace: it pins the path logic, not the isolation. So a conftest that
silently stopped working would have gone on green while the suite quietly wrote
into `workspace/outputs` and `bench_history.jsonl`.

That already happened once (v4.5.0). These are the assertions that would have
caught it.
"""
from pathlib import Path

import pytest

from multi_hive.config import OUTPUTS_DIR, SRC_DIR, WORKSPACE_DIR

LIVE_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def test_workspace_is_not_the_live_one() -> None:
    assert WORKSPACE_DIR.resolve() != LIVE_WORKSPACE.resolve()
    assert not WORKSPACE_DIR.resolve().is_relative_to(LIVE_WORKSPACE.resolve())


def test_workspace_is_managed_by_pytest(tmp_path_factory: pytest.TempPathFactory) -> None:
    """
    The same factory conftest used, so the workspace lives under the basetemp
    pytest prunes. This is what stops the temp dirs piling up: if someone
    reverts to `tempfile.mkdtemp()`, the workspace escapes the basetemp and
    this fails.
    """
    assert WORKSPACE_DIR.resolve().is_relative_to(tmp_path_factory.getbasetemp().resolve())


def test_generated_code_dirs_are_inside_the_temp_workspace() -> None:
    assert SRC_DIR.resolve().is_relative_to(WORKSPACE_DIR.resolve())
    assert OUTPUTS_DIR.resolve().is_relative_to(WORKSPACE_DIR.resolve())
