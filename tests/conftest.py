"""
Isolate the tests from the real workspace.

`config.py` resolves WORKSPACE_DIR at import time, so this has to happen before
anything from `multi_hive` is imported — which is exactly what conftest.py is
for: pytest loads it first.

Without this, tests write to the live workspace. That is not hypothetical: a test
mocking a semantic rejection ("FAIL: uses OrderedDict, not a linked list") wrote
that line into the real rejection ledger, where it sat next to genuine entries
from a semver sprint and made the sprint look like it had failed for a reason
that had nothing to do with it. The same leak would corrupt
bench_history.jsonl — the file whose entire job is to be a trustworthy record.

Tests must never write to the artefacts they are meant to be measuring.

The workspace comes from pytest's `tmp_path_factory` rather than
`tempfile.mkdtemp()`, which never cleans up: 88 abandoned `multi_hive-tests-*`
directories had accumulated in the system temp dir, one per run since the suite
was written. The factory keeps the last three runs — still there to inspect when
a test fails — and prunes everything older.

Two things make this safe to do from a hook rather than at module scope.
`pytest_configure` is the last hook that runs *before* collection imports the
test modules, and therefore before anything imports `multi_hive`; `trylast` puts
it after the tmpdir plugin's own `pytest_configure`, which is what creates the
factory. And the `sys.modules` check below is the tripwire: if a future change
ever imports `multi_hive` earlier than this, the suite fails loudly instead of
quietly writing into the live workspace. `tests/test_workspace_isolation.py`
asserts the outcome, because the old arrangement had no check at all — a broken
isolation would have gone on passing.
"""
import os
import sys

import pytest


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    if "multi_hive" in sys.modules:
        raise pytest.UsageError(
            "multi_hive was imported before the test workspace was set, so "
            "WORKSPACE_DIR is already bound to the live workspace and this run "
            "would write into it. Nothing may import multi_hive before "
            "pytest_configure."
        )

    workspace = config._tmp_path_factory.mktemp("workspace")
    os.environ["HIVE_WORKSPACE"] = str(workspace)
