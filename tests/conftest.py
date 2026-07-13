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
"""
import os
import tempfile
from pathlib import Path

_TEST_WORKSPACE = Path(tempfile.mkdtemp(prefix="multi_hive-tests-"))

os.environ["HIVE_WORKSPACE"] = str(_TEST_WORKSPACE)
