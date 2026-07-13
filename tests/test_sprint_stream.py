"""
The sprint loop consumes whatever LangGraph yields, and LangGraph does not
promise a dict per node.
"""
from unittest.mock import patch

import pytest

from multi_hive.cli import run_sprint


class FakeBroker:
    async def readline(self):
        return None


def fake_stream(events):
    # Accepts the config kwarg the sprint passes (recursion_limit).
    async def _stream(_initial_state, **_kwargs):
        for event in events:
            yield event

    return _stream


@pytest.mark.asyncio
async def test_none_delta_does_not_kill_the_sprint():
    """
    LangGraph yields a None delta for a node that wrote no state. Indexing into
    it raised `TypeError: argument of type 'NoneType' is not iterable` and killed
    the sprint *after* all the work was already done — the code was written, the
    reviewers had passed it, and the run still ended in a crash.
    """
    events = [
        {"sprint_planner": {"sprint_plan": "do the thing"}},
        {"agent_router_node": None},  # <- the crash
        {"async_editor_node": {"model_tier": "fast", "editor_error": None}},
        {"retrospector_node": {}},
    ]

    with (
        patch("multi_hive.cli.hive_app") as app,
        patch("multi_hive.cli.clear_ledger"),
        patch("multi_hive.cli.SprintMetrics") as metrics,
    ):
        app.astream = fake_stream(events)
        metrics.return_value.wall_time = 1.0
        metrics.return_value.peak_rss_mb = 1.0
        metrics.return_value.node_count = len(events)
        metrics.return_value.llm_cache_size = 0

        await run_sprint("build a thing", FakeBroker())
