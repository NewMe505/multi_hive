"""
The stdin broker has two consumers — the REPL and the human gate's
acknowledgement task — and exactly one producer thread. That asymmetry is where
the bugs live.
"""
import asyncio

import pytest

from multi_hive.cli import _EOF, StdinBroker


@pytest.mark.asyncio
async def test_eof_is_sticky_across_consumers():
    """
    Both consumers must observe EOF.

    The gate's ack task and the REPL both call readline(). If the first to reach
    the EOF marker consumes it, the other blocks forever on a queue no producer
    will fill again — the sprint ends and the REPL never returns to the prompt.
    """
    broker = StdinBroker()
    await broker._queue.put(_EOF)

    # The ack task gets there first...
    assert await broker.readline() is None
    # ...and the REPL must still see EOF rather than hanging.
    assert await asyncio.wait_for(broker.readline(), timeout=1.0) is None


@pytest.mark.asyncio
async def test_lines_are_delivered_in_order_and_stripped():
    broker = StdinBroker()
    await broker._queue.put("build a thing")
    await broker._queue.put("exit")

    assert await broker.readline() == "build a thing"
    assert await broker.readline() == "exit"


@pytest.mark.asyncio
async def test_close_is_safe_when_never_started():
    # main() may bail before start(); close() runs in a finally block regardless.
    await StdinBroker().close()
