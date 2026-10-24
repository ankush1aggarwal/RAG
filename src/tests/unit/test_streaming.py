"""Tests for async SSE token bridge."""

from __future__ import annotations

import asyncio
import queue
import time

from app.api.streaming import iter_stream_queue


def test_iter_stream_queue_yields_tokens_before_producer_thread_finishes():
    q: queue.Queue[tuple[str, object]] = queue.Queue()

    def producer() -> None:
        q.put(("token", "first"))
        time.sleep(0.05)
        q.put(("token", "second"))
        q.put(("done", None))

    async def consume() -> list[tuple[str, object]]:
        task = asyncio.create_task(asyncio.to_thread(producer))
        seen: list[tuple[str, object]] = []
        async for kind, payload in iter_stream_queue(q, wall_clock_timeout=5.0):
            seen.append((kind, payload))
            if kind == "token" and payload == "first":
                assert not task.done(), "consumer should read first token before producer exits"
        await task
        return seen

    seen = asyncio.run(consume())
    assert seen == [("token", "first"), ("token", "second")]
