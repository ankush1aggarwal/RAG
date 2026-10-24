"""Cooperative cancel and bounded thread execution."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from app.api.thread_tasks import run_bounded_sync
from app.cooperative_cancel import QueryCancelledError, QueryScope, invoke_in_scope
from app.generation.llm import OllamaLLM


def test_ollama_stream_generate_stops_when_cancel_event_set():
    cancel = threading.Event()
    lines = [
        '{"message":{"content":"a"},"done":false}',
        '{"message":{"content":"b"},"done":false}',
    ]

    stream_resp = MagicMock()
    stream_resp.raise_for_status = MagicMock()
    stream_resp.iter_lines = MagicMock(return_value=iter(lines))
    client = MagicMock()
    client.stream.return_value.__enter__ = MagicMock(return_value=stream_resp)
    client.stream.return_value.__exit__ = MagicMock(return_value=None)
    llm = OllamaLLM(client=client)

    tokens: list[str] = []
    with pytest.raises(QueryCancelledError):
        for token in llm.stream_generate("q", cancel=cancel):
            tokens.append(token)
            cancel.set()
    assert tokens == ["a"]
    stream_resp.close.assert_called_once()


def test_run_bounded_sync_cancels_scope_on_timeout():
    scope = QueryScope(0.05)

    def slow() -> str:
        time.sleep(0.2)
        return "late"

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await run_bounded_sync(scope, slow)

    asyncio.run(run())
    assert scope.is_cancelled()


def test_invoke_in_scope_propagates_to_cooperative_check():
    scope = QueryScope(5.0)
    scope.cancel()

    def worker() -> None:
        from app.cooperative_cancel import cooperative_check

        cooperative_check()

    with pytest.raises(QueryCancelledError):
        invoke_in_scope(scope, worker)
