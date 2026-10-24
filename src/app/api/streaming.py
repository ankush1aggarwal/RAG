"""Bridge sync LLM token iterators to async SSE consumers."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from typing import Any, Literal

from app.cooperative_cancel import QueryCancelledError, QueryScope, cooperative_check
from app.api.resources import AppResources
from app.pipeline.rag import RagGenerationContext, stream_answer_tokens

StreamKind = Literal["token", "done", "error"]
_STREAM_DONE: tuple[StreamKind, None] = ("done", None)


def produce_llm_tokens(
    resources: AppResources,
    context: RagGenerationContext,
    out: queue.Queue[tuple[StreamKind, Any]],
) -> None:
    """Run in a worker thread; push tokens until done or error."""
    try:
        for delta in stream_answer_tokens(context, resources.llm):
            cooperative_check()
            if delta:
                out.put(("token", delta))
        out.put(_STREAM_DONE)
    except QueryCancelledError as exc:
        out.put(("error", exc))
    except BaseException as exc:
        out.put(("error", exc))


def _queue_get(q: queue.Queue[tuple[StreamKind, Any]], timeout: float) -> tuple[StreamKind, Any]:
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError from None


async def iter_stream_queue(
    q: queue.Queue[tuple[StreamKind, Any]],
    *,
    wall_clock_timeout: float,
    scope: QueryScope | None = None,
) -> AsyncIterator[tuple[StreamKind, Any]]:
    """Drain ``q`` until ``done`` or raise on ``error`` / wall-clock timeout."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wall_clock_timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            if scope is not None:
                scope.cancel()
            raise asyncio.TimeoutError()
        wait = min(max(remaining, 0.001), 1.0)
        try:
            kind, payload = await asyncio.wait_for(
                asyncio.to_thread(_queue_get, q, wait),
                timeout=wait + 1.0,
            )
        except TimeoutError:
            continue
        if kind == "done":
            return
        if kind == "error":
            if scope is not None:
                scope.cancel()
            raise payload
        yield kind, payload
