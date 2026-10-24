"""Await sync callables in threads with wall-clock bounds and cooperative cancel."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TypeVar

from app.cooperative_cancel import QueryCancelledError, QueryScope, invoke_in_scope

T = TypeVar("T")

# After cancel, wait briefly for cooperative exit (stream close, scope checks).
_JOIN_AFTER_CANCEL_SECONDS = 2.0
_POLL_INTERVAL_SECONDS = 0.25


async def run_bounded_sync(
    scope: QueryScope,
    func: Callable[..., T],
    /,
    *args: object,
    **kwargs: object,
) -> T:
    """
    Run ``func`` in a thread under ``scope`` until it returns or the deadline passes.

    On timeout, calls ``scope.cancel()`` so in-flight LLM streams can abort; does not
    force-kill the OS thread if ``func`` ignores the cancel flag.
    """
    task = asyncio.create_task(
        asyncio.to_thread(invoke_in_scope, scope, func, *args, **kwargs),
    )
    try:
        while True:
            try:
                scope.check()
            except QueryCancelledError:
                scope.cancel()
                raise asyncio.TimeoutError() from None
            wait = min(max(scope.remaining(), 0.001), _POLL_INTERVAL_SECONDS)
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=wait)
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        scope.cancel()
        raise
    except BaseException:
        scope.cancel()
        raise
    finally:
        if not task.done():
            scope.cancel()
            try:
                await asyncio.wait_for(task, timeout=_JOIN_AFTER_CANCEL_SECONDS)
            except asyncio.TimeoutError:
                pass
