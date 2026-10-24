"""Per-request deadline and cooperative cancel for sync work in thread pools."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Any, TypeVar

T = TypeVar("T")


class QueryCancelledError(Exception):
    """Wall-clock budget exhausted or :meth:`QueryScope.cancel` was called."""


class QueryScope:
    """Thread-safe deadline + cancel flag for one query request."""

    __slots__ = ("_deadline", "_cancel")

    def __init__(self, timeout_seconds: float) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._deadline = time.monotonic() + timeout_seconds
        self._cancel = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        return self._cancel

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def check(self) -> None:
        if self.is_cancelled() or self.remaining() <= 0:
            if not self.is_cancelled():
                self.cancel()
            raise QueryCancelledError()


_query_scope: ContextVar[QueryScope | None] = ContextVar("query_scope", default=None)


def active_scope() -> QueryScope | None:
    return _query_scope.get()


def cooperative_check() -> None:
    scope = _query_scope.get()
    if scope is not None:
        scope.check()


def active_cancel_event() -> threading.Event | None:
    scope = _query_scope.get()
    return scope.cancel_event if scope is not None else None


def set_query_scope(scope: QueryScope) -> Token:
    return _query_scope.set(scope)


def reset_query_scope(token: Token) -> None:
    _query_scope.reset(token)


def invoke_in_scope(
    scope: QueryScope,
    fn: Callable[..., T],
    /,
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run ``fn`` in a worker thread with ``scope`` visible to :func:`cooperative_check`."""
    token = set_query_scope(scope)
    try:
        return fn(*args, **kwargs)
    finally:
        reset_query_scope(token)
