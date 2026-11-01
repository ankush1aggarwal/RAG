"""Request correlation for structured logs (aligns with ``X-Request-ID``)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("observability_request_id", default=None)


def bind_request_id(request_id: str | None) -> Token | None:
    if not request_id:
        return None
    return _request_id.set(request_id)


def reset_request_id(token: Token | None) -> None:
    if token is not None:
        _request_id.reset(token)


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def observability_context(request_id: str | None):
    token = bind_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)
