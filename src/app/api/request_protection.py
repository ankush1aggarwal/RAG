"""Lightweight request protection for the single-instance public demo."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

MAX_REQUEST_BODY_BYTES = 16 * 1024
MAX_QUESTION_CHARS = 2_000
QUERY_RATE_LIMIT = 10
QUERY_RATE_WINDOW_SECONDS = 60
QUERY_PATHS = frozenset({"/query", "/query/stream"})


class RequestProtectionMiddleware:
    """Cap request size and apply a process-local quota to expensive RAG requests.

    The quota is global per process, not per IP. It is intentionally lightweight
    for a single-instance portfolio demo and resets when Cloud Run restarts the
    instance. Keep Cloud Run max instances at one while using this limiter.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = MAX_REQUEST_BODY_BYTES,
        query_rate_limit: int = QUERY_RATE_LIMIT,
        window_seconds: int = QUERY_RATE_WINDOW_SECONDS,
    ) -> None:
        if min(max_body_bytes, query_rate_limit, window_seconds) < 1:
            raise ValueError("Request protection limits must be positive")
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.query_rate_limit = query_rate_limit
        self.window_seconds = window_seconds
        self._query_requests: deque[float] = deque()
        self._lock = Lock()

    def _consume_query_quota(self) -> int | None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            while self._query_requests and self._query_requests[0] <= cutoff:
                self._query_requests.popleft()
            if len(self._query_requests) >= self.query_rate_limit:
                return max(1, int(self._query_requests[0] + self.window_seconds - now + 0.999))
            self._query_requests.append(now)
        return None

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        *,
        status_code: int,
        error: str,
        message: str,
        detail: dict[str, int],
        retry_after: int | None = None,
    ) -> None:
        state = scope.get("state") or {}
        headers = {"Retry-After": str(retry_after)} if retry_after is not None else None
        response = JSONResponse(
            status_code=status_code,
            headers=headers,
            content={
                "error": error,
                "message": message,
                "detail": detail,
                "request_id": state.get("request_id"),
            },
        )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "")

        if method == "POST" and path in QUERY_PATHS:
            retry_after = self._consume_query_quota()
            if retry_after is not None:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=429,
                    error="rate_limit_exceeded",
                    message=(
                        f"Query limit exceeded: {self.query_rate_limit} requests "
                        f"per {self.window_seconds} seconds per running instance."
                    ),
                    detail={
                        "limit": self.query_rate_limit,
                        "window_seconds": self.window_seconds,
                    },
                    retry_after=retry_after,
                )
                return

        if method not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value.decode("latin-1")
                for key, value in scope.get("headers", [])
                if key.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._reject(
                        scope,
                        receive,
                        send,
                        status_code=413,
                        error="payload_too_large",
                        message=f"Request body exceeds {self.max_body_bytes} bytes.",
                        detail={"max_request_body_bytes": self.max_body_bytes},
                    )
                    return
            except ValueError:
                pass

        # Read and cap chunked bodies too, then replay the small buffered body to FastAPI.
        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            total_bytes += len(chunk)
            if total_bytes > self.max_body_bytes:
                await self._reject(
                    scope,
                    receive,
                    send,
                    status_code=413,
                    error="payload_too_large",
                    message=f"Request body exceeds {self.max_body_bytes} bytes.",
                    detail={"max_request_body_bytes": self.max_body_bytes},
                )
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
