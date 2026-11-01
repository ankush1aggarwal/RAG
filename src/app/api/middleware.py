"""HTTP middleware for the FiQA RAG API."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.api.request_id import REQUEST_ID_HEADER, resolve_request_id
from app.observability.context import observability_context

# Re-export for callers that imported from middleware historically.
__all__ = ["REQUEST_ID_HEADER", "RequestIdMiddleware"]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Propagate or assign ``X-Request-ID`` for correlation (logs / support)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = resolve_request_id(incoming)
        request.state.request_id = request_id
        with observability_context(request_id):
            response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
