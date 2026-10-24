"""HTTP error contract and FastAPI exception handlers."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.schemas import ErrorResponse

# Stable machine-readable codes (see docs/api_contract.md).
ERROR_VALIDATION = "validation_error"
ERROR_INVALID_REQUEST = "invalid_request"
ERROR_FROZEN_PROFILE = "frozen_profile_violation"
ERROR_RERANK_UNAVAILABLE = "rerank_unavailable"
ERROR_NOT_READY = "not_ready"
ERROR_INTERNAL = "internal_error"
ERROR_TIMEOUT = "upstream_timeout"


def api_error(
    status_code: int,
    *,
    error: str,
    message: str,
    detail: Any = None,
) -> None:
    """Raise an HTTPException whose body matches :class:`ErrorResponse`."""
    raise HTTPException(
        status_code=status_code,
        detail={
            "error": error,
            "message": message,
            "detail": detail,
        },
    )


def error_response_from_http_exception(exc: HTTPException) -> ErrorResponse:
    raw = exc.detail
    if isinstance(raw, dict) and "error" in raw and "message" in raw:
        return ErrorResponse(
            error=str(raw["error"]),
            message=str(raw["message"]),
            detail=raw.get("detail"),
        )
    if isinstance(raw, dict):
        return ErrorResponse(
            error=ERROR_NOT_READY if exc.status_code == 503 else ERROR_INVALID_REQUEST,
            message="Request could not be completed.",
            detail=raw,
        )
    message = str(raw) if raw is not None else "Request failed."
    code = ERROR_INVALID_REQUEST
    if exc.status_code == 503:
        code = ERROR_NOT_READY
    return ErrorResponse(error=code, message=message, detail=None)


def request_id_from(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def attach_request_id(body: ErrorResponse, request: Request) -> ErrorResponse:
    rid = request_id_from(request)
    if rid:
        return body.model_copy(update={"request_id": rid})
    return body


def is_upstream_timeout(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, TimeoutError):
        return True
    return False


def map_value_error_to_api(exc: ValueError) -> tuple[str, str]:
    text = str(exc)
    if "BM25 index is not available" in text:
        return ERROR_INVALID_REQUEST, text
    if "Production API" in text or "frozen" in text.lower() or "top_k=" in text:
        return ERROR_FROZEN_PROFILE, text
    if "rerank" in text.lower():
        return ERROR_INVALID_REQUEST, text
    return ERROR_INVALID_REQUEST, text


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        body = attach_request_id(error_response_from_http_exception(exc), request)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        body = attach_request_id(
            ErrorResponse(
                error=ERROR_VALIDATION,
                message="Request validation failed.",
                detail=exc.errors(),
            ),
            request,
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request,
        exc: ValidationError,
    ) -> JSONResponse:
        body = attach_request_id(
            ErrorResponse(
                error=ERROR_VALIDATION,
                message="Response validation failed.",
                detail=exc.errors(),
            ),
            request,
        )
        return JSONResponse(status_code=500, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        body = attach_request_id(
            ErrorResponse(
                error=ERROR_INTERNAL,
                message="An unexpected error occurred.",
                detail=None,
            ),
            request,
        )
        return JSONResponse(status_code=500, content=body.model_dump())
