"""Normalize client ``X-Request-ID`` values for safe correlation."""

from __future__ import annotations

import re
import uuid

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_MAX_LENGTH = 128
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def resolve_request_id(incoming: str | None) -> str:
    """Return a sanitized client id, or a new UUID if missing or invalid."""
    if incoming is None:
        return str(uuid.uuid4())
    value = incoming.strip()
    if not value or not _VALID_REQUEST_ID.fullmatch(value):
        return str(uuid.uuid4())
    return value
