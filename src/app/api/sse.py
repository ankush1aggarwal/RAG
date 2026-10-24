"""Server-Sent Events helpers for ``POST /query/stream``."""

from __future__ import annotations

import json
from typing import Any


def format_sse(event: str, data: Any) -> str:
    """One SSE message (event + JSON data line + blank line)."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"
