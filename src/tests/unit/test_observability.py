"""Tests for structured logging, redaction, and stage events."""

from __future__ import annotations

import json
import logging

import pytest

from app.config.settings import Settings
from app.observability.context import bind_request_id, reset_request_id
from app.observability.events import emit, stage_timer
from app.observability.logging import JsonLineFormatter, reset_structured_logging_config
from app.observability.redaction import sanitize_fields


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    reset_structured_logging_config()


def test_sanitize_redacts_api_keys_and_omits_questions():
    fields = sanitize_fields(
        {
            "cohere_api_key": "secret-123",
            "question": "What is APR?",
            "top_k": 10,
        },
        max_text_len=80,
    )
    assert fields["cohere_api_key"] == "[REDACTED]"
    assert fields["question"] == "[omitted len=12]"
    assert fields["top_k"] == 10


def test_sanitize_keeps_token_count_metrics():
    fields = sanitize_fields(
        {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125},
        max_text_len=80,
    )
    assert fields == {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}


def test_emit_includes_request_id_in_structured_payload(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="fiqa.rag")

    token = bind_request_id("trace-week7")
    try:
        with caplog.at_level(logging.INFO, logger="fiqa.rag"):
            emit("test_event", top_k=3)
    finally:
        reset_request_id(token)

    assert caplog.records
    payload = caplog.records[-1].structured
    assert payload["event"] == "test_event"
    assert payload["request_id"] == "trace-week7"
    assert payload["top_k"] == 3


def test_stage_timer_logs_start_and_end(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="fiqa.rag")
    with caplog.at_level(logging.INFO, logger="fiqa.rag"):
        with stage_timer("retrieval", retrieval_method="dense"):
            pass

    events = [r.structured["event"] for r in caplog.records]
    assert events == ["stage_start", "stage_end"]
    end = caplog.records[-1].structured
    assert end["stage"] == "retrieval"
    assert "duration_ms" in end


def test_json_formatter_merges_structured_extra():
    record = logging.LogRecord(
        name="fiqa.rag",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.structured = {"event": "x", "request_id": "1"}
    line = JsonLineFormatter().format(record)
    data = json.loads(line)
    assert data["event"] == "x"
    assert data["request_id"] == "1"
