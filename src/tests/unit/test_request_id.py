"""Tests for ``X-Request-ID`` normalization."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from app.api.request_id import REQUEST_ID_MAX_LENGTH, resolve_request_id


def test_resolve_request_id_accepts_allowed_characters():
    assert resolve_request_id("corr-week6-test") == "corr-week6-test"
    assert resolve_request_id("local.dev:1_2") == "local.dev:1_2"
    assert resolve_request_id("  trimmed-id  ") == "trimmed-id"


def test_resolve_request_id_generates_uuid_when_missing_or_blank():
    with patch("app.api.request_id.uuid.uuid4", return_value=uuid.UUID(int=1)):
        assert resolve_request_id(None) == str(uuid.UUID(int=1))
        assert resolve_request_id("") == str(uuid.UUID(int=1))
        assert resolve_request_id("   ") == str(uuid.UUID(int=1))


def test_resolve_request_id_rejects_excessively_long_values():
    valid_prefix = "a" * REQUEST_ID_MAX_LENGTH
    assert resolve_request_id(valid_prefix) == valid_prefix

    too_long = "a" * (REQUEST_ID_MAX_LENGTH + 1)
    with patch("app.api.request_id.uuid.uuid4", return_value=uuid.UUID(int=2)):
        assert resolve_request_id(too_long) == str(uuid.UUID(int=2))


@pytest.mark.parametrize(
    "invalid",
    [
        "bad\nid",
        "bad\rid",
        "bad\x00id",
        "bad id",
        "bad/id",
        "bad%id",
    ],
)
def test_resolve_request_id_rejects_control_and_disallowed_characters(invalid: str):
    with patch("app.api.request_id.uuid.uuid4", return_value=uuid.UUID(int=3)):
        assert resolve_request_id(invalid) == str(uuid.UUID(int=3))
