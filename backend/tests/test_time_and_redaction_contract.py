from __future__ import annotations

import logging
from datetime import UTC, datetime

from app.core.logging import SecretRedactionFilter, redact_value
from app.core.time import ensure_utc, to_local


def test_naive_athens_datetime_converts_to_utc_with_seasonal_offset() -> None:
    assert ensure_utc(datetime(2026, 7, 15, 12, 0)) == datetime(
        2026, 7, 15, 9, 0, tzinfo=UTC
    )
    assert ensure_utc(datetime(2026, 1, 15, 12, 0)) == datetime(
        2026, 1, 15, 10, 0, tzinfo=UTC
    )


def test_aware_datetime_round_trips_through_configured_local_timezone() -> None:
    original = datetime(2026, 7, 15, 9, 30, tzinfo=UTC)

    local = to_local(original, "Europe/Athens")

    assert local.isoformat() == "2026-07-15T12:30:00+03:00"
    assert ensure_utc(local) == original


def test_to_local_treats_naive_input_as_utc() -> None:
    local = to_local(datetime(2026, 1, 15, 10, 0), "Europe/Athens")

    assert local.isoformat() == "2026-01-15T12:00:00+02:00"


def test_redact_value_masks_nested_secret_keys_queries_and_bearer_tokens() -> None:
    value = {
        "access_token": "access-secret",
        "nested": {
            "client-secret": "client-secret-value",
            "password": "password-value",
            "safe": "visible",
        },
        "url": (
            "https://pbx.example.test/path?access_token=query-secret&api_key=api-secret&ok=1"
        ),
        "message": "Authorization used Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
    }

    redacted = redact_value(value)

    assert redacted["access_token"] == "[REDACTED]"
    assert redacted["nested"] == {
        "client-secret": "[REDACTED]",
        "password": "[REDACTED]",
        "safe": "visible",
    }
    assert "query-secret" not in redacted["url"]
    assert "api-secret" not in redacted["url"]
    assert "access_token=[REDACTED]" in redacted["url"]
    assert "api_key=[REDACTED]" in redacted["url"]
    assert redacted["message"].endswith("Bearer [REDACTED]")


def test_logging_filter_redacts_structured_arguments_before_rendering() -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="payload=%s url=%s",
        args=(
            {"password": "plain-password", "safe": "visible"},
            "https://pbx.example.test/?refresh_token=plain-refresh",
        ),
        exc_info=None,
    )

    assert SecretRedactionFilter().filter(record) is True
    rendered = record.getMessage()

    assert "plain-password" not in rendered
    assert "plain-refresh" not in rendered
    assert "[REDACTED]" in rendered
    assert "visible" in rendered
