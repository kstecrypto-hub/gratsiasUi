from __future__ import annotations

import logging
import sys

from app.core.logging import (
    SecretRedactingFormatter,
    SecretRedactionFilter,
    register_secret,
)


KNOWN_SECRET = "standalone-trace-secret"
QUERY_TOKEN = "trace-query-token"
RELATIVE_TOKEN = "relative-query-token"
TEMPORARY_SIGNATURE = "temporary-download-signature"
TEMPORARY_RESOURCE = "temporary-recording-resource"
OPAQUE_TEMPORARY_RESOURCE = "opaque-temporary-recording-resource"


def exception_record() -> logging.LogRecord:
    try:
        raise RuntimeError(
            "request failed "
            f"{KNOWN_SECRET} "
            "https://pbx.example.test/"
            f"api/download/{TEMPORARY_RESOURCE}"
            f"?access_token={QUERY_TOKEN}&signature={TEMPORARY_SIGNATURE} "
            "https://pbx.example.test/"
            f"api/{OPAQUE_TEMPORARY_RESOURCE}/recording?access_token={QUERY_TOKEN} "
            f"/openapi/v2.0/cdr/search?access_token={RELATIVE_TOKEN}&page=1"
        )
    except RuntimeError:
        return logging.LogRecord(
            name="test.exception-redaction",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="Yeastar request raised an exception",
            args=(),
            exc_info=sys.exc_info(),
        )


def assert_no_trace_secrets(value: str) -> None:
    for secret in (
        KNOWN_SECRET,
        QUERY_TOKEN,
        RELATIVE_TOKEN,
        TEMPORARY_SIGNATURE,
        TEMPORARY_RESOURCE,
        OPAQUE_TEMPORARY_RESOURCE,
    ):
        assert secret not in value
    assert "[REDACTED]" in value


def test_filter_redacts_cached_exc_text_for_subsequent_formatters() -> None:
    register_secret(KNOWN_SECRET)
    record = exception_record()

    assert SecretRedactionFilter().filter(record) is True
    assert record.exc_text is not None
    assert_no_trace_secrets(record.exc_text)

    # Even a later ordinary formatter reuses only the sanitized traceback
    # cache and cannot re-render the original exception message.
    rendered = logging.Formatter("%(levelname)s %(message)s").format(record)
    assert_no_trace_secrets(rendered)


def test_configured_formatter_redacts_full_traceback_and_url_queries() -> None:
    register_secret(KNOWN_SECRET)
    record = exception_record()

    rendered = SecretRedactingFormatter("%(levelname)s %(message)s").format(record)

    assert_no_trace_secrets(rendered)
    assert "https://pbx.example.test/api/download/[REDACTED]?[REDACTED]" in rendered
    assert "https://pbx.example.test/api/[REDACTED]?[REDACTED]" in rendered
    assert "/openapi/v2.0/cdr/search?[REDACTED]" in rendered
