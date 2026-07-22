from __future__ import annotations

import logging
import re
import traceback
from threading import RLock
from typing import Any
from urllib.parse import urlsplit


_SENSITIVE_KEY = re.compile(
    r"(?i)(access[_-]?token|refresh[_-]?token|api[_-]?key|client[_-]?secret|password|authorization)"
)
_TOKEN_QUERY = re.compile(
    r"(?i)(access[_-]?token|refresh[_-]?token|api[_-]?key|client[_-]?secret|password)=([^&\s]+)"
)
_JSON_SECRET = re.compile(
    r'''(?ix)(["']?(?:access[_-]?token|refresh[_-]?token|api[_-]?key|client[_-]?secret|password)["']?\s*:\s*)["'][^"']*["']'''
)
_BEARER = re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+")
_DOWNLOAD_RESOURCE = re.compile(r"(?i)(/api/download/)[^?\s'\"]+")
_OPAQUE_API_RESOURCE = re.compile(r"(?i)(/api/)(?!download/)[^?\s'\"]+")
_ABSOLUTE_URL_QUERY = re.compile(
    r'''(?ix)\b(https?://[^\s?'"<>()]+)\?[^\s'"<>()]+'''
)
_YEASTAR_PATH_QUERY = re.compile(
    r'''(?ix)(/(?:openapi|api)/[^\s?'"<>()]+)\?[^\s'"<>()]+'''
)
_KNOWN_SECRETS: set[str] = set()
_SECRETS_LOCK = RLock()


def register_secret(value: str | None) -> None:
    if value and len(value) >= 4:
        with _SECRETS_LOCK:
            _KNOWN_SECRETS.add(value)


def redact_text(value: str) -> str:
    with _SECRETS_LOCK:
        secrets = tuple(_KNOWN_SECRETS)
    for secret in secrets:
        value = value.replace(secret, "[REDACTED]")
    value = _TOKEN_QUERY.sub(r"\1=[REDACTED]", value)
    value = _JSON_SECRET.sub(r'\1"[REDACTED]"', value)
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _DOWNLOAD_RESOURCE.sub(r"\1[REDACTED]", value)
    return _OPAQUE_API_RESOURCE.sub(r"\1[REDACTED]", value)


def redact_exception_text(value: str) -> str:
    """Redact exception/traceback text, including complete URL queries."""
    value = redact_text(value)
    value = _ABSOLUTE_URL_QUERY.sub(r"\1?[REDACTED]", value)
    return _YEASTAR_PATH_QUERY.sub(r"\1?[REDACTED]", value)


def sanitized_endpoint(value: str) -> str:
    """Return only a safe endpoint path for outbound request logging."""
    parsed = urlsplit(value)
    path = parsed.path or "/"
    path = _DOWNLOAD_RESOURCE.sub(r"\1[REDACTED]", path)
    return _OPAQUE_API_RESOURCE.sub(r"\1[REDACTED]", path)


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(redact_value(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_value(record.msg)
        if record.args:
            record.args = redact_value(record.args)
        if record.exc_info:
            # Format once at the mandatory handler boundary and replace the
            # LogRecord cache with redacted text. This prevents a later
            # formatter/handler from reusing an unsanitized ``exc_text`` value.
            record.exc_text = redact_exception_text(
                "".join(traceback.format_exception(*record.exc_info))
            )
        elif record.exc_text:
            record.exc_text = redact_exception_text(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_exception_text(record.stack_info)
        return True


class SecretRedactingFormatter(logging.Formatter):
    """Redact the fully rendered record, including traceback and stack text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_exception_text(super().format(record))


def configure_logging() -> None:
    from app.core.config import get_settings

    settings = get_settings()
    for secret in (
        settings.APP_SECRET_KEY,
        settings.ADMIN_PASSWORD,
        settings.YEASTAR_CLIENT_SECRET,
        settings.OPENAI_API_KEY,
    ):
        register_secret(secret)
    handler = logging.StreamHandler()
    handler.addFilter(SecretRedactionFilter())
    handler.setFormatter(
        SecretRedactingFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


__all__ = [
    "SecretRedactionFilter",
    "SecretRedactingFormatter",
    "configure_logging",
    "redact_exception_text",
    "redact_text",
    "redact_value",
    "register_secret",
    "sanitized_endpoint",
]
