from __future__ import annotations

import logging

from app.workers.celery_app import celery_app, configure_worker_logging


def test_celery_worker_uses_redacting_application_logging() -> None:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_root_level = root.level
    logger_levels = {
        name: logging.getLogger(name).level for name in ("httpx", "httpcore", "openai")
    }
    try:
        configure_worker_logging()

        assert celery_app.conf.worker_hijack_root_logger is False
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)

        handler = root.handlers[0]
        resource_name = "worker-opaque-resource"
        token = "worker-query-token"
        record = logging.LogRecord(
            name="httpx",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="request failed %s",
            args=(f"https://pbx.example.test/api/{resource_name}/file?access_token={token}",),
            exc_info=None,
        )

        assert handler.filter(record)
        rendered = handler.format(record)

        assert resource_name not in rendered
        assert token not in rendered
        assert "[REDACTED]" in rendered
    finally:
        root.handlers[:] = original_handlers
        root.setLevel(original_root_level)
        for name, level in logger_levels.items():
            logging.getLogger(name).setLevel(level)
