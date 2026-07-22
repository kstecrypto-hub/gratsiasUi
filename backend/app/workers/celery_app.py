from __future__ import annotations

from celery import Celery
from celery.signals import setup_logging

from app.core.config import get_settings
from app.core.logging import configure_logging


settings = get_settings()
celery_app = Celery("yeastar_call_analyzer", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.update(
    accept_content=["json"],
    task_serializer="json",
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    # Keep Celery from replacing the application's redacting root handler.
    worker_hijack_root_logger=False,
    result_expires=86400,
    task_routes={
        "app.workers.tasks.process_analysis_job": {"queue": "analysis"},
        "app.workers.tasks.process_job_item": {"queue": "analysis"},
        "app.workers.tasks.cleanup_retention": {"queue": "maintenance"},
    },
    beat_schedule={
        "daily-retention-cleanup": {
            "task": "app.workers.tasks.cleanup_retention",
            "schedule": 24 * 60 * 60,
        }
    },
)
celery_app.autodiscover_tasks(["app.workers"])


def configure_worker_logging(**_: object) -> None:
    """Use the application's secret-redacting logging configuration in workers."""
    configure_logging()


setup_logging.connect(configure_worker_logging, weak=False)
