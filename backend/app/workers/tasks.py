from __future__ import annotations

import asyncio
from uuid import UUID

from app.workers.celery_app import celery_app
from app.workers.pipeline import (
    DiscoveryResult,
    ProcessingBusyError,
    cleanup_retention_records,
    discover_job_items,
    process_item,
)


_worker_loop: asyncio.AbstractEventLoop | None = None


def _run(coroutine):
    """Use one event loop per forked worker process so async pools never cross loops."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(coroutine)


def _recording_assignment_retry_delay(retries: int) -> int:
    """Back off PBX re-discovery without requiring a user to retry a job."""
    return min(300, 15 * (2 ** min(max(retries, 0), 5)))


def _busy_retry_delay(retries: int) -> int:
    """Keep capacity-constrained recordings in the queue with bounded delays."""
    return min(300, 5 * (2 ** min(max(retries, 0), 6)))


@celery_app.task(name="app.workers.tasks.process_analysis_job", bind=True, max_retries=None)
def process_analysis_job(self, job_id: str) -> dict[str, int]:
    try:
        result: DiscoveryResult = _run(discover_job_items(UUID(job_id)))
    except ProcessingBusyError as exc:
        raise self.retry(
            exc=exc,
            countdown=_busy_retry_delay(self.request.retries),
        )
    for item_id in result.item_ids:
        process_job_item.delay(str(item_id))
    if result.recording_assignment_pending:
        # Do not leave a safe-but-unassigned recording as a user-facing
        # failure.  Celery requeues discovery until the PBX exposes a unique
        # CDR-to-recording correlation or the job is cancelled.
        raise self.retry(
            countdown=_recording_assignment_retry_delay(self.request.retries),
        )
    return {"items_queued": len(result.item_ids)}


@celery_app.task(name="app.workers.tasks.process_job_item", bind=True, max_retries=None)
def process_job_item(self, item_id: str) -> None:
    try:
        _run(process_item(UUID(item_id)))
    except ProcessingBusyError as exc:
        raise self.retry(exc=exc, countdown=_busy_retry_delay(self.request.retries))


@celery_app.task(name="app.workers.tasks.cleanup_retention")
def cleanup_retention() -> dict[str, int]:
    return _run(cleanup_retention_records())
