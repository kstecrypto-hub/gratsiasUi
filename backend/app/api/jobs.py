from __future__ import annotations

import hashlib
import secrets
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.dependencies import CurrentUser
from app.api.dependencies import EffectiveRuntimeSettings
from app.core.config import Settings
from app.core.redis import get_redis
from app.core.time import ensure_utc, utc_now
from app.database.session import get_db
from app.models import KeywordCategory, Operator, ProcessingJob
from app.models.enums import ItemStatus, JobStatus
from app.schemas.common import Page
from app.schemas.jobs import JobCreate, JobDetail, JobItemResponse, JobSummary
from app.services.audit import audit
from app.services.yeastar.circuit_breaker import YeastarCircuitBreaker
from app.services.yeastar.integration import (
    get_integration_status,
    is_connected_configuration,
)
from app.workers.tasks import process_analysis_job, process_job_item


router = APIRouter(prefix="/jobs", tags=["analysis"])

TERMINAL_JOB_STATUSES = (
    JobStatus.COMPLETED,
    JobStatus.COMPLETED_WITH_ERRORS,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
)
FINAL_ITEM_STATUSES = (
    ItemStatus.COMPLETED,
    ItemStatus.FAILED,
    ItemStatus.CANCELLED,
    ItemStatus.SKIPPED,
)


def job_summary(job: ProcessingJob, *, is_current: bool = False) -> JobSummary:
    return JobSummary(
        id=job.id,
        is_current=is_current,
        status=job.status,
        date_from=job.date_from,
        date_to=job.date_to,
        operator_ids=job.selected_operator_ids,
        progress_percent=job.progress_percent,
        current_stage=job.current_stage,
        calls_found=job.calls_found,
        recordings_found=job.recordings_found,
        calls_completed=job.calls_completed,
        calls_failed=job.calls_failed,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def job_detail(job: ProcessingJob, *, is_current: bool = False) -> JobDetail:
    base = job_summary(job, is_current=is_current).model_dump()
    return JobDetail(
        **base,
        direction=job.direction,
        queue=job.queue_name,
        call_status_filter=job.call_status_filter,
        recording_available=job.recording_available,
        include_all_speakers=job.include_all_speakers,
        cancellation_requested=job.cancellation_requested,
        attempt_count=job.attempt_count,
        last_error_category=job.last_error_category,
        last_error_message=job.last_error_message,
        items=[JobItemResponse.model_validate(item) for item in job.items],
    )


async def latest_job_id(db: AsyncSession) -> UUID | None:
    return await db.scalar(
        select(ProcessingJob.id)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
        .limit(1)
    )


async def active_job(
    db: AsyncSession,
    *,
    exclude_id: UUID | None = None,
    lock: bool = False,
) -> ProcessingJob | None:
    statement = select(ProcessingJob).where(
        ProcessingJob.status.not_in(TERMINAL_JOB_STATUSES)
    )
    if exclude_id is not None:
        statement = statement.where(ProcessingJob.id != exclude_id)
    statement = statement.order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
    if lock:
        statement = statement.with_for_update()
    return await db.scalar(statement.limit(1))


def active_job_conflict() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "Another analysis is already running. Open it or wait for it to finish "
            "before starting a new analysis."
        ),
    )


def _require_integrations(settings: Settings) -> None:
    if not settings.yeastar_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Yeastar integration is not configured. "
                "Add the connection details in Settings."
            ),
        )
    if not settings.openai_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI integration is not configured. Add an API key in Settings.",
        )
    if not settings.APP_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application security is not configured.",
        )


async def _require_connected_yeastar(
    db: AsyncSession,
    settings: Settings,
) -> None:
    integration = await get_integration_status(db, create=False)
    try:
        circuit_state = await YeastarCircuitBreaker(get_redis()).get_state()
    except RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Phone-system connection status is temporarily unavailable.",
        ) from exc
    if is_connected_configuration(
        integration,
        settings,
        circuit_state=circuit_state,
    ):
        return
    fingerprint = settings.yeastar_configuration_fingerprint
    if (
        integration is not None
        and integration.configuration_fingerprint is not None
        and integration.configuration_fingerprint != fingerprint
    ):
        detail = "Phone-system settings changed. Test the connection in Settings."
    else:
        detail = "Phone-system connection is not ready. Test the connection in Settings."
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


@router.post("", response_model=JobDetail, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    payload: JobCreate,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveRuntimeSettings,
) -> JobDetail:
    _require_integrations(settings)
    await _require_connected_yeastar(db, settings)
    unique_operator_ids = list(dict.fromkeys(payload.operator_ids))
    operators = (
        await db.scalars(
            select(Operator).where(
                Operator.id.in_(unique_operator_ids),
                Operator.enabled.is_(True),
                Operator.deleted_at.is_(None),
            )
        )
    ).all()
    if len(operators) != len(unique_operator_ids):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="One or more selected operators are unavailable.",
        )
    if payload.keyword_category_ids:
        category_count = await db.scalar(
            select(func.count()).select_from(KeywordCategory).where(
                KeywordCategory.id.in_(payload.keyword_category_ids),
                KeywordCategory.active.is_(True),
                KeywordCategory.deleted_at.is_(None),
            )
        ) or 0
        if category_count != len(set(payload.keyword_category_ids)):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="One or more keyword categories are unavailable.",
            )
    if payload.direction and payload.direction.lower() not in {"inbound", "outbound", "internal"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Direction must be inbound, outbound, or internal.",
        )
    date_from = ensure_utc(payload.date_from, settings.APP_TIMEZONE)
    date_to = ensure_utc(payload.date_to, settings.APP_TIMEZONE)
    supplied_key = payload.idempotency_key or request.headers.get("Idempotency-Key")
    idempotency_key = supplied_key or hashlib.sha256(
        f"{user.id}:{date_from.isoformat()}:{date_to.isoformat()}:{secrets.token_urlsafe(24)}".encode()
    ).hexdigest()
    existing = await db.scalar(
        select(ProcessingJob)
        .options(selectinload(ProcessingJob.items))
        .where(ProcessingJob.idempotency_key == idempotency_key)
    )
    if existing:
        return job_detail(existing, is_current=existing.id == await latest_job_id(db))
    if await active_job(db, lock=True) is not None:
        raise active_job_conflict()
    job = ProcessingJob(
        idempotency_key=idempotency_key,
        requested_by_id=user.id,
        status=JobStatus.QUEUED,
        date_from=date_from,
        date_to=date_to,
        direction=payload.direction.lower() if payload.direction else None,
        queue_name=payload.queue,
        call_status_filter=payload.call_status,
        recording_available=payload.recording_available,
        include_all_speakers=payload.include_all_speakers,
        selected_operator_ids=[str(item) for item in unique_operator_ids],
        selected_category_ids=[str(item) for item in dict.fromkeys(payload.keyword_category_ids)],
        request_filters={
            "direction": payload.direction,
            "queue": payload.queue,
            "call_status": payload.call_status,
            "recording_available": payload.recording_available,
        },
        current_stage="Queued",
    )
    db.add(job)
    await audit(
        db,
        action="analysis.create",
        request=request,
        user=user,
        resource_type="processing_job",
        resource_id=str(job.id),
        details={"date_from": date_from.isoformat(), "date_to": date_to.isoformat()},
    )
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing = await db.scalar(
            select(ProcessingJob)
            .options(selectinload(ProcessingJob.items))
            .where(ProcessingJob.idempotency_key == idempotency_key)
        )
        if existing is not None:
            return job_detail(existing, is_current=existing.id == await latest_job_id(db))
        raise active_job_conflict() from None
    try:
        task = process_analysis_job.delay(str(job.id))
        job.celery_task_id = task.id
        await db.commit()
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.last_error_category = "processing_service"
        job.last_error_message = "Processing service is unavailable."
        job.completed_at = utc_now()
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing service is unavailable.",
        ) from exc
    await db.refresh(job, attribute_names=["items"])
    return job_detail(job, is_current=True)


@router.get("", response_model=Page[JobSummary])
async def list_jobs(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = 1,
    page_size: int = 50,
) -> Page[JobSummary]:
    page, page_size = max(1, page), min(200, max(1, page_size))
    total = await db.scalar(select(func.count()).select_from(ProcessingJob)) or 0
    jobs = (
        await db.scalars(
            select(ProcessingJob)
            .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
    ).all()
    return Page(
        items=[
            job_summary(job, is_current=page == 1 and index == 0)
            for index, job in enumerate(jobs)
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/current", response_model=JobSummary | None)
async def get_current_job(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobSummary | None:
    job = await db.scalar(
        select(ProcessingJob)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
        .limit(1)
    )
    return job_summary(job, is_current=True) if job is not None else None


@router.get("/active", response_model=JobSummary | None)
async def get_active_job(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobSummary | None:
    job = await active_job(db)
    if job is None:
        return None
    return job_summary(job, is_current=job.id == await latest_job_id(db))


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(
    job_id: UUID,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobDetail:
    job = await db.scalar(
        select(ProcessingJob)
        .options(selectinload(ProcessingJob.items))
        .where(ProcessingJob.id == job_id)
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    return job_detail(job, is_current=job.id == await latest_job_id(db))


@router.post("/{job_id}/retry", response_model=JobDetail, status_code=status.HTTP_202_ACCEPTED)
async def retry_job(
    job_id: UUID,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveRuntimeSettings,
) -> JobDetail:
    _require_integrations(settings)
    job = await db.scalar(
        select(ProcessingJob)
        .options(selectinload(ProcessingJob.items))
        .where(ProcessingJob.id == job_id)
        .with_for_update()
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    if job.status not in TERMINAL_JOB_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This analysis is already running.",
        )
    if await active_job(db, exclude_id=job.id, lock=True) is not None:
        raise active_job_conflict()
    failed_items = [item for item in job.items if item.status == ItemStatus.FAILED]
    processable_items = [item for item in failed_items if item.recording_id is not None]
    discovery_items = [item for item in failed_items if item.recording_id is None]
    had_discovery_failures = int(
        (job.request_filters or {}).get("_discovery_failures", 0)
    ) > 0
    if not failed_items and job.status not in {JobStatus.FAILED, JobStatus.COMPLETED_WITH_ERRORS}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="There are no failed calls to retry.")
    job.attempt_count += 1
    job.cancellation_requested = False
    job.status = JobStatus.QUEUED
    job.current_stage = "Queued for retry"
    job.completed_at = None
    job.last_error_category = None
    job.last_error_message = None
    job.request_filters = {**(job.request_filters or {}), "_discovery_failures": 0}
    for item in failed_items:
        item.status = ItemStatus.QUEUED
        item.stage = "queued"
        item.error_category = None
        item.error_message = None
        item.completed_at = None
    statuses_by_call: dict[UUID, list[ItemStatus]] = {}
    for item in job.items:
        statuses_by_call.setdefault(item.call_id, []).append(item.status)
    final_items = sum(
        item.status in {ItemStatus.COMPLETED, ItemStatus.SKIPPED, ItemStatus.CANCELLED}
        for item in job.items
    )
    job.calls_completed = sum(
        bool(states) and all(state == ItemStatus.COMPLETED for state in states)
        for states in statuses_by_call.values()
    )
    job.calls_failed = 0
    job.progress_percent = round(final_items * 100 / len(job.items)) if job.items else 0
    await audit(
        db,
        action="analysis.retry",
        request=request,
        user=user,
        resource_type="processing_job",
        resource_id=str(job.id),
        details={
            "failed_items": len(failed_items),
            "discovery_failures": had_discovery_failures,
        },
    )
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise active_job_conflict() from None
    rerun_discovery = bool(discovery_items or had_discovery_failures or not failed_items)
    try:
        if processable_items and not rerun_discovery:
            for item in processable_items:
                task = process_job_item.delay(str(item.id))
                item.celery_task_id = task.id
        if rerun_discovery:
            task = process_analysis_job.delay(str(job.id))
            job.celery_task_id = task.id
        await db.commit()
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.current_stage = "Retry could not be queued"
        job.last_error_category = "processing_service"
        job.last_error_message = "Processing service is unavailable."
        job.completed_at = utc_now()
        for item in job.items:
            if item.status not in FINAL_ITEM_STATUSES:
                item.status = ItemStatus.FAILED
                item.stage = "failed"
                item.error_category = "processing_service"
                item.error_message = "Processing service is unavailable."
                item.completed_at = utc_now()
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing service is unavailable.",
        ) from exc
    return job_detail(job, is_current=job.id == await latest_job_id(db))


@router.post("/{job_id}/cancel", response_model=JobDetail)
async def cancel_job(
    job_id: UUID,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobDetail:
    job = await db.scalar(
        select(ProcessingJob)
        .options(selectinload(ProcessingJob.items))
        .where(ProcessingJob.id == job_id)
        .with_for_update()
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    if job.status in TERMINAL_JOB_STATUSES:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Analysis is already finished.")
    job.cancellation_requested = True
    for item in job.items:
        if item.status not in FINAL_ITEM_STATUSES:
            item.status = ItemStatus.CANCELLED
            item.stage = "cancelled"
            item.completed_at = utc_now()
    job.status = JobStatus.CANCELLED
    job.current_stage = "Cancelled"
    job.completed_at = utc_now()
    await audit(
        db,
        action="analysis.cancel",
        request=request,
        user=user,
        resource_type="processing_job",
        resource_id=str(job.id),
    )
    await db.commit()
    return job_detail(job, is_current=job.id == await latest_job_id(db))
