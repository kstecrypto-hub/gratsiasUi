from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, delete, distinct, exists, func, not_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, selectinload
from sqlalchemy.orm.attributes import flag_modified

from app.api.jobs import (
    FINAL_ITEM_STATUSES,
    TERMINAL_JOB_STATUSES,
    _require_integrations,
    active_job,
    active_job_conflict,
    job_detail,
    job_summary,
)
from app.api.dependencies import EffectiveRuntimeSettings, EffectiveYeastarSettings
from app.auth.dependencies import CurrentUser
from app.core.config import get_settings
from app.core.time import utc_now
from app.database.session import get_db
from app.models import (
    Call,
    CallParticipant,
    Keyword,
    KeywordCategory,
    KeywordMatch,
    Operator,
    ProcessingJob,
    ProcessingJobItem,
    Recording,
    Transcript,
    TranscriptSegment,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    SpeakerAttributionStatus,
    SpeakerSource,
    TranscriptionMode,
    TranscriptStatus,
)
from app.schemas.common import MessageResponse, Page
from app.schemas.jobs import JobDetail
from app.schemas.results import (
    CallReprocessRequest,
    CallDetailResponse,
    DashboardResponse,
    MatchResponse,
    ResultItem,
    SpeakerAssignmentRequest,
    SpeakerAssignmentResponse,
    TranscriptQualitySummaryResponse,
    TranscriptSegmentResponse,
)
from app.services.audit import audit
from app.services.application_settings import load_application_settings
from app.services.audio import AudioProcessor
from app.services.export import csv_bytes, mask_phone_number
from app.services.keyword_matching import KeywordDefinition, match_text
from app.services.keyword_matching.normalization import normalize_greek
from app.services.yeastar import YeastarClient
from app.workers.tasks import process_job_item


router = APIRouter(tags=["results"])
logger = logging.getLogger(__name__)


def _external_number(call: Call) -> str | None:
    if call.direction.value == "outbound":
        return call.callee_number
    return call.caller_number


def _filter_boundary(
    value: date | datetime | None, timezone: ZoneInfo, *, end: bool
) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone)
        result = value.astimezone(UTC)
        return result + timedelta(microseconds=1) if end else result
    boundary_date = value + timedelta(days=1) if end else value
    return datetime.combine(boundary_date, time.min, tzinfo=timezone).astimezone(UTC)


def _parse_filter_boundary(
    value: str | None, timezone: ZoneInfo, *, field: str, end: bool
) -> datetime | None:
    """Parse date-only filters as local days and datetimes as exact instants.

    FastAPI/Pydantic flattens a ``date | datetime`` query annotation to its first
    primitive type, which rejects valid non-midnight datetimes at the HTTP layer.
    Keeping the wire value as text also lets us preserve the intentionally
    different semantics of a local calendar date and an ISO datetime.
    """
    if value is None:
        return None
    try:
        parsed: date | datetime
        if "T" in value or " " in value:
            parsed = datetime.fromisoformat(value)
        else:
            parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field} must be an ISO 8601 date or datetime.",
        ) from exc
    return _filter_boundary(parsed, timezone, end=end)


def _normalized_transcript_query(value: str | None) -> str | None:
    """Normalize a user-entered transcript search without exposing SQL wildcards."""

    if value is None or not value.strip():
        return None
    normalized = normalize_greek(value)
    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Enter a word or phrase to search in transcripts.",
        )
    if len(normalized) > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Transcript search is too long.",
        )
    return normalized


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DashboardResponse:
    calls_analyzed = await db.scalar(select(func.count()).select_from(Call)) or 0
    calls_with_recordings = (
        await db.scalar(select(func.count()).select_from(Call).where(Call.has_recording.is_(True)))
        or 0
    )
    calls_transcribed = (
        await db.scalar(
            select(func.count(distinct(Transcript.call_id))).where(
                Transcript.status == TranscriptStatus.COMPLETED,
                Transcript.is_current.is_(True),
            )
        )
        or 0
    )
    calls_with_matches = (
        await db.scalar(
            select(func.count(distinct(KeywordMatch.call_id)))
            .join(TranscriptSegment, TranscriptSegment.id == KeywordMatch.transcript_segment_id)
            .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
            .where(
                Transcript.status == TranscriptStatus.COMPLETED,
                Transcript.is_current.is_(True),
            )
        )
        or 0
    )
    failed_call_ids = (
        select(ProcessingJobItem.call_id.label("call_id"))
        .where(ProcessingJobItem.status == ItemStatus.FAILED)
        .union(select(Call.id.label("call_id")).where(Call.processing_status == "failed"))
        .subquery()
    )
    failed_calls = await db.scalar(select(func.count()).select_from(failed_call_ids)) or 0
    processing_jobs = await db.scalar(select(func.count()).select_from(ProcessingJob)) or 0
    recent_jobs = (
        await db.scalars(
            select(ProcessingJob)
            .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
            .limit(10)
        )
    ).all()
    operator_rows = (
        await db.execute(
            select(
                Operator.id,
                Operator.display_name,
                func.count(distinct(KeywordMatch.call_id)).label("call_count"),
                func.count(KeywordMatch.id).label("match_count"),
            )
            .join(KeywordMatch, KeywordMatch.operator_id == Operator.id)
            .join(TranscriptSegment, TranscriptSegment.id == KeywordMatch.transcript_segment_id)
            .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
            .where(
                Transcript.status == TranscriptStatus.COMPLETED,
                Transcript.is_current.is_(True),
            )
            .group_by(Operator.id, Operator.display_name)
            .order_by(func.count(KeywordMatch.id).desc())
            .limit(20)
        )
    ).all()
    category_rows = (
        await db.execute(
            select(
                KeywordCategory.id,
                KeywordCategory.name,
                func.count(distinct(KeywordMatch.call_id)).label("call_count"),
                func.count(KeywordMatch.id).label("match_count"),
            )
            .join(Keyword, Keyword.category_id == KeywordCategory.id)
            .join(KeywordMatch, KeywordMatch.keyword_id == Keyword.id)
            .join(TranscriptSegment, TranscriptSegment.id == KeywordMatch.transcript_segment_id)
            .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
            .where(
                Transcript.status == TranscriptStatus.COMPLETED,
                Transcript.is_current.is_(True),
            )
            .group_by(KeywordCategory.id, KeywordCategory.name)
            .order_by(func.count(KeywordMatch.id).desc())
            .limit(20)
        )
    ).all()
    return DashboardResponse(
        has_data=bool(calls_analyzed or processing_jobs),
        calls_analyzed=calls_analyzed,
        calls_with_recordings=calls_with_recordings,
        calls_transcribed=calls_transcribed,
        calls_with_matches=calls_with_matches,
        failed_calls=failed_calls,
        processing_jobs=processing_jobs,
        results_by_operator=[
            {
                "operator_id": str(row.id),
                "operator_name": row.display_name,
                "call_count": row.call_count,
                "match_count": row.match_count,
                "count": row.match_count,
            }
            for row in operator_rows
        ],
        results_by_keyword_category=[
            {
                "category_id": str(row.id),
                "category_name": row.name,
                "call_count": row.call_count,
                "match_count": row.match_count,
                "count": row.match_count,
            }
            for row in category_rows
        ],
        recent_jobs=[
            job_summary(job, is_current=index == 0) for index, job in enumerate(recent_jobs)
        ],
    )


def _result_query(
    *,
    job_id: UUID,
    include_all_speakers: bool,
    date_from: datetime | None,
    date_to: datetime | None,
    operator_id: UUID | None,
    keyword_id: UUID | None,
    keyword: str | None,
    transcript_query: str | None,
    category_id: UUID | None,
    direction: str | None,
    has_matches: bool | None,
):
    participant_pairs = (
        select(
            ProcessingJobItem.call_id.label("call_id"),
            ProcessingJobItem.operator_id.label("operator_id"),
        )
        .where(ProcessingJobItem.job_id == job_id)
        .distinct()
        .subquery()
    )
    query = (
        select(Call, Operator)
        .join(participant_pairs, participant_pairs.c.call_id == Call.id)
        .join(Operator, Operator.id == participant_pairs.c.operator_id)
    )
    conditions = []
    if date_from:
        conditions.append(Call.started_at >= date_from)
    if date_to:
        conditions.append(Call.started_at < date_to)
    if operator_id:
        conditions.append(Operator.id == operator_id)
    if direction:
        conditions.append(Call.direction == Direction(direction.lower()))
    if transcript_query:
        escaped_query = (
            transcript_query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        transcript_speaker_scope = (
            or_(
                TranscriptSegment.operator_id == Operator.id,
                TranscriptSegment.operator_id.is_(None),
            )
            if include_all_speakers
            else TranscriptSegment.operator_id == Operator.id
        )
        transcript_matches = (
            select(TranscriptSegment.id)
            .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
            .join(
                ProcessingJobItem,
                and_(
                    ProcessingJobItem.job_id == job_id,
                    ProcessingJobItem.call_id == Call.id,
                    ProcessingJobItem.result_transcript_id == Transcript.id,
                    or_(
                        ProcessingJobItem.operator_id == Operator.id,
                        Transcript.operator_id.is_(None),
                    ),
                ),
            )
            .where(
                TranscriptSegment.call_id == Call.id,
                Transcript.status == TranscriptStatus.COMPLETED,
                transcript_speaker_scope,
                TranscriptSegment.normalized_text.ilike(f"%{escaped_query}%", escape="\\"),
            )
        )
        conditions.append(exists(transcript_matches))
    match_speaker_scope = (
        and_(
            or_(
                KeywordMatch.operator_id == Operator.id,
                KeywordMatch.operator_id.is_(None),
            ),
            or_(
                TranscriptSegment.operator_id == Operator.id,
                TranscriptSegment.operator_id.is_(None),
            ),
        )
        if include_all_speakers
        else and_(
            KeywordMatch.operator_id == Operator.id,
            TranscriptSegment.operator_id == Operator.id,
        )
    )
    matching = (
        select(KeywordMatch.id)
        .join(TranscriptSegment, TranscriptSegment.id == KeywordMatch.transcript_segment_id)
        .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
        .join(
            ProcessingJobItem,
            and_(
                ProcessingJobItem.job_id == job_id,
                ProcessingJobItem.call_id == Call.id,
                ProcessingJobItem.result_transcript_id == Transcript.id,
                or_(
                    ProcessingJobItem.operator_id == Operator.id,
                    Transcript.operator_id.is_(None),
                ),
            ),
        )
        .where(
            KeywordMatch.call_id == Call.id,
            Transcript.status == TranscriptStatus.COMPLETED,
            match_speaker_scope,
        )
    )
    if keyword_id or keyword or category_id:
        matching = matching.join(Keyword, Keyword.id == KeywordMatch.keyword_id)
    if keyword_id:
        matching = matching.where(KeywordMatch.keyword_id == keyword_id)
    if keyword:
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        matching = matching.where(Keyword.canonical_phrase.ilike(f"%{escaped}%", escape="\\"))
    if category_id:
        matching = matching.where(Keyword.category_id == category_id)
    if has_matches is True or keyword_id or keyword or category_id:
        conditions.append(exists(matching))
    elif has_matches is False:
        conditions.append(not_(exists(matching)))
    return query.where(*conditions)


async def _matches_for_pairs(
    db: AsyncSession,
    pairs: list[tuple[Call, Operator]],
    *,
    job_id: UUID,
    include_all_speakers: bool,
    keyword_id: UUID | None = None,
    keyword_text: str | None = None,
    category_id: UUID | None = None,
) -> dict[tuple[UUID, UUID], list[tuple[KeywordMatch, Keyword, KeywordCategory]]]:
    if not pairs:
        return {}
    call_ids = {call.id for call, _ in pairs}
    operator_ids = {operator.id for _, operator in pairs}
    target_item = aliased(ProcessingJobItem, name="result_target_item")
    source_item = aliased(ProcessingJobItem, name="result_source_item")
    bindings = (
        select(
            source_item.result_transcript_id.label("transcript_id"),
            target_item.call_id.label("call_id"),
            target_item.operator_id.label("operator_id"),
        )
        .select_from(target_item)
        .join(
            source_item,
            and_(
                source_item.job_id == target_item.job_id,
                source_item.call_id == target_item.call_id,
                source_item.result_transcript_id.is_not(None),
            ),
        )
        .join(Transcript, Transcript.id == source_item.result_transcript_id)
        .where(
            target_item.job_id == job_id,
            target_item.call_id.in_(call_ids),
            target_item.operator_id.in_(operator_ids),
            or_(
                source_item.operator_id == target_item.operator_id,
                Transcript.operator_id.is_(None),
            ),
        )
        .distinct()
        .subquery()
    )
    match_speaker_scope = (
        and_(
            or_(
                KeywordMatch.operator_id == bindings.c.operator_id,
                KeywordMatch.operator_id.is_(None),
            ),
            or_(
                TranscriptSegment.operator_id == bindings.c.operator_id,
                TranscriptSegment.operator_id.is_(None),
            ),
        )
        if include_all_speakers
        else and_(
            KeywordMatch.operator_id == bindings.c.operator_id,
            TranscriptSegment.operator_id == bindings.c.operator_id,
        )
    )
    query = (
        select(KeywordMatch, Keyword, KeywordCategory, bindings.c.operator_id)
        .join(Keyword, Keyword.id == KeywordMatch.keyword_id)
        .join(KeywordCategory, KeywordCategory.id == Keyword.category_id)
        .join(TranscriptSegment, TranscriptSegment.id == KeywordMatch.transcript_segment_id)
        .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
        .join(
            bindings,
            and_(
                bindings.c.transcript_id == Transcript.id,
                bindings.c.call_id == KeywordMatch.call_id,
            ),
        )
        .where(
            Transcript.status == TranscriptStatus.COMPLETED,
            KeywordMatch.call_id.in_(call_ids),
            match_speaker_scope,
        )
    )
    if keyword_id:
        query = query.where(Keyword.id == keyword_id)
    if keyword_text:
        escaped = keyword_text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(Keyword.canonical_phrase.ilike(f"%{escaped}%", escape="\\"))
    if category_id:
        query = query.where(Keyword.category_id == category_id)
    rows = (await db.execute(query.order_by(KeywordMatch.start_seconds))).all()
    grouped: dict[tuple[UUID, UUID], list] = defaultdict(list)
    for match, keyword, category, bound_operator_id in rows:
        grouped[(match.call_id, bound_operator_id)].append((match, keyword, category))
    return grouped


async def _latest_item_statuses(
    db: AsyncSession,
    pairs: list[tuple[Call, Operator]],
    *,
    job_id: UUID,
) -> dict[tuple[UUID, UUID], str]:
    """Return the latest analysis state for each exact call/operator pair."""
    keys = {(call.id, operator.id) for call, operator in pairs}
    if not keys:
        return {}
    conditions = [
        (ProcessingJobItem.call_id == call_id) & (ProcessingJobItem.operator_id == operator_id)
        for call_id, operator_id in keys
    ]
    items = (
        await db.scalars(
            select(ProcessingJobItem)
            .where(ProcessingJobItem.job_id == job_id, or_(*conditions))
            .order_by(
                ProcessingJobItem.updated_at.desc(),
                ProcessingJobItem.created_at.desc(),
                ProcessingJobItem.id.desc(),
            )
        )
    ).all()
    result: dict[tuple[UUID, UUID], str] = {}
    for item in items:
        key = (item.call_id, item.operator_id)
        result.setdefault(key, item.status.value)
    return result


async def _resolve_result_job(db: AsyncSession, job_id: UUID | None) -> ProcessingJob | None:
    if job_id is not None:
        job = await db.get(ProcessingJob, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found.",
            )
        return job
    return await db.scalar(
        select(ProcessingJob)
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
        .limit(1)
    )


def _match_count_expression(
    *,
    job_id: UUID,
    include_all_speakers: bool,
    keyword_id: UUID | None,
    keyword: str | None,
    category_id: UUID | None,
):
    match_speaker_scope = (
        and_(
            or_(
                KeywordMatch.operator_id == Operator.id,
                KeywordMatch.operator_id.is_(None),
            ),
            or_(
                TranscriptSegment.operator_id == Operator.id,
                TranscriptSegment.operator_id.is_(None),
            ),
        )
        if include_all_speakers
        else and_(
            KeywordMatch.operator_id == Operator.id,
            TranscriptSegment.operator_id == Operator.id,
        )
    )
    query = (
        select(func.count(distinct(KeywordMatch.id)))
        .join(TranscriptSegment, TranscriptSegment.id == KeywordMatch.transcript_segment_id)
        .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
        .join(
            ProcessingJobItem,
            and_(
                ProcessingJobItem.job_id == job_id,
                ProcessingJobItem.call_id == Call.id,
                ProcessingJobItem.result_transcript_id == Transcript.id,
                or_(
                    ProcessingJobItem.operator_id == Operator.id,
                    Transcript.operator_id.is_(None),
                ),
            ),
        )
        .where(
            KeywordMatch.call_id == Call.id,
            Transcript.status == TranscriptStatus.COMPLETED,
            match_speaker_scope,
        )
    )
    if keyword_id or keyword or category_id:
        query = query.join(Keyword, Keyword.id == KeywordMatch.keyword_id)
    if keyword_id:
        query = query.where(KeywordMatch.keyword_id == keyword_id)
    if keyword:
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        query = query.where(Keyword.canonical_phrase.ilike(f"%{escaped}%", escape="\\"))
    if category_id:
        query = query.where(Keyword.category_id == category_id)
    return query.correlate(Call, Operator).scalar_subquery()


def _validate_match_filter_combination(
    *,
    has_matches: bool | None,
    keyword_id: UUID | None,
    keyword: str | None,
    category_id: UUID | None,
) -> None:
    if has_matches is False and (keyword_id or keyword or category_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No-match filtering cannot be combined with a saved keyword or keyword category."
            ),
        )


@router.get("/results", response_model=Page[ResultItem])
async def results(
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    job_id: UUID | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    operator_id: UUID | None = None,
    keyword_id: UUID | None = None,
    keyword: str | None = None,
    transcript_query: str | None = None,
    category_id: UUID | None = None,
    direction: str | None = None,
    has_matches: bool | None = None,
    sort: str = "occurred_at",
    order: str = "desc",
    page: int = 1,
    page_size: int = 50,
) -> Page[ResultItem]:
    page, page_size = max(1, page), min(200, max(1, page_size))
    if direction and direction.lower() not in {item.value for item in Direction}:
        raise HTTPException(status_code=422, detail="Invalid direction filter.")
    if sort not in {"occurred_at", "operator", "duration_seconds", "match_count"}:
        raise HTTPException(status_code=422, detail="Invalid sort field.")
    if order not in {"asc", "desc"}:
        raise HTTPException(status_code=422, detail="Invalid sort order.")
    _validate_match_filter_combination(
        has_matches=has_matches,
        keyword_id=keyword_id,
        keyword=keyword,
        category_id=category_id,
    )
    job = await _resolve_result_job(db, job_id)
    if job is None:
        return Page(items=[], total=0, page=page, page_size=page_size)
    application = await load_application_settings(db, get_settings())
    tz = ZoneInfo(str(application["default_timezone"]))
    from_utc = _parse_filter_boundary(date_from, tz, field="date_from", end=False)
    to_utc = _parse_filter_boundary(date_to, tz, field="date_to", end=True)
    normalized_transcript_query = _normalized_transcript_query(transcript_query)
    base = _result_query(
        job_id=job.id,
        include_all_speakers=job.include_all_speakers,
        date_from=from_utc,
        date_to=to_utc,
        operator_id=operator_id,
        keyword_id=keyword_id,
        keyword=keyword,
        transcript_query=normalized_transcript_query,
        category_id=category_id,
        direction=direction,
        has_matches=has_matches,
    )
    total = await db.scalar(select(func.count()).select_from(base.subquery())) or 0
    match_count_sort = _match_count_expression(
        job_id=job.id,
        include_all_speakers=job.include_all_speakers,
        keyword_id=keyword_id,
        keyword=keyword,
        category_id=category_id,
    )
    sort_expression = {
        "occurred_at": Call.started_at,
        "operator": Operator.display_name,
        "duration_seconds": Call.duration_seconds,
        "match_count": match_count_sort,
    }[sort]
    sort_expression = sort_expression.asc() if order == "asc" else sort_expression.desc()
    pairs = (
        await db.execute(
            base.order_by(sort_expression, Call.id).offset((page - 1) * page_size).limit(page_size)
        )
    ).all()
    grouped = await _matches_for_pairs(
        db,
        pairs,
        job_id=job.id,
        include_all_speakers=job.include_all_speakers,
        keyword_id=keyword_id,
        keyword_text=keyword,
        category_id=category_id,
    )
    item_statuses = await _latest_item_statuses(db, pairs, job_id=job.id)
    items = []
    for call, operator in pairs:
        matches = grouped.get((call.id, operator.id), [])
        items.append(
            ResultItem(
                job_id=job.id,
                call_id=call.id,
                started_at=call.started_at,
                operator_id=operator.id,
                operator_name=operator.display_name,
                masked_phone_number=mask_phone_number(_external_number(call)),
                duration_seconds=call.duration_seconds,
                direction=call.direction.value,
                keywords_found=list(
                    dict.fromkeys(keyword.canonical_phrase for _, keyword, _ in matches)
                ),
                match_count=len(matches),
                processing_status=item_statuses.get((call.id, operator.id), call.processing_status),
            )
        )
    return Page(items=items, total=total, page=page, page_size=page_size)


@router.get("/results/export.csv")
async def export_results(
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    job_id: UUID | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    operator_id: UUID | None = None,
    keyword_id: UUID | None = None,
    keyword: str | None = None,
    transcript_query: str | None = None,
    category_id: UUID | None = None,
    direction: str | None = None,
    has_matches: bool | None = None,
    sort: str = "occurred_at",
    order: str = "desc",
) -> StreamingResponse:
    if direction and direction.lower() not in {item.value for item in Direction}:
        raise HTTPException(status_code=422, detail="Invalid direction filter.")
    if sort not in {"occurred_at", "operator", "duration_seconds", "match_count"} or order not in {
        "asc",
        "desc",
    }:
        raise HTTPException(status_code=422, detail="Invalid export sorting.")
    _validate_match_filter_combination(
        has_matches=has_matches,
        keyword_id=keyword_id,
        keyword=keyword,
        category_id=category_id,
    )
    job = await _resolve_result_job(db, job_id)
    application = await load_application_settings(db, get_settings())
    timezone = ZoneInfo(str(application["default_timezone"]))
    from_utc = _parse_filter_boundary(date_from, timezone, field="date_from", end=False)
    to_utc = _parse_filter_boundary(date_to, timezone, field="date_to", end=True)
    normalized_transcript_query = _normalized_transcript_query(transcript_query)
    if job is None:
        pairs = []
    else:
        base = _result_query(
            job_id=job.id,
            include_all_speakers=job.include_all_speakers,
            date_from=from_utc,
            date_to=to_utc,
            operator_id=operator_id,
            keyword_id=keyword_id,
            keyword=keyword,
            transcript_query=normalized_transcript_query,
            category_id=category_id,
            direction=direction,
            has_matches=has_matches,
        )
        match_count_sort = _match_count_expression(
            job_id=job.id,
            include_all_speakers=job.include_all_speakers,
            keyword_id=keyword_id,
            keyword=keyword,
            category_id=category_id,
        )
        sort_expression = {
            "occurred_at": Call.started_at,
            "operator": Operator.display_name,
            "duration_seconds": Call.duration_seconds,
            "match_count": match_count_sort,
        }[sort]
        sort_expression = sort_expression.asc() if order == "asc" else sort_expression.desc()
        pairs = (await db.execute(base.order_by(sort_expression, Call.id).limit(100_001))).all()
    if len(pairs) > 100_000:
        raise HTTPException(
            status_code=422, detail="Export is too large. Choose a narrower date range."
        )
    grouped = (
        await _matches_for_pairs(
            db,
            pairs,
            job_id=job.id,
            include_all_speakers=job.include_all_speakers,
            keyword_id=keyword_id,
            keyword_text=keyword,
            category_id=category_id,
        )
        if job is not None
        else {}
    )

    def rows() -> Iterator[list[Any]]:
        yield [
            "Date",
            "Time",
            "Operator",
            "Phone number",
            "Duration",
            "Direction",
            "Keyword category",
            "Keyword",
            "Transcript excerpt",
            "Timestamp",
            "Match count",
        ]
        for call, operator in pairs:
            started_at = call.started_at
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            local_started = started_at.astimezone(timezone)
            matches = grouped.get((call.id, operator.id), [])
            if not matches:
                yield [
                    local_started.date().isoformat(),
                    local_started.time().replace(microsecond=0).isoformat(),
                    operator.display_name,
                    _external_number(call),
                    call.duration_seconds,
                    call.direction.value,
                    "",
                    "",
                    "",
                    "",
                    0,
                ]
            for match, keyword, category in matches:
                excerpt = " ".join(
                    part
                    for part in (
                        match.context_before,
                        match.original_matched_text,
                        match.context_after,
                    )
                    if part
                )
                yield [
                    local_started.date().isoformat(),
                    local_started.time().replace(microsecond=0).isoformat(),
                    operator.display_name,
                    _external_number(call),
                    call.duration_seconds,
                    call.direction.value,
                    category.name,
                    keyword.canonical_phrase,
                    excerpt,
                    match.start_seconds,
                    len(matches),
                ]

    await audit(
        db,
        action="results.export",
        request=request,
        user=user,
        resource_type="results",
        details={"rows": len(pairs), "truncated": False},
    )
    await db.commit()
    return StreamingResponse(
        csv_bytes(rows()),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="call-results.csv"'},
    )


def _match_response(match: KeywordMatch) -> MatchResponse:
    return MatchResponse(
        id=match.id,
        transcript_segment_id=match.transcript_segment_id,
        keyword_id=match.keyword_id,
        keyword=match.keyword.canonical_phrase,
        category=match.keyword.category.name,
        original_matched_text=match.original_matched_text,
        context_before=match.context_before,
        context_after=match.context_after,
        start_timestamp=match.start_seconds,
        end_timestamp=match.end_seconds,
        match_method=match.match_method.value,
        match_score=match.match_score,
    )


_MANUAL_ASSIGNMENT_ALLOWED_STATUSES = {
    SpeakerAttributionStatus.CHANNEL_UNKNOWN,
    SpeakerAttributionStatus.CALLER_CALLEE_ONLY,
    SpeakerAttributionStatus.MANUALLY_ASSIGNED,
}

_NON_SPEAKER_LABELS = {
    "agent",
    "channel 0",
    "channel 1",
    "channel a",
    "channel b",
    "operator",
    "speaker 0",
    "speaker 1",
    "unknown",
    "unknown speaker",
}


def _available_channels(segments: list[TranscriptSegment]) -> list[int]:
    return sorted({segment.channel_index for segment in segments if segment.channel_index is not None})


def _speaker_assignment_required(
    transcript: Transcript,
    segments: list[TranscriptSegment],
) -> bool:
    if transcript.transcription_mode != TranscriptionMode.DUAL_CHANNEL:
        return False
    if transcript.speaker_attribution_status in {
        SpeakerAttributionStatus.MANUALLY_ASSIGNED,
        SpeakerAttributionStatus.CONFIRMED_BY_PBX,
        SpeakerAttributionStatus.ANONYMOUS_DIARIZATION,
    }:
        return False
    return any(
        segment.operator_id is None and segment.channel_index is not None for segment in segments
    )


def _opposite_channel_label(
    call: Call,
    opposite_channel: int,
    opposite_segments: list[TranscriptSegment],
) -> str:
    """Choose a non-operator label without inventing attribution confidence."""

    preserved = {
        segment.speaker_label
        for segment in opposite_segments
        if segment.channel_index == opposite_channel
        and segment.speaker_label
        and segment.speaker_label.strip().casefold() not in _NON_SPEAKER_LABELS
    }
    if preserved:
        return sorted(preserved)[0]
    if call.direction == Direction.OUTBOUND:
        return "Customer"
    if call.direction == Direction.INBOUND:
        return "Customer"
    return "Customer"


def _transcript_confidence_status(transcript: Transcript) -> str:
    summary = transcript.quality_summary
    if isinstance(summary, dict):
        value = summary.get("confidence_status")
        if isinstance(value, str) and value:
            return value
    return "unavailable"


def _segment_quality_flags(segments: list[TranscriptSegment]) -> list[str]:
    flags: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        for flag in segment.quality_flags or []:
            if flag and flag not in seen:
                seen.add(flag)
                flags.append(flag)
    return flags


async def _keyword_definitions_for_manual(
    db: AsyncSession,
) -> tuple[list[KeywordDefinition], dict[str, Keyword]]:
    keywords = (
        await db.scalars(
            select(Keyword)
            .options(selectinload(Keyword.variants))
            .where(Keyword.active.is_(True), Keyword.deleted_at.is_(None))
        )
    ).all()
    definitions = [
        KeywordDefinition(
            id=str(keyword.id),
            phrase=keyword.canonical_phrase,
            variants=tuple(item.phrase for item in keyword.variants),
            accent_insensitive=keyword.accent_insensitive,
            whole_word=keyword.whole_word,
            exact_phrase=keyword.exact_phrase,
            fuzzy_match=keyword.fuzzy_match,
            fuzzy_threshold=keyword.fuzzy_threshold,
        )
        for keyword in keywords
    ]
    return definitions, {str(item.id): item for item in keywords}


async def _rebuild_operator_matches(
    db: AsyncSession,
    transcript: Transcript,
    operator_id: UUID,
) -> int:
    """Re-run keyword matching under normal operator-only rules."""

    definitions, keyword_by_id = await _keyword_definitions_for_manual(db)
    if not definitions:
        return 0
    segments = (
        await db.scalars(
            select(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript.id)
        )
    ).all()
    added = 0
    for segment in segments:
        if segment.operator_id != operator_id:
            continue
        for found in match_text(segment.original_text, definitions):
            keyword = keyword_by_id[found.keyword_id]
            db.add(
                KeywordMatch(
                    keyword_id=keyword.id,
                    operator_id=operator_id,
                    call_id=transcript.call_id,
                    transcript_segment_id=segment.id,
                    original_matched_text=found.original_matched_text,
                    normalized_match=found.normalized_match,
                    context_before=found.context_before,
                    context_after=found.context_after,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    match_method=found.method,
                    match_score=Decimal(str(round(found.score, 3))),
                )
            )
            added += 1
    await db.flush()
    return added


@router.patch(
    "/calls/{call_id}/speaker-assignment",
    response_model=SpeakerAssignmentResponse,
)
async def assign_operator_channel(
    call_id: UUID,
    payload: SpeakerAssignmentRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SpeakerAssignmentResponse:
    """Correct dual-channel speaker attribution without retranscribing audio."""

    call = await db.scalar(select(Call).where(Call.id == call_id).with_for_update())
    if call is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found.")

    transcript = await db.scalar(
        select(Transcript).where(Transcript.id == payload.transcript_id).with_for_update()
    )
    if transcript is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Transcript not found."
        )
    if transcript.call_id != call.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The transcript does not belong to this call.",
        )
    if not transcript.is_current:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only the current transcript can be manually assigned.",
        )
    if transcript.status != TranscriptStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The transcript must be completed before speaker assignment.",
        )
    if transcript.transcription_mode != TranscriptionMode.DUAL_CHANNEL:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Manual assignment is only available for separated dual-channel transcripts.",
        )
    if (
        transcript.speaker_attribution_status is not None
        and transcript.speaker_attribution_status not in _MANUAL_ASSIGNMENT_ALLOWED_STATUSES
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This transcript's speaker attribution cannot be manually assigned.",
        )

    operator = await db.get(Operator, payload.operator_id)
    if operator is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Operator not found."
        )
    relevant_participant_id = await db.scalar(
        select(CallParticipant.id)
        .where(
            CallParticipant.call_id == call.id,
            CallParticipant.operator_id == operator.id,
        )
        .limit(1)
    )
    if relevant_participant_id is None:
        relevant_participant_id = await db.scalar(
            select(ProcessingJobItem.id)
            .where(
                ProcessingJobItem.call_id == call.id,
                ProcessingJobItem.operator_id == operator.id,
            )
            .limit(1)
        )
    if relevant_participant_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The selected operator was not part of this call.",
        )

    segments = list(
        (
            await db.scalars(
                select(TranscriptSegment)
                .where(TranscriptSegment.transcript_id == transcript.id)
                .order_by(TranscriptSegment.sequence_number, TranscriptSegment.id)
                .with_for_update()
            )
        ).all()
    )
    available_channels = _available_channels(segments)
    if payload.operator_channel_index not in available_channels:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The requested channel is not present in this transcript.",
        )

    requested_channel = payload.operator_channel_index
    opposite_channel = 1 - requested_channel
    opposite_segments = [
        segment for segment in segments if segment.channel_index == opposite_channel
    ]
    opposite_label = _opposite_channel_label(call, opposite_channel, opposite_segments)
    previous_status = (
        transcript.speaker_attribution_status.value
        if transcript.speaker_attribution_status is not None
        else None
    )

    for segment in segments:
        if segment.channel_index == requested_channel:
            segment.operator_id = operator.id
            segment.speaker_label = operator.display_name
            segment.speaker_source = SpeakerSource.MANUAL_OVERRIDE
        elif segment.channel_index == opposite_channel:
            segment.operator_id = None
            segment.speaker_label = opposite_label
            segment.speaker_source = SpeakerSource.UNKNOWN
            flag_modified(segment, "operator_id")

    transcript.operator_id = operator.id
    transcript.speaker_attribution_status = SpeakerAttributionStatus.MANUALLY_ASSIGNED

    segment_ids = [segment.id for segment in segments]
    if segment_ids:
        await db.execute(
            delete(KeywordMatch).where(KeywordMatch.transcript_segment_id.in_(segment_ids))
        )
    await db.flush()
    await _rebuild_operator_matches(db, transcript, operator.id)

    call.processing_status = "completed"
    call.last_error_category = None
    call.last_error_message = None

    await audit(
        db,
        action="call.speaker_assignment",
        request=request,
        user=user,
        resource_type="transcript",
        resource_id=str(transcript.id),
        details={
            "call_id": str(call.id),
            "transcript_id": str(transcript.id),
            "operator_id": str(operator.id),
            "operator_channel_index": requested_channel,
            "previous_attribution_status": previous_status,
        },
    )
    await db.commit()

    return SpeakerAssignmentResponse(
        transcript_id=transcript.id,
        transcription_mode=(
            transcript.transcription_mode.value
            if transcript.transcription_mode is not None
            else None
        ),
        speaker_attribution_status=(
            transcript.speaker_attribution_status.value
            if transcript.speaker_attribution_status is not None
            else None
        ),
        speaker_assignment_required=_speaker_assignment_required(transcript, segments),
        available_channels=available_channels,
        operator_id=operator.id,
        operator_channel_index=requested_channel,
        confidence_status=_transcript_confidence_status(transcript),
        quality_flags=_segment_quality_flags(segments),
        pipeline_version=transcript.pipeline_version,
    )


@router.get("/calls/{call_id}", response_model=CallDetailResponse)
async def call_detail(
    call_id: UUID,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
    job_id: UUID | None = None,
) -> CallDetailResponse:
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found.")
    participants = (
        await db.scalars(
            select(CallParticipant)
            .options(selectinload(CallParticipant.operator))
            .where(CallParticipant.call_id == call.id)
            .order_by(CallParticipant.created_at)
        )
    ).all()
    transcript_conditions = [
        Transcript.call_id == call.id,
        Transcript.status == TranscriptStatus.COMPLETED,
    ]
    if job_id is None:
        transcript_conditions.append(Transcript.is_current.is_(True))
    else:
        await _resolve_result_job(db, job_id)
        bound_transcript_ids = select(ProcessingJobItem.result_transcript_id).where(
            ProcessingJobItem.job_id == job_id,
            ProcessingJobItem.call_id == call.id,
            ProcessingJobItem.result_transcript_id.is_not(None),
        )
        transcript_conditions.append(Transcript.id.in_(bound_transcript_ids))
    selected_transcripts = (
        await db.scalars(
            select(Transcript)
            .where(*transcript_conditions)
            .order_by(Transcript.created_at, Transcript.id)
        )
    ).all()
    segment_conditions = [
        TranscriptSegment.call_id == call.id,
        *transcript_conditions,
    ]
    segments = (
        await db.scalars(
            select(TranscriptSegment)
            .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
            .options(
                selectinload(TranscriptSegment.matches)
                .selectinload(KeywordMatch.keyword)
                .selectinload(Keyword.category)
            )
            .where(*segment_conditions)
            .order_by(TranscriptSegment.start_seconds, TranscriptSegment.sequence_number)
        )
    ).all()
    responses: list[TranscriptSegmentResponse] = []
    top_matches: list[MatchResponse] = []
    for segment in segments:
        matches = [_match_response(match) for match in segment.matches]
        top_matches.extend(matches)
        responses.append(
            TranscriptSegmentResponse(
                id=segment.id,
                operator_id=segment.operator_id,
                speaker_label=segment.speaker_label,
                speaker_source=segment.speaker_source.value,
                start_timestamp=segment.start_seconds,
                end_timestamp=segment.end_seconds,
                original_text=segment.original_text,
                confidence=segment.confidence,
                transcription_model=segment.transcription_model,
                mean_logprob=segment.mean_logprob,
                low_logprob_ratio=segment.low_logprob_ratio,
                quality_flags=list(segment.quality_flags or []),
                audio_variant=segment.audio_variant,
                channel_index=segment.channel_index,
                sequence_number=segment.sequence_number,
                matches=matches,
            )
        )
    history = (
        await db.scalars(
            select(ProcessingJobItem)
            .where(ProcessingJobItem.call_id == call.id)
            .order_by(ProcessingJobItem.created_at.desc())
        )
    ).all()
    recordings = (await db.scalars(select(Recording).where(Recording.call_id == call.id))).all()
    local_audio_available = False
    if len(recordings) == 1 and recordings[0].storage_key:
        try:
            local_audio_available = (
                AudioProcessor(settings).safe_storage_path(recordings[0].storage_key).is_file()
            )
        except Exception:
            local_audio_available = False
    primary_transcript = selected_transcripts[0] if selected_transcripts else None
    available_channels = _available_channels(segments)
    return CallDetailResponse(
        id=call.id,
        transcript_id=primary_transcript.id if primary_transcript is not None else None,
        started_at=call.started_at,
        caller=call.caller_number,
        caller_name=call.caller_name,
        callee=call.callee_number,
        callee_name=call.callee_name,
        duration_seconds=call.duration_seconds,
        direction=call.direction.value,
        call_status=call.call_status,
        queue=call.queue_name,
        processing_status=call.processing_status,
        audio_available=(
            len(recordings) == 1 and (local_audio_available or settings.yeastar_configured)
        ),
        participants=[
            {
                "operator_id": str(item.operator_id) if item.operator_id else None,
                "operator_name": item.operator.display_name if item.operator else None,
                "extension": item.provider_extension_number,
                "role": item.role.value,
                "answered": item.answered,
            }
            for item in participants
        ],
        matches=sorted(top_matches, key=lambda item: item.start_timestamp),
        transcript_segments=responses,
        transcription_mode=(
            primary_transcript.transcription_mode.value
            if primary_transcript is not None and primary_transcript.transcription_mode is not None
            else None
        ),
        speaker_attribution_status=(
            primary_transcript.speaker_attribution_status.value
            if primary_transcript is not None
            and primary_transcript.speaker_attribution_status is not None
            else None
        ),
        speaker_assignment_required=(
            _speaker_assignment_required(primary_transcript, segments)
            if primary_transcript is not None
            else False
        ),
        available_channels=available_channels,
        confidence_status=(
            _transcript_confidence_status(primary_transcript)
            if primary_transcript is not None
            else "unavailable"
        ),
        quality_flags=_segment_quality_flags(segments),
        pipeline_version=primary_transcript.pipeline_version if primary_transcript is not None else None,
        transcript_quality_summaries=[
            TranscriptQualitySummaryResponse(
                transcript_id=transcript.id,
                transcription_mode=(
                    transcript.transcription_mode.value
                    if transcript.transcription_mode is not None
                    else None
                ),
                quality_summary=(
                    dict(transcript.quality_summary)
                    if transcript.quality_summary is not None
                    else None
                ),
            )
            for transcript in selected_transcripts
        ],
        processing_history=[
            {
                "status": item.status.value,
                "stage": item.stage,
                "attempt": item.attempt_count,
                "updated_at": item.updated_at.isoformat(),
                "error": item.error_message,
                "message": item.stage
                if not item.error_message
                else f"{item.stage}: {item.error_message}",
                "created_at": item.created_at.isoformat(),
                "occurred_at": item.updated_at.isoformat(),
            }
            for item in history
        ],
    )


async def _lock_current_reprocess_transcript(
    db: AsyncSession,
    *,
    call_id: UUID,
    transcript_id: UUID | None,
) -> Transcript:
    """Lock a current transcript in the shared Recording-before-Transcript order."""

    statement = (
        select(Transcript)
        .where(
            Transcript.call_id == call_id,
            Transcript.status == TranscriptStatus.COMPLETED,
            Transcript.is_current.is_(True),
        )
        .order_by(
            Transcript.completed_at.desc().nulls_last(),
            Transcript.updated_at.desc().nulls_last(),
            Transcript.created_at.desc().nulls_last(),
            Transcript.id.desc(),
        )
    )
    if transcript_id is not None:
        statement = statement.where(Transcript.id == transcript_id)

    candidates = (await db.scalars(statement)).all()
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This call has no completed current transcript to reprocess.",
        )
    if len(candidates) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Choose the transcript to reprocess for this call.",
        )

    locked_recording_id = await db.scalar(
        select(Recording.id).where(Recording.id == candidates[0].recording_id).with_for_update()
    )
    if locked_recording_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The recording selected for reprocessing is unavailable.",
        )

    # Retention may have won the recording lock and removed the candidate while
    # this request waited. Re-read and lock the authoritative current row only
    # after the Recording lock has been acquired.
    current = (await db.scalars(statement.with_for_update())).all()
    if not current:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This call has no completed current transcript to reprocess.",
        )
    if len(current) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Choose the transcript to reprocess for this call.",
        )
    if current[0].recording_id != locked_recording_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The transcript selected for reprocessing changed. Try again.",
        )
    return current[0]


@router.post(
    "/calls/{call_id}/reprocess",
    response_model=JobDetail,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reprocess_call(
    call_id: UUID,
    payload: CallReprocessRequest,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveRuntimeSettings,
) -> JobDetail:
    """Queue an explicit replacement while keeping the current transcript readable."""
    _require_integrations(settings)
    if payload.pipeline_version == "pipeline-v2" and not (
        settings.TRANSCRIPTION_PIPELINE_V2_ENABLED
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Pipeline V2 is disabled.",
        )
    # An active worker owns ProcessingJob, so reject it before taking target
    # locks. Retention and discovery lock Recording before job/transcript rows;
    # take the same Recording-first path before locking Call so a failed-job
    # retry (job then call) cannot complete a recording/job/call lock cycle.
    if await active_job(db, lock=True) is not None:
        raise active_job_conflict()
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found.")

    current_transcript = await _lock_current_reprocess_transcript(
        db,
        call_id=call.id,
        transcript_id=payload.transcript_id,
    )
    call = await db.scalar(select(Call).where(Call.id == call_id).with_for_update())
    if call is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found.")

    source_item_statement = (
        select(ProcessingJobItem)
        .where(
            ProcessingJobItem.call_id == call.id,
            ProcessingJobItem.recording_id == current_transcript.recording_id,
            ProcessingJobItem.result_transcript_id == current_transcript.id,
            ProcessingJobItem.status == ItemStatus.COMPLETED,
        )
        .order_by(
            ProcessingJobItem.updated_at.desc(),
            ProcessingJobItem.created_at.desc(),
            ProcessingJobItem.id.desc(),
        )
    )
    if current_transcript.operator_id is not None:
        source_item_statement = source_item_statement.where(
            ProcessingJobItem.operator_id == current_transcript.operator_id
        )
    source_items = (await db.scalars(source_item_statement)).all()
    source_item = source_items[0] if source_items else None
    source_operator_ids = {item.operator_id for item in source_items}
    if current_transcript.operator_id is not None:
        if payload.operator_id not in {None, current_transcript.operator_id}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The selected operator does not own this transcript.",
            )
        operator_id = current_transcript.operator_id
    elif payload.operator_id is not None:
        if payload.operator_id not in source_operator_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The selected operator was not part of this transcript's analysis.",
            )
        operator_id = payload.operator_id
        source_item = next(item for item in source_items if item.operator_id == payload.operator_id)
    elif len(source_operator_ids) == 1:
        operator_id = next(iter(source_operator_ids))
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Choose the operator whose unattributed transcript should be reprocessed.",
        )
    source_job = await db.get(ProcessingJob, source_item.job_id) if source_item else None

    job_id = uuid4()
    item_id = uuid4()
    idempotency_key = hashlib.sha256(f"call-reprocess:{job_id}".encode("utf-8")).hexdigest()
    date_from = call.started_at
    minimum_date_to = date_from + timedelta(seconds=1)
    date_to = call.ended_at if call.ended_at and call.ended_at > date_from else minimum_date_to
    job = ProcessingJob(
        id=job_id,
        idempotency_key=idempotency_key,
        requested_by_id=user.id,
        status=JobStatus.QUEUED,
        date_from=date_from,
        date_to=max(date_to, minimum_date_to),
        direction=call.direction.value,
        recording_available=True,
        include_all_speakers=(
            source_job.include_all_speakers
            if source_job is not None
            else current_transcript.operator_id is None
        ),
        selected_operator_ids=[str(operator_id)],
        selected_category_ids=(
            list(source_job.selected_category_ids) if source_job is not None else []
        ),
        request_filters={
            "_reprocess": True,
            "_reprocess_targets": {str(item_id): str(current_transcript.id)},
        },
        current_stage="Queued for reprocessing",
        calls_found=1,
        recordings_found=1,
    )
    db.add(job)
    item = ProcessingJobItem(
        id=item_id,
        job_id=job.id,
        call_id=call.id,
        operator_id=operator_id,
        recording_id=current_transcript.recording_id,
        idempotency_key=hashlib.sha256(
            f"reprocess-item:{job.id}:{current_transcript.id}".encode("utf-8")
        ).hexdigest(),
        requested_pipeline_version=payload.pipeline_version,
        status=ItemStatus.QUEUED,
        stage="queued_for_reprocessing",
    )
    db.add(item)
    try:
        await audit(
            db,
            action="call.reprocess",
            request=request,
            user=user,
            resource_type="call",
            resource_id=str(call.id),
            details={
                "job_id": str(job.id),
                "item_id": str(item.id),
                "source_transcript_id": str(current_transcript.id),
                "operator_id": str(operator_id),
                "pipeline_version": payload.pipeline_version,
            },
        )
        await db.commit()
    except IntegrityError:
        await db.rollback()
        if await active_job(db) is not None:
            raise active_job_conflict() from None
        raise

    await db.refresh(job, attribute_names=["items"])
    try:
        task = process_job_item.delay(str(item.id))
    except Exception as exc:
        completed_at = utc_now()
        job.status = JobStatus.FAILED
        job.current_stage = "Reprocessing could not be queued"
        job.last_error_category = "processing_service"
        job.last_error_message = "Processing service is unavailable."
        job.completed_at = completed_at
        item.status = ItemStatus.FAILED
        item.stage = "failed"
        item.error_category = "processing_service"
        item.error_message = "Processing service is unavailable."
        item.completed_at = completed_at
        try:
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("Could not persist failed reprocess dispatch state")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing service is unavailable.",
        ) from exc
    item.celery_task_id = task.id
    queued_response = job_detail(job, is_current=True)
    try:
        await db.commit()
    except Exception:
        # The task has already been acknowledged by Celery and the durable job
        # was committed before dispatch.  Roll back only the optional task-id
        # update; reporting a failure here could cause a duplicate user retry.
        await db.rollback()
        logger.exception("Reprocess task was queued but its task id was not persisted")
        return queued_response
    await db.refresh(job, attribute_names=["items"])
    return job_detail(job, is_current=True)


@router.post(
    "/calls/{call_id}/retry", response_model=MessageResponse, status_code=status.HTTP_202_ACCEPTED
)
async def retry_call(
    call_id: UUID,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
) -> MessageResponse:
    _require_integrations(settings)
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found.")
    target_job_id = await db.scalar(
        select(ProcessingJobItem.job_id)
        .join(ProcessingJob, ProcessingJob.id == ProcessingJobItem.job_id)
        .where(
            ProcessingJobItem.call_id == call.id,
            ProcessingJobItem.status == ItemStatus.FAILED,
            ProcessingJobItem.recording_id.is_not(None),
        )
        .order_by(ProcessingJob.created_at.desc(), ProcessingJob.id.desc())
        .limit(1)
    )
    if target_job_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="There is no failed call to retry.",
        )
    job = await db.scalar(
        select(ProcessingJob)
        .options(selectinload(ProcessingJob.items))
        .where(ProcessingJob.id == target_job_id)
        .with_for_update()
    )
    if job is None or job.status not in TERMINAL_JOB_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This call is already being processed.",
        )
    if await active_job(db, exclude_id=job.id, lock=True) is not None:
        raise active_job_conflict()
    items = [
        item
        for item in job.items
        if item.call_id == call.id
        and item.status == ItemStatus.FAILED
        and item.recording_id is not None
    ]
    if not items:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="There is no failed call to retry.",
        )
    job.status = JobStatus.QUEUED
    job.current_stage = "Queued for call retry"
    job.attempt_count += 1
    job.cancellation_requested = False
    job.completed_at = None
    job.last_error_category = None
    job.last_error_message = None
    for item in items:
        item.status = ItemStatus.QUEUED
        item.stage = "queued"
        item.error_category = None
        item.error_message = None
        item.completed_at = None
    statuses_by_call: dict[UUID, list[ItemStatus]] = defaultdict(list)
    for item in job.items:
        statuses_by_call[item.call_id].append(item.status)
    final_items = sum(item.status in FINAL_ITEM_STATUSES for item in job.items)
    job.calls_completed = sum(
        bool(states) and all(state == ItemStatus.COMPLETED for state in states)
        for states in statuses_by_call.values()
    )
    job.calls_failed = sum(
        any(state == ItemStatus.FAILED for state in states) for states in statuses_by_call.values()
    ) + int((job.request_filters or {}).get("_discovery_failures", 0))
    job.progress_percent = round(final_items * 100 / len(job.items)) if job.items else 0
    call.processing_status = "queued"
    call.last_error_category = None
    call.last_error_message = None
    await audit(
        db,
        action="call.retry",
        request=request,
        user=user,
        resource_type="call",
        resource_id=str(call.id),
        details={"items": len(items), "job_id": str(job.id)},
    )
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise active_job_conflict() from None
    try:
        for item in items:
            task = process_job_item.delay(str(item.id))
            item.celery_task_id = task.id
        await db.commit()
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.current_stage = "Call retry could not be queued"
        job.last_error_category = "processing_service"
        job.last_error_message = "Processing service is unavailable."
        job.completed_at = utc_now()
        for item in items:
            if item.status not in FINAL_ITEM_STATUSES:
                item.status = ItemStatus.FAILED
                item.stage = "failed"
                item.error_category = "processing_service"
                item.error_message = "Processing service is unavailable."
                item.completed_at = utc_now()
        call.processing_status = "failed"
        call.last_error_category = "processing_service"
        call.last_error_message = "Processing service is unavailable."
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing service is unavailable.",
        ) from exc
    return MessageResponse(message="Call queued for retry.")


def parse_range_header(value: str | None, size: int) -> tuple[int, int, bool]:
    if not value:
        return 0, size - 1, False
    if not value.startswith("bytes=") or "," in value:
        raise ValueError("Unsupported range")
    raw = value[6:].strip()
    if "-" not in raw:
        raise ValueError("Invalid range")
    start_text, end_text = raw.split("-", 1)
    if not start_text:
        suffix = int(end_text)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        start = max(0, size - suffix)
        end = size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError("Unsatisfiable range")
    return start, min(end, size - 1), True


_AUDIO_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}


def recording_media_type(recording: Recording, path: Path) -> str:
    provider_type = (recording.mime_type or "").strip()
    if provider_type.lower().startswith("audio/"):
        return provider_type
    declared_extension = (recording.file_extension or "").strip().lower()
    if declared_extension and not declared_extension.startswith("."):
        declared_extension = f".{declared_extension}"
    suffix = (
        declared_extension
        or Path(recording.yeastar_file_name or "").suffix.lower()
        or path.suffix.lower()
    )
    return _AUDIO_MIME_BY_SUFFIX.get(suffix, "audio/wav")


@router.get("/calls/{call_id}/audio")
async def stream_audio(
    call_id: UUID,
    request: Request,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
):
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Call not found.")
    recordings = (await db.scalars(select(Recording).where(Recording.call_id == call.id))).all()
    if not recordings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Recording is unavailable."
        )
    if len(recordings) != 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This call has multiple recordings and cannot be played safely as one file.",
        )
    recording = recordings[0]
    audio = AudioProcessor(settings)
    ephemeral = False
    path = audio.safe_storage_path(recording.storage_key) if recording.storage_key else None
    if path is None or not path.exists():
        if not settings.yeastar_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Recording is unavailable."
            )
        suffix = Path(recording.yeastar_file_name or "").suffix.lower()
        path = audio.safe_storage_path(f"tmp/stream-{uuid4()}{suffix}")
        async with YeastarClient(settings=settings) as client:
            download = await client.download_recording(recording.yeastar_recording_id, path)
        try:
            await audio.inspect(
                path,
                declared_mime_type=str(download.get("content_type") or ""),
                original_filename=recording.yeastar_file_name,
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        ephemeral = True
    size = path.stat().st_size
    try:
        start, end, partial = parse_range_header(request.headers.get("range"), size)
    except (ValueError, TypeError):
        return Response(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )
    length = end - start + 1

    def body() -> Iterator[bytes]:
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        finally:
            if ephemeral:
                path.unlink(missing_ok=True)

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Disposition": 'inline; filename="recording"',
    }
    if partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        body(),
        status_code=status.HTTP_206_PARTIAL_CONTENT if partial else status.HTTP_200_OK,
        media_type=recording_media_type(recording, path),
        headers=headers,
    )
