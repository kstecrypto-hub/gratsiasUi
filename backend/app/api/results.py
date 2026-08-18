from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, distinct, exists, func, not_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.jobs import (
    FINAL_ITEM_STATUSES,
    TERMINAL_JOB_STATUSES,
    _require_integrations,
    active_job,
    active_job_conflict,
    job_summary,
)
from app.api.dependencies import EffectiveYeastarSettings
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
from app.models.enums import Direction, ItemStatus, JobStatus, TranscriptStatus
from app.schemas.common import MessageResponse, Page
from app.schemas.results import (
    CallDetailResponse,
    DashboardResponse,
    MatchResponse,
    ResultItem,
    TranscriptSegmentResponse,
)
from app.services.audit import audit
from app.services.application_settings import load_application_settings
from app.services.audio import AudioProcessor
from app.services.export import csv_bytes, mask_phone_number
from app.services.keyword_matching.normalization import normalize_greek
from app.services.yeastar import YeastarClient
from app.workers.tasks import process_job_item


router = APIRouter(tags=["results"])


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
    calls_with_recordings = await db.scalar(
        select(func.count()).select_from(Call).where(Call.has_recording.is_(True))
    ) or 0
    calls_transcribed = await db.scalar(
        select(func.count(distinct(Transcript.call_id))).where(
            Transcript.status == TranscriptStatus.COMPLETED
        )
    ) or 0
    calls_with_matches = await db.scalar(
        select(func.count(distinct(KeywordMatch.call_id)))
    ) or 0
    failed_call_ids = (
        select(ProcessingJobItem.call_id.label("call_id"))
        .where(ProcessingJobItem.status == ItemStatus.FAILED)
        .union(select(Call.id.label("call_id")).where(Call.processing_status == "failed"))
        .subquery()
    )
    failed_calls = await db.scalar(
        select(func.count()).select_from(failed_call_ids)
    ) or 0
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
            job_summary(job, is_current=index == 0)
            for index, job in enumerate(recent_jobs)
        ],
    )


def _result_query(
    *,
    job_id: UUID,
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
            transcript_query.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        transcript_matches = (
            select(TranscriptSegment.id)
            .join(Transcript, Transcript.id == TranscriptSegment.transcript_id)
            .where(
                TranscriptSegment.call_id == Call.id,
                Transcript.status == TranscriptStatus.COMPLETED,
                or_(
                    TranscriptSegment.operator_id == Operator.id,
                    and_(
                        TranscriptSegment.operator_id.is_(None),
                        Transcript.operator_id == Operator.id,
                    ),
                    and_(
                        TranscriptSegment.operator_id.is_(None),
                        Transcript.operator_id.is_(None),
                    ),
                ),
                TranscriptSegment.normalized_text.ilike(
                    f"%{escaped_query}%", escape="\\"
                ),
            )
        )
        conditions.append(exists(transcript_matches))
    matching = select(KeywordMatch.id).where(
        KeywordMatch.call_id == Call.id,
        or_(
            KeywordMatch.operator_id == Operator.id,
            KeywordMatch.operator_id.is_(None),
        ),
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
    keyword_id: UUID | None = None,
    keyword_text: str | None = None,
    category_id: UUID | None = None,
) -> dict[tuple[UUID, UUID], list[tuple[KeywordMatch, Keyword, KeywordCategory]]]:
    if not pairs:
        return {}
    call_ids = {call.id for call, _ in pairs}
    operator_ids = {operator.id for _, operator in pairs}
    query = (
            select(KeywordMatch, Keyword, KeywordCategory)
            .join(Keyword, Keyword.id == KeywordMatch.keyword_id)
            .join(KeywordCategory, KeywordCategory.id == Keyword.category_id)
            .where(
                KeywordMatch.call_id.in_(call_ids),
                or_(
                    KeywordMatch.operator_id.in_(operator_ids),
                    KeywordMatch.operator_id.is_(None),
                ),
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
    unknown_by_call: dict[UUID, list] = defaultdict(list)
    for match, keyword, category in rows:
        if match.operator_id is None:
            unknown_by_call[match.call_id].append((match, keyword, category))
        else:
            grouped[(match.call_id, match.operator_id)].append((match, keyword, category))
    for call, operator in pairs:
        if unknown_by_call.get(call.id):
            grouped[(call.id, operator.id)].extend(unknown_by_call[call.id])
            grouped[(call.id, operator.id)].sort(key=lambda item: item[0].start_seconds)
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
        (ProcessingJobItem.call_id == call_id)
        & (ProcessingJobItem.operator_id == operator_id)
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


async def _resolve_result_job(
    db: AsyncSession, job_id: UUID | None
) -> ProcessingJob | None:
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
    keyword_id: UUID | None,
    keyword: str | None,
    category_id: UUID | None,
):
    query = select(func.count(KeywordMatch.id)).where(
        KeywordMatch.call_id == Call.id,
        or_(
            KeywordMatch.operator_id == Operator.id,
            KeywordMatch.operator_id.is_(None),
        ),
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
                "No-match filtering cannot be combined with a saved keyword "
                "or keyword category."
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
        await db.execute(base.order_by(sort_expression, Call.id).offset((page - 1) * page_size).limit(page_size))
    ).all()
    grouped = await _matches_for_pairs(
        db,
        pairs,
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
                keywords_found=list(dict.fromkeys(keyword.canonical_phrase for _, keyword, _ in matches)),
                match_count=len(matches),
                processing_status=item_statuses.get(
                    (call.id, operator.id), call.processing_status
                ),
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
    if sort not in {"occurred_at", "operator", "duration_seconds", "match_count"} or order not in {"asc", "desc"}:
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
            await db.execute(base.order_by(sort_expression, Call.id).limit(100_001))
        ).all()
    if len(pairs) > 100_000:
        raise HTTPException(status_code=422, detail="Export is too large. Choose a narrower date range.")
    grouped = await _matches_for_pairs(
        db,
        pairs,
        keyword_id=keyword_id,
        keyword_text=keyword,
        category_id=category_id,
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
                    part for part in (match.context_before, match.original_matched_text, match.context_after) if part
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


@router.get("/calls/{call_id}", response_model=CallDetailResponse)
async def call_detail(
    call_id: UUID,
    user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
    settings: EffectiveYeastarSettings,
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
    segments = (
        await db.scalars(
            select(TranscriptSegment)
            .options(
                selectinload(TranscriptSegment.matches)
                .selectinload(KeywordMatch.keyword)
                .selectinload(Keyword.category)
            )
            .where(TranscriptSegment.call_id == call.id)
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
            local_audio_available = AudioProcessor(settings).safe_storage_path(
                recordings[0].storage_key
            ).is_file()
        except Exception:
            local_audio_available = False
    return CallDetailResponse(
        id=call.id,
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
            len(recordings) == 1
            and (local_audio_available or settings.yeastar_configured)
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
        processing_history=[
            {
                "status": item.status.value,
                "stage": item.stage,
                "attempt": item.attempt_count,
                "updated_at": item.updated_at.isoformat(),
                "error": item.error_message,
                "message": item.stage if not item.error_message else f"{item.stage}: {item.error_message}",
                "created_at": item.created_at.isoformat(),
                "occurred_at": item.updated_at.isoformat(),
            }
            for item in history
        ],
    )


@router.post("/calls/{call_id}/retry", response_model=MessageResponse, status_code=status.HTTP_202_ACCEPTED)
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
        any(state == ItemStatus.FAILED for state in states)
        for states in statuses_by_call.values()
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recording is unavailable.")
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
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Recording is unavailable.")
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
