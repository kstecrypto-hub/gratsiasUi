from __future__ import annotations

import hashlib
import logging
import secrets
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from redis.exceptions import RedisError
from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.core.logging import redact_value
from app.core.redis import get_redis
from app.core.time import utc_now
from app.database.session import AsyncSessionFactory
from app.models import (
    Call,
    CallLeg,
    CallParticipant,
    AuditLog,
    IntegrationStatus,
    Keyword,
    KeywordMatch,
    Operator,
    ProcessingJob,
    ProcessingJobItem,
    Recording,
    SyncRun,
    Transcript,
    TranscriptSegment,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    RecordingStatus,
    RunStatus,
    SpeakerSource,
    SyncType,
    TranscriptStatus,
)
from app.services.application_settings import load_application_settings
from app.services.audio import AudioChunk, AudioProcessor
from app.services.audio.errors import AudioError
from app.services.keyword_matching import KeywordDefinition, match_text, normalize_greek
from app.services.transcription import OpenAITranscriptionClient
from app.services.transcription.client import (
    TranscriptionCancelledError,
    TranscriptionConfigurationError,
    TranscriptionError,
    TranscriptionResult,
)
from app.services.transcription.client import build_vocabulary_prompt
from app.services.transcription.configuration_store import (
    OpenAIConfigurationStateError,
    load_effective_openai_settings,
)
from app.services.yeastar import YeastarClient
from app.services.yeastar.configuration_store import load_effective_yeastar_settings
from app.services.yeastar.cdr import CDRSummary
from app.services.yeastar.errors import (
    YeastarApiDisabledError,
    YeastarAuthenticationError,
    YeastarCircuitOpenError,
    YeastarConfigurationError,
    YeastarConfigurationStateError,
    YeastarIpBlockedError,
    YeastarIpForbiddenError,
    YeastarPermissionError,
    YeastarRecordingDownloadLimitError,
    YeastarTokenRefreshError,
    YeastarUnsupportedVersionError,
    YeastarError,
)
from app.services.yeastar.interpretation import (
    InterpretedParticipant,
    interpret_call_legs,
    safe_operator_channel,
)
from app.services.yeastar.integration import (
    clear_shared_token_and_require_test,
    reconcile_configuration_fingerprint,
    runtime_cdr_api_version,
)


logger = logging.getLogger(__name__)
FINAL_ITEM_STATES = {ItemStatus.COMPLETED, ItemStatus.FAILED, ItemStatus.SKIPPED, ItemStatus.CANCELLED}
FINAL_JOB_STATES = {
    JobStatus.COMPLETED,
    JobStatus.COMPLETED_WITH_ERRORS,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}
CONNECTION_PAUSE_ERRORS = (
    YeastarConfigurationError,
    YeastarAuthenticationError,
    YeastarTokenRefreshError,
    YeastarCircuitOpenError,
    YeastarIpBlockedError,
    YeastarIpForbiddenError,
    YeastarApiDisabledError,
    YeastarPermissionError,
    YeastarUnsupportedVersionError,
)


class ProcessingBusyError(Exception):
    pass


@dataclass(frozen=True)
class DiscoveryResult:
    """Work that can be dispatched after one safe discovery pass.

    An item without a recording must never be handed to ``process_item``.
    When the phone system has multiple recordings but has not yet supplied a
    deterministic relation to a call leg, discovery asks the task queue to
    revisit the call rather than marking it as failed or guessing.
    """

    item_ids: list[UUID]
    recording_assignment_pending: bool = False


class ProcessingCancelledError(Exception):
    category = "cancelled"


class StaleProcessingTaskError(Exception):
    """A delayed task attempted to mutate a job that is already final."""


def _require_processable_job(job: ProcessingJob) -> None:
    """Make cancellation/finalization authoritative over delayed worker work."""
    if job.cancellation_requested or job.status == JobStatus.CANCELLED:
        raise ProcessingCancelledError("Analysis was cancelled.")
    if job.status in FINAL_JOB_STATES:
        raise StaleProcessingTaskError("Analysis is already finished.")


async def _stop_stale_discovery(session: AsyncSession, job_id: UUID) -> bool:
    """Settle cancelled discovery, or identify a task whose job is already final."""
    job = await session.scalar(
        select(ProcessingJob).where(ProcessingJob.id == job_id).with_for_update()
    )
    if job is None:
        return True
    cancelled = job.cancellation_requested or job.status == JobStatus.CANCELLED
    if job.status in FINAL_JOB_STATES and not cancelled:
        return True
    if not cancelled:
        return False
    if job.status not in FINAL_JOB_STATES:
        job.status = JobStatus.CANCELLED
        job.current_stage = "Cancelled"
        job.completed_at = utc_now()
    run = await session.scalar(
        select(SyncRun)
        .where(
            SyncRun.processing_job_id == job_id,
            SyncRun.status == RunStatus.RUNNING,
        )
        .order_by(SyncRun.created_at.desc())
    )
    if run is not None:
        run.status = RunStatus.CANCELLED
        run.completed_at = utc_now()
    await session.commit()
    return True


async def _settle_stale_item(
    session: AsyncSession,
    item: ProcessingJobItem,
    transcript_id: UUID | None,
) -> Literal["active", "cancelled", "terminal"]:
    """Keep late item exceptions from overwriting cancellation or final history."""
    job = await session.scalar(
        select(ProcessingJob).where(ProcessingJob.id == item.job_id).with_for_update()
    )
    if job is None:
        return "terminal"
    await session.refresh(item, with_for_update=True)
    cancelled = (
        job.cancellation_requested
        or job.status == JobStatus.CANCELLED
        or item.status == ItemStatus.CANCELLED
    )
    if job.status in FINAL_JOB_STATES and not cancelled:
        return "terminal"
    if item.status in FINAL_ITEM_STATES and not cancelled:
        return "terminal"
    if not cancelled:
        return "active"
    if item.status not in {ItemStatus.COMPLETED, ItemStatus.SKIPPED, ItemStatus.CANCELLED}:
        item.status = ItemStatus.CANCELLED
        item.stage = "cancelled"
        item.completed_at = utc_now()
        item.error_category = None
        item.error_message = None
    if transcript_id:
        transcript = await session.get(Transcript, transcript_id)
        if transcript and transcript.status != TranscriptStatus.COMPLETED:
            transcript.status = TranscriptStatus.FAILED
            transcript.error_category = "cancelled"
            transcript.error_message = "Transcription was cancelled."
    await session.commit()
    return "cancelled"


def transcript_idempotency_key(
    recording_id: UUID,
    operator_id: UUID | None,
    model: str,
    checksum: str,
    diarized: bool,
    language: str = "el",
    prompt_version: str | None = None,
) -> str:
    identity = (
        f"{recording_id}:{operator_id or 'unknown'}:{model}:{language}:"
        f"{prompt_version or 'no-prompt'}:{checksum}:{int(diarized)}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def parse_provider_datetime(value: object, settings: Settings) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, settings.YEASTAR_DATETIME_FORMAT)
        except ValueError:
            for fallback in ("%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fallback)
                    break
                except ValueError:
                    continue
            else:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(settings.APP_TIMEZONE))
    return parsed.astimezone(UTC)


def _group_legacy_cdr_summaries(
    summaries: list[CDRSummary], settings: Settings
) -> tuple[dict[str, dict[str, Any]], dict[str, list[CDRSummary]]]:
    """Group legacy CDR legs by UID and choose their earliest call summary.

    The appliance CDR v1 list can contain several rows for a single call UID.
    The runtime client keeps the complete set for its synthetic detail, while
    this mapping supplies one stable call-level summary to persistence.
    """

    grouped: dict[str, list[CDRSummary]] = defaultdict(list)
    for summary in summaries:
        uid = summary.uid.strip()
        if uid:
            grouped[uid].append(summary)

    def time_key(summary: CDRSummary) -> tuple[int, float]:
        parsed = parse_provider_datetime(summary.time, settings)
        return (0, parsed.timestamp()) if parsed is not None else (1, 0.0)

    representatives = {
        uid: dict(min(entries, key=time_key).provider_dict())
        for uid, entries in grouped.items()
    }
    return representatives, grouped


async def _fetch_cdrs(
    client: YeastarClient,
    date_from: datetime,
    date_to: datetime,
    filters: dict[str, Any],
    settings: Settings,
    *,
    cdr_api_version: Literal["v1", "v2"],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[CDRSummary]]]:
    """Fetch call summaries using the persisted, tested CDR API mode."""

    if cdr_api_version == "v1":
        # CDR v1 has no documented server-side date or filter arguments.  Its
        # adapter performs bounded pagination and local filtering safely.
        summaries = await client.search_all_cdrs(date_from, date_to, filters)
        return _group_legacy_cdr_summaries(summaries, settings)

    cdrs: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        response = await client.search_cdrs(date_from, date_to, filters, page)
        data = response["data"]
        for item in data:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("uid") or "")
            if uid:
                cdrs[uid] = item
        if not data or len(cdrs) >= response["total_number"]:
            return cdrs, {}
        page += 1
        if page > 10000:
            raise YeastarError("Phone system returned invalid call pagination.")


def _direction(value: object) -> Direction:
    normalized = str(value or "").strip().casefold()
    return {
        "inbound": Direction.INBOUND,
        "outbound": Direction.OUTBOUND,
        "internal": Direction.INTERNAL,
    }.get(normalized, Direction.UNKNOWN)


def _queue_name(cdr: dict[str, Any]) -> str | None:
    queues = cdr.get("queues") or []
    if isinstance(queues, list):
        names = [str(item.get("name") or item.get("number")) for item in queues if isinstance(item, dict)]
        return ", ".join(dict.fromkeys(name for name in names if name))[:255] or None
    return None


async def _job_stage(
    session: AsyncSession, job: ProcessingJob, status: JobStatus, human_stage: str
) -> None:
    await session.refresh(job, with_for_update=True)
    _require_processable_job(job)
    job.status = status
    job.current_stage = human_stage
    if job.started_at is None:
        job.started_at = utc_now()
    await session.commit()


async def _upsert_call(session: AsyncSession, cdr: dict[str, Any], settings: Settings) -> Call:
    uid = str(cdr.get("uid") or "").strip()
    if not uid:
        raise ValueError("Call is missing its provider UID")
    call = await session.scalar(select(Call).where(Call.yeastar_uid == uid))
    started_at = parse_provider_datetime(cdr.get("time"), settings) or utc_now()
    if call is None:
        call = Call(yeastar_uid=uid, started_at=started_at)
        session.add(call)
    call.yeastar_id = str(cdr.get("id") or "")[:255] or None
    call.started_at = started_at
    call.caller_number = str(cdr.get("call_from_number") or "")[:128] or None
    call.caller_name = str(cdr.get("call_from_name") or "")[:255] or None
    call.callee_number = str(cdr.get("call_to_number") or "")[:128] or None
    call.callee_name = str(cdr.get("call_to_name") or "")[:255] or None
    call.direction = _direction(cdr.get("call_type"))
    call.call_status = str(cdr.get("last_status") or "")[:64] or None
    call.duration_seconds = max(0, int(cdr.get("call_duration") or 0))
    call.queue_name = _queue_name(cdr)
    segments = cdr.get("segments")
    if isinstance(segments, (list, tuple, dict)):
        segment_count = len(segments)
    else:
        try:
            segment_count = int(segments or 1)
        except (TypeError, ValueError):
            segment_count = 1
    call.was_transferred = bool(segment_count > 1 or cdr.get("second_participant"))
    call.provider_payload = redact_value(cdr)
    call.processing_status = "finding_recordings"
    await session.flush()
    return call


async def _upsert_details(
    session: AsyncSession,
    call: Call,
    detail: dict[str, Any],
    operators: list[Operator],
    settings: Settings,
) -> list[CallParticipant]:
    operator_dicts = [
        {
            "id": operator.id,
            "yeastar_extension_id": operator.yeastar_extension_id,
            "extension_number": operator.extension_number,
        }
        for operator in operators
    ]
    interpreted_legs, interpreted_participants = interpret_call_legs(detail, operator_dicts)
    if len(interpreted_legs) > 1:
        # Conservative for channel safety: multi-leg calls are never treated as one-to-one stereo.
        call.was_transferred = True
    existing_legs = {
        leg.yeastar_leg_id: leg
        for leg in (await session.scalars(select(CallLeg).where(CallLeg.call_id == call.id))).all()
    }
    legs_by_provider_id: dict[str, CallLeg] = {}
    for raw in interpreted_legs:
        leg = existing_legs.get(raw["yeastar_leg_id"])
        if leg is None:
            leg = CallLeg(
                call_id=call.id,
                yeastar_leg_id=raw["yeastar_leg_id"],
                sequence_number=raw["sequence_number"],
            )
            session.add(leg)
        for key in (
            "sequence_number",
            "transaction_id",
            "yeastar_cdr_id",
            "provider_leg",
            "caller_extension_id",
            "callee_extension_id",
            "call_from",
            "call_to",
            "caller_number",
            "callee_number",
            "answered_by_extension_id",
            "duration_seconds",
            "ring_duration_seconds",
            "talk_duration_seconds",
            "hold_duration_seconds",
            "call_type",
            "status",
            "event_list",
            "provider_recording_id",
            "provider_payload",
        ):
            setattr(leg, key, raw.get(key))
        leg.started_at = parse_provider_datetime(raw.get("provider_start_time"), settings)
        leg.answered_at = parse_provider_datetime(raw.get("provider_answer_time"), settings)
        leg.ended_at = parse_provider_datetime(raw.get("provider_end_time"), settings)
        await session.flush()
        legs_by_provider_id[leg.yeastar_leg_id] = leg
    existing_participants = (
        await session.scalars(select(CallParticipant).where(CallParticipant.call_id == call.id))
    ).all()
    by_key = {
        (str(item.operator_id), str(item.call_leg_id), item.role.value): item
        for item in existing_participants
    }
    operator_by_id = {str(operator.id): operator for operator in operators}
    saved: list[CallParticipant] = []
    for item in interpreted_participants:
        operator = operator_by_id.get(item.operator_id)
        leg = legs_by_provider_id.get(item.leg_id)
        if operator is None or leg is None:
            continue
        key = (str(operator.id), str(leg.id), item.role.value)
        participant = by_key.get(key)
        if participant is None:
            participant = CallParticipant(
                call_id=call.id,
                operator_id=operator.id,
                call_leg_id=leg.id,
                role=item.role,
            )
            session.add(participant)
        participant.provider_extension_id = item.provider_extension_id
        participant.provider_extension_number = item.extension_number
        participant.operator_was_caller = item.was_caller
        participant.operator_was_callee = item.was_callee
        participant.answered = item.answered
        participant.participated_from = leg.started_at
        participant.participated_until = leg.ended_at
        participant.attribution_source = SpeakerSource.YEASTAR_EXTENSION
        participant.attribution_confidence = Decimal("1.0000")
        saved.append(participant)
    await session.flush()
    return saved


async def _upsert_recordings(
    session: AsyncSession,
    call: Call,
    provider_recordings: list[dict[str, Any]],
    settings: Settings,
) -> list[Recording]:
    existing = {
        item.yeastar_recording_id: item
        for item in (await session.scalars(select(Recording).where(Recording.call_id == call.id))).all()
    }
    saved: list[Recording] = []
    for raw in provider_recordings:
        provider_id = str(raw.get("id") or "").strip()
        if not provider_id:
            continue
        recording = existing.get(provider_id)
        if recording is None:
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=provider_id,
                status=RecordingStatus.DISCOVERED,
            )
            session.add(recording)
        recording.yeastar_file_name = str(raw.get("file") or "")[:512] or None
        recording.yeastar_uid = str(raw.get("uid") or "")[:255] or None
        recording.provider_time = parse_provider_datetime(raw.get("time"), settings)
        recording.call_from = str(raw.get("call_from") or "")[:255] or None
        recording.call_to = str(raw.get("call_to") or "")[:255] or None
        recording.call_from_number = str(raw.get("call_from_number") or "")[:128] or None
        recording.call_to_number = str(raw.get("call_to_number") or "")[:128] or None
        recording.call_type = str(raw.get("call_type") or "")[:64] or None
        recording.archive_status = str(raw.get("archive_status") or "")[:64] or None
        if raw.get("duration") is not None:
            recording.duration_seconds = max(0, float(raw["duration"]))
        if raw.get("size") is not None:
            recording.size_bytes = max(0, int(raw["size"]))
        recording.provider_payload = redact_value(raw)
        saved.append(recording)
    call.has_recording = bool(saved)
    if saved:
        call.processing_status = "queued"
    await session.flush()
    return saved


def _normalise_recording_filename(value: object) -> str | None:
    """Return a provider filename suitable only for an equality comparison."""
    if not isinstance(value, str):
        return None
    filename = value.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    return filename or None


def _normalise_phone_number(value: object) -> str | None:
    """Normalise a phone endpoint without attempting country-specific guesses."""
    if value is None:
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits or None


def _recordings_for_participant(
    recordings: list[Recording], participant: CallParticipant, leg: CallLeg | None
) -> tuple[list[Recording], str | None]:
    """Find every recording with a deterministic relation to a call leg.

    A transferred call can legitimately have more than one recording.  Picking
    the first one would silently attach the wrong conversation to an operator,
    so each fallback below is deliberately exact.  A single leg can safely
    own several recordings when every one has the exact same endpoint pair.
    """
    if len(recordings) == 1:
        return recordings, None
    if not recordings:
        return [], "no_recording"
    explicit_matches: list[Recording] = []
    filename_matches: list[Recording] = []
    endpoint_matches: list[Recording] = []
    if leg and leg.provider_recording_id:
        matched = [
            item for item in recordings if item.yeastar_recording_id == leg.provider_recording_id
        ]
        if len(matched) == 1:
            explicit_matches = matched

    if leg is not None:
        # Legacy CDR 1.0 exposes the recording filename on the CDR row rather
        # than a recording ID.  The recording search result exposes the same
        # filename, which gives us a safe, deterministic correlation.
        payload = leg.provider_payload if isinstance(leg.provider_payload, dict) else {}
        filenames = {
            filename
            for field in ("record_file", "recording_file", "file")
            if (filename := _normalise_recording_filename(payload.get(field))) is not None
        }
        if filenames:
            filename_matches = [
                item
                for item in recordings
                if _normalise_recording_filename(item.yeastar_file_name) in filenames
            ]

        # Some firmware versions omit both IDs and filenames from the CDR
        # detail.  An ordered, two-endpoint match is still deterministic; a
        # one-sided extension match is not and is intentionally never used.
        caller = _normalise_phone_number(leg.caller_number)
        callee = _normalise_phone_number(leg.callee_number)
        if caller is not None and callee is not None:
            endpoint_matches = [
                item
                for item in recordings
                if _normalise_phone_number(item.call_from_number) == caller
                and _normalise_phone_number(item.call_to_number) == callee
            ]

    # Provider IDs and legacy filenames identify one recording directly.  A
    # matching endpoint pair may identify additional split recordings for the
    # same leg, so retain all of those safely attributable files.  If the only
    # direct signals conflict, leave the call queued rather than inventing a
    # relation between them.
    direct_matches = explicit_matches
    if len(filename_matches) == 1:
        if (
            direct_matches
            and filename_matches[0].yeastar_recording_id
            != direct_matches[0].yeastar_recording_id
        ):
            corroborated_ids = {item.yeastar_recording_id for item in endpoint_matches}
            direct_ids = {
                direct_matches[0].yeastar_recording_id,
                filename_matches[0].yeastar_recording_id,
            }
            if not direct_ids.issubset(corroborated_ids):
                return [], "recording_assignment_pending"
        direct_matches = [*direct_matches, *filename_matches]
    matched_provider_ids = {
        item.yeastar_recording_id for item in [*direct_matches, *endpoint_matches]
    }
    if matched_provider_ids:
        return [
            item for item in recordings if item.yeastar_recording_id in matched_provider_ids
        ], None

    # The recording list and CDR details can arrive at different times on the
    # PBX.  Keep the item queued and let the analysis task revisit discovery
    # instead of reporting a user-visible failure or guessing a recording.
    return [], "recording_assignment_pending"


def _recording_for_participant(
    recordings: list[Recording], participant: CallParticipant, leg: CallLeg | None
) -> tuple[Recording | None, str | None]:
    """Compatibility wrapper for callers that need exactly one recording."""
    matched, reason = _recordings_for_participant(recordings, participant, leg)
    return (
        (matched[0], reason)
        if len(matched) == 1
        else (None, reason or "recording_assignment_pending")
    )


def _processing_item_key(
    job_id: UUID,
    call_id: UUID,
    operator_id: UUID,
    recording_id: UUID | None,
    *,
    state: str | None = None,
) -> str:
    """Build stable item keys for both recording work and unassigned states."""
    identity = str(recording_id) if recording_id is not None else state or "unassigned"
    return hashlib.sha256(f"{job_id}:{call_id}:{operator_id}:{identity}".encode()).hexdigest()


def _legacy_processing_item_key(job_id: UUID, call_id: UUID, operator_id: UUID) -> str:
    """Return the pre-recording-aware key used by already-created jobs."""
    return hashlib.sha256(f"{job_id}:{call_id}:{operator_id}".encode()).hexdigest()


async def discover_job_items(job_id: UUID) -> DiscoveryResult:
    lock = get_redis().lock(
        f"yca:job-discovery:{job_id}", timeout=30 * 60 * 60, blocking_timeout=1
    )
    if not await lock.acquire(blocking=False):
        return DiscoveryResult([])
    try:
        return await _discover_job_items_locked(job_id)
    finally:
        try:
            if await lock.owned():
                await lock.release()
        except Exception:
            pass


async def _discover_job_items_locked(job_id: UUID) -> DiscoveryResult:
    redis = get_redis()
    try:
        settings = await load_effective_yeastar_settings(redis, get_settings())
    except (RedisError, YeastarConfigurationError) as exc:
        async with AsyncSessionFactory() as session:
            if await _stop_stale_discovery(session, job_id):
                return DiscoveryResult([])
            job = await session.get(ProcessingJob, job_id)
            if job is not None:
                job.status = JobStatus.WAITING_FOR_CONNECTION
                job.current_stage = "Waiting for phone-system connection"
                job.last_error_category = getattr(
                    exc,
                    "category",
                    "configuration_unavailable",
                )
                job.last_error_message = "Test the phone-system connection in Settings."
                job.completed_at = None
                await session.commit()
        return DiscoveryResult([])
    async with AsyncSessionFactory() as session:
        job = await session.get(ProcessingJob, job_id)
        if job is None or job.status in FINAL_JOB_STATES:
            return DiscoveryResult([])
        if job.cancellation_requested:
            await _stop_stale_discovery(session, job_id)
            return DiscoveryResult([])
        _, configured, changed_from_existing = await reconcile_configuration_fingerprint(
            session,
            settings,
            clear_token_state=lambda: clear_shared_token_and_require_test(redis),
        )
        if not configured or changed_from_existing:
            await session.refresh(job, with_for_update=True)
            if job.cancellation_requested or job.status in FINAL_JOB_STATES:
                await _stop_stale_discovery(session, job_id)
                return DiscoveryResult([])
            job.status = JobStatus.WAITING_FOR_CONNECTION
            job.current_stage = "Waiting for phone-system connection"
            job.last_error_category = (
                "configuration_changed" if changed_from_existing else "not_configured"
            )
            job.last_error_message = "Test the phone-system connection in Settings."
            job.completed_at = None
            await session.commit()
            return DiscoveryResult([])
        run_key = f"job:{job.id}:attempt:{job.attempt_count}"
        existing_run = await session.scalar(
            select(SyncRun).where(SyncRun.idempotency_key == run_key)
        )
        if existing_run is not None:
            if existing_run.status == RunStatus.RUNNING:
                return DiscoveryResult([])
            unassigned = (
                await session.scalars(
                    select(ProcessingJobItem.id).where(
                        ProcessingJobItem.job_id == job.id,
                        ProcessingJobItem.recording_id.is_(None),
                        ProcessingJobItem.status.in_(
                            [
                                ItemStatus.QUEUED,
                                ItemStatus.WAITING_FOR_CONNECTION,
                                ItemStatus.PROCESSING,
                                ItemStatus.FAILED,
                            ]
                        ),
                    )
                )
            ).all()
            if unassigned:
                # Older releases could leave one queued item with no recording
                # ID.  Re-run discovery so it can be converted to either a
                # recording-backed item or the new safe pending placeholder.
                await session.delete(existing_run)
                await session.flush()
            else:
                pending = (
                    await session.scalars(
                        select(ProcessingJobItem.id).where(
                            ProcessingJobItem.job_id == job.id,
                            ProcessingJobItem.status == ItemStatus.QUEUED,
                            ProcessingJobItem.recording_id.is_not(None),
                        )
                    )
                ).all()
                return DiscoveryResult(list(pending))
        run = SyncRun(
            sync_type=SyncType.CALLS,
            status=RunStatus.RUNNING,
            idempotency_key=run_key,
            processing_job_id=job.id,
            started_at=utc_now(),
            date_from=job.date_from,
            date_to=job.date_to,
        )
        session.add(run)
        await _job_stage(session, job, JobStatus.CONNECTING, "Connecting to the phone system")
        try:
            integration = await session.scalar(
                select(IntegrationStatus).where(IntegrationStatus.provider == "yeastar")
            )
            cdr_api_version = runtime_cdr_api_version(
                integration.capabilities_json if integration is not None else None
            )
            async with YeastarClient(
                settings=settings,
                redis=redis,
                system_date_format=(integration.system_date_format if integration else None),
                system_time_format=(integration.system_time_format if integration else None),
                cdr_api_version=cdr_api_version,
            ) as client:
                await client.authenticate()
                await _job_stage(session, job, JobStatus.FETCHING_CALLS, "Finding calls")
                filters: dict[str, Any] = {}
                if job.direction:
                    filters["call_type"] = job.direction.title()
                if job.queue_name:
                    filters["queue"] = job.queue_name
                if job.call_status_filter:
                    filters["status"] = job.call_status_filter
                if job.recording_available is not None:
                    filters["recording_type"] = 1 if job.recording_available else 2
                cdrs, legacy_cdr_summaries = await _fetch_cdrs(
                    client,
                    job.date_from,
                    job.date_to,
                    filters,
                    settings,
                    cdr_api_version=cdr_api_version,
                )
                run.records_seen = len(cdrs)
                await _job_stage(session, job, JobStatus.FINDING_RECORDINGS, "Finding recordings")
                provider_recordings = await client.search_recordings(job.date_from, job.date_to)
                recordings_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for raw in provider_recordings:
                    recordings_by_uid[str(raw.get("uid") or "")].append(raw)
                operators = (
                    await session.scalars(
                        select(Operator).where(Operator.deleted_at.is_(None))
                    )
                ).all()
                selected_ids = {UUID(value) for value in job.selected_operator_ids}
                item_ids: list[UUID] = []
                relevant_calls: set[UUID] = set()
                relevant_recordings: set[UUID] = set()
                discovery_failures = 0
                assignment_pending_calls: set[UUID] = set()
                await _job_stage(
                    session, job, JobStatus.FETCHING_CALL_DETAILS, "Finding call participants"
                )
                for cdr in cdrs.values():
                    await session.refresh(job)
                    _require_processable_job(job)
                    call = await _upsert_call(session, cdr, settings)
                    try:
                        if cdr_api_version == "v1":
                            detail = await client.get_cdr_detail(
                                call.yeastar_uid,
                                summaries=legacy_cdr_summaries.get(call.yeastar_uid),
                            )
                        else:
                            detail = await client.get_cdr_detail(call.yeastar_uid)
                        participants = await _upsert_details(session, call, detail, operators, settings)
                        recordings = await _upsert_recordings(
                            session,
                            call,
                            recordings_by_uid.get(call.yeastar_uid, []),
                            settings,
                        )
                    except CONNECTION_PAUSE_ERRORS:
                        raise
                    except Exception as exc:
                        await session.refresh(job, with_for_update=True)
                        _require_processable_job(job)
                        call.processing_status = "failed"
                        call.last_error_category = getattr(exc, "category", "call_detail")
                        call.last_error_message = "Call details could not be processed."
                        discovery_failures += 1
                        await session.commit()
                        continue
                    selected_participants: dict[UUID, list[CallParticipant]] = defaultdict(list)
                    for participant in participants:
                        if participant.operator_id in selected_ids:
                            selected_participants[participant.operator_id].append(participant)
                    if not selected_participants:
                        await session.refresh(job, with_for_update=True)
                        _require_processable_job(job)
                        await session.commit()
                        continue
                    relevant_calls.add(call.id)
                    relevant_recordings.update(recording.id for recording in recordings)
                    legs_by_id = {
                        leg.id: leg
                        for leg in (await session.scalars(select(CallLeg).where(CallLeg.call_id == call.id))).all()
                    }
                    assigned_by_operator: dict[UUID, dict[UUID, Recording]] = defaultdict(dict)
                    pending_operator_ids: set[UUID] = set()
                    for operator_id, operator_participants in selected_participants.items():
                        for participant in operator_participants:
                            matched, assignment_error = _recordings_for_participant(
                                recordings,
                                participant,
                                legs_by_id.get(participant.call_leg_id),
                            )
                            for recording in matched:
                                assigned_by_operator[operator_id][recording.id] = recording
                            if assignment_error == "recording_assignment_pending":
                                pending_operator_ids.add(operator_id)

                    legacy_unassigned_by_operator: dict[UUID, ProcessingJobItem | None] = {}
                    for operator_id in selected_participants:
                        legacy_key = _legacy_processing_item_key(job.id, call.id, operator_id)
                        legacy_unassigned_by_operator[operator_id] = await session.scalar(
                            select(ProcessingJobItem).where(
                                ProcessingJobItem.idempotency_key == legacy_key,
                                ProcessingJobItem.recording_id.is_(None),
                            )
                        )

                    queued_item_ids: set[UUID] = set()
                    for operator_id, assigned_recordings in assigned_by_operator.items():
                        for recording in assigned_recordings.values():
                            key = _processing_item_key(
                                job.id,
                                call.id,
                                operator_id,
                                recording.id,
                            )
                            item = await session.scalar(
                                select(ProcessingJobItem).where(
                                    ProcessingJobItem.job_id == job.id,
                                    ProcessingJobItem.call_id == call.id,
                                    ProcessingJobItem.operator_id == operator_id,
                                    ProcessingJobItem.recording_id == recording.id,
                                )
                            )
                            if item is None:
                                item = await session.scalar(
                                    select(ProcessingJobItem).where(
                                        ProcessingJobItem.idempotency_key == key
                                    )
                                )
                            converted_legacy_item = False
                            legacy_item = legacy_unassigned_by_operator.get(operator_id)
                            if item is None and legacy_item is not None:
                                item = legacy_item
                                legacy_unassigned_by_operator[operator_id] = None
                                item.recording_id = recording.id
                                item.idempotency_key = key
                                converted_legacy_item = True
                            if item is None:
                                item = ProcessingJobItem(
                                    job_id=job.id,
                                    call_id=call.id,
                                    operator_id=operator_id,
                                    recording_id=recording.id,
                                    idempotency_key=key,
                                    status=ItemStatus.QUEUED,
                                    stage="queued",
                                )
                                session.add(item)
                                await session.flush()
                            if converted_legacy_item:
                                item.status = ItemStatus.QUEUED
                            if item.status == ItemStatus.QUEUED and item.id not in queued_item_ids:
                                item.stage = "queued"
                                item.error_category = None
                                item.error_message = None
                                item.completed_at = None
                                item.locked_at = None
                                item.heartbeat_at = None
                                item_ids.append(item.id)
                                queued_item_ids.add(item.id)

                    for operator_id in selected_participants:
                        pending_key = _processing_item_key(
                            job.id,
                            call.id,
                            operator_id,
                            None,
                            state="recording-assignment-pending",
                        )
                        pending_item = await session.scalar(
                            select(ProcessingJobItem).where(
                                ProcessingJobItem.idempotency_key == pending_key
                            )
                        )
                        legacy_item = legacy_unassigned_by_operator.get(operator_id)
                        if operator_id in pending_operator_ids:
                            if pending_item is None and legacy_item is not None:
                                pending_item = legacy_item
                                legacy_unassigned_by_operator[operator_id] = None
                                pending_item.idempotency_key = pending_key
                            if pending_item is None:
                                pending_item = ProcessingJobItem(
                                    job_id=job.id,
                                    call_id=call.id,
                                    operator_id=operator_id,
                                    recording_id=None,
                                    idempotency_key=pending_key,
                                )
                                session.add(pending_item)
                            pending_item.status = ItemStatus.QUEUED
                            pending_item.stage = "waiting_for_recording_assignment"
                            pending_item.error_category = None
                            pending_item.error_message = None
                            pending_item.completed_at = None
                            pending_item.locked_at = None
                            pending_item.heartbeat_at = None
                            assignment_pending_calls.add(call.id)
                            call.processing_status = "waiting_for_recording_assignment"
                            call.last_error_category = None
                            call.last_error_message = None
                        elif pending_item is not None:
                            await session.delete(pending_item)

                        legacy_item = legacy_unassigned_by_operator.get(operator_id)
                        if legacy_item is not None:
                            await session.delete(legacy_item)
                            legacy_unassigned_by_operator[operator_id] = None

                        if not assigned_by_operator.get(operator_id) and operator_id not in pending_operator_ids:
                            skipped_key = _processing_item_key(
                                job.id,
                                call.id,
                                operator_id,
                                None,
                                state="no-recording",
                            )
                            skipped_item = await session.scalar(
                                select(ProcessingJobItem).where(
                                    ProcessingJobItem.idempotency_key == skipped_key
                                )
                            )
                            if skipped_item is None:
                                session.add(
                                    ProcessingJobItem(
                                        job_id=job.id,
                                        call_id=call.id,
                                        operator_id=operator_id,
                                        recording_id=None,
                                        idempotency_key=skipped_key,
                                        status=ItemStatus.SKIPPED,
                                        stage="no_recording",
                                        completed_at=utc_now(),
                                    )
                                )
                    await session.refresh(job, with_for_update=True)
                    _require_processable_job(job)
                    await session.commit()
                job = await session.get(ProcessingJob, job.id)
                assert job is not None
                await session.refresh(job, with_for_update=True)
                _require_processable_job(job)
                job.calls_found = len(relevant_calls)
                job.recordings_found = len(relevant_recordings)
                item_status_rows = (
                    await session.execute(
                        select(ProcessingJobItem.call_id, ProcessingJobItem.status).where(
                            ProcessingJobItem.job_id == job.id
                        )
                    )
                ).all()
                # A re-discovery can resolve a later recording while work for
                # an earlier recording has already failed.  Preserve that
                # outcome instead of overwriting it with a false success when
                # there is nothing new to dispatch.
                failed_item_calls = {
                    call_id
                    for call_id, item_status in item_status_rows
                    if item_status == ItemStatus.FAILED
                }
                has_active_items = any(
                    item_status not in FINAL_ITEM_STATES
                    for _, item_status in item_status_rows
                )
                total_discovery_errors = discovery_failures + len(failed_item_calls)
                job.calls_failed = total_discovery_errors
                job.request_filters = {
                    **(job.request_filters or {}),
                    "_discovery_failures": discovery_failures,
                    "_recording_assignment_pending": len(assignment_pending_calls),
                }
                if assignment_pending_calls:
                    # Preserve a queued, unassigned item and let the worker
                    # retry full discovery.  There is deliberately no failed
                    # item here: the PBX has not supplied a safe correlation.
                    job.status = JobStatus.FINDING_RECORDINGS
                    job.current_stage = "Waiting for recording assignment"
                    job.completed_at = None
                    job.last_error_category = None
                    job.last_error_message = None
                    await session.delete(run)
                    await session.commit()
                    return DiscoveryResult(item_ids, recording_assignment_pending=True)
                has_work_to_finish = bool(item_ids) or has_active_items
                job.status = JobStatus.DOWNLOADING_RECORDINGS if has_work_to_finish else (
                    JobStatus.COMPLETED_WITH_ERRORS if total_discovery_errors else JobStatus.COMPLETED
                )
                job.current_stage = "Preparing recordings" if has_work_to_finish else "Complete"
                if not has_work_to_finish:
                    job.progress_percent = 100
                    job.completed_at = utc_now()
                run.status = RunStatus.COMPLETED_WITH_ERRORS if total_discovery_errors else RunStatus.COMPLETED
                run.records_created = len(relevant_calls)
                run.records_failed = total_discovery_errors
                run.completed_at = utc_now()
                await session.commit()
                return DiscoveryResult(item_ids)
        except ProcessingCancelledError:
            await session.rollback()
            await _stop_stale_discovery(session, job_id)
            return DiscoveryResult([])
        except StaleProcessingTaskError:
            await session.rollback()
            await _stop_stale_discovery(session, job_id)
            return DiscoveryResult([])
        except CONNECTION_PAUSE_ERRORS as exc:
            await session.rollback()
            if await _stop_stale_discovery(session, job_id):
                return DiscoveryResult([])
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.WAITING_FOR_CONNECTION
                job.current_stage = "Waiting for phone-system connection"
                job.last_error_category = getattr(exc, "category", "connection_paused")
                job.last_error_message = "Test the phone-system connection in Settings."
                job.completed_at = None
            current_run = await session.scalar(
                select(SyncRun)
                .where(SyncRun.processing_job_id == job_id)
                .order_by(SyncRun.created_at.desc())
            )
            if current_run is not None:
                await session.delete(current_run)
            await session.commit()
            return DiscoveryResult([])
        except Exception as exc:
            await session.rollback()
            if await _stop_stale_discovery(session, job_id):
                return DiscoveryResult([])
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.current_stage = "Could not complete analysis"
                job.last_error_category = getattr(exc, "category", "unexpected")
                job.last_error_message = _safe_error_message(exc)
                job.completed_at = utc_now()
            run = await session.scalar(
                select(SyncRun).where(SyncRun.processing_job_id == job_id).order_by(SyncRun.created_at.desc())
            )
            if run:
                run.status = RunStatus.FAILED
                run.error_category = getattr(exc, "category", "unexpected")
                run.error_message = _safe_error_message(exc)
                run.completed_at = utc_now()
            await session.commit()
            raise


def _safe_error_message(exc: Exception) -> str:
    if isinstance(exc, (YeastarError, AudioError, TranscriptionError, ProcessingCancelledError)):
        return str(exc)[:1000]
    return "An unexpected processing error occurred."


async def _cancel_requested(session: AsyncSession, job_id: UUID) -> bool:
    value = await session.scalar(
        select(ProcessingJob.cancellation_requested).where(ProcessingJob.id == job_id)
    )
    return bool(value)


async def _lock_processable_item(
    session: AsyncSession,
    item: ProcessingJobItem,
    job: ProcessingJob,
) -> None:
    """Lock job then item and reject a stale or cancelled delivery."""
    await session.refresh(job, with_for_update=True)
    _require_processable_job(job)
    await session.refresh(item, with_for_update=True)
    if item.status == ItemStatus.CANCELLED:
        raise ProcessingCancelledError("Analysis was cancelled.")
    if item.status in FINAL_ITEM_STATES:
        raise StaleProcessingTaskError("Analysis item is already finished.")


async def _item_stage(
    session: AsyncSession,
    item: ProcessingJobItem,
    job: ProcessingJob,
    status: JobStatus,
    stage: str,
) -> None:
    await _lock_processable_item(session, item, job)
    item.stage = stage
    item.heartbeat_at = utc_now()
    job.status = status
    job.current_stage = stage
    await session.commit()


async def _clone_transcript(
    session: AsyncSession,
    source: Transcript,
    *,
    call: Call,
    recording: Recording,
    operator: Operator | None,
    call_leg_id: UUID | None,
    key: str,
) -> Transcript:
    clone = Transcript(
        call_id=call.id,
        recording_id=recording.id,
        operator_id=operator.id if operator else None,
        idempotency_key=key,
        status=TranscriptStatus.COMPLETED,
        model=source.model,
        language=source.language,
        prompt_version=source.prompt_version,
        processing_duration_seconds=0,
        audio_duration_seconds=recording.duration_seconds,
        api_usage={"audio_reused_by_checksum": True},
        attempt_count=0,
        completed_at=utc_now(),
        source_audio_sha256=source.source_audio_sha256,
        is_diarized=source.is_diarized,
        original_text=source.original_text,
        normalized_text=source.normalized_text,
    )
    session.add(clone)
    await session.flush()
    source_segments = (
        await session.scalars(
            select(TranscriptSegment)
            .where(TranscriptSegment.transcript_id == source.id)
            .order_by(TranscriptSegment.sequence_number)
        )
    ).all()
    for segment in source_segments:
        session.add(
            TranscriptSegment(
                transcript_id=clone.id,
                call_id=call.id,
                call_leg_id=call_leg_id if operator else None,
                operator_id=operator.id if operator else None,
                speaker_label=operator.display_name if operator else segment.speaker_label,
                speaker_source=segment.speaker_source,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                original_text=segment.original_text,
                normalized_text=segment.normalized_text,
                confidence=segment.confidence,
                transcription_model=segment.transcription_model,
                sequence_number=segment.sequence_number,
            )
        )
    await session.flush()
    return clone


async def _keyword_definitions(
    session: AsyncSession, job: ProcessingJob
) -> tuple[list[KeywordDefinition], dict[str, Keyword]]:
    conditions = [Keyword.active.is_(True), Keyword.deleted_at.is_(None)]
    if job.selected_category_ids:
        conditions.append(Keyword.category_id.in_([UUID(item) for item in job.selected_category_ids]))
    keywords = (
        await session.scalars(
            select(Keyword)
            .options(selectinload(Keyword.variants), selectinload(Keyword.category))
            .where(*conditions)
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


async def search_and_persist_matches(
    session: AsyncSession, transcript: Transcript, job: ProcessingJob
) -> int:
    definitions, keyword_by_id = await _keyword_definitions(session, job)
    if not definitions:
        return 0
    segments = (
        await session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript.id)
        )
    ).all()
    segment_ids = [item.id for item in segments]
    existing = set()
    if segment_ids:
        rows = (
            await session.execute(
                select(
                    KeywordMatch.keyword_id,
                    KeywordMatch.transcript_segment_id,
                    KeywordMatch.start_seconds,
                    KeywordMatch.normalized_match,
                ).where(KeywordMatch.transcript_segment_id.in_(segment_ids))
            )
        ).all()
        existing = {(str(a), str(b), c, d) for a, b, c, d in rows}
    added = 0
    for segment in segments:
        # Diarized labels remain unknown. They are intentionally excluded unless all speakers was explicit.
        if not job.include_all_speakers and segment.operator_id is None:
            continue
        if not job.include_all_speakers and segment.operator_id != transcript.operator_id:
            continue
        for found in match_text(segment.original_text, definitions):
            keyword = keyword_by_id[found.keyword_id]
            key = (
                found.keyword_id,
                str(segment.id),
                segment.start_seconds,
                found.normalized_match,
            )
            if key in existing:
                continue
            session.add(
                KeywordMatch(
                    keyword_id=keyword.id,
                    operator_id=segment.operator_id,
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
            existing.add(key)
            added += 1
    await session.flush()
    return added


async def _vocabulary(session: AsyncSession, operator: Operator, settings: Settings) -> list[str]:
    values = [operator.display_name]
    application = await load_application_settings(session, settings)
    company = str(application.get("company_vocabulary") or "")
    values.extend(item.strip() for item in company.replace("\n", ",").split(",") if item.strip())
    keywords = (
        await session.scalars(
            select(Keyword)
            .options(selectinload(Keyword.variants))
            .where(Keyword.active.is_(True))
            .order_by(Keyword.canonical_phrase, Keyword.id)
        )
    ).all()
    for keyword in keywords:
        values.append(keyword.canonical_phrase)
        values.extend(item.phrase for item in keyword.variants)
    operator_names = (
        await session.scalars(select(Operator.display_name).order_by(Operator.display_name, Operator.id))
    ).all()
    values.extend(operator_names)
    return values


async def process_item(item_id: UUID) -> None:
    redis = get_redis()
    settings = get_settings()
    lock = None
    transcription_slot = None
    temporary_files: list[Path] = []
    transcript_id: UUID | None = None
    try:
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item is None or item.status not in {
                ItemStatus.QUEUED,
                ItemStatus.WAITING_FOR_CONNECTION,
                ItemStatus.PROCESSING,
            }:
                return
            job = await session.scalar(
                select(ProcessingJob)
                .where(ProcessingJob.id == item.job_id)
                .with_for_update()
            )
            if job is None:
                return
            _require_processable_job(job)
            item = await session.scalar(
                select(ProcessingJobItem)
                .where(ProcessingJobItem.id == item_id)
                .with_for_update(skip_locked=True)
            )
            if item is None or item.status not in {
                ItemStatus.QUEUED,
                ItemStatus.WAITING_FOR_CONNECTION,
                ItemStatus.PROCESSING,
            }:
                return
            if item.status == ItemStatus.PROCESSING and item.heartbeat_at and item.heartbeat_at > utc_now() - timedelta(hours=2):
                return
            item.status = ItemStatus.PROCESSING
            item.attempt_count += 1
            item.locked_at = utc_now()
            item.heartbeat_at = utc_now()
            await session.commit()
            call = await session.get(Call, item.call_id)
            operator = await session.get(Operator, item.operator_id)
            recording = await session.get(Recording, item.recording_id) if item.recording_id else None
            if not call or not operator:
                raise RuntimeError("Processing item references missing data")
            if recording is None:
                # A task from an older release can still arrive after its
                # one-per-call placeholder has been migrated.  It must not
                # become a failed analysis item simply because no recording
                # has been assigned yet; full discovery owns that transition.
                await _lock_processable_item(session, item, job)
                item.status = ItemStatus.QUEUED
                item.stage = "waiting_for_recording_assignment"
                item.error_category = None
                item.error_message = None
                item.completed_at = None
                await session.commit()
                return
            lock = redis.lock(
                f"yca:recording-processing:{recording.id}",
                timeout=30 * 60 * 60,
                blocking_timeout=1,
            )
            if not await lock.acquire(blocking=False):
                item.status = ItemStatus.QUEUED
                await session.commit()
                raise ProcessingBusyError()
            completed_transcript = await session.scalar(
                select(Transcript)
                .where(
                    Transcript.recording_id == recording.id,
                    Transcript.status == TranscriptStatus.COMPLETED,
                    or_(
                        Transcript.operator_id == operator.id,
                        Transcript.operator_id.is_(None),
                    ),
                )
                .order_by((Transcript.operator_id == operator.id).desc(), Transcript.completed_at.desc())
            )
            if completed_transcript is not None:
                # A successful recording is immutable unless a future explicit reprocess action is added.
                # Vocabulary/model changes may re-run matching but never trigger another paid upload.
                await _item_stage(
                    session,
                    item,
                    job,
                    JobStatus.SEARCHING_KEYWORDS,
                    "Searching for phrases",
                )
                await search_and_persist_matches(session, completed_transcript, job)
                await _lock_processable_item(session, item, job)
                attribution_unknown = completed_transcript.is_diarized
                call.processing_status = (
                    "completed_speaker_attribution_unknown"
                    if attribution_unknown
                    else "completed"
                )
                call.last_error_category = (
                    "speaker_attribution_unknown" if attribution_unknown else None
                )
                call.last_error_message = (
                    "Speakers were separated but the operator could not be identified safely."
                    if attribution_unknown
                    else None
                )
                item.status = ItemStatus.COMPLETED
                item.stage = "completed"
                item.completed_at = utc_now()
                item.error_category = call.last_error_category
                item.error_message = call.last_error_message
                await session.commit()
                return
            try:
                settings = await load_effective_yeastar_settings(redis, settings)
            except RedisError as exc:
                raise YeastarConfigurationStateError(
                    "Phone-system configuration is temporarily unavailable."
                ) from exc
            try:
                settings = await load_effective_openai_settings(redis, settings)
            except (RedisError, OpenAIConfigurationStateError) as exc:
                raise TranscriptionConfigurationError(
                    "OpenAI configuration is temporarily unavailable.",
                    "configuration_state",
                ) from exc
            audio = AudioProcessor(settings)
            await _item_stage(
                session, item, job, JobStatus.DOWNLOADING_RECORDINGS, "Downloading recordings"
            )
            suffix = Path(recording.yeastar_file_name or "").suffix.lower()
            if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".opus"}:
                raise AudioError("Recording file type is not supported.")
            storage_key = recording.storage_key or f"recordings/{recording.id}{suffix}"
            source_path = audio.safe_storage_path(storage_key)
            if not source_path.exists():
                recording.status = RecordingStatus.DOWNLOADING
                try:
                    async with YeastarClient(settings=settings, redis=redis) as yeastar:
                        download = await yeastar.download_recording(
                            recording.yeastar_recording_id, source_path
                        )
                except YeastarRecordingDownloadLimitError as exc:
                    recording.status = RecordingStatus.DISCOVERED
                    item.status = ItemStatus.QUEUED
                    item.stage = "waiting_for_recording_capacity"
                    job.current_stage = "Waiting for phone-system recording capacity"
                    await session.commit()
                    # The Celery task's existing bounded busy retry owns the
                    # deferred queue. The shared token remains cached and the
                    # authentication circuit is untouched by error 70651.
                    raise ProcessingBusyError() from exc
                recording.storage_key = storage_key
                recording.size_bytes = int(download["size_bytes"])
                recording.mime_type = str(download.get("content_type") or "")[:128] or None
                recording.file_extension = suffix
                recording.downloaded_at = utc_now()
                recording.status = RecordingStatus.DOWNLOADED
                await session.commit()
            await _item_stage(session, item, job, JobStatus.INSPECTING_AUDIO, "Checking recording quality")
            info = await audio.inspect(
                source_path,
                declared_mime_type=recording.mime_type,
                original_filename=recording.yeastar_file_name,
            )
            recording.codec_name = info.codec_name
            recording.duration_seconds = info.duration_seconds
            recording.channel_count = info.channel_count
            recording.sample_rate_hz = info.sample_rate_hz
            recording.bit_rate_bps = info.bit_rate_bps
            recording.size_bytes = info.size_bytes
            recording.sha256_checksum = info.sha256_checksum
            recording.status = RecordingStatus.INSPECTED
            recording.inspected_at = utc_now()
            await session.commit()
            participants = (
                await session.scalars(select(CallParticipant).where(CallParticipant.call_id == call.id))
            ).all()
            selected_participants = [item for item in participants if item.operator_id == operator.id]
            legs = (await session.scalars(select(CallLeg).where(CallLeg.call_id == call.id))).all()
            channel: int | None = None
            if info.channel_count == 2 and len(selected_participants) == 1:
                participant = selected_participants[0]
                interpreted = InterpretedParticipant(
                    operator_id=str(operator.id),
                    provider_extension_id=participant.provider_extension_id or "",
                    extension_number=participant.provider_extension_number or operator.extension_number,
                    leg_id=str(participant.call_leg_id or ""),
                    role=participant.role,
                    was_caller=bool(participant.operator_was_caller),
                    was_callee=bool(participant.operator_was_callee),
                    answered=participant.answered,
                )
                try:
                    async with YeastarClient(settings=settings, redis=redis) as yeastar:
                        separated = await yeastar.stereo_separated_recording_enabled()
                except YeastarError:
                    separated = False
                same_side = sum(
                    1
                    for candidate in participants
                    if candidate.operator_id is not None
                    and (
                        bool(candidate.operator_was_caller) == interpreted.was_caller
                        and bool(candidate.operator_was_callee) == interpreted.was_callee
                    )
                )
                channel = safe_operator_channel(
                    interpreted,
                    info.channel_count,
                    separated,
                    one_to_one=(len(legs) == 1 and call.queue_name is None),
                    was_transferred=call.was_transferred,
                    operators_on_same_side=same_side,
                )
            await _item_stage(
                session,
                item,
                job,
                JobStatus.EXTRACTING_OPERATOR_AUDIO,
                "Preparing the operator conversation" if channel is not None else "Separating speakers",
            )
            temp_dir = audio.safe_storage_path(f"tmp/{item.id}-{secrets.token_hex(6)}")
            temp_dir.mkdir(parents=True, exist_ok=False)
            prepared = temp_dir / "prepared.wav"
            temporary_files.append(prepared)
            if channel is not None:
                await audio.extract_channel(source_path, prepared, channel)
                diarized = False
            else:
                await audio.convert_to_mono(source_path, prepared)
                diarized = True
            model = settings.OPENAI_DIARIZATION_MODEL if diarized else settings.OPENAI_TRANSCRIPTION_MODEL
            transcript_operator_id = None if diarized else operator.id
            app_settings = await load_application_settings(session, settings)
            language = str(app_settings["default_language"])
            vocabulary = [] if diarized else await _vocabulary(session, operator, settings)
            prompt_version = None if diarized else build_vocabulary_prompt(vocabulary)[1]
            key = transcript_idempotency_key(
                recording.id,
                transcript_operator_id,
                model,
                info.sha256_checksum,
                diarized,
                language,
                prompt_version,
            )
            transcript = await session.scalar(select(Transcript).where(Transcript.idempotency_key == key))
            transcript_id = transcript.id if transcript else None
            call_leg_id = selected_participants[0].call_leg_id if len(selected_participants) == 1 else None
            needs_transcription = False
            if transcript is not None and transcript.status != TranscriptStatus.COMPLETED:
                await session.execute(
                    delete(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript.id)
                )
                transcript.status = TranscriptStatus.PROCESSING
                transcript.attempt_count += 1
                transcript.error_category = None
                transcript.error_message = None
                transcript.completed_at = None
                needs_transcription = True
                await session.commit()
            if transcript is None:
                duplicate = await session.scalar(
                    select(Transcript)
                    .where(
                        Transcript.source_audio_sha256 == info.sha256_checksum,
                        Transcript.model == model,
                        Transcript.is_diarized.is_(diarized),
                        Transcript.operator_id == transcript_operator_id,
                        Transcript.language == language,
                        Transcript.prompt_version == prompt_version,
                        Transcript.status == TranscriptStatus.COMPLETED,
                    )
                    .order_by(Transcript.completed_at.desc())
                )
                if duplicate:
                    transcript = await _clone_transcript(
                        session,
                        duplicate,
                        call=call,
                        recording=recording,
                        operator=None if diarized else operator,
                        call_leg_id=call_leg_id,
                        key=key,
                    )
                    transcript_id = transcript.id
                    await session.commit()
            if transcript is None:
                transcript = Transcript(
                    call_id=call.id,
                    recording_id=recording.id,
                    operator_id=transcript_operator_id,
                    idempotency_key=key,
                    status=TranscriptStatus.PROCESSING,
                    model=model,
                    language=language,
                    prompt_version=prompt_version,
                    attempt_count=1,
                    source_audio_sha256=info.sha256_checksum,
                    is_diarized=diarized,
                )
                session.add(transcript)
                await session.commit()
                transcript_id = transcript.id
                needs_transcription = True
            if needs_transcription:
                slot_limit = int(app_settings["max_parallel_transcriptions"])
                for slot_number in range(slot_limit):
                    candidate = redis.lock(
                        f"yca:transcription-slot:{slot_number}",
                        timeout=30 * 60 * 60,
                        blocking_timeout=0,
                    )
                    if await candidate.acquire(blocking=False):
                        transcription_slot = candidate
                        break
                if transcription_slot is None:
                    item.status = ItemStatus.QUEUED
                    item.stage = "waiting_for_transcription_capacity"
                    await session.commit()
                    raise ProcessingBusyError()
                await _item_stage(
                    session, item, job, JobStatus.TRANSCRIBING, "Transcribing conversations"
                )
                if diarized and prepared.stat().st_size <= settings.max_transcription_upload_bytes:
                    chunks = [AudioChunk(prepared, 0, info.duration_seconds)]
                else:
                    chunks = await audio.split_audio(
                        prepared,
                        temp_dir,
                        info.duration_seconds,
                        chunk_seconds=480 if diarized else 15,
                    )
                    temporary_files.extend(chunk.path for chunk in chunks)
                async def cancellation_check() -> bool:
                    try:
                        if lock is not None and await lock.owned():
                            await lock.extend(60 * 60, replace_ttl=True)
                        if transcription_slot is not None and await transcription_slot.owned():
                            await transcription_slot.extend(60 * 60, replace_ttl=True)
                    except Exception as exc:
                        raise TranscriptionError(
                            "Processing lock became unavailable.", "processing_lock"
                        ) from exc
                    return await _cancel_requested(session, job.id)

                async with OpenAITranscriptionClient(settings=settings) as transcription_client:
                    if diarized:
                        result = await transcription_client.transcribe_diarized(
                            chunks, language=language, should_cancel=cancellation_check
                        )
                    else:
                        result = await transcription_client.transcribe_isolated(
                            chunks,
                            vocabulary,
                            language=language,
                            should_cancel=cancellation_check,
                        )
                await _persist_transcription_result(
                    session,
                    transcript,
                    result,
                    call,
                    operator if not diarized else None,
                    call_leg_id if not diarized else None,
                    info.duration_seconds,
                )
                await session.commit()
            await _item_stage(
                session, item, job, JobStatus.SEARCHING_KEYWORDS, "Searching for phrases"
            )
            await search_and_persist_matches(session, transcript, job)
            await _lock_processable_item(session, item, job)
            recording.status = RecordingStatus.COMPLETED
            call.processing_status = (
                "completed_speaker_attribution_unknown" if diarized else "completed"
            )
            call.last_error_category = "speaker_attribution_unknown" if diarized else None
            call.last_error_message = (
                "Speakers were separated but the operator could not be identified safely."
                if diarized
                else None
            )
            item.status = ItemStatus.COMPLETED
            item.stage = "completed"
            item.completed_at = utc_now()
            item.error_category = "speaker_attribution_unknown" if diarized else None
            item.error_message = call.last_error_message
            await session.commit()
            outstanding_for_recording = await session.scalar(
                select(func.count())
                .select_from(ProcessingJobItem)
                .where(
                    ProcessingJobItem.recording_id == recording.id,
                    ProcessingJobItem.id != item.id,
                    ProcessingJobItem.status.in_([ItemStatus.QUEUED, ItemStatus.PROCESSING]),
                )
            ) or 0
            if bool(app_settings["delete_audio_after_transcription"]) and not outstanding_for_recording:
                _, failures = audio.remove_files([source_path])
                if not failures:
                    recording.storage_key = None
                    recording.status = RecordingStatus.DELETED
                    recording.deleted_at = utc_now()
                    session.add(
                        AuditLog(
                            created_at=utc_now(),
                            user_id=None,
                            action="audio.delete",
                            resource_type="recording",
                            resource_id=str(recording.id),
                            outcome="success",
                            details={"reason": "after_transcription"},
                        )
                    )
                else:
                    recording.last_error_category = "audio_cleanup_pending"
                    recording.last_error_message = "Audio cleanup will be retried."
                await session.commit()
    except CONNECTION_PAUSE_ERRORS as exc:
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item:
                disposition = await _settle_stale_item(session, item, transcript_id)
                if disposition != "active":
                    return
                item.status = ItemStatus.WAITING_FOR_CONNECTION
                item.stage = "waiting_for_connection"
                item.error_category = getattr(exc, "category", "connection_paused")
                item.error_message = "Test the phone-system connection in Settings."
                item.completed_at = None
                job = await session.get(ProcessingJob, item.job_id)
                if job:
                    job.status = JobStatus.WAITING_FOR_CONNECTION
                    job.current_stage = "Waiting for phone-system connection"
                    job.completed_at = None
                recording = (
                    await session.get(Recording, item.recording_id)
                    if item.recording_id
                    else None
                )
                if recording and recording.status == RecordingStatus.DOWNLOADING:
                    recording.status = RecordingStatus.DISCOVERED
                await session.commit()
    except ProcessingBusyError:
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item and await _settle_stale_item(session, item, transcript_id) != "active":
                return
        raise
    except (ProcessingCancelledError, TranscriptionCancelledError):
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item:
                disposition = await _settle_stale_item(session, item, transcript_id)
                if disposition != "active":
                    return
                item.status = ItemStatus.CANCELLED
                item.stage = "cancelled"
                item.completed_at = utc_now()
                if transcript_id:
                    transcript = await session.get(Transcript, transcript_id)
                    if transcript and transcript.status != TranscriptStatus.COMPLETED:
                        transcript.status = TranscriptStatus.FAILED
                        transcript.error_category = "cancelled"
                        transcript.error_message = "Transcription was cancelled."
                await session.commit()
    except StaleProcessingTaskError:
        return
    except Exception as exc:
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item:
                disposition = await _settle_stale_item(session, item, transcript_id)
                if disposition != "active":
                    return
                item.status = ItemStatus.FAILED
                item.stage = "failed"
                item.error_category = getattr(exc, "category", "unexpected")
                item.error_message = _safe_error_message(exc)
                item.completed_at = utc_now()
                call = await session.get(Call, item.call_id)
                if call:
                    call.processing_status = "failed"
                    call.last_error_category = item.error_category
                    call.last_error_message = item.error_message
                recording = await session.get(Recording, item.recording_id) if item.recording_id else None
                if recording:
                    recording.status = RecordingStatus.FAILED
                    recording.last_error_category = item.error_category
                    recording.last_error_message = item.error_message
                if transcript_id:
                    transcript = await session.get(Transcript, transcript_id)
                    if transcript and transcript.status != TranscriptStatus.COMPLETED:
                        transcript.status = TranscriptStatus.FAILED
                        transcript.error_category = item.error_category
                        transcript.error_message = item.error_message
                await session.commit()
        raise
    finally:
        AudioProcessor(settings).remove_files(temporary_files)
        for path in {item.parent for item in temporary_files}:
            try:
                path.rmdir()
            except OSError:
                pass
        try:
            if lock is not None and await lock.owned():
                await lock.release()
        except Exception:
            pass
        try:
            if transcription_slot is not None and await transcription_slot.owned():
                await transcription_slot.release()
        except Exception:
            pass
        await finalize_job_for_item(item_id)


async def _persist_transcription_result(
    session: AsyncSession,
    transcript: Transcript,
    result: TranscriptionResult,
    call: Call,
    operator: Operator | None,
    call_leg_id: UUID | None,
    audio_duration: float,
) -> None:
    transcript.model = result.model
    transcript.language = result.language
    transcript.prompt_version = result.prompt_version
    transcript.processing_duration_seconds = result.processing_duration_seconds
    transcript.audio_duration_seconds = audio_duration
    transcript.api_usage = result.usage
    transcript.status = TranscriptStatus.COMPLETED
    transcript.completed_at = utc_now()
    transcript.original_text = result.text
    transcript.normalized_text = normalize_greek(result.text)
    transcript.error_category = None
    transcript.error_message = None
    for sequence, segment in enumerate(result.segments, start=1):
        session.add(
            TranscriptSegment(
                transcript_id=transcript.id,
                call_id=call.id,
                call_leg_id=call_leg_id if operator else None,
                operator_id=operator.id if operator else None,
                speaker_label=operator.display_name if operator else segment.speaker_label,
                speaker_source=(
                    SpeakerSource.STEREO_CHANNEL if operator else SpeakerSource.OPENAI_DIARIZATION
                ),
                start_seconds=Decimal(str(round(segment.start_seconds, 3))),
                end_seconds=Decimal(str(round(segment.end_seconds, 3))),
                original_text=segment.text,
                normalized_text=normalize_greek(segment.text),
                confidence=(Decimal(str(segment.confidence)) if segment.confidence is not None else None),
                transcription_model=result.model,
                sequence_number=sequence,
            )
        )
    await session.flush()


async def finalize_job_for_item(item_id: UUID) -> None:
    async with AsyncSessionFactory() as session:
        item = await session.get(ProcessingJobItem, item_id)
        if item is None:
            return
        job = await session.scalar(
            select(ProcessingJob).where(ProcessingJob.id == item.job_id).with_for_update()
        )
        if job is None or job.status in FINAL_JOB_STATES:
            return
        item_rows = (
            await session.execute(
                select(ProcessingJobItem.call_id, ProcessingJobItem.status).where(
                    ProcessingJobItem.job_id == job.id
                )
            )
        ).all()
        statuses_by_call: dict[UUID, list[ItemStatus]] = defaultdict(list)
        for call_id, item_status in item_rows:
            statuses_by_call[call_id].append(item_status)
        total = len(item_rows)
        # A call is complete only when every selected operator completed. This
        # keeps a mixed A=complete/B=failed call out of both success and failure
        # counters at the same time.
        completed = sum(
            bool(states) and all(state == ItemStatus.COMPLETED for state in states)
            for states in statuses_by_call.values()
        )
        failed_items = sum(
            any(state == ItemStatus.FAILED for state in states)
            for states in statuses_by_call.values()
        )
        discovery_failures = int((job.request_filters or {}).get("_discovery_failures", 0))
        failed = failed_items + discovery_failures
        final_count = sum(item_status in FINAL_ITEM_STATES for _, item_status in item_rows)
        job.calls_completed = completed
        job.calls_failed = failed
        job.progress_percent = round(final_count * 100 / total) if total else 100
        if final_count == total:
            job.completed_at = utc_now()
            if job.cancellation_requested:
                job.status = JobStatus.CANCELLED
                job.current_stage = "Cancelled"
            elif failed:
                job.status = JobStatus.COMPLETED_WITH_ERRORS
                job.current_stage = "Complete with some errors"
            else:
                job.status = JobStatus.COMPLETED
                job.current_stage = "Complete"
        await session.commit()


async def cleanup_retention_records() -> dict[str, int]:
    settings = get_settings()
    deleted_transcripts = 0
    deleted_audio = 0
    cleanup_failed = 0
    immediate_audio_retries = 0
    async with AsyncSessionFactory() as session:
        application = await load_application_settings(session, settings)
        immediate_pending = (
            await session.scalars(
                select(Recording).where(
                    Recording.last_error_category == "audio_cleanup_pending",
                    Recording.storage_key.is_not(None),
                )
            )
        ).all()
        for recording in immediate_pending:
            assert recording.storage_key is not None
            path = AudioProcessor(settings).safe_storage_path(recording.storage_key)
            _, failures = AudioProcessor(settings).remove_files([path])
            if failures:
                cleanup_failed += 1
                continue
            immediate_audio_retries += 1
            deleted_audio += 1
            recording.storage_key = None
            recording.status = RecordingStatus.DELETED
            recording.deleted_at = utc_now()
            recording.last_error_category = None
            recording.last_error_message = None
        cutoff = utc_now() - timedelta(days=int(application["transcript_retention_days"]))
        transcripts = (
            await session.scalars(
                select(Transcript).where(
                    Transcript.completed_at.is_not(None), Transcript.completed_at < cutoff
                )
            )
        ).all()
        recording_ids = {item.recording_id for item in transcripts}
        pending_recording_ids = (
            await session.scalars(
                select(Recording.id).where(
                    Recording.last_error_category == "retention_cleanup_pending"
                )
            )
        ).all()
        recording_ids.update(pending_recording_ids)
        for transcript in transcripts:
            await session.delete(transcript)
            deleted_transcripts += 1
        await session.flush()
        for recording_id in recording_ids:
            remaining = await session.scalar(
                select(func.count()).select_from(Transcript).where(Transcript.recording_id == recording_id)
            ) or 0
            recording = await session.get(Recording, recording_id)
            if remaining or not recording or not recording.storage_key:
                continue
            path = AudioProcessor(settings).safe_storage_path(recording.storage_key)
            _, failures = AudioProcessor(settings).remove_files([path])
            if failures:
                cleanup_failed += 1
                recording.last_error_category = "retention_cleanup_pending"
                recording.last_error_message = "Audio retention cleanup will be retried."
            else:
                deleted_audio += 1
                recording.storage_key = None
                recording.status = RecordingStatus.DELETED
                recording.deleted_at = utc_now()
                recording.last_error_category = None
                recording.last_error_message = None
        session.add(
            AuditLog(
                created_at=utc_now(),
                user_id=None,
                action="retention.delete",
                resource_type="retention_cleanup",
                outcome="success" if not cleanup_failed else "partial",
                details={
                    "transcripts_deleted": deleted_transcripts,
                    "audio_deleted": deleted_audio,
                    "cleanup_failed": cleanup_failed,
                    "immediate_audio_retries": immediate_audio_retries,
                },
            )
        )
        await session.commit()
    return {
        "transcripts_deleted": deleted_transcripts,
        "audio_deleted": deleted_audio,
        "cleanup_failed": cleanup_failed,
    }
