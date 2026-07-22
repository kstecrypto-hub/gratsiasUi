from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.workers.pipeline as pipeline_module
from app.database.base import Base
from app.models import (
    Call,
    CallLeg,
    Keyword,
    KeywordCategory,
    KeywordMatch,
    Operator,
    ProcessingJob,
    ProcessingJobItem,
    Recording,
    SyncRun,
    Transcript,
    TranscriptSegment,
    User,
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
from app.services.keyword_matching.normalization import normalize_greek
from app.services.yeastar.cdr import CDRSummary
from app.workers.pipeline import (
    DiscoveryResult,
    _fetch_cdrs,
    _recording_for_participant,
    _recordings_for_participant,
    search_and_persist_matches,
    transcript_idempotency_key,
)
from app.workers.tasks import _busy_retry_delay, _recording_assignment_retry_delay, process_job_item


def test_transcript_idempotency_key_is_deterministic_and_scoped() -> None:
    recording_id = UUID("00000000-0000-0000-0000-000000000001")
    other_recording_id = UUID("00000000-0000-0000-0000-000000000002")
    operator_id = UUID("00000000-0000-0000-0000-000000000101")
    checksum = "a" * 64

    key = transcript_idempotency_key(
        recording_id, operator_id, "gpt-4o-transcribe", checksum, False
    )
    assert key == transcript_idempotency_key(
        recording_id, operator_id, "gpt-4o-transcribe", checksum, False
    )
    assert len(key) == 64
    assert key != transcript_idempotency_key(
        other_recording_id, operator_id, "gpt-4o-transcribe", checksum, False
    )
    assert key != transcript_idempotency_key(
        recording_id, None, "gpt-4o-transcribe", checksum, False
    )
    assert key != transcript_idempotency_key(
        recording_id, operator_id, "gpt-4o-transcribe-diarize", checksum, True
    )
    assert key != transcript_idempotency_key(
        recording_id, operator_id, "gpt-4o-transcribe", "b" * 64, False
    )
    assert key != transcript_idempotency_key(
        recording_id,
        operator_id,
        "gpt-4o-transcribe",
        checksum,
        False,
        language="en",
    )
    assert key != transcript_idempotency_key(
        recording_id,
        operator_id,
        "gpt-4o-transcribe",
        checksum,
        False,
        language="el",
        prompt_version="new-vocabulary",
    )


def _assignment_leg(*, payload: dict[str, object] | None = None) -> CallLeg:
    return CallLeg(
        call_id=uuid4(),
        yeastar_leg_id="leg-1",
        sequence_number=1,
        provider_payload=payload or {},
    )


def _assignment_recording(
    recording_id: str,
    *,
    filename: str | None = None,
    caller: str | None = None,
    callee: str | None = None,
) -> Recording:
    return Recording(
        call_id=uuid4(),
        yeastar_recording_id=recording_id,
        yeastar_file_name=filename,
        call_from_number=caller,
        call_to_number=callee,
        status=RecordingStatus.DISCOVERED,
    )


def test_recording_assignment_matches_legacy_cdr_filename() -> None:
    expected = _assignment_recording("recording-expected", filename="20260720-1001.wav")
    other = _assignment_recording("recording-other", filename="20260720-1002.wav")
    leg = _assignment_leg(payload={"record_file": "archive/20260720-1001.wav"})

    recording, reason = _recording_for_participant([other, expected], SimpleNamespace(), leg)

    assert recording is expected
    assert reason is None


def test_recording_assignment_matches_unique_ordered_endpoints() -> None:
    expected = _assignment_recording(
        "recording-expected",
        caller="+30 (210) 000-0000",
        callee="1001",
    )
    other = _assignment_recording(
        "recording-other",
        caller="+30 210 000 0000",
        callee="1002",
    )
    leg = _assignment_leg()
    leg.caller_number = "302100000000"
    leg.callee_number = "1001"

    recording, reason = _recording_for_participant([other, expected], SimpleNamespace(), leg)

    assert recording is expected
    assert reason is None


def test_ambiguous_recordings_are_left_for_queued_rediscovery() -> None:
    first = _assignment_recording("recording-first")
    second = _assignment_recording("recording-second")

    recording, reason = _recording_for_participant(
        [first, second], SimpleNamespace(), _assignment_leg()
    )

    assert recording is None
    assert reason == "recording_assignment_pending"


def test_conflicting_explicit_id_and_filename_need_endpoint_corroboration() -> None:
    explicit = _assignment_recording(
        "recording-explicit",
        filename="explicit.wav",
        caller="302100000000",
        callee="1001",
    )
    filename_only = _assignment_recording(
        "recording-filename",
        filename="filename.wav",
        caller="302100000000",
        callee="1002",
    )
    leg = _assignment_leg(payload={"record_file": "filename.wav"})
    leg.provider_recording_id = "recording-explicit"
    leg.caller_number = "302100000000"
    leg.callee_number = "1001"

    matched, reason = _recordings_for_participant(
        [explicit, filename_only], SimpleNamespace(), leg
    )

    assert matched == []
    assert reason == "recording_assignment_pending"


def test_recording_assignment_queue_uses_bounded_backoff() -> None:
    assert _recording_assignment_retry_delay(0) == 15
    assert _recording_assignment_retry_delay(1) == 30
    assert _recording_assignment_retry_delay(10) == 300


def test_recording_capacity_queue_retries_without_a_fixed_attempt_limit() -> None:
    assert process_job_item.max_retries is None
    assert _busy_retry_delay(0) == 5
    assert _busy_retry_delay(1) == 10
    assert _busy_retry_delay(10) == 300


async def _assignment_discovery_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recordings: list[dict[str, object]],
    detail: dict[str, object],
    legacy_item: tuple[ItemStatus, str | None] | None = None,
    existing_run: bool = False,
) -> tuple[DiscoveryResult, list[tuple[ItemStatus, str, UUID | None]], JobStatus, int]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    class FakeYeastarClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "FakeYeastarClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def authenticate(self) -> None:
            return None

        async def search_cdrs(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {
                "data": [
                    {
                        "uid": "assignment-call",
                        "id": "cdr-1",
                        "time": int(now.timestamp()),
                        "call_type": "Inbound",
                        "call_from_number": "302100000000",
                        "call_to_number": "1001",
                    }
                ],
                "total_number": 1,
            }

        async def search_recordings(self, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
            return recordings

        async def get_cdr_detail(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return detail

    try:
        async with sessions() as session:
            user = User(
                email="assignment-test@example.test",
                password_hash="not-used-in-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id="extension-1001",
                extension_number="1001",
                display_name="Operator 1001",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            session.add_all([user, operator])
            await session.flush()
            job = ProcessingJob(
                idempotency_key=f"assignment-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.QUEUED,
                date_from=now - timedelta(hours=1),
                date_to=now + timedelta(hours=1),
                include_all_speakers=False,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={},
            )
            session.add(job)
            await session.flush()
            if legacy_item is not None:
                legacy_status, legacy_recording_id = legacy_item
                call = Call(
                    yeastar_uid="assignment-call",
                    started_at=now,
                    direction=Direction.INBOUND,
                    duration_seconds=30,
                )
                session.add(call)
                await session.flush()
                recording_id = None
                if legacy_recording_id is not None:
                    recording = Recording(
                        call_id=call.id,
                        yeastar_recording_id=legacy_recording_id,
                        status=RecordingStatus.COMPLETED,
                    )
                    session.add(recording)
                    await session.flush()
                    recording_id = recording.id
                session.add(
                    ProcessingJobItem(
                        job_id=job.id,
                        call_id=call.id,
                        operator_id=operator.id,
                        recording_id=recording_id,
                        idempotency_key=pipeline_module._legacy_processing_item_key(
                            job.id,
                            call.id,
                            operator.id,
                        ),
                        status=legacy_status,
                        stage="completed" if legacy_status == ItemStatus.COMPLETED else "queued",
                        completed_at=now if legacy_status == ItemStatus.COMPLETED else None,
                    )
                )
            if existing_run:
                session.add(
                    SyncRun(
                        sync_type=SyncType.CALLS,
                        status=RunStatus.COMPLETED,
                        idempotency_key=f"job:{job.id}:attempt:{job.attempt_count}",
                        processing_job_id=job.id,
                        started_at=now,
                        completed_at=now,
                        date_from=job.date_from,
                        date_to=job.date_to,
                    )
                )
            await session.commit()
            job_id = job.id

        async def effective_settings(_redis: object, settings: object) -> object:
            return settings

        async def configured(*_args: object, **_kwargs: object) -> tuple[None, bool, bool]:
            return None, True, False

        settings = SimpleNamespace(APP_TIMEZONE="UTC")
        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline_module, "get_redis", lambda: object())
        monkeypatch.setattr(pipeline_module, "load_effective_yeastar_settings", effective_settings)
        monkeypatch.setattr(pipeline_module, "reconcile_configuration_fingerprint", configured)
        monkeypatch.setattr(pipeline_module, "YeastarClient", FakeYeastarClient)

        result = await pipeline_module._discover_job_items_locked(job_id)
        async with sessions() as session:
            items = (
                await session.scalars(
                    select(ProcessingJobItem).where(ProcessingJobItem.job_id == job_id)
                )
            ).all()
            snapshot = [(item.status, item.stage, item.recording_id) for item in items]
            job = await session.get(ProcessingJob, job_id)
            assert job is not None
            run_count = await session.scalar(select(func.count()).select_from(SyncRun)) or 0
            return result, snapshot, job.status, run_count
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ambiguous_assignment_stays_queued_and_requests_rediscovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, job_status, run_count = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {"id": "recording-1", "uid": "assignment-call", "file": "first.wav"},
            {"id": "recording-2", "uid": "assignment-call", "file": "second.wav"},
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "status": "ANSWERED",
                }
            ]
        },
    )

    assert result.item_ids == []
    assert result.recording_assignment_pending is True
    assert items == [(ItemStatus.QUEUED, "waiting_for_recording_assignment", None)]
    assert job_status == JobStatus.FINDING_RECORDINGS
    # A retry starts a fresh discovery pass rather than reusing a completed run.
    assert run_count == 0


@pytest.mark.asyncio
async def test_resolved_assignment_dispatches_only_recording_backed_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, job_status, run_count = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {"id": "recording-1", "uid": "assignment-call", "file": "first.wav"},
            {"id": "recording-2", "uid": "assignment-call", "file": "second.wav"},
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "recording_id": "recording-2",
                    "status": "ANSWERED",
                }
            ]
        },
    )

    assert result.recording_assignment_pending is False
    assert len(result.item_ids) == 1
    assert items[0][0] == ItemStatus.QUEUED
    assert items[0][1] == "queued"
    assert items[0][2] is not None
    assert job_status == JobStatus.DOWNLOADING_RECORDINGS
    assert run_count == 1


@pytest.mark.asyncio
async def test_every_safely_matched_recording_gets_an_item_for_the_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, job_status, _ = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {
                "id": "recording-1",
                "uid": "assignment-call",
                "file": "first.wav",
                "call_from_number": "302100000000",
                "call_to_number": "1001",
            },
            {
                "id": "recording-2",
                "uid": "assignment-call",
                "file": "second.wav",
                "call_from_number": "302100000000",
                "call_to_number": "1001",
            },
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "recording_id": "recording-1",
                    "status": "ANSWERED",
                }
            ]
        },
    )

    assert result.recording_assignment_pending is False
    assert len(result.item_ids) == 2
    assert len(items) == 2
    assert all(status == ItemStatus.QUEUED for status, _, _ in items)
    assert all(stage == "queued" for _, stage, _ in items)
    assert all(recording_id is not None for _, _, recording_id in items)
    assert job_status == JobStatus.DOWNLOADING_RECORDINGS


@pytest.mark.asyncio
async def test_multiple_participant_legs_do_not_drop_safely_matched_recordings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, _, _ = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {
                "id": "recording-inbound",
                "uid": "assignment-call",
                "file": "inbound.wav",
                "call_from_number": "302100000000",
                "call_to_number": "1001",
            },
            {
                "id": "recording-transfer",
                "uid": "assignment-call",
                "file": "transfer.wav",
                "call_from_number": "1001",
                "call_to_number": "1002",
            },
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-inbound",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "status": "ANSWERED",
                },
                {
                    "transaction_id": "transaction-2",
                    "cdr_id": "leg-transfer",
                    "call_from_ext_id": "extension-1001",
                    "call_from_number": "1001",
                    "call_to_number": "1002",
                    "status": "ANSWERED",
                },
            ]
        },
    )

    assert len(result.item_ids) == 2
    assert len(items) == 2
    assert all(recording_id is not None for _, _, recording_id in items)


@pytest.mark.asyncio
async def test_pre_recording_aware_completed_item_is_reused_without_requeueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, job_status, _ = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {"id": "recording-legacy", "uid": "assignment-call", "file": "legacy.wav"}
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "status": "ANSWERED",
                }
            ]
        },
        legacy_item=(ItemStatus.COMPLETED, "recording-legacy"),
    )

    assert result.item_ids == []
    assert len(items) == 1
    assert items[0][0] == ItemStatus.COMPLETED
    assert items[0][1] == "completed"
    assert items[0][2] is not None
    assert job_status == JobStatus.COMPLETED


@pytest.mark.asyncio
async def test_existing_failed_recording_item_is_not_overwritten_as_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, job_status, _ = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {"id": "recording-legacy", "uid": "assignment-call", "file": "legacy.wav"}
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "status": "ANSWERED",
                }
            ]
        },
        legacy_item=(ItemStatus.FAILED, "recording-legacy"),
    )

    assert result.item_ids == []
    assert len(items) == 1
    assert items[0][0] == ItemStatus.FAILED
    assert job_status == JobStatus.COMPLETED_WITH_ERRORS


@pytest.mark.asyncio
async def test_pre_recording_aware_null_placeholder_is_converted_to_recording_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, _, run_count = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {"id": "recording-now-linked", "uid": "assignment-call", "file": "linked.wav"}
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "status": "ANSWERED",
                }
            ]
        },
        legacy_item=(ItemStatus.QUEUED, None),
        existing_run=True,
    )

    assert len(result.item_ids) == 1
    assert len(items) == 1
    assert items[0][0] == ItemStatus.QUEUED
    assert items[0][1] == "queued"
    assert items[0][2] is not None
    assert run_count == 1


@pytest.mark.asyncio
async def test_legacy_cdr_fetch_uses_local_adapter_and_groups_duplicate_uids() -> None:
    class LegacyClient:
        def __init__(self) -> None:
            self.search_all_calls: list[tuple[datetime, datetime, dict[str, object]]] = []

        async def search_all_cdrs(
            self,
            date_from: datetime,
            date_to: datetime,
            filters: dict[str, object],
        ) -> list[CDRSummary]:
            self.search_all_calls.append((date_from, date_to, filters))
            return [
                CDRSummary(
                    uid="call-1",
                    time=1_784_000_030,
                    call_from_number="1001",
                    call_to_number="2001",
                ),
                CDRSummary(
                    uid="call-1",
                    time=1_784_000_000,
                    call_from_number="1000",
                    call_to_number="2000",
                ),
                CDRSummary(uid="call-2", time=1_784_000_040),
            ]

        async def search_cdrs(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("legacy CDR discovery must not use v2 paging")

    client = LegacyClient()
    date_from = datetime(2026, 7, 1, tzinfo=UTC)
    date_to = datetime(2026, 7, 2, tzinfo=UTC)
    cdrs, legacy_summaries = await _fetch_cdrs(
        client,  # type: ignore[arg-type]
        date_from,
        date_to,
        {"call_type": "Inbound"},
        SimpleNamespace(
            YEASTAR_DATETIME_FORMAT="%m/%d/%Y %H:%M:%S",
            APP_TIMEZONE="UTC",
        ),
        cdr_api_version="v1",
    )

    assert list(cdrs) == ["call-1", "call-2"]
    assert cdrs["call-1"]["call_from_number"] == "1000"
    assert [summary.call_from_number for summary in legacy_summaries["call-1"]] == [
        "1001",
        "1000",
    ]
    assert client.search_all_calls == [(date_from, date_to, {"call_type": "Inbound"})]


@pytest.mark.asyncio
async def test_unknown_diarized_speakers_are_not_searched_by_default() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    try:
        async with sessions() as session:
            call_id = uuid4()
            recording_id = uuid4()
            transcript_id = uuid4()
            segment_id = uuid4()
            category_id = uuid4()
            keyword_id = uuid4()
            job_id = uuid4()
            call = Call(
                id=call_id,
                yeastar_uid="call-1",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
            )
            recording = Recording(
                id=recording_id,
                call_id=call_id,
                yeastar_recording_id="recording-1",
                status=RecordingStatus.COMPLETED,
            )
            transcript = Transcript(
                id=transcript_id,
                call_id=call_id,
                recording_id=recording_id,
                operator_id=None,
                idempotency_key="transcript-1",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe-diarize",
                language="el",
                source_audio_sha256="a" * 64,
                is_diarized=True,
                completed_at=now,
                original_text="Ζητώ επιστροφή χρημάτων.",
                normalized_text=normalize_greek("Ζητώ επιστροφή χρημάτων."),
            )
            segment = TranscriptSegment(
                id=segment_id,
                transcript_id=transcript_id,
                call_id=call_id,
                call_leg_id=None,
                operator_id=None,
                speaker_label="chunk-1:A",
                speaker_source=SpeakerSource.OPENAI_DIARIZATION,
                start_seconds=Decimal("2.000"),
                end_seconds=Decimal("4.000"),
                original_text="Ζητώ επιστροφή χρημάτων.",
                normalized_text=normalize_greek("Ζητώ επιστροφή χρημάτων."),
                transcription_model="gpt-4o-transcribe-diarize",
                sequence_number=1,
            )
            category = KeywordCategory(id=category_id, name="Επιστροφές", active=True)
            keyword = Keyword(
                id=keyword_id,
                category_id=category_id,
                canonical_phrase="επιστροφή χρημάτων",
                normalized_phrase=normalize_greek("επιστροφή χρημάτων"),
                active=True,
                whole_word=True,
                exact_phrase=True,
            )
            job = ProcessingJob(
                id=job_id,
                idempotency_key="job-1",
                requested_by_id=uuid4(),
                status=JobStatus.SEARCHING_KEYWORDS,
                date_from=now - timedelta(hours=1),
                date_to=now,
                include_all_speakers=False,
                selected_operator_ids=[],
                selected_category_ids=[],
                request_filters={},
            )
            session.add_all([call, recording, transcript, segment, category, keyword, job])
            await session.commit()

            assert await search_and_persist_matches(session, transcript, job) == 0
            assert (
                await session.scalar(select(func.count()).select_from(KeywordMatch)) or 0
            ) == 0

            job.include_all_speakers = True
            assert await search_and_persist_matches(session, transcript, job) == 1
            await session.commit()
            match = await session.scalar(select(KeywordMatch))
            assert match is not None
            assert match.operator_id is None
            assert match.transcript_segment_id == segment_id
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_discovery_generic_error_after_cancel_preserves_cancelled_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel_then_fail(
        session: object,
        _cdr: dict[str, object],
        _settings: object,
    ) -> Call:
        job = await session.scalar(select(ProcessingJob))  # type: ignore[attr-defined]
        assert job is not None
        job.cancellation_requested = True
        job.status = JobStatus.CANCELLED
        job.current_stage = "Cancelled"
        job.completed_at = datetime.now(UTC)
        await session.commit()  # type: ignore[attr-defined]
        raise RuntimeError("late discovery failure")

    monkeypatch.setattr(pipeline_module, "_upsert_call", cancel_then_fail)

    result, items, job_status, run_count = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[],
        detail={},
    )

    assert result == DiscoveryResult([])
    assert items == []
    assert job_status == JobStatus.CANCELLED
    assert run_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "late_error",
    [
        pytest.param(RuntimeError("late processing failure"), id="generic"),
        pytest.param(
            pipeline_module.YeastarPermissionError("late connection pause"),
            id="connection-pause",
        ),
    ],
)
async def test_late_item_error_cannot_overwrite_committed_cancellation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    late_error: Exception,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    class FakeLock:
        async def acquire(self, *, blocking: bool = False) -> bool:
            return True

        async def owned(self) -> bool:
            return True

        async def release(self) -> None:
            return None

    class FakeRedis:
        def lock(self, *_args: object, **_kwargs: object) -> FakeLock:
            return FakeLock()

    try:
        async with sessions() as session:
            user = User(
                email=f"lifecycle-{uuid4()}@example.test",
                password_hash="not-used-in-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"extension-{uuid4()}",
                extension_number="1001",
                display_name="Lifecycle Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
                processing_status="pending",
            )
            session.add_all([user, operator, call])
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            job = ProcessingJob(
                idempotency_key=f"lifecycle-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.QUEUED,
                date_from=now - timedelta(hours=1),
                date_to=now,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={},
            )
            session.add_all([recording, job])
            await session.flush()
            transcript = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                idempotency_key=f"lifecycle-transcript-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                source_audio_sha256="c" * 64,
                is_diarized=False,
                completed_at=now,
            )
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                idempotency_key=f"lifecycle-item-{uuid4()}",
                status=ItemStatus.QUEUED,
                stage="queued",
            )
            session.add_all([transcript, item])
            await session.commit()
            job_id = job.id
            item_id = item.id
            call_id = call.id
            recording_id = recording.id
            transcript_id = transcript.id

        cancelled_at = now - timedelta(minutes=1)

        async def cancel_then_fail(
            session: object,
            _transcript: Transcript,
            _job: ProcessingJob,
        ) -> int:
            current_job = await session.get(ProcessingJob, job_id)  # type: ignore[attr-defined]
            current_item = await session.get(ProcessingJobItem, item_id)  # type: ignore[attr-defined]
            assert current_job is not None and current_item is not None
            current_job.cancellation_requested = True
            current_job.status = JobStatus.CANCELLED
            current_job.current_stage = "Cancelled"
            current_job.completed_at = cancelled_at
            current_item.status = ItemStatus.CANCELLED
            current_item.stage = "cancelled"
            current_item.completed_at = cancelled_at
            await session.commit()  # type: ignore[attr-defined]
            raise late_error

        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        monkeypatch.setattr(pipeline_module, "get_redis", lambda: FakeRedis())
        monkeypatch.setattr(
            pipeline_module,
            "get_settings",
            lambda: SimpleNamespace(STORAGE_ROOT=tmp_path),
        )
        monkeypatch.setattr(
            pipeline_module,
            "search_and_persist_matches",
            cancel_then_fail,
        )

        await pipeline_module.process_item(item_id)

        async with sessions() as session:
            final_job = await session.get(ProcessingJob, job_id)
            final_item = await session.get(ProcessingJobItem, item_id)
            final_call = await session.get(Call, call_id)
            final_recording = await session.get(Recording, recording_id)
            final_transcript = await session.get(Transcript, transcript_id)
            assert final_job is not None
            assert final_job.status == JobStatus.CANCELLED
            assert final_job.current_stage == "Cancelled"
            assert final_job.completed_at == cancelled_at.replace(tzinfo=None)
            assert final_item is not None and final_item.status == ItemStatus.CANCELLED
            assert final_call is not None and final_call.processing_status == "pending"
            assert final_recording is not None
            assert final_recording.status == RecordingStatus.COMPLETED
            assert final_transcript is not None
            assert final_transcript.status == TranscriptStatus.COMPLETED
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_status", sorted(pipeline_module.FINAL_JOB_STATES))
async def test_finalize_does_not_mutate_terminal_job(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: JobStatus,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    try:
        async with sessions() as session:
            user = User(
                email=f"finalize-{uuid4()}@example.test",
                password_hash="not-used-in-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"extension-{uuid4()}",
                extension_number="1002",
                display_name="Finalize Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"call-{uuid4()}",
                started_at=now,
                direction=Direction.OUTBOUND,
                duration_seconds=45,
            )
            session.add_all([user, operator, call])
            await session.flush()
            job = ProcessingJob(
                idempotency_key=f"finalize-job-{uuid4()}",
                requested_by_id=user.id,
                status=terminal_status,
                date_from=now - timedelta(hours=1),
                date_to=now,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={"_discovery_failures": 4},
                progress_percent=37,
                current_stage="Historical snapshot",
                calls_completed=7,
                calls_failed=8,
                cancellation_requested=terminal_status == JobStatus.CANCELLED,
                completed_at=now - timedelta(minutes=5),
            )
            session.add(job)
            await session.flush()
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                idempotency_key=f"finalize-item-{uuid4()}",
                status=ItemStatus.COMPLETED,
                stage="completed",
                completed_at=now,
            )
            session.add(item)
            await session.commit()
            item_id = item.id
            job_id = job.id
            before = (
                job.status,
                job.current_stage,
                job.progress_percent,
                job.calls_completed,
                job.calls_failed,
                job.completed_at.replace(tzinfo=None) if job.completed_at else None,
            )

        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        await pipeline_module.finalize_job_for_item(item_id)

        async with sessions() as session:
            job = await session.get(ProcessingJob, job_id)
            assert job is not None
            assert (
                job.status,
                job.current_stage,
                job.progress_percent,
                job.calls_completed,
                job.calls_failed,
                job.completed_at,
            ) == before
    finally:
        await engine.dispose()
