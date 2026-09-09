from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from redis.exceptions import RedisError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.api.results as results_api
import app.workers.pipeline as pipeline
from app.auth.dependencies import get_current_user
from app.core.config import Settings
from app.database.base import Base
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
    User,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    MatchMethod,
    ParticipantRole,
    RecordingStatus,
    SpeakerSource,
    TranscriptStatus,
)


@dataclass
class Harness:
    client: AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    settings: Settings
    user_id: UUID


@pytest.fixture
async def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = (tmp_path / "multi-operator.sqlite3").as_posix()
    settings = Settings(
        APP_ENV="test",
        APP_SECRET_KEY="multi-operator-test-secret-key-with-32-characters",
        APP_TIMEZONE="Europe/Athens",
        STORAGE_ROOT=tmp_path / "storage",
        DATABASE_URL=f"sqlite+aiosqlite:///{database_path}",
        OPENAI_TRANSCRIPTION_MODEL="current-transcription-model",
        TRANSCRIPTION_LANGUAGE="el",
    )
    settings.STORAGE_ROOT.mkdir()
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async with sessions() as session:
        user = User(
            email=f"admin-{uuid4()}@example.test",
            password_hash="unused-in-route-contract-tests",
            is_active=True,
        )
        session.add(user)
        await session.commit()
        user_id = user.id

    app = FastAPI()
    app.include_router(results_api.router, prefix="/api")

    async def override_db():
        async with sessions() as session:
            yield session

    async def override_user() -> User:
        async with sessions() as session:
            user = await session.get(User, user_id)
            assert user is not None
            return user

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    monkeypatch.setattr(results_api, "get_settings", lambda: settings)

    client = AsyncClient(
        transport=ASGITransport(app=app),
        base_url="https://testserver",
    )
    try:
        yield Harness(client=client, sessions=sessions, settings=settings, user_id=user_id)
    finally:
        await client.aclose()
        await engine.dispose()


async def _operator(session: AsyncSession, name: str, extension: str) -> Operator:
    operator = Operator(
        yeastar_extension_id=f"extension-{uuid4()}",
        extension_number=extension,
        display_name=name,
        enabled=True,
        last_synced_at=datetime.now(UTC),
    )
    session.add(operator)
    await session.flush()
    return operator


async def _job(
    session: AsyncSession,
    *,
    user_id: UUID,
    now: datetime,
    operators: list[Operator],
    created_at: datetime,
) -> ProcessingJob:
    job = ProcessingJob(
        idempotency_key=f"job-{uuid4()}",
        requested_by_id=user_id,
        status=JobStatus.COMPLETED_WITH_ERRORS,
        date_from=now - timedelta(hours=1),
        date_to=now + timedelta(hours=1),
        selected_operator_ids=[str(operator.id) for operator in operators],
        selected_category_ids=[],
        request_filters={},
        progress_percent=100,
        current_stage="Complete with some errors",
        created_at=created_at,
        updated_at=created_at,
    )
    session.add(job)
    await session.flush()
    return job


def _job_item(
    *,
    job: ProcessingJob,
    call: Call,
    operator: Operator,
    status: ItemStatus,
    updated_at: datetime,
    recording: Recording | None = None,
    result_transcript_id: UUID | None = None,
) -> ProcessingJobItem:
    return ProcessingJobItem(
        job_id=job.id,
        call_id=call.id,
        operator_id=operator.id,
        recording_id=recording.id if recording else None,
        result_transcript_id=result_transcript_id,
        idempotency_key=f"item-{uuid4()}",
        status=status,
        stage=status.value,
        created_at=updated_at,
        updated_at=updated_at,
        completed_at=updated_at,
    )


@pytest.mark.asyncio
async def test_results_status_is_latest_per_call_operator_and_dashboard_failures_are_distinct(
    harness: Harness,
) -> None:
    now = datetime.now(UTC)
    async with harness.sessions() as session:
        operator_a = await _operator(session, "Operator A", "101")
        operator_b = await _operator(session, "Operator B", "102")
        call = Call(
            yeastar_uid=f"call-{uuid4()}",
            started_at=now,
            caller_number="+302101234567",
            callee_number="101",
            direction=Direction.INBOUND,
            duration_seconds=90,
            has_recording=True,
            processing_status="stale-call-level-status",
        )
        session.add(call)
        await session.flush()
        session.add_all(
            [
                CallParticipant(
                    call_id=call.id,
                    operator_id=operator_a.id,
                    provider_extension_id=operator_a.yeastar_extension_id,
                    provider_extension_number=operator_a.extension_number,
                    role=ParticipantRole.ANSWERING_OPERATOR,
                    answered=True,
                    attribution_source=SpeakerSource.YEASTAR_EXTENSION,
                ),
                CallParticipant(
                    call_id=call.id,
                    operator_id=operator_b.id,
                    provider_extension_id=operator_b.yeastar_extension_id,
                    provider_extension_number=operator_b.extension_number,
                    role=ParticipantRole.TRANSFERRED_OPERATOR,
                    answered=True,
                    attribution_source=SpeakerSource.YEASTAR_EXTENSION,
                ),
            ]
        )

        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"recording-{uuid4()}",
            status=RecordingStatus.COMPLETED,
        )
        session.add(recording)
        await session.flush()
        transcript = Transcript(
            call_id=call.id,
            recording_id=recording.id,
            operator_id=None,
            idempotency_key=f"diarized-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe-diarize",
            language="el",
            completed_at=now,
            source_audio_sha256="a" * 64,
            is_diarized=True,
            original_text="κοινό εύρημα",
            normalized_text="κοινο ευρημα",
        )
        session.add(transcript)
        await session.flush()
        segment = TranscriptSegment(
            transcript_id=transcript.id,
            call_id=call.id,
            operator_id=None,
            speaker_label="speaker_0",
            speaker_source=SpeakerSource.OPENAI_DIARIZATION,
            start_seconds=Decimal("1.000"),
            end_seconds=Decimal("2.000"),
            original_text="κοινό εύρημα",
            normalized_text="κοινο ευρημα",
            transcription_model=transcript.model,
            sequence_number=1,
        )
        category = KeywordCategory(name=f"Category {uuid4()}", active=True)
        session.add_all([segment, category])
        await session.flush()
        keyword = Keyword(
            category_id=category.id,
            canonical_phrase="κοινό εύρημα",
            normalized_phrase="κοινο ευρημα",
            active=True,
        )
        session.add(keyword)
        await session.flush()
        session.add(
            KeywordMatch(
                keyword_id=keyword.id,
                operator_id=None,
                call_id=call.id,
                transcript_segment_id=segment.id,
                original_matched_text="κοινό εύρημα",
                normalized_match="κοινο ευρημα",
                context_before="",
                context_after="",
                start_seconds=Decimal("1.000"),
                end_seconds=Decimal("2.000"),
                match_method=MatchMethod.EXACT_PHRASE,
                match_score=Decimal("1.000"),
            )
        )

        old_job = await _job(
            session,
            user_id=harness.user_id,
            now=now,
            operators=[operator_a, operator_b],
            created_at=now - timedelta(minutes=10),
        )
        new_job = await _job(
            session,
            user_id=harness.user_id,
            now=now,
            operators=[operator_a, operator_b],
            created_at=now - timedelta(minutes=5),
        )
        # This fixture intentionally treats the unattributed diarized segment
        # as shared call context for both selected operators.
        old_job.include_all_speakers = True
        new_job.include_all_speakers = True
        session.add_all(
            [
                _job_item(
                    job=old_job,
                    call=call,
                    operator=operator_a,
                    status=ItemStatus.FAILED,
                    updated_at=now - timedelta(minutes=10),
                ),
                _job_item(
                    job=old_job,
                    call=call,
                    operator=operator_b,
                    status=ItemStatus.COMPLETED,
                    updated_at=now - timedelta(minutes=10),
                    result_transcript_id=transcript.id,
                ),
                _job_item(
                    job=new_job,
                    call=call,
                    operator=operator_a,
                    status=ItemStatus.COMPLETED,
                    updated_at=now - timedelta(minutes=5),
                    result_transcript_id=transcript.id,
                ),
                _job_item(
                    job=new_job,
                    call=call,
                    operator=operator_b,
                    status=ItemStatus.FAILED,
                    updated_at=now - timedelta(minutes=5),
                ),
            ]
        )
        await session.commit()

    response = await harness.client.get("/api/results", params={"has_matches": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 2
    by_operator = {item["operator_id"]: item for item in payload["items"]}
    assert by_operator[str(operator_a.id)]["processing_status"] == "completed"
    assert by_operator[str(operator_b.id)]["processing_status"] == "failed"
    # A diarized/unknown-speaker match is call context for both rows, not relabeled.
    assert by_operator[str(operator_a.id)]["keywords_found"] == ["κοινό εύρημα"]
    assert by_operator[str(operator_b.id)]["keywords_found"] == ["κοινό εύρημα"]
    assert by_operator[str(operator_a.id)]["match_count"] == 1
    assert by_operator[str(operator_b.id)]["match_count"] == 1

    dashboard = await harness.client.get("/api/dashboard")
    assert dashboard.status_code == 200, dashboard.text
    # There are two failed historical items, but they belong to one call.
    assert dashboard.json()["failed_calls"] == 1


@pytest.mark.asyncio
async def test_finalize_mixed_operator_call_counts_only_as_failed(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    async with harness.sessions() as session:
        operator_a = await _operator(session, "Operator A", "201")
        operator_b = await _operator(session, "Operator B", "202")
        call = Call(
            yeastar_uid=f"call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
            duration_seconds=30,
            processing_status="failed",
        )
        session.add(call)
        await session.flush()
        job = await _job(
            session,
            user_id=harness.user_id,
            now=now,
            operators=[operator_a, operator_b],
            created_at=now - timedelta(minutes=1),
        )
        job.status = JobStatus.SEARCHING_KEYWORDS
        completed_item = _job_item(
            job=job,
            call=call,
            operator=operator_a,
            status=ItemStatus.COMPLETED,
            updated_at=now,
        )
        failed_item = _job_item(
            job=job,
            call=call,
            operator=operator_b,
            status=ItemStatus.FAILED,
            updated_at=now,
        )
        session.add_all([completed_item, failed_item])
        await session.commit()
        job_id = job.id
        item_id = failed_item.id

    monkeypatch.setattr(pipeline, "AsyncSessionFactory", harness.sessions)
    await pipeline.finalize_job_for_item(item_id)

    async with harness.sessions() as session:
        finalized = await session.get(ProcessingJob, job_id)
        assert finalized is not None
        assert finalized.calls_completed == 0
        assert finalized.calls_failed == 1
        assert finalized.progress_percent == 100
        assert finalized.status == JobStatus.COMPLETED_WITH_ERRORS


class _Lock:
    def __init__(self) -> None:
        self.acquired = False

    async def acquire(self, *, blocking: bool = False) -> bool:
        self.acquired = True
        return True

    async def owned(self) -> bool:
        return self.acquired

    async def release(self) -> None:
        self.acquired = False

    async def extend(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


class _Redis:
    def __init__(self) -> None:
        self.locks: list[_Lock] = []

    def lock(self, *_args: Any, **_kwargs: Any) -> _Lock:
        lock = _Lock()
        self.locks.append(lock)
        return lock


@pytest.mark.asyncio
async def test_effective_configuration_redis_failure_waits_for_connection(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    async with harness.sessions() as session:
        operator = await _operator(session, "Operator A", "300")
        call = Call(
            yeastar_uid=f"call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
            duration_seconds=30,
            has_recording=True,
            processing_status="queued",
        )
        session.add(call)
        await session.flush()
        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"recording-{uuid4()}",
            yeastar_file_name="recording.wav",
            status=RecordingStatus.DISCOVERED,
        )
        session.add(recording)
        await session.flush()
        job = await _job(
            session,
            user_id=harness.user_id,
            now=now,
            operators=[operator],
            created_at=now,
        )
        job.status = JobStatus.QUEUED
        item = _job_item(
            job=job,
            call=call,
            operator=operator,
            recording=recording,
            status=ItemStatus.QUEUED,
            updated_at=now,
        )
        item.completed_at = None
        session.add(item)
        await session.commit()
        item_id = item.id
        job_id = job.id

    async def unavailable_configuration(*_args: object, **_kwargs: object) -> Settings:
        raise RedisError("configuration store unavailable")

    class CleanupOnlyAudioProcessor:
        def __init__(self, _settings: Settings) -> None:
            pass

        def remove_files(self, _paths: list[Path]) -> tuple[list[Path], list[Path]]:
            return [], []

    redis = _Redis()
    monkeypatch.setattr(pipeline, "AsyncSessionFactory", harness.sessions)
    monkeypatch.setattr(pipeline, "get_settings", lambda: harness.settings)
    monkeypatch.setattr(pipeline, "get_redis", lambda: redis)
    monkeypatch.setattr(
        pipeline,
        "load_effective_yeastar_settings",
        unavailable_configuration,
    )
    monkeypatch.setattr(pipeline, "AudioProcessor", CleanupOnlyAudioProcessor)

    await pipeline.process_item(item_id)

    async with harness.sessions() as session:
        waiting_item = await session.get(ProcessingJobItem, item_id)
        waiting_job = await session.get(ProcessingJob, job_id)
        assert waiting_item is not None
        assert waiting_job is not None
        assert waiting_item.status == ItemStatus.WAITING_FOR_CONNECTION
        assert waiting_item.stage == "waiting_for_connection"
        assert waiting_item.error_category == "configuration_state"
        assert waiting_item.error_message == "Test the phone-system connection in Settings."
        assert waiting_item.completed_at is None
        assert waiting_job.status == JobStatus.WAITING_FOR_CONNECTION
        assert waiting_job.current_stage == "Waiting for phone-system connection"
        assert waiting_job.completed_at is None
    assert len(redis.locks) == 1
    assert redis.locks[0].acquired is False


@pytest.mark.asyncio
async def test_completed_transcript_is_reused_before_audio_or_provider_access(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime.now(UTC)
    async with harness.sessions() as session:
        operator = await _operator(session, "Operator A", "301")
        call = Call(
            yeastar_uid=f"call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
            duration_seconds=45,
            has_recording=True,
            processing_status="failed",
        )
        session.add(call)
        await session.flush()
        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"recording-{uuid4()}",
            yeastar_file_name="must-not-download.wav",
            storage_key=None,
            status=RecordingStatus.COMPLETED,
        )
        session.add(recording)
        await session.flush()
        transcript = Transcript(
            call_id=call.id,
            recording_id=recording.id,
            operator_id=operator.id,
            idempotency_key=f"legacy-transcript-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="retired-transcription-model",
            language="en",
            prompt_version="legacy-vocabulary-prompt",
            completed_at=now - timedelta(days=1),
            source_audio_sha256="b" * 64,
            is_diarized=False,
            original_text="already paid for and transcribed",
            normalized_text="already paid for and transcribed",
        )
        session.add(transcript)
        job = await _job(
            session,
            user_id=harness.user_id,
            now=now,
            operators=[operator],
            created_at=now,
        )
        job.status = JobStatus.QUEUED
        item = _job_item(
            job=job,
            call=call,
            operator=operator,
            recording=recording,
            status=ItemStatus.QUEUED,
            updated_at=now,
        )
        session.add(item)
        await session.commit()
        item_id = item.id
        transcript_id = transcript.id

    searched_transcript_ids: list[UUID] = []
    cleanup_calls: list[list[Path]] = []

    async def record_keyword_search(
        _session: AsyncSession, existing: Transcript, _job: ProcessingJob
    ) -> int:
        searched_transcript_ids.append(existing.id)
        return 0

    async def forbidden_vocabulary(*_args: Any, **_kwargs: Any) -> list[str]:
        raise AssertionError("a completed transcript must not build a new provider prompt")

    class GuardAudioProcessor:
        def __init__(self, _settings: Settings) -> None:
            pass

        def remove_files(self, paths: list[Path]) -> tuple[list[Path], list[Path]]:
            cleanup_calls.append(list(paths))
            return [], []

        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"completed transcript unexpectedly accessed audio: {name}")

    class ForbiddenProviderClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("completed transcript unexpectedly accessed a provider")

    redis = _Redis()
    monkeypatch.setattr(pipeline, "AsyncSessionFactory", harness.sessions)
    monkeypatch.setattr(pipeline, "get_settings", lambda: harness.settings)
    monkeypatch.setattr(pipeline, "get_redis", lambda: redis)
    monkeypatch.setattr(pipeline, "search_and_persist_matches", record_keyword_search)
    monkeypatch.setattr(pipeline, "_vocabulary", forbidden_vocabulary)
    monkeypatch.setattr(pipeline, "AudioProcessor", GuardAudioProcessor)
    monkeypatch.setattr(pipeline, "YeastarClient", ForbiddenProviderClient)
    monkeypatch.setattr(pipeline, "OpenAITranscriptionClient", ForbiddenProviderClient)
    monkeypatch.setattr(pipeline, "TranscriptionOrchestrator", ForbiddenProviderClient)

    await pipeline.process_item(item_id)

    assert searched_transcript_ids == [transcript_id]
    assert cleanup_calls == [[]]
    assert len(redis.locks) == 1
    assert redis.locks[0].acquired is False
    async with harness.sessions() as session:
        completed_item = await session.get(ProcessingJobItem, item_id)
        assert completed_item is not None
        assert completed_item.status == ItemStatus.COMPLETED
        assert completed_item.attempt_count == 1
        assert (await session.scalar(select(func.count()).select_from(Transcript)) or 0) == 1
