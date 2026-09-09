from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.api.results as results_api
from app.api.dependencies import get_effective_yeastar_settings
from app.auth.dependencies import get_current_user
from app.core.config import Settings
from app.database.base import Base
from app.database.session import get_db
from app.models import (
    Call,
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
    RecordingStatus,
    SpeakerSource,
    TranscriptionMode,
    TranscriptStatus,
)


@dataclass
class Harness:
    client: AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    user_id: UUID


@pytest.fixture
async def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    database_path = (tmp_path / "results-speaker-scope.sqlite3").as_posix()
    settings = Settings(
        APP_ENV="test",
        APP_SECRET_KEY="results-speaker-scope-secret-with-32-characters",
        APP_TIMEZONE="Europe/Athens",
        STORAGE_ROOT=tmp_path / "storage",
        DATABASE_URL=f"sqlite+aiosqlite:///{database_path}",
    )
    settings.STORAGE_ROOT.mkdir()
    engine = create_async_engine(settings.DATABASE_URL)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    async with sessions() as session:
        user = User(
            email=f"speaker-scope-{uuid4()}@example.test",
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

    async def override_effective_yeastar_settings() -> Settings:
        return settings

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_effective_yeastar_settings] = override_effective_yeastar_settings
    monkeypatch.setattr(results_api, "get_settings", lambda: settings)

    client = AsyncClient(
        transport=ASGITransport(app=app),
        base_url="https://testserver",
    )
    try:
        yield Harness(client=client, sessions=sessions, user_id=user_id)
    finally:
        await client.aclose()
        await engine.dispose()


@pytest.mark.asyncio
async def test_result_and_export_speaker_scope_follow_analysis_job(harness: Harness) -> None:
    now = datetime(2026, 7, 28, 10, 0, tzinfo=UTC)
    async with harness.sessions() as session:
        operator = Operator(
            yeastar_extension_id=f"extension-{uuid4()}",
            extension_number="101",
            display_name="Scoped Operator",
            enabled=True,
            last_synced_at=now,
        )
        call = Call(
            yeastar_uid=f"call-{uuid4()}",
            started_at=now,
            caller_number="+302101234567",
            callee_number="101",
            direction=Direction.INBOUND,
            duration_seconds=45,
            has_recording=True,
            processing_status="completed",
        )
        session.add_all([operator, call])
        await session.flush()
        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"recording-{uuid4()}",
            status=RecordingStatus.COMPLETED,
        )
        category = KeywordCategory(name=f"Scope {uuid4()}", active=True)
        session.add_all([recording, category])
        await session.flush()
        attributed_keyword = Keyword(
            category_id=category.id,
            canonical_phrase="Attributed Keyword",
            normalized_phrase="attributed keyword",
            active=True,
        )
        unattributed_keyword = Keyword(
            category_id=category.id,
            canonical_phrase="Unattributed Keyword",
            normalized_phrase="unattributed keyword",
            active=True,
        )
        transcript = Transcript(
            call_id=call.id,
            recording_id=recording.id,
            operator_id=None,
            idempotency_key=f"transcript-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe",
            language="en",
            completed_at=now,
            source_audio_sha256="a" * 64,
            is_diarized=False,
            original_text="attributed needle unattributed needle",
            normalized_text="attributed needle unattributed needle",
        )
        session.add_all([attributed_keyword, unattributed_keyword, transcript])
        await session.flush()
        attributed_segment = TranscriptSegment(
            transcript_id=transcript.id,
            call_id=call.id,
            operator_id=operator.id,
            speaker_label=operator.display_name,
            speaker_source=SpeakerSource.YEASTAR_EXTENSION,
            start_seconds=Decimal("1.000"),
            end_seconds=Decimal("2.000"),
            original_text="Attributed needle",
            normalized_text="attributed needle",
            transcription_model=transcript.model,
            sequence_number=1,
        )
        unattributed_segment = TranscriptSegment(
            transcript_id=transcript.id,
            call_id=call.id,
            operator_id=None,
            speaker_label="Channel A",
            speaker_source=SpeakerSource.UNKNOWN,
            start_seconds=Decimal("2.000"),
            end_seconds=Decimal("3.000"),
            original_text="Unattributed needle",
            normalized_text="unattributed needle",
            transcription_model=transcript.model,
            sequence_number=2,
        )
        session.add_all([attributed_segment, unattributed_segment])
        await session.flush()
        session.add_all(
            [
                KeywordMatch(
                    keyword_id=attributed_keyword.id,
                    operator_id=operator.id,
                    call_id=call.id,
                    transcript_segment_id=attributed_segment.id,
                    original_matched_text="Attributed Keyword",
                    normalized_match="attributed keyword",
                    context_before="",
                    context_after="",
                    start_seconds=Decimal("1.000"),
                    end_seconds=Decimal("2.000"),
                    match_method=MatchMethod.EXACT_PHRASE,
                    match_score=Decimal("1.000"),
                ),
                KeywordMatch(
                    keyword_id=unattributed_keyword.id,
                    operator_id=None,
                    call_id=call.id,
                    transcript_segment_id=unattributed_segment.id,
                    original_matched_text="Unattributed Keyword",
                    normalized_match="unattributed keyword",
                    context_before="",
                    context_after="",
                    start_seconds=Decimal("2.000"),
                    end_seconds=Decimal("3.000"),
                    match_method=MatchMethod.EXACT_PHRASE,
                    match_score=Decimal("1.000"),
                ),
            ]
        )

        jobs: list[ProcessingJob] = []
        for include_all_speakers in (False, True):
            job = ProcessingJob(
                idempotency_key=f"scope-job-{include_all_speakers}-{uuid4()}",
                requested_by_id=harness.user_id,
                status=JobStatus.COMPLETED,
                date_from=now - timedelta(hours=1),
                date_to=now + timedelta(hours=1),
                include_all_speakers=include_all_speakers,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={},
                progress_percent=100,
                current_stage="Complete",
                calls_found=1,
                recordings_found=1,
                calls_completed=1,
                completed_at=now,
            )
            session.add(job)
            await session.flush()
            session.add(
                ProcessingJobItem(
                    job_id=job.id,
                    call_id=call.id,
                    operator_id=operator.id,
                    recording_id=recording.id,
                    result_transcript_id=transcript.id,
                    idempotency_key=f"scope-item-{uuid4()}",
                    status=ItemStatus.COMPLETED,
                    stage="completed",
                    completed_at=now,
                )
            )
            jobs.append(job)
        await session.commit()
        operator_only_job_id = jobs[0].id
        all_speakers_job_id = jobs[1].id
        unattributed_keyword_id = unattributed_keyword.id

    operator_only = await harness.client.get(
        "/api/results",
        params={"job_id": str(operator_only_job_id), "sort": "match_count"},
    )
    assert operator_only.status_code == 200, operator_only.text
    assert operator_only.json()["items"][0]["match_count"] == 1
    assert operator_only.json()["items"][0]["keywords_found"] == ["Attributed Keyword"]

    operator_only_keyword = await harness.client.get(
        "/api/results",
        params={
            "job_id": str(operator_only_job_id),
            "keyword_id": str(unattributed_keyword_id),
        },
    )
    operator_only_transcript = await harness.client.get(
        "/api/results",
        params={
            "job_id": str(operator_only_job_id),
            "transcript_query": "unattributed needle",
        },
    )
    assert operator_only_keyword.json()["total"] == 0
    assert operator_only_transcript.json()["total"] == 0

    all_speakers = await harness.client.get(
        "/api/results",
        params={"job_id": str(all_speakers_job_id), "sort": "match_count"},
    )
    all_speakers_keyword = await harness.client.get(
        "/api/results",
        params={
            "job_id": str(all_speakers_job_id),
            "keyword_id": str(unattributed_keyword_id),
        },
    )
    all_speakers_transcript = await harness.client.get(
        "/api/results",
        params={
            "job_id": str(all_speakers_job_id),
            "transcript_query": "unattributed needle",
        },
    )
    assert all_speakers.status_code == 200, all_speakers.text
    assert all_speakers.json()["items"][0]["match_count"] == 2
    assert all_speakers.json()["items"][0]["keywords_found"] == [
        "Attributed Keyword",
        "Unattributed Keyword",
    ]
    assert all_speakers_keyword.json()["total"] == 1
    assert all_speakers_transcript.json()["total"] == 1

    operator_only_export = await harness.client.get(
        "/api/results/export.csv",
        params={"job_id": str(operator_only_job_id), "sort": "match_count"},
    )
    all_speakers_export = await harness.client.get(
        "/api/results/export.csv",
        params={"job_id": str(all_speakers_job_id), "sort": "match_count"},
    )
    assert operator_only_export.status_code == 200, operator_only_export.text
    assert all_speakers_export.status_code == 200, all_speakers_export.text
    operator_only_rows = list(
        csv.reader(io.StringIO(operator_only_export.content.decode("utf-8-sig")))
    )
    all_speakers_rows = list(
        csv.reader(io.StringIO(all_speakers_export.content.decode("utf-8-sig")))
    )
    assert [row[7] for row in operator_only_rows[1:]] == ["Attributed Keyword"]
    assert [row[7] for row in all_speakers_rows[1:]] == [
        "Attributed Keyword",
        "Unattributed Keyword",
    ]


@pytest.mark.asyncio
async def test_call_detail_exposes_bound_quality_and_segment_evidence(
    harness: Harness,
) -> None:
    now = datetime(2026, 7, 29, 9, 0, tzinfo=UTC)
    async with harness.sessions() as session:
        operator = Operator(
            yeastar_extension_id=f"quality-extension-{uuid4()}",
            extension_number="102",
            display_name="Quality Scope Operator",
            enabled=True,
            last_synced_at=now,
        )
        call = Call(
            yeastar_uid=f"quality-call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
            duration_seconds=60,
            has_recording=True,
            processing_status="completed",
        )
        session.add_all([operator, call])
        await session.flush()
        recording_with_segments = Recording(
            call_id=call.id,
            yeastar_recording_id=f"quality-recording-{uuid4()}",
            status=RecordingStatus.COMPLETED,
        )
        recording_without_segments = Recording(
            call_id=call.id,
            yeastar_recording_id=f"empty-quality-recording-{uuid4()}",
            status=RecordingStatus.COMPLETED,
        )
        session.add_all([recording_with_segments, recording_without_segments])
        await session.flush()
        current = Transcript(
            call_id=call.id,
            recording_id=recording_with_segments.id,
            operator_id=None,
            idempotency_key=f"current-quality-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe",
            language="el",
            completed_at=now,
            source_audio_sha256="a" * 64,
            is_diarized=False,
            transcription_mode=TranscriptionMode.DUAL_CHANNEL,
            quality_summary={
                "degraded": False,
                "fallback_duration_ratio": 0.0,
            },
            is_current=True,
        )
        current_without_segments = Transcript(
            call_id=call.id,
            recording_id=recording_without_segments.id,
            operator_id=None,
            idempotency_key=f"empty-current-quality-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="gpt-4o-transcribe",
            language="el",
            completed_at=now,
            source_audio_sha256="b" * 64,
            is_diarized=True,
            transcription_mode=TranscriptionMode.MONO_DIARIZATION,
            quality_summary={
                "degraded": True,
                "fallback_duration_ratio": 0.25,
                "warning": "Refinement fallback was used.",
            },
            is_current=True,
        )
        historical = Transcript(
            call_id=call.id,
            recording_id=recording_with_segments.id,
            operator_id=None,
            idempotency_key=f"historical-quality-{uuid4()}",
            status=TranscriptStatus.COMPLETED,
            model="legacy-model",
            language="el",
            completed_at=now - timedelta(days=1),
            source_audio_sha256="c" * 64,
            is_diarized=True,
            transcription_mode=TranscriptionMode.LEGACY,
            quality_summary={"source": "historical-job"},
            is_current=False,
        )
        session.add_all([current, current_without_segments, historical])
        await session.flush()
        current_segment = TranscriptSegment(
            transcript_id=current.id,
            call_id=call.id,
            operator_id=None,
            speaker_label="Channel A",
            speaker_source=SpeakerSource.STEREO_CHANNEL,
            start_seconds=Decimal("1.000"),
            end_seconds=Decimal("2.000"),
            original_text="Current refined text",
            normalized_text="current refined text",
            confidence=None,
            transcription_model="gpt-4o-transcribe",
            sequence_number=1,
            mean_logprob=Decimal("-0.25000"),
            low_logprob_ratio=Decimal("0.05000"),
            quality_flags=["normalized_retry_used"],
            audio_variant="v2-light-normalized-telephone-v1",
        )
        historical_segment = TranscriptSegment(
            transcript_id=historical.id,
            call_id=call.id,
            operator_id=None,
            speaker_label="chunk-1:A",
            speaker_source=SpeakerSource.OPENAI_DIARIZATION,
            start_seconds=Decimal("3.000"),
            end_seconds=Decimal("4.000"),
            original_text="Historical text",
            normalized_text="historical text",
            confidence=None,
            transcription_model="legacy-model",
            sequence_number=1,
            quality_flags=[],
            audio_variant="legacy-mono",
        )
        job = ProcessingJob(
            idempotency_key=f"historical-quality-job-{uuid4()}",
            requested_by_id=harness.user_id,
            status=JobStatus.COMPLETED,
            date_from=now - timedelta(days=2),
            date_to=now,
            include_all_speakers=True,
            selected_operator_ids=[str(operator.id)],
            selected_category_ids=[],
            request_filters={},
            progress_percent=100,
            current_stage="Complete",
            calls_found=1,
            recordings_found=1,
            calls_completed=1,
            completed_at=now,
        )
        session.add_all([current_segment, historical_segment, job])
        await session.flush()
        session.add(
            ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording_with_segments.id,
                result_transcript_id=historical.id,
                idempotency_key=f"historical-quality-item-{uuid4()}",
                status=ItemStatus.COMPLETED,
                stage="completed",
                completed_at=now,
            )
        )
        await session.commit()
        call_id = call.id
        current_id = current.id
        empty_id = current_without_segments.id
        current_segment_id = current_segment.id
        historical_id = historical.id
        historical_segment_id = historical_segment.id
        job_id = job.id

    current_response = await harness.client.get(f"/api/calls/{call_id}")
    assert current_response.status_code == 200, current_response.text
    current_body = current_response.json()
    assert [item["id"] for item in current_body["transcript_segments"]] == [str(current_segment_id)]
    current_segment_body = current_body["transcript_segments"][0]
    assert current_segment_body["transcription_model"] == "gpt-4o-transcribe"
    assert Decimal(current_segment_body["mean_logprob"]) == Decimal("-0.25000")
    assert Decimal(current_segment_body["low_logprob_ratio"]) == Decimal("0.05000")
    assert current_segment_body["quality_flags"] == ["normalized_retry_used"]
    assert current_segment_body["audio_variant"] == "v2-light-normalized-telephone-v1"
    current_quality = {
        item["transcript_id"]: item for item in current_body["transcript_quality_summaries"]
    }
    assert set(current_quality) == {str(current_id), str(empty_id)}
    assert current_quality[str(current_id)] == {
        "transcript_id": str(current_id),
        "transcription_mode": "dual_channel",
        "quality_summary": {
            "degraded": False,
            "fallback_duration_ratio": 0.0,
        },
    }
    assert current_quality[str(empty_id)]["transcription_mode"] == "mono_diarization"
    assert current_quality[str(empty_id)]["quality_summary"]["degraded"] is True

    historical_response = await harness.client.get(
        f"/api/calls/{call_id}",
        params={"job_id": str(job_id)},
    )
    assert historical_response.status_code == 200, historical_response.text
    historical_body = historical_response.json()
    assert [item["id"] for item in historical_body["transcript_segments"]] == [
        str(historical_segment_id)
    ]
    assert historical_body["transcript_quality_summaries"] == [
        {
            "transcript_id": str(historical_id),
            "transcription_mode": "legacy",
            "quality_summary": {"source": "historical-job"},
        }
    ]
