from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from app.database.base import Base
from app.models.entities import (
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
    TranscriptionAttempt,
    User,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    MatchMethod,
    RecordingStatus,
    Severity,
    SpeakerSource,
    TranscriptStatus,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("load_dependents", [False, True])
async def test_transcript_delete_preserves_set_null_references_and_cascades_dependents(
    load_dependents: bool,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    try:
        async with sessions() as session:
            user = User(
                email=f"retention-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"retention-{uuid4()}",
                extension_number="1010",
                display_name="Retention Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"retention-call-{uuid4()}",
                started_at=now - timedelta(hours=1),
                direction=Direction.INBOUND,
                duration_seconds=60,
                has_recording=True,
            )
            category = KeywordCategory(name=f"Retention category {uuid4()}")
            session.add_all([user, operator, call, category])
            await session.flush()

            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"retention-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            keyword = Keyword(
                category_id=category.id,
                canonical_phrase="refund",
                normalized_phrase="refund",
                severity=Severity.MEDIUM,
            )
            job = ProcessingJob(
                idempotency_key=f"retention-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.COMPLETED,
                date_from=call.started_at,
                date_to=call.started_at + timedelta(hours=1),
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[str(category.id)],
                completed_at=now,
            )
            session.add_all([recording, keyword, job])
            await session.flush()

            source = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                idempotency_key=f"retention-source-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                completed_at=now - timedelta(days=100),
                source_audio_sha256="a" * 64,
                is_diarized=False,
                is_current=True,
            )
            session.add(source)
            await session.flush()

            child = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                idempotency_key=f"retention-child-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                completed_at=now,
                source_audio_sha256="a" * 64,
                is_diarized=False,
                supersedes_transcript_id=source.id,
                is_current=False,
            )
            segment = TranscriptSegment(
                transcript_id=source.id,
                call_id=call.id,
                operator_id=operator.id,
                speaker_label=operator.display_name,
                speaker_source=SpeakerSource.STEREO_CHANNEL,
                start_seconds=Decimal("1.000"),
                end_seconds=Decimal("2.000"),
                original_text="refund",
                normalized_text="refund",
                transcription_model="gpt-4o-transcribe",
                sequence_number=1,
            )
            attempt = TranscriptionAttempt(
                transcript_id=source.id,
                track_id="legacy-operator",
                chunk_index=0,
                start_seconds=Decimal("0.000"),
                end_seconds=Decimal("15.000"),
                model="gpt-4o-transcribe",
                audio_variant="legacy-operator-channel",
                prompt_hash="b" * 64,
                response_text="refund",
                selected=True,
                completed_at=now,
            )
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                result_transcript_id=source.id,
                idempotency_key=f"retention-item-{uuid4()}",
                status=ItemStatus.COMPLETED,
                stage="completed",
                completed_at=now,
            )
            session.add_all([child, segment, attempt, item])
            await session.flush()

            match = KeywordMatch(
                keyword_id=keyword.id,
                operator_id=operator.id,
                call_id=call.id,
                transcript_segment_id=segment.id,
                original_matched_text="refund",
                normalized_match="refund",
                context_before="",
                context_after="",
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                match_method=MatchMethod.EXACT_PHRASE,
                match_score=Decimal("100.000"),
            )
            session.add(match)
            await session.commit()

            source_id = source.id
            child_id = child.id
            segment_id = segment.id
            attempt_id = attempt.id
            match_id = match.id
            item_id = item.id

        async with sessions() as session:
            statement = select(Transcript).where(Transcript.id == source_id)
            if load_dependents:
                statement = statement.options(
                    selectinload(Transcript.segments).selectinload(TranscriptSegment.matches),
                    selectinload(Transcript.attempts),
                )
            source = await session.scalar(statement)
            assert source is not None
            await session.delete(source)
            await session.commit()

        async with sessions() as session:
            assert await session.get(Transcript, source_id) is None
            assert await session.get(TranscriptSegment, segment_id) is None
            assert await session.get(TranscriptionAttempt, attempt_id) is None
            assert await session.get(KeywordMatch, match_id) is None

            surviving_child = await session.get(Transcript, child_id)
            surviving_item = await session.get(ProcessingJobItem, item_id)
            assert surviving_child is not None
            assert surviving_child.supersedes_transcript_id is None
            assert surviving_item is not None
            assert surviving_item.result_transcript_id is None

            assert await session.scalar(select(func.count()).select_from(Keyword)) == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_type", ["recording", "call"])
async def test_loaded_parent_delete_uses_database_transcript_cascade(
    parent_type: str,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def enable_sqlite_foreign_keys(dbapi_connection: object, _record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    try:
        async with sessions() as session:
            call = Call(
                yeastar_uid=f"parent-cascade-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
            )
            session.add(call)
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"parent-cascade-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            session.add(recording)
            await session.flush()
            transcript = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=None,
                idempotency_key=f"parent-cascade-transcript-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe-diarize",
                language="el",
                completed_at=now,
                source_audio_sha256="c" * 64,
                is_diarized=True,
                is_current=True,
            )
            session.add(transcript)
            await session.commit()
            call_id = call.id
            recording_id = recording.id
            transcript_id = transcript.id

        async with sessions() as session:
            if parent_type == "recording":
                parent = await session.scalar(
                    select(Recording)
                    .where(Recording.id == recording_id)
                    .options(selectinload(Recording.transcripts))
                )
            else:
                parent = await session.scalar(
                    select(Call)
                    .where(Call.id == call_id)
                    .options(
                        selectinload(Call.transcripts),
                        selectinload(Call.recordings).selectinload(Recording.transcripts),
                    )
                )
            assert parent is not None
            await session.delete(parent)
            await session.commit()

        async with sessions() as session:
            assert await session.get(Transcript, transcript_id) is None
            if parent_type == "call":
                assert await session.get(Recording, recording_id) is None
    finally:
        await engine.dispose()
