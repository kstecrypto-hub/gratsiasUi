from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.workers.pipeline as pipeline_module
import app.workers.tasks as worker_tasks
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
    TranscriptionAttempt,
    User,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    RecordingStatus,
    RunStatus,
    SpeakerSource,
    SpeakerAttributionStatus,
    SyncType,
    TranscriptionMode,
    TranscriptStatus,
)
from app.services.audio import AudioInfo
from app.services.keyword_matching.normalization import normalize_greek
from app.services.transcription.types import (
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    TrackTranscriptionResult,
    TranscriptionAttemptEvidence,
)
from app.services.yeastar.cdr import CDRSummary
from app.workers.pipeline import (
    DiscoveryResult,
    LEGACY_DIARIZED_PIPELINE_CONFIG_HASH,
    LEGACY_ISOLATED_PIPELINE_CONFIG_HASH,
    LEGACY_PIPELINE_VERSION,
    ReprocessStateError,
    _activate_transcript_replacement,
    _fetch_cdrs,
    _record_reprocess_replacement,
    _recording_for_participant,
    _recordings_for_participant,
    _settle_stale_item,
    _should_mark_transcript_failed,
    search_and_persist_matches,
    transcript_idempotency_key,
)
from app.workers.tasks import (
    _busy_retry_delay,
    _recording_assignment_retry_delay,
    process_analysis_job,
    process_job_item,
)


def test_transcript_idempotency_key_is_deterministic_and_scoped() -> None:
    recording_id = UUID("00000000-0000-0000-0000-000000000001")
    other_recording_id = UUID("00000000-0000-0000-0000-000000000002")
    operator_id = UUID("00000000-0000-0000-0000-000000000101")
    checksum = "a" * 64

    key = transcript_idempotency_key(
        recording_id, operator_id, "gpt-4o-transcribe", checksum, False
    )
    assert key == "a39549975e2755f060d9c53678214ffa9e73e8aab6082e96eb67fe48a001f5be"
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


def test_versioned_transcript_identity_covers_pipeline_configuration() -> None:
    assert LEGACY_DIARIZED_PIPELINE_CONFIG_HASH == (
        "ef5ef358c56c2900297a8233a323a2b295faf08dd8c98dc621ae133560e25e61"
    )
    assert LEGACY_ISOLATED_PIPELINE_CONFIG_HASH == (
        "ad088e9537186db18c69c6781b79ae9e020192cc158790a4ab7e498a94bff2a5"
    )
    recording_id = UUID("00000000-0000-0000-0000-000000000001")
    operator_id = UUID("00000000-0000-0000-0000-000000000101")
    target_id = UUID("00000000-0000-0000-0000-000000000201")
    base = {
        "recording_id": recording_id,
        "operator_id": operator_id,
        "model": "gpt-4o-transcribe",
        "checksum": "a" * 64,
        "diarized": False,
        "language": "el",
        "prompt_version": "prompt-hash",
        "pipeline_version": LEGACY_PIPELINE_VERSION,
        "pipeline_config_hash": LEGACY_ISOLATED_PIPELINE_CONFIG_HASH,
        "transcription_mode": TranscriptionMode.LEGACY,
        "prompt_template_version": "legacy-isolated-vocabulary-v1",
        "vocabulary_hash": "vocabulary-hash",
        "supersedes_transcript_id": target_id,
    }
    key = transcript_idempotency_key(**base)
    assert key == transcript_idempotency_key(**base)

    variants = {
        "recording_id": UUID("00000000-0000-0000-0000-000000000002"),
        "operator_id": UUID("00000000-0000-0000-0000-000000000102"),
        "model": "gpt-4o-transcribe-next",
        "checksum": "b" * 64,
        "diarized": True,
        "language": "en",
        "prompt_version": "other-prompt-hash",
        "pipeline_version": "legacy-v2",
        "pipeline_config_hash": "b" * 64,
        "transcription_mode": TranscriptionMode.DUAL_CHANNEL,
        "prompt_template_version": "other-template",
        "vocabulary_hash": "other-vocabulary",
        "supersedes_transcript_id": UUID("00000000-0000-0000-0000-000000000202"),
    }
    for field, value in variants.items():
        changed = dict(base)
        changed[field] = value
        assert transcript_idempotency_key(**changed) != key, field


async def _replacement_state(
    *, replacement_status: TranscriptStatus
) -> tuple[bool, bool, UUID | None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    try:
        async with sessions() as session:
            operator = Operator(
                yeastar_extension_id=f"replacement-{uuid4()}",
                extension_number="1010",
                display_name="Replacement Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"replacement-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
            )
            session.add_all([operator, call])
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"replacement-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            session.add(recording)
            await session.flush()
            previous = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                idempotency_key=f"replacement-previous-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                source_audio_sha256="a" * 64,
                is_diarized=False,
                transcription_mode=TranscriptionMode.LEGACY,
                speaker_attribution_status=SpeakerAttributionStatus.CONFIRMED_BY_PBX,
                pipeline_version=LEGACY_PIPELINE_VERSION,
                pipeline_config_hash=LEGACY_ISOLATED_PIPELINE_CONFIG_HASH,
                is_current=True,
                completed_at=now,
            )
            replacement = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                idempotency_key=f"replacement-new-{uuid4()}",
                status=replacement_status,
                model="gpt-4o-transcribe",
                language="el",
                source_audio_sha256="a" * 64,
                is_diarized=False,
                transcription_mode=TranscriptionMode.LEGACY,
                speaker_attribution_status=SpeakerAttributionStatus.CONFIRMED_BY_PBX,
                pipeline_version=LEGACY_PIPELINE_VERSION,
                pipeline_config_hash=LEGACY_ISOLATED_PIPELINE_CONFIG_HASH,
                is_current=False,
                completed_at=now if replacement_status == TranscriptStatus.COMPLETED else None,
            )
            session.add_all([previous, replacement])
            await session.commit()
            previous_id = previous.id
            replacement_id = replacement.id

            if replacement_status == TranscriptStatus.COMPLETED:
                await _activate_transcript_replacement(session, replacement, previous_id)
                await session.commit()
            else:
                with pytest.raises(ReprocessStateError):
                    await _activate_transcript_replacement(session, replacement, previous_id)
                await session.rollback()

            current_previous = await session.get(Transcript, previous_id)
            current_replacement = await session.get(Transcript, replacement_id)
            assert current_previous is not None and current_replacement is not None
            return (
                current_previous.is_current,
                current_replacement.is_current,
                current_replacement.supersedes_transcript_id,
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_successful_replacement_swaps_current_transcript_atomically() -> None:
    previous_current, replacement_current, supersedes_id = await _replacement_state(
        replacement_status=TranscriptStatus.COMPLETED
    )
    assert previous_current is False
    assert replacement_current is True
    assert supersedes_id is not None


@pytest.mark.asyncio
async def test_failed_replacement_preserves_previous_current_transcript() -> None:
    previous_current, replacement_current, supersedes_id = await _replacement_state(
        replacement_status=TranscriptStatus.FAILED
    )
    assert previous_current is True
    assert replacement_current is False
    assert supersedes_id is None


@pytest.mark.asyncio
async def test_cancelled_settlement_atomically_persists_partial_provider_evidence() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    try:
        async with sessions() as session:
            user = User(
                email=f"partial-cancel-{uuid4()}@example.test",
                password_hash="unused",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"partial-cancel-{uuid4()}",
                extension_number="1012",
                display_name="Partial Cancel Operator",
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"partial-cancel-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
            )
            session.add_all([user, operator, call])
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"partial-cancel-recording-{uuid4()}",
                status=RecordingStatus.INSPECTED,
            )
            job = ProcessingJob(
                idempotency_key=f"partial-cancel-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.TRANSCRIBING,
                date_from=now - timedelta(days=1),
                date_to=now,
                cancellation_requested=True,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={},
            )
            session.add_all([recording, job])
            await session.flush()
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                idempotency_key=f"partial-cancel-item-{uuid4()}",
                status=ItemStatus.PROCESSING,
                stage="transcribing",
            )
            transcript = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                idempotency_key=f"partial-cancel-transcript-{uuid4()}",
                status=TranscriptStatus.PROCESSING,
                model="gpt-4o-transcribe",
                language="el",
                source_audio_sha256="c" * 64,
                is_diarized=False,
                is_current=True,
                transcription_mode=TranscriptionMode.OPERATOR_CHANNEL,
            )
            session.add_all([item, transcript])
            await session.commit()

            disposition = await _settle_stale_item(
                session,
                item,
                transcript.id,
                (
                    TranscriptionAttemptEvidence(
                        track_id="operator-channel",
                        chunk_index=0,
                        start_seconds=0.0,
                        end_seconds=10.0,
                        model="gpt-4o-transcribe",
                        audio_variant="v2-raw-lossless-pcm16-v1",
                        prompt_hash="a" * 64,
                        response_text="completed before cancellation",
                        mean_logprob=-1.2,
                        low_logprob_ratio=0.5,
                        selected=True,
                        api_usage={"input_tokens": 4},
                        completed_at=now,
                    ),
                ),
            )

            assert disposition == "cancelled"
            settled_item = await session.get(ProcessingJobItem, item.id)
            settled_transcript = await session.get(Transcript, transcript.id)
            attempts = (
                await session.scalars(
                    select(TranscriptionAttempt).where(
                        TranscriptionAttempt.transcript_id == transcript.id
                    )
                )
            ).all()
            assert settled_item is not None
            assert settled_item.status == ItemStatus.CANCELLED
            assert settled_transcript is not None
            assert settled_transcript.status == TranscriptStatus.FAILED
            assert [attempt.response_text for attempt in attempts] == [
                "completed before cancellation"
            ]
            assert attempts[0].selected is True
    finally:
        await engine.dispose()


def test_reprocess_failure_only_mutates_the_item_owned_replacement() -> None:
    item = ProcessingJobItem(
        id=uuid4(),
        requested_pipeline_version=LEGACY_PIPELINE_VERSION,
    )
    job = ProcessingJob(id=uuid4(), request_filters={})
    owned = Transcript(
        id=uuid4(),
        status=TranscriptStatus.COMPLETED,
        is_current=False,
    )
    historical = Transcript(
        id=uuid4(),
        status=TranscriptStatus.COMPLETED,
        is_current=False,
    )

    assert _should_mark_transcript_failed(job, item, owned) is False
    _record_reprocess_replacement(job, item, owned)
    assert _should_mark_transcript_failed(job, item, owned) is False
    assert _should_mark_transcript_failed(job, item, historical) is False
    owned.status = TranscriptStatus.PROCESSING
    assert _should_mark_transcript_failed(job, item, owned) is True
    owned.is_current = True
    assert _should_mark_transcript_failed(job, item, owned) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("lease_extended", [True, False], ids=["owned", "lost"])
async def test_completed_reuse_is_fenced_before_binding_the_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    lease_extended: bool,
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

        async def extend(self, *_args: object, **_kwargs: object) -> bool:
            return lease_extended

        async def release(self) -> None:
            return None

    class FakeRedis:
        def lock(self, *_args: object, **_kwargs: object) -> FakeLock:
            return FakeLock()

    try:
        async with sessions() as session:
            user = User(
                email=f"reuse-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"reuse-{uuid4()}",
                extension_number="1011",
                display_name="Reuse Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"reuse-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
            )
            session.add_all([user, operator, call])
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"reuse-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            job = ProcessingJob(
                idempotency_key=f"reuse-job-{uuid4()}",
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
                idempotency_key=f"reuse-transcript-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                source_audio_sha256="e" * 64,
                is_diarized=False,
                is_current=True,
                completed_at=now,
            )
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                idempotency_key=f"reuse-item-{uuid4()}",
                status=ItemStatus.QUEUED,
                stage="queued",
            )
            session.add_all([transcript, item])
            await session.flush()
            persisted_attempt = TranscriptionAttempt(
                transcript_id=transcript.id,
                track_id="operator-channel",
                chunk_index=0,
                start_seconds=Decimal("0"),
                end_seconds=Decimal("10"),
                model="gpt-4o-transcribe",
                audio_variant="v2-raw-lossless-pcm16-v1",
                prompt_hash="a" * 64,
                response_text="persisted once",
                mean_logprob=Decimal("-0.2"),
                low_logprob_ratio=Decimal("0"),
                selected=True,
                api_usage={"input_tokens": 3},
                completed_at=now,
            )
            session.add(persisted_attempt)
            await session.commit()
            item_id = item.id
            transcript_id = transcript.id
            attempt_id = persisted_attempt.id

        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        monkeypatch.setattr(pipeline_module, "get_redis", lambda: FakeRedis())
        monkeypatch.setattr(
            pipeline_module,
            "get_settings",
            lambda: SimpleNamespace(STORAGE_ROOT=tmp_path),
        )

        if lease_extended:
            await pipeline_module.process_item(item_id)
        else:
            with pytest.raises(pipeline_module.ProcessingLeaseLostError):
                await pipeline_module.process_item(item_id)

        async with sessions() as session:
            completed_item = await session.get(ProcessingJobItem, item_id)
            assert completed_item is not None
            assert completed_item.status == (
                ItemStatus.COMPLETED if lease_extended else ItemStatus.PROCESSING
            )
            assert completed_item.result_transcript_id == (
                transcript_id if lease_extended else None
            )
            attempts = (
                await session.scalars(
                    select(TranscriptionAttempt).where(
                        TranscriptionAttempt.transcript_id == transcript_id
                    )
                )
            ).all()
            assert [attempt.id for attempt in attempts] == [attempt_id]
            assert attempts[0].response_text == "persisted once"
            assert attempts[0].selected is True
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["failure", "cancellation"])
async def test_process_item_failure_or_cancellation_keeps_previous_reprocess_current(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    checksum = "f" * 64

    class FakeLock:
        async def acquire(self, *, blocking: bool = False) -> bool:
            return True

        async def owned(self) -> bool:
            return True

        async def extend(self, *_args: object, **_kwargs: object) -> None:
            return None

        async def release(self) -> None:
            return None

    class FakeRedis:
        def lock(self, *_args: object, **_kwargs: object) -> FakeLock:
            return FakeLock()

    class FakeAudioProcessor:
        def __init__(self, settings: object) -> None:
            self.root = settings.STORAGE_ROOT

        def safe_storage_path(self, relative_key: str):
            return self.root / relative_key

        async def inspect(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                codec_name="pcm_s16le",
                duration_seconds=30.0,
                channel_count=1,
                sample_rate_hz=16_000,
                bit_rate_bps=256_000,
                size_bytes=4,
                sha256_checksum=checksum,
            )

        async def convert_to_mono(self, _source, target) -> None:
            target.write_bytes(b"RIFF")

        def remove_files(self, paths) -> tuple[list, list]:
            removed = []
            for path in paths:
                path.unlink(missing_ok=True)
                removed.append(path)
            return removed, []

    class SuccessfulRetryOrchestrator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def transcribe(self, **_kwargs: object) -> OrchestratedTranscriptionResult:
            raise AssertionError("Completed provider results must not be retranscribed.")

    settings = SimpleNamespace(
        STORAGE_ROOT=tmp_path,
        OPENAI_DIARIZATION_MODEL="gpt-4o-transcribe-diarize",
        OPENAI_TRANSCRIPTION_MODEL="gpt-4o-transcribe",
        max_transcription_upload_bytes=25 * 1024 * 1024,
    )
    source_path = tmp_path / "recordings" / "reprocess.wav"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"RIFF")

    try:
        async with sessions() as session:
            user = User(
                email=f"reprocess-failure-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"reprocess-failure-{uuid4()}",
                extension_number="1012",
                display_name="Reprocess Failure Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"reprocess-failure-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
                processing_status="completed",
            )
            session.add_all([user, operator, call])
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"reprocess-failure-recording-{uuid4()}",
                yeastar_file_name="reprocess.wav",
                storage_key="recordings/reprocess.wav",
                status=RecordingStatus.COMPLETED,
            )
            session.add(recording)
            await session.flush()
            previous_id = uuid4()
            replacement_id = uuid4()
            item_id = uuid4()
            previous = Transcript(
                id=previous_id,
                call_id=call.id,
                recording_id=recording.id,
                operator_id=None,
                idempotency_key=f"reprocess-failure-previous-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model=settings.OPENAI_DIARIZATION_MODEL,
                language="el",
                source_audio_sha256=checksum,
                is_diarized=True,
                transcription_mode=TranscriptionMode.LEGACY,
                speaker_attribution_status=SpeakerAttributionStatus.ANONYMOUS_DIARIZATION,
                pipeline_version=LEGACY_PIPELINE_VERSION,
                pipeline_config_hash=LEGACY_DIARIZED_PIPELINE_CONFIG_HASH,
                preprocessing_profile=pipeline_module.LEGACY_PREPROCESSING_PROFILE,
                is_current=True,
                completed_at=now,
            )
            replacement_key = transcript_idempotency_key(
                recording.id,
                None,
                settings.OPENAI_DIARIZATION_MODEL,
                checksum,
                True,
                "el",
                None,
                pipeline_version=LEGACY_PIPELINE_VERSION,
                pipeline_config_hash=LEGACY_DIARIZED_PIPELINE_CONFIG_HASH,
                transcription_mode=TranscriptionMode.LEGACY,
                prompt_template_version=(pipeline_module.LEGACY_DIARIZED_PROMPT_TEMPLATE_VERSION),
                vocabulary_hash=None,
                supersedes_transcript_id=previous_id,
            )
            replacement = Transcript(
                id=replacement_id,
                call_id=call.id,
                recording_id=recording.id,
                operator_id=None,
                idempotency_key=replacement_key,
                status=TranscriptStatus.COMPLETED,
                model=settings.OPENAI_DIARIZATION_MODEL,
                language="el",
                source_audio_sha256=checksum,
                is_diarized=True,
                transcription_mode=TranscriptionMode.LEGACY,
                speaker_attribution_status=SpeakerAttributionStatus.ANONYMOUS_DIARIZATION,
                pipeline_version=LEGACY_PIPELINE_VERSION,
                pipeline_config_hash=LEGACY_DIARIZED_PIPELINE_CONFIG_HASH,
                preprocessing_profile=pipeline_module.LEGACY_PREPROCESSING_PROFILE,
                is_current=False,
                attempt_count=1,
                completed_at=now,
            )
            job = ProcessingJob(
                idempotency_key=f"reprocess-failure-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.QUEUED,
                date_from=now - timedelta(hours=1),
                date_to=now,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={
                    "_reprocess": True,
                    "_reprocess_targets": {str(item_id): str(previous_id)},
                    "_reprocess_replacements": {str(item_id): str(replacement_id)},
                },
            )
            session.add_all([previous, replacement, job])
            await session.flush()
            item = ProcessingJobItem(
                id=item_id,
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                idempotency_key=f"reprocess-failure-item-{uuid4()}",
                requested_pipeline_version=LEGACY_PIPELINE_VERSION,
                status=ItemStatus.QUEUED,
                stage="queued_for_reprocessing",
            )
            session.add(item)
            await session.commit()
            call_id = call.id
            recording_id = recording.id

        async def effective(_redis: object, current_settings: object) -> object:
            return current_settings

        async def application_settings(*_args: object) -> dict[str, object]:
            return {
                "default_language": "el",
                "delete_audio_after_transcription": False,
                "max_parallel_transcriptions": 1,
            }

        async def finish_keyword_search(session: object, *_args: object) -> int:
            if outcome == "cancellation":
                current_job = await session.get(ProcessingJob, job.id)  # type: ignore[attr-defined]
                current_item = await session.get(ProcessingJobItem, item_id)  # type: ignore[attr-defined]
                assert current_job is not None and current_item is not None
                current_job.cancellation_requested = True
                current_job.status = JobStatus.CANCELLED
                current_job.current_stage = "Cancelled"
                current_job.completed_at = now
                current_item.status = ItemStatus.CANCELLED
                current_item.stage = "cancelled"
                current_item.completed_at = now
                await session.commit()  # type: ignore[attr-defined]
                raise pipeline_module.TranscriptionCancelledError("Transcription was cancelled.")
            raise RuntimeError("late reprocess failure")

        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        monkeypatch.setattr(pipeline_module, "get_redis", lambda: FakeRedis())
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline_module, "AudioProcessor", FakeAudioProcessor)
        monkeypatch.setattr(
            pipeline_module,
            "load_effective_yeastar_settings",
            effective,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_effective_openai_settings",
            effective,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_application_settings",
            application_settings,
        )
        monkeypatch.setattr(
            pipeline_module,
            "search_and_persist_matches",
            finish_keyword_search,
        )

        if outcome == "failure":
            with pytest.raises(RuntimeError, match="late reprocess failure"):
                await pipeline_module.process_item(item_id)
        else:
            await pipeline_module.process_item(item_id)

        async with sessions() as session:
            previous = await session.get(Transcript, previous_id)
            replacement = await session.get(Transcript, replacement_id)
            item = await session.get(ProcessingJobItem, item_id)
            call = await session.get(Call, call_id)
            recording = await session.get(Recording, recording_id)
            assert previous is not None
            assert previous.status == TranscriptStatus.COMPLETED
            assert previous.is_current is True
            assert replacement is not None
            assert replacement.status == TranscriptStatus.COMPLETED
            assert replacement.is_current is False
            assert replacement.supersedes_transcript_id is None
            assert item is not None
            assert item.status == (
                ItemStatus.FAILED if outcome == "failure" else ItemStatus.CANCELLED
            )
            assert item.result_transcript_id is None
            assert call is not None and call.processing_status == "completed"
            assert recording is not None
            assert recording.status == RecordingStatus.COMPLETED

        if outcome == "failure":
            async with sessions() as session:
                retry_job = await session.get(ProcessingJob, job.id)
                retry_item = await session.get(ProcessingJobItem, item_id)
                assert retry_job is not None and retry_item is not None
                retry_job.status = JobStatus.QUEUED
                retry_job.current_stage = "Queued for retry"
                retry_job.completed_at = None
                retry_item.status = ItemStatus.QUEUED
                retry_item.stage = "queued"
                retry_item.completed_at = None
                retry_item.error_category = None
                retry_item.error_message = None
                await session.commit()

            async def successful_keyword_search(*_args: object) -> int:
                return 0

            monkeypatch.setattr(
                pipeline_module,
                "TranscriptionOrchestrator",
                SuccessfulRetryOrchestrator,
            )
            monkeypatch.setattr(
                pipeline_module,
                "search_and_persist_matches",
                successful_keyword_search,
            )

            await pipeline_module.process_item(item_id)

            async with sessions() as session:
                previous = await session.get(Transcript, previous_id)
                replacement = await session.get(Transcript, replacement_id)
                retried_item = await session.get(ProcessingJobItem, item_id)
                transcript_count = (
                    await session.scalar(
                        select(func.count())
                        .select_from(Transcript)
                        .where(Transcript.call_id == call_id)
                    )
                    or 0
                )
                assert previous is not None and previous.is_current is False
                assert replacement is not None
                assert replacement.status == TranscriptStatus.COMPLETED
                assert replacement.is_current is True
                assert replacement.supersedes_transcript_id == previous_id
                assert replacement.attempt_count == 1
                assert retried_item is not None
                assert retried_item.status == ItemStatus.COMPLETED
                assert retried_item.result_transcript_id == replacement_id
                assert transcript_count == 2
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lose_after_inspection",
    [False, True],
    ids=["owned", "lost-after-inspection"],
)
async def test_fresh_process_item_uses_orchestrator_and_persists_legacy_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    lose_after_inspection: bool,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)
    checksum = "c" * 64
    orchestrator_calls: list[dict[str, object]] = []
    searched_transcript_ids: list[UUID] = []

    class FakeLock:
        def __init__(self) -> None:
            self.released = False
            self.extend_calls = 0

        async def acquire(self, *, blocking: bool = False) -> bool:
            del blocking
            return True

        async def owned(self) -> bool:
            return not self.released

        async def extend(self, *_args: object, **_kwargs: object) -> None:
            self.extend_calls += 1
            return None

        async def release(self) -> None:
            self.released = True

    class FakeRedis:
        def __init__(self) -> None:
            self.locks: list[FakeLock] = []

        def lock(self, *_args: object, **_kwargs: object) -> FakeLock:
            lock = FakeLock()
            self.locks.append(lock)
            return lock

    class FakeAudioProcessor:
        def __init__(self, settings: object) -> None:
            self.root = settings.STORAGE_ROOT

        def safe_storage_path(self, relative_key: str):
            return self.root / relative_key

        async def inspect(self, path, **_kwargs: object) -> AudioInfo:
            result = AudioInfo(
                codec_name="pcm_s16le",
                format_name="wav",
                duration_seconds=30.0,
                channel_count=1,
                sample_rate_hz=16_000,
                bit_rate_bps=256_000,
                size_bytes=path.stat().st_size,
                sha256_checksum=checksum,
            )
            if lose_after_inspection:
                assert redis.locks
                redis.locks[0].released = True
            return result

        async def convert_to_mono(self, _source, target) -> None:
            target.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

        def remove_files(self, paths) -> tuple[list, list]:
            removed = []
            for path in paths:
                path.unlink(missing_ok=True)
                removed.append(path)
            return removed, []

    class ForbiddenDirectClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("the worker must not invoke the OpenAI adapter directly")

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["audio_processor"].__class__ is FakeAudioProcessor
            assert callable(kwargs["client_factory"])

        async def transcribe(self, **kwargs: object) -> OrchestratedTranscriptionResult:
            source_path = kwargs["source_path"]
            audio_info = kwargs["audio_info"]
            context = kwargs["context"]
            cancellation_check = kwargs["cancellation_check"]
            assert source_path.exists()
            assert audio_info.duration_seconds == 30.0
            assert context.diarized is True
            assert context.language == "el"
            assert context.vocabulary == ()
            assert context.audio_variant == "legacy-mono"
            assert await cancellation_check() is False
            orchestrator_calls.append(kwargs)
            hypothesis = ChunkHypothesis(
                track_id="legacy-diarized",
                chunk_index=0,
                start_seconds=1.2345,
                end_seconds=2.3456,
                text="Γεια σας",
                speaker_label="chunk-1:A",
            )
            track_result = TrackTranscriptionResult(
                track_id="legacy-diarized",
                model="gpt-4o-transcribe-diarize",
                language="el",
                prompt_version=None,
                processing_duration_seconds=0.5,
                hypotheses=(hypothesis,),
                usage={"totals": {"input_tokens": 7}},
                diarized=True,
            )
            return OrchestratedTranscriptionResult(
                mode="legacy",
                model=track_result.model,
                language=track_result.language,
                prompt_version=None,
                processing_duration_seconds=track_result.processing_duration_seconds,
                segments=track_result.hypotheses,
                tracks=(track_result,),
                usage=track_result.usage,
                diarized=True,
            )

    settings = SimpleNamespace(
        STORAGE_ROOT=tmp_path,
        OPENAI_DIARIZATION_MODEL="gpt-4o-transcribe-diarize",
        OPENAI_TRANSCRIPTION_MODEL="gpt-4o-transcribe",
        max_transcription_upload_bytes=25 * 1024 * 1024,
    )
    source_path = tmp_path / "recordings" / "fresh.wav"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")

    try:
        async with sessions() as session:
            user = User(
                email=f"orchestrator-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"orchestrator-{uuid4()}",
                extension_number="1013",
                display_name="Orchestrated Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"orchestrator-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
                processing_status="selected",
            )
            session.add_all([user, operator, call])
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"orchestrator-recording-{uuid4()}",
                yeastar_file_name="fresh.wav",
                storage_key="recordings/fresh.wav",
                status=RecordingStatus.DOWNLOADED,
            )
            job = ProcessingJob(
                idempotency_key=f"orchestrator-job-{uuid4()}",
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
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                idempotency_key=f"orchestrator-item-{uuid4()}",
                status=ItemStatus.QUEUED,
                stage="queued",
            )
            session.add(item)
            await session.commit()
            item_id = item.id
            call_id = call.id
            job_id = job.id
            recording_id = recording.id

        async def effective(_redis: object, current_settings: object) -> object:
            return current_settings

        async def application_settings(*_args: object) -> dict[str, object]:
            return {
                "default_language": "el",
                "delete_audio_after_transcription": False,
                "max_parallel_transcriptions": 1,
            }

        async def record_keyword_search(
            _session: object,
            transcript: Transcript,
            _job: ProcessingJob,
        ) -> int:
            searched_transcript_ids.append(transcript.id)
            return 0

        redis = FakeRedis()
        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        monkeypatch.setattr(pipeline_module, "get_redis", lambda: redis)
        monkeypatch.setattr(pipeline_module, "get_settings", lambda: settings)
        monkeypatch.setattr(pipeline_module, "AudioProcessor", FakeAudioProcessor)
        monkeypatch.setattr(
            pipeline_module,
            "OpenAITranscriptionClient",
            ForbiddenDirectClient,
        )
        monkeypatch.setattr(
            pipeline_module,
            "TranscriptionOrchestrator",
            FakeOrchestrator,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_effective_yeastar_settings",
            effective,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_effective_openai_settings",
            effective,
        )
        monkeypatch.setattr(
            pipeline_module,
            "load_application_settings",
            application_settings,
        )
        monkeypatch.setattr(
            pipeline_module,
            "search_and_persist_matches",
            record_keyword_search,
        )

        if lose_after_inspection:
            with pytest.raises(pipeline_module.ProcessingLeaseLostError):
                await pipeline_module.process_item(item_id)
        else:
            await pipeline_module.process_item(item_id)

        if lose_after_inspection:
            assert orchestrator_calls == []
            assert len(redis.locks) == 1
            async with sessions() as session:
                item = await session.get(ProcessingJobItem, item_id)
                job = await session.get(ProcessingJob, job_id)
                recording = await session.get(Recording, recording_id)
                transcript_count = (
                    await session.scalar(
                        select(func.count())
                        .select_from(Transcript)
                        .where(Transcript.call_id == call_id)
                    )
                    or 0
                )
                assert item is not None and item.status == ItemStatus.PROCESSING
                assert item.result_transcript_id is None
                assert job is not None and job.status == JobStatus.INSPECTING_AUDIO
                assert recording is not None
                assert recording.status == RecordingStatus.DOWNLOADED
                assert recording.sha256_checksum is None
                assert transcript_count == 0
            return

        assert len(orchestrator_calls) == 1
        assert len(redis.locks) == 2
        assert redis.locks[0].extend_calls > redis.locks[1].extend_calls >= 3
        async with sessions() as session:
            item = await session.get(ProcessingJobItem, item_id)
            transcript = await session.scalar(
                select(Transcript).where(Transcript.call_id == call_id)
            )
            segment = await session.scalar(
                select(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript.id)
            )
            assert item is not None
            assert transcript is not None
            assert segment is not None
            assert item.status == ItemStatus.COMPLETED
            assert item.result_transcript_id == transcript.id
            assert searched_transcript_ids == [transcript.id]
            assert transcript.status == TranscriptStatus.COMPLETED
            assert transcript.original_text == "Γεια σας"
            assert transcript.normalized_text == normalize_greek("Γεια σας")
            assert transcript.api_usage == {"totals": {"input_tokens": 7}}
            assert segment.start_seconds == Decimal("1.234")
            assert segment.end_seconds == Decimal("2.346")
            assert segment.speaker_label == "chunk-1:A"
            assert segment.speaker_source is SpeakerSource.OPENAI_DIARIZATION
            assert segment.operator_id is None
            assert segment.call_leg_id is None
            assert segment.sequence_number == 1
    finally:
        await engine.dispose()


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

    matched, reason = _recordings_for_participant([explicit, filename_only], SimpleNamespace(), leg)

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


def test_discovery_capacity_is_retried_without_acknowledging_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class RetryRequested(Exception):
        pass

    def busy(coroutine: object) -> None:
        coroutine.close()  # type: ignore[attr-defined]
        raise pipeline_module.ProcessingBusyError("discovery busy")

    def retry(*, exc: Exception, countdown: int) -> RetryRequested:
        captured.update(exc=exc, countdown=countdown)
        return RetryRequested()

    monkeypatch.setattr(worker_tasks, "_run", busy)
    monkeypatch.setattr(process_analysis_job, "retry", retry)

    with pytest.raises(RetryRequested):
        process_analysis_job.run(str(uuid4()))

    assert isinstance(captured["exc"], pipeline_module.ProcessingBusyError)
    assert captured["countdown"] == _busy_retry_delay(0)


@pytest.mark.asyncio
async def test_discovery_lock_contention_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BusyLock:
        async def acquire(self, *, blocking: bool = False) -> bool:
            assert blocking is False
            return False

    class BusyRedis:
        def lock(
            self,
            _key: str,
            *,
            timeout: int,
            blocking_timeout: int,
        ) -> BusyLock:
            assert timeout == pipeline_module.WORKER_LEASE_SECONDS
            assert blocking_timeout == 1
            return BusyLock()

    monkeypatch.setattr(pipeline_module, "get_redis", lambda: BusyRedis())

    with pytest.raises(pipeline_module.ProcessingBusyError):
        await pipeline_module.discover_job_items(uuid4())


@pytest.mark.asyncio
async def test_recent_processing_item_redelivery_stays_retryable(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    try:
        async with sessions() as session:
            user = User(
                email=f"redelivery-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"redelivery-{uuid4()}",
                extension_number="1099",
                display_name="Redelivery Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"redelivery-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
            )
            session.add_all([user, operator, call])
            await session.flush()
            job = ProcessingJob(
                idempotency_key=f"redelivery-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.TRANSCRIBING,
                date_from=now - timedelta(hours=1),
                date_to=now,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
                request_filters={},
                current_stage="Transcribing conversations",
            )
            session.add(job)
            await session.flush()
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                idempotency_key=f"redelivery-item-{uuid4()}",
                status=ItemStatus.PROCESSING,
                stage="transcribing",
                attempt_count=1,
                heartbeat_at=now,
            )
            session.add(item)
            await session.commit()
            item_id = item.id

        monkeypatch.setattr(pipeline_module, "AsyncSessionFactory", sessions)
        monkeypatch.setattr(pipeline_module, "get_redis", lambda: object())
        monkeypatch.setattr(
            pipeline_module,
            "get_settings",
            lambda: SimpleNamespace(STORAGE_ROOT=tmp_path),
        )

        with pytest.raises(
            pipeline_module.ProcessingBusyError,
            match="already being processed",
        ):
            await pipeline_module.process_item(item_id)

        async with sessions() as session:
            item = await session.get(ProcessingJobItem, item_id)
            assert item is not None
            assert item.status == ItemStatus.PROCESSING
            assert item.attempt_count == 1
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("recording_owned", "recording_extended", "slot_owned", "slot_extended"),
    [
        (False, True, True, True),
        (True, False, True, True),
        (True, True, False, True),
        (True, True, True, False),
    ],
)
async def test_lost_processing_lease_aborts_before_persistence(
    recording_owned: bool,
    recording_extended: bool,
    slot_owned: bool,
    slot_extended: bool,
) -> None:
    class Lease:
        def __init__(self, *, owned: bool, extended: bool) -> None:
            self.owned_result = owned
            self.extended_result = extended

        async def owned(self) -> bool:
            return self.owned_result

        async def extend(self, *_args: object, **_kwargs: object) -> bool:
            return self.extended_result

    with pytest.raises(pipeline_module.TranscriptionError) as raised:
        await pipeline_module._refresh_processing_leases(
            Lease(owned=recording_owned, extended=recording_extended),
            Lease(owned=slot_owned, extended=slot_extended),
        )

    assert isinstance(raised.value, pipeline_module.ProcessingBusyError)
    assert raised.value.category == "processing_lock"
    assert str(raised.value) == "Processing lock became unavailable."


async def _assignment_discovery_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    *,
    recordings: list[dict[str, object]],
    detail: dict[str, object],
    legacy_item: tuple[ItemStatus, str | None] | None = None,
    existing_run: bool = False,
    existing_run_status: RunStatus = RunStatus.COMPLETED,
    existing_run_updated_at: datetime | None = None,
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

        async def search_recordings(
            self, *_args: object, **_kwargs: object
        ) -> list[dict[str, object]]:
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
                        status=existing_run_status,
                        idempotency_key=f"job:{job.id}:attempt:{job.attempt_count}",
                        processing_job_id=job.id,
                        started_at=now,
                        completed_at=(now if existing_run_status != RunStatus.RUNNING else None),
                        date_from=job.date_from,
                        date_to=job.date_to,
                        updated_at=existing_run_updated_at or now,
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
async def test_recent_running_discovery_attempt_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        pipeline_module.ProcessingBusyError,
        match="discovery is already running",
    ):
        await _assignment_discovery_snapshot(
            monkeypatch,
            recordings=[],
            detail={},
            existing_run=True,
            existing_run_status=RunStatus.RUNNING,
            existing_run_updated_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_stale_running_discovery_attempt_is_safely_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, items, job_status, run_count = await _assignment_discovery_snapshot(
        monkeypatch,
        recordings=[
            {
                "id": "stale-run-recording",
                "uid": "assignment-call",
                "file": "stale-run.wav",
            }
        ],
        detail={
            "timeline": [
                {
                    "transaction_id": "transaction-1",
                    "cdr_id": "leg-1",
                    "call_to_ext_id": "extension-1001",
                    "call_from_number": "302100000000",
                    "call_to_number": "1001",
                    "recording_id": "stale-run-recording",
                    "status": "ANSWERED",
                }
            ]
        },
        existing_run=True,
        existing_run_status=RunStatus.RUNNING,
        existing_run_updated_at=(
            datetime.now(UTC) - pipeline_module.WORKER_STALE_AFTER - timedelta(seconds=1)
        ),
    )

    assert len(result.item_ids) == 1
    assert len(items) == 1
    assert items[0][0] == ItemStatus.QUEUED
    assert items[0][1] == "queued"
    assert items[0][2] is not None
    assert job_status == JobStatus.DOWNLOADING_RECORDINGS
    assert run_count == 1


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
        recordings=[{"id": "recording-legacy", "uid": "assignment-call", "file": "legacy.wav"}],
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
        recordings=[{"id": "recording-legacy", "uid": "assignment-call", "file": "legacy.wav"}],
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
        recordings=[{"id": "recording-now-linked", "uid": "assignment-call", "file": "linked.wav"}],
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
            assert (await session.scalar(select(func.count()).select_from(KeywordMatch)) or 0) == 0

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

        async def extend(self, *_args: object, **_kwargs: object) -> None:
            return None

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
