from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.database.base import Base
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
)
from app.models.enums import (
    Direction,
    JobStatus,
    RecordingStatus,
    SpeakerAttributionStatus,
    SpeakerSource,
    TranscriptionMode,
    TranscriptStatus,
)
from app.schemas.results import CallReprocessRequest
from app.services.keyword_matching import normalize_greek
from app.services.audio.segmentation import SpeechSegmentationConfig
from app.services.transcription.orchestrator import (
    default_speech_segmentation_identity,
)
from app.services.transcription.types import (
    AudioPlan,
    AudioTrack,
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    TranscriptionAttemptEvidence,
)
from app.workers.pipeline import (
    LEGACY_PIPELINE_VERSION,
    PIPELINE_V2_VERSION,
    SUPPORTED_PIPELINE_VERSIONS,
    _attribution_warning_message,
    _cleanup_temporary_audio,
    _effective_pipeline_version,
    _persist_transcription_result,
    _pipeline_v2_runtime_config_hash,
    _transcript_has_attribution_warning,
    _vocabulary,
    search_and_persist_matches,
)


def test_pipeline_v2_cleanup_removes_a_tracked_empty_item_directory(tmp_path: Path) -> None:
    item_directory = tmp_path / "tmp" / "item-token"
    item_directory.mkdir(parents=True)

    class EmptyAudioCleanup:
        def remove_files(self, paths: list[Path]) -> tuple[list[Path], list[Path]]:
            assert paths == []
            return [], []

    _cleanup_temporary_audio(  # type: ignore[arg-type]
        EmptyAudioCleanup(),
        [],
        {item_directory},
    )

    assert not item_directory.exists()


def _runtime_hash(plan: AudioPlan) -> str:
    return _pipeline_v2_runtime_config_hash(
        plan=plan,
        max_upload_bytes=24 * 1024 * 1024,
        prompt_template_version="legacy-isolated-vocabulary-v1",
    )


def _segmentation_hash_plan(mode: str) -> AudioPlan:
    source = Path("recording.wav")
    if mode == "mono_diarization":
        return AudioPlan(
            mode=mode,
            tracks=(
                AudioTrack(
                    track_id="mono-diarization",
                    source_path=source,
                    attribution_status="anonymous_diarization",
                    audio_variant="topology-mono",
                    diarized=True,
                    speaker_source="openai_diarization",
                ),
            ),
            attribution_status="anonymous_diarization",
            reason="stereo-not-confirmed-separated",
        )
    return AudioPlan(
        mode=mode,
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=source,
                channel_index=0,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Ada Agent",
                speaker_source="stereo_channel",
            ),
        ),
        operator_channel=0,
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="confirmed_by_pbx",
        reason="confirmed-separated-stereo-safe-operator",
    )


def test_pipeline_versions_keep_legacy_default_and_explicit_v2_rollback_choices() -> None:
    ordinary = ProcessingJobItem(requested_pipeline_version=None)
    topology_reprocess = ProcessingJobItem(requested_pipeline_version=PIPELINE_V2_VERSION)
    legacy_reprocess = ProcessingJobItem(requested_pipeline_version=LEGACY_PIPELINE_VERSION)

    assert SUPPORTED_PIPELINE_VERSIONS == {
        LEGACY_PIPELINE_VERSION,
        PIPELINE_V2_VERSION,
    }
    assert _effective_pipeline_version(ordinary) == LEGACY_PIPELINE_VERSION
    assert _effective_pipeline_version(topology_reprocess) == PIPELINE_V2_VERSION
    assert _effective_pipeline_version(legacy_reprocess) == LEGACY_PIPELINE_VERSION

    assert (
        CallReprocessRequest(pipeline_version=PIPELINE_V2_VERSION).pipeline_version
        == PIPELINE_V2_VERSION
    )
    assert (
        CallReprocessRequest(pipeline_version=LEGACY_PIPELINE_VERSION).pipeline_version
        == LEGACY_PIPELINE_VERSION
    )
    with pytest.raises(ValidationError):
        CallReprocessRequest(pipeline_version="future-unsafe")  # type: ignore[arg-type]


def test_pipeline_v2_runtime_hash_covers_mode_channel_and_mapping() -> None:
    source = Path("recording.wav")
    operator_zero = AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=source,
                channel_index=0,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Ada Agent",
                speaker_source="stereo_channel",
            ),
        ),
        operator_channel=0,
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="confirmed_by_pbx",
        reason="confirmed-separated-stereo-safe-operator",
    )
    operator_one = AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=source,
                channel_index=1,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Ada Agent",
                speaker_source="stereo_channel",
            ),
        ),
        operator_channel=1,
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="confirmed_by_pbx",
        reason="confirmed-separated-stereo-safe-operator",
    )
    mapped_dual = AudioPlan(
        mode="dual_channel",
        tracks=tuple(
            AudioTrack(
                track_id=f"channel-{channel}",
                source_path=source,
                channel_index=channel,
                attribution_status="caller_callee_only",
                audio_variant=f"topology-channel-{channel}",
                speaker_label=("Caller" if channel == 0 else "Callee"),
                speaker_source="stereo_channel",
            )
            for channel in (0, 1)
        ),
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="caller_callee_only",
        reason="confirmed-separated-stereo-caller-callee",
    )
    reversed_mapping = replace(
        mapped_dual,
        caller_channel=1,
        callee_channel=0,
    )
    unknown_dual = AudioPlan(
        mode="dual_channel",
        tracks=tuple(
            AudioTrack(
                track_id=f"channel-{channel}",
                source_path=source,
                channel_index=channel,
                attribution_status="channel_unknown",
                audio_variant=f"topology-channel-{channel}",
                speaker_label=("Channel A" if channel == 0 else "Channel B"),
                speaker_source="stereo_channel",
            )
            for channel in (0, 1)
        ),
        stereo_separated=True,
        attribution_status="channel_unknown",
        reason="confirmed-separated-stereo-attribution-unknown",
    )
    mono = AudioPlan(
        mode="mono_diarization",
        tracks=(
            AudioTrack(
                track_id="mono-diarization",
                source_path=source,
                attribution_status="anonymous_diarization",
                audio_variant="topology-mono",
                diarized=True,
                speaker_source="openai_diarization",
            ),
        ),
        attribution_status="anonymous_diarization",
        reason="stereo-not-confirmed-separated",
    )

    hashes = {
        _runtime_hash(operator_zero),
        _runtime_hash(operator_one),
        _runtime_hash(mapped_dual),
        _runtime_hash(reversed_mapping),
        _runtime_hash(unknown_dual),
        _runtime_hash(mono),
    }

    assert _runtime_hash(operator_zero) != _runtime_hash(operator_one)
    assert _runtime_hash(mapped_dual) != _runtime_hash(reversed_mapping)
    assert _runtime_hash(operator_zero) != _runtime_hash(mono)
    assert len(hashes) == 6
    assert all(len(value) == 64 for value in hashes)


def test_default_speech_segmentation_identity_is_complete_and_deployable() -> None:
    identity = default_speech_segmentation_identity(max_upload_bytes=24 * 1024 * 1024)

    assert identity["backend"] == {
        "distribution": "webrtcvad-wheels",
        "version": "2.0.14",
    }
    assert identity["analysis_format"] == {
        "channel_count": 1,
        "frame_ms": 20,
        "frame_samples": 320,
        "sample_format": "signed-16-bit-pcm",
        "sample_rate_hz": 16_000,
    }
    assert identity["output_format"] == "pcm-s16le-wav"
    assert identity["configuration"] == SpeechSegmentationConfig().identity()
    assert identity["max_upload_bytes"] == 24 * 1024 * 1024
    assert identity["overlap_join"] == {
        "case_sensitive": True,
        "maximum_tokens": 24,
        "minimum_characters": 8,
        "minimum_tokens": 2,
        "strategy": "exact-case-sensitive-token-overlap-v1",
    }


@pytest.mark.parametrize(
    "configuration_key",
    tuple(SpeechSegmentationConfig().identity()),
)
def test_pipeline_v2_hash_changes_for_every_speech_boundary_configuration_value(
    configuration_key: str,
) -> None:
    plan = _segmentation_hash_plan("operator_channel")
    identity = default_speech_segmentation_identity(max_upload_bytes=24 * 1024 * 1024)
    changed_identity = deepcopy(identity)
    configuration = changed_identity["configuration"]
    assert isinstance(configuration, dict)
    value = configuration[configuration_key]
    assert isinstance(value, (int, float))
    configuration[configuration_key] = value + 1

    original_hash = _pipeline_v2_runtime_config_hash(
        plan=plan,
        max_upload_bytes=24 * 1024 * 1024,
        prompt_template_version="legacy-isolated-vocabulary-v1",
        segmentation_identity=identity,
    )
    changed_hash = _pipeline_v2_runtime_config_hash(
        plan=plan,
        max_upload_bytes=24 * 1024 * 1024,
        prompt_template_version="legacy-isolated-vocabulary-v1",
        segmentation_identity=changed_identity,
    )

    assert changed_hash != original_hash


def test_pipeline_v2_mono_hash_tracks_two_pass_policy_but_not_speech_segmentation() -> None:
    plan = _segmentation_hash_plan("mono_diarization")
    mono_policy = {"version": "mono-two-pass-test-v1", "maximum_span_seconds": 45}

    expected = _pipeline_v2_runtime_config_hash(
        plan=plan,
        max_upload_bytes=24 * 1024 * 1024,
        prompt_template_version="greek-callcenter-mono-v1",
        standard_model="gpt-4o-transcribe",
        diarization_model="gpt-4o-transcribe-diarize",
        mono_refinement_identity=mono_policy,
    )
    with_irrelevant_speech_identity = _pipeline_v2_runtime_config_hash(
        plan=plan,
        max_upload_bytes=24 * 1024 * 1024,
        prompt_template_version="greek-callcenter-mono-v1",
        standard_model="gpt-4o-transcribe",
        diarization_model="gpt-4o-transcribe-diarize",
        mono_refinement_identity=mono_policy,
        segmentation_identity={"strategy": "must-not-affect-mono"},
    )
    with_changed_mono_policy = _pipeline_v2_runtime_config_hash(
        plan=plan,
        max_upload_bytes=24 * 1024 * 1024,
        prompt_template_version="greek-callcenter-mono-v1",
        standard_model="gpt-4o-transcribe",
        diarization_model="gpt-4o-transcribe-diarize",
        mono_refinement_identity={
            "version": "mono-two-pass-test-v1",
            "maximum_span_seconds": 44,
        },
    )

    assert with_irrelevant_speech_identity == expected
    assert with_changed_mono_policy != expected


@pytest.mark.parametrize(
    ("mode", "attribution", "diarized", "expected_warning"),
    [
        (
            TranscriptionMode.DUAL_CHANNEL,
            SpeakerAttributionStatus.CALLER_CALLEE_ONLY,
            False,
            True,
        ),
        (
            TranscriptionMode.DUAL_CHANNEL,
            SpeakerAttributionStatus.CHANNEL_UNKNOWN,
            False,
            True,
        ),
        (
            TranscriptionMode.MONO_DIARIZATION,
            SpeakerAttributionStatus.ANONYMOUS_DIARIZATION,
            True,
            True,
        ),
        (
            TranscriptionMode.LEGACY,
            SpeakerAttributionStatus.ANONYMOUS_DIARIZATION,
            True,
            True,
        ),
        (
            TranscriptionMode.OPERATOR_CHANNEL,
            SpeakerAttributionStatus.CONFIRMED_BY_PBX,
            False,
            False,
        ),
        (
            TranscriptionMode.LEGACY,
            SpeakerAttributionStatus.CONFIRMED_BY_PBX,
            False,
            False,
        ),
    ],
)
def test_attribution_warning_policy_covers_topology_and_legacy(
    mode: TranscriptionMode,
    attribution: SpeakerAttributionStatus,
    diarized: bool,
    expected_warning: bool,
) -> None:
    transcript = SimpleNamespace(
        transcription_mode=mode,
        speaker_attribution_status=attribution,
        is_diarized=diarized,
    )

    assert _transcript_has_attribution_warning(transcript) is expected_warning
    message = _attribution_warning_message(transcript)
    assert (message is not None) is expected_warning
    if mode == TranscriptionMode.DUAL_CHANNEL:
        assert message is not None
        assert "Channels were preserved" in message


class _CapturingSession:
    def __init__(self) -> None:
        self.added: list[TranscriptSegment] = []

    def add(self, value: object) -> None:
        if isinstance(value, TranscriptSegment):
            self.added.append(value)

    async def flush(self) -> None:
        return None


def _result(
    *,
    mode: str,
    attribution_status: str,
    segments: tuple[ChunkHypothesis, ...],
) -> OrchestratedTranscriptionResult:
    attempts = (
        tuple(
            TranscriptionAttemptEvidence(
                track_id=segment.track_id,
                chunk_index=segment.chunk_index or 0,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                model="gpt-4o-transcribe",
                audio_variant=segment.audio_variant,
                prompt_hash="a" * 64,
                response_text=segment.text,
                selected=True,
                completed_at=datetime.now(UTC),
            )
            for segment in segments
        )
        if mode in {"operator_channel", "dual_channel"}
        else ()
    )
    return OrchestratedTranscriptionResult(
        mode=mode,  # type: ignore[arg-type]
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="prompt-v1",
        processing_duration_seconds=1.5,
        segments=segments,
        tracks=(),
        usage={"seconds": 15},
        diarized=mode == "mono_diarization",
        attribution_status=attribution_status,
        plan_reason="test-plan",
        attempts=attempts,
    )


@pytest.mark.asyncio
async def test_topology_persistence_keeps_dual_neutral_and_operator_confirmed() -> None:
    call = SimpleNamespace(id=uuid4())
    dual_transcript = SimpleNamespace(id=uuid4(), operator_id=None)
    dual_result = _result(
        mode="dual_channel",
        attribution_status="caller_callee_only",
        segments=(
            ChunkHypothesis(
                track_id="channel-0",
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=2.5,
                text="Caller text",
                speaker_label="Caller",
                channel_index=0,
                operator_id=None,
                speaker_source="stereo_channel",
                audio_variant="topology-channel-0",
            ),
            ChunkHypothesis(
                track_id="channel-1",
                chunk_index=0,
                start_seconds=0.5,
                end_seconds=3.0,
                text="Callee text",
                speaker_label="Callee",
                channel_index=1,
                operator_id=None,
                speaker_source="stereo_channel",
                audio_variant="topology-channel-1",
            ),
        ),
    )
    dual_session = _CapturingSession()

    await _persist_transcription_result(
        dual_session,  # type: ignore[arg-type]
        dual_transcript,  # type: ignore[arg-type]
        dual_result,
        call,  # type: ignore[arg-type]
        None,
        None,
        30.0,
    )

    assert dual_transcript.transcription_mode is TranscriptionMode.DUAL_CHANNEL
    assert dual_transcript.speaker_attribution_status is SpeakerAttributionStatus.CALLER_CALLEE_ONLY
    assert [
        (
            segment.channel_index,
            segment.track_id,
            segment.chunk_index,
            segment.speaker_label,
            segment.speaker_source,
            segment.operator_id,
            segment.call_leg_id,
            segment.audio_variant,
        )
        for segment in dual_session.added
    ] == [
        (
            0,
            "channel-0",
            0,
            "Caller",
            SpeakerSource.STEREO_CHANNEL,
            None,
            None,
            "topology-channel-0",
        ),
        (
            1,
            "channel-1",
            0,
            "Callee",
            SpeakerSource.STEREO_CHANNEL,
            None,
            None,
            "topology-channel-1",
        ),
    ]

    operator_id = uuid4()
    call_leg_id = uuid4()
    operator = SimpleNamespace(id=operator_id, display_name="Ada Agent")
    operator_transcript = SimpleNamespace(id=uuid4(), operator_id=operator_id)
    operator_result = _result(
        mode="operator_channel",
        attribution_status="confirmed_by_pbx",
        segments=(
            ChunkHypothesis(
                track_id="operator-channel",
                chunk_index=2,
                start_seconds=15.0,
                end_seconds=20.0,
                text="Operator text",
                speaker_label="Ada Agent",
                channel_index=1,
                operator_id=str(operator_id),
                speaker_source="stereo_channel",
                audio_variant="topology-operator-channel",
            ),
        ),
    )
    operator_session = _CapturingSession()

    await _persist_transcription_result(
        operator_session,  # type: ignore[arg-type]
        operator_transcript,  # type: ignore[arg-type]
        operator_result,
        call,  # type: ignore[arg-type]
        operator,  # type: ignore[arg-type]
        call_leg_id,
        30.0,
    )

    segment = operator_session.added[0]
    assert operator_transcript.transcription_mode is TranscriptionMode.OPERATOR_CHANNEL
    assert (
        operator_transcript.speaker_attribution_status is SpeakerAttributionStatus.CONFIRMED_BY_PBX
    )
    assert segment.operator_id == operator_id
    assert segment.call_leg_id == call_leg_id
    assert segment.speaker_label == "Ada Agent"
    assert segment.speaker_source is SpeakerSource.STEREO_CHANNEL
    assert segment.channel_index == 1
    assert segment.track_id == "operator-channel"
    assert segment.chunk_index == 2


async def _new_database() -> tuple[object, async_sessionmaker]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


@pytest.mark.asyncio
async def test_dual_vocabulary_does_not_weight_the_selected_operator() -> None:
    engine, sessions = await _new_database()
    now = datetime.now(UTC)
    try:
        async with sessions() as session:
            ada = Operator(
                yeastar_extension_id=f"ada-{uuid4()}",
                extension_number="101",
                display_name="Ada Agent",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            ben = Operator(
                yeastar_extension_id=f"ben-{uuid4()}",
                extension_number="102",
                display_name="Ben Agent",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            category = KeywordCategory(name=f"Vocabulary {uuid4()}", active=True)
            session.add_all([ada, ben, category])
            await session.flush()
            session.add(
                Keyword(
                    category_id=category.id,
                    canonical_phrase="refund request",
                    normalized_phrase=normalize_greek("refund request"),
                    active=True,
                    whole_word=True,
                    exact_phrase=True,
                )
            )
            await session.commit()

            settings = Settings(APP_ENV="test")
            neutral = await _vocabulary(session, None, settings)
            repeated = await _vocabulary(session, None, settings)
            operator_weighted = await _vocabulary(session, ada, settings)

            assert neutral == repeated
            assert neutral.count("Ada Agent") == 1
            assert neutral.count("Ben Agent") == 1
            assert operator_weighted[0] == "Ada Agent"
            assert operator_weighted.count("Ada Agent") == 2
            assert "refund request" in neutral
    finally:
        await engine.dispose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_keyword_search_excludes_unattributed_segments_unless_include_all() -> None:
    engine, sessions = await _new_database()
    now = datetime.now(UTC)
    operator_id = uuid4()
    try:
        async with sessions() as session:
            operator = Operator(
                id=operator_id,
                yeastar_extension_id=f"keyword-{uuid4()}",
                extension_number="103",
                display_name="Confirmed Agent",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"topology-keyword-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
            )
            session.add_all([operator, call])
            await session.flush()
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
                operator_id=operator.id,
                idempotency_key=f"transcript-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                source_audio_sha256="a" * 64,
                is_diarized=False,
                transcription_mode=TranscriptionMode.OPERATOR_CHANNEL,
                speaker_attribution_status=SpeakerAttributionStatus.CONFIRMED_BY_PBX,
                pipeline_version=PIPELINE_V2_VERSION,
                pipeline_config_hash="b" * 64,
                completed_at=now,
            )
            session.add(transcript)
            await session.flush()
            category = KeywordCategory(name=f"Refunds {uuid4()}", active=True)
            session.add(category)
            await session.flush()
            keyword = Keyword(
                category_id=category.id,
                canonical_phrase="refund request",
                normalized_phrase=normalize_greek("refund request"),
                active=True,
                whole_word=True,
                exact_phrase=True,
            )
            confirmed_segment = TranscriptSegment(
                transcript_id=transcript.id,
                call_id=call.id,
                operator_id=operator.id,
                speaker_label=operator.display_name,
                speaker_source=SpeakerSource.STEREO_CHANNEL,
                start_seconds=0,
                end_seconds=2,
                original_text="A refund request was received.",
                normalized_text=normalize_greek("A refund request was received."),
                transcription_model=transcript.model,
                sequence_number=1,
                channel_index=0,
                track_id="operator-channel",
                chunk_index=0,
                audio_variant="topology-operator-channel",
            )
            unattributed_segment = TranscriptSegment(
                transcript_id=transcript.id,
                call_id=call.id,
                operator_id=None,
                speaker_label="Channel B",
                speaker_source=SpeakerSource.STEREO_CHANNEL,
                start_seconds=3,
                end_seconds=5,
                original_text="Another refund request was received.",
                normalized_text=normalize_greek("Another refund request was received."),
                transcription_model=transcript.model,
                sequence_number=2,
                channel_index=1,
                track_id="channel-1",
                chunk_index=0,
                audio_variant="topology-channel-1",
            )
            job = ProcessingJob(
                idempotency_key=f"job-{uuid4()}",
                requested_by_id=uuid4(),
                status=JobStatus.SEARCHING_KEYWORDS,
                date_from=now - timedelta(hours=1),
                date_to=now + timedelta(seconds=1),
                include_all_speakers=False,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[str(category.id)],
                request_filters={},
            )
            session.add_all([keyword, confirmed_segment, unattributed_segment, job])
            await session.commit()

            assert await search_and_persist_matches(session, transcript, job) == 1
            await session.commit()
            first_match = await session.scalar(select(KeywordMatch))
            assert first_match is not None
            assert first_match.operator_id == operator.id
            assert first_match.transcript_segment_id == confirmed_segment.id

            job.include_all_speakers = True
            assert await search_and_persist_matches(session, transcript, job) == 1
            await session.commit()

            matches = (
                await session.scalars(select(KeywordMatch).order_by(KeywordMatch.start_seconds))
            ).all()
            assert len(matches) == 2
            assert [match.operator_id for match in matches] == [operator.id, None]
            assert await session.scalar(select(func.count()).select_from(KeywordMatch)) == 2
    finally:
        await engine.dispose()  # type: ignore[union-attr]
