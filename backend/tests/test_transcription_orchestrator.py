from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from app.core.config import Settings
from app.models.enums import SpeakerSource
from app.services.audio import AudioChunk, AudioInfo
from app.services.audio.quality import LegacyPassThroughQualityProcessor
from app.services.audio.segmentation import LegacyFixedAudioSegmenter
from app.services.keyword_matching import KeywordDefinition, match_text
from app.services.transcription.client import (
    OpenAITranscriptionClient,
    TranscribedSegment,
    TranscriptionCancelledError,
    TranscriptionError,
    TranscriptionResult,
    build_vocabulary_prompt as client_build_vocabulary_prompt,
)
from app.services.transcription.orchestrator import (
    TranscriptionContext,
    TranscriptionOrchestrator,
)
from app.services.transcription.planning import LegacyAudioPlanner
from app.services.transcription.prompt import (
    LegacyVocabularyPromptBuilder,
    build_vocabulary_prompt,
)
from app.services.transcription.types import (
    AudioTrack,
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    SpeechChunk,
    TrackTranscriptionResult,
)
from app.workers.pipeline import _persist_transcription_result


def _settings(storage_root: Path, *, max_upload_bytes: int | None = None) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=storage_root,
        OPENAI_API_KEY="test-only-key",
        MAX_TRANSCRIPTION_UPLOAD_BYTES=max_upload_bytes,
    )


def _audio_info(*, duration_seconds: float, size_bytes: int = 44) -> AudioInfo:
    return AudioInfo(
        codec_name="pcm_s16le",
        format_name="wav",
        duration_seconds=duration_seconds,
        channel_count=1,
        sample_rate_hz=16_000,
        bit_rate_bps=256_000,
        size_bytes=size_bytes,
        sha256_checksum="a" * 64,
    )


def _chunk(path: Path, start_seconds: float, end_seconds: float) -> AudioChunk:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    return AudioChunk(
        path=path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )


def _fake_openai(
    *responses: object,
) -> tuple[SimpleNamespace, AsyncMock, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    remaining = iter(responses)

    async def create(**kwargs: Any) -> object:
        requests.append({key: value for key, value in kwargs.items() if key != "file"})
        return next(remaining)

    create_mock = AsyncMock(side_effect=create)
    client = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=create_mock),
        ),
        models=SimpleNamespace(retrieve=AsyncMock()),
    )
    return client, create_mock, requests


class StubAudioProcessor:
    def __init__(self, chunks: list[AudioChunk]) -> None:
        self.chunks = chunks
        self.split_calls: list[tuple[Path, Path, float, float]] = []

    async def split_audio(
        self,
        source: Path,
        destination_dir: Path,
        duration_seconds: float,
        *,
        chunk_seconds: float,
    ) -> list[AudioChunk]:
        self.split_calls.append((source, destination_dir, duration_seconds, chunk_seconds))
        return self.chunks


def _context(
    tmp_path: Path,
    *,
    diarized: bool,
    registered: list[Path],
) -> TranscriptionContext:
    return TranscriptionContext(
        diarized=diarized,
        language="el",
        vocabulary=("  Yeastar ", "Γιώργος", "yeastar"),
        temporary_directory=tmp_path / "temporary",
        operator_id=None if diarized else str(uuid4()),
        attribution_status=("anonymous_diarization" if diarized else "confirmed_by_pbx"),
        audio_variant="legacy-mono" if diarized else "legacy-operator-channel",
        register_temporary_file=registered.append,
    )


def _segment_snapshot(result: TranscriptionResult | OrchestratedTranscriptionResult) -> list[tuple]:
    return [
        (
            segment.start_seconds,
            segment.end_seconds,
            segment.text,
            segment.speaker_label,
            segment.confidence,
        )
        for segment in result.segments
    ]


def test_legacy_prompt_contract_is_shared_and_byte_stable() -> None:
    values = ["  Alpha   Beta ", "alpha beta", "", " Γιώργος "]
    expected = "Greek business vocabulary and names: Alpha Beta, Γιώργος"
    expected_version = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:16]

    plan = LegacyVocabularyPromptBuilder().build(values)

    assert (plan.text, plan.version) == (expected, expected_version)
    assert build_vocabulary_prompt(values) == (expected, expected_version)
    assert client_build_vocabulary_prompt(values) == (expected, expected_version)


def _status_error(error_type: type[APIStatusError], status_code: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response = httpx.Response(status_code, request=request)
    return error_type("provider detail", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "category", "message"),
    [
        (
            _status_error(AuthenticationError, 401),
            "openai_authentication",
            "Transcription credentials were rejected.",
        ),
        (
            _status_error(PermissionDeniedError, 403),
            "openai_authentication",
            "Transcription credentials were rejected.",
        ),
        (
            _status_error(RateLimitError, 429),
            "openai_rate_limit",
            "Transcription service is busy. Retry later.",
        ),
        (
            APITimeoutError(
                httpx.Request(
                    "POST",
                    "https://api.openai.com/v1/audio/transcriptions",
                )
            ),
            "openai_timeout",
            "Transcription request timed out.",
        ),
        (
            APIConnectionError(
                request=httpx.Request(
                    "POST",
                    "https://api.openai.com/v1/audio/transcriptions",
                )
            ),
            "openai_connection",
            "Could not connect to the transcription service.",
        ),
        (
            _status_error(APIStatusError, 503),
            "openai_unavailable",
            "Transcription service is unavailable.",
        ),
        (
            _status_error(APIStatusError, 400),
            "openai_request",
            "Transcription request was rejected.",
        ),
        (
            RuntimeError("provider implementation detail"),
            "openai_unexpected",
            "Transcription failed unexpectedly.",
        ),
    ],
)
def test_openai_error_translation_contract_is_unchanged(
    error: Exception,
    category: str,
    message: str,
) -> None:
    translated = OpenAITranscriptionClient._translated_error(error)

    assert translated.category == category
    assert str(translated) == message


def test_existing_transcription_error_is_not_retranslated() -> None:
    error = TranscriptionError("safe message", "safe_category")

    assert OpenAITranscriptionClient._translated_error(error) is error


@pytest.mark.asyncio
async def test_isolated_orchestrator_matches_direct_legacy_requests_and_result(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    chunks = [
        _chunk(tmp_path / "chunks" / "one.wav", 0, 15),
        _chunk(tmp_path / "chunks" / "two.wav", 15, 22),
    ]
    responses = [
        SimpleNamespace(text=" πρώτο ", usage={"input_tokens": 3}),
        SimpleNamespace(text="δεύτερο", usage={"input_tokens": 5}),
    ]
    direct_openai, direct_create, direct_requests = _fake_openai(*responses)
    orchestrated_openai, orchestrated_create, orchestrated_requests = _fake_openai(*responses)
    vocabulary = ["  Yeastar ", "Γιώργος", "yeastar"]

    direct = await OpenAITranscriptionClient(
        settings,
        client=direct_openai,
    ).transcribe_isolated(chunks, vocabulary, language="el")

    processor = StubAudioProcessor(chunks)
    registered: list[Path] = []
    orchestrated = await TranscriptionOrchestrator(
        settings=settings,
        audio_processor=processor,  # type: ignore[arg-type]
        client_factory=lambda: OpenAITranscriptionClient(
            settings,
            client=orchestrated_openai,
        ),
    ).transcribe(
        source_path=tmp_path / "prepared.wav",
        audio_info=_audio_info(duration_seconds=22),
        context=_context(tmp_path, diarized=False, registered=registered),
    )

    assert orchestrated.mode == "legacy"
    assert orchestrated.confidence_status == "unavailable"
    assert orchestrated.model == direct.model
    assert orchestrated.language == direct.language
    assert orchestrated.prompt_version == direct.prompt_version
    assert orchestrated.text == direct.text
    assert orchestrated.usage == direct.usage
    assert _segment_snapshot(orchestrated) == _segment_snapshot(direct)
    assert orchestrated_create.await_count == direct_create.await_count == len(chunks)
    assert orchestrated_requests == direct_requests
    assert processor.split_calls == [(tmp_path / "prepared.wav", tmp_path / "temporary", 22, 15)]
    assert registered == [chunk.path for chunk in chunks]


@pytest.mark.asyncio
async def test_small_diarized_orchestrator_matches_direct_single_upload(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    prepared = tmp_path / "prepared.wav"
    direct_chunk = _chunk(prepared, 0, 30)
    response = {
        "segments": [
            {"start": 1.25, "end": 2.5, "text": "πρώτο", "speaker": "A"},
            {"start": 3.0, "end": 4.0, "text": "δεύτερο"},
        ],
        "usage": {"input_tokens": 11},
    }
    direct_openai, direct_create, direct_requests = _fake_openai(response)
    orchestrated_openai, orchestrated_create, orchestrated_requests = _fake_openai(response)
    direct = await OpenAITranscriptionClient(
        settings,
        client=direct_openai,
    ).transcribe_diarized([direct_chunk], language="el")
    processor = StubAudioProcessor([])
    registered: list[Path] = []

    orchestrated = await TranscriptionOrchestrator(
        settings=settings,
        audio_processor=processor,  # type: ignore[arg-type]
        client_factory=lambda: OpenAITranscriptionClient(
            settings,
            client=orchestrated_openai,
        ),
    ).transcribe(
        source_path=prepared,
        audio_info=_audio_info(duration_seconds=30),
        context=_context(tmp_path, diarized=True, registered=registered),
    )

    assert orchestrated.model == direct.model
    assert orchestrated.language == direct.language
    assert orchestrated.prompt_version is direct.prompt_version is None
    assert orchestrated.text == direct.text
    assert orchestrated.usage == direct.usage
    assert _segment_snapshot(orchestrated) == _segment_snapshot(direct)
    assert orchestrated_create.await_count == direct_create.await_count == 1
    assert orchestrated_requests == direct_requests
    assert processor.split_calls == []
    assert registered == []
    assert "prompt" not in orchestrated_requests[0]


@pytest.mark.asyncio
async def test_legacy_segmenter_preserves_large_diarized_480_second_policy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "large.wav"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    chunks = [
        _chunk(tmp_path / "chunks" / "one.wav", 0, 480),
        _chunk(tmp_path / "chunks" / "two.wav", 480, 600),
    ]
    processor = StubAudioProcessor(chunks)
    registered: list[Path] = []
    track = AudioTrack(
        track_id="legacy-diarized",
        source_path=source,
        diarized=True,
        audio_variant="legacy-mono",
    )

    result = await LegacyFixedAudioSegmenter(  # type: ignore[arg-type]
        processor
    ).segment(
        track,
        audio_info=_audio_info(
            duration_seconds=600,
            size_bytes=source.stat().st_size,
        ),
        destination_dir=tmp_path / "temporary",
        max_upload_bytes=1024 * 1024,
        register_temporary_file=registered.append,
    )

    assert processor.split_calls == [(source, tmp_path / "temporary", 600, 480)]
    assert [(chunk.start_seconds, chunk.end_seconds) for chunk in result] == [
        (0, 480),
        (480, 600),
    ]
    assert [chunk.hard_cut for chunk in result] == [True, False]
    assert registered == [chunk.path for chunk in chunks]


@pytest.mark.asyncio
@pytest.mark.parametrize("diarized", [False, True])
async def test_orchestrator_preserves_cancellation_between_chunk_uploads(
    tmp_path: Path,
    diarized: bool,
) -> None:
    max_upload_bytes = 1024 * 1024 if diarized else None
    settings = _settings(tmp_path, max_upload_bytes=max_upload_bytes)
    duration = 600 if diarized else 30
    boundary = 480 if diarized else 15
    chunks = [
        _chunk(tmp_path / "chunks" / "one.wav", 0, boundary),
        _chunk(tmp_path / "chunks" / "two.wav", boundary, duration),
    ]
    response: object = (
        {"segments": [{"start": 0.0, "end": 1.0, "text": "one", "speaker": "A"}]}
        if diarized
        else SimpleNamespace(text="one", usage={})
    )
    openai, create, _ = _fake_openai(
        response,
        response,
    )
    checks = 0
    source_path = tmp_path / "prepared.wav"
    source_path.write_bytes(
        b"x" * (max_upload_bytes + 1)
        if max_upload_bytes is not None
        else b"RIFF\x00\x00\x00\x00WAVE"
    )

    async def cancellation_check() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(TranscriptionCancelledError) as raised:
        await TranscriptionOrchestrator(
            settings=settings,
            audio_processor=StubAudioProcessor(chunks),  # type: ignore[arg-type]
            client_factory=lambda: OpenAITranscriptionClient(
                settings,
                client=openai,
            ),
        ).transcribe(
            source_path=source_path,
            audio_info=_audio_info(duration_seconds=duration),
            context=_context(tmp_path, diarized=diarized, registered=[]),
            cancellation_check=cancellation_check,
        )

    assert raised.value.category == "cancelled"
    assert checks == 2
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_does_not_wrap_transcription_errors(
    tmp_path: Path,
) -> None:
    failure = TranscriptionError(
        "Transcription service is busy. Retry later.",
        "openai_rate_limit",
    )
    chunk = _chunk(tmp_path / "chunk.wav", 0, 5)

    class FailingClient:
        async def __aenter__(self) -> FailingClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def transcribe_isolated(self, *_args: object, **_kwargs: object) -> None:
            raise failure

    with pytest.raises(TranscriptionError) as raised:
        await TranscriptionOrchestrator(
            settings=_settings(tmp_path),
            audio_processor=StubAudioProcessor([chunk]),  # type: ignore[arg-type]
            client_factory=FailingClient,  # type: ignore[arg-type]
        ).transcribe(
            source_path=tmp_path / "prepared.wav",
            audio_info=_audio_info(duration_seconds=5),
            context=_context(tmp_path, diarized=False, registered=[]),
        )

    assert raised.value is failure
    assert raised.value.category == "openai_rate_limit"


@pytest.mark.asyncio
async def test_legacy_quality_adapter_is_a_true_pass_through(tmp_path: Path) -> None:
    track = AudioTrack(
        track_id="legacy",
        source_path=tmp_path / "prepared.wav",
        diarized=False,
    )

    result = await LegacyPassThroughQualityProcessor().prepare(
        track,
        audio_info=_audio_info(duration_seconds=10),
    )

    assert result is track


def test_legacy_planner_keeps_the_worker_selected_topology(tmp_path: Path) -> None:
    source = tmp_path / "prepared.wav"

    plan = LegacyAudioPlanner().plan(
        source_path=source,
        audio_info=_audio_info(duration_seconds=10),
        diarized=False,
        channel_index=1,
        operator_id="operator-1",
        attribution_status="confirmed_by_pbx",
        audio_variant="legacy-operator-channel",
    )

    assert plan.mode == "legacy"
    assert len(plan.tracks) == 1
    assert plan.tracks[0] == AudioTrack(
        track_id="legacy-operator",
        source_path=source,
        channel_index=1,
        operator_id="operator-1",
        attribution_status="confirmed_by_pbx",
        audio_variant="legacy-operator-channel",
        diarized=False,
    )


class CapturingSession:
    def __init__(self) -> None:
        self.added: list[object] = []

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        return None


def _transcript_state() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4())


def _persisted_segment_snapshot(segment: object) -> tuple:
    return (
        segment.call_leg_id,
        segment.operator_id,
        segment.speaker_label,
        segment.speaker_source,
        segment.start_seconds,
        segment.end_seconds,
        segment.original_text,
        segment.normalized_text,
        segment.confidence,
        segment.transcription_model,
        segment.sequence_number,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_operator", [False, True])
async def test_direct_and_orchestrated_results_persist_equivalent_segments(
    with_operator: bool,
) -> None:
    direct = TranscriptionResult(
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="prompt-v1",
        processing_duration_seconds=1.25,
        segments=[
            TranscribedSegment(
                start_seconds=1.23456,
                end_seconds=2.34567,
                text=" Γεια σας ",
                speaker_label="chunk-1:A",
                confidence=0.75,
            ),
            TranscribedSegment(
                start_seconds=3.0,
                end_seconds=4.0,
                text="δεύτερο",
                speaker_label="chunk-1:B",
            ),
        ],
        usage={"totals": {"input_tokens": 9}},
        diarized=not with_operator,
    )
    hypotheses = tuple(
        ChunkHypothesis(
            track_id="legacy",
            chunk_index=0,
            start_seconds=segment.start_seconds,
            end_seconds=segment.end_seconds,
            text=segment.text,
            speaker_label=segment.speaker_label,
            confidence=segment.confidence,
        )
        for segment in direct.segments
    )
    orchestrated = OrchestratedTranscriptionResult(
        mode="legacy",
        model=direct.model,
        language=direct.language,
        prompt_version=direct.prompt_version,
        processing_duration_seconds=direct.processing_duration_seconds,
        segments=hypotheses,
        tracks=(
            TrackTranscriptionResult(
                track_id="legacy",
                model=direct.model,
                language=direct.language,
                prompt_version=direct.prompt_version,
                processing_duration_seconds=direct.processing_duration_seconds,
                hypotheses=hypotheses,
                usage=direct.usage,
                diarized=direct.diarized,
            ),
        ),
        usage=direct.usage,
        diarized=direct.diarized,
    )
    call = SimpleNamespace(id=uuid4())
    operator = SimpleNamespace(id=uuid4(), display_name="Operator") if with_operator else None
    call_leg_id = uuid4() if with_operator else None
    direct_transcript = _transcript_state()
    orchestrated_transcript = _transcript_state()
    direct_session = CapturingSession()
    orchestrated_session = CapturingSession()

    await _persist_transcription_result(
        direct_session,  # type: ignore[arg-type]
        direct_transcript,  # type: ignore[arg-type]
        direct,
        call,  # type: ignore[arg-type]
        operator,  # type: ignore[arg-type]
        call_leg_id,
        10.0,
    )
    await _persist_transcription_result(
        orchestrated_session,  # type: ignore[arg-type]
        orchestrated_transcript,  # type: ignore[arg-type]
        orchestrated,
        call,  # type: ignore[arg-type]
        operator,  # type: ignore[arg-type]
        call_leg_id,
        10.0,
    )

    transcript_fields = (
        "model",
        "language",
        "prompt_version",
        "processing_duration_seconds",
        "audio_duration_seconds",
        "api_usage",
        "status",
        "original_text",
        "normalized_text",
        "error_category",
        "error_message",
    )
    assert {field: getattr(direct_transcript, field) for field in transcript_fields} == {
        field: getattr(orchestrated_transcript, field) for field in transcript_fields
    }
    assert [_persisted_segment_snapshot(segment) for segment in direct_session.added] == [
        _persisted_segment_snapshot(segment) for segment in orchestrated_session.added
    ]
    assert direct_session.added[0].start_seconds == Decimal("1.235")
    assert direct_session.added[0].speaker_source is (
        SpeakerSource.STEREO_CHANNEL if with_operator else SpeakerSource.OPENAI_DIARIZATION
    )


def test_typed_chunk_contract_accepts_future_quality_evidence(tmp_path: Path) -> None:
    chunk = SpeechChunk(
        track_id="track-1",
        chunk_index=3,
        path=tmp_path / "chunk.wav",
        start_seconds=10,
        end_seconds=20,
        hard_cut=True,
        audio_variant="legacy",
    )

    assert chunk.chunk_index == 3
    assert chunk.hard_cut is True


def test_orchestrated_text_preserves_existing_keyword_matching_semantics() -> None:
    direct = TranscriptionResult(
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="prompt-v1",
        processing_duration_seconds=0.1,
        segments=[
            TranscribedSegment(
                start_seconds=0,
                end_seconds=5,
                text="Ζητώ επιστροφή χρημάτων σήμερα.",
                speaker_label="Operator",
            )
        ],
    )
    hypothesis = ChunkHypothesis(
        track_id="legacy",
        chunk_index=0,
        start_seconds=0,
        end_seconds=5,
        text=direct.segments[0].text,
        speaker_label=direct.segments[0].speaker_label,
    )
    track = TrackTranscriptionResult(
        track_id="legacy",
        model=direct.model,
        language=direct.language,
        prompt_version=direct.prompt_version,
        processing_duration_seconds=direct.processing_duration_seconds,
        hypotheses=(hypothesis,),
    )
    orchestrated = OrchestratedTranscriptionResult(
        mode="legacy",
        model=track.model,
        language=track.language,
        prompt_version=track.prompt_version,
        processing_duration_seconds=track.processing_duration_seconds,
        segments=track.hypotheses,
        tracks=(track,),
    )
    definitions = [
        KeywordDefinition(
            id="refund",
            phrase="επιστροφή χρημάτων",
        )
    ]

    assert match_text(orchestrated.text, definitions) == match_text(
        direct.text,
        definitions,
    )


@pytest.mark.parametrize(
    "relative_path",
    [
        "services/audio/quality.py",
        "services/audio/segmentation.py",
        "services/transcription/confidence.py",
        "services/transcription/merge.py",
        "services/transcription/orchestrator.py",
        "services/transcription/planning.py",
        "services/transcription/prompt.py",
        "services/transcription/types.py",
    ],
)
def test_architecture_modules_keep_orm_out_of_low_level_services(
    relative_path: str,
) -> None:
    source = (Path(__file__).parents[1] / "app" / relative_path).read_text(encoding="utf-8")

    assert "sqlalchemy" not in source
    assert "app.models" not in source
    assert "NotImplementedError" not in source
