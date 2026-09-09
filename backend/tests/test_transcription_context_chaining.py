from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self
import wave

import pytest

from app.core.config import Settings
from app.services.audio import AudioChunk, AudioInfo
from app.services.transcription.client import (
    TranscribedSegment,
    TranscriptionCancelledError,
    TranscriptionResult,
)
from app.services.transcription.orchestrator import (
    TranscriptionContext,
    TranscriptionOrchestrator,
)
from app.services.transcription.prompt import (
    MAX_PREVIOUS_CONTEXT_CHARACTERS,
    LegacyVocabularyPromptBuilder,
    PromptPlan,
)
from app.services.transcription.types import AudioPlan, AudioTrack, SpeechChunk


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=tmp_path,
        OPENAI_API_KEY="test-only-key",
    )


def _audio_info(*, channel_count: int, duration_seconds: float = 40.0) -> AudioInfo:
    return AudioInfo(
        codec_name="pcm_s16le",
        format_name="wav",
        duration_seconds=duration_seconds,
        channel_count=channel_count,
        sample_rate_hz=16_000,
        bit_rate_bps=256_000,
        size_bytes=100,
        sha256_checksum="a" * 64,
    )


def _operator_plan(tmp_path: Path) -> AudioPlan:
    return AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=tmp_path / "source.wav",
                channel_index=0,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Operator One",
                speaker_source="stereo_channel",
            ),
        ),
        operator_channel=0,
        stereo_separated=True,
        attribution_status="confirmed_by_pbx",
    )


def _dual_plan(tmp_path: Path) -> AudioPlan:
    return AudioPlan(
        mode="dual_channel",
        tracks=tuple(
            AudioTrack(
                track_id=f"channel-{channel}",
                source_path=tmp_path / "source.wav",
                channel_index=channel,
                attribution_status="channel_unknown",
                audio_variant=f"topology-channel-{channel}",
                speaker_label=f"Channel {'A' if channel == 0 else 'B'}",
                speaker_source="stereo_channel",
            )
            for channel in (0, 1)
        ),
        stereo_separated=True,
        attribution_status="channel_unknown",
    )


def _mono_plan(tmp_path: Path) -> AudioPlan:
    return AudioPlan(
        mode="mono_diarization",
        tracks=(
            AudioTrack(
                track_id="mono-diarization",
                source_path=tmp_path / "source.wav",
                attribution_status="anonymous_diarization",
                audio_variant="topology-mono",
                diarized=True,
                speaker_source="openai_diarization",
            ),
        ),
        attribution_status="anonymous_diarization",
    )


def _legacy_plan(tmp_path: Path) -> AudioPlan:
    return AudioPlan(
        mode="legacy",
        tracks=(
            AudioTrack(
                track_id="legacy-operator",
                source_path=tmp_path / "prepared.wav",
                channel_index=0,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
                audio_variant="legacy-operator-channel",
            ),
        ),
        operator_channel=0,
        attribution_status="confirmed_by_pbx",
    )


def _context(
    tmp_path: Path,
    *,
    diarized: bool = False,
    vocabulary: tuple[str, ...] = (),
    language: str = "el",
) -> TranscriptionContext:
    return TranscriptionContext(
        diarized=diarized,
        language=language,
        vocabulary=vocabulary,
        temporary_directory=tmp_path / "temporary",
    )


def _chunk(
    tmp_path: Path,
    track_id: str,
    chunk_index: int,
    start_seconds: float,
    end_seconds: float,
    *,
    hard_cut: bool = False,
    overlap_before_ms: int = 0,
) -> SpeechChunk:
    return SpeechChunk(
        track_id=track_id,
        chunk_index=chunk_index,
        path=tmp_path / f"{track_id}-{chunk_index}.wav",
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        hard_cut=hard_cut,
        overlap_before_ms=overlap_before_ms,
        audio_variant=f"{track_id}-pcm16",
    )


class StubAudioProcessor:
    async def extract_channel(
        self,
        source: Path,
        destination: Path,
        channel_index: int,
    ) -> None:
        del source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"channel-{channel_index}".encode())

    async def convert_to_mono(self, source: Path, destination: Path) -> None:
        del source
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(b"\x00\x00" * 30 * 16_000)

    async def extract_pcm_wav_range(
        self,
        source: Path,
        destination: Path,
        *,
        start_sample: int,
        end_sample: int,
        sample_rate_hz: int = 16_000,
    ) -> Path:
        del source
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate_hz)
            writer.writeframes(b"\x00\x00" * (end_sample - start_sample))
        return destination


class ScriptedSegmenter:
    def __init__(self, chunks_by_track: dict[str, tuple[SpeechChunk, ...]]) -> None:
        self.chunks_by_track = chunks_by_track

    async def segment(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
        destination_dir: Path,
        max_upload_bytes: int,
        register_temporary_file: Callable[[Path], None] | None,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[SpeechChunk, ...]:
        del (
            audio_info,
            destination_dir,
            max_upload_bytes,
            register_temporary_file,
            cancellation_check,
        )
        return self.chunks_by_track[track.track_id]


@dataclass(frozen=True, slots=True)
class IsolatedRequest:
    chunks: tuple[AudioChunk, ...]
    vocabulary: tuple[str, ...]
    language: str | None
    prompt_plan: PromptPlan | None
    request_logprobs: bool = False


class ScriptedClient:
    def __init__(
        self,
        isolated_responses: list[tuple[str, ...]] | None = None,
        diarized_responses: list[tuple[str, ...]] | None = None,
    ) -> None:
        self._isolated_responses = iter(isolated_responses or [])
        self._diarized_responses = iter(diarized_responses or [])
        self.isolated_requests: list[IsolatedRequest] = []
        self.diarized_requests: list[tuple[AudioChunk, ...]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def transcribe_isolated(
        self,
        chunks: list[AudioChunk],
        vocabulary: list[str],
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
        prompt_plan: PromptPlan | None = None,
        request_logprobs: bool = False,
    ) -> TranscriptionResult:
        if should_cancel is not None and await should_cancel():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        texts = next(self._isolated_responses)
        if len(texts) != len(chunks):
            raise AssertionError("Each scripted isolated chunk requires one response.")
        self.isolated_requests.append(
            IsolatedRequest(
                chunks=tuple(chunks),
                vocabulary=tuple(vocabulary),
                language=language,
                prompt_plan=prompt_plan,
                request_logprobs=request_logprobs,
            )
        )
        return TranscriptionResult(
            model="gpt-4o-transcribe",
            language=language or "el",
            prompt_version=prompt_plan.version if prompt_plan is not None else None,
            processing_duration_seconds=0.1,
            segments=[
                TranscribedSegment(
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=text,
                    speaker_label="Provider",
                    source_chunk_index=chunk.chunk_index,
                )
                for chunk, text in zip(chunks, texts, strict=True)
                if text
            ],
            usage={
                "chunks": [{"input_tokens": 1} for _ in chunks],
                "totals": {"input_tokens": len(chunks)},
            },
        )

    async def transcribe_diarized(
        self,
        chunks: list[AudioChunk],
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> TranscriptionResult:
        if should_cancel is not None and await should_cancel():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        texts = next(self._diarized_responses)
        if len(texts) != len(chunks):
            raise AssertionError("Each scripted diarized chunk requires one response.")
        self.diarized_requests.append(tuple(chunks))
        return TranscriptionResult(
            model="gpt-4o-transcribe-diarize",
            language=language or "el",
            prompt_version=None,
            processing_duration_seconds=0.1,
            segments=[
                TranscribedSegment(
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=text,
                    speaker_label=f"Speaker {position}",
                    source_chunk_index=chunk.chunk_index,
                )
                for position, (chunk, text) in enumerate(
                    zip(chunks, texts, strict=True),
                    start=1,
                )
                if text
            ],
            usage={
                "chunks": [{"input_tokens": 1} for _ in chunks],
                "totals": {"input_tokens": len(chunks)},
            },
            diarized=True,
        )

    async def transcribe_diarized_complete(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> TranscriptionResult:
        if should_cancel is not None and await should_cancel():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        texts = next(self._diarized_responses)
        self.diarized_requests.append((chunk,))
        turn_duration = (chunk.end_seconds - chunk.start_seconds) / len(texts)
        return TranscriptionResult(
            model="gpt-4o-transcribe-diarize",
            language=language or "el",
            prompt_version=None,
            processing_duration_seconds=0.1,
            segments=[
                TranscribedSegment(
                    start_seconds=chunk.start_seconds + position * turn_duration,
                    end_seconds=chunk.start_seconds + (position + 1) * turn_duration,
                    text=text,
                    speaker_label=chr(ord("A") + position % 2),
                    source_chunk_index=chunk.chunk_index,
                )
                for position, text in enumerate(texts)
                if text
            ],
            usage={
                "chunks": [{"input_tokens": 1}],
                "totals": {"input_tokens": 1},
            },
            diarized=True,
        )


def _orchestrator(
    tmp_path: Path,
    chunks_by_track: dict[str, tuple[SpeechChunk, ...]],
    client: ScriptedClient,
) -> TranscriptionOrchestrator:
    return TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=StubAudioProcessor(),  # type: ignore[arg-type]
        segmenter=ScriptedSegmenter(chunks_by_track),  # type: ignore[arg-type]
        client_factory=lambda: client,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_v2_first_chunk_has_no_context_and_second_gets_bounded_same_track_text(
    tmp_path: Path,
) -> None:
    chunks = (
        _chunk(tmp_path, "operator-channel", 0, 0.0, 10.0),
        _chunk(tmp_path, "operator-channel", 1, 10.0, 20.0),
    )
    accepted = "DISCARD_PREFIX-" + ("x" * 600) + "-KEEP_TAIL"
    client = ScriptedClient([(accepted,), ("second",)])

    await _orchestrator(
        tmp_path,
        {"operator-channel": chunks},
        client,
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=2),
        context=_context(tmp_path, language="en"),
        plan=_operator_plan(tmp_path),
    )

    first_prompt = client.isolated_requests[0].prompt_plan
    second_prompt = client.isolated_requests[1].prompt_plan
    assert first_prompt is not None
    assert second_prompt is not None
    assert first_prompt.previous_context_characters == 0
    assert second_prompt.previous_context_characters == MAX_PREVIOUS_CONTEXT_CHARACTERS
    assert [request.language for request in client.isolated_requests] == ["el", "el"]
    assert "KEEP_TAIL" in second_prompt.text
    assert "DISCARD_PREFIX" not in second_prompt.text


@pytest.mark.asyncio
async def test_exact_hard_cut_overlap_is_trimmed_before_seeding_later_context(
    tmp_path: Path,
) -> None:
    chunks = (
        _chunk(
            tmp_path,
            "operator-channel",
            0,
            0.0,
            10.0,
            hard_cut=True,
        ),
        _chunk(
            tmp_path,
            "operator-channel",
            1,
            9.2,
            20.0,
            overlap_before_ms=800,
        ),
        _chunk(tmp_path, "operator-channel", 2, 20.0, 30.0),
    )
    client = ScriptedClient(
        [
            ("LEAD OVERLAP_ALPHA OVERLAP_BETA",),
            ("OVERLAP_ALPHA OVERLAP_BETA TRIMMED_CONTEXT_ONLY",),
            ("FINAL",),
        ]
    )

    result = await _orchestrator(
        tmp_path,
        {"operator-channel": chunks},
        client,
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=2),
        context=_context(tmp_path),
        plan=_operator_plan(tmp_path),
    )

    third_prompt = client.isolated_requests[2].prompt_plan
    assert third_prompt is not None
    assert "TRIMMED_CONTEXT_ONLY" in third_prompt.text
    assert "OVERLAP_ALPHA" not in third_prompt.text
    assert [segment.text for segment in result.segments] == [
        "LEAD OVERLAP_ALPHA OVERLAP_BETA",
        "TRIMMED_CONTEXT_ONLY",
        "FINAL",
    ]


@pytest.mark.asyncio
async def test_dual_channel_b_never_receives_channel_a_context(
    tmp_path: Path,
) -> None:
    chunks_by_track = {
        "channel-0": (
            _chunk(tmp_path, "channel-0", 0, 0.0, 12.0),
            _chunk(tmp_path, "channel-0", 1, 20.0, 32.0),
        ),
        "channel-1": (
            _chunk(tmp_path, "channel-1", 0, 5.0, 17.0),
            _chunk(tmp_path, "channel-1", 1, 25.0, 37.0),
        ),
    }
    client = ScriptedClient(
        [
            ("CHANNEL_A_PRIVATE_CONTEXT",),
            ("",),
            ("CHANNEL_B_FIRST",),
            ("CHANNEL_B_SECOND",),
        ]
    )

    result = await _orchestrator(tmp_path, chunks_by_track, client).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=2),
        context=_context(tmp_path),
        plan=_dual_plan(tmp_path),
    )

    channel_b_first = client.isolated_requests[2].prompt_plan
    channel_b_second = client.isolated_requests[3].prompt_plan
    assert channel_b_first is not None
    assert channel_b_second is not None
    assert channel_b_first.track_id == "channel-1"
    assert channel_b_first.previous_context_characters == 0
    assert "CHANNEL_A_PRIVATE_CONTEXT" not in channel_b_first.text
    assert "CHANNEL_A_PRIVATE_CONTEXT" not in channel_b_second.text
    assert "CHANNEL_B_FIRST" in channel_b_second.text
    assert [(segment.start_seconds, segment.text) for segment in result.segments] == [
        (0.0, "CHANNEL_A_PRIVATE_CONTEXT"),
        (5.0, "CHANNEL_B_FIRST"),
        (25.0, "CHANNEL_B_SECOND"),
    ]


@pytest.mark.asyncio
async def test_cancellation_between_v2_chunks_prevents_next_provider_request(
    tmp_path: Path,
) -> None:
    chunks = (
        _chunk(tmp_path, "operator-channel", 0, 0.0, 10.0),
        _chunk(tmp_path, "operator-channel", 1, 10.0, 20.0),
    )
    client = ScriptedClient([("first",), ("must-not-be-used",)])
    checks = 0

    async def cancellation_check() -> bool:
        nonlocal checks
        checks += 1
        # One check occurs before track extraction, then one before each
        # provider-equivalent request in the scripted client.
        return checks >= 3

    with pytest.raises(TranscriptionCancelledError):
        await _orchestrator(
            tmp_path,
            {"operator-channel": chunks},
            client,
        ).transcribe(
            source_path=tmp_path / "source.wav",
            audio_info=_audio_info(channel_count=2),
            context=_context(tmp_path),
            plan=_operator_plan(tmp_path),
            cancellation_check=cancellation_check,
        )

    assert len(client.isolated_requests) == 1
    assert client.isolated_requests[0].chunks[0].chunk_index == 0


@pytest.mark.asyncio
async def test_v2_mono_uses_diarization_then_prompted_standard_refinement(
    tmp_path: Path,
) -> None:
    client = ScriptedClient(
        isolated_responses=[("refined mono text",)],
        diarized_responses=[("rough mono text",)],
    )

    result = await _orchestrator(
        tmp_path,
        {},
        client,
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=1, duration_seconds=30.0),
        context=_context(tmp_path, diarized=True),
        plan=_mono_plan(tmp_path),
    )

    assert len(client.isolated_requests) == 1
    assert len(client.diarized_requests) == 1
    assert result.diarized is True
    assert result.prompt_version is not None
    assert result.segments[0].text == "refined mono text"
    assert result.segments[0].speaker_label == "A"
    assert result.segments[0].operator_id is None
    assert len(result.attempts) == 1
    assert result.attempts[0].selected is True
    prompt = client.isolated_requests[0].prompt_plan
    assert prompt is not None
    assert "Τρέχων ανώνυμος ομιλητής: A" in prompt.text
    assert result.quality_summary is not None
    assert result.quality_summary["fallback_span_count"] == 0


@pytest.mark.asyncio
async def test_legacy_isolated_track_remains_one_unchanged_batch_call(
    tmp_path: Path,
) -> None:
    chunks = (
        _chunk(tmp_path, "legacy-operator", 0, 0.0, 15.0),
        _chunk(tmp_path, "legacy-operator", 1, 15.0, 30.0),
    )
    vocabulary = ("  Yeastar ", "Γιώργος", "yeastar")
    expected_prompt = LegacyVocabularyPromptBuilder().build(vocabulary)
    client = ScriptedClient([("legacy one", "legacy two")])

    result = await _orchestrator(
        tmp_path,
        {"legacy-operator": chunks},
        client,
    ).transcribe(
        source_path=tmp_path / "prepared.wav",
        audio_info=_audio_info(channel_count=1, duration_seconds=30.0),
        context=_context(tmp_path, vocabulary=vocabulary),
        plan=_legacy_plan(tmp_path),
    )

    assert len(client.isolated_requests) == 1
    request = client.isolated_requests[0]
    assert len(request.chunks) == 2
    assert request.vocabulary == vocabulary
    assert request.prompt_plan == expected_prompt
    assert client.diarized_requests == []
    assert [segment.text for segment in result.segments] == [
        "legacy one",
        "legacy two",
    ]
