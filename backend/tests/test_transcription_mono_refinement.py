from __future__ import annotations

import shutil
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pytest

from app.core.config import Settings
from app.services.audio import AudioChunk, AudioInfo
from app.services.transcription.client import (
    TranscribedSegment,
    TranscriptionCancelledError,
    TranscriptionResponseEvidence,
    TranscriptionResult,
)
from app.services.transcription.mono import MonoRefinementPolicy
from app.services.transcription.orchestrator import (
    PartialTranscriptionError,
    PartialTranscriptionCancelledError,
    TranscriptionContext,
    TranscriptionOrchestrator,
)
from app.services.transcription.prompt import PromptPlan
from app.services.transcription.types import AudioPlan, AudioTrack, TokenLogprob


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=tmp_path,
        OPENAI_API_KEY="test-only-key",
    )


def _audio_info(duration_seconds: float = 60.0) -> AudioInfo:
    return AudioInfo(
        codec_name="pcm_s16le",
        format_name="wav",
        duration_seconds=duration_seconds,
        channel_count=1,
        sample_rate_hz=16_000,
        bit_rate_bps=256_000,
        size_bytes=100,
        sha256_checksum="a" * 64,
    )


def _plan(tmp_path: Path, duration_seconds: float = 60.0) -> AudioPlan:
    return AudioPlan(
        mode="mono_diarization",
        tracks=(
            AudioTrack(
                track_id="mono-diarization",
                source_path=tmp_path / "source.wav",
                operator_id=None,
                attribution_status="anonymous_diarization",
                audio_variant="topology-mono",
                diarized=True,
                speaker_source="openai_diarization",
                duration_seconds=duration_seconds,
            ),
        ),
        attribution_status="anonymous_diarization",
        reason="mono-source",
    )


class MonoAudioProcessor:
    def __init__(self, duration_seconds: float = 60.0) -> None:
        self.duration_seconds = duration_seconds
        self.extracted_ranges: list[tuple[int, int]] = []

    @staticmethod
    def _write_silence(path: Path, frame_count: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(b"\x00\x00" * frame_count)

    async def convert_to_mono(self, source: Path, destination: Path) -> Path:
        del source
        self._write_silence(destination, round(self.duration_seconds * 16_000))
        return destination

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
        assert sample_rate_hz == 16_000
        self.extracted_ranges.append((start_sample, end_sample))
        self._write_silence(destination, end_sample - start_sample)
        return destination

    async def apply_lossless_audio_filter(
        self,
        source: Path,
        destination: Path,
        *,
        audio_filter: str,
    ) -> Path:
        del audio_filter
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return destination

    @staticmethod
    def partial_wav_path(destination: Path) -> Path:
        return destination.with_suffix(".part.wav")


@dataclass(frozen=True, slots=True)
class StandardScript:
    text: str
    logprobs: tuple[float, ...] = (-0.1,)
    logprobs_available: bool = True


class MonoClient:
    def __init__(
        self,
        pass1_segments: list[TranscribedSegment],
        pass2_scripts: list[StandardScript | Exception],
        *,
        model: str = "gpt-4o-transcribe",
    ) -> None:
        self.pass1_segments = pass1_segments
        self.pass2_scripts = iter(pass2_scripts)
        self.model = model
        self.pass1_requests: list[AudioChunk] = []
        self.pass2_requests: list[tuple[AudioChunk, PromptPlan]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def transcribe_diarized_complete(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> TranscriptionResult:
        if should_cancel is not None and await should_cancel():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        self.pass1_requests.append(chunk)
        return TranscriptionResult(
            model="gpt-4o-transcribe-diarize",
            language=language or "el",
            prompt_version=None,
            processing_duration_seconds=0.2,
            segments=self.pass1_segments,
            usage={
                "chunks": [{"input_tokens": 11, "output_tokens": 4}],
                "totals": {"input_tokens": 11, "output_tokens": 4},
            },
            diarized=True,
        )

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
        del vocabulary
        if should_cancel is not None and await should_cancel():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        assert len(chunks) == 1
        assert language == "el"
        assert prompt_plan is not None
        assert request_logprobs is True
        script = next(self.pass2_scripts)
        self.pass2_requests.append((chunks[0], prompt_plan))
        if isinstance(script, Exception):
            raise script
        tokens = tuple(
            TokenLogprob(token=f"token-{index}", logprob=value)
            for index, value in enumerate(script.logprobs)
        )
        evidence = TranscriptionResponseEvidence(
            source_chunk_index=chunks[0].chunk_index,
            response_text=script.text,
            token_logprobs=tokens,
            logprobs_available=script.logprobs_available,
        )
        return TranscriptionResult(
            model=self.model,
            language="el",
            prompt_version=prompt_plan.version,
            processing_duration_seconds=0.1,
            segments=(
                [
                    TranscribedSegment(
                        start_seconds=chunks[0].start_seconds,
                        end_seconds=chunks[0].end_seconds,
                        text=script.text,
                        speaker_label="Provider",
                        source_chunk_index=chunks[0].chunk_index,
                    )
                ]
                if script.text
                else []
            ),
            usage={
                "chunks": [{"input_tokens": 3, "output_tokens": 2}],
                "totals": {"input_tokens": 3, "output_tokens": 2},
            },
            response_evidence=(evidence,),
        )


def _orchestrator(
    tmp_path: Path,
    client: MonoClient,
    processor: MonoAudioProcessor,
    *,
    policy: MonoRefinementPolicy | None = None,
) -> TranscriptionOrchestrator:
    return TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=processor,  # type: ignore[arg-type]
        mono_refinement_policy=policy or MonoRefinementPolicy(),
        client_factory=lambda: client,  # type: ignore[arg-type]
    )


def _context(tmp_path: Path) -> TranscriptionContext:
    return TranscriptionContext(
        diarized=True,
        language="el",
        vocabulary=(),
        temporary_directory=tmp_path / "temporary",
    )


@pytest.mark.asyncio
async def test_gpt_transcribe_recognizes_complete_conversation_once(tmp_path: Path) -> None:
    from app.workers.pipeline import _validate_transcription_attempts

    text = "Ναι, δεν ξέρω. Είναι μέσα στο service. Εντάξει, ευχαριστώ."
    client = MonoClient(
        [TranscribedSegment(1, 3, "Ναι δεν ξέρω", "A"),
         TranscribedSegment(3, 3.1, "δεν ξέρω", "B"),
         TranscribedSegment(3.1, 3.3, "δεν ξέρω", "A"),
         TranscribedSegment(3.3, 6, "Είναι μέσα στο service", "B")],
        [StandardScript(text, (), False), StandardScript(text.replace("ξέρω", "γνωρίζω"), (), False)], model="gpt-transcribe",
    )
    processor = MonoAudioProcessor()
    orchestrator = _orchestrator(tmp_path, client, processor)
    orchestrator.settings = orchestrator.settings.model_copy(update={"OPENAI_TRANSCRIPTION_MODEL": "gpt-transcribe"})
    result = await orchestrator.transcribe(
        source_path=tmp_path / "source.wav", audio_info=_audio_info(),
        context=_context(tmp_path), plan=_plan(tmp_path),
    )
    assert result.text == text
    assert result.model == "gpt-transcribe"
    assert processor.extracted_ranges == []
    assert len(client.pass2_requests) == 2
    chunk, prompt = client.pass2_requests[0]
    assert (chunk.start_seconds, chunk.end_seconds) == (0, 60)
    assert prompt.previous_context_characters == 0
    assert "όλους τους ομιλητές" in prompt.text
    assert len(result.attempts) == 2
    assert [attempt.selected for attempt in result.attempts] == [True, False]
    assert result.attempts[0].response_text == text
    assert result.segments[-1].speaker_source == "unknown"
    assert all(segment.operator_id is None for segment in result.segments)
    assert result.confidence_status == "unavailable"
    assert result.quality_summary["timestamps_approximate"] is True
    review = result.quality_summary["wording_review"]
    assert review["status"] == "complete"
    assert review["items"][0]["original_text"] == "ξέρω."
    assert review["items"][0]["alternative_text"] == "γνωρίζω."
    assert "transcription_disagreement" in result.segments[0].quality_flags
    assert result.usage["pass2_refinement"]["totals"]["input_tokens"] == 6
    _validate_transcription_attempts(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("script", [StandardScript("", (), False), RuntimeError("provider failure")])
async def test_continuous_failure_preserves_paid_history_and_does_not_publish_rough_text(
    tmp_path: Path, script: StandardScript | Exception,
) -> None:
    client = MonoClient([TranscribedSegment(1, 3, "rough text", "A")], [script])
    orchestrator = _orchestrator(tmp_path, client, MonoAudioProcessor())
    orchestrator.settings = orchestrator.settings.model_copy(update={"OPENAI_TRANSCRIPTION_MODEL": "gpt-transcribe"})
    with pytest.raises(PartialTranscriptionError) as caught:
        await orchestrator.transcribe(
            source_path=tmp_path / "source.wav", audio_info=_audio_info(),
            context=_context(tmp_path), plan=_plan(tmp_path),
        )
    assert caught.value.quality_summary["pass1_completed"] is True
    assert caught.value.quality_summary["status"] == "failed"
    assert caught.value.usage["pass1_diarization"]["totals"]["input_tokens"] == 11
    assert not any(attempt.selected for attempt in caught.value.attempts)


@pytest.mark.asyncio
async def test_continuous_cancellation_after_recognition_retains_attempt(tmp_path: Path) -> None:
    client = MonoClient(
        [TranscribedSegment(1, 3, "Ναι", "A")],
        [StandardScript("Ναι.", (), False)], model="gpt-transcribe",
    )
    orchestrator = _orchestrator(tmp_path, client, MonoAudioProcessor())
    orchestrator.settings = orchestrator.settings.model_copy(update={"OPENAI_TRANSCRIPTION_MODEL": "gpt-transcribe"})

    async def cancel_after_recognition() -> bool:
        return bool(client.pass2_requests)

    with pytest.raises(PartialTranscriptionCancelledError) as caught:
        await orchestrator.transcribe(
            source_path=tmp_path / "source.wav", audio_info=_audio_info(),
            context=_context(tmp_path), plan=_plan(tmp_path),
            cancellation_check=cancel_after_recognition,
        )
    assert len(caught.value.attempts) == 1
    assert caught.value.quality_summary["status"] == "cancelled"
    assert caught.value.usage["pass2_refinement"]["totals"]["input_tokens"] == 3


@pytest.mark.asyncio
async def test_failed_second_reading_keeps_primary_and_discloses_missing_check(tmp_path: Path) -> None:
    client = MonoClient(
        [TranscribedSegment(1, 3, "Original speech", "A")],
        [StandardScript("Original speech", (), False), RuntimeError("provider failure")],
        model="gpt-transcribe",
    )
    orchestrator = _orchestrator(tmp_path, client, MonoAudioProcessor())
    orchestrator.settings = orchestrator.settings.model_copy(update={"OPENAI_TRANSCRIPTION_MODEL": "gpt-transcribe"})
    result = await orchestrator.transcribe(
        source_path=tmp_path / "source.wav", audio_info=_audio_info(),
        context=_context(tmp_path), plan=_plan(tmp_path),
    )
    assert result.text == "Original speech"
    assert result.quality_summary["wording_review"]["status"] == "unavailable"
    assert len(result.attempts) == 1 and result.attempts[0].selected
    assert not list((tmp_path / "temporary").glob("retry-*.wav"))


@pytest.mark.asyncio
async def test_two_pass_mono_keeps_pass1_placement_and_uses_pass2_evidence(
    tmp_path: Path,
) -> None:
    pass1 = [
        TranscribedSegment(1.0, 3.0, "rough one", "A"),
        TranscribedSegment(3.2, 5.0, "rough two", "A"),
        TranscribedSegment(5.1, 7.0, "rough caller", "B"),
        TranscribedSegment(-1.0, 2.0, "invalid time", "A"),
        TranscribedSegment(8.0, 9.0, "invalid label", "Operator"),
    ]
    client = MonoClient(
        pass1,
        [
            StandardScript("refined operator", (-0.1, -0.2)),
            StandardScript("refined caller", (-0.3,)),
        ],
    )
    processor = MonoAudioProcessor()

    result = await _orchestrator(tmp_path, client, processor).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(),
        context=_context(tmp_path),
        plan=_plan(tmp_path),
    )

    assert len(client.pass1_requests) == 1
    assert client.pass1_requests[0].start_seconds == 0
    assert client.pass1_requests[0].end_seconds == 60
    assert processor.extracted_ranges == [
        (12_800, 83_200),
        (78_400, 124_800),
    ]
    assert [
        (
            segment.speaker_label,
            segment.start_seconds,
            segment.end_seconds,
            segment.text,
        )
        for segment in result.segments
    ] == [
        ("A", 1.0, 5.0, "refined operator"),
        ("B", 5.1, 7.0, "refined caller"),
    ]
    assert all(segment.operator_id is None for segment in result.segments)
    assert all(
        segment.speaker_source == "openai_diarization"
        for segment in result.segments
    )
    assert [segment.transcription_model for segment in result.segments] == [
        "gpt-4o-transcribe",
        "gpt-4o-transcribe",
    ]
    assert result.segments[0].mean_logprob == pytest.approx(-0.15)
    assert result.segments[1].mean_logprob == pytest.approx(-0.3)
    assert result.usage["pass1_diarization"]["totals"]["input_tokens"] == 11
    assert len(result.attempts) == 2
    assert all(attempt.selected for attempt in result.attempts)
    assert result.quality_summary is not None
    assert result.quality_summary["rejected_turn_count"] == 2
    assert result.quality_summary["fallback_span_count"] == 0
    second_prompt = client.pass2_requests[1][1].text
    assert "Τρέχων ανώνυμος ομιλητής: B" in second_prompt
    assert "A: refined operator" in second_prompt
    assert "μόνο ως συμφραζόμενο" in second_prompt


@pytest.mark.asyncio
async def test_failed_span_uses_rough_text_and_marks_transcript_degraded(
    tmp_path: Path,
) -> None:
    client = MonoClient(
        [
            TranscribedSegment(0.0, 8.0, "rough fallback", "A"),
            TranscribedSegment(8.0, 10.0, "rough refined", "B"),
        ],
        [RuntimeError("provider failed"), StandardScript("final refined")],
    )

    result = await _orchestrator(
        tmp_path,
        client,
        MonoAudioProcessor(),
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(),
        context=_context(tmp_path),
        plan=_plan(tmp_path),
    )

    fallback, refined = result.segments
    assert fallback.text == "rough fallback"
    assert fallback.transcription_model == "gpt-4o-transcribe-diarize"
    assert fallback.mean_logprob is None
    assert fallback.token_logprobs == ()
    assert {
        "refinement_failed",
        "human_review_recommended",
        "logprobs_unavailable",
        "rough_diarization_fallback",
    }.issubset(fallback.quality_flags)
    assert refined.text == "final refined"
    assert result.quality_summary is not None
    assert result.quality_summary["fallback_duration_ratio"] == pytest.approx(0.8)
    assert result.quality_summary["degraded"] is True
    assert result.quality_summary["status"] == "degraded"
    assert result.quality_summary["warning"]


@pytest.mark.asyncio
async def test_exactly_twenty_percent_fallback_is_not_degraded(
    tmp_path: Path,
) -> None:
    client = MonoClient(
        [
            TranscribedSegment(0.0, 2.0, "rough fallback", "A"),
            TranscribedSegment(2.0, 10.0, "rough refined", "B"),
        ],
        [RuntimeError("provider failed"), StandardScript("final refined")],
    )

    result = await _orchestrator(
        tmp_path,
        client,
        MonoAudioProcessor(),
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(),
        context=_context(tmp_path),
        plan=_plan(tmp_path),
    )

    assert result.quality_summary is not None
    assert result.quality_summary["fallback_duration_ratio"] == pytest.approx(0.2)
    assert result.quality_summary["degraded"] is False
    assert result.quality_summary["status"] == "completed_with_fallback"


@pytest.mark.asyncio
async def test_mono_retry_is_capped_at_one_normalized_attempt(
    tmp_path: Path,
) -> None:
    client = MonoClient(
        [TranscribedSegment(0.0, 5.0, "rough", "A")],
        [
            StandardScript("raw text", (-2.0, -2.0)),
            StandardScript("normalized text", (-0.2, -0.2)),
            AssertionError("a third attempt must never happen"),
        ],
    )

    result = await _orchestrator(
        tmp_path,
        client,
        MonoAudioProcessor(),
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(),
        context=_context(tmp_path),
        plan=_plan(tmp_path),
    )

    assert len(client.pass2_requests) == 2
    assert len(result.attempts) == 2
    assert [attempt.selected for attempt in result.attempts] == [False, True]
    assert result.segments[0].text == "normalized text"
    assert "normalized_retry_used" in result.segments[0].quality_flags


@pytest.mark.asyncio
async def test_refinement_budget_keeps_unprocessed_span_as_flagged_rough_text(
    tmp_path: Path,
) -> None:
    client = MonoClient(
        [
            TranscribedSegment(0.0, 5.0, "rough first", "A"),
            TranscribedSegment(5.0, 10.0, "rough second", "B"),
        ],
        [StandardScript("refined first")],
    )

    result = await _orchestrator(
        tmp_path,
        client,
        MonoAudioProcessor(),
        policy=MonoRefinementPolicy(max_refinement_spans=1),
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(),
        context=_context(tmp_path),
        plan=_plan(tmp_path),
    )

    assert len(client.pass2_requests) == 1
    assert result.segments[1].text == "rough second"
    assert "refinement_budget_exhausted" in result.segments[1].quality_flags


@pytest.mark.asyncio
async def test_cancellation_between_mono_spans_preserves_pass1_and_attempt_audit(
    tmp_path: Path,
) -> None:
    client = MonoClient(
        [
            TranscribedSegment(0.0, 5.0, "rough first", "A"),
            TranscribedSegment(5.0, 10.0, "rough second", "B"),
        ],
        [StandardScript("refined first"), StandardScript("must not run")],
    )
    checks = 0

    async def cancellation_check() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 5

    with pytest.raises(PartialTranscriptionCancelledError) as raised:
        await _orchestrator(
            tmp_path,
            client,
            MonoAudioProcessor(),
        ).transcribe(
            source_path=tmp_path / "source.wav",
            audio_info=_audio_info(),
            context=_context(tmp_path),
            plan=_plan(tmp_path),
            cancellation_check=cancellation_check,
        )

    assert len(client.pass2_requests) == 1
    assert len(raised.value.attempts) == 1
    assert raised.value.usage is not None
    assert "pass1_diarization" in raised.value.usage
    assert raised.value.quality_summary is not None
    assert raised.value.quality_summary["pass1_completed"] is True
