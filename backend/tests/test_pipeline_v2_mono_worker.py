from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models import TranscriptSegment, TranscriptionAttempt
from app.models.enums import (
    SpeakerAttributionStatus,
    SpeakerSource,
    TranscriptionMode,
    TranscriptStatus,
)
from app.services.transcription.orchestrator import PartialTranscriptionCancelledError
from app.services.transcription.types import (
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    TranscriptionAttemptEvidence,
)
from app.workers.pipeline import (
    PIPELINE_V2_MONO_PREPROCESSING_PROFILE,
    PIPELINE_V2_PREPROCESSING_PROFILE,
    PIPELINE_V2_STANDARD_PREPROCESSING_PROFILE,
    _archive_failed_v2_attempt_history,
    _cross_recording_transcript_clone_allowed,
    _has_persisted_mono_pass1_evidence,
    _partial_run_metadata_from_failure,
    _persist_partial_attempt_evidence,
    _persist_transcription_result,
)


class _CapturingSession:
    def __init__(self, *, scalar_result: int = 0) -> None:
        self.added: list[object] = []
        self.flush_count = 0
        self.scalar_result = scalar_result

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        self.flush_count += 1

    async def scalar(self, _statement: object) -> int:
        return self.scalar_result


def _mono_attempt(*, selected: bool) -> TranscriptionAttemptEvidence:
    return TranscriptionAttemptEvidence(
        track_id="mono-diarization",
        chunk_index=0,
        start_seconds=1.0,
        end_seconds=4.0,
        model="gpt-4o-transcribe",
        audio_variant="v2-raw-lossless-pcm16-v1",
        prompt_hash="a" * 64,
        response_text="unsuccessful refinement",
        mean_logprob=-0.9,
        low_logprob_ratio=0.25,
        selected=selected,
        api_usage={"input_tokens": 4},
        completed_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )


def _mono_result(
    *,
    attempts: tuple[TranscriptionAttemptEvidence, ...],
    flags: tuple[str, ...],
) -> OrchestratedTranscriptionResult:
    return OrchestratedTranscriptionResult(
        mode="mono_diarization",
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="greek-mono-v1",
        processing_duration_seconds=1.0,
        segments=(
            ChunkHypothesis(
                track_id="mono-diarization",
                chunk_index=0,
                start_seconds=1.0,
                end_seconds=4.0,
                text="rough pass one text",
                speaker_label="A",
                speaker_source="openai_diarization",
                operator_id=None,
                quality_flags=flags,
                audio_variant="v2-pass1-rough-diarization-v1",
                transcription_model="gpt-4o-transcribe-diarize",
            ),
        ),
        tracks=(),
        usage={
            "pass1_diarization": {"input_tokens": 8},
            "pass2_refinement": {"input_tokens": 4},
        },
        diarized=True,
        attribution_status="anonymous_diarization",
        attempts=attempts,
        quality_summary={
            "pass1_completed": True,
            "degraded": True,
            "fallback_duration_ratio": 1.0,
        },
    )


@pytest.mark.asyncio
async def test_mono_fallback_persists_quality_and_no_selected_refinement() -> None:
    session = _CapturingSession()
    transcript = SimpleNamespace(id=uuid4())

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        _mono_result(
            attempts=(_mono_attempt(selected=False),),
            flags=(
                "refinement_failed",
                "human_review_recommended",
                "logprobs_unavailable",
            ),
        ),
        SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
        None,
        None,
        12.0,
    )

    segment = next(item for item in session.added if isinstance(item, TranscriptSegment))
    attempt = next(item for item in session.added if isinstance(item, TranscriptionAttempt))
    assert transcript.quality_summary == {
        "pass1_completed": True,
        "degraded": True,
        "fallback_duration_ratio": 1.0,
    }
    assert transcript.transcription_mode is TranscriptionMode.MONO_DIARIZATION
    assert transcript.speaker_attribution_status is SpeakerAttributionStatus.ANONYMOUS_DIARIZATION
    assert segment.transcription_model == "gpt-4o-transcribe-diarize"
    assert segment.operator_id is None
    assert segment.speaker_source is SpeakerSource.OPENAI_DIARIZATION
    assert attempt.selected is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected", "flags", "message"),
    [
        (
            True,
            ("refinement_failed",),
            "fallback cannot select",
        ),
        (
            False,
            (),
            "successfully refined mono segment must select exactly one",
        ),
    ],
)
async def test_mono_attempt_selection_is_strictly_tied_to_fallback(
    selected: bool,
    flags: tuple[str, ...],
    message: str,
) -> None:
    session = _CapturingSession()

    with pytest.raises(ValueError, match=message):
        await _persist_transcription_result(
            session,  # type: ignore[arg-type]
            SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
            _mono_result(
                attempts=(_mono_attempt(selected=selected),),
                flags=flags,
            ),
            SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
            None,
            None,
            12.0,
        )

    assert session.added == []
    assert session.flush_count == 0


@pytest.mark.asyncio
async def test_partial_pass1_metadata_is_paid_history_without_attempt_rows() -> None:
    session = _CapturingSession()
    stable_key = "stable-mono-key"
    transcript = SimpleNamespace(
        id=uuid4(),
        transcription_mode=TranscriptionMode.MONO_DIARIZATION,
        status=TranscriptStatus.PROCESSING,
        idempotency_key=stable_key,
        is_current=False,
        api_usage=None,
        quality_summary=None,
    )

    await _persist_partial_attempt_evidence(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        (),
        api_usage={"pass1_diarization": {}},
        quality_summary={"pass1_completed": True},
    )
    transcript.status = TranscriptStatus.FAILED

    assert await _archive_failed_v2_attempt_history(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        stable_key=stable_key,
        transcription_mode=TranscriptionMode.MONO_DIARIZATION,
    )
    assert transcript.api_usage == {"pass1_diarization": {}}
    assert transcript.quality_summary == {"pass1_completed": True}
    assert transcript.idempotency_key != stable_key
    assert transcript.is_current is False
    assert session.flush_count == 2


def test_partial_mono_exception_exposes_pass1_metadata_to_worker() -> None:
    error = PartialTranscriptionCancelledError(
        "cancelled after pass one",
        attempts=(),
        usage={
            "pass1_diarization": {"input_tokens": 8},
            "pass2_refinement": {"input_tokens": 0},
        },
        quality_summary={"pass1_completed": True},
    )

    usage, quality_summary = _partial_run_metadata_from_failure(error, None)

    assert usage == {
        "pass1_diarization": {"input_tokens": 8},
        "pass2_refinement": {"input_tokens": 0},
    }
    assert quality_summary == {"pass1_completed": True}


@pytest.mark.parametrize(
    ("api_usage", "quality_summary", "expected"),
    [
        ({"pass1_diarization": {}}, None, True),
        (None, {"pass1_completed": True}, True),
        ({"pass2_refinement": {}}, None, False),
        ({"diarization": {}}, {"pass1_complete": True}, False),
        (None, {"pass1_completed": False}, False),
    ],
)
def test_paid_mono_history_uses_only_the_fixed_pass1_contract(
    api_usage: dict[str, object] | None,
    quality_summary: dict[str, object] | None,
    expected: bool,
) -> None:
    transcript = SimpleNamespace(
        api_usage=api_usage,
        quality_summary=quality_summary,
    )

    assert (
        _has_persisted_mono_pass1_evidence(transcript)  # type: ignore[arg-type]
        is expected
    )


def test_mono_profile_and_clone_safeguards_are_distinct() -> None:
    assert PIPELINE_V2_MONO_PREPROCESSING_PROFILE not in {
        PIPELINE_V2_PREPROCESSING_PROFILE,
        PIPELINE_V2_STANDARD_PREPROCESSING_PROFILE,
    }
    assert _cross_recording_transcript_clone_allowed(TranscriptionMode.LEGACY)
    assert not _cross_recording_transcript_clone_allowed(TranscriptionMode.OPERATOR_CHANNEL)
    assert not _cross_recording_transcript_clone_allowed(TranscriptionMode.DUAL_CHANNEL)
    assert not _cross_recording_transcript_clone_allowed(TranscriptionMode.MONO_DIARIZATION)
