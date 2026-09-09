from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.services.transcription.confidence import ConfidenceAnalysis
from app.services.transcription.merge import merge_track_results
from app.services.transcription.types import (
    AudioPlan,
    AudioTrack,
    TrackTranscriptionResult,
    TranscriptionAttemptEvidence,
)


def _attempt(
    track_id: str,
    chunk_index: int,
    start_seconds: float,
    end_seconds: float,
    *,
    prompt_hash: str,
) -> TranscriptionAttemptEvidence:
    return TranscriptionAttemptEvidence(
        track_id=track_id,
        chunk_index=chunk_index,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        model="gpt-4o-transcribe",
        audio_variant=f"{track_id}-pcm16",
        prompt_hash=prompt_hash,
        api_usage={"input_tokens": chunk_index + 1},
    )


def _track(
    track_id: str,
    *attempts: TranscriptionAttemptEvidence,
) -> TrackTranscriptionResult:
    return TrackTranscriptionResult(
        track_id=track_id,
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="prompt-identity",
        processing_duration_seconds=0.25,
        hypotheses=(),
        attempts=attempts,
    )


def _dual_plan() -> AudioPlan:
    return AudioPlan(
        mode="dual_channel",
        tracks=(
            AudioTrack(
                track_id="channel-0",
                source_path=Path("channel-0.wav"),
                channel_index=0,
            ),
            AudioTrack(
                track_id="channel-1",
                source_path=Path("channel-1.wav"),
                channel_index=1,
            ),
        ),
    )


def test_attempt_evidence_deeply_freezes_safe_api_usage() -> None:
    source_usage = {
        "output_tokens": 3,
        "details": {
            "input_tokens": [1, 2],
            "cached": False,
        },
    }
    evidence = TranscriptionAttemptEvidence(
        track_id="operator-channel",
        chunk_index=0,
        start_seconds=0.0,
        end_seconds=12.5,
        model="gpt-4o-transcribe",
        audio_variant="operator-pcm16",
        prompt_hash="a" * 64,
        api_usage=source_usage,
    )

    source_usage["details"]["input_tokens"].append(99)

    assert list(evidence.api_usage) == ["details", "output_tokens"]
    assert evidence.api_usage["details"]["input_tokens"] == (1, 2)
    with pytest.raises(TypeError):
        evidence.api_usage["new"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        evidence.api_usage["details"]["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        evidence.prompt_hash = "b" * 64  # type: ignore[misc]


def test_attempt_metadata_defaults_empty_for_existing_result_constructors() -> None:
    track = TrackTranscriptionResult(
        track_id="operator-channel",
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="legacy-compatible",
        processing_duration_seconds=0.1,
        hypotheses=(),
    )
    plan = AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=Path("operator.wav"),
            ),
        ),
    )

    result = merge_track_results(plan, [track], ConfidenceAnalysis())

    assert track.attempts == ()
    assert result.attempts == ()


def test_merge_orders_attempts_by_time_then_plan_track_and_chunk() -> None:
    channel_0_chunk_1 = _attempt("channel-0", 1, 10.0, 20.0, prompt_hash="c" * 64)
    channel_1_chunk_0 = _attempt("channel-1", 0, 0.0, 10.0, prompt_hash="b" * 64)
    channel_0_chunk_0 = _attempt("channel-0", 0, 0.0, 10.0, prompt_hash="a" * 64)
    channel_0 = _track("channel-0", channel_0_chunk_1, channel_0_chunk_0)
    channel_1 = _track("channel-1", channel_1_chunk_0)
    plan = _dual_plan()

    forward = merge_track_results(
        plan,
        [channel_0, channel_1],
        ConfidenceAnalysis(),
    )
    reversed_results = merge_track_results(
        plan,
        [channel_1, channel_0],
        ConfidenceAnalysis(),
    )

    expected = (
        channel_0_chunk_0,
        channel_1_chunk_0,
        channel_0_chunk_1,
    )
    assert forward.attempts == expected
    assert reversed_results.attempts == expected
    assert forward.attempts[0] is channel_0_chunk_0
