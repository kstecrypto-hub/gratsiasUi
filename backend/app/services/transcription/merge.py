from __future__ import annotations

from collections.abc import Sequence

from app.services.transcription.confidence import ConfidenceAnalysis
from app.services.transcription.types import (
    AudioPlan,
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    TrackTranscriptionResult,
    TranscriptionAttemptEvidence,
)


def _merge_attempts(
    plan: AudioPlan,
    results: Sequence[TrackTranscriptionResult],
) -> tuple[TranscriptionAttemptEvidence, ...]:
    track_order = {track.track_id: position for position, track in enumerate(plan.tracks)}
    indexed: list[tuple[int, int, TranscriptionAttemptEvidence]] = []
    for result_position, result in enumerate(results):
        for attempt_position, attempt in enumerate(result.attempts):
            indexed.append((result_position, attempt_position, attempt))
    indexed.sort(
        key=lambda item: (
            item[2].start_seconds,
            item[2].end_seconds,
            track_order.get(item[2].track_id, len(track_order)),
            item[2].track_id,
            item[2].chunk_index,
            item[1],
            item[2].model,
            item[2].audio_variant or "",
            item[2].prompt_hash or "",
            item[0],
        )
    )
    return tuple(item[2] for item in indexed)


def merge_track_results(
    plan: AudioPlan,
    results: Sequence[TrackTranscriptionResult],
    confidence: ConfidenceAnalysis,
) -> OrchestratedTranscriptionResult:
    if not results:
        raise ValueError("At least one track transcription result is required.")

    if len(results) == 1:
        first = results[0]
        segments = first.hypotheses
        usage = dict(first.usage)
        processing_duration = first.processing_duration_seconds
    else:
        track_order = {track.track_id: position for position, track in enumerate(plan.tracks)}
        indexed: list[tuple[int, int, ChunkHypothesis]] = []
        for result in results:
            for sequence, hypothesis in enumerate(result.hypotheses):
                indexed.append(
                    (
                        track_order.get(result.track_id, len(track_order)),
                        sequence,
                        hypothesis,
                    )
                )
        indexed.sort(
            key=lambda item: (
                item[2].start_seconds,
                item[2].end_seconds,
                (item[2].channel_index if item[2].channel_index is not None else len(plan.tracks)),
                (item[2].chunk_index if item[2].chunk_index is not None else 2**31 - 1),
                item[0],
                item[1],
            )
        )
        segments = tuple(item[2] for item in indexed)
        usage = {"tracks": [dict(result.usage) for result in results]}
        processing_duration = sum(result.processing_duration_seconds for result in results)

    first = results[0]
    return OrchestratedTranscriptionResult(
        mode=plan.mode,
        model=first.model,
        language=first.language,
        prompt_version=first.prompt_version,
        processing_duration_seconds=processing_duration,
        segments=segments,
        tracks=tuple(results),
        usage=usage,
        diarized=any(result.diarized for result in results),
        confidence_status=confidence.status,
        attribution_status=plan.attribution_status,
        plan_reason=plan.reason,
        attempts=_merge_attempts(plan, results),
    )
