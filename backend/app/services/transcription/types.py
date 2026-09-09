from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal


TranscriptionModeValue = Literal[
    "legacy",
    "operator_channel",
    "dual_channel",
    "mono_diarization",
]
SpeakerAttributionValue = Literal[
    "confirmed_by_pbx",
    "caller_callee_only",
    "channel_unknown",
    "anonymous_diarization",
]
SpeakerSourceValue = Literal[
    "stereo_channel",
    "openai_diarization",
    "unknown",
]


def _freeze_safe_api_usage(value: Any) -> Any:
    if isinstance(value, Mapping):
        keys = tuple(value)
        if any(not isinstance(key, str) for key in keys):
            raise TypeError("API usage keys must be strings.")
        frozen: dict[str, Any] = {}
        for key in sorted(keys):
            frozen[key] = _freeze_safe_api_usage(value[key])
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_safe_api_usage(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("API usage values must contain only JSON-safe metadata.")


@dataclass(frozen=True, slots=True)
class AudioTrack:
    track_id: str
    source_path: Path
    channel_index: int | None = None
    operator_id: str | None = None
    attribution_status: str | None = None
    audio_variant: str | None = None
    diarized: bool = False
    speaker_label: str | None = None
    speaker_source: SpeakerSourceValue | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class AudioPlan:
    mode: TranscriptionModeValue
    tracks: tuple[AudioTrack, ...]
    operator_channel: int | None = None
    stereo_separated: bool = False
    caller_channel: int | None = None
    callee_channel: int | None = None
    attribution_status: str | None = None
    reason: str = "legacy-worker-selected-topology"


@dataclass(frozen=True, slots=True)
class SpeechChunk:
    track_id: str
    chunk_index: int
    path: Path
    start_seconds: float
    end_seconds: float
    hard_cut: bool = False
    overlap_before_ms: int = 0
    audio_variant: str | None = None


@dataclass(frozen=True, slots=True)
class TokenLogprob:
    token: str
    logprob: float
    start_seconds: float | None = None
    end_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ChunkHypothesis:
    track_id: str
    chunk_index: int | None
    start_seconds: float
    end_seconds: float
    text: str
    speaker_label: str
    confidence: float | None = None
    token_logprobs: tuple[TokenLogprob, ...] = ()
    mean_logprob: float | None = None
    low_logprob_ratio: float | None = None
    quality_flags: tuple[str, ...] = ()
    audio_variant: str | None = None
    hard_cut: bool = False
    overlap_before_ms: int = 0
    channel_index: int | None = None
    operator_id: str | None = None
    speaker_source: SpeakerSourceValue | None = None
    transcription_model: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptionAttemptEvidence:
    track_id: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    model: str
    audio_variant: str | None
    prompt_hash: str | None
    response_text: str | None = field(default=None, repr=False)
    mean_logprob: float | None = None
    low_logprob_ratio: float | None = None
    selected: bool = True
    api_usage: Mapping[str, Any] = field(default_factory=dict)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.mean_logprob is not None and not math.isfinite(self.mean_logprob):
            raise ValueError("Attempt mean logprob must be finite when provided.")
        if self.low_logprob_ratio is not None and not (
            math.isfinite(self.low_logprob_ratio)
            and 0 <= self.low_logprob_ratio <= 1
        ):
            raise ValueError("Attempt low-logprob ratio must be between zero and one.")
        object.__setattr__(
            self,
            "api_usage",
            _freeze_safe_api_usage(self.api_usage),
        )


@dataclass(frozen=True, slots=True)
class TrackTranscriptionResult:
    track_id: str
    model: str
    language: str
    prompt_version: str | None
    processing_duration_seconds: float
    hypotheses: tuple[ChunkHypothesis, ...]
    usage: dict[str, Any] = field(default_factory=dict)
    diarized: bool = False
    attempts: tuple[TranscriptionAttemptEvidence, ...] = ()
    allow_unselected_attempts: bool = False

    def __post_init__(self) -> None:
        grouped: dict[tuple[str, int], list[TranscriptionAttemptEvidence]] = {}
        for attempt in self.attempts:
            if attempt.track_id != self.track_id:
                raise ValueError("Attempt evidence must belong to its track result.")
            grouped.setdefault((attempt.track_id, attempt.chunk_index), []).append(attempt)
        for attempts in grouped.values():
            if len(attempts) > 2:
                raise ValueError("A standard V2 chunk cannot have more than two attempts.")
            selected_count = sum(attempt.selected for attempt in attempts)
            if selected_count > 1 or (
                selected_count == 0 and not self.allow_unselected_attempts
            ):
                raise ValueError("Each attempted V2 chunk must select exactly one attempt.")


@dataclass(frozen=True, slots=True)
class OrchestratedTranscriptionResult:
    mode: TranscriptionModeValue
    model: str
    language: str
    prompt_version: str | None
    processing_duration_seconds: float
    segments: tuple[ChunkHypothesis, ...]
    tracks: tuple[TrackTranscriptionResult, ...]
    usage: dict[str, Any] = field(default_factory=dict)
    diarized: bool = False
    confidence_status: str = "unavailable"
    attribution_status: str | None = None
    plan_reason: str | None = None
    attempts: tuple[TranscriptionAttemptEvidence, ...] = ()
    quality_summary: dict[str, Any] | None = None

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())
