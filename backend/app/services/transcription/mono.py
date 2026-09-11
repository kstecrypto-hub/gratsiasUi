from __future__ import annotations

import math
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Final


MONO_REFINEMENT_POLICY_VERSION: Final = "v2-mono-two-pass-policy-v2"
ANONYMOUS_SPEAKER_LABEL_PATTERN: Final = r"^[A-Z]$"
_SAMPLE_BOUNDARY_EPSILON_SECONDS: Final = 1e-9
_ANONYMOUS_SPEAKER_LABEL = re.compile(ANONYMOUS_SPEAKER_LABEL_PATTERN, re.ASCII)


class MonoPolicyLimitError(ValueError):
    """A safe Prompt 8 execution bound would be exceeded."""


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def is_safe_anonymous_speaker_label(value: object) -> bool:
    """Accept only provider-style anonymous labels, never names or identifiers."""

    return isinstance(value, str) and _ANONYMOUS_SPEAKER_LABEL.fullmatch(value) is not None


@dataclass(frozen=True, slots=True)
class MonoRefinementPolicy:
    version: str = MONO_REFINEMENT_POLICY_VERSION
    sample_rate_hz: int = 16_000
    coalesce_max_gap_seconds: float = 0.4
    max_coalesced_span_seconds: float = 45.0
    extraction_padding_seconds: float = 0.2
    pause_padding_seconds: float = 0.8
    max_refinement_spans: int = 120
    global_context_max_characters: int = 500
    degraded_duration_ratio_threshold: float = 0.20

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("Mono refinement policy version must not be empty.")
        if self.sample_rate_hz != 16_000:
            raise ValueError("Prompt 8 mono extraction must remain 16 kHz.")
        for name, value, allow_zero in (
            ("coalesce_max_gap_seconds", self.coalesce_max_gap_seconds, True),
            ("max_coalesced_span_seconds", self.max_coalesced_span_seconds, False),
            ("extraction_padding_seconds", self.extraction_padding_seconds, True),
            ("pause_padding_seconds", self.pause_padding_seconds, True),
        ):
            result = _finite_float(value, name=name)
            if result < 0 or (not allow_zero and result == 0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be {qualifier}.")
            object.__setattr__(self, name, result)
        _positive_integer(self.max_refinement_spans, name="max_refinement_spans")
        _positive_integer(
            self.global_context_max_characters,
            name="global_context_max_characters",
        )
        degraded_threshold = _finite_float(
            self.degraded_duration_ratio_threshold,
            name="degraded_duration_ratio_threshold",
        )
        if not 0 <= degraded_threshold <= 1:
            raise ValueError(
                "degraded_duration_ratio_threshold must be between zero and one."
            )
        object.__setattr__(
            self,
            "degraded_duration_ratio_threshold",
            degraded_threshold,
        )

    def identity(self) -> dict[str, object]:
        """Return the complete JSON-safe policy identity used by later integration."""

        return {
            "anonymous_speaker_label_pattern": ANONYMOUS_SPEAKER_LABEL_PATTERN,
            "coalescing": {
                "max_gap_seconds": self.coalesce_max_gap_seconds,
                "max_span_seconds": self.max_coalesced_span_seconds,
            },
            "degraded_fallback": {
                "comparison": "strictly_greater_than",
                "duration_ratio_threshold": self.degraded_duration_ratio_threshold,
            },
            "extraction": {
                "padding_seconds": self.extraction_padding_seconds,
                "pause_padding_seconds": self.pause_padding_seconds,
                "pause_padding_limit": "half-gap-to-next-turn",
                "sample_rate_hz": self.sample_rate_hz,
            },
            "refinement": {
                "global_context_max_characters": self.global_context_max_characters,
                "max_spans": self.max_refinement_spans,
            },
            "version": self.version,
        }

    def is_degraded_ratio(self, value: object) -> bool:
        ratio = _finite_float(value, name="degraded duration ratio")
        if not 0 <= ratio <= 1:
            raise ValueError("Degraded duration ratio must be between zero and one.")
        return ratio > self.degraded_duration_ratio_threshold


DEFAULT_MONO_REFINEMENT_POLICY: Final = MonoRefinementPolicy()


@dataclass(frozen=True, slots=True)
class AnonymousDiarizationTurn:
    """Validated anonymous pass-1 evidence with authoritative absolute placement."""

    speaker_label: str
    start_seconds: float
    end_seconds: float
    rough_text: str = field(repr=False)
    recording_duration_seconds: float = field(repr=False)

    def __post_init__(self) -> None:
        if not is_safe_anonymous_speaker_label(self.speaker_label):
            raise ValueError("Speaker label must be one anonymous uppercase ASCII letter.")
        start = _finite_float(self.start_seconds, name="Turn start")
        end = _finite_float(self.end_seconds, name="Turn end")
        recording_duration = _finite_float(
            self.recording_duration_seconds,
            name="Recording duration",
        )
        if recording_duration <= 0:
            raise ValueError("Recording duration must be positive.")
        if start < 0:
            raise ValueError("Turn start must not be negative.")
        if end <= start:
            raise ValueError("Turn end must be greater than its start.")
        if end > recording_duration:
            raise ValueError("Turn timestamps must remain inside the recording.")
        if not isinstance(self.rough_text, str) or not self.rough_text.strip():
            raise ValueError("Diarization rough text must not be empty.")
        object.__setattr__(self, "start_seconds", start)
        object.__setattr__(self, "end_seconds", end)
        object.__setattr__(self, "rough_text", self.rough_text.strip())
        object.__setattr__(
            self,
            "recording_duration_seconds",
            recording_duration,
        )

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class MonoRefinementSpan:
    """One chronological refinement unit while retaining pass-1 timestamps."""

    speaker_label: str
    start_seconds: float
    end_seconds: float
    rough_text: str = field(repr=False)
    recording_duration_seconds: float = field(repr=False)
    source_turn_count: int = 1

    def __post_init__(self) -> None:
        validated = AnonymousDiarizationTurn(
            speaker_label=self.speaker_label,
            start_seconds=self.start_seconds,
            end_seconds=self.end_seconds,
            rough_text=self.rough_text,
            recording_duration_seconds=self.recording_duration_seconds,
        )
        _positive_integer(self.source_turn_count, name="source_turn_count")
        object.__setattr__(self, "start_seconds", validated.start_seconds)
        object.__setattr__(self, "end_seconds", validated.end_seconds)
        object.__setattr__(self, "rough_text", validated.rough_text)
        object.__setattr__(
            self,
            "recording_duration_seconds",
            validated.recording_duration_seconds,
        )

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _span_from_turn(turn: AnonymousDiarizationTurn) -> MonoRefinementSpan:
    return MonoRefinementSpan(
        speaker_label=turn.speaker_label,
        start_seconds=turn.start_seconds,
        end_seconds=turn.end_seconds,
        rough_text=turn.rough_text,
        recording_duration_seconds=turn.recording_duration_seconds,
    )


def coalesce_anonymous_turns(
    turns: Sequence[AnonymousDiarizationTurn],
    policy: MonoRefinementPolicy = DEFAULT_MONO_REFINEMENT_POLICY,
) -> tuple[MonoRefinementSpan, ...]:
    """Order pass-1 turns and merge only safely adjacent same-speaker evidence."""

    if not turns:
        return ()
    ordered = tuple(
        turn
        for _, turn in sorted(
            enumerate(turns),
            key=lambda item: (
                item[1].start_seconds,
                item[1].end_seconds,
                item[0],
            ),
        )
    )
    recording_duration = ordered[0].recording_duration_seconds
    if any(turn.recording_duration_seconds != recording_duration for turn in ordered):
        raise ValueError("All mono turns must belong to the same recording duration.")

    completed: list[MonoRefinementSpan] = []
    current = _span_from_turn(ordered[0])
    for turn in ordered[1:]:
        gap_seconds = turn.start_seconds - current.end_seconds
        combined_end = max(current.end_seconds, turn.end_seconds)
        combined_duration = combined_end - current.start_seconds
        can_merge = (
            turn.speaker_label == current.speaker_label
            and 0 <= gap_seconds <= policy.coalesce_max_gap_seconds
            and combined_duration <= policy.max_coalesced_span_seconds
        )
        if can_merge:
            current = MonoRefinementSpan(
                speaker_label=current.speaker_label,
                start_seconds=current.start_seconds,
                end_seconds=combined_end,
                rough_text=f"{current.rough_text} {turn.rough_text}",
                recording_duration_seconds=recording_duration,
                source_turn_count=current.source_turn_count + 1,
            )
            continue

        completed.append(current)
        current = _span_from_turn(turn)

    completed.append(current)
    return tuple(completed)


@dataclass(frozen=True, slots=True)
class PaddedSampleBounds:
    """16 kHz extraction bounds plus the unchanged authoritative placement."""

    authoritative_start_seconds: float
    authoritative_end_seconds: float
    start_sample: int
    end_sample: int
    recording_sample_count: int
    sample_rate_hz: int = 16_000

    def __post_init__(self) -> None:
        if self.sample_rate_hz != 16_000:
            raise ValueError("Prompt 8 mono extraction bounds must remain 16 kHz.")
        _positive_integer(
            self.recording_sample_count,
            name="recording_sample_count",
        )
        if (
            isinstance(self.start_sample, bool)
            or not isinstance(self.start_sample, int)
            or isinstance(self.end_sample, bool)
            or not isinstance(self.end_sample, int)
            or not 0 <= self.start_sample < self.end_sample <= self.recording_sample_count
        ):
            raise ValueError("Padded sample bounds must remain inside the recording.")
        start = _finite_float(
            self.authoritative_start_seconds,
            name="Authoritative start",
        )
        end = _finite_float(
            self.authoritative_end_seconds,
            name="Authoritative end",
        )
        if start < 0 or end <= start:
            raise ValueError("Authoritative timestamps must define a positive span.")
        object.__setattr__(self, "authoritative_start_seconds", start)
        object.__setattr__(self, "authoritative_end_seconds", end)

    @property
    def extraction_start_seconds(self) -> float:
        return self.start_sample / self.sample_rate_hz

    @property
    def extraction_end_seconds(self) -> float:
        return self.end_sample / self.sample_rate_hz


def padded_sample_bounds(
    span: MonoRefinementSpan,
    policy: MonoRefinementPolicy = DEFAULT_MONO_REFINEMENT_POLICY,
    *,
    following_speech_start_seconds: float | None = None,
) -> PaddedSampleBounds:
    """Convert padded seconds to bounded, lossless 16 kHz sample coordinates."""

    end_padding = policy.extraction_padding_seconds
    if following_speech_start_seconds is not None:
        following_start = _finite_float(
            following_speech_start_seconds, name="Following speech start",
        )
        if not 0 <= following_start <= span.recording_duration_seconds:
            raise ValueError("Following speech start must remain inside the recording.")
        # Diarization may end a turn before the final syllable. Use a little
        # more of an available pause without expanding into the next turn.
        # Keep the original small pad when speakers overlap or switch rapidly.
        end_padding = max(end_padding, min(
            policy.pause_padding_seconds, (following_start - span.end_seconds) / 2,
        ))
    sample_epsilon = _SAMPLE_BOUNDARY_EPSILON_SECONDS * policy.sample_rate_hz
    recording_sample_count = math.floor(
        span.recording_duration_seconds * policy.sample_rate_hz
        + sample_epsilon
    )
    if recording_sample_count < 1:
        raise ValueError("Recording duration does not contain a complete 16 kHz sample.")
    start_sample = max(
        0,
        math.floor(
            (span.start_seconds - policy.extraction_padding_seconds)
            * policy.sample_rate_hz
            + sample_epsilon
        ),
    )
    end_sample = min(
        recording_sample_count,
        math.ceil(
            (span.end_seconds + end_padding)
            * policy.sample_rate_hz
            - sample_epsilon
        ),
    )
    return PaddedSampleBounds(
        authoritative_start_seconds=span.start_seconds,
        authoritative_end_seconds=span.end_seconds,
        start_sample=start_sample,
        end_sample=end_sample,
        recording_sample_count=recording_sample_count,
        sample_rate_hz=policy.sample_rate_hz,
    )


@dataclass(frozen=True, slots=True)
class DegradedDurationSummary:
    spoken_duration_seconds: float
    fallback_duration_seconds: float
    fallback_duration_ratio: float
    degraded: bool
    threshold: float

    def __post_init__(self) -> None:
        spoken = _finite_float(
            self.spoken_duration_seconds,
            name="Spoken duration",
        )
        fallback = _finite_float(
            self.fallback_duration_seconds,
            name="Fallback duration",
        )
        ratio = _finite_float(
            self.fallback_duration_ratio,
            name="Fallback duration ratio",
        )
        threshold = _finite_float(self.threshold, name="Degraded threshold")
        if spoken < 0 or fallback < 0 or fallback > spoken:
            raise ValueError("Fallback duration must be within spoken duration.")
        if not 0 <= ratio <= 1 or not 0 <= threshold <= 1:
            raise ValueError("Degraded duration ratios must be between zero and one.")
        object.__setattr__(self, "spoken_duration_seconds", spoken)
        object.__setattr__(self, "fallback_duration_seconds", fallback)
        object.__setattr__(self, "fallback_duration_ratio", ratio)
        object.__setattr__(self, "threshold", threshold)


def calculate_degraded_duration(
    *,
    spoken_duration_seconds: object,
    fallback_duration_seconds: object,
    policy: MonoRefinementPolicy = DEFAULT_MONO_REFINEMENT_POLICY,
) -> DegradedDurationSummary:
    spoken = _finite_float(spoken_duration_seconds, name="Spoken duration")
    fallback = _finite_float(fallback_duration_seconds, name="Fallback duration")
    if spoken < 0 or fallback < 0 or fallback > spoken:
        raise ValueError("Fallback duration must be within spoken duration.")
    ratio = fallback / spoken if spoken else 0.0
    return DegradedDurationSummary(
        spoken_duration_seconds=spoken,
        fallback_duration_seconds=fallback,
        fallback_duration_ratio=ratio,
        degraded=policy.is_degraded_ratio(ratio),
        threshold=policy.degraded_duration_ratio_threshold,
    )


def calculate_span_degraded_duration(
    spans: Sequence[MonoRefinementSpan],
    *,
    fallback_span_indexes: Collection[int],
    policy: MonoRefinementPolicy = DEFAULT_MONO_REFINEMENT_POLICY,
) -> DegradedDurationSummary:
    fallback_indexes = set(fallback_span_indexes)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(spans)
        for index in fallback_indexes
    ):
        raise ValueError("Fallback span indexes must reference an existing span.")
    return calculate_degraded_duration(
        spoken_duration_seconds=math.fsum(span.duration_seconds for span in spans),
        fallback_duration_seconds=math.fsum(
            spans[index].duration_seconds for index in fallback_indexes
        ),
        policy=policy,
    )


__all__ = [
    "ANONYMOUS_SPEAKER_LABEL_PATTERN",
    "DEFAULT_MONO_REFINEMENT_POLICY",
    "MONO_REFINEMENT_POLICY_VERSION",
    "AnonymousDiarizationTurn",
    "DegradedDurationSummary",
    "MonoPolicyLimitError",
    "MonoRefinementPolicy",
    "MonoRefinementSpan",
    "PaddedSampleBounds",
    "calculate_degraded_duration",
    "calculate_span_degraded_duration",
    "coalesce_anonymous_turns",
    "is_safe_anonymous_speaker_label",
    "padded_sample_bounds",
]
