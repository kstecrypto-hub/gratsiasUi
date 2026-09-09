from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest

from app.services.transcription.mono import (
    DEFAULT_MONO_REFINEMENT_POLICY,
    AnonymousDiarizationTurn,
    MonoRefinementPolicy,
    MonoRefinementSpan,
    calculate_degraded_duration,
    calculate_span_degraded_duration,
    coalesce_anonymous_turns,
    is_safe_anonymous_speaker_label,
    padded_sample_bounds,
)


def _turn(
    speaker: str,
    start: float,
    end: float,
    text: str,
    *,
    recording_duration: float = 120.0,
) -> AnonymousDiarizationTurn:
    return AnonymousDiarizationTurn(
        speaker_label=speaker,
        start_seconds=start,
        end_seconds=end,
        rough_text=text,
        recording_duration_seconds=recording_duration,
    )


def _span(
    start: float,
    end: float,
    *,
    speaker: str = "A",
    recording_duration: float = 10.0,
) -> MonoRefinementSpan:
    return MonoRefinementSpan(
        speaker_label=speaker,
        start_seconds=start,
        end_seconds=end,
        rough_text=f"{speaker}-{start}",
        recording_duration_seconds=recording_duration,
    )


def test_default_policy_is_immutable_json_safe_and_exact() -> None:
    policy = DEFAULT_MONO_REFINEMENT_POLICY

    assert policy.coalesce_max_gap_seconds == 0.4
    assert policy.max_coalesced_span_seconds == 45.0
    assert policy.extraction_padding_seconds == 0.2
    assert policy.max_refinement_spans == 120
    assert policy.global_context_max_characters == 500
    assert policy.degraded_duration_ratio_threshold == 0.20
    assert json.loads(json.dumps(policy.identity())) == policy.identity()
    assert policy.identity()["degraded_fallback"] == {
        "comparison": "strictly_greater_than",
        "duration_ratio_threshold": 0.20,
    }

    with pytest.raises(FrozenInstanceError):
        policy.max_refinement_spans = 121  # type: ignore[misc]


@pytest.mark.parametrize(
    ("label", "valid"),
    [
        ("A", True),
        ("Z", True),
        ("", False),
        ("AA", False),
        ("a", False),
        ("A ", False),
        ("Operator", False),
        ("speaker_0", False),
        (None, False),
    ],
)
def test_only_safe_anonymous_provider_labels_are_accepted(
    label: object,
    valid: bool,
) -> None:
    assert is_safe_anonymous_speaker_label(label) is valid


def test_valid_turn_preserves_absolute_timestamps_and_trimmed_rough_text() -> None:
    turn = _turn("A", 12.25, 16.75, "  rough Greek text  ", recording_duration=60)

    assert turn.start_seconds == 12.25
    assert turn.end_seconds == 16.75
    assert turn.duration_seconds == 4.5
    assert turn.rough_text == "rough Greek text"
    assert turn.recording_duration_seconds == 60.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speaker_label": "Operator"},
        {"start_seconds": math.nan},
        {"start_seconds": math.inf},
        {"start_seconds": -0.001},
        {"end_seconds": math.nan},
        {"end_seconds": math.inf},
        {"start_seconds": 5.0, "end_seconds": 5.0},
        {"start_seconds": 6.0, "end_seconds": 5.0},
        {"end_seconds": 10.001},
        {"rough_text": ""},
        {"rough_text": "   "},
        {"recording_duration_seconds": 0.0},
        {"recording_duration_seconds": math.inf},
    ],
)
def test_invalid_turn_evidence_is_rejected(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "speaker_label": "A",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "rough_text": "rough",
        "recording_duration_seconds": 10.0,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        AnonymousDiarizationTurn(**values)  # type: ignore[arg-type]


def test_same_speaker_adjacent_turns_coalesce_at_inclusive_boundaries() -> None:
    spans = coalesce_anonymous_turns(
        (
            _turn("A", 0.0, 20.0, "first"),
            _turn("A", 20.4, 45.0, "second"),
        )
    )

    assert len(spans) == 1
    assert spans[0].speaker_label == "A"
    assert spans[0].start_seconds == 0.0
    assert spans[0].end_seconds == 45.0
    assert spans[0].rough_text == "first second"
    assert spans[0].source_turn_count == 2


def test_different_speakers_and_gaps_over_limit_remain_separate() -> None:
    spans = coalesce_anonymous_turns(
        (
            _turn("A", 0.0, 5.0, "a1"),
            _turn("B", 5.1, 8.0, "b"),
            _turn("A", 8.5, 10.0, "a2"),
        )
    )

    assert [(span.speaker_label, span.start_seconds, span.end_seconds) for span in spans] == [
        ("A", 0.0, 5.0),
        ("B", 5.1, 8.0),
        ("A", 8.5, 10.0),
    ]


def test_combined_span_over_45_seconds_is_not_coalesced() -> None:
    spans = coalesce_anonymous_turns(
        (
            _turn("A", 0.0, 30.0, "first"),
            _turn("A", 30.1, 46.0, "second"),
        )
    )

    assert [(span.start_seconds, span.end_seconds) for span in spans] == [
        (0.0, 30.0),
        (30.1, 46.0),
    ]


def test_turns_are_processed_in_stable_chronological_order() -> None:
    spans = coalesce_anonymous_turns(
        (
            _turn("B", 20.0, 21.0, "later"),
            _turn("A", 1.0, 2.0, "first"),
            _turn("A", 2.2, 3.0, "second"),
        )
    )

    assert [(span.speaker_label, span.start_seconds, span.rough_text) for span in spans] == [
        ("A", 1.0, "first second"),
        ("B", 20.0, "later"),
    ]


def test_refinement_limits_do_not_discard_pass1_evidence() -> None:
    policy = MonoRefinementPolicy(max_refinement_spans=2)

    spans = coalesce_anonymous_turns(
        (
            _turn("A", 0.0, 1.0, "one"),
            _turn("B", 2.0, 3.0, "two"),
            _turn("A", 4.0, 5.0, "three"),
        ),
        policy,
    )
    overlong = coalesce_anonymous_turns(
        (_turn("A", 0.0, 45.001, "too long"),)
    )

    assert len(spans) == 3
    assert overlong[0].rough_text == "too long"


def test_padding_uses_samples_and_never_crosses_recording_boundaries() -> None:
    at_start = padded_sample_bounds(_span(0.1, 1.0))
    at_end = padded_sample_bounds(_span(9.8, 10.0))

    assert (at_start.start_sample, at_start.end_sample) == (0, 19_200)
    assert at_start.extraction_start_seconds == 0.0
    assert at_start.extraction_end_seconds == 1.2
    assert at_start.authoritative_start_seconds == 0.1
    assert at_start.authoritative_end_seconds == 1.0

    assert (at_end.start_sample, at_end.end_sample) == (153_600, 160_000)
    assert at_end.extraction_start_seconds == 9.6
    assert at_end.extraction_end_seconds == 10.0
    assert at_end.end_sample <= at_end.recording_sample_count
    assert at_end.authoritative_start_seconds == 9.8
    assert at_end.authoritative_end_seconds == 10.0


def test_fractional_padding_floors_start_and_ceils_end_samples() -> None:
    bounds = padded_sample_bounds(_span(1.00001, 1.10001))

    assert bounds.start_sample == math.floor(0.80001 * 16_000)
    assert bounds.end_sample == math.ceil(1.30001 * 16_000)


def test_degraded_threshold_is_strictly_greater_than_twenty_percent() -> None:
    at_threshold = calculate_degraded_duration(
        spoken_duration_seconds=10.0,
        fallback_duration_seconds=2.0,
    )
    over_threshold = calculate_degraded_duration(
        spoken_duration_seconds=10.0,
        fallback_duration_seconds=2.0001,
    )

    assert at_threshold.fallback_duration_ratio == 0.20
    assert at_threshold.degraded is False
    assert over_threshold.fallback_duration_ratio > 0.20
    assert over_threshold.degraded is True


def test_span_degraded_duration_counts_each_fallback_once() -> None:
    spans = (
        _span(0.0, 2.0),
        _span(2.0, 5.0, speaker="B"),
        _span(5.0, 10.0),
    )

    summary = calculate_span_degraded_duration(
        spans,
        fallback_span_indexes=(0, 0, 1),
    )

    assert summary.spoken_duration_seconds == 10.0
    assert summary.fallback_duration_seconds == 5.0
    assert summary.fallback_duration_ratio == 0.5
    assert summary.degraded is True


def test_empty_spoken_duration_is_not_degraded_and_bad_inputs_are_rejected() -> None:
    summary = calculate_degraded_duration(
        spoken_duration_seconds=0.0,
        fallback_duration_seconds=0.0,
    )

    assert summary.fallback_duration_ratio == 0.0
    assert summary.degraded is False

    with pytest.raises(ValueError, match="within spoken duration"):
        calculate_degraded_duration(
            spoken_duration_seconds=1.0,
            fallback_duration_seconds=1.1,
        )
    with pytest.raises(ValueError, match="existing span"):
        calculate_span_degraded_duration(
            (_span(0.0, 1.0),),
            fallback_span_indexes=(1,),
        )
