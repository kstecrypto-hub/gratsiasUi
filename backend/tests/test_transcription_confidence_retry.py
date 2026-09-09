from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from app.services.transcription.confidence import (
    DEFAULT_CONFIDENCE_POLICY,
    ConfidenceAnalysis,
    ConfidenceMetrics,
    ConfidencePolicy,
    LogprobConfidenceAnalyzer,
    UnavailableConfidenceAnalyzer,
    calculate_confidence_metrics,
    classify_confidence,
    is_low_confidence,
    parse_token_logprobs,
    select_preferred_attempt,
)
from app.services.transcription.types import ChunkHypothesis, TokenLogprob


def _metrics(mean_logprob: float) -> ConfidenceMetrics:
    return ConfidenceMetrics(
        mean_logprob=mean_logprob,
        minimum_logprob=mean_logprob,
        low_logprob_ratio=0.0,
        token_count=1,
        geometric_mean_token_probability=math.exp(mean_logprob),
    )


def _hypothesis(*tokens: TokenLogprob) -> ChunkHypothesis:
    return ChunkHypothesis(
        track_id="track-0",
        chunk_index=0,
        start_seconds=0.0,
        end_seconds=1.0,
        text="δοκιμή",
        speaker_label="speaker",
        token_logprobs=tokens,
    )


def test_parse_logprobs_from_sdk_like_response() -> None:
    response = SimpleNamespace(
        logprobs=[
            SimpleNamespace(token="γειά", logprob=-0.2),
            SimpleNamespace(token=" σου", logprob=-0.4),
        ]
    )

    assert parse_token_logprobs(response) == (
        TokenLogprob(token="γειά", logprob=-0.2),
        TokenLogprob(token=" σου", logprob=-0.4),
    )


def test_parse_logprobs_from_mapping_response() -> None:
    response = {
        "logprobs": [
            {
                "token": "ναι",
                "logprob": -0.25,
                "start_seconds": 0.1,
                "end_seconds": 0.3,
            }
        ]
    }

    assert parse_token_logprobs(response) == (
        TokenLogprob(
            token="ναι",
            logprob=-0.25,
            start_seconds=0.1,
            end_seconds=0.3,
        ),
    )


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"logprobs": None},
        {"logprobs": "not-a-token-list"},
        SimpleNamespace(),
    ],
)
def test_missing_or_invalid_logprobs_are_unavailable(response: object) -> None:
    tokens = parse_token_logprobs(response)

    assert tokens == ()
    assert calculate_confidence_metrics(tokens) is None
    assert classify_confidence(None) == "unavailable"
    assert is_low_confidence(None) is False


def test_malformed_nan_infinity_and_positive_logprobs_are_ignored() -> None:
    response = {
        "logprobs": [
            {"token": "valid", "logprob": -0.5},
            {"token": "nan", "logprob": math.nan},
            {"token": "positive-infinity", "logprob": math.inf},
            {"token": "negative-infinity", "logprob": -math.inf},
            {"token": "positive", "logprob": 0.01},
            {"token": "", "logprob": -0.2},
            {"token": "missing"},
            None,
        ]
    }

    assert parse_token_logprobs(response) == (
        TokenLogprob(token="valid", logprob=-0.5),
    )


def test_metrics_include_every_required_internal_signal() -> None:
    metrics = calculate_confidence_metrics(
        (
            TokenLogprob(token="a", logprob=-0.1),
            TokenLogprob(token="b", logprob=-1.1),
            TokenLogprob(token="c", logprob=-0.3),
            TokenLogprob(token="d", logprob=-1.5),
        )
    )

    assert metrics is not None
    assert metrics.mean_logprob == pytest.approx(-0.75)
    assert metrics.minimum_logprob == -1.5
    assert metrics.low_logprob_ratio == 0.5
    assert metrics.token_count == 4
    assert metrics.geometric_mean_token_probability == pytest.approx(math.exp(-0.75))


def test_uncalibrated_threshold_boundaries_are_explicit() -> None:
    policy = DEFAULT_CONFIDENCE_POLICY
    at_boundaries = ConfidenceMetrics(
        mean_logprob=policy.mean_logprob_low_threshold,
        minimum_logprob=-1.0,
        low_logprob_ratio=policy.low_logprob_ratio_threshold,
        token_count=1,
        geometric_mean_token_probability=math.exp(policy.mean_logprob_low_threshold),
    )

    assert policy.calibration_status == "uncalibrated"
    assert policy.max_attempts_per_chunk == 2
    assert classify_confidence(at_boundaries, policy) == "acceptable"
    assert classify_confidence(_metrics(-0.75001), policy) == "low_confidence"

    ratio_low = ConfidenceMetrics(
        mean_logprob=-0.1,
        minimum_logprob=-1.1,
        low_logprob_ratio=0.15001,
        token_count=10,
        geometric_mean_token_probability=math.exp(-0.1),
    )
    assert is_low_confidence(ratio_low, policy) is True


def test_policy_is_frozen_and_cannot_allow_a_third_attempt() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_CONFIDENCE_POLICY.max_attempts_per_chunk = 3  # type: ignore[misc]

    with pytest.raises(ValueError, match="one or two"):
        ConfidencePolicy(max_attempts_per_chunk=3)


def test_extreme_finite_logprobs_degrade_without_overflow() -> None:
    metrics = calculate_confidence_metrics(
        (
            TokenLogprob(token="a", logprob=-1e308),
            TokenLogprob(token="b", logprob=-1e308),
        )
    )

    assert metrics is not None
    assert metrics.mean_logprob == -1e308
    assert metrics.geometric_mean_token_probability == 0.0


def test_valid_metrics_beat_missing_metrics() -> None:
    assert select_preferred_attempt(None, _metrics(-0.5)) == "normalized"
    assert select_preferred_attempt(_metrics(-0.5), None) == "raw"
    assert select_preferred_attempt(None, None) == "raw"


def test_higher_mean_logprob_wins_outside_tolerance() -> None:
    assert select_preferred_attempt(_metrics(-0.8), _metrics(-0.4)) == "normalized"
    assert select_preferred_attempt(_metrics(-0.2), _metrics(-0.8)) == "raw"


@pytest.mark.parametrize("normalized_mean", [-0.55, -0.51, -0.45])
def test_raw_wins_effective_mean_tie(normalized_mean: float) -> None:
    assert (
        select_preferred_attempt(_metrics(-0.5), _metrics(normalized_mean))
        == "raw"
    )


def test_logprob_analyzer_aggregates_selected_hypothesis_tokens() -> None:
    analysis = LogprobConfidenceAnalyzer().analyze(
        (
            _hypothesis(TokenLogprob(token="a", logprob=-0.2)),
            _hypothesis(
                TokenLogprob(token="b", logprob=-0.4),
                TokenLogprob(token="c", logprob=-0.6),
            ),
        )
    )

    assert analysis.status == "available"
    assert analysis.mean_logprob == pytest.approx(-0.4)
    assert analysis.minimum_logprob == -0.6
    assert analysis.low_logprob_ratio == 0.0
    assert analysis.token_count == 3
    assert analysis.geometric_mean_token_probability == pytest.approx(math.exp(-0.4))
    assert analysis.policy_version == DEFAULT_CONFIDENCE_POLICY.version


def test_logprob_analyzer_reports_low_confidence_and_unavailable_honestly() -> None:
    analyzer = LogprobConfidenceAnalyzer()

    assert analyzer.analyze((_hypothesis(),)) == ConfidenceAnalysis()
    assert analyzer.analyze(
        (_hypothesis(TokenLogprob(token="weak", logprob=-2.0)),)
    ).status == "low_confidence"


def test_legacy_unavailable_analyzer_contract_is_unchanged() -> None:
    assert ConfidenceAnalysis() == ConfidenceAnalysis(
        status="unavailable",
        mean_logprob=None,
        low_logprob_ratio=None,
    )
    assert UnavailableConfidenceAnalyzer().analyze(
        (_hypothesis(TokenLogprob(token="ignored", logprob=-0.1)),)
    ) == ConfidenceAnalysis()
