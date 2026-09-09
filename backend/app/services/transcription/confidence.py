from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, Protocol

from app.services.transcription.types import ChunkHypothesis, TokenLogprob


UNCALIBRATED_CONFIDENCE_POLICY_VERSION = "v2-logprob-uncalibrated-v1"
ConfidenceStatus = Literal["unavailable", "available", "low_confidence"]
ConfidenceDecision = Literal["unavailable", "acceptable", "low_confidence"]
AttemptSelection = Literal["raw", "normalized"]


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Centralized provisional policy; its values are not calibrated correctness scores."""

    version: str = UNCALIBRATED_CONFIDENCE_POLICY_VERSION
    calibration_status: Literal["uncalibrated"] = "uncalibrated"
    mean_logprob_low_threshold: float = -0.75
    token_logprob_low_threshold: float = -1.0
    low_logprob_ratio_threshold: float = 0.15
    effective_mean_tie_tolerance: float = 0.05
    max_attempts_per_chunk: int = 2

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("Confidence policy version must not be empty.")
        if self.calibration_status != "uncalibrated":
            raise ValueError("Prompt 7 confidence policy must remain explicitly uncalibrated.")
        for name, value in (
            ("mean_logprob_low_threshold", self.mean_logprob_low_threshold),
            ("token_logprob_low_threshold", self.token_logprob_low_threshold),
            ("low_logprob_ratio_threshold", self.low_logprob_ratio_threshold),
            ("effective_mean_tie_tolerance", self.effective_mean_tie_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if self.mean_logprob_low_threshold > 0:
            raise ValueError("Mean logprob threshold cannot be positive.")
        if self.token_logprob_low_threshold > 0:
            raise ValueError("Per-token logprob threshold cannot be positive.")
        if not 0 <= self.low_logprob_ratio_threshold <= 1:
            raise ValueError("Low-logprob ratio threshold must be between zero and one.")
        if self.effective_mean_tie_tolerance < 0:
            raise ValueError("Effective mean tie tolerance cannot be negative.")
        if (
            isinstance(self.max_attempts_per_chunk, bool)
            or not isinstance(self.max_attempts_per_chunk, int)
            or not 1 <= self.max_attempts_per_chunk <= 2
        ):
            raise ValueError(
                "Confidence policy must permit either one or two attempts per chunk."
            )


DEFAULT_CONFIDENCE_POLICY = ConfidencePolicy()


@dataclass(frozen=True, slots=True)
class ConfidenceMetrics:
    mean_logprob: float
    minimum_logprob: float
    low_logprob_ratio: float
    token_count: int
    geometric_mean_token_probability: float

    def __post_init__(self) -> None:
        for name, value in (
            ("mean_logprob", self.mean_logprob),
            ("minimum_logprob", self.minimum_logprob),
            ("low_logprob_ratio", self.low_logprob_ratio),
            (
                "geometric_mean_token_probability",
                self.geometric_mean_token_probability,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
        if self.mean_logprob > 0 or self.minimum_logprob > 0:
            raise ValueError("Logprob metrics cannot be positive.")
        if not 0 <= self.low_logprob_ratio <= 1:
            raise ValueError("Low-logprob ratio must be between zero and one.")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count < 1
        ):
            raise ValueError("Token count must be positive.")
        if not 0 <= self.geometric_mean_token_probability <= 1:
            raise ValueError("Geometric mean token probability must be between zero and one.")


@dataclass(frozen=True, slots=True)
class ConfidenceAnalysis:
    status: ConfidenceStatus = "unavailable"
    mean_logprob: float | None = None
    low_logprob_ratio: float | None = None
    minimum_logprob: float | None = None
    token_count: int = 0
    geometric_mean_token_probability: float | None = None
    policy_version: str | None = None


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _finite_nonpositive_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not math.isfinite(result) or result > 0:
        return None
    return result


def _optional_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def parse_token_logprobs(response: object) -> tuple[TokenLogprob, ...]:
    """Extract only genuine, finite token logprobs from an SDK or mapping response."""

    entries = _field(response, "logprobs")
    if entries is None or isinstance(entries, (str, bytes, bytearray, Mapping)):
        return ()

    try:
        iterator = iter(entries)
    except TypeError:
        return ()

    parsed: list[TokenLogprob] = []
    for entry in iterator:
        if entry is None:
            continue
        token = _field(entry, "token")
        logprob = _finite_nonpositive_float(_field(entry, "logprob"))
        if not isinstance(token, str) or token == "" or logprob is None:
            continue
        parsed.append(
            TokenLogprob(
                token=token,
                logprob=logprob,
                start_seconds=_optional_finite_float(_field(entry, "start_seconds")),
                end_seconds=_optional_finite_float(_field(entry, "end_seconds")),
            )
        )
    return tuple(parsed)


def calculate_confidence_metrics(
    token_logprobs: Sequence[TokenLogprob],
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> ConfidenceMetrics | None:
    """Calculate uncalibrated token signals without presenting them as accuracy."""

    values = tuple(
        logprob
        for token in token_logprobs
        if (logprob := _finite_nonpositive_float(token.logprob)) is not None
    )
    if not values:
        return None

    token_count = len(values)
    # Dividing before summing prevents otherwise-valid but extreme finite
    # inputs from overflowing ``fsum``. Provider logprobs are ordinarily small,
    # but malformed evidence must degrade safely instead of crashing the run.
    mean_logprob = math.fsum(value / token_count for value in values)
    low_token_count = sum(
        value < policy.token_logprob_low_threshold for value in values
    )
    return ConfidenceMetrics(
        mean_logprob=mean_logprob,
        minimum_logprob=min(values),
        low_logprob_ratio=low_token_count / token_count,
        token_count=token_count,
        geometric_mean_token_probability=math.exp(mean_logprob),
    )


def classify_confidence(
    metrics: ConfidenceMetrics | None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> ConfidenceDecision:
    if metrics is None:
        return "unavailable"
    if (
        metrics.mean_logprob < policy.mean_logprob_low_threshold
        or metrics.low_logprob_ratio > policy.low_logprob_ratio_threshold
    ):
        return "low_confidence"
    return "acceptable"


def is_low_confidence(
    metrics: ConfidenceMetrics | None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> bool:
    """Return true only after a valid logprob-based low-confidence decision."""

    return classify_confidence(metrics, policy) == "low_confidence"


def select_preferred_attempt(
    raw_metrics: ConfidenceMetrics | None,
    normalized_metrics: ConfidenceMetrics | None,
    policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
) -> AttemptSelection:
    """Choose deterministically; missing metrics never defeat valid metrics."""

    if raw_metrics is None:
        return "normalized" if normalized_metrics is not None else "raw"
    if normalized_metrics is None:
        return "raw"

    mean_difference = normalized_metrics.mean_logprob - raw_metrics.mean_logprob
    if math.isclose(
        normalized_metrics.mean_logprob,
        raw_metrics.mean_logprob,
        rel_tol=1e-12,
        abs_tol=policy.effective_mean_tie_tolerance,
    ):
        return "raw"
    return "normalized" if mean_difference > 0 else "raw"


class ConfidenceAnalyzer(Protocol):
    def analyze(
        self,
        hypotheses: Sequence[ChunkHypothesis],
    ) -> ConfidenceAnalysis: ...


class UnavailableConfidenceAnalyzer:
    """Legacy responses do not expose the logprobs needed for confidence."""

    def analyze(
        self,
        hypotheses: Sequence[ChunkHypothesis],
    ) -> ConfidenceAnalysis:
        del hypotheses
        return ConfidenceAnalysis()


class LogprobConfidenceAnalyzer:
    """Aggregate real selected-hypothesis token logprobs using the V2 policy."""

    def __init__(self, policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY) -> None:
        self.policy = policy

    def analyze(
        self,
        hypotheses: Sequence[ChunkHypothesis],
    ) -> ConfidenceAnalysis:
        metrics = calculate_confidence_metrics(
            tuple(
                token
                for hypothesis in hypotheses
                for token in hypothesis.token_logprobs
            ),
            self.policy,
        )
        if metrics is None:
            return ConfidenceAnalysis()

        decision = classify_confidence(metrics, self.policy)
        return ConfidenceAnalysis(
            status="low_confidence" if decision == "low_confidence" else "available",
            mean_logprob=metrics.mean_logprob,
            low_logprob_ratio=metrics.low_logprob_ratio,
            minimum_logprob=metrics.minimum_logprob,
            token_count=metrics.token_count,
            geometric_mean_token_probability=metrics.geometric_mean_token_probability,
            policy_version=self.policy.version,
        )
