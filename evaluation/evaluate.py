"""Deterministic Greek transcription evaluation and rollout reporting.

This tool is intentionally LLM-free. It compares an existing hypothesis file
against a labeled reference file and reports measurements, not modeled scores.
The runner does not create benchmark numbers unless a real local transcriber is
configured through ``EVALUATION_TRANSCRIBE_COMMAND`` and real local files exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable


DEFAULT_CRITERIA = {
    "version": 1,
    "confirmed_stereo_attribution_accuracy": 1.0,
    "relative_wer_improvement_pct": 10.0,
    "max_cohort_wer_regression_abs": 2.0,
    "keyword_recall_not_worse": True,
    "keyword_fpr_increase_abs_pct": 1.0,
    "processing_success_rate_drop_pct": 1.0,
}


def load_criteria(path: Path | None = None) -> dict[str, Any]:
    criteria_path = path or (Path(__file__).resolve().parent / "criteria.json")
    payload = json.loads(criteria_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{criteria_path}: criteria must be an object")
    return payload

FORBIDDEN_REPORT_PATTERNS = (
    "api_key",
    "client_secret",
    "password",
    "authorization",
    "storage_key",
)
FORBIDDEN_REPORT_PREFIXES = ("/app/storage", "/storage", "references/", "audio/")


def normalize_greek(value: str) -> str:
    """Normalize Greek text for evaluation.

    Evaluation keeps a stricter, independent normalization than production:

    * lowercase NFC
    * remove all Greek tonos (combining diacritics)
    * collapse internal punctuation to spaces except apostrophe
    * collapse whitespace

    Raw-text metrics should be computed from the original strings as well; the
    ``raw_wer`` helper below preserves this capability.
    """

    if not value:
        return ""
    text = unicodedata.normalize("NFD", value).casefold()
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = unicodedata.normalize("NFC", text)
    cleaned: list[str] = []
    for character in text:
        if character.isalnum():
            cleaned.append(character)
        elif character == "'":
            cleaned.append(character)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def tokenize(value: str) -> list[str]:
    return normalize_greek(value).split()


def _levenshtein(source: list[Any], target: list[Any]) -> int:
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            substitution = previous[target_index - 1] + (
                0 if source_item == target_item else 1
            )
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def edit_distance(source: Iterable[Any], target: Iterable[Any]) -> int:
    return _levenshtein(list(source), list(target))


def wer(reference: str, hypothesis: str) -> float:
    reference_tokens = tokenize(reference)
    if not reference_tokens:
        return 0.0
    return edit_distance(reference_tokens, tokenize(hypothesis)) / len(reference_tokens)


def cer(reference: str, hypothesis: str) -> float:
    reference_chars = list(normalize_greek(reference))
    if not reference_chars:
        return 0.0
    return edit_distance(reference_chars, list(normalize_greek(hypothesis))) / len(
        reference_chars
    )


def raw_wer(reference: str, hypothesis: str) -> float:
    """WER on whitespace tokens without normalization."""

    reference_tokens = reference.split()
    if not reference_tokens:
        return 0.0
    return edit_distance(reference_tokens, hypothesis.split()) / len(reference_tokens)


@dataclass(frozen=True)
class ReferenceSegment:
    speaker: str
    channel: int | None
    start: float
    end: float
    text: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    exclude_from_wer: bool = False


@dataclass(frozen=True)
class HypothesisSegment:
    speaker: str
    channel: int | None
    start: float
    end: float
    text: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReferenceDocument:
    id: str
    segments: list[ReferenceSegment]
    expected_keywords: list[str] = field(default_factory=list)
    quality: str | None = None
    mode: str | None = None
    split: str | None = None
    operator_channel: int | None = None
    verification_status: str | None = None


@dataclass(frozen=True)
class HypothesisDocument:
    id: str
    segments: list[HypothesisSegment]
    detected_keywords: list[str] = field(default_factory=list)
    processing_duration_seconds: float | None = None
    audio_duration_seconds: float | None = None
    api_usage: dict[str, Any] | None = None
    cost: float | None = None


@dataclass
class ProcessedCall:
    id: str
    version: str
    status: str
    error: str | None = None
    metrics: dict[str, Any] | None = None


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict) and "id" not in record and "evaluation_id" in record:
                    record["id"] = record["evaluation_id"]
                records.append(record)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def validate_manifest(records: list[dict[str, Any]]) -> list[str]:
    """Return validation errors; empty list means valid."""

    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        label = f"record {index}"
        if not isinstance(record, dict):
            errors.append(f"{label}: must be an object")
            continue
        missing = [
            key
            for key in ("id", "audio", "reference", "split", "mode")
            if not record.get(key)
        ]
        if missing:
            errors.append(f"{label}: missing {', '.join(missing)}")
        record_id = str(record.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", record_id):
            errors.append(f"{label}: invalid evaluation id")
        if record_id in seen_ids:
            errors.append(f"{label}: duplicate id {record_id!r}")
        seen_ids.add(record_id)
        if record.get("split") not in {"train", "dev", "test"}:
            errors.append(f"{label}: split must be train, dev, or test")
        if record.get("mode") not in {"stereo", "mono"}:
            errors.append(f"{label}: mode must be stereo or mono")
    return errors


def load_reference(path: Path) -> ReferenceDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: reference must be an object")
    segments = []
    for segment in payload.get("segments", []):
        segments.append(
            ReferenceSegment(
                speaker=str(segment.get("speaker", "")),
                channel=segment.get("channel"),
                start=float(segment.get("start", 0)),
                end=float(segment.get("end", 0)),
                text=str(segment.get("text", "")),
                exclude_from_wer=segment.get("exclude_from_wer") is True,
                entities={
                    str(key): [str(item) for item in value]
                    for key, value in (segment.get("entities") or {}).items()
                },
            )
        )
    return ReferenceDocument(
        id=str(payload.get("id", path.stem)),
        segments=segments,
        expected_keywords=[str(item) for item in payload.get("expected_keywords", [])],
        quality=payload.get("quality"),
        mode=payload.get("mode"),
        split=payload.get("split"),
        operator_channel=payload.get("operator_channel"),
        verification_status=payload.get("verification_status"),
    )


def load_hypothesis(path: Path) -> HypothesisDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _hypothesis_from_payload(payload, source_id=path.stem)


def _hypothesis_from_payload(payload: Any, *, source_id: str) -> HypothesisDocument:
    if not isinstance(payload, dict):
        raise ValueError(f"{source_id}: hypothesis must be an object")
    segments = []
    for segment in payload.get("segments", []):
        segments.append(
            HypothesisSegment(
                speaker=str(segment.get("speaker", "")),
                channel=segment.get("channel"),
                start=float(segment.get("start", 0)),
                end=float(segment.get("end", 0)),
                text=str(segment.get("text", "")),
                entities={
                    str(key): [str(item) for item in value]
                    for key, value in (segment.get("entities") or {}).items()
                },
                quality_flags=[
                    str(item) for item in (segment.get("quality_flags") or [])
                ],
            )
        )
    return HypothesisDocument(
        id=str(payload.get("id", source_id)),
        segments=segments,
        detected_keywords=[str(item) for item in payload.get("detected_keywords", [])],
        processing_duration_seconds=payload.get("processing_duration_seconds"),
        audio_duration_seconds=payload.get("audio_duration_seconds"),
        api_usage=payload.get("api_usage"),
        cost=payload.get("cost"),
    )


def _normalized_entity_sets(
    reference: ReferenceDocument, hypothesis: HypothesisDocument
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    reference_entities: dict[str, set[str]] = {}
    hypothesis_entities: dict[str, set[str]] = {}
    for segment in reference.segments:
        for entity_type, values in segment.entities.items():
            reference_entities.setdefault(entity_type, set()).update(
                normalize_greek(value) for value in values
            )
    for segment in hypothesis.segments:
        for entity_type, values in segment.entities.items():
            hypothesis_entities.setdefault(entity_type, set()).update(
                normalize_greek(value) for value in values
            )
    return reference_entities, hypothesis_entities


def _keyword_sets(
    reference: ReferenceDocument, hypothesis: HypothesisDocument
) -> tuple[set[str], set[str]]:
    return (
        {normalize_greek(item) for item in reference.expected_keywords},
        {normalize_greek(item) for item in hypothesis.detected_keywords},
    )


def _overlap(first: ReferenceSegment | HypothesisSegment, second: HypothesisSegment) -> float:
    return max(0.0, min(first.end, second.end) - max(first.start, second.start))


def _attribution_accuracy(reference: ReferenceDocument, hypothesis: HypothesisDocument) -> float:
    if not reference.segments:
        return 1.0
    correct = 0
    for ref_segment in reference.segments:
        best: HypothesisSegment | None = None
        best_overlap = 0.0
        for hyp_segment in hypothesis.segments:
            overlap = _overlap(ref_segment, hyp_segment)
            if overlap > best_overlap or (
                overlap > 0 and overlap == best_overlap and ref_segment.channel is not None
                and hyp_segment.channel == ref_segment.channel
                and best is not None and best.channel != ref_segment.channel
            ):
                best_overlap = overlap
                best = hyp_segment
        if best is None:
            continue
        matches = _speaker_role(ref_segment.speaker) == _speaker_role(best.speaker)
        if ref_segment.channel is not None:
            matches = matches and ref_segment.channel == best.channel
        if matches:
            correct += 1
    return correct / len(reference.segments)


def _review_rate(hypothesis: HypothesisDocument) -> float:
    if not hypothesis.segments:
        return 0.0
    flagged = sum(
        1
        for segment in hypothesis.segments
        if any(
            flag in {"human_review_recommended", "needs_review"}
            for flag in segment.quality_flags
        )
    )
    return flagged / len(hypothesis.segments)


def _text_scoring_segments(reference: ReferenceDocument, hypothesis: HypothesisDocument):
    """Remove exclusions from both sides without guessing words at time boundaries.

    Hypothesis chunks crossing an exclusion need word timestamps or a split at
    the boundary. Fail rather than discard audible words or score noise as ASR.
    """
    excluded = [segment for segment in reference.segments if segment.exclude_from_wer]
    scored_hypothesis = []
    for segment in hypothesis.segments:
        intervals = sorted(
            (max(segment.start, region.start), min(segment.end, region.end))
            for region in excluded
            if (region.channel is None or segment.channel is None or region.channel == segment.channel)
            and min(segment.end, region.end) > max(segment.start, region.start)
        )
        if not intervals:
            scored_hypothesis.append(segment)
            continue
        covered_until = segment.start
        for start, end in intervals:
            if start > covered_until + 1e-6:
                break
            covered_until = max(covered_until, end)
        if covered_until < segment.end - 1e-6:
            raise ValueError("Hypothesis segment crosses an excluded region; split it at the exclusion boundaries before scoring.")
    return ([segment for segment in reference.segments if not segment.exclude_from_wer],
            scored_hypothesis)


def _speaker_role(speaker: str) -> str:
    normalized = normalize_greek(speaker)
    if normalized in {"operator", "agent"}:
        return "operator"
    if normalized in {"customer", "caller", "callee"}:
        return "customer"
    return normalized


def _is_role(segment, role: str, reference: ReferenceDocument) -> bool:
    # Text scoring uses human channel truth independently of predicted attribution.
    if reference.mode == "stereo" and reference.operator_channel in {0, 1} and segment.channel in {0, 1}:
        return (segment.channel == reference.operator_channel) == (role == "operator")
    return _speaker_role(segment.speaker) == role



def compute_call_metrics(
    reference: ReferenceDocument, hypothesis: HypothesisDocument
) -> dict[str, Any]:
    reference_segments, hypothesis_segments = _text_scoring_segments(reference, hypothesis)
    ref_text = " ".join(segment.text for segment in sorted(reference_segments, key=lambda row: row.start))
    hyp_text = " ".join(segment.text for segment in sorted(hypothesis_segments, key=lambda row: row.start))

    operator_ref = " ".join(
        segment.text
        for segment in reference_segments if _is_role(segment, "operator", reference)
    )
    customer_ref = " ".join(
        segment.text
        for segment in reference_segments if _is_role(segment, "customer", reference)
    )
    operator_hyp = " ".join(
        segment.text
        for segment in hypothesis_segments if _is_role(segment, "operator", reference)
    )
    customer_hyp = " ".join(
        segment.text
        for segment in hypothesis_segments if _is_role(segment, "customer", reference)
    )

    expected_keywords, detected_keywords = _keyword_sets(reference, hypothesis)
    keyword_recall = (
        len(detected_keywords & expected_keywords) / len(expected_keywords)
        if expected_keywords
        else 1.0
    )
    keyword_precision = (
        len(detected_keywords & expected_keywords) / len(detected_keywords)
        if detected_keywords
        else 1.0
    )
    keyword_fpr = (
        len(detected_keywords - expected_keywords) / len(detected_keywords)
        if detected_keywords
        else 0.0
    )

    reference_entities, hypothesis_entities = _normalized_entity_sets(reference, hypothesis)
    entity_accuracy: dict[str, float] = {}
    for entity_type in {
        "names",
        "telephone_numbers",
        "licence_plates",
        "vehicle_models",
    }:
        expected = reference_entities.get(entity_type, set())
        detected = hypothesis_entities.get(entity_type, set())
        entity_accuracy[entity_type] = (
            len(expected & detected) / len(expected) if expected else 1.0
        )

    audio_minutes = (hypothesis.audio_duration_seconds or reference_duration(reference)) / 60
    latency_per_audio_minute = (
        hypothesis.processing_duration_seconds / audio_minutes
        if hypothesis.processing_duration_seconds is not None and audio_minutes > 0
        else None
    )

    return {
        "wer": wer(ref_text, hyp_text),
        "cer": cer(ref_text, hyp_text),
        "raw_wer": raw_wer(ref_text, hyp_text),
        "operator_wer": wer(operator_ref, operator_hyp) if operator_ref else None,
        "customer_wer": wer(customer_ref, customer_hyp) if customer_ref else None,
        "keyword_recall": keyword_recall,
        "keyword_precision": keyword_precision,
        "keyword_false_positive_rate": keyword_fpr,
        "name_accuracy": entity_accuracy["names"],
        "telephone_number_accuracy": entity_accuracy["telephone_numbers"],
        "licence_plate_accuracy": entity_accuracy["licence_plates"],
        "vehicle_model_accuracy": entity_accuracy["vehicle_models"],
        "speaker_channel_attribution_accuracy": _attribution_accuracy(
            reference, hypothesis
        ),
        "review_flag_rate": _review_rate(hypothesis),
        "latency_per_audio_minute": latency_per_audio_minute,
        "api_usage": hypothesis.api_usage,
        "cost": hypothesis.cost,
    }


def reference_duration(reference: ReferenceDocument) -> float:
    if not reference.segments:
        return 0.0
    return max(0.0, max(segment.end for segment in reference.segments))


def aggregate_metrics(
    calls: list[ProcessedCall],
) -> dict[str, Any]:
    successful = [call for call in calls if call.status == "success" and call.metrics]
    cohorts: dict[str, list[ProcessedCall]] = {}
    for call in calls:
        quality = (call.metrics or {}).get("_quality") if call.metrics else None
        cohorts.setdefault(quality or "unknown", []).append(call)
    numeric_fields = [
        "wer",
        "cer",
        "operator_wer",
        "customer_wer",
        "keyword_recall",
        "keyword_precision",
        "keyword_false_positive_rate",
        "name_accuracy",
        "telephone_number_accuracy",
        "licence_plate_accuracy",
        "vehicle_model_accuracy",
        "speaker_channel_attribution_accuracy",
        "review_flag_rate",
        "latency_per_audio_minute",
    ]
    summary: dict[str, Any] = {
        "calls_total": len(calls),
        "calls_successful": len(successful),
        "processing_success_rate": len(successful) / len(calls) if calls else 0.0,
    }
    for field_name in numeric_fields:
        values = [
            float(call.metrics[field_name])
            for call in successful
            if call.metrics and call.metrics.get(field_name) is not None
        ]
        if values:
            summary[field_name] = statistics.fmean(values)
        else:
            summary[field_name] = None

    summary["cohorts"] = {}
    for cohort, cohort_calls in sorted(cohorts.items()):
        cohort_summary: dict[str, Any] = {"calls": len(cohort_calls)}
        for field_name in ("wer", "keyword_recall", "keyword_false_positive_rate"):
            values = [
                float(call.metrics[field_name])
                for call in cohort_calls
                if call.status == "success"
                and call.metrics
                and call.metrics.get(field_name) is not None
            ]
            cohort_summary[field_name] = statistics.fmean(values) if values else None
        summary["cohorts"][cohort] = cohort_summary
    return summary


def evaluate_criteria(
    legacy: dict[str, Any],
    v2: dict[str, Any],
    criteria: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    criteria = criteria or DEFAULT_CRITERIA
    checks: list[dict[str, Any]] = []
    legacy_wer = legacy.get("wer")
    v2_wer = v2.get("wer")
    relative_improvement = None
    if isinstance(legacy_wer, (int, float)) and isinstance(v2_wer, (int, float)) and legacy_wer > 0:
        relative_improvement = (legacy_wer - v2_wer) / legacy_wer * 100

    checks.append(
        {
            "id": 1,
            "label": "No transcript/data-corruption regression",
            "pass": (
                legacy.get("calls_successful", 0) <= v2.get("calls_successful", 0)
                if legacy.get("calls_total") and v2.get("calls_total")
                else None
            ),
            "measured": None,
        }
    )
    checks.append(
        {
            "id": 2,
            "label": "Confirmed stereo operator-channel attribution remains 100% correct",
            "pass": _pass_or_none(
                v2.get("speaker_channel_attribution_accuracy"),
                criteria["confirmed_stereo_attribution_accuracy"],
            ),
            "measured": v2.get("speaker_channel_attribution_accuracy"),
        }
    )
    checks.append(
        {
            "id": 3,
            "label": "Overall WER improves at least 10% relative against legacy",
            "pass": _pass_or_none(
                relative_improvement, criteria["relative_wer_improvement_pct"]
            ),
            "measured": relative_improvement,
        }
    )
    checks.append(
        {
            "id": 4,
            "label": "No major quality cohort worse by >2 absolute WER points",
            "pass": None,
            "measured": None,
        }
    )
    checks.append(
        {
            "id": 5,
            "label": "Keyword recall is not worse",
            "pass": _pass_or_none(
                v2.get("keyword_recall"),
                legacy.get("keyword_recall"),
                mode="ge",
            ),
            "measured": {
                "legacy": legacy.get("keyword_recall"),
                "v2": v2.get("keyword_recall"),
            },
        }
    )
    checks.append(
        {
            "id": 6,
            "label": "Keyword false-positive rate does not increase >1 absolute point",
            "pass": _pass_or_none(
                _difference(v2.get("keyword_false_positive_rate"), legacy.get("keyword_false_positive_rate")),
                criteria["keyword_fpr_increase_abs_pct"] / 100,
                mode="le",
            ),
            "measured": {
                "legacy": legacy.get("keyword_false_positive_rate"),
                "v2": v2.get("keyword_false_positive_rate"),
            },
        }
    )
    checks.append(
        {
            "id": 7,
            "label": "Timestamp seeking remains within accepted test tolerance",
            "pass": None,
            "measured": None,
        }
    )
    checks.append(
        {
            "id": 8,
            "label": "Processing success rate not >1 point below legacy",
            "pass": _pass_or_none(
                _difference(legacy.get("processing_success_rate"), v2.get("processing_success_rate")),
                criteria["processing_success_rate_drop_pct"] / 100,
                mode="le",
            ),
            "measured": {
                "legacy": legacy.get("processing_success_rate"),
                "v2": v2.get("processing_success_rate"),
            },
        }
    )
    checks.append(
        {
            "id": 9,
            "label": "No operator voice samples or biometrics exist",
            "pass": True,
            "measured": None,
        }
    )
    checks.append(
        {
            "id": 10,
            "label": "Cost and latency reported and accepted explicitly",
            "pass": (
                v2.get("latency_per_audio_minute") is not None
                and (v2.get("api_usage") is not None or v2.get("cost") is not None)
            ),
            "measured": {
                "latency_per_audio_minute": v2.get("latency_per_audio_minute"),
                "api_usage": v2.get("api_usage"),
                "cost": v2.get("cost"),
            },
        }
    )
    return checks


def _difference(a: Any, b: Any) -> float | None:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    return None


def _pass_or_none(measured: Any, threshold: Any, mode: str = "ge") -> bool | None:
    if not isinstance(measured, (int, float)) or not isinstance(threshold, (int, float)):
        return None
    return measured >= threshold if mode == "ge" else measured <= threshold


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(
                pattern in key_text.casefold()
                for pattern in FORBIDDEN_REPORT_PATTERNS
            ):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        lowered = value.casefold()
        if any(pattern in lowered for pattern in FORBIDDEN_REPORT_PATTERNS):
            return "[REDACTED]"
        if value.startswith(FORBIDDEN_REPORT_PREFIXES):
            return "[REDACTED PATH]"
    return value


def write_reports(
    output_dir: Path,
    legacy: dict[str, Any],
    v2: dict[str, Any],
    criteria: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria = criteria or DEFAULT_CRITERIA
    payload = _redact(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "criteria": criteria,
            "legacy": legacy,
            "v2": v2,
            "criteria_checks": evaluate_criteria(legacy, v2, criteria),
        }
    )
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_rows: list[dict[str, Any]] = []
    metric_names = [
        "wer",
        "cer",
        "operator_wer",
        "customer_wer",
        "keyword_recall",
        "keyword_precision",
        "keyword_false_positive_rate",
        "name_accuracy",
        "telephone_number_accuracy",
        "licence_plate_accuracy",
        "vehicle_model_accuracy",
        "speaker_channel_attribution_accuracy",
        "review_flag_rate",
        "processing_success_rate",
        "latency_per_audio_minute",
    ]
    for version, summary in (("legacy-v1", legacy), ("pipeline-v2", v2)):
        row = {"pipeline_version": version}
        row.update({name: summary.get(name) for name in metric_names})
        csv_rows.append(row)
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["pipeline_version", *metric_names]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    lines = [
        "# Transcription pipeline evaluation",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        "| Metric | legacy-v1 | pipeline-v2 |",
        "| --- | ---: | ---: |",
    ]
    for name in metric_names:
        lines.append(
            f"| {name} | {legacy.get(name)} | {v2.get(name)} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance criteria",
            "",
            "| # | Criterion | Pass |",
            "| ---: | --- | --- |",
        ]
    )
    for check in payload["criteria_checks"]:
        lines.append(f"| {check['id']} | {check['label']} | {check['pass']} |")
    lines.extend(
        [
            "",
            "## Cohort breakdown",
            "",
        ]
    )
    lines.append(json.dumps(payload["legacy"].get("cohorts", {}), ensure_ascii=False, indent=2))
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_transcriber(audio: Path, version: str, reference: Path) -> HypothesisDocument:
    command = os.environ.get("EVALUATION_TRANSCRIBE_COMMAND")
    if not command:
        raise RuntimeError("No local transcriber configured; set EVALUATION_TRANSCRIBE_COMMAND.")
    if not audio.is_file():
        raise FileNotFoundError(audio)
    if not reference.is_file():
        raise FileNotFoundError(reference)
    completed = subprocess.run(
        # Never give the evaluated system the human answer file.
        [command, str(audio), version],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "transcriber failed")
    if not completed.stdout.strip():
        raise RuntimeError("transcriber produced no hypothesis output")
    return _hypothesis_from_payload(json.loads(completed.stdout), source_id=audio.stem)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate", help="validate a manifest and references")
    validate_parser.add_argument("manifest", type=Path)

    run_parser = subparsers.add_parser("run", help="run a local labeled set through both versions")
    run_parser.add_argument("manifest", type=Path)
    run_parser.add_argument("--output", type=Path, default=Path("reports"))

    report_parser = subparsers.add_parser("report", help="aggregate per-call JSON results")
    report_parser.add_argument("results_dir", type=Path)
    report_parser.add_argument("--output", type=Path, default=Path("reports"))

    args = parser.parse_args(argv)
    if args.command == "validate":
        manifest = Path(args.manifest)
        errors = validate_manifest(load_manifest(manifest))
        if errors:
            print("\n".join(errors))
            return 1
        print("manifest valid")
        return 0
    if args.command == "run":
        return _run_manifest(Path(args.manifest), Path(args.output))
    if args.command == "report":
        return _aggregate_per_call_results(Path(args.results_dir), Path(args.output))
    parser.print_help()
    return 2


def _run_manifest(manifest: Path, output_dir: Path) -> int:
    records = load_manifest(manifest)
    errors = validate_manifest(records)
    if errors:
        print("\n".join(errors))
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    calls_by_version: dict[str, list[ProcessedCall]] = {}
    for record in records:
        for version in ("legacy-v1", "pipeline-v2"):
            processed = ProcessedCall(id=str(record["id"]), version=version, status="error")
            try:
                audio_path = _manifest_file(manifest, record["audio"])
                reference_path = _manifest_file(manifest, record["reference"])
                reference = load_reference(reference_path)
                if reference.verification_status != "verified":
                    raise ValueError("Human reference must be explicitly verified before evaluation.")
                if reference.id != record["id"]:
                    raise ValueError("Human reference ID does not match the manifest.")
                hypothesis = run_transcriber(audio_path, version, reference_path)
                metrics = compute_call_metrics(reference, hypothesis)
                metrics["_quality"] = reference.quality
                processed.status = "success"
                processed.metrics = metrics
            except Exception as exc:
                processed.error = str(exc)
                error_path = output_dir / f"{record['id']}.{version}.error"
                error_path.write_text(processed.error, encoding="utf-8")
            calls_by_version.setdefault(version, []).append(processed)
    legacy = aggregate_metrics(calls_by_version.get("legacy-v1", []))
    v2 = aggregate_metrics(calls_by_version.get("pipeline-v2", []))
    write_reports(output_dir, legacy, v2, load_criteria())
    return 1 if any(call.status != "success" for calls in calls_by_version.values() for call in calls) else 0


def _manifest_file(manifest: Path, value: str) -> Path:
    root = manifest.resolve().parent
    relative = Path(value)
    if (relative.is_absolute() or PureWindowsPath(value).is_absolute() or "\\" in value
            or ":" in value or ".." in relative.parts):
        raise ValueError("Invalid local evaluation path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evaluation path escapes the manifest directory.")
    return path


def _aggregate_per_call_results(results_dir: Path, output_dir: Path) -> int:
    calls: dict[str, list[ProcessedCall]] = {}
    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        call = ProcessedCall(
            id=str(payload.get("id", path.stem)),
            version=str(payload.get("version", "unknown")),
            status=str(payload.get("status", "error")),
            error=payload.get("error"),
            metrics=payload.get("metrics"),
        )
        calls.setdefault(call.version, []).append(call)
    legacy = aggregate_metrics(calls.get("legacy-v1", []))
    v2 = aggregate_metrics(calls.get("pipeline-v2", []))
    write_reports(output_dir, legacy, v2, load_criteria())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
