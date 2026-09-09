from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation import evaluate


def test_normalize_greek_strips_accents_and_case() -> None:
    assert evaluate.normalize_greek("ΚΑΛΗΜΈΡΑ") == "καλημερα"
    assert evaluate.normalize_greek("  Πόλη, στάση!  ") == "πολη σταση"
    assert evaluate.normalize_greek("") == ""


def test_wer_and_cer_are_deterministic() -> None:
    assert evaluate.wer("καλημερα", "καλημερα") == 0.0
    assert evaluate.cer("καλημερα", "καλημερα") == 0.0
    assert evaluate.wer("καλημερα", "καλησπερα") == 1.0
    assert evaluate.cer("αβ", "αγ") == 0.5


def test_manifest_validation(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "call-1",
                        "audio": "audio/call-1.wav",
                        "reference": "references/call-1.json",
                        "split": "test",
                        "quality": "noisy_mobile",
                        "mode": "stereo",
                    }
                ),
                json.dumps({"id": "call-2", "audio": "", "split": "train"}),
                "not-json",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        evaluate.load_manifest(manifest)


def test_manifest_validation_errors(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "call-1",
                        "audio": "audio/call-1.wav",
                        "reference": "references/call-1.json",
                        "split": "test",
                        "quality": "noisy_mobile",
                        "mode": "stereo",
                    }
                ),
                json.dumps({"id": "call-1", "audio": "audio/call-1.wav"}),
            ]
        ),
        encoding="utf-8",
    )
    errors = evaluate.validate_manifest(evaluate.load_manifest(manifest))
    assert any("duplicate id" in error for error in errors)
    assert any("missing" in error for error in errors)


def test_missing_audio_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALUATION_TRANSCRIBE_COMMAND", "unused")
    with pytest.raises(FileNotFoundError):
        evaluate.run_transcriber(
            Path("evaluation/audio/missing.wav"),
            "legacy-v1",
            Path("evaluation/references/missing.json"),
        )


def test_call_metrics_are_deterministic() -> None:
    reference = evaluate.ReferenceDocument(
        id="call-1",
        quality="noisy_mobile",
        mode="stereo",
        split="test",
        expected_keywords=["προσφορά", "ραντεβού"],
        segments=[
            evaluate.ReferenceSegment(
                speaker="Operator",
                channel=0,
                start=0,
                end=3,
                text="Καλημέρα, πώς μπορώ να σας εξυπηρετήσω;",
                entities={"names": []},
            ),
            evaluate.ReferenceSegment(
                speaker="Customer",
                channel=1,
                start=3.5,
                end=8,
                text="Θα ήθελα να κλείσω ένα ραντεβού για service.",
                entities={
                    "names": [],
                    "telephone_numbers": ["2100000000"],
                    "licence_plates": ["ABC-1234"],
                    "vehicle_models": ["Corolla"],
                },
            ),
        ],
    )
    hypothesis = evaluate.HypothesisDocument(
        id="call-1",
        segments=[
            evaluate.HypothesisSegment(
                speaker="Operator",
                channel=0,
                start=0,
                end=3,
                text="Καλημέρα, πώς μπορώ να σας εξυπηρετήσω;",
                entities={"names": []},
            ),
            evaluate.HypothesisSegment(
                speaker="Customer",
                channel=1,
                start=3.5,
                end=8,
                text="Θα ήθελα να κλείσω ένα ραντεβού για service.",
                entities={
                    "names": [],
                    "telephone_numbers": ["2100000000"],
                    "licence_plates": ["ABC-1234"],
                    "vehicle_models": ["Corolla"],
                },
                quality_flags=["human_review_recommended"],
            ),
        ],
        detected_keywords=["προσφορά", "ραντεβού"],
        processing_duration_seconds=12.0,
        audio_duration_seconds=8.0,
    )
    metrics = evaluate.compute_call_metrics(reference, hypothesis)
    assert metrics["wer"] == 0.0
    assert metrics["keyword_recall"] == 1.0
    assert metrics["keyword_precision"] == 1.0
    assert metrics["keyword_false_positive_rate"] == 0.0
    assert metrics["telephone_number_accuracy"] == 1.0
    assert metrics["licence_plate_accuracy"] == 1.0
    assert metrics["vehicle_model_accuracy"] == 1.0
    assert metrics["speaker_channel_attribution_accuracy"] == 1.0
    assert metrics["review_flag_rate"] == 0.5
    assert metrics["latency_per_audio_minute"] == pytest.approx(90.0)


def test_reports_are_generated_and_redacted(tmp_path: Path) -> None:
    legacy = {
        "calls_total": 1,
        "calls_successful": 1,
        "processing_success_rate": 1.0,
        "wer": 0.25,
        "keyword_recall": 0.9,
        "keyword_false_positive_rate": 0.05,
        "cohorts": {"noisy_mobile": {"calls": 1, "wer": 0.25}},
        "api_usage": {"api_key": "super-secret", "storage_key": "/app/storage/customer.wav"},
    }
    v2 = dict(legacy)
    v2["wer"] = 0.20
    v2["latency_per_audio_minute"] = 95.0
    v2["cost"] = 0.004
    evaluate.write_reports(tmp_path, legacy, v2)
    summary_text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "super-secret" not in summary_text
    assert "/app/storage" not in summary_text
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "report.md").exists()
