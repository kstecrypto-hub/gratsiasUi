from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation import evaluate

def test_human_channel_b_truth_scores_text_separately_from_swapped_speaker_roles():
    reference = evaluate.ReferenceDocument(
        id="call", mode="stereo", operator_channel=1,
        segments=[
            evaluate.ReferenceSegment("Operator", 1, 0, 2, "operator words"),
            evaluate.ReferenceSegment("Customer", 0, 2, 4, "customer words"),
        ],
    )
    hypothesis = evaluate.HypothesisDocument(
        id="call", segments=[
            evaluate.HypothesisSegment("Customer", 1, 0, 2, "operator words"),
            evaluate.HypothesisSegment("Operator", 0, 2, 4, "customer words"),
        ],
    )
    metrics = evaluate.compute_call_metrics(reference, hypothesis)
    assert metrics["wer"] == metrics["operator_wer"] == metrics["customer_wer"] == 0
    assert metrics["speaker_channel_attribution_accuracy"] == 0


def test_mono_roles_do_not_require_channels():
    reference = evaluate.ReferenceDocument(id="call", mode="mono", segments=[
        evaluate.ReferenceSegment("Operator", None, 0, 2, "hello"),
        evaluate.ReferenceSegment("Customer", None, 2, 4, "world"),
    ])
    hypothesis = evaluate.HypothesisDocument(id="call", segments=[
        evaluate.HypothesisSegment("Operator", None, 0, 2, "hello"),
        evaluate.HypothesisSegment("Customer", None, 2, 4, "world"),
    ])
    metrics = evaluate.compute_call_metrics(reference, hypothesis)
    assert metrics["operator_wer"] == metrics["customer_wer"] == 0


def test_unintelligible_notes_and_corresponding_hypothesis_are_excluded(tmp_path):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({
        "id": "call", "mode": "stereo", "operator_channel": 1,
        "verification_status": "verified", "segments": [
            {"speaker": "Operator", "channel": 1, "start": 0, "end": 2, "text": "hello"},
            {"speaker": "Customer", "channel": 0, "start": 2, "end": 4,
             "text": "THIS NOTE MUST NOT ENTER WER", "exclude_from_wer": True},
        ],
    }), encoding="utf-8")
    reference = evaluate.load_reference(path)
    assert reference.operator_channel == 1
    assert reference.verification_status == "verified"
    assert reference.segments[1].exclude_from_wer is True
    hypothesis = evaluate.HypothesisDocument(id="call", segments=[
        evaluate.HypothesisSegment("Operator", 1, 0, 2, "hello"),
        evaluate.HypothesisSegment("Customer", 0, 2, 4, "noise hallucination"),
    ])
    metrics = evaluate.compute_call_metrics(reference, hypothesis)
    assert metrics["wer"] == metrics["cer"] == metrics["raw_wer"] == 0


def test_partial_exclusions_require_timestamps_instead_of_silently_biasing_wer():
    reference = evaluate.ReferenceDocument(id="call", segments=[
        evaluate.ReferenceSegment("Operator", None, 0, 1, "hello"),
        evaluate.ReferenceSegment("Operator", None, 1, 2, "", exclude_from_wer=True),
    ])
    hypothesis = evaluate.HypothesisDocument(id="call", segments=[
        evaluate.HypothesisSegment("Operator", None, 0, 2, "hello noise"),
    ])
    with pytest.raises(ValueError, match="exclusion boundaries"):
        evaluate.compute_call_metrics(reference, hypothesis)


def test_exclusions_preserve_the_other_stereo_channel():
    reference = evaluate.ReferenceDocument(id="call", mode="stereo", segments=[
        evaluate.ReferenceSegment("Operator", 1, 0, 2, "hello"),
        evaluate.ReferenceSegment("Customer", 0, 0, 2, "", exclude_from_wer=True),
    ])
    hypothesis = evaluate.HypothesisDocument(id="call", segments=[
        evaluate.HypothesisSegment("Operator", 1, 0, 2, "hello"),
        evaluate.HypothesisSegment("Customer", 0, 0, 2, "noise"),
    ])
    assert evaluate.compute_call_metrics(reference, hypothesis)["wer"] == 0


def test_manifest_relative_paths_verification_gate_and_no_reference_given_to_transcriber(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root = tmp_path / "evaluation"
    (root / "audio").mkdir(parents=True)
    (root / "references").mkdir()
    audio = root / "audio" / "call.wav"
    audio.write_bytes(b"fixture")
    reference = root / "references" / "call.json"
    payload = {"id": "call", "quality": "clean", "mode": "mono", "verification_status": "in_progress",
               "segments": [{"speaker": "Operator", "start": 0, "end": 1, "text": "hello"}]}
    reference.write_text(json.dumps(payload))
    manifest = root / "manifest.local.jsonl"
    manifest.write_text(json.dumps({
        "evaluation_id": "call", "mode": "mono", "split": "test",
        "audio": "audio/call.wav", "reference": "references/call.json",
    }))
    calls = []
    def transcribe(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(payload))
    monkeypatch.setenv("EVALUATION_TRANSCRIBE_COMMAND", "test-transcriber")
    monkeypatch.setattr(evaluate.subprocess, "run", transcribe)
    output = root / "reports"
    assert evaluate._run_manifest(manifest, output) == 1
    assert calls == []
    payload["verification_status"] = "verified"
    reference.write_text(json.dumps(payload))
    assert evaluate._run_manifest(manifest, output) == 0
    assert calls == [["test-transcriber", str(audio), "legacy-v1"],
                     ["test-transcriber", str(audio), "pipeline-v2"]]


def test_overlapping_stereo_attribution_matches_the_correct_channel():
    reference = evaluate.ReferenceDocument(id="call", mode="stereo", segments=[
        evaluate.ReferenceSegment("Operator", 1, 0, 2, "hello"),
        evaluate.ReferenceSegment("Customer", 0, 0, 2, "world"),
    ])
    hypothesis = evaluate.HypothesisDocument(id="call", segments=[
        evaluate.HypothesisSegment("Operator", 1, 0, 2, "hello"),
        evaluate.HypothesisSegment("Customer", 0, 0, 2, "world"),
    ])
    assert evaluate.compute_call_metrics(reference, hypothesis)["speaker_channel_attribution_accuracy"] == 1



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
