from __future__ import annotations

import asyncio
import io
import json
import os
import struct
import subprocess
import wave
from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.models import Keyword, KeywordCategory, KeywordMatch, Transcript, TranscriptSegment
from app.models.enums import MatchMethod
from app.schemas.evaluation import ReferenceDraft
from app.services.evaluation import EvaluationStore, contained_path
from test_api_acceptance import APIHarness, api_harness, _seed_speaker_assignment_call


def wav_bytes(channels=2, seconds=12) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        # Distinct samples prove channel extraction, not merely a successful HTTP response.
        audio.writeframes(struct.pack("<" + "h" * channels, *([1200, -2400][:channels])) * (8000 * seconds))
    return output.getvalue()


@pytest.fixture
def dataset(api_harness: APIHarness, tmp_path: Path):
    root = tmp_path / "evaluation"
    for folder in ("audio", "references", "context"):
        (root / folder).mkdir(parents=True)
    records = []
    for record_id, split, mode in (("eval-001", "dev", "stereo"), ("eval-002", "test", "mono")):
        (root / "audio" / f"{record_id}.wav").write_bytes(wav_bytes(2 if mode == "stereo" else 1))
        context = {
            "direction": "inbound", "queue": "Sales", "is_queue": True,
            "transfer_state": "transferred", "audio_topology": mode,
            "operators": [{"name": "PBX Operator", "extension": "201", "api_key": "hidden"}],
            "caller": "+302101234567", "callee": "+302109876543",
            "production_transcript": "ASR SENTINEL NEVER SHOW",
            "original_text": "ASR SENTINEL NEVER SHOW",
            "api_key": "SECRET SENTINEL", "storage_key": "/private/file",
            "matches": ["PRODUCTION MATCH SENTINEL"],
        }
        (root / "context" / f"{record_id}.json").write_text(json.dumps(context), encoding="utf-8")
        records.append({
            "id": record_id, "audio": f"audio/{record_id}.wav",
            "reference": f"references/{record_id}.json", "context": f"context/{record_id}.json",
            "split": split, "mode": mode,
        })
    manifest = root / "manifest.local.jsonl"
    manifest.write_text("\n".join(map(json.dumps, records)), encoding="utf-8")
    api_harness.settings.EVALUATION_ROOT = root
    api_harness.settings.EVALUATION_UI_ENABLED = True
    return root, records


def valid_draft():
    return {
        "quality": "noisy", "operator_channel": 1, "operator_channel_answered": True,
        "expected_keywords": ["προσφορά"],
        "segments": [
            {"speaker": "Operator", "channel": 1, "start": 0.25, "end": 3.5,
             "text": "Καλημέρα, προσφορά", "exclude_from_wer": False,
             "entities": {"names": ["Test Name"], "telephone_numbers": ["2100000000"],
                          "licence_plates": ["ABC-1234"], "vehicle_models": ["Example Model"]}},
            {"speaker": "Customer", "channel": 0, "start": 1.25, "end": 4.25,
             "text": "", "exclude_from_wer": True, "entities": {}},
        ],
    }


async def get_reference(harness):
    response = await harness.client.get("/api/evaluation/eval-001")
    assert response.status_code == 200, response.text
    return response.json()["reference"]


async def save(harness, draft=None, revision=None):
    revision = revision or (await get_reference(harness))["revision"]
    return await harness.client.put("/api/evaluation/eval-001/reference",
                                    headers=harness.csrf_headers(),
                                    json={**(draft or valid_draft()), "revision": revision})


async def verify(harness, revision=None):
    revision = revision or (await get_reference(harness))["revision"]
    return await harness.client.post("/api/evaluation/eval-001/verify",
                                     headers=harness.csrf_headers(), json={"revision": revision})


async def snapshot(harness):
    async with harness.sessions() as session:
        return {model.__name__: [dict(row) for row in
                (await session.execute(select(model.__table__).order_by(model.id))).mappings()]
                for model in (Transcript, TranscriptSegment, KeywordMatch)}


async def seed_production(harness):
    ids = await _seed_speaker_assignment_call(harness, channel_texts={0: "PRIVATE ASR TEXT", 1: "PRIVATE ASR OTHER"})
    async with harness.sessions() as session:
        category = KeywordCategory(name="Active category")
        inactive_category = KeywordCategory(name="Inactive category", active=False)
        session.add_all([category, inactive_category])
        await session.flush()
        keyword = Keyword(category_id=category.id, canonical_phrase="προσφορά", normalized_phrase="προσφορα")
        session.add_all([
            keyword,
            Keyword(category_id=category.id, canonical_phrase="Inactive keyword",
                    normalized_phrase="inactive", active=False),
            Keyword(category_id=inactive_category.id, canonical_phrase="Hidden category keyword",
                    normalized_phrase="hidden"),
        ])
        await session.flush()
        segment = await session.scalar(select(TranscriptSegment).where(
            TranscriptSegment.transcript_id == ids["transcript_id"]))
        session.add(KeywordMatch(
            keyword_id=keyword.id, call_id=ids["call_id"], transcript_segment_id=segment.id,
            original_matched_text="PRIVATE MATCH", normalized_match="private match",
            start_seconds=0, end_seconds=1, match_method=MatchMethod.EXACT_PHRASE, match_score=1,
        ))
        await session.commit()
    return ids


async def test_feature_defaults_disabled_and_routes_unavailable(api_harness, dataset):
    assert Settings(_env_file=None).EVALUATION_UI_ENABLED is False
    await api_harness.login()
    api_harness.settings.EVALUATION_UI_ENABLED = False
    assert (await api_harness.client.get("/api/features")).json() == {"evaluation_ui_enabled": False}
    for endpoint in ("", "/keywords", "/eval-001", "/eval-001/audio",
                     "/eval-001/audio/channel/0", "/eval-001/audio/channel/1"):
        response = await api_harness.client.get("/api/evaluation" + endpoint)
        assert response.status_code == 404
    assert (await save(api_harness, revision="0" * 64)).status_code == 404
    assert (await verify(api_harness, revision="0" * 64)).status_code == 404
    assert list((dataset[0] / "references").iterdir()) == []


async def test_evaluation_authentication_and_csrf(api_harness, dataset):
    for endpoint in ("/api/features", "/api/evaluation", "/api/evaluation/keywords",
                     "/api/evaluation/eval-001", "/api/evaluation/eval-001/audio"):
        assert (await api_harness.client.get(endpoint)).status_code == 401
    await api_harness.login()
    response = await api_harness.client.put("/api/evaluation/eval-001/reference",
                                           json={**valid_draft(), "revision": "0" * 64})
    assert response.status_code == 403
    assert (await api_harness.client.post("/api/evaluation/eval-001/verify",
                                         json={"revision": "0" * 64})).status_code == 403


async def test_reference_workflow_never_mutates_production_or_returns_asr(api_harness, dataset):
    ids = await seed_production(api_harness)
    before = await snapshot(api_harness)
    await api_harness.login()
    production_before = (await api_harness.client.get(f"/api/calls/{ids['call_id']}")).json()
    response = await api_harness.client.get("/api/evaluation/eval-001")
    assert response.status_code == 200
    for forbidden in ("PRIVATE ASR", "ASR SENTINEL", "SECRET SENTINEL", "hidden",
                      "storage_key", "production_transcript", "original_text", "matches",
                      "+302101234567", "+302109876543", str(ids["call_id"])):
        assert forbidden not in response.text
    assert response.json()["reference"]["segments"] == []
    assert response.json()["reference"]["quality"] is None
    assert response.json()["reference"]["operator_channel_answered"] is False
    assert not (dataset[0] / "references" / "eval-001.json").exists()
    assert await snapshot(api_harness) == before

    catalog = await api_harness.client.get("/api/evaluation/keywords")
    assert [keyword["canonical_phrase"] for keyword in catalog.json()] == ["προσφορά"]
    saved = await save(api_harness)
    assert saved.status_code == 200, saved.text
    assert saved.json()["verification_status"] == "in_progress"
    assert await snapshot(api_harness) == before
    verified = await verify(api_harness)
    assert verified.status_code == 200, verified.text
    assert verified.json()["verification_status"] == "verified"
    assert verified.json()["verified_at"]
    assert await snapshot(api_harness) == before
    assert (await api_harness.client.get(f"/api/calls/{ids['call_id']}")).json() == production_before

    # Every human field round-trips, including overlap, exclusions, entities and Channel B truth.
    reference = await get_reference(api_harness)
    for key in ("quality", "operator_channel", "operator_channel_answered", "expected_keywords"):
        assert reference[key] == valid_draft()[key]
    assert reference["segments"][0]["entities"] == valid_draft()["segments"][0]["entities"]
    assert reference["segments"][1]["exclude_from_wer"] is True
    assert reference["segments"][1]["text"] == ""
    assert reference["segments"][1]["start"] < reference["segments"][0]["end"]
    raw = json.loads((dataset[0] / "references" / "eval-001.json").read_text(encoding="utf-8"))
    assert raw["id"] == "eval-001" and raw["evaluation_id"] == "eval-001"
    assert str(api_harness.admin_id) not in json.dumps(raw)
    assert "verified_by" not in raw and "annotator" not in raw

    changed = valid_draft()
    changed["segments"][0]["text"] = "Manually edited"
    changed["operator_channel"] = 0
    changed["segments"].pop()
    edited = await save(api_harness, changed)
    assert edited.status_code == 200
    assert edited.json()["verification_status"] == "in_progress"
    assert edited.json()["verified_at"] is None
    assert (await get_reference(api_harness))["segments"][0]["text"] == "Manually edited"
    assert len((await get_reference(api_harness))["segments"]) == 1
    assert await snapshot(api_harness) == before


async def test_progress_filters_and_frozen_audio_survive_no_production_call(api_harness, dataset):
    await api_harness.login()
    await save(api_harness)
    await verify(api_harness)
    result = (await api_harness.client.get("/api/evaluation")).json()
    assert result["progress"] == {"all": {"verified": 1, "total": 2},
                                   "dev": {"verified": 1, "total": 1},
                                   "test": {"verified": 0, "total": 1}}
    for filter, ids in (("dev", ["eval-001"]), ("verified", ["eval-001"]),
                        ("test", ["eval-002"]), ("unverified", ["eval-002"])):
        result = (await api_harness.client.get("/api/evaluation", params={"filter": filter})).json()
        assert [row["evaluation_id"] for row in result["items"]] == ids
    assert (await api_harness.client.get("/api/evaluation/eval-001/audio")).content == wav_bytes()


@pytest.mark.parametrize("channel,sample", [(0, 1200), (1, -2400)])
async def test_stereo_channel_preview_is_correct_and_original_unchanged(api_harness, dataset, channel, sample):
    await api_harness.login()
    original = (dataset[0] / "audio" / "eval-001.wav").read_bytes()
    response = await api_harness.client.get(f"/api/evaluation/eval-001/audio/channel/{channel}")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(response.content), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getnframes() / audio.getframerate() == 12
        assert struct.unpack("<h", audio.readframes(1))[0] == sample
    assert (dataset[0] / "audio" / "eval-001.wav").read_bytes() == original
    assert list((dataset[0] / "previews").glob("*.wav"))
    assert (await api_harness.client.get(f"/api/evaluation/eval-001/audio/channel/{channel}")).content == response.content


@pytest.mark.parametrize("suffix", ["audio", "audio/channel/0", "audio/channel/1"])
async def test_audio_range_scrubbing_head_and_invalid_ranges(api_harness, dataset, suffix):
    await api_harness.login()
    url = "/api/evaluation/eval-001/" + suffix
    full = await api_harness.client.get(url)
    partial = await api_harness.client.get(url, headers={"Range": "bytes=12-43"})
    assert partial.status_code == 206
    assert partial.content == full.content[12:44]
    assert partial.headers["content-range"].startswith("bytes 12-43/")
    assert partial.headers["cache-control"] == "no-store"
    assert (await api_harness.client.head(url)).content == b""
    assert (await api_harness.client.get(url, headers={"Range": "bytes=9999999-"})).status_code == 416


@pytest.mark.parametrize("field,value", [
    ("quality", None), ("segments", []), ("operator_channel_answered", False),
])
async def test_verification_required_answers(api_harness, dataset, field, value):
    await api_harness.login()
    draft = valid_draft()
    draft[field] = value
    assert (await save(api_harness, draft)).status_code == 200
    assert (await verify(api_harness)).status_code == 422
    assert (await get_reference(api_harness))["verified_at"] is None


@pytest.mark.parametrize("patch", [
    {"start": 3.5, "end": 3.5}, {"start": 4, "end": 2},
    {"start": 12, "end": 13}, {"end": 12.001}, {"text": " "},
])
async def test_verification_rejects_invalid_segments(api_harness, dataset, patch):
    await api_harness.login()
    draft = valid_draft()
    draft["segments"][0].update(patch)
    assert (await save(api_harness, draft)).status_code == 200
    assert (await verify(api_harness)).status_code == 422


@pytest.mark.parametrize("patch", [
    {"quality": "made_up"}, {"operator_channel": 2}, {"operator_channel": True},
    {"production_transcript": "Cannot write"}, {"verified_at": "2026-01-01"},
    {"verification_status": "verified"}, {"reference": "../outside.json"},
])
async def test_reference_rejects_unknown_or_invalid_fields(api_harness, dataset, patch):
    await api_harness.login()
    response = await save(api_harness, {**valid_draft(), **patch})
    assert response.status_code == 422
    assert not (dataset[0] / "references" / "eval-001.json").exists()


async def test_explicit_unknown_operator_channel_is_valid(api_harness, dataset):
    await api_harness.login()
    draft = valid_draft()
    draft["operator_channel"] = None
    await save(api_harness, draft)
    assert (await verify(api_harness)).status_code == 200


async def test_stale_save_verify_and_concurrent_writes_are_rejected(api_harness, dataset):
    await api_harness.login()
    revision = (await get_reference(api_harness))["revision"]
    first, second = await asyncio.gather(
        save(api_harness, revision=revision), save(api_harness, revision=revision))
    assert sorted([first.status_code, second.status_code]) == [200, 409]
    assert (await verify(api_harness, revision)).status_code == 409
    assert (await save(api_harness, revision=revision)).status_code == 409
    assert (await verify(api_harness)).status_code == 200


@pytest.mark.parametrize("path", [
    "../outside.json", "/tmp/outside.json", "C:/private.json", "C:\\private.json",
    "//server/share/file", "references/../../private.json", "references/link:secret",
    "references\\..\\private.json",
])
def test_contained_paths_reject_traversal(tmp_path, path):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        contained_path(tmp_path, path)
    assert error.value.status_code == 400


@pytest.mark.parametrize("endpoint", [
    "/api/evaluation/%2E%2E%5Cprivate", "/api/evaluation/C%3A%5Cprivate",
    "/api/evaluation/%2Fetc%2Fpasswd", "/api/evaluation/eval-001/audio/channel/2",
    "/api/evaluation/eval-002/audio/channel/1",
])
async def test_browser_cannot_supply_paths_or_invalid_channels(api_harness, dataset, endpoint):
    await api_harness.login()
    assert (await api_harness.client.get(endpoint)).status_code in {400, 404, 422}


async def test_manifest_path_traversal_does_not_expose_local_file(api_harness, dataset, tmp_path):
    await api_harness.login()
    root, records = dataset
    secret = tmp_path / "outside.wav"
    secret.write_bytes(b"LOCAL SECRET")
    records[0]["audio"] = "../outside.wav"
    (root / "manifest.local.jsonl").write_text("\n".join(map(json.dumps, records)))
    response = await api_harness.client.get("/api/evaluation/eval-001/audio")
    assert response.status_code == 400
    assert "LOCAL SECRET" not in response.text
    assert str(tmp_path) not in response.text


@pytest.mark.parametrize("folder", ["audio", "references", "context", "previews"])
def test_symlink_or_windows_junction_escape_is_rejected(tmp_path, folder):
    from fastapi import HTTPException
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / folder
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                       check=True, capture_output=True)
        assert link.is_junction()
    else:
        link.symlink_to(outside, target_is_directory=True)
        assert link.is_symlink()
    with pytest.raises(HTTPException) as error:
        contained_path(root, f"{folder}/file.json", folder=folder)
    assert error.value.status_code == 400


def test_real_evaluation_data_is_git_ignored():
    repo = Path(__file__).resolve().parents[2]
    paths = ["evaluation/audio/test.wav", "evaluation/references/test.json",
             "evaluation/context/test.json", "evaluation/previews/test.wav",
             "evaluation/previews/test.json", "evaluation/manifest.local.jsonl",
             "evaluation/reports/test.json"]
    result = subprocess.run(["git", "check-ignore", "--stdin", "-z"], input=("\0".join(paths) + "\0").encode(),
                            capture_output=True, cwd=repo, check=True)
    assert set(result.stdout.decode().strip("\0").split("\0")) == set(paths)
    assert not (repo / "evaluation" / "label_server.py").exists()


def test_no_production_schema_in_reference_service():
    from app.services import evaluation
    source = Path(evaluation.__file__).read_text(encoding="utf-8")
    assert "from app.models" not in source
    assert "from app.services.transcription" not in source


def test_mono_verification_does_not_require_operator_channel():
    draft = ReferenceDraft(quality="normal", segments=[{
        "speaker": "Operator", "start": 0, "end": 2, "text": "hello",
    }])
    assert EvaluationStore.verification_errors(draft, 3, "mono") == []
