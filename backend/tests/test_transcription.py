from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.services.audio import AudioChunk
from app.services.transcription.client import (
    OpenAITranscriptionClient,
    TranscriptionCancelledError,
)


def _settings(storage_root: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=storage_root,
        OPENAI_API_KEY="test-only-key",
    )


def _fake_openai(*responses: object) -> tuple[SimpleNamespace, AsyncMock]:
    create = AsyncMock(side_effect=responses)
    client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)),
        models=SimpleNamespace(retrieve=AsyncMock()),
    )
    client.with_options = lambda **_: client
    return client, create


def _chunk(tmp_path: Path, sequence: int, start: float, end: float) -> AudioChunk:
    path = tmp_path / f"chunk-{sequence}.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    return AudioChunk(path=path, start_seconds=start, end_seconds=end)


@pytest.mark.asyncio
async def test_context_manager_closes_only_the_client_it_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owned = SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        "app.services.transcription.client.AsyncOpenAI", lambda **_: owned
    )

    async with OpenAITranscriptionClient(_settings(tmp_path)) as client:
        assert client._client is owned

    owned.close.assert_awaited_once()
    assert client._client is None

    injected = SimpleNamespace(close=AsyncMock())
    async with OpenAITranscriptionClient(_settings(tmp_path), client=injected):
        pass
    injected.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_isolated_transcription_uses_chunk_boundaries_and_supported_contract(
    tmp_path: Path,
) -> None:
    response = SimpleNamespace(text="  Γεια σας  ", usage={"input_tokens": 7, "output_tokens": 3})
    fake, create = _fake_openai(response)
    client = OpenAITranscriptionClient(_settings(tmp_path), client=fake)
    chunk = _chunk(tmp_path, 1, 30.0, 45.0)

    result = await client.transcribe_isolated([chunk], ["Yeastar", "Γιώργος"])

    assert result.diarized is False
    assert len(result.segments) == 1
    assert result.segments[0].start_seconds == 30.0
    assert result.segments[0].end_seconds == 45.0
    assert result.segments[0].speaker_label == "Operator"
    assert result.usage["totals"] == {"input_tokens": 7, "output_tokens": 3}
    request = create.await_args.kwargs
    assert request["model"] == "gpt-4o-transcribe"
    assert request["language"] == "el"
    assert request["response_format"] == "json"
    assert "Γιώργος" in request["prompt"]
    assert "timestamp_granularities" not in request


@pytest.mark.asyncio
async def test_diarization_namespaces_chunk_speakers_and_keeps_unknown_unknown(
    tmp_path: Path,
) -> None:
    first = {
        "segments": [
            {"start": 2.0, "end": 4.0, "text": "πρώτο"},
        ],
        "usage": {"input_tokens": 11},
    }
    second = {
        "segments": [
            {"start": 1.5, "end": 3.5, "text": "δεύτερο", "speaker": "A"},
        ],
        "usage": {"input_tokens": 13},
    }
    fake, create = _fake_openai(first, second)
    client = OpenAITranscriptionClient(_settings(tmp_path), client=fake)
    chunks = [_chunk(tmp_path, 1, 60.0, 90.0), _chunk(tmp_path, 2, 90.0, 120.0)]

    result = await client.transcribe_diarized(chunks)

    assert result.diarized is True
    assert [(segment.start_seconds, segment.end_seconds) for segment in result.segments] == [
        (62.0, 64.0),
        (91.5, 93.5),
    ]
    assert [segment.speaker_label for segment in result.segments] == [
        "chunk-1:Unknown speaker",
        "chunk-2:A",
    ]
    assert result.usage["totals"] == {"input_tokens": 24}
    for awaited in create.await_args_list:
        request = awaited.kwargs
        assert request["model"] == "gpt-4o-transcribe-diarize"
        assert request["response_format"] == "diarized_json"
        assert request["chunking_strategy"] == "auto"
        assert "prompt" not in request
        assert "timestamp_granularities" not in request


@pytest.mark.asyncio
async def test_diarization_clamps_timestamps_and_ignores_non_finite_values(tmp_path: Path) -> None:
    response = {
        "segments": [
            {"start": -2.0, "end": 99.0, "text": "bounded", "speaker": "A"},
            {"start": float("nan"), "end": 2.0, "text": "invalid", "speaker": "B"},
        ]
    }
    fake, _ = _fake_openai(response)
    client = OpenAITranscriptionClient(_settings(tmp_path), client=fake)

    result = await client.transcribe_diarized([_chunk(tmp_path, 1, 100.0, 110.0)])

    assert [(segment.start_seconds, segment.end_seconds) for segment in result.segments] == [
        (100.0, 110.0)
    ]


@pytest.mark.asyncio
async def test_complete_diarization_uses_one_anonymous_supported_request(
    tmp_path: Path,
) -> None:
    response = {
        "duration": 30.0,
        "segments": [
            {"start": 1.0, "end": 4.0, "text": "rough text", "speaker": "A"},
        ],
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }
    fake, create = _fake_openai(response)
    client = OpenAITranscriptionClient(_settings(tmp_path), client=fake)
    chunk = _chunk(tmp_path, 1, 0.0, 30.0)

    result = await client.transcribe_diarized_complete(chunk)

    assert create.await_count == 1
    assert result.segments[0].speaker_label == "A"
    assert result.segments[0].start_seconds == 1.0
    assert result.segments[0].end_seconds == 4.0
    assert result.usage["totals"] == {"input_tokens": 11, "output_tokens": 3}
    request = create.await_args.kwargs
    assert request["model"] == "gpt-4o-transcribe-diarize"
    assert request["language"] == "el"
    assert request["response_format"] == "diarized_json"
    assert request["chunking_strategy"] == "auto"
    for forbidden in (
        "prompt",
        "include",
        "known_speaker_names",
        "known_speaker_references",
        "timestamp_granularities",
    ):
        assert forbidden not in request


@pytest.mark.asyncio
@pytest.mark.parametrize("diarized", [False, True])
async def test_cancellation_is_checked_between_chunks_before_another_upload(
    tmp_path: Path, diarized: bool
) -> None:
    first_response: object = (
        {"segments": [{"start": 0.0, "end": 1.0, "text": "ένα", "speaker": "A"}]}
        if diarized
        else SimpleNamespace(text="ένα", usage={})
    )
    # A second response is provided deliberately; cancellation must prevent it from being consumed.
    second_response: object = (
        {"segments": [{"start": 0.0, "end": 1.0, "text": "δύο", "speaker": "A"}]}
        if diarized
        else SimpleNamespace(text="δύο", usage={})
    )
    fake, create = _fake_openai(first_response, second_response)
    client = OpenAITranscriptionClient(_settings(tmp_path), client=fake)
    chunks = [_chunk(tmp_path, 1, 0.0, 10.0), _chunk(tmp_path, 2, 10.0, 20.0)]
    checks = 0

    async def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(TranscriptionCancelledError) as raised:
        if diarized:
            await client.transcribe_diarized(chunks, should_cancel=should_cancel)
        else:
            await client.transcribe_isolated(chunks, [], should_cancel=should_cancel)

    assert raised.value.category == "cancelled"
    assert checks == 2
    assert create.await_count == 1
