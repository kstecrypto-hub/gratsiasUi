from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.enums import ParticipantRole
from app.services.audio.errors import AudioToolError
from app.services.audio.processor import AudioProcessor
from app.services.yeastar.interpretation import InterpretedParticipant, safe_operator_channel


def _settings(storage_root: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=storage_root,
        OPENAI_API_KEY="test-only-key",
    )


def _minimal_wav_header() -> bytes:
    return b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt "


def _ffprobe_payload(*, duration: float = 2.0, channels: int = 2) -> bytes:
    return json.dumps(
        {
            "streams": [
                {
                    "codec_name": "pcm_s16le",
                    "channels": channels,
                    "sample_rate": "16000",
                    "bit_rate": "512000",
                }
            ],
            "format": {
                "format_name": "wav",
                "duration": str(duration),
                "size": "16",
                "bit_rate": "512000",
            },
        }
    ).encode()


@pytest.mark.asyncio
async def test_inspect_requires_a_successful_full_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(_minimal_wav_header())
    processor = AudioProcessor(_settings(tmp_path))
    calls: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        calls.append(args)
        if args[0] == "ffprobe":
            return _ffprobe_payload(), b""
        raise AudioToolError("Audio file could not be decoded.")

    monkeypatch.setattr(processor, "_run", fake_run)

    with pytest.raises(AudioToolError):
        await processor.inspect(source, declared_mime_type="audio/wav")

    decode = calls[1]
    assert decode[0] == "ffmpeg"
    assert "-xerror" in decode
    assert decode[decode.index("-map") + 1] == "0:a:0"
    assert decode[-2:] == ("null", "-")


@pytest.mark.asyncio
async def test_inspect_returns_probe_metadata_after_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(_minimal_wav_header())
    processor = AudioProcessor(_settings(tmp_path))

    async def fake_run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        if args[0] == "ffprobe":
            return _ffprobe_payload(duration=7.25, channels=1), b""
        return b"", b""

    monkeypatch.setattr(processor, "_run", fake_run)

    info = await processor.inspect(source, declared_mime_type="audio/wav")

    assert info.duration_seconds == 7.25
    assert info.channel_count == 1
    assert info.sample_rate_hz == 16000
    assert info.sha256_checksum == AudioProcessor.sha256(source)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", [0, 1])
async def test_extract_channel_maps_the_first_audio_stream_with_pan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, channel: int
) -> None:
    source = tmp_path / "stereo.wav"
    source.write_bytes(_minimal_wav_header())
    destination = tmp_path / f"operator-{channel}.wav"
    processor = AudioProcessor(_settings(tmp_path))
    command: tuple[str, ...] | None = None

    async def fake_run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        nonlocal command
        command = args
        Path(args[-1]).write_bytes(_minimal_wav_header())
        return b"", b""

    monkeypatch.setattr(processor, "_run", fake_run)

    result = await processor.extract_channel(source, destination, channel)

    assert result == destination
    assert destination.is_file()
    assert command is not None
    assert command[command.index("-map") + 1] == "0:a:0"
    assert command[command.index("-filter:a") + 1] == f"pan=mono|c0=c{channel}"
    assert command[command.index("-ac") + 1] == "1"


@pytest.mark.asyncio
async def test_split_audio_removes_completed_and_partial_chunks_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "recording.wav"
    source.write_bytes(_minimal_wav_header())
    destination_dir = tmp_path / "chunks"
    processor = AudioProcessor(_settings(tmp_path))
    invocation = 0

    async def fake_run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        nonlocal invocation
        invocation += 1
        Path(args[-1]).write_bytes(_minimal_wav_header())
        if invocation == 2:
            raise AudioToolError("Audio file could not be decoded.")
        return b"", b""

    monkeypatch.setattr(processor, "_run", fake_run)

    with pytest.raises(AudioToolError):
        await processor.split_audio(source, destination_dir, 2.0, chunk_seconds=1.0)

    assert list(destination_dir.glob("chunk-*.wav")) == []


def test_channel_selection_falls_back_when_attribution_is_not_exclusive() -> None:
    participant = InterpretedParticipant(
        operator_id="operator-1",
        provider_extension_id="extension-1",
        extension_number="101",
        leg_id="leg-1",
        role=ParticipantRole.CALLER,
        was_caller=True,
        was_callee=False,
        answered=True,
    )

    assert safe_operator_channel(participant, 2, True) is None
    assert safe_operator_channel(participant, 2, True, one_to_one=True, was_transferred=True) is None
    assert (
        safe_operator_channel(
            participant,
            2,
            True,
            one_to_one=True,
            operators_on_same_side=2,
        )
        is None
    )
    assert safe_operator_channel(participant, 1, True, one_to_one=True) is None


def test_channel_selection_allows_only_explicit_one_to_one_stereo() -> None:
    caller = InterpretedParticipant(
        operator_id="operator-1",
        provider_extension_id="extension-1",
        extension_number="101",
        leg_id="leg-1",
        role=ParticipantRole.CALLER,
        was_caller=True,
        was_callee=False,
        answered=True,
    )
    callee = InterpretedParticipant(
        operator_id="operator-2",
        provider_extension_id="extension-2",
        extension_number="102",
        leg_id="leg-1",
        role=ParticipantRole.CALLEE,
        was_caller=False,
        was_callee=True,
        answered=True,
    )

    assert safe_operator_channel(caller, 2, True, one_to_one=True) == 0
    assert safe_operator_channel(callee, 2, True, one_to_one=True) == 1
