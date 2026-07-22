from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.services.audio.errors import AudioToolError, InvalidAudioError


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".opus"}
ALLOWED_MIME_TYPES = {
    "application/octet-stream",
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp3",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "audio/opus",
    "audio/vnd.wave",
    "audio/wav",
    "audio/wave",
    "audio/x-flac",
    "audio/x-m4a",
    "audio/x-wav",
}


@dataclass(frozen=True)
class AudioInfo:
    codec_name: str
    format_name: str
    duration_seconds: float
    channel_count: int
    sample_rate_hz: int
    bit_rate_bps: int | None
    size_bytes: int
    sha256_checksum: str


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start_seconds: float
    end_seconds: float


class AudioProcessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def safe_storage_path(self, relative_key: str) -> Path:
        if not relative_key or "\x00" in relative_key:
            raise InvalidAudioError("Invalid audio storage key.")
        root = self.settings.STORAGE_ROOT.resolve()
        result = (root / relative_key).resolve()
        if not result.is_relative_to(root):
            raise InvalidAudioError("Audio path is outside application storage.")
        return result

    @staticmethod
    async def _run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AudioToolError("Required audio processing tool is unavailable.") from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise AudioToolError("Audio processing timed out.") from exc
        if process.returncode != 0:
            # FFmpeg output can contain local paths; do not pass it through to logs or clients.
            raise AudioToolError("Audio file could not be decoded.")
        return stdout, stderr

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    async def inspect(
        self,
        path: Path,
        *,
        declared_mime_type: str | None = None,
        original_filename: str | None = None,
    ) -> AudioInfo:
        original_path = path
        if original_path.is_symlink():
            raise InvalidAudioError("Audio file location is invalid.")
        path = original_path.resolve()
        root = self.settings.STORAGE_ROOT.resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise InvalidAudioError("Audio file location is invalid.")
        size = path.stat().st_size
        if size <= 0 or size > self.settings.MAX_RECORDING_BYTES:
            raise InvalidAudioError("Audio file size is invalid.")
        extension = Path(original_filename or path.name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise InvalidAudioError("Recording file type is not supported.")
        mime = (declared_mime_type or mimetypes.guess_type(original_filename or path.name)[0] or "").lower()
        if mime and mime not in ALLOWED_MIME_TYPES:
            raise InvalidAudioError("Recording content type is not audio.")
        with path.open("rb") as handle:
            header = handle.read(16)
        signature_ok = (
            (extension == ".wav" and header[:4] in {b"RIFF", b"RF64"} and header[8:12] == b"WAVE")
            or (extension == ".mp3" and (header[:3] == b"ID3" or (len(header) > 1 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0)))
            or (extension == ".flac" and header[:4] == b"fLaC")
            or (extension in {".ogg", ".oga", ".opus"} and header[:4] == b"OggS")
            or (extension == ".m4a" and header[4:8] == b"ftyp")
            or (extension == ".aac" and len(header) > 1 and header[0] == 0xFF and header[1] & 0xF6 == 0xF0)
        )
        if not signature_ok:
            raise InvalidAudioError("Recording content does not match its file type.")
        stdout, _ = await self._run(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels,sample_rate,bit_rate:format=format_name,duration,size,bit_rate",
            "-of",
            "json",
            str(path),
            timeout=60,
        )
        try:
            payload: dict[str, Any] = json.loads(stdout)
            stream = payload["streams"][0]
            format_data = payload["format"]
            duration = float(format_data["duration"])
            channels = int(stream["channels"])
            sample_rate = int(stream["sample_rate"])
            codec = str(stream["codec_name"])
            format_name = str(format_data["format_name"])
            bit_rate_raw = stream.get("bit_rate") or format_data.get("bit_rate")
            bit_rate = int(bit_rate_raw) if bit_rate_raw else None
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InvalidAudioError("Recording does not contain decodable audio.") from exc
        if not (0 < duration <= 24 * 60 * 60):
            raise InvalidAudioError("Audio duration is outside the supported range.")
        if channels < 1 or channels > 32 or sample_rate < 4000:
            raise InvalidAudioError("Audio stream properties are invalid.")
        # ffprobe reads headers; a complete decode catches truncated or corrupt payloads.
        await self._run(
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
            timeout=min(900.0, max(60.0, duration * 2)),
        )
        return AudioInfo(
            codec_name=codec,
            format_name=format_name,
            duration_seconds=duration,
            channel_count=channels,
            sample_rate_hz=sample_rate,
            bit_rate_bps=bit_rate,
            size_bytes=size,
            sha256_checksum=self.sha256(path),
        )

    async def extract_channel(self, source: Path, destination: Path, channel: int) -> Path:
        if channel not in {0, 1}:
            raise InvalidAudioError("Audio channel selection is invalid.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".part.wav")
        temporary.unlink(missing_ok=True)
        try:
            await self._run(
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-filter:a",
                f"pan=mono|c0=c{channel}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary),
            )
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    async def convert_to_mono(self, source: Path, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".part.wav")
        temporary.unlink(missing_ok=True)
        try:
            await self._run(
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(temporary),
            )
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    async def split_audio(
        self,
        source: Path,
        destination_dir: Path,
        duration_seconds: float,
        *,
        chunk_seconds: float = 30.0,
    ) -> list[AudioChunk]:
        if duration_seconds <= 0 or chunk_seconds < 1:
            raise InvalidAudioError("Audio chunk parameters are invalid.")
        destination_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[AudioChunk] = []
        attempted: list[Path] = []
        start = 0.0
        sequence = 0
        try:
            while start < duration_seconds:
                end = min(duration_seconds, start + chunk_seconds)
                destination = destination_dir / f"chunk-{sequence:05d}-{secrets.token_hex(4)}.wav"
                attempted.append(destination)
                await self._run(
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{end - start:.3f}",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                str(destination),
                )
                if destination.stat().st_size > self.settings.max_transcription_upload_bytes:
                    destination.unlink(missing_ok=True)
                    raise InvalidAudioError("Audio chunk exceeds the transcription upload limit.")
                chunks.append(AudioChunk(destination, start, end))
                start = end
                sequence += 1
        except Exception:
            _, cleanup_failures = self.remove_files(attempted)
            if cleanup_failures:
                raise AudioToolError("Audio chunk creation failed and temporary cleanup is incomplete.")
            raise
        return chunks

    @staticmethod
    def remove_files(paths: list[Path]) -> tuple[list[Path], list[Path]]:
        removed: list[Path] = []
        failed: list[Path] = []
        for path in paths:
            try:
                path.unlink(missing_ok=True)
                if path.exists():
                    failed.append(path)
                else:
                    removed.append(path)
            except OSError:
                failed.append(path)
        return removed, failed
