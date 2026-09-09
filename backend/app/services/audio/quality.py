from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from app.services.audio import AudioInfo, AudioProcessor
from app.services.transcription.types import AudioTrack, SpeechChunk


RAW_LOSSLESS_AUDIO_VARIANT = "v2-raw-lossless-pcm16-v1"
LIGHT_NORMALIZED_AUDIO_VARIANT = "v2-light-normalized-telephone-v1"


@dataclass(frozen=True, slots=True)
class LightNormalizationProfile:
    version: str = "ffmpeg-light-normalized-v1"
    highpass_hz: int = 100
    lowpass_hz: int = 3400
    loudness_i: int = -23
    loudness_lra: int = 7
    true_peak_db: int = -2
    sample_rate_hz: int = 16_000
    channels: int = 1
    codec: str = "pcm_s16le"
    audio_variant: str = LIGHT_NORMALIZED_AUDIO_VARIANT

    @property
    def filter_chain(self) -> str:
        return (
            f"highpass=f={self.highpass_hz},"
            f"lowpass=f={self.lowpass_hz},"
            f"loudnorm=I={self.loudness_i}:"
            f"LRA={self.loudness_lra}:TP={self.true_peak_db}"
        )

    def identity(self) -> dict[str, object]:
        return {
            "audio_variant": self.audio_variant,
            "channels": self.channels,
            "codec": self.codec,
            "filter_chain": self.filter_chain,
            "sample_rate_hz": self.sample_rate_hz,
            "version": self.version,
        }


LIGHT_NORMALIZATION_PROFILE = LightNormalizationProfile()


class AudioQualityProcessor(Protocol):
    async def prepare(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
    ) -> AudioTrack: ...


class LegacyPassThroughQualityProcessor:
    """Keep the worker-prepared legacy WAV unchanged."""

    async def prepare(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
    ) -> AudioTrack:
        del audio_info
        return track


class RetryAudioQualityProcessor(Protocol):
    async def prepare_retry(
        self,
        chunk: SpeechChunk,
        *,
        destination_dir: Path,
        register_temporary_file: Callable[[Path], None] | None = None,
    ) -> SpeechChunk: ...

    def cleanup_retry(self, chunk: SpeechChunk) -> None: ...


class LightNormalizedRetryQualityProcessor:
    """Create the single versioned, lossless-output retry variant."""

    def __init__(
        self,
        processor: AudioProcessor,
        *,
        profile: LightNormalizationProfile = LIGHT_NORMALIZATION_PROFILE,
    ) -> None:
        self.processor = processor
        self.profile = profile

    async def prepare_retry(
        self,
        chunk: SpeechChunk,
        *,
        destination_dir: Path,
        register_temporary_file: Callable[[Path], None] | None = None,
    ) -> SpeechChunk:
        track_digest = hashlib.sha256(chunk.track_id.encode("utf-8")).hexdigest()[:12]
        destination = (
            destination_dir
            / f"retry-{track_digest}-{chunk.chunk_index}-{self.profile.version}.wav"
        )
        partial = self.processor.partial_wav_path(destination)
        # Register before FFmpeg so the worker owns even a partially written
        # destination if the audio side effect fails.
        if register_temporary_file is not None:
            register_temporary_file(destination)
            register_temporary_file(partial)
        try:
            await self.processor.apply_lossless_audio_filter(
                chunk.path,
                destination,
                audio_filter=self.profile.filter_chain,
            )
        except Exception:
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            raise
        return replace(
            chunk,
            path=destination,
            audio_variant=self.profile.audio_variant,
        )

    def cleanup_retry(self, chunk: SpeechChunk) -> None:
        for path in (self.processor.partial_wav_path(chunk.path), chunk.path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # The worker-wide registered-file cleanup remains the fallback.
                continue
