from __future__ import annotations

import hashlib
import math
import sys
import wave
from array import array
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import webrtcvad

from app.services.audio import AudioInfo, AudioProcessor
from app.services.audio.errors import (
    AudioSegmentationCancelledError,
    AudioToolError,
    InvalidAudioError,
)
from app.services.transcription.types import AudioTrack, SpeechChunk


WEBRTC_VAD_DISTRIBUTION = "webrtcvad-wheels"
WEBRTC_VAD_VERSION = "2.0.14"
LOCAL_SPEECH_SEGMENTATION_VERSION = "local-webrtc-speech-v1"


@dataclass(frozen=True, slots=True)
class SpeechSegmentationConfig:
    sample_rate_hz: int = 16_000
    channel_count: int = 1
    sample_width_bytes: int = 2
    frame_ms: int = 20
    vad_aggressiveness: int = 2
    start_window_ms: int = 300
    start_voiced_ratio: float = 0.60
    end_silence_ms: int = 600
    prefix_padding_ms: int = 250
    suffix_padding_ms: int = 250
    merge_gap_ms: int = 300
    minimum_segment_ms: int = 1_200
    target_segment_ms: int = 40_000
    maximum_segment_ms: int = 60_000
    hard_cut_overlap_ms: int = 800

    def __post_init__(self) -> None:
        if (
            self.sample_rate_hz != 16_000
            or self.channel_count != 1
            or self.sample_width_bytes != 2
            or self.frame_ms != 20
        ):
            raise ValueError("Speech segmentation requires 16 kHz mono PCM16 in 20 ms frames.")
        if self.vad_aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("WebRTC VAD aggressiveness must be between 0 and 3.")
        if not 0 < self.start_voiced_ratio <= 1:
            raise ValueError("The voiced start ratio must be greater than zero and at most one.")
        for name in ("start_window_ms", "end_silence_ms", "merge_gap_ms"):
            value = int(getattr(self, name))
            if value <= 0 or value % self.frame_ms:
                raise ValueError(f"{name} must be a positive multiple of frame_ms.")
        for name in ("prefix_padding_ms", "suffix_padding_ms", "hard_cut_overlap_ms"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative.")
        if not (0 < self.minimum_segment_ms <= self.target_segment_ms <= self.maximum_segment_ms):
            raise ValueError("Segment durations must satisfy minimum <= target <= maximum.")
        if self.maximum_segment_ms - self.target_segment_ms < self.minimum_segment_ms:
            raise ValueError("A target split must leave room for a minimum trailing segment.")
        if self.hard_cut_overlap_ms >= self.minimum_segment_ms:
            raise ValueError("Hard-cut overlap must be smaller than the minimum segment.")
        if self.target_segment_ms + self.hard_cut_overlap_ms > self.maximum_segment_ms:
            raise ValueError("Hard-cut overlap must keep the next target chunk within the maximum.")

    @property
    def frame_samples(self) -> int:
        return self.sample_rate_hz * self.frame_ms // 1_000

    def samples_for_ms(self, value: int) -> int:
        return self.sample_rate_hz * value // 1_000

    def identity(self) -> dict[str, int | float]:
        return {
            "channel_count": self.channel_count,
            "end_silence_ms": self.end_silence_ms,
            "frame_ms": self.frame_ms,
            "hard_cut_overlap_ms": self.hard_cut_overlap_ms,
            "maximum_segment_ms": self.maximum_segment_ms,
            "merge_gap_ms": self.merge_gap_ms,
            "minimum_segment_ms": self.minimum_segment_ms,
            "prefix_padding_ms": self.prefix_padding_ms,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_width_bytes": self.sample_width_bytes,
            "start_voiced_ratio": self.start_voiced_ratio,
            "start_window_ms": self.start_window_ms,
            "suffix_padding_ms": self.suffix_padding_ms,
            "target_segment_ms": self.target_segment_ms,
            "vad_aggressiveness": self.vad_aggressiveness,
        }


DEFAULT_SPEECH_SEGMENTATION_CONFIG = SpeechSegmentationConfig()


def local_speech_segmentation_identity(
    config: SpeechSegmentationConfig = DEFAULT_SPEECH_SEGMENTATION_CONFIG,
) -> dict[str, object]:
    return {
        "algorithm_version": LOCAL_SPEECH_SEGMENTATION_VERSION,
        "analysis_format": {
            "channel_count": config.channel_count,
            "frame_ms": config.frame_ms,
            "frame_samples": config.frame_samples,
            "sample_format": "signed-16-bit-pcm",
            "sample_rate_hz": config.sample_rate_hz,
        },
        "backend": {
            "distribution": WEBRTC_VAD_DISTRIBUTION,
            "version": WEBRTC_VAD_VERSION,
        },
        "configuration": config.identity(),
        "output_format": "pcm-s16le-wav",
        "strategy": "local-speech-boundaries",
    }


class FrameVoiceClassifier(Protocol):
    def is_speech(self, frame: bytes, sample_rate_hz: int) -> bool: ...


class WebRtcVadFrameClassifier:
    def __init__(self, aggressiveness: int) -> None:
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: bytes, sample_rate_hz: int) -> bool:
        return bool(self._vad.is_speech(frame, sample_rate_hz))


class AudioSegmenter(Protocol):
    async def segment(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
        destination_dir: Path,
        max_upload_bytes: int,
        register_temporary_file: Callable[[Path], None] | None = None,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[SpeechChunk, ...]: ...


class LegacyFixedAudioSegmenter:
    """Adapter for the frozen one-upload/15-second/480-second policy."""

    def __init__(self, processor: AudioProcessor) -> None:
        self.processor = processor

    async def segment(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
        destination_dir: Path,
        max_upload_bytes: int,
        register_temporary_file: Callable[[Path], None] | None = None,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[SpeechChunk, ...]:
        del cancellation_check
        if track.diarized and track.source_path.stat().st_size <= max_upload_bytes:
            return (
                SpeechChunk(
                    track_id=track.track_id,
                    chunk_index=0,
                    path=track.source_path,
                    start_seconds=0,
                    end_seconds=audio_info.duration_seconds,
                    audio_variant=track.audio_variant,
                ),
            )

        chunk_seconds = 480 if track.diarized else 15
        legacy_chunks = await self.processor.split_audio(
            track.source_path,
            destination_dir,
            audio_info.duration_seconds,
            chunk_seconds=chunk_seconds,
        )
        chunks: list[SpeechChunk] = []
        for chunk_index, chunk in enumerate(legacy_chunks):
            if register_temporary_file is not None:
                register_temporary_file(chunk.path)
            chunks.append(
                SpeechChunk(
                    track_id=track.track_id,
                    chunk_index=chunk_index,
                    path=chunk.path,
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    hard_cut=chunk.end_seconds < audio_info.duration_seconds,
                    audio_variant=track.audio_variant,
                )
            )
        return tuple(chunks)


@dataclass(frozen=True, slots=True)
class _Region:
    start_sample: int
    end_sample: int


@dataclass(frozen=True, slots=True)
class _PlannedChunk:
    start_sample: int
    end_sample: int
    hard_cut: bool
    overlap_before_ms: int


class LocalSpeechAudioSegmenter:
    """Deterministic local WebRTC VAD with exact integer-sample boundaries."""

    def __init__(
        self,
        processor: AudioProcessor,
        *,
        config: SpeechSegmentationConfig = DEFAULT_SPEECH_SEGMENTATION_CONFIG,
        classifier_factory: Callable[[], FrameVoiceClassifier] | None = None,
    ) -> None:
        self.processor = processor
        self.config = config
        self.classifier_factory = classifier_factory or (
            lambda: WebRtcVadFrameClassifier(self.config.vad_aggressiveness)
        )

    def identity(self) -> dict[str, object]:
        return local_speech_segmentation_identity(self.config)

    async def segment(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
        destination_dir: Path,
        max_upload_bytes: int,
        register_temporary_file: Callable[[Path], None] | None = None,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[SpeechChunk, ...]:
        del audio_info
        if max_upload_bytes <= 0:
            raise InvalidAudioError("Audio upload limit is invalid.")
        await self._check_cancellation(cancellation_check)
        decisions, energies, duration_samples = await self._analyze(
            track.source_path,
            cancellation_check=cancellation_check,
        )
        planned = await self._plan_chunks(
            decisions,
            energies,
            duration_samples,
            cancellation_check=cancellation_check,
        )
        if not planned:
            return ()

        destination_dir.mkdir(parents=True, exist_ok=True)
        track_token = hashlib.sha256(track.track_id.encode("utf-8")).hexdigest()[:12]
        attempted: list[Path] = []
        chunks: list[SpeechChunk] = []
        try:
            for chunk_index, boundary in enumerate(planned):
                await self._check_cancellation(cancellation_check)
                destination = destination_dir / f"speech-{track_token}-{chunk_index:05d}.wav"
                partial = self.processor.partial_wav_path(destination)
                attempted.extend((destination, partial))
                if register_temporary_file is not None:
                    register_temporary_file(destination)
                    register_temporary_file(partial)
                await self.processor.extract_pcm_wav_range(
                    track.source_path,
                    destination,
                    start_sample=boundary.start_sample,
                    end_sample=boundary.end_sample,
                    sample_rate_hz=self.config.sample_rate_hz,
                )
                await self._check_cancellation(cancellation_check)
                if destination.stat().st_size > max_upload_bytes:
                    raise InvalidAudioError("Audio chunk exceeds the transcription upload limit.")
                chunks.append(
                    SpeechChunk(
                        track_id=track.track_id,
                        chunk_index=chunk_index,
                        path=destination,
                        start_seconds=(boundary.start_sample / self.config.sample_rate_hz),
                        end_seconds=boundary.end_sample / self.config.sample_rate_hz,
                        hard_cut=boundary.hard_cut,
                        overlap_before_ms=boundary.overlap_before_ms,
                        audio_variant=track.audio_variant,
                    )
                )
        except Exception as exc:
            _, cleanup_failures = self.processor.remove_files(attempted)
            if cleanup_failures:
                raise AudioToolError(
                    "Speech chunk creation failed and temporary cleanup is incomplete."
                ) from exc
            raise
        return tuple(chunks)

    async def _analyze(
        self,
        source: Path,
        *,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> tuple[bytearray, array[int], int]:
        if source.is_symlink() or not source.is_file():
            raise InvalidAudioError("Audio file location is invalid.")
        classifier = self.classifier_factory()
        decisions = bytearray()
        energies = array("I")
        try:
            with wave.open(str(source), "rb") as reader:
                if (
                    reader.getnchannels() != self.config.channel_count
                    or reader.getsampwidth() != self.config.sample_width_bytes
                    or reader.getframerate() != self.config.sample_rate_hz
                    or reader.getcomptype() != "NONE"
                ):
                    raise InvalidAudioError(
                        "Speech segmentation requires 16 kHz mono PCM16 WAV audio."
                    )
                duration_samples = reader.getnframes()
                if duration_samples <= 0:
                    raise InvalidAudioError("Audio file contains no PCM samples.")
                frame_index = 0
                consumed_samples = 0
                cancellation_interval = max(
                    1,
                    self.config.target_segment_ms // self.config.frame_ms,
                )
                while consumed_samples < duration_samples:
                    if frame_index and frame_index % cancellation_interval == 0:
                        await self._check_cancellation(cancellation_check)
                    expected_samples = min(
                        self.config.frame_samples,
                        duration_samples - consumed_samples,
                    )
                    frame = reader.readframes(expected_samples)
                    if len(frame) != expected_samples * self.config.sample_width_bytes:
                        raise InvalidAudioError("Audio frames could not be read completely.")
                    energy = self._mean_square_energy(frame)
                    vad_frame = frame.ljust(
                        self.config.frame_samples * self.config.sample_width_bytes,
                        b"\x00",
                    )
                    try:
                        voiced = classifier.is_speech(
                            vad_frame,
                            self.config.sample_rate_hz,
                        )
                    except Exception as exc:
                        raise InvalidAudioError(
                            "Audio frames could not be classified safely."
                        ) from exc
                    decisions.append(1 if voiced else 0)
                    energies.append(energy)
                    consumed_samples += expected_samples
                    frame_index += 1
        except AudioSegmentationCancelledError:
            raise
        except InvalidAudioError:
            raise
        except (EOFError, OSError, wave.Error) as exc:
            raise InvalidAudioError("Audio file could not be decoded.") from exc
        await self._check_cancellation(cancellation_check)
        return decisions, energies, duration_samples

    @staticmethod
    def _mean_square_energy(frame: bytes) -> int:
        samples = array("h")
        samples.frombytes(frame)
        if sys.byteorder != "little":
            samples.byteswap()
        if not samples:
            return 0
        return sum(sample * sample for sample in samples) // len(samples)

    @staticmethod
    async def _check_cancellation(
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> None:
        if cancellation_check is not None and await cancellation_check():
            raise AudioSegmentationCancelledError("Audio segmentation was cancelled.")

    async def _plan_chunks(
        self,
        decisions: Sequence[int],
        energies: Sequence[int],
        duration_samples: int,
        *,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> tuple[_PlannedChunk, ...]:
        if len(decisions) != len(energies):
            raise ValueError("Frame decisions and energies must have equal lengths.")
        await self._check_cancellation(cancellation_check)
        regions = self._detect_regions(decisions, duration_samples)
        if not regions:
            return ()
        regions = self._apply_padding(regions, duration_samples)
        regions = self._merge_close_regions(regions)
        regions = self._merge_tiny_regions(regions)
        await self._check_cancellation(cancellation_check)
        chunks: list[_PlannedChunk] = []
        for region in regions:
            chunks.extend(
                await self._split_region(
                    region,
                    decisions,
                    energies,
                    cancellation_check=cancellation_check,
                )
            )
        return tuple(chunks)

    def _detect_regions(
        self,
        decisions: Sequence[int],
        duration_samples: int,
    ) -> list[_Region]:
        window_frames = self.config.start_window_ms // self.config.frame_ms
        required_voiced = math.ceil(window_frames * self.config.start_voiced_ratio)
        end_silence_frames = self.config.end_silence_ms // self.config.frame_ms
        window: deque[tuple[int, bool]] = deque(maxlen=window_frames)
        regions: list[_Region] = []
        active = False
        region_start = 0
        last_voiced_end = 0
        silence_frames = 0

        for frame_index, decision in enumerate(decisions):
            voiced = bool(decision)
            if not active:
                window.append((frame_index, voiced))
                if (
                    len(window) == window_frames
                    and sum(1 for _, item_voiced in window if item_voiced) >= required_voiced
                ):
                    voiced_frames = [index for index, item_voiced in window if item_voiced]
                    region_start = voiced_frames[0] * self.config.frame_samples
                    last_voiced_end = min(
                        duration_samples,
                        (voiced_frames[-1] + 1) * self.config.frame_samples,
                    )
                    silence_frames = frame_index - voiced_frames[-1]
                    active = True
                    window.clear()
                continue

            if voiced:
                last_voiced_end = min(
                    duration_samples,
                    (frame_index + 1) * self.config.frame_samples,
                )
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= end_silence_frames:
                    if last_voiced_end > region_start:
                        regions.append(_Region(region_start, last_voiced_end))
                    active = False
                    window.clear()
                    silence_frames = 0

        if active and last_voiced_end > region_start:
            regions.append(_Region(region_start, last_voiced_end))
        return regions

    def _apply_padding(
        self,
        regions: Sequence[_Region],
        duration_samples: int,
    ) -> list[_Region]:
        prefix = self.config.samples_for_ms(self.config.prefix_padding_ms)
        suffix = self.config.samples_for_ms(self.config.suffix_padding_ms)
        return [
            _Region(
                max(0, region.start_sample - prefix),
                min(duration_samples, region.end_sample + suffix),
            )
            for region in regions
        ]

    def _merge_close_regions(self, regions: Sequence[_Region]) -> list[_Region]:
        merge_gap = self.config.samples_for_ms(self.config.merge_gap_ms)
        merged: list[_Region] = []
        for region in regions:
            if merged and region.start_sample - merged[-1].end_sample < merge_gap:
                previous = merged[-1]
                merged[-1] = _Region(
                    previous.start_sample,
                    max(previous.end_sample, region.end_sample),
                )
            else:
                merged.append(region)
        return merged

    def _merge_tiny_regions(self, regions: Sequence[_Region]) -> list[_Region]:
        if len(regions) < 2:
            return list(regions)

        minimum = self.config.samples_for_ms(self.config.minimum_segment_ms)
        safe_gap = self.config.samples_for_ms(self.config.end_silence_ms)
        maximum = self.config.samples_for_ms(self.config.maximum_segment_ms)
        starts = [region.start_sample for region in regions]
        ends = [region.end_sample for region in regions]
        previous = [index - 1 for index in range(len(regions))]
        following = [index + 1 if index + 1 < len(regions) else -1 for index in range(len(regions))]
        cursor = 0
        while cursor != -1:
            if ends[cursor] - starts[cursor] >= minimum:
                cursor = following[cursor]
                continue

            candidates: list[tuple[int, int, int]] = []
            previous_index = previous[cursor]
            if previous_index != -1:
                gap = max(0, starts[cursor] - ends[previous_index])
                if gap <= safe_gap and ends[cursor] - starts[previous_index] <= maximum:
                    candidates.append((gap, 0, previous_index))
            following_index = following[cursor]
            if following_index != -1:
                gap = max(0, starts[following_index] - ends[cursor])
                if gap <= safe_gap and ends[following_index] - starts[cursor] <= maximum:
                    candidates.append((gap, 1, following_index))
            if not candidates:
                cursor = following[cursor]
                continue

            _, _, neighbor_index = min(candidates)
            kept_index = min(cursor, neighbor_index)
            removed_index = max(cursor, neighbor_index)
            starts[kept_index] = min(starts[cursor], starts[neighbor_index])
            ends[kept_index] = max(ends[cursor], ends[neighbor_index])
            after_removed = following[removed_index]
            following[kept_index] = after_removed
            if after_removed != -1:
                previous[after_removed] = kept_index
            previous[removed_index] = -1
            following[removed_index] = -1

            # Only the merged node and its live predecessor can have different
            # merge eligibility. Resuming there preserves the prior
            # left-to-right/restart semantics without rescanning the prefix.
            cursor = previous[kept_index] if previous[kept_index] != -1 else kept_index

        merged: list[_Region] = []
        cursor = 0
        while cursor != -1:
            merged.append(_Region(starts[cursor], ends[cursor]))
            cursor = following[cursor]
        return merged

    async def _split_region(
        self,
        region: _Region,
        decisions: Sequence[int],
        energies: Sequence[int],
        *,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> list[_PlannedChunk]:
        maximum = self.config.samples_for_ms(self.config.maximum_segment_ms)
        target = self.config.samples_for_ms(self.config.target_segment_ms)
        overlap = self.config.samples_for_ms(self.config.hard_cut_overlap_ms)
        logical_start = region.start_sample
        overlap_before_samples = 0
        overlap_before_ms = 0
        chunks: list[_PlannedChunk] = []

        while region.end_sample - (logical_start - overlap_before_samples) > maximum:
            await self._check_cancellation(cancellation_check)
            safe_cut = self._safe_cut(
                logical_start=logical_start,
                region_end=region.end_sample,
                overlap_before_samples=overlap_before_samples,
                decisions=decisions,
                energies=energies,
            )
            hard_cut = safe_cut is None
            cut = safe_cut if safe_cut is not None else logical_start + target
            chunk_start = max(region.start_sample, logical_start - overlap_before_samples)
            chunks.append(
                _PlannedChunk(
                    start_sample=chunk_start,
                    end_sample=cut,
                    hard_cut=hard_cut,
                    overlap_before_ms=overlap_before_ms,
                )
            )
            logical_start = cut
            overlap_before_samples = overlap if hard_cut else 0
            overlap_before_ms = self.config.hard_cut_overlap_ms if hard_cut else 0

        await self._check_cancellation(cancellation_check)
        chunks.append(
            _PlannedChunk(
                start_sample=max(
                    region.start_sample,
                    logical_start - overlap_before_samples,
                ),
                end_sample=region.end_sample,
                hard_cut=False,
                overlap_before_ms=overlap_before_ms,
            )
        )
        return chunks

    def _safe_cut(
        self,
        *,
        logical_start: int,
        region_end: int,
        overlap_before_samples: int,
        decisions: Sequence[int],
        energies: Sequence[int],
    ) -> int | None:
        frame_samples = self.config.frame_samples
        minimum = self.config.samples_for_ms(self.config.minimum_segment_ms)
        target_length = self.config.samples_for_ms(self.config.target_segment_ms)
        maximum = self.config.samples_for_ms(self.config.maximum_segment_ms)
        flexibility = maximum - target_length
        earliest = logical_start + max(minimum, target_length - flexibility)
        latest = min(
            logical_start + maximum - overlap_before_samples,
            region_end - minimum,
        )
        if earliest > latest:
            return None
        desired = logical_start + target_length
        minimum_run_frames = self.config.merge_gap_ms // self.config.frame_ms
        candidate_first_frame = earliest // frame_samples
        candidate_last_frame = latest // frame_samples
        first_frame = max(0, candidate_first_frame - minimum_run_frames)
        last_frame = min(
            len(decisions),
            candidate_last_frame + minimum_run_frames + 1,
        )
        candidates: list[tuple[int, int, int]] = []
        frame_index = first_frame
        while frame_index < last_frame:
            if decisions[frame_index]:
                frame_index += 1
                continue
            run_start = frame_index
            while frame_index < last_frame and not decisions[frame_index]:
                frame_index += 1
            run_end = frame_index
            if run_end - run_start < minimum_run_frames:
                continue
            low = max(earliest, run_start * frame_samples)
            high = min(latest, run_end * frame_samples)
            if low > high:
                continue

            # Choose the lowest-energy actual non-speech frame inside this
            # valid run. Ties prefer the point nearest the target, then the
            # earlier absolute sample. A run near the target still wins over
            # a distant run before energy is considered across runs.
            first_energy_frame = max(
                run_start,
                math.ceil(low / frame_samples) - 1,
            )
            last_energy_frame = min(run_end - 1, high // frame_samples)
            run_candidates: list[tuple[int, int, int]] = []
            for energy_index in range(first_energy_frame, last_energy_frame + 1):
                candidate = min(
                    high,
                    max(low, energy_index * frame_samples),
                )
                run_candidates.append(
                    (
                        int(energies[energy_index]),
                        abs(candidate - desired),
                        candidate,
                    )
                )
            if not run_candidates:
                continue
            energy, distance, candidate = min(run_candidates)
            candidates.append(
                (
                    distance,
                    energy,
                    candidate,
                )
            )
        if not candidates:
            return None
        return min(candidates)[2]
