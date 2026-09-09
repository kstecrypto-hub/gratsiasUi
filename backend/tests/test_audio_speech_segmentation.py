from __future__ import annotations

import hashlib
import struct
import wave
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.services.audio import AudioInfo, AudioProcessor
from app.services.audio.errors import (
    AudioSegmentationCancelledError,
    AudioToolError,
    InvalidAudioError,
)
from app.services.audio.segmentation import (
    LocalSpeechAudioSegmenter,
    SpeechSegmentationConfig,
)
from app.services.transcription.types import AudioTrack, SpeechChunk


FRAME_MS = 20
SAMPLE_RATE_HZ = 16_000
SAMPLES_PER_FRAME = SAMPLE_RATE_HZ * FRAME_MS // 1000
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2


def _settings(storage_root: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=storage_root,
        OPENAI_API_KEY="test-only-key",
    )


def _config(**changes: Any) -> SpeechSegmentationConfig:
    """A scaled configuration that keeps boundary tests fast."""

    base = SpeechSegmentationConfig(
        frame_ms=FRAME_MS,
        vad_aggressiveness=2,
        start_window_ms=60,
        start_voiced_ratio=0.60,
        end_silence_ms=40,
        prefix_padding_ms=40,
        suffix_padding_ms=40,
        merge_gap_ms=80,
        minimum_segment_ms=100,
        target_segment_ms=400,
        maximum_segment_ms=600,
        hard_cut_overlap_ms=80,
    )
    return replace(base, **changes)


def _write_wav(
    path: Path,
    frame_amplitudes: list[int],
    *,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    channel_count: int = 1,
    sample_width_bytes: int = 2,
) -> Path:
    """Write a real frame-aligned PCM WAV without relying on FFmpeg."""

    path.parent.mkdir(parents=True, exist_ok=True)
    samples_per_frame = sample_rate_hz * FRAME_MS // 1000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channel_count)
        output.setsampwidth(sample_width_bytes)
        output.setframerate(sample_rate_hz)
        for amplitude in frame_amplitudes:
            if sample_width_bytes == 2:
                sample = max(-32_768, min(32_767, amplitude))
                frame = struct.pack(
                    f"<{samples_per_frame * channel_count}h",
                    *([sample] * samples_per_frame * channel_count),
                )
            else:
                sample = max(0, min(255, 128 + amplitude))
                frame = bytes([sample]) * samples_per_frame * channel_count
            output.writeframesraw(frame)
    return path


def _read_pcm16_wav(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as source:
        return (
            source.getframerate(),
            source.getnchannels(),
            source.getnframes(),
            source.readframes(source.getnframes()),
        )


def _audio_info(path: Path, *, duration_seconds: float, channel_count: int = 1) -> AudioInfo:
    return AudioInfo(
        codec_name="pcm_s16le",
        format_name="wav",
        duration_seconds=duration_seconds,
        channel_count=channel_count,
        sample_rate_hz=SAMPLE_RATE_HZ,
        bit_rate_bps=256_000,
        size_bytes=path.stat().st_size,
        sha256_checksum=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


class _DecisionClassifier:
    def __init__(self, decisions: tuple[bool, ...]) -> None:
        self.decisions = decisions
        self.frames_seen = 0

    def is_speech(self, frame: bytes, sample_rate: int) -> bool:
        assert sample_rate == SAMPLE_RATE_HZ
        assert len(frame) == PCM16_BYTES_PER_FRAME
        if self.frames_seen >= len(self.decisions):
            raise AssertionError("The segmenter classified more frames than the WAV contains.")
        decision = self.decisions[self.frames_seen]
        self.frames_seen += 1
        return decision


class _ClassifierFactory:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = tuple(decisions)
        self.instances: list[_DecisionClassifier] = []

    def __call__(self, *_args: object, **_kwargs: object) -> _DecisionClassifier:
        classifier = _DecisionClassifier(self.decisions)
        self.instances.append(classifier)
        return classifier


def _segmenter(
    tmp_path: Path,
    decisions: list[bool],
    *,
    config: SpeechSegmentationConfig | None = None,
    processor: AudioProcessor | None = None,
) -> tuple[LocalSpeechAudioSegmenter, _ClassifierFactory]:
    factory = _ClassifierFactory(decisions)
    return (
        LocalSpeechAudioSegmenter(
            processor or AudioProcessor(_settings(tmp_path)),
            config=config or _config(),
            classifier_factory=factory,
        ),
        factory,
    )


async def _segment(
    tmp_path: Path,
    decisions: list[bool],
    *,
    amplitudes: list[int] | None = None,
    config: SpeechSegmentationConfig | None = None,
    destination_name: str = "chunks",
    processor: AudioProcessor | None = None,
    cancellation_check: Any = None,
    registered: list[Path] | None = None,
    track_id: str = "track-0",
) -> tuple[tuple[SpeechChunk, ...], _ClassifierFactory, Path]:
    amplitudes = amplitudes or [6000 if item else 0 for item in decisions]
    source = _write_wav(tmp_path / f"{track_id}.wav", amplitudes)
    segmenter, factory = _segmenter(
        tmp_path,
        decisions,
        config=config,
        processor=processor,
    )
    duration_seconds = len(decisions) * FRAME_MS / 1000
    chunks = await segmenter.segment(
        AudioTrack(
            track_id=track_id,
            source_path=source,
            duration_seconds=duration_seconds,
            audio_variant="test-pcm16",
        ),
        audio_info=_audio_info(source, duration_seconds=duration_seconds),
        destination_dir=tmp_path / destination_name,
        max_upload_bytes=24 * 1024 * 1024,
        register_temporary_file=(registered.append if registered is not None else None),
        cancellation_check=cancellation_check,
    )
    return chunks, factory, source


def _signature(chunks: tuple[SpeechChunk, ...]) -> list[tuple[int, float, float, bool, int]]:
    return [
        (
            chunk.chunk_index,
            chunk.start_seconds,
            chunk.end_seconds,
            chunk.hard_cut,
            chunk.overlap_before_ms,
        )
        for chunk in chunks
    ]


def test_speech_segmentation_defaults_are_typed_and_complete() -> None:
    config = SpeechSegmentationConfig()

    assert config.frame_ms == 20
    assert config.vad_aggressiveness == 2
    assert config.start_window_ms == 300
    assert config.start_voiced_ratio == 0.60
    assert config.end_silence_ms == 600
    assert config.prefix_padding_ms == 250
    assert config.suffix_padding_ms == 250
    assert config.merge_gap_ms == 300
    assert config.minimum_segment_ms == 1200
    assert config.target_segment_ms == 40_000
    assert config.maximum_segment_ms == 60_000
    assert config.hard_cut_overlap_ms == 800


@pytest.mark.asyncio
async def test_silence_only_audio_produces_zero_chunks(tmp_path: Path) -> None:
    decisions = [False] * 20

    chunks, factory, _ = await _segment(tmp_path, decisions, registered=[])

    assert chunks == ()
    assert len(factory.instances) == 1
    assert factory.instances[0].frames_seen == len(decisions)
    assert list((tmp_path / "chunks").glob("*.wav")) == []


@pytest.mark.asyncio
async def test_each_track_uses_a_fresh_classifier_instance(tmp_path: Path) -> None:
    decisions = [False] * 3 + [True] * 5 + [False] * 3
    amplitudes = [6000 if item else 0 for item in decisions]
    first_source = _write_wav(tmp_path / "channel-0.wav", amplitudes)
    second_source = _write_wav(tmp_path / "channel-1.wav", amplitudes)
    segmenter, factory = _segmenter(tmp_path, decisions)
    duration_seconds = len(decisions) * FRAME_MS / 1000

    for channel_index, source in enumerate((first_source, second_source)):
        chunks = await segmenter.segment(
            AudioTrack(
                track_id=f"channel-{channel_index}",
                source_path=source,
                duration_seconds=duration_seconds,
                channel_index=channel_index,
            ),
            audio_info=_audio_info(source, duration_seconds=duration_seconds),
            destination_dir=tmp_path / f"chunks-{channel_index}",
            max_upload_bytes=24 * 1024 * 1024,
        )
        assert len(chunks) == 1

    assert len(factory.instances) == 2
    assert factory.instances[0] is not factory.instances[1]
    assert [instance.frames_seen for instance in factory.instances] == [
        len(decisions),
        len(decisions),
    ]


@pytest.mark.asyncio
async def test_one_speech_region_produces_one_exact_padded_pcm_chunk(tmp_path: Path) -> None:
    decisions = [False] * 5 + [True] * 10 + [False] * 5
    registered: list[Path] = []

    chunks, _, source = await _segment(tmp_path, decisions, registered=registered)

    assert _signature(chunks) == [(0, 0.06, 0.34, False, 0)]
    assert registered == [
        chunks[0].path,
        AudioProcessor.partial_wav_path(chunks[0].path),
    ]
    assert not registered[1].exists()
    sample_rate, channels, sample_count, chunk_pcm = _read_pcm16_wav(chunks[0].path)
    _, _, _, source_pcm = _read_pcm16_wav(source)
    assert (sample_rate, channels) == (SAMPLE_RATE_HZ, 1)
    assert sample_count == round((0.34 - 0.06) * SAMPLE_RATE_HZ)
    assert (
        chunk_pcm == source_pcm[round(0.06 * SAMPLE_RATE_HZ) * 2 : round(0.34 * SAMPLE_RATE_HZ) * 2]
    )


@pytest.mark.asyncio
async def test_short_gap_regions_merge_after_padding(tmp_path: Path) -> None:
    decisions = [False] * 3 + [True] * 5 + [False] * 6 + [True] * 5 + [False] * 3

    chunks, _, _ = await _segment(tmp_path, decisions)

    assert len(chunks) == 1
    assert chunks[0].start_seconds == pytest.approx(0.02)
    assert chunks[0].end_seconds == pytest.approx(0.42)


@pytest.mark.asyncio
async def test_long_gap_regions_remain_separate(tmp_path: Path) -> None:
    decisions = [False] * 3 + [True] * 5 + [False] * 12 + [True] * 5 + [False] * 3

    chunks, _, _ = await _segment(tmp_path, decisions)

    assert len(chunks) == 2
    assert chunks[0].end_seconds < chunks[1].start_seconds


@pytest.mark.asyncio
async def test_prefix_padding_is_exact_in_the_middle_and_bounded_at_zero(
    tmp_path: Path,
) -> None:
    at_start = [True] * 5 + [False] * 3
    in_middle = [False] * 5 + [True] * 5 + [False] * 3

    start_chunks, _, _ = await _segment(
        tmp_path,
        at_start,
        destination_name="at-start",
        track_id="start",
    )
    middle_chunks, _, _ = await _segment(
        tmp_path,
        in_middle,
        destination_name="in-middle",
        track_id="middle",
    )

    assert start_chunks[0].start_seconds == 0
    assert middle_chunks[0].start_seconds == pytest.approx(0.06)


@pytest.mark.asyncio
async def test_suffix_padding_is_exact_in_the_middle_and_bounded_at_duration(
    tmp_path: Path,
) -> None:
    in_middle = [False] * 3 + [True] * 5 + [False] * 5
    at_end = [False] * 3 + [True] * 5

    middle_chunks, _, _ = await _segment(
        tmp_path,
        in_middle,
        destination_name="in-middle",
        track_id="middle",
    )
    end_chunks, _, _ = await _segment(
        tmp_path,
        at_end,
        destination_name="at-end",
        track_id="end",
    )

    assert middle_chunks[0].end_seconds == pytest.approx(0.20)
    assert end_chunks[0].end_seconds == pytest.approx(len(at_end) * FRAME_MS / 1000)


@pytest.mark.asyncio
async def test_tiny_region_merges_with_a_safe_neighbor(tmp_path: Path) -> None:
    config = _config(
        merge_gap_ms=20,
        minimum_segment_ms=200,
    )
    decisions = [False] * 3 + [True] * 3 + [False] * 5 + [True] * 10 + [False] * 3

    chunks, _, _ = await _segment(tmp_path, decisions, config=config)

    assert len(chunks) == 1
    assert chunks[0].start_seconds < 0.1
    assert chunks[0].end_seconds > 0.35


@pytest.mark.asyncio
async def test_long_speech_prefers_a_safe_low_energy_split(tmp_path: Path) -> None:
    config = _config(
        end_silence_ms=200,
        prefix_padding_ms=0,
        suffix_padding_ms=0,
        merge_gap_ms=40,
    )
    decisions = [True] * 50
    decisions[19:22] = [False, False, False]
    amplitudes = [7000] * len(decisions)
    amplitudes[19:22] = [50, 0, 50]

    chunks, _, _ = await _segment(
        tmp_path,
        decisions,
        amplitudes=amplitudes,
        config=config,
    )

    assert len(chunks) == 2
    assert 0.38 <= chunks[0].end_seconds <= 0.44
    assert chunks[0].hard_cut is False
    assert chunks[1].start_seconds == chunks[0].end_seconds
    assert chunks[1].overlap_before_ms == 0


@pytest.mark.asyncio
async def test_safe_split_chooses_the_lowest_energy_frame_inside_the_nearest_run(
    tmp_path: Path,
) -> None:
    config = _config(
        end_silence_ms=200,
        prefix_padding_ms=0,
        suffix_padding_ms=0,
        merge_gap_ms=40,
    )
    decisions = [True] * 50
    decisions[18:23] = [False] * 5
    amplitudes = [7000] * len(decisions)
    amplitudes[18:23] = [0, 3000, 6000, 3000, 1000]

    chunks, _, _ = await _segment(
        tmp_path,
        decisions,
        amplitudes=amplitudes,
        config=config,
    )

    assert chunks[0].end_seconds == pytest.approx(0.36)
    assert chunks[0].hard_cut is False


@pytest.mark.asyncio
async def test_safe_split_prefers_a_near_pause_over_a_distant_lower_energy_pause(
    tmp_path: Path,
) -> None:
    config = _config(
        end_silence_ms=200,
        prefix_padding_ms=0,
        suffix_padding_ms=0,
        merge_gap_ms=40,
    )
    decisions = [True] * 50
    decisions[12:15] = [False] * 3
    decisions[19:22] = [False] * 3
    amplitudes = [7000] * len(decisions)
    amplitudes[12:15] = [100, 0, 100]
    amplitudes[19:22] = [500, 500, 500]

    chunks, _, _ = await _segment(
        tmp_path,
        decisions,
        amplitudes=amplitudes,
        config=config,
    )

    assert chunks[0].end_seconds == pytest.approx(0.40)
    assert chunks[0].hard_cut is False


@pytest.mark.asyncio
async def test_equal_safe_cut_candidates_choose_the_earlier_absolute_sample(
    tmp_path: Path,
) -> None:
    config = _config(
        end_silence_ms=200,
        prefix_padding_ms=0,
        suffix_padding_ms=0,
        merge_gap_ms=40,
    )
    decisions = [True] * 50
    decisions[16:18] = [False] * 2
    decisions[23:25] = [False] * 2
    amplitudes = [7000] * len(decisions)
    amplitudes[16:18] = [0, 0]
    amplitudes[23:25] = [0, 0]

    chunks, _, _ = await _segment(
        tmp_path,
        decisions,
        amplitudes=amplitudes,
        config=config,
    )

    assert chunks[0].end_seconds == pytest.approx(0.34)
    assert chunks[0].hard_cut is False


@pytest.mark.asyncio
async def test_no_safe_gap_uses_a_hard_cut_near_target(tmp_path: Path) -> None:
    config = _config(prefix_padding_ms=0, suffix_padding_ms=0)
    decisions = [True] * 50

    chunks, _, _ = await _segment(tmp_path, decisions, config=config)

    assert len(chunks) >= 2
    assert chunks[0].end_seconds == pytest.approx(0.4)
    assert chunks[0].hard_cut is True


@pytest.mark.asyncio
async def test_hard_cut_applies_only_the_configured_overlap_to_following_chunk(
    tmp_path: Path,
) -> None:
    config = _config(prefix_padding_ms=0, suffix_padding_ms=0)
    decisions = [True] * 70

    chunks, _, _ = await _segment(tmp_path, decisions, config=config)

    assert chunks[0].overlap_before_ms == 0
    overlapping = [chunk for chunk in chunks if chunk.overlap_before_ms]
    assert overlapping
    assert {chunk.overlap_before_ms for chunk in overlapping} == {80}
    for previous, current in zip(chunks, chunks[1:]):
        if current.overlap_before_ms:
            assert previous.hard_cut is True
            assert current.start_seconds == pytest.approx(previous.end_seconds - 0.08)


@pytest.mark.asyncio
async def test_chunk_timestamps_are_absolute_and_clamped_to_track_duration(
    tmp_path: Path,
) -> None:
    decisions = [False] * 15 + [True] * 10 + [False] * 5

    chunks, _, _ = await _segment(tmp_path, decisions)

    assert _signature(chunks) == [(0, 0.26, 0.54, False, 0)]
    assert 0 <= chunks[0].start_seconds < chunks[0].end_seconds <= 0.60


@pytest.mark.asyncio
async def test_chunk_indexes_are_stable_and_contiguous(tmp_path: Path) -> None:
    decisions = [False] * 3 + [True] * 5 + [False] * 12 + [True] * 5 + [False] * 3

    chunks, _, _ = await _segment(tmp_path, decisions)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


@pytest.mark.asyncio
async def test_repeated_runs_produce_identical_boundaries(tmp_path: Path) -> None:
    decisions = [True] * 50
    config = _config(prefix_padding_ms=0, suffix_padding_ms=0)

    first, _, _ = await _segment(
        tmp_path,
        decisions,
        config=config,
        destination_name="first",
        track_id="first",
    )
    second, _, _ = await _segment(
        tmp_path,
        decisions,
        config=config,
        destination_name="second",
        track_id="second",
    )

    assert _signature(first) == _signature(second)


class _CountingDecisions(Sequence[int]):
    def __init__(self, frame_count: int, value: int) -> None:
        self.frame_count = frame_count
        self.value = value
        self.reads = 0

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index: int | slice) -> int | list[int]:
        if isinstance(index, slice):
            indexes = range(*index.indices(self.frame_count))
            self.reads += len(indexes)
            return [self.value for _ in indexes]
        if index < 0:
            index += self.frame_count
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        self.reads += 1
        return self.value

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.frame_count):
            self.reads += 1
            yield self.value


@pytest.mark.asyncio
async def test_long_recording_planning_reads_frame_decisions_linearly(
    tmp_path: Path,
) -> None:
    frame_count = 90_000
    decisions = _CountingDecisions(frame_count, 1)
    segmenter, _ = _segmenter(tmp_path, [])

    chunks = await segmenter._plan_chunks(  # noqa: SLF001
        decisions,
        [1] * frame_count,
        frame_count * SAMPLES_PER_FRAME,
    )

    assert len(chunks) > 10
    assert decisions.reads < frame_count * 6


@pytest.mark.asyncio
async def test_cancellation_is_checked_during_long_region_planning(
    tmp_path: Path,
) -> None:
    frame_count = 5_000
    checks = 0
    segmenter, _ = _segmenter(tmp_path, [])

    async def cancellation_check() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 3

    with pytest.raises(AudioSegmentationCancelledError):
        await segmenter._plan_chunks(  # noqa: SLF001
            [1] * frame_count,
            [1] * frame_count,
            frame_count * SAMPLES_PER_FRAME,
            cancellation_check=cancellation_check,
        )

    assert checks == 3


class _RegionReadCounter:
    def __init__(self) -> None:
        self.reads = 0


class _CountingRegion:
    def __init__(
        self,
        start_sample: int,
        end_sample: int,
        counter: _RegionReadCounter,
    ) -> None:
        self._start_sample = start_sample
        self._end_sample = end_sample
        self._counter = counter

    @property
    def start_sample(self) -> int:
        self._counter.reads += 1
        return self._start_sample

    @property
    def end_sample(self) -> int:
        self._counter.reads += 1
        return self._end_sample


def test_tiny_region_merging_does_not_rescan_an_ineligible_prefix(
    tmp_path: Path,
) -> None:
    config = _config(end_silence_ms=80, merge_gap_ms=40)
    segmenter, _ = _segmenter(tmp_path, [], config=config)
    counter = _RegionReadCounter()
    regions: list[_CountingRegion] = []
    unsafe_gap = config.samples_for_ms(config.end_silence_ms) + 1
    eligible_gap = config.samples_for_ms(config.merge_gap_ms) + 1
    cursor = 0

    for _ in range(600):
        regions.append(
            _CountingRegion(
                cursor,
                cursor + SAMPLES_PER_FRAME,
                counter,
            )
        )
        cursor += SAMPLES_PER_FRAME + unsafe_gap
    for _ in range(300):
        regions.extend(
            (
                _CountingRegion(
                    cursor,
                    cursor + SAMPLES_PER_FRAME,
                    counter,
                ),
                _CountingRegion(
                    cursor + SAMPLES_PER_FRAME + eligible_gap,
                    cursor + (2 * SAMPLES_PER_FRAME) + eligible_gap,
                    counter,
                ),
            )
        )
        cursor += (2 * SAMPLES_PER_FRAME) + eligible_gap + unsafe_gap

    merged = segmenter._merge_tiny_regions(regions)  # type: ignore[arg-type]  # noqa: SLF001

    assert len(merged) == 900
    assert counter.reads < len(regions) * 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sample_rate_hz", "channel_count", "sample_width_bytes"),
    [
        (8000, 1, 2),
        (16_000, 2, 2),
        (16_000, 1, 1),
    ],
)
async def test_unsupported_wav_format_fails_without_creating_chunks(
    tmp_path: Path,
    sample_rate_hz: int,
    channel_count: int,
    sample_width_bytes: int,
) -> None:
    source = _write_wav(
        tmp_path / "unsupported.wav",
        [1000] * 10,
        sample_rate_hz=sample_rate_hz,
        channel_count=channel_count,
        sample_width_bytes=sample_width_bytes,
    )
    segmenter, factory = _segmenter(tmp_path, [True] * 10)

    with pytest.raises(InvalidAudioError):
        await segmenter.segment(
            AudioTrack(track_id="invalid", source_path=source),
            audio_info=_audio_info(
                source,
                duration_seconds=0.2,
                channel_count=channel_count,
            ),
            destination_dir=tmp_path / "chunks",
            max_upload_bytes=24 * 1024 * 1024,
        )

    assert len(factory.instances) == 1
    assert factory.instances[0].frames_seen == 0
    assert list((tmp_path / "chunks").glob("*")) == []


@pytest.mark.asyncio
async def test_corrupt_wav_fails_safely_without_creating_chunks(tmp_path: Path) -> None:
    source = tmp_path / "corrupt.wav"
    source.write_bytes(b"not a wave file")
    segmenter, factory = _segmenter(tmp_path, [True] * 10)

    with pytest.raises(InvalidAudioError):
        await segmenter.segment(
            AudioTrack(track_id="corrupt", source_path=source),
            audio_info=_audio_info(source, duration_seconds=0.2),
            destination_dir=tmp_path / "chunks",
            max_upload_bytes=24 * 1024 * 1024,
        )

    assert len(factory.instances) == 1
    assert factory.instances[0].frames_seen == 0
    assert list((tmp_path / "chunks").glob("*")) == []


class _CountingAudioProcessor(AudioProcessor):
    def __init__(self, settings: Settings, *, fail_on_call: int | None = None) -> None:
        super().__init__(settings)
        self.fail_on_call = fail_on_call
        self.range_calls = 0

    async def extract_pcm_wav_range(self, *args: Any, **kwargs: Any) -> Path:
        self.range_calls += 1
        if self.range_calls == self.fail_on_call:
            raise AudioToolError("simulated chunk creation failure")
        return await super().extract_pcm_wav_range(*args, **kwargs)


@pytest.mark.asyncio
async def test_cancellation_is_checked_during_chunk_creation_and_cleans_outputs(
    tmp_path: Path,
) -> None:
    processor = _CountingAudioProcessor(_settings(tmp_path))
    decisions = [False] * 3 + [True] * 5 + [False] * 12 + [True] * 5 + [False] * 3
    registered: list[Path] = []

    async def cancellation_check() -> bool:
        return processor.range_calls >= 1

    with pytest.raises(AudioSegmentationCancelledError):
        await _segment(
            tmp_path,
            decisions,
            processor=processor,
            cancellation_check=cancellation_check,
            registered=registered,
        )

    assert processor.range_calls == 1
    assert registered
    assert all(not path.exists() for path in registered)
    assert list((tmp_path / "chunks").glob("*")) == []


@pytest.mark.asyncio
async def test_chunk_creation_failure_removes_completed_temporary_files(
    tmp_path: Path,
) -> None:
    processor = _CountingAudioProcessor(_settings(tmp_path), fail_on_call=2)
    decisions = [False] * 3 + [True] * 5 + [False] * 12 + [True] * 5 + [False] * 3
    registered: list[Path] = []

    with pytest.raises(AudioToolError, match="simulated chunk creation failure"):
        await _segment(
            tmp_path,
            decisions,
            processor=processor,
            registered=registered,
        )

    assert processor.range_calls == 2
    assert len(registered) == 4
    assert all(not path.exists() for path in registered)
    assert list((tmp_path / "chunks").glob("*")) == []
