from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self
from uuid import uuid4
import wave

import pytest

from app.core.config import Settings
from app.models import TranscriptSegment
from app.models.enums import (
    SpeakerAttributionStatus,
    SpeakerSource,
    TranscriptionMode,
)
from app.services.audio import AudioInfo
from app.services.transcription.client import (
    TranscribedSegment,
    TranscriptionCancelledError,
    TranscriptionResult,
)
from app.services.transcription.confidence import ConfidenceAnalysis
from app.services.transcription.merge import merge_track_results
from app.services.transcription.orchestrator import (
    TranscriptionContext,
    TranscriptionOrchestrator,
)
from app.services.transcription.planning import (
    LegacyAudioPlanner,
    TopologyAudioPlanner,
)
from app.services.transcription.types import (
    AudioPlan,
    AudioTrack,
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    SpeechChunk,
    TrackTranscriptionResult,
    TranscriptionAttemptEvidence,
)
from app.workers.pipeline import _persist_transcription_result


def _settings(storage_root: Path) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=storage_root,
        OPENAI_API_KEY="test-only-key",
    )


def _audio_info(
    *,
    channel_count: int,
    duration_seconds: float = 30.0,
) -> AudioInfo:
    return AudioInfo(
        codec_name="pcm_s16le",
        format_name="wav",
        duration_seconds=duration_seconds,
        channel_count=channel_count,
        sample_rate_hz=16_000,
        bit_rate_bps=256_000,
        size_bytes=100,
        sha256_checksum="a" * 64,
    )


def _topology_plan(
    tmp_path: Path,
    *,
    channel_count: int = 2,
    stereo_separated: bool = True,
    operator_channel: int | None = None,
    operator_id: str | None = None,
    operator_display_name: str | None = None,
    caller_channel: int | None = None,
    callee_channel: int | None = None,
) -> AudioPlan:
    return TopologyAudioPlanner().plan(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=channel_count),
        diarized=False,
        channel_index=1,
        operator_id=operator_id,
        attribution_status="confirmed_by_pbx",
        audio_variant="stale-worker-hint",
        stereo_separated=stereo_separated,
        operator_channel=operator_channel,
        caller_channel=caller_channel,
        callee_channel=callee_channel,
        operator_display_name=operator_display_name,
    )


def test_confirmed_separated_stereo_operator_uses_only_the_safe_channel(
    tmp_path: Path,
) -> None:
    operator_id = str(uuid4())

    plan = _topology_plan(
        tmp_path,
        operator_channel=1,
        operator_id=operator_id,
        operator_display_name="Maria",
        caller_channel=0,
        callee_channel=1,
    )

    assert plan == AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=tmp_path / "source.wav",
                channel_index=1,
                operator_id=operator_id,
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Maria",
                speaker_source="stereo_channel",
                duration_seconds=30.0,
            ),
        ),
        operator_channel=1,
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="confirmed_by_pbx",
        reason="confirmed-separated-stereo-safe-operator",
    )


@pytest.mark.parametrize(
    ("operator_channel", "operator_id", "display_name"),
    [
        (None, None, None),
        (0, None, "Maria"),
        (1, str(uuid4()), None),
        (9, str(uuid4()), "Maria"),
    ],
)
def test_uncertain_separated_stereo_preserves_both_tracks_without_operator_ids(
    tmp_path: Path,
    operator_channel: int | None,
    operator_id: str | None,
    display_name: str | None,
) -> None:
    plan = _topology_plan(
        tmp_path,
        operator_channel=operator_channel,
        operator_id=operator_id,
        operator_display_name=display_name,
    )

    assert plan.mode == "dual_channel"
    assert plan.operator_channel is None
    assert plan.stereo_separated is True
    assert plan.attribution_status == "channel_unknown"
    assert [(track.channel_index, track.speaker_label) for track in plan.tracks] == [
        (0, "Channel A"),
        (1, "Channel B"),
    ]
    assert all(track.operator_id is None for track in plan.tracks)
    assert all(track.diarized is False for track in plan.tracks)


def test_caller_and_callee_labels_require_a_complete_two_channel_proof(
    tmp_path: Path,
) -> None:
    proven = _topology_plan(
        tmp_path,
        caller_channel=1,
        callee_channel=0,
    )

    assert proven.mode == "dual_channel"
    assert proven.attribution_status == "caller_callee_only"
    assert proven.caller_channel == 1
    assert proven.callee_channel == 0
    assert [(track.channel_index, track.speaker_label) for track in proven.tracks] == [
        (0, "Callee"),
        (1, "Caller"),
    ]


@pytest.mark.parametrize(
    ("caller_channel", "callee_channel"),
    [
        (0, None),
        (None, 1),
        (0, 0),
        (1, 1),
        (-1, 1),
        (0, 2),
    ],
)
def test_partial_or_invalid_caller_callee_hints_are_cleared(
    tmp_path: Path,
    caller_channel: int | None,
    callee_channel: int | None,
) -> None:
    plan = _topology_plan(
        tmp_path,
        caller_channel=caller_channel,
        callee_channel=callee_channel,
    )

    assert plan.mode == "dual_channel"
    assert plan.attribution_status == "channel_unknown"
    assert plan.caller_channel is None
    assert plan.callee_channel is None
    assert [track.speaker_label for track in plan.tracks] == [
        "Channel A",
        "Channel B",
    ]


def test_mono_uses_diarization_and_clears_stale_stereo_hints(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(
        tmp_path,
        channel_count=1,
        stereo_separated=True,
        operator_channel=1,
        operator_id=str(uuid4()),
        operator_display_name="Maria",
        caller_channel=0,
        callee_channel=1,
    )

    assert plan.mode == "mono_diarization"
    assert plan.operator_channel is None
    assert plan.stereo_separated is False
    assert plan.caller_channel is None
    assert plan.callee_channel is None
    assert plan.attribution_status == "anonymous_diarization"
    assert plan.reason == "mono-source"
    assert len(plan.tracks) == 1
    assert plan.tracks[0].channel_index is None
    assert plan.tracks[0].operator_id is None
    assert plan.tracks[0].diarized is True


def test_unconfirmed_stereo_uses_diarization_without_assuming_mapping(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(
        tmp_path,
        channel_count=2,
        stereo_separated=False,
        operator_channel=0,
        operator_id=str(uuid4()),
        operator_display_name="Maria",
        caller_channel=0,
        callee_channel=1,
    )

    assert plan.mode == "mono_diarization"
    assert plan.operator_channel is None
    assert plan.stereo_separated is False
    assert plan.caller_channel is None
    assert plan.callee_channel is None
    assert plan.attribution_status == "anonymous_diarization"
    assert plan.reason == "stereo-not-confirmed-separated"
    assert plan.tracks[0].channel_index is None
    assert plan.tracks[0].operator_id is None


def test_unsupported_multichannel_audio_uses_a_truthful_safe_fallback_reason(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(
        tmp_path,
        channel_count=3,
        stereo_separated=True,
        operator_channel=0,
        operator_id=str(uuid4()),
        operator_display_name="Maria",
        caller_channel=0,
        callee_channel=1,
    )

    assert plan.mode == "mono_diarization"
    assert plan.reason == "unsupported-channel-count"
    assert plan.operator_channel is None
    assert plan.caller_channel is None
    assert plan.callee_channel is None
    assert plan.tracks[0].operator_id is None


def test_legacy_planner_remains_selectable(tmp_path: Path) -> None:
    operator_id = str(uuid4())

    plan = LegacyAudioPlanner().plan(
        source_path=tmp_path / "prepared.wav",
        audio_info=_audio_info(channel_count=1),
        diarized=False,
        channel_index=1,
        operator_id=operator_id,
        attribution_status="confirmed_by_pbx",
        audio_variant="legacy-operator-channel",
    )

    assert plan.mode == "legacy"
    assert plan.reason == "legacy-worker-selected-topology"
    assert plan.tracks == (
        AudioTrack(
            track_id="legacy-operator",
            source_path=tmp_path / "prepared.wav",
            channel_index=1,
            operator_id=operator_id,
            attribution_status="confirmed_by_pbx",
            audio_variant="legacy-operator-channel",
        ),
    )


class RecordingAudioProcessor:
    def __init__(self, *, fail_channel: int | None = None) -> None:
        self.fail_channel = fail_channel
        self.extract_calls: list[tuple[Path, Path, int]] = []
        self.mono_calls: list[tuple[Path, Path]] = []

    async def extract_channel(
        self,
        source: Path,
        destination: Path,
        channel_index: int,
    ) -> None:
        self.extract_calls.append((source, destination, channel_index))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(f"channel-{channel_index}".encode())
        if channel_index == self.fail_channel:
            raise RuntimeError("simulated extraction failure")

    async def convert_to_mono(self, source: Path, destination: Path) -> None:
        self.mono_calls.append((source, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16_000)
            writer.writeframes(b"\x00\x00" * 30 * 16_000)

    async def extract_pcm_wav_range(
        self,
        source: Path,
        destination: Path,
        *,
        start_sample: int,
        end_sample: int,
        sample_rate_hz: int = 16_000,
    ) -> Path:
        del source
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate_hz)
            writer.writeframes(b"\x00\x00" * (end_sample - start_sample))
        return destination


class IndependentFixedSegmenter:
    def __init__(self) -> None:
        self.calls: list[AudioTrack] = []

    async def segment(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
        destination_dir: Path,
        max_upload_bytes: int,
        register_temporary_file: Any,
        cancellation_check: Any = None,
    ) -> tuple[SpeechChunk, ...]:
        del (
            audio_info,
            max_upload_bytes,
            register_temporary_file,
            cancellation_check,
        )
        self.calls.append(track)
        channel = track.channel_index if track.channel_index is not None else 0
        return (
            SpeechChunk(
                track_id=track.track_id,
                chunk_index=0,
                path=destination_dir / f"{track.track_id}-0.wav",
                start_seconds=0.0,
                end_seconds=15.0,
                hard_cut=True,
                audio_variant=track.audio_variant,
            ),
            SpeechChunk(
                track_id=track.track_id,
                chunk_index=1,
                path=destination_dir / f"{track.track_id}-1.wav",
                start_seconds=15.0,
                end_seconds=30.0,
                hard_cut=False,
                audio_variant=f"{track.audio_variant}-{channel}",
            ),
        )


class CapturingTranscriptionClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.isolated_calls: list[list[Any]] = []
        self.diarized_calls: list[list[Any]] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def transcribe_isolated(
        self,
        chunks: list[Any],
        *_args: object,
        **_kwargs: object,
    ) -> TranscriptionResult:
        self.isolated_calls.append(chunks)
        if self.fail:
            raise RuntimeError("simulated transcription failure")
        call_index = len(self.isolated_calls) - 1
        return TranscriptionResult(
            model="test-isolated",
            language="el",
            prompt_version="legacy-prompt",
            processing_duration_seconds=0.25,
            segments=[
                TranscribedSegment(
                    start_seconds=chunk.start_seconds + 1.0,
                    end_seconds=chunk.end_seconds,
                    text=f"track-{call_index}-chunk-{position}",
                    speaker_label="provider-label",
                )
                for position, chunk in enumerate(chunks)
            ],
            usage={"track": call_index},
            diarized=False,
        )

    async def transcribe_diarized(
        self,
        chunks: list[Any],
        *_args: object,
        **_kwargs: object,
    ) -> TranscriptionResult:
        self.diarized_calls.append(chunks)
        return TranscriptionResult(
            model="test-diarized",
            language="el",
            prompt_version=None,
            processing_duration_seconds=0.25,
            segments=[
                TranscribedSegment(
                    start_seconds=chunk.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=f"chunk-{position}",
                    speaker_label=f"Speaker {position}",
                )
                for position, chunk in enumerate(chunks)
            ],
            usage={},
            diarized=True,
        )

    async def transcribe_diarized_complete(
        self,
        chunk: Any,
        *_args: object,
        **_kwargs: object,
    ) -> TranscriptionResult:
        self.diarized_calls.append([chunk])
        return TranscriptionResult(
            model="test-diarized",
            language="el",
            prompt_version=None,
            processing_duration_seconds=0.25,
            segments=[
                TranscribedSegment(
                    start_seconds=0.0,
                    end_seconds=15.0,
                    text="rough mono text",
                    speaker_label="A",
                )
            ],
            usage={"chunks": [{"input_tokens": 2}], "totals": {"input_tokens": 2}},
            diarized=True,
        )


class SilentSegmenter:
    def __init__(self) -> None:
        self.calls: list[AudioTrack] = []

    async def segment(
        self,
        track: AudioTrack,
        *,
        audio_info: AudioInfo,
        destination_dir: Path,
        max_upload_bytes: int,
        register_temporary_file: Any,
        cancellation_check: Any = None,
    ) -> tuple[SpeechChunk, ...]:
        del (
            audio_info,
            destination_dir,
            max_upload_bytes,
            register_temporary_file,
            cancellation_check,
        )
        self.calls.append(track)
        return ()


def _context(tmp_path: Path, registered: list[Path]) -> TranscriptionContext:
    return TranscriptionContext(
        diarized=False,
        language="el",
        vocabulary=("Yeastar",),
        temporary_directory=tmp_path / "temporary",
        register_temporary_file=registered.append,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_speech_tracks", "expected_legacy_tracks"),
    [
        ("operator_channel", ["operator-channel"], []),
        ("dual_channel", ["channel-0", "channel-1"], []),
        ("mono_diarization", [], []),
        ("legacy", [], ["legacy-operator"]),
    ],
)
async def test_orchestrator_routes_only_operator_and_dual_modes_to_speech_segmentation(
    tmp_path: Path,
    mode: str,
    expected_speech_tracks: list[str],
    expected_legacy_tracks: list[str],
) -> None:
    if mode == "operator_channel":
        plan = _topology_plan(
            tmp_path,
            operator_channel=0,
            operator_id=str(uuid4()),
            operator_display_name="Maria",
        )
        channel_count = 2
    elif mode == "dual_channel":
        plan = _topology_plan(tmp_path)
        channel_count = 2
    elif mode == "mono_diarization":
        plan = _topology_plan(tmp_path, channel_count=1)
        channel_count = 1
    else:
        plan = LegacyAudioPlanner().plan(
            source_path=tmp_path / "prepared.wav",
            audio_info=_audio_info(channel_count=1),
            diarized=False,
            channel_index=0,
            operator_id=str(uuid4()),
            attribution_status="confirmed_by_pbx",
            audio_variant="legacy-operator-channel",
        )
        channel_count = 1

    speech_segmenter = IndependentFixedSegmenter()
    legacy_segmenter = IndependentFixedSegmenter()

    await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=RecordingAudioProcessor(),  # type: ignore[arg-type]
        speech_segmenter=speech_segmenter,  # type: ignore[arg-type]
        legacy_segmenter=legacy_segmenter,  # type: ignore[arg-type]
        client_factory=CapturingTranscriptionClient,  # type: ignore[arg-type]
    ).transcribe(
        source_path=plan.tracks[0].source_path,
        audio_info=_audio_info(channel_count=channel_count),
        context=_context(tmp_path, []),
        plan=plan,
    )

    assert [track.track_id for track in speech_segmenter.calls] == expected_speech_tracks
    assert [track.track_id for track in legacy_segmenter.calls] == expected_legacy_tracks


@pytest.mark.asyncio
async def test_silence_on_both_dual_tracks_skips_client_creation_and_provider_work(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(tmp_path)
    segmenter = SilentSegmenter()
    factory_calls = 0

    def client_factory() -> CapturingTranscriptionClient:
        nonlocal factory_calls
        factory_calls += 1
        return CapturingTranscriptionClient()

    result = await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=RecordingAudioProcessor(),  # type: ignore[arg-type]
        speech_segmenter=segmenter,  # type: ignore[arg-type]
        legacy_segmenter=IndependentFixedSegmenter(),  # type: ignore[arg-type]
        client_factory=client_factory,  # type: ignore[arg-type]
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=2),
        context=_context(tmp_path, []),
        plan=plan,
    )

    assert [track.track_id for track in segmenter.calls] == ["channel-0", "channel-1"]
    assert factory_calls == 0
    assert result.text == ""
    assert result.segments == ()
    assert [track.hypotheses for track in result.tracks] == [(), ()]


@pytest.mark.asyncio
async def test_dual_channel_orchestration_extracts_both_channels_and_never_mixes_to_mono(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(tmp_path)
    processor = RecordingAudioProcessor()
    segmenter = IndependentFixedSegmenter()
    client = CapturingTranscriptionClient()
    registered: list[Path] = []

    result = await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=processor,  # type: ignore[arg-type]
        planner=TopologyAudioPlanner(),
        segmenter=segmenter,  # type: ignore[arg-type]
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=2),
        context=_context(tmp_path, registered),
        plan=plan,
    )

    assert [call[2] for call in processor.extract_calls] == [0, 1]
    assert processor.mono_calls == []
    assert [track.track_id for track in segmenter.calls] == ["channel-0", "channel-1"]
    assert [track.source_path.name for track in segmenter.calls] == [
        "channel-0.wav",
        "channel-1.wav",
    ]
    assert len(client.isolated_calls) == 4
    assert client.diarized_calls == []
    assert [
        [(chunk.start_seconds, chunk.end_seconds) for chunk in chunks]
        for chunks in client.isolated_calls
    ] == [
        [(0.0, 15.0)],
        [(15.0, 30.0)],
        [(0.0, 15.0)],
        [(15.0, 30.0)],
    ]
    assert [(segment.start_seconds, segment.end_seconds) for segment in result.segments] == [
        (1.0, 15.0),
        (1.0, 15.0),
        (16.0, 30.0),
        (16.0, 30.0),
    ]
    assert [(segment.channel_index, segment.chunk_index) for segment in result.segments] == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    ]
    assert [segment.speaker_label for segment in result.segments] == [
        "Channel A",
        "Channel B",
        "Channel A",
        "Channel B",
    ]
    assert all(segment.operator_id is None for segment in result.segments)
    assert registered == [
        tmp_path / "temporary" / "channel-0.wav",
        tmp_path / "temporary" / "channel-1.wav",
    ]


def _overlapping_chunks(tmp_path: Path) -> tuple[SpeechChunk, ...]:
    return (
        SpeechChunk(
            track_id="channel-0",
            chunk_index=0,
            path=tmp_path / "chunk-0.wav",
            start_seconds=0.0,
            end_seconds=40.0,
            hard_cut=True,
        ),
        SpeechChunk(
            track_id="channel-0",
            chunk_index=1,
            path=tmp_path / "chunk-1.wav",
            start_seconds=39.2,
            end_seconds=60.0,
            overlap_before_ms=800,
        ),
    )


def test_explicit_source_chunk_index_resolves_ambiguous_overlap_intervals(
    tmp_path: Path,
) -> None:
    chunks = _overlapping_chunks(tmp_path)
    result = TranscriptionResult(
        model="test-model",
        language="el",
        prompt_version="test-prompt",
        processing_duration_seconds=0.1,
        segments=[
            TranscribedSegment(
                start_seconds=39.2,
                end_seconds=40.0,
                text="end of first",
                speaker_label="Channel A",
                source_chunk_index=0,
            ),
            TranscribedSegment(
                start_seconds=39.2,
                end_seconds=60.0,
                text="start of second",
                speaker_label="Channel A",
                source_chunk_index=1,
            ),
        ],
    )

    track_result = TranscriptionOrchestrator._track_result(
        AudioTrack(
            track_id="channel-0",
            source_path=tmp_path / "channel-0.wav",
            channel_index=0,
            speaker_label="Channel A",
        ),
        chunks,
        result,
    )

    assert [
        (
            hypothesis.chunk_index,
            hypothesis.hard_cut,
            hypothesis.overlap_before_ms,
        )
        for hypothesis in track_result.hypotheses
    ] == [
        (0, True, 0),
        (1, False, 800),
    ]


def test_exact_adjacent_same_track_hard_cut_prefix_is_trimmed() -> None:
    hypotheses = [
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=40.0,
            text="alpha repeated phrase",
            speaker_label="Channel A",
            channel_index=0,
            hard_cut=True,
        ),
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=1,
            start_seconds=39.2,
            end_seconds=60.0,
            text="repeated phrase continues",
            speaker_label="Channel A",
            channel_index=0,
            overlap_before_ms=800,
        ),
    ]

    result = TranscriptionOrchestrator._remove_exact_hard_cut_overlap(hypotheses)

    assert [hypothesis.text for hypothesis in result] == [
        "alpha repeated phrase",
        "continues",
    ]


@pytest.mark.parametrize(
    ("previous_text", "current_text", "current_chunk_index"),
    [
        ("alpha repeated phrase", "Repeated phrase continues", 1),
        ("alpha repeated phrase", "repeated phrase, continues", 1),
        ("alpha repeated", "repeated continues", 1),
        ("alpha repeated phrase", "repeated phrase continues", 2),
    ],
)
def test_overlap_join_retains_case_punctuation_one_token_and_nonadjacent_repetition(
    previous_text: str,
    current_text: str,
    current_chunk_index: int,
) -> None:
    hypotheses = [
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=40.0,
            text=previous_text,
            speaker_label="Channel A",
            channel_index=0,
            hard_cut=True,
        ),
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=current_chunk_index,
            start_seconds=39.2,
            end_seconds=60.0,
            text=current_text,
            speaker_label="Channel A",
            channel_index=0,
            overlap_before_ms=800,
        ),
    ]

    result = TranscriptionOrchestrator._remove_exact_hard_cut_overlap(hypotheses)

    assert [hypothesis.text for hypothesis in result] == [
        previous_text,
        current_text,
    ]


def test_overlap_join_never_removes_identical_text_across_channels() -> None:
    hypotheses = [
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=40.0,
            text="identical repeated phrase",
            speaker_label="Channel A",
            channel_index=0,
            hard_cut=True,
        ),
        ChunkHypothesis(
            track_id="channel-1",
            chunk_index=1,
            start_seconds=39.2,
            end_seconds=60.0,
            text="identical repeated phrase",
            speaker_label="Channel B",
            channel_index=1,
            overlap_before_ms=800,
        ),
    ]

    result = TranscriptionOrchestrator._remove_exact_hard_cut_overlap(hypotheses)

    assert [hypothesis.text for hypothesis in result] == [
        "identical repeated phrase",
        "identical repeated phrase",
    ]


@pytest.mark.asyncio
async def test_operator_channel_orchestration_extracts_only_the_confirmed_track(
    tmp_path: Path,
) -> None:
    operator_id = str(uuid4())
    plan = _topology_plan(
        tmp_path,
        operator_channel=1,
        operator_id=operator_id,
        operator_display_name="Maria",
    )
    processor = RecordingAudioProcessor()
    segmenter = IndependentFixedSegmenter()
    client = CapturingTranscriptionClient()

    result = await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=processor,  # type: ignore[arg-type]
        planner=TopologyAudioPlanner(),
        segmenter=segmenter,  # type: ignore[arg-type]
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=2),
        context=_context(tmp_path, []),
        plan=plan,
    )

    assert [call[2] for call in processor.extract_calls] == [1]
    assert processor.mono_calls == []
    assert len(client.isolated_calls) == 2
    assert client.diarized_calls == []
    assert {segment.operator_id for segment in result.segments} == {operator_id}
    assert {segment.speaker_label for segment in result.segments} == {"Maria"}
    assert {segment.channel_index for segment in result.segments} == {1}


@pytest.mark.asyncio
async def test_mono_plan_uses_conversion_and_diarized_client(tmp_path: Path) -> None:
    plan = _topology_plan(tmp_path, channel_count=1)
    processor = RecordingAudioProcessor()
    segmenter = IndependentFixedSegmenter()
    client = CapturingTranscriptionClient()

    result = await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        audio_processor=processor,  # type: ignore[arg-type]
        planner=TopologyAudioPlanner(),
        segmenter=segmenter,  # type: ignore[arg-type]
        client_factory=lambda: client,  # type: ignore[arg-type]
    ).transcribe(
        source_path=tmp_path / "source.wav",
        audio_info=_audio_info(channel_count=1),
        context=_context(tmp_path, []),
        plan=plan,
    )

    assert processor.extract_calls == []
    assert processor.mono_calls == [
        (
            tmp_path / "source.wav",
            tmp_path / "temporary" / "mono-diarization.wav",
        )
    ]
    assert len(client.isolated_calls) == 1
    assert len(client.diarized_calls) == 1
    assert result.mode == "mono_diarization"
    assert result.diarized is True
    assert result.segments[0].text == "track-0-chunk-0"
    assert result.segments[0].speaker_label == "A"
    assert result.segments[0].operator_id is None


@pytest.mark.asyncio
async def test_cancellation_after_one_dual_extraction_leaves_that_file_registered(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(tmp_path)
    processor = RecordingAudioProcessor()
    registered: list[Path] = []
    checks = 0

    async def cancel_before_second_track() -> bool:
        nonlocal checks
        checks += 1
        return checks == 2

    with pytest.raises(TranscriptionCancelledError):
        await TranscriptionOrchestrator(
            settings=_settings(tmp_path),
            audio_processor=processor,  # type: ignore[arg-type]
            planner=TopologyAudioPlanner(),
            segmenter=IndependentFixedSegmenter(),  # type: ignore[arg-type]
            client_factory=lambda: CapturingTranscriptionClient(),  # type: ignore[arg-type]
        ).transcribe(
            source_path=tmp_path / "source.wav",
            audio_info=_audio_info(channel_count=2),
            context=_context(tmp_path, registered),
            plan=plan,
            cancellation_check=cancel_before_second_track,
        )

    assert [call[2] for call in processor.extract_calls] == [0]
    assert registered == [tmp_path / "temporary" / "channel-0.wav"]


@pytest.mark.asyncio
async def test_partial_output_is_registered_even_when_channel_extraction_fails(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(tmp_path)
    processor = RecordingAudioProcessor(fail_channel=1)
    registered: list[Path] = []

    with pytest.raises(RuntimeError, match="simulated extraction failure"):
        await TranscriptionOrchestrator(
            settings=_settings(tmp_path),
            audio_processor=processor,  # type: ignore[arg-type]
            planner=TopologyAudioPlanner(),
            segmenter=IndependentFixedSegmenter(),  # type: ignore[arg-type]
            client_factory=lambda: CapturingTranscriptionClient(),  # type: ignore[arg-type]
        ).transcribe(
            source_path=tmp_path / "source.wav",
            audio_info=_audio_info(channel_count=2),
            context=_context(tmp_path, registered),
            plan=plan,
        )

    assert [call[2] for call in processor.extract_calls] == [0, 1]
    assert registered == [
        tmp_path / "temporary" / "channel-0.wav",
        tmp_path / "temporary" / "channel-1.wav",
    ]


def _track_result(
    track_id: str,
    *hypotheses: ChunkHypothesis,
) -> TrackTranscriptionResult:
    return TrackTranscriptionResult(
        track_id=track_id,
        model="test-model",
        language="el",
        prompt_version="test-prompt",
        processing_duration_seconds=0.5,
        hypotheses=tuple(hypotheses),
        usage={"track_id": track_id},
    )


def test_dual_merge_orders_by_time_channel_and_chunk_without_deduplicating_overlap(
    tmp_path: Path,
) -> None:
    plan = _topology_plan(tmp_path)
    channel_1 = _track_result(
        "channel-1",
        ChunkHypothesis(
            track_id="channel-1",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=3.0,
            text="shorter end",
            speaker_label="Channel B",
            channel_index=1,
        ),
        ChunkHypothesis(
            track_id="channel-1",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=4.0,
            text="same words",
            speaker_label="Channel B",
            channel_index=1,
        ),
    )
    channel_0 = _track_result(
        "channel-0",
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=1,
            start_seconds=0.0,
            end_seconds=4.0,
            text="same words",
            speaker_label="Channel A",
            channel_index=0,
        ),
        ChunkHypothesis(
            track_id="channel-0",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=4.0,
            text="earlier chunk",
            speaker_label="Channel A",
            channel_index=0,
        ),
    )

    result = merge_track_results(
        plan,
        [channel_1, channel_0],
        ConfidenceAnalysis(),
    )

    assert [
        (segment.end_seconds, segment.channel_index, segment.chunk_index, segment.text)
        for segment in result.segments
    ] == [
        (3.0, 1, 0, "shorter end"),
        (4.0, 0, 0, "earlier chunk"),
        (4.0, 0, 1, "same words"),
        (4.0, 1, 0, "same words"),
    ]
    assert len(result.segments) == 4
    assert [segment.text for segment in result.segments].count("same words") == 2


class CapturingSession:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, value: Any) -> None:
        if isinstance(value, TranscriptSegment):
            self.added.append(value)

    async def flush(self) -> None:
        return None


def _transcript_state() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        transcription_mode=TranscriptionMode.LEGACY,
        speaker_attribution_status=None,
    )


def _orchestrated_result(
    *,
    mode: str,
    attribution_status: str | None,
    segments: tuple[ChunkHypothesis, ...],
) -> OrchestratedTranscriptionResult:
    track = _track_result(segments[0].track_id, *segments)
    attempts = (
        tuple(
            TranscriptionAttemptEvidence(
                track_id=segment.track_id,
                chunk_index=segment.chunk_index or 0,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                model=track.model,
                audio_variant=segment.audio_variant,
                prompt_hash="a" * 64,
                response_text=segment.text,
                selected=True,
            )
            for segment in segments
        )
        if mode in {"operator_channel", "dual_channel"}
        else ()
    )
    return OrchestratedTranscriptionResult(
        mode=mode,  # type: ignore[arg-type]
        model=track.model,
        language=track.language,
        prompt_version=track.prompt_version,
        processing_duration_seconds=track.processing_duration_seconds,
        segments=segments,
        tracks=(track,),
        usage={"tracks": [track.usage]},
        attribution_status=attribution_status,
        attempts=attempts,
    )


@pytest.mark.asyncio
async def test_dual_channel_persistence_keeps_topology_metadata_and_null_operator_ids() -> None:
    transcript = _transcript_state()
    session = CapturingSession()
    call = SimpleNamespace(id=uuid4())
    selected_operator = SimpleNamespace(id=uuid4(), display_name="Selected operator")
    call_leg_id = uuid4()
    result = _orchestrated_result(
        mode="dual_channel",
        attribution_status="channel_unknown",
        segments=(
            ChunkHypothesis(
                track_id="channel-0",
                chunk_index=2,
                start_seconds=1.2345,
                end_seconds=3.4567,
                text="alpha",
                speaker_label="Channel A",
                channel_index=0,
                operator_id=None,
                speaker_source="stereo_channel",
                audio_variant="topology-channel-0",
            ),
            ChunkHypothesis(
                track_id="channel-1",
                chunk_index=2,
                start_seconds=1.5,
                end_seconds=3.75,
                text="beta",
                speaker_label="Channel B",
                channel_index=1,
                operator_id=None,
                speaker_source="stereo_channel",
                audio_variant="topology-channel-1",
            ),
        ),
    )

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        result,
        call,  # type: ignore[arg-type]
        selected_operator,  # type: ignore[arg-type]
        call_leg_id,
        30.0,
    )

    assert transcript.transcription_mode is TranscriptionMode.DUAL_CHANNEL
    assert transcript.speaker_attribution_status is SpeakerAttributionStatus.CHANNEL_UNKNOWN
    assert [
        (
            segment.channel_index,
            segment.track_id,
            segment.chunk_index,
            segment.speaker_label,
            segment.speaker_source,
            segment.operator_id,
            segment.call_leg_id,
            segment.audio_variant,
        )
        for segment in session.added
    ] == [
        (
            0,
            "channel-0",
            2,
            "Channel A",
            SpeakerSource.STEREO_CHANNEL,
            None,
            None,
            "topology-channel-0",
        ),
        (
            1,
            "channel-1",
            2,
            "Channel B",
            SpeakerSource.STEREO_CHANNEL,
            None,
            None,
            "topology-channel-1",
        ),
    ]
    assert session.added[0].start_seconds == Decimal("1.234")
    assert session.added[0].end_seconds == Decimal("3.457")


@pytest.mark.asyncio
async def test_confirmed_operator_persistence_retains_operator_and_call_leg() -> None:
    operator_id = uuid4()
    transcript = _transcript_state()
    session = CapturingSession()
    call = SimpleNamespace(id=uuid4())
    operator = SimpleNamespace(id=operator_id, display_name="Maria")
    call_leg_id = uuid4()
    result = _orchestrated_result(
        mode="operator_channel",
        attribution_status="confirmed_by_pbx",
        segments=(
            ChunkHypothesis(
                track_id="operator-channel",
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=5.0,
                text="operator words",
                speaker_label="Maria",
                channel_index=1,
                operator_id=str(operator_id),
                speaker_source="stereo_channel",
                audio_variant="topology-operator-channel",
            ),
        ),
    )

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        result,
        call,  # type: ignore[arg-type]
        operator,  # type: ignore[arg-type]
        call_leg_id,
        5.0,
    )

    segment = session.added[0]
    assert transcript.transcription_mode is TranscriptionMode.OPERATOR_CHANNEL
    assert transcript.speaker_attribution_status is SpeakerAttributionStatus.CONFIRMED_BY_PBX
    assert segment.operator_id == operator_id
    assert segment.call_leg_id == call_leg_id
    assert segment.channel_index == 1
    assert segment.track_id == "operator-channel"


@pytest.mark.asyncio
async def test_legacy_persistence_ignores_v2_segment_fields_and_preserves_legacy_mapping() -> None:
    operator_id = uuid4()
    transcript = _transcript_state()
    transcript.speaker_attribution_status = SpeakerAttributionStatus.CONFIRMED_BY_PBX
    session = CapturingSession()
    call = SimpleNamespace(id=uuid4())
    operator = SimpleNamespace(id=operator_id, display_name="Legacy Maria")
    call_leg_id = uuid4()
    result = _orchestrated_result(
        mode="legacy",
        attribution_status=None,
        segments=(
            ChunkHypothesis(
                track_id="future-track-ignored",
                chunk_index=7,
                start_seconds=0.0,
                end_seconds=5.0,
                text="legacy words",
                speaker_label="provider-label",
                channel_index=1,
                operator_id=str(uuid4()),
                speaker_source="unknown",
                audio_variant="future-variant-ignored",
            ),
        ),
    )

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        result,
        call,  # type: ignore[arg-type]
        operator,  # type: ignore[arg-type]
        call_leg_id,
        5.0,
    )

    segment = session.added[0]
    assert transcript.transcription_mode is TranscriptionMode.LEGACY
    assert transcript.speaker_attribution_status is SpeakerAttributionStatus.CONFIRMED_BY_PBX
    assert segment.operator_id == operator_id
    assert segment.call_leg_id == call_leg_id
    assert segment.speaker_label == "Legacy Maria"
    assert segment.speaker_source is SpeakerSource.STEREO_CHANNEL
    assert segment.channel_index is None
    assert segment.track_id is None
    assert segment.chunk_index is None
    assert segment.audio_variant is None
