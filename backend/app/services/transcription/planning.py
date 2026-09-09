from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.services.audio import AudioInfo
from app.services.transcription.types import AudioPlan, AudioTrack


class AudioPlanner(Protocol):
    def plan(
        self,
        *,
        source_path: Path,
        audio_info: AudioInfo,
        diarized: bool,
        channel_index: int | None,
        operator_id: str | None,
        attribution_status: str | None,
        audio_variant: str | None,
        stereo_separated: bool,
        operator_channel: int | None,
        caller_channel: int | None,
        callee_channel: int | None,
        operator_display_name: str | None,
    ) -> AudioPlan: ...


class LegacyAudioPlanner:
    """Describe the topology already selected by the legacy worker."""

    def plan(
        self,
        *,
        source_path: Path,
        audio_info: AudioInfo,
        diarized: bool,
        channel_index: int | None,
        operator_id: str | None,
        attribution_status: str | None,
        audio_variant: str | None,
        stereo_separated: bool = False,
        operator_channel: int | None = None,
        caller_channel: int | None = None,
        callee_channel: int | None = None,
        operator_display_name: str | None = None,
    ) -> AudioPlan:
        del (
            audio_info,
            stereo_separated,
            operator_channel,
            caller_channel,
            callee_channel,
            operator_display_name,
        )
        track_id = "legacy-diarized" if diarized else "legacy-operator"
        return AudioPlan(
            mode="legacy",
            tracks=(
                AudioTrack(
                    track_id=track_id,
                    source_path=source_path,
                    channel_index=channel_index,
                    operator_id=operator_id,
                    attribution_status=attribution_status,
                    audio_variant=audio_variant,
                    diarized=diarized,
                ),
            ),
            operator_channel=channel_index if not diarized else None,
            stereo_separated=channel_index is not None,
            attribution_status=attribution_status,
            reason="legacy-worker-selected-topology",
        )


class TopologyAudioPlanner:
    """Select topology only from sanitized file and PBX evidence."""

    def plan(
        self,
        *,
        source_path: Path,
        audio_info: AudioInfo,
        diarized: bool,
        channel_index: int | None,
        operator_id: str | None,
        attribution_status: str | None,
        audio_variant: str | None,
        stereo_separated: bool,
        operator_channel: int | None,
        caller_channel: int | None,
        callee_channel: int | None,
        operator_display_name: str | None,
    ) -> AudioPlan:
        del diarized, channel_index, attribution_status, audio_variant
        is_confirmed_separated_stereo = (
            audio_info.channel_count == 2 and stereo_separated
        )
        safe_operator_channel = (
            operator_channel
            if operator_channel in {0, 1}
            and operator_id is not None
            and bool(operator_display_name)
            else None
        )
        if is_confirmed_separated_stereo and safe_operator_channel is not None:
            return AudioPlan(
                mode="operator_channel",
                tracks=(
                    AudioTrack(
                        track_id="operator-channel",
                        source_path=source_path,
                        channel_index=safe_operator_channel,
                        operator_id=operator_id,
                        attribution_status="confirmed_by_pbx",
                        audio_variant="topology-operator-channel",
                        speaker_label=operator_display_name,
                        speaker_source="stereo_channel",
                        duration_seconds=audio_info.duration_seconds,
                    ),
                ),
                operator_channel=safe_operator_channel,
                stereo_separated=True,
                caller_channel=(
                    caller_channel if caller_channel in {0, 1} else None
                ),
                callee_channel=(
                    callee_channel if callee_channel in {0, 1} else None
                ),
                attribution_status="confirmed_by_pbx",
                reason="confirmed-separated-stereo-safe-operator",
            )

        if is_confirmed_separated_stereo:
            mapping_proven = (
                caller_channel in {0, 1}
                and callee_channel in {0, 1}
                and caller_channel != callee_channel
                and {caller_channel, callee_channel} == {0, 1}
            )
            labels = (
                {
                    int(caller_channel): "Caller",
                    int(callee_channel): "Callee",
                }
                if mapping_proven
                else {0: "Channel A", 1: "Channel B"}
            )
            status = "caller_callee_only" if mapping_proven else "channel_unknown"
            reason = (
                "confirmed-separated-stereo-caller-callee"
                if mapping_proven
                else "confirmed-separated-stereo-attribution-unknown"
            )
            return AudioPlan(
                mode="dual_channel",
                tracks=tuple(
                    AudioTrack(
                        track_id=f"channel-{channel}",
                        source_path=source_path,
                        channel_index=channel,
                        operator_id=None,
                        attribution_status=status,
                        audio_variant=f"topology-channel-{channel}",
                        speaker_label=labels[channel],
                        speaker_source="stereo_channel",
                        duration_seconds=audio_info.duration_seconds,
                    )
                    for channel in (0, 1)
                ),
                operator_channel=None,
                stereo_separated=True,
                caller_channel=int(caller_channel) if mapping_proven else None,
                callee_channel=int(callee_channel) if mapping_proven else None,
                attribution_status=status,
                reason=reason,
            )

        if audio_info.channel_count == 1:
            reason = "mono-source"
        elif audio_info.channel_count == 2:
            reason = "stereo-not-confirmed-separated"
        else:
            reason = "unsupported-channel-count"
        return AudioPlan(
            mode="mono_diarization",
            tracks=(
                AudioTrack(
                    track_id="mono-diarization",
                    source_path=source_path,
                    operator_id=None,
                    attribution_status="anonymous_diarization",
                    audio_variant="topology-mono",
                    diarized=True,
                    speaker_source="openai_diarization",
                    duration_seconds=audio_info.duration_seconds,
                ),
            ),
            operator_channel=None,
            stereo_separated=False,
            caller_channel=None,
            callee_channel=None,
            attribution_status="anonymous_diarization",
            reason=reason,
        )
