from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from app.schemas.common import APIModel
from app.schemas.jobs import JobSummary


class DashboardResponse(APIModel):
    has_data: bool
    calls_analyzed: int
    calls_with_recordings: int
    calls_transcribed: int
    calls_with_matches: int
    failed_calls: int
    processing_jobs: int
    results_by_operator: list[dict]
    results_by_keyword_category: list[dict]
    recent_jobs: list[JobSummary] = Field(default_factory=list)


class ResultItem(APIModel):
    job_id: UUID
    call_id: UUID
    started_at: datetime
    operator_id: UUID | None
    operator_name: str | None
    masked_phone_number: str | None
    duration_seconds: int
    direction: str
    keywords_found: list[str]
    match_count: int
    processing_status: str


class MatchResponse(APIModel):
    id: UUID
    transcript_segment_id: UUID
    keyword_id: UUID
    keyword: str
    category: str
    original_matched_text: str
    context_before: str
    context_after: str
    start_timestamp: Decimal
    end_timestamp: Decimal
    match_method: str
    match_score: Decimal


class TranscriptSegmentResponse(APIModel):
    id: UUID
    operator_id: UUID | None
    speaker_label: str
    speaker_source: str
    start_timestamp: Decimal
    end_timestamp: Decimal
    original_text: str
    confidence: Decimal | None
    transcription_model: str
    mean_logprob: Decimal | None = None
    low_logprob_ratio: Decimal | None = None
    quality_flags: list[str] = Field(default_factory=list)
    audio_variant: str | None = None
    channel_index: int | None = None
    sequence_number: int | None = None
    matches: list[MatchResponse]


class TranscriptQualitySummaryResponse(APIModel):
    transcript_id: UUID
    transcription_mode: (
        Literal[
            "legacy",
            "operator_channel",
            "dual_channel",
            "mono_diarization",
        ]
        | None
    )
    quality_summary: dict[str, Any] | None = None


class CallReprocessRequest(APIModel):
    pipeline_version: Literal["legacy-v1", "pipeline-v2"]
    transcript_id: UUID | None = None
    operator_id: UUID | None = None


class SpeakerAssignmentRequest(APIModel):
    transcript_id: UUID
    operator_id: UUID
    operator_channel_index: Literal[0, 1]


class SpeakerAssignmentResponse(APIModel):
    transcript_id: UUID
    transcription_mode: (
        Literal[
            "legacy",
            "operator_channel",
            "dual_channel",
            "mono_diarization",
        ]
        | None
    )
    speaker_attribution_status: (
        Literal[
            "confirmed_by_pbx",
            "caller_callee_only",
            "channel_unknown",
            "anonymous_diarization",
            "manually_assigned",
        ]
        | None
    )
    speaker_assignment_required: bool
    available_channels: list[int] = Field(default_factory=list)
    operator_id: UUID | None = None
    operator_channel_index: int | None = None
    confidence_status: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    pipeline_version: str | None = None


class CallDetailResponse(APIModel):
    id: UUID
    transcript_id: UUID | None = None
    started_at: datetime
    caller: str | None
    caller_name: str | None
    callee: str | None
    callee_name: str | None
    duration_seconds: int
    direction: str
    call_status: str | None
    queue: str | None
    processing_status: str
    audio_available: bool
    participants: list[dict]
    matches: list[MatchResponse]
    transcript_segments: list[TranscriptSegmentResponse]
    transcription_mode: (
        Literal[
            "legacy",
            "operator_channel",
            "dual_channel",
            "mono_diarization",
        ]
        | None
    ) = None
    speaker_attribution_status: str | None = None
    speaker_assignment_required: bool = False
    available_channels: list[int] = Field(default_factory=list)
    confidence_status: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    pipeline_version: str | None = None
    transcript_quality_summaries: list[TranscriptQualitySummaryResponse] = Field(
        default_factory=list
    )
    processing_history: list[dict]
