from __future__ import annotations

from datetime import datetime
from decimal import Decimal
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
    matches: list[MatchResponse]


class CallDetailResponse(APIModel):
    id: UUID
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
    processing_history: list[dict]
