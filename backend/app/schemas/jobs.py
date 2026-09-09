from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, model_validator

from app.models.enums import JobStatus
from app.schemas.common import APIModel


class JobCreate(APIModel):
    date_from: datetime
    date_to: datetime
    operator_ids: list[UUID] = Field(min_length=1, max_length=500)
    direction: str | None = Field(default=None, max_length=32)
    queue: str | None = Field(default=None, max_length=255)
    call_status: str | None = Field(default=None, max_length=64)
    recording_available: bool | None = None
    keyword_category_ids: list[UUID] = Field(default_factory=list, max_length=500)
    include_all_speakers: bool = False
    idempotency_key: str | None = Field(default=None, min_length=16, max_length=128)

    @model_validator(mode="after")
    def check_dates(self) -> "JobCreate":
        if self.date_to <= self.date_from:
            raise ValueError("date_to must be after date_from")
        if (self.date_to - self.date_from).days > 366:
            raise ValueError("Date range cannot exceed 366 days")
        return self


class JobSummary(APIModel):
    id: UUID
    is_current: bool = False
    status: JobStatus
    date_from: datetime
    date_to: datetime
    operator_ids: list[str]
    progress_percent: int
    current_stage: str
    calls_found: int
    recordings_found: int
    calls_completed: int
    calls_failed: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class JobItemResponse(APIModel):
    id: UUID
    call_id: UUID
    operator_id: UUID
    recording_id: UUID | None
    requested_pipeline_version: str | None = None
    result_transcript_id: UUID | None = None
    status: str
    stage: str
    attempt_count: int
    error_category: str | None
    error_message: str | None


class JobDetail(JobSummary):
    direction: str | None
    queue: str | None
    call_status_filter: str | None
    recording_available: bool | None
    include_all_speakers: bool
    cancellation_requested: bool
    attempt_count: int
    last_error_category: str | None
    last_error_message: str | None
    items: list[JobItemResponse]
