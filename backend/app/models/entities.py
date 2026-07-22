from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    MatchMethod,
    ParticipantRole,
    RecordingStatus,
    RunStatus,
    Severity,
    SpeakerSource,
    SyncType,
    TranscriptStatus,
    YeastarConnectionStatus,
)


def enum_column(enum_class: type, name: str) -> Enum:
    return Enum(
        enum_class,
        name=name,
        native_enum=False,
        validate_strings=True,
        create_constraint=True,
    )


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Operator(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "operators"
    __table_args__ = (
        Index(
            "uq_operators_active_extension_number",
            "extension_number",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
    )

    yeastar_extension_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    extension_number: Mapped[str] = mapped_column(String(64), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    mobile_number: Mapped[str | None] = mapped_column(String(64))
    presence_status: Mapped[str | None] = mapped_column(String(64))
    provider_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    last_synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    participants: Mapped[list[CallParticipant]] = relationship(back_populates="operator")


class Call(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "calls"
    __table_args__ = (
        Index("ix_calls_started_at_operator_results", "started_at", "processing_status"),
    )

    yeastar_uid: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    yeastar_id: Mapped[str | None] = mapped_column(String(255), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    caller_number: Mapped[str | None] = mapped_column(String(128))
    caller_name: Mapped[str | None] = mapped_column(String(255))
    callee_number: Mapped[str | None] = mapped_column(String(128))
    callee_name: Mapped[str | None] = mapped_column(String(255))
    direction: Mapped[Direction] = mapped_column(
        enum_column(Direction, "direction"), nullable=False, default=Direction.UNKNOWN, index=True
    )
    call_status: Mapped[str | None] = mapped_column(String(64), index=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    queue_name: Mapped[str | None] = mapped_column(String(255), index=True)
    has_recording: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    was_transferred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    processing_status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending", index=True)
    last_error_category: Mapped[str | None] = mapped_column(String(128))
    last_error_message: Mapped[str | None] = mapped_column(String(1000))
    provider_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    participants: Mapped[list[CallParticipant]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    legs: Mapped[list[CallLeg]] = relationship(back_populates="call", cascade="all, delete-orphan")
    recordings: Mapped[list[Recording]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    transcripts: Mapped[list[Transcript]] = relationship(back_populates="call")


class CallLeg(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "call_legs"
    __table_args__ = (
        UniqueConstraint("call_id", "yeastar_leg_id", name="uq_call_legs_call_provider_leg"),
        UniqueConstraint("call_id", "sequence_number", name="uq_call_legs_call_sequence"),
    )

    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    yeastar_leg_id: Mapped[str] = mapped_column(String(255), nullable=False)
    transaction_id: Mapped[str | None] = mapped_column(String(255), index=True)
    yeastar_cdr_id: Mapped[str | None] = mapped_column(String(255), index=True)
    provider_leg: Mapped[str | None] = mapped_column(String(128))
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    caller_extension_id: Mapped[str | None] = mapped_column(String(128))
    callee_extension_id: Mapped[str | None] = mapped_column(String(128))
    call_from: Mapped[str | None] = mapped_column(String(255))
    call_to: Mapped[str | None] = mapped_column(String(255))
    caller_number: Mapped[str | None] = mapped_column(String(128))
    callee_number: Mapped[str | None] = mapped_column(String(128))
    answered_by_extension_id: Mapped[str | None] = mapped_column(String(128), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ring_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    talk_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hold_duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    call_type: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str | None] = mapped_column(String(64))
    event_list: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    provider_recording_id: Mapped[str | None] = mapped_column(String(255))
    provider_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    call: Mapped[Call] = relationship(back_populates="legs")
    participants: Mapped[list[CallParticipant]] = relationship(back_populates="call_leg")


class CallParticipant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "call_participants"
    __table_args__ = (
        UniqueConstraint(
            "call_id", "operator_id", "call_leg_id", "role", name="uq_participant_call_operator_leg_role"
        ),
    )

    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    operator_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("operators.id", ondelete="RESTRICT"), index=True
    )
    call_leg_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("call_legs.id", ondelete="CASCADE"), index=True
    )
    provider_extension_id: Mapped[str | None] = mapped_column(String(128), index=True)
    provider_extension_number: Mapped[str | None] = mapped_column(String(64))
    role: Mapped[ParticipantRole] = mapped_column(
        enum_column(ParticipantRole, "participant_role"), nullable=False
    )
    operator_was_caller: Mapped[bool | None] = mapped_column(Boolean)
    operator_was_callee: Mapped[bool | None] = mapped_column(Boolean)
    answered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    participated_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    participated_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attribution_source: Mapped[SpeakerSource] = mapped_column(
        enum_column(SpeakerSource, "participant_attribution_source"),
        nullable=False,
        default=SpeakerSource.UNKNOWN,
    )
    attribution_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))

    call: Mapped[Call] = relationship(back_populates="participants")
    operator: Mapped[Operator | None] = relationship(back_populates="participants")
    call_leg: Mapped[CallLeg | None] = relationship(back_populates="participants")


class Recording(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "recordings"
    __table_args__ = (
        UniqueConstraint("call_id", "yeastar_recording_id", name="uq_recording_call_provider_id"),
        Index("ix_recordings_sha256_checksum", "sha256_checksum"),
    )

    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    yeastar_recording_id: Mapped[str] = mapped_column(String(255), nullable=False)
    yeastar_uid: Mapped[str | None] = mapped_column(String(255), index=True)
    provider_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    call_from: Mapped[str | None] = mapped_column(String(255))
    call_to: Mapped[str | None] = mapped_column(String(255))
    call_from_number: Mapped[str | None] = mapped_column(String(128))
    call_to_number: Mapped[str | None] = mapped_column(String(128))
    call_type: Mapped[str | None] = mapped_column(String(64))
    archive_status: Mapped[str | None] = mapped_column(String(64))
    yeastar_file_name: Mapped[str | None] = mapped_column(String(512))
    storage_key: Mapped[str | None] = mapped_column(String(512), unique=True)
    mime_type: Mapped[str | None] = mapped_column(String(128))
    file_extension: Mapped[str | None] = mapped_column(String(16))
    size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    channel_count: Mapped[int | None] = mapped_column(Integer)
    codec_name: Mapped[str | None] = mapped_column(String(64))
    sample_rate_hz: Mapped[int | None] = mapped_column(Integer)
    bit_rate_bps: Mapped[int | None] = mapped_column(Integer)
    sha256_checksum: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[RecordingStatus] = mapped_column(
        enum_column(RecordingStatus, "recording_status"),
        nullable=False,
        default=RecordingStatus.DISCOVERED,
        index=True,
    )
    channel_assignment: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    provider_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    inspected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_category: Mapped[str | None] = mapped_column(String(128))
    last_error_message: Mapped[str | None] = mapped_column(String(1000))

    call: Mapped[Call] = relationship(back_populates="recordings")
    transcripts: Mapped[list[Transcript]] = relationship(back_populates="recording")


class IntegrationStatus(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "integration_status"

    provider: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[YeastarConnectionStatus] = mapped_column(
        enum_column(YeastarConnectionStatus, "yeastar_connection_status"),
        nullable=False,
        default=YeastarConnectionStatus.NOT_TESTED,
        index=True,
    )
    configuration_fingerprint: Mapped[str | None] = mapped_column(String(64))
    configured_date_format: Mapped[str | None] = mapped_column(String(128))
    device_name: Mapped[str | None] = mapped_column(String(255))
    model_name: Mapped[str | None] = mapped_column(String(255))
    firmware_version: Mapped[str | None] = mapped_column(String(128))
    system_time: Mapped[str | None] = mapped_column(String(128))
    system_date_format: Mapped[str | None] = mapped_column(String(128))
    system_time_format: Mapped[str | None] = mapped_column(String(128))
    provider_timestamp: Mapped[int | None] = mapped_column(BigInteger)
    capabilities_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_successful_connection_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_error_category: Mapped[str | None] = mapped_column(String(128))
    last_error_reference: Mapped[str | None] = mapped_column(String(128))


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"
    __table_args__ = (
        CheckConstraint("date_to > date_from", name="valid_date_range"),
        CheckConstraint("progress_percent >= 0 AND progress_percent <= 100", name="progress_range"),
        # The database, rather than a worker lease, owns the one-analysis-at-a-time
        # invariant.  The enum stores member names (uppercase) in both PostgreSQL
        # and SQLite.
        Index(
            "uq_processing_jobs_one_active",
            text("(1)"),
            unique=True,
            postgresql_where=text(
                "status NOT IN ('COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED', 'CANCELLED')"
            ),
            sqlite_where=text(
                "status NOT IN ('COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED', 'CANCELLED')"
            ),
        ),
    )

    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    requested_by_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    status: Mapped[JobStatus] = mapped_column(
        enum_column(JobStatus, "processing_job_status"),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True,
    )
    date_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    date_to: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    direction: Mapped[str | None] = mapped_column(String(32))
    queue_name: Mapped[str | None] = mapped_column(String(255))
    call_status_filter: Mapped[str | None] = mapped_column(String(64))
    recording_available: Mapped[bool | None] = mapped_column(Boolean)
    include_all_speakers: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    selected_operator_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    selected_category_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    request_filters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_stage: Mapped[str] = mapped_column(String(128), nullable=False, default="queued")
    calls_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recordings_found: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    calls_completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    calls_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancellation_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_category: Mapped[str | None] = mapped_column(String(128))
    last_error_message: Mapped[str | None] = mapped_column(String(1000))

    items: Mapped[list[ProcessingJobItem]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class ProcessingJobItem(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processing_job_items"
    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "call_id",
            "operator_id",
            "recording_id",
            name="uq_job_item_call_operator_recording",
        ),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    operator_id: Mapped[UUID] = mapped_column(
        ForeignKey("operators.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    recording_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("recordings.id", ondelete="SET NULL"), index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    status: Mapped[ItemStatus] = mapped_column(
        enum_column(ItemStatus, "processing_job_item_status"),
        nullable=False,
        default=ItemStatus.QUEUED,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(128), nullable=False, default="queued")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    celery_task_id: Mapped[str | None] = mapped_column(String(255))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_category: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))

    job: Mapped[ProcessingJob] = relationship(back_populates="items")
    call: Mapped[Call] = relationship()
    operator: Mapped[Operator] = relationship()
    recording: Mapped[Recording | None] = relationship()


class Transcript(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "transcripts"

    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    recording_id: Mapped[UUID] = mapped_column(
        ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    operator_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("operators.id", ondelete="RESTRICT"), index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(192), nullable=False, unique=True)
    status: Mapped[TranscriptStatus] = mapped_column(
        enum_column(TranscriptStatus, "transcript_status"),
        nullable=False,
        default=TranscriptStatus.QUEUED,
        index=True,
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="el")
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    processing_duration_seconds: Mapped[float | None] = mapped_column(Float)
    audio_duration_seconds: Mapped[float | None] = mapped_column(Float)
    api_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    error_category: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_diarized: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    original_text: Mapped[str | None] = mapped_column(Text)
    normalized_text: Mapped[str | None] = mapped_column(Text)

    call: Mapped[Call] = relationship(back_populates="transcripts")
    recording: Mapped[Recording] = relationship(back_populates="transcripts")
    operator: Mapped[Operator | None] = relationship()
    segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="transcript", cascade="all, delete-orphan", order_by="TranscriptSegment.sequence_number"
    )


class TranscriptSegment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "transcript_segments"
    __table_args__ = (
        UniqueConstraint("transcript_id", "sequence_number", name="uq_segment_transcript_sequence"),
        CheckConstraint("end_seconds >= start_seconds", name="segment_time_order"),
    )

    transcript_id: Mapped[UUID] = mapped_column(
        ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    call_leg_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("call_legs.id", ondelete="SET NULL"), index=True
    )
    operator_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("operators.id", ondelete="RESTRICT"), index=True
    )
    speaker_label: Mapped[str] = mapped_column(String(128), nullable=False)
    speaker_source: Mapped[SpeakerSource] = mapped_column(
        enum_column(SpeakerSource, "speaker_source"), nullable=False, index=True
    )
    start_seconds: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    end_seconds: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(6, 5))
    transcription_model: Mapped[str] = mapped_column(String(128), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)

    transcript: Mapped[Transcript] = relationship(back_populates="segments")
    matches: Mapped[list[KeywordMatch]] = relationship(back_populates="segment")


class KeywordCategory(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "keyword_categories"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    keywords: Mapped[list[Keyword]] = relationship(
        back_populates="category", cascade="all, delete-orphan"
    )


class Keyword(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "keywords"
    __table_args__ = (
        UniqueConstraint("category_id", "normalized_phrase", name="uq_keyword_category_phrase"),
        CheckConstraint(
            "fuzzy_threshold >= 0 AND fuzzy_threshold <= 100", name="fuzzy_threshold_range"
        ),
    )

    category_id: Mapped[UUID] = mapped_column(
        ForeignKey("keyword_categories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    canonical_phrase: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_phrase: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    accent_insensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    whole_word: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    exact_phrase: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    fuzzy_match: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fuzzy_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=90)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    severity: Mapped[Severity] = mapped_column(
        enum_column(Severity, "keyword_severity"), nullable=False, default=Severity.MEDIUM
    )
    notes: Mapped[str | None] = mapped_column(Text)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    category: Mapped[KeywordCategory] = relationship(back_populates="keywords")
    variants: Mapped[list[KeywordVariant]] = relationship(
        back_populates="keyword", cascade="all, delete-orphan"
    )


class KeywordVariant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "keyword_variants"
    __table_args__ = (
        UniqueConstraint("keyword_id", "normalized_phrase", name="uq_variant_keyword_phrase"),
    )

    keyword_id: Mapped[UUID] = mapped_column(
        ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False, index=True
    )
    phrase: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_phrase: Mapped[str] = mapped_column(String(500), nullable=False)

    keyword: Mapped[Keyword] = relationship(back_populates="variants")


class KeywordMatch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "keyword_matches"
    __table_args__ = (
        UniqueConstraint(
            "keyword_id",
            "transcript_segment_id",
            "start_seconds",
            "normalized_match",
            name="uq_match_keyword_segment_position",
        ),
        CheckConstraint("end_seconds >= start_seconds", name="match_time_order"),
    )

    keyword_id: Mapped[UUID] = mapped_column(
        ForeignKey("keywords.id", ondelete="CASCADE"), nullable=False, index=True
    )
    operator_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("operators.id", ondelete="RESTRICT"), index=True
    )
    call_id: Mapped[UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False, index=True
    )
    transcript_segment_id: Mapped[UUID] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    original_matched_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_match: Mapped[str] = mapped_column(Text, nullable=False)
    context_before: Mapped[str] = mapped_column(Text, nullable=False, default="")
    context_after: Mapped[str] = mapped_column(Text, nullable=False, default="")
    start_seconds: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    end_seconds: Mapped[Decimal] = mapped_column(Numeric(12, 3), nullable=False)
    match_method: Mapped[MatchMethod] = mapped_column(
        enum_column(MatchMethod, "keyword_match_method"), nullable=False
    )
    match_score: Mapped[Decimal] = mapped_column(Numeric(6, 3), nullable=False)

    keyword: Mapped[Keyword] = relationship()
    segment: Mapped[TranscriptSegment] = relationship(back_populates="matches")
    call: Mapped[Call] = relationship()
    operator: Mapped[Operator | None] = relationship()


class SyncRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "sync_runs"

    sync_type: Mapped[SyncType] = mapped_column(
        enum_column(SyncType, "sync_type"), nullable=False, index=True
    )
    status: Mapped[RunStatus] = mapped_column(
        enum_column(RunStatus, "sync_run_status"), nullable=False, index=True
    )
    idempotency_key: Mapped[str] = mapped_column(String(160), nullable=False, unique=True)
    processing_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("processing_jobs.id", ondelete="SET NULL"), index=True
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    date_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    date_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    records_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_created: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_category: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(String(1000))


class ApplicationSetting(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "application_settings"

    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)
    updated_by_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )


class AuditLog(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_created_action", "created_at", "action"),)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    action: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(128))
    resource_id: Mapped[str | None] = mapped_column(String(255))
    outcome: Mapped[str] = mapped_column(String(32), nullable=False, default="success")
    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(512))
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
