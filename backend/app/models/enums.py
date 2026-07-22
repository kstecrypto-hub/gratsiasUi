from __future__ import annotations

from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    WAITING_FOR_CONNECTION = "waiting_for_connection"
    CONNECTING = "connecting"
    FETCHING_CALLS = "fetching_calls"
    FETCHING_CALL_DETAILS = "fetching_call_details"
    FINDING_RECORDINGS = "finding_recordings"
    DOWNLOADING_RECORDINGS = "downloading_recordings"
    INSPECTING_AUDIO = "inspecting_audio"
    EXTRACTING_OPERATOR_AUDIO = "extracting_operator_audio"
    TRANSCRIBING = "transcribing"
    SEARCHING_KEYWORDS = "searching_keywords"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ItemStatus(StrEnum):
    QUEUED = "queued"
    WAITING_FOR_CONNECTION = "waiting_for_connection"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class RecordingStatus(StrEnum):
    DISCOVERED = "discovered"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    INSPECTED = "inspected"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class TranscriptStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SpeakerSource(StrEnum):
    YEASTAR_EXTENSION = "yeastar_extension"
    STEREO_CHANNEL = "stereo_channel"
    OPENAI_DIARIZATION = "openai_diarization"
    UNKNOWN = "unknown"
    MANUAL_OVERRIDE = "manual_override"


class MatchMethod(StrEnum):
    EXACT_PHRASE = "exact_phrase"
    WHOLE_WORD = "whole_word"
    VARIANT = "variant"
    ORDERED_TERMS = "ordered_terms"
    FUZZY = "fuzzy"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SyncType(StrEnum):
    OPERATORS = "operators"
    CALLS = "calls"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ParticipantRole(StrEnum):
    CALLER = "caller"
    CALLEE = "callee"
    ANSWERING_OPERATOR = "answering_operator"
    TRANSFERRED_OPERATOR = "transferred_operator"
    UNKNOWN = "unknown"


class Direction(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class YeastarConnectionStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    NOT_TESTED = "not_tested"
    CONNECTED = "connected"
    AUTH_REJECTED = "auth_rejected"
    TOKEN_REFRESH_FAILED = "token_refresh_failed"
    IP_NOT_ALLOWED = "ip_not_allowed"
    IP_BLOCKED = "ip_blocked"
    API_DISABLED = "api_disabled"
    PERMISSION_DENIED = "permission_denied"
    UNSUPPORTED_API_VERSION = "unsupported_api_version"
    UNSUPPORTED_FIRMWARE = "unsupported_firmware"
    NETWORK_UNAVAILABLE = "network_unavailable"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
