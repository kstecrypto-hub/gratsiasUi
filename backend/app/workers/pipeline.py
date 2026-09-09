from __future__ import annotations

import hashlib
import json
import logging
import secrets
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from redis.exceptions import RedisError
from sqlalchemy import Select, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import Settings, get_settings
from app.core.logging import redact_value
from app.core.redis import get_redis
from app.core.time import utc_now
from app.database.session import AsyncSessionFactory
from app.models import (
    Call,
    CallLeg,
    CallParticipant,
    AuditLog,
    IntegrationStatus,
    Keyword,
    KeywordMatch,
    Operator,
    ProcessingJob,
    ProcessingJobItem,
    Recording,
    SyncRun,
    Transcript,
    TranscriptionAttempt,
    TranscriptSegment,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    RecordingStatus,
    RunStatus,
    SpeakerSource,
    SpeakerAttributionStatus,
    SyncType,
    TranscriptionMode,
    TranscriptStatus,
)
from app.services.application_settings import load_application_settings
from app.services.audio import AudioInfo, AudioProcessor
from app.services.audio.errors import AudioError
from app.services.audio.quality import (
    LIGHT_NORMALIZATION_PROFILE,
    RAW_LOSSLESS_AUDIO_VARIANT,
)
from app.services.keyword_matching import KeywordDefinition, match_text, normalize_greek
from app.services.transcription import OpenAITranscriptionClient
from app.services.transcription.client import (
    TRANSCRIPTION_LOGPROB_CONTRACT_VERSION,
    V2_TRANSCRIPTION_TEMPERATURE,
    TranscriptionCancelledError,
    TranscriptionConfigurationError,
    TranscriptionError,
    TranscriptionResult,
    model_supports_transcription_logprobs,
)
from app.services.transcription.confidence import DEFAULT_CONFIDENCE_POLICY
from app.services.transcription.configuration_store import (
    OpenAIConfigurationStateError,
    load_effective_openai_settings,
)
from app.services.transcription.mono import DEFAULT_MONO_REFINEMENT_POLICY
from app.services.transcription.orchestrator import (
    PartialTranscriptionCancelledError,
    PartialTranscriptionError,
    TranscriptionContext,
    TranscriptionOrchestrator,
    default_speech_segmentation_identity,
)
from app.services.transcription.planning import (
    LegacyAudioPlanner,
    TopologyAudioPlanner,
)
from app.services.transcription.prompt import (
    PRIORITY_COMPANY,
    PRIORITY_CURRENT_CALL,
    PRIORITY_CURRENT_PARTY,
    PRIORITY_GENERAL,
    PRIORITY_QUEUE,
    PRIORITY_SELECTED_KEYWORD,
    PRIORITY_SELECTED_OPERATOR,
    RankedVocabularyTerm,
    TRACK_ROLE_CALLEE,
    TRACK_ROLE_CALLER,
    TRACK_ROLE_OPERATOR,
    VOCABULARY_SOURCE_COMPANY,
    VOCABULARY_SOURCE_CURRENT_CALL,
    VOCABULARY_SOURCE_CURRENT_PARTY,
    VOCABULARY_SOURCE_GENERAL,
    VOCABULARY_SOURCE_QUEUE,
    VOCABULARY_SOURCE_SELECTED_KEYWORD,
    VOCABULARY_SOURCE_SELECTED_OPERATOR,
    V2PromptManifest,
    build_vocabulary_prompt,
)
from app.services.transcription.types import (
    AudioPlan,
    OrchestratedTranscriptionResult,
    TranscriptionAttemptEvidence,
)
from app.services.yeastar import YeastarClient
from app.services.yeastar.configuration_store import load_effective_yeastar_settings
from app.services.yeastar.cdr import CDRSummary
from app.services.yeastar.errors import (
    YeastarApiDisabledError,
    YeastarAuthenticationError,
    YeastarCircuitOpenError,
    YeastarConfigurationError,
    YeastarConfigurationStateError,
    YeastarIpBlockedError,
    YeastarIpForbiddenError,
    YeastarPermissionError,
    YeastarRecordingDownloadLimitError,
    YeastarTokenRefreshError,
    YeastarUnsupportedVersionError,
    YeastarError,
)
from app.services.yeastar.interpretation import (
    InterpretedParticipant,
    interpret_call_legs,
    safe_caller_callee_channels,
    safe_operator_channel,
)
from app.services.yeastar.integration import (
    clear_shared_token_and_require_test,
    reconcile_configuration_fingerprint,
    runtime_cdr_api_version,
)


logger = logging.getLogger(__name__)
FINAL_ITEM_STATES = {
    ItemStatus.COMPLETED,
    ItemStatus.FAILED,
    ItemStatus.SKIPPED,
    ItemStatus.CANCELLED,
}
FINAL_JOB_STATES = {
    JobStatus.COMPLETED,
    JobStatus.COMPLETED_WITH_ERRORS,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}
WORKER_STALE_AFTER = timedelta(hours=2)
WORKER_LEASE_SECONDS = int(WORKER_STALE_AFTER.total_seconds())
WORKER_LEASE_REFRESH_SECONDS = 60 * 60
CONNECTION_PAUSE_ERRORS = (
    YeastarConfigurationError,
    YeastarAuthenticationError,
    YeastarTokenRefreshError,
    YeastarCircuitOpenError,
    YeastarIpBlockedError,
    YeastarIpForbiddenError,
    YeastarApiDisabledError,
    YeastarPermissionError,
    YeastarUnsupportedVersionError,
)

LEGACY_PIPELINE_VERSION = "legacy-v1"
PIPELINE_V2_VERSION = "pipeline-v2"
SUPPORTED_PIPELINE_VERSIONS = frozenset({LEGACY_PIPELINE_VERSION, PIPELINE_V2_VERSION})
LEGACY_PREPROCESSING_PROFILE = "legacy-current"
PIPELINE_V2_PREPROCESSING_PROFILE = "topology-v2-pcm16k-mono-tracks-v1"
PIPELINE_V2_STANDARD_PREPROCESSING_PROFILE = "topology-v2-pcm16k-standard-logprob-retry-v1"
PIPELINE_V2_MONO_PREPROCESSING_PROFILE = "topology-v2-pcm16k-mono-two-pass-refinement-v1"
LEGACY_DIARIZED_PROMPT_TEMPLATE_VERSION = "legacy-diarized-no-prompt-v1"
LEGACY_ISOLATED_PROMPT_TEMPLATE_VERSION = "legacy-isolated-vocabulary-v1"
GENERAL_DEALERSHIP_VOCABULARY = (
    "αντιπροσωπεία",
    "συνεργείο",
    "service",
    "ραντεβού",
    "όχημα",
    "ανταλλακτικά",
    "εγγύηση",
)


def _legacy_pipeline_config_hash(prompt_template_version: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "audio_flow": "existing-channel-selection",
                "diarized_chunk_seconds": 480,
                "isolated_chunk_seconds": 15,
                "preprocessing_profile": LEGACY_PREPROCESSING_PROFILE,
                "prompt_template_version": prompt_template_version,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


LEGACY_DIARIZED_PIPELINE_CONFIG_HASH = _legacy_pipeline_config_hash(
    LEGACY_DIARIZED_PROMPT_TEMPLATE_VERSION
)
LEGACY_ISOLATED_PIPELINE_CONFIG_HASH = _legacy_pipeline_config_hash(
    LEGACY_ISOLATED_PROMPT_TEMPLATE_VERSION
)


def _legacy_runtime_pipeline_config_hash(
    *,
    diarized: bool,
    channel_index: int | None,
    max_upload_bytes: int,
    prompt_template_version: str,
    preprocessing_profile: str = LEGACY_PREPROCESSING_PROFILE,
    diarized_chunk_seconds: int = 480,
    isolated_chunk_seconds: int = 15,
    isolated_response_format: str = "json",
    diarized_response_format: str = "diarized_json",
    diarized_chunking_strategy: str = "auto",
) -> str:
    """Identify every legacy execution choice that can change provider work."""

    canonical = {
        "audio_preparation": {
            "channel_index": channel_index,
            "codec": "pcm_s16le",
            "sample_rate_hz": 16000,
            "track": "mono-fallback" if diarized else "operator-channel",
        },
        "preprocessing_profile": preprocessing_profile,
        "prompt_template_version": prompt_template_version,
        "request": {
            "chunking_strategy": diarized_chunking_strategy if diarized else None,
            "response_format": (diarized_response_format if diarized else isolated_response_format),
        },
        "segmentation": {
            "chunk_seconds": (diarized_chunk_seconds if diarized else isolated_chunk_seconds),
            "max_upload_bytes": max_upload_bytes,
            "single_upload_when_within_limit": diarized,
            "strategy": "legacy-fixed",
        },
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _pipeline_v2_runtime_config_hash(
    *,
    plan: AudioPlan,
    max_upload_bytes: int,
    prompt_template_version: str,
    segmentation_identity: dict[str, object] | None = None,
    prompt_identity: str | None = None,
    vocabulary_hash: str | None = None,
    standard_model: str | None = None,
    diarization_model: str | None = None,
    mono_refinement_identity: dict[str, object] | None = None,
) -> str:
    """Identify every topology-phase choice that changes work or attribution."""

    standard_v2 = plan.mode in {"operator_channel", "dual_channel"}
    mono_v2 = plan.mode == "mono_diarization"
    prompted_standard_v2 = standard_v2 or mono_v2
    resolved_standard_model = standard_model or "gpt-4o-transcribe"
    logprobs_supported = prompted_standard_v2 and model_supports_transcription_logprobs(
        resolved_standard_model
    )
    canonical = {
        "audio_plan": {
            "attribution_status": plan.attribution_status,
            "callee_channel": plan.callee_channel,
            "caller_channel": plan.caller_channel,
            "mode": plan.mode,
            "operator_channel": plan.operator_channel,
            "reason": plan.reason,
            "stereo_separated": plan.stereo_separated,
            "tracks": [
                {
                    "audio_variant": track.audio_variant,
                    "attribution_status": track.attribution_status,
                    "channel_index": track.channel_index,
                    "diarized": track.diarized,
                    "speaker_label_policy": (
                        "operator_display_name"
                        if track.operator_id is not None
                        else track.speaker_label
                    ),
                    "speaker_source": track.speaker_source,
                    "track_id": track.track_id,
                }
                for track in plan.tracks
            ],
        },
        "planner": "pbx-topology-v1",
        "preprocessing_profile": (
            PIPELINE_V2_STANDARD_PREPROCESSING_PROFILE
            if standard_v2
            else (
                PIPELINE_V2_MONO_PREPROCESSING_PROFILE
                if mono_v2
                else PIPELINE_V2_PREPROCESSING_PROFILE
            )
        ),
        "prompt_template_version": prompt_template_version,
        "request": {
            "chunking_strategy": "auto" if plan.mode == "mono_diarization" else None,
            "include": ["logprobs"] if logprobs_supported else None,
            "logprob_contract": (TRANSCRIPTION_LOGPROB_CONTRACT_VERSION if standard_v2 else None),
            "logprobs_supported": logprobs_supported if standard_v2 else None,
            "sdk_transport_max_retries": 0 if standard_v2 else None,
            "response_format": ("diarized_json" if plan.mode == "mono_diarization" else "json"),
            "temperature": (V2_TRANSCRIPTION_TEMPERATURE if logprobs_supported else None),
        },
        "segmentation": (
            (
                segmentation_identity
                or default_speech_segmentation_identity(max_upload_bytes=max_upload_bytes)
            )
            if plan.mode in {"operator_channel", "dual_channel"}
            else {
                "chunk_seconds": 480,
                "max_upload_bytes": max_upload_bytes,
                "single_upload_when_within_limit": True,
                "strategy": "legacy-fixed",
            }
        ),
    }
    if mono_v2:
        canonical["request"] = {
            "diarization": {
                "chunking_strategy": "auto",
                "complete_prepared_audio": True,
                "language": "el",
                "model": diarization_model or "gpt-4o-transcribe-diarize",
                "response_format": "diarized_json",
                "sdk_transport_max_retries": 0,
                "sends_prompt": False,
                "sends_speaker_references": False,
            },
            "refinement": {
                "include": ["logprobs"] if logprobs_supported else None,
                "language": "el",
                "logprob_contract": TRANSCRIPTION_LOGPROB_CONTRACT_VERSION,
                "logprobs_supported": logprobs_supported,
                "model": resolved_standard_model,
                "response_format": "json",
                "sdk_transport_max_retries": 0,
                "temperature": (V2_TRANSCRIPTION_TEMPERATURE if logprobs_supported else None),
            },
        }
        canonical["segmentation"] = {
            "complete_prepared_audio": True,
            "max_upload_bytes": max_upload_bytes,
            "strategy": "provider-diarization-auto-v1",
        }
        canonical["mono_refinement"] = (
            mono_refinement_identity or DEFAULT_MONO_REFINEMENT_POLICY.identity()
        )
    if prompted_standard_v2:
        policy = DEFAULT_CONFIDENCE_POLICY
        canonical["confidence_retry"] = {
            "audio_variants": {
                "normalized": LIGHT_NORMALIZATION_PROFILE.identity(),
                "raw": RAW_LOSSLESS_AUDIO_VARIANT,
            },
            "effective_mean_tie_tolerance": policy.effective_mean_tie_tolerance,
            "low_logprob_ratio_threshold": policy.low_logprob_ratio_threshold,
            "max_attempts_per_chunk": policy.max_attempts_per_chunk,
            "mean_logprob_low_threshold": policy.mean_logprob_low_threshold,
            "policy_version": policy.version,
            "selection": "valid-metrics-then-higher-mean-raw-on-tie-v1",
            "token_logprob_low_threshold": policy.token_logprob_low_threshold,
        }
    if prompt_identity is not None or vocabulary_hash is not None:
        canonical["prompt_policy"] = {
            "prompt_identity": prompt_identity,
            "vocabulary_hash": vocabulary_hash,
        }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _segmentation_identity_for_plan(
    orchestrator: TranscriptionOrchestrator,
    plan: AudioPlan,
    *,
    max_upload_bytes: int,
) -> dict[str, object] | None:
    if plan.mode not in {"operator_channel", "dual_channel"}:
        return None
    identity_method = getattr(orchestrator, "segmentation_identity", None)
    if callable(identity_method):
        return identity_method(
            plan.mode,
            max_upload_bytes=max_upload_bytes,
        )
    # Older injected test orchestrators predate Prompt 5. Production instances
    # expose the executing adapter identity and do not use this compatibility path.
    return default_speech_segmentation_identity(max_upload_bytes=max_upload_bytes)


def _effective_pipeline_version(
    item: ProcessingJobItem,
    settings: Settings | None = None,
) -> str:
    """Resolve the pipeline version with configuration-owned rollback.

    Ordinary work uses ``TRANSCRIPTION_PIPELINE_DEFAULT``. V2 stays gated by
    ``TRANSCRIPTION_PIPELINE_V2_ENABLED``; disabling it blocks new V2 work but
    never deletes or rewrites existing V2 rows.
    """

    requested = item.requested_pipeline_version
    if requested is not None:
        if requested not in SUPPORTED_PIPELINE_VERSIONS:
            raise ReprocessStateError(
                f"Pipeline version {requested!r} is not available."
            )
        if requested == PIPELINE_V2_VERSION and settings is not None and not (
            getattr(settings, "TRANSCRIPTION_PIPELINE_V2_ENABLED", False)
        ):
            raise ReprocessStateError("Pipeline V2 is disabled.")
        return requested
    default = (
        getattr(settings, "TRANSCRIPTION_PIPELINE_DEFAULT", LEGACY_PIPELINE_VERSION)
        if settings is not None
        else LEGACY_PIPELINE_VERSION
    )
    if default == PIPELINE_V2_VERSION and settings is not None and not (
        getattr(settings, "TRANSCRIPTION_PIPELINE_V2_ENABLED", False)
    ):
        return LEGACY_PIPELINE_VERSION
    return default


def _transcript_has_attribution_warning(transcript: Transcript) -> bool:
    if transcript.transcription_mode in {
        TranscriptionMode.DUAL_CHANNEL,
        TranscriptionMode.MONO_DIARIZATION,
    }:
        return True
    return bool(
        transcript.transcription_mode in {None, TranscriptionMode.LEGACY} and transcript.is_diarized
    )


def _attribution_warning_message(transcript: Transcript) -> str | None:
    if not _transcript_has_attribution_warning(transcript):
        return None
    if transcript.transcription_mode == TranscriptionMode.DUAL_CHANNEL:
        return "Channels were preserved, but the operator could not be identified safely."
    return "Speakers were separated but the operator could not be identified safely."


def _plan_with_orchestrator(
    orchestrator: object,
    planner: LegacyAudioPlanner | TopologyAudioPlanner,
    *,
    source_path: Path,
    audio_info: AudioInfo,
    context: TranscriptionContext,
) -> AudioPlan:
    """Plan once before identity creation; keep older injected test adapters usable."""

    plan_method = getattr(orchestrator, "plan", None)
    if callable(plan_method):
        return plan_method(
            source_path=source_path,
            audio_info=audio_info,
            context=context,
        )
    return planner.plan(
        source_path=source_path,
        audio_info=audio_info,
        diarized=context.diarized,
        channel_index=context.channel_index,
        operator_id=context.operator_id,
        attribution_status=context.attribution_status,
        audio_variant=context.audio_variant,
        stereo_separated=context.stereo_separated,
        operator_channel=context.operator_channel,
        caller_channel=context.caller_channel,
        callee_channel=context.callee_channel,
        operator_display_name=context.operator_display_name,
    )


class ProcessingBusyError(Exception):
    pass


class ProcessingLeaseLostError(ProcessingBusyError, TranscriptionError):
    """A stale worker lost ownership and must exit without mutating shared state."""

    def __init__(self) -> None:
        TranscriptionError.__init__(
            self,
            "Processing lock became unavailable.",
            "processing_lock",
        )


@dataclass(frozen=True)
class DiscoveryResult:
    """Work that can be dispatched after one safe discovery pass.

    An item without a recording must never be handed to ``process_item``.
    When the phone system has multiple recordings but has not yet supplied a
    deterministic relation to a call leg, discovery asks the task queue to
    revisit the call rather than marking it as failed or guessing.
    """

    item_ids: list[UUID]
    recording_assignment_pending: bool = False


class ProcessingCancelledError(Exception):
    category = "cancelled"


class StaleProcessingTaskError(Exception):
    """A delayed task attempted to mutate a job that is already final."""


class ReprocessStateError(Exception):
    """A completed-call replacement no longer has a safe source transcript."""

    category = "reprocess_state"


def _worker_timestamp_is_recent(value: datetime | None) -> bool:
    if value is None:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    else:
        value = value.astimezone(UTC)
    return value > utc_now() - WORKER_STALE_AFTER


async def _refresh_processing_leases(
    recording_lock: Any | None,
    transcription_slot: Any | None,
) -> None:
    for lease in (recording_lock, transcription_slot):
        await _refresh_worker_lease(lease)


async def _refresh_available_processing_leases(
    recording_lock: Any | None,
    transcription_slot: Any | None,
) -> None:
    """Fence failure/cleanup paths once the recording lease has been acquired."""

    if recording_lock is None:
        return
    await _refresh_worker_lease(recording_lock)
    if transcription_slot is not None:
        await _refresh_worker_lease(transcription_slot)


async def _refresh_worker_lease(lease: Any | None) -> None:
    if lease is None:
        raise ProcessingLeaseLostError
    try:
        if not await lease.owned():
            raise ProcessingLeaseLostError
        extended = await lease.extend(
            WORKER_LEASE_REFRESH_SECONDS,
            replace_ttl=True,
        )
        if extended is False:
            raise ProcessingLeaseLostError
    except ProcessingLeaseLostError:
        raise
    except Exception as exc:
        raise ProcessingLeaseLostError from exc


def _require_processable_job(job: ProcessingJob) -> None:
    """Make cancellation/finalization authoritative over delayed worker work."""
    if job.cancellation_requested or job.status == JobStatus.CANCELLED:
        raise ProcessingCancelledError("Analysis was cancelled.")
    if job.status in FINAL_JOB_STATES:
        raise StaleProcessingTaskError("Analysis is already finished.")


async def _stop_stale_discovery(session: AsyncSession, job_id: UUID) -> bool:
    """Settle cancelled discovery, or identify a task whose job is already final."""
    job = await session.scalar(
        select(ProcessingJob).where(ProcessingJob.id == job_id).with_for_update()
    )
    if job is None:
        return True
    cancelled = job.cancellation_requested or job.status == JobStatus.CANCELLED
    if job.status in FINAL_JOB_STATES and not cancelled:
        return True
    if not cancelled:
        return False
    if job.status not in FINAL_JOB_STATES:
        job.status = JobStatus.CANCELLED
        job.current_stage = "Cancelled"
        job.completed_at = utc_now()
    run = await session.scalar(
        select(SyncRun)
        .where(
            SyncRun.processing_job_id == job_id,
            SyncRun.status == RunStatus.RUNNING,
        )
        .order_by(SyncRun.created_at.desc())
    )
    if run is not None:
        run.status = RunStatus.CANCELLED
        run.completed_at = utc_now()
    await session.commit()
    return True


async def _settle_stale_item(
    session: AsyncSession,
    item: ProcessingJobItem,
    transcript_id: UUID | None,
    partial_attempts: tuple[TranscriptionAttemptEvidence, ...] = (),
    partial_usage: Mapping[str, Any] | None = None,
    partial_quality_summary: Mapping[str, Any] | None = None,
) -> Literal["active", "cancelled", "terminal"]:
    """Keep late item exceptions from overwriting cancellation or final history."""
    if item.recording_id is not None:
        await _lock_recording_rows(session, [item.recording_id])
    job = await session.scalar(
        select(ProcessingJob).where(ProcessingJob.id == item.job_id).with_for_update()
    )
    if job is None:
        return "terminal"
    await session.refresh(item, with_for_update=True)
    cancelled = (
        job.cancellation_requested
        or job.status == JobStatus.CANCELLED
        or item.status == ItemStatus.CANCELLED
    )
    if job.status in FINAL_JOB_STATES and not cancelled:
        return "terminal"
    if item.status in FINAL_ITEM_STATES and not cancelled:
        return "terminal"
    if not cancelled:
        return "active"
    if item.status not in {ItemStatus.COMPLETED, ItemStatus.SKIPPED, ItemStatus.CANCELLED}:
        item.status = ItemStatus.CANCELLED
        item.stage = "cancelled"
        item.completed_at = utc_now()
        item.error_category = None
        item.error_message = None
    if transcript_id:
        transcript = await session.get(Transcript, transcript_id)
        if transcript and _should_mark_transcript_failed(job, item, transcript):
            await _persist_partial_attempt_evidence(
                session,
                transcript,
                partial_attempts,
                api_usage=partial_usage,
                quality_summary=partial_quality_summary,
            )
            transcript.status = TranscriptStatus.FAILED
            transcript.error_category = "cancelled"
            transcript.error_message = "Transcription was cancelled."
    await session.commit()
    return "cancelled"


def transcript_idempotency_key(
    recording_id: UUID,
    operator_id: UUID | None,
    model: str,
    checksum: str,
    diarized: bool,
    language: str = "el",
    prompt_version: str | None = None,
    *,
    pipeline_version: str | None = None,
    pipeline_config_hash: str | None = None,
    transcription_mode: TranscriptionMode | str | None = None,
    prompt_template_version: str | None = None,
    vocabulary_hash: str | None = None,
    supersedes_transcript_id: UUID | None = None,
) -> str:
    # Preserve the exact pre-V2 identity for callers that have not opted in to
    # versioned pipeline metadata.  Production workers pass explicit stable
    # legacy values below, while integrations using the old helper signature do
    # not silently change their persisted keys.
    if pipeline_version is None:
        identity = (
            f"{recording_id}:{operator_id or 'unknown'}:{model}:{language}:"
            f"{prompt_version or 'no-prompt'}:{checksum}:{int(diarized)}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    mode = (
        transcription_mode.value
        if isinstance(transcription_mode, TranscriptionMode)
        else transcription_mode
    )
    identity = {
        "diarized": diarized,
        "language": language,
        "model": model,
        "operator_id": str(operator_id) if operator_id is not None else None,
        "pipeline_config_hash": pipeline_config_hash,
        "pipeline_version": pipeline_version,
        "prompt_template_version": prompt_template_version,
        "prompt_version": prompt_version,
        "recording_checksum": checksum,
        "recording_id": str(recording_id),
        "supersedes_transcript_id": (
            str(supersedes_transcript_id) if supersedes_transcript_id is not None else None
        ),
        "transcription_mode": mode,
        "vocabulary_hash": vocabulary_hash,
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _transcript_recency_order() -> tuple[Any, ...]:
    """Return a total, NULL-safe order for deterministic transcript reuse."""

    return (
        Transcript.completed_at.desc().nulls_last(),
        Transcript.created_at.desc(),
        Transcript.id.desc(),
    )


def parse_provider_datetime(value: object, settings: Settings) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(text, settings.YEASTAR_DATETIME_FORMAT)
        except ValueError:
            for fallback in ("%Y/%m/%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fallback)
                    break
                except ValueError:
                    continue
            else:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(settings.APP_TIMEZONE))
    return parsed.astimezone(UTC)


def _group_legacy_cdr_summaries(
    summaries: list[CDRSummary], settings: Settings
) -> tuple[dict[str, dict[str, Any]], dict[str, list[CDRSummary]]]:
    """Group legacy CDR legs by UID and choose their earliest call summary.

    The appliance CDR v1 list can contain several rows for a single call UID.
    The runtime client keeps the complete set for its synthetic detail, while
    this mapping supplies one stable call-level summary to persistence.
    """

    grouped: dict[str, list[CDRSummary]] = defaultdict(list)
    for summary in summaries:
        uid = summary.uid.strip()
        if uid:
            grouped[uid].append(summary)

    def time_key(summary: CDRSummary) -> tuple[int, float]:
        parsed = parse_provider_datetime(summary.time, settings)
        return (0, parsed.timestamp()) if parsed is not None else (1, 0.0)

    representatives = {
        uid: dict(min(entries, key=time_key).provider_dict()) for uid, entries in grouped.items()
    }
    return representatives, grouped


async def _fetch_cdrs(
    client: YeastarClient,
    date_from: datetime,
    date_to: datetime,
    filters: dict[str, Any],
    settings: Settings,
    *,
    cdr_api_version: Literal["v1", "v2"],
    progress_check: Callable[[], Awaitable[None]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[CDRSummary]]]:
    """Fetch call summaries using the persisted, tested CDR API mode."""

    if cdr_api_version == "v1":
        # CDR v1 has no documented server-side date or filter arguments.  Its
        # adapter performs bounded pagination and local filtering safely.
        if progress_check is not None:
            await progress_check()
        summaries = await client.search_all_cdrs(date_from, date_to, filters)
        if progress_check is not None:
            await progress_check()
        return _group_legacy_cdr_summaries(summaries, settings)

    cdrs: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        if progress_check is not None:
            await progress_check()
        response = await client.search_cdrs(date_from, date_to, filters, page)
        if progress_check is not None:
            await progress_check()
        data = response["data"]
        for item in data:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("uid") or "")
            if uid:
                cdrs[uid] = item
        if not data or len(cdrs) >= response["total_number"]:
            return cdrs, {}
        page += 1
        if page > 10000:
            raise YeastarError("Phone system returned invalid call pagination.")


def _direction(value: object) -> Direction:
    normalized = str(value or "").strip().casefold()
    return {
        "inbound": Direction.INBOUND,
        "outbound": Direction.OUTBOUND,
        "internal": Direction.INTERNAL,
    }.get(normalized, Direction.UNKNOWN)


def _queue_name(cdr: dict[str, Any]) -> str | None:
    queues = cdr.get("queues") or []
    if isinstance(queues, list):
        names = [
            str(item.get("name") or item.get("number")) for item in queues if isinstance(item, dict)
        ]
        return ", ".join(dict.fromkeys(name for name in names if name))[:255] or None
    return None


async def _job_stage(
    session: AsyncSession,
    job: ProcessingJob,
    status: JobStatus,
    human_stage: str,
    *,
    before_commit: Callable[[], Awaitable[None]] | None = None,
) -> None:
    await session.refresh(job, with_for_update=True)
    _require_processable_job(job)
    job.status = status
    job.current_stage = human_stage
    if job.started_at is None:
        job.started_at = utc_now()
    if before_commit is not None:
        await before_commit()
    await session.commit()


async def _commit_discovery_progress(
    session: AsyncSession,
    run: SyncRun,
    lease_check: Callable[[], Awaitable[None]] | None,
) -> None:
    """Fence a discovery commit and persist a durable liveness heartbeat."""

    if lease_check is not None:
        await lease_check()
    run.updated_at = utc_now()
    await session.commit()


async def _lock_recording_rows(
    session: AsyncSession,
    recording_ids: list[UUID] | set[UUID],
) -> None:
    """Serialize discovery and retention by recording in deterministic order."""

    ordered_ids = sorted(set(recording_ids), key=str)
    if not ordered_ids:
        return
    (
        await session.scalars(
            select(Recording.id)
            .where(Recording.id.in_(ordered_ids))
            .order_by(Recording.id)
            .with_for_update()
        )
    ).all()


async def _upsert_call(session: AsyncSession, cdr: dict[str, Any], settings: Settings) -> Call:
    uid = str(cdr.get("uid") or "").strip()
    if not uid:
        raise ValueError("Call is missing its provider UID")
    call = await session.scalar(select(Call).where(Call.yeastar_uid == uid))
    started_at = parse_provider_datetime(cdr.get("time"), settings) or utc_now()
    if call is None:
        call = Call(yeastar_uid=uid, started_at=started_at)
        session.add(call)
    call.yeastar_id = str(cdr.get("id") or "")[:255] or None
    call.started_at = started_at
    call.caller_number = str(cdr.get("call_from_number") or "")[:128] or None
    call.caller_name = str(cdr.get("call_from_name") or "")[:255] or None
    call.callee_number = str(cdr.get("call_to_number") or "")[:128] or None
    call.callee_name = str(cdr.get("call_to_name") or "")[:255] or None
    call.direction = _direction(cdr.get("call_type"))
    call.call_status = str(cdr.get("last_status") or "")[:64] or None
    call.duration_seconds = max(0, int(cdr.get("call_duration") or 0))
    call.queue_name = _queue_name(cdr)
    segments = cdr.get("segments")
    if isinstance(segments, (list, tuple, dict)):
        segment_count = len(segments)
    else:
        try:
            segment_count = int(segments or 1)
        except (TypeError, ValueError):
            segment_count = 1
    call.was_transferred = bool(segment_count > 1 or cdr.get("second_participant"))
    call.provider_payload = redact_value(cdr)
    call.processing_status = "finding_recordings"
    await session.flush()
    return call


async def _upsert_details(
    session: AsyncSession,
    call: Call,
    detail: dict[str, Any],
    operators: list[Operator],
    settings: Settings,
) -> list[CallParticipant]:
    operator_dicts = [
        {
            "id": operator.id,
            "yeastar_extension_id": operator.yeastar_extension_id,
            "extension_number": operator.extension_number,
        }
        for operator in operators
    ]
    interpreted_legs, interpreted_participants = interpret_call_legs(detail, operator_dicts)
    if len(interpreted_legs) > 1:
        # Conservative for channel safety: multi-leg calls are never treated as one-to-one stereo.
        call.was_transferred = True
    existing_legs = {
        leg.yeastar_leg_id: leg
        for leg in (await session.scalars(select(CallLeg).where(CallLeg.call_id == call.id))).all()
    }
    legs_by_provider_id: dict[str, CallLeg] = {}
    for raw in interpreted_legs:
        leg = existing_legs.get(raw["yeastar_leg_id"])
        if leg is None:
            leg = CallLeg(
                call_id=call.id,
                yeastar_leg_id=raw["yeastar_leg_id"],
                sequence_number=raw["sequence_number"],
            )
            session.add(leg)
        for key in (
            "sequence_number",
            "transaction_id",
            "yeastar_cdr_id",
            "provider_leg",
            "caller_extension_id",
            "callee_extension_id",
            "call_from",
            "call_to",
            "caller_number",
            "callee_number",
            "answered_by_extension_id",
            "duration_seconds",
            "ring_duration_seconds",
            "talk_duration_seconds",
            "hold_duration_seconds",
            "call_type",
            "status",
            "event_list",
            "provider_recording_id",
            "provider_payload",
        ):
            setattr(leg, key, raw.get(key))
        leg.started_at = parse_provider_datetime(raw.get("provider_start_time"), settings)
        leg.answered_at = parse_provider_datetime(raw.get("provider_answer_time"), settings)
        leg.ended_at = parse_provider_datetime(raw.get("provider_end_time"), settings)
        await session.flush()
        legs_by_provider_id[leg.yeastar_leg_id] = leg
    existing_participants = (
        await session.scalars(select(CallParticipant).where(CallParticipant.call_id == call.id))
    ).all()
    by_key = {
        (str(item.operator_id), str(item.call_leg_id), item.role.value): item
        for item in existing_participants
    }
    operator_by_id = {str(operator.id): operator for operator in operators}
    saved: list[CallParticipant] = []
    for item in interpreted_participants:
        operator = operator_by_id.get(item.operator_id)
        leg = legs_by_provider_id.get(item.leg_id)
        if operator is None or leg is None:
            continue
        key = (str(operator.id), str(leg.id), item.role.value)
        participant = by_key.get(key)
        if participant is None:
            participant = CallParticipant(
                call_id=call.id,
                operator_id=operator.id,
                call_leg_id=leg.id,
                role=item.role,
            )
            session.add(participant)
        participant.provider_extension_id = item.provider_extension_id
        participant.provider_extension_number = item.extension_number
        participant.operator_was_caller = item.was_caller
        participant.operator_was_callee = item.was_callee
        participant.answered = item.answered
        participant.participated_from = leg.started_at
        participant.participated_until = leg.ended_at
        participant.attribution_source = SpeakerSource.YEASTAR_EXTENSION
        participant.attribution_confidence = Decimal("1.0000")
        saved.append(participant)
    await session.flush()
    return saved


async def _upsert_recordings(
    session: AsyncSession,
    call: Call,
    provider_recordings: list[dict[str, Any]],
    settings: Settings,
) -> list[Recording]:
    existing = {
        item.yeastar_recording_id: item
        for item in (
            await session.scalars(select(Recording).where(Recording.call_id == call.id))
        ).all()
    }
    saved: list[Recording] = []
    for raw in provider_recordings:
        provider_id = str(raw.get("id") or "").strip()
        if not provider_id:
            continue
        recording = existing.get(provider_id)
        if recording is None:
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=provider_id,
                status=RecordingStatus.DISCOVERED,
            )
            session.add(recording)
        recording.yeastar_file_name = str(raw.get("file") or "")[:512] or None
        recording.yeastar_uid = str(raw.get("uid") or "")[:255] or None
        recording.provider_time = parse_provider_datetime(raw.get("time"), settings)
        recording.call_from = str(raw.get("call_from") or "")[:255] or None
        recording.call_to = str(raw.get("call_to") or "")[:255] or None
        recording.call_from_number = str(raw.get("call_from_number") or "")[:128] or None
        recording.call_to_number = str(raw.get("call_to_number") or "")[:128] or None
        recording.call_type = str(raw.get("call_type") or "")[:64] or None
        recording.archive_status = str(raw.get("archive_status") or "")[:64] or None
        if raw.get("duration") is not None:
            recording.duration_seconds = max(0, float(raw["duration"]))
        if raw.get("size") is not None:
            recording.size_bytes = max(0, int(raw["size"]))
        recording.provider_payload = redact_value(raw)
        saved.append(recording)
    call.has_recording = bool(saved)
    if saved:
        call.processing_status = "queued"
    await session.flush()
    return saved


def _normalise_recording_filename(value: object) -> str | None:
    """Return a provider filename suitable only for an equality comparison."""
    if not isinstance(value, str):
        return None
    filename = value.strip().replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    return filename or None


def _normalise_phone_number(value: object) -> str | None:
    """Normalise a phone endpoint without attempting country-specific guesses."""
    if value is None:
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits or None


def _recordings_for_participant(
    recordings: list[Recording], participant: CallParticipant, leg: CallLeg | None
) -> tuple[list[Recording], str | None]:
    """Find every recording with a deterministic relation to a call leg.

    A transferred call can legitimately have more than one recording.  Picking
    the first one would silently attach the wrong conversation to an operator,
    so each fallback below is deliberately exact.  A single leg can safely
    own several recordings when every one has the exact same endpoint pair.
    """
    if len(recordings) == 1:
        return recordings, None
    if not recordings:
        return [], "no_recording"
    explicit_matches: list[Recording] = []
    filename_matches: list[Recording] = []
    endpoint_matches: list[Recording] = []
    if leg and leg.provider_recording_id:
        matched = [
            item for item in recordings if item.yeastar_recording_id == leg.provider_recording_id
        ]
        if len(matched) == 1:
            explicit_matches = matched

    if leg is not None:
        # Legacy CDR 1.0 exposes the recording filename on the CDR row rather
        # than a recording ID.  The recording search result exposes the same
        # filename, which gives us a safe, deterministic correlation.
        payload = leg.provider_payload if isinstance(leg.provider_payload, dict) else {}
        filenames = {
            filename
            for field in ("record_file", "recording_file", "file")
            if (filename := _normalise_recording_filename(payload.get(field))) is not None
        }
        if filenames:
            filename_matches = [
                item
                for item in recordings
                if _normalise_recording_filename(item.yeastar_file_name) in filenames
            ]

        # Some firmware versions omit both IDs and filenames from the CDR
        # detail.  An ordered, two-endpoint match is still deterministic; a
        # one-sided extension match is not and is intentionally never used.
        caller = _normalise_phone_number(leg.caller_number)
        callee = _normalise_phone_number(leg.callee_number)
        if caller is not None and callee is not None:
            endpoint_matches = [
                item
                for item in recordings
                if _normalise_phone_number(item.call_from_number) == caller
                and _normalise_phone_number(item.call_to_number) == callee
            ]

    # Provider IDs and legacy filenames identify one recording directly.  A
    # matching endpoint pair may identify additional split recordings for the
    # same leg, so retain all of those safely attributable files.  If the only
    # direct signals conflict, leave the call queued rather than inventing a
    # relation between them.
    direct_matches = explicit_matches
    if len(filename_matches) == 1:
        if (
            direct_matches
            and filename_matches[0].yeastar_recording_id != direct_matches[0].yeastar_recording_id
        ):
            corroborated_ids = {item.yeastar_recording_id for item in endpoint_matches}
            direct_ids = {
                direct_matches[0].yeastar_recording_id,
                filename_matches[0].yeastar_recording_id,
            }
            if not direct_ids.issubset(corroborated_ids):
                return [], "recording_assignment_pending"
        direct_matches = [*direct_matches, *filename_matches]
    matched_provider_ids = {
        item.yeastar_recording_id for item in [*direct_matches, *endpoint_matches]
    }
    if matched_provider_ids:
        return [
            item for item in recordings if item.yeastar_recording_id in matched_provider_ids
        ], None

    # The recording list and CDR details can arrive at different times on the
    # PBX.  Keep the item queued and let the analysis task revisit discovery
    # instead of reporting a user-visible failure or guessing a recording.
    return [], "recording_assignment_pending"


def _recording_for_participant(
    recordings: list[Recording], participant: CallParticipant, leg: CallLeg | None
) -> tuple[Recording | None, str | None]:
    """Compatibility wrapper for callers that need exactly one recording."""
    matched, reason = _recordings_for_participant(recordings, participant, leg)
    return (
        (matched[0], reason)
        if len(matched) == 1
        else (None, reason or "recording_assignment_pending")
    )


def _processing_item_key(
    job_id: UUID,
    call_id: UUID,
    operator_id: UUID,
    recording_id: UUID | None,
    *,
    state: str | None = None,
) -> str:
    """Build stable item keys for both recording work and unassigned states."""
    identity = str(recording_id) if recording_id is not None else state or "unassigned"
    return hashlib.sha256(f"{job_id}:{call_id}:{operator_id}:{identity}".encode()).hexdigest()


def _legacy_processing_item_key(job_id: UUID, call_id: UUID, operator_id: UUID) -> str:
    """Return the pre-recording-aware key used by already-created jobs."""
    return hashlib.sha256(f"{job_id}:{call_id}:{operator_id}".encode()).hexdigest()


async def discover_job_items(job_id: UUID) -> DiscoveryResult:
    lock = get_redis().lock(
        f"yca:job-discovery:{job_id}",
        timeout=WORKER_LEASE_SECONDS,
        blocking_timeout=1,
    )
    if not await lock.acquire(blocking=False):
        raise ProcessingBusyError("Analysis discovery is already running.")

    async def lease_check() -> None:
        await _refresh_worker_lease(lock)

    try:
        await lease_check()
        return await _discover_job_items_locked(job_id, lease_check=lease_check)
    finally:
        try:
            if await lock.owned():
                await lock.release()
        except Exception:
            pass


async def _discover_job_items_locked(
    job_id: UUID,
    *,
    lease_check: Callable[[], Awaitable[None]] | None = None,
) -> DiscoveryResult:
    redis = get_redis()
    try:
        settings = await load_effective_yeastar_settings(redis, get_settings())
    except (RedisError, YeastarConfigurationError) as exc:
        async with AsyncSessionFactory() as session:
            if await _stop_stale_discovery(session, job_id):
                return DiscoveryResult([])
            job = await session.get(ProcessingJob, job_id)
            if job is not None:
                job.status = JobStatus.WAITING_FOR_CONNECTION
                job.current_stage = "Waiting for phone-system connection"
                job.last_error_category = getattr(
                    exc,
                    "category",
                    "configuration_unavailable",
                )
                job.last_error_message = "Test the phone-system connection in Settings."
                job.completed_at = None
                if lease_check is not None:
                    await lease_check()
                await session.commit()
        return DiscoveryResult([])
    async with AsyncSessionFactory() as session:
        job = await session.get(ProcessingJob, job_id)
        if job is None or job.status in FINAL_JOB_STATES:
            return DiscoveryResult([])
        if job.cancellation_requested:
            await _stop_stale_discovery(session, job_id)
            return DiscoveryResult([])
        _, configured, changed_from_existing = await reconcile_configuration_fingerprint(
            session,
            settings,
            clear_token_state=lambda: clear_shared_token_and_require_test(redis),
        )
        if not configured or changed_from_existing:
            await session.refresh(job, with_for_update=True)
            if job.cancellation_requested or job.status in FINAL_JOB_STATES:
                await _stop_stale_discovery(session, job_id)
                return DiscoveryResult([])
            job.status = JobStatus.WAITING_FOR_CONNECTION
            job.current_stage = "Waiting for phone-system connection"
            job.last_error_category = (
                "configuration_changed" if changed_from_existing else "not_configured"
            )
            job.last_error_message = "Test the phone-system connection in Settings."
            job.completed_at = None
            if lease_check is not None:
                await lease_check()
            await session.commit()
            return DiscoveryResult([])
        run_key = f"job:{job.id}:attempt:{job.attempt_count}"
        run: SyncRun | None = None
        existing_run = await session.scalar(
            select(SyncRun).where(SyncRun.idempotency_key == run_key)
        )
        if existing_run is not None:
            if existing_run.status == RunStatus.RUNNING:
                if _worker_timestamp_is_recent(existing_run.updated_at):
                    raise ProcessingBusyError("Analysis discovery is already running.")
                # The Redis discovery lease is held by this task and the
                # durable run has not advanced within the worker-stale window.
                # Reuse the same logical attempt marker so crash recovery does
                # not erase durable run history or violate its idempotency key.
                existing_run.started_at = utc_now()
                existing_run.completed_at = None
                existing_run.records_seen = 0
                existing_run.records_created = 0
                existing_run.records_updated = 0
                existing_run.records_failed = 0
                existing_run.error_category = None
                existing_run.error_message = None
                run = existing_run
                existing_run = None
        if existing_run is not None:
            unassigned = (
                await session.scalars(
                    select(ProcessingJobItem.id).where(
                        ProcessingJobItem.job_id == job.id,
                        ProcessingJobItem.recording_id.is_(None),
                        ProcessingJobItem.status.in_(
                            [
                                ItemStatus.QUEUED,
                                ItemStatus.WAITING_FOR_CONNECTION,
                                ItemStatus.PROCESSING,
                                ItemStatus.FAILED,
                            ]
                        ),
                    )
                )
            ).all()
            if unassigned:
                # Older releases could leave one queued item with no recording
                # ID.  Re-run discovery so it can be converted to either a
                # recording-backed item or the new safe pending placeholder.
                await session.delete(existing_run)
                await session.flush()
            else:
                pending = (
                    await session.scalars(
                        select(ProcessingJobItem.id).where(
                            ProcessingJobItem.job_id == job.id,
                            ProcessingJobItem.status == ItemStatus.QUEUED,
                            ProcessingJobItem.recording_id.is_not(None),
                        )
                    )
                ).all()
                if lease_check is not None:
                    await lease_check()
                return DiscoveryResult(list(pending))
        if run is None:
            run = SyncRun(
                sync_type=SyncType.CALLS,
                status=RunStatus.RUNNING,
                idempotency_key=run_key,
                processing_job_id=job.id,
                started_at=utc_now(),
                date_from=job.date_from,
                date_to=job.date_to,
            )
            session.add(run)

        async def discovery_heartbeat() -> None:
            if lease_check is not None:
                await lease_check()
            run.updated_at = utc_now()

        await _job_stage(
            session,
            job,
            JobStatus.CONNECTING,
            "Connecting to the phone system",
            before_commit=discovery_heartbeat,
        )
        try:
            integration = await session.scalar(
                select(IntegrationStatus).where(IntegrationStatus.provider == "yeastar")
            )
            cdr_api_version = runtime_cdr_api_version(
                integration.capabilities_json if integration is not None else None
            )
            async with YeastarClient(
                settings=settings,
                redis=redis,
                system_date_format=(integration.system_date_format if integration else None),
                system_time_format=(integration.system_time_format if integration else None),
                cdr_api_version=cdr_api_version,
            ) as client:
                await client.authenticate()
                await _job_stage(
                    session,
                    job,
                    JobStatus.FETCHING_CALLS,
                    "Finding calls",
                    before_commit=discovery_heartbeat,
                )
                filters: dict[str, Any] = {}
                if job.direction:
                    filters["call_type"] = job.direction.title()
                if job.queue_name:
                    filters["queue"] = job.queue_name
                if job.call_status_filter:
                    filters["status"] = job.call_status_filter
                if job.recording_available is not None:
                    filters["recording_type"] = 1 if job.recording_available else 2
                cdrs, legacy_cdr_summaries = await _fetch_cdrs(
                    client,
                    job.date_from,
                    job.date_to,
                    filters,
                    settings,
                    cdr_api_version=cdr_api_version,
                    progress_check=lease_check,
                )
                run.records_seen = len(cdrs)
                await _job_stage(
                    session,
                    job,
                    JobStatus.FINDING_RECORDINGS,
                    "Finding recordings",
                    before_commit=discovery_heartbeat,
                )
                provider_recordings = await client.search_recordings(job.date_from, job.date_to)
                recordings_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for raw in provider_recordings:
                    recordings_by_uid[str(raw.get("uid") or "")].append(raw)
                operators = (
                    await session.scalars(select(Operator).where(Operator.deleted_at.is_(None)))
                ).all()
                selected_ids = {UUID(value) for value in job.selected_operator_ids}
                item_ids: list[UUID] = []
                relevant_calls: set[UUID] = set()
                relevant_recordings: set[UUID] = set()
                discovery_failures = 0
                assignment_pending_calls: set[UUID] = set()
                await _job_stage(
                    session,
                    job,
                    JobStatus.FETCHING_CALL_DETAILS,
                    "Finding call participants",
                    before_commit=discovery_heartbeat,
                )
                for cdr in cdrs.values():
                    if lease_check is not None:
                        await lease_check()
                    await session.refresh(job)
                    _require_processable_job(job)
                    call = await _upsert_call(session, cdr, settings)
                    try:
                        if cdr_api_version == "v1":
                            detail = await client.get_cdr_detail(
                                call.yeastar_uid,
                                summaries=legacy_cdr_summaries.get(call.yeastar_uid),
                            )
                        else:
                            detail = await client.get_cdr_detail(call.yeastar_uid)
                        participants = await _upsert_details(
                            session, call, detail, operators, settings
                        )
                        recordings = await _upsert_recordings(
                            session,
                            call,
                            recordings_by_uid.get(call.yeastar_uid, []),
                            settings,
                        )
                    except CONNECTION_PAUSE_ERRORS:
                        raise
                    except Exception as exc:
                        await session.refresh(job, with_for_update=True)
                        _require_processable_job(job)
                        call.processing_status = "failed"
                        call.last_error_category = getattr(exc, "category", "call_detail")
                        call.last_error_message = "Call details could not be processed."
                        discovery_failures += 1
                        await _commit_discovery_progress(session, run, lease_check)
                        continue
                    selected_participants: dict[UUID, list[CallParticipant]] = defaultdict(list)
                    for participant in participants:
                        if participant.operator_id in selected_ids:
                            selected_participants[participant.operator_id].append(participant)
                    if not selected_participants:
                        await session.refresh(job, with_for_update=True)
                        _require_processable_job(job)
                        await _commit_discovery_progress(session, run, lease_check)
                        continue
                    if lease_check is not None:
                        await lease_check()
                    await _lock_recording_rows(
                        session,
                        [recording.id for recording in recordings],
                    )
                    if lease_check is not None:
                        await lease_check()
                    relevant_calls.add(call.id)
                    relevant_recordings.update(recording.id for recording in recordings)
                    legs_by_id = {
                        leg.id: leg
                        for leg in (
                            await session.scalars(select(CallLeg).where(CallLeg.call_id == call.id))
                        ).all()
                    }
                    assigned_by_operator: dict[UUID, dict[UUID, Recording]] = defaultdict(dict)
                    pending_operator_ids: set[UUID] = set()
                    for operator_id, operator_participants in selected_participants.items():
                        for participant in operator_participants:
                            matched, assignment_error = _recordings_for_participant(
                                recordings,
                                participant,
                                legs_by_id.get(participant.call_leg_id),
                            )
                            for recording in matched:
                                assigned_by_operator[operator_id][recording.id] = recording
                            if assignment_error == "recording_assignment_pending":
                                pending_operator_ids.add(operator_id)

                    legacy_unassigned_by_operator: dict[UUID, ProcessingJobItem | None] = {}
                    for operator_id in selected_participants:
                        legacy_key = _legacy_processing_item_key(job.id, call.id, operator_id)
                        legacy_unassigned_by_operator[operator_id] = await session.scalar(
                            select(ProcessingJobItem).where(
                                ProcessingJobItem.idempotency_key == legacy_key,
                                ProcessingJobItem.recording_id.is_(None),
                            )
                        )

                    queued_item_ids: set[UUID] = set()
                    for operator_id, assigned_recordings in assigned_by_operator.items():
                        for recording in assigned_recordings.values():
                            key = _processing_item_key(
                                job.id,
                                call.id,
                                operator_id,
                                recording.id,
                            )
                            item = await session.scalar(
                                select(ProcessingJobItem).where(
                                    ProcessingJobItem.job_id == job.id,
                                    ProcessingJobItem.call_id == call.id,
                                    ProcessingJobItem.operator_id == operator_id,
                                    ProcessingJobItem.recording_id == recording.id,
                                )
                            )
                            if item is None:
                                item = await session.scalar(
                                    select(ProcessingJobItem).where(
                                        ProcessingJobItem.idempotency_key == key
                                    )
                                )
                            converted_legacy_item = False
                            legacy_item = legacy_unassigned_by_operator.get(operator_id)
                            if item is None and legacy_item is not None:
                                item = legacy_item
                                legacy_unassigned_by_operator[operator_id] = None
                                item.recording_id = recording.id
                                item.idempotency_key = key
                                converted_legacy_item = True
                            if item is None:
                                item = ProcessingJobItem(
                                    job_id=job.id,
                                    call_id=call.id,
                                    operator_id=operator_id,
                                    recording_id=recording.id,
                                    idempotency_key=key,
                                    status=ItemStatus.QUEUED,
                                    stage="queued",
                                )
                                session.add(item)
                                await session.flush()
                            if converted_legacy_item:
                                item.status = ItemStatus.QUEUED
                            if item.status == ItemStatus.QUEUED and item.id not in queued_item_ids:
                                item.stage = "queued"
                                item.error_category = None
                                item.error_message = None
                                item.completed_at = None
                                item.locked_at = None
                                item.heartbeat_at = None
                                item_ids.append(item.id)
                                queued_item_ids.add(item.id)

                    for operator_id in selected_participants:
                        pending_key = _processing_item_key(
                            job.id,
                            call.id,
                            operator_id,
                            None,
                            state="recording-assignment-pending",
                        )
                        pending_item = await session.scalar(
                            select(ProcessingJobItem).where(
                                ProcessingJobItem.idempotency_key == pending_key
                            )
                        )
                        legacy_item = legacy_unassigned_by_operator.get(operator_id)
                        if operator_id in pending_operator_ids:
                            if pending_item is None and legacy_item is not None:
                                pending_item = legacy_item
                                legacy_unassigned_by_operator[operator_id] = None
                                pending_item.idempotency_key = pending_key
                            if pending_item is None:
                                pending_item = ProcessingJobItem(
                                    job_id=job.id,
                                    call_id=call.id,
                                    operator_id=operator_id,
                                    recording_id=None,
                                    idempotency_key=pending_key,
                                )
                                session.add(pending_item)
                            pending_item.status = ItemStatus.QUEUED
                            pending_item.stage = "waiting_for_recording_assignment"
                            pending_item.error_category = None
                            pending_item.error_message = None
                            pending_item.completed_at = None
                            pending_item.locked_at = None
                            pending_item.heartbeat_at = None
                            assignment_pending_calls.add(call.id)
                            call.processing_status = "waiting_for_recording_assignment"
                            call.last_error_category = None
                            call.last_error_message = None
                        elif pending_item is not None:
                            await session.delete(pending_item)

                        legacy_item = legacy_unassigned_by_operator.get(operator_id)
                        if legacy_item is not None:
                            await session.delete(legacy_item)
                            legacy_unassigned_by_operator[operator_id] = None

                        if (
                            not assigned_by_operator.get(operator_id)
                            and operator_id not in pending_operator_ids
                        ):
                            skipped_key = _processing_item_key(
                                job.id,
                                call.id,
                                operator_id,
                                None,
                                state="no-recording",
                            )
                            skipped_item = await session.scalar(
                                select(ProcessingJobItem).where(
                                    ProcessingJobItem.idempotency_key == skipped_key
                                )
                            )
                            if skipped_item is None:
                                session.add(
                                    ProcessingJobItem(
                                        job_id=job.id,
                                        call_id=call.id,
                                        operator_id=operator_id,
                                        recording_id=None,
                                        idempotency_key=skipped_key,
                                        status=ItemStatus.SKIPPED,
                                        stage="no_recording",
                                        completed_at=utc_now(),
                                    )
                                )
                    await session.refresh(job, with_for_update=True)
                    _require_processable_job(job)
                    await _commit_discovery_progress(session, run, lease_check)
                job = await session.get(ProcessingJob, job.id)
                assert job is not None
                await session.refresh(job, with_for_update=True)
                _require_processable_job(job)
                job.calls_found = len(relevant_calls)
                job.recordings_found = len(relevant_recordings)
                item_status_rows = (
                    await session.execute(
                        select(ProcessingJobItem.call_id, ProcessingJobItem.status).where(
                            ProcessingJobItem.job_id == job.id
                        )
                    )
                ).all()
                # A re-discovery can resolve a later recording while work for
                # an earlier recording has already failed.  Preserve that
                # outcome instead of overwriting it with a false success when
                # there is nothing new to dispatch.
                failed_item_calls = {
                    call_id
                    for call_id, item_status in item_status_rows
                    if item_status == ItemStatus.FAILED
                }
                has_active_items = any(
                    item_status not in FINAL_ITEM_STATES for _, item_status in item_status_rows
                )
                total_discovery_errors = discovery_failures + len(failed_item_calls)
                job.calls_failed = total_discovery_errors
                job.request_filters = {
                    **(job.request_filters or {}),
                    "_discovery_failures": discovery_failures,
                    "_recording_assignment_pending": len(assignment_pending_calls),
                }
                if assignment_pending_calls:
                    # Preserve a queued, unassigned item and let the worker
                    # retry full discovery.  There is deliberately no failed
                    # item here: the PBX has not supplied a safe correlation.
                    job.status = JobStatus.FINDING_RECORDINGS
                    job.current_stage = "Waiting for recording assignment"
                    job.completed_at = None
                    job.last_error_category = None
                    job.last_error_message = None
                    await session.delete(run)
                    if lease_check is not None:
                        await lease_check()
                    await session.commit()
                    return DiscoveryResult(item_ids, recording_assignment_pending=True)
                has_work_to_finish = bool(item_ids) or has_active_items
                job.status = (
                    JobStatus.DOWNLOADING_RECORDINGS
                    if has_work_to_finish
                    else (
                        JobStatus.COMPLETED_WITH_ERRORS
                        if total_discovery_errors
                        else JobStatus.COMPLETED
                    )
                )
                job.current_stage = "Preparing recordings" if has_work_to_finish else "Complete"
                if not has_work_to_finish:
                    job.progress_percent = 100
                    job.completed_at = utc_now()
                run.status = (
                    RunStatus.COMPLETED_WITH_ERRORS
                    if total_discovery_errors
                    else RunStatus.COMPLETED
                )
                run.records_created = len(relevant_calls)
                run.records_failed = total_discovery_errors
                run.completed_at = utc_now()
                await _commit_discovery_progress(session, run, lease_check)
                return DiscoveryResult(item_ids)
        except ProcessingCancelledError:
            await session.rollback()
            await _stop_stale_discovery(session, job_id)
            return DiscoveryResult([])
        except StaleProcessingTaskError:
            await session.rollback()
            await _stop_stale_discovery(session, job_id)
            return DiscoveryResult([])
        except ProcessingBusyError:
            await session.rollback()
            raise
        except CONNECTION_PAUSE_ERRORS as exc:
            await session.rollback()
            if lease_check is not None:
                await lease_check()
            if await _stop_stale_discovery(session, job_id):
                return DiscoveryResult([])
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.WAITING_FOR_CONNECTION
                job.current_stage = "Waiting for phone-system connection"
                job.last_error_category = getattr(exc, "category", "connection_paused")
                job.last_error_message = "Test the phone-system connection in Settings."
                job.completed_at = None
            current_run = await session.scalar(
                select(SyncRun)
                .where(SyncRun.processing_job_id == job_id)
                .order_by(SyncRun.created_at.desc())
            )
            if current_run is not None:
                await session.delete(current_run)
            await session.commit()
            return DiscoveryResult([])
        except Exception as exc:
            await session.rollback()
            if lease_check is not None:
                await lease_check()
            if await _stop_stale_discovery(session, job_id):
                return DiscoveryResult([])
            job = await session.get(ProcessingJob, job_id)
            if job:
                job.status = JobStatus.FAILED
                job.current_stage = "Could not complete analysis"
                job.last_error_category = getattr(exc, "category", "unexpected")
                job.last_error_message = _safe_error_message(exc)
                job.completed_at = utc_now()
            run = await session.scalar(
                select(SyncRun)
                .where(SyncRun.processing_job_id == job_id)
                .order_by(SyncRun.created_at.desc())
            )
            if run:
                run.status = RunStatus.FAILED
                run.error_category = getattr(exc, "category", "unexpected")
                run.error_message = _safe_error_message(exc)
                run.completed_at = utc_now()
            await session.commit()
            raise


def _safe_error_message(exc: Exception) -> str:
    if isinstance(
        exc,
        (
            YeastarError,
            AudioError,
            TranscriptionError,
            ProcessingCancelledError,
            ReprocessStateError,
        ),
    ):
        return str(exc)[:1000]
    return "An unexpected processing error occurred."


def _cleanup_temporary_audio(
    audio: AudioProcessor,
    temporary_files: list[Path],
    temporary_directories: set[Path],
) -> None:
    audio.remove_files(temporary_files)
    for path in temporary_directories | {item.parent for item in temporary_files}:
        try:
            path.rmdir()
        except OSError:
            pass


def _reprocess_target_id(
    job: ProcessingJob,
    item: ProcessingJobItem,
    settings: Settings | None = None,
) -> UUID | None:
    """Return the persisted replacement target for an explicit reprocess item."""

    if item.requested_pipeline_version is None:
        return None
    if item.requested_pipeline_version not in SUPPORTED_PIPELINE_VERSIONS:
        raise ReprocessStateError(
            f"Pipeline version {item.requested_pipeline_version!r} is not available."
        )
    if (
        item.requested_pipeline_version == PIPELINE_V2_VERSION
        and settings is not None
        and not getattr(settings, "TRANSCRIPTION_PIPELINE_V2_ENABLED", False)
    ):
        raise ReprocessStateError("Pipeline V2 is disabled.")
    targets = (job.request_filters or {}).get("_reprocess_targets")
    raw_target = targets.get(str(item.id)) if isinstance(targets, dict) else None
    try:
        return UUID(str(raw_target))
    except (TypeError, ValueError) as exc:
        raise ReprocessStateError(
            "The completed transcript selected for reprocessing is unavailable."
        ) from exc


def _record_reprocess_replacement(
    job: ProcessingJob,
    item: ProcessingJobItem,
    transcript: Transcript,
) -> None:
    """Persist which non-current row this item is allowed to fail or promote."""

    if item.requested_pipeline_version is None:
        return
    request_filters = dict(job.request_filters or {})
    replacements = dict(request_filters.get("_reprocess_replacements") or {})
    replacements[str(item.id)] = str(transcript.id)
    request_filters["_reprocess_replacements"] = replacements
    job.request_filters = request_filters


def _item_owns_reprocess_replacement(
    job: ProcessingJob,
    item: ProcessingJobItem,
    transcript: Transcript,
) -> bool:
    replacements = (job.request_filters or {}).get("_reprocess_replacements")
    return isinstance(replacements, dict) and replacements.get(str(item.id)) == str(transcript.id)


def _should_mark_transcript_failed(
    job: ProcessingJob,
    item: ProcessingJobItem,
    transcript: Transcript,
) -> bool:
    # Provider results are committed before keyword search and replacement
    # activation. A later failure must not erase completed text or auditable
    # attempts; a subsequent delivery can resume those downstream steps.
    if transcript.status == TranscriptStatus.COMPLETED:
        return False
    if item.requested_pipeline_version is None:
        return True
    return not transcript.is_current and _item_owns_reprocess_replacement(job, item, transcript)


def _has_persisted_mono_pass1_evidence(transcript: Transcript) -> bool:
    api_usage = transcript.api_usage
    if isinstance(api_usage, Mapping) and "pass1_diarization" in api_usage:
        return True
    quality_summary = transcript.quality_summary
    return isinstance(quality_summary, Mapping) and quality_summary.get("pass1_completed") is True


async def _archive_failed_v2_attempt_history(
    session: AsyncSession,
    transcript: Transcript,
    *,
    stable_key: str,
    transcription_mode: TranscriptionMode,
) -> bool:
    """Release a retry key without deleting an interrupted run's audit rows."""

    if transcript.status != TranscriptStatus.FAILED or transcription_mode not in {
        TranscriptionMode.OPERATOR_CHANNEL,
        TranscriptionMode.DUAL_CHANNEL,
        TranscriptionMode.MONO_DIARIZATION,
    }:
        return False
    attempt_count = (
        await session.scalar(
            select(func.count())
            .select_from(TranscriptionAttempt)
            .where(TranscriptionAttempt.transcript_id == transcript.id)
        )
        or 0
    )
    if attempt_count == 0 and not (
        transcription_mode == TranscriptionMode.MONO_DIARIZATION
        and _has_persisted_mono_pass1_evidence(transcript)
    ):
        return False
    transcript.idempotency_key = hashlib.sha256(
        f"failed-attempt-history:{stable_key}:{transcript.id}".encode()
    ).hexdigest()
    transcript.is_current = False
    await session.flush()
    return True


async def _retire_noncompleted_current_target(
    session: AsyncSession,
    *,
    recording_id: UUID,
    operator_id: UUID | None,
) -> None:
    """Make room for a normal retry when only a failed current row exists."""

    operator_condition = (
        Transcript.operator_id == operator_id
        if operator_id is not None
        else Transcript.operator_id.is_(None)
    )
    current = await session.scalar(
        select(Transcript)
        .where(
            Transcript.recording_id == recording_id,
            operator_condition,
            Transcript.is_current.is_(True),
        )
        .with_for_update()
    )
    if current is None:
        return
    if current.status == TranscriptStatus.COMPLETED:
        raise ReprocessStateError("A completed transcript already exists for this recording.")
    current.is_current = False
    await session.flush()


async def _activate_transcript_replacement(
    session: AsyncSession,
    replacement: Transcript,
    previous_transcript_id: UUID,
) -> None:
    """Atomically promote a completed replacement after demoting its source."""

    if replacement.id == previous_transcript_id:
        raise ReprocessStateError("A transcript cannot replace itself.")
    recording_id = await session.scalar(
        select(Recording.id).where(Recording.id == replacement.recording_id).with_for_update()
    )
    if recording_id is None:
        raise ReprocessStateError("The replacement recording is unavailable.")
    previous = await session.scalar(
        select(Transcript).where(Transcript.id == previous_transcript_id).with_for_update()
    )
    if previous is None:
        raise ReprocessStateError("The transcript selected for replacement is unavailable.")
    if previous.call_id != replacement.call_id or previous.recording_id != replacement.recording_id:
        raise ReprocessStateError("A replacement must remain within the same call and recording.")
    await _validate_supersession_lineage(session, replacement, previous)
    if replacement.status != TranscriptStatus.COMPLETED:
        raise ReprocessStateError("Only a completed transcript can become current.")
    if replacement.is_current and replacement.supersedes_transcript_id == previous_transcript_id:
        return
    if previous.status != TranscriptStatus.COMPLETED or not previous.is_current:
        raise ReprocessStateError("The current transcript changed before reprocessing completed.")

    # PostgreSQL partial unique indexes are immediate rather than deferrable.
    # Flush the demotion first, then promote the replacement in the same
    # transaction so readers never observe a committed gap.
    previous.is_current = False
    await session.flush()
    replacement.supersedes_transcript_id = previous.id
    replacement.is_current = True
    await session.flush()


async def _validate_supersession_lineage(
    session: AsyncSession,
    replacement: Transcript,
    previous: Transcript,
) -> None:
    """Reject cycles and target changes before linking transcript history."""

    visited = {replacement.id}
    current: Transcript | None = previous
    while current is not None:
        if current.id in visited:
            raise ReprocessStateError("Transcript supersession would create a cycle.")
        visited.add(current.id)
        if (
            current.call_id != replacement.call_id
            or current.recording_id != replacement.recording_id
        ):
            raise ReprocessStateError("Transcript history crosses a call or recording boundary.")
        # An anonymous source may become attributed when PBX topology becomes
        # available. Once attributed, history cannot move to another operator
        # or back to an anonymous target.
        if current.operator_id not in {None, replacement.operator_id}:
            raise ReprocessStateError("Transcript history crosses an operator boundary.")
        if current.supersedes_transcript_id is None:
            return
        current = await session.scalar(
            select(Transcript)
            .where(Transcript.id == current.supersedes_transcript_id)
            .with_for_update()
        )
        if current is None:
            raise ReprocessStateError("Transcript supersession history is incomplete.")


async def _cancel_requested(session: AsyncSession, job_id: UUID) -> bool:
    value = await session.scalar(
        select(ProcessingJob.cancellation_requested).where(ProcessingJob.id == job_id)
    )
    return bool(value)


async def _lock_processable_item(
    session: AsyncSession,
    item: ProcessingJobItem,
    job: ProcessingJob,
) -> None:
    """Lock recording, job, then item and reject stale or cancelled delivery."""
    if item.recording_id is not None:
        await _lock_recording_rows(session, [item.recording_id])
    await session.refresh(job, with_for_update=True)
    _require_processable_job(job)
    await session.refresh(item, with_for_update=True)
    if item.status == ItemStatus.CANCELLED:
        raise ProcessingCancelledError("Analysis was cancelled.")
    if item.status in FINAL_ITEM_STATES:
        raise StaleProcessingTaskError("Analysis item is already finished.")


async def _item_stage(
    session: AsyncSession,
    item: ProcessingJobItem,
    job: ProcessingJob,
    status: JobStatus,
    stage: str,
    *,
    before_commit: Callable[[], Awaitable[None]] | None = None,
) -> None:
    if before_commit is not None:
        await before_commit()
    await _lock_processable_item(session, item, job)
    if before_commit is not None:
        await before_commit()
    item.stage = stage
    item.heartbeat_at = utc_now()
    job.status = status
    job.current_stage = stage
    await session.commit()


async def _clone_transcript(
    session: AsyncSession,
    source: Transcript,
    source_segments: tuple[TranscriptSegment, ...],
    *,
    call: Call,
    recording: Recording,
    operator: Operator | None,
    call_leg_id: UUID | None,
    key: str,
) -> Transcript:
    clone = Transcript(
        call_id=call.id,
        recording_id=recording.id,
        operator_id=operator.id if operator else None,
        idempotency_key=key,
        status=TranscriptStatus.COMPLETED,
        model=source.model,
        language=source.language,
        prompt_version=source.prompt_version,
        prompt_template_version=source.prompt_template_version,
        vocabulary_hash=source.vocabulary_hash,
        processing_duration_seconds=0,
        audio_duration_seconds=recording.duration_seconds,
        api_usage={"audio_reused_by_checksum": True},
        attempt_count=0,
        completed_at=utc_now(),
        source_audio_sha256=source.source_audio_sha256,
        is_diarized=source.is_diarized,
        transcription_mode=source.transcription_mode,
        speaker_attribution_status=source.speaker_attribution_status,
        pipeline_version=source.pipeline_version,
        pipeline_config_hash=source.pipeline_config_hash,
        preprocessing_profile=source.preprocessing_profile,
        quality_summary=source.quality_summary,
        is_current=True,
        original_text=source.original_text,
        normalized_text=source.normalized_text,
    )
    session.add(clone)
    await session.flush()
    for segment in source_segments:
        session.add(
            TranscriptSegment(
                transcript_id=clone.id,
                call_id=call.id,
                call_leg_id=call_leg_id if operator else None,
                operator_id=operator.id if operator else None,
                speaker_label=operator.display_name if operator else segment.speaker_label,
                speaker_source=segment.speaker_source,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                original_text=segment.original_text,
                normalized_text=segment.normalized_text,
                confidence=segment.confidence,
                transcription_model=segment.transcription_model,
                sequence_number=segment.sequence_number,
                channel_index=segment.channel_index,
                track_id=segment.track_id,
                chunk_index=segment.chunk_index,
                mean_logprob=segment.mean_logprob,
                low_logprob_ratio=segment.low_logprob_ratio,
                quality_flags=list(segment.quality_flags or []),
                audio_variant=segment.audio_variant,
            )
        )
    await session.flush()
    return clone


def _cross_recording_transcript_clone_allowed(
    transcription_mode: TranscriptionMode,
) -> bool:
    """Only clone modes whose complete audit contract is copied with segments."""

    return transcription_mode not in {
        TranscriptionMode.OPERATOR_CHANNEL,
        TranscriptionMode.DUAL_CHANNEL,
        TranscriptionMode.MONO_DIARIZATION,
    }


async def _completed_duplicate_snapshot(
    session: AsyncSession,
    *,
    checksum: str,
    model: str,
    diarized: bool,
    operator_id: UUID | None,
    language: str,
    prompt_version: str | None,
    pipeline_version: str,
    pipeline_config_hash: str,
    preprocessing_profile: str,
    transcription_mode: TranscriptionMode,
    prompt_template_version: str | None = None,
    vocabulary_hash: str | None = None,
) -> tuple[Transcript, tuple[TranscriptSegment, ...]] | None:
    """Read a reusable transcript and all of its segments from one MVCC snapshot."""

    latest_id = (
        select(Transcript.id)
        .where(
            Transcript.source_audio_sha256 == checksum,
            Transcript.model == model,
            Transcript.is_diarized.is_(diarized),
            Transcript.operator_id == operator_id,
            Transcript.language == language,
            Transcript.prompt_version == prompt_version,
            Transcript.prompt_template_version == prompt_template_version,
            Transcript.vocabulary_hash == vocabulary_hash,
            Transcript.pipeline_version == pipeline_version,
            Transcript.pipeline_config_hash == pipeline_config_hash,
            Transcript.preprocessing_profile == preprocessing_profile,
            Transcript.transcription_mode == transcription_mode,
            Transcript.status == TranscriptStatus.COMPLETED,
        )
        .order_by(*_transcript_recency_order())
        .limit(1)
        .correlate(None)
        .scalar_subquery()
    )
    rows = (
        await session.execute(
            select(Transcript, TranscriptSegment)
            .outerjoin(
                TranscriptSegment,
                TranscriptSegment.transcript_id == Transcript.id,
            )
            .where(Transcript.id == latest_id)
            .order_by(
                TranscriptSegment.sequence_number,
                TranscriptSegment.id,
            )
        )
    ).all()
    if not rows:
        return None
    source = rows[0][0]
    segments = tuple(row[1] for row in rows if row[1] is not None)
    return source, segments


async def _keyword_definitions(
    session: AsyncSession, job: ProcessingJob
) -> tuple[list[KeywordDefinition], dict[str, Keyword]]:
    conditions = [Keyword.active.is_(True), Keyword.deleted_at.is_(None)]
    if job.selected_category_ids:
        conditions.append(
            Keyword.category_id.in_([UUID(item) for item in job.selected_category_ids])
        )
    keywords = (
        await session.scalars(
            select(Keyword)
            .options(selectinload(Keyword.variants), selectinload(Keyword.category))
            .where(*conditions)
        )
    ).all()
    definitions = [
        KeywordDefinition(
            id=str(keyword.id),
            phrase=keyword.canonical_phrase,
            variants=tuple(item.phrase for item in keyword.variants),
            accent_insensitive=keyword.accent_insensitive,
            whole_word=keyword.whole_word,
            exact_phrase=keyword.exact_phrase,
            fuzzy_match=keyword.fuzzy_match,
            fuzzy_threshold=keyword.fuzzy_threshold,
        )
        for keyword in keywords
    ]
    return definitions, {str(item.id): item for item in keywords}


async def search_and_persist_matches(
    session: AsyncSession, transcript: Transcript, job: ProcessingJob
) -> int:
    definitions, keyword_by_id = await _keyword_definitions(session, job)
    if not definitions:
        return 0
    segments = (
        await session.scalars(
            select(TranscriptSegment).where(TranscriptSegment.transcript_id == transcript.id)
        )
    ).all()
    segment_ids = [item.id for item in segments]
    existing = set()
    if segment_ids:
        rows = (
            await session.execute(
                select(
                    KeywordMatch.keyword_id,
                    KeywordMatch.transcript_segment_id,
                    KeywordMatch.start_seconds,
                    KeywordMatch.normalized_match,
                ).where(KeywordMatch.transcript_segment_id.in_(segment_ids))
            )
        ).all()
        existing = {(str(a), str(b), c, d) for a, b, c, d in rows}
    added = 0
    for segment in segments:
        # Diarized labels remain unknown. They are intentionally excluded unless all speakers was explicit.
        if not job.include_all_speakers and segment.operator_id is None:
            continue
        if not job.include_all_speakers and segment.operator_id != transcript.operator_id:
            continue
        for found in match_text(segment.original_text, definitions):
            keyword = keyword_by_id[found.keyword_id]
            key = (
                found.keyword_id,
                str(segment.id),
                segment.start_seconds,
                found.normalized_match,
            )
            if key in existing:
                continue
            session.add(
                KeywordMatch(
                    keyword_id=keyword.id,
                    operator_id=segment.operator_id,
                    call_id=transcript.call_id,
                    transcript_segment_id=segment.id,
                    original_matched_text=found.original_matched_text,
                    normalized_match=found.normalized_match,
                    context_before=found.context_before,
                    context_after=found.context_after,
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    match_method=found.method,
                    match_score=Decimal(str(round(found.score, 3))),
                )
            )
            existing.add(key)
            added += 1
    await session.flush()
    return added


async def _vocabulary(
    session: AsyncSession,
    operator: Operator | None,
    settings: Settings,
) -> list[str]:
    # An unattributed dual-channel transcript can be shared by multiple selected
    # operator items.  Do not make its prompt identity depend on which item wins
    # the recording lease.
    values = [operator.display_name] if operator is not None else []
    application = await load_application_settings(session, settings)
    company = str(application.get("company_vocabulary") or "")
    values.extend(item.strip() for item in company.replace("\n", ",").split(",") if item.strip())
    keywords = (
        await session.scalars(
            select(Keyword)
            .options(selectinload(Keyword.variants))
            .where(Keyword.active.is_(True))
            .order_by(Keyword.canonical_phrase, Keyword.id)
        )
    ).all()
    for keyword in keywords:
        values.append(keyword.canonical_phrase)
        values.extend(_ordered_keyword_variant_phrases(keyword))
    operator_names = (
        await session.scalars(
            select(Operator.display_name).order_by(Operator.display_name, Operator.id)
        )
    ).all()
    values.extend(operator_names)
    return values


async def _v2_ranked_vocabulary(
    session: AsyncSession,
    *,
    job: ProcessingJob,
    call: Call,
    operator: Operator,
    settings: Settings,
) -> tuple[RankedVocabularyTerm, ...]:
    """Collect only current-call terms for the V2 contextual prompt."""

    terms: list[RankedVocabularyTerm] = [
        RankedVocabularyTerm(
            value=operator.display_name,
            priority=PRIORITY_SELECTED_OPERATOR,
            source=VOCABULARY_SOURCE_SELECTED_OPERATOR,
            roles=(TRACK_ROLE_OPERATOR,),
        )
    ]
    if call.caller_name:
        terms.append(
            RankedVocabularyTerm(
                value=call.caller_name,
                priority=PRIORITY_CURRENT_PARTY,
                source=VOCABULARY_SOURCE_CURRENT_PARTY,
                roles=(TRACK_ROLE_CALLER,),
            )
        )
    if call.callee_name:
        terms.append(
            RankedVocabularyTerm(
                value=call.callee_name,
                priority=PRIORITY_CURRENT_PARTY,
                source=VOCABULARY_SOURCE_CURRENT_PARTY,
                roles=(TRACK_ROLE_CALLEE,),
            )
        )
    for current_call_value in (call.caller_number, call.callee_number):
        if current_call_value:
            terms.append(
                RankedVocabularyTerm(
                    value=current_call_value,
                    priority=PRIORITY_CURRENT_CALL,
                    source=VOCABULARY_SOURCE_CURRENT_CALL,
                )
            )

    if job.selected_category_ids:
        selected_category_ids = [UUID(category_id) for category_id in job.selected_category_ids]
        keywords = (
            await session.scalars(
                select(Keyword)
                .options(selectinload(Keyword.variants))
                .where(
                    Keyword.active.is_(True),
                    Keyword.deleted_at.is_(None),
                    Keyword.category_id.in_(selected_category_ids),
                )
                .order_by(Keyword.canonical_phrase, Keyword.id)
            )
        ).all()
        for keyword in keywords:
            terms.append(
                RankedVocabularyTerm(
                    value=keyword.canonical_phrase,
                    priority=PRIORITY_SELECTED_KEYWORD,
                    source=VOCABULARY_SOURCE_SELECTED_KEYWORD,
                )
            )
            terms.extend(
                RankedVocabularyTerm(
                    value=variant,
                    priority=PRIORITY_SELECTED_KEYWORD,
                    source=VOCABULARY_SOURCE_SELECTED_KEYWORD,
                )
                for variant in _ordered_keyword_variant_phrases(keyword)
            )

    application = await load_application_settings(session, settings)
    company = str(application.get("company_vocabulary") or "")
    terms.extend(
        RankedVocabularyTerm(
            value=value,
            priority=PRIORITY_COMPANY,
            source=VOCABULARY_SOURCE_COMPANY,
        )
        for value in (item.strip() for item in company.replace("\n", ",").split(","))
        if value
    )
    if call.queue_name:
        terms.append(
            RankedVocabularyTerm(
                value=call.queue_name,
                priority=PRIORITY_QUEUE,
                source=VOCABULARY_SOURCE_QUEUE,
            )
        )
    terms.extend(
        RankedVocabularyTerm(
            value=value,
            priority=PRIORITY_GENERAL,
            source=VOCABULARY_SOURCE_GENERAL,
        )
        for value in GENERAL_DEALERSHIP_VOCABULARY
    )
    return tuple(terms)


def _ordered_keyword_variant_phrases(keyword: Keyword) -> list[str]:
    """Stabilize the legacy prompt without semantically reordering its variants."""

    return [
        item.phrase
        for item in sorted(
            keyword.variants,
            key=lambda item: (
                item.created_at,
                str(item.id),
            ),
        )
    ]


async def process_item(item_id: UUID) -> None:
    redis = get_redis()
    settings = get_settings()
    lock = None
    transcription_slot = None
    temporary_files: list[Path] = []
    temporary_directories: set[Path] = set()
    transcript_id: UUID | None = None
    provider_result: TranscriptionResult | OrchestratedTranscriptionResult | None = None
    try:
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item is None or item.status not in {
                ItemStatus.QUEUED,
                ItemStatus.WAITING_FOR_CONNECTION,
                ItemStatus.PROCESSING,
            }:
                return
            job = await session.scalar(
                select(ProcessingJob).where(ProcessingJob.id == item.job_id).with_for_update()
            )
            if job is None:
                return
            _require_processable_job(job)
            item = await session.scalar(
                select(ProcessingJobItem)
                .where(ProcessingJobItem.id == item_id)
                .with_for_update(skip_locked=True)
            )
            if item is None or item.status not in {
                ItemStatus.QUEUED,
                ItemStatus.WAITING_FOR_CONNECTION,
                ItemStatus.PROCESSING,
            }:
                return
            if item.status == ItemStatus.PROCESSING and _worker_timestamp_is_recent(
                item.heartbeat_at
            ):
                raise ProcessingBusyError("Analysis item is already being processed.")
            if job.started_at is None:
                job.started_at = utc_now()
            item.status = ItemStatus.PROCESSING
            item.attempt_count += 1
            item.locked_at = utc_now()
            item.heartbeat_at = utc_now()
            await session.commit()
            call = await session.get(Call, item.call_id)
            operator = await session.get(Operator, item.operator_id)
            recording = (
                await session.get(Recording, item.recording_id) if item.recording_id else None
            )
            if not call or not operator:
                raise RuntimeError("Processing item references missing data")
            if recording is None:
                # A task from an older release can still arrive after its
                # one-per-call placeholder has been migrated.  It must not
                # become a failed analysis item simply because no recording
                # has been assigned yet; full discovery owns that transition.
                await _lock_processable_item(session, item, job)
                item.status = ItemStatus.QUEUED
                item.stage = "waiting_for_recording_assignment"
                item.error_category = None
                item.error_message = None
                item.completed_at = None
                await session.commit()
                return
            reprocess_target_id = _reprocess_target_id(job, item, settings)
            reprocess_target: Transcript | None = None
            if reprocess_target_id is not None:
                reprocess_target = await session.scalar(
                    select(Transcript).where(
                        Transcript.id == reprocess_target_id,
                        Transcript.call_id == call.id,
                        Transcript.recording_id == recording.id,
                        Transcript.status == TranscriptStatus.COMPLETED,
                    )
                )
                if reprocess_target is None:
                    raise ReprocessStateError(
                        "The completed transcript selected for reprocessing is unavailable."
                    )
                if reprocess_target.operator_id not in {None, operator.id}:
                    raise ReprocessStateError(
                        "The selected transcript does not match the requested operator."
                    )
                if not reprocess_target.is_current:
                    completed_replacement = await session.scalar(
                        select(Transcript.id).where(
                            Transcript.supersedes_transcript_id == reprocess_target.id,
                            Transcript.call_id == call.id,
                            Transcript.recording_id == recording.id,
                            Transcript.pipeline_version == item.requested_pipeline_version,
                            Transcript.status == TranscriptStatus.COMPLETED,
                            Transcript.is_current.is_(True),
                        )
                    )
                    if completed_replacement is None:
                        raise ReprocessStateError("The selected transcript is no longer current.")
            lock = redis.lock(
                f"yca:recording-processing:{recording.id}",
                timeout=WORKER_LEASE_SECONDS,
                blocking_timeout=1,
            )
            if not await lock.acquire(blocking=False):
                await _lock_processable_item(session, item, job)
                item.status = ItemStatus.QUEUED
                await session.commit()
                raise ProcessingBusyError()

            async def lease_check() -> None:
                await _refresh_available_processing_leases(lock, transcription_slot)

            completed_transcript = None
            if reprocess_target_id is None:
                completed_transcript = await session.scalar(
                    select(Transcript)
                    .where(
                        Transcript.recording_id == recording.id,
                        Transcript.status == TranscriptStatus.COMPLETED,
                        Transcript.is_current.is_(True),
                        or_(
                            Transcript.operator_id == operator.id,
                            Transcript.operator_id.is_(None),
                        ),
                    )
                    .order_by(
                        (Transcript.operator_id == operator.id).desc(),
                        *_transcript_recency_order(),
                    )
                )
            if completed_transcript is not None:
                # Ordinary analysis preserves the paid-transcript reuse policy.
                # Explicit reprocessing bypasses this branch and creates a
                # separately versioned replacement.
                await _item_stage(
                    session,
                    item,
                    job,
                    JobStatus.SEARCHING_KEYWORDS,
                    "Searching for phrases",
                    before_commit=lease_check,
                )
                await _lock_recording_rows(session, [recording.id])
                await lease_check()
                await search_and_persist_matches(session, completed_transcript, job)
                await lease_check()
                await _lock_processable_item(session, item, job)
                await lease_check()
                attribution_unknown = _transcript_has_attribution_warning(completed_transcript)
                call.processing_status = (
                    "completed_speaker_attribution_unknown" if attribution_unknown else "completed"
                )
                call.last_error_category = (
                    "speaker_attribution_unknown" if attribution_unknown else None
                )
                call.last_error_message = _attribution_warning_message(completed_transcript)
                item.status = ItemStatus.COMPLETED
                item.stage = "completed"
                item.completed_at = utc_now()
                item.result_transcript_id = completed_transcript.id
                item.error_category = call.last_error_category
                item.error_message = call.last_error_message
                await session.commit()
                return
            try:
                settings = await load_effective_yeastar_settings(redis, settings)
            except RedisError as exc:
                raise YeastarConfigurationStateError(
                    "Phone-system configuration is temporarily unavailable."
                ) from exc
            try:
                settings = await load_effective_openai_settings(redis, settings)
            except (RedisError, OpenAIConfigurationStateError) as exc:
                raise TranscriptionConfigurationError(
                    "OpenAI configuration is temporarily unavailable.",
                    "configuration_state",
                ) from exc
            audio = AudioProcessor(settings)
            await _item_stage(
                session,
                item,
                job,
                JobStatus.DOWNLOADING_RECORDINGS,
                "Downloading recordings",
                before_commit=lease_check,
            )
            suffix = Path(recording.yeastar_file_name or "").suffix.lower()
            if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".opus"}:
                raise AudioError("Recording file type is not supported.")
            storage_key = recording.storage_key or f"recordings/{recording.id}{suffix}"
            source_path = audio.safe_storage_path(storage_key)
            if not source_path.exists():
                if reprocess_target_id is None:
                    recording.status = RecordingStatus.DOWNLOADING
                try:
                    async with YeastarClient(settings=settings, redis=redis) as yeastar:
                        download = await yeastar.download_recording(
                            recording.yeastar_recording_id, source_path
                        )
                except YeastarRecordingDownloadLimitError as exc:
                    await lease_check()
                    await _lock_processable_item(session, item, job)
                    await lease_check()
                    if reprocess_target_id is None:
                        recording.status = RecordingStatus.DISCOVERED
                    item.status = ItemStatus.QUEUED
                    item.stage = "waiting_for_recording_capacity"
                    job.current_stage = "Waiting for phone-system recording capacity"
                    await session.commit()
                    # The Celery task's existing bounded busy retry owns the
                    # deferred queue. The shared token remains cached and the
                    # authentication circuit is untouched by error 70651.
                    raise ProcessingBusyError() from exc
                await lease_check()
                await _lock_recording_rows(session, [recording.id])
                await lease_check()
                recording.storage_key = storage_key
                recording.size_bytes = int(download["size_bytes"])
                recording.mime_type = str(download.get("content_type") or "")[:128] or None
                recording.file_extension = suffix
                recording.downloaded_at = utc_now()
                if reprocess_target_id is None:
                    recording.status = RecordingStatus.DOWNLOADED
                await session.commit()
            await _item_stage(
                session,
                item,
                job,
                JobStatus.INSPECTING_AUDIO,
                "Checking recording quality",
                before_commit=lease_check,
            )
            info = await audio.inspect(
                source_path,
                declared_mime_type=recording.mime_type,
                original_filename=recording.yeastar_file_name,
            )
            await lease_check()
            await _lock_recording_rows(session, [recording.id])
            await lease_check()
            recording.codec_name = info.codec_name
            recording.duration_seconds = info.duration_seconds
            recording.channel_count = info.channel_count
            recording.sample_rate_hz = info.sample_rate_hz
            recording.bit_rate_bps = info.bit_rate_bps
            recording.size_bytes = info.size_bytes
            recording.sha256_checksum = info.sha256_checksum
            if reprocess_target_id is None:
                recording.status = RecordingStatus.INSPECTED
            recording.inspected_at = utc_now()
            await session.commit()
            participants = (
                await session.scalars(
                    select(CallParticipant).where(CallParticipant.call_id == call.id)
                )
            ).all()
            selected_participants = [
                item for item in participants if item.operator_id == operator.id
            ]
            legs = (await session.scalars(select(CallLeg).where(CallLeg.call_id == call.id))).all()
            pipeline_version = _effective_pipeline_version(item, settings)
            pipeline_v2 = pipeline_version == PIPELINE_V2_VERSION
            separated = False
            operator_channel: int | None = None
            caller_channel: int | None = None
            callee_channel: int | None = None
            should_read_stereo_capability = info.channel_count == 2 and (
                pipeline_v2 or len(selected_participants) == 1
            )
            if should_read_stereo_capability:
                try:
                    async with YeastarClient(settings=settings, redis=redis) as yeastar:
                        separated = await yeastar.stereo_separated_recording_enabled()
                except YeastarError:
                    separated = False
            caller_callee_mapping = safe_caller_callee_channels(
                info.channel_count,
                separated,
                one_to_one=(len(legs) == 1 and call.queue_name is None),
                was_transferred=call.was_transferred,
            )
            if caller_callee_mapping is not None:
                caller_channel, callee_channel = caller_callee_mapping
            if info.channel_count == 2 and len(selected_participants) == 1:
                participant = selected_participants[0]
                interpreted = InterpretedParticipant(
                    operator_id=str(operator.id),
                    provider_extension_id=participant.provider_extension_id or "",
                    extension_number=participant.provider_extension_number
                    or operator.extension_number,
                    leg_id=str(participant.call_leg_id or ""),
                    role=participant.role,
                    was_caller=bool(participant.operator_was_caller),
                    was_callee=bool(participant.operator_was_callee),
                    answered=participant.answered,
                )
                same_side = sum(
                    1
                    for candidate in participants
                    if candidate.operator_id is not None
                    and (
                        bool(candidate.operator_was_caller) == interpreted.was_caller
                        and bool(candidate.operator_was_callee) == interpreted.was_callee
                    )
                )
                operator_channel = safe_operator_channel(
                    interpreted,
                    info.channel_count,
                    separated,
                    one_to_one=(len(legs) == 1 and call.queue_name is None),
                    was_transferred=call.was_transferred,
                    operators_on_same_side=same_side,
                )
            temp_dir = audio.safe_storage_path(f"tmp/{item.id}-{secrets.token_hex(6)}")
            temp_dir.mkdir(parents=True, exist_ok=False)
            temporary_directories.add(temp_dir)
            app_settings = await load_application_settings(session, settings)
            language = str(app_settings["default_language"])
            if pipeline_v2:
                prepared = source_path
                planner = TopologyAudioPlanner()
                orchestrator = TranscriptionOrchestrator(
                    settings=settings,
                    audio_processor=audio,
                    planner=planner,
                    client_factory=lambda: OpenAITranscriptionClient(settings=settings),
                )
                context = TranscriptionContext(
                    diarized=False,
                    language=language,
                    vocabulary=(),
                    temporary_directory=temp_dir,
                    operator_id=str(operator.id),
                    stereo_separated=separated,
                    operator_channel=operator_channel,
                    caller_channel=caller_channel,
                    callee_channel=callee_channel,
                    operator_display_name=operator.display_name,
                    register_temporary_file=temporary_files.append,
                )
                audio_plan = _plan_with_orchestrator(
                    orchestrator,
                    planner,
                    source_path=prepared,
                    audio_info=info,
                    context=context,
                )
            else:
                await _item_stage(
                    session,
                    item,
                    job,
                    JobStatus.EXTRACTING_OPERATOR_AUDIO,
                    (
                        "Preparing the operator conversation"
                        if operator_channel is not None
                        else "Separating speakers"
                    ),
                    before_commit=lease_check,
                )
                prepared = temp_dir / "prepared.wav"
                temporary_files.append(prepared)
                if operator_channel is not None:
                    await audio.extract_channel(
                        source_path,
                        prepared,
                        operator_channel,
                    )
                    legacy_diarized = False
                else:
                    await audio.convert_to_mono(source_path, prepared)
                    legacy_diarized = True
                await lease_check()
                legacy_attribution = (
                    SpeakerAttributionStatus.ANONYMOUS_DIARIZATION
                    if legacy_diarized
                    else SpeakerAttributionStatus.CONFIRMED_BY_PBX
                )
                planner = LegacyAudioPlanner()
                orchestrator = TranscriptionOrchestrator(
                    settings=settings,
                    audio_processor=audio,
                    planner=planner,
                    client_factory=lambda: OpenAITranscriptionClient(settings=settings),
                )
                context = TranscriptionContext(
                    diarized=legacy_diarized,
                    language=language,
                    vocabulary=(),
                    temporary_directory=temp_dir,
                    channel_index=operator_channel,
                    operator_id=(None if legacy_diarized else str(operator.id)),
                    attribution_status=legacy_attribution.value,
                    audio_variant=("legacy-mono" if legacy_diarized else "legacy-operator-channel"),
                    operator_display_name=operator.display_name,
                    register_temporary_file=temporary_files.append,
                )
                audio_plan = _plan_with_orchestrator(
                    orchestrator,
                    planner,
                    source_path=prepared,
                    audio_info=info,
                    context=context,
                )
            if pipeline_v2:
                await _item_stage(
                    session,
                    item,
                    job,
                    JobStatus.EXTRACTING_OPERATOR_AUDIO,
                    (
                        "Preparing the operator conversation"
                        if audio_plan.mode == "operator_channel"
                        else (
                            "Preparing separated audio channels"
                            if audio_plan.mode == "dual_channel"
                            else "Separating speakers"
                        )
                    ),
                    before_commit=lease_check,
                )
            await lease_check()
            transcription_mode = TranscriptionMode(audio_plan.mode)
            diarized = transcription_mode in {
                TranscriptionMode.MONO_DIARIZATION,
            } or (transcription_mode == TranscriptionMode.LEGACY and audio_plan.tracks[0].diarized)
            # Pipeline V2 mono uses diarization only for authoritative speaker
            # placement; its final text comes from the configured standard
            # transcription model. Legacy diarization keeps its original model.
            model = (
                settings.OPENAI_TRANSCRIPTION_MODEL
                if pipeline_v2 or not diarized
                else settings.OPENAI_DIARIZATION_MODEL
            )
            operator_attributed = transcription_mode in {
                TranscriptionMode.OPERATOR_CHANNEL,
            } or (transcription_mode == TranscriptionMode.LEGACY and not diarized)
            transcript_operator_id = operator.id if operator_attributed else None
            if (
                reprocess_target_id is not None
                and reprocess_target is not None
                and reprocess_target.operator_id
                not in {
                    None,
                    transcript_operator_id,
                }
            ):
                raise ReprocessStateError(
                    "Reprocessing cannot remove or change an attributed operator."
                )
            v2_prompt_manifest: V2PromptManifest | None = None
            if pipeline_v2:
                language = "el"
                ranked_vocabulary = await _v2_ranked_vocabulary(
                    session,
                    job=job,
                    call=call,
                    operator=operator,
                    settings=settings,
                )
                v2_prompt_manifest = orchestrator.v2_prompt_builder.build_manifest(
                    audio_plan,
                    ranked_vocabulary,
                )
                vocabulary: list[str] = []
                prompt_version = v2_prompt_manifest.prompt_identity
                prompt_template_version = v2_prompt_manifest.template_version
                vocabulary_hash = v2_prompt_manifest.vocabulary_hash
            else:
                vocabulary = (
                    []
                    if diarized
                    else await _vocabulary(
                        session,
                        operator if operator_attributed else None,
                        settings,
                    )
                )
                prompt_version = None if diarized else build_vocabulary_prompt(vocabulary)[1]
                prompt_template_version = (
                    LEGACY_DIARIZED_PROMPT_TEMPLATE_VERSION
                    if diarized
                    else LEGACY_ISOLATED_PROMPT_TEMPLATE_VERSION
                )
                vocabulary_hash = None if diarized else prompt_version
            persisted_prompt_template_version = (
                prompt_template_version if v2_prompt_manifest is not None else None
            )
            persisted_vocabulary_hash = vocabulary_hash if v2_prompt_manifest is not None else None
            context = replace(
                context,
                diarized=diarized,
                language=language,
                vocabulary=tuple(vocabulary),
                attribution_status=audio_plan.attribution_status,
                v2_prompt_manifest=v2_prompt_manifest,
            )
            attribution_status = SpeakerAttributionStatus(audio_plan.attribution_status)
            preprocessing_profile = (
                (
                    PIPELINE_V2_STANDARD_PREPROCESSING_PROFILE
                    if audio_plan.mode in {"operator_channel", "dual_channel"}
                    else PIPELINE_V2_MONO_PREPROCESSING_PROFILE
                )
                if pipeline_v2
                else LEGACY_PREPROCESSING_PROFILE
            )
            historical_pipeline_config_hash = (
                (
                    LEGACY_DIARIZED_PIPELINE_CONFIG_HASH
                    if diarized
                    else LEGACY_ISOLATED_PIPELINE_CONFIG_HASH
                )
                if not pipeline_v2
                else None
            )
            segmentation_identity = (
                _segmentation_identity_for_plan(
                    orchestrator,
                    audio_plan,
                    max_upload_bytes=settings.max_transcription_upload_bytes,
                )
                if pipeline_v2
                else None
            )
            pipeline_config_hash = (
                _pipeline_v2_runtime_config_hash(
                    plan=audio_plan,
                    max_upload_bytes=settings.max_transcription_upload_bytes,
                    prompt_template_version=prompt_template_version,
                    segmentation_identity=segmentation_identity,
                    prompt_identity=(
                        v2_prompt_manifest.prompt_identity
                        if v2_prompt_manifest is not None
                        else None
                    ),
                    vocabulary_hash=vocabulary_hash,
                    standard_model=settings.OPENAI_TRANSCRIPTION_MODEL,
                    diarization_model=settings.OPENAI_DIARIZATION_MODEL,
                )
                if pipeline_v2
                else _legacy_runtime_pipeline_config_hash(
                    diarized=diarized,
                    channel_index=operator_channel,
                    max_upload_bytes=settings.max_transcription_upload_bytes,
                    prompt_template_version=prompt_template_version,
                )
            )
            key = transcript_idempotency_key(
                recording.id,
                transcript_operator_id,
                model,
                info.sha256_checksum,
                diarized,
                language,
                prompt_version,
                pipeline_version=pipeline_version,
                pipeline_config_hash=pipeline_config_hash,
                transcription_mode=transcription_mode,
                prompt_template_version=prompt_template_version,
                vocabulary_hash=vocabulary_hash,
                supersedes_transcript_id=reprocess_target_id,
            )
            # Rediscovery and retention use Recording -> Job -> Transcript.
            # Take the same database fence before mutating transcript identity
            # or reprocess ownership so an ORM flush cannot invert that order.
            await _lock_recording_rows(session, [recording.id])
            await lease_check()
            transcript = await session.scalar(
                select(Transcript).where(Transcript.idempotency_key == key)
            )
            if transcript is None and not pipeline_v2:
                historical_versioned_key = transcript_idempotency_key(
                    recording.id,
                    transcript_operator_id,
                    model,
                    info.sha256_checksum,
                    diarized,
                    language,
                    prompt_version,
                    pipeline_version=pipeline_version,
                    pipeline_config_hash=historical_pipeline_config_hash,
                    transcription_mode=TranscriptionMode.LEGACY,
                    prompt_template_version=prompt_template_version,
                    vocabulary_hash=vocabulary_hash,
                    supersedes_transcript_id=reprocess_target_id,
                )
                transcript = await session.scalar(
                    select(Transcript).where(Transcript.idempotency_key == historical_versioned_key)
                )
                if transcript is not None:
                    # Phase 1 persisted a coarser legacy configuration identity.
                    # Upgrade an exact in-flight/completed match in place so
                    # hardening does not create another provider attempt.
                    transcript.idempotency_key = key
                    transcript.pipeline_config_hash = pipeline_config_hash
            if transcript is None and reprocess_target_id is None and not pipeline_v2:
                legacy_key = transcript_idempotency_key(
                    recording.id,
                    transcript_operator_id,
                    model,
                    info.sha256_checksum,
                    diarized,
                    language,
                    prompt_version,
                )
                transcript = await session.scalar(
                    select(Transcript).where(
                        Transcript.idempotency_key == legacy_key,
                        Transcript.status != TranscriptStatus.COMPLETED,
                    )
                )
                if transcript is not None:
                    # Carry an unfinished pre-versioning attempt forward rather
                    # than silently creating an extra row/provider attempt.
                    transcript.idempotency_key = key
                    transcript.transcription_mode = TranscriptionMode.LEGACY
                    transcript.speaker_attribution_status = attribution_status
                    transcript.pipeline_version = pipeline_version
                    transcript.pipeline_config_hash = pipeline_config_hash
                    transcript.preprocessing_profile = LEGACY_PREPROCESSING_PROFILE
            if transcript is not None and await _archive_failed_v2_attempt_history(
                session,
                transcript,
                stable_key=key,
                transcription_mode=transcription_mode,
            ):
                # A failed run with paid provider evidence is immutable audit
                # history. Release the stable execution key and current slot so
                # a later retry receives a fresh transcript row instead of
                # deleting or mixing attempts from separate runs.
                transcript = None
            if transcript is not None and reprocess_target_id is not None:
                _record_reprocess_replacement(job, item, transcript)
            transcript_id = transcript.id if transcript else None
            call_leg_id = (
                selected_participants[0].call_leg_id if len(selected_participants) == 1 else None
            )
            needs_transcription = False
            if transcript is not None and transcript.status != TranscriptStatus.COMPLETED:
                if reprocess_target_id is None and not transcript.is_current:
                    await _retire_noncompleted_current_target(
                        session,
                        recording_id=recording.id,
                        operator_id=transcript_operator_id,
                    )
                    transcript.is_current = True
                await session.execute(
                    delete(TranscriptSegment).where(
                        TranscriptSegment.transcript_id == transcript.id
                    )
                )
                await session.execute(
                    delete(TranscriptionAttempt).where(
                        TranscriptionAttempt.transcript_id == transcript.id
                    )
                )
                transcript.prompt_template_version = persisted_prompt_template_version
                transcript.vocabulary_hash = persisted_vocabulary_hash
                transcript.status = TranscriptStatus.PROCESSING
                transcript.attempt_count += 1
                transcript.error_category = None
                transcript.error_message = None
                transcript.completed_at = None
                needs_transcription = True
                await lease_check()
                await session.commit()
            if (
                transcript is None
                and reprocess_target_id is None
                and _cross_recording_transcript_clone_allowed(transcription_mode)
            ):
                duplicate_snapshot = await _completed_duplicate_snapshot(
                    session,
                    checksum=info.sha256_checksum,
                    model=model,
                    diarized=diarized,
                    operator_id=transcript_operator_id,
                    language=language,
                    prompt_version=prompt_version,
                    pipeline_version=pipeline_version,
                    pipeline_config_hash=pipeline_config_hash,
                    preprocessing_profile=preprocessing_profile,
                    transcription_mode=transcription_mode,
                    prompt_template_version=persisted_prompt_template_version,
                    vocabulary_hash=persisted_vocabulary_hash,
                )
                if duplicate_snapshot is not None:
                    duplicate, duplicate_segments = duplicate_snapshot
                    await _retire_noncompleted_current_target(
                        session,
                        recording_id=recording.id,
                        operator_id=transcript_operator_id,
                    )
                    transcript = await _clone_transcript(
                        session,
                        duplicate,
                        duplicate_segments,
                        call=call,
                        recording=recording,
                        operator=operator if operator_attributed else None,
                        call_leg_id=call_leg_id if operator_attributed else None,
                        key=key,
                    )
                    transcript.pipeline_config_hash = pipeline_config_hash
                    transcript_id = transcript.id
                    await lease_check()
                    await session.commit()
            if transcript is None:
                if reprocess_target_id is None:
                    await _retire_noncompleted_current_target(
                        session,
                        recording_id=recording.id,
                        operator_id=transcript_operator_id,
                    )
                transcript = Transcript(
                    call_id=call.id,
                    recording_id=recording.id,
                    operator_id=transcript_operator_id,
                    idempotency_key=key,
                    status=TranscriptStatus.PROCESSING,
                    model=model,
                    language=language,
                    prompt_version=prompt_version,
                    prompt_template_version=persisted_prompt_template_version,
                    vocabulary_hash=persisted_vocabulary_hash,
                    attempt_count=1,
                    source_audio_sha256=info.sha256_checksum,
                    is_diarized=diarized,
                    transcription_mode=transcription_mode,
                    speaker_attribution_status=attribution_status,
                    pipeline_version=pipeline_version,
                    pipeline_config_hash=pipeline_config_hash,
                    preprocessing_profile=preprocessing_profile,
                    is_current=reprocess_target_id is None,
                )
                session.add(transcript)
                await session.flush()
                if reprocess_target_id is not None:
                    _record_reprocess_replacement(job, item, transcript)
                await lease_check()
                await session.commit()
                transcript_id = transcript.id
                needs_transcription = True
            if needs_transcription:
                slot_limit = int(app_settings["max_parallel_transcriptions"])
                for slot_number in range(slot_limit):
                    candidate = redis.lock(
                        f"yca:transcription-slot:{slot_number}",
                        timeout=WORKER_LEASE_SECONDS,
                        blocking_timeout=0,
                    )
                    if await candidate.acquire(blocking=False):
                        transcription_slot = candidate
                        break
                if transcription_slot is None:
                    await lease_check()
                    await _lock_processable_item(session, item, job)
                    await lease_check()
                    item.status = ItemStatus.QUEUED
                    item.stage = "waiting_for_transcription_capacity"
                    await session.commit()
                    raise ProcessingBusyError()
                await _item_stage(
                    session,
                    item,
                    job,
                    JobStatus.TRANSCRIBING,
                    "Transcribing conversations",
                    before_commit=lease_check,
                )

                async def cancellation_check() -> bool:
                    await _refresh_processing_leases(lock, transcription_slot)
                    return await _cancel_requested(session, job.id)

                provider_result = await orchestrator.transcribe(
                    source_path=prepared,
                    audio_info=info,
                    context=context,
                    plan=audio_plan,
                    cancellation_check=cancellation_check,
                )
                if await cancellation_check():
                    raise TranscriptionCancelledError("Transcription was cancelled.")
                await _lock_recording_rows(session, [recording.id])
                await lease_check()
                await _persist_transcription_result(
                    session,
                    transcript,
                    provider_result,
                    call,
                    operator if operator_attributed else None,
                    call_leg_id if operator_attributed else None,
                    info.duration_seconds,
                )
                await lease_check()
                await session.commit()
            await _item_stage(
                session,
                item,
                job,
                JobStatus.SEARCHING_KEYWORDS,
                "Searching for phrases",
                before_commit=lease_check,
            )
            await _lock_recording_rows(session, [recording.id])
            await lease_check()
            await search_and_persist_matches(session, transcript, job)
            await lease_check()
            await _lock_processable_item(session, item, job)
            await lease_check()
            if reprocess_target_id is not None and not transcript.is_current:
                if not _item_owns_reprocess_replacement(job, item, transcript):
                    raise ReprocessStateError(
                        "The replacement transcript is not owned by this processing item."
                    )
                await _activate_transcript_replacement(
                    session,
                    transcript,
                    reprocess_target_id,
                )
            await lease_check()
            recording.status = RecordingStatus.COMPLETED
            attribution_unknown = _transcript_has_attribution_warning(transcript)
            call.processing_status = (
                "completed_speaker_attribution_unknown" if attribution_unknown else "completed"
            )
            call.last_error_category = (
                "speaker_attribution_unknown" if attribution_unknown else None
            )
            call.last_error_message = _attribution_warning_message(transcript)
            item.status = ItemStatus.COMPLETED
            item.stage = "completed"
            item.completed_at = utc_now()
            item.result_transcript_id = transcript.id
            item.error_category = "speaker_attribution_unknown" if attribution_unknown else None
            item.error_message = call.last_error_message
            await session.commit()
            if bool(app_settings["delete_audio_after_transcription"]):
                await lease_check()
                await _lock_recording_rows(session, [recording.id])
                await _lock_recording_job_rows(session, {recording.id})
                await lease_check()
                await session.refresh(recording)
                outstanding_for_recording = (
                    await session.scalar(
                        select(func.count())
                        .select_from(ProcessingJobItem)
                        .where(
                            ProcessingJobItem.recording_id == recording.id,
                            ProcessingJobItem.id != item.id,
                            ProcessingJobItem.status.not_in(FINAL_ITEM_STATES),
                        )
                    )
                    or 0
                )
                if not outstanding_for_recording and recording.storage_key:
                    cleanup_path = audio.safe_storage_path(recording.storage_key)
                    _, failures = audio.remove_files([cleanup_path])
                    await lease_check()
                    if not failures:
                        recording.storage_key = None
                        recording.status = RecordingStatus.DELETED
                        recording.deleted_at = utc_now()
                        session.add(
                            AuditLog(
                                created_at=utc_now(),
                                user_id=None,
                                action="audio.delete",
                                resource_type="recording",
                                resource_id=str(recording.id),
                                outcome="success",
                                details={"reason": "after_transcription"},
                            )
                        )
                    else:
                        recording.last_error_category = "audio_cleanup_pending"
                        recording.last_error_message = "Audio cleanup will be retried."
                await lease_check()
                await session.commit()
    except CONNECTION_PAUSE_ERRORS as exc:
        await _refresh_available_processing_leases(lock, transcription_slot)
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item:
                disposition = await _settle_stale_item(session, item, transcript_id)
                if disposition != "active":
                    return
                await _refresh_available_processing_leases(lock, transcription_slot)
                item.status = ItemStatus.WAITING_FOR_CONNECTION
                item.stage = "waiting_for_connection"
                item.error_category = getattr(exc, "category", "connection_paused")
                item.error_message = "Test the phone-system connection in Settings."
                item.completed_at = None
                job = await session.get(ProcessingJob, item.job_id)
                if job:
                    job.status = JobStatus.WAITING_FOR_CONNECTION
                    job.current_stage = "Waiting for phone-system connection"
                    job.completed_at = None
                recording = (
                    await session.get(Recording, item.recording_id) if item.recording_id else None
                )
                if recording and recording.status == RecordingStatus.DOWNLOADING:
                    recording.status = RecordingStatus.DISCOVERED
                await session.commit()
    except ProcessingBusyError:
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item and await _settle_stale_item(session, item, transcript_id) != "active":
                return
        raise
    except (ProcessingCancelledError, TranscriptionCancelledError) as exc:
        partial_attempts = _partial_attempts_from_failure(exc, provider_result)
        partial_usage, partial_quality_summary = _partial_run_metadata_from_failure(
            exc,
            provider_result,
        )
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item:
                disposition = await _settle_stale_item(
                    session,
                    item,
                    transcript_id,
                    partial_attempts,
                    partial_usage,
                    partial_quality_summary,
                )
                if disposition != "active":
                    return
                item.status = ItemStatus.CANCELLED
                item.stage = "cancelled"
                item.completed_at = utc_now()
                if transcript_id:
                    transcript = await session.get(Transcript, transcript_id)
                    job = await session.get(ProcessingJob, item.job_id)
                    if transcript and job and _should_mark_transcript_failed(job, item, transcript):
                        await _persist_partial_attempt_evidence(
                            session,
                            transcript,
                            partial_attempts,
                            api_usage=partial_usage,
                            quality_summary=partial_quality_summary,
                        )
                        transcript.status = TranscriptStatus.FAILED
                        transcript.error_category = "cancelled"
                        transcript.error_message = "Transcription was cancelled."
                await session.commit()
    except StaleProcessingTaskError:
        return
    except Exception as exc:
        partial_attempts = _partial_attempts_from_failure(exc, provider_result)
        partial_usage, partial_quality_summary = _partial_run_metadata_from_failure(
            exc,
            provider_result,
        )
        await _refresh_available_processing_leases(lock, transcription_slot)
        async with AsyncSessionFactory() as session:
            item = await session.get(ProcessingJobItem, item_id)
            if item:
                disposition = await _settle_stale_item(
                    session,
                    item,
                    transcript_id,
                    partial_attempts,
                    partial_usage,
                    partial_quality_summary,
                )
                if disposition != "active":
                    return
                await _refresh_available_processing_leases(lock, transcription_slot)
                item.status = ItemStatus.FAILED
                item.stage = "failed"
                item.error_category = getattr(exc, "category", "unexpected")
                item.error_message = _safe_error_message(exc)
                item.completed_at = utc_now()
                if item.requested_pipeline_version is None:
                    call = await session.get(Call, item.call_id)
                    if call:
                        call.processing_status = "failed"
                        call.last_error_category = item.error_category
                        call.last_error_message = item.error_message
                    recording = (
                        await session.get(Recording, item.recording_id)
                        if item.recording_id
                        else None
                    )
                    if recording:
                        recording.status = RecordingStatus.FAILED
                        recording.last_error_category = item.error_category
                        recording.last_error_message = item.error_message
                if transcript_id:
                    transcript = await session.get(Transcript, transcript_id)
                    job = await session.get(ProcessingJob, item.job_id)
                    if transcript and job and _should_mark_transcript_failed(job, item, transcript):
                        await _persist_partial_attempt_evidence(
                            session,
                            transcript,
                            partial_attempts,
                            api_usage=partial_usage,
                            quality_summary=partial_quality_summary,
                        )
                        transcript.status = TranscriptStatus.FAILED
                        transcript.error_category = item.error_category
                        transcript.error_message = item.error_message
                await session.commit()
        raise
    finally:
        _cleanup_temporary_audio(
            AudioProcessor(settings),
            temporary_files,
            temporary_directories,
        )
        try:
            if lock is not None and await lock.owned():
                await lock.release()
        except Exception:
            pass
        try:
            if transcription_slot is not None and await transcription_slot.owned():
                await transcription_slot.release()
        except Exception:
            pass
        await finalize_job_for_item(item_id)


def _partial_attempts_from_failure(
    exc: BaseException,
    provider_result: TranscriptionResult | OrchestratedTranscriptionResult | None,
) -> tuple[TranscriptionAttemptEvidence, ...]:
    if isinstance(
        exc,
        (PartialTranscriptionError, PartialTranscriptionCancelledError),
    ):
        return exc.attempts
    if isinstance(provider_result, OrchestratedTranscriptionResult):
        return provider_result.attempts
    return ()


def _partial_run_metadata_from_failure(
    exc: BaseException,
    provider_result: TranscriptionResult | OrchestratedTranscriptionResult | None,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if isinstance(
        exc,
        (PartialTranscriptionError, PartialTranscriptionCancelledError),
    ):
        usage = getattr(exc, "usage", None)
        quality_summary = getattr(exc, "quality_summary", None)
        return (
            usage if isinstance(usage, Mapping) else None,
            quality_summary if isinstance(quality_summary, Mapping) else None,
        )
    if isinstance(provider_result, OrchestratedTranscriptionResult):
        quality_summary = provider_result.quality_summary
        return (
            provider_result.usage,
            quality_summary if isinstance(quality_summary, Mapping) else None,
        )
    return None, None


def _transcription_attempt_entity(
    transcript_id: UUID,
    attempt: TranscriptionAttemptEvidence,
) -> TranscriptionAttempt:
    return TranscriptionAttempt(
        transcript_id=transcript_id,
        track_id=attempt.track_id,
        chunk_index=attempt.chunk_index,
        start_seconds=Decimal(str(round(attempt.start_seconds, 3))),
        end_seconds=Decimal(str(round(attempt.end_seconds, 3))),
        model=attempt.model,
        audio_variant=attempt.audio_variant or "unknown",
        prompt_hash=attempt.prompt_hash,
        response_text=attempt.response_text,
        mean_logprob=(
            Decimal(str(attempt.mean_logprob)) if attempt.mean_logprob is not None else None
        ),
        low_logprob_ratio=(
            Decimal(str(attempt.low_logprob_ratio))
            if attempt.low_logprob_ratio is not None
            else None
        ),
        selected=attempt.selected,
        api_usage=_mutable_json_metadata(attempt.api_usage),
        completed_at=attempt.completed_at or utc_now(),
    )


async def _persist_partial_attempt_evidence(
    session: AsyncSession,
    transcript: Transcript,
    attempts: tuple[TranscriptionAttemptEvidence, ...],
    *,
    api_usage: Mapping[str, Any] | None = None,
    quality_summary: Mapping[str, Any] | None = None,
) -> None:
    """Atomically retain completed provider work from an interrupted V2 run."""

    mutated = False
    if api_usage is not None:
        transcript.api_usage = _mutable_json_metadata(api_usage)
        mutated = True
    if quality_summary is not None:
        transcript.quality_summary = _mutable_json_metadata(quality_summary)
        mutated = True
    if attempts:
        _validate_attempt_evidence(
            attempts,
            allow_unselected=(transcript.transcription_mode == TranscriptionMode.MONO_DIARIZATION),
        )
        existing_count = (
            await session.scalar(
                select(func.count())
                .select_from(TranscriptionAttempt)
                .where(TranscriptionAttempt.transcript_id == transcript.id)
            )
            or 0
        )
        if not existing_count:
            for attempt in attempts:
                session.add(_transcription_attempt_entity(transcript.id, attempt))
            mutated = True
    if mutated:
        await session.flush()


async def _persist_transcription_result(
    session: AsyncSession,
    transcript: Transcript,
    result: TranscriptionResult | OrchestratedTranscriptionResult,
    call: Call,
    operator: Operator | None,
    call_leg_id: UUID | None,
    audio_duration: float,
) -> None:
    topology_result = (
        isinstance(result, OrchestratedTranscriptionResult) and result.mode != "legacy"
    )
    if topology_result:
        _validate_transcription_attempts(result)
    transcript.model = result.model
    transcript.language = result.language
    transcript.prompt_version = result.prompt_version
    transcript.processing_duration_seconds = result.processing_duration_seconds
    transcript.audio_duration_seconds = audio_duration
    transcript.api_usage = result.usage
    transcript.quality_summary = (
        _mutable_json_metadata(result.quality_summary)
        if isinstance(result, OrchestratedTranscriptionResult)
        and result.quality_summary is not None
        else None
    )
    transcript.status = TranscriptStatus.COMPLETED
    transcript.completed_at = utc_now()
    transcript.original_text = result.text
    transcript.normalized_text = normalize_greek(result.text)
    transcript.error_category = None
    transcript.error_message = None
    if topology_result:
        transcript.transcription_mode = TranscriptionMode(result.mode)
        if result.attribution_status is not None:
            transcript.speaker_attribution_status = SpeakerAttributionStatus(
                result.attribution_status
            )
        for attempt in result.attempts:
            session.add(_transcription_attempt_entity(transcript.id, attempt))
    for sequence, segment in enumerate(result.segments, start=1):
        if topology_result:
            segment_operator_id = (
                UUID(segment.operator_id) if segment.operator_id is not None else None
            )
            speaker_source = SpeakerSource(segment.speaker_source or SpeakerSource.UNKNOWN.value)
            speaker_label = segment.speaker_label
            segment_call_leg_id = (
                call_leg_id if operator is not None and segment_operator_id == operator.id else None
            )
        else:
            segment_operator_id = operator.id if operator else None
            speaker_source = (
                SpeakerSource.STEREO_CHANNEL if operator else SpeakerSource.OPENAI_DIARIZATION
            )
            speaker_label = operator.display_name if operator else segment.speaker_label
            segment_call_leg_id = call_leg_id if operator else None
        session.add(
            TranscriptSegment(
                transcript_id=transcript.id,
                call_id=call.id,
                call_leg_id=segment_call_leg_id,
                operator_id=segment_operator_id,
                speaker_label=speaker_label,
                speaker_source=speaker_source,
                start_seconds=Decimal(str(round(segment.start_seconds, 3))),
                end_seconds=Decimal(str(round(segment.end_seconds, 3))),
                original_text=segment.text,
                normalized_text=normalize_greek(segment.text),
                confidence=(
                    Decimal(str(segment.confidence)) if segment.confidence is not None else None
                ),
                transcription_model=(
                    (segment.transcription_model or result.model)
                    if topology_result
                    else result.model
                ),
                sequence_number=sequence,
                channel_index=segment.channel_index if topology_result else None,
                track_id=segment.track_id if topology_result else None,
                chunk_index=segment.chunk_index if topology_result else None,
                mean_logprob=(
                    Decimal(str(segment.mean_logprob))
                    if topology_result and segment.mean_logprob is not None
                    else None
                ),
                low_logprob_ratio=(
                    Decimal(str(segment.low_logprob_ratio))
                    if topology_result and segment.low_logprob_ratio is not None
                    else None
                ),
                quality_flags=(list(segment.quality_flags) if topology_result else []),
                audio_variant=segment.audio_variant if topology_result else None,
            )
        )
    await session.flush()


def _validate_transcription_attempts(
    result: OrchestratedTranscriptionResult,
) -> None:
    grouped: dict[tuple[str, int], list[TranscriptionAttemptEvidence]] = {}
    for attempt in result.attempts:
        grouped.setdefault((attempt.track_id, attempt.chunk_index), []).append(attempt)
    if result.mode == "mono_diarization":
        _validate_attempt_evidence(result.attempts, allow_unselected=True)
        segment_groups: set[tuple[str, int]] = set()
        for segment in result.segments:
            if segment.chunk_index is None:
                raise ValueError("Each refined mono segment must retain its span identity.")
            if segment.operator_id is not None:
                raise ValueError("Anonymous mono segments cannot be assigned to an operator.")
            if segment.speaker_source != SpeakerSource.OPENAI_DIARIZATION.value:
                raise ValueError("Refined mono segments must retain diarization speaker evidence.")
            identity = (segment.track_id, segment.chunk_index)
            segment_groups.add(identity)
            selected_count = sum(bool(attempt.selected) for attempt in grouped.get(identity, ()))
            refinement_failed = "refinement_failed" in segment.quality_flags
            if refinement_failed:
                if selected_count:
                    raise ValueError(
                        "A rough diarization fallback cannot select a refinement attempt."
                    )
            elif selected_count != 1:
                raise ValueError(
                    "Each successfully refined mono segment must select exactly one attempt."
                )
        if orphaned := set(grouped).difference(segment_groups):
            orphaned_identity = sorted(orphaned)[0]
            raise ValueError(
                "Mono refinement attempt evidence has no matching final segment "
                f"for {orphaned_identity[0]} span {orphaned_identity[1]}."
            )
        return
    _validate_attempt_evidence(result.attempts)
    if result.mode not in {"operator_channel", "dual_channel"}:
        return
    segment_groups: set[tuple[str, int]] = set()
    for segment in result.segments:
        if segment.chunk_index is None:
            raise ValueError("Each standard V2 segment must retain its chunk identity.")
        segment_groups.add((segment.track_id, segment.chunk_index))
    if missing := segment_groups.difference(grouped):
        missing_identity = sorted(missing)[0]
        raise ValueError(
            "Each standard V2 segment must have selected attempt evidence "
            f"for {missing_identity[0]} chunk {missing_identity[1]}."
        )


def _validate_attempt_evidence(
    attempts: tuple[TranscriptionAttemptEvidence, ...],
    *,
    allow_unselected: bool = False,
) -> None:
    grouped: dict[tuple[str, int], list[TranscriptionAttemptEvidence]] = {}
    for attempt in attempts:
        grouped.setdefault((attempt.track_id, attempt.chunk_index), []).append(attempt)
    for grouped_attempts in grouped.values():
        if not 1 <= len(grouped_attempts) <= 2:
            raise ValueError("Each attempted V2 chunk must have one or two attempts.")
        selected_count = sum(bool(attempt.selected) for attempt in grouped_attempts)
        if allow_unselected and selected_count > 1:
            raise ValueError("Each attempted V2 chunk can select at most one attempt.")
        if not allow_unselected and selected_count != 1:
            raise ValueError("Each attempted V2 chunk must select exactly one attempt.")


def _mutable_json_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _mutable_json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_json_metadata(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Attempt usage contains unsupported metadata.")


async def finalize_job_for_item(item_id: UUID) -> None:
    async with AsyncSessionFactory() as session:
        item = await session.get(ProcessingJobItem, item_id)
        if item is None:
            return
        job = await session.scalar(
            select(ProcessingJob).where(ProcessingJob.id == item.job_id).with_for_update()
        )
        if job is None or job.status in FINAL_JOB_STATES:
            return
        item_rows = (
            await session.execute(
                select(ProcessingJobItem.call_id, ProcessingJobItem.status).where(
                    ProcessingJobItem.job_id == job.id
                )
            )
        ).all()
        statuses_by_call: dict[UUID, list[ItemStatus]] = defaultdict(list)
        for call_id, item_status in item_rows:
            statuses_by_call[call_id].append(item_status)
        total = len(item_rows)
        # A call is complete only when every selected operator completed. This
        # keeps a mixed A=complete/B=failed call out of both success and failure
        # counters at the same time.
        completed = sum(
            bool(states) and all(state == ItemStatus.COMPLETED for state in states)
            for states in statuses_by_call.values()
        )
        failed_items = sum(
            any(state == ItemStatus.FAILED for state in states)
            for states in statuses_by_call.values()
        )
        discovery_failures = int((job.request_filters or {}).get("_discovery_failures", 0))
        failed = failed_items + discovery_failures
        final_count = sum(item_status in FINAL_ITEM_STATES for _, item_status in item_rows)
        job.calls_completed = completed
        job.calls_failed = failed
        job.progress_percent = round(final_count * 100 / total) if total else 100
        if final_count == total:
            job.completed_at = utc_now()
            if job.cancellation_requested:
                job.status = JobStatus.CANCELLED
                job.current_stage = "Cancelled"
            elif failed:
                job.status = JobStatus.COMPLETED_WITH_ERRORS
                job.current_stage = "Complete with some errors"
            else:
                job.status = JobStatus.COMPLETED
                job.current_stage = "Complete"
        await session.commit()


def _active_processing_recording_ids() -> Select[tuple[UUID | None]]:
    """Return recordings protected by a live job item, including reuse work."""

    return (
        select(ProcessingJobItem.recording_id)
        .join(ProcessingJob, ProcessingJob.id == ProcessingJobItem.job_id)
        .where(
            ProcessingJobItem.recording_id.is_not(None),
            ProcessingJobItem.status.not_in(FINAL_ITEM_STATES),
            ProcessingJob.status.not_in(FINAL_JOB_STATES),
        )
    )


async def _recording_has_active_processing(
    session: AsyncSession,
    recording_id: UUID,
) -> bool:
    count = await session.scalar(
        select(func.count())
        .select_from(ProcessingJobItem)
        .join(ProcessingJob, ProcessingJob.id == ProcessingJobItem.job_id)
        .where(
            ProcessingJobItem.recording_id == recording_id,
            ProcessingJobItem.status.not_in(FINAL_ITEM_STATES),
            ProcessingJob.status.not_in(FINAL_JOB_STATES),
        )
    )
    return bool(count)


async def _lock_recording_job_rows(
    session: AsyncSession,
    recording_ids: set[UUID],
) -> None:
    """Fence retries that can reactivate an existing item during retention."""

    if not recording_ids:
        return
    job_ids = (
        select(ProcessingJobItem.job_id)
        .where(ProcessingJobItem.recording_id.in_(recording_ids))
        .distinct()
    )
    (
        await session.scalars(
            select(ProcessingJob.id)
            .where(ProcessingJob.id.in_(job_ids))
            .order_by(ProcessingJob.id)
            .with_for_update()
        )
    ).all()


async def cleanup_retention_records() -> dict[str, int]:
    settings = get_settings()
    deleted_transcripts = 0
    deleted_audio = 0
    cleanup_failed = 0
    immediate_audio_retries = 0
    async with AsyncSessionFactory() as session:
        application = await load_application_settings(session, settings)
        active_processing_recording_ids = _active_processing_recording_ids()
        immediate_pending = (
            await session.scalars(
                select(Recording).where(
                    Recording.last_error_category == "audio_cleanup_pending",
                    Recording.storage_key.is_not(None),
                    Recording.id.not_in(active_processing_recording_ids),
                )
            )
        ).all()
        cutoff = utc_now() - timedelta(days=int(application["transcript_retention_days"]))
        transcripts = (
            await session.scalars(
                select(Transcript)
                .where(
                    Transcript.completed_at.is_not(None),
                    Transcript.completed_at < cutoff,
                    Transcript.recording_id.not_in(active_processing_recording_ids),
                )
                .order_by(Transcript.recording_id, Transcript.id)
            )
        ).all()
        pending_recording_ids = (
            await session.scalars(
                select(Recording.id).where(
                    Recording.last_error_category == "retention_cleanup_pending",
                    Recording.id.not_in(active_processing_recording_ids),
                )
            )
        ).all()
        maintenance_recording_ids = {
            *(recording.id for recording in immediate_pending),
            *(transcript.recording_id for transcript in transcripts),
            *pending_recording_ids,
        }
        # Discovery takes the same recording locks before it can commit new
        # recording-backed items. Existing retries are serialized by job row.
        # Whichever transaction locks first therefore linearizes completely
        # before the other can recheck active work.
        await _lock_recording_rows(session, maintenance_recording_ids)
        await _lock_recording_job_rows(session, maintenance_recording_ids)

        for recording in immediate_pending:
            await session.refresh(recording, with_for_update=True)
            if await _recording_has_active_processing(session, recording.id):
                continue
            if not recording.storage_key:
                continue
            path = AudioProcessor(settings).safe_storage_path(recording.storage_key)
            _, failures = AudioProcessor(settings).remove_files([path])
            if failures:
                cleanup_failed += 1
                continue
            immediate_audio_retries += 1
            deleted_audio += 1
            recording.storage_key = None
            recording.status = RecordingStatus.DELETED
            recording.deleted_at = utc_now()
            recording.last_error_category = None
            recording.last_error_message = None

        recording_ids: set[UUID] = set()
        recording_ids.update(pending_recording_ids)
        for candidate in transcripts:
            # The recording/job locks above close the check/delete race with
            # ordinary discovery and retry. The transcript lock also serializes
            # explicit reprocess creation.
            transcript = await session.scalar(
                select(Transcript)
                .where(
                    Transcript.id == candidate.id,
                    Transcript.completed_at.is_not(None),
                    Transcript.completed_at < cutoff,
                )
                .with_for_update()
            )
            if transcript is None:
                continue
            if await _recording_has_active_processing(
                session,
                transcript.recording_id,
            ):
                continue
            recording_ids.add(transcript.recording_id)
            await session.delete(transcript)
            deleted_transcripts += 1
        await session.flush()
        for recording_id in recording_ids:
            if await _recording_has_active_processing(session, recording_id):
                continue
            remaining = (
                await session.scalar(
                    select(func.count())
                    .select_from(Transcript)
                    .where(Transcript.recording_id == recording_id)
                )
                or 0
            )
            recording = await session.get(Recording, recording_id)
            if remaining or not recording or not recording.storage_key:
                continue
            path = AudioProcessor(settings).safe_storage_path(recording.storage_key)
            _, failures = AudioProcessor(settings).remove_files([path])
            if failures:
                cleanup_failed += 1
                recording.last_error_category = "retention_cleanup_pending"
                recording.last_error_message = "Audio retention cleanup will be retried."
            else:
                deleted_audio += 1
                recording.storage_key = None
                recording.status = RecordingStatus.DELETED
                recording.deleted_at = utc_now()
                recording.last_error_category = None
                recording.last_error_message = None
        session.add(
            AuditLog(
                created_at=utc_now(),
                user_id=None,
                action="retention.delete",
                resource_type="retention_cleanup",
                outcome="success" if not cleanup_failed else "partial",
                details={
                    "transcripts_deleted": deleted_transcripts,
                    "audio_deleted": deleted_audio,
                    "cleanup_failed": cleanup_failed,
                    "immediate_audio_retries": immediate_audio_retries,
                },
            )
        )
        await session.commit()
    return {
        "transcripts_deleted": deleted_transcripts,
        "audio_deleted": deleted_audio,
        "cleanup_failed": cleanup_failed,
    }
