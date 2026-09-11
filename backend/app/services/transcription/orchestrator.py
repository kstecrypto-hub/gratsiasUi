from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
import re
import wave

from app.core.config import Settings, get_settings
from app.core.time import utc_now
from app.services.audio import AudioChunk, AudioInfo, AudioProcessor
from app.services.audio.quality import (
    AudioQualityProcessor,
    LegacyPassThroughQualityProcessor,
    LightNormalizedRetryQualityProcessor,
    RAW_LOSSLESS_AUDIO_VARIANT,
    RetryAudioQualityProcessor,
)
from app.services.audio.segmentation import (
    AudioSegmenter,
    LegacyFixedAudioSegmenter,
    LocalSpeechAudioSegmenter,
    local_speech_segmentation_identity,
)
from app.services.audio.errors import AudioSegmentationCancelledError
from app.services.transcription.client import (
    OpenAITranscriptionClient,
    TranscriptionCancelledError,
    TranscriptionResponseEvidence,
    TranscriptionResult,
)
from app.services.transcription.confidence import (
    DEFAULT_CONFIDENCE_POLICY,
    ConfidenceMetrics,
    ConfidencePolicy,
    ConfidenceAnalyzer,
    LogprobConfidenceAnalyzer,
    calculate_confidence_metrics,
    is_low_confidence,
    select_preferred_attempt,
)
from app.services.transcription.merge import merge_track_results
from app.services.transcription.conversation import (
    CONVERSATION_ALIGNMENT_VERSION,
    align_conversation,
)
from app.services.transcription.mono import (
    DEFAULT_MONO_REFINEMENT_POLICY,
    AnonymousDiarizationTurn,
    MonoRefinementPolicy,
    MonoRefinementSpan,
    calculate_span_degraded_duration,
    coalesce_anonymous_turns,
    padded_sample_bounds,
)
from app.services.transcription.planning import AudioPlanner, LegacyAudioPlanner
from app.services.transcription.prompt import (
    LegacyVocabularyPromptBuilder,
    PromptPlan,
    PromptBuilder,
    V2GreekPromptBuilder,
    V2PromptManifest,
)
from app.services.transcription.types import (
    AudioTrack,
    AudioPlan,
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    SpeechChunk,
    TokenLogprob,
    TrackTranscriptionResult,
    TranscriptionAttemptEvidence,
)


EXACT_OVERLAP_JOIN_VERSION = "exact-case-sensitive-token-overlap-v1"
EXACT_OVERLAP_MIN_TOKENS = 2
EXACT_OVERLAP_MIN_CHARACTERS = 8
EXACT_OVERLAP_MAX_TOKENS = 24
QUALITY_FLAG_LOW_CONFIDENCE = "low_confidence"
QUALITY_FLAG_NORMALIZED_RETRY_USED = "normalized_retry_used"
QUALITY_FLAG_BOTH_ATTEMPTS_LOW_CONFIDENCE = "both_attempts_low_confidence"
QUALITY_FLAG_HUMAN_REVIEW_RECOMMENDED = "human_review_recommended"
QUALITY_FLAG_LOGPROBS_UNAVAILABLE = "logprobs_unavailable"
QUALITY_FLAG_REFINEMENT_FAILED = "refinement_failed"
QUALITY_FLAG_REFINEMENT_BUDGET_EXHAUSTED = "refinement_budget_exhausted"
QUALITY_FLAG_REFINEMENT_SPAN_TOO_LONG = "refinement_span_too_long"
QUALITY_FLAG_NORMALIZED_RETRY_FAILED = "normalized_retry_failed"
QUALITY_FLAG_ROUGH_DIARIZATION_FALLBACK = "rough_diarization_fallback"
MONO_ROUGH_FALLBACK_AUDIO_VARIANT = "pass1-diarization-rough-fallback-v1"


@dataclass(frozen=True, slots=True)
class _V2StandardAttempt:
    chunk: SpeechChunk
    result: TranscriptionResult
    response: TranscriptionResponseEvidence
    metrics: ConfidenceMetrics | None
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class _MonoSpanRefinement:
    text: str | None
    model: str | None
    token_logprobs: tuple[TokenLogprob, ...]
    mean_logprob: float | None
    low_logprob_ratio: float | None
    quality_flags: tuple[str, ...]
    audio_variant: str | None
    attempts: tuple[TranscriptionAttemptEvidence, ...]
    usage: dict[str, object]
    processing_duration_seconds: float
    failure_reason: str | None = None


class _MonoSpanRefinementCancelled(TranscriptionCancelledError):
    def __init__(
        self,
        *,
        attempts: tuple[TranscriptionAttemptEvidence, ...],
        usage: dict[str, object],
        processing_duration_seconds: float,
    ) -> None:
        super().__init__("Transcription was cancelled.")
        self.attempts = attempts
        self.usage = usage
        self.processing_duration_seconds = processing_duration_seconds


class PartialTranscriptionError(Exception):
    """A failed V2 run that still contains completed provider attempts."""

    def __init__(
        self,
        message: str,
        *,
        attempts: tuple[TranscriptionAttemptEvidence, ...],
        category: str | None = None,
        usage: dict[str, object] | None = None,
        quality_summary: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        self.attempts = attempts
        self.usage = usage
        self.quality_summary = quality_summary


class PartialTranscriptionCancelledError(TranscriptionCancelledError):
    """A cancelled V2 run that still contains completed provider attempts."""

    def __init__(
        self,
        message: str,
        *,
        attempts: tuple[TranscriptionAttemptEvidence, ...],
        usage: dict[str, object] | None = None,
        quality_summary: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.usage = usage
        self.quality_summary = quality_summary


def default_speech_segmentation_identity(
    *,
    max_upload_bytes: int,
) -> dict[str, object]:
    identity = local_speech_segmentation_identity()
    identity["max_upload_bytes"] = max_upload_bytes
    identity["overlap_join"] = {
        "case_sensitive": True,
        "maximum_tokens": EXACT_OVERLAP_MAX_TOKENS,
        "minimum_characters": EXACT_OVERLAP_MIN_CHARACTERS,
        "minimum_tokens": EXACT_OVERLAP_MIN_TOKENS,
        "strategy": EXACT_OVERLAP_JOIN_VERSION,
    }
    return identity


@dataclass(frozen=True, slots=True)
class TranscriptionContext:
    diarized: bool
    language: str
    vocabulary: tuple[str, ...]
    temporary_directory: Path
    channel_index: int | None = None
    operator_id: str | None = None
    attribution_status: str | None = None
    audio_variant: str | None = None
    stereo_separated: bool = False
    operator_channel: int | None = None
    caller_channel: int | None = None
    callee_channel: int | None = None
    operator_display_name: str | None = None
    v2_prompt_manifest: V2PromptManifest | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    register_temporary_file: Callable[[Path], None] | None = field(
        default=None,
        compare=False,
        repr=False,
    )


class TranscriptionOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        audio_processor: AudioProcessor | None = None,
        planner: AudioPlanner | None = None,
        segmenter: AudioSegmenter | None = None,
        speech_segmenter: AudioSegmenter | None = None,
        legacy_segmenter: AudioSegmenter | None = None,
        quality_processor: AudioQualityProcessor | None = None,
        retry_quality_processor: RetryAudioQualityProcessor | None = None,
        prompt_builder: PromptBuilder | None = None,
        v2_prompt_builder: V2GreekPromptBuilder | None = None,
        confidence_analyzer: ConfidenceAnalyzer | None = None,
        confidence_policy: ConfidencePolicy = DEFAULT_CONFIDENCE_POLICY,
        mono_refinement_policy: MonoRefinementPolicy = DEFAULT_MONO_REFINEMENT_POLICY,
        client_factory: Callable[[], OpenAITranscriptionClient] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.audio_processor = audio_processor or AudioProcessor(self.settings)
        self.planner = planner or LegacyAudioPlanner()
        self._segmenter_override = segmenter
        self.legacy_segmenter = legacy_segmenter or LegacyFixedAudioSegmenter(self.audio_processor)
        self.speech_segmenter = speech_segmenter or LocalSpeechAudioSegmenter(self.audio_processor)
        # Preserve the original public attribute for injected legacy tests.
        self.segmenter = segmenter or self.legacy_segmenter
        self.quality_processor = quality_processor or LegacyPassThroughQualityProcessor()
        self.retry_quality_processor = (
            retry_quality_processor
            or LightNormalizedRetryQualityProcessor(self.audio_processor)
        )
        self.prompt_builder = prompt_builder or LegacyVocabularyPromptBuilder()
        self.v2_prompt_builder = v2_prompt_builder or V2GreekPromptBuilder()
        self.confidence_policy = confidence_policy
        self.mono_refinement_policy = mono_refinement_policy
        self.confidence_analyzer = confidence_analyzer or LogprobConfidenceAnalyzer(
            confidence_policy
        )
        self.client_factory = client_factory or (
            lambda: OpenAITranscriptionClient(settings=self.settings)
        )

    def segmentation_identity(
        self,
        mode: str,
        *,
        max_upload_bytes: int,
    ) -> dict[str, object]:
        if mode not in {"operator_channel", "dual_channel"}:
            return {
                "chunk_seconds": 480,
                "max_upload_bytes": max_upload_bytes,
                "single_upload_when_within_limit": True,
                "strategy": "legacy-fixed",
            }
        segmenter = self._segmenter_for_mode(mode)
        identity_method = getattr(segmenter, "identity", None)
        if not callable(identity_method):
            raise ValueError("The executing speech segmenter has no versioned identity.")
        identity = dict(identity_method())
        identity["max_upload_bytes"] = max_upload_bytes
        identity["overlap_join"] = {
            "case_sensitive": True,
            "maximum_tokens": EXACT_OVERLAP_MAX_TOKENS,
            "minimum_characters": EXACT_OVERLAP_MIN_CHARACTERS,
            "minimum_tokens": EXACT_OVERLAP_MIN_TOKENS,
            "strategy": EXACT_OVERLAP_JOIN_VERSION,
        }
        return identity

    def plan(
        self,
        *,
        source_path: Path,
        audio_info: AudioInfo,
        context: TranscriptionContext,
    ) -> AudioPlan:
        return self.planner.plan(
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

    async def transcribe(
        self,
        *,
        source_path: Path,
        audio_info: AudioInfo,
        context: TranscriptionContext,
        plan: AudioPlan | None = None,
        cancellation_check: Callable[[], Awaitable[bool]] | None = None,
    ) -> OrchestratedTranscriptionResult:
        plan = plan or self.plan(
            source_path=source_path,
            audio_info=audio_info,
            context=context,
        )
        if plan.mode in {"operator_channel", "dual_channel", "mono_diarization"}:
            prompt_manifest = context.v2_prompt_manifest
            expected_manifest = self.v2_prompt_builder.build_manifest(plan, ())
            if prompt_manifest is None:
                prompt_manifest = expected_manifest
            elif {(track.track_id, track.track_role) for track in prompt_manifest.tracks} != {
                (track.track_id, track.track_role) for track in expected_manifest.tracks
            }:
                raise ValueError("The V2 prompt manifest does not match the audio plan.")
            context = replace(
                context,
                language="el",
                v2_prompt_manifest=prompt_manifest,
            )
        materialized_tracks = await self._prepare_tracks(
            plan,
            destination_dir=context.temporary_directory,
            register_temporary_file=context.register_temporary_file,
            cancellation_check=cancellation_check,
        )
        prepared_tracks = [
            await self.quality_processor.prepare(track, audio_info=audio_info)
            for track in materialized_tracks
        ]
        if plan.mode == "mono_diarization":
            if len(prepared_tracks) != 1:
                raise ValueError("Mono diarization requires exactly one prepared track.")
            async with self.client_factory() as client:
                return await self._transcribe_v2_mono(
                    client,
                    plan,
                    prepared_tracks[0],
                    audio_info,
                    context,
                    cancellation_check,
                )

        chunks_by_track: dict[str, tuple[SpeechChunk, ...]] = {}
        segmenter = self._segmenter_for_mode(plan.mode)
        for track in prepared_tracks:
            try:
                chunks_by_track[track.track_id] = await segmenter.segment(
                    track,
                    audio_info=audio_info,
                    destination_dir=context.temporary_directory,
                    max_upload_bytes=self.settings.max_transcription_upload_bytes,
                    register_temporary_file=context.register_temporary_file,
                    cancellation_check=cancellation_check,
                )
            except AudioSegmentationCancelledError as exc:
                raise TranscriptionCancelledError("Transcription was cancelled.") from exc

        track_results: list[TrackTranscriptionResult] = []
        has_provider_work = any(chunks_by_track[track.track_id] for track in prepared_tracks)
        if has_provider_work:
            async with self.client_factory() as client:
                for track in prepared_tracks:
                    speech_chunks = chunks_by_track[track.track_id]
                    if not speech_chunks:
                        track_results.append(self._empty_track_result(track, context))
                        continue
                    try:
                        track_results.append(
                            await self._transcribe_track(
                                client,
                                track,
                                speech_chunks,
                                context,
                                cancellation_check,
                            )
                        )
                    except (
                        PartialTranscriptionError,
                        PartialTranscriptionCancelledError,
                    ) as exc:
                        prior_attempts = tuple(
                            attempt
                            for track_result in track_results
                            for attempt in track_result.attempts
                        )
                        self._raise_partial_transcription(
                            exc,
                            (*prior_attempts, *exc.attempts),
                        )
        else:
            for track in prepared_tracks:
                track_results.append(self._empty_track_result(track, context))

        all_hypotheses = [
            hypothesis for track_result in track_results for hypothesis in track_result.hypotheses
        ]
        confidence = self.confidence_analyzer.analyze(all_hypotheses)
        return merge_track_results(plan, track_results, confidence)

    def _segmenter_for_mode(self, mode: str) -> AudioSegmenter:
        if self._segmenter_override is not None:
            return self._segmenter_override
        if mode in {"operator_channel", "dual_channel"}:
            return self.speech_segmenter
        return self.legacy_segmenter

    async def _transcribe_track(
        self,
        client: OpenAITranscriptionClient,
        track: AudioTrack,
        speech_chunks: tuple[SpeechChunk, ...],
        context: TranscriptionContext,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> TrackTranscriptionResult:
        provider_chunks = [
            AudioChunk(
                path=chunk.path,
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                chunk_index=chunk.chunk_index,
                hard_cut=chunk.hard_cut,
                overlap_before_ms=chunk.overlap_before_ms,
            )
            for chunk in speech_chunks
        ]
        if track.diarized:
            result = await client.transcribe_diarized(
                provider_chunks,
                language=context.language,
                should_cancel=cancellation_check,
            )
        elif context.v2_prompt_manifest is not None:
            return await self._transcribe_v2_track(
                client,
                track,
                speech_chunks,
                context,
                cancellation_check,
            )
        else:
            prompt_plan = self.prompt_builder.build(context.vocabulary)
            result = await client.transcribe_isolated(
                provider_chunks,
                list(context.vocabulary),
                language=context.language,
                should_cancel=cancellation_check,
                prompt_plan=prompt_plan,
            )
        return self._track_result(track, speech_chunks, result)

    async def _transcribe_v2_mono(
        self,
        client: OpenAITranscriptionClient,
        plan: AudioPlan,
        track: AudioTrack,
        audio_info: AudioInfo,
        context: TranscriptionContext,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> OrchestratedTranscriptionResult:
        del audio_info
        manifest = context.v2_prompt_manifest
        if manifest is None:
            raise ValueError("V2 mono refinement requires a prompt manifest.")
        frame_count = self._prepared_mono_frame_count(track.source_path)
        recording_duration = frame_count / self.mono_refinement_policy.sample_rate_hz
        pass1_chunk = AudioChunk(
            path=track.source_path,
            start_seconds=0.0,
            end_seconds=recording_duration,
            chunk_index=0,
        )
        pass1 = await client.transcribe_diarized_complete(
            pass1_chunk,
            language="el",
            should_cancel=cancellation_check,
        )
        pass2_usage: dict[str, object] = {"chunks": [], "totals": {}}
        usage = self._mono_usage(pass1.usage, pass2_usage)

        valid_turns: list[AnonymousDiarizationTurn] = []
        for segment in pass1.segments:
            try:
                valid_turns.append(
                    AnonymousDiarizationTurn(
                        speaker_label=segment.speaker_label,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        rough_text=segment.text,
                        recording_duration_seconds=recording_duration,
                    )
                )
            except (TypeError, ValueError):
                continue
        rejected_turn_count = len(pass1.segments) - len(valid_turns)
        if self.settings.OPENAI_TRANSCRIPTION_MODEL == "gpt-transcribe":
            return await self._transcribe_mono_conversation(
                client, plan, track, context,
                pass1=pass1,
                valid_turns=tuple(valid_turns),
                recording_duration=recording_duration,
                cancellation_check=cancellation_check,
            )
        spans = coalesce_anonymous_turns(
            valid_turns,
            self.mono_refinement_policy,
        )

        hypotheses: list[ChunkHypothesis] = []
        attempts: list[TranscriptionAttemptEvidence] = []
        fallback_span_indexes: set[int] = set()
        refined_span_count = 0
        previous_context = ""
        processing_duration = pass1.processing_duration_seconds

        for span_index, span in enumerate(spans):
            if cancellation_check is not None and await cancellation_check():
                self._raise_mono_partial_cancellation(
                    attempts=tuple(attempts),
                    usage=usage,
                    quality_summary=self._mono_quality_summary(
                        pass1=pass1,
                        pass1_turn_count=len(pass1.segments),
                        valid_turn_count=len(valid_turns),
                        rejected_turn_count=rejected_turn_count,
                        spans=spans,
                        fallback_span_indexes=fallback_span_indexes,
                        refined_span_count=refined_span_count,
                        processed_span_count=len(hypotheses),
                        status_override="cancelled",
                    ),
                )

            fallback_reason: str | None = None
            refinement: _MonoSpanRefinement | None = None
            if span_index >= self.mono_refinement_policy.max_refinement_spans:
                fallback_reason = QUALITY_FLAG_REFINEMENT_BUDGET_EXHAUSTED
            elif (
                span.duration_seconds
                > self.mono_refinement_policy.max_coalesced_span_seconds
            ):
                fallback_reason = QUALITY_FLAG_REFINEMENT_SPAN_TOO_LONG
            else:
                try:
                    bounds = padded_sample_bounds(
                        span,
                        self.mono_refinement_policy,
                        following_speech_start_seconds=(
                            spans[span_index + 1].start_seconds
                            if span_index + 1 < len(spans)
                            else recording_duration
                        ),
                    )
                    span_path = (
                        context.temporary_directory
                        / f"mono-refinement-{span_index:05d}.wav"
                    )
                    if context.register_temporary_file is not None:
                        context.register_temporary_file(span_path)
                    await self.audio_processor.extract_pcm_wav_range(
                        track.source_path,
                        span_path,
                        start_sample=bounds.start_sample,
                        end_sample=bounds.end_sample,
                        sample_rate_hz=bounds.sample_rate_hz,
                    )
                    speech_chunk = SpeechChunk(
                        track_id=track.track_id,
                        chunk_index=span_index,
                        path=span_path,
                        start_seconds=bounds.authoritative_start_seconds,
                        end_seconds=bounds.authoritative_end_seconds,
                        audio_variant=RAW_LOSSLESS_AUDIO_VARIANT,
                    )
                    prompt_plan = manifest.build_anonymous(
                        track.track_id,
                        span.speaker_label,
                        previous_context=previous_context,
                    )
                    refinement = await self._refine_mono_span(
                        client,
                        speech_chunk,
                        prompt_plan=prompt_plan,
                        context=context,
                        cancellation_check=cancellation_check,
                    )
                except _MonoSpanRefinementCancelled as exc:
                    self._merge_v2_usage(pass2_usage, exc.usage)
                    processing_duration += exc.processing_duration_seconds
                    attempts.extend(exc.attempts)
                    self._raise_mono_partial_cancellation(
                        attempts=tuple(attempts),
                        usage=usage,
                        quality_summary=self._mono_quality_summary(
                            pass1=pass1,
                            pass1_turn_count=len(pass1.segments),
                            valid_turn_count=len(valid_turns),
                            rejected_turn_count=rejected_turn_count,
                            spans=spans,
                            fallback_span_indexes=fallback_span_indexes,
                            refined_span_count=refined_span_count,
                            processed_span_count=len(hypotheses),
                            status_override="cancelled",
                        ),
                    )
                except TranscriptionCancelledError:
                    self._raise_mono_partial_cancellation(
                        attempts=tuple(attempts),
                        usage=usage,
                        quality_summary=self._mono_quality_summary(
                            pass1=pass1,
                            pass1_turn_count=len(pass1.segments),
                            valid_turn_count=len(valid_turns),
                            rejected_turn_count=rejected_turn_count,
                            spans=spans,
                            fallback_span_indexes=fallback_span_indexes,
                            refined_span_count=refined_span_count,
                            processed_span_count=len(hypotheses),
                            status_override="cancelled",
                        ),
                    )
                except Exception:
                    fallback_reason = "refinement_audio_or_prompt_failed"

            if refinement is not None:
                self._merge_v2_usage(pass2_usage, refinement.usage)
                processing_duration += refinement.processing_duration_seconds
                attempts.extend(refinement.attempts)
                if refinement.text is None:
                    fallback_reason = (
                        refinement.failure_reason or QUALITY_FLAG_REFINEMENT_FAILED
                    )

            if refinement is None or refinement.text is None:
                fallback_span_indexes.add(span_index)
                flags = [
                    QUALITY_FLAG_REFINEMENT_FAILED,
                    QUALITY_FLAG_HUMAN_REVIEW_RECOMMENDED,
                    QUALITY_FLAG_LOGPROBS_UNAVAILABLE,
                    QUALITY_FLAG_ROUGH_DIARIZATION_FALLBACK,
                ]
                if fallback_reason and fallback_reason not in flags:
                    flags.append(fallback_reason)
                text = span.rough_text
                hypothesis = self._mono_hypothesis(
                    track=track,
                    span=span,
                    span_index=span_index,
                    text=text,
                    model=pass1.model,
                    quality_flags=tuple(flags),
                    audio_variant=MONO_ROUGH_FALLBACK_AUDIO_VARIANT,
                )
            else:
                refined_span_count += 1
                text = refinement.text
                hypothesis = self._mono_hypothesis(
                    track=track,
                    span=span,
                    span_index=span_index,
                    text=text,
                    model=refinement.model
                    or self.settings.OPENAI_TRANSCRIPTION_MODEL,
                    token_logprobs=refinement.token_logprobs,
                    mean_logprob=refinement.mean_logprob,
                    low_logprob_ratio=refinement.low_logprob_ratio,
                    quality_flags=refinement.quality_flags,
                    audio_variant=refinement.audio_variant,
                )
            hypotheses.append(hypothesis)
            previous_context = self._append_mono_context(
                previous_context,
                span.speaker_label,
                text,
                self.mono_refinement_policy.global_context_max_characters,
            )

        quality_summary = self._mono_quality_summary(
            pass1=pass1,
            pass1_turn_count=len(pass1.segments),
            valid_turn_count=len(valid_turns),
            rejected_turn_count=rejected_turn_count,
            spans=spans,
            fallback_span_indexes=fallback_span_indexes,
            refined_span_count=refined_span_count,
            processed_span_count=len(hypotheses),
        )
        confidence = self.confidence_analyzer.analyze(hypotheses)
        track_result = TrackTranscriptionResult(
            track_id=track.track_id,
            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
            language="el",
            prompt_version=manifest.prompt_identity,
            processing_duration_seconds=processing_duration,
            hypotheses=tuple(hypotheses),
            usage=usage,
            diarized=True,
            attempts=tuple(attempts),
            allow_unselected_attempts=True,
        )
        return OrchestratedTranscriptionResult(
            mode=plan.mode,
            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
            language="el",
            prompt_version=manifest.prompt_identity,
            processing_duration_seconds=processing_duration,
            segments=tuple(hypotheses),
            tracks=(track_result,),
            usage=usage,
            diarized=True,
            confidence_status=confidence.status,
            attribution_status=plan.attribution_status,
            plan_reason=plan.reason,
            attempts=tuple(attempts),
            quality_summary=quality_summary,
        )

    async def _transcribe_mono_conversation(
        self,
        client: OpenAITranscriptionClient,
        plan: AudioPlan,
        track: AudioTrack,
        context: TranscriptionContext,
        *,
        pass1: TranscriptionResult,
        valid_turns: tuple[AnonymousDiarizationTurn, ...],
        recording_duration: float,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> OrchestratedTranscriptionResult:
        manifest = context.v2_prompt_manifest
        assert manifest is not None
        quality: dict[str, object] = {
            "strategy": CONVERSATION_ALIGNMENT_VERSION,
            "pass1_completed": True,
            "pass1_model": pass1.model,
            "pass2_model": self.settings.OPENAI_TRANSCRIPTION_MODEL,
            "pass1_turn_count": len(pass1.segments),
            "valid_turn_count": len(valid_turns),
            "rejected_turn_count": len(pass1.segments) - len(valid_turns),
            "timestamps_approximate": True,
        }
        try:
            refinement = await self._refine_mono_span(
                client,
                SpeechChunk(
                    track_id=track.track_id,
                    chunk_index=0,
                    path=track.source_path,
                    start_seconds=0.0,
                    end_seconds=recording_duration,
                    audio_variant=RAW_LOSSLESS_AUDIO_VARIANT,
                ),
                prompt_plan=manifest.build_conversation(track.track_id),
                context=context,
                cancellation_check=cancellation_check,
            )
        except _MonoSpanRefinementCancelled as exc:
            self._raise_mono_partial_cancellation(
                attempts=exc.attempts,
                usage=self._mono_usage(pass1.usage, exc.usage),
                quality_summary={**quality, "status": "cancelled"},
            )
        usage = self._mono_usage(pass1.usage, refinement.usage)
        if cancellation_check is not None and await cancellation_check():
            self._raise_mono_partial_cancellation(
                attempts=refinement.attempts, usage=usage,
                quality_summary={**quality, "status": "cancelled"},
            )
        if not refinement.text:
            # Preserve the current transcript if recognition fails; rough speaker
            # detection must not silently replace a successful continuous transcript.
            raise PartialTranscriptionError(
                "Continuous transcription returned no usable text.",
                attempts=refinement.attempts,
                usage=usage,
                quality_summary={**quality, "status": "failed"},
            )
        alignment = align_conversation(
            refinement.text, valid_turns,
            recording_duration_seconds=recording_duration,
        )
        hypotheses = tuple(
            ChunkHypothesis(
                track_id=track.track_id,
                # All segments reference the one complete-audio attempt. They
                # are placements of its words, not separately recognized crops.
                chunk_index=0,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                text=segment.text,
                speaker_label=segment.speaker_label,
                speaker_source=("unknown" if segment.speaker_label == "Unknown" else "openai_diarization"),
                transcription_model=refinement.model,
                audio_variant=refinement.audio_variant,
                quality_flags=tuple(dict.fromkeys((
                    *refinement.quality_flags,
                    "approximate_timestamps",
                    *(("speaker_alignment_uncertain", QUALITY_FLAG_HUMAN_REVIEW_RECOMMENDED)
                      if segment.speaker_label == "Unknown" else ()),
                ))),
            )
            for segment in alignment.segments
        )
        uncertain = alignment.uncertain_word_count > 0
        quality.update({
            "status": "completed_with_warning" if uncertain else "complete",
            "warning": (
                "Text was transcribed with the complete conversation. Speaker timing is approximate. "
                "Some words could not be assigned to a speaker reliably."
                if uncertain else
                "Text was transcribed with the complete conversation. Speaker timing is approximate."
            ),
            "alignment_match_ratio": alignment.match_ratio,
            "alignment_limit_exceeded": alignment.limit_exceeded,
            "uncertain_word_count": alignment.uncertain_word_count,
            "recognized_word_count": alignment.word_count,
            "degraded": False,
        })
        duration = pass1.processing_duration_seconds + refinement.processing_duration_seconds
        track_result = TrackTranscriptionResult(
            track_id=track.track_id,
            model=refinement.model or self.settings.OPENAI_TRANSCRIPTION_MODEL,
            language="el", prompt_version=manifest.prompt_identity,
            processing_duration_seconds=duration, hypotheses=hypotheses,
            usage=usage, diarized=True, attempts=refinement.attempts,
        )
        return OrchestratedTranscriptionResult(
            mode=plan.mode, model=track_result.model, language="el",
            prompt_version=manifest.prompt_identity,
            processing_duration_seconds=duration,
            segments=hypotheses, tracks=(track_result,), usage=usage,
            diarized=True, confidence_status="unavailable",
            attribution_status=plan.attribution_status, plan_reason=plan.reason,
            attempts=refinement.attempts, quality_summary=quality,
        )

    async def _refine_mono_span(
        self,
        client: OpenAITranscriptionClient,
        speech_chunk: SpeechChunk,
        *,
        prompt_plan: PromptPlan,
        context: TranscriptionContext,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> _MonoSpanRefinement:
        usage: dict[str, object] = {"chunks": [], "totals": {}}
        processing_duration = 0.0
        try:
            raw_attempt = await self._transcribe_v2_attempt(
                client,
                replace(
                    speech_chunk,
                    audio_variant=RAW_LOSSLESS_AUDIO_VARIANT,
                ),
                prompt_plan=prompt_plan,
                cancellation_check=cancellation_check,
            )
        except TranscriptionCancelledError as exc:
            raise _MonoSpanRefinementCancelled(
                attempts=(),
                usage=usage,
                processing_duration_seconds=processing_duration,
            ) from exc
        except Exception:
            return _MonoSpanRefinement(
                text=None,
                model=None,
                token_logprobs=(),
                mean_logprob=None,
                low_logprob_ratio=None,
                quality_flags=(),
                audio_variant=None,
                attempts=(),
                usage=usage,
                processing_duration_seconds=processing_duration,
                failure_reason="refinement_provider_failed",
            )

        raw_usage = self._merge_v2_usage(usage, raw_attempt.result.usage)
        processing_duration += raw_attempt.result.processing_duration_seconds
        normalized_attempt: _V2StandardAttempt | None = None
        normalized_usage: dict[str, object] = {}
        normalized_retry_failed = False
        if (
            self.confidence_policy.max_attempts_per_chunk > 1
            and is_low_confidence(raw_attempt.metrics, self.confidence_policy)
        ):
            if cancellation_check is not None and await cancellation_check():
                raise self._mono_cancelled_after_raw(
                    raw_attempt,
                    prompt_hash=prompt_plan.prompt_hash,
                    raw_usage=raw_usage,
                    usage=usage,
                    processing_duration_seconds=processing_duration,
                )
            try:
                normalized_chunk = await self.retry_quality_processor.prepare_retry(
                    speech_chunk,
                    destination_dir=context.temporary_directory,
                    register_temporary_file=context.register_temporary_file,
                )
                try:
                    normalized_attempt = await self._transcribe_v2_attempt(
                        client,
                        normalized_chunk,
                        prompt_plan=prompt_plan,
                        cancellation_check=cancellation_check,
                    )
                finally:
                    self.retry_quality_processor.cleanup_retry(normalized_chunk)
            except TranscriptionCancelledError as exc:
                raise self._mono_cancelled_after_raw(
                    raw_attempt,
                    prompt_hash=prompt_plan.prompt_hash,
                    raw_usage=raw_usage,
                    usage=usage,
                    processing_duration_seconds=processing_duration,
                ) from exc
            except Exception:
                normalized_retry_failed = True
            if normalized_attempt is not None:
                normalized_usage = self._merge_v2_usage(
                    usage,
                    normalized_attempt.result.usage,
                )
                processing_duration += (
                    normalized_attempt.result.processing_duration_seconds
                )

        raw_usable = bool(raw_attempt.response.response_text.strip())
        normalized_usable = bool(
            normalized_attempt is not None
            and normalized_attempt.response.response_text.strip()
        )
        selected_name: str | None
        if raw_usable and normalized_usable:
            selected_name = self._selected_v2_attempt_name(
                raw_attempt,
                normalized_attempt,
            )
        elif normalized_usable:
            selected_name = "normalized"
        elif raw_usable:
            selected_name = "raw"
        else:
            selected_name = None
        selected_attempt = (
            normalized_attempt
            if selected_name == "normalized" and normalized_attempt is not None
            else raw_attempt
        )
        evidence = [
            self._attempt_evidence(
                raw_attempt,
                prompt_hash=prompt_plan.prompt_hash,
                api_usage=raw_usage,
                selected=selected_name == "raw",
            )
        ]
        if normalized_attempt is not None:
            evidence.append(
                self._attempt_evidence(
                    normalized_attempt,
                    prompt_hash=prompt_plan.prompt_hash,
                    api_usage=normalized_usage,
                    selected=selected_name == "normalized",
                )
            )
        if selected_name is None:
            return _MonoSpanRefinement(
                text=None,
                model=None,
                token_logprobs=(),
                mean_logprob=None,
                low_logprob_ratio=None,
                quality_flags=(),
                audio_variant=None,
                attempts=tuple(evidence),
                usage=usage,
                processing_duration_seconds=processing_duration,
                failure_reason="refinement_empty_standard_text",
            )

        quality_flags = list(
            self._v2_quality_flags(
                raw_attempt.metrics,
                (
                    normalized_attempt.metrics
                    if normalized_attempt is not None
                    else None
                ),
                selected_attempt.metrics,
                normalized_retry_used=normalized_attempt is not None,
            )
        )
        if normalized_retry_failed:
            quality_flags.append(QUALITY_FLAG_NORMALIZED_RETRY_FAILED)
        return _MonoSpanRefinement(
            text=selected_attempt.response.response_text.strip(),
            model=selected_attempt.result.model,
            token_logprobs=selected_attempt.response.token_logprobs,
            mean_logprob=(
                selected_attempt.metrics.mean_logprob
                if selected_attempt.metrics is not None
                else None
            ),
            low_logprob_ratio=(
                selected_attempt.metrics.low_logprob_ratio
                if selected_attempt.metrics is not None
                else None
            ),
            quality_flags=tuple(quality_flags),
            audio_variant=selected_attempt.chunk.audio_variant,
            attempts=tuple(evidence),
            usage=usage,
            processing_duration_seconds=processing_duration,
        )

    def _mono_cancelled_after_raw(
        self,
        raw_attempt: _V2StandardAttempt,
        *,
        prompt_hash: str | None,
        raw_usage: dict[str, object],
        usage: dict[str, object],
        processing_duration_seconds: float,
    ) -> _MonoSpanRefinementCancelled:
        return _MonoSpanRefinementCancelled(
            attempts=(
                self._attempt_evidence(
                    raw_attempt,
                    prompt_hash=prompt_hash,
                    api_usage=raw_usage,
                    selected=True,
                ),
            ),
            usage=usage,
            processing_duration_seconds=processing_duration_seconds,
        )

    @staticmethod
    def _prepared_mono_frame_count(path: Path) -> int:
        try:
            with wave.open(str(path), "rb") as reader:
                if (
                    reader.getnchannels() != 1
                    or reader.getsampwidth() != 2
                    or reader.getframerate() != 16_000
                    or reader.getcomptype() != "NONE"
                ):
                    raise ValueError(
                        "Mono refinement requires 16 kHz mono PCM16 prepared audio."
                    )
                frame_count = reader.getnframes()
        except (EOFError, wave.Error) as exc:
            raise ValueError("Prepared mono audio is not a valid PCM WAV.") from exc
        if frame_count < 1:
            raise ValueError("Prepared mono audio has no samples.")
        return frame_count

    @staticmethod
    def _append_mono_context(
        previous_context: str,
        speaker_label: str,
        text: str,
        maximum_characters: int,
    ) -> str:
        entry = f"{speaker_label}: {' '.join(text.strip().split())}"
        combined = " ".join(value for value in (previous_context, entry) if value)
        return combined[-maximum_characters:].strip()

    @staticmethod
    def _mono_usage(
        pass1_usage: dict[str, object],
        pass2_usage: dict[str, object],
    ) -> dict[str, object]:
        return {
            "pass1_diarization": dict(pass1_usage),
            "pass2_refinement": pass2_usage,
        }

    def _mono_quality_summary(
        self,
        *,
        pass1: TranscriptionResult,
        pass1_turn_count: int,
        valid_turn_count: int,
        rejected_turn_count: int,
        spans: tuple[MonoRefinementSpan, ...],
        fallback_span_indexes: set[int],
        refined_span_count: int,
        processed_span_count: int,
        status_override: str | None = None,
    ) -> dict[str, object]:
        degraded = calculate_span_degraded_duration(
            spans,
            fallback_span_indexes=fallback_span_indexes,
            policy=self.mono_refinement_policy,
        )
        if status_override is not None:
            status = status_override
            warning: str | None = "Two-pass mono transcription did not finish."
        elif degraded.degraded:
            status = "degraded"
            warning = (
                "More than 20% of spoken duration uses rough diarization text "
                "because standard refinement was unavailable."
            )
        elif fallback_span_indexes:
            status = "completed_with_fallback"
            warning = (
                "Some spans use explicitly flagged rough diarization text "
                "without logprob confidence."
            )
        elif rejected_turn_count:
            status = "completed_with_warning"
            warning = "Malformed diarization turns were ignored safely."
        else:
            status = "complete"
            warning = None
        return {
            "policy_version": self.mono_refinement_policy.version,
            "status": status,
            "warning": warning,
            "pass1_completed": True,
            "pass1_model": pass1.model,
            "pass2_model": self.settings.OPENAI_TRANSCRIPTION_MODEL,
            "pass1_turn_count": pass1_turn_count,
            "valid_turn_count": valid_turn_count,
            "rejected_turn_count": rejected_turn_count,
            "coalesced_span_count": len(spans),
            "processed_span_count": processed_span_count,
            "refined_span_count": refined_span_count,
            "fallback_span_count": len(fallback_span_indexes),
            "spoken_duration_seconds": degraded.spoken_duration_seconds,
            "fallback_duration_seconds": degraded.fallback_duration_seconds,
            "fallback_duration_ratio": degraded.fallback_duration_ratio,
            "degraded_duration_ratio_threshold": degraded.threshold,
            "degraded": degraded.degraded,
            "max_refinement_spans": self.mono_refinement_policy.max_refinement_spans,
            "max_refinement_span_seconds": (
                self.mono_refinement_policy.max_coalesced_span_seconds
            ),
        }

    @staticmethod
    def _raise_mono_partial_cancellation(
        *,
        attempts: tuple[TranscriptionAttemptEvidence, ...],
        usage: dict[str, object],
        quality_summary: dict[str, object],
    ) -> None:
        raise PartialTranscriptionCancelledError(
            "Transcription was cancelled.",
            attempts=attempts,
            usage=usage,
            quality_summary=quality_summary,
        )

    @staticmethod
    def _mono_hypothesis(
        *,
        track: AudioTrack,
        span: MonoRefinementSpan,
        span_index: int,
        text: str,
        model: str,
        token_logprobs: tuple[TokenLogprob, ...] = (),
        mean_logprob: float | None = None,
        low_logprob_ratio: float | None = None,
        quality_flags: tuple[str, ...] = (),
        audio_variant: str | None = None,
    ) -> ChunkHypothesis:
        return ChunkHypothesis(
            track_id=track.track_id,
            chunk_index=span_index,
            start_seconds=span.start_seconds,
            end_seconds=span.end_seconds,
            text=text,
            speaker_label=span.speaker_label,
            token_logprobs=token_logprobs,
            mean_logprob=mean_logprob,
            low_logprob_ratio=low_logprob_ratio,
            quality_flags=quality_flags,
            audio_variant=audio_variant,
            channel_index=None,
            operator_id=None,
            speaker_source="openai_diarization",
            transcription_model=model,
        )

    async def _transcribe_v2_track(
        self,
        client: OpenAITranscriptionClient,
        track: AudioTrack,
        speech_chunks: tuple[SpeechChunk, ...],
        context: TranscriptionContext,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> TrackTranscriptionResult:
        manifest = context.v2_prompt_manifest
        if manifest is None:
            raise ValueError("V2 standard transcription requires a prompt manifest.")

        hypotheses: list[ChunkHypothesis] = []
        attempts: list[TranscriptionAttemptEvidence] = []
        usage: dict[str, object] = {"chunks": [], "totals": {}}
        processing_duration_seconds = 0.0
        model = self.settings.OPENAI_TRANSCRIPTION_MODEL
        language = "el"
        previous_context: str | None = None

        for speech_chunk in speech_chunks:
            prompt_plan = manifest.build(track.track_id, previous_context)
            raw_chunk = replace(
                speech_chunk,
                audio_variant=RAW_LOSSLESS_AUDIO_VARIANT,
            )
            try:
                raw_attempt = await self._transcribe_v2_attempt(
                    client,
                    raw_chunk,
                    prompt_plan=prompt_plan,
                    cancellation_check=cancellation_check,
                )
            except Exception as exc:
                self._raise_partial_transcription(exc, tuple(attempts))
            raw_usage: dict[str, object] = {}
            normalized_attempt: _V2StandardAttempt | None = None
            normalized_usage: dict[str, object] = {}
            try:
                raw_usage = self._merge_v2_usage(usage, raw_attempt.result.usage)
                processing_duration_seconds += (
                    raw_attempt.result.processing_duration_seconds
                )
                if (
                    self.confidence_policy.max_attempts_per_chunk > 1
                    and is_low_confidence(raw_attempt.metrics, self.confidence_policy)
                ):
                    if cancellation_check is not None and await cancellation_check():
                        raise TranscriptionCancelledError("Transcription was cancelled.")
                    normalized_chunk = await self.retry_quality_processor.prepare_retry(
                        speech_chunk,
                        destination_dir=context.temporary_directory,
                        register_temporary_file=context.register_temporary_file,
                    )
                    try:
                        normalized_attempt = await self._transcribe_v2_attempt(
                            client,
                            normalized_chunk,
                            prompt_plan=prompt_plan,
                            cancellation_check=cancellation_check,
                        )
                    finally:
                        self.retry_quality_processor.cleanup_retry(normalized_chunk)
                    normalized_usage = self._merge_v2_usage(
                        usage,
                        normalized_attempt.result.usage,
                    )
                    processing_duration_seconds += (
                        normalized_attempt.result.processing_duration_seconds
                    )

                selected_name = self._selected_v2_attempt_name(
                    raw_attempt,
                    normalized_attempt,
                )
                selected_attempt = (
                    normalized_attempt
                    if selected_name == "normalized" and normalized_attempt is not None
                    else raw_attempt
                )
                quality_flags = self._v2_quality_flags(
                    raw_attempt.metrics,
                    (
                        normalized_attempt.metrics
                        if normalized_attempt is not None
                        else None
                    ),
                    selected_attempt.metrics,
                    normalized_retry_used=normalized_attempt is not None,
                )
                model = selected_attempt.result.model
                language = selected_attempt.result.language

                chunk_result = self._track_result(
                    track,
                    (selected_attempt.chunk,),
                    selected_attempt.result,
                )
                selected_hypotheses = tuple(
                    replace(
                        hypothesis,
                        token_logprobs=selected_attempt.response.token_logprobs,
                        mean_logprob=(
                            selected_attempt.metrics.mean_logprob
                            if selected_attempt.metrics is not None
                            else None
                        ),
                        low_logprob_ratio=(
                            selected_attempt.metrics.low_logprob_ratio
                            if selected_attempt.metrics is not None
                            else None
                        ),
                        quality_flags=quality_flags,
                        audio_variant=selected_attempt.chunk.audio_variant,
                    )
                    for hypothesis in chunk_result.hypotheses
                )
                updated_hypotheses = list(
                    self._remove_exact_hard_cut_overlap(
                        [*hypotheses, *selected_hypotheses]
                    )
                )
                if len(updated_hypotheses) > len(hypotheses):
                    previous_context = updated_hypotheses[-1].text
                hypotheses = updated_hypotheses

                chunk_attempts = [
                    self._attempt_evidence(
                        raw_attempt,
                        prompt_hash=prompt_plan.prompt_hash,
                        api_usage=raw_usage,
                        selected=selected_name == "raw",
                    )
                ]
                if normalized_attempt is not None:
                    chunk_attempts.append(
                        self._attempt_evidence(
                            normalized_attempt,
                            prompt_hash=prompt_plan.prompt_hash,
                            api_usage=normalized_usage,
                            selected=selected_name == "normalized",
                        )
                    )
                attempts.extend(chunk_attempts)
            except (
                PartialTranscriptionError,
                PartialTranscriptionCancelledError,
            ):
                raise
            except Exception as exc:
                partial_chunk_attempts = self._partial_chunk_attempts(
                    raw_attempt,
                    normalized_attempt,
                    prompt_hash=prompt_plan.prompt_hash,
                    raw_usage=raw_usage,
                    normalized_usage=normalized_usage,
                )
                self._raise_partial_transcription(
                    exc,
                    (*attempts, *partial_chunk_attempts),
                )

        return TrackTranscriptionResult(
            track_id=track.track_id,
            model=model,
            language=language,
            prompt_version=manifest.prompt_identity,
            processing_duration_seconds=processing_duration_seconds,
            hypotheses=tuple(hypotheses),
            usage=usage,
            diarized=False,
            attempts=tuple(attempts),
        )

    async def _transcribe_v2_attempt(
        self,
        client: OpenAITranscriptionClient,
        chunk: SpeechChunk,
        *,
        prompt_plan: PromptPlan,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> _V2StandardAttempt:
        provider_chunk = AudioChunk(
            path=chunk.path,
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds,
            chunk_index=chunk.chunk_index,
            hard_cut=chunk.hard_cut,
            overlap_before_ms=chunk.overlap_before_ms,
        )
        result = await client.transcribe_isolated(
            [provider_chunk],
            [],
            language="el",
            should_cancel=cancellation_check,
            prompt_plan=prompt_plan,
            request_logprobs=True,
        )
        response = self._v2_response_evidence(result, chunk.chunk_index)
        metrics = (
            calculate_confidence_metrics(
                response.token_logprobs,
                self.confidence_policy,
            )
            if response.logprobs_available
            else None
        )
        return _V2StandardAttempt(
            chunk=chunk,
            result=result,
            response=response,
            metrics=metrics,
            completed_at=utc_now(),
        )

    @staticmethod
    def _v2_response_evidence(
        result: TranscriptionResult,
        chunk_index: int,
    ) -> TranscriptionResponseEvidence:
        for response in result.response_evidence:
            if response.source_chunk_index == chunk_index:
                return response
        if len(result.response_evidence) == 1:
            return result.response_evidence[0]
        return TranscriptionResponseEvidence(
            source_chunk_index=chunk_index,
            response_text=result.text,
            token_logprobs=(),
            logprobs_available=False,
        )

    @staticmethod
    def _attempt_evidence(
        attempt: _V2StandardAttempt,
        *,
        prompt_hash: str | None,
        api_usage: dict[str, object],
        selected: bool,
    ) -> TranscriptionAttemptEvidence:
        return TranscriptionAttemptEvidence(
            track_id=attempt.chunk.track_id,
            chunk_index=attempt.chunk.chunk_index,
            start_seconds=attempt.chunk.start_seconds,
            end_seconds=attempt.chunk.end_seconds,
            model=attempt.result.model,
            audio_variant=attempt.chunk.audio_variant,
            prompt_hash=prompt_hash,
            response_text=attempt.response.response_text,
            mean_logprob=(
                attempt.metrics.mean_logprob if attempt.metrics is not None else None
            ),
            low_logprob_ratio=(
                attempt.metrics.low_logprob_ratio
                if attempt.metrics is not None
                else None
            ),
            selected=selected,
            api_usage=api_usage,
            completed_at=attempt.completed_at,
        )

    def _selected_v2_attempt_name(
        self,
        raw_attempt: _V2StandardAttempt,
        normalized_attempt: _V2StandardAttempt | None,
    ) -> str:
        if normalized_attempt is None:
            return "raw"
        return select_preferred_attempt(
            raw_attempt.metrics,
            normalized_attempt.metrics,
            self.confidence_policy,
        )

    def _partial_chunk_attempts(
        self,
        raw_attempt: _V2StandardAttempt,
        normalized_attempt: _V2StandardAttempt | None,
        *,
        prompt_hash: str | None,
        raw_usage: dict[str, object],
        normalized_usage: dict[str, object],
    ) -> tuple[TranscriptionAttemptEvidence, ...]:
        selected_name = self._selected_v2_attempt_name(raw_attempt, normalized_attempt)
        evidence = [
            self._attempt_evidence(
                raw_attempt,
                prompt_hash=prompt_hash,
                api_usage=raw_usage,
                selected=selected_name == "raw",
            )
        ]
        if normalized_attempt is not None:
            evidence.append(
                self._attempt_evidence(
                    normalized_attempt,
                    prompt_hash=prompt_hash,
                    api_usage=normalized_usage,
                    selected=selected_name == "normalized",
                )
            )
        return tuple(evidence)

    @staticmethod
    def _raise_partial_transcription(
        exc: Exception,
        attempts: tuple[TranscriptionAttemptEvidence, ...],
    ) -> None:
        message = str(exc) or "Transcription did not complete."
        if isinstance(exc, PartialTranscriptionCancelledError):
            raise PartialTranscriptionCancelledError(
                message,
                attempts=attempts,
                usage=exc.usage,
                quality_summary=exc.quality_summary,
            ) from exc
        if isinstance(exc, TranscriptionCancelledError):
            raise PartialTranscriptionCancelledError(
                message,
                attempts=attempts,
            ) from exc
        raise PartialTranscriptionError(
            message,
            attempts=attempts,
            category=getattr(exc, "category", None),
            usage=(
                exc.usage if isinstance(exc, PartialTranscriptionError) else None
            ),
            quality_summary=(
                exc.quality_summary
                if isinstance(exc, PartialTranscriptionError)
                else None
            ),
        ) from exc

    def _v2_quality_flags(
        self,
        raw_metrics: ConfidenceMetrics | None,
        normalized_metrics: ConfidenceMetrics | None,
        selected_metrics: ConfidenceMetrics | None,
        *,
        normalized_retry_used: bool,
    ) -> tuple[str, ...]:
        flags: list[str] = []
        if normalized_retry_used:
            flags.append(QUALITY_FLAG_NORMALIZED_RETRY_USED)
        if selected_metrics is None:
            flags.append(QUALITY_FLAG_LOGPROBS_UNAVAILABLE)
            return tuple(flags)
        if is_low_confidence(selected_metrics, self.confidence_policy):
            flags.append(QUALITY_FLAG_LOW_CONFIDENCE)
        if (
            normalized_retry_used
            and is_low_confidence(raw_metrics, self.confidence_policy)
            and is_low_confidence(normalized_metrics, self.confidence_policy)
        ):
            flags.extend(
                (
                    QUALITY_FLAG_BOTH_ATTEMPTS_LOW_CONFIDENCE,
                    QUALITY_FLAG_HUMAN_REVIEW_RECOMMENDED,
                )
            )
        return tuple(flags)

    @staticmethod
    def _merge_v2_usage(
        aggregate: dict[str, object],
        current: dict[str, object],
    ) -> dict[str, object]:
        aggregate_chunks = aggregate["chunks"]
        aggregate_totals = aggregate["totals"]
        if not isinstance(aggregate_chunks, list) or not isinstance(aggregate_totals, dict):
            raise TypeError("V2 usage aggregation was initialized incorrectly.")

        raw_chunks = current.get("chunks")
        chunk_usage: dict[str, object] = {}
        if isinstance(raw_chunks, list):
            for raw_chunk in raw_chunks:
                if isinstance(raw_chunk, dict):
                    safe_chunk = dict(raw_chunk)
                    aggregate_chunks.append(safe_chunk)
                    if not chunk_usage:
                        chunk_usage = safe_chunk
        elif current:
            chunk_usage = dict(current)
            aggregate_chunks.append(chunk_usage)

        raw_totals = current.get("totals")
        if isinstance(raw_totals, dict):
            for key, value in raw_totals.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    aggregate_totals[key] = aggregate_totals.get(key, 0) + value
        else:
            for key, value in chunk_usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    aggregate_totals[key] = aggregate_totals.get(key, 0) + value
        return chunk_usage

    def _empty_track_result(
        self,
        track: AudioTrack,
        context: TranscriptionContext,
    ) -> TrackTranscriptionResult:
        prompt_version = (
            None
            if track.diarized
            else (
                context.v2_prompt_manifest.prompt_identity
                if context.v2_prompt_manifest is not None
                else self.prompt_builder.build(context.vocabulary).version
            )
        )
        return TrackTranscriptionResult(
            track_id=track.track_id,
            model=(
                self.settings.OPENAI_DIARIZATION_MODEL
                if track.diarized
                else self.settings.OPENAI_TRANSCRIPTION_MODEL
            ),
            language=context.language,
            prompt_version=prompt_version,
            processing_duration_seconds=0,
            hypotheses=(),
            usage={"chunks": [], "totals": {}},
            diarized=track.diarized,
        )

    async def _prepare_tracks(
        self,
        plan: AudioPlan,
        *,
        destination_dir: Path,
        register_temporary_file: Callable[[Path], None] | None,
        cancellation_check: Callable[[], Awaitable[bool]] | None,
    ) -> tuple[AudioTrack, ...]:
        if plan.mode == "legacy":
            return plan.tracks
        destination_dir.mkdir(parents=True, exist_ok=True)
        prepared: list[AudioTrack] = []
        for track in plan.tracks:
            if cancellation_check is not None and await cancellation_check():
                raise TranscriptionCancelledError("Transcription was cancelled.")
            destination = destination_dir / f"{track.track_id}.wav"
            # Register before the audio side effect so cleanup also owns a
            # destination left behind by a failing processor implementation.
            if register_temporary_file is not None:
                register_temporary_file(destination)
            if plan.mode in {"operator_channel", "dual_channel"}:
                if track.channel_index not in {0, 1}:
                    raise ValueError("A separated-stereo track requires channel 0 or 1.")
                await self.audio_processor.extract_channel(
                    track.source_path,
                    destination,
                    track.channel_index,
                )
            elif plan.mode == "mono_diarization":
                await self.audio_processor.convert_to_mono(track.source_path, destination)
            else:
                raise ValueError(f"Unsupported transcription mode: {plan.mode}")
            prepared.append(replace(track, source_path=destination))
        return tuple(prepared)

    @staticmethod
    def _track_result(
        track: AudioTrack,
        chunks: tuple[SpeechChunk, ...],
        result: TranscriptionResult,
    ) -> TrackTranscriptionResult:
        hypotheses: list[ChunkHypothesis] = []
        for segment in result.segments:
            source_chunk = TranscriptionOrchestrator._source_chunk(
                chunks,
                segment.start_seconds,
                segment.end_seconds,
                segment.source_chunk_index,
            )
            hypotheses.append(
                ChunkHypothesis(
                    track_id=track.track_id,
                    chunk_index=(source_chunk.chunk_index if source_chunk is not None else None),
                    start_seconds=segment.start_seconds,
                    end_seconds=segment.end_seconds,
                    text=segment.text,
                    speaker_label=track.speaker_label or segment.speaker_label,
                    confidence=segment.confidence,
                    audio_variant=(
                        source_chunk.audio_variant
                        if source_chunk is not None
                        else track.audio_variant
                    ),
                    hard_cut=source_chunk.hard_cut if source_chunk is not None else False,
                    overlap_before_ms=(
                        source_chunk.overlap_before_ms if source_chunk is not None else 0
                    ),
                    channel_index=track.channel_index,
                    operator_id=track.operator_id,
                    speaker_source=track.speaker_source,
                )
            )
        hypotheses = list(TranscriptionOrchestrator._remove_exact_hard_cut_overlap(hypotheses))
        return TrackTranscriptionResult(
            track_id=track.track_id,
            model=result.model,
            language=result.language,
            prompt_version=result.prompt_version,
            processing_duration_seconds=result.processing_duration_seconds,
            hypotheses=tuple(hypotheses),
            usage=result.usage,
            diarized=result.diarized,
        )

    @staticmethod
    def _source_chunk(
        chunks: tuple[SpeechChunk, ...],
        start_seconds: float,
        end_seconds: float,
        source_chunk_index: int | None,
    ) -> SpeechChunk | None:
        if source_chunk_index is not None:
            for chunk in chunks:
                if chunk.chunk_index == source_chunk_index:
                    return chunk
        for chunk in chunks:
            if chunk.start_seconds == start_seconds and chunk.end_seconds == end_seconds:
                return chunk
        for position, chunk in reversed(tuple(enumerate(chunks))):
            is_last = position == len(chunks) - 1
            if chunk.start_seconds <= start_seconds < chunk.end_seconds:
                return chunk
            if is_last and start_seconds == chunk.end_seconds:
                return chunk
        return None

    @staticmethod
    def _remove_exact_hard_cut_overlap(
        hypotheses: list[ChunkHypothesis],
    ) -> tuple[ChunkHypothesis, ...]:
        result: list[ChunkHypothesis] = []
        for hypothesis in hypotheses:
            if result:
                previous = result[-1]
                adjacent = (
                    previous.track_id == hypothesis.track_id
                    and previous.chunk_index is not None
                    and hypothesis.chunk_index == previous.chunk_index + 1
                    and previous.hard_cut
                    and hypothesis.overlap_before_ms > 0
                )
                if adjacent:
                    trimmed = TranscriptionOrchestrator._trim_exact_token_overlap(
                        previous.text,
                        hypothesis.text,
                    )
                    hypothesis = replace(hypothesis, text=trimmed)
            if hypothesis.text.strip():
                result.append(hypothesis)
        return tuple(result)

    @staticmethod
    def _trim_exact_token_overlap(previous: str, current: str) -> str:
        previous_tokens = list(re.finditer(r"\S+", previous))
        current_tokens = list(re.finditer(r"\S+", current))
        maximum = min(
            len(previous_tokens),
            len(current_tokens),
            EXACT_OVERLAP_MAX_TOKENS,
        )
        for count in range(maximum, EXACT_OVERLAP_MIN_TOKENS - 1, -1):
            previous_values = [match.group(0) for match in previous_tokens[-count:]]
            current_values = [match.group(0) for match in current_tokens[:count]]
            if previous_values != current_values:
                continue
            if len(" ".join(current_values)) < EXACT_OVERLAP_MIN_CHARACTERS:
                continue
            return current[current_tokens[count - 1].end() :].lstrip()
        return current
