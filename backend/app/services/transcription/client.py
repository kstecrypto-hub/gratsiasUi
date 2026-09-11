from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)

from app.core.config import Settings, get_settings
from app.services.audio import AudioChunk
from app.services.transcription.prompt import (
    LegacyVocabularyPromptBuilder,
    PromptPlan,
    build_vocabulary_prompt as _build_vocabulary_prompt,
)
from app.services.transcription.confidence import parse_token_logprobs
from app.services.transcription.types import TokenLogprob


LOGPROB_SUPPORTED_TRANSCRIPTION_MODELS = frozenset(
    {
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-12-15",
    }
)
TRANSCRIPTION_LOGPROB_CONTRACT_VERSION = "openai-transcription-json-logprobs-v1"
V2_TRANSCRIPTION_TEMPERATURE = 0.0


class TranscriptionError(Exception):
    category = "transcription"

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        if category:
            self.category = category


class TranscriptionConfigurationError(TranscriptionError):
    category = "not_configured"


class TranscriptionCancelledError(TranscriptionError):
    category = "cancelled"


@dataclass(frozen=True)
class TranscribedSegment:
    start_seconds: float
    end_seconds: float
    text: str
    speaker_label: str
    confidence: float | None = None
    source_chunk_index: int | None = None


@dataclass(frozen=True)
class TranscriptionResponseEvidence:
    source_chunk_index: int | None
    response_text: str = field(repr=False)
    token_logprobs: tuple[TokenLogprob, ...] = ()
    logprobs_available: bool = False


@dataclass(frozen=True)
class TranscriptionResult:
    model: str
    language: str
    prompt_version: str | None
    processing_duration_seconds: float
    segments: list[TranscribedSegment]
    usage: dict[str, Any] = field(default_factory=dict)
    diarized: bool = False
    response_evidence: tuple[TranscriptionResponseEvidence, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


def build_vocabulary_prompt(
    values: list[str] | tuple[str, ...],
    *,
    max_characters: int = 4000,
) -> tuple[str, str]:
    """Keep the original public helper available for existing callers."""

    return _build_vocabulary_prompt(values, max_characters=max_characters)


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _response_value(value: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def model_supports_transcription_logprobs(model: str) -> bool:
    return model in LOGPROB_SUPPORTED_TRANSCRIPTION_MODELS


class OpenAITranscriptionClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "OpenAITranscriptionClient":
        self._require_client()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.close()
            self._client = None

    def _require_client(self) -> AsyncOpenAI:
        if not self.settings.openai_configured:
            raise TranscriptionConfigurationError(
                "OpenAI integration is not configured. Add an API key in Settings."
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.settings.OPENAI_API_KEY, timeout=120.0, max_retries=2
            )
        return self._client

    async def test_connection(self) -> bool:
        client = self._require_client()
        # Models retrieval verifies credentials without uploading audio or exposing model details to users.
        try:
            await client.models.retrieve(self.settings.OPENAI_TRANSCRIPTION_MODEL)
            if self.settings.OPENAI_DIARIZATION_MODEL != self.settings.OPENAI_TRANSCRIPTION_MODEL:
                await client.models.retrieve(self.settings.OPENAI_DIARIZATION_MODEL)
        except Exception as exc:
            raise self._translated_error(exc) from exc
        return True

    @staticmethod
    def _translated_error(exc: Exception) -> TranscriptionError:
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return TranscriptionError(
                "Transcription credentials were rejected.", "openai_authentication"
            )
        if isinstance(exc, RateLimitError):
            return TranscriptionError(
                "Transcription service is busy. Retry later.", "openai_rate_limit"
            )
        if isinstance(exc, APITimeoutError):
            return TranscriptionError("Transcription request timed out.", "openai_timeout")
        if isinstance(exc, APIConnectionError):
            return TranscriptionError(
                "Could not connect to the transcription service.", "openai_connection"
            )
        if isinstance(exc, APIStatusError):
            if exc.status_code >= 500:
                return TranscriptionError(
                    "Transcription service is unavailable.", "openai_unavailable"
                )
            return TranscriptionError("Transcription request was rejected.", "openai_request")
        if isinstance(exc, TranscriptionError):
            return exc
        return TranscriptionError("Transcription failed unexpectedly.", "openai_unexpected")

    async def transcribe_isolated(
        self,
        chunks: list[AudioChunk],
        vocabulary: list[str],
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
        prompt_plan: PromptPlan | None = None,
        request_logprobs: bool = False,
    ) -> TranscriptionResult:
        client = self._require_client()
        language = language or self.settings.TRANSCRIPTION_LANGUAGE
        prompt_plan = prompt_plan or LegacyVocabularyPromptBuilder().build(vocabulary)
        prompt = prompt_plan.text
        prompt_version = prompt_plan.version
        segments: list[TranscribedSegment] = []
        response_evidence: list[TranscriptionResponseEvidence] = []
        usage: dict[str, Any] = {"chunks": [], "totals": {}}
        started = time.monotonic()
        logprob_contract_supported = (
            request_logprobs
            and model_supports_transcription_logprobs(
                self.settings.OPENAI_TRANSCRIPTION_MODEL
            )
        )
        for chunk in chunks:
            if should_cancel is not None and await should_cancel():
                raise TranscriptionCancelledError("Transcription was cancelled.")
            if chunk.path.stat().st_size > self.settings.max_transcription_upload_bytes:
                raise TranscriptionError("Audio chunk exceeds the transcription upload limit.")
            with chunk.path.open("rb") as audio_file:
                try:
                    if self.settings.OPENAI_TRANSCRIPTION_MODEL == "gpt-transcribe":
                        # GPT Transcribe uses plural language hints and does not
                        # expose token logprobs. Preserve the V2 upload budget
                        # without sending unsupported legacy request fields.
                        request_client = (
                            client.with_options(max_retries=0) if request_logprobs else client
                        )
                        response = await request_client.audio.transcriptions.create(
                            file=audio_file,
                            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
                            prompt=prompt or None,
                            response_format="json",
                            extra_body={"languages": [language]},
                        )
                    elif logprob_contract_supported:
                        response = await client.with_options(
                            max_retries=0
                        ).audio.transcriptions.create(
                            file=audio_file,
                            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
                            language=language,
                            prompt=prompt or None,
                            response_format="json",
                            include=["logprobs"],
                            temperature=V2_TRANSCRIPTION_TEMPERATURE,
                        )
                    elif request_logprobs:
                        # Unsupported configured models remain usable without
                        # fabricating evidence. Transport retries are still
                        # disabled so a V2 chunk cannot exceed its upload cap.
                        response = await client.with_options(
                            max_retries=0
                        ).audio.transcriptions.create(
                            file=audio_file,
                            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
                            language=language,
                            prompt=prompt or None,
                            response_format="json",
                        )
                    else:
                        # This branch is the frozen legacy request contract.
                        response = await client.audio.transcriptions.create(
                            file=audio_file,
                            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
                            language=language,
                            prompt=prompt or None,
                            response_format="json",
                        )
                except Exception as exc:
                    raise self._translated_error(exc) from exc
            response_text = str(_response_value(response, "text", "") or "")
            text = response_text.strip()
            if request_logprobs:
                raw_logprobs = _response_value(response, "logprobs")
                logprobs_available = (
                    logprob_contract_supported
                    and isinstance(raw_logprobs, Sequence)
                    and not isinstance(raw_logprobs, (str, bytes, bytearray))
                )
                response_evidence.append(
                    TranscriptionResponseEvidence(
                        source_chunk_index=chunk.chunk_index,
                        response_text=response_text,
                        token_logprobs=(
                            parse_token_logprobs(response)
                            if logprob_contract_supported
                            else ()
                        ),
                        logprobs_available=logprobs_available,
                    )
                )
            if text:
                # This model does not expose word timestamps. Exact source chunk boundaries are retained.
                segments.append(
                    TranscribedSegment(
                        chunk.start_seconds,
                        chunk.end_seconds,
                        text,
                        "Operator",
                        source_chunk_index=chunk.chunk_index,
                    )
                )
            chunk_usage = _model_dump(_response_value(response, "usage"))
            usage["chunks"].append(chunk_usage)
            for key, value in chunk_usage.items():
                if isinstance(value, (int, float)):
                    usage["totals"][key] = usage["totals"].get(key, 0) + value
        return TranscriptionResult(
            model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
            language=language,
            prompt_version=prompt_version,
            processing_duration_seconds=time.monotonic() - started,
            segments=segments,
            usage=usage,
            diarized=False,
            response_evidence=tuple(response_evidence),
        )

    async def transcribe_diarized(
        self,
        chunks: list[AudioChunk],
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> TranscriptionResult:
        client = self._require_client()
        language = language or self.settings.TRANSCRIPTION_LANGUAGE
        segments: list[TranscribedSegment] = []
        usage: dict[str, Any] = {"chunks": [], "totals": {}}
        started = time.monotonic()
        for chunk_index, chunk in enumerate(chunks):
            if should_cancel is not None and await should_cancel():
                raise TranscriptionCancelledError("Transcription was cancelled.")
            if chunk.path.stat().st_size > self.settings.max_transcription_upload_bytes:
                raise TranscriptionError("Audio chunk exceeds the transcription upload limit.")
            with chunk.path.open("rb") as audio_file:
                try:
                    response = await client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.settings.OPENAI_DIARIZATION_MODEL,
                        language=language,
                        response_format="diarized_json",
                        chunking_strategy="auto",
                    )
                except Exception as exc:
                    raise self._translated_error(exc) from exc
            response_data = _model_dump(response)
            for raw in response_data.get("segments", []) or []:
                if not isinstance(raw, dict) or not str(raw.get("text", "")).strip():
                    continue
                chunk_duration = max(0.0, chunk.end_seconds - chunk.start_seconds)
                relative_start = float(raw.get("start", 0))
                relative_end = float(raw.get("end", relative_start))
                if not math.isfinite(relative_start) or not math.isfinite(relative_end):
                    continue
                relative_start = min(chunk_duration, max(0.0, relative_start))
                relative_end = min(chunk_duration, max(relative_start, relative_end))
                segments.append(
                    TranscribedSegment(
                        chunk.start_seconds + relative_start,
                        min(chunk.end_seconds, chunk.start_seconds + relative_end),
                        str(raw["text"]).strip(),
                        f"chunk-{chunk_index + 1}:{str(raw.get('speaker') or 'Unknown speaker')}",
                        source_chunk_index=chunk.chunk_index,
                    )
                )
            chunk_usage = _model_dump(response_data.get("usage"))
            usage["chunks"].append(chunk_usage)
            for key, value in chunk_usage.items():
                if isinstance(value, (int, float)):
                    usage["totals"][key] = usage["totals"].get(key, 0) + value
        return TranscriptionResult(
            model=self.settings.OPENAI_DIARIZATION_MODEL,
            language=language,
            prompt_version=None,
            processing_duration_seconds=time.monotonic() - started,
            segments=segments,
            usage=usage,
            diarized=True,
        )

    async def transcribe_diarized_complete(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
    ) -> TranscriptionResult:
        """Run the V2 anonymous pass over one complete prepared recording.

        This intentionally remains separate from the frozen legacy multi-chunk
        method above. Anonymous labels are meaningful only within this one
        provider response, so the V2 path must never split and namespace them.
        """

        client = self._require_client()
        language = language or self.settings.TRANSCRIPTION_LANGUAGE
        if should_cancel is not None and await should_cancel():
            raise TranscriptionCancelledError("Transcription was cancelled.")
        if chunk.path.stat().st_size > self.settings.max_transcription_upload_bytes:
            raise TranscriptionError(
                "The complete prepared recording exceeds the diarization upload limit.",
                "mono_diarization_upload_limit",
            )

        started = time.monotonic()
        with chunk.path.open("rb") as audio_file:
            try:
                response = await client.with_options(
                    max_retries=0
                ).audio.transcriptions.create(
                    file=audio_file,
                    model=self.settings.OPENAI_DIARIZATION_MODEL,
                    language=language,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
            except Exception as exc:
                raise self._translated_error(exc) from exc

        response_data = _model_dump(response)
        segments: list[TranscribedSegment] = []
        raw_segments = response_data.get("segments", [])
        if isinstance(raw_segments, Sequence) and not isinstance(
            raw_segments,
            (str, bytes, bytearray),
        ):
            for raw in raw_segments:
                if not isinstance(raw, Mapping):
                    continue
                try:
                    start_seconds = float(raw.get("start"))
                    end_seconds = float(raw.get("end"))
                except (TypeError, ValueError):
                    continue
                speaker = raw.get("speaker")
                text = raw.get("text")
                segments.append(
                    TranscribedSegment(
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        text=text if isinstance(text, str) else "",
                        speaker_label=speaker if isinstance(speaker, str) else "",
                        source_chunk_index=chunk.chunk_index,
                    )
                )

        provider_usage = _model_dump(response_data.get("usage"))
        totals = {
            key: value
            for key, value in provider_usage.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        usage: dict[str, Any] = {
            "chunks": [provider_usage],
            "totals": totals,
            "provider_segment_count": len(segments),
        }
        try:
            provider_duration = float(response_data.get("duration"))
        except (TypeError, ValueError):
            provider_duration = math.nan
        if math.isfinite(provider_duration) and provider_duration >= 0:
            usage["provider_duration_seconds"] = provider_duration

        return TranscriptionResult(
            model=self.settings.OPENAI_DIARIZATION_MODEL,
            language=language,
            prompt_version=None,
            processing_duration_seconds=time.monotonic() - started,
            segments=segments,
            usage=usage,
            diarized=True,
        )
