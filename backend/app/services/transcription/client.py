from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Awaitable, Callable
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


@dataclass(frozen=True)
class TranscriptionResult:
    model: str
    language: str
    prompt_version: str | None
    processing_duration_seconds: float
    segments: list[TranscribedSegment]
    usage: dict[str, Any] = field(default_factory=dict)
    diarized: bool = False

    @property
    def text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


def build_vocabulary_prompt(values: list[str], *, max_characters: int = 4000) -> tuple[str, str]:
    clean: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = " ".join(value.strip().split())
        key = value.casefold()
        if value and key not in seen:
            clean.append(value)
            seen.add(key)
    prompt = "Greek business vocabulary and names: " + ", ".join(clean)
    prompt = prompt[:max_characters]
    version = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return prompt, version


def _model_dump(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return value
    return {}


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
            self._client = AsyncOpenAI(api_key=self.settings.OPENAI_API_KEY, timeout=120.0, max_retries=2)
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
            return TranscriptionError("Transcription credentials were rejected.", "openai_authentication")
        if isinstance(exc, RateLimitError):
            return TranscriptionError("Transcription service is busy. Retry later.", "openai_rate_limit")
        if isinstance(exc, APITimeoutError):
            return TranscriptionError("Transcription request timed out.", "openai_timeout")
        if isinstance(exc, APIConnectionError):
            return TranscriptionError("Could not connect to the transcription service.", "openai_connection")
        if isinstance(exc, APIStatusError):
            if exc.status_code >= 500:
                return TranscriptionError("Transcription service is unavailable.", "openai_unavailable")
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
    ) -> TranscriptionResult:
        client = self._require_client()
        language = language or self.settings.TRANSCRIPTION_LANGUAGE
        prompt, prompt_version = build_vocabulary_prompt(vocabulary)
        segments: list[TranscribedSegment] = []
        usage: dict[str, Any] = {"chunks": [], "totals": {}}
        started = time.monotonic()
        for chunk in chunks:
            if should_cancel is not None and await should_cancel():
                raise TranscriptionCancelledError("Transcription was cancelled.")
            if chunk.path.stat().st_size > self.settings.max_transcription_upload_bytes:
                raise TranscriptionError("Audio chunk exceeds the transcription upload limit.")
            with chunk.path.open("rb") as audio_file:
                try:
                    response = await client.audio.transcriptions.create(
                        file=audio_file,
                        model=self.settings.OPENAI_TRANSCRIPTION_MODEL,
                        language=language,
                        prompt=prompt or None,
                        response_format="json",
                    )
                except Exception as exc:
                    raise self._translated_error(exc) from exc
            text = str(getattr(response, "text", "") or "").strip()
            if text:
                # This model does not expose word timestamps. Exact source chunk boundaries are retained.
                segments.append(
                    TranscribedSegment(chunk.start_seconds, chunk.end_seconds, text, "Operator")
                )
            chunk_usage = _model_dump(getattr(response, "usage", None))
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
