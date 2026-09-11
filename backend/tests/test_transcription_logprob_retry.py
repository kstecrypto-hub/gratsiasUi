from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.models.entities import TranscriptSegment, TranscriptionAttempt
from app.services.audio import AudioChunk
from app.services.audio.processor import AudioProcessor
from app.services.audio.quality import (
    LIGHT_NORMALIZATION_PROFILE,
    LIGHT_NORMALIZED_AUDIO_VARIANT,
    RAW_LOSSLESS_AUDIO_VARIANT,
    LightNormalizedRetryQualityProcessor,
)
from app.services.transcription.client import (
    OpenAITranscriptionClient,
    TranscribedSegment,
    TranscriptionResponseEvidence,
    TranscriptionResult,
)
from app.services.transcription.confidence import ConfidencePolicy
from app.services.transcription.orchestrator import (
    QUALITY_FLAG_BOTH_ATTEMPTS_LOW_CONFIDENCE,
    QUALITY_FLAG_HUMAN_REVIEW_RECOMMENDED,
    QUALITY_FLAG_LOGPROBS_UNAVAILABLE,
    QUALITY_FLAG_LOW_CONFIDENCE,
    QUALITY_FLAG_NORMALIZED_RETRY_USED,
    PartialTranscriptionCancelledError,
    PartialTranscriptionError,
    TranscriptionContext,
    TranscriptionOrchestrator,
)
from app.services.transcription.prompt import PromptPlan, V2GreekPromptBuilder
from app.services.transcription.types import (
    AudioPlan,
    AudioTrack,
    ChunkHypothesis,
    OrchestratedTranscriptionResult,
    SpeechChunk,
    TokenLogprob,
    TranscriptionAttemptEvidence,
)
from app.workers.pipeline import (
    _persist_transcription_result,
    _pipeline_v2_runtime_config_hash,
)


def _settings(
    tmp_path: Path,
    *,
    model: str = "gpt-4o-transcribe",
) -> Settings:
    return Settings(
        APP_ENV="test",
        STORAGE_ROOT=tmp_path,
        OPENAI_API_KEY="test-only-key",
        OPENAI_TRANSCRIPTION_MODEL=model,
    )


def _audio_chunk(tmp_path: Path, *, chunk_index: int = 0) -> AudioChunk:
    path = tmp_path / f"provider-{chunk_index}.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    return AudioChunk(
        path=path,
        start_seconds=float(chunk_index * 10),
        end_seconds=float((chunk_index + 1) * 10),
        chunk_index=chunk_index,
    )


class _ProviderClient:
    def __init__(self, *responses: object) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []
        self.with_options_calls: list[dict[str, Any]] = []
        self.audio = SimpleNamespace(
            transcriptions=SimpleNamespace(create=self._create),
        )
        self.models = SimpleNamespace()

    async def _create(self, **kwargs: Any) -> object:
        self.requests.append(dict(kwargs))
        return next(self._responses)

    def with_options(self, **kwargs: Any) -> SimpleNamespace:
        self.with_options_calls.append(dict(kwargs))
        return SimpleNamespace(
            audio=SimpleNamespace(
                transcriptions=SimpleNamespace(create=self._create),
            )
        )


def _prompt_plan() -> PromptPlan:
    return PromptPlan(
        text="Ελληνικό τηλεφωνικό αίτημα",
        version="prompt-contract-v1",
        prompt_hash="a" * 64,
    )


def _request_without_file(request: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in request.items() if key != "file"}


@pytest.mark.asyncio
@pytest.mark.parametrize("request_logprobs", [False, True])
async def test_gpt_transcribe_uses_plural_languages_without_fabricated_confidence(
    tmp_path: Path, request_logprobs: bool,
) -> None:
    provider = _ProviderClient({
        "text": "Το αυτοκίνητο έχει επισκευαστεί.",
        "languages": [{"code": "el"}],
        "usage": {"type": "duration", "seconds": 10},
    })
    client = OpenAITranscriptionClient(
        _settings(tmp_path, model="gpt-transcribe"),
        client=provider,  # type: ignore[arg-type]
    )
    result = await client.transcribe_isolated(
        [_audio_chunk(tmp_path)], [], language="el",
        prompt_plan=_prompt_plan(), request_logprobs=request_logprobs,
    )
    assert _request_without_file(provider.requests[0]) == {
        "model": "gpt-transcribe",
        "prompt": "Ελληνικό τηλεφωνικό αίτημα",
        "response_format": "json",
        "extra_body": {"languages": ["el"]},
    }
    assert provider.with_options_calls == ([{"max_retries": 0}] if request_logprobs else [])
    assert result.text == "Το αυτοκίνητο έχει επισκευαστεί."
    assert result.usage["totals"] == {"seconds": 10}
    if request_logprobs:
        assert len(result.response_evidence) == 1
        assert result.response_evidence[0].logprobs_available is False
        assert result.response_evidence[0].token_logprobs == ()
    else:
        assert result.response_evidence == ()


@pytest.mark.asyncio
async def test_v2_client_uses_exact_logprob_request_and_parses_sdk_evidence(
    tmp_path: Path,
) -> None:
    response = SimpleNamespace(
        text=" Γεια σας ",
        logprobs=[
            SimpleNamespace(token="Γεια", logprob=-0.2),
            SimpleNamespace(token=" σας", logprob=-0.4),
        ],
        usage=SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "type": "tokens",
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            }
        ),
    )
    provider = _ProviderClient(response)
    client = OpenAITranscriptionClient(
        _settings(tmp_path),
        client=provider,  # type: ignore[arg-type]
    )

    result = await client.transcribe_isolated(
        [_audio_chunk(tmp_path)],
        [],
        language="el",
        prompt_plan=_prompt_plan(),
        request_logprobs=True,
    )

    assert provider.with_options_calls == [{"max_retries": 0}]
    assert len(provider.requests) == 1
    assert _request_without_file(provider.requests[0]) == {
        "model": "gpt-4o-transcribe",
        "language": "el",
        "prompt": "Ελληνικό τηλεφωνικό αίτημα",
        "response_format": "json",
        "include": ["logprobs"],
        "temperature": 0.0,
    }
    assert result.text == "Γεια σας"
    assert result.usage == {
        "chunks": [
            {
                "type": "tokens",
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            }
        ],
        "totals": {
            "input_tokens": 4,
            "output_tokens": 2,
            "total_tokens": 6,
        },
    }
    assert result.response_evidence == (
        TranscriptionResponseEvidence(
            source_chunk_index=0,
            response_text=" Γεια σας ",
            token_logprobs=(
                TokenLogprob(token="Γεια", logprob=-0.2),
                TokenLogprob(token=" σας", logprob=-0.4),
            ),
            logprobs_available=True,
        ),
    )


@pytest.mark.asyncio
async def test_v2_client_parses_mapping_response_and_ignores_malformed_logprobs(
    tmp_path: Path,
) -> None:
    provider = _ProviderClient(
        {
            "text": "ναι",
            "logprobs": [
                {"token": "ναι", "logprob": -0.25},
                {"token": "nan", "logprob": float("nan")},
                {"token": "infinite", "logprob": float("inf")},
                {"token": "positive", "logprob": 0.1},
                {"token": "", "logprob": -0.1},
                {"token": "missing"},
            ],
            "usage": {"input_tokens": 2},
        }
    )

    result = await OpenAITranscriptionClient(
        _settings(tmp_path),
        client=provider,  # type: ignore[arg-type]
    ).transcribe_isolated(
        [_audio_chunk(tmp_path)],
        [],
        language="el",
        prompt_plan=_prompt_plan(),
        request_logprobs=True,
    )

    assert result.response_evidence[0].logprobs_available is True
    assert result.response_evidence[0].token_logprobs == (TokenLogprob(token="ναι", logprob=-0.25),)


@pytest.mark.asyncio
async def test_v2_client_reports_missing_logprobs_honestly(tmp_path: Path) -> None:
    provider = _ProviderClient({"text": "χωρίς στοιχεία", "usage": {}})

    result = await OpenAITranscriptionClient(
        _settings(tmp_path),
        client=provider,  # type: ignore[arg-type]
    ).transcribe_isolated(
        [_audio_chunk(tmp_path)],
        [],
        language="el",
        prompt_plan=_prompt_plan(),
        request_logprobs=True,
    )

    evidence = result.response_evidence[0]
    assert evidence.response_text == "χωρίς στοιχεία"
    assert evidence.token_logprobs == ()
    assert evidence.logprobs_available is False


@pytest.mark.asyncio
async def test_v2_unsupported_model_does_not_fabricate_logprob_request_or_evidence(
    tmp_path: Path,
) -> None:
    provider = _ProviderClient(
        {
            "text": "whisper fallback",
            "logprobs": [{"token": "must-ignore", "logprob": -0.1}],
            "usage": {},
        }
    )

    result = await OpenAITranscriptionClient(
        _settings(tmp_path, model="whisper-1"),
        client=provider,  # type: ignore[arg-type]
    ).transcribe_isolated(
        [_audio_chunk(tmp_path)],
        [],
        language="el",
        prompt_plan=_prompt_plan(),
        request_logprobs=True,
    )

    assert provider.with_options_calls == [{"max_retries": 0}]
    assert _request_without_file(provider.requests[0]) == {
        "model": "whisper-1",
        "language": "el",
        "prompt": "Ελληνικό τηλεφωνικό αίτημα",
        "response_format": "json",
    }
    assert result.response_evidence[0] == TranscriptionResponseEvidence(
        source_chunk_index=0,
        response_text="whisper fallback",
        token_logprobs=(),
        logprobs_available=False,
    )


@pytest.mark.asyncio
async def test_legacy_client_request_contract_is_unchanged(tmp_path: Path) -> None:
    provider = _ProviderClient(
        {
            "text": "legacy",
            "logprobs": [{"token": "ignored", "logprob": -0.1}],
            "usage": {},
        }
    )

    result = await OpenAITranscriptionClient(
        _settings(tmp_path),
        client=provider,  # type: ignore[arg-type]
    ).transcribe_isolated(
        [_audio_chunk(tmp_path)],
        ["Yeastar"],
        language="el",
    )

    assert provider.with_options_calls == []
    request = _request_without_file(provider.requests[0])
    assert request == {
        "model": "gpt-4o-transcribe",
        "language": "el",
        "prompt": "Greek business vocabulary and names: Yeastar",
        "response_format": "json",
    }
    assert "include" not in request
    assert "temperature" not in request
    assert result.response_evidence == ()


@dataclass(frozen=True, slots=True)
class _ScriptedAttempt:
    text: str
    logprobs: tuple[float, ...] | None
    input_tokens: int = 1
    output_tokens: int = 1
    duration_seconds: float = 0.1
    error: Exception | None = None
    logprobs_available: bool | None = None


@dataclass(frozen=True, slots=True)
class _CapturedAttemptRequest:
    chunk: AudioChunk
    language: str | None
    prompt_plan: PromptPlan | None
    request_logprobs: bool


class _ScriptedLogprobClient:
    def __init__(self, attempts: list[_ScriptedAttempt]) -> None:
        self._attempts = iter(attempts)
        self.requests: list[_CapturedAttemptRequest] = []

    async def transcribe_isolated(
        self,
        chunks: list[AudioChunk],
        _vocabulary: list[str],
        *,
        language: str | None = None,
        should_cancel: Callable[[], Awaitable[bool]] | None = None,
        prompt_plan: PromptPlan | None = None,
        request_logprobs: bool = False,
    ) -> TranscriptionResult:
        del should_cancel
        assert len(chunks) == 1
        chunk = chunks[0]
        self.requests.append(
            _CapturedAttemptRequest(
                chunk=chunk,
                language=language,
                prompt_plan=prompt_plan,
                request_logprobs=request_logprobs,
            )
        )
        attempt = next(self._attempts)
        if attempt.error is not None:
            raise attempt.error
        token_logprobs = (
            ()
            if attempt.logprobs is None
            else tuple(
                TokenLogprob(token=f"token-{position}", logprob=value)
                for position, value in enumerate(attempt.logprobs)
            )
        )
        usage = {
            "chunks": [
                {
                    "input_tokens": attempt.input_tokens,
                    "output_tokens": attempt.output_tokens,
                    "total_tokens": attempt.input_tokens + attempt.output_tokens,
                }
            ],
            "totals": {
                "input_tokens": attempt.input_tokens,
                "output_tokens": attempt.output_tokens,
                "total_tokens": attempt.input_tokens + attempt.output_tokens,
            },
        }
        return TranscriptionResult(
            model="gpt-4o-transcribe",
            language=language or "el",
            prompt_version=prompt_plan.version if prompt_plan is not None else None,
            processing_duration_seconds=attempt.duration_seconds,
            segments=(
                [
                    TranscribedSegment(
                        start_seconds=chunk.start_seconds,
                        end_seconds=chunk.end_seconds,
                        text=attempt.text,
                        speaker_label="Provider",
                        source_chunk_index=chunk.chunk_index,
                    )
                ]
                if attempt.text
                else []
            ),
            usage=usage,
            response_evidence=(
                TranscriptionResponseEvidence(
                    source_chunk_index=chunk.chunk_index,
                    response_text=attempt.text,
                    token_logprobs=token_logprobs,
                    logprobs_available=(
                        attempt.logprobs is not None
                        if attempt.logprobs_available is None
                        else attempt.logprobs_available
                    ),
                ),
            ),
        )


class _RecordingRetryProcessor:
    def __init__(self) -> None:
        self.prepared: list[SpeechChunk] = []
        self.cleaned: list[Path] = []

    async def prepare_retry(
        self,
        chunk: SpeechChunk,
        *,
        destination_dir: Path,
        register_temporary_file: Callable[[Path], None] | None = None,
    ) -> SpeechChunk:
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"normalized-{chunk.track_id}-{chunk.chunk_index}.wav"
        if register_temporary_file is not None:
            register_temporary_file(destination)
        destination.write_bytes(b"normalized")
        normalized = replace(
            chunk,
            path=destination,
            audio_variant=LIGHT_NORMALIZED_AUDIO_VARIANT,
        )
        self.prepared.append(normalized)
        return normalized

    def cleanup_retry(self, chunk: SpeechChunk) -> None:
        self.cleaned.append(chunk.path)
        chunk.path.unlink(missing_ok=True)


class _FailingRetryProcessor:
    async def prepare_retry(
        self,
        chunk: SpeechChunk,
        *,
        destination_dir: Path,
        register_temporary_file: Callable[[Path], None] | None = None,
    ) -> SpeechChunk:
        del chunk, destination_dir, register_temporary_file
        raise RuntimeError("normalization failed")

    def cleanup_retry(self, chunk: SpeechChunk) -> None:
        raise AssertionError(f"Nothing was prepared for cleanup: {chunk.path}")


def _operator_plan(tmp_path: Path) -> AudioPlan:
    return AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=tmp_path / "source.wav",
                channel_index=0,
                operator_id="operator-1",
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Operator One",
                speaker_source="stereo_channel",
            ),
        ),
        operator_channel=0,
        stereo_separated=True,
        attribution_status="confirmed_by_pbx",
        reason="test-operator-channel",
    )


def _dual_plan(tmp_path: Path) -> AudioPlan:
    return AudioPlan(
        mode="dual_channel",
        tracks=tuple(
            AudioTrack(
                track_id=f"channel-{channel}",
                source_path=tmp_path / "source.wav",
                channel_index=channel,
                attribution_status="channel_unknown",
                audio_variant=f"topology-channel-{channel}",
                speaker_label=f"Channel {'A' if channel == 0 else 'B'}",
                speaker_source="stereo_channel",
            )
            for channel in (0, 1)
        ),
        stereo_separated=True,
        attribution_status="channel_unknown",
        reason="test-dual-channel",
    )


def test_runtime_identity_tracks_real_standard_model_logprob_support(
    tmp_path: Path,
) -> None:
    plan = _operator_plan(tmp_path)
    common = {
        "plan": plan,
        "max_upload_bytes": 25_000_000,
        "prompt_template_version": "greek-context-v1",
    }

    supported = _pipeline_v2_runtime_config_hash(
        **common,
        standard_model="gpt-4o-transcribe",
    )
    unsupported = _pipeline_v2_runtime_config_hash(
        **common,
        standard_model="whisper-1",
    )
    assert supported != unsupported

    mono_plan = replace(
        plan,
        mode="mono_diarization",
        tracks=(
            replace(
                plan.tracks[0],
                track_id="mono",
                channel_index=None,
                operator_id=None,
                diarized=True,
            ),
        ),
    )
    mono_common = {
        "plan": mono_plan,
        "max_upload_bytes": 25_000_000,
        "prompt_template_version": "greek-context-v1",
    }
    assert _pipeline_v2_runtime_config_hash(
        **mono_common,
        standard_model="gpt-4o-transcribe",
        diarization_model="gpt-4o-transcribe-diarize",
    ) != _pipeline_v2_runtime_config_hash(
        **mono_common,
        standard_model="whisper-1",
        diarization_model="gpt-4o-transcribe-diarize",
    )
    assert _pipeline_v2_runtime_config_hash(
        **mono_common,
        standard_model="gpt-4o-transcribe",
        diarization_model="gpt-4o-transcribe-diarize",
    ) != _pipeline_v2_runtime_config_hash(
        **mono_common,
        standard_model="gpt-4o-transcribe",
        diarization_model="another-diarization-model",
    )


def _speech_chunk(
    tmp_path: Path,
    *,
    track_id: str = "operator-channel",
    chunk_index: int = 0,
) -> SpeechChunk:
    path = tmp_path / f"{track_id}-{chunk_index}.wav"
    path.write_bytes(b"raw")
    return SpeechChunk(
        track_id=track_id,
        chunk_index=chunk_index,
        path=path,
        start_seconds=float(chunk_index * 10),
        end_seconds=float((chunk_index + 1) * 10),
        audio_variant=f"{track_id}-segmented-pcm16",
    )


def _v2_context(
    tmp_path: Path,
    plan: AudioPlan,
    *,
    registered: list[Path] | None = None,
) -> TranscriptionContext:
    return TranscriptionContext(
        diarized=False,
        language="el",
        vocabulary=(),
        temporary_directory=tmp_path / "temporary",
        v2_prompt_manifest=V2GreekPromptBuilder().build_manifest(plan, ()),
        register_temporary_file=(registered.append if registered is not None else None),
    )


async def _transcribe_operator_track(
    tmp_path: Path,
    attempts: list[_ScriptedAttempt],
    *,
    chunks: tuple[SpeechChunk, ...] | None = None,
    cancellation_check: Callable[[], Awaitable[bool]] | None = None,
) -> tuple[
    Any,
    _ScriptedLogprobClient,
    _RecordingRetryProcessor,
    TranscriptionContext,
]:
    plan = _operator_plan(tmp_path)
    context = _v2_context(tmp_path, plan)
    client = _ScriptedLogprobClient(attempts)
    retry_processor = _RecordingRetryProcessor()
    orchestrator = TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        retry_quality_processor=retry_processor,  # type: ignore[arg-type]
    )
    result = await orchestrator._transcribe_v2_track(
        client,  # type: ignore[arg-type]
        plan.tracks[0],
        chunks or (_speech_chunk(tmp_path),),
        context,
        cancellation_check,
    )
    return result, client, retry_processor, context


@pytest.mark.asyncio
async def test_v2_high_confidence_chunk_makes_exactly_one_raw_attempt(
    tmp_path: Path,
) -> None:
    result, client, retry_processor, _ = await _transcribe_operator_track(
        tmp_path,
        [_ScriptedAttempt("raw accepted", (-0.1, -0.2), input_tokens=3)],
    )

    assert len(client.requests) == 1
    assert client.requests[0].language == "el"
    assert client.requests[0].request_logprobs is True
    assert client.requests[0].chunk.path.name == "operator-channel-0.wav"
    assert retry_processor.prepared == []
    assert result.hypotheses[0].text == "raw accepted"
    assert result.hypotheses[0].audio_variant == RAW_LOSSLESS_AUDIO_VARIANT
    assert result.hypotheses[0].quality_flags == ()
    assert len(result.attempts) == 1
    assert result.attempts[0].selected is True
    assert result.attempts[0].audio_variant == RAW_LOSSLESS_AUDIO_VARIANT


@pytest.mark.asyncio
async def test_v2_missing_logprobs_is_flagged_without_retry(tmp_path: Path) -> None:
    result, client, retry_processor, _ = await _transcribe_operator_track(
        tmp_path,
        [_ScriptedAttempt("raw without metrics", None)],
    )

    assert len(client.requests) == 1
    assert retry_processor.prepared == []
    assert result.hypotheses[0].mean_logprob is None
    assert result.hypotheses[0].low_logprob_ratio is None
    assert result.hypotheses[0].quality_flags == (QUALITY_FLAG_LOGPROBS_UNAVAILABLE,)
    assert result.attempts[0].mean_logprob is None
    assert result.attempts[0].selected is True


@pytest.mark.asyncio
async def test_unavailable_evidence_cannot_drive_confidence_or_retry(
    tmp_path: Path,
) -> None:
    result, client, retry_processor, _ = await _transcribe_operator_track(
        tmp_path,
        [
            _ScriptedAttempt(
                "unavailable despite payload",
                (-2.0,),
                logprobs_available=False,
            )
        ],
    )

    assert len(client.requests) == 1
    assert retry_processor.prepared == []
    assert result.attempts[0].mean_logprob is None
    assert result.hypotheses[0].quality_flags == (QUALITY_FLAG_LOGPROBS_UNAVAILABLE,)


@pytest.mark.asyncio
async def test_configured_single_attempt_cap_prevents_normalized_retry(
    tmp_path: Path,
) -> None:
    plan = _operator_plan(tmp_path)
    client = _ScriptedLogprobClient(
        [
            _ScriptedAttempt("weak but capped", (-2.0,)),
            _ScriptedAttempt("forbidden retry", (-0.1,)),
        ]
    )
    retry_processor = _RecordingRetryProcessor()
    result = await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        confidence_policy=ConfidencePolicy(max_attempts_per_chunk=1),
        retry_quality_processor=retry_processor,  # type: ignore[arg-type]
    )._transcribe_v2_track(
        client,  # type: ignore[arg-type]
        plan.tracks[0],
        (_speech_chunk(tmp_path),),
        _v2_context(tmp_path, plan),
        None,
    )

    assert len(client.requests) == 1
    assert retry_processor.prepared == []
    assert len(result.attempts) == 1
    assert result.attempts[0].selected is True
    assert result.hypotheses[0].quality_flags == (QUALITY_FLAG_LOW_CONFIDENCE,)


@pytest.mark.asyncio
async def test_low_raw_retries_once_selects_stronger_normalized_and_counts_both_costs(
    tmp_path: Path,
) -> None:
    registered: list[Path] = []
    plan = _operator_plan(tmp_path)
    context = _v2_context(tmp_path, plan, registered=registered)
    client = _ScriptedLogprobClient(
        [
            _ScriptedAttempt(
                "raw must not survive",
                (-1.2, -1.1),
                input_tokens=3,
                output_tokens=2,
                duration_seconds=0.2,
            ),
            _ScriptedAttempt(
                "normalized selected",
                (-0.2, -0.3),
                input_tokens=5,
                output_tokens=4,
                duration_seconds=0.4,
            ),
        ]
    )
    retry_processor = _RecordingRetryProcessor()
    result = await TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        retry_quality_processor=retry_processor,  # type: ignore[arg-type]
    )._transcribe_v2_track(
        client,  # type: ignore[arg-type]
        plan.tracks[0],
        (_speech_chunk(tmp_path),),
        context,
        None,
    )

    assert len(client.requests) == 2
    assert [attempt.selected for attempt in result.attempts] == [False, True]
    assert [attempt.audio_variant for attempt in result.attempts] == [
        RAW_LOSSLESS_AUDIO_VARIANT,
        LIGHT_NORMALIZED_AUDIO_VARIANT,
    ]
    assert [attempt.response_text for attempt in result.attempts] == [
        "raw must not survive",
        "normalized selected",
    ]
    assert all(attempt.prompt_hash for attempt in result.attempts)
    assert result.attempts[0].prompt_hash == result.attempts[1].prompt_hash
    assert all(attempt.completed_at is not None for attempt in result.attempts)
    assert result.attempts[0].mean_logprob == pytest.approx(-1.15)
    assert result.attempts[1].mean_logprob == pytest.approx(-0.25)
    assert result.hypotheses[0].text == "normalized selected"
    assert "raw must not survive" not in result.hypotheses[0].text
    assert result.hypotheses[0].audio_variant == LIGHT_NORMALIZED_AUDIO_VARIANT
    assert result.hypotheses[0].quality_flags == (QUALITY_FLAG_NORMALIZED_RETRY_USED,)
    assert result.processing_duration_seconds == pytest.approx(0.6)
    assert result.usage == {
        "chunks": [
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
        ],
        "totals": {
            "input_tokens": 8,
            "output_tokens": 6,
            "total_tokens": 14,
        },
    }
    assert result.attempts[0].api_usage == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
    }
    assert result.attempts[1].api_usage == {
        "input_tokens": 5,
        "output_tokens": 4,
        "total_tokens": 9,
    }
    assert registered == [retry_processor.prepared[0].path]
    assert retry_processor.cleaned == [retry_processor.prepared[0].path]
    assert retry_processor.prepared[0].path.exists() is False


@pytest.mark.asyncio
async def test_effective_logprob_tie_prefers_raw_even_if_normalized_is_acceptable(
    tmp_path: Path,
) -> None:
    result, client, _, _ = await _transcribe_operator_track(
        tmp_path,
        [
            _ScriptedAttempt("raw tie winner", (-0.76,)),
            _ScriptedAttempt("normalized close", (-0.72,)),
        ],
    )

    assert len(client.requests) == 2
    assert [attempt.selected for attempt in result.attempts] == [True, False]
    assert result.hypotheses[0].text == "raw tie winner"
    assert result.hypotheses[0].audio_variant == RAW_LOSSLESS_AUDIO_VARIANT
    assert result.hypotheses[0].quality_flags == (
        QUALITY_FLAG_NORMALIZED_RETRY_USED,
        QUALITY_FLAG_LOW_CONFIDENCE,
    )


@pytest.mark.asyncio
async def test_both_low_attempts_stop_at_two_and_recommend_human_review(
    tmp_path: Path,
) -> None:
    result, client, _, _ = await _transcribe_operator_track(
        tmp_path,
        [
            _ScriptedAttempt("weak raw", (-1.4,)),
            _ScriptedAttempt("less weak normalized", (-0.9,)),
            _ScriptedAttempt("forbidden third", (-0.1,)),
        ],
    )

    assert len(client.requests) == 2
    assert len(result.attempts) == 2
    assert sum(attempt.selected for attempt in result.attempts) == 1
    assert result.hypotheses[0].text == "less weak normalized"
    assert result.hypotheses[0].quality_flags == (
        QUALITY_FLAG_NORMALIZED_RETRY_USED,
        QUALITY_FLAG_LOW_CONFIDENCE,
        QUALITY_FLAG_BOTH_ATTEMPTS_LOW_CONFIDENCE,
        QUALITY_FLAG_HUMAN_REVIEW_RECOMMENDED,
    )


@pytest.mark.asyncio
async def test_cancellation_is_rechecked_after_low_raw_and_before_retry(
    tmp_path: Path,
) -> None:
    checks = 0

    async def cancelled() -> bool:
        nonlocal checks
        checks += 1
        return True

    client = _ScriptedLogprobClient(
        [
            _ScriptedAttempt("weak raw", (-1.2,)),
            _ScriptedAttempt("must never upload", (-0.1,)),
        ]
    )
    retry_processor = _RecordingRetryProcessor()
    plan = _operator_plan(tmp_path)
    orchestrator = TranscriptionOrchestrator(
        settings=_settings(tmp_path),
        retry_quality_processor=retry_processor,  # type: ignore[arg-type]
    )

    with pytest.raises(
        PartialTranscriptionCancelledError,
        match="cancelled",
    ) as raised:
        await orchestrator._transcribe_v2_track(
            client,  # type: ignore[arg-type]
            plan.tracks[0],
            (_speech_chunk(tmp_path),),
            _v2_context(tmp_path, plan),
            cancelled,
        )

    assert checks == 1
    assert len(client.requests) == 1
    assert retry_processor.prepared == []
    assert len(raised.value.attempts) == 1
    assert raised.value.attempts[0].response_text == "weak raw"
    assert raised.value.attempts[0].selected is True


@pytest.mark.asyncio
async def test_retry_reuses_exact_prompt_and_next_context_uses_selected_text_only(
    tmp_path: Path,
) -> None:
    chunks = (
        _speech_chunk(tmp_path, chunk_index=0),
        _speech_chunk(tmp_path, chunk_index=1),
    )
    result, client, _, _ = await _transcribe_operator_track(
        tmp_path,
        [
            _ScriptedAttempt("RAW_PRIVATE_CONTEXT", (-1.2,)),
            _ScriptedAttempt("SELECTED_CONTEXT", (-0.2,)),
            _ScriptedAttempt("SECOND_CHUNK", (-0.1,)),
        ],
        chunks=chunks,
    )

    assert len(client.requests) == 3
    raw_prompt = client.requests[0].prompt_plan
    retry_prompt = client.requests[1].prompt_plan
    next_prompt = client.requests[2].prompt_plan
    assert raw_prompt is not None
    assert retry_prompt is not None
    assert next_prompt is not None
    assert raw_prompt == retry_prompt
    assert raw_prompt.prompt_hash == retry_prompt.prompt_hash
    assert "SELECTED_CONTEXT" in next_prompt.text
    assert "RAW_PRIVATE_CONTEXT" not in next_prompt.text
    assert [hypothesis.text for hypothesis in result.hypotheses] == [
        "SELECTED_CONTEXT",
        "SECOND_CHUNK",
    ]
    assert [(attempt.chunk_index, attempt.selected) for attempt in result.attempts] == [
        (0, False),
        (0, True),
        (1, True),
    ]


@pytest.mark.asyncio
async def test_normalized_retry_file_is_cleaned_when_second_request_fails(
    tmp_path: Path,
) -> None:
    client = _ScriptedLogprobClient(
        [
            _ScriptedAttempt("weak raw", (-1.2,)),
            _ScriptedAttempt("", None, error=RuntimeError("provider failed")),
        ]
    )
    retry_processor = _RecordingRetryProcessor()
    plan = _operator_plan(tmp_path)

    with pytest.raises(PartialTranscriptionError, match="provider failed") as raised:
        await TranscriptionOrchestrator(
            settings=_settings(tmp_path),
            retry_quality_processor=retry_processor,  # type: ignore[arg-type]
        )._transcribe_v2_track(
            client,  # type: ignore[arg-type]
            plan.tracks[0],
            (_speech_chunk(tmp_path),),
            _v2_context(tmp_path, plan),
            None,
        )

    assert len(client.requests) == 2
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert len(raised.value.attempts) == 1
    assert raised.value.attempts[0].response_text == "weak raw"
    assert raised.value.attempts[0].selected is True
    assert len(retry_processor.prepared) == 1
    assert retry_processor.cleaned == [retry_processor.prepared[0].path]
    assert retry_processor.prepared[0].path.exists() is False


@pytest.mark.asyncio
async def test_later_raw_failure_carries_all_prior_completed_chunk_attempts(
    tmp_path: Path,
) -> None:
    chunks = (
        _speech_chunk(tmp_path, chunk_index=0),
        _speech_chunk(tmp_path, chunk_index=1),
    )
    plan = _operator_plan(tmp_path)
    client = _ScriptedLogprobClient(
        [
            _ScriptedAttempt("first completed chunk", (-0.1,)),
            _ScriptedAttempt("", None, error=RuntimeError("later raw failed")),
        ]
    )

    with pytest.raises(PartialTranscriptionError, match="later raw failed") as raised:
        await TranscriptionOrchestrator(
            settings=_settings(tmp_path),
        )._transcribe_v2_track(
            client,  # type: ignore[arg-type]
            plan.tracks[0],
            chunks,
            _v2_context(tmp_path, plan),
            None,
        )

    assert len(client.requests) == 2
    assert [
        (attempt.chunk_index, attempt.response_text, attempt.selected)
        for attempt in raised.value.attempts
    ] == [(0, "first completed chunk", True)]


@pytest.mark.asyncio
async def test_normalization_failure_carries_completed_raw_attempt(
    tmp_path: Path,
) -> None:
    plan = _operator_plan(tmp_path)
    client = _ScriptedLogprobClient([_ScriptedAttempt("weak raw", (-1.2,))])

    with pytest.raises(PartialTranscriptionError, match="normalization failed") as raised:
        await TranscriptionOrchestrator(
            settings=_settings(tmp_path),
            retry_quality_processor=_FailingRetryProcessor(),  # type: ignore[arg-type]
        )._transcribe_v2_track(
            client,  # type: ignore[arg-type]
            plan.tracks[0],
            (_speech_chunk(tmp_path),),
            _v2_context(tmp_path, plan),
            None,
        )

    assert len(client.requests) == 1
    assert len(raised.value.attempts) == 1
    assert raised.value.attempts[0].response_text == "weak raw"
    assert raised.value.attempts[0].selected is True


@pytest.mark.asyncio
async def test_dual_track_context_and_retry_state_remain_isolated(
    tmp_path: Path,
) -> None:
    plan = _dual_plan(tmp_path)
    context = _v2_context(tmp_path, plan)
    client = _ScriptedLogprobClient(
        [
            _ScriptedAttempt("CHANNEL_A_PRIVATE", (-0.1,)),
            _ScriptedAttempt("CHANNEL_A_SECOND", (-0.1,)),
            _ScriptedAttempt("CHANNEL_B_FIRST", (-0.1,)),
            _ScriptedAttempt("CHANNEL_B_SECOND", (-0.1,)),
        ]
    )
    orchestrator = TranscriptionOrchestrator(settings=_settings(tmp_path))

    for track in plan.tracks:
        await orchestrator._transcribe_v2_track(
            client,  # type: ignore[arg-type]
            track,
            (
                _speech_chunk(tmp_path, track_id=track.track_id, chunk_index=0),
                _speech_chunk(tmp_path, track_id=track.track_id, chunk_index=1),
            ),
            context,
            None,
        )

    channel_b_first = client.requests[2].prompt_plan
    channel_b_second = client.requests[3].prompt_plan
    assert channel_b_first is not None
    assert channel_b_second is not None
    assert channel_b_first.track_id == "channel-1"
    assert channel_b_first.previous_context_characters == 0
    assert "CHANNEL_A_PRIVATE" not in channel_b_first.text
    assert "CHANNEL_A_PRIVATE" not in channel_b_second.text
    assert "CHANNEL_B_FIRST" in channel_b_second.text


class _CapturingSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.flush_count = 0

    def add(self, value: object) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        self.flush_count += 1


def _attempt_evidence(
    *,
    audio_variant: str,
    response_text: str,
    mean_logprob: float,
    low_logprob_ratio: float,
    selected: bool,
    completed_at: datetime,
    input_tokens: int,
) -> TranscriptionAttemptEvidence:
    return TranscriptionAttemptEvidence(
        track_id="operator-channel",
        chunk_index=0,
        start_seconds=1.23456,
        end_seconds=9.87654,
        model="gpt-4o-transcribe",
        audio_variant=audio_variant,
        prompt_hash="b" * 64,
        response_text=response_text,
        mean_logprob=mean_logprob,
        low_logprob_ratio=low_logprob_ratio,
        selected=selected,
        api_usage={
            "input_tokens": input_tokens,
            "nested": {"safe": [1, True, None]},
        },
        completed_at=completed_at,
    )


def _persistence_result(
    attempts: tuple[TranscriptionAttemptEvidence, ...],
) -> OrchestratedTranscriptionResult:
    return OrchestratedTranscriptionResult(
        mode="operator_channel",
        model="gpt-4o-transcribe",
        language="el",
        prompt_version="prompt-v2",
        processing_duration_seconds=0.6,
        segments=(),
        tracks=(),
        usage={"chunks": [], "totals": {"input_tokens": 8}},
        attribution_status="confirmed_by_pbx",
        attempts=attempts,
    )


@pytest.mark.asyncio
async def test_worker_persists_both_attempts_with_exactly_one_selected(
    tmp_path: Path,
) -> None:
    del tmp_path
    raw_completed = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    normalized_completed = datetime(2026, 1, 2, 3, 5, tzinfo=UTC)
    raw = _attempt_evidence(
        audio_variant=RAW_LOSSLESS_AUDIO_VARIANT,
        response_text="raw response",
        mean_logprob=-1.2,
        low_logprob_ratio=0.5,
        selected=False,
        completed_at=raw_completed,
        input_tokens=3,
    )
    normalized = _attempt_evidence(
        audio_variant=LIGHT_NORMALIZED_AUDIO_VARIANT,
        response_text="selected response",
        mean_logprob=-0.2,
        low_logprob_ratio=0.0,
        selected=True,
        completed_at=normalized_completed,
        input_tokens=5,
    )
    session = _CapturingSession()
    transcript = SimpleNamespace(id=uuid4())

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        _persistence_result((raw, normalized)),
        SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
        None,
        None,
        10.0,
    )

    persisted = [value for value in session.added if isinstance(value, TranscriptionAttempt)]
    assert len(persisted) == 2
    assert [attempt.selected for attempt in persisted] == [False, True]
    assert [attempt.audio_variant for attempt in persisted] == [
        RAW_LOSSLESS_AUDIO_VARIANT,
        LIGHT_NORMALIZED_AUDIO_VARIANT,
    ]
    assert [attempt.response_text for attempt in persisted] == [
        "raw response",
        "selected response",
    ]
    assert [attempt.mean_logprob for attempt in persisted] == [
        Decimal("-1.2"),
        Decimal("-0.2"),
    ]
    assert [attempt.low_logprob_ratio for attempt in persisted] == [
        Decimal("0.5"),
        Decimal("0.0"),
    ]
    assert [attempt.start_seconds for attempt in persisted] == [
        Decimal("1.235"),
        Decimal("1.235"),
    ]
    assert [attempt.end_seconds for attempt in persisted] == [
        Decimal("9.877"),
        Decimal("9.877"),
    ]
    assert [attempt.completed_at for attempt in persisted] == [
        raw_completed,
        normalized_completed,
    ]
    assert persisted[0].api_usage == {
        "input_tokens": 3,
        "nested": {"safe": [1, True, None]},
    }
    assert persisted[1].api_usage == {
        "input_tokens": 5,
        "nested": {"safe": [1, True, None]},
    }
    assert session.flush_count == 1


@pytest.mark.asyncio
async def test_worker_persists_only_selected_attempt_text_as_final_segment() -> None:
    completed = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    raw = _attempt_evidence(
        audio_variant=RAW_LOSSLESS_AUDIO_VARIANT,
        response_text="unselected raw text",
        mean_logprob=-1.2,
        low_logprob_ratio=0.5,
        selected=False,
        completed_at=completed,
        input_tokens=3,
    )
    normalized = _attempt_evidence(
        audio_variant=LIGHT_NORMALIZED_AUDIO_VARIANT,
        response_text="selected normalized text",
        mean_logprob=-0.2,
        low_logprob_ratio=0.0,
        selected=True,
        completed_at=completed,
        input_tokens=5,
    )
    result = replace(
        _persistence_result((raw, normalized)),
        segments=(
            ChunkHypothesis(
                track_id="operator-channel",
                chunk_index=0,
                start_seconds=1.23456,
                end_seconds=9.87654,
                text="selected normalized text",
                speaker_label="Operator",
                speaker_source="stereo_channel",
                audio_variant=LIGHT_NORMALIZED_AUDIO_VARIANT,
            ),
        ),
    )
    session = _CapturingSession()

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
        result,
        SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
        None,
        None,
        10.0,
    )

    segments = [value for value in session.added if isinstance(value, TranscriptSegment)]
    attempts = [value for value in session.added if isinstance(value, TranscriptionAttempt)]
    assert [segment.original_text for segment in segments] == ["selected normalized text"]
    assert "unselected raw text" not in segments[0].original_text
    assert [attempt.response_text for attempt in attempts] == [
        "unselected raw text",
        "selected normalized text",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_values", [(False, False), (True, True)])
async def test_worker_rejects_invalid_selection_before_adding_attempts(
    selected_values: tuple[bool, bool],
) -> None:
    completed = datetime(2026, 1, 2, 3, 4, tzinfo=UTC)
    attempts = tuple(
        _attempt_evidence(
            audio_variant=(
                RAW_LOSSLESS_AUDIO_VARIANT if position == 0 else LIGHT_NORMALIZED_AUDIO_VARIANT
            ),
            response_text=f"attempt-{position}",
            mean_logprob=-1.0 + position * 0.1,
            low_logprob_ratio=0.5,
            selected=selected,
            completed_at=completed,
            input_tokens=position + 1,
        )
        for position, selected in enumerate(selected_values)
    )
    session = _CapturingSession()
    transcript = SimpleNamespace(id=uuid4())

    with pytest.raises(ValueError, match="select exactly one"):
        await _persist_transcription_result(
            session,  # type: ignore[arg-type]
            transcript,  # type: ignore[arg-type]
            _persistence_result(attempts),
            SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
            None,
            None,
            10.0,
        )

    assert session.added == []
    assert session.flush_count == 0
    assert not hasattr(transcript, "status")


@pytest.mark.asyncio
async def test_worker_requires_attempt_evidence_for_every_standard_v2_segment() -> None:
    result = replace(
        _persistence_result(()),
        segments=(
            ChunkHypothesis(
                track_id="operator-channel",
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=10.0,
                text="selected text",
                speaker_label="Operator",
            ),
        ),
    )
    session = _CapturingSession()
    transcript = SimpleNamespace(id=uuid4())

    with pytest.raises(ValueError, match="must have selected attempt evidence"):
        await _persist_transcription_result(
            session,  # type: ignore[arg-type]
            transcript,  # type: ignore[arg-type]
            result,
            SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
            None,
            None,
            10.0,
        )

    assert session.added == []
    assert session.flush_count == 0
    assert not hasattr(transcript, "status")


def test_light_normalization_profile_has_one_exact_versioned_filter_contract() -> None:
    assert LIGHT_NORMALIZATION_PROFILE.version == "ffmpeg-light-normalized-v1"
    assert LIGHT_NORMALIZATION_PROFILE.audio_variant == LIGHT_NORMALIZED_AUDIO_VARIANT
    assert (
        LIGHT_NORMALIZATION_PROFILE.filter_chain
        == "highpass=f=100,lowpass=f=3400,loudnorm=I=-23:LRA=7:TP=-2"
    )
    assert LIGHT_NORMALIZATION_PROFILE.identity() == {
        "audio_variant": LIGHT_NORMALIZED_AUDIO_VARIANT,
        "channels": 1,
        "codec": "pcm_s16le",
        "filter_chain": ("highpass=f=100,lowpass=f=3400,loudnorm=I=-23:LRA=7:TP=-2"),
        "sample_rate_hz": 16_000,
        "version": "ffmpeg-light-normalized-v1",
    }


@pytest.mark.asyncio
async def test_audio_filter_command_is_exact_lossless_pcm16_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")
    destination = tmp_path / "normalized.wav"
    processor = AudioProcessor(_settings(tmp_path))
    commands: list[tuple[str, ...]] = []

    async def fake_run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        del timeout
        commands.append(args)
        Path(args[-1]).write_bytes(b"complete-pcm16")
        return b"", b""

    monkeypatch.setattr(processor, "_run", fake_run)

    result = await processor.apply_lossless_audio_filter(
        source,
        destination,
        audio_filter=LIGHT_NORMALIZATION_PROFILE.filter_chain,
    )

    partial = processor.partial_wav_path(destination)
    assert result == destination
    assert destination.read_bytes() == b"complete-pcm16"
    assert partial.exists() is False
    assert commands == [
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-filter:a",
            LIGHT_NORMALIZATION_PROFILE.filter_chain,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(partial),
        )
    ]


@pytest.mark.asyncio
async def test_audio_filter_failure_removes_partial_and_destination_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "raw.wav"
    source.write_bytes(b"raw")
    destination = tmp_path / "normalized.wav"
    destination.write_bytes(b"stale")
    processor = AudioProcessor(_settings(tmp_path))
    partial = processor.partial_wav_path(destination)

    async def failing_run(*args: str, timeout: float = 180.0) -> tuple[bytes, bytes]:
        del timeout
        Path(args[-1]).write_bytes(b"partial")
        destination.write_bytes(b"unexpected-side-effect")
        raise RuntimeError("ffmpeg failed")

    monkeypatch.setattr(processor, "_run", failing_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        await processor.apply_lossless_audio_filter(
            source,
            destination,
            audio_filter=LIGHT_NORMALIZATION_PROFILE.filter_chain,
        )

    assert partial.exists() is False
    assert destination.exists() is False
    assert source.read_bytes() == b"raw"


class _CapturingFilterProcessor:
    def __init__(self) -> None:
        self.filters: list[tuple[Path, Path, str]] = []

    @staticmethod
    def partial_wav_path(destination: Path) -> Path:
        return destination.with_suffix(".part.wav")

    async def apply_lossless_audio_filter(
        self,
        source: Path,
        destination: Path,
        *,
        audio_filter: str,
    ) -> Path:
        self.filters.append((source, destination, audio_filter))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"normalized")
        return destination


@pytest.mark.asyncio
async def test_retry_quality_processor_registers_outputs_before_use_and_cleans_them(
    tmp_path: Path,
) -> None:
    source_chunk = _speech_chunk(tmp_path)
    processor = _CapturingFilterProcessor()
    quality = LightNormalizedRetryQualityProcessor(
        processor,  # type: ignore[arg-type]
    )
    registered: list[Path] = []

    normalized = await quality.prepare_retry(
        source_chunk,
        destination_dir=tmp_path / "retry",
        register_temporary_file=registered.append,
    )

    partial = processor.partial_wav_path(normalized.path)
    assert normalized.path != source_chunk.path
    assert normalized.audio_variant == LIGHT_NORMALIZED_AUDIO_VARIANT
    assert registered == [normalized.path, partial]
    assert processor.filters == [
        (
            source_chunk.path,
            normalized.path,
            LIGHT_NORMALIZATION_PROFILE.filter_chain,
        )
    ]
    assert normalized.path.exists() is True
    quality.cleanup_retry(normalized)
    assert normalized.path.exists() is False
    assert partial.exists() is False
    assert source_chunk.path.exists() is True
