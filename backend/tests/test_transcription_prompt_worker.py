from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.database.base import Base
from app.models import (
    ApplicationSetting,
    Call,
    Keyword,
    KeywordCategory,
    KeywordVariant,
    Operator,
    ProcessingJob,
    Recording,
    Transcript,
    TranscriptionAttempt,
    User,
)
from app.models.enums import (
    Direction,
    JobStatus,
    RecordingStatus,
    SpeakerAttributionStatus,
    TranscriptionMode,
    TranscriptStatus,
)
from app.services.keyword_matching import normalize_greek
from app.services.transcription.prompt import (
    GREEK_CALLCENTER_PROMPT_VERSION,
    PRIORITY_COMPANY,
    PRIORITY_CURRENT_CALL,
    PRIORITY_CURRENT_PARTY,
    PRIORITY_QUEUE,
    PRIORITY_SELECTED_KEYWORD,
    PRIORITY_SELECTED_OPERATOR,
    TRACK_ROLE_CALLEE,
    TRACK_ROLE_CALLER,
    TRACK_ROLE_OPERATOR,
    VOCABULARY_SOURCE_COMPANY,
    VOCABULARY_SOURCE_CURRENT_CALL,
    VOCABULARY_SOURCE_CURRENT_PARTY,
    VOCABULARY_SOURCE_QUEUE,
    VOCABULARY_SOURCE_SELECTED_KEYWORD,
    VOCABULARY_SOURCE_SELECTED_OPERATOR,
    RankedVocabularyTerm,
    V2GreekPromptBuilder,
)
from app.services.transcription.types import (
    AudioPlan,
    AudioTrack,
    OrchestratedTranscriptionResult,
    TranscriptionAttemptEvidence,
)
from app.workers.pipeline import (
    PIPELINE_V2_PREPROCESSING_PROFILE,
    PIPELINE_V2_VERSION,
    _completed_duplicate_snapshot,
    _persist_transcription_result,
    _pipeline_v2_runtime_config_hash,
    _v2_ranked_vocabulary,
    transcript_idempotency_key,
)


@dataclass(frozen=True)
class _VocabularyFixture:
    call: Call
    job: ProcessingJob
    operator: Operator
    unrelated_operator: Operator
    selected_keyword: Keyword
    unselected_keyword: Keyword


async def _new_database() -> tuple[Any, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_vocabulary_fixture(session: AsyncSession) -> _VocabularyFixture:
    now = datetime.now(UTC)
    user = User(
        email=f"prompt-worker-{uuid4()}@example.test",
        password_hash="not-used",
    )
    operator = Operator(
        yeastar_extension_id=f"selected-{uuid4()}",
        extension_number="101",
        display_name="Selected Operator",
        provider_active=True,
        enabled=True,
        last_synced_at=now,
    )
    unrelated_operator = Operator(
        yeastar_extension_id=f"unrelated-{uuid4()}",
        extension_number="102",
        display_name="Unrelated Operator",
        provider_active=True,
        enabled=True,
        last_synced_at=now,
    )
    call = Call(
        yeastar_uid=f"prompt-worker-call-{uuid4()}",
        started_at=now,
        caller_number="+30 210 111 1111",
        caller_name="Current Caller",
        callee_number="+30 210 222 2222",
        callee_name="Current Callee",
        direction=Direction.INBOUND,
        call_status="answered",
        duration_seconds=90,
        queue_name="Service Queue",
        has_recording=True,
        provider_payload={},
    )
    selected_category = KeywordCategory(
        name=f"Selected vocabulary {uuid4()}",
        active=True,
    )
    unselected_category = KeywordCategory(
        name=f"Unselected vocabulary {uuid4()}",
        active=True,
    )
    session.add_all(
        [
            user,
            operator,
            unrelated_operator,
            call,
            selected_category,
            unselected_category,
            ApplicationSetting(
                key="company_vocabulary",
                value="Gratsias Motors, Hybrid Service",
            ),
        ]
    )
    await session.flush()

    selected_keyword = Keyword(
        category_id=selected_category.id,
        canonical_phrase="Selected warranty",
        normalized_phrase=normalize_greek("Selected warranty"),
        active=True,
        variants=[
            KeywordVariant(
                phrase="Selected guarantee",
                normalized_phrase=normalize_greek("Selected guarantee"),
            )
        ],
    )
    unselected_keyword = Keyword(
        category_id=unselected_category.id,
        canonical_phrase="Unselected finance",
        normalized_phrase=normalize_greek("Unselected finance"),
        active=True,
        variants=[
            KeywordVariant(
                phrase="Unselected loan",
                normalized_phrase=normalize_greek("Unselected loan"),
            )
        ],
    )
    job = ProcessingJob(
        idempotency_key=f"prompt-worker-job-{uuid4()}",
        requested_by_id=user.id,
        status=JobStatus.COMPLETED,
        date_from=now - timedelta(hours=1),
        date_to=now,
        selected_operator_ids=[str(operator.id)],
        selected_category_ids=[str(selected_category.id)],
        request_filters={},
    )
    session.add_all([selected_keyword, unselected_keyword, job])
    await session.commit()
    return _VocabularyFixture(
        call=call,
        job=job,
        operator=operator,
        unrelated_operator=unrelated_operator,
        selected_keyword=selected_keyword,
        unselected_keyword=unselected_keyword,
    )


def _operator_plan(
    operator_id: UUID | str = "operator-id",
    *,
    source_path: Path = Path("recording.wav"),
) -> AudioPlan:
    return AudioPlan(
        mode="operator_channel",
        tracks=(
            AudioTrack(
                track_id="operator-channel",
                source_path=source_path,
                channel_index=0,
                operator_id=str(operator_id),
                attribution_status="confirmed_by_pbx",
                audio_variant="topology-operator-channel",
                speaker_label="Selected Operator",
                speaker_source="stereo_channel",
            ),
        ),
        operator_channel=0,
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="confirmed_by_pbx",
        reason="confirmed-separated-stereo-safe-operator",
    )


def _caller_callee_plan() -> AudioPlan:
    return AudioPlan(
        mode="dual_channel",
        tracks=(
            AudioTrack(
                track_id="channel-0",
                source_path=Path("recording.wav"),
                channel_index=0,
                attribution_status="caller_callee_only",
                audio_variant="topology-channel-0",
                speaker_label="Caller",
                speaker_source="stereo_channel",
            ),
            AudioTrack(
                track_id="channel-1",
                source_path=Path("recording.wav"),
                channel_index=1,
                attribution_status="caller_callee_only",
                audio_variant="topology-channel-1",
                speaker_label="Callee",
                speaker_source="stereo_channel",
            ),
        ),
        stereo_separated=True,
        caller_channel=0,
        callee_channel=1,
        attribution_status="caller_callee_only",
        reason="confirmed-separated-stereo-caller-callee",
    )


def _values(terms: tuple[RankedVocabularyTerm, ...]) -> set[str]:
    return {term.value for term in terms}


@pytest.mark.asyncio
async def test_v2_worker_collects_ranked_current_call_vocabulary_and_scopes_roles() -> None:
    engine, sessions = await _new_database()
    try:
        async with sessions() as session:
            fixture = await _seed_vocabulary_fixture(session)
            terms = await _v2_ranked_vocabulary(
                session,
                job=fixture.job,
                call=fixture.call,
                operator=fixture.operator,
                settings=Settings(APP_ENV="test"),
            )

            by_value = {term.value: term for term in terms}
            assert by_value["Selected Operator"].priority == PRIORITY_SELECTED_OPERATOR
            assert by_value["Selected Operator"].source == (VOCABULARY_SOURCE_SELECTED_OPERATOR)
            assert by_value["Selected Operator"].roles == (TRACK_ROLE_OPERATOR,)
            assert fixture.unrelated_operator.display_name not in by_value

            assert by_value["Current Caller"].priority == PRIORITY_CURRENT_PARTY
            assert by_value["Current Caller"].source == VOCABULARY_SOURCE_CURRENT_PARTY
            assert by_value["Current Caller"].roles == (TRACK_ROLE_CALLER,)
            assert by_value["Current Callee"].roles == (TRACK_ROLE_CALLEE,)
            assert by_value["+30 210 111 1111"].priority == PRIORITY_CURRENT_CALL
            assert by_value["+30 210 222 2222"].source == (VOCABULARY_SOURCE_CURRENT_CALL)

            assert by_value["Selected warranty"].priority == PRIORITY_SELECTED_KEYWORD
            assert by_value["Selected guarantee"].source == (VOCABULARY_SOURCE_SELECTED_KEYWORD)
            assert fixture.unselected_keyword.canonical_phrase not in by_value
            assert "Unselected loan" not in by_value
            assert by_value["Gratsias Motors"].priority == PRIORITY_COMPANY
            assert by_value["Hybrid Service"].source == VOCABULARY_SOURCE_COMPANY
            assert by_value["Service Queue"].priority == PRIORITY_QUEUE
            assert by_value["Service Queue"].source == VOCABULARY_SOURCE_QUEUE

            builder = V2GreekPromptBuilder()
            operator_manifest = builder.build_manifest(
                _operator_plan(fixture.operator.id),
                terms,
            )
            dual_manifest = builder.build_manifest(_caller_callee_plan(), terms)

            operator_values = _values(operator_manifest.track("operator-channel").terms)
            caller_values = _values(dual_manifest.track("channel-0").terms)
            callee_values = _values(dual_manifest.track("channel-1").terms)
            assert "Selected Operator" in operator_values
            assert "Current Caller" not in operator_values
            assert "Current Callee" not in operator_values
            assert "Current Caller" in caller_values
            assert "Current Callee" not in caller_values
            assert "Selected Operator" not in caller_values
            assert "Current Callee" in callee_values
            assert "Current Caller" not in callee_values
            assert "Selected Operator" not in callee_values
            assert fixture.unrelated_operator.display_name not in (
                operator_values | caller_values | callee_values
            )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_v2_worker_with_no_selected_category_excludes_every_saved_keyword() -> None:
    engine, sessions = await _new_database()
    try:
        async with sessions() as session:
            fixture = await _seed_vocabulary_fixture(session)
            fixture.job.selected_category_ids = []
            await session.flush()

            terms = await _v2_ranked_vocabulary(
                session,
                job=fixture.job,
                call=fixture.call,
                operator=fixture.operator,
                settings=Settings(APP_ENV="test"),
            )

            values = _values(terms)
            assert fixture.selected_keyword.canonical_phrase not in values
            assert "Selected guarantee" not in values
            assert fixture.unselected_keyword.canonical_phrase not in values
            assert "Unselected loan" not in values
            assert "Selected Operator" in values
            assert "Gratsias Motors" in values
            assert "Service Queue" in values
    finally:
        await engine.dispose()


def test_v2_prompt_metadata_changes_config_and_transcript_idempotency() -> None:
    builder = V2GreekPromptBuilder()
    plan = _operator_plan()
    base_manifest = builder.build_manifest(
        plan,
        (
            RankedVocabularyTerm(
                value="Gratsias Motors",
                priority=PRIORITY_COMPANY,
                source=VOCABULARY_SOURCE_COMPANY,
            ),
        ),
    )
    changed_manifest = builder.build_manifest(
        plan,
        (
            RankedVocabularyTerm(
                value="Gratsias Motors",
                priority=PRIORITY_COMPANY,
                source=VOCABULARY_SOURCE_COMPANY,
            ),
            RankedVocabularyTerm(
                value="Selected warranty",
                priority=PRIORITY_SELECTED_KEYWORD,
                source=VOCABULARY_SOURCE_SELECTED_KEYWORD,
            ),
        ),
    )
    assert changed_manifest.vocabulary_hash != base_manifest.vocabulary_hash
    assert changed_manifest.prompt_identity != base_manifest.prompt_identity

    config_base = {
        "plan": plan,
        "max_upload_bytes": 24 * 1024 * 1024,
        "prompt_template_version": base_manifest.template_version,
        "segmentation_identity": {"strategy": "prompt-worker-test"},
        "prompt_identity": base_manifest.prompt_identity,
        "vocabulary_hash": base_manifest.vocabulary_hash,
    }
    base_config_hash = _pipeline_v2_runtime_config_hash(**config_base)
    template_config_hash = _pipeline_v2_runtime_config_hash(
        **{
            **config_base,
            "prompt_template_version": "greek-callcenter-v3",
        }
    )
    vocabulary_config_hash = _pipeline_v2_runtime_config_hash(
        **{
            **config_base,
            "vocabulary_hash": changed_manifest.vocabulary_hash,
        }
    )
    aggregate_config_hash = _pipeline_v2_runtime_config_hash(
        **{
            **config_base,
            "prompt_identity": changed_manifest.prompt_identity,
        }
    )
    assert (
        len(
            {
                base_config_hash,
                template_config_hash,
                vocabulary_config_hash,
                aggregate_config_hash,
            }
        )
        == 4
    )

    idempotency_base = {
        "recording_id": uuid4(),
        "operator_id": uuid4(),
        "model": "gpt-4o-transcribe",
        "checksum": "a" * 64,
        "diarized": False,
        "language": "el",
        "prompt_version": base_manifest.prompt_identity,
        "pipeline_version": PIPELINE_V2_VERSION,
        "pipeline_config_hash": base_config_hash,
        "transcription_mode": TranscriptionMode.OPERATOR_CHANNEL,
        "prompt_template_version": base_manifest.template_version,
        "vocabulary_hash": base_manifest.vocabulary_hash,
    }
    base_key = transcript_idempotency_key(**idempotency_base)
    template_key = transcript_idempotency_key(
        **{
            **idempotency_base,
            "prompt_template_version": "greek-callcenter-v3",
            "pipeline_config_hash": template_config_hash,
        }
    )
    vocabulary_key = transcript_idempotency_key(
        **{
            **idempotency_base,
            "prompt_version": changed_manifest.prompt_identity,
            "vocabulary_hash": changed_manifest.vocabulary_hash,
            "pipeline_config_hash": vocabulary_config_hash,
        }
    )
    aggregate_key = transcript_idempotency_key(
        **{
            **idempotency_base,
            "prompt_version": "f" * 64,
            "pipeline_config_hash": aggregate_config_hash,
        }
    )
    assert len({base_key, template_key, vocabulary_key, aggregate_key}) == 4


class _CapturingSession:
    def __init__(self) -> None:
        self.added: list[Any] = []

    def add(self, value: Any) -> None:
        self.added.append(value)

    async def flush(self) -> None:
        return None


@pytest.mark.asyncio
async def test_v2_attempt_persistence_keeps_only_safe_prompt_hash_metadata() -> None:
    raw_audio_path = Path("C:/private/audio/customer-call.wav")
    api_key = "sk-this-must-never-be-persisted-1234567890"
    private_term = "Private Customer Name"
    private_context = "Private previous transcript context"
    operator_id = uuid4()
    manifest = V2GreekPromptBuilder().build_manifest(
        _operator_plan(operator_id, source_path=raw_audio_path),
        (
            RankedVocabularyTerm(
                value=private_term,
                priority=PRIORITY_SELECTED_OPERATOR,
                source=VOCABULARY_SOURCE_SELECTED_OPERATOR,
                roles=(TRACK_ROLE_OPERATOR,),
            ),
            RankedVocabularyTerm(
                value=api_key,
                priority=PRIORITY_CURRENT_CALL,
                source=VOCABULARY_SOURCE_CURRENT_CALL,
            ),
        ),
    )
    request_prompt = manifest.build(
        "operator-channel",
        previous_context=private_context,
    )
    assert request_prompt.prompt_hash is not None
    assert private_term in request_prompt.text
    assert private_context in request_prompt.text
    assert api_key not in request_prompt.text

    attempt = TranscriptionAttemptEvidence(
        track_id="operator-channel",
        chunk_index=0,
        start_seconds=0.0,
        end_seconds=12.5,
        model="gpt-4o-transcribe",
        audio_variant="topology-operator-channel",
        prompt_hash=request_prompt.prompt_hash,
        api_usage={"input_tokens": 7, "duration_seconds": 12.5},
    )
    result = OrchestratedTranscriptionResult(
        mode="operator_channel",
        model="gpt-4o-transcribe",
        language="el",
        prompt_version=manifest.prompt_identity,
        processing_duration_seconds=0.25,
        segments=(),
        tracks=(),
        usage={"chunks": [{"input_tokens": 7}], "totals": {"input_tokens": 7}},
        attribution_status="confirmed_by_pbx",
        attempts=(attempt,),
    )
    transcript = SimpleNamespace(
        id=uuid4(),
        operator_id=operator_id,
        prompt_template_version=manifest.template_version,
        vocabulary_hash=manifest.vocabulary_hash,
    )
    session = _CapturingSession()

    await _persist_transcription_result(
        session,  # type: ignore[arg-type]
        transcript,  # type: ignore[arg-type]
        result,
        SimpleNamespace(id=uuid4()),  # type: ignore[arg-type]
        None,
        None,
        12.5,
    )

    assert len(session.added) == 1
    persisted = session.added[0]
    assert isinstance(persisted, TranscriptionAttempt)
    assert persisted.prompt_hash == request_prompt.prompt_hash
    assert len(persisted.prompt_hash or "") == 64
    assert persisted.response_text is None
    assert persisted.selected is True
    assert persisted.api_usage == {
        "duration_seconds": 12.5,
        "input_tokens": 7,
    }
    assert transcript.prompt_version == manifest.prompt_identity
    assert transcript.prompt_template_version == GREEK_CALLCENTER_PROMPT_VERSION
    assert transcript.vocabulary_hash == manifest.vocabulary_hash

    safe_snapshot = json.dumps(
        {
            "audio_variant": persisted.audio_variant,
            "api_usage": persisted.api_usage,
            "model": persisted.model,
            "prompt_hash": persisted.prompt_hash,
            "response_text": persisted.response_text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    assert request_prompt.text not in safe_snapshot
    assert private_term not in safe_snapshot
    assert private_context not in safe_snapshot
    assert str(raw_audio_path) not in safe_snapshot
    assert api_key not in safe_snapshot
    assert {"prompt", "path", "api_key"}.isdisjoint(persisted.api_usage)


@pytest.mark.asyncio
async def test_duplicate_reuse_requires_matching_template_and_vocabulary_metadata() -> None:
    engine, sessions = await _new_database()
    checksum = "a" * 64
    aggregate_identity = "b" * 64
    template_version = GREEK_CALLCENTER_PROMPT_VERSION
    vocabulary_hash = "c" * 64
    pipeline_config_hash = "d" * 64
    now = datetime.now(UTC)
    try:
        async with sessions() as session:
            call = Call(
                yeastar_uid=f"duplicate-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
                duration_seconds=30,
                has_recording=True,
                provider_payload={},
            )
            session.add(call)
            await session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"duplicate-recording-{uuid4()}",
                status=RecordingStatus.INSPECTED,
                provider_payload={},
            )
            session.add(recording)
            await session.flush()
            transcript = Transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=None,
                idempotency_key=f"duplicate-transcript-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                prompt_version=aggregate_identity,
                prompt_template_version=template_version,
                vocabulary_hash=vocabulary_hash,
                completed_at=now,
                source_audio_sha256=checksum,
                is_diarized=False,
                transcription_mode=TranscriptionMode.DUAL_CHANNEL,
                speaker_attribution_status=SpeakerAttributionStatus.CHANNEL_UNKNOWN,
                pipeline_version=PIPELINE_V2_VERSION,
                pipeline_config_hash=pipeline_config_hash,
                preprocessing_profile=PIPELINE_V2_PREPROCESSING_PROFILE,
                is_current=True,
            )
            session.add(transcript)
            await session.commit()

            query = {
                "checksum": checksum,
                "model": "gpt-4o-transcribe",
                "diarized": False,
                "operator_id": None,
                "language": "el",
                "prompt_version": aggregate_identity,
                "pipeline_version": PIPELINE_V2_VERSION,
                "pipeline_config_hash": pipeline_config_hash,
                "preprocessing_profile": PIPELINE_V2_PREPROCESSING_PROFILE,
                "transcription_mode": TranscriptionMode.DUAL_CHANNEL,
                "prompt_template_version": template_version,
                "vocabulary_hash": vocabulary_hash,
            }
            exact = await _completed_duplicate_snapshot(session, **query)
            wrong_template = await _completed_duplicate_snapshot(
                session,
                **{
                    **query,
                    "prompt_template_version": "greek-callcenter-v3",
                },
            )
            wrong_vocabulary = await _completed_duplicate_snapshot(
                session,
                **{
                    **query,
                    "vocabulary_hash": "e" * 64,
                },
            )

            assert exact is not None
            assert exact[0].id == transcript.id
            assert exact[1] == ()
            assert wrong_template is None
            assert wrong_vocabulary is None
    finally:
        await engine.dispose()
