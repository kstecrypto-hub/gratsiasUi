from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.workers.pipeline as pipeline_module
from app.database.base import Base
from app.models import Call, Operator, Recording, SyncRun, Transcript
from app.models.enums import (
    Direction,
    RecordingStatus,
    RunStatus,
    SyncType,
    TranscriptStatus,
)
from app.services.transcription.prompt import build_vocabulary_prompt
from app.workers.pipeline import (
    LEGACY_ISOLATED_PROMPT_TEMPLATE_VERSION,
    ReprocessStateError,
    _activate_transcript_replacement,
    _commit_discovery_progress,
    _legacy_runtime_pipeline_config_hash,
    _ordered_keyword_variant_phrases,
    _transcript_recency_order,
)


def test_runtime_pipeline_identity_covers_every_legacy_execution_choice() -> None:
    base = {
        "diarized": False,
        "channel_index": 0,
        "max_upload_bytes": 25 * 1024 * 1024,
        "prompt_template_version": LEGACY_ISOLATED_PROMPT_TEMPLATE_VERSION,
        "preprocessing_profile": "legacy-current",
        "diarized_chunk_seconds": 480,
        "isolated_chunk_seconds": 15,
        "isolated_response_format": "json",
        "diarized_response_format": "diarized_json",
        "diarized_chunking_strategy": "auto",
    }
    identity = _legacy_runtime_pipeline_config_hash(**base)

    assert len(identity) == 64
    assert identity == _legacy_runtime_pipeline_config_hash(**base)
    variants = {
        "diarized": True,
        "channel_index": 1,
        "max_upload_bytes": 24 * 1024 * 1024,
        "prompt_template_version": "legacy-template-v2",
        "preprocessing_profile": "legacy-normalized-v2",
        "isolated_chunk_seconds": 14,
        "isolated_response_format": "verbose_json",
    }
    for field, value in variants.items():
        changed = dict(base)
        changed[field] = value
        assert _legacy_runtime_pipeline_config_hash(**changed) != identity, field

    diarized_base = dict(base)
    diarized_base.update(diarized=True, channel_index=None)
    diarized_identity = _legacy_runtime_pipeline_config_hash(**diarized_base)
    for field, value in {
        "diarized_chunk_seconds": 479,
        "diarized_response_format": "json",
        "diarized_chunking_strategy": "server_v2",
    }.items():
        changed = dict(diarized_base)
        changed[field] = value
        assert _legacy_runtime_pipeline_config_hash(**changed) != diarized_identity, field


def test_keyword_variant_order_makes_prompt_identity_stable() -> None:
    first_id = UUID("00000000-0000-0000-0000-000000000001")
    second_id = UUID("00000000-0000-0000-0000-000000000002")
    first_created = datetime(2026, 1, 1, tzinfo=UTC)
    second_created = first_created + timedelta(seconds=1)
    variants = [
        SimpleNamespace(
            id=second_id,
            phrase="Entered second",
            created_at=second_created,
        ),
        SimpleNamespace(
            id=second_id,
            phrase="Same-time second",
            created_at=first_created,
        ),
        SimpleNamespace(
            id=first_id,
            phrase="Entered first",
            created_at=first_created,
        ),
    ]
    keyword = SimpleNamespace(variants=variants)

    ordered = _ordered_keyword_variant_phrases(keyword)
    reversed_ordered = _ordered_keyword_variant_phrases(
        SimpleNamespace(variants=list(reversed(variants)))
    )

    assert ordered == ["Entered first", "Same-time second", "Entered second"]
    assert reversed_ordered == ordered
    assert build_vocabulary_prompt(ordered) == build_vocabulary_prompt(reversed_ordered)


async def _new_lineage_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA foreign_keys=ON"))
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    return engine, sessions()


async def _lineage_target(session):
    now = datetime.now(UTC)
    operator = Operator(
        yeastar_extension_id=f"lineage-operator-{uuid4()}",
        extension_number=str(uuid4()),
        display_name="Lineage Operator",
        last_synced_at=now,
    )
    call = Call(
        yeastar_uid=f"lineage-call-{uuid4()}",
        started_at=now,
        direction=Direction.INBOUND,
    )
    session.add_all([operator, call])
    await session.flush()
    recording = Recording(
        call_id=call.id,
        yeastar_recording_id=f"lineage-recording-{uuid4()}",
        status=RecordingStatus.COMPLETED,
    )
    session.add(recording)
    await session.flush()
    return call, recording, operator


def _completed_transcript(
    *,
    call: Call,
    recording: Recording,
    operator: Operator | None,
    is_current: bool,
    supersedes_transcript_id: UUID | None = None,
) -> Transcript:
    return Transcript(
        call_id=call.id,
        recording_id=recording.id,
        operator_id=operator.id if operator is not None else None,
        idempotency_key=f"lineage-transcript-{uuid4()}",
        status=TranscriptStatus.COMPLETED,
        model="gpt-4o-transcribe",
        language="el",
        source_audio_sha256=uuid4().hex * 2,
        completed_at=datetime.now(UTC),
        is_current=is_current,
        supersedes_transcript_id=supersedes_transcript_id,
    )


@pytest.mark.parametrize("cycle_size", [2, 3])
async def test_replacement_rejects_multi_hop_supersession_cycles(
    cycle_size: int,
) -> None:
    engine, session = await _new_lineage_session()
    try:
        async with session:
            call, recording, operator = await _lineage_target(session)
            replacement = _completed_transcript(
                call=call,
                recording=recording,
                operator=operator,
                is_current=False,
            )
            session.add(replacement)
            await session.flush()
            predecessor_id = replacement.id
            if cycle_size == 3:
                middle = _completed_transcript(
                    call=call,
                    recording=recording,
                    operator=operator,
                    is_current=False,
                    supersedes_transcript_id=replacement.id,
                )
                session.add(middle)
                await session.flush()
                predecessor_id = middle.id
            previous = _completed_transcript(
                call=call,
                recording=recording,
                operator=operator,
                is_current=True,
                supersedes_transcript_id=predecessor_id,
            )
            session.add(previous)
            await session.commit()

            with pytest.raises(ReprocessStateError, match="cycle"):
                await _activate_transcript_replacement(
                    session,
                    replacement,
                    previous.id,
                )
    finally:
        await engine.dispose()


async def test_replacement_rejects_cross_operator_history() -> None:
    engine, session = await _new_lineage_session()
    try:
        async with session:
            call, recording, replacement_operator = await _lineage_target(session)
            previous_operator = Operator(
                yeastar_extension_id=f"other-lineage-operator-{uuid4()}",
                extension_number=str(uuid4()),
                display_name="Other Lineage Operator",
                last_synced_at=datetime.now(UTC),
            )
            session.add(previous_operator)
            await session.flush()
            replacement = _completed_transcript(
                call=call,
                recording=recording,
                operator=replacement_operator,
                is_current=False,
            )
            previous = _completed_transcript(
                call=call,
                recording=recording,
                operator=previous_operator,
                is_current=True,
            )
            session.add_all([replacement, previous])
            await session.commit()

            with pytest.raises(ReprocessStateError, match="operator boundary"):
                await _activate_transcript_replacement(
                    session,
                    replacement,
                    previous.id,
                )
    finally:
        await engine.dispose()


async def test_replacement_accepts_valid_linear_history() -> None:
    engine, session = await _new_lineage_session()
    try:
        async with session:
            call, recording, operator = await _lineage_target(session)
            ancestor = _completed_transcript(
                call=call,
                recording=recording,
                operator=operator,
                is_current=False,
            )
            session.add(ancestor)
            await session.flush()
            previous = _completed_transcript(
                call=call,
                recording=recording,
                operator=operator,
                is_current=True,
                supersedes_transcript_id=ancestor.id,
            )
            replacement = _completed_transcript(
                call=call,
                recording=recording,
                operator=operator,
                is_current=False,
            )
            session.add_all([previous, replacement])
            await session.commit()

            await _activate_transcript_replacement(
                session,
                replacement,
                previous.id,
            )
            await session.commit()

            assert replacement.is_current is True
            assert replacement.supersedes_transcript_id == previous.id
            assert previous.is_current is False
    finally:
        await engine.dispose()


async def test_transcript_recency_order_is_null_safe_and_total() -> None:
    engine, session = await _new_lineage_session()
    try:
        async with session:
            call, recording, operator = await _lineage_target(session)
            timestamp = datetime(2026, 1, 2, tzinfo=UTC)
            lower_id = UUID("00000000-0000-0000-0000-000000000001")
            higher_id = UUID("00000000-0000-0000-0000-000000000002")
            rows = [
                _completed_transcript(
                    call=call,
                    recording=recording,
                    operator=operator,
                    is_current=False,
                ),
                _completed_transcript(
                    call=call,
                    recording=recording,
                    operator=operator,
                    is_current=False,
                ),
                _completed_transcript(
                    call=call,
                    recording=recording,
                    operator=operator,
                    is_current=False,
                ),
            ]
            rows[0].id = lower_id
            rows[1].id = higher_id
            rows[2].id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
            for row in rows[:2]:
                row.completed_at = timestamp
                row.created_at = timestamp
            rows[2].completed_at = None
            rows[2].created_at = timestamp
            session.add_all(rows)
            await session.commit()

            ordered = (
                await session.scalars(
                    select(Transcript)
                    .where(Transcript.id.in_([row.id for row in rows]))
                    .order_by(*_transcript_recency_order())
                )
            ).all()

            assert [row.id for row in ordered] == [
                higher_id,
                lower_id,
                UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
            ]
    finally:
        await engine.dispose()


async def test_discovery_aborts_when_its_redis_lease_is_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LostLease:
        async def acquire(self, *, blocking: bool) -> bool:
            assert blocking is False
            return True

        async def owned(self) -> bool:
            return False

        async def release(self) -> None:
            raise AssertionError("A lost lease must not be released.")

    lease = LostLease()
    redis = SimpleNamespace(lock=lambda *_args, **_kwargs: lease)

    async def completed_discovery(_job_id: UUID) -> pipeline_module.DiscoveryResult:
        return pipeline_module.DiscoveryResult([])

    monkeypatch.setattr(pipeline_module, "get_redis", lambda: redis)
    monkeypatch.setattr(
        pipeline_module,
        "_discover_job_items_locked",
        completed_discovery,
    )

    with pytest.raises(
        pipeline_module.ProcessingBusyError,
        match="lock became unavailable",
    ):
        await pipeline_module.discover_job_items(uuid4())


@pytest.mark.asyncio
async def test_discovery_progress_commit_is_lease_fenced_and_heartbeated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    initial = datetime(2026, 1, 1, tzinfo=UTC)
    heartbeat = initial + timedelta(minutes=5)
    checks = 0

    async def owned_lease() -> None:
        nonlocal checks
        checks += 1

    async def lost_lease() -> None:
        raise pipeline_module.ProcessingLeaseLostError

    monkeypatch.setattr(pipeline_module, "utc_now", lambda: heartbeat)
    try:
        async with sessions() as session:
            run = SyncRun(
                sync_type=SyncType.CALLS,
                status=RunStatus.RUNNING,
                idempotency_key=f"heartbeat-{uuid4()}",
                started_at=initial,
                updated_at=initial,
            )
            session.add(run)
            await session.commit()
            run_id = run.id

            await _commit_discovery_progress(session, run, owned_lease)
            assert checks == 1

            run.records_seen = 7
            with pytest.raises(pipeline_module.ProcessingLeaseLostError):
                await _commit_discovery_progress(session, run, lost_lease)
            await session.rollback()

        async with sessions() as session:
            persisted = await session.get(SyncRun, run_id)
            assert persisted is not None
            assert persisted.updated_at == heartbeat.replace(tzinfo=None)
            assert persisted.records_seen == 0
    finally:
        await engine.dispose()
