from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Barrier
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from app.database.base import Base
from app.models import (
    Call,
    Operator,
    ProcessingJob,
    ProcessingJobItem,
    Recording,
    Transcript,
    TranscriptSegment,
    TranscriptionAttempt,
    User,
)
from app.models.enums import (
    Direction,
    ItemStatus,
    JobStatus,
    RecordingStatus,
    SpeakerAttributionStatus,
    SpeakerSource,
    TranscriptionMode,
    TranscriptStatus,
)
from app.services.transcription.types import TranscriptionAttemptEvidence
from app.workers.pipeline import (
    _archive_failed_v2_attempt_history,
    _lock_recording_rows,
    _persist_partial_attempt_evidence,
)


MIGRATION_REVISION = "f3b9c7d1a620"
HARDENING_MIGRATION_REVISION = "a91d4e7c2b30"
PROMPT_METADATA_MIGRATION_REVISION = "d62c8f3a1b40"
ATTEMPT_SELECTION_MIGRATION_REVISION = "e74f9b2c6d10"
LEGACY_DIARIZED_PIPELINE_CONFIG_HASH = (
    "ef5ef358c56c2900297a8233a323a2b295faf08dd8c98dc621ae133560e25e61"
)
LEGACY_ISOLATED_PIPELINE_CONFIG_HASH = (
    "ad088e9537186db18c69c6781b79ae9e020192cc158790a4ab7e498a94bff2a5"
)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
POSTGRES_MIGRATION_TEST = pytest.mark.skipif(
    os.environ.get("RUN_POSTGRES_MIGRATION_TESTS") != "1",
    reason="Set RUN_POSTGRES_MIGRATION_TESTS=1 to run isolated PostgreSQL migration tests.",
)


def _normalized_sql(value: object) -> str:
    return " ".join(str(value).lower().split())


def test_pipeline_v2_schema_contract() -> None:
    transcript = Base.metadata.tables["transcripts"]
    segment = Base.metadata.tables["transcript_segments"]
    attempt = Base.metadata.tables["transcription_attempts"]
    item = Base.metadata.tables["processing_job_items"]

    assert set(TranscriptionMode) == {
        TranscriptionMode.LEGACY,
        TranscriptionMode.OPERATOR_CHANNEL,
        TranscriptionMode.DUAL_CHANNEL,
        TranscriptionMode.MONO_DIARIZATION,
    }
    assert set(SpeakerAttributionStatus) == {
        SpeakerAttributionStatus.CONFIRMED_BY_PBX,
        SpeakerAttributionStatus.CALLER_CALLEE_ONLY,
        SpeakerAttributionStatus.CHANNEL_UNKNOWN,
        SpeakerAttributionStatus.ANONYMOUS_DIARIZATION,
        SpeakerAttributionStatus.MANUALLY_ASSIGNED,
    }
    assert "requested_pipeline_version" in item.c
    assert item.c.requested_pipeline_version.type.length == 64
    assert "result_transcript_id" in item.c
    result_transcript_fk = next(
        foreign_key
        for foreign_key in item.foreign_key_constraints
        if list(foreign_key.columns)[0].name == "result_transcript_id"
    )
    assert result_transcript_fk.ondelete == "SET NULL"

    expected_transcript_columns = {
        "transcription_mode",
        "speaker_attribution_status",
        "pipeline_version",
        "pipeline_config_hash",
        "prompt_template_version",
        "vocabulary_hash",
        "preprocessing_profile",
        "quality_summary",
        "supersedes_transcript_id",
        "is_current",
    }
    assert expected_transcript_columns <= set(transcript.c.keys())
    assert transcript.c.pipeline_config_hash.type.length == 64
    assert transcript.c.pipeline_version.type.length == 64
    assert transcript.c.prompt_template_version.type.length == 64
    assert transcript.c.prompt_template_version.nullable is True
    assert transcript.c.vocabulary_hash.type.length == 64
    assert transcript.c.vocabulary_hash.nullable is True
    assert {
        "prompt",
        "prompt_text",
        "prompt_body",
        "api_key",
        "raw_audio_path",
    }.isdisjoint(transcript.c.keys())
    assert transcript.c.is_current.nullable is False
    supersedes = next(
        foreign_key
        for foreign_key in transcript.foreign_key_constraints
        if list(foreign_key.columns)[0].name == "supersedes_transcript_id"
    )
    assert supersedes.ondelete == "SET NULL"
    assert supersedes.referred_table is transcript
    assert {
        constraint.name
        for constraint in transcript.constraints
        if isinstance(constraint, sa.CheckConstraint)
    } >= {"ck_transcripts_transcript_not_self_superseding"}
    assert {index.name for index in transcript.indexes if index.unique} >= {
        "uq_transcripts_current_attributed",
        "uq_transcripts_current_unattributed",
    }

    expected_segment_columns = {
        "channel_index",
        "track_id",
        "chunk_index",
        "mean_logprob",
        "low_logprob_ratio",
        "quality_flags",
        "audio_variant",
    }
    assert expected_segment_columns <= set(segment.c.keys())
    assert isinstance(segment.c.channel_index.type, sa.SmallInteger)
    assert segment.c.track_id.type.length == 128
    assert segment.c.mean_logprob.type.precision == 8
    assert segment.c.mean_logprob.type.scale == 5
    assert segment.c.low_logprob_ratio.type.precision == 6
    assert segment.c.low_logprob_ratio.type.scale == 5
    assert segment.c.quality_flags.nullable is False

    assert "prompt" not in attempt.c
    assert set(attempt.c.keys()) >= {
        "id",
        "transcript_id",
        "track_id",
        "chunk_index",
        "start_seconds",
        "end_seconds",
        "model",
        "audio_variant",
        "prompt_hash",
        "response_text",
        "mean_logprob",
        "low_logprob_ratio",
        "selected",
        "api_usage",
        "completed_at",
        "created_at",
        "updated_at",
    }
    transcript_fk = next(iter(attempt.foreign_key_constraints))
    assert transcript_fk.ondelete == "CASCADE"
    assert {index.name for index in attempt.indexes} == {
        "ix_transcription_attempts_transcript_track_chunk",
        "uq_transcription_attempts_selected_chunk",
    }
    selected_attempt = next(
        index
        for index in attempt.indexes
        if index.name == "uq_transcription_attempts_selected_chunk"
    )
    assert selected_attempt.unique is True
    assert _normalized_sql(
        selected_attempt.dialect_options["postgresql"]["where"]
    ) == "selected is true"
    assert _normalized_sql(
        selected_attempt.dialect_options["sqlite"]["where"]
    ) == "selected = 1"
    assert TranscriptionAttempt.__tablename__ == "transcription_attempts"


@pytest.mark.parametrize("attributed", [True, False], ids=["attributed", "unattributed"])
async def test_current_transcript_uniqueness_is_enforced_for_nullable_operator(
    attributed: bool,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA foreign_keys=ON"))
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    async with sessions() as session:
        operator = Operator(
            yeastar_extension_id=f"extension-{uuid4()}",
            extension_number=str(uuid4()),
            display_name="Schema Test Operator",
            last_synced_at=now,
        )
        call = Call(
            yeastar_uid=f"call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
        )
        session.add_all([operator, call])
        await session.flush()
        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"recording-{uuid4()}",
            status=RecordingStatus.INSPECTED,
        )
        session.add(recording)
        await session.flush()
        call_id = call.id
        recording_id = recording.id
        operator_id = operator.id if attributed else None
        first = _completed_transcript(
            call_id=call_id,
            recording_id=recording_id,
            operator_id=operator_id,
            is_current=True,
        )
        session.add(first)
        await session.commit()

        assert first.transcription_mode is TranscriptionMode.LEGACY
        assert first.pipeline_version == "legacy-v1"
        stored_mode = await session.scalar(
            text("SELECT transcription_mode FROM transcripts WHERE id = :id"),
            {"id": first.id.hex},
        )
        assert stored_mode == "LEGACY"

        session.add(
            _completed_transcript(
                call_id=call_id,
                recording_id=recording_id,
                operator_id=operator_id,
                is_current=True,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        session.add(
            _completed_transcript(
                call_id=call_id,
                recording_id=recording_id,
                operator_id=operator_id,
                is_current=False,
            )
        )
        await session.commit()

    await engine.dispose()


@pytest.mark.asyncio
async def test_selected_attempt_uniqueness_is_enforced_by_a_real_transaction() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA foreign_keys=ON"))
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    async with sessions() as session:
        call = Call(
            yeastar_uid=f"attempt-call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
        )
        session.add(call)
        await session.flush()
        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"attempt-recording-{uuid4()}",
            status=RecordingStatus.INSPECTED,
        )
        session.add(recording)
        await session.flush()
        transcript = _completed_transcript(
            call_id=call.id,
            recording_id=recording.id,
            operator_id=None,
            is_current=True,
        )
        session.add(transcript)
        await session.flush()
        transcript_id = transcript.id

        session.add(
            _attempt_row(
                transcript_id=transcript_id,
                audio_variant="v2-raw-lossless-pcm16-v1",
                selected=True,
            )
        )
        await session.commit()

        session.add(
            _attempt_row(
                transcript_id=transcript_id,
                audio_variant="v2-light-normalized-telephone-v1",
                selected=True,
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()
        await session.rollback()

        session.add(
            _attempt_row(
                transcript_id=transcript_id,
                audio_variant="v2-light-normalized-telephone-v1",
                selected=False,
            )
        )
        await session.commit()
        attempts = (
            await session.scalars(
                select(TranscriptionAttempt)
                .where(TranscriptionAttempt.transcript_id == transcript_id)
                .order_by(TranscriptionAttempt.audio_variant)
            )
        ).all()
        assert len(attempts) == 2
        assert sum(attempt.selected for attempt in attempts) == 1

    await engine.dispose()


@pytest.mark.asyncio
async def test_partial_attempt_and_failed_status_persist_atomically() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.execute(text("PRAGMA foreign_keys=ON"))
        await connection.run_sync(Base.metadata.create_all)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    now = datetime.now(UTC)

    async with sessions() as session:
        call = Call(
            yeastar_uid=f"partial-call-{uuid4()}",
            started_at=now,
            direction=Direction.INBOUND,
        )
        session.add(call)
        await session.flush()
        recording = Recording(
            call_id=call.id,
            yeastar_recording_id=f"partial-recording-{uuid4()}",
            status=RecordingStatus.INSPECTED,
        )
        session.add(recording)
        await session.flush()
        stable_key = f"partial-transcript-{uuid4()}"
        transcript = Transcript(
            call_id=call.id,
            recording_id=recording.id,
            operator_id=None,
            idempotency_key=stable_key,
            status=TranscriptStatus.PROCESSING,
            model="gpt-4o-transcribe",
            language="el",
            source_audio_sha256=uuid4().hex * 2,
            is_diarized=False,
            is_current=True,
            transcription_mode=TranscriptionMode.OPERATOR_CHANNEL,
        )
        session.add(transcript)
        await session.commit()
        transcript_id = transcript.id
        call_id = call.id
        recording_id = recording.id
        evidence = (
            TranscriptionAttemptEvidence(
                track_id="operator-channel",
                chunk_index=0,
                start_seconds=0.0,
                end_seconds=10.0,
                model="gpt-4o-transcribe",
                audio_variant="v2-raw-lossless-pcm16-v1",
                prompt_hash="b" * 64,
                response_text="paid raw response",
                mean_logprob=-1.2,
                low_logprob_ratio=0.5,
                selected=True,
                api_usage={"input_tokens": 4},
                completed_at=now,
            ),
        )

        transcript.status = TranscriptStatus.FAILED
        await _persist_partial_attempt_evidence(session, transcript, evidence)
        await session.rollback()

        rolled_back = await session.get(Transcript, transcript_id)
        assert rolled_back is not None
        assert rolled_back.status == TranscriptStatus.PROCESSING
        assert (
            await session.scalar(
                select(sa.func.count())
                .select_from(TranscriptionAttempt)
                .where(TranscriptionAttempt.transcript_id == transcript_id)
            )
            == 0
        )

        rolled_back.status = TranscriptStatus.FAILED
        await _persist_partial_attempt_evidence(session, rolled_back, evidence)
        await session.commit()
        persisted = (
            await session.scalars(
                select(TranscriptionAttempt).where(
                    TranscriptionAttempt.transcript_id == transcript_id
                )
            )
        ).all()
        assert len(persisted) == 1
        assert persisted[0].response_text == "paid raw response"
        assert persisted[0].selected is True
        failed = await session.get(Transcript, transcript_id)
        assert failed is not None
        assert failed.status == TranscriptStatus.FAILED
        await _persist_partial_attempt_evidence(session, failed, evidence)
        await session.commit()
        assert (
            await session.scalar(
                select(sa.func.count())
                .select_from(TranscriptionAttempt)
                .where(TranscriptionAttempt.transcript_id == transcript_id)
            )
            == 1
        )

        assert await _archive_failed_v2_attempt_history(
            session,
            failed,
            stable_key=stable_key,
            transcription_mode=TranscriptionMode.OPERATOR_CHANNEL,
        )
        replacement = Transcript(
            call_id=call_id,
            recording_id=recording_id,
            operator_id=None,
            idempotency_key=stable_key,
            status=TranscriptStatus.PROCESSING,
            model="gpt-4o-transcribe",
            language="el",
            source_audio_sha256=uuid4().hex * 2,
            is_diarized=False,
            is_current=True,
            transcription_mode=TranscriptionMode.OPERATOR_CHANNEL,
        )
        session.add(replacement)
        await session.commit()

        historical = await session.get(Transcript, transcript_id)
        assert historical is not None
        assert historical.idempotency_key != stable_key
        assert historical.is_current is False
        assert replacement.idempotency_key == stable_key
        assert replacement.id != historical.id
        assert (
            await session.scalar(
                select(sa.func.count())
                .select_from(TranscriptionAttempt)
                .where(TranscriptionAttempt.transcript_id == historical.id)
            )
            == 1
        )

    await engine.dispose()


def _attempt_row(
    *,
    transcript_id: UUID,
    audio_variant: str,
    selected: bool,
) -> TranscriptionAttempt:
    return TranscriptionAttempt(
        transcript_id=transcript_id,
        track_id="operator-channel",
        chunk_index=0,
        start_seconds=Decimal("0.000"),
        end_seconds=Decimal("10.000"),
        model="gpt-4o-transcribe",
        audio_variant=audio_variant,
        prompt_hash="a" * 64,
        response_text=audio_variant,
        mean_logprob=Decimal("-0.50000"),
        low_logprob_ratio=Decimal("0.10000"),
        selected=selected,
        api_usage={"input_tokens": 1},
        completed_at=datetime.now(UTC),
    )


def _completed_transcript(
    *,
    call_id: UUID,
    recording_id: UUID,
    operator_id: UUID | None,
    is_current: bool,
) -> Transcript:
    return Transcript(
        call_id=call_id,
        recording_id=recording_id,
        operator_id=operator_id,
        idempotency_key=f"schema-transcript-{uuid4()}",
        status=TranscriptStatus.COMPLETED,
        model="gpt-4o-transcribe",
        language="el",
        source_audio_sha256=uuid4().hex * 2,
        is_diarized=operator_id is None,
        is_current=is_current,
        completed_at=datetime.now(UTC),
    )


@contextmanager
def _isolated_postgres_schema() -> Iterator[tuple[sa.Connection, str]]:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://app:app@postgres:5432/yeastar",
    )
    if not database_url.startswith("postgresql"):
        pytest.skip("PostgreSQL is required for migration behavior tests.")
    engine = sa.create_engine(database_url, poolclass=sa.pool.NullPool)
    schema = f"test_pipeline_v2_{uuid4().hex}"
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin:
        admin.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}"')
            yield connection, schema
    finally:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin:
            admin.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        engine.dispose()


@contextmanager
def _committed_postgres_schema() -> Iterator[tuple[sa.Engine, str]]:
    """Create a migrated schema whose objects are visible to concurrent sessions."""

    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://app:app@postgres:5432/yeastar",
    )
    if not database_url.startswith("postgresql"):
        pytest.skip("PostgreSQL is required for migration behavior tests.")
    engine = sa.create_engine(database_url, poolclass=sa.pool.NullPool)
    schema = f"test_pipeline_v2_{uuid4().hex}"
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin:
        admin.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}"')
            _run_upgrades(connection)
        yield engine, schema
    finally:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as admin:
            admin.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        engine.dispose()


def _revision_chain() -> list:
    config = Config()
    config.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    script = ScriptDirectory.from_config(config)
    return list(reversed(list(script.walk_revisions(base="base", head="heads"))))


def test_safe_prompt_metadata_migration_is_additive_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = next(
        item
        for item in _revision_chain()
        if item.revision == PROMPT_METADATA_MIGRATION_REVISION
    )
    assert migration.down_revision == HARDENING_MIGRATION_REVISION

    added: list[tuple[str, sa.Column]] = []
    dropped: list[tuple[str, str]] = []
    monkeypatch.setattr(
        migration.module.op,
        "add_column",
        lambda table, column: added.append((table, column)),
    )
    monkeypatch.setattr(
        migration.module.op,
        "drop_column",
        lambda table, column: dropped.append((table, column)),
    )

    migration.module.upgrade()
    migration.module.downgrade()

    assert [(table, column.name) for table, column in added] == [
        ("transcripts", "prompt_template_version"),
        ("transcripts", "vocabulary_hash"),
    ]
    assert all(isinstance(column.type, sa.String) for _, column in added)
    assert all(column.type.length == 64 for _, column in added)
    assert all(column.nullable is True for _, column in added)
    assert dropped == [
        ("transcripts", "vocabulary_hash"),
        ("transcripts", "prompt_template_version"),
    ]


def test_selected_attempt_migration_is_partial_unique_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = next(
        item
        for item in _revision_chain()
        if item.revision == ATTEMPT_SELECTION_MIGRATION_REVISION
    )
    assert migration.down_revision == PROMPT_METADATA_MIGRATION_REVISION

    created: list[tuple[tuple[object, ...], dict[str, object]]] = []
    dropped: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        migration.module.op,
        "create_index",
        lambda *args, **kwargs: created.append((args, kwargs)),
    )
    monkeypatch.setattr(
        migration.module.op,
        "drop_index",
        lambda *args, **kwargs: dropped.append((args, kwargs)),
    )

    migration.module.upgrade()
    migration.module.downgrade()

    assert created[0][0] == (
        "uq_transcription_attempts_selected_chunk",
        "transcription_attempts",
        ["transcript_id", "track_id", "chunk_index"],
    )
    assert created[0][1]["unique"] is True
    assert _normalized_sql(created[0][1]["postgresql_where"]) == "selected is true"
    assert _normalized_sql(created[0][1]["sqlite_where"]) == "selected = 1"
    assert dropped == [
        (
            ("uq_transcription_attempts_selected_chunk",),
            {"table_name": "transcription_attempts"},
        )
    ]


def _run_upgrades(connection: sa.Connection, *, stop_before: str | None = None):
    migration_context = MigrationContext.configure(
        connection,
        opts={"target_metadata": Base.metadata},
    )
    target_revision = None
    with Operations.context(migration_context):
        for revision in _revision_chain():
            if revision.revision == stop_before:
                target_revision = revision
                break
            revision.module.upgrade()
    return target_revision


def _run_downgrade(connection: sa.Connection, revision) -> None:
    migration_context = MigrationContext.configure(
        connection,
        opts={"target_metadata": Base.metadata},
    )
    with Operations.context(migration_context):
        revision.module.downgrade()


@POSTGRES_MIGRATION_TEST
def test_fresh_postgres_upgrade_and_pipeline_v2_downgrade() -> None:
    with _isolated_postgres_schema() as (connection, schema):
        _run_upgrades(connection)
        inspector = inspect(connection)
        assert "transcription_attempts" in inspector.get_table_names(schema=schema)
        assert {
            column["name"] for column in inspector.get_columns("transcripts", schema=schema)
        } >= {
            "transcription_mode",
            "speaker_attribution_status",
            "pipeline_version",
            "pipeline_config_hash",
            "prompt_template_version",
            "vocabulary_hash",
            "preprocessing_profile",
            "quality_summary",
            "supersedes_transcript_id",
            "is_current",
        }
        assert {
            column["name"]
            for column in inspector.get_columns("processing_job_items", schema=schema)
        } >= {"requested_pipeline_version", "result_transcript_id"}
        assert {index["name"] for index in inspector.get_indexes("transcripts", schema=schema)} >= {
            "uq_transcripts_current_attributed",
            "uq_transcripts_current_unattributed",
        }
        assert {
            index["name"]
            for index in inspector.get_indexes(
                "transcription_attempts",
                schema=schema,
            )
        } >= {"uq_transcription_attempts_selected_chunk"}
        check_names = {
            item["name"] for item in inspector.get_check_constraints("transcripts", schema=schema)
        }
        assert {
            "ck_transcripts_transcription_mode",
            "ck_transcripts_speaker_attribution_status",
            "ck_transcripts_transcript_not_self_superseding",
        } <= check_names

        prompt_metadata_revision = next(
            item
            for item in _revision_chain()
            if item.revision == PROMPT_METADATA_MIGRATION_REVISION
        )
        _run_downgrade(connection, prompt_metadata_revision)
        inspector = inspect(connection)
        assert {
            "prompt_template_version",
            "vocabulary_hash",
        }.isdisjoint(
            column["name"]
            for column in inspector.get_columns("transcripts", schema=schema)
        )

        hardening_revision = next(
            item for item in _revision_chain() if item.revision == HARDENING_MIGRATION_REVISION
        )
        _run_downgrade(connection, hardening_revision)
        foundation_revision = next(
            item for item in _revision_chain() if item.revision == MIGRATION_REVISION
        )
        _run_downgrade(connection, foundation_revision)
        inspector = inspect(connection)
        assert "transcription_attempts" not in inspector.get_table_names(schema=schema)
        assert "transcription_mode" not in {
            column["name"] for column in inspector.get_columns("transcripts", schema=schema)
        }
        assert "requested_pipeline_version" not in {
            column["name"]
            for column in inspector.get_columns("processing_job_items", schema=schema)
        }
        assert "result_transcript_id" not in {
            column["name"]
            for column in inspector.get_columns("processing_job_items", schema=schema)
        }


@POSTGRES_MIGRATION_TEST
def test_postgres_rejects_invalid_supersession_cycles_and_boundaries() -> None:
    with _isolated_postgres_schema() as (connection, _schema):
        foundation_revision = _run_upgrades(
            connection,
            stop_before=MIGRATION_REVISION,
        )
        assert foundation_revision is not None
        fixtures = _insert_legacy_transcripts(connection)
        migration_context = MigrationContext.configure(
            connection,
            opts={"target_metadata": Base.metadata},
        )
        with Operations.context(migration_context):
            foundation_revision.module.upgrade()
            hardening_revision = next(
                item for item in _revision_chain() if item.revision == HARDENING_MIGRATION_REVISION
            )
            hardening_revision.module.upgrade()

        connection.execute(
            text(
                """
                UPDATE transcripts
                SET supersedes_transcript_id = :newer
                WHERE id = :older
                """
            ),
            {
                "older": fixtures["attributed_old"],
                "newer": fixtures["attributed_new"],
            },
        )
        with pytest.raises(IntegrityError, match="supersession would create a cycle"):
            with connection.begin_nested():
                connection.execute(
                    text(
                        """
                        UPDATE transcripts
                        SET supersedes_transcript_id = :older
                        WHERE id = :newer
                        """
                    ),
                    {
                        "older": fixtures["attributed_old"],
                        "newer": fixtures["attributed_new"],
                    },
                )

        with pytest.raises(IntegrityError, match="logical target boundary"):
            with connection.begin_nested():
                connection.execute(
                    text(
                        """
                        UPDATE transcripts
                        SET supersedes_transcript_id = :other_target
                        WHERE id = :transcript_id
                        """
                    ),
                    {
                        "transcript_id": fixtures["channel_unknown"],
                        "other_target": fixtures["attributed_new"],
                    },
                )

        attributed_operator_id = connection.scalar(
            text("SELECT operator_id FROM transcripts WHERE id = :transcript_id"),
            {"transcript_id": fixtures["attributed_old"]},
        )
        assert attributed_operator_id is not None
        connection.execute(
            text(
                """
                UPDATE transcripts
                SET operator_id = :operator_id
                WHERE id = :transcript_id
                """
            ),
            {
                "operator_id": attributed_operator_id,
                "transcript_id": fixtures["channel_unknown"],
            },
        )
        for forbidden_operator_id in (None, uuid4()):
            with pytest.raises(
                IntegrityError,
                match="immutable logical target boundary",
            ):
                with connection.begin_nested():
                    connection.execute(
                        text(
                            """
                            UPDATE transcripts
                            SET operator_id = :operator_id
                            WHERE id = :transcript_id
                            """
                        ),
                        {
                            "operator_id": forbidden_operator_id,
                            "transcript_id": fixtures["channel_unknown"],
                        },
                    )


@POSTGRES_MIGRATION_TEST
def test_legacy_transcript_backfill_and_partial_uniqueness_on_postgres() -> None:
    with _isolated_postgres_schema() as (connection, _schema):
        revision = _run_upgrades(connection, stop_before=MIGRATION_REVISION)
        assert revision is not None
        fixtures = _insert_legacy_transcripts(connection)
        migration_context = MigrationContext.configure(
            connection,
            opts={"target_metadata": Base.metadata},
        )
        with Operations.context(migration_context):
            revision.module.upgrade()

        rows = {
            row.id: row
            for row in connection.execute(
                text(
                    """
                    SELECT id, transcription_mode, speaker_attribution_status,
                           pipeline_version, pipeline_config_hash,
                           preprocessing_profile, is_current
                    FROM transcripts
                    """
                )
            ).mappings()
        }
        assert all(row.transcription_mode == "LEGACY" for row in rows.values())
        assert all(row.pipeline_version == "legacy-v1" for row in rows.values())
        assert rows[fixtures["attributed_new"]].pipeline_config_hash == (
            LEGACY_ISOLATED_PIPELINE_CONFIG_HASH
        )
        assert rows[fixtures["diarized_new"]].pipeline_config_hash == (
            LEGACY_DIARIZED_PIPELINE_CONFIG_HASH
        )
        assert rows[fixtures["channel_unknown"]].pipeline_config_hash == (
            LEGACY_ISOLATED_PIPELINE_CONFIG_HASH
        )
        assert all(row.preprocessing_profile == "legacy-current" for row in rows.values())
        assert rows[fixtures["attributed_new"]].speaker_attribution_status == ("CONFIRMED_BY_PBX")
        assert rows[fixtures["diarized_new"]].speaker_attribution_status == (
            "ANONYMOUS_DIARIZATION"
        )
        assert rows[fixtures["channel_unknown"]].speaker_attribution_status == ("CHANNEL_UNKNOWN")
        current_ids = {row.id for row in rows.values() if row.is_current}
        assert current_ids == {
            fixtures["attributed_new"],
            fixtures["diarized_new"],
            fixtures["channel_unknown"],
        }
        item_bindings = dict(
            connection.execute(
                text(
                    """
                    SELECT id, result_transcript_id
                    FROM processing_job_items
                    WHERE id IN (:attributed_item, :diarized_item)
                    """
                ),
                {
                    "attributed_item": fixtures["attributed_item"],
                    "diarized_item": fixtures["diarized_item"],
                },
            ).all()
        )
        assert item_bindings == {
            fixtures["attributed_item"]: fixtures["attributed_new"],
            fixtures["diarized_item"]: fixtures["diarized_new"],
        }

        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    text("UPDATE transcripts SET is_current = true WHERE id = :id"),
                    {"id": fixtures["attributed_old"]},
                )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    text("UPDATE transcripts SET is_current = true WHERE id = :id"),
                    {"id": fixtures["diarized_old"]},
                )


@POSTGRES_MIGRATION_TEST
def test_concurrent_reprocess_jobs_are_database_serialized() -> None:
    """Only one request can win when two reprocess jobs race to become active."""

    with _committed_postgres_schema() as (engine, schema):
        now = datetime.now(UTC)
        user_id = uuid4()
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            session.add(
                User(
                    id=user_id,
                    email=f"reprocess-race-{uuid4()}@example.test",
                    password_hash="not-used-by-this-test",
                    is_active=True,
                )
            )
            session.commit()

        ready = Barrier(2)

        def submit(position: int) -> str:
            with Session(engine) as session:
                session.execute(text(f'SET search_path TO "{schema}"'))
                session.add(
                    ProcessingJob(
                        idempotency_key=f"concurrent-reprocess-{position}-{uuid4()}",
                        requested_by_id=user_id,
                        status=JobStatus.QUEUED,
                        date_from=now - timedelta(hours=1),
                        date_to=now,
                        selected_operator_ids=[],
                        selected_category_ids=[],
                        request_filters={"_reprocess": True},
                        current_stage="Queued for reprocessing",
                    )
                )
                ready.wait(timeout=10)
                try:
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    return "rejected"
                return "created"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = sorted(executor.map(submit, range(2)))

        assert outcomes == ["created", "rejected"]


@POSTGRES_MIGRATION_TEST
def test_concurrent_supersession_writes_cannot_commit_a_cycle() -> None:
    with _committed_postgres_schema() as (engine, schema):
        now = datetime.now(UTC)
        call_id = uuid4()
        recording_id = uuid4()
        first_id = uuid4()
        second_id = uuid4()
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            session.add(
                Call(
                    id=call_id,
                    yeastar_uid=f"supersession-race-call-{uuid4()}",
                    started_at=now,
                    direction=Direction.INBOUND,
                )
            )
            session.flush()
            session.add(
                Recording(
                    id=recording_id,
                    call_id=call_id,
                    yeastar_recording_id=f"supersession-race-recording-{uuid4()}",
                    status=RecordingStatus.COMPLETED,
                )
            )
            session.flush()
            session.add_all(
                [
                    _completed_transcript(
                        call_id=call_id,
                        recording_id=recording_id,
                        operator_id=None,
                        is_current=False,
                    ),
                    _completed_transcript(
                        call_id=call_id,
                        recording_id=recording_id,
                        operator_id=None,
                        is_current=False,
                    ),
                ]
            )
            session.flush()
            transcripts = session.scalars(
                select(Transcript)
                .where(Transcript.recording_id == recording_id)
                .order_by(Transcript.id)
            ).all()
            transcripts[0].id = first_id
            transcripts[1].id = second_id
            session.commit()

        ready = Barrier(2)

        def link(child_id: UUID, parent_id: UUID) -> str:
            with Session(engine) as session:
                session.execute(text(f'SET search_path TO "{schema}"'))
                ready.wait(timeout=10)
                try:
                    session.execute(
                        text(
                            """
                            UPDATE transcripts
                            SET supersedes_transcript_id = :parent_id
                            WHERE id = :child_id
                            """
                        ),
                        {"child_id": child_id, "parent_id": parent_id},
                    )
                    session.commit()
                except IntegrityError:
                    session.rollback()
                    return "rejected"
                return "committed"

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(link, first_id, second_id),
                executor.submit(link, second_id, first_id),
            ]
            outcomes = sorted(future.result(timeout=20) for future in futures)

        assert outcomes == ["committed", "rejected"]
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            links = dict(
                session.execute(
                    select(Transcript.id, Transcript.supersedes_transcript_id).where(
                        Transcript.id.in_([first_id, second_id])
                    )
                ).all()
            )
        assert sum(parent_id is not None for parent_id in links.values()) == 1
        assert not (links[first_id] == second_id and links[second_id] == first_id)


@POSTGRES_MIGRATION_TEST
@pytest.mark.asyncio
async def test_recording_row_lock_linearizes_retention_before_discovery() -> None:
    with _committed_postgres_schema() as (engine, schema):
        now = datetime.now(UTC)
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            user = User(
                email=f"retention-race-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"retention-race-{uuid4()}",
                extension_number=uuid4().hex,
                display_name="Retention Race Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"retention-race-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
            )
            session.add_all([user, operator, call])
            session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"retention-race-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            job = ProcessingJob(
                idempotency_key=f"retention-race-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.QUEUED,
                date_from=now - timedelta(hours=1),
                date_to=now,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
            )
            session.add_all([recording, job])
            session.flush()
            transcript = _completed_transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                is_current=True,
            )
            session.add(transcript)
            session.flush()
            recording_id = recording.id
            transcript_id = transcript.id
            job_id = job.id
            call_id = call.id
            operator_id = operator.id
            session.commit()

        database_url = os.environ["DATABASE_URL"]
        async_engine = create_async_engine(database_url, poolclass=sa.pool.NullPool)
        sessions = async_sessionmaker(async_engine, expire_on_commit=False)
        retention_locked = asyncio.Event()
        release_retention = asyncio.Event()
        retention_committed = asyncio.Event()
        discovery_started = asyncio.Event()
        item_id = uuid4()
        retention_task: asyncio.Task[None] | None = None
        discovery_task: asyncio.Task[None] | None = None

        async def retention_transaction() -> None:
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                await _lock_recording_rows(session, {recording_id})
                retention_locked.set()
                await asyncio.wait_for(release_retention.wait(), timeout=10)
                current = await session.get(Transcript, transcript_id)
                assert current is not None
                await session.delete(current)
                await session.commit()
                retention_committed.set()

        async def discovery_transaction() -> None:
            await retention_locked.wait()
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                discovery_started.set()
                await _lock_recording_rows(session, [recording_id])
                assert retention_committed.is_set()
                session.add(
                    ProcessingJobItem(
                        id=item_id,
                        job_id=job_id,
                        call_id=call_id,
                        operator_id=operator_id,
                        recording_id=recording_id,
                        idempotency_key=f"retention-race-item-{uuid4()}",
                        status=ItemStatus.QUEUED,
                        stage="queued",
                    )
                )
                await session.commit()

        try:
            retention_task = asyncio.create_task(retention_transaction())
            await asyncio.wait_for(retention_locked.wait(), timeout=10)
            discovery_task = asyncio.create_task(discovery_transaction())
            await asyncio.wait_for(discovery_started.wait(), timeout=10)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(discovery_task), timeout=0.1)
            release_retention.set()
            await asyncio.wait_for(
                asyncio.gather(retention_task, discovery_task),
                timeout=10,
            )

            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                assert await session.get(Transcript, transcript_id) is None
                assert await session.get(ProcessingJobItem, item_id) is not None
        finally:
            release_retention.set()
            pending_tasks = [
                task
                for task in (retention_task, discovery_task)
                if task is not None and not task.done()
            ]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await async_engine.dispose()


@POSTGRES_MIGRATION_TEST
@pytest.mark.asyncio
async def test_reprocess_and_retention_share_recording_first_lock_order() -> None:
    from app.api.results import _lock_current_reprocess_transcript
    from app.workers.pipeline import _lock_recording_job_rows

    with _committed_postgres_schema() as (engine, schema):
        now = datetime.now(UTC)
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            user = User(
                email=f"reprocess-retention-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"reprocess-retention-{uuid4()}",
                extension_number=uuid4().hex,
                display_name="Reprocess Retention Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"reprocess-retention-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
            )
            session.add_all([user, operator, call])
            session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"reprocess-retention-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            session.add(recording)
            session.flush()
            transcript = _completed_transcript(
                call_id=call.id,
                recording_id=recording.id,
                operator_id=operator.id,
                is_current=True,
            )
            session.add(transcript)
            session.flush()
            user_id = user.id
            operator_id = operator.id
            call_id = call.id
            recording_id = recording.id
            transcript_id = transcript.id
            session.commit()

        database_url = os.environ["DATABASE_URL"]
        async_engine = create_async_engine(database_url, poolclass=sa.pool.NullPool)
        sessions = async_sessionmaker(async_engine, expire_on_commit=False)
        reprocess_locked = asyncio.Event()
        retention_started = asyncio.Event()
        replacement_item_id = uuid4()
        reprocess_task: asyncio.Task[None] | None = None
        retention_task: asyncio.Task[None] | None = None

        async def reprocess_transaction() -> None:
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                current = await _lock_current_reprocess_transcript(
                    session,
                    call_id=call_id,
                    transcript_id=transcript_id,
                )
                assert current.id == transcript_id
                reprocess_locked.set()
                await asyncio.wait_for(retention_started.wait(), timeout=10)
                # Give retention a chance to request its Recording lock. With
                # the historical Transcript-first order this creates the exact
                # Recording/Transcript cycle PostgreSQL rejected as a deadlock.
                await asyncio.sleep(0.1)
                job = ProcessingJob(
                    idempotency_key=f"reprocess-retention-job-{uuid4()}",
                    requested_by_id=user_id,
                    status=JobStatus.QUEUED,
                    date_from=now - timedelta(hours=1),
                    date_to=now,
                    selected_operator_ids=[str(operator_id)],
                    selected_category_ids=[],
                    request_filters={
                        "_reprocess": True,
                        "_reprocess_targets": {str(replacement_item_id): str(transcript_id)},
                    },
                )
                session.add(job)
                await session.flush()
                session.add(
                    ProcessingJobItem(
                        id=replacement_item_id,
                        job_id=job.id,
                        call_id=call_id,
                        operator_id=operator_id,
                        recording_id=recording_id,
                        idempotency_key=f"reprocess-retention-item-{uuid4()}",
                        requested_pipeline_version="legacy-v1",
                        status=ItemStatus.QUEUED,
                        stage="queued_for_reprocessing",
                    )
                )
                await session.commit()

        async def retention_transaction() -> None:
            await asyncio.wait_for(reprocess_locked.wait(), timeout=10)
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                retention_started.set()
                await _lock_recording_rows(session, {recording_id})
                await _lock_recording_job_rows(session, {recording_id})
                locked = await session.scalar(
                    select(Transcript).where(Transcript.id == transcript_id).with_for_update()
                )
                assert locked is not None
                await session.commit()

        try:
            reprocess_task = asyncio.create_task(reprocess_transaction())
            retention_task = asyncio.create_task(retention_transaction())
            await asyncio.wait_for(
                asyncio.gather(reprocess_task, retention_task),
                timeout=10,
            )
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                assert await session.get(ProcessingJobItem, replacement_item_id) is not None
        finally:
            pending_tasks = [
                task
                for task in (reprocess_task, retention_task)
                if task is not None and not task.done()
            ]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await async_engine.dispose()


@POSTGRES_MIGRATION_TEST
@pytest.mark.asyncio
async def test_worker_and_rediscovery_share_recording_first_lock_order() -> None:
    from app.workers.pipeline import _lock_processable_item

    with _committed_postgres_schema() as (engine, schema):
        now = datetime.now(UTC)
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            user = User(
                email=f"worker-discovery-{uuid4()}@example.test",
                password_hash="not-used-by-this-test",
                is_active=True,
            )
            operator = Operator(
                yeastar_extension_id=f"worker-discovery-{uuid4()}",
                extension_number=uuid4().hex,
                display_name="Worker Discovery Operator",
                provider_active=True,
                enabled=True,
                last_synced_at=now,
            )
            call = Call(
                yeastar_uid=f"worker-discovery-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
            )
            session.add_all([user, operator, call])
            session.flush()
            recording = Recording(
                call_id=call.id,
                yeastar_recording_id=f"worker-discovery-recording-{uuid4()}",
                status=RecordingStatus.DOWNLOADED,
            )
            job = ProcessingJob(
                idempotency_key=f"worker-discovery-job-{uuid4()}",
                requested_by_id=user.id,
                status=JobStatus.TRANSCRIBING,
                date_from=now - timedelta(hours=1),
                date_to=now,
                selected_operator_ids=[str(operator.id)],
                selected_category_ids=[],
            )
            session.add_all([recording, job])
            session.flush()
            item = ProcessingJobItem(
                job_id=job.id,
                call_id=call.id,
                operator_id=operator.id,
                recording_id=recording.id,
                idempotency_key=f"worker-discovery-item-{uuid4()}",
                status=ItemStatus.PROCESSING,
                stage="transcribing",
            )
            session.add(item)
            session.flush()
            recording_id = recording.id
            job_id = job.id
            item_id = item.id
            session.commit()

        database_url = os.environ["DATABASE_URL"]
        async_engine = create_async_engine(database_url, poolclass=sa.pool.NullPool)
        sessions = async_sessionmaker(async_engine, expire_on_commit=False)
        worker_locked = asyncio.Event()
        discovery_started = asyncio.Event()
        worker_task: asyncio.Task[None] | None = None
        discovery_task: asyncio.Task[None] | None = None

        async def worker_transaction() -> None:
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                item = await session.get(ProcessingJobItem, item_id)
                job = await session.get(ProcessingJob, job_id)
                assert item is not None and job is not None
                await _lock_processable_item(session, item, job)
                worker_locked.set()
                await asyncio.wait_for(discovery_started.wait(), timeout=10)
                # With the historical Job-first worker order, rediscovery owns
                # Recording here and waits for Job while this update waits for
                # Recording, producing a PostgreSQL deadlock.
                await asyncio.sleep(0.1)
                recording = await session.get(Recording, recording_id)
                assert recording is not None
                recording.status = RecordingStatus.INSPECTED
                await session.commit()

        async def discovery_transaction() -> None:
            await asyncio.wait_for(worker_locked.wait(), timeout=10)
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                discovery_started.set()
                await _lock_recording_rows(session, {recording_id})
                locked_job = await session.scalar(
                    select(ProcessingJob).where(ProcessingJob.id == job_id).with_for_update()
                )
                assert locked_job is not None
                await session.commit()

        try:
            worker_task = asyncio.create_task(worker_transaction())
            discovery_task = asyncio.create_task(discovery_transaction())
            await asyncio.wait_for(
                asyncio.gather(worker_task, discovery_task),
                timeout=10,
            )
            async with sessions() as session:
                await session.execute(text(f'SET search_path TO "{schema}"'))
                recording = await session.get(Recording, recording_id)
                assert recording is not None
                assert recording.status == RecordingStatus.INSPECTED
        finally:
            pending_tasks = [
                task
                for task in (worker_task, discovery_task)
                if task is not None and not task.done()
            ]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await async_engine.dispose()


@POSTGRES_MIGRATION_TEST
@pytest.mark.asyncio
async def test_checksum_reuse_snapshot_survives_source_retention_commit() -> None:
    from app.workers.pipeline import _clone_transcript, _completed_duplicate_snapshot

    with _committed_postgres_schema() as (engine, schema):
        now = datetime.now(UTC)
        checksum = "d" * 64
        pipeline_config_hash = "e" * 64
        with Session(engine) as session:
            session.execute(text(f'SET search_path TO "{schema}"'))
            source_call = Call(
                yeastar_uid=f"reuse-source-call-{uuid4()}",
                started_at=now,
                direction=Direction.INBOUND,
            )
            destination_call = Call(
                yeastar_uid=f"reuse-destination-call-{uuid4()}",
                started_at=now,
                direction=Direction.OUTBOUND,
            )
            session.add_all([source_call, destination_call])
            session.flush()
            source_recording = Recording(
                call_id=source_call.id,
                yeastar_recording_id=f"reuse-source-recording-{uuid4()}",
                status=RecordingStatus.COMPLETED,
            )
            destination_recording = Recording(
                call_id=destination_call.id,
                yeastar_recording_id=f"reuse-destination-recording-{uuid4()}",
                status=RecordingStatus.INSPECTED,
                duration_seconds=30,
            )
            session.add_all([source_recording, destination_recording])
            session.flush()
            source = Transcript(
                call_id=source_call.id,
                recording_id=source_recording.id,
                operator_id=None,
                idempotency_key=f"reuse-source-transcript-{uuid4()}",
                status=TranscriptStatus.COMPLETED,
                model="gpt-4o-transcribe",
                language="el",
                prompt_version="prompt-hash",
                completed_at=now - timedelta(days=31),
                source_audio_sha256=checksum,
                is_diarized=False,
                transcription_mode=TranscriptionMode.DUAL_CHANNEL,
                speaker_attribution_status=SpeakerAttributionStatus.CHANNEL_UNKNOWN,
                pipeline_version="pipeline-v2",
                pipeline_config_hash=pipeline_config_hash,
                preprocessing_profile="topology-v2-pcm16k-mono-tracks-v1",
                quality_summary={"attribution_warning": True},
                is_current=True,
                original_text="Channel A then Channel B",
                normalized_text="channel a then channel b",
            )
            session.add(source)
            session.flush()
            session.add_all(
                [
                    TranscriptSegment(
                        transcript_id=source.id,
                        call_id=source_call.id,
                        operator_id=None,
                        speaker_label="Channel A",
                        speaker_source=SpeakerSource.STEREO_CHANNEL,
                        start_seconds=Decimal("0.000"),
                        end_seconds=Decimal("2.500"),
                        original_text="Channel A",
                        normalized_text="channel a",
                        confidence=Decimal("0.90000"),
                        transcription_model=source.model,
                        sequence_number=1,
                        channel_index=0,
                        track_id="channel-0",
                        chunk_index=0,
                        mean_logprob=Decimal("-0.10000"),
                        low_logprob_ratio=Decimal("0.05000"),
                        quality_flags=["overlap-preserved"],
                        audio_variant="topology-channel-0",
                    ),
                    TranscriptSegment(
                        transcript_id=source.id,
                        call_id=source_call.id,
                        operator_id=None,
                        speaker_label="Channel B",
                        speaker_source=SpeakerSource.STEREO_CHANNEL,
                        start_seconds=Decimal("1.500"),
                        end_seconds=Decimal("3.000"),
                        original_text="Channel B",
                        normalized_text="channel b",
                        confidence=Decimal("0.80000"),
                        transcription_model=source.model,
                        sequence_number=2,
                        channel_index=1,
                        track_id="channel-1",
                        chunk_index=0,
                        mean_logprob=Decimal("-0.20000"),
                        low_logprob_ratio=Decimal("0.10000"),
                        quality_flags=[],
                        audio_variant="topology-channel-1",
                    ),
                ]
            )
            source_recording_id = source_recording.id
            source_transcript_id = source.id
            destination_call_id = destination_call.id
            destination_recording_id = destination_recording.id
            session.commit()

        database_url = os.environ["DATABASE_URL"]
        async_engine = create_async_engine(database_url, poolclass=sa.pool.NullPool)
        sessions = async_sessionmaker(async_engine, expire_on_commit=False)
        try:
            async with sessions() as clone_session:
                await clone_session.execute(text(f'SET search_path TO "{schema}"'))
                await _lock_recording_rows(clone_session, {destination_recording_id})
                destination_call = await clone_session.get(Call, destination_call_id)
                destination_recording = await clone_session.get(
                    Recording,
                    destination_recording_id,
                )
                assert destination_call is not None
                assert destination_recording is not None
                snapshot = await _completed_duplicate_snapshot(
                    clone_session,
                    checksum=checksum,
                    model="gpt-4o-transcribe",
                    diarized=False,
                    operator_id=None,
                    language="el",
                    prompt_version="prompt-hash",
                    pipeline_version="pipeline-v2",
                    pipeline_config_hash=pipeline_config_hash,
                    preprocessing_profile="topology-v2-pcm16k-mono-tracks-v1",
                    transcription_mode=TranscriptionMode.DUAL_CHANNEL,
                )
                assert snapshot is not None
                source_snapshot, segment_snapshot = snapshot
                assert source_snapshot.id == source_transcript_id
                assert [segment.sequence_number for segment in segment_snapshot] == [1, 2]

                async with sessions() as retention_session:
                    await retention_session.execute(text(f'SET search_path TO "{schema}"'))
                    await _lock_recording_rows(retention_session, {source_recording_id})
                    retained_source = await retention_session.scalar(
                        select(Transcript)
                        .where(Transcript.id == source_transcript_id)
                        .with_for_update()
                    )
                    assert retained_source is not None
                    await retention_session.delete(retained_source)
                    await retention_session.commit()

                assert (
                    await clone_session.scalar(
                        select(Transcript.id).where(Transcript.id == source_transcript_id)
                    )
                    is None
                )
                clone = await _clone_transcript(
                    clone_session,
                    source_snapshot,
                    segment_snapshot,
                    call=destination_call,
                    recording=destination_recording,
                    operator=None,
                    call_leg_id=None,
                    key=f"reuse-clone-{uuid4()}",
                )
                clone_id = clone.id
                await clone_session.commit()

            with Session(engine) as session:
                session.execute(text(f'SET search_path TO "{schema}"'))
                assert session.get(Transcript, source_transcript_id) is None
                clone = session.get(Transcript, clone_id)
                assert clone is not None
                assert clone.status == TranscriptStatus.COMPLETED
                assert clone.original_text == "Channel A then Channel B"
                assert clone.transcription_mode == TranscriptionMode.DUAL_CHANNEL
                cloned_segments = session.scalars(
                    select(TranscriptSegment)
                    .where(TranscriptSegment.transcript_id == clone_id)
                    .order_by(TranscriptSegment.sequence_number)
                ).all()
                assert [
                    (
                        segment.sequence_number,
                        segment.channel_index,
                        segment.track_id,
                        segment.chunk_index,
                        segment.speaker_label,
                        segment.speaker_source,
                        segment.start_seconds,
                        segment.end_seconds,
                        segment.quality_flags,
                        segment.audio_variant,
                    )
                    for segment in cloned_segments
                ] == [
                    (
                        1,
                        0,
                        "channel-0",
                        0,
                        "Channel A",
                        SpeakerSource.STEREO_CHANNEL,
                        Decimal("0.000"),
                        Decimal("2.500"),
                        ["overlap-preserved"],
                        "topology-channel-0",
                    ),
                    (
                        2,
                        1,
                        "channel-1",
                        0,
                        "Channel B",
                        SpeakerSource.STEREO_CHANNEL,
                        Decimal("1.500"),
                        Decimal("3.000"),
                        [],
                        "topology-channel-1",
                    ),
                ]
        finally:
            await async_engine.dispose()


def _insert_legacy_transcripts(connection: sa.Connection) -> dict[str, UUID]:
    now = datetime.now(UTC)
    operator_id = uuid4()
    user_id = uuid4()
    job_id = uuid4()
    connection.execute(
        text(
            """
            INSERT INTO operators (
                id, yeastar_extension_id, extension_number, display_name,
                provider_active, enabled, last_synced_at, created_at, updated_at
            ) VALUES (
                :id, :extension_id, :extension_number, 'Migration Operator',
                true, true, :now, :now, :now
            )
            """
        ),
        {
            "id": operator_id,
            "extension_id": f"extension-{uuid4()}",
            "extension_number": uuid4().hex,
            "now": now,
        },
    )
    connection.execute(
        text(
            """
            INSERT INTO users (
                id, email, password_hash, is_active, created_at, updated_at
            ) VALUES (
                :id, :email, 'not-used-by-this-test', true, :now, :now
            )
            """
        ),
        {
            "id": user_id,
            "email": f"migration-history-{uuid4()}@example.test",
            "now": now,
        },
    )
    connection.execute(
        text(
            """
            INSERT INTO processing_jobs (
                id, idempotency_key, requested_by_id, status, date_from, date_to,
                include_all_speakers, selected_operator_ids, selected_category_ids,
                request_filters, progress_percent, current_stage, calls_found,
                recordings_found, calls_completed, calls_failed,
                cancellation_requested, attempt_count, completed_at,
                created_at, updated_at
            ) VALUES (
                :id, :idempotency_key, :user_id, 'COMPLETED', :date_from, :date_to,
                true, '[]'::json, '[]'::json, '{}'::json, 100, 'Complete', 2,
                2, 2, 0, false, 1, :now, :now, :now
            )
            """
        ),
        {
            "id": job_id,
            "idempotency_key": f"migration-history-{uuid4()}",
            "user_id": user_id,
            "date_from": now - timedelta(hours=1),
            "date_to": now + timedelta(hours=1),
            "now": now,
        },
    )

    recordings: list[tuple[UUID, UUID]] = []
    for position in range(3):
        call_id = uuid4()
        recording_id = uuid4()
        connection.execute(
            text(
                """
                INSERT INTO calls (
                    id, yeastar_uid, started_at, direction, duration_seconds,
                    has_recording, was_transferred, processing_status,
                    provider_payload, created_at, updated_at
                ) VALUES (
                    :id, :uid, :now, 'INBOUND', 60, true, false, 'completed',
                    '{}'::json, :now, :now
                )
                """
            ),
            {"id": call_id, "uid": f"migration-call-{uuid4()}", "now": now},
        )
        connection.execute(
            text(
                """
                INSERT INTO recordings (
                    id, call_id, yeastar_recording_id, status, provider_payload,
                    created_at, updated_at
                ) VALUES (
                    :id, :call_id, :provider_id, 'INSPECTED', '{}'::json, :now, :now
                )
                """
            ),
            {
                "id": recording_id,
                "call_id": call_id,
                "provider_id": f"migration-recording-{position}-{uuid4()}",
                "now": now,
            },
        )
        recordings.append((call_id, recording_id))

    ids = {
        "attributed_old": uuid4(),
        "attributed_new": uuid4(),
        "attributed_failed": uuid4(),
        "diarized_old": uuid4(),
        "diarized_new": uuid4(),
        "channel_unknown": uuid4(),
        "attributed_item": uuid4(),
        "diarized_item": uuid4(),
    }
    rows = [
        (
            ids["attributed_old"],
            *recordings[0],
            operator_id,
            "COMPLETED",
            False,
            now - timedelta(days=3),
            now - timedelta(days=4),
        ),
        (
            ids["attributed_new"],
            *recordings[0],
            operator_id,
            "COMPLETED",
            False,
            now - timedelta(days=2),
            now - timedelta(days=3),
        ),
        (
            ids["attributed_failed"],
            *recordings[0],
            operator_id,
            "FAILED",
            False,
            None,
            now - timedelta(days=1),
        ),
        (
            ids["diarized_old"],
            *recordings[1],
            None,
            "COMPLETED",
            True,
            now - timedelta(days=3),
            now - timedelta(days=4),
        ),
        (
            ids["diarized_new"],
            *recordings[1],
            None,
            "COMPLETED",
            True,
            now - timedelta(days=1),
            now - timedelta(days=2),
        ),
        (
            ids["channel_unknown"],
            *recordings[2],
            None,
            "COMPLETED",
            False,
            now,
            now - timedelta(days=1),
        ),
    ]
    for (
        transcript_id,
        call_id,
        recording_id,
        row_operator_id,
        status,
        is_diarized,
        completed_at,
        created_at,
    ) in rows:
        connection.execute(
            text(
                """
                INSERT INTO transcripts (
                    id, call_id, recording_id, operator_id, idempotency_key,
                    status, model, language, attempt_count, completed_at,
                    source_audio_sha256, is_diarized, created_at, updated_at
                ) VALUES (
                    :id, :call_id, :recording_id, :operator_id, :idempotency_key,
                    :status, 'legacy-model', 'el', 1, :completed_at,
                    :checksum, :is_diarized, :created_at, :created_at
                )
                """
            ),
            {
                "id": transcript_id,
                "call_id": call_id,
                "recording_id": recording_id,
                "operator_id": row_operator_id,
                "idempotency_key": f"legacy-{transcript_id}",
                "status": status,
                "completed_at": completed_at,
                "checksum": transcript_id.hex * 2,
                "is_diarized": is_diarized,
                "created_at": created_at,
            },
        )
    for item_id, (call_id, recording_id) in (
        (ids["attributed_item"], recordings[0]),
        (ids["diarized_item"], recordings[1]),
    ):
        connection.execute(
            text(
                """
                INSERT INTO processing_job_items (
                    id, job_id, call_id, operator_id, recording_id,
                    idempotency_key, status, stage, attempt_count,
                    completed_at, created_at, updated_at
                ) VALUES (
                    :id, :job_id, :call_id, :operator_id, :recording_id,
                    :idempotency_key, 'COMPLETED', 'completed', 1,
                    :now, :now, :now
                )
                """
            ),
            {
                "id": item_id,
                "job_id": job_id,
                "call_id": call_id,
                "operator_id": operator_id,
                "recording_id": recording_id,
                "idempotency_key": f"migration-history-item-{item_id}",
                "now": now,
            },
        )
    return ids
