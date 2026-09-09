from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
from uuid import UUID

import pytest
import sqlalchemy as sa

from app.database.base import Base
from app.models import Call, Recording, Transcript, TranscriptSegment
from app.models.enums import SpeakerAttributionStatus, TranscriptionMode
from app.services.audio import AudioInfo
from app.services.audio.quality import LegacyPassThroughQualityProcessor
from app.services.transcription.confidence import UnavailableConfidenceAnalyzer
from app.services.transcription.planning import LegacyAudioPlanner
from app.services.transcription.prompt import LegacyVocabularyPromptBuilder
from app.services.transcription.types import AudioTrack
from app.workers.pipeline import (
    _legacy_runtime_pipeline_config_hash,
    transcript_idempotency_key,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = BACKEND_ROOT.parent
APP_ROOT = BACKEND_ROOT / "app"
HANDOFF_PATH = REPOSITORY_ROOT / "docs" / "transcription-pipeline-v2-handoff.md"
HARDENING_MIGRATION_PATH = (
    BACKEND_ROOT
    / "migrations"
    / "versions"
    / "a91d4e7c2b30_enforce_transcript_supersession_integrity.py"
)

REQUIRED_V2_MODULES = (
    "app.models.entities",
    "app.models.enums",
    "app.services.audio.quality",
    "app.services.audio.segmentation",
    "app.services.transcription.client",
    "app.services.transcription.confidence",
    "app.services.transcription.merge",
    "app.services.transcription.orchestrator",
    "app.services.transcription.planning",
    "app.services.transcription.prompt",
    "app.services.transcription.types",
    "app.workers.pipeline",
    "app.api.results",
)

LOW_LEVEL_MODULE_PATHS = (
    Path("services/audio/quality.py"),
    Path("services/audio/segmentation.py"),
    Path("services/transcription/client.py"),
    Path("services/transcription/confidence.py"),
    Path("services/transcription/merge.py"),
    Path("services/transcription/orchestrator.py"),
    Path("services/transcription/planning.py"),
    Path("services/transcription/prompt.py"),
    Path("services/transcription/types.py"),
)

HANDOFF_HEADINGS = (
    "## 1. Current completed foundation",
    "## 2. Database contract",
    "## 3. Orchestrator contract",
    "## 4. Current legacy adapters",
    "## 5. Invariants later phases must preserve",
    "## 6. Subsequent implementation sequence and module map",
    "## 7. Speaker identity and security prohibition",
    "## 8. Known technical debt",
    "## 9. Rollback considerations",
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _called_attributes(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func)
            if name:
                names.add(name)
    return names


def _function_node(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} was not found.")


def _compact(value: str) -> str:
    return "".join(value.split())


def _foreign_key_for_column(table: sa.Table, column_name: str) -> sa.ForeignKeyConstraint:
    for constraint in table.foreign_key_constraints:
        if [column.name for column in constraint.columns] == [column_name]:
            return constraint
    raise AssertionError(f"No foreign key found for {table.name}.{column_name}.")


def _normalized_sql(value: object) -> str:
    return " ".join(str(value).lower().split())


@pytest.mark.parametrize("module_name", REQUIRED_V2_MODULES)
def test_required_foundation_modules_import_without_cycles(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_pipeline_v2_schema_and_index_manifest() -> None:
    transcript = Base.metadata.tables["transcripts"]
    segment = Base.metadata.tables["transcript_segments"]
    attempt = Base.metadata.tables["transcription_attempts"]
    item = Base.metadata.tables["processing_job_items"]
    job = Base.metadata.tables["processing_jobs"]

    assert [mode.value for mode in TranscriptionMode] == [
        "legacy",
        "operator_channel",
        "dual_channel",
        "mono_diarization",
    ]
    assert [status.value for status in SpeakerAttributionStatus] == [
        "confirmed_by_pbx",
        "caller_callee_only",
        "channel_unknown",
        "anonymous_diarization",
        "manually_assigned",
    ]

    assert {
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
    } <= set(transcript.c.keys())
    assert transcript.c.prompt_template_version.type.length == 64
    assert transcript.c.prompt_template_version.nullable is True
    assert transcript.c.vocabulary_hash.type.length == 64
    assert transcript.c.vocabulary_hash.nullable is True
    assert transcript.c.is_current.nullable is False
    assert (
        _foreign_key_for_column(
            transcript,
            "supersedes_transcript_id",
        ).ondelete
        == "SET NULL"
    )
    assert {
        constraint.name
        for constraint in transcript.constraints
        if isinstance(constraint, sa.CheckConstraint)
    } >= {"ck_transcripts_transcript_not_self_superseding"}

    transcript_indexes = {index.name: index for index in transcript.indexes}
    attributed = transcript_indexes["uq_transcripts_current_attributed"]
    unattributed = transcript_indexes["uq_transcripts_current_unattributed"]
    assert attributed.unique is True
    assert [column.name for column in attributed.columns] == [
        "recording_id",
        "operator_id",
    ]
    assert (
        _normalized_sql(attributed.dialect_options["postgresql"]["where"])
        == "is_current is true and operator_id is not null"
    )
    assert (
        _normalized_sql(attributed.dialect_options["sqlite"]["where"])
        == "is_current = 1 and operator_id is not null"
    )
    assert unattributed.unique is True
    assert [column.name for column in unattributed.columns] == ["recording_id"]
    assert (
        _normalized_sql(unattributed.dialect_options["postgresql"]["where"])
        == "is_current is true and operator_id is null"
    )
    assert (
        _normalized_sql(unattributed.dialect_options["sqlite"]["where"])
        == "is_current = 1 and operator_id is null"
    )

    assert {
        "channel_index",
        "track_id",
        "chunk_index",
        "mean_logprob",
        "low_logprob_ratio",
        "quality_flags",
        "audio_variant",
    } <= set(segment.c.keys())

    attempt_columns = set(attempt.c.keys())
    assert {
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
    } <= attempt_columns
    assert {"prompt", "api_key", "secret", "credentials"}.isdisjoint(attempt_columns)
    assert _foreign_key_for_column(attempt, "transcript_id").ondelete == "CASCADE"
    assert {index.name for index in attempt.indexes} == {
        "ix_transcription_attempts_transcript_track_chunk",
        "uq_transcription_attempts_selected_chunk",
    }

    assert {"requested_pipeline_version", "result_transcript_id"} <= set(item.c.keys())
    assert _foreign_key_for_column(item, "result_transcript_id").ondelete == "SET NULL"

    job_indexes = {index.name: index for index in job.indexes}
    one_active = job_indexes["uq_processing_jobs_one_active"]
    assert one_active.unique is True
    active_predicate = _normalized_sql(one_active.dialect_options["postgresql"]["where"])
    for final_state in ("completed", "completed_with_errors", "failed", "cancelled"):
        assert final_state in active_predicate


def test_deletion_and_supersession_integrity_guards_are_present() -> None:
    assert Transcript.segments.property.passive_deletes is True
    assert Transcript.attempts.property.passive_deletes is True
    assert TranscriptSegment.matches.property.passive_deletes is True
    assert "delete-orphan" in TranscriptSegment.matches.property.cascade
    assert Call.transcripts.property.passive_deletes == "all"
    assert Recording.transcripts.property.passive_deletes == "all"

    migration_tree = _parse(HARDENING_MIGRATION_PATH)
    assignments = {
        node.target.id: ast.literal_eval(node.value)
        for node in migration_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"revision", "down_revision"}
        and node.value is not None
    }
    migration_source = HARDENING_MIGRATION_PATH.read_text(encoding="utf-8")
    assert assignments == {
        "revision": "a91d4e7c2b30",
        "down_revision": "f3b9c7d1a620",
    }
    for required_sql in (
        "WITH RECURSIVE lineage",
        "FROM recordings AS recording",
        "FOR UPDATE",
        "trg_transcripts_supersession_integrity",
        "transcript supersession would create a cycle",
        "transcript supersession crosses a logical target boundary",
        "transcript update crosses an immutable logical target boundary",
    ):
        assert required_sql in migration_source


def test_versioned_idempotency_identity_manifest() -> None:
    base = {
        "recording_id": UUID("00000000-0000-0000-0000-000000000001"),
        "operator_id": UUID("00000000-0000-0000-0000-000000000101"),
        "model": "gpt-4o-transcribe",
        "checksum": "a" * 64,
        "diarized": False,
        "language": "el",
        "prompt_version": "prompt-hash",
        "pipeline_version": "legacy-v1",
        "pipeline_config_hash": "b" * 64,
        "transcription_mode": TranscriptionMode.LEGACY,
        "prompt_template_version": "legacy-isolated-vocabulary-v1",
        "vocabulary_hash": "vocabulary-hash",
        "supersedes_transcript_id": UUID("00000000-0000-0000-0000-000000000201"),
    }
    identity = transcript_idempotency_key(**base)

    assert identity == transcript_idempotency_key(**base)
    assert len(identity) == 64
    assert set(identity) <= set("0123456789abcdef")

    variants = {
        "recording_id": UUID("00000000-0000-0000-0000-000000000002"),
        "operator_id": UUID("00000000-0000-0000-0000-000000000102"),
        "model": "gpt-4o-transcribe-next",
        "checksum": "c" * 64,
        "diarized": True,
        "language": "en",
        "prompt_version": "other-prompt-hash",
        "pipeline_version": "pipeline-v2",
        "pipeline_config_hash": "d" * 64,
        "transcription_mode": TranscriptionMode.DUAL_CHANNEL,
        "prompt_template_version": "contextual-prompt-v2",
        "vocabulary_hash": "other-vocabulary-hash",
        "supersedes_transcript_id": UUID("00000000-0000-0000-0000-000000000202"),
    }
    for field, value in variants.items():
        changed = dict(base)
        changed[field] = value
        assert transcript_idempotency_key(**changed) != identity, field


def test_runtime_configuration_identity_covers_effective_legacy_behavior() -> None:
    base = {
        "diarized": False,
        "channel_index": 0,
        "max_upload_bytes": 25 * 1024 * 1024,
        "prompt_template_version": "legacy-isolated-vocabulary-v1",
    }
    identity = _legacy_runtime_pipeline_config_hash(**base)
    assert identity == _legacy_runtime_pipeline_config_hash(**base)
    for field, value in {
        "channel_index": 1,
        "max_upload_bytes": 24 * 1024 * 1024,
        "prompt_template_version": "legacy-template-v2",
        "isolated_chunk_seconds": 14,
        "isolated_response_format": "verbose_json",
    }.items():
        changed = dict(base)
        changed[field] = value
        assert _legacy_runtime_pipeline_config_hash(**changed) != identity, field


@pytest.mark.parametrize("relative_path", LOW_LEVEL_MODULE_PATHS)
def test_low_level_import_boundary_is_orm_and_database_free(
    relative_path: Path,
) -> None:
    tree = _parse(APP_ROOT / relative_path)
    imports = _imported_modules(tree)

    assert not {
        module
        for module in imports
        if module == "sqlalchemy"
        or module.startswith("sqlalchemy.")
        or module == "app.models"
        or module.startswith("app.models.")
        or module == "app.database"
        or module.startswith("app.database.")
    }
    assert not any(
        isinstance(node, ast.Name) and node.id == "NotImplementedError" for node in ast.walk(tree)
    )


def test_openai_and_transcription_call_boundaries_are_explicit() -> None:
    openai_importers: set[Path] = set()
    provider_create_callers: set[Path] = set()
    transcription_callers: set[Path] = set()
    connection_test_callers: set[Path] = set()

    for path in APP_ROOT.rglob("*.py"):
        relative = path.relative_to(APP_ROOT)
        tree = _parse(path)
        if any(
            module == "openai" or module.startswith("openai.") for module in _imported_modules(tree)
        ):
            openai_importers.add(relative)
        calls = _called_attributes(tree)
        if any(name.endswith(".audio.transcriptions.create") for name in calls):
            provider_create_callers.add(relative)
        if any(
            name.endswith(".transcribe_isolated") or name.endswith(".transcribe_diarized")
            for name in calls
        ):
            transcription_callers.add(relative)
        if any(name.endswith(".test_connection") for name in calls):
            connection_test_callers.add(relative)

    client_path = Path("services/transcription/client.py")
    orchestrator_path = Path("services/transcription/orchestrator.py")
    worker_calls = _called_attributes(_parse(APP_ROOT / "workers" / "pipeline.py"))
    assert openai_importers == {client_path}
    assert provider_create_callers == {client_path}
    assert transcription_callers == {orchestrator_path}
    assert connection_test_callers == {
        Path("api/health.py"),
        Path("api/settings.py"),
    }
    assert "TranscriptionOrchestrator" in worker_calls
    assert any(name.endswith(".transcribe") for name in worker_calls)


@pytest.mark.asyncio
async def test_legacy_adapter_defaults_remain_pass_through(tmp_path: Path) -> None:
    source = tmp_path / "prepared.wav"
    audio_info = AudioInfo(
        codec_name="pcm_s16le",
        format_name="wav",
        duration_seconds=30,
        channel_count=1,
        sample_rate_hz=16_000,
        bit_rate_bps=256_000,
        size_bytes=44,
        sha256_checksum="a" * 64,
    )
    planner = LegacyAudioPlanner()

    isolated = planner.plan(
        source_path=source,
        audio_info=audio_info,
        diarized=False,
        channel_index=1,
        operator_id="operator-1",
        attribution_status="confirmed_by_pbx",
        audio_variant="legacy-operator-channel",
    )
    diarized = planner.plan(
        source_path=source,
        audio_info=audio_info,
        diarized=True,
        channel_index=None,
        operator_id=None,
        attribution_status="anonymous_diarization",
        audio_variant="legacy-mono",
    )

    assert isolated.mode == diarized.mode == "legacy"
    assert len(isolated.tracks) == len(diarized.tracks) == 1
    assert isolated.tracks[0].diarized is False
    assert diarized.tracks[0].diarized is True

    track = AudioTrack(track_id="legacy", source_path=source, diarized=False)
    assert (
        await LegacyPassThroughQualityProcessor().prepare(
            track,
            audio_info=audio_info,
        )
        is track
    )

    confidence = UnavailableConfidenceAnalyzer().analyze(())
    assert confidence.status == "unavailable"
    assert confidence.mean_logprob is None
    assert confidence.low_logprob_ratio is None

    prompt = LegacyVocabularyPromptBuilder().build(["  Alpha   Beta ", "alpha beta"])
    expected = "Greek business vocabulary and names: Alpha Beta"
    assert prompt.text == expected
    assert prompt.version == hashlib.sha256(expected.encode("utf-8")).hexdigest()[:16]


def test_reprocess_activation_and_result_binding_source_invariants() -> None:
    results_tree = _parse(APP_ROOT / "api" / "results.py")
    pipeline_tree = _parse(APP_ROOT / "workers" / "pipeline.py")
    reprocess = _compact(ast.unparse(_function_node(results_tree, "reprocess_call")))
    reprocess_lock = _compact(
        ast.unparse(_function_node(results_tree, "_lock_current_reprocess_transcript"))
    )
    process_item = _compact(ast.unparse(_function_node(pipeline_tree, "process_item")))
    activation = _compact(
        ast.unparse(_function_node(pipeline_tree, "_activate_transcript_replacement"))
    )
    results_source = _compact((APP_ROOT / "api" / "results.py").read_text(encoding="utf-8"))

    assert "/calls/{call_id}/reprocess" in reprocess
    assert "_lock_current_reprocess_transcript(" in reprocess
    assert "Transcript.status==TranscriptStatus.COMPLETED" in reprocess_lock
    assert "Transcript.is_current.is_(True)" in reprocess_lock
    assert ".with_for_update()" in reprocess_lock
    assert reprocess_lock.index("select(Recording.id)") < reprocess_lock.index(
        "statement.with_for_update()"
    )
    assert reprocess.index("_lock_current_reprocess_transcript(") < reprocess.index(
        "select(Call).where(Call.id==call_id).with_for_update()"
    )
    assert "_reprocess_targets" in reprocess
    assert "call.reprocess" in reprocess
    assert "process_job_item.delay" in reprocess
    assert "is_current=False" not in reprocess

    assert "ProcessingJobItem.result_transcript_id==current_transcript.id" in results_source
    assert "_activate_transcript_replacement(" in process_item
    assert "item.result_transcript_id=transcript.id" in process_item
    assert process_item.index("_activate_transcript_replacement(") < process_item.index(
        "item.result_transcript_id=transcript.id"
    )
    assert (
        "awaitsession.commit()"
        in process_item[process_item.index("item.result_transcript_id=transcript.id") :]
    )

    assert "previous.is_current=False" in activation
    assert "_validate_supersession_lineage(" in activation
    assert "replacement.supersedes_transcript_id=previous.id" in activation
    assert "replacement.is_current=True" in activation
    assert activation.count("awaitsession.flush()") >= 2


def test_worker_redelivery_and_lease_boundaries_are_explicit() -> None:
    pipeline_tree = _parse(APP_ROOT / "workers" / "pipeline.py")
    tasks_tree = _parse(APP_ROOT / "workers" / "tasks.py")
    discovery = _compact(ast.unparse(_function_node(pipeline_tree, "discover_job_items")))
    locked_discovery = _compact(
        ast.unparse(_function_node(pipeline_tree, "_discover_job_items_locked"))
    )
    discovery_commit = _compact(
        ast.unparse(_function_node(pipeline_tree, "_commit_discovery_progress"))
    )
    retention_guard = _compact(
        ast.unparse(_function_node(pipeline_tree, "_active_processing_recording_ids"))
    )
    retention = _compact(ast.unparse(_function_node(pipeline_tree, "cleanup_retention_records")))
    item_lock = _compact(ast.unparse(_function_node(pipeline_tree, "_lock_processable_item")))
    stale_settlement = _compact(ast.unparse(_function_node(pipeline_tree, "_settle_stale_item")))
    item_stage = _compact(ast.unparse(_function_node(pipeline_tree, "_item_stage")))
    process_item = _compact(ast.unparse(_function_node(pipeline_tree, "process_item")))
    analysis_task = _compact(ast.unparse(_function_node(tasks_tree, "process_analysis_job")))

    assert "raiseProcessingBusyError" in discovery
    assert "_refresh_worker_lease(lock)" in discovery
    assert "_discover_job_items_locked(job_id,lease_check=lease_check)" in discovery
    assert "progress_check=lease_check" in locked_discovery
    assert "before_commit=discovery_heartbeat" in locked_discovery
    assert "_commit_discovery_progress(session,run,lease_check)" in locked_discovery
    assert "_lock_recording_rows(" in locked_discovery
    assert "exceptProcessingBusyError" in locked_discovery
    assert discovery_commit.index("awaitlease_check()") < discovery_commit.index(
        "awaitsession.commit()"
    )
    assert "requested_pipeline_version" not in retention_guard
    assert "ProcessingJobItem.status.not_in(FINAL_ITEM_STATES)" in retention_guard
    assert "ProcessingJob.status.not_in(FINAL_JOB_STATES)" in retention_guard
    assert "_lock_recording_rows(session,maintenance_recording_ids)" in retention
    assert "_lock_recording_job_rows(session,maintenance_recording_ids)" in retention
    assert (
        "Recording.last_error_category=='retention_cleanup_pending',"
        "Recording.id.not_in(active_processing_recording_ids)"
    ) in retention
    assert item_lock.index("_lock_recording_rows(") < item_lock.index(
        "session.refresh(job,with_for_update=True)"
    )
    assert stale_settlement.index("_lock_recording_rows(") < stale_settlement.index(
        "select(ProcessingJob)"
    )
    assert item_stage.count("ifbefore_commitisnotNone:awaitbefore_commit()") == 2
    assert "exceptProcessingBusyError" in analysis_task
    assert "_refresh_processing_leases(lock,transcription_slot)" in process_item
    assert process_item.count("before_commit=lease_check") >= 6
    assert process_item.count("_refresh_available_processing_leases(lock,transcription_slot)") >= 3
    first_refresh = process_item.index("_refresh_processing_leases(lock,transcription_slot)")
    final_refresh = process_item.rindex("awaitlease_check()")
    persistence = process_item.index("_persist_transcription_result(")
    assert first_refresh < persistence < final_refresh
    assert "ifawaitcancellation_check()" in process_item
    assert "item.status==ItemStatus.PROCESSING" in process_item
    assert "raiseProcessingBusyError" in process_item
    audio_cleanup = process_item.index("cleanup_path=audio.safe_storage_path")
    assert process_item.rfind("_lock_recording_rows(", 0, audio_cleanup) >= 0
    assert (
        process_item.rfind("_lock_recording_job_rows(session,{recording.id})", 0, audio_cleanup)
        >= 0
    )
    assert "ProcessingJobItem.status.not_in(FINAL_ITEM_STATES)" in process_item
    activation = process_item.index("_activate_transcript_replacement(")
    completion = process_item.index("recording.status=RecordingStatus.COMPLETED", activation)
    assert "awaitlease_check()" in process_item[activation:completion]


def test_handoff_contract_has_required_order_modules_and_prohibitions() -> None:
    handoff = HANDOFF_PATH.read_text(encoding="utf-8")
    lower = handoff.lower()

    heading_positions = [handoff.index(heading) for heading in HANDOFF_HEADINGS]
    assert heading_positions == sorted(heading_positions)

    sequence_start = handoff.index(HANDOFF_HEADINGS[5])
    sequence_end = handoff.index(HANDOFF_HEADINGS[6])
    sequence = handoff[sequence_start:sequence_end]
    sequence_labels = (
        "1. **Topology**",
        "2. **Segmentation**",
        "3. **Prompts**",
        "4. **Confidence**",
        "5. **Mono refinement**",
        "6. **Manual assignment**",
        "7. **Evaluation and activation**",
    )
    sequence_positions = [sequence.index(label) for label in sequence_labels]
    assert sequence_positions == sorted(sequence_positions)

    for required_path in (
        "backend/app/services/transcription/types.py",
        "backend/app/services/transcription/planning.py",
        "backend/app/services/audio/segmentation.py",
        "backend/app/services/transcription/prompt.py",
        "backend/app/services/transcription/confidence.py",
        "backend/app/services/transcription/merge.py",
        "backend/app/services/transcription/orchestrator.py",
        "backend/app/services/transcription/client.py",
        "backend/app/workers/pipeline.py",
        "backend/app/api/results.py",
        "backend/app/schemas/results.py",
        "backend/app/api/settings.py",
    ):
        assert required_path in handoff

    for prohibition in (
        "no operator voice samples",
        "no voiceprints",
        "no speaker recognition",
        "no speaker biometrics",
    ):
        assert prohibition in lower
