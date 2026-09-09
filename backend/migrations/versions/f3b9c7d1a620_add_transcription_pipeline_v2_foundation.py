"""add transcription pipeline v2 persistence foundation

Revision ID: f3b9c7d1a620
Revises: e6a2c9f4b810
Create Date: 2026-07-22 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "f3b9c7d1a620"
down_revision: Union[str, None] = "e6a2c9f4b810"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TRANSCRIPTION_MODES = (
    "LEGACY",
    "OPERATOR_CHANNEL",
    "DUAL_CHANNEL",
    "MONO_DIARIZATION",
)
_SPEAKER_ATTRIBUTION_STATUSES = (
    "CONFIRMED_BY_PBX",
    "CALLER_CALLEE_ONLY",
    "CHANNEL_UNKNOWN",
    "ANONYMOUS_DIARIZATION",
    "MANUALLY_ASSIGNED",
)
_LEGACY_DIARIZED_PIPELINE_CONFIG_HASH = (
    "ef5ef358c56c2900297a8233a323a2b295faf08dd8c98dc621ae133560e25e61"
)
_LEGACY_ISOLATED_PIPELINE_CONFIG_HASH = (
    "ad088e9537186db18c69c6781b79ae9e020192cc158790a4ab7e498a94bff2a5"
)


def _enum_values(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def upgrade() -> None:
    op.add_column(
        "processing_job_items",
        sa.Column("requested_pipeline_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "processing_job_items",
        sa.Column("result_transcript_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_processing_job_items_result_transcript_id_transcripts"),
        "processing_job_items",
        "transcripts",
        ["result_transcript_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_processing_job_items_result_transcript_id"),
        "processing_job_items",
        ["result_transcript_id"],
        unique=False,
    )

    op.add_column(
        "transcripts",
        sa.Column(
            "transcription_mode",
            sa.Enum(
                *_TRANSCRIPTION_MODES,
                name="transcription_mode",
                native_enum=False,
                create_constraint=False,
            ),
            server_default=sa.text("'LEGACY'"),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "transcription_mode",
        "transcripts",
        f"transcription_mode IN ({_enum_values(_TRANSCRIPTION_MODES)})",
    )
    op.add_column(
        "transcripts",
        sa.Column(
            "speaker_attribution_status",
            sa.Enum(
                *_SPEAKER_ATTRIBUTION_STATUSES,
                name="speaker_attribution_status",
                native_enum=False,
                create_constraint=False,
            ),
            nullable=True,
        ),
    )
    op.create_check_constraint(
        "speaker_attribution_status",
        "transcripts",
        f"speaker_attribution_status IN ({_enum_values(_SPEAKER_ATTRIBUTION_STATUSES)})",
    )
    op.add_column(
        "transcripts",
        sa.Column(
            "pipeline_version",
            sa.String(length=64),
            server_default=sa.text("'legacy-v1'"),
            nullable=True,
        ),
    )
    op.add_column(
        "transcripts",
        sa.Column("pipeline_config_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "transcripts",
        sa.Column("preprocessing_profile", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "transcripts",
        sa.Column("quality_summary", sa.JSON(), nullable=True),
    )
    op.add_column(
        "transcripts",
        sa.Column("supersedes_transcript_id", sa.Uuid(), nullable=True),
    )
    op.create_foreign_key(
        op.f("fk_transcripts_supersedes_transcript_id_transcripts"),
        "transcripts",
        "transcripts",
        ["supersedes_transcript_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        op.f("ix_transcripts_supersedes_transcript_id"),
        "transcripts",
        ["supersedes_transcript_id"],
        unique=False,
    )
    op.create_check_constraint(
        "transcript_not_self_superseding",
        "transcripts",
        "supersedes_transcript_id IS NULL OR supersedes_transcript_id <> id",
    )
    # Start false so pre-existing duplicates cannot violate the partial indexes
    # before the deterministic ranking below has selected one logical current row.
    op.add_column(
        "transcripts",
        sa.Column(
            "is_current",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )

    op.add_column(
        "transcript_segments",
        sa.Column("channel_index", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        "transcript_segments",
        sa.Column("track_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "transcript_segments",
        sa.Column("chunk_index", sa.Integer(), nullable=True),
    )
    op.add_column(
        "transcript_segments",
        sa.Column("mean_logprob", sa.Numeric(precision=8, scale=5), nullable=True),
    )
    op.add_column(
        "transcript_segments",
        sa.Column("low_logprob_ratio", sa.Numeric(precision=6, scale=5), nullable=True),
    )
    op.add_column(
        "transcript_segments",
        sa.Column(
            "quality_flags",
            sa.JSON(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
    )
    op.add_column(
        "transcript_segments",
        sa.Column("audio_variant", sa.String(length=128), nullable=True),
    )
    op.create_check_constraint(
        "segment_channel_index_nonnegative",
        "transcript_segments",
        "channel_index IS NULL OR channel_index >= 0",
    )
    op.create_check_constraint(
        "segment_chunk_index_nonnegative",
        "transcript_segments",
        "chunk_index IS NULL OR chunk_index >= 0",
    )
    op.create_check_constraint(
        "segment_low_logprob_ratio_range",
        "transcript_segments",
        "low_logprob_ratio IS NULL OR (low_logprob_ratio >= 0 AND low_logprob_ratio <= 1)",
    )

    op.create_table(
        "transcription_attempts",
        sa.Column("transcript_id", sa.Uuid(), nullable=False),
        sa.Column("track_id", sa.String(length=128), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Numeric(precision=12, scale=3), nullable=False),
        sa.Column("end_seconds", sa.Numeric(precision=12, scale=3), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("audio_variant", sa.String(length=128), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("mean_logprob", sa.Numeric(precision=8, scale=5), nullable=True),
        sa.Column("low_logprob_ratio", sa.Numeric(precision=6, scale=5), nullable=True),
        sa.Column("selected", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("api_usage", sa.JSON(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "end_seconds >= start_seconds",
            name=op.f("ck_transcription_attempts_attempt_time_order"),
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name=op.f("ck_transcription_attempts_attempt_chunk_index_nonnegative"),
        ),
        sa.CheckConstraint(
            "low_logprob_ratio IS NULL OR (low_logprob_ratio >= 0 AND low_logprob_ratio <= 1)",
            name=op.f("ck_transcription_attempts_attempt_low_logprob_ratio_range"),
        ),
        sa.ForeignKeyConstraint(
            ["transcript_id"],
            ["transcripts.id"],
            name=op.f("fk_transcription_attempts_transcript_id_transcripts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transcription_attempts")),
    )
    op.create_index(
        "ix_transcription_attempts_transcript_track_chunk",
        "transcription_attempts",
        ["transcript_id", "track_id", "chunk_index"],
        unique=False,
    )

    # Enum columns persist member names in this repository.  Rank completed
    # rows ahead of failed/in-progress rows, then use stable tie-breakers so a
    # legacy duplicate set always chooses the same current transcript.
    op.execute(
        sa.text(
            """
            UPDATE transcripts
            SET transcription_mode = 'LEGACY',
                speaker_attribution_status = CASE
                    WHEN is_diarized THEN 'ANONYMOUS_DIARIZATION'
                    WHEN operator_id IS NOT NULL THEN 'CONFIRMED_BY_PBX'
                    ELSE 'CHANNEL_UNKNOWN'
                END,
                pipeline_version = 'legacy-v1',
                pipeline_config_hash = CASE
                    WHEN is_diarized THEN :legacy_diarized_pipeline_config_hash
                    ELSE :legacy_isolated_pipeline_config_hash
                END,
                preprocessing_profile = 'legacy-current',
                is_current = false
            """
        ).bindparams(
            legacy_diarized_pipeline_config_hash=(_LEGACY_DIARIZED_PIPELINE_CONFIG_HASH),
            legacy_isolated_pipeline_config_hash=(_LEGACY_ISOLATED_PIPELINE_CONFIG_HASH),
        )
    )
    op.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY recording_id, operator_id
                           ORDER BY
                               CASE WHEN status = 'COMPLETED' THEN 0 ELSE 1 END,
                               completed_at DESC NULLS LAST,
                               created_at DESC,
                               id DESC
                       ) AS position
                FROM transcripts
            )
            UPDATE transcripts AS transcript
            SET is_current = true
            FROM ranked
            WHERE transcript.id = ranked.id
              AND ranked.position = 1
            """
        )
    )
    # Preserve historical result provenance before reprocessing can create a
    # replacement.  Exact operator transcripts win; shared unattributed
    # transcripts are the deterministic fallback for all-speaker job items.
    op.execute(
        sa.text(
            """
            UPDATE processing_job_items AS item
            SET result_transcript_id = (
                SELECT transcript.id
                FROM transcripts AS transcript
                WHERE transcript.call_id = item.call_id
                  AND transcript.recording_id = item.recording_id
                  AND (
                      transcript.operator_id = item.operator_id
                      OR transcript.operator_id IS NULL
                  )
                  AND transcript.status = 'COMPLETED'
                  AND transcript.is_current IS TRUE
                ORDER BY
                    CASE
                        WHEN transcript.operator_id = item.operator_id THEN 0
                        ELSE 1
                    END,
                    transcript.completed_at DESC NULLS LAST,
                    transcript.created_at DESC,
                    transcript.id DESC
                LIMIT 1
            )
            WHERE item.status = 'COMPLETED'
              AND item.recording_id IS NOT NULL
            """
        )
    )
    op.create_index(
        "uq_transcripts_current_attributed",
        "transcripts",
        ["recording_id", "operator_id"],
        unique=True,
        postgresql_where=sa.text("is_current IS TRUE AND operator_id IS NOT NULL"),
        sqlite_where=sa.text("is_current = 1 AND operator_id IS NOT NULL"),
    )
    op.create_index(
        "uq_transcripts_current_unattributed",
        "transcripts",
        ["recording_id"],
        unique=True,
        postgresql_where=sa.text("is_current IS TRUE AND operator_id IS NULL"),
        sqlite_where=sa.text("is_current = 1 AND operator_id IS NULL"),
    )
    op.alter_column(
        "transcripts",
        "is_current",
        existing_type=sa.Boolean(),
        server_default=sa.text("true"),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.drop_index("uq_transcripts_current_unattributed", table_name="transcripts")
    op.drop_index("uq_transcripts_current_attributed", table_name="transcripts")

    op.drop_index(
        "ix_transcription_attempts_transcript_track_chunk",
        table_name="transcription_attempts",
    )
    op.drop_table("transcription_attempts")

    op.drop_constraint(
        op.f("ck_transcript_segments_segment_low_logprob_ratio_range"),
        "transcript_segments",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_transcript_segments_segment_chunk_index_nonnegative"),
        "transcript_segments",
        type_="check",
    )
    op.drop_constraint(
        op.f("ck_transcript_segments_segment_channel_index_nonnegative"),
        "transcript_segments",
        type_="check",
    )
    op.drop_column("transcript_segments", "audio_variant")
    op.drop_column("transcript_segments", "quality_flags")
    op.drop_column("transcript_segments", "low_logprob_ratio")
    op.drop_column("transcript_segments", "mean_logprob")
    op.drop_column("transcript_segments", "chunk_index")
    op.drop_column("transcript_segments", "track_id")
    op.drop_column("transcript_segments", "channel_index")

    op.drop_column("transcripts", "is_current")
    op.drop_constraint(
        op.f("ck_transcripts_transcript_not_self_superseding"),
        "transcripts",
        type_="check",
    )
    op.drop_index(
        op.f("ix_transcripts_supersedes_transcript_id"),
        table_name="transcripts",
    )
    op.drop_constraint(
        op.f("fk_transcripts_supersedes_transcript_id_transcripts"),
        "transcripts",
        type_="foreignkey",
    )
    op.drop_column("transcripts", "supersedes_transcript_id")
    op.drop_column("transcripts", "quality_summary")
    op.drop_column("transcripts", "preprocessing_profile")
    op.drop_column("transcripts", "pipeline_config_hash")
    op.drop_column("transcripts", "pipeline_version")
    op.drop_constraint(
        op.f("ck_transcripts_speaker_attribution_status"),
        "transcripts",
        type_="check",
    )
    op.drop_column("transcripts", "speaker_attribution_status")
    op.drop_constraint(
        op.f("ck_transcripts_transcription_mode"),
        "transcripts",
        type_="check",
    )
    op.drop_column("transcripts", "transcription_mode")

    op.drop_index(
        op.f("ix_processing_job_items_result_transcript_id"),
        table_name="processing_job_items",
    )
    op.drop_constraint(
        op.f("fk_processing_job_items_result_transcript_id_transcripts"),
        "processing_job_items",
        type_="foreignkey",
    )
    op.drop_column("processing_job_items", "result_transcript_id")
    op.drop_column("processing_job_items", "requested_pipeline_version")
