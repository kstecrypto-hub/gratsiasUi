"""add Yeastar integration metadata and provider identifiers

Revision ID: 8d4a7c2e1f90
Revises: c1b0957b4d6c
Create Date: 2026-07-15 16:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "8d4a7c2e1f90"
down_revision: Union[str, None] = "c1b0957b4d6c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("operators", sa.Column("mobile_number", sa.String(length=64)))
    op.add_column("operators", sa.Column("presence_status", sa.String(length=64)))
    op.add_column(
        "operators",
        sa.Column(
            "provider_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )
    op.create_index("ix_operators_provider_active", "operators", ["provider_active"])

    op.add_column("call_legs", sa.Column("transaction_id", sa.String(length=255)))
    op.add_column("call_legs", sa.Column("yeastar_cdr_id", sa.String(length=255)))
    op.add_column("call_legs", sa.Column("provider_leg", sa.String(length=128)))
    op.add_column("call_legs", sa.Column("call_from", sa.String(length=255)))
    op.add_column("call_legs", sa.Column("call_to", sa.String(length=255)))
    op.add_column(
        "call_legs",
        sa.Column("ring_duration_seconds", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "call_legs",
        sa.Column("hold_duration_seconds", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("call_legs", sa.Column("call_type", sa.String(length=64)))
    op.add_column(
        "call_legs",
        sa.Column("event_list", sa.JSON(), nullable=False, server_default=sa.text("'[]'")),
    )
    op.create_index("ix_call_legs_transaction_id", "call_legs", ["transaction_id"])
    op.create_index("ix_call_legs_yeastar_cdr_id", "call_legs", ["yeastar_cdr_id"])

    op.add_column("recordings", sa.Column("yeastar_uid", sa.String(length=255)))
    op.add_column("recordings", sa.Column("provider_time", sa.DateTime(timezone=True)))
    op.add_column("recordings", sa.Column("call_from", sa.String(length=255)))
    op.add_column("recordings", sa.Column("call_to", sa.String(length=255)))
    op.add_column("recordings", sa.Column("call_from_number", sa.String(length=128)))
    op.add_column("recordings", sa.Column("call_to_number", sa.String(length=128)))
    op.add_column("recordings", sa.Column("call_type", sa.String(length=64)))
    op.add_column("recordings", sa.Column("archive_status", sa.String(length=64)))
    op.add_column(
        "recordings",
        sa.Column("provider_payload", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
    )
    op.create_index("ix_recordings_yeastar_uid", "recordings", ["yeastar_uid"])

    op.create_table(
        "integration_status",
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "NOT_CONFIGURED",
                "NOT_TESTED",
                "CONNECTED",
                "AUTH_REJECTED",
                "TOKEN_REFRESH_FAILED",
                "IP_NOT_ALLOWED",
                "IP_BLOCKED",
                "API_DISABLED",
                "PERMISSION_DENIED",
                "UNSUPPORTED_API_VERSION",
                "UNSUPPORTED_FIRMWARE",
                "NETWORK_UNAVAILABLE",
                "TEMPORARILY_UNAVAILABLE",
                name="yeastar_connection_status",
                native_enum=False,
                create_constraint=True,
            ),
            nullable=False,
        ),
        sa.Column("configuration_fingerprint", sa.String(length=64)),
        sa.Column("configured_date_format", sa.String(length=128)),
        sa.Column("device_name", sa.String(length=255)),
        sa.Column("model_name", sa.String(length=255)),
        sa.Column("firmware_version", sa.String(length=128)),
        sa.Column("system_time", sa.String(length=128)),
        sa.Column("system_date_format", sa.String(length=128)),
        sa.Column("system_time_format", sa.String(length=128)),
        sa.Column("provider_timestamp", sa.BigInteger()),
        sa.Column("capabilities_json", sa.JSON(), nullable=False, server_default=sa.text("'{}'")),
        sa.Column("last_tested_at", sa.DateTime(timezone=True)),
        sa.Column("last_successful_connection_at", sa.DateTime(timezone=True)),
        sa.Column("last_error_category", sa.String(length=128)),
        sa.Column("last_error_reference", sa.String(length=128)),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_integration_status")),
        sa.UniqueConstraint("provider", name=op.f("uq_integration_status_provider")),
    )
    op.create_index("ix_integration_status_status", "integration_status", ["status"])

    op.drop_constraint(
        op.f("ck_processing_jobs_processing_job_status"),
        "processing_jobs",
        type_="check",
    )
    op.create_check_constraint(
        "processing_job_status",
        "processing_jobs",
        "status IN ('QUEUED', 'WAITING_FOR_CONNECTION', 'CONNECTING', "
        "'FETCHING_CALLS', 'FETCHING_CALL_DETAILS', 'FINDING_RECORDINGS', "
        "'DOWNLOADING_RECORDINGS', 'INSPECTING_AUDIO', 'EXTRACTING_OPERATOR_AUDIO', "
        "'TRANSCRIBING', 'SEARCHING_KEYWORDS', 'COMPLETED', "
        "'COMPLETED_WITH_ERRORS', 'FAILED', 'CANCELLED')",
    )
    op.drop_constraint(
        op.f("ck_processing_job_items_processing_job_item_status"),
        "processing_job_items",
        type_="check",
    )
    op.alter_column(
        "processing_job_items",
        "status",
        existing_type=sa.String(length=10),
        type_=sa.String(length=22),
        existing_nullable=False,
    )
    op.create_check_constraint(
        "processing_job_item_status",
        "processing_job_items",
        "status IN ('QUEUED', 'WAITING_FOR_CONNECTION', 'PROCESSING', 'COMPLETED', "
        "'FAILED', 'CANCELLED', 'SKIPPED')",
    )


def downgrade() -> None:
    op.drop_constraint(
        op.f("ck_processing_job_items_processing_job_item_status"),
        "processing_job_items",
        type_="check",
    )
    op.execute(
        sa.text(
            "UPDATE processing_job_items SET status = 'QUEUED' "
            "WHERE status = 'WAITING_FOR_CONNECTION'"
        )
    )
    op.alter_column(
        "processing_job_items",
        "status",
        existing_type=sa.String(length=22),
        type_=sa.String(length=10),
        existing_nullable=False,
    )
    op.create_check_constraint(
        "processing_job_item_status",
        "processing_job_items",
        "status IN ('QUEUED', 'PROCESSING', 'COMPLETED', 'FAILED', 'CANCELLED', 'SKIPPED')",
    )
    op.drop_constraint(
        op.f("ck_processing_jobs_processing_job_status"),
        "processing_jobs",
        type_="check",
    )
    op.execute(
        sa.text(
            "UPDATE processing_jobs SET status = 'QUEUED' "
            "WHERE status = 'WAITING_FOR_CONNECTION'"
        )
    )
    op.create_check_constraint(
        "processing_job_status",
        "processing_jobs",
        "status IN ('QUEUED', 'CONNECTING', 'FETCHING_CALLS', 'FETCHING_CALL_DETAILS', "
        "'FINDING_RECORDINGS', 'DOWNLOADING_RECORDINGS', 'INSPECTING_AUDIO', "
        "'EXTRACTING_OPERATOR_AUDIO', 'TRANSCRIBING', 'SEARCHING_KEYWORDS', "
        "'COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED', 'CANCELLED')",
    )

    op.drop_index("ix_integration_status_status", table_name="integration_status")
    op.drop_table("integration_status")

    op.drop_index("ix_recordings_yeastar_uid", table_name="recordings")
    op.drop_column("recordings", "provider_payload")
    op.drop_column("recordings", "archive_status")
    op.drop_column("recordings", "call_type")
    op.drop_column("recordings", "call_to_number")
    op.drop_column("recordings", "call_from_number")
    op.drop_column("recordings", "call_to")
    op.drop_column("recordings", "call_from")
    op.drop_column("recordings", "provider_time")
    op.drop_column("recordings", "yeastar_uid")

    op.drop_index("ix_call_legs_yeastar_cdr_id", table_name="call_legs")
    op.drop_index("ix_call_legs_transaction_id", table_name="call_legs")
    op.drop_column("call_legs", "event_list")
    op.drop_column("call_legs", "call_type")
    op.drop_column("call_legs", "hold_duration_seconds")
    op.drop_column("call_legs", "ring_duration_seconds")
    op.drop_column("call_legs", "provider_leg")
    op.drop_column("call_legs", "call_to")
    op.drop_column("call_legs", "call_from")
    op.drop_column("call_legs", "yeastar_cdr_id")
    op.drop_column("call_legs", "transaction_id")

    op.drop_index("ix_operators_provider_active", table_name="operators")
    op.drop_column("operators", "provider_active")
    op.drop_column("operators", "presence_status")
    op.drop_column("operators", "mobile_number")
