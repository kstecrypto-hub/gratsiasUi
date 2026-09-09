"""enforce one selected transcription attempt

Revision ID: e74f9b2c6d10
Revises: d62c8f3a1b40
Create Date: 2026-07-30 15:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "e74f9b2c6d10"
down_revision: Union[str, None] = "d62c8f3a1b40"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "uq_transcription_attempts_selected_chunk",
        "transcription_attempts",
        ["transcript_id", "track_id", "chunk_index"],
        unique=True,
        postgresql_where=sa.text("selected IS TRUE"),
        sqlite_where=sa.text("selected = 1"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_transcription_attempts_selected_chunk",
        table_name="transcription_attempts",
    )
