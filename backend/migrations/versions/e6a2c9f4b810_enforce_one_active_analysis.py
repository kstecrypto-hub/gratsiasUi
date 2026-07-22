"""enforce one active analysis

Revision ID: e6a2c9f4b810
Revises: d4f8a6b2e917
Create Date: 2026-07-20 16:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "e6a2c9f4b810"
down_revision: Union[str, None] = "d4f8a6b2e917"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TERMINAL = "'COMPLETED', 'COMPLETED_WITH_ERRORS', 'FAILED', 'CANCELLED'"


def upgrade() -> None:
    # Older releases allowed several queued/running jobs.  Preserve every row,
    # keep the newest active job, and close older work before adding the hard
    # database invariant.  Stale task deliveries are ignored by the worker.
    op.execute(
        sa.text(
            f"""
            UPDATE processing_jobs
            SET status = 'CANCELLED',
                cancellation_requested = true,
                current_stage = 'Superseded by single-analysis mode',
                completed_at = COALESCE(completed_at, CURRENT_TIMESTAMP)
            WHERE status NOT IN ({_TERMINAL})
              AND id <> (
                  SELECT id
                  FROM processing_jobs
                  WHERE status NOT IN ({_TERMINAL})
                  ORDER BY created_at DESC, id DESC
                  LIMIT 1
              )
            """
        )
    )
    op.create_index(
        "uq_processing_jobs_one_active",
        "processing_jobs",
        [sa.text("(1)")],
        unique=True,
        postgresql_where=sa.text(f"status NOT IN ({_TERMINAL})"),
        sqlite_where=sa.text(f"status NOT IN ({_TERMINAL})"),
    )


def downgrade() -> None:
    op.drop_index("uq_processing_jobs_one_active", table_name="processing_jobs")
