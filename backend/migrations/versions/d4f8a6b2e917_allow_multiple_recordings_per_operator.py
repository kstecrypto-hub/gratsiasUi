"""allow one analysis item per recording for an operator

Revision ID: d4f8a6b2e917
Revises: b7e2f4a9c310
Create Date: 2026-07-20 12:00:00.000000
"""

from typing import Sequence, Union

from alembic import op


revision: str = "d4f8a6b2e917"
down_revision: Union[str, None] = "b7e2f4a9c310"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "uq_job_item_call_operator",
        "processing_job_items",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_job_item_call_operator_recording",
        "processing_job_items",
        ["job_id", "call_id", "operator_id", "recording_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_job_item_call_operator_recording",
        "processing_job_items",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_job_item_call_operator",
        "processing_job_items",
        ["job_id", "call_id", "operator_id"],
    )
