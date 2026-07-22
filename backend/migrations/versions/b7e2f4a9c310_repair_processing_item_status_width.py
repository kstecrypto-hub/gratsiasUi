"""repair processing job item status width

Revision ID: b7e2f4a9c310
Revises: 8d4a7c2e1f90
Create Date: 2026-07-15 17:30:00.000000

Some development deployments applied the preceding revision before its
``WAITING_FOR_CONNECTION`` column-width correction was added.  A new forward
revision is required because Alembic correctly does not rerun a revision that
is already stamped as applied.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b7e2f4a9c310"
down_revision: Union[str, None] = "8d4a7c2e1f90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "processing_job_items",
        "status",
        existing_type=sa.String(length=10),
        type_=sa.String(length=22),
        existing_nullable=False,
    )


def downgrade() -> None:
    # Revision 8d4a7c2e1f90 itself now defines VARCHAR(22).  This repair is a
    # compatibility bridge for databases stamped by its earlier draft, so
    # returning to that revision must preserve its corrected schema.
    pass
