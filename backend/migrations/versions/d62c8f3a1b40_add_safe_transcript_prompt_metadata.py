"""add safe transcript prompt metadata

Revision ID: d62c8f3a1b40
Revises: a91d4e7c2b30
Create Date: 2026-07-30 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "d62c8f3a1b40"
down_revision: Union[str, None] = "a91d4e7c2b30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "transcripts",
        sa.Column("prompt_template_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "transcripts",
        sa.Column("vocabulary_hash", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("transcripts", "vocabulary_hash")
    op.drop_column("transcripts", "prompt_template_version")
