"""create voice_note table

Revision ID: 0001
Revises:
Create Date: 2026-07-25
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "voice_note",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("transcription", sa.Text(), nullable=True),
        sa.Column("transcription_error", sa.Text(), nullable=True),
        sa.Column(
            "transcription_retries",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column("transcription_last_retry", sa.DateTime(), nullable=True),
        sa.Column("vikunja_id", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("voice_note")
