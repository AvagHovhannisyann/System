"""Ingestion-run tracking table (P3.1).

Revision ID: 0005
Revises: 0004
Create Date: 2026-08-01

Operational metadata about our own pipeline: one row per connector execution,
recording source, kind (``backfill``/``live``), lifecycle status, start/end,
rows written, resume position before and after, error detail, and the emitted
data-quality metrics.

**Deliberately not a bitemporal fact table**, and therefore deliberately
without the append-only triggers revisions 0003/0004 install on the fact
tables and the identity anchor. A run legitimately transitions ``running`` ->
``succeeded``/``failed``, which is a state change of an operational record and
not a later correction to a historical belief; append-only triggers would make
a run impossible to close. The full reasoning — including why inventing a
``knowledge_time`` for "we started a job" would itself violate invariant I3 —
is in the model docstring (``backend/ingest/runs.py``).

The CHECK constraints encode the lifecycle rather than leaving it to
application code: ``run_kind`` and ``status`` are closed vocabularies,
``ended_at`` is set exactly when the run is no longer ``running``, an ended run
cannot end before it started, and ``rows_written`` cannot be negative.

Downgrade drops the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ingestion_run table with its lifecycle constraints and index."""
    op.create_table(
        "ingestion_run",
        sa.Column("run_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("run_kind", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("ended_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("rows_written", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column("checkpoint_before", postgresql.JSONB(), nullable=True),
        sa.Column("checkpoint_after", postgresql.JSONB(), nullable=True),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column(
            "quality_metrics",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.PrimaryKeyConstraint("run_id", name="pk_ingestion_run"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so passing an
        # already-prefixed name would produce ck_ingestion_run_ck_ingestion_run_...
        # and diverge from the names the ORM model declares.
        sa.CheckConstraint("run_kind IN ('backfill', 'live')", name="run_kind"),
        sa.CheckConstraint("status IN ('running', 'succeeded', 'failed')", name="status"),
        sa.CheckConstraint("rows_written >= 0", name="rows_written_non_negative"),
        sa.CheckConstraint("ended_at IS NULL OR ended_at >= started_at", name="end_after_start"),
        sa.CheckConstraint("(status = 'running') = (ended_at IS NULL)", name="running_iff_open"),
    )
    op.execute(
        "CREATE INDEX ix_ingestion_run_source_started ON ingestion_run (source, started_at DESC)"
    )


def downgrade() -> None:
    """Drop the ingestion_run table (its index goes with it)."""
    op.drop_table("ingestion_run")
