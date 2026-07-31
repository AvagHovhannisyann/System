"""Bitemporal fact tables: security anchor, securities master, price bars.

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-31

Implements DECISIONS.md D-011 (P2.2): every fact table carries
``valid_from``/``valid_to`` (half-open event-time interval, ``'infinity'``
for open-ended), ``knowledge_time`` (writer-supplied, **no default**),
``ingested_at`` (audit only, server default ``now()``), and
``is_retraction``. Primary keys are (logical key, valid_from,
knowledge_time) so corrections are new rows and equal-knowledge-time
duplicates are impossible. Hypertable conversion and the as-of indices
follow in revision 0003 (P2.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _bitemporal_columns() -> list[sa.Column[Any]]:
    """Return the five D-011 mixin columns, freshly constructed.

    ``knowledge_time`` deliberately has no default of any kind: writers must
    supply it explicitly (D-011). ``ingested_at`` is audit-only.
    """
    return [
        sa.Column("valid_from", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "valid_to",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("'infinity'::timestamptz"),
        ),
        sa.Column("knowledge_time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "ingested_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "is_retraction",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    ]


def upgrade() -> None:
    """Create security (identity anchor), security_master, price_bar."""
    op.create_table(
        "security",
        sa.Column("security_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.PrimaryKeyConstraint("security_id", name="pk_security"),
    )
    op.create_table(
        "security_master",
        sa.Column("security_id", sa.BigInteger(), nullable=False),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("exchange", sa.Text(), nullable=False),
        sa.Column("first_listed_on", sa.Date(), nullable=True),
        sa.Column("delisted_on", sa.Date(), nullable=True),
        *_bitemporal_columns(),
        sa.PrimaryKeyConstraint(
            "security_id", "valid_from", "knowledge_time", name="pk_security_master"
        ),
        sa.ForeignKeyConstraint(
            ["security_id"],
            ["security.security_id"],
            name="fk_security_master_security_id_security",
        ),
        sa.CheckConstraint("valid_from < valid_to", name="ck_security_master_valid_interval"),
    )
    op.create_table(
        "price_bar",
        sa.Column("security_id", sa.BigInteger(), nullable=False),
        # Prices in USD per share; volume in shares; adjustment_factor
        # dimensionless (close_raw_usd * adjustment_factor == close_usd).
        sa.Column("open_usd", sa.Numeric(20, 6), nullable=False),
        sa.Column("high_usd", sa.Numeric(20, 6), nullable=False),
        sa.Column("low_usd", sa.Numeric(20, 6), nullable=False),
        sa.Column("close_usd", sa.Numeric(20, 6), nullable=False),
        sa.Column("volume_shares", sa.BigInteger(), nullable=False),
        sa.Column("close_raw_usd", sa.Numeric(20, 6), nullable=False),
        sa.Column("adjustment_factor", sa.Numeric(20, 10), nullable=False),
        *_bitemporal_columns(),
        sa.PrimaryKeyConstraint("security_id", "valid_from", "knowledge_time", name="pk_price_bar"),
        sa.ForeignKeyConstraint(
            ["security_id"],
            ["security.security_id"],
            name="fk_price_bar_security_id_security",
        ),
        sa.CheckConstraint("valid_from < valid_to", name="ck_price_bar_valid_interval"),
    )


def downgrade() -> None:
    """Drop the fact tables in dependency order."""
    op.drop_table("price_bar")
    op.drop_table("security_master")
    op.drop_table("security")
