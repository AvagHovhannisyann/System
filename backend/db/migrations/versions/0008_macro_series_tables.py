"""FRED macro series and observation fact tables (P3.8, D-011).

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-01

Two bitemporal fact tables in the same D-011 shape as revisions 0002/0003/0006:
the five mixin columns, a primary key of (logical key, ``valid_from``,
``knowledge_time``) so a revision is a new row, a ``valid_from < valid_to``
CHECK, the composite as-of index in exactly the ``DISTINCT ON``/``ORDER BY``
order the query layer emits, and a ``BEFORE UPDATE OR DELETE`` row trigger
making them append-only in the database.

**Neither table is a hypertable**, and that is a deliberate departure from
``price_bar``/``edgar_filing`` rather than an omission. Macro data is small —
tens of thousands of observations per series across its whole vintage history,
for a series list measured in tens — while its ``valid_from`` axis (the
observation date) stretches back decades. One-month chunks over that span would
create on the order of a thousand near-empty chunks whose planning overhead
exceeds any pruning benefit, and the access pattern is a point or short-range
lookup on ``(series_id, observation_date)`` rather than a wide event-time scan.
This is the same reasoning that left ``edgar_filing_document`` unpartitioned in
revision 0006. Revisit if the series list grows by orders of magnitude.

Two CHECKs beyond the shared interval constraint, both encoding invariants the
connector also enforces, so a future writer cannot quietly break them:

- ``ck_macro_observation_missing_value`` — ``value IS NULL`` exactly when
  ``is_missing``. FRED's missing marker is the string ``"."``; this makes it
  structurally impossible for it to land as ``0`` or as a real number.
- ``ck_macro_observation_valid_from_matches_date`` — ``valid_from`` is
  ``observation_date`` at 00:00 UTC. The two columns are deliberately
  redundant (one is the calendar period label, one is the event-time
  coordinate the as-of layer indexes), and this keeps the redundancy from
  becoming a divergence.

Constraint names here are given **bare** (``missing_value``, not
``ck_macro_observation_missing_value``) because ``Base.metadata``'s naming
convention is ``ck_%(table_name)s_%(constraint_name)s`` and Alembic applies it
to these definitions. Passing an already-prefixed name yields
``ck_macro_observation_ck_macro_observation_...``, which PostgreSQL then
truncates to 63 characters with a hash suffix — an unreadable, unpredictable
name that no error message or test can match on. Revision 0006 pre-dates this
observation and carries the doubled form; its names happen to fit under 63
characters, so they are merely ugly rather than broken.

Downgrade drops the triggers and both tables. The trigger function is shared
with revisions 0003/0004/0006 and is left alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACT_TABLES = ("macro_series", "macro_observation")


def _bitemporal_columns() -> list[sa.Column[Any]]:
    """Return the five D-011 mixin columns, freshly constructed.

    Identical to revisions 0002/0006's helper — repeated rather than imported
    so a later edit to those revisions cannot retroactively change this one's
    DDL. ``knowledge_time`` has no default of any kind: writers supply it.
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
        sa.Column("is_retraction", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    ]


def upgrade() -> None:
    """Create the macro fact tables, their as-of indices and append-only triggers."""
    op.create_table(
        "macro_series",
        sa.Column("series_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("frequency", sa.Text(), nullable=False),
        sa.Column("frequency_short", sa.Text(), nullable=False),
        # The authoritative unit of every value of this series (directive §8).
        sa.Column("units", sa.Text(), nullable=False),
        sa.Column("units_short", sa.Text(), nullable=False),
        sa.Column("seasonal_adjustment_short", sa.Text(), nullable=False),
        sa.Column("observation_start", sa.Date(), nullable=False),
        sa.Column("observation_end", sa.Date(), nullable=False),
        # FRED real-time period as stated at ingestion. vintage_start_date is the
        # raw date knowledge_time was derived from; it is never a knowledge time.
        sa.Column("vintage_start_date", sa.Date(), nullable=False),
        sa.Column("vintage_end_date", sa.Date(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        *_bitemporal_columns(),
        sa.PrimaryKeyConstraint(
            "series_id", "valid_from", "knowledge_time", name="pk_macro_series"
        ),
        sa.CheckConstraint("valid_from < valid_to", name="valid_interval"),
        sa.CheckConstraint(
            "vintage_end_date IS NULL OR vintage_end_date >= vintage_start_date",
            name="vintage_interval",
        ),
    )
    op.create_table(
        "macro_observation",
        sa.Column("series_id", sa.Text(), nullable=False),
        sa.Column("observation_date", sa.Date(), nullable=False),
        # Arbitrary precision on purpose: a declared scale would silently round
        # source values. The unit lives on macro_series.units, never here.
        sa.Column("value", sa.Numeric(), nullable=True),
        sa.Column("is_missing", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("vintage_start_date", sa.Date(), nullable=False),
        sa.Column("vintage_end_date", sa.Date(), nullable=True),
        *_bitemporal_columns(),
        sa.PrimaryKeyConstraint(
            "series_id",
            "observation_date",
            "valid_from",
            "knowledge_time",
            name="pk_macro_observation",
        ),
        sa.CheckConstraint("valid_from < valid_to", name="valid_interval"),
        # FRED's "." marker must never become 0 or NaN: NULL exactly when missing.
        sa.CheckConstraint(
            "(is_missing AND value IS NULL) OR (NOT is_missing AND value IS NOT NULL)",
            name="missing_value",
        ),
        sa.CheckConstraint(
            "valid_from = timezone('UTC', observation_date::timestamp)",
            name="valid_from_matches_date",
        ),
        sa.CheckConstraint(
            "vintage_end_date IS NULL OR vintage_end_date >= vintage_start_date",
            name="vintage_interval",
        ),
    )
    op.execute(
        "CREATE INDEX ix_macro_series_asof_lookup "
        "ON macro_series (series_id, valid_from, knowledge_time DESC)"
    )
    op.execute(
        "CREATE INDEX ix_macro_observation_asof_lookup "
        "ON macro_observation (series_id, observation_date, valid_from, knowledge_time DESC)"
    )
    # P3.9 coverage/gap/staleness reporting scans by vintage date per source.
    op.execute(
        "CREATE INDEX ix_macro_observation_vintage_start ON macro_observation (vintage_start_date)"
    )
    for table in _FACT_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table}_append_only "
            f"BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION bitemporal_append_only_guard()"
        )


def downgrade() -> None:
    """Drop the append-only triggers and both macro fact tables."""
    for table in _FACT_TABLES:
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.drop_table("macro_observation")
    op.drop_table("macro_series")
