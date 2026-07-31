"""TimescaleDB hypertable, as-of indices, append-only triggers (P2.5, D-011).

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-31

Physical layout per DECISIONS.md D-011:

- ``price_bar`` becomes a hypertable partitioned on ``valid_from`` (the
  event-time axis): research queries always bound event time, so chunk
  pruning bites on every query, while ``knowledge_time <= as_of`` predicates
  are half-unbounded and prune poorly. **Chunk interval 1 month**: bars are
  daily (one row per security per day), so monthly chunks hold enough rows to
  amortize per-chunk overhead while research reads, which typically span
  months of lookback, touch few chunks. The primary key
  ``(security_id, valid_from, knowledge_time)`` already contains the
  partition column, satisfying ``create_hypertable``'s requirement that every
  unique constraint include it — this is why the PK was designed composite
  in revision 0002.
- ``security_master`` stays a plain table (small, not time-series bulk).
- Composite as-of index per fact table ``(entity, valid_from,
  knowledge_time DESC)`` — exactly the ``DISTINCT ON``/``ORDER BY`` shape of
  the versioned read in ``backend.db.asof``.
- **Append-only enforced in the database**: BEFORE UPDATE OR DELETE row
  triggers raise on both fact tables. Triggers rather than revoked grants
  because grants do not bind superusers or future roles, while a trigger is
  role-independent; FOR EACH ROW rather than FOR EACH STATEMENT because
  Timescale copies row triggers onto every chunk, so even direct-chunk
  mutations are rejected. TRUNCATE is deliberately not blocked: it is the
  sanctioned admin/test reset path and never masquerades as a correction.

Downgrade drops triggers, function, and indices. The hypertable conversion
itself is not reversed (Timescale has no supported un-convert); dropping the
table in revision 0002's downgrade works on hypertables, so the chain stays
walkable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACT_TABLES = ("security_master", "price_bar")


def upgrade() -> None:
    """Convert price_bar to a hypertable; add as-of indices and append-only triggers."""
    # Hypertable first so the indices below are created on (and propagated to)
    # chunks by Timescale. Chunk interval: 1 month — daily bars,
    # month-spanning research reads (see module docstring).
    op.execute(
        "SELECT create_hypertable("
        "'price_bar', 'valid_from', chunk_time_interval => INTERVAL '1 month')"
    )
    for table in _FACT_TABLES:
        op.execute(
            f"CREATE INDEX ix_{table}_asof_lookup "
            f"ON {table} (security_id, valid_from, knowledge_time DESC)"
        )
    op.execute(
        """
        CREATE FUNCTION bitemporal_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'bitemporal fact table % is append-only (D-011): % rejected; '
                'corrections and retractions are new rows with a later knowledge_time',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    for table in _FACT_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table}_append_only "
            f"BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION bitemporal_append_only_guard()"
        )


def downgrade() -> None:
    """Drop triggers, guard function, and as-of indices (hypertable stays converted)."""
    for table in _FACT_TABLES:
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.execute("DROP FUNCTION bitemporal_append_only_guard()")
    for table in _FACT_TABLES:
        op.execute(f"DROP INDEX ix_{table}_asof_lookup")
