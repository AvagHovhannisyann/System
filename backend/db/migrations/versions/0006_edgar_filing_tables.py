"""SEC EDGAR filing and document fact tables (P3.2, D-011).

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-01

Two bitemporal fact tables, built to the same D-011 shape as revisions
0002/0003: the five mixin columns, a primary key of (logical key,
``valid_from``, ``knowledge_time``) so corrections are new rows, a
``valid_from < valid_to`` CHECK, the composite as-of index in exactly the
``DISTINCT ON``/``ORDER BY`` order the query layer emits, and a BEFORE UPDATE
OR DELETE row trigger making them append-only in the database.

``edgar_filing`` is a **hypertable** partitioned on ``valid_from`` with 1-month
chunks, for the same reason ``price_bar`` is: ``valid_from`` here is the
filing's acceptance instant, filings are an append-only event stream that
accumulates on the order of a million rows a year, and every research read
bounds event time (a training window, a rebalance date's lookback), so chunk
pruning bites on every query. The primary key
``(accession_number, cik, valid_from, knowledge_time)`` already contains the
partition column, satisfying ``create_hypertable``'s requirement.

``edgar_filing_document`` is deliberately **not** a hypertable. Its access
pattern is "the documents of these accessions" — a point lookup on the logical
key, not an event-time range — so chunk pruning would not bite, and
partitioning would only spread every accession lookup across every chunk. It
carries the same append-only trigger and the same as-of index.

Neither table has a foreign key: ``edgar_filing_document`` cannot reference
``edgar_filing`` because a bitemporal table's logical key is not unique per
row, and neither can reference ``security`` because no CIK-to-security mapping
exists yet (it arrives with the connectors blocked on B1). Both omissions are
documented on the models rather than silently present.

Downgrade drops the triggers and both tables. The trigger function itself is
shared with revisions 0003/0004 and is left alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_FACT_TABLES = ("edgar_filing", "edgar_filing_document")


def _bitemporal_columns() -> list[sa.Column[Any]]:
    """Return the five D-011 mixin columns, freshly constructed.

    Identical to revision 0002's helper — repeated rather than imported so a
    later edit to that revision cannot retroactively change this one's DDL.
    ``knowledge_time`` has no default of any kind: writers supply it.
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
    """Create the EDGAR fact tables, the hypertable, indices and triggers."""
    op.create_table(
        "edgar_filing",
        sa.Column("accession_number", sa.Text(), nullable=False),
        sa.Column("cik", sa.BigInteger(), nullable=False),
        sa.Column("company_name", sa.Text(), nullable=False),
        sa.Column("form_type", sa.Text(), nullable=False),
        # Calendar dates, no time component. filing_date is EDGAR's assigned
        # filing date and is never a knowledge time (17 CFR 232.13); index_date
        # is the daily index the filing was disseminated in.
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("index_date", sa.Date(), nullable=False),
        sa.Column("period_of_report", sa.Date(), nullable=True),
        sa.Column("declared_document_count", sa.Integer(), nullable=True),
        sa.Column("document_count", sa.Integer(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        *_bitemporal_columns(),
        # (accession_number, cik) is the logical key: EDGAR lists one accession
        # once per associated filer, and the majority of accessions in a daily
        # index have more than one. See the model docstring.
        sa.PrimaryKeyConstraint(
            "accession_number", "cik", "valid_from", "knowledge_time", name="pk_edgar_filing"
        ),
        sa.CheckConstraint("valid_from < valid_to", name="ck_edgar_filing_valid_interval"),
        sa.CheckConstraint("document_count >= 0", name="ck_edgar_filing_document_count"),
    )
    op.create_table(
        "edgar_filing_document",
        sa.Column("accession_number", sa.Text(), nullable=False),
        sa.Column("document_sequence", sa.Integer(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("document_url", sa.Text(), nullable=False),
        *_bitemporal_columns(),
        sa.PrimaryKeyConstraint(
            "accession_number",
            "document_sequence",
            "valid_from",
            "knowledge_time",
            name="pk_edgar_filing_document",
        ),
        sa.CheckConstraint("valid_from < valid_to", name="ck_edgar_filing_document_valid_interval"),
    )
    # Hypertable before the indices so Timescale propagates them to chunks.
    op.execute(
        "SELECT create_hypertable("
        "'edgar_filing', 'valid_from', chunk_time_interval => INTERVAL '1 month')"
    )
    op.execute(
        "CREATE INDEX ix_edgar_filing_asof_lookup "
        "ON edgar_filing (accession_number, cik, valid_from, knowledge_time DESC)"
    )
    op.execute(
        "CREATE INDEX ix_edgar_filing_document_asof_lookup "
        "ON edgar_filing_document "
        "(accession_number, document_sequence, valid_from, knowledge_time DESC)"
    )
    # Coverage/gap/staleness reporting (P3.9) scans by the daily-index date;
    # without this it would sequential-scan every chunk on every report.
    op.execute("CREATE INDEX ix_edgar_filing_index_date ON edgar_filing (index_date)")
    for table in _FACT_TABLES:
        op.execute(
            f"CREATE TRIGGER trg_{table}_append_only "
            f"BEFORE UPDATE OR DELETE ON {table} "
            f"FOR EACH ROW EXECUTE FUNCTION bitemporal_append_only_guard()"
        )


def downgrade() -> None:
    """Drop the append-only triggers and both EDGAR fact tables."""
    for table in _FACT_TABLES:
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.drop_table("edgar_filing_document")
    op.drop_table("edgar_filing")
