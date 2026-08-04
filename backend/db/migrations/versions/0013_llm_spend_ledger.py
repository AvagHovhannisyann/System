"""LLM spend ledger: per-provider cap enforcement and its audit trail (P7.7).

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-02

One table, ``llm_spend_ledger``, not bitemporal, not a hypertable, and
**append-only** by the same ``BEFORE UPDATE OR DELETE`` row trigger shape
revisions 0003/0004/0007/0009/0010 use.

It is an **event log**, not a balance. Three row kinds — ``reserved``,
``released``, ``settled`` — tied by ``reservation_id``, each carrying a signed
``delta_amount``, so a window's committed spend is ``SUM(delta_amount)``. The
shape follows from where the cap is enforced: *before* the call, against an
upper bound on what the call will cost, reconciled afterwards against what it
actually cost. Both numbers matter — the bound is what the control acted on, the
actual is what happened, and the gap between them is the measurement that says
whether the estimator is doing its job or quietly strangling a budget. A running
balance would keep neither.

Two settlements of one reservation would double-count a delta and corrupt every
later cap check, so ``(reservation_id, event)`` is unique: the application
refuses it with a clear message and the database refuses it regardless of which
process asked.

Not bitemporal, matching revisions 0005, 0007, 0009 and 0010: the bitemporal
columns describe when a fact was true in the world and when it became knowable
to the *market* (D-011). Spending our own money on our own extraction has no
market knowability, and a ``knowledge_time`` invented for it would be a
fabricated value in the one column whose meaning is that it is not fabricated
(I3). Consequently it is absent from the bitemporal registry, unscoped by the
Core-level read guard, and read without an as-of.

Not a hypertable: reads are "this provider, this window", which the two partial
indices below serve directly, not scans along an event-time axis worth chunking
on.

TRUNCATE stays unblocked, matching 0003/0004/0007/0009/0010: it is the
sanctioned admin/test reset path and never masquerades as an edit.

Downgrade drops the trigger, its function, the indices and the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROVIDER_NAMES_SQL = "'anthropic', 'openai'"
"""Provider vocabulary, spelled as the CHECK constraint spells it.

Duplicated from ``backend.db.models.PROVIDER_NAMES_SQL`` and
:class:`backend.extraction.providers.catalog.Provider` rather than imported: a
migration must describe the schema at *its* revision and must not change meaning
because application code changed later. Revision 0009 does the same, and a unit
test asserts the three spellings still name the same set.
"""


def upgrade() -> None:
    """Create ``llm_spend_ledger``, its constraints, indices and append-only trigger."""
    op.create_table(
        "llm_spend_ledger",
        sa.Column("ledger_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("reservation_id", sa.Text(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("requested_model", sa.Text(), nullable=False),
        sa.Column("served_model", sa.Text(), nullable=False),
        sa.Column("currency", sa.Text(), nullable=False),
        sa.Column("estimated_cost", sa.Numeric(20, 10), nullable=False),
        sa.Column("actual_cost", sa.Numeric(20, 10), nullable=True),
        sa.Column("delta_amount", sa.Numeric(20, 10), nullable=False),
        sa.Column("input_tokens_bound", sa.Integer(), nullable=False),
        sa.Column("output_tokens_bound", sa.Integer(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("daily_window", sa.Text(), nullable=False),
        sa.Column("monthly_window", sa.Text(), nullable=False),
        sa.Column("daily_limit", sa.Numeric(20, 10), nullable=False),
        sa.Column("monthly_limit", sa.Numeric(20, 10), nullable=False),
        sa.Column("policy", sa.Text(), nullable=False),
        sa.Column("degraded", sa.Boolean(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=True),
        sa.Column("reconciled", sa.Boolean(), nullable=False),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("ledger_id", name="pk_llm_spend_ledger"),
        # One settlement and one release per reservation. The application also
        # checks, in a different transaction; this is the guarantee that holds
        # when two processes check at once.
        sa.UniqueConstraint(
            "reservation_id", "event", name="uq_llm_spend_ledger_reservation_event"
        ),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005, 0007, 0009, 0010).
        sa.CheckConstraint(f"provider IN ({_PROVIDER_NAMES_SQL})", name="provider_known"),
        sa.CheckConstraint("event IN ('reserved', 'released', 'settled')", name="event_known"),
        sa.CheckConstraint("policy IN ('halt', 'degrade')", name="policy_known"),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('succeeded', 'failed')", name="outcome_known"
        ),
        sa.CheckConstraint("reservation_id <> ''", name="reservation_id_not_empty"),
        sa.CheckConstraint("requested_model <> ''", name="requested_model_not_empty"),
        sa.CheckConstraint("served_model <> ''", name="served_model_not_empty"),
        # Currency is a shape check, not a list: enumerating the world's
        # currencies here is a table nobody maintains, while refusing 'usd' and
        # '' catches the mistake that actually happens — two spellings of one
        # currency becoming two budgets.
        sa.CheckConstraint("currency ~ '^[A-Z]{3}$'", name="currency_is_iso4217_alpha"),
        sa.CheckConstraint(r"daily_window ~ '^\d{4}-\d{2}-\d{2}$'", name="daily_window_shape"),
        sa.CheckConstraint(r"monthly_window ~ '^\d{4}-\d{2}$'", name="monthly_window_shape"),
        sa.CheckConstraint("estimated_cost >= 0", name="estimated_cost_non_negative"),
        sa.CheckConstraint(
            "actual_cost IS NULL OR actual_cost >= 0", name="actual_cost_non_negative"
        ),
        sa.CheckConstraint("daily_limit >= 0", name="daily_limit_non_negative"),
        sa.CheckConstraint("monthly_limit >= 0", name="monthly_limit_non_negative"),
        sa.CheckConstraint("input_tokens_bound >= 0", name="input_tokens_bound_non_negative"),
        sa.CheckConstraint("output_tokens_bound >= 1", name="output_tokens_bound_positive"),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0", name="input_tokens_counted"
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0", name="output_tokens_counted"
        ),
        # The event algebra, in the schema, so a row cannot claim arithmetic the
        # ledger does not do:
        #   reserved -> delta = +estimate
        #   released -> delta = -estimate
        #   settled  -> delta = coalesce(actual, estimate) - estimate
        sa.CheckConstraint(
            "(event = 'reserved' AND delta_amount = estimated_cost) "
            "OR (event = 'released' AND delta_amount = -estimated_cost) "
            "OR (event = 'settled' "
            "    AND delta_amount = COALESCE(actual_cost, estimated_cost) - estimated_cost)",
            name="delta_matches_event",
        ),
        sa.CheckConstraint(
            "(outcome IS NOT NULL) = (event = 'settled')", name="outcome_iff_settled"
        ),
        sa.CheckConstraint(
            "reconciled = (actual_cost IS NOT NULL)", name="reconciled_iff_measured"
        ),
        sa.CheckConstraint(
            "event = 'settled' OR (input_tokens IS NULL AND output_tokens IS NULL)",
            name="tokens_only_on_settlement",
        ),
        sa.CheckConstraint(
            "degraded = (requested_model <> served_model)", name="degraded_iff_substituted"
        ),
    )
    # The two access patterns, and the only two: "what has this provider spent
    # today" and "what has it spent this month". Both are read on the path to
    # every call, so both get an index rather than a sequential scan that grows
    # with the ledger.
    op.create_index(
        "ix_llm_spend_ledger_provider_daily",
        "llm_spend_ledger",
        ["provider", "daily_window"],
    )
    op.create_index(
        "ix_llm_spend_ledger_provider_monthly",
        "llm_spend_ledger",
        ["provider", "monthly_window"],
    )
    # Partial, matching config_change_event's: rows made outside a request carry
    # no correlation id and are never the answer to "what did this request cost".
    op.create_index(
        "ix_llm_spend_ledger_correlation_id",
        "llm_spend_ledger",
        ["correlation_id"],
        postgresql_where=sa.text("correlation_id IS NOT NULL"),
    )
    op.execute(
        """
        CREATE FUNCTION llm_spend_ledger_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'llm_spend_ledger is append-only: % rejected. Spend is recorded as reserved / '
                'settled / released events whose signed deltas sum to committed spend; editing '
                'one would rewrite what a cap check saw at the moment it admitted a call '
                '(DIRECTIVE 6.5 and 6.11, P7.7)',
                TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_llm_spend_ledger_append_only "
        "BEFORE UPDATE OR DELETE ON llm_spend_ledger "
        "FOR EACH ROW EXECUTE FUNCTION llm_spend_ledger_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the append-only trigger, its function, the indices and the table."""
    op.execute("DROP TRIGGER trg_llm_spend_ledger_append_only ON llm_spend_ledger")
    op.execute("DROP FUNCTION llm_spend_ledger_append_only_guard()")
    op.drop_index("ix_llm_spend_ledger_correlation_id", table_name="llm_spend_ledger")
    op.drop_index("ix_llm_spend_ledger_provider_monthly", table_name="llm_spend_ledger")
    op.drop_index("ix_llm_spend_ledger_provider_daily", table_name="llm_spend_ledger")
    op.drop_table("llm_spend_ledger")
