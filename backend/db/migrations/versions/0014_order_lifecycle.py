"""Order management: orders, the append-only transition log, and their guards (P11.2).

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-02

Two tables, neither bitemporal, neither a hypertable, both **append-only** by the
same ``BEFORE UPDATE OR DELETE`` row-trigger shape revisions
0003/0004/0007/0009/0010/0011 use:

- ``execution_order`` — the immutable content of one order plus the
  content-derived idempotency key. **No state column**: an append-only table
  cannot update one, and a denormalized state that could drift from the history
  is the condition the transition log exists to make impossible.
- ``execution_order_transition`` — one row per state change, in a gapless
  per-order sequence, carrying the fill payload on fill events and nothing on any
  other.

Three enforcement points, none of which the application can waive
-----------------------------------------------------------------

**1. Idempotency: ``UNIQUE (idempotency_key)``.** The key is a SHA-256 digest of
the order's own canonical content (``backend.execution.idempotency``). Checking
for a duplicate in Python would be a read followed by a write with a window
between them, and two workers racing through that window both find the key free
and both insert. The constraint has no window: whatever else is running, in
whatever process, exactly one insert survives and the loser gets an
``IntegrityError``. A CHECK additionally requires the key to *look* like a
digest, because a key that is not derived from content is a counter wearing a
digest's clothes, and a counter cannot survive the restart it exists for.

**2. Illegal transitions: two CHECKs.** ``from_state_not_terminal`` bans any
transition out of ``filled``/``cancelled``/``rejected``/``expired`` — this is
what makes ``FILLED -> PENDING_NEW`` unrepresentable rather than merely unlikely
— and ``legal_transition`` enumerates the 25 legal
``(from_state, event, to_state)`` triples of
``backend.execution.lifecycle.TRANSITIONS``. A test compares the SQL enumeration
against the Python table so the two cannot drift.

**3. Chain integrity: the ``execution_transition_chain_guard`` trigger.** A CHECK
sees one row; the properties that make a history replayable are relations between
rows. The trigger checks each insert against its predecessor: sequence numbers
are ``1..n`` with no gaps, ``from_state`` equals the previous row's ``to_state``,
the cumulative filled quantity is the running sum of the per-event quantities, it
never exceeds the order's quantity, and an order reaching ``filled`` has traded
its whole quantity. Under concurrency two writers both pass the trigger and the
primary key rejects the loser — which is the retryable
``ConcurrentTransitionError``, not corruption.

Paper-only, at the database
---------------------------

``venue`` carries a server default of ``'paper'`` and a ``CHECK (venue =
'paper')``; no writer supplies the column at all. ``fill_source`` admits only
``'simulated'`` and ``'paper_broker'``, so a live execution has no representation
(I3). ``fill_cost_basis`` admits only ``'lower_bound'``, so no fill can claim to
be a calibrated estimate of cost (D-013). These restate in SQL what
``backend/execution/orders.py`` makes true in the type system; the point of
restating it is that a writer bypassing Python entirely is still bound.

Neither table is bitemporal, and that is deliberate — same reasoning as revisions
0005, 0007, 0009, 0010 and 0011. The bitemporal columns describe when a fact was
true in the world and when it became knowable to the *market* (D-011). An order
is a decision **we** took and an event stream **we** were sent; its
point-in-time content is the ``as_of`` already baked into the reproducibility
stamp of the plan that produced it. A ``knowledge_time`` invented for it would be
a fabricated value in the one column whose meaning is that it is not fabricated
(I3). So these tables are absent from the bitemporal registry, unscoped by the
Core-level read guard, and read without an as-of.

TRUNCATE stays unblocked on both, matching every earlier revision: it is the
sanctioned admin and test reset path and never masquerades as an edit.

Downgrade drops the triggers, their functions, the indices and both tables in
dependency order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = ("execution_order", "execution_order_transition")
"""Tables that get the append-only trigger, in creation order.

Named here *and* spelled out in full ``CREATE TRIGGER`` statements below rather
than generated from this tuple. The duplication is deliberate: "which tables in
this schema are append-only" is a question a human answers by grepping for
``BEFORE UPDATE OR DELETE ON``, and a loop that assembles the statement makes
that grep return nothing. Revisions 0003, 0004, 0007, 0009, 0010 and 0011 write
theirs out for the same reason, and a test asserts the statements are present.
"""

_LEGAL_TRANSITIONS_SQL = (
    "(from_state, event, to_state) IN ("
    "('draft', 'release', 'pending_new'), "
    "('draft', 'abandon', 'cancelled'), "
    "('pending_new', 'acknowledge', 'acknowledged'), "
    "('pending_new', 'reject', 'rejected'), "
    "('pending_new', 'expire', 'expired'), "
    "('acknowledged', 'partial_fill', 'partially_filled'), "
    "('acknowledged', 'fill_complete', 'filled'), "
    "('acknowledged', 'request_cancel', 'pending_cancel'), "
    "('acknowledged', 'expire', 'expired'), "
    "('acknowledged', 'venue_cancel', 'cancelled'), "
    "('partially_filled', 'partial_fill', 'partially_filled'), "
    "('partially_filled', 'fill_complete', 'filled'), "
    "('partially_filled', 'request_cancel', 'pending_cancel_partial'), "
    "('partially_filled', 'expire', 'expired'), "
    "('partially_filled', 'venue_cancel', 'cancelled'), "
    "('pending_cancel', 'cancel_confirmed', 'cancelled'), "
    "('pending_cancel', 'cancel_rejected', 'acknowledged'), "
    "('pending_cancel', 'partial_fill', 'pending_cancel_partial'), "
    "('pending_cancel', 'fill_complete', 'filled'), "
    "('pending_cancel', 'expire', 'expired'), "
    "('pending_cancel_partial', 'cancel_confirmed', 'cancelled'), "
    "('pending_cancel_partial', 'cancel_rejected', 'partially_filled'), "
    "('pending_cancel_partial', 'partial_fill', 'pending_cancel_partial'), "
    "('pending_cancel_partial', 'fill_complete', 'filled'), "
    "('pending_cancel_partial', 'expire', 'expired'))"
)
"""The 25 legal triples of ``backend.execution.lifecycle.TRANSITIONS``, in SQL.

Compared against the Python table by
``backend/tests/execution/test_migration.py``: a transition added to one and not
the other is a schema that admits a state the machine refuses, or refuses one it
produces, and both are silent until a real order hits them.
"""


def upgrade() -> None:
    """Create both execution tables, their constraints, indices and triggers."""
    op.create_table(
        "execution_order",
        sa.Column("order_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("idempotency_schema", sa.Text(), nullable=False),
        sa.Column("idempotency_preimage", sa.Text(), nullable=False),
        sa.Column("venue", sa.Text(), nullable=False, server_default=sa.text("'paper'")),
        sa.Column("security_id", sa.BigInteger(), nullable=False),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("quantity_shares", sa.BigInteger(), nullable=False),
        sa.Column("order_type", sa.Text(), nullable=False),
        sa.Column("time_in_force", sa.Text(), nullable=False),
        sa.Column("limit_price_usd", sa.Numeric(18, 6), nullable=True),
        sa.Column("rebalance_date", sa.Date(), nullable=False),
        sa.Column("slice_index", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("slice_count", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("git_commit", sa.Text(), nullable=False),
        sa.Column("git_dirty", sa.Boolean(), nullable=False),
        sa.Column("data_version", sa.Text(), nullable=False),
        sa.Column("config_hash", sa.Text(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("order_id", name="pk_execution_order"),
        sa.ForeignKeyConstraint(
            ("security_id",),
            ("security.security_id",),
            name="fk_execution_order_security_id_security",
        ),
        # The idempotency enforcement point (module docstring). A duplicate
        # submission fails here, in the database, not in a check-then-insert
        # window that two concurrent workers both walk through.
        sa.UniqueConstraint("idempotency_key", name="uq_execution_order_idempotency_key"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005, 0007, 0009,
        # 0010, 0011).
        #
        # Paper-only at the database (directive §1.1, §9.5). No writer supplies
        # this column; the server default does, and this refuses anything else.
        sa.CheckConstraint("venue = 'paper'", name="venue_is_paper"),
        sa.CheckConstraint("quantity_shares > 0", name="quantity_shares_positive"),
        sa.CheckConstraint("security_id > 0", name="security_id_positive"),
        sa.CheckConstraint("side IN ('buy', 'sell')", name="side_is_known"),
        sa.CheckConstraint("order_type IN ('market', 'limit')", name="order_type_is_known"),
        sa.CheckConstraint("time_in_force IN ('day', 'gtc')", name="time_in_force_is_known"),
        # The type and the price together are the instruction; either half alone
        # is ambiguous. Two-sided in the D-030 shape, because forbidding only one
        # direction trades a fabrication bug for a silent-absence bug.
        sa.CheckConstraint(
            "(order_type = 'limit') = (limit_price_usd IS NOT NULL)",
            name="limit_price_iff_limit_order",
        ),
        sa.CheckConstraint(
            "limit_price_usd IS NULL OR limit_price_usd > 0", name="limit_price_positive"
        ),
        sa.CheckConstraint(
            "slice_count >= 1 AND slice_index >= 0 AND slice_index < slice_count",
            name="slice_within_count",
        ),
        sa.CheckConstraint("idempotency_key ~ '^[0-9a-f]{64}$'", name="idempotency_key_is_sha256"),
        sa.CheckConstraint("idempotency_preimage <> ''", name="idempotency_preimage_present"),
        sa.CheckConstraint("idempotency_schema <> ''", name="idempotency_schema_present"),
        # I2: all four stamp components present on every order, so every fill
        # traces to the commit and config that produced the decision.
        sa.CheckConstraint(
            "git_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'", name="git_commit_is_sha"
        ),
        sa.CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        sa.CheckConstraint("data_version <> ''", name="data_version_present"),
        sa.CheckConstraint("seed >= 0", name="seed_non_negative"),
    )
    # §6.9's blotter reads "what did we trade on this rebalance", and P11.3
    # reconciles a cycle's orders against broker positions per name. The primary
    # key is a surrogate and answers neither.
    op.create_index(
        "ix_execution_order_rebalance",
        "execution_order",
        ["rebalance_date", "security_id"],
    )
    op.create_table(
        "execution_order_transition",
        sa.Column("order_id", sa.BigInteger(), nullable=False),
        sa.Column("sequence_number", sa.Integer(), nullable=False),
        sa.Column("from_state", sa.Text(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("to_state", sa.Text(), nullable=False),
        sa.Column("filled_quantity_after_shares", sa.BigInteger(), nullable=False),
        sa.Column("fill_quantity_shares", sa.BigInteger(), nullable=True),
        sa.Column("fill_price_usd", sa.Numeric(18, 6), nullable=True),
        sa.Column("fill_source", sa.Text(), nullable=True),
        sa.Column("fill_cost_basis", sa.Text(), nullable=True),
        sa.Column("venue_fill_id", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("note", sa.Text(), nullable=True),
        # (order_id, sequence_number) is also the concurrency control: two
        # writers computing last+1 from the same tail claim the same number and
        # the loser is rejected here.
        sa.PrimaryKeyConstraint(
            "order_id", "sequence_number", name="pk_execution_order_transition"
        ),
        sa.ForeignKeyConstraint(
            ("order_id",),
            ("execution_order.order_id",),
            name="fk_execution_order_transition_order_id_execution_order",
        ),
        sa.CheckConstraint("sequence_number >= 1", name="sequence_number_positive"),
        # The named property this table exists to guarantee. Implied by
        # legal_transition below, and stated separately anyway: it is the one
        # constraint whose violation produces a phantom position, and a reader
        # grepping for it should find it by name.
        sa.CheckConstraint(
            "from_state NOT IN ('filled', 'cancelled', 'rejected', 'expired')",
            name="from_state_not_terminal",
        ),
        sa.CheckConstraint(_LEGAL_TRANSITIONS_SQL, name="legal_transition"),
        sa.CheckConstraint("filled_quantity_after_shares >= 0", name="filled_after_non_negative"),
        # D-030 shape, both directions: the payload is present exactly on fill
        # events and absent on every other. A non-fill row carrying a quantity
        # would be a trade nobody reported; a fill row missing one would be a
        # trade whose size is a gap.
        sa.CheckConstraint(
            "event NOT IN ('partial_fill', 'fill_complete') OR ("
            "fill_quantity_shares IS NOT NULL AND fill_price_usd IS NOT NULL "
            "AND fill_source IS NOT NULL AND fill_cost_basis IS NOT NULL)",
            name="fill_payload_present",
        ),
        sa.CheckConstraint(
            "event IN ('partial_fill', 'fill_complete') OR ("
            "fill_quantity_shares IS NULL AND fill_price_usd IS NULL "
            "AND fill_source IS NULL AND fill_cost_basis IS NULL "
            "AND venue_fill_id IS NULL)",
            name="fill_payload_absent",
        ),
        sa.CheckConstraint(
            "fill_quantity_shares IS NULL OR fill_quantity_shares > 0",
            name="fill_quantity_positive",
        ),
        sa.CheckConstraint(
            "fill_price_usd IS NULL OR fill_price_usd > 0", name="fill_price_positive"
        ),
        # I3: a live fill has no representation here. Not a disabled option — an
        # absent one. A third value needs a migration, not a configuration edit.
        sa.CheckConstraint(
            "fill_source IS NULL OR fill_source IN ('simulated', 'paper_broker')",
            name="fill_source_is_not_live",
        ),
        # D-013: paper and simulated fills bound slippage from below. A row
        # claiming any other basis would let a later calibration treat an
        # optimistic fill as a measured estimate.
        sa.CheckConstraint(
            "fill_cost_basis IS NULL OR fill_cost_basis = 'lower_bound'",
            name="fill_cost_basis_is_lower_bound",
        ),
        sa.CheckConstraint(
            "venue_fill_id IS NULL OR venue_fill_id <> ''", name="venue_fill_id_set"
        ),
    )
    op.execute(
        """
        CREATE FUNCTION execution_transition_chain_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            previous execution_order_transition%ROWTYPE;
            ordered_shares bigint;
            traded bigint := COALESCE(NEW.fill_quantity_shares, 0);
        BEGIN
            SELECT quantity_shares INTO ordered_shares
              FROM execution_order WHERE order_id = NEW.order_id;
            IF ordered_shares IS NULL THEN
                RAISE EXCEPTION
                    'transition names order_id % which does not exist', NEW.order_id;
            END IF;
            SELECT * INTO previous FROM execution_order_transition
             WHERE order_id = NEW.order_id
             ORDER BY sequence_number DESC LIMIT 1;
            IF previous.order_id IS NULL THEN
                IF NEW.sequence_number <> 1 THEN
                    RAISE EXCEPTION
                        'first transition of order % must be sequence 1, got %',
                        NEW.order_id, NEW.sequence_number;
                END IF;
                IF NEW.from_state <> 'draft' THEN
                    RAISE EXCEPTION
                        'first transition of order % must start in draft, got %',
                        NEW.order_id, NEW.from_state;
                END IF;
                IF NEW.filled_quantity_after_shares <> traded THEN
                    RAISE EXCEPTION
                        'first transition of order % claims % shares filled, but this event '
                        'trades %', NEW.order_id, NEW.filled_quantity_after_shares, traded;
                END IF;
            ELSE
                -- Position already held. Raised as unique_violation on purpose:
                -- it is the same refusal the primary key gives, arriving by a
                -- different route, and the caller must not have to tell them
                -- apart. Under READ COMMITTED a writer whose INSERT starts
                -- after the winner committed reaches *this* branch rather than
                -- the index, because each statement takes a fresh snapshot.
                IF NEW.sequence_number <= previous.sequence_number THEN
                    RAISE EXCEPTION
                        'order % is already at sequence %, so position % is taken; another '
                        'writer claimed it first — re-read the history and decide again',
                        NEW.order_id, previous.sequence_number, NEW.sequence_number
                        USING ERRCODE = 'unique_violation';
                END IF;
                -- A gap is a different failure: nobody holds the position, the
                -- writer skipped one. Not retryable, so it keeps the default
                -- raise_exception code and reaches the caller as a chain error.
                IF NEW.sequence_number <> previous.sequence_number + 1 THEN
                    RAISE EXCEPTION
                        'order % is at sequence %, so the next transition is %, not %',
                        NEW.order_id, previous.sequence_number,
                        previous.sequence_number + 1, NEW.sequence_number;
                END IF;
                IF NEW.from_state <> previous.to_state THEN
                    RAISE EXCEPTION
                        'order % had reached %, but this transition starts in %; the chain '
                        'does not connect', NEW.order_id, previous.to_state, NEW.from_state;
                END IF;
                IF NEW.filled_quantity_after_shares
                   <> previous.filled_quantity_after_shares + traded THEN
                    RAISE EXCEPTION
                        'order % had % shares filled and this event trades %, so the '
                        'cumulative quantity is %, not %',
                        NEW.order_id, previous.filled_quantity_after_shares, traded,
                        previous.filled_quantity_after_shares + traded,
                        NEW.filled_quantity_after_shares;
                END IF;
            END IF;
            IF NEW.filled_quantity_after_shares > ordered_shares THEN
                RAISE EXCEPTION
                    'order % is for % shares but this transition takes the filled quantity '
                    'to %; refusing rather than clamping, because clamping discards shares '
                    'the venue says it traded',
                    NEW.order_id, ordered_shares, NEW.filled_quantity_after_shares;
            END IF;
            IF NEW.to_state = 'filled'
               AND NEW.filled_quantity_after_shares <> ordered_shares THEN
                RAISE EXCEPTION
                    'order % reaches filled with % of % shares traded; a filled order has '
                    'traded its whole quantity, and a shortfall here is a position the '
                    'blotter would not show',
                    NEW.order_id, NEW.filled_quantity_after_shares, ordered_shares;
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_execution_order_transition_chain "
        "BEFORE INSERT ON execution_order_transition "
        "FOR EACH ROW EXECUTE FUNCTION execution_transition_chain_guard()"
    )
    op.execute(
        """
        CREATE FUNCTION execution_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'execution table % is append-only (P11.2): % rejected; an order records an '
                'instruction and a transition records something that happened, and neither '
                'stops being true later — a correction is a new transition, and erasing one '
                'would destroy the audit trail a fill has to be traced through (I2)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_execution_order_append_only "
        "BEFORE UPDATE OR DELETE ON execution_order "
        "FOR EACH ROW EXECUTE FUNCTION execution_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_execution_order_transition_append_only "
        "BEFORE UPDATE OR DELETE ON execution_order_transition "
        "FOR EACH ROW EXECUTE FUNCTION execution_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the triggers, their functions, the index, and both tables."""
    for table in reversed(_APPEND_ONLY_TABLES):
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.execute("DROP FUNCTION execution_append_only_guard()")
    op.execute("DROP TRIGGER trg_execution_order_transition_chain ON execution_order_transition")
    op.execute("DROP FUNCTION execution_transition_chain_guard()")
    op.drop_table("execution_order_transition")
    op.drop_index("ix_execution_order_rebalance", table_name="execution_order")
    op.drop_table("execution_order")
