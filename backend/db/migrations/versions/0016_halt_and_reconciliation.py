"""Reconciliation verdicts and the halt log, both append-only (P11.3, P11.5).

Revision ID: 0016
Revises: 0015
Create Date: 2026-08-02

Two tables, neither bitemporal, neither a hypertable, both **append-only** by the
same ``BEFORE UPDATE OR DELETE`` row-trigger shape revisions
0003/0004/0007/0009/0010/0011/0014 use:

- ``execution_reconciliation`` — one row per cycle's comparison of our book
  against a statement, holding **both snapshots in full** beside their digests so
  the verdict can be re-derived from the row alone. A row that held only
  "3 breaks found" would be a claim nobody could check, and a break nobody can
  re-examine cannot be investigated.
- ``execution_halt`` — engagements and clearances. A halt is a row, not a flag,
  because a halt held in memory evaporates on restart, at exactly the moment it
  matters.

Where the guarantees live, and where they deliberately do not
--------------------------------------------------------------

**Nothing constrains a halt engagement.** ``execution_halt`` has a unique index
and a ``BEFORE INSERT`` trigger, and both act only on *clearances*. That
asymmetry is the single most important decision in this revision: a constraint
that can reject a halt-engage row is a constraint that can stop the kill switch
from firing. Duplicate engagements, concurrent engagements, engagements for a
cycle that already halted — all accepted. A redundant halt row costs nothing.

**A halt is cleared at most once, and only by naming an engagement.**
``UNIQUE (clears_halt_id)`` gives the first; the ``execution_halt_clearance_guard``
trigger gives the second. NULLs are distinct in Postgres, so the unique index
leaves every engagement untouched — *provided* no engagement can carry a
``clears_halt_id`` at all, which is what ``clearance_fields_iff_cleared``
enforces per column. Its first version stated the rule over the conjunction of
the three clearance columns and so admitted a stray id on an engagement; that
row consumed the unique slot and made the halt permanently un-clearable while
reporting "already cleared". Corrected in place rather than in a follow-up
revision: the constraint was wrong on arrival, no database outside ephemeral CI
containers has ever applied this revision, and splitting one constraint's
origin across two revisions would hide that.

**Deciding on SQLSTATE, never on the exception class (D-034).** A ``BEFORE
INSERT`` trigger fires ahead of every CHECK and every index, and a plpgsql
``RAISE`` reaches SQLAlchemy as a generic ``DBAPIError`` rather than an
``IntegrityError``. So the guard splits its two refusals by code, exactly as
revision 0014 split a taken sequence position from a gap:

- a clearance for a halt that is **already cleared** raises
  ``USING ERRCODE = 'unique_violation'`` — the same refusal the unique index
  gives, arriving by a different route. Under ``READ COMMITTED`` the loser of a
  race whose statement starts after the winner committed reaches the *trigger*
  rather than the index, because each statement takes a fresh snapshot, and the
  caller must not have to tell the two apart;
- a clearance naming a row that does not exist or is **not an engagement** keeps
  the default ``P0001``, because it is not retryable: nothing about retrying it
  can succeed, and translating it into a conflict would loop forever.

Paper-only, at the database
---------------------------

``execution_reconciliation.reported_origin`` admits only ``'paper_broker'`` and
``'simulated'`` — a statement from a real-money account has no representation
(I3, the same shape 0014 uses for ``fill_source``) — and
``internal_origin`` is pinned to ``'internal_ledger'`` so a reconciliation of a
snapshot against itself, which always passes and proves nothing, cannot be
recorded.

``cash_tolerance_usd`` carries the ceiling from
``backend.execution.reconciliation.MAX_CASH_TOLERANCE_USD`` as a CHECK, so no
Python can waive it: above five cents a "tolerance" absorbs an event rather than
one quantisation step between a two-decimal statement and a six-decimal ledger.

Neither table is bitemporal, and that is deliberate — same reasoning as revisions
0005, 0007, 0009, 0010, 0011 and 0014. A reconciliation is an observation *we*
made and a halt is a decision *we* took; neither is a fact about the world whose
knowledge time the market determined, and a ``knowledge_time`` invented for one
would be a fabricated value in the one column whose meaning is that it is not
(I3).

TRUNCATE stays unblocked on both, matching every earlier revision: it is the
sanctioned admin and test reset path and never masquerades as an edit.

Downgrade drops the triggers, their functions, the indices and both tables in
dependency order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = ("execution_reconciliation", "execution_halt")
"""Tables that get the append-only trigger, in creation order.

Named here *and* spelled out in full ``CREATE TRIGGER`` statements below rather
than generated from this tuple. The duplication is deliberate: "which tables in
this schema are append-only" is a question a human answers by grepping for
``BEFORE UPDATE OR DELETE ON``, and a loop that assembles the statement makes
that grep return nothing. Revisions 0003, 0004, 0007, 0009, 0010, 0011 and 0014
write theirs out for the same reason, and a test asserts the statements are
present.
"""

_HALT_TRIGGERS_SQL = (
    "halt_trigger IS NULL OR halt_trigger IN ("
    "'drawdown_breach', 'stale_data', 'reconciliation_mismatch', 'manual', "
    "'unknown_condition')"
)
"""The five members of ``backend.execution.halt.HaltTrigger``, in SQL.

Compared against the Python enum by
``backend/tests/execution/test_control_migration.py``: a trigger in one and not the
other is either a halt the schema refuses to record — which is a kill switch that
cannot fire — or a stored cause the reader cannot interpret.
"""


def upgrade() -> None:
    """Create both tables, their constraints, indices and triggers."""
    op.create_table(
        "execution_reconciliation",
        sa.Column("reconciliation_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("cycle_id", sa.Text(), nullable=False),
        sa.Column("internal_origin", sa.Text(), nullable=False),
        sa.Column("reported_origin", sa.Text(), nullable=False),
        sa.Column("internal_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("internal_digest", sa.Text(), nullable=False),
        sa.Column("reported_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("reported_digest", sa.Text(), nullable=False),
        sa.Column("internal_observed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("reported_observed_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("cash_tolerance_usd", sa.Numeric(18, 6), nullable=False),
        sa.Column("findings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("finding_count", sa.Integer(), nullable=False),
        sa.Column("break_count", sa.Integer(), nullable=False),
        sa.Column("matched", sa.Boolean(), nullable=False),
        sa.Column("result_digest", sa.Text(), nullable=False),
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
        sa.PrimaryKeyConstraint("reconciliation_id", name="pk_execution_reconciliation"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005, 0007, 0009, 0010,
        # 0011, 0014).
        sa.CheckConstraint("cycle_id <> ''", name="cycle_id_present"),
        # A reconciliation of our ledger against itself always passes and proves
        # nothing; pinning the internal side makes that unrecordable rather than
        # merely discouraged.
        sa.CheckConstraint("internal_origin = 'internal_ledger'", name="internal_origin_is_ledger"),
        # I3: a statement from a real-money account has no representation. Not a
        # disabled option — an absent one. A third value needs a migration.
        sa.CheckConstraint(
            "reported_origin IN ('paper_broker', 'simulated')",
            name="reported_origin_is_not_live",
        ),
        sa.CheckConstraint("internal_digest ~ '^[0-9a-f]{64}$'", name="internal_digest_is_sha256"),
        sa.CheckConstraint("reported_digest ~ '^[0-9a-f]{64}$'", name="reported_digest_is_sha256"),
        sa.CheckConstraint("result_digest ~ '^[0-9a-f]{64}$'", name="result_digest_is_sha256"),
        # The tolerance ceiling, where no Python can waive it. Zero is allowed:
        # tightening is always permitted, widening past a quantisation step is not.
        sa.CheckConstraint(
            "cash_tolerance_usd >= 0 AND cash_tolerance_usd <= 0.05",
            name="tolerance_within_ceiling",
        ),
        sa.CheckConstraint(
            "finding_count >= 0 AND break_count >= 0 AND break_count <= finding_count",
            name="counts_consistent",
        ),
        sa.CheckConstraint("matched = (break_count = 0)", name="matched_iff_no_breaks"),
        # I2: all four stamp components on every verdict.
        sa.CheckConstraint(
            "git_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'", name="git_commit_is_sha"
        ),
        sa.CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        sa.CheckConstraint("data_version <> ''", name="data_version_present"),
        sa.CheckConstraint("seed >= 0", name="seed_non_negative"),
    )
    # §6.9's reconciliation status reads "how did this cycle reconcile", and an
    # investigation reads "show me the cycles that broke". The primary key is a
    # surrogate and answers neither.
    op.create_index(
        "ix_execution_reconciliation_cycle",
        "execution_reconciliation",
        ["cycle_id", "reconciliation_id"],
    )
    op.create_table(
        "execution_halt",
        sa.Column("halt_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("event", sa.Text(), nullable=False),
        sa.Column("halt_trigger", sa.Text(), nullable=True),
        sa.Column("cycle_id", sa.Text(), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("clears_halt_id", sa.BigInteger(), nullable=True),
        sa.Column("cleared_by", sa.Text(), nullable=True),
        sa.Column("clearance_reason", sa.Text(), nullable=True),
        sa.Column("occurred_at", sa.TIMESTAMP(timezone=True), nullable=False),
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
        sa.PrimaryKeyConstraint("halt_id", name="pk_execution_halt"),
        sa.ForeignKeyConstraint(
            ("clears_halt_id",),
            ("execution_halt.halt_id",),
            name="fk_execution_halt_clears_halt_id_execution_halt",
        ),
        # A halt is cleared at most once. NULLs are distinct in Postgres, so every
        # engagement — all of which carry NULL here — is left unconstrained. That
        # is the point: nothing may refuse a halt.
        sa.UniqueConstraint("clears_halt_id", name="uq_execution_halt_clears_halt_id"),
        sa.CheckConstraint("event IN ('engaged', 'cleared')", name="event_is_known"),
        # D-030 shape, both directions: the trigger names the condition on an
        # engagement and is absent on a clearance.
        sa.CheckConstraint(
            "(event = 'engaged') = (halt_trigger IS NOT NULL)", name="trigger_iff_engaged"
        ),
        sa.CheckConstraint(_HALT_TRIGGERS_SQL, name="trigger_is_known"),
        # The three clearance columns are present exactly on a clearance, stated
        # **per column**. The conjunction form this replaced only forbade an
        # engagement carrying all three; a stray clears_halt_id alone was
        # accepted and consumed the unique slot for that halt, making it
        # permanently un-clearable. See the class docstring in
        # backend/db/models.py for the full failure.
        sa.CheckConstraint(
            "(event = 'cleared') = (clears_halt_id IS NOT NULL) "
            "AND (event = 'cleared') = (cleared_by IS NOT NULL) "
            "AND (event = 'cleared') = (clearance_reason IS NOT NULL)",
            name="clearance_fields_iff_cleared",
        ),
        sa.CheckConstraint("cleared_by IS NULL OR cleared_by <> ''", name="cleared_by_present"),
        sa.CheckConstraint(
            "clearance_reason IS NULL OR clearance_reason <> ''", name="clearance_reason_present"
        ),
        sa.CheckConstraint("cycle_id <> ''", name="cycle_id_present"),
        sa.CheckConstraint("detail <> ''", name="detail_present"),
        sa.CheckConstraint(
            "git_commit ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'", name="git_commit_is_sha"
        ),
        sa.CheckConstraint("config_hash ~ '^[0-9a-f]{64}$'", name="config_hash_is_sha256"),
        sa.CheckConstraint("data_version <> ''", name="data_version_present"),
        sa.CheckConstraint("seed >= 0", name="seed_non_negative"),
    )
    # "Is anything open right now" is the question every release path asks, and
    # it asks it on every cycle. The partial index covers exactly the engagements.
    op.execute(
        "CREATE INDEX ix_execution_halt_open ON execution_halt (halt_id) WHERE event = 'engaged'"
    )
    op.execute(
        """
        CREATE FUNCTION execution_halt_clearance_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            target execution_halt%ROWTYPE;
            existing_clearance bigint;
        BEGIN
            -- An engagement is never examined and never refused. A guard that
            -- can reject a halt-engage row is a guard that can stop the kill
            -- switch from firing, and a redundant halt row costs nothing.
            IF NEW.event <> 'cleared' THEN
                RETURN NEW;
            END IF;
            SELECT * INTO target FROM execution_halt WHERE halt_id = NEW.clears_halt_id;
            IF target.halt_id IS NULL THEN
                RAISE EXCEPTION
                    'clearance names halt_id % which does not exist', NEW.clears_halt_id;
            END IF;
            IF target.event <> 'engaged' THEN
                RAISE EXCEPTION
                    'clearance names halt_id %, whose event is %; only an engagement can be '
                    'cleared', NEW.clears_halt_id, target.event;
            END IF;
            -- Already cleared. Raised as unique_violation on purpose: it is the
            -- same refusal the unique index gives, arriving by a different route,
            -- and the caller must not have to tell them apart. Under READ
            -- COMMITTED a writer whose INSERT starts after the winner committed
            -- reaches *this* branch rather than the index, because each statement
            -- takes a fresh snapshot (D-034).
            SELECT halt_id INTO existing_clearance FROM execution_halt
             WHERE event = 'cleared' AND clears_halt_id = NEW.clears_halt_id
             LIMIT 1;
            IF existing_clearance IS NOT NULL THEN
                RAISE EXCEPTION
                    'halt_id % was already cleared by halt_id %; a second clearance would '
                    'leave the log with two answers to who turned it back on',
                    NEW.clears_halt_id, existing_clearance
                    USING ERRCODE = 'unique_violation';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_execution_halt_clearance "
        "BEFORE INSERT ON execution_halt "
        "FOR EACH ROW EXECUTE FUNCTION execution_halt_clearance_guard()"
    )
    op.execute(
        """
        CREATE FUNCTION execution_control_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'execution control table % is append-only (P11.3/P11.5): % rejected; a '
                'reconciliation records what two snapshots said at a moment and a halt '
                'records something that happened, and neither stops being true later — a '
                'correction is a new reconciliation, a re-halt is a new engagement, and '
                'erasing either would destroy the trail an operator reasons backwards '
                'through (I2)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_execution_reconciliation_append_only "
        "BEFORE UPDATE OR DELETE ON execution_reconciliation "
        "FOR EACH ROW EXECUTE FUNCTION execution_control_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_execution_halt_append_only "
        "BEFORE UPDATE OR DELETE ON execution_halt "
        "FOR EACH ROW EXECUTE FUNCTION execution_control_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the triggers, their functions, the indices, and both tables."""
    for table in reversed(_APPEND_ONLY_TABLES):
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.execute("DROP FUNCTION execution_control_append_only_guard()")
    op.execute("DROP TRIGGER trg_execution_halt_clearance ON execution_halt")
    op.execute("DROP FUNCTION execution_halt_clearance_guard()")
    op.execute("DROP INDEX ix_execution_halt_open")
    op.drop_table("execution_halt")
    op.drop_index("ix_execution_reconciliation_cycle", table_name="execution_reconciliation")
    op.drop_table("execution_reconciliation")
