"""Point-in-time universe snapshots and per-candidate screening outcomes (P4.1/P4.2).

Revision ID: 0011
Revises: 0010
Create Date: 2026-08-02

Two tables, neither bitemporal, neither a hypertable, both **append-only** by
the same ``BEFORE UPDATE OR DELETE`` row-trigger shape revisions
0003/0004/0007/0009/0010 use:

- ``universe_snapshot`` — one row per execution of
  :func:`backend.universe.builder.build_universe`. Identity is
  ``(rebalance_date, criteria_hash, as_of)`` and it is UNIQUE: those three
  determine the universe completely given the contents of the bitemporal store,
  which is the reproducibility claim I2 asks for. Re-running after more data has
  been ingested is a *different* ``as_of`` and therefore a new row rather than a
  correction — which is what makes "the universe as we knew it on date X"
  answerable after the fact.
- ``universe_member`` — one row per **candidate considered**, not per member.
  §6.3 requires a filter-impact waterfall showing how many names each screen
  removed, and a table holding only survivors cannot answer that: a universe of
  480 names says nothing about whether the market-cap floor removed 40 names or
  4,000. Rebuilding the answer later is not an option either, because rebuilding
  requires the store as it stood at the original ``as_of``. So an excluded name
  gets a row carrying ``failed_filters``, and the CHECK below ties ``included``
  to that list being empty so the two can never come to disagree.

Neither is bitemporal, and that is deliberate. The bitemporal columns describe
when a fact was true in the world and when it became knowable to the *market*
(D-011). A universe snapshot is a computation **we** ran over facts that are
already bitemporal: its inputs carry the knowledge times, and it carries the
``as_of`` instant it read them at, which is the whole of its point-in-time
content. A ``knowledge_time`` invented for it would be a fabricated value in the
one column whose meaning is that it is not fabricated (I3). Same reasoning as
revisions 0005, 0007, 0009 and 0010, so these tables are absent from the
bitemporal registry, unscoped by the Core-level read guard, and read without an
as-of.

Neither is a hypertable either: rebalance dates are monthly-to-weekly, so a
decade of history is hundreds of snapshot rows, not an event-time stream worth
chunking.

The foreign key on ``universe_member.security_id`` targets the ``security``
identity anchor rather than the bitemporal ``security_master``: a versioned
table's logical key is not unique per row, so it cannot be a foreign-key target
(D-011). Which *version* of the identity was in force is a function of the
snapshot's ``rebalance_date`` and ``as_of`` and is re-derivable through
``as_of()``.

TRUNCATE stays unblocked on both, matching 0003/0004/0007/0009/0010: it is the
sanctioned admin/test reset path and never masquerades as an edit.

Downgrade drops the triggers, their function, the indices and both tables in
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
revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = ("universe_snapshot", "universe_member")
"""Tables that get the append-only trigger, in creation order.

Named here *and* spelled out in full ``CREATE TRIGGER`` statements below rather
than generated from this tuple. The duplication is deliberate: "which tables in
this schema are append-only" is a question a human answers by grepping for
``BEFORE UPDATE OR DELETE ON``, and a loop that assembles the statement makes
that grep return nothing. Revisions 0003, 0004, 0007, 0009 and 0010 write theirs
out for the same reason, and a test asserts the statements are present.
"""


def upgrade() -> None:
    """Create both universe tables, their constraints, indices, and append-only triggers."""
    op.create_table(
        "universe_snapshot",
        sa.Column("snapshot_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("rebalance_date", sa.Date(), nullable=False),
        sa.Column("criteria_hash", sa.Text(), nullable=False),
        sa.Column("criteria", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("as_of", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("candidate_count", sa.Integer(), nullable=False),
        sa.Column("member_count", sa.Integer(), nullable=False),
        sa.Column(
            "built_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("snapshot_id", name="pk_universe_snapshot"),
        # The snapshot's identity (module docstring). Also the uniqueness that
        # turns a repeated build into an IntegrityError instead of a second row
        # nobody could choose between.
        sa.UniqueConstraint(
            "rebalance_date", "criteria_hash", "as_of", name="uq_universe_snapshot_build"
        ),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005, 0007, 0009, 0010).
        sa.CheckConstraint("criteria_hash <> ''", name="criteria_hash_not_empty"),
        sa.CheckConstraint("candidate_count >= 0", name="candidate_count_non_negative"),
        sa.CheckConstraint("member_count >= 0", name="member_count_non_negative"),
        # A member is a candidate that failed nothing, so members can never
        # outnumber the names considered. A violation here means the screen and
        # the counts disagree, which is the one arithmetic error that would make
        # every waterfall built from this row wrong.
        sa.CheckConstraint("member_count <= candidate_count", name="members_within_candidates"),
    )
    # backend.universe.snapshot.load_snapshots filters on criteria_hash first,
    # then bounds rebalance_date and as_of. The UNIQUE constraint above leads
    # with rebalance_date, so it cannot serve that access pattern; this index
    # is the one that does. Physical layout lives in the migration rather than
    # the ORM, as in revisions 0003 and 0010.
    op.create_index(
        "ix_universe_snapshot_criteria_lookup",
        "universe_snapshot",
        ["criteria_hash", "rebalance_date", "as_of"],
    )
    op.create_table(
        "universe_member",
        sa.Column("snapshot_id", sa.BigInteger(), nullable=False),
        sa.Column("security_id", sa.BigInteger(), nullable=False),
        sa.Column("included", sa.Boolean(), nullable=False),
        sa.Column(
            "failed_filters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.PrimaryKeyConstraint("snapshot_id", "security_id", name="pk_universe_member"),
        sa.ForeignKeyConstraint(
            ("snapshot_id",),
            ("universe_snapshot.snapshot_id",),
            name="fk_universe_member_snapshot_id_universe_snapshot",
        ),
        sa.ForeignKeyConstraint(
            ("security_id",),
            ("security.security_id",),
            name="fk_universe_member_security_id_security",
        ),
        # Membership and the screening record are the same statement: a member
        # is exactly a candidate that failed nothing. Two independently settable
        # columns would permit "an included name that failed the price screen",
        # which no consumer — least of all the waterfall — could act on.
        sa.CheckConstraint(
            "included = (jsonb_array_length(failed_filters) = 0)",
            name="included_iff_no_failures",
        ),
    )
    # "Which snapshots was this name a candidate in" — §6.3's constituent
    # entry/exit history and the point-in-time browser's per-name view. The
    # primary key leads with snapshot_id and cannot answer it.
    op.create_index("ix_universe_member_security", "universe_member", ["security_id"])
    op.execute(
        """
        CREATE FUNCTION universe_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'universe table % is append-only (P4.1): % rejected; a snapshot records '
                'what the screens decided at one as_of instant, and re-running after more '
                'data arrived is a new snapshot rather than a correction — overwriting one '
                'would destroy the evidence that the answer changed (I2, DIRECTIVE 5-P4)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_universe_snapshot_append_only "
        "BEFORE UPDATE OR DELETE ON universe_snapshot "
        "FOR EACH ROW EXECUTE FUNCTION universe_append_only_guard()"
    )
    op.execute(
        "CREATE TRIGGER trg_universe_member_append_only "
        "BEFORE UPDATE OR DELETE ON universe_member "
        "FOR EACH ROW EXECUTE FUNCTION universe_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the append-only triggers, their function, the indices, and both tables."""
    for table in reversed(_APPEND_ONLY_TABLES):
        op.execute(f"DROP TRIGGER trg_{table}_append_only ON {table}")
    op.execute("DROP FUNCTION universe_append_only_guard()")
    op.drop_index("ix_universe_member_security", table_name="universe_member")
    op.drop_table("universe_member")
    op.drop_index("ix_universe_snapshot_criteria_lookup", table_name="universe_snapshot")
    op.drop_table("universe_snapshot")
