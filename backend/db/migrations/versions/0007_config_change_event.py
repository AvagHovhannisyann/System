"""Configuration audit log: append-only ``config_change_event`` (CC.1, §6.11).

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-01

One row per configuration change — who (``actor``), when (``recorded_at``, from
the database clock), what changed (``scope``/``target``/``field``), the previous
value and the new value, plus the ``correlation_id`` of the request that made
it. Configuration has no mutable table anywhere in this system: the current
value of a key is the ``new_value`` of its most recent event
(``backend.db.audit.current_value``), so a change that is not in this table did
not happen.

**Deliberately not a bitemporal fact table.** The bitemporal columns describe
when a fact was true in the world and when it became knowable to the market
(D-011); a configuration change is something *we* did to our own system, has no
market knowability, and any ``knowledge_time`` invented for it would be a
fabricated value in the one column whose meaning is that it is not fabricated
(I3). The full reasoning is in the module docstring
(``backend/db/audit.py``). The table is therefore absent from the bitemporal
registry and is read without an as-of.

**Append-only, enforced the way revisions 0003/0004 enforce it**: a
``BEFORE UPDATE OR DELETE`` FOR EACH ROW trigger, role-independent, so no
session or role edits a past event. A dedicated trigger function rather than
reuse of ``bitemporal_append_only_guard`` keeps the message accurate — this
table is append-only but *not* bitemporal, so "corrections are new rows with a
later knowledge_time" would be the wrong advice; the right advice is that a
later configuration change is a later event.

TRUNCATE stays unblocked, matching 0003/0004: it is the sanctioned admin/test
reset path and never masquerades as an edit.

**This is append-only, not immutable, and §6.11's word is not yet true.** Per
D-012 the stack runs a single database role that owns this table and this
trigger, and can ``DISABLE``/``DROP`` the trigger or ``TRUNCATE`` the log. D-017
makes role separation (CC.9) a prerequisite for presenting this log to the
operator as trustworthy. Recorded here as well as in the module so a reader of
the schema alone is not misled.

Indices: ``(scope, target, field, event_id DESC)`` is exactly the
``DISTINCT ON``/``ORDER BY`` shape of the current-value reconstruction;
``recorded_at DESC`` serves CC.4's newest-first browser; the partial index on
``correlation_id`` answers "what did this request change" without scanning the
rows that were not made under a request.

Downgrade drops the trigger, its function and the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create config_change_event with its constraints, indices and append-only trigger."""
    op.create_table(
        "config_change_event",
        sa.Column("event_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("target", sa.Text(), nullable=False),
        sa.Column("field", sa.Text(), nullable=False),
        # Both NOT NULL: a value of None is JSON null, never SQL NULL. SQL NULL
        # and JSON null both decode to Python None, so using SQL NULL to mean
        # "there was no previous value" would be a distinction invisible in
        # Python — is_initial carries that instead.
        sa.Column("previous_value", postgresql.JSONB(), nullable=False),
        sa.Column("new_value", postgresql.JSONB(), nullable=False),
        sa.Column("is_initial", sa.Boolean(), nullable=False),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("event_id", name="pk_config_change_event"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_config_change_event_ck_... and
        # diverge from the names the ORM model declares (same as revision 0005).
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        sa.CheckConstraint("scope <> ''", name="scope_not_empty"),
        sa.CheckConstraint("target <> ''", name="target_not_empty"),
        sa.CheckConstraint("field <> ''", name="field_not_empty"),
        sa.CheckConstraint(
            "NOT is_initial OR previous_value = 'null'::jsonb",
            name="initial_event_has_no_previous_value",
        ),
    )
    op.execute(
        "CREATE INDEX ix_config_change_event_key "
        "ON config_change_event (scope, target, field, event_id DESC)"
    )
    op.execute(
        "CREATE INDEX ix_config_change_event_recorded_at ON config_change_event (recorded_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_config_change_event_correlation_id "
        "ON config_change_event (correlation_id) WHERE correlation_id IS NOT NULL"
    )
    op.execute(
        """
        CREATE FUNCTION config_event_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'configuration audit table % is append-only: % rejected; a '
                'configuration change is recorded as a new event, never by editing '
                'or removing a past one (DIRECTIVE 6.11, CC.1)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_config_change_event_append_only "
        "BEFORE UPDATE OR DELETE ON config_change_event "
        "FOR EACH ROW EXECUTE FUNCTION config_event_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the append-only trigger, its function, and the audit table."""
    op.execute("DROP TRIGGER trg_config_change_event_append_only ON config_change_event")
    op.execute("DROP FUNCTION config_event_append_only_guard()")
    op.drop_table("config_change_event")
