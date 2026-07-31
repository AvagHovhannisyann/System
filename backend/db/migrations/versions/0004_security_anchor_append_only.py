"""Append-only triggers on the ``security`` identity anchor (perimeter hardening).

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-31

The identity anchor is documented as create-once-never-change
(``backend/db/models.py``): every descriptive attribute lives in the
bitemporal ``security_master``, and fact rows reference ``security_id`` by
foreign key. Revision 0003 enforced append-only on the *fact* tables only;
this revision closes the perimeter by giving the anchor the same
BEFORE UPDATE OR DELETE row-trigger treatment — a mutated or deleted anchor
would silently re-key or orphan versioned history, which no role should be
able to do by convention alone.

A dedicated trigger function (rather than reusing
``bitemporal_append_only_guard``) keeps the error message accurate: the
anchor is append-only but *not* bitemporal — there is no
"correction with a later knowledge_time" for it. TRUNCATE stays deliberately
unblocked (0003 rationale): it is the sanctioned admin/test reset path.

Downgrade drops the trigger and function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the BEFORE UPDATE OR DELETE append-only trigger to ``security``."""
    op.execute(
        """
        CREATE FUNCTION security_anchor_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'identity anchor table % is append-only: % rejected; anchors are '
                'created once per entity and never change — versioned attributes '
                'belong in security_master (D-011)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_security_append_only "
        "BEFORE UPDATE OR DELETE ON security "
        "FOR EACH ROW EXECUTE FUNCTION security_anchor_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the anchor append-only trigger and its function."""
    op.execute("DROP TRIGGER trg_security_append_only ON security")
    op.execute("DROP FUNCTION security_anchor_append_only_guard()")
