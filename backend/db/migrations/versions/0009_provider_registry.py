"""LLM provider registry: encrypted keys and versioned per-task model assignments (P7.1).

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-01

Two tables, neither bitemporal and neither a hypertable:

- ``llm_provider_credential`` — one row per provider, holding the API key as a
  Fernet token (§7: encrypted at rest, KEK from the environment) plus the
  permanently masked rendering the operator sees. **Deliberately mutable**: a
  rotation replaces the row's ciphertext and a deletion removes the row, so a
  retired key stops being decryptable. Keeping superseded ciphertext as history
  would widen a KEK compromise from "every key in use" to "every key ever
  used"; the history §6.5 asks for is the record of the operator's *actions*,
  which lives in the append-only ``config_change_event`` table as masked values
  and is safe to keep forever precisely because nothing in it is recoverable.
- ``extraction_model_assignment`` — one row per **version** of a task's
  assignment (§6.5: "changing an assignment creates a new configuration version
  rather than mutating the current one"). Append-only, enforced by a
  ``BEFORE UPDATE OR DELETE`` row trigger in the same shape revisions
  0003/0004/0007 use, so a superseded version cannot be edited or removed by
  any session or role.

Neither table is bitemporal, deliberately. The bitemporal columns describe when
a fact was true in the world and when it became knowable to the market (D-011);
an operator's credential and an operator's model assignment are things *we* did
to our own system, have no market knowability, and any ``knowledge_time``
invented for them would be a fabricated value in the one column whose meaning
is that it is not fabricated (I3). Same reasoning as revisions 0005 and 0007.
They are therefore absent from the bitemporal registry, unscoped by the
Core-level read guard, and read without an as-of.

Neither is a hypertable either: one row per provider and a handful of versions
per task is not time-series volume, and there is no event-time axis to chunk on.

The provider vocabulary is pinned by CHECK constraints rather than left open.
Adding a provider is a code *and* migration change on purpose — a provider the
platform cannot construct a request for is a provider it cannot use, so a bare
database row would never be sufficient. ``backend/db/models.py`` declares the
same vocabulary, and a unit test asserts it matches the ``Provider`` enum.

TRUNCATE stays unblocked on the append-only table, matching 0003/0004/0007: it
is the sanctioned admin/test reset path and never masquerades as an edit.

Downgrade drops the trigger, its function and both tables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PROVIDER_NAMES_SQL = "'anthropic', 'openai'"
"""Provider vocabulary; mirrors ``backend.db.models.PROVIDER_NAMES_SQL`` (test-asserted)."""


def upgrade() -> None:
    """Create the credential and assignment tables with their constraints and trigger."""
    op.create_table(
        "llm_provider_credential",
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("ciphertext", sa.Text(), nullable=False),
        sa.Column("masked_display", sa.Text(), nullable=False),
        sa.Column("key_version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "rotated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("provider", name="pk_llm_provider_credential"),
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would produce ck_..._ck_... and diverge from the
        # names the ORM model declares (same as revisions 0005 and 0007).
        sa.CheckConstraint(f"provider IN ({_PROVIDER_NAMES_SQL})", name="provider_known"),
        sa.CheckConstraint("ciphertext <> ''", name="ciphertext_not_empty"),
        sa.CheckConstraint("masked_display <> ''", name="masked_display_not_empty"),
        sa.CheckConstraint("key_version >= 1", name="key_version_positive"),
    )
    op.create_table(
        "extraction_model_assignment",
        sa.Column("assignment_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("task", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("temperature", sa.Numeric(4, 3), nullable=False),
        sa.Column("max_tokens", sa.Integer(), nullable=False),
        sa.Column("timeout_s", sa.Numeric(6, 3), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("correlation_id", sa.Text(), nullable=True),
        sa.Column(
            "recorded_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("assignment_id", name="pk_extraction_model_assignment"),
        # The version sequence per task is the whole point of the table (§6.5):
        # this unique constraint is what makes "version 3 of this task" a single
        # unambiguous row, and its index is also the lookup path for "the
        # greatest version of this task", so no separate index is created.
        sa.UniqueConstraint("task", "version", name="uq_extraction_model_assignment_task_version"),
        sa.CheckConstraint("task <> ''", name="task_not_empty"),
        sa.CheckConstraint("model <> ''", name="model_not_empty"),
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        sa.CheckConstraint(f"provider IN ({_PROVIDER_NAMES_SQL})", name="provider_known"),
        sa.CheckConstraint("version >= 1", name="version_positive"),
        sa.CheckConstraint("temperature >= 0 AND temperature <= 2", name="temperature_in_range"),
        sa.CheckConstraint("max_tokens >= 1", name="max_tokens_positive"),
        sa.CheckConstraint("timeout_s > 0", name="timeout_positive"),
    )
    op.execute(
        """
        CREATE FUNCTION model_assignment_append_only_guard() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION
                'model assignment table % is append-only: % rejected; changing a '
                'task''s assignment records a new version, never an edit to an '
                'existing one (DIRECTIVE 6.5, P7.1)',
                TG_TABLE_NAME, TG_OP;
        END;
        $$
        """
    )
    op.execute(
        "CREATE TRIGGER trg_extraction_model_assignment_append_only "
        "BEFORE UPDATE OR DELETE ON extraction_model_assignment "
        "FOR EACH ROW EXECUTE FUNCTION model_assignment_append_only_guard()"
    )


def downgrade() -> None:
    """Drop the append-only trigger, its function, and both registry tables."""
    op.execute(
        "DROP TRIGGER trg_extraction_model_assignment_append_only ON extraction_model_assignment"
    )
    op.execute("DROP FUNCTION model_assignment_append_only_guard()")
    op.drop_table("extraction_model_assignment")
    op.drop_table("llm_provider_credential")
