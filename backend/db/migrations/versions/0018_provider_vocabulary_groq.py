"""Widen the provider CHECK vocabulary to admit Groq (B4, D-047).

Revision ID: 0018
Revises: 0017
Create Date: 2026-08-04

Why a new revision rather than an edit to 0009
----------------------------------------------

The provider vocabulary is a CHECK constraint declared by revision 0009
(``llm_provider_credential``, ``extraction_model_assignment``) and 0013
(``llm_spend_ledger``). Adding ``groq`` to the enum could have been done by
editing those files, and a fresh database would have come out correct — which
is exactly what makes the temptation worth naming. **A migration records what
the schema was at its point in the chain, not what it is now.** Editing an
applied revision makes the chain describe a history that never happened: a
database migrated last week and a database created today would disagree about
what 0009 did, and nothing in the chain would say so. Every earlier revision
therefore keeps the vocabulary it shipped with, and each new member arrives as
its own widening.

The cost is that the vocabulary is now declared in four places rather than
three, and the drift tests in
``backend/tests/extraction/test_providers_registry.py`` are what keep them
honest: 0009 and 0013 are pinned to the *historical* set, this revision to the
*current* set, and the current set to the enum.

What this does
--------------

Drops and recreates ``provider_known`` on all three tables. Two of them
(``extraction_model_assignment``, ``llm_spend_ledger``) carry
``BEFORE UPDATE OR DELETE`` append-only row triggers; those are row triggers and
do not fire for DDL, so the constraint swap is unaffected by them. No row is
read, written, or moved.

There is a window, inside the transaction, in which no vocabulary constraint
exists on these tables. It is closed by the transaction: Postgres runs DDL
transactionally, so a concurrent writer either sees the old constraint or the
new one, never neither.

On the downgrade
----------------

Narrowing back will **fail loudly** if any row names ``groq`` by then, because
``ADD CONSTRAINT`` validates existing rows. That is the desired behaviour and
not a defect to work around: the alternative is a downgrade that silently
deletes an operator's credential or a settled spend record to make room for
itself. A downgrade that refuses is a downgrade that tells you the truth about
what is in the table.
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


_CONSTRAINT_NAME = "provider_known"

_PROVIDER_TABLES = (
    "llm_provider_credential",
    "extraction_model_assignment",
    "llm_spend_ledger",
)
"""Every table whose ``provider`` column is bound to the closed vocabulary.

Listed here rather than discovered, so that a table added later with a
``provider`` column and no entry in this tuple is a visible omission in the next
widening rather than a constraint that quietly falls behind.
"""


def _constraint_name(table: str) -> str:
    """Return the constraint's name **as the database spells it**.

    Revisions 0009 and 0013 declare this CHECK as the bare ``provider_known``
    and rely on the metadata naming convention
    (``ck_%(table_name)s_%(constraint_name)s``) to expand it at emit time. That
    expansion happens for a constraint attached to a table being *created*; a
    bare ``ALTER TABLE ... DROP CONSTRAINT provider_known`` gets no such
    treatment and fails with "constraint does not exist".

    Verified against rendered SQL rather than assumed: ``alembic upgrade 0009
    --sql`` emits ``CONSTRAINT ck_llm_provider_credential_provider_known``.
    ``test_the_widening_names_constraints_the_way_the_database_does`` holds this
    to the ORM metadata so it cannot drift.
    """
    return f"ck_{table}_{_CONSTRAINT_NAME}"


_PROVIDER_NAMES_SQL = "'anthropic', 'groq', 'openai'"
"""The vocabulary *as of this revision*. Must match ``backend.db.models``."""

_PREVIOUS_PROVIDER_NAMES_SQL = "'anthropic', 'openai'"
"""The vocabulary before this revision, for the downgrade."""


def _set_vocabulary(names_sql: str) -> None:
    """Point ``provider_known`` at *names_sql* on every provider-bearing table."""
    for table in _PROVIDER_TABLES:
        name = _constraint_name(table)
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")
        op.execute(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK (provider IN ({names_sql}))")


def upgrade() -> None:
    """Admit ``groq`` alongside the existing providers."""
    _set_vocabulary(_PROVIDER_NAMES_SQL)


def downgrade() -> None:
    """Refuse ``groq`` again; fails if any row already names it."""
    _set_vocabulary(_PREVIOUS_PROVIDER_NAMES_SQL)
