"""Baseline — empty anchor revision.

Revision ID: 0001
Revises: (root)
Create Date: 2026-07-31

Establishes the root of the migration chain so every later revision has a
stable ``down_revision``. Intentionally performs no schema changes; the
first real tables arrive with Phase 2 (bitemporal store).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the baseline revision — intentionally a no-op."""


def downgrade() -> None:
    """Revert the baseline revision — intentionally a no-op."""
