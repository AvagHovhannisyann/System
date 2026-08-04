"""Migration 0011 against the ORM declarations it has to match (no database).

Alembic revisions in this project are hand-written, so nothing automatically
keeps a revision and its models in step. A divergence between them does not fail
at import time and does not fail in unit tests — it fails when someone runs
``alembic upgrade head`` against a real database and gets a schema the ORM cannot
map, which is usually a long way from whoever caused it. So the columns,
constraints and their exact names are compared here, structurally.

What the statements *do* — the append-only triggers rejecting an UPDATE, the
unique constraint rejecting a duplicate build — needs Postgres and lives in
``backend/tests/integration/test_universe_db.py``.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import cast

import sqlalchemy as sa

from backend.db import models

_SNAPSHOT_TABLE = cast("sa.Table", models.UniverseSnapshot.__table__)
_MEMBER_TABLE = cast("sa.Table", models.UniverseMember.__table__)

_APPEND_ONLY_SHAPE = "BEFORE UPDATE OR DELETE"


def _revision(module_name: str) -> ModuleType:
    return importlib.import_module(f"backend.db.migrations.versions.{module_name}")


def _source() -> str:
    module = _revision("0011_universe_snapshot")
    return Path(str(module.__file__)).read_text()


def test_0011_follows_0010_in_a_linear_chain() -> None:
    revision = _revision("0011_universe_snapshot")
    assert revision.revision == "0011"
    assert revision.down_revision == _revision("0010_extraction_framework").revision


def test_the_revision_creates_exactly_the_two_universe_tables() -> None:
    source = _source()
    for table in ("universe_snapshot", "universe_member"):
        assert f'op.create_table(\n        "{table}"' in source


def test_both_tables_carry_the_append_only_trigger() -> None:
    # The statements are written out in full rather than generated from a loop
    # so that grepping for the trigger shape finds every append-only table. The
    # count is asserted too: a third table added without a trigger, or a trigger
    # added for a table nobody declared, both show up here.
    source = _source()
    assert source.count(f"{_APPEND_ONLY_SHAPE} ON ") == 2
    assert f"{_APPEND_ONLY_SHAPE} ON universe_snapshot" in source
    assert f"{_APPEND_ONLY_SHAPE} ON universe_member" in source


def test_truncate_is_deliberately_not_blocked() -> None:
    # Matching revisions 0003/0004/0007/0009/0010: TRUNCATE is the sanctioned
    # admin and test reset path, and it never masquerades as an edit.
    assert "OR TRUNCATE" not in _source()


def test_neither_table_is_bitemporal() -> None:
    # A snapshot is a computation *we* ran over facts that are already
    # bitemporal; a knowledge_time invented for it would be a fabricated value
    # in the one column whose meaning is that it is not fabricated.
    for table in (_SNAPSHOT_TABLE, _MEMBER_TABLE):
        assert {"valid_from", "valid_to", "knowledge_time"}.isdisjoint(table.columns.keys())


def test_the_snapshot_identity_is_unique() -> None:
    constraint = next(
        item for item in _SNAPSHOT_TABLE.constraints if isinstance(item, sa.UniqueConstraint)
    )
    assert constraint.name == "uq_universe_snapshot_build"
    assert [column.name for column in constraint.columns] == [
        "rebalance_date",
        "criteria_hash",
        "as_of",
    ]
    declared = 'sa.UniqueConstraint("rebalance_date","criteria_hash","as_of"'
    assert declared in re.sub(r"\s+", "", _source())


def test_every_declared_column_appears_in_the_revision() -> None:
    # Whitespace-normalised so a column the formatter split across lines still
    # matches; the assertion is about the declaration existing, not its layout.
    source = re.sub(r"\s+", "", _source())
    for table in (_SNAPSHOT_TABLE, _MEMBER_TABLE):
        for column in table.columns:
            assert f'sa.Column("{column.name}"' in source, (table.name, column.name)


def test_every_declared_check_constraint_appears_in_the_revision() -> None:
    # The names are compared, not only the expressions: the metadata naming
    # convention expands an unprefixed name, and a revision that spells the name
    # out in full would create a constraint the ORM does not know about.
    source = _source()
    for table in (_SNAPSHOT_TABLE, _MEMBER_TABLE):
        for constraint in table.constraints:
            if isinstance(constraint, sa.CheckConstraint):
                unprefixed = str(constraint.name).removeprefix(f"ck_{table.name}_")
                assert f'name="{unprefixed}"' in source, constraint.name


def test_the_member_table_points_at_the_identity_anchor_not_the_versioned_master() -> None:
    # A versioned table's logical key is not unique per row, so it cannot be a
    # foreign-key target (D-011).
    targets = {foreign_key.column.table.name for foreign_key in _MEMBER_TABLE.foreign_keys}
    assert targets == {"security", "universe_snapshot"}
    assert '"security.security_id"' in _source()
    assert "security_master.security_id" not in _source()


def test_the_revision_indexes_the_two_access_patterns_the_package_uses() -> None:
    source = _source()
    # load_snapshots filters on criteria_hash first; the unique constraint leads
    # with rebalance_date and cannot serve that.
    assert '"ix_universe_snapshot_criteria_lookup"' in source
    assert '["criteria_hash", "rebalance_date", "as_of"]' in source
    # "which snapshots was this name in" — the primary key leads with snapshot_id.
    assert '"ix_universe_member_security"' in source


def test_the_downgrade_drops_everything_the_upgrade_created() -> None:
    source = _source()
    for statement in (
        'op.drop_table("universe_member")',
        'op.drop_table("universe_snapshot")',
        'op.drop_index("ix_universe_member_security"',
        'op.drop_index("ix_universe_snapshot_criteria_lookup"',
        "DROP FUNCTION universe_append_only_guard()",
    ):
        assert statement in source
