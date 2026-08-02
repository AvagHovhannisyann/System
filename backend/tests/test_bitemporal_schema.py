"""P2.2 unit tests: BitemporalMixin columns/constraints and the bitemporal registry.

These tests inspect SQLAlchemy metadata only — no database is needed. The
same schema is exercised against a real TimescaleDB in
``backend/tests/integration/``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, cast

import pytest
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DefaultClause,
    Numeric,
    Table,
    Text,
)
from sqlalchemy.types import DateTime, TypeDecorator

from backend.db.base import Base
from backend.db.bitemporal import (
    OBSERVATION_PAYLOAD_PRESENT,
    RETRACTION_PAYLOAD_ABSENT,
    TEMPORAL_COLUMNS,
    BitemporalMixin,
    bitemporal_classes,
    bitemporal_mappers,
    bitemporal_tables,
)
from backend.db.models import MacroObservation, PriceBar, Security, SecurityMaster

_TEMPORAL_COLUMNS = ("valid_from", "valid_to", "knowledge_time", "ingested_at")

_MIGRATION_0012 = (
    Path(__file__).resolve().parents[1]
    / "db"
    / "migrations"
    / "versions"
    / "0012_retraction_payload.py"
)


def _table(model: type) -> Table:
    """Return the mapped Table of a declarative model (typed for mypy)."""
    return cast("Table", cast("Any", model).__table__)


def _server_default_text(table: Table, column_name: str) -> str:
    """Return the rendered server-default expression of a column (asserting one exists)."""
    server_default = table.c[column_name].server_default
    assert server_default is not None, f"{column_name} must have a server default"
    assert isinstance(server_default, DefaultClause)
    return str(server_default.arg)


@pytest.mark.parametrize("model", [SecurityMaster, PriceBar])
def test_mixin_temporal_columns_are_timestamptz_not_null(model: type[BitemporalMixin]) -> None:
    """Every temporal column is a NOT NULL TIMESTAMPTZ, decorators included.

    The event-time columns are wrapped in the InfinityDateTime TypeDecorator
    (aware <-> PG 'infinity' round-trip); the DDL type is what must be
    TIMESTAMPTZ, so assert against the resolved implementation type rather
    than the outer decorator.
    """
    table = _table(model)
    for column_name in _TEMPORAL_COLUMNS:
        column = table.c[column_name]
        column_type = column.type
        ddl_type = (
            column_type.impl_instance if isinstance(column_type, TypeDecorator) else column_type
        )
        assert isinstance(ddl_type, DateTime), f"{column_name} must be a timestamp"
        assert ddl_type.timezone is True, f"{column_name} must be TIMESTAMPTZ (all UTC, D-011)"
        assert column.nullable is False, f"{column_name} must be NOT NULL"


@pytest.mark.parametrize("model", [SecurityMaster, PriceBar])
def test_knowledge_time_has_no_server_default(model: type[BitemporalMixin]) -> None:
    """D-011: knowledge_time is always supplied explicitly by writers — never defaulted."""
    column = _table(model).c["knowledge_time"]
    assert column.server_default is None
    assert column.default is None


@pytest.mark.parametrize("model", [SecurityMaster, PriceBar])
def test_audit_and_flag_defaults(model: type[BitemporalMixin]) -> None:
    """ingested_at defaults to now() server-side; valid_to to 'infinity'; is_retraction to false."""
    table = _table(model)
    assert "now" in _server_default_text(table, "ingested_at").lower()
    assert "infinity" in _server_default_text(table, "valid_to")
    retraction = table.c["is_retraction"]
    assert isinstance(retraction.type, Boolean)
    assert retraction.nullable is False
    assert "false" in _server_default_text(table, "is_retraction").lower()


@pytest.mark.parametrize("model", [SecurityMaster, PriceBar])
def test_valid_interval_check_constraint_present(model: type[BitemporalMixin]) -> None:
    """CHECK (valid_from < valid_to) — half-open interval sanity (D-011)."""
    checks = [c for c in _table(model).constraints if isinstance(c, CheckConstraint)]
    assert any("valid_from < valid_to" in str(c.sqltext) for c in checks)


@pytest.mark.parametrize("model", [SecurityMaster, PriceBar])
def test_primary_key_is_logical_key_plus_versioning_axes(model: type[BitemporalMixin]) -> None:
    """PK = (logical key, valid_from, knowledge_time): one row per version, ties impossible."""
    table = _table(model)
    expected = (*model.__bitemporal_key__, "valid_from", "knowledge_time")
    assert tuple(c.name for c in table.primary_key.columns) == expected


def _check_constraint(table: Table, unprefixed_name: str) -> CheckConstraint:
    """Return the named CHECK on ``table`` (naming convention applied), asserting it exists."""
    expected = f"ck_{table.name}_{unprefixed_name}"
    for constraint in table.constraints:
        if isinstance(constraint, CheckConstraint) and constraint.name == expected:
            return constraint
    message = f"{table.name} has no CHECK constraint {expected}"
    raise AssertionError(message)


def _migration_0012_statements() -> list[str]:
    """Return every SQL string literal revision 0012 passes to ``op.execute``."""
    tree = ast.parse(_MIGRATION_0012.read_text())
    return [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ]


@pytest.mark.parametrize("model", bitemporal_classes())
def test_payload_is_every_column_that_is_neither_key_nor_temporal(
    model: type[BitemporalMixin],
) -> None:
    """P2.10: ``__bitemporal_payload__`` is derived by subtraction, not by hand.

    Derived from the table itself so a fact table added by a later phase is
    covered without anyone editing this test — the same reasoning as the
    registry test below. A hand-maintained list would stop being the invariant
    the moment a column is added.
    """
    table = _table(model)
    expected = tuple(
        column.name
        for column in table.columns
        if column.name not in {*TEMPORAL_COLUMNS, *model.__bitemporal_key__}
    )
    assert model.__bitemporal_payload__ == expected
    assert set(model.__bitemporal_required_payload__) <= set(expected)


@pytest.mark.parametrize("model", bitemporal_classes())
def test_payload_columns_are_ddl_nullable(model: type[BitemporalMixin]) -> None:
    """A retraction has no payload, so payload columns must be able to hold NULL.

    The key and temporal columns stay NOT NULL: a retraction still addresses a
    fact, so it repeats the logical key and the valid interval.
    """
    table = _table(model)
    for name in model.__bitemporal_payload__:
        assert table.c[name].nullable is True, f"{table.name}.{name} must be DDL-nullable (P2.10)"
    for name in (*model.__bitemporal_key__, *TEMPORAL_COLUMNS):
        assert table.c[name].nullable is False, f"{table.name}.{name} must stay NOT NULL"


@pytest.mark.parametrize("model", bitemporal_classes())
def test_retraction_payload_absent_check_covers_every_payload_column(
    model: type[BitemporalMixin],
) -> None:
    """The CHECK that makes a retraction structurally incapable of carrying a value.

    Every payload column, not a subset: a single column left out is a column
    a retraction could still put a fabricated number in (I3).
    """
    table = _table(model)
    assert model.__bitemporal_payload__, (
        f"{table.name} has no payload columns; a fact table with nothing but key and "
        "temporal columns records no fact and the P2.10 constraints would be tautologies"
    )
    constraint = _check_constraint(table, RETRACTION_PAYLOAD_ABSENT)
    clauses = " AND ".join(f"{name} IS NULL" for name in model.__bitemporal_payload__)
    assert str(constraint.sqltext) == f"NOT is_retraction OR ({clauses})"


@pytest.mark.parametrize("model", bitemporal_classes())
def test_observation_payload_present_check_preserves_the_old_not_null(
    model: type[BitemporalMixin],
) -> None:
    """Observations keep exactly the guarantee ``NOT NULL`` gave them, no less.

    ``__bitemporal_required_payload__`` is read off the model's own
    ``Mapped[...]`` annotations, so this asserts the CHECK covers precisely the
    columns the model declares non-optional.
    """
    table = _table(model)
    constraint = _check_constraint(table, OBSERVATION_PAYLOAD_PRESENT)
    clauses = " AND ".join(
        f"{name} IS NOT NULL" for name in model.__bitemporal_required_payload__
    )
    assert str(constraint.sqltext) == f"is_retraction OR ({clauses})"


def test_optional_payload_columns_are_the_ones_the_source_may_not_state() -> None:
    """A column outside the required set is one the *source* may leave unstated.

    Pinned on two models with a genuinely optional payload so the distinction
    cannot quietly collapse into "everything is optional now", which is what a
    careless widening of the required set would look like.
    """
    assert SecurityMaster.__bitemporal_required_payload__ == ("ticker", "name", "exchange")
    assert set(SecurityMaster.__bitemporal_payload__) - set(
        SecurityMaster.__bitemporal_required_payload__
    ) == {"first_listed_on", "delisted_on"}
    # macro_observation.value is NULL exactly when FRED reported the "." marker,
    # which is knowledge, not absence of it (see the model docstring).
    assert MacroObservation.__bitemporal_required_payload__ == (
        "is_missing",
        "vintage_start_date",
    )


@pytest.mark.parametrize("model", bitemporal_classes())
def test_migration_0012_matches_the_model_payload_constraints(
    model: type[BitemporalMixin],
) -> None:
    """Revision 0012's DDL must state the same CHECK expressions the models do.

    The migration writes the expressions out verbatim (so they are greppable
    in the file that installs them) while the models generate them; this is
    what stops the two from drifting into a database whose constraints differ
    from the metadata every test above inspects.
    """
    table = _table(model)
    statements = _migration_0012_statements()
    for unprefixed in (RETRACTION_PAYLOAD_ABSENT, OBSERVATION_PAYLOAD_PRESENT):
        constraint = _check_constraint(table, unprefixed)
        fragment = f"CONSTRAINT {constraint.name} CHECK ({constraint.sqltext})"
        assert any(fragment in statement for statement in statements), (
            f"revision 0012 does not install {constraint.name} with the expression the "
            f"model declares: {fragment}"
        )
    for name in model.__bitemporal_required_payload__:
        fragment = f"ALTER COLUMN {name} DROP NOT NULL"
        assert any(
            fragment in statement and f"ALTER TABLE {table.name} " in statement
            for statement in statements
        ), f"revision 0012 does not drop NOT NULL on {table.name}.{name}"


def test_registry_contains_exactly_the_bitemporal_mappers() -> None:
    """The registry the query layer consumes lists every bitemporal mapper, nothing else.

    The expectation is *derived* from the ORM's own mapper registry rather than
    frozen as a list of models. A frozen list would say "these two tables
    exist", which stops being the invariant the moment a phase adds a fact
    table (P3.2 adds two) — and a registry that silently missed a new mapper is
    precisely what would let a fact table escape the as-of read enforcement, so
    the check has to keep biting as tables are added rather than be edited each
    time.
    """
    mapped_bitemporal = {
        mapper.class_
        for mapper in Base.registry.mappers
        if issubclass(mapper.class_, BitemporalMixin)
    }
    assert {SecurityMaster, PriceBar} <= mapped_bitemporal
    assert set(bitemporal_classes()) == mapped_bitemporal
    assert _table(Security).name not in {t.name for t in bitemporal_tables()}
    assert {m.class_ for m in bitemporal_mappers()} == mapped_bitemporal
    assert bitemporal_tables() == frozenset(_table(model) for model in mapped_bitemporal)


def test_bitemporal_key_declared_per_model() -> None:
    assert SecurityMaster.__bitemporal_key__ == ("security_id",)
    assert PriceBar.__bitemporal_key__ == ("security_id",)


def test_subclass_without_bitemporal_key_is_rejected() -> None:
    """A bitemporal model must declare its logical key or it cannot be defined at all."""
    with pytest.raises(TypeError, match="__bitemporal_key__"):

        class _MissingKey(BitemporalMixin, Base):  # pyright: ignore[reportUnusedClass]
            __tablename__ = "_missing_key_probe"

    assert "_missing_key_probe" not in Base.metadata.tables


def test_subclass_with_unknown_key_column_is_rejected() -> None:
    with pytest.raises(TypeError, match="no_such_column"):

        class _BadKey(BitemporalMixin, Base):  # pyright: ignore[reportUnusedClass]
            __tablename__ = "_bad_key_probe"
            __bitemporal_key__ = ("no_such_column",)

    assert "_bad_key_probe" not in Base.metadata.tables


def test_security_anchor_shape() -> None:
    """The identity anchor is a single surrogate-key column, generated by the database."""
    table = _table(Security)
    assert [c.name for c in table.columns] == ["security_id"]
    column = table.c["security_id"]
    assert isinstance(column.type, BigInteger)
    assert column.identity is not None


def test_price_bar_unit_bearing_columns() -> None:
    """Column names carry units (directive §8): *_usd, volume_shares, dimensionless factor.

    The price columns are DDL-nullable since P2.10 — a retraction has no
    payload to put in them — but every one of them is still *required on an
    observation*, which is asserted below via ``__bitemporal_required_payload__``
    rather than by ``NOT NULL``. The guarantee moved; it did not weaken.
    """
    table = _table(PriceBar)
    for name in ("open_usd", "high_usd", "low_usd", "close_usd", "close_raw_usd"):
        assert isinstance(table.c[name].type, Numeric)
        assert name in PriceBar.__bitemporal_required_payload__
    assert isinstance(table.c["volume_shares"].type, BigInteger)
    assert isinstance(table.c["adjustment_factor"].type, Numeric)
    fks = list(table.c["security_id"].foreign_keys)
    assert len(fks) == 1
    assert fks[0].column.table.name == "security"


def test_security_master_identity_columns() -> None:
    table = _table(SecurityMaster)
    assert isinstance(table.c["ticker"].type, Text)
    assert isinstance(table.c["name"].type, Text)
    assert isinstance(table.c["exchange"].type, Text)
    assert isinstance(table.c["first_listed_on"].type, Date)
    assert isinstance(table.c["delisted_on"].type, Date)
    assert table.c["first_listed_on"].nullable is True
    assert table.c["delisted_on"].nullable is True
