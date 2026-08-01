"""P2.2 unit tests: BitemporalMixin columns/constraints and the bitemporal registry.

These tests inspect SQLAlchemy metadata only — no database is needed. The
same schema is exercised against a real TimescaleDB in
``backend/tests/integration/``.
"""

from __future__ import annotations

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
    BitemporalMixin,
    bitemporal_classes,
    bitemporal_mappers,
    bitemporal_tables,
)
from backend.db.models import PriceBar, Security, SecurityMaster

_TEMPORAL_COLUMNS = ("valid_from", "valid_to", "knowledge_time", "ingested_at")


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
    """Column names carry units (directive §8): *_usd, volume_shares, dimensionless factor."""
    table = _table(PriceBar)
    for name in ("open_usd", "high_usd", "low_usd", "close_usd", "close_raw_usd"):
        assert isinstance(table.c[name].type, Numeric)
        assert table.c[name].nullable is False
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
