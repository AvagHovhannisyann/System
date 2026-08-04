"""P7.7: migration 0013 and the ORM agree, and the durable ledger's SQL is what it claims.

This repository hand-writes its migrations rather than autogenerating them, so
the migration and the ORM model are two independent statements of one schema and
their agreement is asserted rather than assumed — the same reasoning
``backend/tests/integration/test_providers_db.py`` records for revision 0009.

**What is unrun here, and why.** There is no Docker daemon in this environment,
so nothing below reaches PostgreSQL: no table is created, no trigger fires, no
advisory lock is taken. These are structural and construction-level checks, and
they are labelled as such. The behaviour they cannot reach — that the trigger
actually refuses an UPDATE, that the advisory lock actually serializes two
connections — needs the integration suite, and is reported as unrun rather than
claimed.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Protocol, cast

import sqlalchemy as sa

from backend.db.models import PROVIDER_NAMES_SQL, LlmSpendLedger
from backend.extraction.governor.caps import SpendWindow
from backend.extraction.governor.postgres import (
    PostgresSpendLedger,
    _provider_lock_key,
    committed_statement,
)
from backend.extraction.providers.catalog import Provider

if TYPE_CHECKING:
    from collections.abc import Callable

_MIGRATION = "0013_llm_spend_ledger"


class _MigrationModule(Protocol):
    """The parts of a migration revision this file reads.

    A protocol rather than ``ModuleType`` so the attributes are named and typed:
    ``importlib`` returns an untyped module, and reading ``revision`` off it
    would otherwise be unchecked.
    """

    revision: str
    down_revision: str | None
    op: object
    upgrade: Callable[[], None]
    _PROVIDER_NAMES_SQL: str


def _load() -> _MigrationModule:
    """Import the migration module by its revision file name."""
    return cast(
        "_MigrationModule",
        importlib.import_module(f"backend.db.migrations.versions.{_MIGRATION}"),
    )


def _table() -> sa.Table:
    """Return the ORM table for the spend ledger."""
    table = LlmSpendLedger.__table__
    assert isinstance(table, sa.Table)
    return table


class _RecordingOp:
    """Captures what a migration's ``upgrade()`` declares, without a database.

    Substituted for ``alembic.op`` so the migration's own statements can be
    inspected. This checks the *declaration*, which is the half that can be
    checked without a server; the other half — whether PostgreSQL accepts and
    enforces it — is what the integration suite exists for.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.tables: dict[str, tuple[sa.schema.SchemaItem, ...]] = {}
        self.indexes: list[tuple[str, str, tuple[str, ...]]] = []
        self.statements: list[str] = []

    def create_table(self, name: str, *elements: sa.schema.SchemaItem) -> None:
        """Record a table declaration."""
        self.tables[name] = elements

    def create_index(self, name: str, table: str, columns: list[str], **_: object) -> None:
        """Record an index declaration."""
        self.indexes.append((name, table, tuple(columns)))

    def execute(self, statement: object) -> None:
        """Record raw SQL."""
        self.statements.append(str(statement))


def _declared() -> _RecordingOp:
    """Run migration 0013's ``upgrade()`` against a recorder and return it."""
    module = _load()
    recorder = _RecordingOp()
    original = module.op
    module.op = recorder
    try:
        module.upgrade()
    finally:
        module.op = original
    return recorder


# --------------------------------------------------------------------------
# Chain position
# --------------------------------------------------------------------------


def test_0013_follows_0012() -> None:
    module = _load()
    assert module.revision == "0013"
    assert module.down_revision == "0012"


# --------------------------------------------------------------------------
# Migration and ORM describe the same table
# --------------------------------------------------------------------------


def test_the_migration_declares_exactly_the_orm_columns_with_the_same_types() -> None:
    """Two independent statements of one schema; drift fails here, not in production."""
    elements = _declared().tables["llm_spend_ledger"]
    declared = {element.name: element for element in elements if isinstance(element, sa.Column)}
    mapped = {column.name: column for column in _table().columns}

    assert set(declared) == set(mapped)
    for name, column in mapped.items():
        assert str(declared[name].type) == str(column.type), name
        assert declared[name].nullable == column.nullable, name


def test_the_migration_and_the_orm_declare_the_same_named_constraints() -> None:
    elements = _declared().tables["llm_spend_ledger"]
    declared_checks = {
        element.name for element in elements if isinstance(element, sa.CheckConstraint)
    }
    # The ORM metadata naming convention expands unprefixed CHECK names, so the
    # comparison is made on the unprefixed form the two files actually spell.
    mapped_checks = {
        constraint.name
        for constraint in _table().constraints
        if isinstance(constraint, sa.CheckConstraint) and isinstance(constraint.name, str)
    }
    unprefixed = {name.removeprefix("ck_llm_spend_ledger_") for name in mapped_checks}

    assert declared_checks == unprefixed
    assert "delta_matches_event" in declared_checks
    assert "degraded_iff_substituted" in declared_checks
    assert "reconciled_iff_measured" in declared_checks


def test_a_reservation_may_be_settled_or_released_only_once_in_the_schema() -> None:
    """The application refuses it too; this is the half that holds across processes."""
    elements = _declared().tables["llm_spend_ledger"]
    declared = {element.name for element in elements if isinstance(element, sa.UniqueConstraint)}
    mapped = {
        constraint.name: tuple(constraint.columns.keys())
        for constraint in _table().constraints
        if isinstance(constraint, sa.UniqueConstraint)
    }

    assert declared == set(mapped)
    assert mapped["uq_llm_spend_ledger_reservation_event"] == ("reservation_id", "event")


def test_the_window_lookups_are_indexed() -> None:
    """Both are read on the path to every call."""
    indexes = {name: columns for name, _, columns in _declared().indexes}
    assert indexes["ix_llm_spend_ledger_provider_daily"] == ("provider", "daily_window")
    assert indexes["ix_llm_spend_ledger_provider_monthly"] == ("provider", "monthly_window")


def test_the_table_is_append_only_by_trigger() -> None:
    """A spend event that can be edited afterwards is not evidence."""
    statements = " ".join(_declared().statements)
    assert "BEFORE UPDATE OR DELETE ON llm_spend_ledger" in statements
    assert "CREATE FUNCTION llm_spend_ledger_append_only_guard()" in statements


def test_the_enum_and_the_orm_spell_one_vocabulary() -> None:
    """Drift between the two live spellings fails the suite rather than a database."""
    from_models = {value.strip().strip("'") for value in PROVIDER_NAMES_SQL.split(",")}
    from_enum = {member.value for member in Provider}
    assert from_models == from_enum


def test_this_migrations_vocabulary_is_frozen_and_the_current_one_only_widens_it() -> None:
    """0013 keeps what it shipped, and the vocabulary may grow but never shrink.

    This used to assert 0013 == ORM == enum. It no longer can, and the reason is
    the point: a migration records the schema at its own point in the chain, so
    when a provider is added the earlier revisions keep their set and a new
    revision widens it (0018 for ``groq``). Re-pointing this at the enum would
    have required editing an applied migration.

    The replacement is not weaker. *Frozen* pins 0013 against a helpful edit, and
    the **superset** clause asserts something the old equality could not: the
    vocabulary is append-only. Removing a provider would leave rows in
    ``llm_spend_ledger`` — an append-only table nothing may edit or delete —
    naming a value the CHECK no longer admits, so the table could not be restored
    from its own contents and the next widening's ``ADD CONSTRAINT`` would fail
    against rows nobody can remove.
    """
    module = _load()
    from_migration = {value.strip().strip("'") for value in module._PROVIDER_NAMES_SQL.split(",")}
    from_models = {value.strip().strip("'") for value in PROVIDER_NAMES_SQL.split(",")}

    assert from_migration == {"anthropic", "openai"}
    assert from_migration <= from_models, (
        "a provider was removed from the vocabulary; rows already written to the "
        "append-only spend ledger would name a value the CHECK now rejects"
    )


# --------------------------------------------------------------------------
# The durable ledger's SQL, compiled but never executed
# --------------------------------------------------------------------------


def test_the_committed_total_is_scoped_to_one_provider_and_one_window() -> None:
    """A sum that leaked across providers or days would be a cap over the wrong thing."""
    compiled = str(
        committed_statement(Provider.ANTHROPIC, SpendWindow.DAILY, "2026-08-02").compile(
            compile_kwargs={"literal_binds": True}
        )
    )
    assert "sum(llm_spend_ledger.delta_amount)" in compiled
    assert "llm_spend_ledger.provider = 'anthropic'" in compiled
    assert "llm_spend_ledger.daily_window = '2026-08-02'" in compiled
    assert "monthly_window" not in compiled


def test_the_monthly_total_reads_the_monthly_column() -> None:
    compiled = str(
        committed_statement(Provider.OPENAI, SpendWindow.MONTHLY, "2026-08").compile(
            compile_kwargs={"literal_binds": True}
        )
    )
    assert "llm_spend_ledger.monthly_window = '2026-08'" in compiled
    assert "daily_window" not in compiled


def test_the_advisory_lock_is_taken_before_the_committed_read() -> None:
    """The ordering *is* the cross-process correctness argument.

    A lock held only around the insert would still let two transactions read the
    same headroom and both find room. Asserted on the source because the
    property is an ordering of statements inside one transaction, and there is
    no database here to observe it against.
    """
    source = inspect.getsource(PostgresSpendLedger.reserve)
    lock_at = source.index("pg_advisory_xact_lock")
    read_at = source.index("self._committed(")
    insert_at = source.index("session.add(")
    assert lock_at < read_at < insert_at


def test_the_lock_key_is_deterministic_and_per_provider() -> None:
    """Different providers never contend; the same provider always uses one key."""
    assert _provider_lock_key(Provider.ANTHROPIC) == _provider_lock_key(Provider.ANTHROPIC)
    assert _provider_lock_key(Provider.ANTHROPIC) != _provider_lock_key(Provider.OPENAI)
    for provider in Provider:
        assert -(2**63) <= _provider_lock_key(provider) < 2**63
