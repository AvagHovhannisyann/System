"""Migration 0016 against the ORM and the Python vocabularies it must match (no database).

Alembic revisions in this project are hand-written, so nothing automatically
keeps a revision, its models, and — here — the halt-trigger enum and the
tolerance ceiling in step. A divergence does not fail at import and does not fail
in unit tests: it fails when someone runs ``alembic upgrade head`` and gets a
schema the ORM cannot map, or a schema that refuses to record a halt the kill
switch wants to engage. The second is the dangerous one, so the trigger
vocabulary is compared element by element.

What the statements *do* — the CHECKs refusing a raw ``INSERT``, the clearance
guard refusing a second clearance, the append-only triggers refusing ``UPDATE`` —
needs Postgres and lives in ``backend/tests/integration/test_reconciliation.py``.
"""

from __future__ import annotations

import importlib
import re
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest
import sqlalchemy as sa

from backend.db import models
from backend.execution.halt import HaltEventKind, HaltTrigger
from backend.execution.reconciliation import MAX_CASH_TOLERANCE_USD, SnapshotOrigin
from backend.tests.integration import test_reconciliation as integration

if TYPE_CHECKING:
    from collections.abc import Iterable

_RECONCILIATION_TABLE = cast("sa.Table", models.ExecutionReconciliation.__table__)
_HALT_TABLE = cast("sa.Table", models.ExecutionHalt.__table__)

_APPEND_ONLY_SHAPE = "BEFORE UPDATE OR DELETE"


def _revision() -> ModuleType:
    return importlib.import_module("backend.db.migrations.versions.0016_halt_and_reconciliation")


def _source() -> str:
    return Path(str(_revision().__file__)).read_text()


def _squashed() -> str:
    """Return the revision source with quotes dropped and whitespace collapsed.

    Adjacent string literals are concatenated at compile time, so a statement
    split across several of them reaches Postgres as one. Normalising that way
    lets these tests assert the SQL a database would see rather than the source
    layout the formatter happened to choose.
    """
    return re.sub(r"\s+", " ", _source().replace('"', ""))


def _clearance_guard_body() -> str:
    """Return the source of the clearance guard function only."""
    source = _source()
    start = source.index("CREATE FUNCTION execution_halt_clearance_guard")
    end = source.index("CREATE TRIGGER trg_execution_halt_clearance")
    return source[start:end]


def _check(table: sa.Table, name: str) -> str:
    """Return the SQL text of one named CHECK constraint on ``table``."""
    for constraint in table.constraints:
        if isinstance(constraint, sa.CheckConstraint) and constraint.name == name:
            return str(constraint.sqltext)
    msg = f"{table.name} has no CHECK named {name}"
    raise AssertionError(msg)


def test_0016_declares_0015_as_its_parent_by_id() -> None:
    # By id only. Revision 0015 is another track's file and is deliberately not
    # imported here: depending on the identifier is the contract, reading the
    # module would be a dependency on its contents.
    revision = _revision()
    assert revision.revision == "0016"
    assert revision.down_revision == "0015"


def test_the_revision_creates_exactly_the_two_control_tables() -> None:
    source = _source()
    for table in ("execution_reconciliation", "execution_halt"):
        assert f'op.create_table(\n        "{table}"' in source
    assert source.count("op.create_table(") == 2


def test_both_tables_carry_the_append_only_trigger() -> None:
    source = _source()
    assert source.count(f"{_APPEND_ONLY_SHAPE} ON ") == 2
    assert f"{_APPEND_ONLY_SHAPE} ON execution_reconciliation " in source
    assert f"{_APPEND_ONLY_SHAPE} ON execution_halt " in source


def test_truncate_is_deliberately_not_blocked() -> None:
    assert "OR TRUNCATE" not in _source()


def test_neither_table_is_bitemporal() -> None:
    # A reconciliation is an observation we made and a halt is a decision we took;
    # a knowledge_time invented for either would be a fabricated value in the one
    # column whose meaning is that it is not (I3).
    for table in (_RECONCILIATION_TABLE, _HALT_TABLE):
        assert {"valid_from", "valid_to", "knowledge_time"}.isdisjoint(table.columns.keys())


def test_every_declared_column_appears_in_the_revision() -> None:
    squashed = re.sub(r"\s+", "", _source())
    for table in (_RECONCILIATION_TABLE, _HALT_TABLE):
        for column in table.columns:
            assert f'sa.Column("{column.name}"' in squashed, (table.name, column.name)


def test_every_declared_check_constraint_appears_in_the_revision() -> None:
    # Names are compared, not only expressions: the metadata naming convention
    # expands an unprefixed name, and a revision spelling the name out in full
    # would create a constraint the ORM does not know about.
    source = _source()
    for table in (_RECONCILIATION_TABLE, _HALT_TABLE):
        for constraint in table.constraints:
            if isinstance(constraint, sa.CheckConstraint):
                unprefixed = str(constraint.name).removeprefix(f"ck_{table.name}_")
                assert f'name="{unprefixed}"' in source, constraint.name


def test_every_constraint_name_fits_postgres_identifier_limit() -> None:
    # Postgres truncates identifiers at 63 characters *silently*, which would
    # leave the ORM and the schema disagreeing about a constraint's name with no
    # error anywhere.
    for table in (_RECONCILIATION_TABLE, _HALT_TABLE):
        for constraint in table.constraints:
            assert len(str(constraint.name)) <= 63, constraint.name


# ---------------------------------------------------------------------------
# The vocabularies that must not drift.
# ---------------------------------------------------------------------------


def test_the_sql_trigger_vocabulary_is_exactly_the_python_enum() -> None:
    # The comparison this file exists for. A trigger the Python side can engage
    # and the schema refuses is a kill switch that cannot fire; a value the schema
    # admits and Python cannot read is a stored cause nobody can interpret.
    revision = _revision()
    sql_values = set(re.findall(r"'([a-z_]+)'", revision._HALT_TRIGGERS_SQL))
    assert sql_values == {member.value for member in HaltTrigger}
    assert len(sql_values) == 5


def test_the_orm_trigger_check_matches_the_python_enum() -> None:
    text = _check(_HALT_TABLE, "ck_execution_halt_trigger_is_known")
    assert set(re.findall(r"'([a-z_]+)'", text)) == {member.value for member in HaltTrigger}


def test_the_halt_event_check_names_exactly_the_two_kinds() -> None:
    text = _check(_HALT_TABLE, "ck_execution_halt_event_is_known")
    assert set(re.findall(r"'([a-z_]+)'", text)) == {member.value for member in HaltEventKind}


def test_the_reported_origin_check_names_exactly_the_non_internal_origins() -> None:
    # I3 at the schema: a statement from a real-money account has no
    # representation, and neither does our own ledger standing in for one.
    text = _check(_RECONCILIATION_TABLE, "ck_execution_reconciliation_reported_origin_is_not_live")
    named = set(re.findall(r"'([a-z_]+)'", text))
    assert named == {SnapshotOrigin.PAPER_BROKER.value, SnapshotOrigin.SIMULATED.value}
    assert SnapshotOrigin.INTERNAL_LEDGER.value not in named
    assert "'live'" not in text


def test_the_internal_origin_check_pins_our_own_ledger() -> None:
    text = _check(_RECONCILIATION_TABLE, "ck_execution_reconciliation_internal_origin_is_ledger")
    assert text == f"internal_origin = '{SnapshotOrigin.INTERNAL_LEDGER.value}'"


def test_the_schema_tolerance_ceiling_equals_the_python_one() -> None:
    # A ceiling the code enforces and the schema does not is a ceiling one raw
    # INSERT away from being absent.
    text = _check(_RECONCILIATION_TABLE, "ck_execution_reconciliation_tolerance_within_ceiling")
    ceiling = re.search(r"cash_tolerance_usd <= ([0-9.]+)", text)
    assert ceiling is not None
    assert Decimal(ceiling.group(1)) == MAX_CASH_TOLERANCE_USD
    assert "cash_tolerance_usd >= 0" in text


def test_the_verdict_is_a_function_of_the_findings() -> None:
    assert (
        _check(_RECONCILIATION_TABLE, "ck_execution_reconciliation_matched_iff_no_breaks")
        == "matched = (break_count = 0)"
    )


# ---------------------------------------------------------------------------
# The clearance guard: refuses clearances, never engagements.
# ---------------------------------------------------------------------------


def test_the_clearance_guard_is_a_before_insert_row_trigger() -> None:
    squashed = _squashed()
    assert "CREATE FUNCTION execution_halt_clearance_guard() RETURNS trigger" in squashed
    assert (
        "CREATE TRIGGER trg_execution_halt_clearance BEFORE INSERT ON execution_halt "
        "FOR EACH ROW EXECUTE FUNCTION execution_halt_clearance_guard()" in squashed
    )


def test_the_clearance_guard_returns_immediately_for_an_engagement() -> None:
    # The single most important line in the revision: nothing may refuse a
    # halt-engage row, so the guard leaves before it can examine one.
    assert "IF NEW.event <> 'cleared' THEN RETURN NEW; END IF;" in _squashed()


def test_every_raise_in_the_guard_comes_after_the_engagement_returns() -> None:
    # Ordering matters as much as presence: an engagement must reach the RETURN
    # before any branch that could refuse it exists in the control flow.
    body = _clearance_guard_body()
    early_return = body.index("IF NEW.event <> 'cleared' THEN")
    raises = [match.start() for match in re.finditer(r"RAISE EXCEPTION", body)]
    assert raises, "a guard with no refusals is not a guard"
    assert all(position > early_return for position in raises)


def test_only_the_already_cleared_branch_is_raised_as_a_conflict() -> None:
    # D-034's split, restated: "already cleared" is the same refusal the unique
    # index gives and carries its code; "not an engagement" is not retryable and
    # keeps the default P0001, because retrying it would loop forever.
    body = _clearance_guard_body()
    assert body.count("USING ERRCODE") == 1
    assert body.count("USING ERRCODE = 'unique_violation'") == 1
    conflict = body.index("USING ERRCODE = 'unique_violation'")
    assert body.index("was already cleared by halt_id") < conflict
    for message in ("which does not exist", "only an engagement can be"):
        assert body.index(message) < body.index("was already cleared by halt_id")


def test_a_halt_can_be_cleared_at_most_once_in_the_orm_and_the_revision() -> None:
    constraint = next(
        item for item in _HALT_TABLE.constraints if isinstance(item, sa.UniqueConstraint)
    )
    assert constraint.name == "uq_execution_halt_clears_halt_id"
    assert [column.name for column in constraint.columns] == ["clears_halt_id"]
    declared = 'sa.UniqueConstraint("clears_halt_id",name="uq_execution_halt_clears_halt_id")'
    assert declared in re.sub(r"\s+", "", _source())


def test_a_clearance_points_at_an_earlier_row_of_the_same_table() -> None:
    targets = {foreign_key.column.table.name for foreign_key in _HALT_TABLE.foreign_keys}
    assert targets == {"execution_halt"}
    assert '"execution_halt.halt_id"' in _source()


def test_each_clearance_column_is_tied_to_the_event_on_its_own() -> None:
    # Stated per column, not over their conjunction.
    #
    # The first version read `(event = 'cleared') = (a IS NOT NULL AND b IS NOT
    # NULL AND c IS NOT NULL)`, which only refuses an engagement carrying all
    # three. A stray clears_halt_id alone was accepted — and because
    # uq_execution_halt_clears_halt_id is on the column unconditionally, that row
    # consumed the unique slot for the halt it named, so the genuine clearance was
    # refused as "already cleared" while open_halts kept reporting the halt open.
    # Permanently un-clearable, with an error that misdescribed why.
    text = _check(_HALT_TABLE, "ck_execution_halt_clearance_fields_iff_cleared")
    for column in ("clears_halt_id", "cleared_by", "clearance_reason"):
        assert f"(event = 'cleared') = ({column} IS NOT NULL)" in text, column
    # The conjunction form must not come back: it would satisfy the loop above
    # only if each column also appeared in its own biconditional, but pinning the
    # count keeps the expression from growing a weaker extra clause.
    assert text.count("(event = 'cleared') = (") == 3
    assert "IS NOT NULL AND" not in text


def test_the_clearance_rule_is_stated_identically_in_the_revision() -> None:
    squashed = _squashed()
    for column in ("clears_halt_id", "cleared_by", "clearance_reason"):
        assert f"(event = 'cleared') = ({column} IS NOT NULL)" in squashed, column


def test_the_open_halt_index_covers_exactly_the_engagements() -> None:
    # Every release path asks "is anything open" on every cycle; the partial index
    # is what keeps that from scanning the whole history.
    assert (
        "CREATE INDEX ix_execution_halt_open ON execution_halt (halt_id) "
        "WHERE event = 'engaged'" in _squashed()
    )


def test_the_downgrade_drops_everything_the_upgrade_created() -> None:
    source = _source()
    for statement in (
        "DROP TRIGGER trg_execution_halt_clearance ON execution_halt",
        "DROP FUNCTION execution_halt_clearance_guard()",
        "DROP FUNCTION execution_control_append_only_guard()",
        "DROP INDEX ix_execution_halt_open",
    ):
        assert statement in source
    assert 'op.drop_table("execution_halt")' in source
    assert 'op.drop_table("execution_reconciliation")' in source


# ---------------------------------------------------------------------------
# Probe isolation, checkable without a database.
# ---------------------------------------------------------------------------


def _checks_of(table: sa.Table) -> dict[str, str]:
    """Return every CHECK on ``table`` as unprefixed name to SQL text."""
    return {
        str(constraint.name).removeprefix(f"ck_{table.name}_"): str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, sa.CheckConstraint)
    }


def _mentioning(checks: dict[str, str], columns: Iterable[str]) -> set[str]:
    """Return the constraints whose SQL names any of ``columns``."""
    wanted = set(columns)
    return {
        name
        for name, text in checks.items()
        if any(re.search(rf"\b{re.escape(column)}\b", text) for column in wanted)
    }


@pytest.mark.parametrize(
    ("probes", "table"),
    [
        (integration.RECONCILIATION_PROBES, _RECONCILIATION_TABLE),
        (integration.HALT_PROBES, _HALT_TABLE),
    ],
    ids=["reconciliation", "halt"],
)
def test_every_raw_insert_probe_declares_the_constraints_its_columns_touch(
    probes: tuple[integration.ConstraintProbe, ...], table: sa.Table
) -> None:
    """A probe must know which other CHECKs its overridden columns reach.

    CHECK evaluation order is unspecified in Postgres, so a probe row that breaks
    two constraints reports whichever the server reaches first: the test then
    fails for the wrong reason, or passes by luck and stops meaning anything. That
    cost P11.2 four tests and P11.3 one, and left a third passing on luck.

    This cannot verify that the probe's *values* satisfy the neighbours — only a
    database can, and the integration test's assertion on the constraint **name**
    is what does it. What it can do, without Docker, is make the coupling
    impossible to overlook: if an override touches a column named by another
    constraint, the probe has to say so.
    """
    checks = _checks_of(table)
    assert checks, table.name
    for probe in probes:
        assert probe.constraint in checks, (table.name, probe.constraint)
        reached = _mentioning(checks, probe.overrides)
        assert probe.constraint in reached, (
            f"{probe.constraint} does not name any column {sorted(probe.overrides)} overrides; "
            f"the probe cannot be aiming at it"
        )
        neighbours = reached - {probe.constraint}
        assert neighbours == set(probe.also_mentions), (
            f"probe for {probe.constraint} overrides {sorted(probe.overrides)}, which also "
            f"reaches {sorted(neighbours)}; declared {sorted(probe.also_mentions)}"
        )


def test_the_probe_lists_cover_the_constraints_worth_probing() -> None:
    # Guards against a probe list that quietly shrinks. Not every CHECK needs a
    # probe — some are unreachable in isolation — but the ones that are aimed at
    # must stay aimed at.
    reconciliation = {probe.constraint for probe in integration.RECONCILIATION_PROBES}
    halt = {probe.constraint for probe in integration.HALT_PROBES}
    assert {"matched_iff_no_breaks", "counts_consistent", "tolerance_within_ceiling"} <= (
        reconciliation
    )
    assert {"clearance_fields_iff_cleared", "trigger_iff_engaged", "event_is_known"} <= halt
