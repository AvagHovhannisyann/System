"""Migration 0014 against the ORM and the state machine it has to match (no database).

Alembic revisions in this project are hand-written, so nothing automatically
keeps a revision, its models, and — here — the Python transition table in step. A
divergence does not fail at import and does not fail in unit tests: it fails when
someone runs ``alembic upgrade head`` and gets a schema the ORM cannot map, or
worse, a schema that admits a transition the machine refuses. Both are compared
structurally here.

What the statements *do* — the unique constraint rejecting a concurrent
duplicate, the CHECK rejecting ``FILLED -> PENDING_NEW``, the chain trigger
rejecting a gap — needs Postgres and lives in
``backend/tests/integration/test_order_lifecycle.py``.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import cast

import sqlalchemy as sa

from backend.db import models
from backend.execution.lifecycle import TERMINAL_STATES, TRANSITIONS
from backend.execution.orders import PAPER_FILL_COST_BASIS, ExecutionVenue, FillSource

_ORDER_TABLE = cast("sa.Table", models.ExecutionOrder.__table__)
_TRANSITION_TABLE = cast("sa.Table", models.ExecutionOrderTransition.__table__)

_APPEND_ONLY_SHAPE = "BEFORE UPDATE OR DELETE"
_TRIPLE = re.compile(r"\('([a-z_]+)', '([a-z_]+)', '([a-z_]+)'\)")


def _revision() -> ModuleType:
    return importlib.import_module("backend.db.migrations.versions.0014_order_lifecycle")


def _source() -> str:
    return Path(str(_revision().__file__)).read_text()


def test_0014_declares_0013_as_its_parent_by_id() -> None:
    # By id only. Revision 0013 is another track's file and is deliberately not
    # imported here: depending on the identifier is the contract, reading the
    # module would be a dependency on its contents.
    revision = _revision()
    assert revision.revision == "0014"
    assert revision.down_revision == "0013"


def test_the_revision_creates_exactly_the_two_execution_tables() -> None:
    source = _source()
    for table in ("execution_order", "execution_order_transition"):
        assert f'op.create_table(\n        "{table}"' in source
    assert source.count("op.create_table(") == 2


def test_both_tables_carry_the_append_only_trigger() -> None:
    # Written out in full rather than generated, so grepping for the trigger
    # shape finds every append-only table. The count is asserted too: a third
    # table without a trigger, or a trigger for a table nobody declared, shows up
    # here.
    source = _source()
    assert source.count(f"{_APPEND_ONLY_SHAPE} ON ") == 2
    assert f"{_APPEND_ONLY_SHAPE} ON execution_order " in source
    assert f"{_APPEND_ONLY_SHAPE} ON execution_order_transition " in source


def test_truncate_is_deliberately_not_blocked() -> None:
    # Matching revisions 0003/0004/0007/0009/0010/0011: TRUNCATE is the
    # sanctioned admin and test reset path and never masquerades as an edit.
    assert "OR TRUNCATE" not in _source()


def test_neither_table_is_bitemporal() -> None:
    # An order is a decision we took, not a fact about the world whose knowledge
    # time the market determined; a knowledge_time invented for it would be a
    # fabricated value in the one column whose meaning is that it is not.
    for table in (_ORDER_TABLE, _TRANSITION_TABLE):
        assert {"valid_from", "valid_to", "knowledge_time"}.isdisjoint(table.columns.keys())


def test_the_order_has_no_state_column() -> None:
    # Append-only tables cannot update one, and a denormalized state that drifts
    # from the history is the condition the transition log exists to prevent.
    names = [column.name for column in _ORDER_TABLE.columns]
    assert "state" not in names
    assert not any(name.endswith("_state") for name in names)


def test_the_idempotency_key_is_unique_in_the_orm_and_in_the_revision() -> None:
    # The enforcement point. If this constraint is absent, two concurrent
    # submissions both land and the position doubles.
    constraint = next(
        item for item in _ORDER_TABLE.constraints if isinstance(item, sa.UniqueConstraint)
    )
    assert constraint.name == "uq_execution_order_idempotency_key"
    assert [column.name for column in constraint.columns] == ["idempotency_key"]
    declared = 'sa.UniqueConstraint("idempotency_key",name="uq_execution_order_idempotency_key")'
    assert declared in re.sub(r"\s+", "", _source())


def test_the_transition_primary_key_is_the_per_order_sequence() -> None:
    # Also the concurrency control: two writers claiming last+1 collide here.
    assert [column.name for column in _TRANSITION_TABLE.primary_key.columns] == [
        "order_id",
        "sequence_number",
    ]


def test_every_declared_column_appears_in_the_revision() -> None:
    source = re.sub(r"\s+", "", _source())
    for table in (_ORDER_TABLE, _TRANSITION_TABLE):
        for column in table.columns:
            assert f'sa.Column("{column.name}"' in source, (table.name, column.name)


def test_every_declared_check_constraint_appears_in_the_revision() -> None:
    # Names are compared, not only expressions: the metadata naming convention
    # expands an unprefixed name, and a revision spelling the name out in full
    # would create a constraint the ORM does not know about.
    source = _source()
    for table in (_ORDER_TABLE, _TRANSITION_TABLE):
        for constraint in table.constraints:
            if isinstance(constraint, sa.CheckConstraint):
                unprefixed = str(constraint.name).removeprefix(f"ck_{table.name}_")
                assert f'name="{unprefixed}"' in source, constraint.name


def test_the_order_points_at_the_identity_anchor_not_the_versioned_master() -> None:
    targets = {foreign_key.column.table.name for foreign_key in _ORDER_TABLE.foreign_keys}
    assert targets == {"security"}
    assert '"security.security_id"' in _source()
    assert "security_master.security_id" not in _source()


def _check(table: sa.Table, name: str) -> str:
    """Return the SQL text of one named CHECK constraint on ``table``."""
    for constraint in table.constraints:
        if isinstance(constraint, sa.CheckConstraint) and constraint.name == name:
            return str(constraint.sqltext)
    msg = f"{table.name} has no CHECK named {name}"
    raise AssertionError(msg)


def test_the_sql_transition_enumeration_is_exactly_the_python_table() -> None:
    # The comparison this file exists for. A triple in one and not the other is
    # a schema that admits a state the machine refuses, or refuses one it
    # produces, and both stay silent until a real order hits them.
    revision = _revision()
    sql_triples = set(_TRIPLE.findall(revision._LEGAL_TRANSITIONS_SQL))
    python_triples = {
        (state.value, event.value, target.value) for (state, event), target in TRANSITIONS.items()
    }
    assert sql_triples == python_triples
    assert len(sql_triples) == len(TRANSITIONS) == 25


def test_the_orm_legal_transition_check_matches_the_python_table() -> None:
    text = _check(_TRANSITION_TABLE, "ck_execution_order_transition_legal_transition")
    orm_triples = set(_TRIPLE.findall(text))
    python_triples = {
        (state.value, event.value, target.value) for (state, event), target in TRANSITIONS.items()
    }
    assert orm_triples == python_triples


def test_the_terminal_states_check_names_exactly_the_terminal_states() -> None:
    text = _check(_TRANSITION_TABLE, "ck_execution_order_transition_from_state_not_terminal")
    named = set(re.findall(r"'([a-z_]+)'", text))
    assert named == {state.value for state in TERMINAL_STATES}
    assert "NOT IN" in text
    assert "from_state NOT IN" in _source()


def test_the_fill_payload_rule_is_stated_in_both_directions() -> None:
    # D-030's shape: forbidding only one direction trades a fabrication bug for
    # a silent-absence bug pointing the other way.
    present = _check(_TRANSITION_TABLE, "ck_execution_order_transition_fill_payload_present")
    absent = _check(_TRANSITION_TABLE, "ck_execution_order_transition_fill_payload_absent")
    for column in ("fill_quantity_shares", "fill_price_usd", "fill_source", "fill_cost_basis"):
        assert f"{column} IS NOT NULL" in present, column
        assert f"{column} IS NULL" in absent, column
    assert "venue_fill_id IS NULL" in absent


def test_the_venue_check_names_the_only_venue_the_enum_has() -> None:
    text = _check(_ORDER_TABLE, "ck_execution_order_venue_is_paper")
    assert text == f"venue = '{ExecutionVenue.PAPER.value}'"


def test_the_fill_source_check_names_exactly_the_non_live_sources() -> None:
    text = _check(_TRANSITION_TABLE, "ck_execution_order_transition_fill_source_is_not_live")
    named = set(re.findall(r"'([a-z_]+)'", text))
    assert named == {member.value for member in FillSource}


def test_the_cost_basis_check_pins_the_lower_bound_label() -> None:
    text = _check(_TRANSITION_TABLE, "ck_execution_order_transition_fill_cost_basis_is_lower_bound")
    assert f"'{PAPER_FILL_COST_BASIS}'" in text
    assert re.findall(r"'([a-z_]+)'", text) == [PAPER_FILL_COST_BASIS]


def test_the_chain_guard_runs_before_every_insert() -> None:
    # A CHECK sees one row; the properties that make a history replayable are
    # relations between rows, so they live in a trigger. Each of the four is
    # asserted present by the condition it tests, not by its message.
    source = _source()
    assert "BEFORE INSERT ON execution_order_transition" in source
    assert "CREATE FUNCTION execution_transition_chain_guard()" in source
    flattened = re.sub(r"\s+", " ", source)
    for condition in (
        "IF NEW.sequence_number <> 1 THEN",
        "IF NEW.from_state <> 'draft' THEN",
        "IF NEW.sequence_number <> previous.sequence_number + 1 THEN",
        "IF NEW.from_state <> previous.to_state THEN",
        (
            "IF NEW.filled_quantity_after_shares <> "
            "previous.filled_quantity_after_shares + traded THEN"
        ),
        "IF NEW.filled_quantity_after_shares > ordered_shares THEN",
        ("IF NEW.to_state = 'filled' AND NEW.filled_quantity_after_shares <> ordered_shares THEN"),
    ):
        assert condition in flattened, condition


def test_the_downgrade_removes_everything_the_upgrade_creates() -> None:
    source = _source()
    for statement in (
        "DROP FUNCTION execution_append_only_guard()",
        "DROP FUNCTION execution_transition_chain_guard()",
        "DROP TRIGGER trg_execution_order_transition_chain",
        'op.drop_table("execution_order_transition")',
        'op.drop_table("execution_order")',
        'op.drop_index("ix_execution_order_rebalance"',
    ):
        assert statement in source, statement
