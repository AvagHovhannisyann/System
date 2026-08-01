"""CC.1 unit tests: audit-log schema shape, value/address validation, migration 0007.

Metadata and pure-function checks only — no database. The behaviour that needs
a real PostgreSQL (event round trip, previous-value derivation, current-value
reconstruction, and the append-only triggers) is exercised in
``backend/tests/integration/test_audit_log.py``.
"""

from __future__ import annotations

import datetime as dt
import importlib
import math
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import structlog
from sqlalchemy import BigInteger, Boolean, CheckConstraint, DefaultClause, Table, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import DateTime

import backend.db.audit as audit_module
from backend.db.audit import (
    SYSTEM_ACTOR,
    ConfigChangeEvent,
    ConfigNotSetError,
    _advisory_lock_key,
    _current_correlation_id,
    _require_identifier,
    _require_json_value,
)
from backend.db.base import Base
from backend.db.bitemporal import BitemporalMixin, bitemporal_classes, bitemporal_tables

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSIONS_DIR = _REPO_ROOT / "backend" / "db" / "migrations" / "versions"

_TABLE = cast("Table", cast("Any", ConfigChangeEvent).__table__)


def _server_default_text(column_name: str) -> str:
    """Return the rendered server-default expression of a column (asserting one exists)."""
    server_default = _TABLE.c[column_name].server_default
    assert server_default is not None, f"{column_name} must have a server default"
    assert isinstance(server_default, DefaultClause)
    return str(server_default.arg)


def _check_constraint_texts() -> set[str]:
    """Return the SQL text of every CHECK constraint on the audit table."""
    return {
        str(constraint.sqltext)
        for constraint in _TABLE.constraints
        if isinstance(constraint, CheckConstraint)
    }


# --- Schema shape ----------------------------------------------------------


def test_table_name_and_columns() -> None:
    assert _TABLE.name == "config_change_event"
    assert [column.name for column in _TABLE.columns] == [
        "event_id",
        "recorded_at",
        "actor",
        "scope",
        "target",
        "field",
        "previous_value",
        "new_value",
        "is_initial",
        "correlation_id",
    ]


def test_audit_log_is_not_bitemporal() -> None:
    """A config change is something we did, not a fact about the world (module docstring).

    Asserted three ways because each is a different way the mistake could be
    made later: inheriting the mixin, acquiring its columns by hand, or slipping
    into the registry that scopes the as-of read guard.
    """
    assert not issubclass(ConfigChangeEvent, BitemporalMixin)
    registered = {str(cast("Any", cls).__table__.name) for cls in bitemporal_classes()}
    assert "config_change_event" not in registered
    assert _TABLE not in bitemporal_tables()
    for temporal_column in ("valid_from", "valid_to", "knowledge_time", "is_retraction"):
        assert temporal_column not in _TABLE.c


def test_event_id_is_a_database_generated_identity_primary_key() -> None:
    column = _TABLE.c["event_id"]
    assert isinstance(column.type, BigInteger)
    assert column.identity is not None
    assert tuple(c.name for c in _TABLE.primary_key.columns) == ("event_id",)


def test_recorded_at_is_timestamptz_defaulted_from_the_database_clock() -> None:
    """When comes from the server, not the caller: one clock, and not caller-forgeable."""
    column = _TABLE.c["recorded_at"]
    assert isinstance(column.type, DateTime)
    assert column.type.timezone is True
    assert column.nullable is False
    assert "now" in _server_default_text("recorded_at").lower()
    assert column.default is None


@pytest.mark.parametrize(
    "column_name",
    ["actor", "scope", "target", "field", "previous_value", "new_value", "is_initial"],
)
def test_every_column_but_correlation_id_is_not_null(column_name: str) -> None:
    """Only the correlation id is optional — a change made outside a request has none."""
    assert _TABLE.c[column_name].nullable is False


def test_correlation_id_is_optional_text() -> None:
    column = _TABLE.c["correlation_id"]
    assert isinstance(column.type, Text)
    assert column.nullable is True


def test_value_columns_are_not_null_jsonb() -> None:
    """NOT NULL JSONB on both: a value of None is JSON null, never SQL NULL.

    This is what makes ``is_initial`` load-bearing rather than redundant — SQL
    NULL and JSON null both decode to Python ``None``, so the "no previous
    value" case cannot be encoded as SQL NULL without becoming invisible in
    Python.
    """
    for column_name in ("previous_value", "new_value"):
        column = _TABLE.c[column_name]
        assert isinstance(column.type, JSONB)
        assert column.nullable is False


def test_is_initial_is_a_not_null_boolean_with_no_default() -> None:
    """Every writer must state whether the key existed before; there is no safe default."""
    column = _TABLE.c["is_initial"]
    assert isinstance(column.type, Boolean)
    assert column.nullable is False
    assert column.server_default is None
    assert column.default is None


@pytest.mark.parametrize("component", ["actor", "scope", "target", "field"])
def test_address_components_have_a_non_empty_check(component: str) -> None:
    assert f"{component} <> ''" in _check_constraint_texts()


def test_initial_event_check_ties_is_initial_to_previous_value() -> None:
    """The database, not just the writer, refuses an "initial" event with a previous value."""
    assert "NOT is_initial OR previous_value = 'null'::jsonb" in _check_constraint_texts()


def test_declared_indices_match_the_query_shapes_they_serve() -> None:
    """Reconstruction, newest-first browsing, and 'what did this request change'."""
    indices = {str(index.name): index for index in _TABLE.indexes}
    assert set(indices) == {
        "ix_config_change_event_key",
        "ix_config_change_event_recorded_at",
        "ix_config_change_event_correlation_id",
    }
    key_index = [
        str(expression) for expression in indices["ix_config_change_event_key"].expressions
    ]
    assert key_index == [
        "config_change_event.scope",
        "config_change_event.target",
        "config_change_event.field",
        "event_id DESC",
    ]
    recorded_at_index = [
        str(expression) for expression in indices["ix_config_change_event_recorded_at"].expressions
    ]
    assert recorded_at_index == ["recorded_at DESC"]
    correlation_index = indices["ix_config_change_event_correlation_id"]
    assert "correlation_id IS NOT NULL" in str(
        correlation_index.dialect_options["postgresql"]["where"]
    )


def test_model_is_registered_on_the_shared_metadata() -> None:
    """Alembic and every schema tool see one MetaData; a stray Base would be invisible."""
    assert Base.metadata.tables["config_change_event"] is _TABLE


def test_system_actor_constant_is_a_valid_actor() -> None:
    assert _require_identifier(SYSTEM_ACTOR, "actor") == SYSTEM_ACTOR


# --- Address validation ----------------------------------------------------


@pytest.mark.parametrize("bad", ["", " ", " enabled", "enabled ", "\tenabled", "enabled\n"])
def test_empty_or_padded_address_components_are_rejected(bad: str) -> None:
    """Whitespace is rejected, not trimmed: two spellings of one key must not both be writable."""
    with pytest.raises(ValueError, match="field"):
        _require_identifier(bad, "field")


def test_non_string_address_component_is_rejected() -> None:
    with pytest.raises(TypeError, match="scope must be a str"):
        _require_identifier(cast("str", 7), "scope")


def test_ordinary_identifiers_pass_through_unchanged() -> None:
    for good in ("feature_toggle", "momentum_12_1", "enabled", "a"):
        assert _require_identifier(good, "field") == good


# --- Value validation ------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -12,
        3.5,
        "text",
        [],
        {},
        [1, "two", None, {"three": [False, 4.0]}],
        {"a": {"b": {"c": [1, 2, 3]}}},
    ],
)
def test_json_values_are_accepted(value: object) -> None:
    assert _require_json_value(cast("Any", value), "new_value") is value


@pytest.mark.parametrize(
    "value",
    [
        dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        dt.date(2026, 1, 1),
        Decimal("1.5"),
        {"x"},
        (1, 2),
        object(),
        b"bytes",
    ],
)
def test_non_json_values_are_rejected(value: object) -> None:
    """A value that does not round-trip through JSONB is a silently-changing setting."""
    with pytest.raises(TypeError, match="not a JSON value"):
        _require_json_value(cast("Any", value), "new_value")


def test_nested_non_json_value_is_rejected_with_its_path() -> None:
    with pytest.raises(TypeError, match=r"new_value\['limits'\]\[1\]"):
        _require_json_value(cast("Any", {"limits": [1, Decimal("2")]}), "new_value")


def test_non_string_mapping_key_is_rejected() -> None:
    with pytest.raises(TypeError, match="non-str key"):
        _require_json_value(cast("Any", {1: "one"}), "new_value")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_are_rejected(value: float) -> None:
    """PostgreSQL rejects NaN/Infinity in jsonb; catching it here names the field."""
    with pytest.raises(ValueError, match="jsonb"):
        _require_json_value(value, "new_value")


def test_excessively_nested_value_is_rejected() -> None:
    deep: Any = "leaf"
    for _ in range(audit_module._MAX_JSON_DEPTH + 2):
        deep = [deep]
    with pytest.raises(ValueError, match="nests deeper"):
        _require_json_value(deep, "new_value")


def test_self_referential_container_is_rejected_rather_than_recursing_forever() -> None:
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="nests deeper"):
        _require_json_value(cast("Any", cyclic), "new_value")


# --- Advisory lock key -----------------------------------------------------


def test_advisory_lock_key_is_deterministic_and_fits_a_bigint() -> None:
    key = _advisory_lock_key("feature_toggle", "momentum", "enabled")
    assert key == _advisory_lock_key("feature_toggle", "momentum", "enabled")
    assert -(2**63) <= key < 2**63


def test_advisory_lock_key_separates_components() -> None:
    """The NUL separator means no two distinct addresses concatenate to the same material."""
    assert _advisory_lock_key("ab", "c", "d") != _advisory_lock_key("a", "bc", "d")
    assert _advisory_lock_key("a", "b", "cd") != _advisory_lock_key("a", "bc", "d")


def test_distinct_keys_get_distinct_lock_ids() -> None:
    keys = {_advisory_lock_key("s", "t", field) for field in ("enabled", "model", "prompt_version")}
    assert len(keys) == 3


# --- Correlation id resolution ---------------------------------------------


def test_correlation_id_is_none_when_no_request_is_bound() -> None:
    """Outside a request there is no correlation id, and none is invented."""
    structlog.contextvars.clear_contextvars()
    assert _current_correlation_id() is None


def test_correlation_id_comes_from_the_bound_request_id() -> None:
    structlog.contextvars.clear_contextvars()
    tokens = structlog.contextvars.bind_contextvars(request_id="req-abc")
    try:
        assert _current_correlation_id() == "req-abc"
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


@pytest.mark.parametrize("bound", ["", 42, None])
def test_unusable_bound_request_id_yields_none(bound: object) -> None:
    """A blank or non-string binding is treated as absent rather than stored as-is."""
    structlog.contextvars.clear_contextvars()
    tokens = structlog.contextvars.bind_contextvars(request_id=bound)
    try:
        assert _current_correlation_id() is None
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


# --- Errors ----------------------------------------------------------------


def test_config_not_set_error_is_a_lookup_error() -> None:
    """Callers can catch it as the ordinary "key is absent" failure it is."""
    assert issubclass(ConfigNotSetError, LookupError)


# --- Migration 0007 --------------------------------------------------------


def _load(module_name: str) -> ModuleType:
    return importlib.import_module(f"backend.db.migrations.versions.{module_name}")


def _version_modules() -> list[ModuleType]:
    """Import and return every alembic revision module in the versions directory."""
    return [
        _load(path.stem)
        for path in sorted(_VERSIONS_DIR.glob("[0-9]*.py"))
        if path.stem != "__init__"
    ]


def test_0007_follows_0006() -> None:
    module = _load("0007_config_change_event")
    assert module.revision == "0007"
    assert module.down_revision == "0006"


def test_revision_chain_is_linear_with_a_single_head() -> None:
    """No revision may have two children, and exactly one revision may have none.

    Stated as a property over the whole directory rather than as a hard-coded
    list, because the failure this guards against is precisely a *new* revision
    branching off an existing one — which a list of known revisions would not
    see until someone remembered to update it.
    """
    modules = _version_modules()
    revisions = [module.revision for module in modules]
    assert len(set(revisions)) == len(revisions), f"duplicate revision ids: {revisions}"
    children: dict[str | None, list[str]] = {}
    for module in modules:
        children.setdefault(module.down_revision, []).append(module.revision)
    branched = {parent: kids for parent, kids in children.items() if len(kids) > 1}
    assert branched == {}, f"alembic chain branches: {branched}"
    heads = sorted(set(revisions) - set(children))
    assert len(heads) == 1, f"expected a single alembic head, found {heads}"


def test_0007_installs_the_established_append_only_trigger_shape() -> None:
    """Presence check on the raw-SQL migration; the trigger's behaviour is integration-tested.

    The shape matters as much as the fact: ``BEFORE UPDATE OR DELETE`` FOR EACH
    ROW is what revisions 0003/0004 install, and ``FOR EACH ROW`` is what makes
    the guard reach mutations aimed at a table directly rather than only at
    statement level.
    """
    source = Path(str(_load("0007_config_change_event").__file__)).read_text()
    assert "BEFORE UPDATE OR DELETE ON config_change_event" in source
    assert "FOR EACH ROW EXECUTE FUNCTION config_event_append_only_guard()" in source
    assert "CREATE FUNCTION config_event_append_only_guard()" in source


def test_0007_does_not_block_truncate() -> None:
    """TRUNCATE stays the sanctioned admin/test reset path, exactly as in 0003/0004."""
    source = Path(str(_load("0007_config_change_event").__file__)).read_text()
    trigger_clause = source.split("BEFORE UPDATE OR DELETE ON config_change_event")[1]
    assert "TRUNCATE" not in trigger_clause.split("FOR EACH ROW")[0]


def test_0007_creates_the_indices_the_read_paths_need() -> None:
    source = Path(str(_load("0007_config_change_event").__file__)).read_text()
    assert "ix_config_change_event_key" in source
    assert "(scope, target, field, event_id DESC)" in source
    assert "ix_config_change_event_recorded_at" in source
    assert "ix_config_change_event_correlation_id" in source
    assert "WHERE correlation_id IS NOT NULL" in source


# --- Honesty of the immutability claim (D-017) -----------------------------


def test_module_documents_that_append_only_is_not_immutability() -> None:
    """D-017: the docstrings must not let "immutable audit log" stand as a claim.

    Append-only here is enforced by a trigger owned by the same role the
    application connects as, which can drop it. D-017 makes role separation
    (CC.9) a prerequisite for presenting this log as trustworthy, and explicitly
    warns against overclaiming in docstrings. This test exists so that a later
    edit that quietly upgrades "append-only" to "immutable" — the exact
    regression D-017 was written to prevent — fails the build instead of
    shipping a false claim to the operator.
    """
    module_doc = audit_module.__doc__ or ""
    migration_doc = _load("0007_config_change_event").__doc__ or ""
    for doc, where in ((module_doc, "backend/db/audit.py"), (migration_doc, "migration 0007")):
        assert "append-only" in doc, where
        assert "not immutable" in doc.lower(), where
        assert "CC.9" in doc, where
        assert "D-017" in doc, where
    assert "not authenticated" in (ConfigChangeEvent.__table__.c["actor"].doc or "")
