"""Migration 0017 against the code it has to agree with (P12.1/P12.4).

No database here — the schema's *behaviour* is exercised in
``test_alerts_store_db.py``. What is checked is the agreement that a container
run cannot check for you, because a divergence makes both sides internally
consistent and jointly wrong:

* the revision links to 0016 by identifier, which is how this track depends on a
  migration written on another one;
* the enumerated halt causes in SQL are exactly
  :class:`~backend.monitoring.expectation.HaltCause`. A cause the enum has and
  the CHECK does not is a halt the database refuses to record — the worst
  possible time to lose a row;
* the constraint names in the revision match the ones the ORM model declares,
  since this repository hand-writes migrations and the two are only equal
  because somebody kept them so;
* the append-only trigger has the established shape and does **not** block
  TRUNCATE, which is the sanctioned admin and test reset path.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from sqlalchemy import CheckConstraint, UniqueConstraint

from backend.db.models import (
    MonitoringAlert,
    MonitoringAlertAcknowledgement,
    MonitoringAlertDelivery,
    MonitoringHaltEvent,
)
from backend.monitoring.expectation import HaltCause

_TABLES = (
    "monitoring_alert",
    "monitoring_alert_delivery",
    "monitoring_alert_acknowledgement",
    "monitoring_halt_event",
)


def _module() -> ModuleType:
    return importlib.import_module("backend.db.migrations.versions.0017_alerts_and_halts")


def _source() -> str:
    return Path(str(_module().__file__)).read_text()


def test_0017_follows_0016() -> None:
    module = _module()
    assert module.revision == "0017"
    assert module.down_revision == "0016"


def test_every_table_is_created_and_dropped() -> None:
    source = _source()
    for table in _TABLES:
        assert f'op.create_table(\n        "{table}"' in source, table
        assert f'op.drop_table("{table}")' in source, table


def test_the_append_only_trigger_has_the_established_shape() -> None:
    """``BEFORE UPDATE OR DELETE ... FOR EACH ROW``, as in 0003/0004/0007/0013/0014."""
    source = _source()
    assert "CREATE FUNCTION monitoring_append_only_guard()" in source
    for table in _TABLES:
        assert f"BEFORE UPDATE OR DELETE ON {table} " in source, table
    assert source.count("FOR EACH ROW EXECUTE FUNCTION monitoring_append_only_guard()") == len(
        _TABLES
    )


def test_the_trigger_does_not_block_truncate() -> None:
    """TRUNCATE stays the sanctioned admin/test reset, exactly as in every earlier revision."""
    source = _source()
    for table in _TABLES:
        clause = source.split(f"BEFORE UPDATE OR DELETE ON {table} ")[1]
        assert "TRUNCATE" not in clause.split("FOR EACH ROW")[0]


def test_the_sql_cause_list_is_exactly_the_python_enum() -> None:
    """A cause the enum has and the CHECK does not is a halt the database refuses."""
    causes_sql = _module()._HALT_CAUSES_SQL
    in_sql = {piece.strip().strip("'") for piece in causes_sql.split(",")}
    assert in_sql == {str(cause) for cause in HaltCause}


def test_the_model_check_names_match_the_revision() -> None:
    """Hand-written migrations: agreement is asserted, not assumed.

    Both sides declare CHECK names *unprefixed* and let the metadata naming
    convention expand them to ``ck_%(table_name)s_%(constraint_name)s``. The
    model's objects carry the expanded form by the time they are readable here,
    so the prefix is removed before comparing — which also asserts the
    convention is the one in force.
    """
    source = _source()
    for model in (
        MonitoringAlert,
        MonitoringAlertDelivery,
        MonitoringAlertAcknowledgement,
        MonitoringHaltEvent,
    ):
        prefix = f"ck_{model.__tablename__}_"
        for constraint in model.__table_args__:
            assert isinstance(constraint, CheckConstraint | UniqueConstraint)
            name = str(constraint.name)
            declared = (
                name.removeprefix(prefix) if isinstance(constraint, CheckConstraint) else name
            )
            assert f'name="{declared}"' in source, (model.__tablename__, name)


def test_the_unique_constraints_that_carry_the_guarantees_are_present() -> None:
    source = _source()
    # One condition, one alert.
    assert 'sa.UniqueConstraint("dedup_key", name="uq_monitoring_alert_dedup_key")' in source
    # One signature per alert, never replaced.
    assert 'sa.UniqueConstraint("alert_id", name="uq_monitoring_alert_acknowledgement_alert")' in (
        source
    )
    # One resume per halt, so the derived halt state cannot be ambiguous.
    assert '"resolves_halt_event_id", name="uq_monitoring_halt_event_resolves"' in source


def test_no_state_column_exists_on_any_of_the_four_tables() -> None:
    """Delivered / acknowledged / is_active are derived. A stored flag would drift."""
    forbidden = {"delivered", "acknowledged", "is_active", "active", "escalated", "state"}
    for model in (
        MonitoringAlert,
        MonitoringAlertDelivery,
        MonitoringAlertAcknowledgement,
        MonitoringHaltEvent,
    ):
        columns = {column.name for column in model.__table__.columns}
        assert not (columns & forbidden), (model.__tablename__, columns & forbidden)


def test_every_row_carries_the_four_i2_components() -> None:
    """A halt or an alert that cannot say which run produced it cannot be audited."""
    for model in (MonitoringAlert, MonitoringHaltEvent):
        columns = {column.name for column in model.__table__.columns}
        assert {"git_commit", "git_dirty", "data_version", "config_hash", "seed"} <= columns
