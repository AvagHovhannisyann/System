"""CC.1 integration: the configuration audit log against real PostgreSQL.

What is proved here and nowhere else:

- an event written through :func:`backend.db.audit.record_config_change` reads
  back with every field §6.11 requires — actor, instant, address, previous
  value, new value — plus the correlation id that ties it to a request;
- ``previous_value`` is derived from the log itself and is right across the
  cases a naive implementation gets wrong: the first event for a key,
  ``None`` -> value, and value -> ``None``;
- the current configuration is **reconstructed from the event history** and
  from nothing else, which is what "config changes are events, not mutations"
  has to mean to be worth anything;
- UPDATE and DELETE are refused **by the database trigger**, on every path
  tested: raw SQL on a connection, and ORM DML through the writer session;
- migration 0007 and the ORM model — written independently, since this repo
  hand-writes its migrations rather than autogenerating them — agree on the
  columns, the index shapes and the trigger's timing and orientation.

On reaching the trigger: unlike the bitemporal fact tables, this table needs no
special engine to be reached. The Core-level guard's default-deny scan is
scoped to bitemporal fact-table names (``backend.db._guard.fact_table_names``)
and ``config_change_event`` is deliberately not one, so an ordinary admin
connection *is* a direct line to Postgres here — the mutation really does reach
the server and really is refused there, which is the property under test.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import TYPE_CHECKING, Any, cast

import pytest
import sqlalchemy as sa
import structlog
from sqlalchemy import Table
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import create_admin_engine, ingest_writer_session
from backend.db.audit import (
    SYSTEM_ACTOR,
    ConfigChangeEvent,
    ConfigNotSetError,
    config_history,
    current_scope_values,
    current_value,
    current_values,
    record_config_change,
    record_config_changes,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_SCOPE = "feature_toggle"
_TARGET = "momentum_12_1"
_ACTOR = "operator"


@pytest.fixture(autouse=True)
async def _clean_config_events() -> AsyncIterator[None]:
    """Truncate the audit table after every test in this module.

    The shared integration fixture truncates the fact tables and the ingestion
    log; this table is new in migration 0007 and holds no foreign key for its
    CASCADE to follow. TRUNCATE is deliberately left unblocked by the
    append-only trigger (migration 0007, following 0003/0004) precisely so the
    sanctioned reset path exists. An ordinary admin engine suffices: the Core
    guard only refuses SQL naming a *bitemporal* fact table, and this is not
    one.
    """
    yield
    engine = create_admin_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(sa.text("TRUNCATE TABLE config_change_event RESTART IDENTITY"))
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
def _no_ambient_request() -> Iterator[None]:
    """Start every test with no bound request id, so correlation defaults are explicit.

    The shared ``backend/tests/conftest.py`` fixture already does this for the
    whole suite. Repeated here on purpose: what these tests assert about the
    *default* correlation id is only meaningful with nothing bound, and that
    precondition should be visible in the file that depends on it rather than
    inherited silently from a conftest two directories up.
    """
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


async def _raw_execute(statement: str) -> None:
    """Run one textual statement on an admin connection (see the module docstring)."""
    engine = create_admin_engine()
    try:
        async with engine.begin() as connection:
            await connection.execute(sa.text(statement))
    finally:
        await engine.dispose()


# --- Round trip ------------------------------------------------------------


async def test_event_round_trip_records_who_when_what_and_both_values() -> None:
    """Every field §6.11 names, written once and read back."""
    before = dt.datetime.now(dt.UTC)
    event = await record_config_change(
        _SCOPE, _TARGET, "enabled", True, actor=_ACTOR, correlation_id="req-1"
    )
    after = dt.datetime.now(dt.UTC)

    assert event.event_id > 0
    assert event.actor == _ACTOR
    assert (event.scope, event.target, event.field) == (_SCOPE, _TARGET, "enabled")
    assert event.new_value is True
    assert event.previous_value is None
    assert event.is_initial is True
    assert event.correlation_id == "req-1"
    assert event.recorded_at.tzinfo is not None
    assert event.recorded_at.utcoffset() == dt.timedelta(0)
    assert before <= event.recorded_at <= after

    (stored,) = await config_history(scope=_SCOPE, target=_TARGET)
    assert stored.event_id == event.event_id
    assert stored.new_value is True
    assert stored.actor == _ACTOR
    assert stored.correlation_id == "req-1"


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        0,
        -7,
        2.5,
        "gpt-tier-cheap",
        [],
        {},
        ["a", 1, None, {"nested": [True, 2.25]}],
        {"models": ["a", "b"], "threshold": 0.75, "notes": None},
    ],
)
async def test_json_values_round_trip_unchanged(value: Any) -> None:  # noqa: ANN401 — JSON value
    """A setting that changes type or shape across the store is a setting nobody can trust."""
    await record_config_change(_SCOPE, _TARGET, "payload", value, actor=_ACTOR)
    assert await current_value(_SCOPE, _TARGET, "payload") == value


# --- Previous value derivation ---------------------------------------------


async def test_previous_value_is_the_prior_events_new_value() -> None:
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    second = await record_config_change(_SCOPE, _TARGET, "enabled", False, actor=_ACTOR)
    assert second.is_initial is False
    assert second.previous_value is True
    assert second.new_value is False


async def test_concurrent_changes_to_one_key_still_form_an_honest_chain() -> None:
    """Two writers racing on one key must not both record the same previous value.

    ``previous_value`` is read from the log and written back to it, so without
    the per-key advisory lock two concurrent changes would both observe the same
    prior state and one of them would record a previous value that was never the
    value — a lie in the field whose whole purpose is to be checkable. Which
    writer wins is deliberately not asserted; that the chain is intact is.
    """
    await asyncio.gather(
        record_config_change(_SCOPE, _TARGET, "weight", "a", actor="writer_a"),
        record_config_change(_SCOPE, _TARGET, "weight", "b", actor="writer_b"),
    )
    oldest_first = tuple(reversed(await config_history(scope=_SCOPE, target=_TARGET)))
    assert len(oldest_first) == 2
    first, second = oldest_first
    assert first.is_initial is True
    assert first.previous_value is None
    assert second.is_initial is False
    assert second.previous_value == first.new_value
    assert {first.new_value, second.new_value} == {"a", "b"}
    assert await current_value(_SCOPE, _TARGET, "weight") == second.new_value


async def test_none_to_value_and_value_to_none_are_both_captured() -> None:
    """The case a naive encoding loses: SQL NULL and JSON null both decode to Python None.

    ``is_initial`` is what keeps "the key had no value" distinguishable from
    "the key's value was None"; without it the first and second events below
    would be indistinguishable in Python.
    """
    first = await record_config_change(_SCOPE, _TARGET, "threshold", None, actor=_ACTOR)
    assert first.is_initial is True
    assert first.previous_value is None
    assert first.new_value is None

    second = await record_config_change(_SCOPE, _TARGET, "threshold", 0.8, actor=_ACTOR)
    assert second.is_initial is False
    assert second.previous_value is None  # the value genuinely was None
    assert second.new_value == 0.8

    third = await record_config_change(_SCOPE, _TARGET, "threshold", None, actor=_ACTOR)
    assert third.is_initial is False
    assert third.previous_value == 0.8
    assert third.new_value is None

    assert await current_value(_SCOPE, _TARGET, "threshold") is None


async def test_previous_value_is_stored_as_json_null_not_sql_null() -> None:
    """The encoding the whole previous/new scheme rests on, asserted in the database.

    If ``None`` were persisted as SQL NULL the NOT NULL constraint would reject
    the write; this asserts the positive form directly so the encoding is
    pinned rather than merely not-crashing.
    """
    await record_config_change(_SCOPE, _TARGET, "threshold", None, actor=_ACTOR)
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            row = (
                await connection.execute(
                    sa.text(
                        "SELECT previous_value IS NULL, previous_value = 'null'::jsonb, "
                        "new_value = 'null'::jsonb FROM config_change_event"
                    )
                )
            ).one()
    finally:
        await engine.dispose()
    assert row == (False, True, True)


async def test_setting_a_value_it_already_has_is_still_recorded() -> None:
    """The log records what the operator did, and pressing the button is a thing they did."""
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    repeat = await record_config_change(_SCOPE, _TARGET, "enabled", True, actor="someone_else")
    assert repeat.is_initial is False
    assert repeat.previous_value is True
    assert repeat.new_value is True
    assert len(await config_history(scope=_SCOPE, target=_TARGET, field="enabled")) == 2


# --- Reconstruction --------------------------------------------------------


async def test_current_value_is_reconstructed_from_the_event_history() -> None:
    """There is no stored current value: the latest event *is* the value."""
    for value in (True, False, True, False):
        await record_config_change(_SCOPE, _TARGET, "enabled", value, actor=_ACTOR)
    assert await current_value(_SCOPE, _TARGET, "enabled") is False
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    assert await current_value(_SCOPE, _TARGET, "enabled") is True


async def test_current_value_raises_when_the_key_has_never_been_set() -> None:
    with pytest.raises(ConfigNotSetError, match="no configuration event"):
        await current_value(_SCOPE, "never_configured", "enabled")


async def test_current_values_returns_the_latest_of_each_field() -> None:
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    await record_config_change(_SCOPE, _TARGET, "weight", 0.1, actor=_ACTOR)
    await record_config_change(_SCOPE, _TARGET, "weight", 0.4, actor=_ACTOR)
    assert await current_values(_SCOPE, _TARGET) == {"enabled": True, "weight": 0.4}
    assert await current_values(_SCOPE, "unconfigured") == {}


async def test_current_scope_values_groups_every_target_in_one_subsystem() -> None:
    """The shape P5.5's catalog and CC.4's settings page read in one round trip."""
    await record_config_change(_SCOPE, "momentum", "enabled", True, actor=_ACTOR)
    await record_config_change(_SCOPE, "momentum", "enabled", False, actor=_ACTOR)
    await record_config_change(_SCOPE, "value", "enabled", True, actor=_ACTOR)
    await record_config_change("settings", "scheduler", "enabled", True, actor=SYSTEM_ACTOR)

    assert await current_scope_values(_SCOPE) == {
        "momentum": {"enabled": False},
        "value": {"enabled": True},
    }
    assert await current_scope_values("settings") == {"scheduler": {"enabled": True}}
    assert await current_scope_values("nothing_here") == {}


async def test_keys_are_isolated_across_scope_target_and_field() -> None:
    """Three components address one key; a change to one must not move another."""
    await record_config_change("a", "t", "f", 1, actor=_ACTOR)
    await record_config_change("b", "t", "f", 2, actor=_ACTOR)
    await record_config_change("a", "u", "f", 3, actor=_ACTOR)
    await record_config_change("a", "t", "g", 4, actor=_ACTOR)
    assert await current_value("a", "t", "f") == 1
    assert await current_value("b", "t", "f") == 2
    assert await current_value("a", "u", "f") == 3
    assert await current_value("a", "t", "g") == 4


# --- Actor and correlation id ----------------------------------------------


async def test_actor_is_recorded_verbatim_per_event() -> None:
    first = await record_config_change(_SCOPE, _TARGET, "enabled", True, actor="alice")
    second = await record_config_change(_SCOPE, _TARGET, "enabled", False, actor=SYSTEM_ACTOR)
    assert first.actor == "alice"
    assert second.actor == SYSTEM_ACTOR


async def test_correlation_id_defaults_to_the_request_in_flight() -> None:
    """A change made during a request is joinable to that request's log lines (D-003)."""
    tokens = structlog.contextvars.bind_contextvars(request_id="req-from-middleware")
    try:
        event = await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
    assert event.correlation_id == "req-from-middleware"


async def test_correlation_id_is_null_outside_a_request_rather_than_invented() -> None:
    event = await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=SYSTEM_ACTOR)
    assert event.correlation_id is None


async def test_history_can_be_filtered_to_one_request() -> None:
    """Answering "what did this request change" is why the correlation id is stored."""
    tokens = structlog.contextvars.bind_contextvars(request_id="req-A")
    try:
        await record_config_changes(_SCOPE, _TARGET, {"enabled": True, "weight": 0.5}, actor=_ACTOR)
    finally:
        structlog.contextvars.reset_contextvars(**tokens)
    await record_config_change(_SCOPE, _TARGET, "enabled", False, actor=_ACTOR)

    from_request = await config_history(correlation_id="req-A")
    assert {(event.field, event.new_value) for event in from_request} == {
        ("enabled", True),
        ("weight", 0.5),
    }
    assert len(await config_history()) == 3


# --- Batch recording -------------------------------------------------------


async def test_batch_records_one_event_per_field() -> None:
    events = await record_config_changes(
        _SCOPE,
        _TARGET,
        {"enabled": True, "weight": 0.25, "notes": None},
        actor=_ACTOR,
        correlation_id="req-batch",
    )
    assert len(events) == 3
    assert {event.field for event in events} == {"enabled", "weight", "notes"}
    assert all(event.correlation_id == "req-batch" for event in events)
    assert all(event.is_initial for event in events)
    assert await current_values(_SCOPE, _TARGET) == {
        "enabled": True,
        "weight": 0.25,
        "notes": None,
    }


async def test_batch_derives_previous_values_from_earlier_events() -> None:
    await record_config_changes(_SCOPE, _TARGET, {"enabled": False, "weight": 0.1}, actor=_ACTOR)
    events = await record_config_changes(
        _SCOPE, _TARGET, {"enabled": True, "weight": 0.9}, actor=_ACTOR
    )
    by_field = {event.field: event for event in events}
    assert by_field["enabled"].previous_value is False
    assert by_field["weight"].previous_value == 0.1
    assert not any(event.is_initial for event in events)


async def test_rejected_batch_writes_nothing() -> None:
    """Validation runs before the transaction opens, so a bad field leaves no half-change."""
    with pytest.raises(TypeError, match="not a JSON value"):
        await record_config_changes(
            _SCOPE,
            _TARGET,
            {"enabled": True, "bad": dt.datetime.now(dt.UTC)},  # type: ignore[dict-item]
            actor=_ACTOR,
        )
    assert await config_history() == ()


async def test_batch_events_share_one_instant_and_are_still_ordered_by_id() -> None:
    """One operator action is one instant; sequence is carried by the id, not the clock.

    ``recorded_at`` defaults to ``now()``, the *transaction* start, so all three
    events below carry the same timestamp — which is the honest rendering of a
    single act, and simultaneously the reason nothing in this module orders by
    the clock. The history order must still be exact, which only ``event_id``
    can deliver.
    """
    events = await record_config_changes(
        _SCOPE, _TARGET, {"enabled": True, "weight": 0.25, "notes": None}, actor=_ACTOR
    )
    assert len({event.recorded_at for event in events}) == 1

    newest_first = await config_history(scope=_SCOPE, target=_TARGET)
    ids = [event.event_id for event in newest_first]
    assert ids == sorted(ids, reverse=True)
    assert ids == sorted((event.event_id for event in events), reverse=True)


async def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one field"):
        await record_config_changes(_SCOPE, _TARGET, {}, actor=_ACTOR)


# --- History ordering and limits -------------------------------------------


async def test_history_is_newest_first_and_respects_its_limit() -> None:
    for value in range(5):
        await record_config_change(_SCOPE, _TARGET, "weight", value, actor=_ACTOR)
    newest_first = await config_history(scope=_SCOPE, target=_TARGET, field="weight")
    assert [event.new_value for event in newest_first] == [4, 3, 2, 1, 0]
    assert [event.new_value for event in await config_history(limit=2)] == [4, 3]


async def test_history_pages_by_keyset_without_repeating_or_skipping_an_event() -> None:
    """Paging an append-only log must not shift rows between pages (CC.4's browser).

    Appends land *between* page reads in any live system, which is exactly when
    ``OFFSET`` paging shows one entry twice and hides another — indistinguishable,
    to a reader of an audit trail, from tampering. The keyset cursor is bounded by
    an id that does not move, so a change recorded mid-scroll appears on a later
    refresh and never disturbs the page being read.
    """
    for value in range(5):
        await record_config_change(_SCOPE, _TARGET, "weight", value, actor=_ACTOR)

    first_page = await config_history(limit=2)
    assert [event.new_value for event in first_page] == [4, 3]

    await record_config_change(_SCOPE, _TARGET, "weight", 99, actor=_ACTOR)

    second_page = await config_history(limit=2, before_event_id=first_page[-1].event_id)
    assert [event.new_value for event in second_page] == [2, 1]
    last_page = await config_history(limit=2, before_event_id=second_page[-1].event_id)
    assert [event.new_value for event in last_page] == [0]
    assert await config_history(limit=2, before_event_id=last_page[-1].event_id) == ()


async def test_history_cursor_composes_with_the_other_filters() -> None:
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    weights = [
        await record_config_change(_SCOPE, _TARGET, "weight", value, actor=_ACTOR)
        for value in (1, 2, 3)
    ]
    page = await config_history(
        scope=_SCOPE, target=_TARGET, field="weight", before_event_id=weights[-1].event_id
    )
    assert [event.new_value for event in page] == [2, 1]


async def test_history_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit must be >= 1"):
        await config_history(limit=0)


# --- Append-only enforcement at the database -------------------------------


async def test_update_is_rejected_by_the_database_trigger() -> None:
    """A past event cannot be rewritten, whatever issues the statement."""
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    with pytest.raises(DBAPIError, match="append-only"):
        await _raw_execute("UPDATE config_change_event SET new_value = 'false'::jsonb")


async def test_actor_cannot_be_rewritten_after_the_fact() -> None:
    """The specific tampering an audit log exists to prevent, named as its own case."""
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    with pytest.raises(DBAPIError, match="append-only"):
        await _raw_execute("UPDATE config_change_event SET actor = 'somebody_else'")
    (event,) = await config_history()
    assert event.actor == _ACTOR


async def test_delete_is_rejected_by_the_database_trigger() -> None:
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    with pytest.raises(DBAPIError, match="append-only"):
        await _raw_execute("DELETE FROM config_change_event")
    assert len(await config_history()) == 1


async def test_orm_dml_through_the_writer_session_is_rejected_too() -> None:
    """The trigger is path-independent: ORM UPDATE/DELETE die at the database as well."""
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(sa.update(ConfigChangeEvent).values(actor="forged"))
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(sa.delete(ConfigChangeEvent))


# --- Database-level constraints --------------------------------------------


@pytest.mark.parametrize("component", ["actor", "scope", "target", "field"])
async def test_empty_address_components_are_refused_by_the_database(component: str) -> None:
    """Not only by the writer's validation: a direct INSERT is refused too."""
    columns = {"actor": _ACTOR, "scope": _SCOPE, "target": _TARGET, "field": "enabled"}
    columns[component] = ""
    async with ingest_writer_session() as session:
        session.add(
            ConfigChangeEvent(
                **columns, previous_value=None, new_value=True, is_initial=True, correlation_id=None
            )
        )
        with pytest.raises(IntegrityError, match=f"{component}_not_empty"):
            await session.flush()


async def test_initial_event_claiming_a_previous_value_is_refused_by_the_database() -> None:
    """``is_initial`` and ``previous_value`` cannot be made to disagree, even by hand."""
    async with ingest_writer_session() as session:
        session.add(
            ConfigChangeEvent(
                actor=_ACTOR,
                scope=_SCOPE,
                target=_TARGET,
                field="enabled",
                previous_value=False,
                new_value=True,
                is_initial=True,
                correlation_id=None,
            )
        )
        with pytest.raises(IntegrityError, match="initial_event_has_no_previous_value"):
            await session.flush()


async def test_recorded_at_comes_from_the_database_clock() -> None:
    """No caller supplies the instant, so no caller can misdate an entry."""
    async with ingest_writer_session() as session:
        event = ConfigChangeEvent(
            actor=_ACTOR,
            scope=_SCOPE,
            target=_TARGET,
            field="enabled",
            previous_value=None,
            new_value=True,
            is_initial=True,
            correlation_id=None,
        )
        session.add(event)
        await session.flush()
        await session.refresh(event)
        recorded_at = event.recorded_at
        await session.commit()
    assert recorded_at.tzinfo is not None
    assert recorded_at.utcoffset() == dt.timedelta(0)
    assert abs((dt.datetime.now(dt.UTC) - recorded_at).total_seconds()) < 60


# --- Schema installed by migration 0007 ------------------------------------


async def _catalog_scalar(query: str) -> object:
    """Return one scalar from a catalog query on an admin connection."""
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            return (await connection.execute(sa.text(query))).scalar()
    finally:
        await engine.dispose()


async def test_append_only_trigger_is_installed_for_update_and_delete_only() -> None:
    """TRUNCATE stays the sanctioned reset path (0003/0004 rationale), so it is not blocked."""
    events = await _catalog_scalar(
        "SELECT string_agg(event_manipulation, ',' ORDER BY event_manipulation) "
        "FROM information_schema.triggers "
        "WHERE trigger_name = 'trg_config_change_event_append_only'"
    )
    assert events == "DELETE,UPDATE"


async def test_append_only_trigger_fires_before_each_row_like_0003_and_0004() -> None:
    """The *shape* of the guard, asserted at the database rather than in the source text.

    ``BEFORE`` so the mutation never happens, and ``FOR EACH ROW`` rather than
    per statement — the established pattern from revisions 0003/0004, where row
    triggers are also what makes Timescale propagate the guard onto chunks. A
    statement-level or AFTER trigger would still make the tests above pass while
    being a materially weaker guard, so the shape is pinned on its own.
    """
    shape = await _catalog_scalar(
        "SELECT DISTINCT action_timing || '|' || action_orientation "
        "FROM information_schema.triggers "
        "WHERE trigger_name = 'trg_config_change_event_append_only'"
    )
    assert shape == "BEFORE|ROW"


@pytest.mark.parametrize(
    ("index_name", "expected"),
    [
        ("ix_config_change_event_key", "(scope, target, field, event_id DESC)"),
        ("ix_config_change_event_recorded_at", "(recorded_at DESC)"),
        ("ix_config_change_event_correlation_id", "WHERE (correlation_id IS NOT NULL)"),
    ],
)
async def test_migration_created_the_read_path_indices(index_name: str, expected: str) -> None:
    """Each index exists *and* has the shape the query it serves needs.

    Existence alone would pass for an index on the wrong columns or in the
    wrong order, which is exactly the drift that turns the current-value
    reconstruction from an index seek into a sequential scan without failing
    anything.
    """
    definition = await _catalog_scalar(
        # S608: fixed parametrize literals, not external input.
        f"SELECT indexdef FROM pg_indexes WHERE indexname = '{index_name}'"  # noqa: S608
    )
    assert definition is not None
    assert expected in str(definition)


async def test_orm_model_and_hand_written_migration_agree_on_the_columns() -> None:
    """Migration 0007 and the ORM model are written twice and must say the same thing.

    Nothing else forces them to agree: the migration is hand-written DDL (this
    repo does not run autogenerate) and the model is declared independently, so
    a column added to one and forgotten in the other would first surface as a
    runtime error inside whichever consumer touched it — the worst place to
    learn about it. Nullability is included because it carries meaning here:
    only ``correlation_id`` may be NULL, and both value columns being NOT NULL
    is what forces ``None`` to be stored as JSON null.
    """
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    sa.text(
                        "SELECT column_name, data_type, is_nullable "
                        "FROM information_schema.columns "
                        "WHERE table_name = 'config_change_event'"
                    )
                )
            ).all()
    finally:
        await engine.dispose()

    in_database = {str(row[0]): (str(row[1]), row[2] == "YES") for row in rows}
    table = cast("Table", cast("Any", ConfigChangeEvent).__table__)
    in_model = {column.name: column.nullable for column in table.columns}

    assert set(in_database) == set(in_model)
    assert {name: nullable for name, (_, nullable) in in_database.items()} == in_model
    assert in_database["recorded_at"][0] == "timestamp with time zone"
    assert in_database["previous_value"][0] == "jsonb"
    assert in_database["new_value"][0] == "jsonb"


async def test_audit_table_is_readable_without_an_as_of() -> None:
    """It is not in the bitemporal registry, so the as-of read guard does not scope it.

    Stated as a test because the opposite mistake — adding the mixin to make the
    table "consistent" with its neighbours — would make every configuration read
    require an as-of instant that configuration does not have.
    """
    await record_config_change(_SCOPE, _TARGET, "enabled", True, actor=_ACTOR)
    async with ingest_writer_session() as session:
        rows = (await session.scalars(sa.select(ConfigChangeEvent))).all()
    assert len(rows) == 1
