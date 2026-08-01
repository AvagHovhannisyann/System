"""P2.7 gate (end-to-end): bypass impossibility proven against real TimescaleDB.

Directive Phase 2 gate: "a test proving that direct table access outside the
query layer is impossible from application code." These tests attempt real
bypasses through *publicly reachable* API only, against the live database:

- the ``as_of`` control leg: the sanctioned path works and is versioned;
- textual SQL naming a fact table raises even on a bound as-of session;
- a writer-session ``INSERT ... FROM SELECT`` sourcing a bitemporal table
  must raise — the embedded SELECT is a read, and it must not execute
  unversioned just because the top-level statement is DML;
- ``Session.connection()`` — public ``AsyncSession`` API on a session
  obtained from the public ``as_of``/writer paths — must not yield a raw
  Core read of a fact table;
- ``create_admin_engine()`` — a public ``backend.db`` export — must not
  permit a Core SELECT of a fact table (its sanctioned uses are health
  probes and migration tooling; D-011 explicitly rejects restriction by
  convention, so the restriction must be enforced, not documented);
- ``exec_driver_sql`` naming a fact table must raise;
- and the enforcement model itself: with the structural walker **stubbed
  blind**, a real fact-table read is still refused at the SQL boundary,
  while every legitimate path (writer ``INSERT ... VALUES``, ``as_of``
  reads, column loads, ``SELECT 1``, catalog queries) keeps working.

A failure here is a real unversioned read path from application code — a
gate failure to fix in ``backend/db``, never by weakening the test (I6).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
import sqlalchemy as sa
from sqlalchemy import select

from backend.db import (
    BitemporalBypassError,
    _guard,
    as_of,
    create_admin_engine,
    ingest_writer_session,
)
from backend.db import asof as asof_module
from backend.db._guard import References
from backend.db.models import PriceBar, Security
from backend.tests.integration.factories import bar_version, create_security, insert_rows

_DAY = dt.date(2024, 1, 5)
_K1 = dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC)  # original version knowable
_K2 = dt.datetime(2024, 1, 9, 21, 0, tzinfo=dt.UTC)  # restatement knowable
_BETWEEN = dt.datetime(2024, 1, 7, 0, 0, tzinfo=dt.UTC)  # K1 < _BETWEEN < K2
_OTHER_DAY = dt.datetime(2024, 1, 12, tzinfo=dt.UTC)  # distinct fact for the write tests


async def _seed_bar_with_later_restatement() -> int:
    """One fact, two versions: only the K1 version is knowable at ``_BETWEEN``."""
    security_id = await create_security()
    await insert_rows(
        bar_version(security_id, _DAY, _K1, "100"),
        bar_version(security_id, _DAY, _K2, "999"),
    )
    return security_id


async def test_control_leg_as_of_session_reads_versioned_data() -> None:
    """The sanctioned path passes — and returns the version knowable then."""
    await _seed_bar_with_later_restatement()
    async with as_of(_BETWEEN) as session:
        bars = list((await session.scalars(select(PriceBar))).all())
    assert [str(bar.close_usd) for bar in bars] == ["100.000000"]


async def test_textual_sql_on_bound_as_of_session_raises() -> None:
    """Textual SQL cannot be versioned, so it fails closed even under as_of."""
    async with as_of(_BETWEEN) as session:
        with pytest.raises(BitemporalBypassError, match="price_bar"):
            await session.execute(sa.text("SELECT close_usd FROM price_bar"))


async def _writer_insert_from_select_on_fact_table() -> None:
    """Attempted bypass: DML whose source is an unversioned bitemporal SELECT."""
    statement = sa.insert(Security).from_select(
        ["security_id"],
        select(PriceBar.security_id + sa.literal(1_000_000)).distinct(),
    )
    async with ingest_writer_session() as session:
        await session.execute(statement)


async def test_writer_insert_from_select_sourcing_fact_table_raises() -> None:
    """An embedded SELECT inside INSERT must not read a fact table unversioned.

    ``ingest_writer_session`` permits INSERT, but the FROM-SELECT clause here
    *reads* ``price_bar`` with no as-of bound — all versions, future
    knowledge included. If this executes instead of raising, the writer path
    is an unversioned read path (I1 violation).
    """
    await _seed_bar_with_later_restatement()
    with pytest.raises(BitemporalBypassError):
        await _writer_insert_from_select_on_fact_table()


async def _read_fact_table_via_session_connection() -> int:
    """Attempted bypass: raw Core connection obtained from a public session."""
    async with as_of(_BETWEEN) as session:
        connection = await session.connection()
        result = await connection.execute(
            sa.text("SELECT count(*) FROM price_bar WHERE knowledge_time > :as_of"),
            {"as_of": _BETWEEN},
        )
        return int(result.scalar_one())


async def test_session_connection_cannot_read_fact_tables_raw() -> None:
    """``Session.connection()`` must not open an unversioned read path.

    ``AsyncSession.connection()`` is public API on the session that the
    public ``as_of``/``ingest_writer_session`` paths hand out; Core
    statements on it never pass the ORM enforcement hook. If this query runs
    it returns rows whose ``knowledge_time`` exceeds the session's own as-of
    — future knowledge, through nothing but sanctioned imports.
    """
    await _seed_bar_with_later_restatement()
    with pytest.raises(BitemporalBypassError):
        await _read_fact_table_via_session_connection()


async def _read_fact_table_via_admin_engine() -> None:
    """Attempted bypass: Core SELECT on a fact table via the public admin engine."""
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            await connection.execute(sa.text("SELECT knowledge_time FROM price_bar"))
    finally:
        await engine.dispose()


async def test_admin_engine_cannot_read_fact_tables() -> None:
    """The exported admin engine must reject fact-table SELECTs.

    ``create_admin_engine`` is on the curated public surface for health
    probes and migration tooling. Neither use reads fact rows, so a SELECT
    touching a bitemporal fact table through it must raise — otherwise the
    public surface itself is the bypass, guarded only by a docstring, which
    D-011 rejects ("application-level filtering by convention ... is how
    lookahead bias actually happens").
    """
    await _seed_bar_with_later_restatement()
    with pytest.raises(BitemporalBypassError):
        await _read_fact_table_via_admin_engine()


async def test_exec_driver_sql_naming_a_fact_table_raises() -> None:
    """``exec_driver_sql`` never fires ``before_execute`` — the SQL scan catches it."""
    await _seed_bar_with_later_restatement()
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            with pytest.raises(BitemporalBypassError, match="price_bar"):
                await connection.exec_driver_sql("SELECT count(*) FROM price_bar")
    finally:
        await engine.dispose()


# --- default-deny: the walker is no longer the thing I1 rests on ------------

_NO_REFERENCES = References(frozenset(), frozenset(), frozenset())

_SQL_BOUNDARY_REFUSAL = "unsanctioned SQL naming bitemporal fact table"
"""Substring identifying a refusal raised by the final-SQL arbiter.

Asserted instead of just the table name so the test cannot be satisfied by
some *other* layer catching the statement: the structural pre-checks word
their rejections differently.
"""


def _blind_walker(element: object, table_names: frozenset[str]) -> References:  # noqa: ARG001
    """Stand in for a clause shape the structural walker cannot see.

    Reports "no bitemporal references" for every statement — which is exactly
    what a walker blind spot looks like from the enforcement layer: not an
    error, just silence. Stubbing the walker tests the *architecture*;
    enumerating known-tricky shapes would only ever test the shapes we already
    thought of.
    """
    return _NO_REFERENCES


async def test_structural_blind_spot_is_refused_at_the_sql_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read the walker cannot see must raise, not return unversioned rows (I1).

    Both consumers of the walker are blinded, so the ORM hook declines to
    rewrite (and declines to sanction) and the Core structural pre-check sees
    nothing to object to. The execution is still refused, because the guard
    denies by default on the compiled SQL.

    On the pre-inversion code this call *succeeded* and returned both
    versions of the fact — including the ``999`` restatement whose
    ``knowledge_time`` is after this session's as-of, i.e. exactly the
    lookahead I1 exists to forbid — because the cursor-level name scan was
    skipped for anything compiled from a clause element.
    """
    await _seed_bar_with_later_restatement()
    monkeypatch.setattr(_guard, "collect_references", _blind_walker)
    monkeypatch.setattr(asof_module, "collect_references", _blind_walker)

    async with as_of(_BETWEEN) as session:
        with pytest.raises(BitemporalBypassError, match=_SQL_BOUNDARY_REFUSAL):
            await session.scalars(select(PriceBar))


# --- positive controls: every legitimate path still works -------------------


def _core_bar_values(security_id: int) -> dict[str, object]:
    """Column values for one bar, for a Core ``insert().values()`` statement."""
    price = Decimal("7")
    return {
        "security_id": security_id,
        "valid_from": _OTHER_DAY,
        "valid_to": _OTHER_DAY + dt.timedelta(days=1),
        "knowledge_time": _K1,
        "open_usd": price,
        "high_usd": price,
        "low_usd": price,
        "close_usd": price,
        "close_raw_usd": price,
        "adjustment_factor": Decimal("1"),
        "volume_shares": 1000,
    }


async def test_writer_insert_values_into_a_fact_table_succeeds() -> None:
    """The ingestion write path is sanctioned, in both of its shapes.

    ``INSERT INTO price_bar ... VALUES`` names a fact table in its SQL, so
    under default-deny it runs only because ``before_execute`` vets it (plain
    DML, no embedded read) and grants it a sanction naming that one table.
    Both shapes go through that grant: the ORM flush of ``session.add`` and
    an explicit Core ``insert().values()`` on the writer session.
    """
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _K1, "100"))  # ORM flush path

    async with ingest_writer_session() as session:
        await session.execute(sa.insert(PriceBar).values(**_core_bar_values(security_id)))
        await session.commit()

    async with as_of(_BETWEEN) as session:
        closes = sorted(
            str(bar.close_usd) for bar in (await session.scalars(select(PriceBar))).all()
        )
    assert closes == ["100.000000", "7.000000"]


async def test_as_of_read_returns_only_versions_knowable_then() -> None:
    """The sanctioned read path passes the SQL boundary and stays correct.

    The rewritten statement names ``price_bar`` inside its versioned
    subquery; it executes because the ORM hook stamped it, and the rows it
    returns are still the as-of-correct ones on both sides of the
    restatement.
    """
    await _seed_bar_with_later_restatement()

    async with as_of(_BETWEEN) as session:
        before = [str(bar.close_usd) for bar in (await session.scalars(select(PriceBar))).all()]
    async with as_of(_K2) as session:
        after = [str(bar.close_usd) for bar in (await session.scalars(select(PriceBar))).all()]

    assert before == ["100.000000"]
    assert after == ["999.000000"]


async def test_session_refresh_column_load_succeeds() -> None:
    """The documented column-load exemption survives default-deny.

    ``session.refresh`` re-reads one physical row addressed by the full
    primary key (``knowledge_time`` included), so it cannot time-travel; the
    writer needs it to read server-generated audit columns. It reaches
    Postgres only because the ORM hook stamps column loads sanctioned.
    """
    security_id = await create_security()
    async with ingest_writer_session() as session:
        bar = bar_version(security_id, _DAY, _K1, "100")
        session.add(bar)
        await session.flush()
        await session.refresh(bar)
        assert bar.ingested_at is not None
        await session.commit()


async def test_health_style_select_one_succeeds() -> None:
    """``/api/health``'s probe (``SELECT 1`` on an admin engine) is unaffected."""
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            assert (await connection.execute(sa.text("SELECT 1"))).scalar_one() == 1
    finally:
        await engine.dispose()


async def test_catalog_query_naming_no_fact_table_succeeds() -> None:
    """Catalog/monitoring queries pass, as long as their SQL names no fact table.

    Default-deny keys on the SQL text, not on the statement's provenance, so
    an admin-engine catalog query is admitted without any sanction. (A
    catalog query that *quotes* a fact-table name is the documented
    fail-closed false positive and belongs on the migration engine.)
    """
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                sa.text("SELECT count(*) FROM timescaledb_information.hypertables")
            )
            assert result.scalar_one() >= 1
    finally:
        await engine.dispose()
