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
  convention, so the restriction must be enforced, not documented).

A failure here is a real unversioned read path from application code — a
gate failure to fix in ``backend/db``, never by weakening the test (I6).
"""

from __future__ import annotations

import datetime as dt

import pytest
import sqlalchemy as sa
from sqlalchemy import select

from backend.db import (
    BitemporalBypassError,
    as_of,
    create_admin_engine,
    ingest_writer_session,
)
from backend.db.models import PriceBar, Security
from backend.tests.integration.factories import bar_version, create_security, insert_rows

_DAY = dt.date(2024, 1, 5)
_K1 = dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC)  # original version knowable
_K2 = dt.datetime(2024, 1, 9, 21, 0, tzinfo=dt.UTC)  # restatement knowable
_BETWEEN = dt.datetime(2024, 1, 7, 0, 0, tzinfo=dt.UTC)  # K1 < _BETWEEN < K2


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
