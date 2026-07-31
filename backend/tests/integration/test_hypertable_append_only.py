"""P2.5 integration: hypertable layout and database-enforced append-only (D-011).

Verifies against real TimescaleDB that ``price_bar`` is a hypertable
partitioned on ``valid_from`` with 1-month chunks, that ``security_master``
stays a plain table, that the as-of composite indices exist, and that
UPDATE/DELETE on the fact tables are rejected by triggers regardless of the
access path (raw admin connection or ORM session) — plus the schema
constraints migration 0002 promised (``knowledge_time`` NOT NULL with no
default, ``valid_from < valid_to``).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import ingest_writer_session

# TID251: DB-level trigger tests must reach Postgres *beneath* the application
# guard — see the helper docstrings below for why this is the correct engine.
from backend.db.engine import _create_migration_engine as _migration_engine  # noqa: TID251
from backend.db.models import PriceBar
from backend.tests.integration.factories import (
    bar_version,
    create_security,
    insert_rows,
    master_version,
)

_DAY = dt.date(2024, 1, 5)
_K1 = dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC)


async def _unguarded_scalar(query: str) -> object:
    """Run a catalog query on the unguarded migration engine, return one scalar.

    Catalog introspection quotes fact-table names as string literals, which
    the Core guard's conservative word-boundary scan rejects by design
    (documented fail-closed false positive, D-011). Infrastructure queries are
    a sanctioned use of the module-private migration engine.
    """
    engine = _migration_engine()
    try:
        async with engine.connect() as connection:
            return (await connection.execute(sa.text(query))).scalar()
    finally:
        await engine.dispose()


async def _expect_append_only_rejection(statement: str) -> None:
    """Assert the *database trigger* rejects a raw mutation of a fact table.

    Deliberately issued through the unguarded migration engine. The append-only
    triggers are the defense-in-depth layer *beneath* the application-level
    Core guard: anything that reaches Postgres at all — psql, a future service,
    a mistaken admin script — must still be refused. Routing this through a
    guarded engine would only re-test the app guard (covered separately in
    ``test_bypass_impossibility_db.py``) and would leave the trigger itself
    unexercised.
    """
    engine = _migration_engine()
    try:
        with pytest.raises(DBAPIError, match="append-only"):
            async with engine.begin() as connection:
                await connection.execute(sa.text(statement))
    finally:
        await engine.dispose()


async def test_price_bar_is_a_hypertable_and_security_master_is_not() -> None:
    hypertables = await _unguarded_scalar(
        "SELECT array_agg(hypertable_name) FROM timescaledb_information.hypertables"
    )
    assert hypertables == ["price_bar"]


async def test_hypertable_partitions_on_valid_from_with_one_month_chunks() -> None:
    dimension = await _unguarded_scalar(
        "SELECT column_name || '|' || time_interval::text "
        "FROM timescaledb_information.dimensions WHERE hypertable_name = 'price_bar'"
    )
    # TimescaleDB stores chunk intervals as fixed durations: the migration's
    # INTERVAL '1 month' is normalized to 30 days for timestamptz dimensions.
    assert dimension == "valid_from|30 days"


@pytest.mark.parametrize("table", ["price_bar", "security_master"])
async def test_asof_composite_index_exists(table: str) -> None:
    index_definition = await _unguarded_scalar(
        # S608 suppressed: `table` is a fixed parametrize literal, not external input.
        f"SELECT indexdef FROM pg_indexes WHERE indexname = 'ix_{table}_asof_lookup'"  # noqa: S608
    )
    assert index_definition is not None
    assert "(security_id, valid_from, knowledge_time DESC)" in str(index_definition)


async def test_update_on_price_bar_rejected_by_trigger() -> None:
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _K1, "100"))
    await _expect_append_only_rejection("UPDATE price_bar SET close_usd = 0")


async def test_delete_on_price_bar_rejected_by_trigger() -> None:
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _K1, "100"))
    await _expect_append_only_rejection("DELETE FROM price_bar")


async def test_update_and_delete_on_security_master_rejected_by_trigger() -> None:
    security_id = await create_security()
    await insert_rows(master_version(security_id, "ACME", _K1))
    await _expect_append_only_rejection("UPDATE security_master SET ticker = 'EVIL'")
    await _expect_append_only_rejection("DELETE FROM security_master")


async def test_orm_update_through_writer_session_also_rejected() -> None:
    """The trigger is role- and path-independent: ORM DML dies at the database."""
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _K1, "100"))
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError, match="append-only"):
            await session.execute(sa.update(PriceBar).values(close_usd=Decimal("0")))


async def test_knowledge_time_is_not_null_and_has_no_default() -> None:
    """A writer that forgets knowledge_time is rejected by the database (D-011)."""
    security_id = await create_security()
    incomplete = bar_version(security_id, _DAY, _K1, "100")
    del incomplete.knowledge_time  # unset: the INSERT omits the column, no default exists
    async with ingest_writer_session() as session:
        session.add(incomplete)
        with pytest.raises(IntegrityError, match="knowledge_time"):
            await session.flush()


async def test_empty_valid_interval_rejected_by_check_constraint() -> None:
    security_id = await create_security()
    degenerate = bar_version(security_id, _DAY, _K1, "100")
    degenerate.valid_to = degenerate.valid_from  # empty half-open interval
    async with ingest_writer_session() as session:
        session.add(degenerate)
        with pytest.raises(IntegrityError, match="valid_interval"):
            await session.flush()
