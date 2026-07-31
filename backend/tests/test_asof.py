"""P2.3 unit tests: as_of timestamp guard, session binding, runtime enforcement.

Nothing here needs a database: the ``do_orm_execute`` hook raises *before*
any connection is acquired, and creating an ``AsyncSession`` performs no
I/O until first use. End-to-end behavior against real TimescaleDB lives in
``backend/tests/integration/test_asof_layer.py``.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session

from backend.db.asof import (
    AsOfTimestampError,
    BitemporalBypassError,
    _enforce_bitemporal_reads,
    as_of,
    ingest_writer_session,
)
from backend.db.models import PriceBar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, tzinfo=dt.UTC)


async def test_naive_timestamp_rejected() -> None:
    naive = dt.datetime(2020, 1, 1)  # noqa: DTZ001 — the point is to pass a naive value
    with pytest.raises(AsOfTimestampError, match="timezone-aware"):
        async with as_of(naive):
            pass


async def test_non_utc_offset_rejected() -> None:
    cet = dt.datetime(2020, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    with pytest.raises(AsOfTimestampError, match="UTC"):
        async with as_of(cet):
            pass


async def test_future_timestamp_rejected() -> None:
    future = dt.datetime.now(dt.UTC) + dt.timedelta(days=1)
    with pytest.raises(AsOfTimestampError, match="future"):
        async with as_of(future):
            pass


async def test_as_of_yields_session_bound_to_timestamp() -> None:
    """The yielded session carries the validated as-of in its info mapping."""
    ts = _utc(2020, 6, 1, 12)
    async with as_of(ts) as session:
        assert session.sync_session.info["bitemporal_as_of"] == ts


async def test_current_view_is_as_of_now() -> None:
    """D-011: 'current view' is simply as_of(now()) — now() itself must be accepted."""
    now = dt.datetime.now(dt.UTC)
    async with as_of(now) as session:
        assert session.sync_session.info["bitemporal_as_of"] == now


async def test_writer_session_is_marked_and_unversioned_select_raises() -> None:
    """Writer sessions permit INSERT (flush path) but reject bitemporal SELECTs."""
    async with ingest_writer_session() as session:
        assert session.sync_session.info["bitemporal_writer"] is True
        with pytest.raises(BitemporalBypassError, match="as_of"):
            await session.execute(select(PriceBar))


def test_enforcement_hook_registered_on_session_class() -> None:
    """D-011 layer 2: the hook listens on the ORM Session *class*.

    Every session in the process — including ones built by future code paths
    and the sync sessions inside every AsyncSession — is covered. It is
    registered as an import side effect of backend.db, which any model use
    triggers.
    """
    assert event.contains(Session, "do_orm_execute", _enforce_bitemporal_reads)


@pytest.fixture
async def hand_built_session() -> AsyncIterator[AsyncSession]:
    """A session deliberately created outside the sanctioned factories.

    The engine points at a closed port; no connection is ever attempted
    because enforcement raises first.
    """
    engine = create_async_engine("postgresql+asyncpg://nobody:nothing@127.0.0.1:9/none")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    session = factory()
    yield session
    await session.close()
    await engine.dispose()


async def test_hand_built_session_select_on_bitemporal_table_raises(
    hand_built_session: AsyncSession,
) -> None:
    with pytest.raises(BitemporalBypassError, match="price_bar"):
        await hand_built_session.execute(select(PriceBar))


async def test_hand_built_session_filtered_select_raises_too(
    hand_built_session: AsyncSession,
) -> None:
    """WHERE clauses do not launder a bypass; any statement shape is caught."""
    with pytest.raises(BitemporalBypassError, match="price_bar"):
        await hand_built_session.execute(
            select(PriceBar.close_usd).where(PriceBar.security_id == 1)
        )


async def test_error_message_names_the_read_path(
    hand_built_session: AsyncSession,
) -> None:
    with pytest.raises(BitemporalBypassError, match="as_of"):
        await hand_built_session.execute(select(PriceBar))
