"""P2.3 integration: as-of read semantics against real TimescaleDB (D-011).

Covers the full versioned-read contract end to end: knowledge-time boundary
(inclusive), corrections flipping with ``as_of``, retraction masking,
latest-wins per (logical key, ``valid_from``), joins across two bitemporal
tables, and the writer-session read prohibition.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy import select

from backend.db import BitemporalBypassError, as_of, ingest_writer_session
from backend.db.models import PriceBar, SecurityMaster
from backend.tests.integration.factories import (
    bar_version,
    create_security,
    insert_rows,
    master_version,
)

_DAY = dt.date(2024, 1, 5)
_K1 = dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC)  # original bar knowable
_K2 = dt.datetime(2024, 1, 6, 9, 0, tzinfo=dt.UTC)  # correction knowable
_K3 = dt.datetime(2024, 1, 7, 0, 0, tzinfo=dt.UTC)  # retraction knowable


async def _bars_at(as_of_ts: dt.datetime) -> list[PriceBar]:
    async with as_of(as_of_ts) as session:
        result = await session.scalars(select(PriceBar).order_by(PriceBar.valid_from))
        return list(result.all())


async def test_fact_visible_from_knowledge_time_inclusive_invisible_before() -> None:
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _K1, "100"))

    assert await _bars_at(_K1 - dt.timedelta(microseconds=1)) == []

    at_boundary = await _bars_at(_K1)  # knowledge_time == as_of IS visible (D-011)
    assert len(at_boundary) == 1
    assert isinstance(at_boundary[0], PriceBar)
    assert at_boundary[0].close_usd == Decimal("100")

    later = await _bars_at(dt.datetime(2024, 2, 1, tzinfo=dt.UTC))
    assert len(later) == 1
    assert later[0].close_usd == Decimal("100")


async def test_correction_flips_with_as_of() -> None:
    security_id = await create_security()
    await insert_rows(
        bar_version(security_id, _DAY, _K1, "100"),
        bar_version(security_id, _DAY, _K2, "101"),
    )

    before_correction = await _bars_at(dt.datetime(2024, 1, 6, 0, 0, tzinfo=dt.UTC))
    assert [bar.close_usd for bar in before_correction] == [Decimal("100")]

    at_correction_boundary = await _bars_at(_K2)
    assert [bar.close_usd for bar in at_correction_boundary] == [Decimal("101")]

    long_after = await _bars_at(dt.datetime(2024, 3, 1, tzinfo=dt.UTC))
    assert [bar.close_usd for bar in long_after] == [Decimal("101")]


async def test_retraction_masks_the_fact_from_its_knowledge_time() -> None:
    security_id = await create_security()
    await insert_rows(
        bar_version(security_id, _DAY, _K1, "100"),
        bar_version(security_id, _DAY, _K2, "101"),
        bar_version(security_id, _DAY, _K3, "101", is_retraction=True),
    )

    still_visible = await _bars_at(dt.datetime(2024, 1, 6, 12, 0, tzinfo=dt.UTC))
    assert [bar.close_usd for bar in still_visible] == [Decimal("101")]

    assert await _bars_at(_K3) == []  # winning retraction hides the fact
    assert await _bars_at(dt.datetime(2024, 6, 1, tzinfo=dt.UTC)) == []


async def test_latest_wins_is_grouped_per_valid_interval() -> None:
    """A correction to one day must not disturb another day's bar."""
    security_id = await create_security()
    other_day = dt.date(2024, 1, 8)
    await insert_rows(
        bar_version(security_id, _DAY, _K1, "100"),
        bar_version(security_id, _DAY, _K2, "101"),
        bar_version(security_id, other_day, dt.datetime(2024, 1, 8, 21, 0, tzinfo=dt.UTC), "50"),
    )

    bars = await _bars_at(dt.datetime(2024, 2, 1, tzinfo=dt.UTC))
    assert [(bar.valid_from.date(), bar.close_usd) for bar in bars] == [
        (_DAY, Decimal("101")),
        (other_day, Decimal("50")),
    ]


async def test_column_only_select_is_versioned_too() -> None:
    security_id = await create_security()
    await insert_rows(
        bar_version(security_id, _DAY, _K1, "100"),
        bar_version(security_id, _DAY, _K2, "101"),
    )

    async with as_of(dt.datetime(2024, 2, 1, tzinfo=dt.UTC)) as session:
        closes = (await session.scalars(select(PriceBar.close_usd))).all()
    assert list(closes) == [Decimal("101")]


async def test_join_across_bitemporal_tables_respects_as_of() -> None:
    """Identity (master) and prices version independently; a join sees both as-of."""
    security_id = await create_security()
    rename_knowable = dt.datetime(2024, 6, 1, tzinfo=dt.UTC)
    await insert_rows(
        master_version(security_id, "ACME", dt.datetime(2024, 1, 2, tzinfo=dt.UTC)),
        master_version(security_id, "ACMEX", rename_knowable),
        bar_version(security_id, _DAY, _K1, "100"),
    )
    statement = select(SecurityMaster.ticker, PriceBar.close_usd).join_from(
        SecurityMaster,
        PriceBar,
        PriceBar.security_id == SecurityMaster.security_id,
    )

    async with as_of(dt.datetime(2024, 1, 10, tzinfo=dt.UTC)) as session:
        rows = list((await session.execute(statement)).tuples().all())
        assert rows == [("ACME", Decimal("100"))]

    async with as_of(dt.datetime(2024, 7, 1, tzinfo=dt.UTC)) as session:
        rows = list((await session.execute(statement)).tuples().all())
        assert rows == [("ACMEX", Decimal("100"))]


async def test_writer_session_unversioned_select_raises_against_real_db() -> None:
    security_id = await create_security()
    async with ingest_writer_session() as session:
        session.add(bar_version(security_id, _DAY, _K1, "100"))
        await session.flush()  # INSERT is permitted
        with pytest.raises(BitemporalBypassError, match="as_of"):
            await session.execute(select(PriceBar))
