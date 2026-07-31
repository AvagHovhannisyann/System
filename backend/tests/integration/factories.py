"""Row-construction and insertion helpers for the bitemporal integration tests.

All writes go through the sanctioned ``ingest_writer_session`` path; every
row's ``knowledge_time`` is supplied explicitly (D-011 — there is no
default). Prices are USD per share, volume in shares, adjustment factor
dimensionless.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING

from backend.db import ingest_writer_session
from backend.db.models import PriceBar, Security, SecurityMaster

if TYPE_CHECKING:
    from backend.db.base import Base


async def create_security() -> int:
    """Create one identity-anchor row and return its database-generated id."""
    async with ingest_writer_session() as session:
        security = Security()
        session.add(security)
        await session.flush()
        security_id = security.security_id
        await session.commit()
    return security_id


async def insert_rows(*rows: Base) -> None:
    """Insert the given ORM rows in one writer session and commit."""
    async with ingest_writer_session() as session:
        session.add_all(rows)
        await session.commit()


def bar_version(
    security_id: int,
    day: dt.date,
    knowledge_time: dt.datetime,
    close: str,
    *,
    is_retraction: bool = False,
) -> PriceBar:
    """Build one PriceBar version for trading day ``day``.

    Event time is the half-open day interval ``[D 00:00Z, D+1 00:00Z)``
    (D-011). ``close`` (USD/share, decimal string) fills every price column
    so assertions can key on a single value; volume fixed at 1000 shares,
    adjustment factor 1 (raw == adjusted).
    """
    valid_from = dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC)
    price = Decimal(close)
    return PriceBar(
        security_id=security_id,
        valid_from=valid_from,
        valid_to=valid_from + dt.timedelta(days=1),
        knowledge_time=knowledge_time,
        is_retraction=is_retraction,
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal("1"),
        volume_shares=1000,
    )


def master_version(
    security_id: int,
    ticker: str,
    knowledge_time: dt.datetime,
    *,
    is_retraction: bool = False,
) -> SecurityMaster:
    """Build one SecurityMaster identity version (open-ended valid interval).

    ``valid_to`` is left to its ``'infinity'`` server default; the identity
    applies from 2020-01-01Z until superseded by a later-knowledge version.
    """
    return SecurityMaster(
        security_id=security_id,
        ticker=ticker,
        name=f"{ticker} Corp.",
        exchange="XNAS",
        valid_from=dt.datetime(2020, 1, 1, tzinfo=dt.UTC),
        knowledge_time=knowledge_time,
        is_retraction=is_retraction,
    )
