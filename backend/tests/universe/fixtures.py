"""Hand-built inputs for the universe tests: nothing here is market data.

Every value below was chosen by a person to make an expected answer derivable by
hand: prices are round dollars, volumes are round share counts, and the
resulting dollar volumes are powers of ten. No number here was measured, sampled,
downloaded, or generated to resemble a measurement, and none of it is ever
written to the store — the fixtures exist purely so the screening arithmetic can
be checked against answers a reader can verify without running anything.

The identifiers are deliberately meaningless integers rather than tickers. A
fixture named ``AAPL`` invites a reader to check the numbers against what Apple
actually did, and they would not match, because these are not Apple's numbers.

``FIXTURE_REBALANCE_DATE`` is a Tuesday, and :func:`weekday_bars` lays bars on
consecutive weekdays, so the fixtures have the *shape* of an exchange calendar
without pretending to be one — public holidays are not modelled, which is
exactly why the ADV window is a generous calendar span rather than a calendar
lookup.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from backend.db.asof import AS_OF_INFO_KEY
from backend.db.models import PriceBar, SecurityMaster
from backend.universe.builder import DailyBar, ListedSecurity
from backend.universe.criteria import UniverseCandidate, UniverseCriteria

if TYPE_CHECKING:
    from collections.abc import Iterable

FIXTURE_AS_OF = dt.datetime(2026, 1, 2, tzinfo=dt.UTC)
"""Knowledge instant the fixture reads are made at — years after the rebalance dates.

Chosen far in the future on purpose: it is the situation a research read is
always in, and the one where survivorship bias enters. At this instant the store
*knows* which names were later delisted, so a builder that consults the current
identity version instead of the historical one produces a universe missing every
name that has since disappeared.
"""

FIXTURE_REBALANCE_DATE = dt.date(2020, 3, 31)
"""A Tuesday, chosen so the weekday bar generator ends on a trading-shaped day."""

FIXTURE_EXCHANGE = "XNYS"
"""A MIC code inside the default fixture whitelist."""


def fixture_criteria(
    *,
    min_adv_usd: str = "1000000",
    min_price_usd: str = "5",
    min_market_cap_usd: str = "300000000",
    allowed_exchanges: frozenset[str] = frozenset({"XNYS", "XNAS"}),
    require_borrow: bool = False,
    adv_lookback_days: int = 20,
) -> UniverseCriteria:
    """Build criteria from round numbers, with every threshold overridable.

    Defaults: $1,000,000/day ADV floor, $5/share price floor, $300,000,000
    market-cap floor, NYSE and Nasdaq, borrow screen off, 20-trading-day ADV
    window. They are ordinary-looking round numbers, not a recommended
    configuration.
    """
    return UniverseCriteria(
        min_adv_usd=Decimal(min_adv_usd),
        min_price_usd=Decimal(min_price_usd),
        min_market_cap_usd=Decimal(min_market_cap_usd),
        allowed_exchanges=allowed_exchanges,
        require_borrow=require_borrow,
        adv_lookback_days=adv_lookback_days,
    )


def listed_security(
    security_id: int,
    *,
    exchange: str = FIXTURE_EXCHANGE,
    first_listed_on: dt.date | None = None,
    delisted_on: dt.date | None = None,
) -> ListedSecurity:
    """Build one identity version, listed by default and never delisted."""
    return ListedSecurity(
        security_id=security_id,
        exchange=exchange,
        first_listed_on=first_listed_on,
        delisted_on=delisted_on,
    )


def weekday_bars(
    security_id: int,
    *,
    last_date: dt.date = FIXTURE_REBALANCE_DATE,
    count: int = 20,
    close_usd: str = "10",
    volume_shares: int = 1_000_000,
) -> tuple[DailyBar, ...]:
    """Build ``count`` identical bars on consecutive weekdays ending at ``last_date``.

    Identical on purpose: the median of a constant series is that constant, so
    the expected ADV is ``close_usd * volume_shares`` and needs no arithmetic to
    verify. Defaults give $10 x 1,000,000 shares = **$10,000,000/day**, ten
    times the default ADV floor.

    Returns:
        Bars ascending by trade date. ``last_date`` must itself be a weekday;
        the generator walks backwards skipping Saturdays and Sundays.
    """
    dates: list[dt.date] = []
    cursor = last_date
    while len(dates) < count:
        if cursor.weekday() < 5:  # Monday..Friday
            dates.append(cursor)
        cursor -= dt.timedelta(days=1)
    return tuple(
        DailyBar(
            security_id=security_id,
            trade_date=trade_date,
            close_raw_usd=Decimal(close_usd),
            volume_shares=volume_shares,
        )
        for trade_date in reversed(dates)
    )


def passing_candidate(
    security_id: int,
    *,
    exchange: str = FIXTURE_EXCHANGE,
    price_usd: str | None = "10",
    adv_usd: str | None = "10000000",
    market_cap_usd: str | None = "1000000000",
    borrow_available: bool | None = True,
) -> UniverseCandidate:
    """Build a candidate that passes every default screen, with each input overridable.

    The market cap and the borrow flag are supplied **by hand**. No code path in
    :mod:`backend.universe` can produce them: their sources do not exist
    (BLOCKERS.md B1 and B2), which is why every real build refuses. Supplying
    them here is what lets the screening arithmetic downstream of that refusal be
    exercised at all, and it is legitimate precisely because these are test
    inputs written out in full rather than values invented inside the system and
    presented as measurements.
    """
    return UniverseCandidate(
        security_id=security_id,
        exchange=exchange,
        price_usd=None if price_usd is None else Decimal(price_usd),
        adv_usd=None if adv_usd is None else Decimal(adv_usd),
        market_cap_usd=None if market_cap_usd is None else Decimal(market_cap_usd),
        borrow_available=borrow_available,
    )


@dataclass(frozen=True, slots=True)
class CannedResult:
    """The little of SQLAlchemy's ``Result`` that the builder's reads actually use."""

    rows: tuple[tuple[object, ...], ...]

    def all(self) -> tuple[tuple[object, ...], ...]:
        """Return the canned rows, positionally indexable exactly like ``Row``."""
        return self.rows


class FakeAsOfSession:
    """A stand-in for an ``as_of()``-scoped session that serves canned rows.

    **What this proves and what it does not.** It proves the builder's
    *orchestration*: that it validates the session and the criteria before
    issuing any statement, that it reads both source tables through the session
    it was handed, that it never constructs a session or an engine of its own
    (this object has neither, so a builder that tried would fail loudly), and
    that the rows flow into the assembly and the screen. It proves **nothing**
    about the SQL — no statement here is ever executed, no as-of rewrite runs,
    no append-only trigger fires. That is what
    ``backend/tests/integration/test_universe_db.py`` is for, against a real
    TimescaleDB.

    Statements are routed by the table they name, so the double does not depend
    on the order in which the builder happens to issue its reads.
    """

    def __init__(
        self,
        *,
        as_of: dt.datetime = FIXTURE_AS_OF,
        listings: Iterable[ListedSecurity] = (),
        bars: Iterable[DailyBar] = (),
    ) -> None:
        """Bind an as-of instant and the rows each of the two reads should return."""
        self.info: dict[str, object] = {AS_OF_INFO_KEY: as_of}
        self.executed: list[str] = []
        self._listing_rows = tuple(
            (item.security_id, item.exchange, item.first_listed_on, item.delisted_on)
            for item in listings
        )
        self._bar_rows = tuple(
            (
                bar.security_id,
                dt.datetime(
                    bar.trade_date.year, bar.trade_date.month, bar.trade_date.day, tzinfo=dt.UTC
                ),
                bar.close_raw_usd,
                bar.volume_shares,
            )
            for bar in bars
        )

    async def execute(self, statement: object) -> CannedResult:
        """Return the canned rows for whichever table the statement names."""
        sql = str(statement)
        self.executed.append(sql)
        if SecurityMaster.__tablename__ in sql:
            return CannedResult(self._listing_rows)
        if PriceBar.__tablename__ in sql:
            return CannedResult(self._bar_rows)
        message = f"the universe builder issued an unexpected statement: {sql}"
        raise AssertionError(message)


class RawRowSession:
    """A scoped-session stand-in serving arbitrary row tuples, whatever is asked.

    :class:`FakeAsOfSession` builds its rows from value objects and so can only
    produce well-formed ones. This one takes the tuples directly, which is what
    the reads' own guards — duplicate identity versions, a bar whose event-time
    interval does not start at midnight — have to be shown rows for.
    """

    def __init__(
        self,
        *,
        rows: Iterable[tuple[object, ...]] = (),
        as_of: dt.datetime = FIXTURE_AS_OF,
    ) -> None:
        """Bind an as-of instant and the rows every read should return."""
        self.info: dict[str, object] = {AS_OF_INFO_KEY: as_of}
        self._rows = tuple(rows)

    async def execute(self, statement: object) -> CannedResult:  # noqa: ARG002 — same rows always
        """Return the canned rows regardless of the statement."""
        return CannedResult(self._rows)
