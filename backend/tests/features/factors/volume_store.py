"""A fixture price store that also carries traded volume, for Amihud illiquidity.

The sibling of :mod:`backend.tests.features.factors.store`, which stands in for
``price_bar`` under as-of semantics but hands back only the three columns the
shared price loader selects. Amihud needs five — it is the only factor in the
package that reads ``close_raw_usd`` and ``volume_shares`` — so
:func:`backend.features.factors.liquidity.load_dollar_volume_bars` issues its own
query and needs its own double.

Everything that module says about what a double like this proves applies here
unchanged, and is not repeated: the as-of *read contract* is reproduced (versions
with ``knowledge_time <= as_of`` only, latest knowledge wins per (security,
trading day), a winning retraction hides the fact) so that the factor above it
can be tested without a database, while the claim that PostgreSQL enforces that
contract belongs to ``backend/tests/integration`` and is not made here.

The calendar helpers and the version-resolution rule are **imported** from that
module rather than re-derived, so the two doubles cannot drift into disagreeing
about what "visible" means. What is added is two columns and one convenience:
``raw_closes`` defaults to the adjusted closes, i.e. an adjustment factor of 1,
which is the uninteresting case for most tests and lets the interesting ones —
where the two bases differ and the factor has to pick the right one — stand out.

Every number written here is a hand-chosen fixture (round volumes, exact price
ratios) precisely so it could not be mistaken for a market observation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from backend.tests.features.factors.store import (
    DEFAULT_KNOWLEDGE_DELAY,
    business_days_ending,
    midnight_utc,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "FixtureVolumeBar",
    "FixtureVolumeResult",
    "FixtureVolumeSession",
    "FixtureVolumeStore",
    "volume_bars",
]

type VolumeRow = tuple[int, dt.datetime, Decimal, Decimal, int]
"""The row shape ``load_dollar_volume_bars``'s ``select()`` asks for."""


@dataclass(frozen=True, slots=True)
class FixtureVolumeBar:
    """One hand-written version of one daily price bar, with volume.

    Attributes:
        security_id: the security the bar belongs to.
        trading_date: the trading day, stored by D-011 as
            ``valid_from = trading_date 00:00Z``.
        close_usd: adjusted close, USD per share, as a ``Decimal`` because that
            is what the ``Numeric`` column yields.
        close_raw_usd: unadjusted close, USD per share, as printed on the trade
            date. Paired with ``volume_shares`` it gives the day's traded USD.
        volume_shares: unadjusted share count traded that day.
        knowledge_time: when this version became knowable (tz-aware UTC).
        is_retraction: whether this version retracts the fact.
    """

    security_id: int
    trading_date: dt.date
    close_usd: Decimal
    close_raw_usd: Decimal
    volume_shares: int
    knowledge_time: dt.datetime
    is_retraction: bool = False


def volume_bars(
    security_id: int,
    closes: Sequence[float],
    volumes: Sequence[int],
    *,
    last_trading_day: dt.date,
    raw_closes: Sequence[float] | None = None,
    knowledge_delay: dt.timedelta = DEFAULT_KNOWLEDGE_DELAY,
) -> tuple[FixtureVolumeBar, ...]:
    """Lay closes and volumes onto the business days ending at ``last_trading_day``.

    Args:
        security_id: the security these bars belong to.
        closes: adjusted closes in USD per share, oldest first. The last element
            lands on ``last_trading_day``.
        volumes: unadjusted share counts, same length and order as ``closes``.
        last_trading_day: the newest trading day in the series (a weekday).
        raw_closes: unadjusted closes in USD per share, same length and order.
            Defaults to ``closes`` — an adjustment factor of 1, the case where
            the two bases coincide and neither can stand in for the other by
            accident.
        knowledge_delay: how long after midnight opening a bar's trading day the
            bar becomes knowable. Defaults to
            :data:`~backend.tests.features.factors.store.DEFAULT_KNOWLEDGE_DELAY`.

    Returns:
        One bar per close, ascending by trading day.
    """
    unadjusted = list(closes) if raw_closes is None else list(raw_closes)
    days = business_days_ending(last_trading_day, len(closes))
    return tuple(
        FixtureVolumeBar(
            security_id=security_id,
            trading_date=day,
            close_usd=Decimal(str(close)),
            close_raw_usd=Decimal(str(raw)),
            volume_shares=volume,
            knowledge_time=midnight_utc(day) + knowledge_delay,
        )
        for day, close, raw, volume in zip(days, closes, unadjusted, volumes, strict=True)
    )


@dataclass(frozen=True, slots=True)
class FixtureVolumeStore:
    """Hand-written bars with volume, readable only through the as-of read contract.

    Attributes:
        bars: every version of every bar, in any order. Several versions of one
            (security, trading day) are the normal case — that is how a
            correction, a re-adjustment or a volume restatement is represented.
    """

    bars: tuple[FixtureVolumeBar, ...] = field(default_factory=tuple)

    def with_bars(self, extra: Iterable[FixtureVolumeBar]) -> FixtureVolumeStore:
        """Return a new store with ``extra`` versions added.

        Returns a copy rather than mutating, so a test can hold the "before"
        store and the "after" store at once and compare what each produces —
        the shape every lookahead test in this package takes.

        Args:
            extra: additional bar versions.

        Returns:
            A new store containing this store's bars followed by ``extra``.
        """
        return replace(self, bars=(*self.bars, *extra))

    def visible_rows(self, as_of: dt.datetime) -> list[VolumeRow]:
        """Return the rows a session pinned at ``as_of`` would see.

        Implements the D-011 read contract: only versions with
        ``knowledge_time <= as_of`` (boundary inclusive), the greatest
        ``knowledge_time`` wins per (security, trading day), and a winning
        retraction hides the fact entirely.

        Args:
            as_of: the pinned instant (tz-aware UTC).

        Returns:
            ``(security_id, valid_from, close_usd, close_raw_usd, volume_shares)``
            tuples in **descending** key order, so a loader that assumed the
            query's ``ORDER BY`` had been applied would get the reverse of what
            it expected rather than accidentally the right thing.
        """
        winners: dict[tuple[int, dt.date], FixtureVolumeBar] = {}
        for bar in self.bars:
            if bar.knowledge_time > as_of:
                continue
            key = (bar.security_id, bar.trading_date)
            current = winners.get(key)
            if current is None or bar.knowledge_time > current.knowledge_time:
                winners[key] = bar
        return [
            (
                bar.security_id,
                midnight_utc(bar.trading_date),
                bar.close_usd,
                bar.close_raw_usd,
                bar.volume_shares,
            )
            for _, bar in sorted(winners.items(), reverse=True)
            if not bar.is_retraction
        ]


class FixtureVolumeResult:
    """The slice of SQLAlchemy's ``Result`` the dollar-volume loader uses."""

    def __init__(self, rows: list[VolumeRow]) -> None:
        """Hold the rows this result will hand back.

        Args:
            rows: ``(security_id, valid_from, close_usd, close_raw_usd,
                volume_shares)`` tuples.
        """
        self._rows = rows

    def all(self) -> list[VolumeRow]:
        """Return every row, as ``Result.all()`` does."""
        return self._rows


class FixtureVolumeSession:
    """The slice of ``AsyncSession`` the dollar-volume loader uses, pinned.

    Answers every statement with **all** rows visible at its pinned instant,
    ignoring the statement's filters and ordering, so the loader cannot pass by
    relying on the database to have filtered by security, by calendar window or
    by order — it has to do those itself.

    Attributes:
        statements: every statement the code under test executed, so a test can
            assert what was asked for as well as what came back.
    """

    def __init__(self, store: FixtureVolumeStore, as_of: dt.datetime) -> None:
        """Pin a fixture store at an instant.

        Args:
            store: the hand-written bars.
            as_of: the instant this session is pinned at (tz-aware UTC).
        """
        self._store = store
        self._as_of = as_of
        self.statements: list[Any] = []

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> FixtureVolumeResult:  # noqa: ANN401 — mirrors AsyncSession.execute, which is itself untyped in its statement parameter
        """Record the statement and return every row visible at the pinned instant.

        Args:
            statement: the SQLAlchemy statement, recorded and otherwise ignored.
            *_args: ignored; present so the call shape matches ``AsyncSession``.
            **_kwargs: ignored; present so the call shape matches ``AsyncSession``.

        Returns:
            A result over the visible rows.
        """
        self.statements.append(statement)
        return FixtureVolumeResult(self._store.visible_rows(self._as_of))
