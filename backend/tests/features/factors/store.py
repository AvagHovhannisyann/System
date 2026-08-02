"""A fixture price store standing in for ``price_bar`` under as-of semantics.

Every bar in these tests is written out by hand: a security, a trading day, a
close, and the instant that bar became knowable. Nothing is sampled, simulated
or downloaded, and no number here is presented as a market observation — the
values are chosen to make arithmetic checkable (round numbers, exact ratios)
precisely so that they could not be mistaken for real prices.

--------------------------------------------------------------------------
What this double stands in for, and what it therefore proves
--------------------------------------------------------------------------

:func:`backend.db.as_of` enforces invariant I1 in PostgreSQL, with a statement
rewrite and a cursor-level guard. Exercising it needs a TimescaleDB container,
which is what ``backend/tests/integration`` is for; those tests are where the
as-of layer itself is proven, including the property test over 10,000 random
knowledge times required by the Phase 2 gate.

:class:`FixtureAsOfSession` reproduces that layer's *read contract* — versions
with ``knowledge_time <= as_of`` only, latest knowledge wins per (security,
trading day), a winning retraction hides the fact — so that the factors on top
of it can be tested without a database. The division of labour is deliberate:

- **proven here**: that a factor's value is a function of the *visible* set and
  nothing else; that the loader applies its own calendar window rather than
  trusting the query to have applied it; that a bar which becomes visible too
  early is refused rather than consumed.
- **not proven here, and not claimed**: that PostgreSQL applies the knowledge
  bound correctly. That is the integration suite's job, and duplicating it
  against a hand-written double would prove only that the double agrees with
  itself.

The double **ignores the SQL statement entirely** and returns every visible row.
That is on purpose. It means the loader cannot pass these tests by relying on
the database to have filtered by security, by calendar window or by order — it
has to do those itself, which is exactly the property worth having when the real
query is one planner change away from returning rows in a different order.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_KNOWLEDGE_DELAY",
    "FixtureAsOfSession",
    "FixtureBar",
    "FixturePriceStore",
    "FixtureResult",
    "bars_from_closes",
    "business_days_ending",
    "midnight_utc",
]

_SATURDAY: Final = 5
"""``date.weekday()`` of Saturday; 5 and 6 are the weekend."""

DEFAULT_KNOWLEDGE_DELAY: Final = dt.timedelta(days=1)
"""When a fixture bar becomes knowable, relative to midnight opening its trading day.

One calendar day, i.e. midnight UTC *after* the trading day, which is one
plausible reading of D-011's documented default for a date-only source ("the
next trading day at 00:00Z, never the report date itself"). It is deliberately
later than the real close (16:00 ET, ~20:00-21:00 UTC on the trading day), so
these fixtures are conservative about what a factor is allowed to see and the
inclusive boundary at ``midnight_utc(compute_date)`` is exercised exactly: the
bar for ``compute_date - 1`` is knowable at the very instant a zero-lag feature
reads, and not a moment earlier.
"""


def midnight_utc(day: dt.date) -> dt.datetime:
    """Return midnight UTC opening ``day`` (tz-aware).

    Args:
        day: a calendar date.

    Returns:
        ``day`` at ``00:00:00+00:00`` — the instant D-011 stores as a daily
        bar's ``valid_from``.
    """
    return dt.datetime.combine(day, dt.time.min, tzinfo=dt.UTC)


def business_days_ending(last_day: dt.date, count: int) -> tuple[dt.date, ...]:
    """Return ``count`` consecutive Monday-Friday dates ending at ``last_day``.

    A stand-in for an exchange calendar with no holidays, which is the right
    simplification here: the factors count trading *bars*, so what matters to
    them is that consecutive bars are not consecutive calendar days. Holidays
    would only make the fixtures harder to read without exercising anything new.

    Args:
        last_day: the newest date in the result. Must itself be a weekday.
        count: how many dates to return (count), at least 1.

    Returns:
        The dates ascending, ending at ``last_day``.

    Raises:
        ValueError: if ``last_day`` is a weekend or ``count`` is not positive.
    """
    if last_day.weekday() >= _SATURDAY:
        msg = f"last_day {last_day.isoformat()} is a weekend; fixtures use trading days"
        raise ValueError(msg)
    if count < 1:
        msg = f"count must be at least 1; got {count}"
        raise ValueError(msg)
    days: list[dt.date] = []
    day = last_day
    while len(days) < count:
        if day.weekday() < _SATURDAY:
            days.append(day)
        day -= dt.timedelta(days=1)
    return tuple(reversed(days))


@dataclass(frozen=True, slots=True)
class FixtureBar:
    """One hand-written version of one daily price bar.

    Attributes:
        security_id: the security the bar belongs to.
        trading_date: the trading day, stored by D-011 as
            ``valid_from = trading_date 00:00Z``.
        close_usd: adjusted close, USD per share, as a ``Decimal`` because that
            is what the ``Numeric`` column yields and the loader has to convert
            it.
        knowledge_time: when this version became knowable (tz-aware UTC).
        is_retraction: whether this version retracts the fact.
    """

    security_id: int
    trading_date: dt.date
    close_usd: Decimal
    knowledge_time: dt.datetime
    is_retraction: bool = False


def bars_from_closes(
    security_id: int,
    closes: Sequence[float],
    *,
    last_trading_day: dt.date,
    knowledge_delay: dt.timedelta = DEFAULT_KNOWLEDGE_DELAY,
) -> tuple[FixtureBar, ...]:
    """Lay a sequence of closes onto the business days ending at ``last_trading_day``.

    Args:
        security_id: the security these bars belong to.
        closes: adjusted closes in USD per share, oldest first. The last element
            lands on ``last_trading_day``.
        last_trading_day: the newest trading day in the series (a weekday).
        knowledge_delay: how long after midnight opening a bar's trading day the
            bar becomes knowable. Defaults to
            :data:`DEFAULT_KNOWLEDGE_DELAY`.

    Returns:
        One bar per close, ascending by trading day.
    """
    days = business_days_ending(last_trading_day, len(closes))
    return tuple(
        FixtureBar(
            security_id=security_id,
            trading_date=day,
            close_usd=Decimal(str(close)),
            knowledge_time=midnight_utc(day) + knowledge_delay,
        )
        for day, close in zip(days, closes, strict=True)
    )


@dataclass(frozen=True, slots=True)
class FixturePriceStore:
    """Hand-written price bars, readable only through the as-of read contract.

    Attributes:
        bars: every version of every bar, in any order. Several versions of one
            (security, trading day) are the normal case — that is how a
            correction or a re-adjustment is represented.
    """

    bars: tuple[FixtureBar, ...] = field(default_factory=tuple)

    def with_bars(self, extra: Iterable[FixtureBar]) -> FixturePriceStore:
        """Return a new store with ``extra`` versions added.

        Returns a copy rather than mutating, so a test can hold the "before"
        store and the "after" store at once and compare what each produces —
        which is the shape every lookahead test in this package takes.

        Args:
            extra: additional bar versions.

        Returns:
            A new store containing this store's bars followed by ``extra``.
        """
        return replace(self, bars=(*self.bars, *extra))

    def visible_rows(self, as_of: dt.datetime) -> list[tuple[int, dt.datetime, Decimal]]:
        """Return the rows a session pinned at ``as_of`` would see.

        Implements the D-011 read contract: only versions with
        ``knowledge_time <= as_of`` (boundary inclusive), the greatest
        ``knowledge_time`` wins per (security, trading day), and a winning
        retraction hides the fact entirely.

        Args:
            as_of: the pinned instant (tz-aware UTC).

        Returns:
            ``(security_id, valid_from, close_usd)`` triples — the shape the
            loader's ``select()`` asks for — in **descending** key order, so a
            loader that assumed the query's ``ORDER BY`` had been applied would
            get the reverse of what it expected rather than accidentally the
            right thing.
        """
        winners: dict[tuple[int, dt.date], FixtureBar] = {}
        for bar in self.bars:
            if bar.knowledge_time > as_of:
                continue
            key = (bar.security_id, bar.trading_date)
            current = winners.get(key)
            if current is None or bar.knowledge_time > current.knowledge_time:
                winners[key] = bar
        return [
            (bar.security_id, midnight_utc(bar.trading_date), bar.close_usd)
            for _, bar in sorted(winners.items(), reverse=True)
            if not bar.is_retraction
        ]


class FixtureResult:
    """The slice of SQLAlchemy's ``Result`` the price loader uses."""

    def __init__(self, rows: list[tuple[int, dt.datetime, Decimal]]) -> None:
        """Hold the rows this result will hand back.

        Args:
            rows: ``(security_id, valid_from, close_usd)`` triples.
        """
        self._rows = rows

    def all(self) -> list[tuple[int, dt.datetime, Decimal]]:
        """Return every row, as ``Result.all()`` does."""
        return self._rows


class FixtureAsOfSession:
    """The slice of ``AsyncSession`` the price loader uses, pinned to an instant.

    Stands in for a session produced by :func:`backend.db.as_of`. It answers
    every statement with **all** rows visible at its pinned instant, ignoring the
    statement's filters and ordering — see this module's docstring for why that
    makes the tests stronger rather than weaker.

    Attributes:
        statements: every statement the code under test executed, so a test can
            assert what was asked for as well as what came back.
    """

    def __init__(self, store: FixturePriceStore, as_of: dt.datetime) -> None:
        """Pin a fixture store at an instant.

        Args:
            store: the hand-written bars.
            as_of: the instant this session is pinned at (tz-aware UTC).
        """
        self._store = store
        self._as_of = as_of
        self.statements: list[Any] = []

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> FixtureResult:  # noqa: ANN401 — mirrors AsyncSession.execute, which is itself untyped in its statement parameter
        """Record the statement and return every row visible at the pinned instant.

        Args:
            statement: the SQLAlchemy statement, recorded and otherwise ignored.
            *_args: ignored; present so the call shape matches ``AsyncSession``.
            **_kwargs: ignored; present so the call shape matches ``AsyncSession``.

        Returns:
            A result over the visible rows.
        """
        self.statements.append(statement)
        return FixtureResult(self._store.visible_rows(self._as_of))
