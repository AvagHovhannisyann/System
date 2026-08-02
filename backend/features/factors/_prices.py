"""Point-in-time price history for the price-based baseline factors (P5.3).

Momentum, short-term reversal and low volatility are all functions of one
object: the sequence of adjusted closes a security printed *before* the
rebalance date. This module produces that sequence and nothing else, so the
three factors share one temporal contract rather than three near-identical
queries.

--------------------------------------------------------------------------
Where the temporal safety actually comes from
--------------------------------------------------------------------------

**The knowledge bound is the session's, not this module's.** The session handed
to a feature computation is already pinned by :func:`backend.db.as_of` to
``midnight_utc(compute_date) - availability_lag``; rows whose ``knowledge_time``
is later simply do not exist as far as this query is concerned (I1). This module
therefore issues **no** ``knowledge_time`` predicate of its own. Re-implementing
the bound here would either duplicate it (harmless but misleading about where
enforcement lives) or contradict it (a second, weaker filter that quietly wins).

**What this module adds is a check on the store's honesty.** A price bar for
trading day ``D`` cannot be knowable before ``D``'s close, which is hours after
``midnight_utc(D)``. So on a correctly-pinned session no visible bar may be
dated on or after ``compute_date``. If one is, the price connector has stamped
an optimistic ``knowledge_time`` — most plausibly by reusing ``valid_from``
(``D 00:00Z``, the *open* of the trading day) as the knowledge time — and every
factor built on it would be prescient by a day. That is not a data condition to
be smoothed over with a ``NaN``; it is a defect in the store, so
:func:`load_adjusted_closes` raises :class:`PriceTemporalIntegrityError` and the
run stops. **This check is live in production**, which is why the event-time
upper bound is deliberately *not* pushed into the SQL: a ``WHERE valid_from <
midnight(D)`` predicate would silently discard exactly the rows that prove the
connector is wrong.

**Blocked on B1 — the price connector does not exist yet.** P3.4 (daily OHLCV
via Sharadar SEP) is blocked on the data-vendor decision, so ``price_bar`` is an
empty table and its ``knowledge_time`` policy is not yet fixed in code. The
consequence for this module is bounded and stated rather than guessed: with no
rows, every security gets an empty series and every price factor returns ``NaN``
— "not available", which is the truth — and the integrity check above is what
will catch a knowledge-time policy that turns out to be optimistic when the
connector does land.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**Closes are ``close_usd``: USD per share, split- and dividend-adjusted** on the
basis known at the pinned instant. Adjusted rather than raw because a factor
built on raw closes reads a 2-for-1 split as a -50% return. Point-in-time
rather than latest because a re-adjustment arrives as a new version with a later
``knowledge_time`` (D-011), so a backtest dated before a split sees the
pre-split series — which is what the operator would have had.

*Assumption, and it belongs to the connector:* that a re-adjustment re-versions
the whole affected history, so every bar visible at one instant is on one
adjustment basis. A connector that re-versioned only the bars after a split
would leave a series with a discontinuity at the split, and this module cannot
detect that from the bars alone.

**Trading dates are UTC calendar dates** taken from ``valid_from``, which a
daily bar for trading day ``D`` stores as ``D 00:00Z`` (D-011). They are
exchange trading days, not a continuous calendar: gaps are weekends, holidays
and halts, and nothing here fills them.

**Windows are counted in bars, bounded in calendar days.** A factor asks for a
number of trading observations (252 daily returns) but the store is addressed by
date, so :func:`load_adjusted_closes` takes a calendar ``lookback_days`` bound
and the caller checks it received enough bars. The bound is what stops a thinly
traded name from reaching back four years to find 253 prints and calling the
result a twelve-month momentum.

**The calendar bound is applied twice on purpose** — once in SQL, so the
database does not ship a decade of history per security, and again in Python, so
the arithmetic does not depend on the database having applied it. The Python
pass is the one the tests exercise. Rows are likewise re-sorted here rather than
trusted to arrive in ``ORDER BY`` order, because an implicitly ordered result is
a correctness dependency on the query planner.

**Missing means missing.** A security with too few bars, or a stale last print,
gets ``NaN`` from the factor that asked. Nothing is interpolated, forward-filled
or averaged in: under invariant I3 a filled price is fabricated data, and
downstream it is indistinguishable from a measurement.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from sqlalchemy import select

from backend.db.models import PriceBar
from backend.features.errors import FeatureComputeError
from backend.features.spec import compute_instant

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FloatArray

__all__ = [
    "MAX_PRICE_STALENESS",
    "PRICE_SOURCE_TABLE",
    "PriceSeries",
    "PriceSeriesError",
    "PriceTemporalIntegrityError",
    "load_adjusted_closes",
    "log_price_changes",
]

PRICE_SOURCE_TABLE: Final = "price_bar"
"""Physical table the price factors read; the value they put in ``source_tables``."""

MAX_PRICE_STALENESS: Final = dt.timedelta(days=10)
"""Largest gap between a security's last print and the compute date (wall clock).

A US listing that has not traded for ten calendar days is halted, suspended or
already gone — nine consecutive market holidays do not exist. Beyond this gap a
price factor returns ``NaN`` rather than a number computed from prints that
predate whatever stopped the stock trading, because a stale momentum score looks
exactly like a live one to every consumer downstream.

Deliberately generous rather than tight: the cost of a gap too large is that a
suspended name keeps a stale score for a few extra days, while the cost of a gap
too small is a spurious ``NaN`` for every name over a long holiday weekend, and
the second is the error that would be blamed on the data rather than on this
constant.
"""


class PriceSeriesError(FeatureComputeError):
    """Raised when loaded price rows cannot form one series per security.

    The case this exists for is two visible bars for the same security and the
    same trading day. The as-of layer resolves versions before returning rows
    (latest ``knowledge_time`` wins per logical key and ``valid_from``), so a
    duplicate means either the version resolution did not happen or the bar's
    event time is not what D-011 says it is. Silently keeping one of the two
    would pick a price by result order, which is not a choice this module is
    entitled to make.
    """


class PriceTemporalIntegrityError(FeatureComputeError):
    """Raised when a visible price bar is dated on or after the compute date.

    A lookahead detector, not a validation nicety. The bar for trading day ``D``
    settles at ``D``'s close, hours after ``midnight_utc(D)``, so no honest
    ``knowledge_time`` makes it visible to a session pinned at or before that
    instant. Seeing one means the price connector's knowledge-time policy is
    optimistic — the likeliest way being to reuse ``valid_from`` (``D 00:00Z``,
    the open) as the knowledge time — and every factor computed from it would be
    a day prescient.

    Failing closed is the point: the alternative is a backtest that improves and
    never explains why (I1).

    Attributes:
        security_id: the security whose bar was too fresh.
        bar_date: the offending bar's trading day.
        compute_date: the date the feature was being computed for.
        as_of: the instant the session was pinned at (tz-aware UTC).
    """

    def __init__(
        self,
        *,
        security_id: int,
        bar_date: dt.date,
        compute_date: dt.date,
        as_of: dt.datetime,
    ) -> None:
        """Build the error from the offending bar and the pinned instant.

        Args:
            security_id: the security whose bar was visible too early.
            bar_date: the offending bar's trading day.
            compute_date: the date the feature was being computed for.
            as_of: the instant the session was pinned at (tz-aware UTC).
        """
        self.security_id = security_id
        self.bar_date = bar_date
        self.compute_date = compute_date
        self.as_of = as_of
        super().__init__(
            f"price bar for security {security_id} dated {bar_date.isoformat()} is "
            f"visible to a session pinned at {as_of.isoformat()}, while computing "
            f"for {compute_date.isoformat()}. That bar settles at its own close, "
            f"hours after midnight UTC opening its trading day, so it cannot have "
            f"been knowable at the pinned instant: the price connector has stamped "
            f"an optimistic knowledge_time (reusing valid_from, the trading day's "
            f"open, is the usual cause). Refusing to compute a factor from it — a "
            f"day of foresight does not fail any distribution check (I1)."
        )


@dataclass(frozen=True, slots=True)
class PriceSeries:
    """One security's adjusted closes over a bounded window before a compute date.

    Attributes:
        security_id: the security these prints belong to (dimensionless key).
        trading_dates: the trading days, **ascending and strictly increasing**,
            every one strictly before the compute date. Exchange trading days,
            so consecutive entries are not consecutive calendar days.
        adjusted_close: USD per share, split- and dividend-adjusted on the basis
            known at the pinned instant, aligned positionally with
            :attr:`trading_dates`. Never ``NaN``: an absent bar is an absent
            element, not a filled one.
    """

    security_id: int
    trading_dates: tuple[dt.date, ...]
    adjusted_close: FloatArray

    @property
    def bar_count(self) -> int:
        """Number of prints in the window (count)."""
        return len(self.trading_dates)

    def is_stale(self, compute_date: dt.date, *, tolerance: dt.timedelta) -> bool:
        """Return whether the last print is too old to describe a tradable name.

        Args:
            compute_date: the date the feature is being computed for.
            tolerance: largest acceptable gap between the last print and
                ``compute_date`` (wall clock). See :data:`MAX_PRICE_STALENESS`.

        Returns:
            ``True`` when there are no prints at all, or when the newest print
            predates ``compute_date`` by more than ``tolerance``.
        """
        if not self.trading_dates:
            return True
        return compute_date - self.trading_dates[-1] > tolerance


def log_price_changes(closes: FloatArray) -> FloatArray:
    """Return successive log price changes, in dimensionless log-return units.

    Log rather than simple returns because these are summed and standard-deviated
    downstream: log returns add across time, so a 252-bar volatility is the
    standard deviation of the quantity that actually accumulates, and a symmetric
    move up and down cancels exactly instead of leaving a residual.

    Args:
        closes: adjusted closes in USD per share, ascending in time, one per
            trading day. Length ``n`` produces ``n - 1`` returns; a length of 0
            or 1 produces an empty array.

    Returns:
        ``ln(close[i + 1] / close[i])`` for each consecutive pair, dimensionless.
        An element is ``NaN`` wherever either close is non-positive — a
        non-positive equity price is a corrupt print rather than a measurement,
        and ``NaN`` (not available) is the honest value for the return that would
        have spanned it. No warning is emitted and no value is imputed.

    Example:
        >>> import numpy as np
        >>> np.round(log_price_changes(np.array([100.0, 110.0, 99.0])), 6)
        array([ 0.09531, -0.10536])
    """
    if closes.size < 2:
        return np.zeros(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log(closes)
    usable = np.asarray(np.where(closes > 0.0, logged, np.nan), dtype=np.float64)
    return np.asarray(np.diff(usable), dtype=np.float64)


async def load_adjusted_closes(
    session: AsyncSession,
    *,
    security_ids: Sequence[int],
    compute_date: dt.date,
    as_of: dt.datetime,
    lookback_days: int,
) -> dict[int, PriceSeries]:
    """Load each security's adjusted closes over the window before ``compute_date``.

    Issues one query through the **already-pinned** session, so the set of rows
    it can see is fixed by the caller's availability lag before this function is
    entered (I1). What it adds on top is the honesty check described in the
    module docstring: any visible bar dated on or after ``compute_date`` raises.

    Args:
        session: an ``AsyncSession`` already scoped by :func:`backend.db.as_of`.
            Never opened here; a computation is given exactly one session and it
            is the only one that reflects what the feature may know.
        security_ids: securities to load, in any order. Duplicates are
            tolerated and collapse to one series each. An empty sequence returns
            an empty mapping without querying.
        compute_date: the rebalance date. Bars dated on or after it must not be
            visible; if one is, that is a temporal-integrity failure, not a
            filter condition.
        as_of: the instant the session is pinned at (tz-aware UTC). Used only in
            the error message, so a failure states which knowledge set produced
            it; this function does not filter on it.
        lookback_days: how far back to look, in **calendar** days before
            ``compute_date``. Bounds the query and, applied again in Python,
            bounds the returned series.

    Returns:
        One :class:`PriceSeries` per requested security, keyed by
        ``security_id``. A security with no visible bars in the window gets an
        empty series rather than being omitted, so callers index the mapping
        without a membership test and get ``NaN`` from the arithmetic instead of
        a ``KeyError``.

    Raises:
        FeatureComputeError: if ``lookback_days`` is not positive, or a bar's
            ``valid_from`` is not timezone-aware (a naive event time cannot be
            placed on the UTC calendar the compute date lives on).
        PriceTemporalIntegrityError: if a visible bar is dated on or after
            ``compute_date``.
        PriceSeriesError: if two visible bars share a security and a trading day.
    """
    if lookback_days <= 0:
        msg = (
            f"lookback_days must be positive; got {lookback_days}. A window with "
            f"no width cannot contain the trading days a price factor needs."
        )
        raise FeatureComputeError(msg)
    requested = tuple(dict.fromkeys(security_ids))
    if not requested:
        return {}

    window_end = compute_instant(compute_date)
    window_start = window_end - dt.timedelta(days=lookback_days)
    statement = (
        select(PriceBar.security_id, PriceBar.valid_from, PriceBar.close_usd)
        .where(PriceBar.security_id.in_(requested))
        .where(PriceBar.valid_from >= window_start)
        .order_by(PriceBar.security_id, PriceBar.valid_from)
    )
    rows = (await session.execute(statement)).all()

    wanted = set(requested)
    collected: dict[int, dict[dt.date, float]] = {security_id: {} for security_id in requested}
    for security_id, valid_from, close_usd in rows:
        if security_id not in wanted:
            continue
        bar_date = _trading_date(valid_from, security_id=security_id)
        if bar_date >= compute_date:
            raise PriceTemporalIntegrityError(
                security_id=security_id,
                bar_date=bar_date,
                compute_date=compute_date,
                as_of=as_of,
            )
        if valid_from < window_start:
            continue
        series = collected[security_id]
        if bar_date in series:
            msg = (
                f"two visible price bars for security {security_id} on trading day "
                f"{bar_date.isoformat()}. The as-of layer resolves versions before "
                f"returning rows (latest knowledge_time wins per key and valid_from), "
                f"so a duplicate means version resolution did not happen. Refusing to "
                f"pick one by result order."
            )
            raise PriceSeriesError(msg)
        series[bar_date] = float(close_usd)

    return {
        security_id: _to_series(security_id, collected[security_id]) for security_id in requested
    }


def _trading_date(valid_from: dt.datetime, *, security_id: int) -> dt.date:
    """Return the UTC calendar date a bar's event-time start denotes.

    D-011 stores a daily bar for trading day ``D`` with ``valid_from = D
    00:00Z``, so the trading day is the UTC date of that instant.

    Args:
        valid_from: the bar's event-time interval start.
        security_id: the owning security, for the error message.

    Returns:
        The trading day as a UTC calendar date.

    Raises:
        FeatureComputeError: if ``valid_from`` is naive. A naive event time
            cannot be placed on the UTC calendar that ``compute_date`` lives on,
            and guessing a zone would put bars on the wrong side of a date
            boundary — which is exactly the one-day error this module exists to
            refuse.
    """
    if valid_from.tzinfo is None or valid_from.utcoffset() is None:
        msg = (
            f"price bar for security {security_id} has a naive valid_from "
            f"({valid_from!r}); event times are timezone-aware UTC (D-011). A naive "
            f"instant cannot be placed on the UTC calendar the compute date uses."
        )
        raise FeatureComputeError(msg)
    return valid_from.astimezone(dt.UTC).date()


def _to_series(security_id: int, closes_by_date: dict[dt.date, float]) -> PriceSeries:
    """Build an ascending :class:`PriceSeries` from a date-keyed price mapping.

    Sorts here rather than relying on the query's ``ORDER BY``: every window in
    this package is positional (``the 253rd close back``), so an ordering
    assumption that the database happened to satisfy would be a silent
    correctness dependency on the planner.

    Args:
        security_id: the owning security.
        closes_by_date: adjusted close (USD per share) per trading day.

    Returns:
        The series, ascending by trading day.
    """
    ordered = sorted(closes_by_date.items())
    return PriceSeries(
        security_id=security_id,
        trading_dates=tuple(day for day, _ in ordered),
        adjusted_close=np.array([close for _, close in ordered], dtype=np.float64),
    )
