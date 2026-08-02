"""Liquidity factor: Amihud illiquidity, price impact per dollar traded (P5.3).

Amihud (2002) measures illiquidity as the average daily ratio of absolute return
to dollar volume — how far the price moves for each dollar that changes hands.
It is a price-impact estimate built entirely from daily bars, which is its whole
appeal: the theoretically better measures (effective spread, Kyle's lambda,
order-flow imbalance) need intraday or quote data this platform has no intention
of buying, and Amihud's own paper and the replication literature since show the
daily proxy tracks them closely enough to price the liquidity premium.

The premium is the point. Illiquid stocks have historically earned higher
returns as compensation for the cost of getting in and out of them, so the factor
is expected to earn a **positive** premium in the units below. It is also the
factor most likely to be *unrealisable*: the names it ranks highest are the ones
the P11 cost model will charge the most to trade. That tension is not resolved
here — it is exactly what the Phase 10/11 net-of-cost evaluation exists to
measure — but it is why this column is worth carrying even if it never earns a
tradable premium: a strategy whose alpha is concentrated in the illiquid tail
needs to be able to see that about itself.

--------------------------------------------------------------------------
Availability lag: zero, for the reasons D-027 gives
--------------------------------------------------------------------------

``price_bar`` is the only source, so this factor sits in D-027's price regime and
declares the same zero lag as momentum, short-term reversal and low volatility.
:mod:`backend.features.factors.momentum` argues it in full. In short: a daily bar
is knowable at that day's close (16:00 ET, three to four hours before midnight
UTC opening the next date); any vendor delivery delay is already carried by the
connector's ``knowledge_time`` under D-011 and enforced by the as-of session, so
a margin here would discard bars that genuinely were available; and the residual
risk — a connector that stamps a bar knowable before its own close — is
**detected rather than absorbed**, by the same
:class:`~backend.features.factors._prices.PriceTemporalIntegrityError` guard,
re-applied in :func:`load_dollar_volume_bars`.

The guard matters slightly more here than for a 252-bar volatility. Volume is far
more autocorrelated and far more skewed than return: a single extra day at the
end of the window can move an Amihud estimate materially if that day was thin,
because the statistic averages ``1/volume`` and that function is convex. So the
"one extra day changes little" intuition that holds for a standard deviation does
**not** hold for this factor, and it is stated rather than assumed.

**Blocked on B1.** The price connector (P3.4, Sharadar SEP) does not exist, so
``price_bar`` is empty and this factor returns ``NaN`` for every security today.
``NaN`` is honest here in a way it would not be for a factor with no table at all
(see :mod:`backend.features.factors._fundamentals`): the query runs, the store
answers, and the answer is "no bars".

--------------------------------------------------------------------------
Units — the awkward part, stated precisely
--------------------------------------------------------------------------

**The value is in units of 1/USD: a dimensionless log return per USD of daily
dollar volume, averaged over 252 trading bars.** A value of ``2.5e-10`` says that
over the estimation window, one dollar of trading moved the price by
``2.5e-10`` of itself on average — equivalently, a one-basis-point move
accompanied about ``4e5`` dollars of volume.

**It is not scaled by ``1e6``**, as Amihud's original paper and most published
tables are ("ILLIQ times 1e6"). That scaling is a presentation convention, and a
factor that carried a hidden multiplier inside a column labelled as a measurement
is precisely the silent units error directive §8 is written about. The raw
numbers are small — a liquid mega-cap lands around ``1e-12`` to ``1e-11``, a
thin micro-cap around ``1e-7`` — and that is fine: float64 represents them with
full relative precision (its smallest normal is ~``2.2e-308``), the P5.2
pipeline z-scores the cross-section so the scale cancels, and a reader comparing
this column against a published table has one documented conversion to apply
rather than an undocumented one to discover.

**The numerator is the absolute daily log return of the *adjusted* close.** Log
for the reason :func:`~backend.features.factors._prices.log_price_changes` gives,
adjusted so a 2-for-1 split is not read as a 50% price impact.

**The denominator is USD of dollar volume: the *unadjusted* close times the
*unadjusted* share volume.** This is the pairing that yields real traded dollars,
and it is the same rule :mod:`backend.features.factors.value` applies to market
capitalization: ``close_raw_usd`` and ``volume_shares`` are both quoted on the
unadjusted basis, and mixing an adjusted price with an unadjusted count misstates
the product by the adjustment factor. Using ``close_usd`` here would leave a
factor that looked entirely healthy and was wrong by a per-security constant that
drifts every time a corporate action occurs.

The numerator and denominator are deliberately on *different* adjustment bases,
which reads like an error and is not: a return must be adjustment-consistent
across two days, while a dollar traded is a dollar traded on the day it traded.

--------------------------------------------------------------------------
Window and availability
--------------------------------------------------------------------------

**252 daily observations — one trading year — requiring 253 closes**, every one
inside :data:`AMIHUD_LOOKBACK_DAYS` calendar days of the compute date. This is
Amihud's own annual estimation window, and the length is doing real work rather
than following convention: ``|r| / dollar_volume`` is extremely heavy-tailed
because the denominator can approach zero, so the sample mean of a short window
is dominated by its thinnest day or two. A 21-bar version of this factor is
largely a measurement of whether the window happened to contain a half-session
before a holiday.

**Any non-positive dollar volume inside the window makes the estimate ``NaN``.**
A zero-volume day has no price impact to measure — the ratio is ``0/0`` or a
division by zero — and dropping such days instead would be a silent selection on
the dependent variable: it removes exactly the least liquid observations and
biases the security's score toward looking *more* liquid than it was. The whole
estimate is voided instead, and the security is reported as not available.
Negative volume or a non-positive unadjusted close is corrupt data and is treated
identically.

**A stale name is ``NaN``** rather than carrying a year-old liquidity score, on
the same :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS` rule the
other price factors use.

--------------------------------------------------------------------------
Why this module loads its own bars
--------------------------------------------------------------------------

:func:`~backend.features.factors._prices.load_adjusted_closes` selects three
columns and this factor needs five: it is the only factor in the package that
reads ``volume_shares`` and ``close_raw_usd``. So :func:`load_dollar_volume_bars`
issues its own query, and reuses everything that is genuinely shared — the error
types, the source-table constant, the staleness tolerance, the log-return
helper, and :class:`~backend.features.factors._prices.PriceSeries` itself, which
:class:`DollarVolumeSeries` wraps rather than re-declares, so the invariants
documented there hold here verbatim. What is duplicated is the query shape and
the row loop, which differ because the columns differ; the temporal contract they
enforce is written once, in :mod:`backend.features.factors._prices`, and is
imported.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from sqlalchemy import select

from backend.db.models import PriceBar
from backend.features.errors import FeatureComputeError
from backend.features.factors._prices import (
    MAX_PRICE_STALENESS,
    PRICE_SOURCE_TABLE,
    PriceSeries,
    PriceSeriesError,
    PriceTemporalIntegrityError,
    log_price_changes,
)
from backend.features.registry import feature
from backend.features.spec import FeatureSpec, compute_instant

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray

__all__ = [
    "AMIHUD_ILLIQUIDITY",
    "AMIHUD_LOOKBACK_DAYS",
    "AMIHUD_OBSERVATIONS",
    "DollarVolumeSeries",
    "amihud_illiquidity",
    "amihud_illiquidity_value",
    "load_dollar_volume_bars",
]

AMIHUD_OBSERVATIONS: Final = 252
"""Daily ``|return| / dollar volume`` observations the estimate averages (count).

One trading year, Amihud's own estimation window, and it requires 253 closes: an
observation needs both the day's dollar volume and the return into that day.

The length is load-bearing rather than conventional. The ratio's denominator can
approach zero, so its distribution is far heavier-tailed than a return's and the
sample mean of a short window is dominated by its two thinnest days. Over 252
observations one anomalous session contributes at most 1/252 of the answer.
"""

AMIHUD_LOOKBACK_DAYS: Final = 400
"""Calendar days of history loaded for the illiquidity estimate (wall clock).

The 253 closes needed span ~367 calendar days at 252 trading days per year; 400
leaves roughly a month of slack for holidays and missed prints, and refuses a
name too thin to have printed 253 times in thirteen months. Matches
:data:`~backend.features.factors.risk.VOLATILITY_LOOKBACK_DAYS`, which bounds the
same number of closes for the same reason.
"""

AMIHUD_ILLIQUIDITY = FeatureSpec(
    name="amihud_illiquidity",
    definition=(
        "Amihud (2002) illiquidity: the mean over the 252 trading bars ending at "
        "the last print before the rebalance date of |daily log return of the "
        "split- and dividend-adjusted close| divided by that day's dollar volume "
        "in USD, where dollar volume is the UNADJUSTED close times the UNADJUSTED "
        "share volume — the pairing that yields real traded dollars, since a share "
        "count and an adjusted price are on different bases. The return leg is "
        "adjusted so a corporate action is not read as price impact; the two legs "
        "are deliberately on different adjustment bases. Reported RAW, in units of "
        "1/USD, and NOT scaled by 1e6 as published tables usually are: a hidden "
        "multiplier inside a column labelled as a measurement is the silent units "
        "error. Expected premium sign: POSITIVE — illiquid stocks have "
        "historically earned a liquidity premium, so the factor is named after the "
        "quantity it measures and is NOT negated. NaN when the security has fewer "
        "than 253 adjusted closes within 400 calendar days of the rebalance date, "
        "has not printed within 10 calendar days of it, or has any non-positive "
        "dollar volume or close inside the window — dropping thin days instead "
        "would select on the least liquid observations and understate illiquidity."
    ),
    units=(
        "dimensionless log return per USD of daily dollar volume, averaged over 252 "
        "trading bars; units of 1/USD (2.5e-10 = one dollar of trading moved the "
        "price by 2.5e-10 of itself, on average). Raw, not scaled by 1e6."
    ),
    availability_lag=dt.timedelta(0),
    source_tables=frozenset({PRICE_SOURCE_TABLE}),
)
"""Declaration of ``amihud_illiquidity``. Lag zero — see the module docstring.

Like ``short_term_reversal`` and unlike ``momentum_12_1``, this factor's freshest
input is the most recent bar, so it is one of the two whose value would move
first if the price connector's knowledge-time policy (P3.4, blocked on B1) turned
out to be optimistic — and it is the more sensitive of the two, because it
averages ``1/volume`` rather than a return. The loader's temporal-integrity check
is what refuses that case outright.
"""


@dataclass(frozen=True, slots=True)
class DollarVolumeSeries:
    """One security's closes and traded dollars over a window before a compute date.

    Wraps a :class:`~backend.features.factors._prices.PriceSeries` rather than
    re-declaring its fields, so everything that module documents about trading
    dates, ordering and adjustment basis holds here without restatement, and the
    staleness rule is the same code rather than the same idea written twice.

    Attributes:
        prices: the adjusted closes and their trading days, ascending, every one
            strictly before the compute date.
        dollar_volume: USD traded per day — the unadjusted close times the
            unadjusted share volume — aligned positionally with
            ``prices.trading_dates``. Non-positive entries are possible and mean
            "no trading, or a corrupt bar"; they are not filtered here, because
            the decision about what a zero-volume day does to an estimate belongs
            to the factor and not to the container.
    """

    prices: PriceSeries
    dollar_volume: FloatArray

    def __post_init__(self) -> None:
        """Check that the two aligned sequences are in fact aligned.

        A length mismatch would silently pair each return with the wrong day's
        volume — every value plausible, every value wrong — which is the one
        defect in this factor that no distribution check would reveal.

        Raises:
            FeatureComputeError: if ``dollar_volume`` has a different length from
                ``prices.trading_dates``.
        """
        if self.dollar_volume.shape != (len(self.prices.trading_dates),):
            msg = (
                f"security {self.prices.security_id} has "
                f"{len(self.prices.trading_dates)} trading day(s) but "
                f"{self.dollar_volume.size} dollar-volume entr(y/ies) of shape "
                f"{self.dollar_volume.shape}. The two are paired positionally, so a "
                f"mismatch would divide each return by another day's volume."
            )
            raise FeatureComputeError(msg)

    @property
    def bar_count(self) -> int:
        """Number of prints in the window (count)."""
        return self.prices.bar_count


def amihud_illiquidity_value(series: DollarVolumeSeries, *, compute_date: dt.date) -> float:
    """Compute Amihud illiquidity for one security.

    The arithmetic, separated from the query so it can be checked against a
    hand-worked example without a database::

        mean over the last 252 bars of ( |ln(close[t] / close[t-1])| / dollar_volume[t] )

    The ``t`` alignment is the part worth stating: each observation pairs the
    return *into* a day with that same day's traded dollars, so 252 observations
    consume 253 closes and the oldest close contributes only as a denominator of
    the first return.

    Args:
        series: the security's adjusted closes and daily dollar volumes over the
            loaded window, ascending by trading day, every one strictly before
            ``compute_date``.
        compute_date: the rebalance date, used only for the staleness check.

    Returns:
        Mean price impact in units of 1/USD — a dimensionless log return per USD
        of dollar volume. ``NaN`` when the estimate cannot be formed: fewer than
        253 prints in the loaded window, a last print more than
        :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS` old, a
        non-positive close anywhere in the window (which makes its returns
        ``NaN``), or a non-positive dollar volume on any day of the window.
        Never ``±inf``: a zero denominator becomes ``NaN`` before the division
        rather than an infinity afterwards, because the P5.2 pipeline refuses
        infinities outright and one bad tick would abort a whole cross-section.

    Example:
        >>> import datetime as dt
        >>> import numpy as np
        >>> from backend.features.factors._prices import PriceSeries
        >>> closes = np.where(np.arange(253) % 2 == 0, 100.0, 101.0)
        >>> days = tuple(dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(253))
        >>> series = DollarVolumeSeries(
        ...     PriceSeries(1, days, closes), np.full(253, 101_000_000.0)
        ... )
        >>> value = amihud_illiquidity_value(series, compute_date=dt.date(2025, 9, 11))
        >>> round(value * 1e9, 6)
        0.098518
    """
    prices = series.prices
    closes = prices.adjusted_close
    required = AMIHUD_OBSERVATIONS + 1
    if closes.size < required or prices.is_stale(compute_date, tolerance=MAX_PRICE_STALENESS):
        return float("nan")
    returns = log_price_changes(closes[-required:])
    traded = series.dollar_volume[-AMIHUD_OBSERVATIONS:]
    # A non-positive denominator is turned into NaN *before* the division rather
    # than being caught after it: `x / 0.0` is `inf`, and an infinity here would
    # propagate into a cross-section the transform pipeline rejects wholesale
    # (backend.features._stats.reject_infinities), turning one untraded day for
    # one security into a failure for every security on that date.
    usable = np.asarray(np.where(traded > 0.0, traded, np.nan), dtype=np.float64)
    impact = np.asarray(np.abs(returns) / usable, dtype=np.float64)
    return float(np.mean(impact))


async def load_dollar_volume_bars(
    session: AsyncSession,
    *,
    security_ids: Sequence[int],
    compute_date: dt.date,
    as_of: dt.datetime,
    lookback_days: int,
) -> dict[int, DollarVolumeSeries]:
    """Load each security's closes and daily traded dollars before ``compute_date``.

    The volume-carrying sibling of
    :func:`~backend.features.factors._prices.load_adjusted_closes`, with the
    identical temporal contract: one query through the **already-pinned**
    session, no ``knowledge_time`` predicate of its own (the bound is the as-of
    layer's and restating it here would either duplicate or contradict it), the
    calendar window re-applied in Python, rows re-sorted rather than trusted to
    arrive ordered, and any visible bar dated on or after ``compute_date`` treated
    as a store defect rather than as something to filter away.

    Args:
        session: an ``AsyncSession`` already scoped by :func:`backend.db.as_of`.
            Never opened here; a computation is given exactly one session and it
            is the only one that reflects what the feature may know.
        security_ids: securities to load, in any order. Duplicates are tolerated
            and collapse to one series each. An empty sequence returns an empty
            mapping without querying.
        compute_date: the rebalance date. Bars dated on or after it must not be
            visible; if one is, that is a temporal-integrity failure, not a
            filter condition.
        as_of: the instant the session is pinned at (tz-aware UTC). Used only in
            the error message, so a failure states which knowledge set produced
            it; this function does not filter on it.
        lookback_days: how far back to look, in **calendar** days before
            ``compute_date``.

    Returns:
        One :class:`DollarVolumeSeries` per requested security, keyed by
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
        select(
            PriceBar.security_id,
            PriceBar.valid_from,
            PriceBar.close_usd,
            PriceBar.close_raw_usd,
            PriceBar.volume_shares,
        )
        .where(PriceBar.security_id.in_(requested))
        .where(PriceBar.valid_from >= window_start)
        .order_by(PriceBar.security_id, PriceBar.valid_from)
    )
    rows = (await session.execute(statement)).all()

    wanted = set(requested)
    collected: dict[int, dict[dt.date, tuple[float, float]]] = {
        security_id: {} for security_id in requested
    }
    for security_id, valid_from, close_usd, close_raw_usd, volume_shares in rows:
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
        series[bar_date] = (float(close_usd), float(close_raw_usd) * float(volume_shares))

    return {
        security_id: _to_series(security_id, collected[security_id]) for security_id in requested
    }


def _trading_date(valid_from: dt.datetime, *, security_id: int) -> dt.date:
    """Return the UTC calendar date a bar's event-time start denotes.

    Mirrors the identically-named helper in
    :mod:`backend.features.factors._prices`, which is module-private there and so
    is not imported across modules. Six lines of duplication is the cheaper of
    the two costs; the alternative is widening that module's public surface for
    one caller.

    Args:
        valid_from: the bar's event-time interval start. D-011 stores a daily bar
            for trading day ``D`` with ``valid_from = D 00:00Z``.
        security_id: the owning security, for the error message.

    Returns:
        The trading day as a UTC calendar date.

    Raises:
        FeatureComputeError: if ``valid_from`` is naive. A naive event time
            cannot be placed on the UTC calendar that ``compute_date`` lives on,
            and guessing a zone would put bars on the wrong side of a date
            boundary — the one-day error the integrity check exists to refuse.
    """
    if valid_from.tzinfo is None or valid_from.utcoffset() is None:
        msg = (
            f"price bar for security {security_id} has a naive valid_from "
            f"({valid_from!r}); event times are timezone-aware UTC (D-011). A naive "
            f"instant cannot be placed on the UTC calendar the compute date uses."
        )
        raise FeatureComputeError(msg)
    return valid_from.astimezone(dt.UTC).date()


def _to_series(
    security_id: int, bars_by_date: dict[dt.date, tuple[float, float]]
) -> DollarVolumeSeries:
    """Build an ascending :class:`DollarVolumeSeries` from a date-keyed mapping.

    Sorts here rather than relying on the query's ``ORDER BY``: the window is
    positional (``the 253rd close back``), so an ordering assumption the database
    happened to satisfy would be a silent correctness dependency on the planner.

    Args:
        security_id: the owning security.
        bars_by_date: ``(adjusted close in USD per share, dollar volume in USD)``
            per trading day.

    Returns:
        The series, ascending by trading day.
    """
    ordered = sorted(bars_by_date.items())
    return DollarVolumeSeries(
        prices=PriceSeries(
            security_id=security_id,
            trading_dates=tuple(day for day, _ in ordered),
            adjusted_close=np.array([close for _, (close, _) in ordered], dtype=np.float64),
        ),
        dollar_volume=np.array([traded for _, (_, traded) in ordered], dtype=np.float64),
    )


@feature(AMIHUD_ILLIQUIDITY)
async def amihud_illiquidity(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Compute ``amihud_illiquidity`` for the requested securities.

    Args:
        session: an ``AsyncSession`` already pinned by :func:`backend.db.as_of`
            to the instant this feature's declaration permits. Read through it;
            do not open another.
        request: the securities, the compute date, and the pinned instant.

    Returns:
        One ``float64`` per requested security, in ``request.security_ids``
        order, in :data:`AMIHUD_ILLIQUIDITY`'s units — mean price impact in
        1/USD, so a high value is an illiquid name. ``NaN`` where the estimate
        could not be formed.

    Raises:
        PriceTemporalIntegrityError: if a visible price bar is dated on or after
            the compute date — the price connector's knowledge-time policy is
            optimistic and every value here would be prescient (I1).
        PriceSeriesError: if two visible bars share a security and a trading day.
    """
    history = await load_dollar_volume_bars(
        session,
        security_ids=request.security_ids,
        compute_date=request.compute_date,
        as_of=request.as_of,
        lookback_days=AMIHUD_LOOKBACK_DAYS,
    )
    return np.array(
        [
            amihud_illiquidity_value(history[security_id], compute_date=request.compute_date)
            for security_id in request.security_ids
        ],
        dtype=np.float64,
    )
