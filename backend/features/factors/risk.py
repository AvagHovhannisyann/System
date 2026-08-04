"""Risk-based factor: the low-volatility anomaly (P5.3).

Stocks with low realized volatility have historically earned higher
risk-adjusted returns than the CAPM allows, and often higher raw returns than
their high-volatility peers (Black, Jensen & Scholes 1972; Ang, Hodrick, Xing &
Zhang 2006; Frazzini & Pedersen 2014). The usual explanations are leverage
constraints and a preference for lottery-like payoffs rather than any
compensation for risk, which is why the effect survives in a factor library
built to rank stocks cross-sectionally.

--------------------------------------------------------------------------
Sign convention: the value carries the "low"
--------------------------------------------------------------------------

The feature is named ``low_volatility`` and its value is the **negative** of
trailing realized volatility, so a high score means a calm stock — the leg the
premium accrues to. The alternative (name the feature ``volatility`` and let the
model discover the sign) was rejected because the name would then be
directionless while the plan, the gate and the Features page all speak of "low
volatility", and because P5.4's premium check needs a stated expected sign per
factor rather than a convention held in someone's head. The package's rule is in
:mod:`backend.features.factors`: a directionally named factor carries its
direction in the value; a factor named after a measured quantity does not.

The cost is that the units are stated as a negative standard deviation, which
reads oddly. That is preferable to a silent sign: directive §8 makes units
mandatory precisely because a sign or scale error in this domain produces no
symptom.

--------------------------------------------------------------------------
Availability lag: zero
--------------------------------------------------------------------------

Same reasoning as :mod:`backend.features.factors.momentum`, which states it in
full. In short: a daily close is knowable at that close (16:00 ET, three to four
hours before midnight UTC opening the next date), the price connector's
``knowledge_time`` already carries any vendor delivery delay under D-011, and a
bar that becomes visible earlier than its own close is a store defect that
:func:`backend.features.factors._prices.load_adjusted_closes` raises on rather
than a margin this declaration should absorb.

This factor sits between the two momentum factors in exposure to that risk: its
window ends at the most recent print, like short-term reversal, but one extra
day at the end of a 252-day standard deviation moves the value by a negligible
amount. A wrong lag here would be a real lookahead all the same — volatility
clusters, so knowing today's move is knowing something about today's return —
which is why the guard is at the loader and covers every price factor equally.

**Blocked on B1.** The price connector (P3.4) does not exist, so ``price_bar``
is empty and this factor returns ``NaN`` for every security today.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**The value is a negative annualized standard deviation of daily log returns, a
dimensionless fraction.** ``-0.24`` means 24% annualized volatility. Not
percent, not basis points, not variance.

**Annualization multiplies by ``sqrt(252)``** — the square-root-of-time rule,
which assumes daily log returns are serially uncorrelated. They are not exactly:
short-horizon return autocorrelation is small but non-zero, so the annualized
number is a convention rather than a forecast. It is applied anyway because the
factor is used cross-sectionally, where a constant multiplier changes no ranking
and cancels in the z-score, and because a number labelled "annualized
volatility" is what every consumer expects to compare against a 20%-ish
intuition.

**The standard deviation is the sample one (``ddof=1``)** and is taken about the
sample mean rather than about zero. Over 252 observations the difference is
immaterial; it is stated because the two conventions differ by a factor of
``sqrt(n/(n-1))`` and silently mixing them across a codebase is the kind of
error this section exists to prevent.

**252 daily returns require 253 closes**, all within
:data:`~backend.features.factors.momentum.MOMENTUM_LOOKBACK_DAYS`-equivalent
calendar bounds (:data:`VOLATILITY_LOOKBACK_DAYS`). Fewer prints, or a last
print older than
:data:`~backend.features.factors._prices.MAX_PRICE_STALENESS`, gives ``NaN``.
Nothing is annualized up from a shorter window: a volatility estimated from 40
days and one estimated from 252 are different statistics, and quietly mixing
them across the cross-section would make the factor partly a measurement of data
coverage.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.features.factors._prices import (
    MAX_PRICE_STALENESS,
    PRICE_SOURCE_TABLE,
    load_adjusted_closes,
    log_price_changes,
)
from backend.features.registry import feature
from backend.features.spec import FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray
    from backend.features.factors._prices import PriceSeries

__all__ = [
    "LOW_VOLATILITY",
    "TRADING_DAYS_PER_YEAR",
    "VOLATILITY_LOOKBACK_DAYS",
    "VOLATILITY_RETURNS",
    "low_volatility",
    "low_volatility_value",
]

VOLATILITY_RETURNS: Final = 252
"""Daily log returns the volatility estimate is taken over (count).

One trading year. Requires 253 closes. Long enough that the sample standard
deviation is a usable estimate (its own relative standard error is ~4.5% at this
length) and short enough to track a regime change within months rather than
years.
"""

VOLATILITY_LOOKBACK_DAYS: Final = 400
"""Calendar days of history loaded for the volatility estimate (wall clock).

The 253 closes needed span ~367 calendar days at 252 trading days per year; 400
leaves roughly a month of slack for holidays and missed prints, and refuses a
name too thin to have printed 253 times in thirteen months.
"""

TRADING_DAYS_PER_YEAR: Final = 252
"""Trading days per year used to annualize (count).

Separate constant from :data:`VOLATILITY_RETURNS` although both are 252: one is
the estimation window and the other the time-scaling convention. They coincide
today and there is no reason a change to either should silently change the
other.
"""

LOW_VOLATILITY = FeatureSpec(
    name="low_volatility",
    definition=(
        "The low-volatility anomaly (Black/Jensen/Scholes 1972; Ang et al. 2006; "
        "Frazzini & Pedersen 2014), stated so that a HIGH score is a CALM stock: "
        "the NEGATIVE of the annualized sample standard deviation (ddof=1) of the "
        "252 daily log returns of the split- and dividend-adjusted close ending at "
        "the last print before the rebalance date, annualized by sqrt(252). "
        "Expected premium sign: POSITIVE in these units — low-volatility stocks "
        "have historically earned higher risk-adjusted, and often higher raw, "
        "returns than high-volatility stocks. NaN when the security has fewer than "
        "253 adjusted closes within 400 calendar days of the rebalance date, or "
        "has not printed within 10 calendar days of it."
    ),
    units=(
        "negative annualized standard deviation of daily log returns, dimensionless "
        "fraction (-0.24 = 24% annualized volatility; higher = calmer)"
    ),
    availability_lag=dt.timedelta(0),
    source_tables=frozenset({PRICE_SOURCE_TABLE}),
)
"""Declaration of ``low_volatility``. Lag zero — see the module docstring."""


def low_volatility_value(series: PriceSeries, *, compute_date: dt.date) -> float:
    """Compute the negated annualized realized volatility for one security.

    The arithmetic, separated from the query so it can be checked against a
    hand-worked example without a database::

        -std( ln(close[i+1] / close[i]) over the last 252 pairs, ddof=1 ) * sqrt(252)

    Args:
        series: the security's adjusted closes over the loaded window, ascending
            by trading day, every one strictly before ``compute_date``.
        compute_date: the rebalance date, used only for the staleness check.

    Returns:
        Negative annualized volatility (dimensionless fraction), so ``-0.24`` is
        a stock at 24% annualized. ``NaN`` when the estimate cannot be formed:
        fewer than 253 prints in the loaded window, a last print more than
        :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS` old, or any
        non-positive close inside the window (a corrupt print makes its returns
        ``NaN``, which propagates through the standard deviation — a deviation
        taken over a window containing one is not a measurement of anything).

    Example:
        >>> import datetime as dt
        >>> import numpy as np
        >>> from backend.features.factors._prices import PriceSeries
        >>> steps = np.where(np.arange(252) % 2 == 0, 1.01, 1 / 1.01)
        >>> closes = np.concatenate(([100.0], 100.0 * np.cumprod(steps)))
        >>> days = tuple(dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(253))
        >>> series = PriceSeries(1, days, closes)
        >>> round(low_volatility_value(series, compute_date=dt.date(2025, 9, 11)), 6)
        -0.158271
    """
    closes = series.adjusted_close
    required = VOLATILITY_RETURNS + 1
    if closes.size < required or series.is_stale(compute_date, tolerance=MAX_PRICE_STALENESS):
        return float("nan")
    # A corrupt print makes its two returns NaN (see log_price_changes), and NaN
    # propagates through np.std, so the whole estimate becomes NaN without an
    # explicit check. That is the wanted behaviour and it is stated rather than
    # re-implemented: a guard here could not fail, and a line that cannot fail
    # is a line a reader has to reason about for nothing.
    returns = log_price_changes(closes[-required:])
    deviation = float(np.std(returns, ddof=1))
    return -deviation * float(np.sqrt(TRADING_DAYS_PER_YEAR))


@feature(LOW_VOLATILITY)
async def low_volatility(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Compute ``low_volatility`` for the requested securities.

    Args:
        session: an ``AsyncSession`` already pinned by :func:`backend.db.as_of`
            to the instant this feature's declaration permits. Read through it;
            do not open another.
        request: the securities, the compute date, and the pinned instant.

    Returns:
        One ``float64`` per requested security, in ``request.security_ids``
        order, in :data:`LOW_VOLATILITY`'s units — negative annualized
        volatility, so a high value is a calm stock. ``NaN`` where the estimate
        could not be formed.

    Raises:
        PriceTemporalIntegrityError: if a visible price bar is dated on or after
            the compute date — the price connector's knowledge-time policy is
            optimistic and every value here would be prescient (I1).
        PriceSeriesError: if two visible bars share a security and a trading day.
    """
    history = await load_adjusted_closes(
        session,
        security_ids=request.security_ids,
        compute_date=request.compute_date,
        as_of=request.as_of,
        lookback_days=VOLATILITY_LOOKBACK_DAYS,
    )
    return np.array(
        [
            low_volatility_value(history[security_id], compute_date=request.compute_date)
            for security_id in request.security_ids
        ],
        dtype=np.float64,
    )
