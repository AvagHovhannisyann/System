"""Price-trend factors: twelve-month momentum and one-month reversal (P5.3).

Two factors built from the same input — the adjusted closes a security printed
before the rebalance date — pointing in opposite directions, which is why they
belong in one module where the boundary between them is visible.

**Momentum 12-1** (Jegadeesh & Titman 1993) is the return over the twelve months
ending *one month* before the compute date. The skipped month is not a detail:
without it the factor is contaminated by the one-month reversal below, and the
premium measured on the contaminated version is the two effects netting against
each other rather than momentum.

**Short-term reversal** (Jegadeesh 1990, Lehmann 1990) is the *negative* of the
most recent month's return. Last month's losers tend to outperform last month's
winners over the following month — microstructure and liquidity provision, not
a fundamental view — so the factor carries the reversal sign already applied and
a high score means a recent loser. That is the sign convention this package
uses for a directionally named factor; see :mod:`backend.features.factors`.

--------------------------------------------------------------------------
Availability lag: zero, and why that is the correct number, not a shortcut
--------------------------------------------------------------------------

Both factors declare ``availability_lag = 0``, meaning they read the store as of
``midnight UTC opening the compute date``. The justification has three parts,
and the third is the one that makes zero safe rather than merely convenient.

**1. A daily close is knowable at that close.** The US equity close is 16:00
ET, which is 20:00-21:00 UTC on the same calendar day — three to four hours
*before* midnight UTC opening the next date. So every bar dated on or before
``compute_date - 1`` was genuinely knowable at the instant a zero-lag feature
reads. This is the sense in which a price factor differs from a fundamental
one: nothing has to be filed, accepted or published for a close to exist.

**2. Adding a margin would double-count a delay the store already carries.**
Under D-011 the price connector stamps each bar's ``knowledge_time`` under its
own declared policy, and the as-of session filters on it. If the vendor's daily
file lands after midnight UTC, an honest connector stamps a later
``knowledge_time``, the bar is invisible to this feature, and the window simply
ends a day earlier — automatically, in the conservative direction, without any
constant here. A non-zero lag on top would discard bars that *were* available,
costing signal to protect against something already handled.

**3. The failure mode a lag could not fix is checked directly.** The risk worth
worrying about is a connector that stamps a bar's ``knowledge_time`` optimistically
— reusing ``valid_from`` (``D 00:00Z``, the trading day's *open*) is the obvious
way — which would let the bar for ``compute_date`` itself become visible. A lag
of one day would hide that defect while leaving it in the store for every other
consumer. Instead :func:`backend.features.factors._prices.load_adjusted_closes`
raises :class:`~backend.features.factors._prices.PriceTemporalIntegrityError` on
any visible bar dated on or after the compute date, so the defect is surfaced
rather than absorbed.

**Blocked on B1, and stated rather than guessed.** The price connector (P3.4,
Sharadar SEP) does not exist, so its knowledge-time policy is not yet fixed in
code and ``price_bar`` is empty. Today both factors return ``NaN`` for every
security — no bars, no value, which is the truth. The declarations above are
what P3.4 must satisfy; if it lands with a policy that makes a bar knowable
before its own close, part 3 turns that into a loud failure on the first
computation rather than a silent day of foresight.

Momentum 12-1 is additionally insulated by its own definition: its freshest
input is a close 21 trading days old, so no plausible vendor delivery delay can
reach it. Short-term reversal is not — it reads the most recent print — and is
therefore the factor whose value would move first if the connector's policy were
wrong. That asymmetry is the reason the guard exists at the loader rather than
in one factor.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**Both factors are dimensionless log returns (fractions, not percent, not basis
points).** ``0.15`` means a 15% cumulative move. Log rather than simple returns
because the pipeline winsorizes and z-scores these values: a log return is
symmetric under inversion (a doubling and a halving are ``+0.693`` and
``-0.693``), so a 1/99 winsorization treats winners and losers alike, where
simple returns are bounded below by ``-1`` and unbounded above and would clip
the two tails at different economic magnitudes. Cross-sectional *ranks* are
identical either way — the log is monotone in the price ratio — so nothing is
lost.

**Windows are counted in trading bars, bounded in calendar days.** One month is
21 trading bars and one year is 252; the calendar bound
(:data:`MOMENTUM_LOOKBACK_DAYS`) is what stops a thinly traded name from
reaching back years to collect 253 prints and calling the result a twelve-month
return. A security without enough prints inside the bound gets ``NaN``.

**Closes are adjusted for splits and dividends** on the basis known at the
pinned instant, so a corporate action is not read as a return and a
re-adjustment published after the compute date is not applied retroactively.

**Staleness is ``NaN``, not a stale number.** A security whose last print
predates the compute date by more than
:data:`~backend.features.factors._prices.MAX_PRICE_STALENESS` gets ``NaN``: a
halted name's momentum score is a measurement of a stock that is not trading.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.features.factors._prices import (
    MAX_PRICE_STALENESS,
    PRICE_SOURCE_TABLE,
    load_adjusted_closes,
)
from backend.features.registry import feature
from backend.features.spec import FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray
    from backend.features.factors._prices import PriceSeries

__all__ = [
    "MOMENTUM_12_1",
    "MOMENTUM_FORMATION_BARS",
    "MOMENTUM_LOOKBACK_DAYS",
    "MOMENTUM_SKIP_BARS",
    "REVERSAL_BARS",
    "REVERSAL_LOOKBACK_DAYS",
    "SHORT_TERM_REVERSAL",
    "momentum_12_1",
    "momentum_12_1_value",
    "short_term_reversal",
    "short_term_reversal_value",
]

MOMENTUM_SKIP_BARS: Final = 21
"""Trading bars skipped between the formation window and the compute date (count).

One month at ~21 trading days. Skipping it is what separates this factor from
:data:`SHORT_TERM_REVERSAL`: the most recent month's return reverses over the
following month, so including it nets the two effects and understates both.
"""

MOMENTUM_FORMATION_BARS: Final = 231
"""Trading bars the formation return spans (count): 252 - 21, eleven months.

The "12-1" convention measures from twelve months back to one month back, so the
span is a year minus the skipped month, not a full year.
"""

MOMENTUM_LOOKBACK_DAYS: Final = 400
"""Calendar days of history loaded for momentum (wall clock).

The 253 closes the factor needs span ~367 calendar days at 252 trading days per
year. 400 leaves roughly a month of slack for holidays and a handful of missed
prints, and refuses a name that needs to reach back further — one that thin has
not had a twelve-month return to measure.
"""

REVERSAL_BARS: Final = 21
"""Trading bars the reversal return spans (count): one month.

Matched deliberately to :data:`MOMENTUM_SKIP_BARS` — the window momentum skips
is exactly the window reversal measures, so the two factors partition the last
twelve months rather than overlapping.
"""

REVERSAL_LOOKBACK_DAYS: Final = 60
"""Calendar days of history loaded for short-term reversal (wall clock).

The 22 closes needed span ~31 calendar days; 60 leaves room for holidays and a
few missed prints without admitting a name that last traded two months ago.
"""

MOMENTUM_12_1 = FeatureSpec(
    name="momentum_12_1",
    definition=(
        "Twelve-month price momentum skipping the most recent month "
        "(Jegadeesh & Titman 1993): the log return of the split- and "
        "dividend-adjusted close from 252 trading bars before the rebalance date "
        "to 21 trading bars before it, so the formation window spans 231 bars "
        "(~11 months) and the most recent month is excluded. The skipped month is "
        "the short_term_reversal window; including it would net momentum against "
        "reversal and understate both. Expected premium sign: POSITIVE — past "
        "winners have historically continued to outperform over the following "
        "month. NaN when the security has fewer than 253 adjusted closes within "
        "400 calendar days of the rebalance date, or has not printed within 10 "
        "calendar days of it (a halted name has no live momentum)."
    ),
    units="log return over the 231-bar formation window, dimensionless fraction",
    availability_lag=dt.timedelta(0),
    source_tables=frozenset({PRICE_SOURCE_TABLE}),
)
"""Declaration of ``momentum_12_1``. Lag zero — see the module docstring.

The freshest input is a close 21 trading bars old, so this factor is the least
exposed of the three price factors to any vendor delivery delay: even a
multi-day delay in the price feed cannot reach the formation window.
"""

SHORT_TERM_REVERSAL = FeatureSpec(
    name="short_term_reversal",
    definition=(
        "One-month short-term reversal (Jegadeesh 1990, Lehmann 1990): the "
        "NEGATIVE of the log return of the split- and dividend-adjusted close "
        "over the 21 trading bars ending at the last print before the rebalance "
        "date. The reversal sign is applied in the value, so a HIGH score is a "
        "recent LOSER — the leg the reversal premium accrues to — matching the "
        "factor's directional name. Expected premium sign: POSITIVE in these "
        "units (equivalently, the raw prior-month return earns a negative "
        "premium). NaN when the security has fewer than 22 adjusted closes within "
        "60 calendar days of the rebalance date, or has not printed within 10 "
        "calendar days of it."
    ),
    units=(
        "negative log return over the prior 21 trading bars, dimensionless fraction "
        "(higher = larger recent loss)"
    ),
    availability_lag=dt.timedelta(0),
    source_tables=frozenset({PRICE_SOURCE_TABLE}),
)
"""Declaration of ``short_term_reversal``. Lag zero — see the module docstring.

Unlike :data:`MOMENTUM_12_1` this factor's freshest input is the most recent
close, so it is the one whose value would move first if the price connector's
knowledge-time policy (P3.4, blocked on B1) turned out to be optimistic. The
loader's temporal-integrity check is what refuses that case outright.
"""


def momentum_12_1_value(series: PriceSeries, *, compute_date: dt.date) -> float:
    """Compute twelve-month-minus-one-month momentum for one security.

    The arithmetic, separated from the query so it can be checked against a
    hand-worked example without a database::

        ln( close[-(21 + 1)] / close[-(231 + 21 + 1)] )

    Negative indices count back from the last print before ``compute_date``, so
    the numerator is the close 21 bars before that print and the denominator the
    close 252 bars before it.

    Args:
        series: the security's adjusted closes over the loaded window, ascending
            by trading day, every one strictly before ``compute_date``.
        compute_date: the rebalance date, used only for the staleness check.

    Returns:
        The formation-window log return (dimensionless fraction), or ``NaN``
        when the window cannot be formed: fewer than 253 prints in the loaded
        window, a last print more than
        :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS` old, or a
        non-positive close at either endpoint (a corrupt print, for which no
        substitute is invented).

    Example:
        >>> import datetime as dt
        >>> import numpy as np
        >>> from backend.features.factors._prices import PriceSeries
        >>> closes = np.linspace(100.0, 200.0, 253)
        >>> days = tuple(dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(253))
        >>> series = PriceSeries(1, days, closes)
        >>> round(momentum_12_1_value(series, compute_date=dt.date(2025, 9, 11)), 6)
        0.650588
    """
    closes = series.adjusted_close
    required = MOMENTUM_SKIP_BARS + MOMENTUM_FORMATION_BARS + 1
    if closes.size < required or series.is_stale(compute_date, tolerance=MAX_PRICE_STALENESS):
        return float("nan")
    end = float(closes[-(MOMENTUM_SKIP_BARS + 1)])
    start = float(closes[-required])
    if end <= 0.0 or start <= 0.0:
        return float("nan")
    return float(np.log(end / start))


def short_term_reversal_value(series: PriceSeries, *, compute_date: dt.date) -> float:
    """Compute one-month short-term reversal for one security.

    The arithmetic, with the reversal sign applied::

        -ln( close[-1] / close[-(21 + 1)] )

    Args:
        series: the security's adjusted closes over the loaded window, ascending
            by trading day, every one strictly before ``compute_date``.
        compute_date: the rebalance date, used only for the staleness check.

    Returns:
        The negated prior-month log return (dimensionless fraction), so a
        positive value is a recent loser. ``NaN`` when the window cannot be
        formed: fewer than 22 prints in the loaded window, a last print more than
        :data:`~backend.features.factors._prices.MAX_PRICE_STALENESS` old, or a
        non-positive close at either endpoint.

    Example:
        >>> import datetime as dt
        >>> import numpy as np
        >>> from backend.features.factors._prices import PriceSeries
        >>> closes = np.full(22, 100.0)
        >>> closes[-1] = 110.0
        >>> days = tuple(dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(22))
        >>> series = PriceSeries(1, days, closes)
        >>> round(short_term_reversal_value(series, compute_date=dt.date(2025, 1, 23)), 6)
        -0.09531
    """
    closes = series.adjusted_close
    required = REVERSAL_BARS + 1
    if closes.size < required or series.is_stale(compute_date, tolerance=MAX_PRICE_STALENESS):
        return float("nan")
    end = float(closes[-1])
    start = float(closes[-required])
    if end <= 0.0 or start <= 0.0:
        return float("nan")
    return -float(np.log(end / start))


@feature(MOMENTUM_12_1)
async def momentum_12_1(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Compute ``momentum_12_1`` for the requested securities.

    Args:
        session: an ``AsyncSession`` already pinned by :func:`backend.db.as_of`
            to the instant this feature's declaration permits. Read through it;
            do not open another.
        request: the securities, the compute date, and the pinned instant.

    Returns:
        One ``float64`` per requested security, in ``request.security_ids``
        order, in :data:`MOMENTUM_12_1`'s units (dimensionless log return).
        ``NaN`` where the formation window could not be built.

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
        lookback_days=MOMENTUM_LOOKBACK_DAYS,
    )
    return np.array(
        [
            momentum_12_1_value(history[security_id], compute_date=request.compute_date)
            for security_id in request.security_ids
        ],
        dtype=np.float64,
    )


@feature(SHORT_TERM_REVERSAL)
async def short_term_reversal(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Compute ``short_term_reversal`` for the requested securities.

    Args:
        session: an ``AsyncSession`` already pinned by :func:`backend.db.as_of`
            to the instant this feature's declaration permits.
        request: the securities, the compute date, and the pinned instant.

    Returns:
        One ``float64`` per requested security, in ``request.security_ids``
        order, in :data:`SHORT_TERM_REVERSAL`'s units — the negated prior-month
        log return, so a high value is a recent loser. ``NaN`` where the window
        could not be built.

    Raises:
        PriceTemporalIntegrityError: if a visible price bar is dated on or after
            the compute date.
        PriceSeriesError: if two visible bars share a security and a trading day.
    """
    history = await load_adjusted_closes(
        session,
        security_ids=request.security_ids,
        compute_date=request.compute_date,
        as_of=request.as_of,
        lookback_days=REVERSAL_LOOKBACK_DAYS,
    )
    return np.array(
        [
            short_term_reversal_value(history[security_id], compute_date=request.compute_date)
            for security_id in request.security_ids
        ],
        dtype=np.float64,
    )
