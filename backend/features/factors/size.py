"""Size: log market capitalization, and the share count that does not exist (P5.3).

The oldest documented cross-sectional anomaly (Banz 1981) and one leg of the
Fama & French (1993) three-factor model: small companies have historically
earned higher returns than large ones by more than their market betas explain.
The effect is weaker and more contested than it was in 1981 — much of it
concentrates in January, in the smallest decile, and in names too illiquid to
trade at size — which is a reason to measure it honestly rather than a reason to
leave it out. It is also the single most-used *control* in this literature: a
value or quality premium that turns out to be a size premium in disguise is the
standard way a factor study goes wrong, and P5.6's correlation review needs this
column to be able to see that.

**This factor does not compute today.** Market capitalization is a price times a
**share count**, and the share count comes from the point-in-time fundamentals
feed (P3.5, Sharadar SF1 on the as-reported ARQ/ARY dimensions) which is blocked
on B1. The computation raises
:class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`
through the same gate the six value/quality/growth factors use. See
:mod:`backend.features.factors._fundamentals` for why a plausible number here
would be worse than an exception.

--------------------------------------------------------------------------
The tempting shortcut, named so it is not taken
--------------------------------------------------------------------------

``price_bar`` exists. It is therefore possible to write a "size" factor from
prices alone, and the result would be plausible, well-typed, correctly shaped and
completely wrong: **price per share is not market value**. A $500 share of a
mid-cap and a $20 share of a mega-cap rank in the wrong order, and the resulting
column would be a measurement of share-splitting policy rather than of company
size. Nothing downstream could tell the difference — the distribution would look
fine, the z-scores would be well behaved, and the backtest would report a
premium on an ordering nobody intended.

There is no share count anywhere in this store. ``security_master`` carries
identity, not capitalization; ``price_bar`` carries prices and traded volume,
not shares outstanding. So the refusal is not a policy choice about tidiness, it
is the arithmetic: the numerator of this factor has no source. A "shares
outstanding" table is not invented here for the same reason
:mod:`~backend.features.factors._fundamentals` does not invent one — a
fabricated share count produces a fabricated market cap, and I3 forbids exactly
that (directive §9.1-9.2).

--------------------------------------------------------------------------
Availability lag: 7 days, the fundamentals regime
--------------------------------------------------------------------------

Declares :data:`~backend.features.factors.value.FUNDAMENTAL_AVAILABILITY_LAG`,
whose three components are derived in :mod:`backend.features.factors.value`:
date-only ``datekey`` granularity resolved to the next trading day (up to 4 days
across a holiday weekend), filing date versus acceptance instant (1 day), and
vendor delivery after the filing (2 days, a labelled guess).

Size sits on the *fundamentals* side of D-027's two regimes even though half its
arithmetic is a price, because a lag bounds the freshest input a feature may
read and the share count is the slower of the two. The cost is the one D-027
names explicitly: ``FeatureSpec`` carries one lag per feature, so the 7 days
applies to the price leg as well and market cap on date ``D`` uses the close from
``D - 7 days``. For a factor whose cross-sectional ordering spans five orders of
magnitude that is a rounding error — a week of price drift reorders adjacent
names, never a micro-cap past a mega-cap — and it is the safe direction.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**The value is the natural log of market capitalization measured in USD.**
``ln(1e9) = 20.72`` is a one-billion-dollar company. Log rather than raw USD
because market capitalization spans four to five orders of magnitude across any
realistic universe: a raw-USD cross-section is so right-skewed that the 1/99
winsorization clips the top of the distribution at an arbitrary economic size and
the subsequent z-score turns the column into approximately a dummy for "is this
one of the two largest names". The log makes the cross-section roughly symmetric,
which is what the P5.2 transform pipeline assumes, and it is the form the
literature publishes (log market equity, "ME").

**"Dimensionless" here means the log of a count of dollars, not a pure number
whose value is currency-independent.** ``ln(market capitalization / 1 USD)``.
Restating the same company in another currency shifts *every* value by the same
constant, which cancels exactly in the cross-sectional demeaning — so rankings
and z-scores are unit-independent while raw levels are not. This is stated
because a level compared across two runs that disagreed about the unit would
differ by a plausible-looking constant rather than by an obvious factor.

**Market capitalization is the unadjusted close times common shares
outstanding**, matching :mod:`backend.features.factors.value` exactly:
``close_raw_usd``, not ``close_usd``. Shares outstanding is an unadjusted count
as of the report, and multiplying it by a split-adjusted price misstates market
value by the adjustment factor — the silent units error directive §8 exists for.
The two legs must be on one adjustment basis and unadjusted is the basis the
share count is quoted in.

**Common shares outstanding, as reported**, from the most recent fiscal period
whose filing was knowable at the cutoff — not the current count, and not a
restated one. A buyback or a secondary offering changes the count, and the figure
that was knowable at the cutoff is the one the operator could have used. A
restated share count is the lookahead I1 exists to prevent.

**A non-positive market capitalization is ``NaN``, not a large negative log.**
Market value is positive by construction for a listed common share, so a
non-positive product is corrupt data rather than a measurement, and ``ln`` of it
is undefined. A missing filing, a missing price, or a missing share count is
likewise ``NaN``: not available, never imputed from a peer, a sector median or a
previous period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.features.factors._fundamentals import (
    FUNDAMENTALS_TABLE,
    require_fundamentals_source,
)
from backend.features.factors._prices import PRICE_SOURCE_TABLE
from backend.features.factors.value import FUNDAMENTAL_AVAILABILITY_LAG
from backend.features.registry import feature
from backend.features.spec import FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray

__all__ = [
    "SIZE",
    "size",
]

SIZE = FeatureSpec(
    name="size",
    definition=(
        "Company size (Banz 1981; the ME leg of Fama & French 1993), as the "
        "natural log of market capitalization in USD: the unadjusted close on the "
        "last trading day at or before the cutoff, times common shares outstanding "
        "as reported for the most recent fiscal period whose filing was knowable "
        "at the cutoff. Unadjusted price rather than adjusted, because the share "
        "count is an unadjusted count and multiplying it by a split-adjusted price "
        "misstates market value by the adjustment factor. The log is taken because "
        "market capitalization spans four to five orders of magnitude, and the raw "
        "cross-section winsorizes and standardizes into approximately a dummy for "
        "the largest name. Expected premium sign: NEGATIVE — large companies have "
        "historically underperformed small ones, so the factor is named after the "
        "quantity it measures and is NOT negated. A non-positive market "
        "capitalization is corrupt data and produces NaN, as does a missing price, "
        "a missing share count or a missing filing. "
        "BLOCKED: the share count comes from the point-in-time fundamentals "
        "connector (P3.5), which is blocked on B1, so this feature raises rather "
        "than returning a value. Price alone is NOT a substitute: price per share "
        "is not market value."
    ),
    units=(
        "natural log of market capitalization measured in USD, dimensionless "
        "(ln(1e9) = 20.72 for a one-billion-dollar company; a change of currency "
        "unit shifts every value by one constant and cancels in the cross-section)"
    ),
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE}),
)
"""Declaration of ``size``. Lag 7 days — see the module docstring.

Declares both source tables because market capitalization needs both legs: the
share count from the fundamentals feed, the price from ``price_bar``. Both are
read at the same cutoff, which is what the single declared lag means. The
fundamentals leg is the one that does not exist, and it is the numerator.
"""


@feature(SIZE)
async def size(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``size``: there is no share count to multiply a price by.

    Routed through the same gate as the six fundamentals factors rather than
    through a new one, because it is blocked for the same reason by the same
    connector, and two error families for one blocker would make the platform's
    state harder to read rather than more precise.

    Args:
        _session: the pinned session, unused — there is no share count to read.
            Refusing before any I/O also means the refusal cannot be half-done.
        request: the securities and compute date, used only to name the feature
            in the error.

    Returns:
        Never returns.

    Raises:
        FundamentalsSourceUnavailableError: always, while P3.5 is blocked on B1.
        FundamentalsComputationNotWrittenError: instead of the above once the
            fundamentals table is mapped and this arithmetic is still unwritten.
    """
    require_fundamentals_source(request.feature)
