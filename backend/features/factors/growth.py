"""Balance-sheet growth factors: accruals and asset growth (P5.3).

Both are anomalies of *expansion*: the finding that companies whose balance
sheets have recently grown — through accounting accruals or through outright
investment — subsequently underperform.

**Accruals** (Sloan 1996). Earnings are the sum of a cash component and an
accrual component, and the accrual component is far less persistent. Investors
who fixate on the earnings total therefore over-extrapolate the accrual part,
and firms with high accruals disappoint. Measured here in the cash-flow form —
net income less operating cash flow — rather than Sloan's original
balance-sheet-difference form, because the cash-flow form is a single-period
identity that does not require reconstructing working-capital deltas across a
restatement, an acquisition or a change in the presentation of the balance
sheet. The two agree in ordinary cases and differ in exactly the cases where the
balance-sheet form is unreliable.

**Asset growth** (Cooper, Gulen & Schill 2008). The year-over-year growth in
total assets is one of the strongest known cross-sectional predictors of
returns, and its sign is negative: firms that expand their asset base
underperform, whether the expansion came from capital expenditure, acquisitions
or working capital. Empire-building, over-extrapolation of recent success, or
simply the point in the investment cycle at which capital is cheapest — the
factor is agnostic about which.

**Both keep their natural sign.** A high ``accruals`` and a high
``asset_growth`` are the *underperforming* legs, so the expected premium sign on
both is NEGATIVE. They are named after the quantities they measure rather than
after the strategies that trade them, so — unlike ``low_volatility`` and
``short_term_reversal`` — the value is not negated. The package's convention is
stated once in :mod:`backend.features.factors`; the point of writing it down is
that a factor library in which the sign convention has to be inferred is a
factor library that will eventually be traded backwards.

**Neither factor computes today.** The point-in-time fundamentals connector
(P3.5, Sharadar SF1 on the as-reported ARQ/ARY dimensions) is blocked on B1, so
there is no fundamentals table. Each computation raises
:class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`
naming the blocker rather than returning a plausible ratio (I3).

--------------------------------------------------------------------------
Availability lag: 7 days, with one wrinkle these two factors add
--------------------------------------------------------------------------

Both declare
:data:`~backend.features.factors.value.FUNDAMENTAL_AVAILABILITY_LAG`, whose
three components are derived in :mod:`backend.features.factors.value`.

The wrinkle is that both factors are **differences across a year**, so each
needs two observations rather than one: the current period and the same period a
year earlier. That does not change the lag — a lag bounds how fresh the
*freshest* input may be, and the older observation is a year staler still — but
it does change what a missing filing costs. A single unfiled quarter makes the
factor unavailable for a full year of rebalance dates rather than for one, and
that shows up as coverage rather than as error. The declaration says ``NaN``
where either observation is missing, and the Features page's coverage column
(§6.4) is where the consequence becomes visible.

The prior-year observation must be the one that was knowable **at the same
cutoff**, not the one believed correct today. A restatement published after the
cutoff must not reach back and change last year's number, or the factor becomes
a measurement of what was later admitted — the exact lookahead I1 exists to
prevent, and the reason B1 rejected the two vendors that serve restated
fundamentals.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**Accruals is a dimensionless ratio**, USD of annual accruals per USD of average
total assets. The denominator is the *average* of the current and prior-year
total assets, not the current level: a company that doubled its asset base
during the year would otherwise have its accruals scaled by a base it held for
only part of the period.

**Asset growth is a dimensionless fraction**, not a percent and not a log
change. ``0.25`` means total assets grew 25% year over year. The simple growth
rate rather than the log is deliberate here — the factor is conventionally
defined and published that way, and the asymmetry of the simple form (bounded
below at ``-1``, unbounded above) is handled by the pipeline's 1/99
winsorization.

**Flows are trailing twelve months, stocks are point-in-time.** Net income and
operating cash flow are summed over the four most recent consecutive as-reported
quarters; total assets are balance-sheet levels at the latest of those quarters
and at the corresponding quarter a year earlier.

**Non-positive denominators are ``NaN``.** Average total assets and prior-year
total assets are positive for a going concern; a non-positive one is corrupt
data or a company this ratio does not describe. Negative accruals (cash earnings
exceeding reported earnings) and negative asset growth (a shrinking balance
sheet) are ordinary facts and are reported as the negative numbers they are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.features.factors._fundamentals import (
    FUNDAMENTALS_TABLE,
    require_fundamentals_source,
)
from backend.features.factors.value import FUNDAMENTAL_AVAILABILITY_LAG
from backend.features.registry import feature
from backend.features.spec import FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray

__all__ = [
    "ACCRUALS",
    "ASSET_GROWTH",
    "accruals",
    "asset_growth",
]

ACCRUALS = FeatureSpec(
    name="accruals",
    definition=(
        "Accounting accruals (Sloan 1996), in the cash-flow form: as-reported net "
        "income less net cash from operating activities, each summed over the four "
        "most recent consecutive fiscal quarters knowable at the cutoff, divided by "
        "the average of total assets at the latest of those quarters and at the "
        "corresponding quarter one year earlier. The cash-flow form is used rather "
        "than Sloan's balance-sheet-difference form because it is a single-period "
        "identity that survives restatements, acquisitions and changes in balance "
        "sheet presentation, which the difference form does not. The prior-year "
        "figure is the one knowable at the same cutoff, never a later restatement. "
        "Expected premium sign: NEGATIVE — high-accrual firms have historically "
        "underperformed, so the factor keeps its natural sign and is NOT negated. "
        "Negative accruals (cash earnings above reported earnings) are a real fact "
        "and produce a negative value; non-positive average total assets, a missing "
        "prior-year observation, or fewer than four consecutive knowable quarters "
        "produce NaN. "
        "BLOCKED: the point-in-time fundamentals connector (P3.5) is blocked on "
        "B1, so this feature raises rather than returning a value."
    ),
    units=(
        "dimensionless ratio: USD of trailing-twelve-month accruals per USD of average total assets"
    ),
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE}),
)
"""Declaration of ``accruals``. Lag 7 days — see the module docstring.

Needs two observations a year apart, both as they were knowable at the cutoff.
That does not lengthen the lag (the older one is a year staler already) but it
does mean a single unfiled quarter costs a year of coverage rather than a day.
"""

ASSET_GROWTH = FeatureSpec(
    name="asset_growth",
    definition=(
        "Year-over-year total asset growth (Cooper, Gulen & Schill 2008): "
        "as-reported total assets at the most recent fiscal quarter knowable at "
        "the cutoff, less total assets at the corresponding quarter one year "
        "earlier, divided by that prior-year level. Expressed as a simple growth "
        "rate rather than a log change, matching the published definition; the "
        "asymmetry of the simple form is handled by the pipeline's 1/99 "
        "winsorization. The prior-year figure is the one knowable at the same "
        "cutoff, never a later restatement. Expected premium sign: NEGATIVE — "
        "firms that expand their asset base have historically underperformed, so "
        "the factor keeps its natural sign and is NOT negated. A shrinking balance "
        "sheet is a real fact and produces a negative value; non-positive "
        "prior-year total assets, or a missing observation at either end, produce "
        "NaN. "
        "BLOCKED: the point-in-time fundamentals connector (P3.5) is blocked on "
        "B1, so this feature raises rather than returning a value."
    ),
    units=(
        "dimensionless fraction: year-over-year change in total assets per USD of "
        "prior-year total assets (0.25 = 25% growth)"
    ),
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE}),
)
"""Declaration of ``asset_growth``. Lag 7 days — see the module docstring."""


@feature(ACCRUALS)
async def accruals(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``accruals``: its fundamentals feed does not exist.

    Args:
        _session: the pinned session, unused — there is no table to read.
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


@feature(ASSET_GROWTH)
async def asset_growth(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``asset_growth``: its fundamentals feed does not exist.

    Args:
        _session: the pinned session, unused — there is no table to read.
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
