"""Quality factors: gross profitability and return on invested capital (P5.3).

Two ways of asking whether a business earns more than the capital it consumes,
chosen because they fail differently and so are worth carrying together.

**Gross profitability** (Novy-Marx 2013) is revenue minus cost of goods sold,
scaled by total assets. Its whole argument is that it sits at the *top* of the
income statement, above the lines management has the most discretion over —
depreciation policy, capitalization choices, restructuring charges, tax
structuring. That makes it noisy about economics but comparatively clean about
accounting, and it is why it is measured against assets rather than equity: a
gross-profit-to-equity ratio would mostly measure leverage.

**Return on invested capital** is the opposite trade. It is a genuine economic
return — after-tax operating profit over the capital actually funding
operations — and it is the number a business owner would care about, but every
one of its components has been through the discretionary part of the statements.
Carrying both, and reviewing their correlation (the P5.6 gate), is the point.

**Neither factor computes today.** The point-in-time fundamentals connector
(P3.5, Sharadar SF1 on the as-reported ARQ/ARY dimensions) is blocked on B1, so
there is no fundamentals table. Each computation raises
:class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`
naming the blocker rather than returning a plausible ratio (I3).

--------------------------------------------------------------------------
Availability lag: 7 days
--------------------------------------------------------------------------

Both declare
:data:`~backend.features.factors.value.FUNDAMENTAL_AVAILABILITY_LAG`, the shared
margin whose three components — date-only ``datekey`` granularity resolved to
the next trading day, filing date versus acceptance instant, and vendor delivery
after filing — are derived in :mod:`backend.features.factors.value`. It is a
margin on top of the store's own ``knowledge_time``, it is a B1-blocked estimate
chosen too long rather than too short, and it may be reduced only on evidence
from the Phase 3 gate.

One point is specific to these two factors and worth stating: unlike the value
factors, neither reads a price. So the lag costs them nothing but staleness in
the fundamentals themselves, which move on a filing calendar anyway — a
seven-day margin on a quarterly figure is close to free. If the shared lag ever
needs to differ per factor, these are the two where a *longer* one would be
cheapest, not the two where a shorter one is most tempting.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**Both factors are dimensionless ratios** — USD of annual profit per USD of a
capital base. Not percent, not basis points: an ROIC of ``0.12`` is 12%.

**Flows are trailing twelve months, stocks are point-in-time.** A ratio of a
flow (a year of profit) to a stock (assets at an instant) needs both stated:
revenue, cost of goods sold, operating profit and taxes are summed over the four
most recent consecutive as-reported quarters; total assets and invested capital
are taken from the balance sheet of the latest of those quarters. Mixing a
quarterly flow with an annual one across the cross-section would make the factor
partly a measurement of fiscal calendars.

**Invested capital is total debt plus common and preferred equity, less cash and
short-term investments.** The excess-cash deduction is the part with a
judgement in it: cash earns no operating return, so leaving it in the
denominator penalizes a cash-rich company for holding cash rather than for
running a poor business. The alternative convention (no deduction) is defensible
and would produce a different, more leverage-sensitive factor; this library
picks one and states it, because a ROIC whose denominator is unspecified is not
comparable to anything.

**NOPAT is operating profit after an effective tax rate**, the effective rate
being trailing tax expense over trailing pre-tax income, clamped to ``[0, 1]``.
The clamp is not cosmetic: a company with a pre-tax loss and a tax benefit
produces an effective rate outside that interval, and an unclamped rate would
flip the sign of NOPAT and report a profitable operation for a loss-making one.

**Non-positive denominators are ``NaN``; negative numerators are values.** Total
assets and invested capital are positive for a going concern, so a non-positive
one is either corrupt or a capital structure this ratio does not describe —
either way not a measurement. Negative gross profit and negative NOPAT are
ordinary facts and are reported as the negative ratios they produce.
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
    "GROSS_PROFITABILITY",
    "ROIC",
    "gross_profitability",
    "roic",
]

GROSS_PROFITABILITY = FeatureSpec(
    name="gross_profitability",
    definition=(
        "Gross profitability (Novy-Marx 2013): as-reported revenue less cost of "
        "goods sold, summed over the four most recent consecutive fiscal quarters "
        "whose filings were knowable at the cutoff, divided by total assets from "
        "the balance sheet of the latest of those quarters. Measured at the top of "
        "the income statement, above the lines most exposed to accounting "
        "discretion, and scaled by assets rather than equity so the ratio does not "
        "become a measurement of leverage. Expected premium sign: POSITIVE — "
        "profitable firms have historically outperformed unprofitable ones of "
        "similar valuation. Negative gross profit is a real fact and produces a "
        "negative ratio; non-positive total assets, or fewer than four consecutive "
        "knowable quarters, produce NaN. "
        "BLOCKED: the point-in-time fundamentals connector (P3.5) is blocked on "
        "B1, so this feature raises rather than returning a value."
    ),
    units="dimensionless ratio: USD of trailing-twelve-month gross profit per USD of total assets",
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE}),
)
"""Declaration of ``gross_profitability``. Lag 7 days — see the module docstring.

Reads only the fundamentals feed: no price leg, so the declared lag costs
nothing beyond staleness in figures that move on a filing calendar anyway.
"""

ROIC = FeatureSpec(
    name="roic",
    definition=(
        "Return on invested capital: net operating profit after tax divided by "
        "invested capital. NOPAT is as-reported operating profit (EBIT) summed "
        "over the four most recent consecutive fiscal quarters knowable at the "
        "cutoff, times one minus the effective tax rate (trailing tax expense over "
        "trailing pre-tax income, clamped to [0, 1] so a tax benefit on a pre-tax "
        "loss cannot flip the sign of NOPAT). Invested capital is total debt plus "
        "common and preferred equity less cash and short-term investments, from "
        "the balance sheet of the latest of those quarters; cash is deducted "
        "because it earns no operating return and would otherwise penalize a "
        "cash-rich firm for its balance sheet rather than its business. Expected "
        "premium sign: POSITIVE. Negative NOPAT is a real fact and produces a "
        "negative ratio; non-positive invested capital, or fewer than four "
        "consecutive knowable quarters, produce NaN. "
        "BLOCKED: the point-in-time fundamentals connector (P3.5) is blocked on "
        "B1, so this feature raises rather than returning a value."
    ),
    units="dimensionless ratio: USD of trailing-twelve-month NOPAT per USD of invested capital",
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE}),
)
"""Declaration of ``roic``. Lag 7 days — see the module docstring."""


@feature(GROSS_PROFITABILITY)
async def gross_profitability(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``gross_profitability``: its fundamentals feed does not exist.

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


@feature(ROIC)
async def roic(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``roic``: its fundamentals feed does not exist.

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
