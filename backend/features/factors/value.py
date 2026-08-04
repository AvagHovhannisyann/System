"""Value factors: book-to-price and earnings yield (P5.3).

Both ask the same question — how much of an accounting quantity does a dollar of
market value buy — and both are ratios of a **fundamental** (slow, published on
a filing calendar) to a **market price** (fast, published every day). That
mismatch is the whole difficulty of declaring them correctly, and it is what the
availability-lag section below is about.

**Neither factor computes today.** The point-in-time fundamentals connector
(P3.5, Sharadar SF1 on the as-reported ARQ/ARY dimensions) is blocked on B1, so
there is no fundamentals table. Each computation raises
:class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`
naming the blocker. The declarations are complete and are the contract P3.5 will
be built against; the arithmetic is the part that cannot be honest yet. See
:mod:`backend.features.factors._fundamentals` for why a plausible number here
would be worse than an exception.

--------------------------------------------------------------------------
Availability lag: 7 days, and where the number comes from
--------------------------------------------------------------------------

A fundamental is knowable when its filing becomes public, not when its fiscal
period ends — a December quarter end is knowable in February, not on 31
December. That much is not in dispute and it is not what the lag is for: under
D-011 the fundamentals connector stamps each row's ``knowledge_time`` from the
original filing, and the as-of session already refuses anything later. The
declared lag is a **margin on top of that**, and it is sized to cover the
uncertainty in a connector that has not been written.

Three sources of uncertainty, each in the lookahead direction, each rounded up:

**1. Date-only granularity (up to 4 days).** Sharadar SF1 gives ``datekey``, a
calendar date with no time of day. D-011's documented default for a date-only
source is the next trading day at ``00:00Z`` — never the report date itself,
since same-day availability at the open cannot be assumed. Across a Friday
filing followed by a Monday holiday that is four calendar days.

**2. Filing date is not acceptance instant (1 day).** Under 17 CFR 232.13 a
submission accepted after 17:30 ET is deemed filed the *next* business day, and
— as :class:`backend.db.models.EdgarFiling` documents with real accession
numbers — a filing date can also *precede* the acceptance instant once both are
expressed in UTC. The second direction is the lookahead one: a connector keying
on the filing date would claim a filing was knowable roughly a day before it
existed.

**3. Vendor delivery (2 days).** A row appears in the vendor's file some time
after the filing it summarizes. That delay is not documented in code, and
``BLOCKERS.md`` B1 warns that the point-in-time claim itself must be validated
on trial data before purchase — vendors describe as point-in-time things that
are not. Two days is a guess in the conservative direction, and it is labelled
as one.

Seven days total. **This is a B1-blocked estimate and it is deliberately too
long rather than too short.** A lag that is too long costs signal — a value
score is a week staler than it needed to be — while a lag that is too short
fabricates foresight, and no test of the arithmetic detects the second. When
P3.5 lands with a documented knowledge-time policy and the gate has validated it
against a known restatement, this number can be *reduced* on that evidence and
the change logged in ``DECISIONS.md``. It should never be reduced on the grounds
that it is costing signal.

**What the lag costs, stated plainly.** ``FeatureSpec`` carries one lag per
feature, so the 7 days applies to the *whole* computation, including the price
leg — book-to-price on date ``D`` uses the close from ``D - 7 days``, not
yesterday's. For a value factor that is a small and acceptable loss: book-to-
price moves on filings and slowly on price, and a week of price staleness
changes rankings marginally. It is a real cost and it is accepted knowingly; the
alternative, a per-source lag, is not expressible in the frozen wave-1 schema.

--------------------------------------------------------------------------
Units and assumptions
--------------------------------------------------------------------------

**Both factors are dimensionless ratios** — USD of an accounting quantity per
USD of market value. Not percent, not basis points. A book-to-price of ``0.4``
means forty cents of common equity per dollar of market capitalization.

**Yields, not multiples.** The accounting quantity is the numerator and market
value the denominator, deliberately. The inverted forms (P/B, P/E) are
discontinuous through zero and unbounded for a company with near-zero earnings,
so a cross-section of them cannot be winsorized or standardized sensibly. The
yield form is continuous through zero and takes a meaningful negative value for
a loss-making company, which is a fact about the company rather than an artifact.

**Market capitalization is the unadjusted close times common shares
outstanding.** ``close_raw_usd``, not ``close_usd``: shares outstanding is an
unadjusted count as of the report, and multiplying it by a split-adjusted price
would misstate market value by the adjustment factor — the exact silent
units error directive §8 is written about. The two must be on the same
adjustment basis, and unadjusted is the basis the share count is quoted in.

**Common equity excludes preferred stock and is as reported**, not restated: the
figure the filing carried at the time, which is what the operator could have
computed. Restated fundamentals are the lookahead that invariant I1 exists to
prevent, and are why B1 rejected two cheaper vendors.

**Trailing twelve months means four consecutive as-reported quarters**, summed.
Not an annualized latest quarter, and not a fiscal-year figure stretched to fit:
either would mix a seasonal quarter with a full year across the cross-section
and make the factor partly a measurement of fiscal calendars.

**Negative denominators are ``NaN``, negative numerators are values.** Market
capitalization is positive by construction, so a non-positive one is corrupt
data rather than a measurement. Negative common equity and negative trailing
earnings are ordinary facts about real companies and are reported as the
negative ratios they produce.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

from backend.features.factors._fundamentals import (
    FUNDAMENTALS_TABLE,
    require_fundamentals_source,
)
from backend.features.factors._prices import PRICE_SOURCE_TABLE
from backend.features.registry import feature
from backend.features.spec import FeatureSpec

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FeatureComputeRequest, FloatArray

__all__ = [
    "BOOK_TO_PRICE",
    "EARNINGS_YIELD",
    "FUNDAMENTAL_AVAILABILITY_LAG",
    "TRAILING_QUARTERS",
    "book_to_price",
    "earnings_yield",
]

FUNDAMENTAL_AVAILABILITY_LAG: Final = dt.timedelta(days=7)
"""Availability-lag margin every fundamentals-derived factor declares (wall clock).

A margin **on top of** the store's own ``knowledge_time``, not a replacement for
it. Sized from three B1-blocked uncertainties in the unwritten P3.5 connector,
each rounded up in the lookahead direction: date-only ``datekey`` granularity
resolved to the next trading day (up to 4 days across a holiday weekend), filing
date versus acceptance instant (1 day), and vendor delivery after the filing
(2 days, a labelled guess). The module docstring derives each.

Shared by all six fundamental factors on purpose: they read one table under one
undecided policy, so a per-factor number would imply a distinction that does not
exist. Reducible only on evidence — a documented connector policy validated by
the Phase 3 gate against a known restatement — and the reduction goes in
``DECISIONS.md``. Never reducible on the grounds that it costs signal.
"""

TRAILING_QUARTERS: Final = 4
"""Consecutive as-reported quarters summed for a trailing-twelve-month figure (count)."""

BOOK_TO_PRICE = FeatureSpec(
    name="book_to_price",
    definition=(
        "Book-to-price (the value factor of Fama & French 1992, in yield form): "
        "as-reported common shareholders' equity — total equity less preferred "
        "stock — from the most recent fiscal period whose filing was knowable at "
        "the cutoff, divided by market capitalization (unadjusted close on the "
        "last trading day at or before the cutoff, times common shares "
        "outstanding as reported for that same period). Stated as a yield rather "
        "than as price-to-book because the inverted form is discontinuous through "
        "zero and cannot be winsorized or standardized across a cross-section. "
        "Expected premium sign: POSITIVE — cheap stocks have historically "
        "outperformed expensive ones. Negative common equity is a real fact and "
        "produces a negative ratio; a non-positive market capitalization is "
        "corrupt data and produces NaN, as does a missing filing. "
        "BLOCKED: the point-in-time fundamentals connector (P3.5) is blocked on "
        "B1, so this feature raises rather than returning a value."
    ),
    units=(
        "dimensionless ratio: USD of as-reported common equity per USD of market capitalization"
    ),
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE}),
)
"""Declaration of ``book_to_price``. Lag 7 days — see the module docstring.

Declares both source tables because the ratio needs both legs: the equity from
the fundamentals feed, the market value from ``price_bar``. Both are read at the
same cutoff, which is what the single declared lag means.
"""

EARNINGS_YIELD = FeatureSpec(
    name="earnings_yield",
    definition=(
        "Trailing earnings yield (Basu 1977, in yield form): as-reported net "
        "income available to common shareholders summed over the four most recent "
        "consecutive fiscal quarters whose filings were knowable at the cutoff, "
        "divided by market capitalization (unadjusted close on the last trading "
        "day at or before the cutoff, times common shares outstanding as reported "
        "for the latest of those quarters). Four quarters are summed rather than "
        "annualizing the latest one, so the factor does not become a measurement "
        "of seasonality and fiscal-calendar alignment. Expected premium sign: "
        "POSITIVE. Negative trailing earnings are a real fact and produce a "
        "negative yield; fewer than four consecutive knowable quarters, or a "
        "non-positive market capitalization, produce NaN. "
        "BLOCKED: the point-in-time fundamentals connector (P3.5) is blocked on "
        "B1, so this feature raises rather than returning a value."
    ),
    units=(
        "dimensionless ratio: USD of trailing-twelve-month net income to common per "
        "USD of market capitalization"
    ),
    availability_lag=FUNDAMENTAL_AVAILABILITY_LAG,
    source_tables=frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE}),
)
"""Declaration of ``earnings_yield``. Lag 7 days — see the module docstring."""


@feature(BOOK_TO_PRICE)
async def book_to_price(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``book_to_price``: its fundamentals feed does not exist.

    The declaration above is the deliverable; this is the honest computation for
    a feature whose input is blocked. It reads nothing and returns nothing.

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


@feature(EARNINGS_YIELD)
async def earnings_yield(_session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
    """Refuse to compute ``earnings_yield``: its fundamentals feed does not exist.

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
