"""The baseline factor library (P5.3): nine declarations and their computations.

Importing this package registers all nine factors into the process-wide
:func:`~backend.features.registry.default_registry`. That is the intended
mechanism — ``registry.py`` says the catalog is *"populated by the factor
modules on import"* — and it means the 30-feature cap is enforced against a
catalog that actually reflects what the platform offers a model.

======================  ===========  ==============  =========================
Feature                 Lag          Computes today  Module
======================  ===========  ==============  =========================
``momentum_12_1``       0            yes             :mod:`~.momentum`
``short_term_reversal`` 0            yes             :mod:`~.momentum`
``low_volatility``      0            yes             :mod:`~.risk`
``book_to_price``       7 days       **no — B1**     :mod:`~.value`
``earnings_yield``      7 days       **no — B1**     :mod:`~.value`
``gross_profitability`` 7 days       **no — B1**     :mod:`~.quality`
``roic``                7 days       **no — B1**     :mod:`~.quality`
``accruals``            7 days       **no — B1**     :mod:`~.growth`
``asset_growth``        7 days       **no — B1**     :mod:`~.growth`
======================  ===========  ==============  =========================

--------------------------------------------------------------------------
Why six of the nine refuse to compute
--------------------------------------------------------------------------

The point-in-time fundamentals connector (P3.5, Sharadar SF1 on the as-reported
ARQ/ARY dimensions) is blocked on B1: no vendor credentials, therefore no table,
no column names, no knowledge-time policy. The six fundamentals-derived factors
are declared in full and their computations raise
:class:`~backend.features.factors._fundamentals.FundamentalsSourceUnavailableError`.

This is not a gap waiting to be filled with something approximate. Directive §2
I3 and §9.1-9.2 forbid a stub that returns a plausible value, and here the
prohibition has teeth: a fabricated book-to-price is indistinguishable from a
measured one at every point downstream — the transform pipeline, the design
matrix, the model, the optimizer, the backtest — and every one of them would
report healthy numbers. The output would be a backtest that looks real. An
exception naming the blocker is the only honest state.

The three price factors do compute, against ``price_bar``. That table exists
(migration 0002) but is empty, because its connector (P3.4) is *also* blocked on
B1 — so today they return ``NaN`` for every security. ``NaN`` here is honest in
a way it would not be for the six above: the query runs, the store answers, and
the answer is "no data". For a table that does not exist there is no query to
run and nothing to be not-available about.

--------------------------------------------------------------------------
Sign convention: read this before adding a factor
--------------------------------------------------------------------------

Every factor states its **expected premium sign** in its ``definition``, and the
value's sign follows one rule:

- a factor named after a **direction** carries that direction in its value.
  ``low_volatility`` is the negative of volatility, so a high score is a calm
  stock; ``short_term_reversal`` is the negative of last month's return, so a
  high score is a recent loser. In both cases the name promises a direction and
  the value keeps the promise.
- a factor named after a **measured quantity** does not negate. ``accruals`` and
  ``asset_growth`` are the quantities themselves, and both carry an expected
  premium sign of NEGATIVE — high accruals and fast-growing balance sheets are
  the underperforming legs. Renaming them to make every sign positive would put
  a strategy's name on a measurement.

A gradient-boosted ranker learns the sign either way, so the convention is not
for the model. It is for P5.4's premium check, which needs a stated expectation
per factor to test against, and for the correlation review at the P5.6 gate,
where two factors that look uncorrelated because one is inverted is a failure of
reading rather than of statistics.

--------------------------------------------------------------------------
Availability lags: two regimes, both justified where they are declared
--------------------------------------------------------------------------

**Price factors declare zero.** A daily close is knowable at that close — 16:00
ET, three to four hours before midnight UTC opening the next date — so nothing
has to be filed or published for it to exist. Any vendor delivery delay is
already carried by the connector's ``knowledge_time`` under D-011 and enforced
by the as-of session, so a margin on top would discard bars that genuinely were
available. The residual risk — a connector that stamps a bar knowable before its
own close — is checked directly rather than absorbed: any visible bar dated on
or after the compute date raises
:class:`~backend.features.factors._prices.PriceTemporalIntegrityError`.
:mod:`backend.features.factors.momentum` argues this in full.

**Fundamental factors declare seven days.** A margin on top of the store's
``knowledge_time``, sized from three uncertainties in a connector that has not
been written: date-only ``datekey`` granularity resolved to the next trading day
(up to 4 days across a holiday weekend), filing date versus acceptance instant
(1 day), and vendor delivery after the filing (2 days, a labelled guess).
:mod:`backend.features.factors.value` derives each. It is a B1-blocked estimate
deliberately chosen too long rather than too short — a lag too long costs signal,
a lag too short fabricates foresight, and only the second is invisible to every
test of the arithmetic. It may be reduced when P3.5 lands with a documented
policy validated by the Phase 3 gate against a known restatement, with the
change logged in ``DECISIONS.md``; it may not be reduced because it is costing
signal.

--------------------------------------------------------------------------
Feature budget
--------------------------------------------------------------------------

Nine of the thirty slots. ``PLAN.md`` P5.3 names two further baseline factors
not implemented here (size, short interest) plus Amihud illiquidity, and Phase 7
adds LLM-derived features against the same cap. Registration order decides only
*which* registration is refused once the catalog is full; the cap itself holds
after every operation (see :mod:`backend.features.registry`).
"""

from __future__ import annotations

from backend.features.factors._fundamentals import (
    FUNDAMENTALS_TABLE,
    FundamentalsComputationNotWrittenError,
    FundamentalsSourceUnavailableError,
)
from backend.features.factors._prices import (
    MAX_PRICE_STALENESS,
    PRICE_SOURCE_TABLE,
    PriceSeries,
    PriceSeriesError,
    PriceTemporalIntegrityError,
)
from backend.features.factors.growth import ACCRUALS, ASSET_GROWTH, accruals, asset_growth
from backend.features.factors.momentum import (
    MOMENTUM_12_1,
    SHORT_TERM_REVERSAL,
    momentum_12_1,
    short_term_reversal,
)
from backend.features.factors.quality import (
    GROSS_PROFITABILITY,
    ROIC,
    gross_profitability,
    roic,
)
from backend.features.factors.risk import LOW_VOLATILITY, low_volatility
from backend.features.factors.value import (
    BOOK_TO_PRICE,
    EARNINGS_YIELD,
    FUNDAMENTAL_AVAILABILITY_LAG,
    book_to_price,
    earnings_yield,
)
from backend.features.spec import FeatureSpec

__all__ = [
    "ACCRUALS",
    "ASSET_GROWTH",
    "BASELINE_FACTORS",
    "BOOK_TO_PRICE",
    "EARNINGS_YIELD",
    "FUNDAMENTALS_TABLE",
    "FUNDAMENTAL_AVAILABILITY_LAG",
    "GROSS_PROFITABILITY",
    "LOW_VOLATILITY",
    "MAX_PRICE_STALENESS",
    "MOMENTUM_12_1",
    "PRICE_SOURCE_TABLE",
    "ROIC",
    "SHORT_TERM_REVERSAL",
    "FundamentalsComputationNotWrittenError",
    "FundamentalsSourceUnavailableError",
    "PriceSeries",
    "PriceSeriesError",
    "PriceTemporalIntegrityError",
    "accruals",
    "asset_growth",
    "book_to_price",
    "earnings_yield",
    "gross_profitability",
    "low_volatility",
    "momentum_12_1",
    "roic",
    "short_term_reversal",
]

BASELINE_FACTORS: tuple[FeatureSpec, ...] = (
    ACCRUALS,
    ASSET_GROWTH,
    BOOK_TO_PRICE,
    EARNINGS_YIELD,
    GROSS_PROFITABILITY,
    LOW_VOLATILITY,
    MOMENTUM_12_1,
    ROIC,
    SHORT_TERM_REVERSAL,
)
"""Every declaration this package registers, sorted by name.

Sorted rather than grouped by module so it matches
:meth:`~backend.features.registry.FeatureRegistry.specs`, which enumerates by
name; a catalog that reads differently from the registry it describes is a
catalog that will drift from it.
"""
