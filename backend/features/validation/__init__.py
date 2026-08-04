"""Factor-premia validation: do the baseline factors reproduce what is known? (P5.4).

Gate G5's first clause — *"known factor premia reproduce (sign + plausible
magnitude)"* — is the only check in Phase 5 that can catch a factor which is
computed cleanly, typed correctly, transformed correctly, and simply measures the
wrong thing. A sign error survives every other test in this package: the array is
float64, the availability lag is honoured, the distribution is plausible, the
correlation matrix looks reasonable, and the factor is pointing backwards.

The package is two halves that are deliberately kept apart:

:mod:`~backend.features.validation.expectations`
    What the literature says, per factor, with the citation attached — the sign,
    a plausible magnitude range as an annualized fraction, the portfolio the
    published number refers to, and the minimum sample below which no verdict is
    rendered. **Written before any return data existed**, because an expectation
    formed after the fact agrees with the data by construction. The table is
    hashed into every report's config hash, so an edit is visible as a different
    run rather than a quieter verdict.

:mod:`~backend.features.validation.premia`
    The harness: it measures the realized premium of a supplied return panel,
    compares sign and magnitude, reports a t-statistic, and flags any factor that
    contradicts its expectation. It never builds the data it validates, and an
    empty panel raises
    :class:`~backend.features.validation.errors.FactorReturnsUnavailableError`
    rather than reporting a premium of zero (I3).

**Status: the harness is complete; the measurement is blocked on B1.** There is
no price history in ``price_bar`` (P3.4) and no point-in-time fundamentals table
(P3.5), so six of the nine baseline factors refuse to compute and the other three
return ``NaN`` for every security. No factor return series can be built from this
repository today, and the harness says so with a named error instead of an empty
table.
:data:`~backend.features.validation.premia.WHAT_G5_STILL_NEEDS` enumerates what
the gate is waiting for.
"""

from __future__ import annotations

from backend.features.validation.errors import (
    DegenerateReturnSeriesError,
    DuplicateFactorSeriesError,
    ExpectationDeclarationError,
    FactorReturnsUnavailableError,
    InsufficientHistoryError,
    MalformedReturnSeriesError,
    PremiaValidationError,
    UnknownFactorExpectationError,
)
from backend.features.validation.expectations import (
    EXPECTATIONS,
    MAX_PLAUSIBLE_ANNUALIZED_PREMIUM,
    REFERENCE_CONSTRUCTION,
    CostBasis,
    FactorPremiumExpectation,
    PremiumSign,
    expectation_for,
    expectations_config,
    factors_without_expectations,
)
from backend.features.validation.premia import (
    DEFAULT_SIGNIFICANCE_T,
    MAX_PERIODS_PER_YEAR,
    MAX_PLAUSIBLE_PERIOD_RETURN,
    WHAT_G5_STILL_NEEDS,
    FactorPremiumCheck,
    FactorReturnPanel,
    FactorReturnSeries,
    PremiaValidationReport,
    PremiumVerdict,
    RealizedPremium,
    check_factor_premium,
    premia_validation_stamp,
    realized_premium,
    validate_premia,
)

__all__ = [
    "DEFAULT_SIGNIFICANCE_T",
    "EXPECTATIONS",
    "MAX_PERIODS_PER_YEAR",
    "MAX_PLAUSIBLE_ANNUALIZED_PREMIUM",
    "MAX_PLAUSIBLE_PERIOD_RETURN",
    "REFERENCE_CONSTRUCTION",
    "WHAT_G5_STILL_NEEDS",
    "CostBasis",
    "DegenerateReturnSeriesError",
    "DuplicateFactorSeriesError",
    "ExpectationDeclarationError",
    "FactorPremiumCheck",
    "FactorPremiumExpectation",
    "FactorReturnPanel",
    "FactorReturnSeries",
    "FactorReturnsUnavailableError",
    "InsufficientHistoryError",
    "MalformedReturnSeriesError",
    "PremiaValidationError",
    "PremiaValidationReport",
    "PremiumSign",
    "PremiumVerdict",
    "RealizedPremium",
    "UnknownFactorExpectationError",
    "check_factor_premium",
    "expectation_for",
    "expectations_config",
    "factors_without_expectations",
    "premia_validation_stamp",
    "realized_premium",
    "validate_premia",
]
