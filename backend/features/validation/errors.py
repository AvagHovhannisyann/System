"""Failure taxonomy for the factor-premia validation harness (P5.4).

Every refusal here exists to keep one sentence true: **an absent measurement is
never reported as a measured absence.** The harness compares a realized factor
premium against a published expectation, and the two ways that comparison can be
made dishonest are

1. running it on nothing and reporting the resulting zero as "no premium", and
2. running it on a sample too short to distinguish a broken factor from a bad
   decade, then reporting the verdict as though the sample had been long enough.

Both are indistinguishable, downstream, from an honest negative result: a table
row reading ``momentum_12_1  0.00%  t=0.00  NO_PREMIUM_DETECTED`` looks like
evidence and is not. So both raise — :class:`FactorReturnsUnavailableError` and
:class:`InsufficientHistoryError` — and the harness produces no report at all
rather than a report with a hole in it.

This is the same rule the factor library applies one layer down: six of the nine
baseline factors raise ``FundamentalsSourceUnavailableError`` rather than return
a plausible number while their connector is blocked on B1 (see
:mod:`backend.features.factors`). The harness is where that discipline has to be
repeated, because a validation result is precisely the artifact whose whole
purpose is to be believed.

Every error carries the numbers that produced it, so the message alone is enough
to diagnose the cause without re-running under a debugger.
"""

from __future__ import annotations

from backend.features.errors import FeatureError

__all__ = [
    "DegenerateReturnSeriesError",
    "DuplicateFactorSeriesError",
    "ExpectationDeclarationError",
    "FactorReturnsUnavailableError",
    "InsufficientHistoryError",
    "MalformedReturnSeriesError",
    "PremiaValidationError",
    "UnknownFactorExpectationError",
]


class PremiaValidationError(FeatureError):
    """Base class for every failure raised by :mod:`backend.features.validation`.

    A subclass of :class:`~backend.features.errors.FeatureError` so a caller
    sweeping the feature library can catch the whole package's failures with one
    ``except`` clause, and so that a premia-validation failure cannot be mistaken
    for an unrelated ``ValueError`` from numpy.

    Raised directly — rather than through a subclass — for an argument error
    that fits no narrower category, currently only a non-positive or non-finite
    significance threshold. Nothing in this package raises a bare ``ValueError``:
    the promise that every failure here is a ``PremiaValidationError`` is what
    lets a caller distinguish "the harness refused" from "numpy disagreed".
    """


class ExpectationDeclarationError(PremiaValidationError):
    """Raised when a published expectation cannot describe a checkable claim.

    Examples: a magnitude range whose endpoints straddle zero (which would make
    "the sign agrees" and "the magnitude is in range" contradict each other), a
    range wider than 100% per year (a units error — the range is a fraction, not
    a percent), an empty citation, or a non-positive minimum sample length.

    Raised at import time, because :data:`~backend.features.validation.
    expectations.EXPECTATIONS` is built at module scope: a malformed expectation
    cannot reach a comparison.
    """


class UnknownFactorExpectationError(PremiaValidationError):
    """Raised when a factor's realized premium has no written-down expectation.

    This is deliberately fatal rather than a skipped row. The value of the
    expectation table is that it was written **before** the data existed; an
    expectation invented at the moment a realized premium is in hand is not a
    test of the factor, it is a description of the sample. So a factor with no
    prior expectation cannot be validated at all, and the harness says so
    instead of quietly dropping it from a report that then reads as complete.

    Attributes:
        factor: the factor name that had no expectation.
        known: factor names that do have one, sorted.
    """

    def __init__(self, factor: str, known: tuple[str, ...]) -> None:
        """Build the error from the missing factor and the table's contents.

        Args:
            factor: the factor name that was looked up.
            known: factor names carrying an expectation, sorted, for the message.
        """
        self.factor = factor
        self.known = known
        rendered = ", ".join(known) if known else "<the expectation table is empty>"
        super().__init__(
            f"no published premium expectation is declared for factor {factor!r}, so "
            f"its realized premium cannot be validated. Expectations are written "
            f"from the literature before any data exists, precisely so the "
            f"comparison cannot be tuned to what the data turns out to show; adding "
            f"one now, with the realized number already in hand, would make the "
            f"check circular. Declare it in backend/features/validation/"
            f"expectations.py with its citation. Declared: {rendered}"
        )


class FactorReturnsUnavailableError(PremiaValidationError):
    """Raised when there are no factor returns to validate.

    The empty case is the one the harness exists to refuse. An empty panel does
    not mean the premia are zero, it means nothing was measured — and a harness
    that answered it with ``premium = 0.00, t = 0.00, NO_PREMIUM_DETECTED``
    would be manufacturing the strongest possible claim (a precise zero) out of
    the weakest possible evidence (none). Directive §9.2: a function that returns
    a plausible value without doing the work is worse than one that raises.

    Today this is the *only* reachable outcome of running the harness against the
    store, and that is a statement about B1 rather than about the factors: the
    price connector (P3.4) and the fundamentals connector (P3.5) both lack
    credentials, ``price_bar`` is empty, and six of the nine baseline factors
    refuse to compute at all. The harness is therefore complete and the
    measurement is blocked, which is a state the gate must be able to see.

    Attributes:
        detail: what was empty — the panel, or a named factor's series.
    """

    def __init__(self, detail: str) -> None:
        """Build the error from a description of what was empty.

        Args:
            detail: what held no observations, e.g. ``"the panel"`` or
                ``"series for 'momentum_12_1'"``.
        """
        self.detail = detail
        super().__init__(
            f"{detail} contains no return observations, so no factor premium can be "
            f"measured. An empty sample is not evidence of a zero premium; it is the "
            f"absence of evidence, and this harness will not render it as a number "
            f"(I3, directive §9.2). Today this is the expected outcome: the price and "
            f"fundamentals connectors are blocked on B1, so no factor return series "
            f"can be built from the store yet."
        )


class InsufficientHistoryError(PremiaValidationError):
    """Raised when a return series is too short to test a long-sample premium.

    Gate G5 asks whether *long-sample* premia reproduce, and the distinction
    matters more here than anywhere else in the feature library: every one of
    these factors has multi-year stretches over which its premium is zero or
    negative and the security prices were not lying. HML earned approximately
    nothing from 2007 to 2020; momentum lost roughly a third of its value in
    2009. A five-year window that shows no value premium is a fact about
    2007-2012, not about ``book_to_price``, and a harness that reported it as a
    factor failure would teach its operator to distrust a correct implementation.

    So the minimum sample is part of each published expectation, and a series
    below it raises rather than producing a verdict. The refusal is symmetric:
    a short sample cannot confirm a premium either.

    Attributes:
        factor: the factor whose series was too short.
        sample_years: the series length in years.
        required_years: the minimum the expectation declares.
        observations: the number of return observations supplied.
        periods_per_year: the series' declared periodicity.
    """

    def __init__(
        self,
        *,
        factor: str,
        sample_years: float,
        required_years: float,
        observations: int,
        periods_per_year: float,
    ) -> None:
        """Build the error from the series and the expectation it fell short of.

        Args:
            factor: the factor whose series was too short.
            sample_years: ``observations / periods_per_year``.
            required_years: the expectation's ``min_sample_years``.
            observations: number of return observations supplied.
            periods_per_year: the series' declared periodicity.
        """
        self.factor = factor
        self.sample_years = sample_years
        self.required_years = required_years
        self.observations = observations
        self.periods_per_year = periods_per_year
        super().__init__(
            f"factor {factor!r} was supplied {observations} return observations at "
            f"{periods_per_year} periods per year, which is {sample_years:.2f} years "
            f"of history; its published expectation requires at least "
            f"{required_years:.2f} years. Factor premia have multi-year drawdowns "
            f"during which the correct implementation shows nothing, so a short "
            f"window can neither confirm nor contradict a long-sample premium — and "
            f"a verdict from one would be a statement about the window wearing the "
            f"factor's name."
        )


class MalformedReturnSeriesError(PremiaValidationError):
    """Raised when a supplied return series cannot be a return series.

    Covers shape (not one-dimensional), content (``NaN`` or infinity — a missing
    period is not a return of zero and must not be silently averaged in),
    periodicity (non-positive, or above daily, which this platform does not
    trade), and magnitude. The magnitude bound is a **units guard**: returns here
    are fractions, so ``0.01`` is one percent, and a series of ``0.31`` monthly
    values is far more likely to be percent mistakenly passed as fraction than a
    31%-per-month factor. It cannot catch the reverse mistake — a percent series
    whose values are small — which is why every entry point restates the unit.

    Attributes:
        factor: the factor the series was supplied for.
        detail: what was wrong, in words.
    """

    def __init__(self, *, factor: str, detail: str) -> None:
        """Build the error from the factor name and the defect.

        Args:
            factor: the factor the series was supplied for.
            detail: what was wrong with the series.
        """
        self.factor = factor
        self.detail = detail
        super().__init__(
            f"the return series supplied for factor {factor!r} is malformed: "
            f"{detail}. Returns are simple per-period returns of the factor's "
            f"long-short portfolio, expressed as fractions (0.01 is one percent), "
            f"one value per rebalance period, finite, in chronological order."
        )


class DegenerateReturnSeriesError(PremiaValidationError):
    """Raised when a return series has zero dispersion, leaving ``t`` undefined.

    ``t = mean / (stdev / sqrt(n))`` divides by zero when every period's return
    is identical. Left alone that yields ``inf`` for a non-zero constant and
    ``nan`` for a constant zero — an infinitely significant premium, or a blank
    cell, from a series no real portfolio produces. Both render as something; the
    error renders as the truth.

    Attributes:
        factor: the factor whose series was constant.
        constant_return: the repeated per-period value (fraction).
        observations: number of observations in the series.
    """

    def __init__(self, *, factor: str, constant_return: float, observations: int) -> None:
        """Build the error from the constant series.

        Args:
            factor: the factor whose series was constant.
            constant_return: the repeated per-period return, as a fraction.
            observations: number of observations in the series.
        """
        self.factor = factor
        self.constant_return = constant_return
        self.observations = observations
        super().__init__(
            f"the return series supplied for factor {factor!r} is constant at "
            f"{constant_return!r} across all {observations} periods, so its standard "
            f"error is zero and its t-statistic is undefined. A constant long-short "
            f"return is not a portfolio's history; it is a placeholder, and this "
            f"harness will not report an infinite t-statistic for one."
        )


class DuplicateFactorSeriesError(PremiaValidationError):
    """Raised when one validation run covers the same factor twice.

    Which one wins would decide the verdict, and it would be decided by
    construction order — the kind of order dependence that makes a result
    unreproducible from a config hash (I2). There is also no defensible merge:
    two series for one factor are two different portfolio constructions, and
    averaging their premia produces a number that describes neither.

    Attributes:
        factor: the duplicated factor name.
    """

    def __init__(self, factor: str) -> None:
        """Build the error from the duplicated factor name.

        Args:
            factor: the factor name supplied more than once.
        """
        self.factor = factor
        super().__init__(
            f"factor {factor!r} appears more than once in a single validation run. "
            f"Two series are two portfolio constructions; picking one by position "
            f"would make the verdict depend on construction order, and averaging "
            f"them would produce a premium that describes neither. Supply one series "
            f"per factor, or validate the constructions in separate runs."
        )
