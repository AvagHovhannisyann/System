"""The factor-premia validation harness (P5.4) — gate G5's first clause.

G5 asks whether *known factor premia reproduce with the correct sign and a
plausible magnitude over long samples*. This module is the machinery that asks
that question of a realized return panel and answers it per factor, with a sign
comparison, a magnitude comparison and a t-statistic.

--------------------------------------------------------------------------
The expectations are written before the data, and that is the whole point
--------------------------------------------------------------------------

:mod:`backend.features.validation.expectations` declares, for each of the nine
baseline factors, the sign and magnitude range the literature reports, with the
citation attached. That table was written while **no return data existed**, and
it had to be: an expectation formed after a realized premium is in hand is not a
test of the factor, it is a description of the sample, and it agrees with the
data by construction.

Two mechanisms keep it that way rather than trusting a convention. The full
expectation table — ranges, citations, caveats, minimum sample lengths — is
hashed into every report's :class:`~backend.tracking.stamp.ReproducibilityStamp`
via :func:`~backend.features.validation.expectations.expectations_config`, so a
range widened after seeing a disappointing number produces a report with a
different ``config_hash``, visibly a different run rather than a continuation of
the old one. And a factor with no declared expectation cannot be validated at
all: :class:`~backend.features.validation.errors.UnknownFactorExpectationError`
refuses the comparison rather than accepting an expectation invented on the spot.

--------------------------------------------------------------------------
This module never manufactures the data it validates (I3)
--------------------------------------------------------------------------

There is no function here that generates, simulates, imputes, or fills a return.
The only way a premium is computed is from a :class:`FactorReturnPanel` the
caller supplies, and an empty one raises
:class:`~backend.features.validation.errors.FactorReturnsUnavailableError`
rather than reporting ``0.00%, t=0.00, NO_PREMIUM_DETECTED``. That row would be
the strongest possible claim — a precise zero — drawn from the weakest possible
evidence, and it is indistinguishable in a table from an honest negative result.

**Today that error is the only reachable outcome**, and it is a statement about
B1 rather than about the factors. The price connector (P3.4) and the
point-in-time fundamentals connector (P3.5) both lack vendor credentials;
``price_bar`` exists but is empty, so the three price factors return ``NaN`` for
every security; the six fundamentals factors raise outright. Nothing in this
repository can build a non-empty panel, so the harness is complete and the
measurement is blocked. What G5's first clause still needs from B1 is listed in
:data:`WHAT_G5_STILL_NEEDS`.

--------------------------------------------------------------------------
Units, stated once and repeated at every entry point
--------------------------------------------------------------------------

* A **return** is a simple (arithmetic) return of the factor's long-short
  portfolio over one rebalance period, as a **fraction**: ``0.01`` is one
  percent, never ``1``. One value per period, chronological, finite. A period
  with no observation is absent from the series, never a zero.
* A **premium** is annualized **arithmetically** — mean per-period return times
  ``periods_per_year`` — because that is how the sources quote their results
  ("0.31% per month"), and because geometric compounding would subtract a
  variance drag those figures do not contain. For a long-short equity factor the
  two differ by roughly 100-200 basis points a year, enough to move a comparison
  across a range boundary.
* A premium is expressed in the **factor's own sign convention**, the one its
  :class:`~backend.features.spec.FeatureSpec` declares. A high ``low_volatility``
  score is a calm stock, so its long leg is the calm stocks; a high ``accruals``
  score is a high-accrual firm, so its long leg is the one expected to *lose*
  and its premium is expected to be negative. The harness never re-signs a
  series to make a verdict come out positive.
* The **t-statistic** is ``mean / (stdev / sqrt(n))`` with ``ddof=1``, on the
  per-period series. It is periodicity-consistent: annualizing the mean and the
  standard error by the same convention leaves it unchanged, so the t reported
  here is the t of the annualized premium too.

--------------------------------------------------------------------------
Gross versus net, and why the harness insists you say which (I4)
--------------------------------------------------------------------------

Every published premium in the expectation table is **gross** of transaction
costs, because essentially all of them are. A realized series must therefore
declare its own :class:`~backend.features.validation.expectations.CostBasis`,
with no default, and the report carries the comparison basis onto every
rendering:

* A **gross** realized premium may be compared like-for-like — and may never be
  quoted as a performance result. Directive §9.6 forbids reporting a gross
  return; a factor-reproduction diagnostic is not a performance claim, but the
  number is exactly the one that flatters a factor, so every report saying so
  carries the disclosure rather than relying on a reader's memory.
* A **net** realized premium landing *below* a gross published range has not
  contradicted the literature. It has reproduced it and then paid for it. The
  harness flags the basis mismatch so the magnitude verdict is read correctly;
  the sign verdict is unaffected. The gap is largest for
  ``short_term_reversal``, whose gross premium is among the largest in the table
  and whose net premium at 100%-plus monthly turnover is approximately nothing.

--------------------------------------------------------------------------
Statistical assumptions, stated because they are not testable here
--------------------------------------------------------------------------

The t-statistic assumes the per-period returns are serially independent and
identically distributed. Non-overlapping monthly long-short returns are close
enough to that for the reproduction check this module performs; **overlapping**
windows are not, and a series built from them carries an inflated t that nothing
in this module can detect. No Newey-West or other autocorrelation correction is
applied, deliberately: applying one would require an assumption about the lag
structure that the caller is better placed to make, and a correction applied
silently is worse than one applied visibly upstream.

:data:`DEFAULT_SIGNIFICANCE_T` is 2.0, the convention these papers themselves
use. Harvey, Liu & Zhu (2016), "…and the Cross-Section of Expected Returns",
*Review of Financial Studies* 29(1), 5-68, argue for roughly 3.0 when *declaring
a new* factor, to account for the hundreds tested and unreported. That argument
does not apply here — nothing in this table is being discovered, and the
multiple-testing correction this platform owes lives in
:mod:`backend.backtest.dsr`, fed by ``TESTING_LEDGER.md``. Raising the threshold
would make the harness *less* likely to notice a factor that reproduces
correctly, which is the error direction that matters for a reproduction check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.features.validation.errors import (
    DegenerateReturnSeriesError,
    DuplicateFactorSeriesError,
    FactorReturnsUnavailableError,
    InsufficientHistoryError,
    MalformedReturnSeriesError,
    PremiaValidationError,
    UnknownFactorExpectationError,
)
from backend.features.validation.expectations import (
    EXPECTATIONS,
    CostBasis,
    FactorPremiumExpectation,
    expectations_config,
)
from backend.tracking.stamp import ReproducibilityStamp

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    import numpy.typing as npt

    from backend.features._stats import FloatArray

__all__ = [
    "DEFAULT_SIGNIFICANCE_T",
    "MAX_PERIODS_PER_YEAR",
    "MAX_PLAUSIBLE_PERIOD_RETURN",
    "WHAT_G5_STILL_NEEDS",
    "FactorPremiumCheck",
    "FactorReturnPanel",
    "FactorReturnSeries",
    "PremiaValidationReport",
    "PremiumVerdict",
    "RealizedPremium",
    "check_factor_premium",
    "premia_validation_stamp",
    "realized_premium",
    "validate_premia",
]

DEFAULT_SIGNIFICANCE_T: Final = 2.0
"""``|t|`` at or above which a premium is called distinguishable from zero.

Dimensionless. Two standard errors — the convention the source papers use. See
the module docstring for why this is not raised to Harvey, Liu & Zhu's 3.0.
"""

MAX_PLAUSIBLE_PERIOD_RETURN: Final = 10.0
"""Largest accepted absolute per-period return (fraction), a units guard.

A 1000% move in one rebalance period. A series of values like ``0.31`` meaning
"0.31 percent" is far likelier to be percent passed where a fraction was meant
than a factor earning 31% a month, and this bound catches the direction of that
mistake that matters — it cannot catch a percent series whose values happen to be
small, which is why every entry point restates the unit instead of relying on it.

It also makes overflow unreachable: with returns bounded by 10 and a panel of any
representable length, no mean, variance or standard error in this module can
leave the ``float64`` range, so there is no non-finite branch to guard (and, per
D-028, no unreachable guard pretending to be tested).
"""

MAX_PERIODS_PER_YEAR: Final = 366.0
"""Largest accepted periodicity (periods per year).

Daily. Directive §1.1 puts intraday strategies out of scope permanently, so a
periodicity above this is a units error — 252 written as 25200, or a per-period
count where a per-year one was meant — and it would silently divide the sample
length by 100 in the long-sample check.
"""

WHAT_G5_STILL_NEEDS: Final = (
    (
        "A price history in `price_bar` (P3.4, blocked on B1): the three price factors "
        "compute today and return NaN for every security because the table is empty."
    ),
    (
        "A point-in-time fundamentals table (P3.5, blocked on B1): the six "
        "fundamentals-derived factors raise FundamentalsSourceUnavailableError, so no "
        "cross-section of scores exists for them at any date."
    ),
    (
        "Point-in-time universe membership across the long sample (P4), so each "
        "rebalance date's cross-section is the investable set as it stood then rather "
        "than as it stands now — a survivorship-biased universe inflates every premium "
        "in the table and reproduces the literature for the wrong reason."
    ),
    (
        "A portfolio construction step turning factor scores into the long-short decile "
        "spread these expectations are calibrated for, and a decision on whether the "
        "check is run gross (like-for-like against the literature) or net (P9.3 costs "
        "applied), recorded per series as its CostBasis."
    ),
    (
        "Expectations for `size`, `short_interest` and Amihud illiquidity once those "
        "factors are declared with fixed registry names; the coverage check "
        "`factors_without_expectations` reports the gap and the gate cannot pass while "
        "it is non-empty."
    ),
)
"""What gate G5's first clause needs that this module cannot supply.

The harness is complete; the measurement is blocked. This tuple is the standing
answer to "why is there no premia table in ``PROGRESS.md``", so that the absence
is a recorded blocker rather than an oversight nobody wrote down.
"""


@dataclass(frozen=True, slots=True, eq=False)
class FactorReturnSeries:
    """One factor's realized long-short portfolio returns.

    Not compared by equality (``eq=False``): the payload is a numpy array, and a
    dataclass ``__eq__`` over one raises rather than answering. Compare the
    fields you mean.

    Attributes:
        factor: the factor's registry name, matching its
            :class:`~backend.features.spec.FeatureSpec` and its declared
            expectation.
        returns: simple per-period returns of the factor's long-short portfolio
            as **fractions** (``0.01`` is one percent), one per rebalance period,
            chronological, finite, at least two of them. Stored as a read-only
            ``float64`` copy, so a later mutation of the caller's array cannot
            change a premium already reported.
        periods_per_year: the series' periodicity — 12 for monthly, 252 for
            daily. Used to annualize the premium and to measure the sample
            length in years.
        cost_basis: whether these returns are gross or net of modelled costs
            (I4). Required, with no default: the published expectations are
            gross, and which basis a realized premium carries decides how its
            magnitude verdict may be read.
        construction: prose describing the portfolio these returns came from —
            breakpoints, weighting, rebalance frequency, universe. Compared by a
            human against
            :data:`~backend.features.validation.expectations.REFERENCE_CONSTRUCTION`;
            the harness prints both rather than pretending to machine-verify
            prose.

    Raises:
        FactorReturnsUnavailableError: if the series holds no observations.
        MalformedReturnSeriesError: if it is not a one-dimensional finite float
            series of at least two values, if the periodicity is out of range, or
            if a value exceeds :data:`MAX_PLAUSIBLE_PERIOD_RETURN`.
    """

    factor: str
    returns: FloatArray
    periods_per_year: float
    cost_basis: CostBasis
    construction: str

    def __post_init__(self) -> None:
        """Validate and freeze the series.

        Raises:
            FactorReturnsUnavailableError: if the series is empty.
            MalformedReturnSeriesError: if the series or its periodicity cannot
                describe a return history.
        """
        if not self.factor.strip():
            msg = "a return series must name the factor it belongs to; got an empty name"
            raise MalformedReturnSeriesError(factor=self.factor, detail=msg)
        if not self.construction.strip():
            detail = (
                "no construction was stated. A premium is only comparable to a "
                "published range if the portfolio behind it is comparable, and a "
                "decile spread and a tercile spread on the same factor differ by "
                "more than the range is wide"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        supplied: npt.ArrayLike = self.returns
        try:
            values = np.array(supplied, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            detail = f"the values are not coercible to float64 ({exc})"
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail) from exc
        if values.ndim != 1:
            detail = (
                f"the values have shape {values.shape}, not one dimension. A panel of "
                f"several factors is a FactorReturnPanel of one-dimensional series, "
                f"one per factor"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        if values.size == 0:
            raise FactorReturnsUnavailableError(f"the return series for factor {self.factor!r}")
        if values.size == 1:
            detail = (
                "a single observation has no standard error, so no premium computed "
                "from it can be distinguished from zero"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        if not bool(np.all(np.isfinite(values))):
            detail = (
                "it contains a NaN or an infinity. A period with no observation is "
                "absent from the series, never a zero and never a NaN averaged in as "
                "though it were data (I3)"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        extreme = float(np.max(np.abs(values)))
        if extreme > MAX_PLAUSIBLE_PERIOD_RETURN:
            detail = (
                f"its largest absolute value is {extreme!r}, above the "
                f"{MAX_PLAUSIBLE_PERIOD_RETURN} units guard. Returns are fractions "
                f"(0.01 is one percent); a value this large is almost always percent "
                f"passed where a fraction was meant"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        periodicity: object = self.periods_per_year
        if isinstance(periodicity, bool) or not isinstance(periodicity, int | float):
            detail = f"periods_per_year={periodicity!r} is not a number"
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        if not math.isfinite(self.periods_per_year) or self.periods_per_year <= 0.0:
            detail = (
                f"periods_per_year={self.periods_per_year!r} is not positive and "
                f"finite; it is 12 for a monthly series, 252 for a daily one"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        if self.periods_per_year > MAX_PERIODS_PER_YEAR:
            detail = (
                f"periods_per_year={self.periods_per_year!r} is above "
                f"{MAX_PERIODS_PER_YEAR} (daily). Intraday is out of scope (§1.1), so "
                f"this is a units error, and it would divide the measured sample "
                f"length by the same factor in the long-sample check"
            )
            raise MalformedReturnSeriesError(factor=self.factor, detail=detail)
        values.setflags(write=False)
        object.__setattr__(self, "returns", values)
        object.__setattr__(self, "periods_per_year", float(self.periods_per_year))

    @property
    def observations(self) -> int:
        """Return the number of per-period returns in the series (count)."""
        return int(self.returns.size)

    @property
    def sample_years(self) -> float:
        """Return the series length in years: ``observations / periods_per_year``."""
        return self.observations / self.periods_per_year


@dataclass(frozen=True, slots=True, eq=False)
class FactorReturnPanel:
    """The realized long-short return series for one or more factors.

    The harness's data boundary. Constructing one is the only way to get returns
    into :func:`validate_premia`, and an empty one is refused here rather than
    producing an empty report downstream — an empty panel is the state B1 leaves
    the platform in, and it must arrive as an error rather than as a table of
    zeros.

    Attributes:
        series: one :class:`FactorReturnSeries` per factor, at least one, with no
            factor appearing twice.

    Raises:
        FactorReturnsUnavailableError: if the panel holds no series.
        DuplicateFactorSeriesError: if two series name the same factor.
    """

    series: tuple[FactorReturnSeries, ...]

    def __post_init__(self) -> None:
        """Validate the panel.

        Raises:
            FactorReturnsUnavailableError: if the panel is empty.
            DuplicateFactorSeriesError: if a factor appears more than once.
        """
        entries = tuple(self.series)
        if not entries:
            raise FactorReturnsUnavailableError("the factor return panel")
        _require_unique_factors(entry.factor for entry in entries)
        object.__setattr__(self, "series", entries)

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factor names covered by this panel, in series order."""
        return tuple(entry.factor for entry in self.series)


def _require_unique_factors(factors: Iterable[str]) -> None:
    """Raise if a factor name appears more than once.

    Args:
        factors: the factor names to check, in order.

    Raises:
        DuplicateFactorSeriesError: naming the first repeated factor.
    """
    seen: set[str] = set()
    for factor in factors:
        if factor in seen:
            raise DuplicateFactorSeriesError(factor)
        seen.add(factor)


@dataclass(frozen=True, slots=True)
class RealizedPremium:
    """What a factor's return series actually earned, with its precision.

    Attributes:
        factor: the factor's registry name.
        mean_per_period: arithmetic mean of the per-period returns (fraction per
            period).
        annualized: ``mean_per_period * periods_per_year`` (fraction per year),
            arithmetic — see the module docstring on why not geometric.
        standard_error_per_period: ``stdev(ddof=1) / sqrt(n)`` (fraction per
            period).
        t_statistic: ``mean_per_period / standard_error_per_period``,
            dimensionless. Unchanged by annualizing both numerator and
            denominator, so it is equally the t of :attr:`annualized`. Assumes
            serially independent returns; see the module docstring.
        observations: number of per-period returns (count).
        periods_per_year: the series' periodicity.
        sample_years: ``observations / periods_per_year``.
        cost_basis: gross or net, carried from the series (I4).
        construction: the series' stated portfolio construction.
    """

    factor: str
    mean_per_period: float
    annualized: float
    standard_error_per_period: float
    t_statistic: float
    observations: int
    periods_per_year: float
    sample_years: float
    cost_basis: CostBasis
    construction: str


def realized_premium(series: FactorReturnSeries) -> RealizedPremium:
    """Measure the premium a factor's realized return series earned.

    No expectation is consulted here: this function reports what the series did,
    and :func:`check_factor_premium` decides what that means. Keeping the two
    apart is what makes the measurement checkable against hand arithmetic
    independently of the verdict logic.

    Args:
        series: the factor's realized long-short returns. Simple per-period
            returns as fractions, already **net or gross as the series declares**
            — nothing here can verify which, which is why the basis is a required
            field that travels onto the result.

    Returns:
        A :class:`RealizedPremium` carrying the mean, its arithmetic
        annualization, the standard error, the t-statistic and the sample size.

    Raises:
        DegenerateReturnSeriesError: if every period's return is identical, which
            leaves the standard error at zero and the t-statistic undefined.
            Reported as an error rather than as ``inf`` or ``nan``.

    Example:
        >>> import numpy as np
        >>> from backend.features.validation.expectations import CostBasis
        >>> alternating = np.array([0.03, -0.01] * 60, dtype=np.float64)
        >>> series = FactorReturnSeries(
        ...     factor="momentum_12_1",
        ...     returns=alternating,
        ...     periods_per_year=12.0,
        ...     cost_basis=CostBasis.GROSS_OF_COSTS,
        ...     construction="hand-built example, not a measurement",
        ... )
        >>> measured = realized_premium(series)
        >>> round(measured.mean_per_period, 10), round(measured.annualized, 10)
        (0.01, 0.12)
    """
    values = series.returns
    count = series.observations
    mean = float(np.mean(values))
    deviation = float(np.std(values, ddof=1))
    if deviation == 0.0:
        raise DegenerateReturnSeriesError(
            factor=series.factor, constant_return=mean, observations=count
        )
    # No overflow branch: MAX_PLAUSIBLE_PERIOD_RETURN bounds every value by 10,
    # so the mean, the variance and the standard error are all representable for
    # any panel length float64 can index. An unreachable guard here would be
    # untested code claiming to be a safeguard (D-028).
    standard_error = deviation / math.sqrt(count)
    return RealizedPremium(
        factor=series.factor,
        mean_per_period=mean,
        annualized=mean * series.periods_per_year,
        standard_error_per_period=standard_error,
        t_statistic=mean / standard_error,
        observations=count,
        periods_per_year=series.periods_per_year,
        sample_years=series.sample_years,
        cost_basis=series.cost_basis,
        construction=series.construction,
    )


class PremiumVerdict(StrEnum):
    """The outcome of comparing one realized premium against its expectation.

    Four outcomes, and the order in which they are decided matters:
    **significance is settled first**. A premium the sample cannot distinguish
    from zero is not evidence in either direction, so it is never called a
    contradiction — reading noise as a refutation of the literature is the same
    error as reading noise as a confirmation of it, and this harness exists to
    refuse both.
    """

    REPRODUCES = "REPRODUCES"
    """Significant, correctly signed, and inside the published magnitude range."""

    SIGN_CONTRADICTS = "SIGN_CONTRADICTS"
    """Significant and pointing the wrong way. Flagged.

    The finding this harness is built to catch. A significant premium of the
    wrong sign is usually a defect in the factor — an un-negated directional
    factor, a reversed long-short convention, a misaligned forward return — and
    it is far more informative than a magnitude miss, because the literature's
    signs are the part that replicates most reliably across samples.
    """

    MAGNITUDE_IMPLAUSIBLE = "MAGNITUDE_IMPLAUSIBLE"
    """Significant and correctly signed, but outside the published range. Flagged.

    Both directions are flagged, and the *high* side is the more alarming one: a
    premium several times the published figure is the signature of a lookahead,
    of a survivorship-biased universe, or of a microcap-loaded equal-weighted
    construction — never of a better implementation of a documented effect.
    """

    NO_PREMIUM_DETECTED = "NO_PREMIUM_DETECTED"
    """``|t|`` below the threshold: the sample does not distinguish it from zero.

    Not a flag and not a pass. It is a statement about the evidence, and it does
    not satisfy G5's first clause — a factor whose premium cannot be measured has
    not reproduced.
    """


@dataclass(frozen=True, slots=True)
class FactorPremiumCheck:
    """One factor's realized premium set against its published expectation.

    Every judgement is a property derived from the three stored fields, so a
    check cannot be constructed carrying a verdict its own numbers contradict.

    Attributes:
        expectation: the published claim, declared before any data existed.
        realized: what the supplied series actually earned.
        significance_threshold: the ``|t|`` at or above which the premium is
            called distinguishable from zero.

    Raises:
        PremiaValidationError: if the factors do not match, or the threshold is
            not positive and finite.
    """

    expectation: FactorPremiumExpectation
    realized: RealizedPremium
    significance_threshold: float

    def __post_init__(self) -> None:
        """Validate that the two sides describe the same factor.

        Raises:
            PremiaValidationError: if the expectation and the realized premium
                name different factors, or the threshold is unusable.
        """
        if self.expectation.factor != self.realized.factor:
            msg = (
                f"cannot check the realized premium of {self.realized.factor!r} "
                f"against the published expectation for {self.expectation.factor!r}; "
                f"a comparison across two factors is not a weaker check, it is a "
                f"different one wearing the wrong name"
            )
            raise PremiaValidationError(msg)
        _require_usable_threshold(self.significance_threshold)

    @property
    def factor(self) -> str:
        """Return the factor's registry name."""
        return self.expectation.factor

    @property
    def is_significant(self) -> bool:
        """Return whether ``|t|`` reaches :attr:`significance_threshold`."""
        return abs(self.realized.t_statistic) >= self.significance_threshold

    @property
    def sign_agrees(self) -> bool:
        """Return whether the realized premium points the way the literature says."""
        return self.expectation.sign_agrees(self.realized.annualized)

    @property
    def magnitude_within_range(self) -> bool:
        """Return whether the realized premium falls inside the published range.

        Reported as a fact regardless of significance; the verdict is what
        decides whether it may be read as a conclusion.
        """
        return self.expectation.contains(self.realized.annualized)

    @property
    def basis_matches_published(self) -> bool:
        """Return whether realized and published premia share a cost basis (I4).

        When they do not — a net realized premium against a gross published range
        — a magnitude verdict of
        :attr:`~PremiumVerdict.MAGNITUDE_IMPLAUSIBLE` on the low side is the
        expected outcome rather than a defect, and the report says so.
        """
        return self.realized.cost_basis is self.expectation.published_cost_basis

    @property
    def verdict(self) -> PremiumVerdict:
        """Return the outcome of the comparison.

        Significance is decided first: a premium indistinguishable from zero is
        :attr:`~PremiumVerdict.NO_PREMIUM_DETECTED` whatever its sign, because
        calling an insignificant negative number a contradiction of the
        literature over-reads the sample in exactly the direction this harness
        exists to prevent.
        """
        if not self.is_significant:
            return PremiumVerdict.NO_PREMIUM_DETECTED
        if not self.sign_agrees:
            return PremiumVerdict.SIGN_CONTRADICTS
        if not self.magnitude_within_range:
            return PremiumVerdict.MAGNITUDE_IMPLAUSIBLE
        return PremiumVerdict.REPRODUCES

    @property
    def flagged(self) -> bool:
        """Return whether this check contradicts the published expectation.

        ``True`` for :attr:`~PremiumVerdict.SIGN_CONTRADICTS` and
        :attr:`~PremiumVerdict.MAGNITUDE_IMPLAUSIBLE`. A flagged factor is never
        silently reported: :attr:`PremiaValidationReport.flagged` collects them
        and :attr:`PremiaValidationReport.reproduces` is ``False`` while any
        exists.
        """
        return self.verdict in (
            PremiumVerdict.SIGN_CONTRADICTS,
            PremiumVerdict.MAGNITUDE_IMPLAUSIBLE,
        )

    def to_dict(self) -> dict[str, object]:
        """Render the check for a report, a log line or the dashboard.

        Returns:
            A JSON-serializable mapping carrying the numbers **and** the
            expectation they were judged against, so a row cannot be read without
            the claim it tested. Premia are annualized fractions.
        """
        return {
            "factor": self.factor,
            "verdict": str(self.verdict),
            "flagged": self.flagged,
            "realized_annualized_premium": self.realized.annualized,
            "realized_mean_per_period": self.realized.mean_per_period,
            "standard_error_per_period": self.realized.standard_error_per_period,
            "t_statistic": self.realized.t_statistic,
            "significance_threshold": self.significance_threshold,
            "is_significant": self.is_significant,
            "sign_agrees": self.sign_agrees,
            "magnitude_within_range": self.magnitude_within_range,
            "expected_sign": str(self.expectation.sign),
            "expected_range": [self.expectation.annualized_low, self.expectation.annualized_high],
            "observations": self.realized.observations,
            "periods_per_year": self.realized.periods_per_year,
            "sample_years": self.realized.sample_years,
            "realized_cost_basis": str(self.realized.cost_basis),
            "published_cost_basis": str(self.expectation.published_cost_basis),
            "basis_matches_published": self.basis_matches_published,
            "realized_construction": self.realized.construction,
            "published_construction": self.expectation.construction,
            "source": self.expectation.source,
        }


def _require_usable_threshold(threshold: float) -> None:
    """Raise if a significance threshold cannot separate signal from noise.

    Args:
        threshold: the ``|t|`` cutoff.

    Raises:
        PremiaValidationError: if it is not positive and finite. Zero would call
            every premium significant, including one measured at ``t = 0.001``,
            and the harness would then report sign agreement from noise.
    """
    supplied: object = threshold
    if isinstance(supplied, bool) or not isinstance(supplied, int | float):
        msg = f"significance_threshold={supplied!r} is not a number"
        raise PremiaValidationError(msg)
    if not math.isfinite(threshold) or threshold <= 0.0:
        msg = (
            f"significance_threshold={threshold!r} must be positive and finite. A "
            f"threshold of zero calls every premium significant, including one "
            f"measured at t = 0.001, and the harness would then report sign "
            f"agreement drawn entirely from noise."
        )
        raise PremiaValidationError(msg)


def check_factor_premium(
    series: FactorReturnSeries,
    expectation: FactorPremiumExpectation,
    *,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_T,
) -> FactorPremiumCheck:
    """Measure one factor's premium and compare it against its expectation.

    Args:
        series: the factor's realized long-short returns, as fractions per
            period, on the cost basis the series declares.
        expectation: the published claim for the same factor.
        significance_threshold: ``|t|`` at or above which the premium is called
            distinguishable from zero. Defaults to
            :data:`DEFAULT_SIGNIFICANCE_T`.

    Returns:
        A :class:`FactorPremiumCheck` carrying the measurement, the expectation
        and the derived verdict.

    Raises:
        InsufficientHistoryError: if the series is shorter than the expectation's
            ``min_sample_years``. Refused rather than reported: every factor here
            has multi-year stretches over which its premium is absent, so a short
            window can neither confirm nor contradict a long-sample claim.
        DegenerateReturnSeriesError: if the series is constant.
        PremiaValidationError: if the series and the expectation name different
            factors, or the threshold is unusable.
    """
    _require_usable_threshold(significance_threshold)
    if series.sample_years < expectation.min_sample_years:
        raise InsufficientHistoryError(
            factor=series.factor,
            sample_years=series.sample_years,
            required_years=expectation.min_sample_years,
            observations=series.observations,
            periods_per_year=series.periods_per_year,
        )
    return FactorPremiumCheck(
        expectation=expectation,
        realized=realized_premium(series),
        significance_threshold=significance_threshold,
    )


@dataclass(frozen=True, slots=True)
class PremiaValidationReport:
    """Every factor's verdict, plus the stamp that makes the run regenerable.

    Attributes:
        checks: one :class:`FactorPremiumCheck` per validated factor, at least
            one, no factor twice.
        expectations: the table the checks were judged against — carried so the
            report can name the expectations that were *not* exercised, which is
            what stops a partial run from reading as a complete one.
        stamp: the I2 reproducibility stamp. Build it with
            :func:`premia_validation_stamp` so the expectation table is inside
            the config hash.

    Raises:
        FactorReturnsUnavailableError: if the report holds no checks.
        DuplicateFactorSeriesError: if a factor is checked twice.
    """

    checks: tuple[FactorPremiumCheck, ...]
    expectations: Mapping[str, FactorPremiumExpectation]
    stamp: ReproducibilityStamp

    def __post_init__(self) -> None:
        """Validate the report's contents.

        Raises:
            FactorReturnsUnavailableError: if there are no checks.
            DuplicateFactorSeriesError: if a factor appears twice.
        """
        checks = tuple(self.checks)
        if not checks:
            raise FactorReturnsUnavailableError("the validation report")
        _require_unique_factors(check.factor for check in checks)
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "expectations", MappingProxyType(dict(self.expectations)))

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factor names checked, in check order."""
        return tuple(check.factor for check in self.checks)

    @property
    def flagged(self) -> tuple[FactorPremiumCheck, ...]:
        """Return the checks that contradict their published expectation.

        Sign contradictions and implausible magnitudes. Never empty when
        :attr:`reproduces` is ``False`` for a contradiction reason — the report
        does not have a way to fail quietly.
        """
        return tuple(check for check in self.checks if check.flagged)

    @property
    def undetermined(self) -> tuple[FactorPremiumCheck, ...]:
        """Return the checks whose premium is indistinguishable from zero."""
        return tuple(
            check for check in self.checks if check.verdict is PremiumVerdict.NO_PREMIUM_DETECTED
        )

    @property
    def unchecked_factors(self) -> tuple[str, ...]:
        """Return the expectations no series exercised, sorted.

        A report covering three of nine factors is not three-ninths of a gate
        pass; it is a gate that has not been run. This is what makes that
        visible.
        """
        checked = set(self.factors)
        return tuple(sorted(factor for factor in self.expectations if factor not in checked))

    @property
    def basis_mismatches(self) -> tuple[FactorPremiumCheck, ...]:
        """Return the checks comparing across cost bases (I4)."""
        return tuple(check for check in self.checks if not check.basis_matches_published)

    @property
    def reproduces(self) -> bool:
        """Return whether G5's first clause is satisfied by this report.

        ``True`` only when **every** declared expectation was exercised and every
        check reproduces. Two ways this is deliberately strict:

        * an unexercised expectation fails it, because a factor nobody measured
          has not reproduced, and a report listing only the factors that happened
          to have data reads as complete when it is not;
        * :attr:`~PremiumVerdict.NO_PREMIUM_DETECTED` fails it, because "we could
          not tell" is not "it reproduced".
        """
        return not self.unchecked_factors and all(
            check.verdict is PremiumVerdict.REPRODUCES for check in self.checks
        )

    @property
    def disclosures(self) -> tuple[str, ...]:
        """Return the statements that must accompany these numbers anywhere.

        Assembled from the report's own contents rather than written by the
        caller, so a rendering cannot drop the one that applies to it. Covers the
        cost basis (I4), the pre-registration of the expectations, the
        t-statistic's independence assumption, incomplete coverage, and a stamp
        taken from a dirty working tree.
        """
        lines = [
            (
                "Premia are long-short factor returns in each factor's own sign "
                "convention, annualized arithmetically as a fraction of notional "
                "(0.04 is 4% a year)."
            ),
            (
                "Expected signs and ranges were declared from the cited literature "
                "before any return data existed and are hashed into this report's "
                "config_hash; a range edited after the fact produces a different hash, "
                "not a quieter verdict."
            ),
            (
                "t-statistics assume serially independent per-period returns. "
                "Overlapping windows inflate them and no autocorrelation correction is "
                "applied."
            ),
            (
                "Reproducing a historical premium is a test of this implementation, not "
                "a forecast: several of these premia have decayed since publication."
            ),
        ]
        if any(check.realized.cost_basis is CostBasis.GROSS_OF_COSTS for check in self.checks):
            lines.append(
                "Some premia here are GROSS of transaction costs, which is the basis the "
                "published expectations use. A gross long-short premium is a "
                "factor-reproduction diagnostic only and may never be quoted as a "
                "performance result (directive §9.6, I4)."
            )
        if self.basis_mismatches:
            mismatched = ", ".join(check.factor for check in self.basis_mismatches)
            lines.append(
                f"Cost-basis mismatch for: {mismatched}. A NET realized premium compared "
                f"against a GROSS published range is expected to land below it; that is "
                f"the cost of trading the factor, not a defect. The sign verdict is "
                f"unaffected by the basis."
            )
        if self.unchecked_factors:
            unchecked = ", ".join(self.unchecked_factors)
            lines.append(
                f"INCOMPLETE: {len(self.unchecked_factors)} of {len(self.expectations)} "
                f"declared expectations were not exercised ({unchecked}). Gate G5's first "
                f"clause is not satisfied by a partial panel."
            )
        if self.stamp.git_dirty:
            lines.append(
                "Produced from a dirty working tree: this result is not regenerable from "
                f"commit {self.stamp.git_commit} alone (I2)."
            )
        return tuple(lines)

    def to_dict(self) -> dict[str, object]:
        """Render the whole report for a log, an artifact or the dashboard.

        Returns:
            A JSON-serializable mapping. The disclosures and the stamp are part
            of it, not an optional decoration a consumer may leave out.
        """
        return {
            "reproduces": self.reproduces,
            "checks": [check.to_dict() for check in self.checks],
            "flagged_factors": [check.factor for check in self.flagged],
            "undetermined_factors": [check.factor for check in self.undetermined],
            "unchecked_factors": list(self.unchecked_factors),
            "disclosures": list(self.disclosures),
            "stamp": self.stamp.as_tags(),
        }


def validate_premia(
    panel: FactorReturnPanel,
    *,
    stamp: ReproducibilityStamp,
    expectations: Mapping[str, FactorPremiumExpectation] = EXPECTATIONS,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_T,
) -> PremiaValidationReport:
    """Validate every factor in ``panel`` against its published expectation.

    The harness's entry point. It measures each factor's realized premium,
    compares sign and magnitude against the pre-registered claim, and returns one
    report carrying the verdicts, the coverage gap and the reproducibility stamp.

    Args:
        panel: the realized long-short return series, one per factor. Supplied by
            the caller and never constructed here — nothing in this module
            generates, simulates or imputes a return (I3). An empty panel cannot
            be constructed, so "no data" arrives as
            :class:`~backend.features.validation.errors.FactorReturnsUnavailableError`
            rather than as a report of zeros.
        stamp: the I2 reproducibility stamp for this run. Build it with
            :func:`premia_validation_stamp` so the expectation table is inside
            the config hash.
        expectations: the table to judge against. Defaults to
            :data:`~backend.features.validation.expectations.EXPECTATIONS`; pass
            a different one only for a test, and note that doing so without
            rebuilding the stamp makes the report's config hash a lie.
        significance_threshold: ``|t|`` at or above which a premium is called
            distinguishable from zero. Defaults to
            :data:`DEFAULT_SIGNIFICANCE_T`.

    Returns:
        A :class:`PremiaValidationReport`. Its :attr:`~PremiaValidationReport.
        reproduces` property is the answer to G5's first clause, and it is
        ``False`` unless every declared expectation was exercised.

    Raises:
        UnknownFactorExpectationError: if the panel carries a factor with no
            declared expectation. Not skipped — a factor validated against an
            expectation invented after the fact is not validated.
        InsufficientHistoryError: if any series is shorter than its expectation's
            minimum sample.
        DegenerateReturnSeriesError: if any series is constant.
        PremiaValidationError: if the threshold is unusable.
    """
    _require_usable_threshold(significance_threshold)
    table = MappingProxyType(dict(expectations))
    checks = tuple(
        check_factor_premium(
            series,
            _expectation_from(table, series.factor),
            significance_threshold=significance_threshold,
        )
        for series in panel.series
    )
    return PremiaValidationReport(checks=checks, expectations=table, stamp=stamp)


def _expectation_from(
    expectations: Mapping[str, FactorPremiumExpectation], factor: str
) -> FactorPremiumExpectation:
    """Look ``factor`` up in ``expectations`` or refuse.

    Args:
        expectations: the table in use for this run.
        factor: the factor name to find.

    Returns:
        The declared expectation.

    Raises:
        UnknownFactorExpectationError: if the factor has none.
    """
    try:
        return expectations[factor]
    except KeyError as exc:
        raise UnknownFactorExpectationError(factor, tuple(sorted(expectations))) from exc


def premia_validation_stamp(
    *,
    data_version: str,
    seed: int = 0,
    expectations: Mapping[str, FactorPremiumExpectation] = EXPECTATIONS,
    significance_threshold: float = DEFAULT_SIGNIFICANCE_T,
    extra_config: Mapping[str, object] | None = None,
    repo_root: Path | None = None,
) -> ReproducibilityStamp:
    """Build the I2 stamp for a premia-validation run, expectations included.

    The config this hashes is the *whole* expectation table plus the significance
    threshold, so two reports carrying the same ``config_hash`` were judged
    against the same claims at the same standard. That is what makes the
    pre-registration enforceable rather than merely stated: an edit to a range, a
    citation or a caveat changes the hash of every report produced afterwards.

    Args:
        data_version: identifier of the data snapshot the returns came from.
            Obtain it from
            :func:`backend.tracking.data_version.resolve_data_version` rather
            than typing one in.
        seed: recorded verbatim. Defaults to 0 because this harness has no
            stochastic component — no bootstrap, no sampling, no shuffling — and
            I2 requires all four components to be present rather than three.
        expectations: the table the run will use. Defaults to
            :data:`~backend.features.validation.expectations.EXPECTATIONS`.
        significance_threshold: the threshold the run will use.
        extra_config: anything else that changes the result and must therefore
            change the hash — the universe criteria hash, the portfolio
            construction parameters, the cost model parameters. Merged at the top
            level under its own keys; a key colliding with one this function sets
            is refused.
        repo_root: repository to read git state from. Defaults to the platform
            root.

    Returns:
        A fully populated :class:`~backend.tracking.stamp.ReproducibilityStamp`.

    Raises:
        PremiaValidationError: if ``extra_config`` collides with a reserved key,
            or the threshold is unusable.
        ConfigHashError: if the config cannot be canonicalised.
        GitStateUnavailableError: if git state cannot be read.
    """
    _require_usable_threshold(significance_threshold)
    config: dict[str, object] = {
        "harness": "backend.features.validation.premia",
        "expectations": expectations_config(expectations),
        "significance_threshold": significance_threshold,
    }
    for key, value in (extra_config or {}).items():
        if key in config:
            msg = (
                f"extra_config key {key!r} collides with a key this harness sets "
                f"itself; overwriting it would let a run change what it claims to "
                f"have been judged against while keeping the same hash shape"
            )
            raise PremiaValidationError(msg)
        config[key] = value
    return ReproducibilityStamp.create(
        config=config, seed=seed, data_version=data_version, repo_root=repo_root
    )
