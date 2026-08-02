"""Published premium expectations, one per baseline factor (P5.4).

This module is a **prior**, and it is written down before the data exists on
purpose. Gate G5's first clause asks whether known factor premia reproduce with
the correct sign and a plausible magnitude, and the only way that question has
an answer is if "correct" and "plausible" were fixed in advance. An expectation
written after a realized premium is in hand is not a test of the factor; it is a
description of the sample, and it will agree with the data every time.

So each entry states, from the literature and with its citation:

* the **sign** the premium should carry, in the units the factor library's own
  declaration uses (see the sign convention in
  :mod:`backend.features.factors` — a factor named after a direction carries
  that direction, a factor named after a measured quantity does not, which is
  why ``accruals`` and ``asset_growth`` expect a NEGATIVE premium);
* a **plausible magnitude range**, as a signed interval of annualized fraction;
* the **published estimate** the range is centered on, quoted in the units the
  source itself uses, so the arithmetic that produced the range is checkable;
* the **portfolio construction** the published number refers to, because a
  decile spread and a tercile spread on the same factor differ by a factor of
  two or more and comparing across them is a units error in all but name;
* the **sample** the published number was measured over;
* the **minimum sample length** below which the harness refuses to render a
  verdict at all.

Two properties make this table hard to bend after the fact. Its contents are
hashed into the reproducibility stamp of every report
(:func:`expectations_config` feeds :func:`~backend.tracking.stamp.
canonical_config_hash`), so an edit changes the config hash of every result
produced afterwards and a tuned comparison cannot masquerade as the original
one. And :mod:`backend.tests.features.validation.test_expectations` asserts the
sign of every entry against the *factor's own declaration*, so the two
statements of a factor's direction cannot drift apart silently.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**Magnitudes are annualized simple returns, as fractions.** ``0.04`` is four
percent per year, never ``4``. Annualization is **arithmetic** — the mean
per-period return times the number of periods per year — because that is how
every source below quotes its result ("0.31% per month"), and because the
geometric alternative would subtract a variance drag the published figures do
not contain. The two differ by roughly ``sigma ** 2 / 2``, which for a
long-short equity factor is 100-200 basis points a year: enough to move a
comparison across a range boundary, which is why the convention is fixed here
rather than left to the caller.

**Ranges are signed intervals, not magnitudes with a separate sign.** The entry
for ``accruals`` is ``(-0.12, -0.02)``, not ``(0.02, 0.12)`` plus a NEGATIVE
flag. The two encodings are equivalent until someone compares a realized ``+8%``
against the second one, gets "magnitude in range", and reports agreement with a
premium of the wrong sign. Construction enforces that both endpoints carry the
declared sign.

**Every published estimate below is GROSS of transaction costs**
(:data:`~backend.features.validation.expectations.CostBasis.GROSS_OF_COSTS`),
because essentially all of them are: the academic long-short portfolio is formed
at closing prices and rebalanced without a spread, a commission, or a borrow
fee. This is recorded per entry rather than assumed, because it decides how a
comparison may be read. A realized premium measured **net** of modelled costs
that lands *below* a gross published range has not contradicted the literature —
it has reproduced it and then paid for it. The harness carries that caveat onto
any report where the two bases differ (I4), and it is at its largest for
``short_term_reversal``, whose gross premium is among the biggest in this table
and whose net premium at 100%-plus monthly turnover is approximately nothing.

**Ranges are deliberately wide.** They are a bug detector, not an estimate. The
job is to catch a factor computed with the wrong sign, a lookahead that turns a
4% premium into a 40% one, or a construction that measured something else
entirely — not to adjudicate whether the value premium is 3.4% or 4.6%. A range
narrow enough to reject a correct implementation in a different decade is worse
than useless, because the operator learns to override it.

--------------------------------------------------------------------------
What these numbers are not
--------------------------------------------------------------------------

They are not forecasts, and reproducing one is not evidence that the factor will
earn it. Several of these premia have decayed sharply since publication —
notably accruals (Green, Hand & Soliman 2011) and, over 2007-2020, the value
premium — and post-publication decay is the norm rather than the exception
(McLean & Pontiff 2016, *Journal of Finance* 71(1), 5-32). The gate this table
serves asks whether the *implementation* reproduces a documented historical
fact over a long sample. That is a test of the code, not of the future.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from backend.features.validation.errors import (
    ExpectationDeclarationError,
    UnknownFactorExpectationError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "EXPECTATIONS",
    "MAX_PLAUSIBLE_ANNUALIZED_PREMIUM",
    "REFERENCE_CONSTRUCTION",
    "CostBasis",
    "FactorPremiumExpectation",
    "PremiumSign",
    "expectation_for",
    "expectations_config",
    "factors_without_expectations",
]

MAX_PLAUSIBLE_ANNUALIZED_PREMIUM: Final = 1.0
"""Largest declarable magnitude for a range endpoint (annualized fraction).

100% per year. No long-sample published equity factor premium approaches it, so
an endpoint above this bound is a units error — a percent written where a
fraction was meant — and it fails at construction rather than silently widening
a range until everything passes.
"""

REFERENCE_CONSTRUCTION: Final = (
    "decile long-short spread, value-weighted within each leg, rebalanced monthly, "
    "US common equity excluding microcaps, gross of transaction costs"
)
"""The portfolio construction the magnitude ranges below are calibrated for.

A realized premium is comparable to a range only if it was measured on a
comparably constructed portfolio. The two differences that matter most are
**breakpoint width** (a decile spread is materially larger than a tercile spread
on the same factor, and quoting one against the other's range is a units error
wearing a different name) and **weighting** (equal weighting loads on microcaps,
where most of these premia are two to three times larger and least tradeable —
Cooper, Gulen & Schill report roughly 20% per year equal-weighted against 8%
value-weighted for the same sort).

Each expectation records the construction its *own* source used, which is not
always this one; where they differ the range was widened rather than rescaled,
because rescaling across constructions is a guess. A realized series states its
own construction, and the harness prints both, so a reviewer can see whether the
magnitude comparison was like-for-like. The harness does not attempt to
machine-verify prose.
"""


class PremiumSign(StrEnum):
    """The sign a factor's long-short premium is expected to carry.

    Stated in the **factor's own units**, as declared in
    :mod:`backend.features.factors`. ``low_volatility`` is the negative of
    volatility, so its expected premium is POSITIVE; ``accruals`` is the accruals
    measure itself, so its expected premium is NEGATIVE. Renaming the second kind
    to make every sign positive would put a strategy's name on a measurement.
    """

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"

    @property
    def direction(self) -> int:
        """Return ``+1`` for POSITIVE and ``-1`` for NEGATIVE (dimensionless)."""
        return 1 if self is PremiumSign.POSITIVE else -1


class CostBasis(StrEnum):
    """Whether a premium is quoted before or after trading costs (I4).

    Required, with no default, on both a published expectation and a realized
    return series. The directive's §9.6 — never report a gross return — is about
    performance claims, and a factor-reproduction diagnostic is not one; but a
    gross long-short premium *is* exactly the number that flatters a factor, so
    the basis travels with the number everywhere rather than living in a
    convention someone has to remember.
    """

    GROSS_OF_COSTS = "gross_of_costs"
    """Before spread, commission, market impact and borrow. What papers report."""

    NET_OF_MODELLED_COSTS = "net_of_modelled_costs"
    """After the P9.3 cost model. The only basis a performance claim may use."""


@dataclass(frozen=True, slots=True)
class FactorPremiumExpectation:
    """One factor's published premium: sign, plausible range, and provenance.

    Frozen and hashable so it can be embedded in a config hash (I2) and cannot be
    edited after a comparison has been made against it.

    Attributes:
        factor: the factor's registry name, matching its
            :class:`~backend.features.spec.FeatureSpec`.
        sign: the expected premium's direction, in the factor's own units.
        annualized_low: lower endpoint of the plausible range (annualized simple
            return, fraction). Must carry ``sign`` and be non-zero.
        annualized_high: upper endpoint, strictly greater than ``annualized_low``
            and likewise signed. For a NEGATIVE expectation the interval is e.g.
            ``(-0.12, -0.02)``: "low" is the more negative end, so the ordering
            is arithmetic throughout and a range check is a plain
            ``low <= x <= high``.
        source: full citation of the paper the expectation is taken from.
        published_estimate: what that source actually reports, in its own units,
            plus the arithmetic converting it to the annualized fraction used
            here. This is what makes the range auditable rather than asserted.
        construction: the portfolio the published number refers to. Compare with
            :data:`REFERENCE_CONSTRUCTION`.
        sample: the market and period the published number was measured over.
        published_cost_basis: whether the published number is gross or net.
            Gross for every entry in this table; recorded rather than assumed.
        min_sample_years: the shortest realized sample from which the harness
            will render a verdict (years). Below it, a check raises
            :class:`~backend.features.validation.errors.InsufficientHistoryError`.
        notes: caveats a reviewer needs in order to read a verdict correctly —
            post-publication decay, known dead decades, whether the effect is a
            raw return or an alpha.

    Raises:
        ExpectationDeclarationError: if the declaration cannot describe a
            checkable claim. Every check runs at construction, so a malformed
            expectation cannot reach a comparison.
    """

    factor: str
    sign: PremiumSign
    annualized_low: float
    annualized_high: float
    source: str
    published_estimate: str
    construction: str
    sample: str
    published_cost_basis: CostBasis
    min_sample_years: float
    notes: str

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            ExpectationDeclarationError: naming the offending field and what it
                must be.
        """
        if not self.factor.strip():
            msg = "an expectation must name the factor it applies to; got an empty name"
            raise ExpectationDeclarationError(msg)
        for field, text in (
            ("source", self.source),
            ("published_estimate", self.published_estimate),
            ("construction", self.construction),
            ("sample", self.sample),
        ):
            if not text.strip():
                msg = (
                    f"expectation for {self.factor!r} has an empty {field}. An "
                    f"expectation without provenance is an assertion, and the whole "
                    f"point of this table is that each entry can be checked against "
                    f"the literature it came from."
                )
                raise ExpectationDeclarationError(msg)
        for field, endpoint in (
            ("annualized_low", self.annualized_low),
            ("annualized_high", self.annualized_high),
        ):
            if not math.isfinite(endpoint):
                msg = f"expectation for {self.factor!r} has a non-finite {field}={endpoint!r}"
                raise ExpectationDeclarationError(msg)
            if endpoint == 0.0:
                msg = (
                    f"expectation for {self.factor!r} has {field}=0. A range endpoint "
                    f"at zero makes 'the sign agrees' and 'the magnitude is in range' "
                    f"contradict each other for a premium of exactly zero, and admits "
                    f"a premium of the wrong sign at the boundary."
                )
                raise ExpectationDeclarationError(msg)
            if endpoint * self.sign.direction < 0.0:
                msg = (
                    f"expectation for {self.factor!r} declares sign {self.sign} but "
                    f"{field}={endpoint!r} carries the opposite sign. The range is a "
                    f"signed interval: a NEGATIVE expectation reads (-0.12, -0.02), "
                    f"not (0.02, 0.12), so that a range check is a plain comparison "
                    f"and cannot accept a premium pointing the wrong way."
                )
                raise ExpectationDeclarationError(msg)
            if abs(endpoint) > MAX_PLAUSIBLE_ANNUALIZED_PREMIUM:
                msg = (
                    f"expectation for {self.factor!r} declares {field}={endpoint!r}, "
                    f"above {MAX_PLAUSIBLE_ANNUALIZED_PREMIUM} in absolute value. "
                    f"Ranges are annualized fractions (0.04 is four percent a year); "
                    f"no long-sample published equity premium reaches 100% a year, so "
                    f"this is a percent written where a fraction was meant."
                )
                raise ExpectationDeclarationError(msg)
        if self.annualized_low >= self.annualized_high:
            msg = (
                f"expectation for {self.factor!r} declares an empty or inverted range "
                f"[{self.annualized_low!r}, {self.annualized_high!r}]. Endpoints are "
                f"ordered arithmetically, so for a NEGATIVE expectation the low "
                f"endpoint is the more negative one."
            )
            raise ExpectationDeclarationError(msg)
        if not math.isfinite(self.min_sample_years) or self.min_sample_years <= 0.0:
            msg = (
                f"expectation for {self.factor!r} declares "
                f"min_sample_years={self.min_sample_years!r}, which is not a positive "
                f"finite number of years. Gate G5 asks about long-sample premia, and "
                f"a minimum of zero would let a single quarter render a verdict."
            )
            raise ExpectationDeclarationError(msg)

    def contains(self, annualized_premium: float) -> bool:
        """Return whether a realized annualized premium falls inside the range.

        Args:
            annualized_premium: the realized premium, annualized simple return as
                a fraction, in the factor's own units and sign convention.

        Returns:
            ``True`` if ``annualized_low <= annualized_premium <=
            annualized_high``. Endpoints are inclusive: the range is a plausibility
            band, and a value landing exactly on a boundary is not evidence of a
            defect.
        """
        return self.annualized_low <= annualized_premium <= self.annualized_high

    def sign_agrees(self, annualized_premium: float) -> bool:
        """Return whether a realized premium points the way the literature says.

        Args:
            annualized_premium: the realized premium, annualized fraction, in the
                factor's own units.

        Returns:
            ``True`` when the realized premium is non-zero and shares the declared
            sign. Exactly zero agrees with neither direction and returns ``False``
            — though a premium that close to zero is separately reported as
            indistinguishable from it, which is the reading that matters.
        """
        return annualized_premium * self.sign.direction > 0.0

    def as_config(self) -> dict[str, object]:
        """Return the expectation as a JSON-serializable mapping for hashing.

        Every field is included, prose and all. A weakened caveat or a corrected
        citation changes the config hash of subsequent reports just as a widened
        range does, which is the intended behaviour: the prose is what a reviewer
        reads to decide whether a verdict means anything.

        Returns:
            A mapping accepted by
            :func:`backend.tracking.stamp.canonical_config_hash`.
        """
        return {
            "annualized_high": self.annualized_high,
            "annualized_low": self.annualized_low,
            "construction": self.construction,
            "factor": self.factor,
            "min_sample_years": self.min_sample_years,
            "notes": self.notes,
            "published_cost_basis": str(self.published_cost_basis),
            "published_estimate": self.published_estimate,
            "sample": self.sample,
            "sign": str(self.sign),
            "source": self.source,
        }


_DECLARED: Final = (
    FactorPremiumExpectation(
        factor="momentum_12_1",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.03,
        annualized_high=0.15,
        source=(
            "Jegadeesh, N., & Titman, S. (1993). 'Returns to Buying Winners and "
            "Selling Losers: Implications for Stock Market Efficiency.' Journal of "
            "Finance 48(1), 65-91. Corroborated out of sample by Jegadeesh & Titman "
            "(2001), Journal of Finance 56(2), 699-720, and across markets and asset "
            "classes by Asness, Moskowitz & Pedersen (2013), 'Value and Momentum "
            "Everywhere', Journal of Finance 68(3), 929-985."
        ),
        published_estimate=(
            "Jegadeesh & Titman report roughly 0.95% per month (about 12% a year) for "
            "the zero-cost winner-minus-loser portfolio on 6-month formation and "
            "6-month holding. The 12-1 formation with monthly holding used here is the "
            "convention behind Fama & French's UMD factor, which averages roughly "
            "0.65% per month (about 8% a year) over 1927-2023. The range spans both "
            "and allows for the wide dispersion of momentum's decade-by-decade means."
        ),
        construction=(
            "decile winner-minus-loser spread, equal-weighted in the original paper; "
            "the 12-1 corroborating figure is Fama-French's 2x3 value-weighted UMD"
        ),
        sample="NYSE/AMEX 1965-1989 (original); US 1927-2023 (UMD)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "Momentum's returns are strongly negatively skewed and its drawdowns are "
            "crashes rather than fades: 1932, 2001 and 2009 each cost the factor a "
            "third or more, and 2009 alone would dominate a sample beginning in 2005. "
            "Turnover is high, so the gross-to-net gap is large — Frazzini, Israel & "
            "Moskowitz (2015) find real-world implementation costs materially below "
            "earlier academic estimates, but not zero. A realized premium far ABOVE "
            "this range is not good news: the most common cause is contamination by "
            "the skipped month or by the compute date's own bar."
        ),
    ),
    FactorPremiumExpectation(
        factor="short_term_reversal",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.03,
        annualized_high=0.30,
        source=(
            "Jegadeesh, N. (1990). 'Evidence of Predictable Behavior of Security "
            "Returns.' Journal of Finance 45(3), 881-898. Weekly analogue: Lehmann, "
            "B. N. (1990). 'Fads, Martingales, and Market Efficiency.' Quarterly "
            "Journal of Economics 105(1), 1-28."
        ),
        published_estimate=(
            "Jegadeesh reports a decile spread on the prior month's return of roughly "
            "2% per month (over 20% a year) for the equal-weighted extreme portfolios. "
            "Fama-French's value-weighted 2x3 ST_Rev factor averages roughly 0.5% per "
            "month (about 6% a year). The range is the widest in this table because "
            "the two constructions differ by more than a factor of three."
        ),
        construction=(
            "decile spread on the prior month's return, equal-weighted (original); "
            "2x3 value-weighted ST_Rev (corroborating figure)"
        ),
        sample="NYSE 1934-1987 (original); US 1926-2023 (ST_Rev)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The factor with by far the largest gross-to-net gap in this table. It "
            "turns over essentially the whole book every month, its premium is "
            "concentrated in small and illiquid names, and a substantial part of the "
            "measured spread is bid-ask bounce rather than a tradeable return. A "
            "NET-of-costs realized premium near zero is the expected outcome and does "
            "NOT contradict the literature; the sign check is the informative one "
            "here, and the magnitude check is only meaningful gross."
        ),
    ),
    FactorPremiumExpectation(
        factor="low_volatility",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.01,
        annualized_high=0.12,
        source=(
            "Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). 'The Cross-Section "
            "of Volatility and Expected Returns.' Journal of Finance 61(1), 259-299, "
            "and (2009) 'High Idiosyncratic Volatility and Low Returns: International "
            "and Further U.S. Evidence.' Journal of Financial Economics 91(1), 1-23. "
            "See also Baker, M., Bradley, B., & Wurgler, J. (2011). 'Benchmarks as "
            "Limits to Arbitrage.' Financial Analysts Journal 67(1), 40-54, and "
            "Frazzini, A., & Pedersen, L. H. (2014). 'Betting Against Beta.' Journal "
            "of Financial Economics 111(1), 1-25."
        ),
        published_estimate=(
            "Ang et al. report the highest-volatility quintile underperforming the "
            "lowest by roughly 1% per month in risk-adjusted terms (about 12% a year), "
            "with t-statistics near 3. Frazzini & Pedersen's beta-hedged BAB factor "
            "earns roughly 0.7% per month (about 8% a year) with a Sharpe ratio near "
            "0.8. The low end of the range reflects the RAW long-short spread, which "
            "is materially smaller than either."
        ),
        construction=(
            "quintile spread on trailing idiosyncratic volatility, value-weighted, "
            "reported as a Fama-French three-factor alpha rather than a raw return"
        ),
        sample="US 1963-2000 (Ang et al. 2006); 23 developed markets (2009)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The one entry whose published figure is an ALPHA, not a raw return. An "
            "unhedged low-minus-high volatility portfolio carries a large negative "
            "market beta, so part of any raw premium it earns is a short-the-market "
            "position rather than the anomaly, and the raw spread is smaller than the "
            "alpha in rising markets and larger in falling ones. A raw realized "
            "premium landing at the bottom of this range is therefore consistent with "
            "the literature; one at the top over a bull-market sample is more likely "
            "to be the beta than the effect. Beta-neutralizing the factor before "
            "measuring (available in the P5.2 transform pipeline) makes the "
            "comparison sharper."
        ),
    ),
    FactorPremiumExpectation(
        factor="book_to_price",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.01,
        annualized_high=0.08,
        source=(
            "Fama, E. F., & French, K. R. (1992). 'The Cross-Section of Expected Stock "
            "Returns.' Journal of Finance 47(2), 427-465, and (1993) 'Common Risk "
            "Factors in the Returns on Stocks and Bonds.' Journal of Financial "
            "Economics 33(1), 3-56. Earlier: Rosenberg, B., Reid, K., & Lanstein, R. "
            "(1985). 'Persuasive Evidence of Market Inefficiency.' Journal of "
            "Portfolio Management 11(3), 9-16."
        ),
        published_estimate=(
            "The HML factor averages roughly 0.35-0.40% per month over 1963-1991 "
            "(about 4-5% a year) and roughly 0.33% per month over the full 1926-2023 "
            "history (about 4% a year)."
        ),
        construction="2x3 size/book-to-market sorts, value-weighted, rebalanced annually (HML)",
        sample="US 1963-1991 (original); 1926-2023 (full history)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The factor with the best-documented dead decade: HML earned approximately "
            "nothing, and by several constructions less than nothing, from 2007 to "
            "2020 (Fama & French, 'The Value Premium', 2021). A sample that begins "
            "after 2005 will not reproduce this premium no matter how correct the "
            "implementation, which is the single strongest argument for the 20-year "
            "minimum. Note also D-027: this factor's 7-day availability lag applies to "
            "its price leg too, so the realized premium here is measured on a slightly "
            "staler book-to-price than the published one — signal loss in the safe "
            "direction, and a reason to expect the low end rather than the high."
        ),
    ),
    FactorPremiumExpectation(
        factor="earnings_yield",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.02,
        annualized_high=0.10,
        source=(
            "Basu, S. (1977). 'Investment Performance of Common Stocks in Relation to "
            "Their Price-Earnings Ratios: A Test of the Efficient Market Hypothesis.' "
            "Journal of Finance 32(3), 663-682, and Basu, S. (1983). 'The "
            "Relationship between Earnings Yield, Market Value and Return for NYSE "
            "Common Stocks.' Journal of Financial Economics 12(1), 129-156."
        ),
        published_estimate=(
            "Basu (1977) reports the highest-earnings-yield quintile returning about "
            "16.3% a year against about 9.3% for the lowest — a spread near 7% a year "
            "— over 1957-1971. Basu (1983) shows the effect survives a size control, "
            "at a smaller magnitude."
        ),
        construction="quintile spread on trailing earnings-to-price, equal-weighted",
        sample="NYSE 1957-1971 (original); 1963-1980 (size-controlled)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "Correlated with book_to_price at the 0.5-0.7 level historically, so the "
            "two are not independent confirmations of each other; the P5.6 correlation "
            "review is where that redundancy is judged. Loss-making firms have a "
            "negative earnings yield and sit in the short leg mechanically, which makes "
            "this factor's magnitude sensitive to how the negative-earnings tail is "
            "handled — a plausible cause of a realized premium outside this range that "
            "is a construction difference rather than a defect."
        ),
    ),
    FactorPremiumExpectation(
        factor="gross_profitability",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.01,
        annualized_high=0.08,
        source=(
            "Novy-Marx, R. (2013). 'The Other Side of Value: The Gross Profitability "
            "Premium.' Journal of Financial Economics 108(1), 1-28."
        ),
        published_estimate=(
            "Novy-Marx reports roughly 0.31% per month (about 3.7% a year, t near 2.5) "
            "for the value-weighted long-short portfolio sorted on gross profits to "
            "assets over 1963-2010."
        ),
        construction=(
            "decile spread on gross profits-to-assets, value-weighted, rebalanced annually"
        ),
        sample="US 1963-2010",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The premium is substantially larger within value stocks and roughly "
            "absent among the most expensive names, so a realized premium here is "
            "sensitive to whether the sort was made value-neutral. The measure is "
            "taken at the top of the income statement deliberately (Novy-Marx's whole "
            "argument), so a realized premium far below this range is worth checking "
            "against the definition actually computed — a bottom-line profitability "
            "measure is a different factor with a smaller premium."
        ),
    ),
    FactorPremiumExpectation(
        factor="roic",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.01,
        annualized_high=0.09,
        source=(
            "No paper prices this exact definition. The closest published analogues "
            "are Fama, E. F., & French, K. R. (2015). 'A Five-Factor Asset Pricing "
            "Model.' Journal of Financial Economics 116(1), 1-22 (the RMW "
            "operating-profitability factor); Hou, K., Xue, C., & Zhang, L. (2015). "
            "'Digesting Anomalies: An Investment Approach.' Review of Financial "
            "Studies 28(3), 650-705 (the ROE factor); and Haugen, R. A., & Baker, "
            "N. L. (1996). 'Commonality in the Determinants of Expected Stock "
            "Returns.' Journal of Financial Economics 41(3), 401-439."
        ),
        published_estimate=(
            "RMW averages roughly 0.25-0.28% per month (about 3.0-3.4% a year) over "
            "1963-2013. The q-factor ROE premium is larger, roughly 0.55% per month "
            "(about 6.6% a year, t near 4.8) over 1967-2014, partly because it is "
            "formed on the most recently announced quarterly earnings rather than on "
            "annual accounting data."
        ),
        construction=(
            "2x3 value-weighted sorts on operating profitability (RMW); 2x3x3 "
            "value-weighted sorts on quarterly ROE, rebalanced monthly (q-factor)"
        ),
        sample="US 1963-2013 (RMW); 1967-2014 (ROE)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The weakest magnitude claim in this table, and the range is wide because "
            "of it: ROIC deducts cash from the denominator and taxes the numerator, "
            "which neither RMW nor ROE does, so no published number is measured on "
            "this quantity. The SIGN is the load-bearing part of this entry. A "
            "realized premium inside the range is consistent with the profitability "
            "literature; one outside it is a prompt to check the definition before "
            "concluding anything about the code."
        ),
    ),
    FactorPremiumExpectation(
        factor="accruals",
        sign=PremiumSign.NEGATIVE,
        annualized_low=-0.12,
        annualized_high=-0.02,
        source=(
            "Sloan, R. G. (1996). 'Do Stock Prices Fully Reflect Information in "
            "Accruals and Cash Flows About Future Earnings?' The Accounting Review "
            "71(3), 289-315. Decay documented by Green, J., Hand, J. R. M., & Soliman, "
            "M. T. (2011). 'Going, Going, Gone? The Apparent Demise of the Accruals "
            "Anomaly.' Management Science 57(5), 797-816."
        ),
        published_estimate=(
            "Sloan reports roughly 10.4% size-adjusted abnormal return over the year "
            "following formation for a hedge portfolio long the lowest-accrual decile "
            "and short the highest, over 1962-1991. In this factor's units — the value "
            "IS the accruals measure, so the premium accrues to the SHORT leg — that "
            "is about -10% a year."
        ),
        construction=(
            "decile hedge portfolio on total accruals scaled by average total assets, "
            "equal-weighted"
        ),
        sample="US 1962-1991",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The clearest case of post-publication decay in this table: Green, Hand & "
            "Soliman find the premium indistinguishable from zero after roughly 2003, "
            "attributed to hedge-fund capital entering the trade. A long sample "
            "beginning in the 1960s should still show it; a sample beginning in 2005 "
            "should not, and a strong realized premium over a recent-only sample is "
            "more likely to be a lookahead in the fundamentals lag than a revival. "
            "Note the sign convention: a POSITIVE realized premium here means high-"
            "accrual firms outperformed, which contradicts the literature and is "
            "flagged."
        ),
    ),
    FactorPremiumExpectation(
        factor="asset_growth",
        sign=PremiumSign.NEGATIVE,
        annualized_low=-0.13,
        annualized_high=-0.02,
        source=(
            "Cooper, M. J., Gulen, H., & Schill, M. J. (2008). 'Asset Growth and the "
            "Cross-Section of Stock Returns.' Journal of Finance 63(4), 1609-1651. "
            "Corroborated by the CMA factor in Fama, E. F., & French, K. R. (2015). "
            "'A Five-Factor Asset Pricing Model.' Journal of Financial Economics "
            "116(1), 1-22."
        ),
        published_estimate=(
            "Cooper, Gulen & Schill report roughly 8% a year value-weighted (and "
            "roughly 20% equal-weighted) for the low-minus-high asset-growth decile "
            "spread over 1968-2003. CMA averages roughly 0.33% per month (about 4% a "
            "year) over 1963-2013. In this factor's units — the value IS asset growth "
            "— both are negative premia, so the range spans about -4% to -8% with "
            "margin at each end."
        ),
        construction=(
            "decile spread on year-over-year total asset growth, value-weighted, "
            "rebalanced annually (original); 2x3 value-weighted investment sorts (CMA)"
        ),
        sample="US 1968-2003 (original); 1963-2013 (CMA)",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes=(
            "The equal-weighted figure is roughly two and a half times the "
            "value-weighted one, which is why the reference construction matters: a "
            "realized premium near -20% a year is not a stronger result, it is "
            "evidence the portfolio is loading on microcaps. Correlated with accruals "
            "(both are balance-sheet growth measures) and with book_to_price by "
            "construction, since a firm that has grown its assets without growing its "
            "market value is mechanically cheap."
        ),
    ),
)

EXPECTATIONS: Final[Mapping[str, FactorPremiumExpectation]] = MappingProxyType(
    {expectation.factor: expectation for expectation in _DECLARED}
)
"""The published expectation for each of the nine declared baseline factors.

A read-only mapping from factor name to :class:`FactorPremiumExpectation`.
Immutable at runtime so a caller cannot widen a range between two comparisons
inside one process — an edit has to be a visible change to this file, which
changes the config hash of every report produced afterwards.

Coverage is nine of the eleven baseline factors ``PLAN.md`` P5.3 names: ``size``,
``short_interest`` and Amihud illiquidity are not declared in
:mod:`backend.features.factors` yet, and an expectation for a factor whose
registry name is not yet fixed would be a guess about a name rather than a claim
about the literature. :func:`factors_without_expectations` is how the P5.6 gate
detects the gap; the gate's first clause cannot pass while it is non-empty.
"""


def expectation_for(factor: str) -> FactorPremiumExpectation:
    """Return the published expectation declared for ``factor``.

    Args:
        factor: the factor's registry name.

    Returns:
        Its :class:`FactorPremiumExpectation`.

    Raises:
        UnknownFactorExpectationError: if no expectation is declared. Never
            returns a permissive default: a factor with no prior expectation
            cannot be validated, and inventing one with the realized number
            already in hand would make the check circular.
    """
    try:
        return EXPECTATIONS[factor]
    except KeyError as exc:
        raise UnknownFactorExpectationError(factor, tuple(sorted(EXPECTATIONS))) from exc


def factors_without_expectations(factors: Iterable[str]) -> tuple[str, ...]:
    """Return the names in ``factors`` that carry no published expectation.

    The coverage check gate G5 needs: every registered factor must have a
    written-down expectation before "known factor premia reproduce" can mean
    anything, because a factor absent from the table is a factor nobody checked
    and a report listing only the covered ones reads as complete.

    Args:
        factors: factor names — typically
            ``registry.names()`` or ``spec.name for spec in BASELINE_FACTORS``.

    Returns:
        The uncovered names, sorted and de-duplicated. Empty when every name is
        covered.
    """
    return tuple(sorted({factor for factor in factors if factor not in EXPECTATIONS}))


def expectations_config(
    expectations: Mapping[str, FactorPremiumExpectation] = EXPECTATIONS,
) -> dict[str, object]:
    """Render an expectation table as a JSON-serializable config fragment.

    This is what puts the expectations inside the reproducibility stamp (I2). A
    report's ``config_hash`` covers the full table it was judged against —
    ranges, citations, caveats and minimum sample lengths — so two reports with
    the same hash were held to the same standard, and a report produced after any
    edit to the table is visibly a different run rather than a continuation of
    the old one.

    Args:
        expectations: the table to render. Defaults to :data:`EXPECTATIONS`.

    Returns:
        A mapping of factor name to that expectation's fields, accepted by
        :func:`backend.tracking.stamp.canonical_config_hash` (which sorts keys,
        so the ordering of this dict does not affect the digest).
    """
    return {factor: expectation.as_config() for factor, expectation in expectations.items()}
