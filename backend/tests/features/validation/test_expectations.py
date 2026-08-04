"""The expectation table: signs, ranges, provenance, and the anti-tuning hash (P5.4).

The expectations are the reusable artefact of P5.4 — the part that survives B1 —
so the checks here are about the *table*, not about the arithmetic that consumes
it (that is ``test_premia.py``).

Two of them earn their place by catching a class of error nothing else can:

* :class:`TestSignsAgreeWithTheFactorLibrary` reads the expected sign out of each
  factor's own :class:`~backend.features.spec.FeatureSpec` definition and
  compares it with the expectation declared here. Two statements of a factor's
  direction that live in different files will eventually disagree, and the
  disagreement is invisible — both files read plausibly on their own, and the
  harness would then validate a correctly-computed factor against a backwards
  expectation and flag it, or validate a backwards factor against a backwards
  expectation and pass it.
* :class:`TestTheTableCannotBeTunedSilently` pins the config-hash coupling. The
  claim in the module docstring — that an edited range shows up as a different
  ``config_hash`` — is only true if the table is actually inside the hashed
  config, and that is a property of code, not of intent.

:data:`EXPECTED` restates every entry's sign and range as literals, written out
by hand rather than derived from the table under test. Widening a range
therefore requires editing this file too, which is the point: it makes a
loosened expectation a visible, deliberate diff rather than a one-character
change nobody reviews.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from backend.features.factors import BASELINE_FACTORS
from backend.features.validation.errors import (
    ExpectationDeclarationError,
    UnknownFactorExpectationError,
)
from backend.features.validation.expectations import (
    EXPECTATIONS,
    MAX_PLAUSIBLE_ANNUALIZED_PREMIUM,
    CostBasis,
    FactorPremiumExpectation,
    PremiumSign,
    expectation_for,
    expectations_config,
    factors_without_expectations,
)
from backend.tracking.stamp import canonical_config_hash

EXPECTED: dict[str, tuple[PremiumSign, float, float]] = {
    # Sign and signed range (annualized fraction), written out by hand from the
    # published estimates cited in expectations.py. See the module docstring on
    # why these are literals rather than reads of the table under test.
    "momentum_12_1": (PremiumSign.POSITIVE, 0.03, 0.15),
    "short_term_reversal": (PremiumSign.POSITIVE, 0.03, 0.30),
    "low_volatility": (PremiumSign.POSITIVE, 0.01, 0.12),
    "book_to_price": (PremiumSign.POSITIVE, 0.01, 0.08),
    "earnings_yield": (PremiumSign.POSITIVE, 0.02, 0.10),
    "gross_profitability": (PremiumSign.POSITIVE, 0.01, 0.08),
    "roic": (PremiumSign.POSITIVE, 0.01, 0.09),
    "accruals": (PremiumSign.NEGATIVE, -0.12, -0.02),
    "asset_growth": (PremiumSign.NEGATIVE, -0.13, -0.02),
}

MINIMUM_SAMPLE_YEARS = 20.0
"""Every entry declares this. G5 asks about long samples; see expectations.py."""

_DECLARED_SIGN = re.compile(r"Expected premium sign: (POSITIVE|NEGATIVE)")
"""Pulls the sign out of a factor's own ``FeatureSpec.definition`` prose."""


def sample_expectation(**overrides: object) -> FactorPremiumExpectation:
    """Build a well-formed expectation, with fields overridden for a refusal test."""
    base = FactorPremiumExpectation(
        factor="fixture_factor",
        sign=PremiumSign.POSITIVE,
        annualized_low=0.02,
        annualized_high=0.08,
        source="Fixture, A. (1999). 'A Constructed Citation.' Journal of Fixtures 1(1), 1-2.",
        published_estimate="0.5% per month, about 6% a year",
        construction="decile long-short, value-weighted",
        sample="a constructed fixture, not a market",
        published_cost_basis=CostBasis.GROSS_OF_COSTS,
        min_sample_years=20.0,
        notes="",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class TestTheDeclaredTable:
    """The nine entries, checked against literals restated here."""

    def test_the_table_holds_exactly_the_nine_expected_factors(self) -> None:
        assert set(EXPECTATIONS) == set(EXPECTED)

    @pytest.mark.parametrize("factor", sorted(EXPECTED))
    def test_sign_and_range_match_the_hand_written_table(self, factor: str) -> None:
        sign, low, high = EXPECTED[factor]
        expectation = EXPECTATIONS[factor]
        assert expectation.sign is sign
        assert expectation.annualized_low == low
        assert expectation.annualized_high == high

    @pytest.mark.parametrize("factor", sorted(EXPECTED))
    def test_every_entry_carries_a_citation_with_a_year(self, factor: str) -> None:
        # A "source" of "Fama-French" is not a citation. Requiring a
        # parenthesized year is the cheapest check that some real reference was
        # written down, and it is what makes a range auditable rather than
        # asserted.
        expectation = EXPECTATIONS[factor]
        assert re.search(r"\(\d{4}\)", expectation.source), expectation.source
        assert expectation.published_estimate.strip()
        assert expectation.construction.strip()
        assert expectation.sample.strip()
        assert expectation.notes.strip()

    @pytest.mark.parametrize("factor", sorted(EXPECTED))
    def test_every_published_estimate_is_gross_of_costs(self, factor: str) -> None:
        # I4: the basis is recorded per entry rather than assumed, because it
        # decides how a net realized premium may be compared against the range.
        assert EXPECTATIONS[factor].published_cost_basis is CostBasis.GROSS_OF_COSTS

    @pytest.mark.parametrize("factor", sorted(EXPECTED))
    def test_every_entry_demands_a_long_sample(self, factor: str) -> None:
        assert EXPECTATIONS[factor].min_sample_years == MINIMUM_SAMPLE_YEARS

    def test_the_table_is_read_only_at_runtime(self) -> None:
        # A range widened between two comparisons inside one process would make
        # the config hash a lie: the hash is taken once, at stamp time.
        with pytest.raises(TypeError):
            EXPECTATIONS["momentum_12_1"] = sample_expectation()  # type: ignore[index]


class TestSignsAgreeWithTheFactorLibrary:
    """The expected sign here must match the sign the factor itself declares.

    Scoped to the entries this table declares, not to the live registry. Whether
    *every registered factor* carries an expectation is gate G5's coverage
    question, answered by
    :func:`~backend.features.validation.expectations.factors_without_expectations`
    at P5.6 — the three baseline factors ``PLAN.md`` still lists as unwritten
    (size, short interest, Amihud illiquidity) have no fixed registry name yet,
    and an expectation guessing at a name would be a claim about spelling rather
    than about the literature.
    """

    def test_every_expectation_names_a_declared_factor(self) -> None:
        declared = {spec.name for spec in BASELINE_FACTORS}
        assert set(EXPECTATIONS) <= declared

    @pytest.mark.parametrize("factor", sorted(EXPECTED))
    def test_the_factor_definition_states_the_same_sign(self, factor: str) -> None:
        specs = {spec.name: spec for spec in BASELINE_FACTORS}
        match = _DECLARED_SIGN.search(specs[factor].definition)
        assert match is not None, f"{factor} does not state an expected premium sign"
        assert EXPECTATIONS[factor].sign is PremiumSign(match.group(1))


class TestMalformedExpectationsAreRefused:
    """A declaration that cannot describe a checkable claim never reaches a comparison."""

    def test_a_range_endpoint_of_the_wrong_sign_is_refused(self) -> None:
        with pytest.raises(ExpectationDeclarationError, match="carries the opposite sign"):
            sample_expectation(sign=PremiumSign.NEGATIVE, annualized_low=0.02)

    def test_a_negative_expectation_needs_a_negative_range(self) -> None:
        # The encoding that would otherwise let a realized +8% satisfy an
        # expectation of -8%.
        with pytest.raises(ExpectationDeclarationError, match="carries the opposite sign"):
            sample_expectation(sign=PremiumSign.NEGATIVE, annualized_low=0.02, annualized_high=0.12)

    def test_a_zero_endpoint_is_refused(self) -> None:
        with pytest.raises(ExpectationDeclarationError, match="annualized_low=0"):
            sample_expectation(annualized_low=0.0)

    def test_an_inverted_range_is_refused(self) -> None:
        with pytest.raises(ExpectationDeclarationError, match="empty or inverted range"):
            sample_expectation(annualized_low=0.08, annualized_high=0.02)

    def test_a_percent_scale_range_is_refused_as_a_units_error(self) -> None:
        with pytest.raises(ExpectationDeclarationError, match="percent written where a fraction"):
            sample_expectation(annualized_high=MAX_PLAUSIBLE_ANNUALIZED_PREMIUM + 1e-9)

    def test_a_non_finite_endpoint_is_refused(self) -> None:
        with pytest.raises(ExpectationDeclarationError, match="non-finite"):
            sample_expectation(annualized_high=float("inf"))

    @pytest.mark.parametrize("field", ["source", "published_estimate", "construction", "sample"])
    def test_an_expectation_without_provenance_is_refused(self, field: str) -> None:
        with pytest.raises(ExpectationDeclarationError, match=f"empty {field}"):
            sample_expectation(**{field: "   "})

    def test_an_unnamed_expectation_is_refused(self) -> None:
        with pytest.raises(ExpectationDeclarationError, match="must name the factor"):
            sample_expectation(factor="")

    @pytest.mark.parametrize("years", [0.0, -1.0, float("nan")])
    def test_a_non_positive_minimum_sample_is_refused(self, years: float) -> None:
        with pytest.raises(ExpectationDeclarationError, match="min_sample_years"):
            sample_expectation(min_sample_years=years)


class TestRangeAndSignComparisons:
    """``contains`` and ``sign_agrees``, including the negative-expectation case."""

    def test_a_positive_expectation_accepts_only_positive_premia(self) -> None:
        expectation = EXPECTATIONS["momentum_12_1"]
        assert expectation.sign_agrees(0.06)
        assert not expectation.sign_agrees(-0.06)
        assert not expectation.sign_agrees(0.0)

    def test_a_negative_expectation_accepts_only_negative_premia(self) -> None:
        expectation = EXPECTATIONS["accruals"]
        assert expectation.sign_agrees(-0.06)
        assert not expectation.sign_agrees(0.06)

    def test_the_range_is_inclusive_at_both_endpoints(self) -> None:
        expectation = EXPECTATIONS["momentum_12_1"]
        assert expectation.contains(0.03)
        assert expectation.contains(0.15)
        assert not expectation.contains(0.0299)
        assert not expectation.contains(0.1501)

    def test_a_negative_range_orders_arithmetically(self) -> None:
        expectation = EXPECTATIONS["asset_growth"]
        assert expectation.contains(-0.13)
        assert expectation.contains(-0.02)
        assert not expectation.contains(-0.14)
        assert not expectation.contains(-0.01)

    def test_a_premium_of_the_wrong_sign_is_never_in_range(self) -> None:
        # The property the signed-interval encoding exists to guarantee.
        for expectation in EXPECTATIONS.values():
            wrong_way = -0.5 * (expectation.annualized_low + expectation.annualized_high)
            assert not expectation.contains(wrong_way)
            assert not expectation.sign_agrees(wrong_way)


class TestLookup:
    """A factor with no prior expectation cannot be validated at all."""

    def test_lookup_returns_the_declared_expectation(self) -> None:
        assert expectation_for("roic") is EXPECTATIONS["roic"]

    def test_an_undeclared_factor_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(UnknownFactorExpectationError) as raised:
            expectation_for("amihud_illiquidity")
        assert raised.value.factor == "amihud_illiquidity"
        assert "momentum_12_1" in str(raised.value)

    def test_coverage_reports_the_names_with_no_expectation(self) -> None:
        gap = factors_without_expectations(
            ["momentum_12_1", "size", "short_interest", "size", "roic"]
        )
        assert gap == ("short_interest", "size")

    def test_coverage_is_empty_when_every_name_is_declared(self) -> None:
        assert factors_without_expectations(EXPECTATIONS) == ()


class TestTheTableCannotBeTunedSilently:
    """The config-hash coupling that makes the pre-registration enforceable."""

    def test_the_config_carries_every_field_of_every_entry(self) -> None:
        config = expectations_config()
        assert set(config) == set(EXPECTED)
        momentum = config["momentum_12_1"]
        assert isinstance(momentum, dict)
        assert momentum["sign"] == "POSITIVE"
        assert momentum["annualized_low"] == 0.03
        assert momentum["annualized_high"] == 0.15
        assert "Jegadeesh" in str(momentum["source"])

    def test_the_config_is_canonically_hashable(self) -> None:
        digest = canonical_config_hash({"expectations": expectations_config()})
        assert re.fullmatch(r"[0-9a-f]{64}", digest)

    def test_widening_a_range_changes_the_hash(self) -> None:
        original = canonical_config_hash({"expectations": expectations_config()})
        widened = dict(EXPECTATIONS)
        widened["momentum_12_1"] = replace(EXPECTATIONS["momentum_12_1"], annualized_low=0.001)
        assert canonical_config_hash({"expectations": expectations_config(widened)}) != original

    def test_weakening_a_caveat_changes_the_hash(self) -> None:
        # The prose is hashed too: it is what a reviewer reads to decide whether
        # a verdict means anything, so quietly deleting a caveat must not leave
        # the report looking like the same run.
        original = canonical_config_hash({"expectations": expectations_config()})
        edited = dict(EXPECTATIONS)
        edited["book_to_price"] = replace(EXPECTATIONS["book_to_price"], notes="")
        assert canonical_config_hash({"expectations": expectations_config(edited)}) != original
