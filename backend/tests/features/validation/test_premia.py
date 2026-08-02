"""The harness: does it detect what it claims to detect? (P5.4).

**Nothing in this file is data.** Every return series here is a hand-built
alternating two-value sequence whose mean, standard deviation and t-statistic
have closed forms written out below, constructed so the harness can be shown to
recover a premium that was *put there on purpose*. None of it is a measurement
of any factor, none of it is presented as one, and no verdict produced here says
anything about ``momentum_12_1`` or any other factor (I3). The real panel does
not exist: ``price_bar`` is empty and six of the nine factors refuse outright,
both blocked on B1.

The closed forms, for a series of ``n`` values (``n`` even) alternating
``mu + d`` and ``mu - d``:

* mean ``= mu`` — the ``+d`` and ``-d`` deviations cancel exactly in pairs;
* ``stdev(ddof=1) = d * sqrt(n / (n - 1))`` — every deviation is ``±d``, so the
  sum of squares is ``n * d ** 2``;
* standard error ``= stdev / sqrt(n) = d / sqrt(n - 1)``;
* ``t = mu * sqrt(n - 1) / d``.

These are derived here from the definition of the estimators, not read back out
of the implementation, so a sign or ``ddof`` error in the module would show up as
a mismatch rather than agreeing with itself.

What the suite has to prove, in the language of the task:

* a fixture with a **known injected premium** is recovered with the right sign
  and magnitude (:class:`TestAnInjectedPremiumIsRecovered`);
* a fixture with **no premium** reports no premium rather than a spurious one
  (:class:`TestNoPremiumIsReportedAsNoPremium`) — including the case where the
  sign happens to agree, which is the one a lazy implementation passes;
* a factor whose realized sign **contradicts** its published expectation is
  flagged rather than silently reported (:class:`TestContradictionsAreFlagged`);
* running with **no data** raises rather than reporting a plausible zero
  (:class:`TestNoDataRaises`).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

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
    expectation_for,
    expectations_config,
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
    check_factor_premium,
    premia_validation_stamp,
    realized_premium,
    validate_premia,
)
from backend.tracking.stamp import ReproducibilityStamp, canonical_config_hash

MONTHS_PER_YEAR = 12.0
LONG_SAMPLE_MONTHS = 240
"""Twenty years of monthly observations: the shortest sample any entry accepts."""

FIXTURE_CONSTRUCTION = (
    "CONSTRUCTED TEST FIXTURE: alternating two-value sequence with a known mean; "
    "not a measurement of anything"
)

CLEAN_STAMP = ReproducibilityStamp(
    git_commit="0" * 40,
    git_dirty=False,
    data_version="fixture-no-data",
    config_hash="0" * 64,
    seed=0,
)
"""A stamp for tests that are not about stamping. Obviously not a real commit."""

DIRTY_STAMP = ReproducibilityStamp(
    git_commit="1" * 40,
    git_dirty=True,
    data_version="fixture-no-data",
    config_hash="0" * 64,
    seed=0,
)


def constructed_series(
    factor: str,
    *,
    mean_per_period: float,
    half_spread: float,
    observations: int = LONG_SAMPLE_MONTHS,
    periods_per_year: float = MONTHS_PER_YEAR,
    cost_basis: CostBasis = CostBasis.GROSS_OF_COSTS,
) -> FactorReturnSeries:
    """Build a fixture series alternating ``mean ± half_spread``.

    Deliberately not random and not plausible-looking: a real long-short return
    series never alternates. The point is an exactly known mean and standard
    error, so the harness's arithmetic is checked against the closed forms in the
    module docstring rather than against a second copy of itself.
    """
    if observations % 2 != 0:
        msg = "the closed forms in this file assume an even number of observations"
        raise ValueError(msg)
    values = np.empty(observations, dtype=np.float64)
    values[0::2] = mean_per_period + half_spread
    values[1::2] = mean_per_period - half_spread
    return FactorReturnSeries(
        factor=factor,
        returns=values,
        periods_per_year=periods_per_year,
        cost_basis=cost_basis,
        construction=FIXTURE_CONSTRUCTION,
    )


def expected_t_statistic(*, mean_per_period: float, half_spread: float, observations: int) -> float:
    """Return ``mu * sqrt(n - 1) / d`` — the closed form, derived independently."""
    return mean_per_period * math.sqrt(observations - 1) / half_spread


def midrange_series(factor: str) -> FactorReturnSeries:
    """Build a fixture landing in the middle of ``factor``'s published range.

    The half-spread is set to the mean's magnitude, which puts ``|t|`` at
    ``sqrt(n - 1)`` — about 15.5 for a 240-month fixture, comfortably significant
    — so the verdict turns on the sign and magnitude rather than on precision.
    """
    expectation = EXPECTATIONS[factor]
    annualized = 0.5 * (expectation.annualized_low + expectation.annualized_high)
    mean = annualized / MONTHS_PER_YEAR
    return constructed_series(factor, mean_per_period=mean, half_spread=abs(mean))


class TestTheFixtureItself:
    """The closed forms hold, so the rest of the file is checking the harness."""

    def test_the_fixture_has_the_mean_and_standard_error_the_module_docstring_states(
        self,
    ) -> None:
        series = constructed_series("momentum_12_1", mean_per_period=0.005, half_spread=0.02)
        values = series.returns
        assert values.size == LONG_SAMPLE_MONTHS
        assert float(np.mean(values)) == pytest.approx(0.005, abs=1e-15)
        assert float(np.std(values, ddof=1)) == pytest.approx(
            0.02 * math.sqrt(240 / 239), rel=1e-12
        )


class TestAnInjectedPremiumIsRecovered:
    """A premium put there on purpose comes back with the right sign and size."""

    def test_a_six_percent_annual_premium_is_measured_as_six_percent(self) -> None:
        # 0.5% a month over 240 months. Annualized arithmetically: 0.005 * 12.
        series = constructed_series("momentum_12_1", mean_per_period=0.005, half_spread=0.02)
        measured = realized_premium(series)
        assert measured.mean_per_period == pytest.approx(0.005, abs=1e-15)
        assert measured.annualized == pytest.approx(0.06, abs=1e-14)
        assert measured.standard_error_per_period == pytest.approx(0.02 / math.sqrt(239), rel=1e-12)
        assert measured.t_statistic == pytest.approx(
            expected_t_statistic(mean_per_period=0.005, half_spread=0.02, observations=240),
            rel=1e-12,
        )
        assert measured.observations == 240
        assert measured.sample_years == 20.0
        assert measured.cost_basis is CostBasis.GROSS_OF_COSTS

    def test_the_verdict_on_that_premium_is_that_it_reproduces(self) -> None:
        series = constructed_series("momentum_12_1", mean_per_period=0.005, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert check.is_significant
        assert check.sign_agrees
        assert check.magnitude_within_range
        assert check.verdict is PremiumVerdict.REPRODUCES
        assert not check.flagged

    def test_a_negative_expectation_factor_reproduces_with_a_negative_premium(self) -> None:
        # accruals expects a NEGATIVE premium: the factor's value IS the accruals
        # measure, so the premium accrues to the short leg. A harness that
        # compared magnitudes would pass this on +6% too.
        series = constructed_series("accruals", mean_per_period=-0.005, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("accruals"))
        assert check.realized.annualized == pytest.approx(-0.06, abs=1e-14)
        assert check.verdict is PremiumVerdict.REPRODUCES

    def test_the_same_premium_with_the_wrong_sign_does_not_reproduce(self) -> None:
        series = constructed_series("accruals", mean_per_period=0.005, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("accruals"))
        assert check.verdict is PremiumVerdict.SIGN_CONTRADICTS

    @pytest.mark.parametrize("factor", sorted(EXPECTATIONS))
    def test_a_midrange_premium_reproduces_for_every_declared_factor(self, factor: str) -> None:
        check = check_factor_premium(midrange_series(factor), expectation_for(factor))
        assert check.verdict is PremiumVerdict.REPRODUCES

    def test_the_t_statistic_does_not_depend_on_the_declared_periodicity(self) -> None:
        # Annualizing the mean and the standard error by the same convention
        # leaves their ratio alone, so the reported t is equally the t of the
        # annualized premium. Same values, two periodicities, both long enough.
        monthly = constructed_series("momentum_12_1", mean_per_period=0.005, half_spread=0.02)
        annual = constructed_series(
            "momentum_12_1", mean_per_period=0.005, half_spread=0.02, periods_per_year=1.0
        )
        assert realized_premium(monthly).t_statistic == pytest.approx(
            realized_premium(annual).t_statistic, rel=1e-12
        )
        assert realized_premium(monthly).annualized == pytest.approx(
            12.0 * realized_premium(annual).annualized, rel=1e-12
        )


class TestNoPremiumIsReportedAsNoPremium:
    """An insignificant premium is never dressed up as a finding."""

    def test_a_premium_indistinguishable_from_zero_is_reported_as_such(self) -> None:
        # mu = 0.0002, d = 0.02, n = 240 -> t = 0.155: the right sign, a plausible
        # magnitude, and no evidence whatsoever.
        series = constructed_series("momentum_12_1", mean_per_period=0.0002, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert abs(check.realized.t_statistic) < DEFAULT_SIGNIFICANCE_T
        assert check.sign_agrees, "the fixture's sign does agree — that must not be enough"
        assert check.verdict is PremiumVerdict.NO_PREMIUM_DETECTED
        assert not check.flagged

    def test_an_exactly_zero_premium_is_reported_as_no_premium(self) -> None:
        series = constructed_series("momentum_12_1", mean_per_period=0.0, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert check.realized.annualized == pytest.approx(0.0, abs=1e-18)
        assert check.realized.t_statistic == pytest.approx(0.0, abs=1e-12)
        assert not check.sign_agrees
        assert check.verdict is PremiumVerdict.NO_PREMIUM_DETECTED

    def test_an_insignificant_wrong_sign_is_not_called_a_contradiction(self) -> None:
        # Reading noise as a refutation is the same error as reading noise as a
        # confirmation. -0.02% a month at d = 0.02 is t = -0.155.
        series = constructed_series("momentum_12_1", mean_per_period=-0.0002, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert not check.sign_agrees
        assert check.verdict is PremiumVerdict.NO_PREMIUM_DETECTED
        assert not check.flagged

    def test_no_premium_detected_does_not_satisfy_the_gate(self) -> None:
        panel = FactorReturnPanel(
            tuple(
                constructed_series(factor, mean_per_period=0.0002, half_spread=0.02)
                for factor in sorted(EXPECTATIONS)
            )
        )
        report = validate_premia(panel, stamp=CLEAN_STAMP)
        assert len(report.undetermined) == len(EXPECTATIONS)
        assert not report.reproduces

    def test_the_significance_threshold_is_what_separates_the_two(self) -> None:
        # The same series is significant at a lower threshold and not at the
        # default: the boundary is real and it is the threshold, not an accident
        # of the fixture.
        series = constructed_series("momentum_12_1", mean_per_period=0.0002, half_spread=0.02)
        expectation = expectation_for("momentum_12_1")
        assert (
            check_factor_premium(series, expectation).verdict is PremiumVerdict.NO_PREMIUM_DETECTED
        )
        loose = check_factor_premium(series, expectation, significance_threshold=0.1)
        assert loose.verdict is PremiumVerdict.MAGNITUDE_IMPLAUSIBLE


class TestContradictionsAreFlagged:
    """A realized sign that contradicts the literature is never reported quietly."""

    def test_a_significant_wrong_sign_is_flagged(self) -> None:
        series = constructed_series("momentum_12_1", mean_per_period=-0.005, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert check.is_significant
        assert not check.sign_agrees
        assert check.verdict is PremiumVerdict.SIGN_CONTRADICTS
        assert check.flagged

    def test_a_flagged_factor_appears_in_the_report_and_fails_the_gate(self) -> None:
        good = [midrange_series(factor) for factor in sorted(EXPECTATIONS) if factor != "roic"]
        bad = constructed_series("roic", mean_per_period=-0.005, half_spread=0.02)
        report = validate_premia(FactorReturnPanel((*good, bad)), stamp=CLEAN_STAMP)
        assert [check.factor for check in report.flagged] == ["roic"]
        assert not report.reproduces
        assert "roic" in report.to_dict()["flagged_factors"]  # type: ignore[operator]

    def test_a_premium_far_above_the_published_range_is_flagged(self) -> None:
        # 4% a month is 48% a year against a range topping out at 15%. Not a
        # better momentum implementation: a lookahead, or a different universe.
        series = constructed_series("momentum_12_1", mean_per_period=0.04, half_spread=0.02)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert check.sign_agrees
        assert not check.magnitude_within_range
        assert check.verdict is PremiumVerdict.MAGNITUDE_IMPLAUSIBLE
        assert check.flagged

    def test_a_significant_premium_far_below_the_published_range_is_flagged(self) -> None:
        # 0.1% a month is 1.2% a year, below momentum's 3% floor, and measured
        # precisely enough to be significant.
        series = constructed_series("momentum_12_1", mean_per_period=0.001, half_spread=0.002)
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert check.is_significant
        assert check.sign_agrees
        assert check.verdict is PremiumVerdict.MAGNITUDE_IMPLAUSIBLE

    def test_a_premium_on_a_range_endpoint_is_not_flagged(self) -> None:
        series = constructed_series(
            "momentum_12_1", mean_per_period=0.15 / MONTHS_PER_YEAR, half_spread=0.02
        )
        check = check_factor_premium(series, expectation_for("momentum_12_1"))
        assert check.realized.annualized == pytest.approx(0.15, rel=1e-12)
        assert check.verdict is PremiumVerdict.REPRODUCES


class TestNoDataRaises:
    """The empty case — today's only reachable one — is an error, not a zero."""

    def test_an_empty_panel_raises(self) -> None:
        with pytest.raises(FactorReturnsUnavailableError, match="no return observations"):
            FactorReturnPanel(())

    def test_the_empty_panel_error_names_the_blocker(self) -> None:
        with pytest.raises(FactorReturnsUnavailableError) as raised:
            FactorReturnPanel(())
        message = str(raised.value)
        assert "B1" in message
        assert "not evidence of a zero premium" in message

    def test_an_empty_series_raises(self) -> None:
        with pytest.raises(FactorReturnsUnavailableError, match="momentum_12_1"):
            FactorReturnSeries(
                factor="momentum_12_1",
                returns=np.array([], dtype=np.float64),
                periods_per_year=MONTHS_PER_YEAR,
                cost_basis=CostBasis.GROSS_OF_COSTS,
                construction=FIXTURE_CONSTRUCTION,
            )

    def test_a_report_with_no_checks_raises(self) -> None:
        with pytest.raises(FactorReturnsUnavailableError):
            PremiaValidationReport(checks=(), expectations=EXPECTATIONS, stamp=CLEAN_STAMP)

    def test_a_sample_shorter_than_the_expectation_demands_raises(self) -> None:
        # Five years of monthly returns. HML earned nothing from 2007 to 2020;
        # a five-year window cannot tell a broken factor from a bad decade.
        short = constructed_series(
            "book_to_price", mean_per_period=0.003, half_spread=0.02, observations=60
        )
        with pytest.raises(InsufficientHistoryError) as raised:
            check_factor_premium(short, expectation_for("book_to_price"))
        assert raised.value.sample_years == pytest.approx(5.0)
        assert raised.value.required_years == 20.0

    def test_the_minimum_sample_boundary_is_inclusive(self) -> None:
        exactly_twenty = constructed_series(
            "book_to_price", mean_per_period=0.003, half_spread=0.02, observations=240
        )
        assert check_factor_premium(exactly_twenty, expectation_for("book_to_price"))
        one_month_short = constructed_series(
            "book_to_price", mean_per_period=0.003, half_spread=0.02, observations=238
        )
        with pytest.raises(InsufficientHistoryError):
            check_factor_premium(one_month_short, expectation_for("book_to_price"))

    def test_what_the_gate_still_needs_is_written_down(self) -> None:
        # The standing answer to "why is there no premia table in PROGRESS.md".
        assert len(WHAT_G5_STILL_NEEDS) >= 4
        joined = " ".join(WHAT_G5_STILL_NEEDS)
        assert "price_bar" in joined
        assert "B1" in joined


class TestMalformedInputIsRefused:
    """Nothing unusable becomes a number."""

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
    def test_a_non_finite_return_is_refused(self, bad: float) -> None:
        values = np.full(LONG_SAMPLE_MONTHS, 0.01, dtype=np.float64)
        values[7] = bad
        with pytest.raises(MalformedReturnSeriesError, match="NaN or an infinity"):
            FactorReturnSeries(
                factor="momentum_12_1",
                returns=values,
                periods_per_year=MONTHS_PER_YEAR,
                cost_basis=CostBasis.GROSS_OF_COSTS,
                construction=FIXTURE_CONSTRUCTION,
            )

    def test_a_percent_scale_series_is_refused_as_a_units_error(self) -> None:
        values = np.full(LONG_SAMPLE_MONTHS, 0.01, dtype=np.float64)
        values[0] = MAX_PLAUSIBLE_PERIOD_RETURN + 0.1
        with pytest.raises(MalformedReturnSeriesError, match="units guard"):
            FactorReturnSeries(
                factor="momentum_12_1",
                returns=values,
                periods_per_year=MONTHS_PER_YEAR,
                cost_basis=CostBasis.GROSS_OF_COSTS,
                construction=FIXTURE_CONSTRUCTION,
            )

    def test_a_two_dimensional_panel_is_refused(self) -> None:
        with pytest.raises(MalformedReturnSeriesError, match="not one dimension"):
            FactorReturnSeries(
                factor="momentum_12_1",
                returns=np.zeros((12, 20), dtype=np.float64),
                periods_per_year=MONTHS_PER_YEAR,
                cost_basis=CostBasis.GROSS_OF_COSTS,
                construction=FIXTURE_CONSTRUCTION,
            )

    def test_a_single_observation_is_refused(self) -> None:
        with pytest.raises(MalformedReturnSeriesError, match="no standard error"):
            FactorReturnSeries(
                factor="momentum_12_1",
                returns=np.array([0.01], dtype=np.float64),
                periods_per_year=MONTHS_PER_YEAR,
                cost_basis=CostBasis.GROSS_OF_COSTS,
                construction=FIXTURE_CONSTRUCTION,
            )

    @pytest.mark.parametrize("periodicity", [0.0, -12.0, float("nan"), MAX_PERIODS_PER_YEAR + 1.0])
    def test_an_unusable_periodicity_is_refused(self, periodicity: float) -> None:
        with pytest.raises(MalformedReturnSeriesError, match="periods_per_year"):
            constructed_series(
                "momentum_12_1",
                mean_per_period=0.005,
                half_spread=0.02,
                periods_per_year=periodicity,
            )

    def test_a_series_without_a_stated_construction_is_refused(self) -> None:
        with pytest.raises(MalformedReturnSeriesError, match="no construction was stated"):
            FactorReturnSeries(
                factor="momentum_12_1",
                returns=np.full(LONG_SAMPLE_MONTHS, 0.005, dtype=np.float64),
                periods_per_year=MONTHS_PER_YEAR,
                cost_basis=CostBasis.GROSS_OF_COSTS,
                construction="  ",
            )

    def test_a_constant_series_raises_rather_than_reporting_an_infinite_t(self) -> None:
        constant = FactorReturnSeries(
            factor="momentum_12_1",
            returns=np.full(LONG_SAMPLE_MONTHS, 0.005, dtype=np.float64),
            periods_per_year=MONTHS_PER_YEAR,
            cost_basis=CostBasis.GROSS_OF_COSTS,
            construction=FIXTURE_CONSTRUCTION,
        )
        with pytest.raises(DegenerateReturnSeriesError, match="t-statistic is undefined"):
            realized_premium(constant)

    def test_a_factor_with_no_expectation_cannot_be_validated(self) -> None:
        series = constructed_series("amihud_illiquidity", mean_per_period=0.005, half_spread=0.02)
        with pytest.raises(UnknownFactorExpectationError, match="written from the literature"):
            validate_premia(FactorReturnPanel((series,)), stamp=CLEAN_STAMP)

    def test_two_series_for_one_factor_are_refused(self) -> None:
        with pytest.raises(DuplicateFactorSeriesError, match="momentum_12_1"):
            FactorReturnPanel((midrange_series("momentum_12_1"), midrange_series("momentum_12_1")))

    def test_a_check_across_two_different_factors_is_refused(self) -> None:
        with pytest.raises(PremiaValidationError, match="different one wearing the wrong name"):
            FactorPremiumCheck(
                expectation=expectation_for("accruals"),
                realized=realized_premium(midrange_series("momentum_12_1")),
                significance_threshold=DEFAULT_SIGNIFICANCE_T,
            )

    @pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
    def test_an_unusable_significance_threshold_is_refused(self, threshold: float) -> None:
        with pytest.raises(PremiaValidationError, match="significance_threshold"):
            check_factor_premium(
                midrange_series("roic"),
                expectation_for("roic"),
                significance_threshold=threshold,
            )

    def test_the_series_copies_its_input_so_a_later_mutation_cannot_move_a_premium(self) -> None:
        values = np.full(LONG_SAMPLE_MONTHS, 0.005, dtype=np.float64)
        values[0::2] = 0.025
        series = FactorReturnSeries(
            factor="momentum_12_1",
            returns=values,
            periods_per_year=MONTHS_PER_YEAR,
            cost_basis=CostBasis.GROSS_OF_COSTS,
            construction=FIXTURE_CONSTRUCTION,
        )
        before = realized_premium(series).annualized
        values[:] = 999.0
        assert realized_premium(series).annualized == before
        assert not series.returns.flags.writeable


class TestTheReport:
    """Coverage, disclosures and the stamp travel with the verdicts."""

    def test_a_full_panel_of_reproducing_factors_satisfies_the_clause(self) -> None:
        panel = FactorReturnPanel(tuple(midrange_series(factor) for factor in sorted(EXPECTATIONS)))
        report = validate_premia(panel, stamp=CLEAN_STAMP)
        assert report.unchecked_factors == ()
        assert report.flagged == ()
        assert report.undetermined == ()
        assert report.reproduces

    def test_a_partial_panel_does_not_satisfy_the_clause(self) -> None:
        # Three of nine factors reproducing is not a third of a gate pass; it is
        # a gate that has not been run. Today only the three price factors could
        # even in principle produce a series.
        covered = ("momentum_12_1", "short_term_reversal", "low_volatility")
        panel = FactorReturnPanel(tuple(midrange_series(factor) for factor in covered))
        report = validate_premia(panel, stamp=CLEAN_STAMP)
        assert all(check.verdict is PremiumVerdict.REPRODUCES for check in report.checks)
        assert set(report.unchecked_factors) == set(EXPECTATIONS) - set(covered)
        assert not report.reproduces
        assert any("INCOMPLETE" in line for line in report.disclosures)

    def test_a_gross_report_says_so_and_refuses_to_be_a_performance_claim(self) -> None:
        panel = FactorReturnPanel(tuple(midrange_series(factor) for factor in sorted(EXPECTATIONS)))
        report = validate_premia(panel, stamp=CLEAN_STAMP)
        gross = [line for line in report.disclosures if "GROSS" in line]
        assert len(gross) == 1
        assert "may never be quoted as a performance result" in gross[0]

    def test_a_net_series_against_a_gross_expectation_is_disclosed_as_a_mismatch(self) -> None:
        net = constructed_series(
            "short_term_reversal",
            mean_per_period=0.005,
            half_spread=0.02,
            cost_basis=CostBasis.NET_OF_MODELLED_COSTS,
        )
        report = validate_premia(FactorReturnPanel((net,)), stamp=CLEAN_STAMP)
        assert [check.factor for check in report.basis_mismatches] == ["short_term_reversal"]
        assert not report.checks[0].basis_matches_published
        assert any("Cost-basis mismatch" in line for line in report.disclosures)

    def test_a_gross_series_against_a_gross_expectation_is_not_a_mismatch(self) -> None:
        report = validate_premia(
            FactorReturnPanel((midrange_series("momentum_12_1"),)), stamp=CLEAN_STAMP
        )
        assert report.basis_mismatches == ()
        assert not any("Cost-basis mismatch" in line for line in report.disclosures)

    def test_a_dirty_tree_is_disclosed(self) -> None:
        panel = FactorReturnPanel((midrange_series("momentum_12_1"),))
        clean = validate_premia(panel, stamp=CLEAN_STAMP)
        dirty = validate_premia(panel, stamp=DIRTY_STAMP)
        assert not any("dirty working tree" in line for line in clean.disclosures)
        assert any("dirty working tree" in line for line in dirty.disclosures)

    def test_every_report_states_the_pre_registration_and_the_t_assumption(self) -> None:
        report = validate_premia(
            FactorReturnPanel((midrange_series("momentum_12_1"),)), stamp=CLEAN_STAMP
        )
        joined = " ".join(report.disclosures)
        assert "before any return data existed" in joined
        assert "serially independent" in joined
        assert "not a forecast" in joined

    def test_the_rendered_report_carries_the_expectation_beside_the_number(self) -> None:
        report = validate_premia(
            FactorReturnPanel((midrange_series("accruals"),)), stamp=CLEAN_STAMP
        )
        rendered = report.to_dict()
        assert json.dumps(rendered)  # JSON-serializable end to end
        checks = rendered["checks"]
        assert isinstance(checks, list)
        row = checks[0]
        assert row["factor"] == "accruals"
        assert row["expected_sign"] == "NEGATIVE"
        assert row["expected_range"] == [-0.12, -0.02]
        assert row["realized_cost_basis"] == "gross_of_costs"
        assert "Sloan" in str(row["source"])
        assert rendered["disclosures"]
        assert isinstance(rendered["stamp"], dict)

    def test_a_report_cannot_check_one_factor_twice(self) -> None:
        check = check_factor_premium(midrange_series("roic"), expectation_for("roic"))
        with pytest.raises(DuplicateFactorSeriesError, match="roic"):
            PremiaValidationReport(
                checks=(check, check), expectations=EXPECTATIONS, stamp=CLEAN_STAMP
            )


class TestTheStamp:
    """I2: the report's config hash covers the expectations it was judged against."""

    def test_the_stamp_hashes_the_expectation_table_and_the_threshold(self) -> None:
        stamp = premia_validation_stamp(data_version="fixture-no-data")
        assert stamp.config_hash == canonical_config_hash(
            {
                "harness": "backend.features.validation.premia",
                "expectations": expectations_config(),
                "significance_threshold": DEFAULT_SIGNIFICANCE_T,
            }
        )
        assert stamp.data_version == "fixture-no-data"
        assert stamp.seed == 0

    def test_a_different_threshold_produces_a_different_hash(self) -> None:
        default = premia_validation_stamp(data_version="fixture-no-data")
        loosened = premia_validation_stamp(
            data_version="fixture-no-data", significance_threshold=1.0
        )
        assert loosened.config_hash != default.config_hash

    def test_a_narrower_expectation_table_produces_a_different_hash(self) -> None:
        default = premia_validation_stamp(data_version="fixture-no-data")
        subset = premia_validation_stamp(
            data_version="fixture-no-data",
            expectations={"roic": EXPECTATIONS["roic"]},
        )
        assert subset.config_hash != default.config_hash

    def test_extra_config_changes_the_hash_and_cannot_overwrite_a_reserved_key(self) -> None:
        with_universe = premia_validation_stamp(
            data_version="fixture-no-data", extra_config={"universe_criteria_hash": "abc"}
        )
        assert (
            with_universe.config_hash
            != premia_validation_stamp(data_version="fixture-no-data").config_hash
        )
        with pytest.raises(PremiaValidationError, match="collides"):
            premia_validation_stamp(
                data_version="fixture-no-data", extra_config={"expectations": {}}
            )
