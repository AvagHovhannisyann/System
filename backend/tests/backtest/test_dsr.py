"""Known-answer and property tests for the Deflated Sharpe Ratio (P10.3).

Every expected number below is either

* arithmetic written out in the test body, evaluated step by step so a human can
  check it without running anything, or
* an *independently known* constant — the exact expectation of the maximum of
  ``n`` iid standard normals, from the order-statistics literature — which the
  implementation is never allowed to have produced.

Nothing here is checked against a second copy of the same formula.

Formulas under test (module docstrings carry the full citations):

* ``PSR(SR*) = Z[(SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4-1)/4 * SR**2)]``
  — Bailey & López de Prado (2012), *Journal of Risk* 15(2).
* ``SR*_0 = sqrt(V) * [(1-gamma) * Z^-1[1 - 1/N] + gamma * Z^-1[1 - 1/(N e)]]``
  and ``DSR = PSR(SR*_0)`` — Bailey & López de Prado (2014), *Journal of
  Portfolio Management* 40(5).

All Sharpe ratios here are **per observation period**, never annualized.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.backtest import dsr as dsr_module
from backend.backtest.dsr import (
    EULER_MASCHERONI,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    deflated_sharpe_ratio_from_trials,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from backend.backtest.metrics import kurtosis, sharpe_ratio, skewness

# ---------------------------------------------------------------------------
# Probabilistic Sharpe Ratio — the inner piece
# ---------------------------------------------------------------------------


def test_psr_matches_hand_computation_for_a_gaussian_sample() -> None:
    # SR = 0.10, SR* = 0, n = 101, g3 = 0, g4 = 3 (Gaussian).
    #   variance term = 1 - 0 * 0.1 + (3 - 1) / 4 * 0.1 ** 2
    #                 = 1 + 0.5 * 0.01 = 1.005
    #   sqrt(n - 1)   = sqrt(100) = 10
    #   z             = (0.10 - 0) * 10 / sqrt(1.005)
    #                 = 1.0 / 1.0024968827881711 = 0.9975093361076329
    #   PSR           = Z[0.9975093361076329] = 0.8407413278013518
    value = probabilistic_sharpe_ratio(
        observed_sharpe=0.1,
        benchmark_sharpe=0.0,
        n_observations=101,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert value == pytest.approx(0.8407413278013518, rel=1e-12)


def test_psr_matches_hand_computation_with_negative_skew_and_fat_tails() -> None:
    # SR = 0.20, SR* = 0.05, n = 257, g3 = -0.5, g4 = 6.
    #   variance term = 1 - (-0.5) * 0.2 + (6 - 1) / 4 * 0.2 ** 2
    #                 = 1 + 0.1 + 1.25 * 0.04 = 1.15
    #   sqrt(n - 1)   = sqrt(256) = 16
    #   z             = (0.20 - 0.05) * 16 / sqrt(1.15)
    #                 = 2.4 / 1.0723805294763609 = 2.2380115397767533
    #   PSR           = Z[2.2380115397767533] = 0.9873898486973046
    value = probabilistic_sharpe_ratio(
        observed_sharpe=0.2,
        benchmark_sharpe=0.05,
        n_observations=257,
        skewness=-0.5,
        kurtosis=6.0,
    )
    assert value == pytest.approx(0.9873898486973046, rel=1e-12)


def test_psr_is_one_half_when_the_observed_sharpe_equals_the_benchmark() -> None:
    # The numerator vanishes, so z = 0 and Z[0] = 0.5 whatever the moments are.
    for skew, kurt in ((0.0, 3.0), (-1.5, 9.0), (2.0, 12.0)):
        value = probabilistic_sharpe_ratio(
            observed_sharpe=0.3,
            benchmark_sharpe=0.3,
            n_observations=500,
            skewness=skew,
            kurtosis=kurt,
        )
        assert value == pytest.approx(0.5, abs=1e-15)


def test_psr_increases_strictly_with_the_observed_sharpe() -> None:
    values = [
        probabilistic_sharpe_ratio(
            observed_sharpe=sharpe,
            benchmark_sharpe=0.05,
            n_observations=1000,
            skewness=0.0,
            kurtosis=3.0,
        )
        for sharpe in (0.0, 0.02, 0.05, 0.08, 0.2)
    ]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_psr_increases_with_the_sample_length_at_a_fixed_sharpe() -> None:
    # sqrt(n - 1) multiplies the numerator: the same Sharpe ratio earned over a
    # longer sample is stronger evidence.
    values = [
        probabilistic_sharpe_ratio(
            observed_sharpe=0.1,
            benchmark_sharpe=0.0,
            n_observations=n,
            skewness=0.0,
            kurtosis=3.0,
        )
        for n in (26, 101, 401, 1601)
    ]
    assert values == sorted(values)


def test_fat_tails_lower_the_psr_when_the_sharpe_beats_its_benchmark() -> None:
    # Kurtosis enters only through the variance term, which is the estimator's
    # standard error. A larger standard error pulls the probability towards 0.5
    # from whichever side it sits on. Above the benchmark — the regime in which a
    # result is being certified — that means fat tails strictly lower the PSR.
    above = [
        probabilistic_sharpe_ratio(
            observed_sharpe=0.15,
            benchmark_sharpe=0.0,
            n_observations=1000,
            skewness=0.0,
            kurtosis=kurt,
        )
        for kurt in (1.0, 3.0, 6.0, 12.0, 30.0)
    ]
    assert above == sorted(above, reverse=True)
    assert all(value > 0.5 for value in above)

    # Below the benchmark the same widening pulls the probability *up* towards
    # 0.5. Asserting this too keeps the test honest about what kurtosis does,
    # rather than implying a monotonicity that does not exist.
    below = [
        probabilistic_sharpe_ratio(
            observed_sharpe=0.0,
            benchmark_sharpe=0.15,
            n_observations=1000,
            skewness=0.0,
            kurtosis=kurt,
        )
        for kurt in (1.0, 3.0, 6.0, 12.0, 30.0)
    ]
    assert below == sorted(below)
    assert all(value < 0.5 for value in below)


def test_negative_skew_lowers_the_psr_at_an_identical_sharpe() -> None:
    # variance term = 1 - g3 * SR + ...: with SR > 0, more negative skew enlarges
    # it. A Sharpe ratio earned by selling tail risk is worth less than the same
    # number earned symmetrically.
    values = [
        probabilistic_sharpe_ratio(
            observed_sharpe=0.15,
            benchmark_sharpe=0.0,
            n_observations=1000,
            skewness=skew,
            kurtosis=6.0,
        )
        for skew in (1.0, 0.5, 0.0, -0.5, -1.0)
    ]
    assert values == sorted(values, reverse=True)


def test_psr_rejects_an_excess_kurtosis_passed_where_a_non_excess_one_belongs() -> None:
    # A Gaussian sample has excess kurtosis 0 and non-excess kurtosis 3. Passing
    # the former is the single most likely unit error in this formula, and no
    # real distribution has non-excess kurtosis below 1.
    with pytest.raises(ValueError, match="excess kurtosis was probably passed"):
        probabilistic_sharpe_ratio(
            observed_sharpe=0.1,
            benchmark_sharpe=0.0,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.0,
        )


def test_psr_accepts_the_attainable_minimum_kurtosis_of_exactly_one() -> None:
    # The two-point symmetric distribution attains g4 = 1 exactly, and computing
    # m4 / m2 ** 2 on such a sample in floating point lands a few ulps below it.
    # Refusing that would reject a correctly computed moment; the guard exists to
    # catch an excess-kurtosis mix-up, which is off by about 3.
    minimum = kurtosis([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    assert minimum == pytest.approx(1.0, rel=1e-12)
    value = probabilistic_sharpe_ratio(
        observed_sharpe=0.1,
        benchmark_sharpe=0.0,
        n_observations=500,
        skewness=0.0,
        kurtosis=min(minimum, 1.0 - 1e-16),
    )
    assert 0.0 < value < 1.0
    # A value below 1 by more than floating-point slack is still refused.
    with pytest.raises(ValueError, match="excess kurtosis was probably passed"):
        probabilistic_sharpe_ratio(
            observed_sharpe=0.1,
            benchmark_sharpe=0.0,
            n_observations=500,
            skewness=0.0,
            kurtosis=0.999,
        )


def test_psr_rejects_moments_that_make_the_variance_term_non_positive() -> None:
    # g3 = 4, SR = 0.5, g4 = 1: 1 - 4 * 0.5 + 0 * 0.25 = -1. The moments are
    # mutually impossible; the formula has no meaning there.
    with pytest.raises(ValueError, match="not positive"):
        probabilistic_sharpe_ratio(
            observed_sharpe=0.5,
            benchmark_sharpe=0.0,
            n_observations=500,
            skewness=4.0,
            kurtosis=1.0,
        )


@pytest.mark.parametrize("n_observations", [-1, 0, 1])
def test_psr_rejects_a_sample_too_short_to_have_a_standard_error(n_observations: int) -> None:
    with pytest.raises(ValueError, match="n_observations must be at least 2"):
        probabilistic_sharpe_ratio(
            observed_sharpe=0.1,
            benchmark_sharpe=0.0,
            n_observations=n_observations,
            skewness=0.0,
            kurtosis=3.0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"observed_sharpe": float("nan")},
        {"benchmark_sharpe": float("inf")},
        {"skewness": float("nan")},
        {"kurtosis": float("inf")},
    ],
)
def test_psr_rejects_non_finite_inputs(kwargs: dict[str, float]) -> None:
    arguments: dict[str, float | int] = {
        "observed_sharpe": 0.1,
        "benchmark_sharpe": 0.0,
        "n_observations": 500,
        "skewness": 0.0,
        "kurtosis": 3.0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="must be finite"):
        probabilistic_sharpe_ratio(**arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The deflated benchmark — the expected maximum under the null
# ---------------------------------------------------------------------------


def test_expected_maximum_matches_hand_computation() -> None:
    # N = 10 trials, V = 0.25 so sqrt(V) = 0.5.
    #   Z^-1[1 - 1/10]        = Z^-1[0.9]              = 1.2815515655446008
    #   Z^-1[1 - 1/(10 * e)]  = Z^-1[0.9632120558828558] = 1.7892417645816279
    #   gamma                 = 0.5772156649015329
    #   bracket = (1 - gamma) * 1.2815515655446008 + gamma * 1.7892417645816279
    #           = 0.5417234...  + 1.0328748...  = 1.5745983013457500
    #   SR*_0   = 0.5 * 1.5745983013457500 = 0.787299150672875
    value = expected_maximum_sharpe_ratio(trials=10, trial_sharpe_variance=0.25)
    assert value == pytest.approx(0.787299150672875, rel=1e-12)


# Exact expectations of the maximum of n iid N(0, 1) draws. These are properties
# of the normal distribution, not of this code: E[max] = 1/sqrt(pi) for n = 2 and
# 3/(2 sqrt(pi)) for n = 3 are classical closed forms, and the rest are the
# standard tabulated normal order-statistic means (Harter, H. L. (1961),
# "Expected values of normal order statistics", Biometrika 48, 151-165).
_EXACT_EXPECTED_MAXIMUM = {
    2: 0.5641895835477563,  # 1 / sqrt(pi)
    3: 0.8462843753216345,  # 3 / (2 * sqrt(pi))
    4: 1.0293753730039641,
    5: 1.1629644736405196,
    10: 1.5387527308351728,
    20: 1.8674815878559086,
    50: 2.2490743062653502,
    100: 2.5075853723316522,
}


@pytest.mark.parametrize("trials", sorted(_EXACT_EXPECTED_MAXIMUM))
def test_the_extreme_value_approximation_tracks_the_exact_order_statistic(trials: int) -> None:
    # The Bailey-López de Prado benchmark is an *approximation* to E[max of N
    # iid N(0, V)]. Checking it against the exact value is the only way to know
    # the approximation was transcribed correctly; checking it against another
    # copy of itself would prove nothing.
    approximation = expected_maximum_sharpe_ratio(trials=trials, trial_sharpe_variance=1.0)
    exact = _EXACT_EXPECTED_MAXIMUM[trials]
    assert approximation == pytest.approx(exact, rel=0.10)
    # The approximation is known to run slightly high for small N (it is exact
    # only asymptotically); at N = 2 it is ~8% low, everywhere else within 3%.
    if trials >= 3:
        assert approximation == pytest.approx(exact, rel=0.03)


def test_the_expected_maximum_scales_with_the_square_root_of_the_trial_variance() -> None:
    unit = expected_maximum_sharpe_ratio(trials=25, trial_sharpe_variance=1.0)
    for variance in (0.01, 0.25, 4.0):
        scaled = expected_maximum_sharpe_ratio(trials=25, trial_sharpe_variance=variance)
        assert scaled == pytest.approx(math.sqrt(variance) * unit, rel=1e-12)


def test_the_expected_maximum_is_exactly_zero_for_a_single_trial() -> None:
    # Z^-1[1 - 1/1] = Z^-1[0] is -infinity; the quantity being approximated is
    # exactly 0 there (the expected maximum of one zero-mean draw), so the exact
    # value is returned rather than the approximation's singularity.
    assert expected_maximum_sharpe_ratio(trials=1, trial_sharpe_variance=0.25) == 0.0
    assert expected_maximum_sharpe_ratio(trials=7, trial_sharpe_variance=0.0) == 0.0


def test_the_expected_maximum_increases_strictly_with_the_trial_count() -> None:
    values = [
        expected_maximum_sharpe_ratio(trials=n, trial_sharpe_variance=0.04)
        for n in (1, 2, 5, 10, 50, 200, 1000)
    ]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"trials": 0, "trial_sharpe_variance": 1.0}, "trials must be at least 1"),
        ({"trials": -3, "trial_sharpe_variance": 1.0}, "trials must be at least 1"),
        ({"trials": 5, "trial_sharpe_variance": -1e-9}, "finite and non-negative"),
        ({"trials": 5, "trial_sharpe_variance": float("nan")}, "finite and non-negative"),
    ],
)
def test_the_expected_maximum_refuses_impossible_arguments(
    kwargs: dict[str, float], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        expected_maximum_sharpe_ratio(**kwargs)  # type: ignore[arg-type]


def test_the_euler_mascheroni_constant_is_the_published_value() -> None:
    # 0.57721566490153286060651209008240243... (OEIS A001620).
    assert abs(EULER_MASCHERONI - 0.5772156649015328606) < 1e-18


# ---------------------------------------------------------------------------
# The Deflated Sharpe Ratio itself
# ---------------------------------------------------------------------------


def test_deflated_sharpe_ratio_matches_hand_computation_end_to_end() -> None:
    # observed SR = 0.15 per period, n = 1001, g3 = 0, g4 = 3,
    # N = 10 trials with trial Sharpe variance V = 0.0025 (sd 0.05).
    #   SR*_0 = 0.05 * 1.5745983013457500 = 0.0787299150672875   (see above)
    #   variance term = 1 - 0 * 0.15 + (3 - 1) / 4 * 0.15 ** 2
    #                 = 1 + 0.5 * 0.0225 = 1.01125
    #   sqrt(n - 1)   = sqrt(1000) = 31.622776601683793
    #   z = (0.15 - 0.0787299150672875) * 31.622776601683793 / sqrt(1.01125)
    #     = 0.0712700849327125 * 31.622776601683793 / 1.0056092680559383
    #     = 2.2411865580427697
    #   DSR = Z[2.2411865580427697] = 0.9874930034039904
    result = deflated_sharpe_ratio(
        observed_sharpe=0.15,
        n_observations=1001,
        skewness=0.0,
        kurtosis=3.0,
        trials=10,
        trial_sharpe_variance=0.0025,
    )
    assert result.expected_maximum_sharpe == pytest.approx(0.0787299150672875, rel=1e-12)
    assert result.value == pytest.approx(0.9874930034039904, rel=1e-12)
    # The inputs travel with the number: directive §6.7 requires the trial count
    # to be displayed alongside the result, and a DSR without one is not
    # interpretable.
    assert result.trials == 10
    assert result.trial_sharpe_variance == 0.0025
    assert result.observed_sharpe == 0.15
    assert result.n_observations == 1001
    assert result.skewness == 0.0
    assert result.kurtosis == 3.0


def test_the_deflated_sharpe_ratio_is_strictly_decreasing_in_the_trial_count() -> None:
    # This is the whole point of the statistic: the same observed Sharpe ratio is
    # worth less the harder you searched for it.
    values = [
        deflated_sharpe_ratio(
            observed_sharpe=0.15,
            n_observations=1001,
            skewness=0.0,
            kurtosis=3.0,
            trials=n,
            trial_sharpe_variance=0.0025,
        ).value
        for n in (1, 2, 5, 10, 25, 50, 100, 200, 500, 1000)
    ]
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)
    # And the gap is not cosmetic: pretending one thing was tried when 200 were
    # turns this result from "certain" into "coin flip".
    assert values[0] > 0.99
    assert values[-2] < 0.6


def test_a_single_trial_leaves_the_probabilistic_sharpe_ratio_undeflated() -> None:
    undeflated = probabilistic_sharpe_ratio(
        observed_sharpe=0.15,
        benchmark_sharpe=0.0,
        n_observations=1001,
        skewness=0.0,
        kurtosis=3.0,
    )
    deflated = deflated_sharpe_ratio(
        observed_sharpe=0.15,
        n_observations=1001,
        skewness=0.0,
        kurtosis=3.0,
        trials=1,
        trial_sharpe_variance=0.0,
    )
    assert deflated.value == pytest.approx(undeflated, rel=1e-15)
    assert deflated.expected_maximum_sharpe == 0.0


def _equal_mean_and_variance_samples(
    *, mean: float, deviation: float, size: int, tail_fraction: float
) -> tuple[list[float], list[float]]:
    """Build two samples with identical mean and variance but different kurtosis.

    The thin one is a two-point symmetric sample at ``mean +- deviation``:
    population variance ``deviation ** 2`` and kurtosis exactly 1, the lowest any
    distribution can attain. The fat one puts a ``tail_fraction`` of its mass at
    ``mean +- deviation / sqrt(tail_fraction)`` and the rest exactly at ``mean``:
    the same population variance by construction, and kurtosis exactly
    ``1 / tail_fraction``.
    """
    half = size // 2
    thin = [mean + deviation] * half + [mean - deviation] * half

    tail = round(size * tail_fraction)
    amplitude = deviation / math.sqrt(tail_fraction)
    fat = (
        [mean + amplitude] * (tail // 2) + [mean - amplitude] * (tail // 2) + [mean] * (size - tail)
    )
    return thin, fat


def test_fat_tails_lower_the_dsr_at_an_identical_mean_and_variance() -> None:
    # Two 100-observation samples, both with mean 0.001 and population standard
    # deviation 0.01, so both have exactly the same Sharpe ratio and exactly the
    # same skewness (zero). They differ only in kurtosis: 1 against 5.
    thin, fat = _equal_mean_and_variance_samples(
        mean=0.001, deviation=0.01, size=100, tail_fraction=0.2
    )

    assert sharpe_ratio(thin) == pytest.approx(sharpe_ratio(fat), rel=1e-12)
    assert skewness(thin) == pytest.approx(0.0, abs=1e-12)
    assert skewness(fat) == pytest.approx(0.0, abs=1e-12)
    assert kurtosis(thin) == pytest.approx(1.0, rel=1e-12)
    assert kurtosis(fat) == pytest.approx(5.0, rel=1e-12)  # 1 / 0.2

    # A benchmark below the observed Sharpe ratio — the regime where a result is
    # actually being certified, and where a wider estimator standard error is
    # unambiguously bad news.
    thin_result = deflated_sharpe_ratio_from_returns(thin, trials=5, trial_sharpe_variance=0.0004)
    fat_result = deflated_sharpe_ratio_from_returns(fat, trials=5, trial_sharpe_variance=0.0004)
    assert thin_result.observed_sharpe == pytest.approx(fat_result.observed_sharpe, rel=1e-12)
    assert thin_result.expected_maximum_sharpe == fat_result.expected_maximum_sharpe
    assert thin_result.observed_sharpe > thin_result.expected_maximum_sharpe
    assert fat_result.value < thin_result.value


def test_zero_trial_variance_with_many_trials_is_refused_rather_than_silently_undeflated() -> None:
    # With V = 0 the deflation term vanishes however many trials are declared,
    # so the "deflated" number would silently equal the undeflated PSR. That is
    # almost always missing data, and it is exactly the failure this statistic
    # exists to prevent.
    with pytest.raises(ValueError, match="deflation term would vanish"):
        deflated_sharpe_ratio(
            observed_sharpe=0.15,
            n_observations=1001,
            skewness=0.0,
            kurtosis=3.0,
            trials=200,
            trial_sharpe_variance=0.0,
        )


# ---------------------------------------------------------------------------
# The trial count is required — never defaulted to 1
# ---------------------------------------------------------------------------

_REQUIRED_PARAMETERS = frozenset({"trials", "trial_sharpe_variance", "trial_sharpes"})


@pytest.mark.parametrize("name", sorted(dsr_module.__all__))
def test_no_public_dsr_entry_point_defaults_the_trial_count(name: str) -> None:
    # A DSR computed as though one configuration was tried when two hundred were
    # is the most misleading number this system can emit. It must be impossible
    # to obtain one by omission, so this walks every exported callable and fails
    # if a future edit ever gives these parameters a default.
    attribute = getattr(dsr_module, name)
    if not callable(attribute) or isinstance(attribute, type):
        return
    signature = inspect.signature(attribute)
    for parameter_name, parameter in signature.parameters.items():
        if parameter_name in _REQUIRED_PARAMETERS:
            assert parameter.default is inspect.Parameter.empty, (
                f"{name}({parameter_name}=...) has acquired a default; the trial count "
                "must always come from the caller"
            )
            assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
                f"{name}({parameter_name}=...) must be keyword-only so it cannot be "
                "supplied positionally by accident"
            )


def test_omitting_the_trial_count_is_an_error_rather_than_an_implicit_one() -> None:
    with pytest.raises(TypeError, match="trials"):
        deflated_sharpe_ratio(  # type: ignore[call-arg]
            observed_sharpe=0.15,
            n_observations=1001,
            skewness=0.0,
            kurtosis=3.0,
        )
    with pytest.raises(TypeError, match="trials"):
        deflated_sharpe_ratio_from_returns([0.01, -0.02, 0.03, 0.005])  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# The convenience entry points
# ---------------------------------------------------------------------------


def test_from_returns_derives_the_same_moments_the_explicit_form_expects() -> None:
    returns = np.random.default_rng(20260801).normal(0.0005, 0.01, size=750)
    derived = deflated_sharpe_ratio_from_returns(returns, trials=40, trial_sharpe_variance=0.0009)
    explicit = deflated_sharpe_ratio(
        observed_sharpe=sharpe_ratio(returns),
        n_observations=750,
        skewness=skewness(returns),
        kurtosis=kurtosis(returns),
        trials=40,
        trial_sharpe_variance=0.0009,
    )
    assert derived == explicit


def test_from_returns_never_annualizes_the_sharpe_it_deflates() -> None:
    # The deflation formula is written in per-period units. An annualized Sharpe
    # ratio fed into it inflates the answer by roughly sqrt(252) with nothing
    # downstream complaining, so the entry point must read the per-period value.
    returns = np.random.default_rng(5).normal(0.0005, 0.01, size=500)
    result = deflated_sharpe_ratio_from_returns(returns, trials=10, trial_sharpe_variance=0.0004)
    assert result.observed_sharpe == pytest.approx(sharpe_ratio(returns), rel=1e-15)
    assert result.observed_sharpe != pytest.approx(
        sharpe_ratio(returns, periods_per_year=252), rel=1e-6
    )


def test_from_trials_uses_the_row_count_and_the_sample_variance_of_the_trials() -> None:
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0004, 0.01, size=600)
    observed = sharpe_ratio(returns)
    # A trial set that contains the reported strategy and 29 weaker ones.
    trial_sharpes = np.concatenate([[observed], observed - rng.uniform(0.001, 0.05, size=29)])

    result = deflated_sharpe_ratio_from_trials(returns, trial_sharpes=trial_sharpes)
    assert result.trials == 30
    assert result.trial_sharpe_variance == pytest.approx(
        float(np.var(trial_sharpes, ddof=1)), rel=1e-15
    )
    assert result.observed_sharpe == pytest.approx(observed, rel=1e-15)


def test_from_trials_refuses_a_strategy_better_than_every_recorded_trial() -> None:
    # If the reported strategy is not in the trial set, the trial count
    # understates the search — the exact condition directive §9.7 exists to
    # prevent — or the two are in different units. Either way the deflation
    # would be wrong.
    returns = np.random.default_rng(3).normal(0.002, 0.01, size=400)
    with pytest.raises(ValueError, match="exceeds the best recorded trial"):
        deflated_sharpe_ratio_from_trials(returns, trial_sharpes=[-0.5, -0.2, 0.0])


def test_from_trials_accepts_the_reported_strategy_as_the_best_recorded_trial() -> None:
    returns = np.random.default_rng(4).normal(0.001, 0.01, size=400)
    observed = sharpe_ratio(returns)
    result = deflated_sharpe_ratio_from_trials(
        returns, trial_sharpes=[observed, observed - 0.01, observed - 0.05]
    )
    assert result.trials == 3


# ---------------------------------------------------------------------------
# Properties — checked over randomized *structure*, not one fixed shape
# ---------------------------------------------------------------------------

_sharpes = st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False)


@given(
    observed=_sharpes,
    n_observations=st.integers(min_value=2, max_value=10_000),
    skew=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    kurt=st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    variance=st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False),
    trial_pairs=st.lists(st.integers(min_value=1, max_value=100_000), min_size=2, max_size=6),
)
@hypothesis_settings(max_examples=300, deadline=None)
def test_the_dsr_never_rises_when_more_trials_are_declared(
    observed: float,
    n_observations: int,
    skew: float,
    kurt: float,
    variance: float,
    trial_pairs: list[int],
) -> None:
    """More search can only ever make the same result less impressive."""
    values = [
        deflated_sharpe_ratio(
            observed_sharpe=observed,
            n_observations=n_observations,
            skewness=skew,
            kurtosis=kurt,
            trials=trials,
            trial_sharpe_variance=variance,
        ).value
        for trials in sorted(trial_pairs)
    ]
    assert values == sorted(values, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in values)


@given(
    returns=st.lists(
        st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False),
        min_size=8,
        max_size=200,
    ),
    trials=st.integers(min_value=2, max_value=5_000),
    variance=st.floats(min_value=1e-6, max_value=0.5, allow_nan=False, allow_infinity=False),
)
@hypothesis_settings(max_examples=300, deadline=None)
def test_the_dsr_of_any_return_series_is_a_probability_or_a_refusal(
    returns: list[float], trials: int, variance: float
) -> None:
    """No input may produce a number outside [0, 1], a NaN, or an infinity."""
    try:
        result = deflated_sharpe_ratio_from_returns(
            returns, trials=trials, trial_sharpe_variance=variance
        )
    except ValueError:
        # Degenerate samples (zero variance, or moments the formula cannot
        # accept) are refused. A refusal is a correct answer; a NaN is not.
        return
    assert 0.0 <= result.value <= 1.0
    assert math.isfinite(result.value)
    assert math.isfinite(result.expected_maximum_sharpe)
