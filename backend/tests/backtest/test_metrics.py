"""Known-answer tests for the shared return statistics.

Every expected value here is computed by hand in the test body's comment, not by
calling a second implementation of the same formula. A statistics module checked
only against itself is checked against nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.backtest.metrics import (
    as_float_array,
    average_ranks,
    kurtosis,
    sharpe_ratio,
    skewness,
    standard_normal_cdf,
    standard_normal_ppf,
)


def test_sharpe_ratio_matches_hand_computation() -> None:
    # returns = [0.01, 0.02, 0.03, 0.04]
    #   mean       = 0.10 / 4                       = 0.025
    #   deviations = [-0.015, -0.005, 0.005, 0.015]
    #   sum sq     = 2.25e-4 + 0.25e-4 + 0.25e-4 + 2.25e-4 = 5.0e-4
    #   var (ddof=1) = 5.0e-4 / 3                   = 1.6666...e-4
    #   sd         = sqrt(1.6666e-4)                = 0.01290994448735806
    #   SR         = 0.025 / 0.01290994448735806    = 1.9364916731037085
    assert sharpe_ratio([0.01, 0.02, 0.03, 0.04]) == pytest.approx(1.9364916731037085, rel=1e-12)


def test_annualization_multiplies_by_the_square_root_of_the_period_count() -> None:
    returns = [0.01, 0.02, 0.03, 0.04]
    per_period = sharpe_ratio(returns)
    annual = sharpe_ratio(returns, periods_per_year=252)
    assert annual == pytest.approx(per_period * math.sqrt(252), rel=1e-12)


def test_the_risk_free_rate_shifts_the_numerator_only() -> None:
    returns = [0.01, 0.02, 0.03, 0.04]
    # Subtracting a constant moves the mean by that constant and leaves the
    # standard deviation untouched: SR = (0.025 - 0.005) / 0.01290994448735806.
    assert sharpe_ratio(returns, risk_free_rate=0.005) == pytest.approx(
        0.020 / 0.01290994448735806, rel=1e-12
    )


def test_a_constant_series_has_no_sharpe_ratio_rather_than_an_infinite_one() -> None:
    with pytest.raises(ValueError, match="zero standard deviation"):
        sharpe_ratio([0.01, 0.01, 0.01, 0.01])


def test_sharpe_ratio_rejects_a_sample_too_short_for_its_degrees_of_freedom() -> None:
    with pytest.raises(ValueError, match="more than ddof"):
        sharpe_ratio([0.01])


def test_sharpe_ratio_rejects_a_non_positive_annualization_factor() -> None:
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        sharpe_ratio([0.01, 0.02, 0.03], periods_per_year=0.0)


def test_skewness_matches_hand_computation() -> None:
    # x = [0, 0, 0, 4]; mean = 1; deviations = [-1, -1, -1, 3]
    #   m2 = (1 + 1 + 1 + 9) / 4 = 3
    #   m3 = (-1 - 1 - 1 + 27) / 4 = 6
    #   g3 = 6 / 3 ** 1.5 = 6 / 5.196152422706632 = 1.1547005383792515
    assert skewness([0.0, 0.0, 0.0, 4.0]) == pytest.approx(1.1547005383792515, rel=1e-12)


def test_skewness_of_a_symmetric_sample_is_zero() -> None:
    assert skewness([-2.0, -1.0, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-15)


def test_kurtosis_of_a_two_point_symmetric_sample_is_its_lower_bound_of_one() -> None:
    # x = [-1, 1, -1, 1]; m2 = 1, m4 = 1, so g4 = 1 — the minimum any
    # distribution can attain, which is why the PSR rejects g4 < 1.
    assert kurtosis([-1.0, 1.0, -1.0, 1.0]) == pytest.approx(1.0, rel=1e-12)


def test_kurtosis_is_non_excess_so_a_gaussian_sample_sits_near_three() -> None:
    sample = np.random.default_rng(20260801).normal(size=200_000)
    assert kurtosis(sample) == pytest.approx(3.0, abs=0.05)
    # The excess convention would give ~0 here; the distinction is the whole
    # reason the function is named for the non-excess one.
    assert kurtosis(sample) > 2.0


def test_the_moments_are_scale_free_across_extreme_magnitudes() -> None:
    # g3 and g4 are ratios of central moments chosen precisely so that rescaling
    # the sample leaves them unchanged. Evaluating them as m3 / m2 ** 1.5 and
    # m4 / m2 ** 2 does not preserve that: at 1e-100 the squared variance
    # underflows to exactly zero and the division fails, and at 1e100 it
    # overflows to infinity and the ratio is a NaN. Hypothesis found the first of
    # those against the original implementation.
    base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]
    reference_skew = skewness(base)
    reference_kurt = kurtosis(base)
    for factor in (1e-100, 1e-30, 1e-3, 1e3, 1e30, 1e100):
        scaled = [value * factor for value in base]
        assert skewness(scaled) == pytest.approx(reference_skew, rel=1e-12)
        assert kurtosis(scaled) == pytest.approx(reference_kurt, rel=1e-12)
    # Shifting the sample must not move them either.
    shifted = [value + 5.0 for value in base]
    assert skewness(shifted) == pytest.approx(reference_skew, rel=1e-12)
    assert kurtosis(shifted) == pytest.approx(reference_kurt, rel=1e-12)


def test_the_sharpe_ratio_refuses_a_series_that_overflows_the_double_range() -> None:
    # A NaN Sharpe ratio renders as a blank cell rather than as an error, which
    # is the failure mode this project cannot afford.
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="not finite"):
        sharpe_ratio([1e308, -1e308, 1e308, -1e308])


def test_moments_of_a_constant_series_are_refused() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        skewness([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="zero variance"):
        kurtosis([1.0, 1.0, 1.0])


def test_standard_normal_functions_hit_textbook_values() -> None:
    assert standard_normal_cdf(0.0) == pytest.approx(0.5, abs=1e-15)
    assert standard_normal_cdf(1.959963984540054) == pytest.approx(0.975, abs=1e-12)
    assert standard_normal_ppf(0.975) == pytest.approx(1.959963984540054, abs=1e-9)
    assert standard_normal_ppf(0.5) == pytest.approx(0.0, abs=1e-15)


def test_the_inverse_normal_refuses_the_endpoints_instead_of_returning_infinity() -> None:
    for probability in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="requires 0 < p < 1"):
            standard_normal_ppf(probability)


def test_average_ranks_are_one_based_ascending_and_share_ties() -> None:
    ranks = average_ranks([10.0, 30.0, 20.0])
    assert ranks.tolist() == [1.0, 3.0, 2.0]

    tied = average_ranks([10.0, 20.0, 20.0, 30.0])
    # The two 20s occupy ranks 2 and 3, so both take 2.5.
    assert tied.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_average_ranks_always_sum_to_the_triangular_number() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        values = rng.integers(0, 5, size=12).astype(np.float64)  # many ties on purpose
        ranks = average_ranks(values)
        assert float(ranks.sum()) == pytest.approx(12 * 13 / 2)


def test_the_closed_form_rank_used_by_pbo_agrees_with_average_ranks() -> None:
    # backend.backtest.pbo computes the in-sample winner's out-of-sample rank as
    #     (# strictly worse) + (# tied + 1) / 2
    # to stay vectorized. That shortcut must equal the general average rank.
    rng = np.random.default_rng(11)
    for _ in range(200):
        values = rng.integers(0, 4, size=9).astype(np.float64)
        ranks = average_ranks(values)
        for index, value in enumerate(values):
            strictly_worse = int(np.count_nonzero(values < value))
            tied = int(np.count_nonzero(values == value))
            assert ranks[index] == pytest.approx(strictly_worse + (tied + 1) / 2.0)


def test_array_validation_rejects_the_shapes_that_would_silently_produce_nan() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        as_float_array([[1.0, 2.0]], name="returns")
    with pytest.raises(ValueError, match="non-empty"):
        as_float_array([], name="returns")
    with pytest.raises(ValueError, match="finite"):
        as_float_array([1.0, float("nan")], name="returns")
    with pytest.raises(ValueError, match="finite"):
        as_float_array([1.0, float("inf")], name="returns")


def test_array_validation_names_the_offending_argument() -> None:
    with pytest.raises(ValueError, match="trial_sharpes"):
        as_float_array([float("nan")], name="trial_sharpes")
