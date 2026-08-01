"""Trailing-volatility estimator: alignment, units, and the trailing property."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.labels.errors import LabelConfigurationError, LabelInputError
from backend.labels.volatility import daily_log_returns, trailing_volatility
from backend.tests.labels.paths import prices_from_log_returns


class TestDailyLogReturns:
    def test_first_bar_has_no_return(self) -> None:
        returns = daily_log_returns([100.0, 101.0, 102.0])
        assert math.isnan(returns[0])

    def test_returns_are_natural_log_ratios(self) -> None:
        prices = prices_from_log_returns([0.02, -0.01, 0.005])
        returns = daily_log_returns(prices)
        assert returns[1:] == pytest.approx([0.02, -0.01, 0.005])

    def test_length_matches_the_price_series(self) -> None:
        assert daily_log_returns([1.0, 2.0, 3.0, 4.0]).shape == (4,)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_or_missing_prices_are_refused(self, bad: float) -> None:
        with pytest.raises(LabelInputError, match="strictly positive"):
            daily_log_returns([100.0, bad, 102.0])

    def test_two_dimensional_input_is_refused_rather_than_flattened(self) -> None:
        with pytest.raises(LabelInputError, match="one-dimensional"):
            daily_log_returns([[100.0, 101.0], [102.0, 103.0]])


class TestTrailingVolatility:
    def test_matches_the_hand_computed_sample_standard_deviation(self) -> None:
        # Two returns r1, r2 have sample std |r1 - r2| / sqrt(2) with ddof=1.
        returns = np.array([np.nan, 0.03, 0.01])
        expected = abs(0.03 - 0.01) / math.sqrt(2.0)
        assert trailing_volatility(returns, window=2)[2] == pytest.approx(expected)

    def test_uses_ddof_one_not_the_population_estimator(self) -> None:
        returns = np.array([np.nan, 0.02, -0.02, 0.02, -0.02])
        # Population std of four +/-0.02 values is 0.02; the sample std is
        # 0.02 * sqrt(4/3). The two differ by 15% here, so this is not a
        # cosmetic distinction.
        assert trailing_volatility(returns, window=4)[4] == pytest.approx(0.02 * math.sqrt(4 / 3))

    def test_no_partial_window_estimate_is_produced(self) -> None:
        returns = np.array([np.nan, 0.01, -0.01, 0.02, -0.02])
        volatility = trailing_volatility(returns, window=4)
        assert np.all(np.isnan(volatility[:4]))
        assert math.isfinite(volatility[4])

    def test_a_short_series_is_all_nan_rather_than_an_error(self) -> None:
        volatility = trailing_volatility(np.array([np.nan, 0.01]), window=5)
        assert volatility.shape == (2,)
        assert np.all(np.isnan(volatility))

    def test_estimate_at_t_uses_only_bars_up_to_t(self) -> None:
        returns = np.array([np.nan, 0.01, -0.01, 0.02, -0.02, 0.03])
        baseline = trailing_volatility(returns, window=3)
        altered = returns.copy()
        altered[4:] = [5.0, -5.0]  # violent moves strictly after bar 3
        assert trailing_volatility(altered, window=3)[3] == pytest.approx(baseline[3])

    def test_estimate_at_t_does_include_the_return_realized_into_t(self) -> None:
        # The return into bar t is known at bar t's close, so it belongs in the
        # estimate. Changing it must change sigma[t] — the complement of the
        # test above, and what keeps that one from passing vacuously.
        returns = np.array([np.nan, 0.01, -0.01, 0.02])
        baseline = trailing_volatility(returns, window=3)[3]
        altered = returns.copy()
        altered[3] = 0.09
        assert trailing_volatility(altered, window=3)[3] != pytest.approx(baseline)

    def test_scaling_every_return_scales_volatility_by_the_same_factor(self) -> None:
        returns = np.array([np.nan, 0.01, -0.02, 0.015, -0.005, 0.02])
        base = trailing_volatility(returns, window=3)
        doubled = trailing_volatility(returns * 2.0, window=3)
        finite = ~np.isnan(base)
        assert doubled[finite] == pytest.approx(2.0 * base[finite])

    def test_a_missing_return_inside_the_window_poisons_the_estimate(self) -> None:
        returns = np.array([np.nan, 0.01, np.nan, 0.02, -0.01, 0.015])
        volatility = trailing_volatility(returns, window=3)
        assert np.all(np.isnan(volatility[:5]))
        assert math.isfinite(volatility[5])

    @pytest.mark.parametrize("window", [-1, 0, 1])
    def test_a_window_below_two_is_refused(self, window: int) -> None:
        with pytest.raises(LabelConfigurationError, match="window must be >= 2"):
            trailing_volatility(np.array([np.nan, 0.01, 0.02]), window=window)

    def test_a_flat_window_gives_zero_not_an_error(self) -> None:
        # The estimator's job is to report zero; refusing to label a
        # zero-volatility event is the barrier engine's job, and keeping the
        # two separate is what lets that refusal be tested on its own.
        assert trailing_volatility(np.array([np.nan, 0.0, 0.0, 0.0]), window=3)[3] == 0.0
