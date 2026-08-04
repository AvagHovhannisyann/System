"""Residualization: exact beta recovery, frozen forward application, and refusals.

The regression is tested on an **orthogonal design** rather than on noisy data.
Four mutually orthogonal ±1 patterns of length 8 (an intercept, a market factor,
a sector factor and an idiosyncratic term) make the ordinary-least-squares
solution exact in closed form: the fitted coefficients must come back as the
ones that were put in, to floating-point precision, and the in-sample residuals
must equal the idiosyncratic term exactly. A test on noisy data could only have
asserted "approximately", and would have passed with a subtly wrong estimator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.labels.barriers import (
    BarrierSpec,
    LabelBasis,
    TripleBarrierOutcome,
    triple_barrier_labels,
)
from backend.labels.errors import (
    DegenerateVolatilityError,
    InsufficientHistoryError,
    LabelConfigurationError,
    LabelInputError,
    RankDeficientFactorError,
)
from backend.labels.residualize import (
    ResidualSpec,
    fit_residual_model,
    residualized_triple_barrier_labels,
    usable_residual_event_indices,
)
from backend.labels.volatility import daily_log_returns
from backend.tests.labels.paths import factor_series, prices_from_log_returns

# Four mutually orthogonal ±1 vectors of length 8. Orthogonality is what makes
# the least-squares solution exact: each coefficient is an independent
# projection, so OLS returns precisely the coefficients used to build the data.
ONES = np.ones(8)
MARKET_PATTERN = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
SECTOR_PATTERN = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
IDIOSYNCRATIC_PATTERN = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

TRUE_ALPHA = 0.0005
TRUE_BETA_MARKET = 1.5
TRUE_BETA_SECTOR = 0.5
MARKET_SCALE = 0.01
SECTOR_SCALE = 0.008
IDIOSYNCRATIC_SCALE = 0.004

EVENT_BAR = 8
HORIZON = 3
WINDOW = 8
SPEC = BarrierSpec(horizon=HORIZON, volatility_window=WINDOW)
RESIDUAL_SPEC = ResidualSpec(estimation_window=WINDOW)


def test_the_patterns_really_are_orthogonal() -> None:
    """Guard the premise of every exactness assertion in this module."""
    columns = (ONES, MARKET_PATTERN, SECTOR_PATTERN, IDIOSYNCRATIC_PATTERN)
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            assert float(columns[left] @ columns[right]) == 0.0


def _build_series(
    *,
    forward_market: list[float],
    forward_sector: list[float],
    forward_idiosyncratic: list[float],
    extra_market_loading: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build asset/market/sector closes with a known factor structure.

    Args:
        forward_market: market log returns after the event (fractions).
        forward_sector: sector log returns after the event.
        forward_idiosyncratic: idiosyncratic log returns after the event; the
            forward residual path is exactly the cumulative sum of these.
        extra_market_loading: added to the asset's market beta throughout, used
            to show that the residual is invariant to it.

    Returns:
        ``(asset_close, market_close, sector_close)``.
    """
    market = np.concatenate((MARKET_SCALE * MARKET_PATTERN, forward_market))
    sector = np.concatenate((SECTOR_SCALE * SECTOR_PATTERN, forward_sector))
    idiosyncratic = np.concatenate(
        (IDIOSYNCRATIC_SCALE * IDIOSYNCRATIC_PATTERN, forward_idiosyncratic)
    )
    in_sample_alpha = np.concatenate((np.full(8, TRUE_ALPHA), np.zeros(len(forward_market))))
    asset = (
        in_sample_alpha
        + (TRUE_BETA_MARKET + extra_market_loading) * market
        + TRUE_BETA_SECTOR * sector
        + idiosyncratic
    )
    return (
        prices_from_log_returns(asset),
        prices_from_log_returns(market),
        prices_from_log_returns(sector),
    )


class TestExactFit:
    def test_the_fitted_coefficients_are_the_ones_that_built_the_data(self) -> None:
        asset, market, sector = _build_series(
            forward_market=[0.0] * HORIZON,
            forward_sector=[0.0] * HORIZON,
            forward_idiosyncratic=[0.0] * HORIZON,
        )
        fit = fit_residual_model(
            daily_log_returns(asset),
            daily_log_returns(market),
            daily_log_returns(sector),
            event_index=EVENT_BAR,
            residual_spec=RESIDUAL_SPEC,
            volatility_window=WINDOW,
        )
        assert fit.alpha == pytest.approx(TRUE_ALPHA)
        assert fit.beta_market == pytest.approx(TRUE_BETA_MARKET)
        assert fit.beta_sector == pytest.approx(TRUE_BETA_SECTOR)
        assert fit.n_observations == WINDOW

    def test_residual_volatility_is_the_idiosyncratic_standard_deviation(self) -> None:
        asset, market, sector = _build_series(
            forward_market=[0.0] * HORIZON,
            forward_sector=[0.0] * HORIZON,
            forward_idiosyncratic=[0.0] * HORIZON,
        )
        fit = fit_residual_model(
            daily_log_returns(asset),
            daily_log_returns(market),
            daily_log_returns(sector),
            event_index=EVENT_BAR,
            residual_spec=RESIDUAL_SPEC,
            volatility_window=WINDOW,
        )
        # Eight alternating +/-s values have sample std s * sqrt(8/7).
        expected = IDIOSYNCRATIC_SCALE * math.sqrt(8 / 7)
        assert fit.residual_volatility == pytest.approx(expected)

    def test_residual_volatility_is_far_below_total_volatility(self) -> None:
        # The point of sizing the barrier in idiosyncratic units: for this
        # security most of the variance is factor variance, and a barrier placed
        # at total volatility would be roughly four times too wide.
        asset, market, sector = _build_series(
            forward_market=[0.0] * HORIZON,
            forward_sector=[0.0] * HORIZON,
            forward_idiosyncratic=[0.0] * HORIZON,
        )
        fit = fit_residual_model(
            daily_log_returns(asset),
            daily_log_returns(market),
            daily_log_returns(sector),
            event_index=EVENT_BAR,
            residual_spec=RESIDUAL_SPEC,
            volatility_window=WINDOW,
        )
        raw = triple_barrier_labels(asset, [EVENT_BAR], SPEC)
        assert fit.residual_volatility < 0.4 * float(raw.trailing_volatility[0])

    def test_the_conditioning_of_an_orthogonal_design_is_one(self) -> None:
        asset, market, sector = _build_series(
            forward_market=[0.0] * HORIZON,
            forward_sector=[0.0] * HORIZON,
            forward_idiosyncratic=[0.0] * HORIZON,
        )
        fit = fit_residual_model(
            daily_log_returns(asset),
            daily_log_returns(market),
            daily_log_returns(sector),
            event_index=EVENT_BAR,
            residual_spec=RESIDUAL_SPEC,
            volatility_window=WINDOW,
        )
        assert fit.condition_number == pytest.approx(1.0)


class TestTheLabelMeasuresIdiosyncraticMovement:
    def test_a_pure_factor_move_produces_no_idiosyncratic_label(self) -> None:
        # A big market rally that the stock follows exactly at its beta. The raw
        # price path clears the upper barrier; the residual path never moves.
        # This is the whole reason P6.2 exists.
        asset, market, sector = _build_series(
            forward_market=[0.03, 0.0, 0.0],
            forward_sector=[0.0] * HORIZON,
            forward_idiosyncratic=[0.0] * HORIZON,
        )
        raw = triple_barrier_labels(asset, [EVENT_BAR], SPEC)
        residual = residualized_triple_barrier_labels(
            asset, market, sector, [EVENT_BAR], SPEC, RESIDUAL_SPEC
        )
        assert raw.outcome[0] == TripleBarrierOutcome.UPPER_FIRST
        assert residual.labels.outcome[0] == TripleBarrierOutcome.VERTICAL
        assert residual.labels.realized_log_return[0] == pytest.approx(0.0, abs=1e-15)
        assert residual.labels.basis is LabelBasis.RESIDUAL

    def test_an_idiosyncratic_move_of_the_same_size_does_produce_a_label(self) -> None:
        # Non-vacuity for the test above: the residual path must still be able
        # to reach a barrier. The idiosyncratic barrier here is
        # 0.004*sqrt(8/7)*sqrt(3) = 0.0074, so +0.02 clears it on bar 1.
        asset, market, sector = _build_series(
            forward_market=[0.0] * HORIZON,
            forward_sector=[0.0] * HORIZON,
            forward_idiosyncratic=[0.02, 0.0, 0.0],
        )
        residual = residualized_triple_barrier_labels(
            asset, market, sector, [EVENT_BAR], SPEC, RESIDUAL_SPEC
        )
        assert residual.labels.outcome[0] == TripleBarrierOutcome.UPPER_FIRST
        assert residual.labels.resolution_index[0] == EVENT_BAR + 1
        assert residual.labels.realized_log_return[0] == pytest.approx(0.02)

    def test_the_forward_residual_is_the_frozen_betas_applied_to_future_returns(self) -> None:
        asset, market, sector = _build_series(
            forward_market=[0.01, -0.02, 0.005],
            forward_sector=[-0.004, 0.006, 0.0],
            forward_idiosyncratic=[0.001, -0.002, 0.0015],
        )
        residual = residualized_triple_barrier_labels(
            asset, market, sector, [EVENT_BAR], SPEC, RESIDUAL_SPEC
        )
        # Nothing touches a barrier, so the label runs to the deadline and its
        # realized return is the full cumulative residual: exactly the sum of
        # the idiosyncratic terms that were put in.
        assert residual.labels.outcome[0] == TripleBarrierOutcome.VERTICAL
        assert residual.labels.realized_log_return[0] == pytest.approx(0.001 - 0.002 + 0.0015)

    @pytest.mark.parametrize("extra_loading", [0.5, -0.75, 3.0])
    def test_the_label_is_invariant_to_the_stock_s_beta(self, extra_loading: float) -> None:
        # Adding c * market to the asset's returns shifts the fitted market beta
        # by exactly c and leaves the residual — and therefore the label —
        # untouched. If beta leaked into the label, this would move it.
        forward_market = [0.01, -0.02, 0.005]
        forward_sector = [-0.004, 0.006, 0.0]
        forward_idiosyncratic = [0.006, -0.002, 0.0015]
        base_asset, market, sector = _build_series(
            forward_market=forward_market,
            forward_sector=forward_sector,
            forward_idiosyncratic=forward_idiosyncratic,
        )
        tilted_asset, _, _ = _build_series(
            forward_market=forward_market,
            forward_sector=forward_sector,
            forward_idiosyncratic=forward_idiosyncratic,
            extra_market_loading=extra_loading,
        )

        base = residualized_triple_barrier_labels(
            base_asset, market, sector, [EVENT_BAR], SPEC, RESIDUAL_SPEC
        )
        tilted = residualized_triple_barrier_labels(
            tilted_asset, market, sector, [EVENT_BAR], SPEC, RESIDUAL_SPEC
        )
        assert tilted.beta_market[0] == pytest.approx(base.beta_market[0] + extra_loading)
        assert tilted.beta_sector[0] == pytest.approx(base.beta_sector[0])
        assert tilted.labels.outcome[0] == base.labels.outcome[0]
        assert tilted.labels.realized_log_return[0] == pytest.approx(
            base.labels.realized_log_return[0]
        )
        assert tilted.labels.trailing_volatility[0] == pytest.approx(
            base.labels.trailing_volatility[0]
        )


class TestRefusals:
    def _wave_series(self, n_bars: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return factor_series(n_bars)

    def test_a_sector_series_identical_to_the_market_is_refused(self) -> None:
        asset, market, _ = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(RankDeficientFactorError, match="condition number"):
            residualized_triple_barrier_labels(asset, market, market, [60], spec, residual_spec)

    def test_a_near_duplicate_sector_series_is_also_refused(self) -> None:
        # Not literally equal: the sector is the market plus a nine-orders-of-
        # magnitude-smaller wiggle. The design is technically full rank, which
        # is exactly why a rank check alone would let this through and hand back
        # meaningless betas.
        asset, market, _ = self._wave_series()
        nudge = 1.0 + 1e-11 * np.sin(np.arange(market.size))
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(RankDeficientFactorError):
            residualized_triple_barrier_labels(
                asset, market, market * nudge, [60], spec, residual_spec
            )

    def test_a_constant_factor_is_refused(self) -> None:
        asset, market, _ = self._wave_series()
        flat = np.full(market.size, 50.0)
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(RankDeficientFactorError):
            residualized_triple_barrier_labels(asset, market, flat, [60], spec, residual_spec)

    def test_a_perfect_in_sample_fit_is_refused_rather_than_labelled(self) -> None:
        # asset == market exactly: every residual is zero, the barrier has zero
        # width, and every bar would "touch" it.
        _, market, sector = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(DegenerateVolatilityError):
            residualized_triple_barrier_labels(market, market, sector, [60], spec, residual_spec)

    def test_an_asset_that_is_a_combination_of_the_factors_is_refused(self) -> None:
        # The general form of the case above, and the one that actually happens:
        # a security with no idiosyncratic component at all leaves residuals of
        # order 1e-16, which are a *positive* volatility and would size a barrier
        # resolved entirely by the sign of the rounding error.
        _, market, sector = self._wave_series()
        combination = prices_from_log_returns(
            1.3 * daily_log_returns(market)[1:] + 0.7 * daily_log_returns(sector)[1:]
        )
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(DegenerateVolatilityError, match="rounding error"):
            residualized_triple_barrier_labels(
                combination, market, sector, [60], spec, residual_spec
            )

    def test_an_event_without_the_full_estimation_window_is_refused(self) -> None:
        asset, market, sector = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(InsufficientHistoryError, match="fewer than the 40"):
            residualized_triple_barrier_labels(asset, market, sector, [39], spec, residual_spec)

    def test_an_event_without_the_full_horizon_is_refused(self) -> None:
        asset, market, sector = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        last = asset.size - 1
        with pytest.raises(InsufficientHistoryError, match="past the end of the series"):
            residualized_triple_barrier_labels(
                asset, market, sector, [last - 2], spec, residual_spec
            )

    def test_mismatched_series_lengths_are_refused(self) -> None:
        asset, market, sector = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=10)
        with pytest.raises(LabelInputError, match="same length"):
            residualized_triple_barrier_labels(asset, market[:-1], sector, [60], spec)

    def test_no_events_gives_an_empty_result_not_an_error(self) -> None:
        asset, market, sector = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=10)
        result = residualized_triple_barrier_labels(asset, market, sector, [], spec)
        assert result.labels.n_labels == 0
        assert result.beta_market.shape == (0,)
        assert result.condition_number.shape == (0,)

    def test_a_volatility_window_longer_than_the_estimation_window_is_refused(self) -> None:
        asset, market, sector = self._wave_series()
        spec = BarrierSpec(horizon=5, volatility_window=50)
        residual_spec = ResidualSpec(estimation_window=40)
        with pytest.raises(LabelConfigurationError, match="cannot exceed estimation_window"):
            residualized_triple_barrier_labels(asset, market, sector, [60], spec, residual_spec)


class TestResidualSpecValidation:
    @pytest.mark.parametrize("window", [0, 1, 4])
    def test_too_short_an_estimation_window_is_refused(self, window: int) -> None:
        with pytest.raises(LabelConfigurationError, match="estimation_window must be"):
            ResidualSpec(estimation_window=window)

    @pytest.mark.parametrize("limit", [0.0, 1.0, -5.0, math.nan, math.inf])
    def test_an_impossible_condition_number_limit_is_refused(self, limit: float) -> None:
        with pytest.raises(LabelConfigurationError, match="condition_number_limit"):
            ResidualSpec(condition_number_limit=limit)

    def test_the_default_window_is_a_trading_year(self) -> None:
        assert ResidualSpec().estimation_window == 252


class TestUsableResidualEventIndices:
    def test_it_respects_the_longer_trailing_requirement(self) -> None:
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        usable = usable_residual_event_indices(60, spec, residual_spec)
        assert usable.tolist() == list(range(40, 55))

    def test_every_selected_bar_actually_labels(self) -> None:
        spec = BarrierSpec(horizon=5, volatility_window=10)
        residual_spec = ResidualSpec(estimation_window=40)
        prices = factor_series(80)
        usable = usable_residual_event_indices(prices[0].size, spec, residual_spec)
        assert usable.size > 0
        result = residualized_triple_barrier_labels(*prices, usable, spec, residual_spec)
        assert result.labels.n_labels == usable.size
        assert result.labels.n_ambiguous == 0, "close-only labelling cannot be ambiguous"
