"""Tests for the buy-and-hold benchmark and the shared portfolio accounting.

Every expected number here is derived from the cost model's documented formulas
in a comment above the assertion, never from a second implementation of the
thing being tested. The formulas, once:

* half-spread and commission are flat basis points of traded notional;
* impact is ``impact_coefficient * daily_volatility_bps * sqrt(notional / adv)``
  basis points of traded notional;
* borrow accrues ACT/360 on the **short exposure held**, not on traded notional.

The load-bearing assertions are the ones about the *asymmetry* between the two
sides: the benchmark pays its own entry cost (so it is not handed a head start
over the strategy) and pays nothing thereafter (because it does not trade).
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.backtest.artifact import EquityCurve, RealizedAccounting
from backend.backtest.benchmark import (
    SECONDS_PER_DAY,
    BuyAndHoldSpec,
    ExecutionCosts,
    PortfolioWipeoutError,
    WeightError,
    accrue_borrow_usd,
    buy_and_hold,
    cost_weight_changes,
    drift_weights,
    gross_exposure,
    validate_weights,
)
from backend.costs.model import UNCALIBRATED_DEFAULTS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.costs.model import CostModelParams

CAPITAL = 1_000_000.0
ADV = 1_000_000_000.0

# One order of the full book at 0.1% of ADV:
#   participation = 1e6 / 1e9                  = 1e-3
#   impact        = 1.0 * 200 * sqrt(1e-3)     = 6.324555320336759 bps
FULL_BOOK_IMPACT_BPS = 200.0 * np.sqrt(1e-3)
FULL_BOOK_COST_BPS = 5.0 + 1.0 + FULL_BOOK_IMPACT_BPS
FULL_BOOK_COST_USD = FULL_BOOK_COST_BPS / 10_000.0 * CAPITAL


def dates(count: int) -> tuple[dt.datetime, ...]:
    """Return ``count`` consecutive daily UTC instants."""
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    return tuple(base + dt.timedelta(days=index) for index in range(count))


def free_params() -> CostModelParams:
    """Return a cost parameter set with every component zeroed.

    Used only to isolate the arithmetic under test from the cost model. No
    result computed with it is ever reported as a strategy result.
    """
    return UNCALIBRATED_DEFAULTS.with_parameters(
        half_spread_bps=0.0,
        commission_bps=0.0,
        impact_coefficient=0.0,
        default_daily_volatility_bps=0.0,
        borrow_rate_bps_per_year=0.0,
    )


def hold(
    weights: dict[str, float],
    asset_returns: dict[str, Sequence[float]],
    *,
    params: CostModelParams | None = None,
    n_periods: int = 3,
) -> tuple[EquityCurve, RealizedAccounting]:
    """Run ``buy_and_hold`` over ``n_periods`` with flat ADV and no volatility override."""
    return buy_and_hold(
        spec=BuyAndHoldSpec(weights=weights),
        dates=dates(n_periods + 1),
        asset_returns=asset_returns,
        adv_usd=dict.fromkeys(weights, ADV),
        daily_volatility_bps=dict.fromkeys(weights, None),
        initial_capital_usd=CAPITAL,
        params=params if params is not None else UNCALIBRATED_DEFAULTS,
    )


# ---------------------------------------------------------------------------
# Weight arithmetic
# ---------------------------------------------------------------------------


def test_gross_exposure_counts_both_sides_of_a_long_short_book() -> None:
    assert gross_exposure({"A": 0.6, "B": -0.4}) == pytest.approx(1.0)
    assert gross_exposure({}) == 0.0


def test_validate_weights_drops_zero_positions() -> None:
    cleaned = validate_weights(
        {"A": 0.5, "B": 0.0},
        known_assets=frozenset({"A", "B"}),
        max_gross_exposure=1.0,
        context="fixture",
    )
    assert cleaned == {"A": 0.5}


def test_validate_weights_refuses_an_asset_that_was_not_knowable() -> None:
    with pytest.raises(WeightError, match="lookahead"):
        validate_weights(
            {"GHOST": 0.5},
            known_assets=frozenset({"A"}),
            max_gross_exposure=1.0,
            context="fixture",
        )


def test_validate_weights_refuses_a_non_finite_weight() -> None:
    with pytest.raises(WeightError, match="not finite"):
        validate_weights(
            {"A": float("nan")},
            known_assets=frozenset({"A"}),
            max_gross_exposure=1.0,
            context="fixture",
        )


def test_validate_weights_refuses_leverage_at_the_default_cap() -> None:
    with pytest.raises(WeightError, match="gross exposure"):
        validate_weights(
            {"A": 0.8, "B": -0.4},
            known_assets=frozenset({"A", "B"}),
            max_gross_exposure=1.0,
            context="fixture",
        )


def test_a_fully_invested_single_name_book_needs_no_rebalancing_trade() -> None:
    # w(1 + r) / (1 + w r) with w = 1 is exactly 1, so a buy-and-hold of one
    # name never trades again. Getting this wrong manufactures turnover that
    # costs money in the simulation and never happened.
    drifted, portfolio_return = drift_weights({"A": 1.0}, {"A": 0.25})
    assert portfolio_return == pytest.approx(0.25)
    assert drifted == pytest.approx({"A": 1.0})


def test_drift_moves_weights_towards_the_winner() -> None:
    # r_p = 0.5*0.20 + 0.5*(-0.10) = 0.05
    # w_A = 0.5 * 1.20 / 1.05 = 0.5714285714...
    # w_B = 0.5 * 0.90 / 1.05 = 0.4285714285...
    drifted, portfolio_return = drift_weights({"A": 0.5, "B": 0.5}, {"A": 0.20, "B": -0.10})
    assert portfolio_return == pytest.approx(0.05)
    assert drifted["A"] == pytest.approx(0.6 / 1.05)
    assert drifted["B"] == pytest.approx(0.45 / 1.05)
    assert sum(drifted.values()) == pytest.approx(1.0)


def test_drift_refuses_a_held_asset_with_no_return_for_the_period() -> None:
    with pytest.raises(KeyError, match="no return for held asset"):
        drift_weights({"A": 1.0}, {"B": 0.1})


def test_drift_stops_at_a_wipeout_rather_than_continuing_through_zero() -> None:
    with pytest.raises(PortfolioWipeoutError, match="wipes out the book"):
        drift_weights({"A": 1.0}, {"A": -1.0})


def test_an_empty_book_returns_zero_and_stays_empty() -> None:
    drifted, portfolio_return = drift_weights({}, {"A": 0.5})
    assert drifted == {}
    assert portfolio_return == 0.0


# ---------------------------------------------------------------------------
# Costing a basket
# ---------------------------------------------------------------------------


def test_costing_one_full_book_order_matches_the_documented_formulas() -> None:
    costs = cost_weight_changes(
        weight_changes={"A": 1.0},
        portfolio_value_usd=CAPITAL,
        adv_usd={"A": ADV},
        daily_volatility_bps={"A": None},
        params=UNCALIBRATED_DEFAULTS,
    )
    assert costs.half_spread_usd == pytest.approx(5.0 / 10_000.0 * CAPITAL)
    assert costs.commission_usd == pytest.approx(1.0 / 10_000.0 * CAPITAL)
    assert costs.impact_usd == pytest.approx(FULL_BOOK_IMPACT_BPS / 10_000.0 * CAPITAL)
    assert costs.total_usd == pytest.approx(FULL_BOOK_COST_USD)
    assert costs.traded_notional_usd == pytest.approx(CAPITAL)
    assert costs.n_orders == 1


def test_a_sell_costs_exactly_what_the_matching_buy_costs() -> None:
    buy = cost_weight_changes(
        weight_changes={"A": 0.25},
        portfolio_value_usd=CAPITAL,
        adv_usd={"A": ADV},
        daily_volatility_bps={"A": None},
        params=UNCALIBRATED_DEFAULTS,
    )
    sell = cost_weight_changes(
        weight_changes={"A": -0.25},
        portfolio_value_usd=CAPITAL,
        adv_usd={"A": ADV},
        daily_volatility_bps={"A": None},
        params=UNCALIBRATED_DEFAULTS,
    )
    assert buy.total_usd == pytest.approx(sell.total_usd)


def test_a_zero_change_is_not_an_order() -> None:
    costs = cost_weight_changes(
        weight_changes={"A": 0.0},
        portfolio_value_usd=CAPITAL,
        adv_usd={"A": ADV},
        daily_volatility_bps={"A": None},
        params=UNCALIBRATED_DEFAULTS,
    )
    assert costs.n_orders == 0
    assert costs.total_usd == 0.0


def test_a_traded_asset_without_average_daily_volume_is_refused() -> None:
    with pytest.raises(KeyError, match="no average daily volume"):
        cost_weight_changes(
            weight_changes={"A": 0.5},
            portfolio_value_usd=CAPITAL,
            adv_usd={},
            daily_volatility_bps={},
            params=UNCALIBRATED_DEFAULTS,
        )


def test_impact_grows_with_the_square_root_of_participation() -> None:
    # Quadrupling the order should exactly double the impact *rate*, so the
    # dollar impact rises eightfold: 4 (notional) x 2 (rate).
    small = cost_weight_changes(
        weight_changes={"A": 0.1},
        portfolio_value_usd=CAPITAL,
        adv_usd={"A": ADV},
        daily_volatility_bps={"A": None},
        params=UNCALIBRATED_DEFAULTS,
    )
    large = cost_weight_changes(
        weight_changes={"A": 0.4},
        portfolio_value_usd=CAPITAL,
        adv_usd={"A": ADV},
        daily_volatility_bps={"A": None},
        params=UNCALIBRATED_DEFAULTS,
    )
    assert large.impact_usd == pytest.approx(8.0 * small.impact_usd)


def test_execution_costs_add_componentwise() -> None:
    left = ExecutionCosts(
        half_spread_usd=1.0,
        commission_usd=2.0,
        impact_usd=3.0,
        traded_notional_usd=10.0,
        n_orders=1,
    )
    right = ExecutionCosts(
        half_spread_usd=4.0,
        commission_usd=5.0,
        impact_usd=6.0,
        traded_notional_usd=20.0,
        n_orders=2,
    )
    total = left + right
    assert (total.half_spread_usd, total.commission_usd, total.impact_usd) == (5.0, 7.0, 9.0)
    assert total.traded_notional_usd == 30.0
    assert total.n_orders == 3
    assert total.total_usd == pytest.approx(21.0)


# ---------------------------------------------------------------------------
# Borrow accrues on held exposure, not on traded notional
# ---------------------------------------------------------------------------


def test_borrow_accrues_only_on_the_short_side_act_360() -> None:
    # 100 bps/yr * 1 day / 360 = 0.2777... bps on a $500k short exposure.
    accrued = accrue_borrow_usd(
        weights={"A": 0.5, "B": -0.5},
        portfolio_value_usd=CAPITAL,
        holding_period_days=1.0,
        params=UNCALIBRATED_DEFAULTS,
    )
    assert accrued == pytest.approx(100.0 / 360.0 / 10_000.0 * 0.5 * CAPITAL)


def test_a_long_only_book_accrues_no_borrow() -> None:
    assert (
        accrue_borrow_usd(
            weights={"A": 1.0},
            portfolio_value_usd=CAPITAL,
            holding_period_days=30.0,
            params=UNCALIBRATED_DEFAULTS,
        )
        == 0.0
    )


def test_borrow_is_proportional_to_the_holding_period() -> None:
    one_day = accrue_borrow_usd(
        weights={"A": -1.0},
        portfolio_value_usd=CAPITAL,
        holding_period_days=1.0,
        params=UNCALIBRATED_DEFAULTS,
    )
    three_days = accrue_borrow_usd(
        weights={"A": -1.0},
        portfolio_value_usd=CAPITAL,
        holding_period_days=3.0,
        params=UNCALIBRATED_DEFAULTS,
    )
    assert three_days == pytest.approx(3.0 * one_day)


def test_a_day_is_the_same_number_of_seconds_on_both_sides_of_the_accounting() -> None:
    assert SECONDS_PER_DAY == 86_400.0


# ---------------------------------------------------------------------------
# The benchmark itself
# ---------------------------------------------------------------------------


def test_the_benchmark_pays_an_entry_cost_and_never_trades_again() -> None:
    curve, accounting = hold({"A": 1.0}, {"A": [0.10, -0.20, 0.25]})
    after_entry = CAPITAL - FULL_BOOK_COST_USD
    assert curve.equity_usd == pytest.approx(
        [
            CAPITAL,
            after_entry * 1.10,
            after_entry * 1.10 * 0.80,
            after_entry * 1.10 * 0.80 * 1.25,
        ],
        rel=1e-12,
    )
    assert accounting.n_orders == 1
    assert accounting.n_rebalances == 1
    assert accounting.total_cost_usd == pytest.approx(FULL_BOOK_COST_USD)
    assert accounting.traded_notional_usd == pytest.approx(CAPITAL)


def test_a_free_cost_model_isolates_the_pure_market_path() -> None:
    curve, accounting = hold({"A": 1.0}, {"A": [0.10, -0.20, 0.25]}, params=free_params())
    assert accounting.total_cost_usd == 0.0
    assert curve.equity_usd[-1] == pytest.approx(CAPITAL * 1.10 * 0.80 * 1.25, rel=1e-12)


def test_the_costed_benchmark_is_strictly_worse_than_the_free_one() -> None:
    costed, _ = hold({"A": 1.0}, {"A": [0.01, 0.01, 0.01]})
    free, _ = hold({"A": 1.0}, {"A": [0.01, 0.01, 0.01]}, params=free_params())
    assert costed.equity_usd[-1] < free.equity_usd[-1]
    assert np.all(costed.equity_usd <= free.equity_usd + 1e-9)


def test_a_two_name_benchmark_places_one_order_per_name() -> None:
    _, accounting = hold(
        {"A": 0.6, "B": 0.4},
        {"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.01]},
    )
    assert accounting.n_orders == 2
    assert accounting.traded_notional_usd == pytest.approx(CAPITAL)


def test_a_short_leg_of_the_benchmark_accrues_borrow_every_period() -> None:
    _, accounting = hold(
        {"A": 0.5, "B": -0.5},
        {"A": [0.0, 0.0, 0.0], "B": [0.0, 0.0, 0.0]},
    )
    # Flat returns, so the weights never drift: three daily accruals of
    # 100 bps/yr ACT/360 on half of the post-entry book. The book itself shrinks
    # only by the borrow already paid, hence the loose relative tolerance.
    entry_cost = accounting.half_spread_usd + accounting.commission_usd + accounting.impact_usd
    after_entry = CAPITAL - entry_cost
    assert accounting.borrow_usd == pytest.approx(
        3.0 * (100.0 / 360.0 / 10_000.0) * 0.5 * after_entry, rel=1e-4
    )


def test_the_benchmark_is_measured_on_exactly_the_dates_it_was_given() -> None:
    calendar = dates(4)
    curve, _ = hold({"A": 1.0}, {"A": [0.01, 0.01, 0.01]})
    assert curve.dates == calendar
    assert curve.n_periods == 3


def test_a_benchmark_asset_without_returns_is_refused() -> None:
    with pytest.raises(KeyError, match="no return series"):
        hold({"A": 1.0}, {"B": [0.01, 0.01, 0.01]})


def test_a_return_series_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(ValueError, match="returns but the calendar"):
        hold({"A": 1.0}, {"A": [0.01, 0.01]})


def test_a_benchmark_needs_positive_capital() -> None:
    with pytest.raises(ValueError, match="initial_capital_usd must be positive"):
        buy_and_hold(
            spec=BuyAndHoldSpec(weights={"A": 1.0}),
            dates=dates(4),
            asset_returns={"A": [0.0, 0.0, 0.0]},
            adv_usd={"A": ADV},
            daily_volatility_bps={"A": None},
            initial_capital_usd=0.0,
            params=UNCALIBRATED_DEFAULTS,
        )


def test_a_benchmark_wiped_out_by_the_market_stops_rather_than_reporting_a_path() -> None:
    with pytest.raises(PortfolioWipeoutError):
        hold({"A": 1.0}, {"A": [0.0, -1.0, 0.0]})


def test_a_benchmark_whose_entry_cost_exhausts_its_capital_stops() -> None:
    ruinous = UNCALIBRATED_DEFAULTS.with_parameters(half_spread_bps=20_000.0)
    with pytest.raises(PortfolioWipeoutError, match="entry cost"):
        hold({"A": 1.0}, {"A": [0.0, 0.0, 0.0]}, params=ruinous)


# ---------------------------------------------------------------------------
# The specification describes itself
# ---------------------------------------------------------------------------


def test_a_benchmark_of_nothing_is_refused() -> None:
    with pytest.raises(WeightError, match="must hold something"):
        BuyAndHoldSpec(weights={})


def test_a_non_finite_benchmark_weight_is_refused() -> None:
    with pytest.raises(WeightError, match="not finite"):
        BuyAndHoldSpec(weights={"A": float("inf")})


def test_a_blank_description_is_generated_from_the_holdings() -> None:
    spec = BuyAndHoldSpec(weights={"SPY": 1.0})
    assert "SPY 100.0%" in spec.description
    assert "never" in spec.description
    assert "net of its own entry cost" in spec.description


def test_an_explicit_description_is_kept_verbatim() -> None:
    spec = BuyAndHoldSpec(weights={"SPY": 1.0}, description="S&P 500 total return")
    assert spec.description == "S&P 500 total return"
