"""The composed transaction cost model: properties, defaults, calibration fence (P9.3).

The properties pinned here are the ones a backtest cannot check for itself and
that fail *silently* when they are wrong — a cost model that is wrong in the
optimistic direction turns a losing strategy into a winning backtest, which is
the failure invariant I4 and DECISIONS.md D-013 exist to prevent:

- **monotone in size** — a bigger order never costs less;
- **the square-root exponent**, verified through the composed model as well as
  the component, because a wrong exponent is invisible in any output;
- **zero size costs zero**, in both units;
- **a short costs exactly the borrow term more than the identical long** — not
  approximately, exactly, so the term cannot silently vanish or double;
- **a round trip costs at least twice the half-spread**, the floor below which
  the model would be claiming free liquidity;
- **the uncalibrated flag reaches the result**, so a backtest can state from
  its own artifacts that its costs are assumptions.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from backend.costs import (
    UNCALIBRATED_BASIS,
    UNCALIBRATED_DEFAULTS,
    CostCalibrationError,
    CostModelParams,
    CostParameterError,
    Order,
    Side,
    borrow_cost_bps,
    bps_of_notional_to_usd,
    estimate_trade_cost,
    estimate_trade_costs,
)

_ADV_USD = 100_000_000.0
"""$100m average daily dollar volume — a liquid US large cap."""


def _order(
    notional_usd: float,
    *,
    side: Side = Side.BUY,
    is_short_position: bool = False,
    holding_period_days: float = 0.0,
    daily_volatility_bps: float | None = None,
    adv_usd: float = _ADV_USD,
) -> Order:
    return Order(
        side=side,
        notional_usd=notional_usd,
        adv_usd=adv_usd,
        daily_volatility_bps=daily_volatility_bps,
        is_short_position=is_short_position,
        holding_period_days=holding_period_days,
    )


_notionals = st.floats(
    min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False, allow_subnormal=False
)


# --------------------------------------------------------------------------
# Composition: the total is the sum of the parts, in one unit
# --------------------------------------------------------------------------


@given(notional=_notionals)
def test_the_total_is_exactly_the_sum_of_the_four_components(notional: float) -> None:
    """No fifth term, no double count, no rounding fudge."""
    cost = estimate_trade_cost(_order(notional, is_short_position=True, holding_period_days=21.0))
    assert cost.total_bps == pytest.approx(
        cost.half_spread_bps + cost.commission_bps + cost.impact_bps + cost.borrow_bps,
        rel=1e-12,
    )


@given(notional=_notionals)
def test_the_dollar_total_is_the_bps_total_applied_to_the_notional(notional: float) -> None:
    """The only unit boundary in the model, checked against the conversion itself."""
    cost = estimate_trade_cost(_order(notional))
    assert cost.total_usd == pytest.approx(
        bps_of_notional_to_usd(cost.total_bps, notional), rel=1e-12
    )
    assert cost.total_usd == pytest.approx(cost.total_bps * 1e-4 * notional, rel=1e-12)


def test_the_composed_cost_is_hand_computable_from_the_defaults() -> None:
    """A $1m order in a $100m-ADV name, 1% of volume, with the shipped defaults.

    half-spread 5 + commission 1 + impact (1.0 * 200 * sqrt(0.01) = 20) = 26 bps,
    which on $1,000,000 is $2,600. If any default moves, this test states the
    consequence in the units an operator reads.
    """
    cost = estimate_trade_cost(_order(1_000_000.0))
    assert cost.participation == pytest.approx(0.01)
    assert cost.half_spread_bps == 5.0
    assert cost.commission_bps == 1.0
    assert cost.impact_bps == pytest.approx(20.0)
    assert cost.borrow_bps == 0.0
    assert cost.total_bps == pytest.approx(26.0)
    assert cost.total_usd == pytest.approx(2_600.0)


# --------------------------------------------------------------------------
# Property: monotone in size
# --------------------------------------------------------------------------


@given(
    notional=_notionals,
    factor=st.floats(min_value=1.01, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_cost_is_strictly_monotone_in_order_size(notional: float, factor: float) -> None:
    """A bigger order is more expensive, in bps of its own notional and in dollars.

    Both statements matter and they are not the same: the dollar cost would
    rise with size even under a flat bps model, so only the bps comparison
    tests that impact is present at all.
    """
    smaller = estimate_trade_cost(_order(notional))
    larger = estimate_trade_cost(_order(notional * factor))

    assert larger.total_bps > smaller.total_bps
    assert larger.total_usd > smaller.total_usd


@given(notional=_notionals, extra=st.floats(min_value=0.0, max_value=1e9, allow_nan=False))
def test_cost_is_non_decreasing_in_order_size_everywhere(notional: float, extra: float) -> None:
    """The weak form, including ties: no size increase ever reduces the cost."""
    assert (
        estimate_trade_cost(_order(notional + extra)).total_bps
        >= estimate_trade_cost(_order(notional)).total_bps
    )


def test_cost_is_monotone_across_orders_of_magnitude_of_participation() -> None:
    """A concrete ladder, so the direction is legible without running Hypothesis."""
    totals = [
        estimate_trade_cost(_order(_ADV_USD * participation)).total_bps
        for participation in (0.0001, 0.001, 0.01, 0.1, 1.0)
    ]
    assert totals == sorted(totals)
    assert totals[0] < totals[-1]


# --------------------------------------------------------------------------
# Property: the square-root exponent, through the composed model
# --------------------------------------------------------------------------


def test_a_one_percent_and_a_hundred_percent_adv_order_differ_by_the_square_root_factor() -> None:
    """The exponent check the task specifies, on the composed model's own output.

    Only the impact term varies with size, so the difference in totals is the
    difference in impact, and that must be a factor of ``sqrt(100) == 10``.
    Asserted on the isolated impact component *and* on the total, because the
    fixed terms dilute the ratio in the total and an exponent bug could hide in
    that dilution if only the total were checked.
    """
    one_percent = estimate_trade_cost(_order(_ADV_USD * 0.01))
    full_day = estimate_trade_cost(_order(_ADV_USD * 1.0))

    assert one_percent.participation == pytest.approx(0.01)
    assert full_day.participation == pytest.approx(1.0)

    # The impact term itself: exactly 10x.
    assert full_day.impact_bps / one_percent.impact_bps == pytest.approx(10.0, rel=1e-12)

    # A linear model would have made it 100x. It must not be.
    assert not math.isclose(full_day.impact_bps / one_percent.impact_bps, 100.0, rel_tol=1e-6)

    # And the totals, with the fixed 6 bps of spread and commission in place:
    # 5 + 1 + 20 = 26 against 5 + 1 + 200 = 206.
    assert one_percent.total_bps == pytest.approx(26.0)
    assert full_day.total_bps == pytest.approx(206.0)


@given(
    participation=st.floats(min_value=1e-6, max_value=0.5, allow_nan=False, allow_infinity=False),
    factor=st.floats(min_value=2.0, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_the_exponent_holds_through_the_model_at_every_size(
    participation: float, factor: float
) -> None:
    """Property form of the exponent check, on the model rather than the component."""
    smaller = estimate_trade_cost(_order(_ADV_USD * participation))
    larger = estimate_trade_cost(_order(_ADV_USD * participation * factor))
    assert larger.impact_bps / smaller.impact_bps == pytest.approx(math.sqrt(factor), rel=1e-9)


# --------------------------------------------------------------------------
# Property: zero size costs zero
# --------------------------------------------------------------------------


def test_a_zero_sized_order_costs_zero_in_both_units() -> None:
    """Nothing traded: no spread crossed, no commission, no impact, no dollars."""
    cost = estimate_trade_cost(_order(0.0))
    assert cost.half_spread_bps == 0.0
    assert cost.commission_bps == 0.0
    assert cost.impact_bps == 0.0
    assert cost.borrow_bps == 0.0
    assert cost.total_bps == 0.0
    assert cost.total_usd == 0.0
    assert cost.participation == 0.0


def test_a_zero_sized_short_costs_zero_even_with_a_holding_period() -> None:
    """No position was opened, so there is nothing to finance."""
    cost = estimate_trade_cost(
        _order(0.0, side=Side.SELL, is_short_position=True, holding_period_days=90.0)
    )
    assert cost.total_bps == 0.0
    assert cost.total_usd == 0.0


def test_a_zero_sized_order_still_reports_its_side_and_calibration_status() -> None:
    """A free trade is still an auditable record."""
    cost = estimate_trade_cost(_order(0.0, side=Side.SELL))
    assert cost.side is Side.SELL
    assert cost.uncalibrated is True
    assert cost.calibration_basis == UNCALIBRATED_BASIS


def test_cost_in_dollars_is_continuous_at_zero_while_cost_in_bps_is_not() -> None:
    """The honest shape, and the reason both units are reported.

    An arbitrarily small order pays the full fixed rate in bps — 26 bps whether
    it is $1 or $1m — but its dollar cost goes to zero with its size. Neither
    statement alone describes the model.
    """
    tiny = estimate_trade_cost(_order(1e-6))
    assert tiny.total_bps > 6.0
    assert tiny.total_usd < 1e-8
    assert estimate_trade_cost(_order(0.0)).total_bps == 0.0


# --------------------------------------------------------------------------
# Property: a short costs exactly the borrow term more than the same long
# --------------------------------------------------------------------------


@given(
    notional=_notionals,
    days=st.floats(min_value=0.5, max_value=750.0, allow_nan=False, allow_infinity=False),
)
def test_a_short_exceeds_the_identical_long_by_exactly_the_borrow_term(
    notional: float, days: float
) -> None:
    """Exactly, not approximately: borrow is the only term the side changes."""
    long_cost = estimate_trade_cost(_order(notional, side=Side.BUY))
    short_cost = estimate_trade_cost(
        _order(notional, side=Side.SELL, is_short_position=True, holding_period_days=days)
    )
    expected_borrow = borrow_cost_bps(
        borrow_rate_bps_per_year=UNCALIBRATED_DEFAULTS.borrow_rate_bps_per_year,
        holding_period_days=days,
    )

    assert short_cost.borrow_bps == pytest.approx(expected_borrow, rel=1e-12, abs=1e-15)
    assert long_cost.borrow_bps == 0.0
    assert short_cost.total_bps - long_cost.total_bps == pytest.approx(
        expected_borrow, rel=1e-12, abs=1e-12
    )
    assert short_cost.half_spread_bps == long_cost.half_spread_bps
    assert short_cost.commission_bps == long_cost.commission_bps
    assert short_cost.impact_bps == pytest.approx(long_cost.impact_bps, rel=1e-12)


def test_the_borrow_premium_is_hand_computable() -> None:
    """$1m short of a 1%-ADV name held 36 days: 26 bps + 10 bps of borrow."""
    short_cost = estimate_trade_cost(
        _order(1_000_000.0, side=Side.SELL, is_short_position=True, holding_period_days=36.0)
    )
    assert short_cost.borrow_bps == pytest.approx(10.0)
    assert short_cost.total_bps == pytest.approx(36.0)


def test_a_short_must_state_its_holding_period() -> None:
    """Silence is not zero borrow: an unstated horizon is refused, not defaulted.

    Defaulting the horizon to zero would report a short book net of everything
    except its financing — an understatement, which is the one direction D-013
    forbids. The platform rebalances daily and intraday is out of scope
    (directive §1.1), so no real short here is held for zero days.
    """
    with pytest.raises(CostParameterError, match="holding_period_days"):
        _order(1_000_000.0, side=Side.SELL, is_short_position=True)

    with pytest.raises(CostParameterError, match="holding_period_days"):
        _order(1_000_000.0, side=Side.SELL, is_short_position=True, holding_period_days=0.0)


def test_the_shortest_admissible_short_is_one_day_and_costs_one_day_of_borrow() -> None:
    """The boundary is open at zero and immediately usable above it."""
    cost = estimate_trade_cost(
        _order(1_000_000.0, side=Side.SELL, is_short_position=True, holding_period_days=1.0)
    )
    assert cost.borrow_bps == pytest.approx(100.0 / 360.0)


def test_a_long_order_needs_no_holding_period() -> None:
    """The requirement is on the borrow term, not on every order."""
    assert estimate_trade_cost(_order(1_000_000.0)).borrow_bps == 0.0


def test_a_sell_that_is_not_a_short_pays_no_borrow() -> None:
    """Selling a long position borrows nothing; the flag is the position, not the side."""
    cost = estimate_trade_cost(_order(1_000_000.0, side=Side.SELL, holding_period_days=90.0))
    assert cost.borrow_bps == 0.0


def test_the_closing_leg_of_a_short_is_not_charged_borrow_again() -> None:
    """Borrow accrues once, on the leg that holds it — not on both legs."""
    opening = estimate_trade_cost(
        _order(1_000_000.0, side=Side.SELL, is_short_position=True, holding_period_days=30.0)
    )
    closing = estimate_trade_cost(_order(1_000_000.0, side=Side.BUY))
    assert opening.borrow_bps > 0.0
    assert closing.borrow_bps == 0.0


@given(notional=_notionals)
def test_buying_and_selling_cost_the_same_absent_borrow(notional: float) -> None:
    """Spread, commission and impact are symmetric; only borrow is not."""
    buy = estimate_trade_cost(_order(notional, side=Side.BUY))
    sell = estimate_trade_cost(_order(notional, side=Side.SELL))
    assert buy.total_bps == pytest.approx(sell.total_bps, rel=1e-15)


# --------------------------------------------------------------------------
# Property: a round trip costs at least twice the half-spread
# --------------------------------------------------------------------------


@given(notional=_notionals)
def test_a_round_trip_costs_at_least_two_half_spreads(notional: float) -> None:
    """Buy then sell crosses the spread twice, so it pays the full quoted spread.

    The floor is the model's minimum claim about liquidity. Falling below it
    would mean the model had found a way to trade inside the touch, which is
    exactly the optimism D-013 forbids.
    """
    params = UNCALIBRATED_DEFAULTS
    opening = estimate_trade_cost(_order(notional, side=Side.BUY))
    closing = estimate_trade_cost(_order(notional, side=Side.SELL))
    round_trip_bps = opening.total_bps + closing.total_bps

    assert round_trip_bps >= 2.0 * params.half_spread_bps
    # And the same statement in dollars, which is what a P&L actually pays.
    assert opening.total_usd + closing.total_usd >= bps_of_notional_to_usd(
        2.0 * params.half_spread_bps, notional
    )


def test_a_round_trip_pays_the_full_quoted_spread_plus_the_variable_terms() -> None:
    """The floor is 10 bps of quoted spread; the model charges more, never less."""
    params = UNCALIBRATED_DEFAULTS
    opening = estimate_trade_cost(_order(1_000_000.0, side=Side.BUY))
    closing = estimate_trade_cost(_order(1_000_000.0, side=Side.SELL))
    round_trip_bps = opening.total_bps + closing.total_bps

    assert 2.0 * params.half_spread_bps == 10.0
    assert round_trip_bps == pytest.approx(52.0)
    assert round_trip_bps > 2.0 * params.half_spread_bps


def test_a_round_trip_in_a_zero_impact_zero_commission_world_is_exactly_two_half_spreads() -> None:
    """The floor is attained, so it is a real bound rather than a loose inequality."""
    params = UNCALIBRATED_DEFAULTS.with_parameters(
        commission_bps=0.0, impact_coefficient=0.0, borrow_rate_bps_per_year=0.0
    )
    opening = estimate_trade_cost(_order(1_000_000.0, side=Side.BUY), params)
    closing = estimate_trade_cost(_order(1_000_000.0, side=Side.SELL), params)
    assert opening.total_bps + closing.total_bps == pytest.approx(2.0 * params.half_spread_bps)


# --------------------------------------------------------------------------
# Volatility: per-name when supplied, conservative default when not
# --------------------------------------------------------------------------


def test_an_order_without_volatility_uses_the_conservative_default() -> None:
    with_default = estimate_trade_cost(_order(1_000_000.0))
    explicit = estimate_trade_cost(
        _order(1_000_000.0, daily_volatility_bps=UNCALIBRATED_DEFAULTS.default_daily_volatility_bps)
    )
    assert with_default.impact_bps == pytest.approx(explicit.impact_bps)


def test_a_more_volatile_name_costs_more_to_trade() -> None:
    """Impact scales linearly in volatility — the other half of the impact law."""
    calm = estimate_trade_cost(_order(1_000_000.0, daily_volatility_bps=100.0))
    wild = estimate_trade_cost(_order(1_000_000.0, daily_volatility_bps=400.0))
    assert wild.impact_bps == pytest.approx(4.0 * calm.impact_bps)


# --------------------------------------------------------------------------
# The UNCALIBRATED flag reaches the result (directive §5-P9, I4)
# --------------------------------------------------------------------------


def test_the_shipped_defaults_are_flagged_uncalibrated() -> None:
    assert UNCALIBRATED_DEFAULTS.uncalibrated is True
    assert "UNCALIBRATED" in UNCALIBRATED_DEFAULTS.calibration_basis
    assert UNCALIBRATED_DEFAULTS.calibration_basis == UNCALIBRATED_BASIS


def test_every_estimate_carries_the_calibration_status_forward() -> None:
    """A backtest must be able to say "these costs were assumed" from the artifact alone."""
    cost = estimate_trade_cost(_order(1_000_000.0))
    assert cost.uncalibrated is True
    assert cost.calibration_basis == UNCALIBRATED_BASIS

    summary = cost.summary()
    assert summary["uncalibrated"] is True
    assert summary["calibration_basis"] == UNCALIBRATED_BASIS
    assert summary["units"] == "basis points of traded notional (1 bp = 1e-4); total_usd in USD"


def test_the_defaults_are_the_default_argument() -> None:
    """A caller who supplies no parameters gets pessimistic costs, not free ones."""
    assert estimate_trade_cost(_order(1_000_000.0)).total_bps == pytest.approx(
        estimate_trade_cost(_order(1_000_000.0), UNCALIBRATED_DEFAULTS).total_bps
    )


def test_the_shipped_default_values_are_the_documented_ones() -> None:
    """Pins each default so a change is a deliberate edit with a test to update.

    Every one of these is a conservative literature value, not a measurement;
    the justification for each is on :class:`CostModelParams`.
    """
    assert UNCALIBRATED_DEFAULTS.half_spread_bps == 5.0
    assert UNCALIBRATED_DEFAULTS.commission_bps == 1.0
    assert UNCALIBRATED_DEFAULTS.impact_coefficient == 1.0
    assert UNCALIBRATED_DEFAULTS.default_daily_volatility_bps == 200.0
    assert UNCALIBRATED_DEFAULTS.borrow_rate_bps_per_year == 100.0


def test_a_basket_reports_the_calibration_status_on_every_leg() -> None:
    orders = [_order(1e6), _order(2e6, side=Side.SELL), _order(0.0)]
    costs = estimate_trade_costs(orders)
    assert len(costs) == len(orders)
    assert all(cost.uncalibrated for cost in costs)
    assert [cost.side for cost in costs] == [Side.BUY, Side.SELL, Side.BUY]


# --------------------------------------------------------------------------
# The D-013 calibration fence
# --------------------------------------------------------------------------


def test_a_calibration_may_not_be_cheaper_than_the_conservative_default() -> None:
    """D-013: paper fills are a lower bound on slippage, never an estimate.

    A calibration fitted to optimistic fills would tighten the model and flatter
    every downstream Sharpe. Cheaper-than-default parameters are refused at
    construction rather than caveated in prose.
    """
    with pytest.raises(CostCalibrationError) as raised:
        CostModelParams(
            half_spread_bps=2.0,
            uncalibrated=False,
            calibration_basis="fitted to 10,000 paper fills, 2026-Q3",
        )
    assert raised.value.parameter == "half_spread_bps"
    assert raised.value.value == 2.0
    assert raised.value.floor == UNCALIBRATED_DEFAULTS.half_spread_bps
    assert "lower bound" in str(raised.value)


def test_a_calibration_that_is_more_expensive_than_the_default_is_allowed() -> None:
    """The fence is one-directional: costs may only be revised upward."""
    params = CostModelParams(
        half_spread_bps=8.0,
        commission_bps=1.5,
        impact_coefficient=1.2,
        default_daily_volatility_bps=250.0,
        borrow_rate_bps_per_year=150.0,
        uncalibrated=False,
        calibration_basis="fitted to 10,000 paper fills with a 1.5x haircut, 2026-Q3",
    )
    assert params.uncalibrated is False
    cost = estimate_trade_cost(_order(1_000_000.0), params)
    assert cost.uncalibrated is False
    assert "haircut" in cost.calibration_basis


def test_a_calibrated_set_may_not_reuse_the_uncalibrated_basis_string() -> None:
    """Otherwise a "calibrated" result could carry the word UNCALIBRATED into a report."""
    with pytest.raises(CostParameterError, match="calibration basis"):
        CostModelParams(uncalibrated=False, calibration_basis=UNCALIBRATED_BASIS)


def test_an_empty_calibration_basis_is_refused() -> None:
    with pytest.raises(CostParameterError, match="calibration_basis"):
        CostModelParams(calibration_basis="   ")


def test_adjusting_parameters_cannot_widen_the_calibration_claim() -> None:
    """``with_parameters`` moves numbers only; marking a set calibrated is P11.8's job."""
    adjusted = UNCALIBRATED_DEFAULTS.with_parameters(half_spread_bps=12.0)
    assert adjusted.half_spread_bps == 12.0
    assert adjusted.uncalibrated is True
    assert adjusted.calibration_basis == UNCALIBRATED_BASIS


def test_sensitivity_analysis_downward_stays_uncalibrated_and_is_permitted() -> None:
    """Cheaper *uncalibrated* parameters are a what-if, not a claim about fills."""
    cheaper = UNCALIBRATED_DEFAULTS.with_parameters(half_spread_bps=1.0)
    assert cheaper.half_spread_bps == 1.0
    assert cheaper.uncalibrated is True


def test_lowering_a_calibrated_parameter_below_the_floor_is_refused() -> None:
    """The fence survives ``with_parameters`` too."""
    calibrated = CostModelParams(
        half_spread_bps=8.0,
        commission_bps=1.0,
        impact_coefficient=1.0,
        default_daily_volatility_bps=200.0,
        borrow_rate_bps_per_year=100.0,
        uncalibrated=False,
        calibration_basis="fitted with haircut, 2026-Q3",
    )
    with pytest.raises(CostCalibrationError):
        calibrated.with_parameters(half_spread_bps=4.0)


def test_an_unknown_parameter_name_is_refused() -> None:
    """A typo'd override would otherwise be silently ignored."""
    with pytest.raises(CostParameterError, match="unknown cost parameter"):
        UNCALIBRATED_DEFAULTS.with_parameters(halfspread_bps=7.0)


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_a_negative_notional_is_refused() -> None:
    """Direction lives in ``side``; a negative notional would flip a cost into a credit."""
    with pytest.raises(CostParameterError, match="notional_usd"):
        _order(-1.0)


@pytest.mark.parametrize("adv", [0.0, -1.0, math.inf])
def test_an_unusable_adv_is_refused(adv: float) -> None:
    with pytest.raises(CostParameterError, match="adv_usd"):
        _order(1_000.0, adv_usd=adv)


def test_a_negative_holding_period_is_refused() -> None:
    with pytest.raises(CostParameterError, match="holding_period_days"):
        _order(1_000.0, is_short_position=True, holding_period_days=-1.0)


def test_a_non_finite_volatility_is_refused() -> None:
    with pytest.raises(CostParameterError, match="daily_volatility_bps"):
        _order(1_000.0, daily_volatility_bps=math.nan)


@pytest.mark.parametrize(
    "field",
    [
        "half_spread_bps",
        "commission_bps",
        "impact_coefficient",
        "default_daily_volatility_bps",
        "borrow_rate_bps_per_year",
    ],
)
def test_a_negative_cost_parameter_is_refused(field: str) -> None:
    """A negative cost parameter is a subsidy, and there is no such thing."""
    with pytest.raises(CostParameterError, match=field):
        CostModelParams(**{field: -1.0})  # type: ignore[arg-type]


def test_an_empty_basket_costs_nothing_and_returns_nothing() -> None:
    assert estimate_trade_costs([]) == ()
