"""Property-based tests for cost-model monotonicity, scaling and units (P9.3, P9.4).

Directive §8 requires a Hypothesis suite on the cost model, and names the reason:
*"basis points versus percent versus fraction is the most common bug class in
this domain and it is silent"*. A cost model does not raise when it is wrong. It
returns a number that is plausible at every order size, flows into a net return,
then into a Sharpe ratio, and first becomes visible as a chart nobody can
falsify. So the properties asserted here are the structural ones a reader of a
backtest relies on and cannot check from the output:

1. **Monotonicity.** Cost never decreases as the order grows, never increases as
   the available volume grows, and never decreases as any single parameter is
   raised. A model that is non-monotone somewhere is one an optimizer can game.
2. **Square-root scaling.** Multiplying the order by ``k`` multiplies impact by
   ``sqrt(k)`` — measured from the output, at every pair of sizes, rather than
   read off the source. The exponent is the one parameter of this model whose
   error is invisible: a linear and a square-root law agree exactly at 100% of
   ADV and disagree by a factor of ten at 1%, which is where every real order
   lives.
3. **Non-negativity.** No component, and no total, is ever a subsidy.
4. **Units.** Every ``*_bps`` field is in basis points of the order's own
   notional, and the only bridge to dollars is the exact identity
   ``total_usd == bps_to_fraction(total_bps) * notional_usd``. Two dimensional
   properties pin this down without trusting a name: costs quoted in bps are
   invariant to the currency scale, while costs quoted in dollars are linear in
   it.

Inputs are drawn from ranges a US large/mid-cap universe actually produces
(``$1``-``$5bn`` orders against ``$100k``-``$100bn`` ADV, 0-2000 bps of daily
volatility, 0-10,000 bps/year of borrow), so every generated example is a
genuinely valid order rather than a shape the validators reject before any
arithmetic happens.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from backend.costs import (
    BPS_PER_UNIT,
    IMPACT_EXPONENT,
    CostModelParams,
    Order,
    Side,
    TradeCost,
    borrow_cost_bps,
    bps_to_fraction,
    estimate_trade_cost,
    estimate_trade_costs,
    usd_to_bps_of_notional,
)

# --------------------------------------------------------------------------
# Strategies — every draw is a valid order, so the suite measures arithmetic
# rather than validation
# --------------------------------------------------------------------------

_NOTIONALS = st.floats(min_value=1.0, max_value=5e9, allow_nan=False, allow_infinity=False)
"""Traded notional in USD. From a $1 odd-lot to a $5bn program trade."""

_ADVS = st.floats(min_value=1e5, max_value=1e11, allow_nan=False, allow_infinity=False)
"""Average daily dollar volume in USD, from a thin microcap to a megacap."""

_VOLATILITIES = st.floats(min_value=0.0, max_value=2_000.0, allow_nan=False, allow_infinity=False)
"""Daily return standard deviation in bps. 2000 bps is 20% per day."""

_HOLDING_PERIODS = st.floats(min_value=0.5, max_value=250.0, allow_nan=False, allow_infinity=False)
"""Days a short is held. Strictly positive, as :class:`Order` requires."""

_SCALE_FACTORS = st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False)
"""Multiplier applied to an order size or a currency scale."""


@st.composite
def _parameter_sets(draw: st.DrawFn) -> CostModelParams:
    """Draw a valid, uncalibrated parameter set spanning the plausible ranges.

    ``uncalibrated`` is left at ``True`` throughout: marking a set calibrated is
    fenced by the D-013 floor, and a suite that flipped the flag would be
    testing the fence rather than the arithmetic.
    """
    return CostModelParams(
        half_spread_bps=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        commission_bps=draw(
            st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False)
        ),
        impact_coefficient=draw(
            st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False)
        ),
        default_daily_volatility_bps=draw(_VOLATILITIES),
        borrow_rate_bps_per_year=draw(
            st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def _orders(draw: st.DrawFn, *, allow_zero: bool = True) -> Order:
    """Draw a valid order, short or long, with or without its own volatility."""
    is_short = draw(st.booleans())
    notional = draw(st.just(0.0) | _NOTIONALS) if allow_zero else draw(_NOTIONALS)
    return Order(
        side=draw(st.sampled_from(Side)),
        notional_usd=notional,
        adv_usd=draw(_ADVS),
        daily_volatility_bps=draw(st.none() | _VOLATILITIES),
        is_short_position=is_short,
        holding_period_days=draw(_HOLDING_PERIODS) if is_short else 0.0,
    )


def _components(cost: TradeCost) -> tuple[float, ...]:
    """The four component costs, in basis points of notional."""
    return (cost.half_spread_bps, cost.commission_bps, cost.impact_bps, cost.borrow_bps)


# --------------------------------------------------------------------------
# Non-negativity and finiteness
# --------------------------------------------------------------------------


@given(order=_orders(), params=_parameter_sets())
def test_no_component_of_a_cost_is_ever_negative(order: Order, params: CostModelParams) -> None:
    """A negative cost is a subsidy, and there is no such thing.

    Asserted on every component separately rather than on the total, because a
    total can be positive while a component beneath it has gone negative — and
    the component breakdown is what dashboard §6.7's cost waterfall displays.
    """
    cost = estimate_trade_cost(order, params)
    for component in _components(cost):
        assert component >= 0.0
    assert cost.total_bps >= 0.0
    assert cost.total_usd >= 0.0


@given(order=_orders(), params=_parameter_sets())
def test_a_finite_order_produces_a_finite_cost(order: Order, params: CostModelParams) -> None:
    """No realistic order may produce a NaN or an infinity.

    A non-finite cost survives every downstream sum and first becomes visible
    as a blank cell in a report, which is exactly how it reaches production.
    """
    cost = estimate_trade_cost(order, params)
    for component in (*_components(cost), cost.total_bps, cost.total_usd, cost.participation):
        assert math.isfinite(component)


# --------------------------------------------------------------------------
# Units — bps throughout, and the one exact bridge to dollars
# --------------------------------------------------------------------------


@given(order=_orders(), params=_parameter_sets())
def test_the_total_is_exactly_its_components_and_the_dollar_total_its_conversion(
    order: Order, params: CostModelParams
) -> None:
    """The two identities that make the breakdown readable, asserted exactly.

    ``total_bps`` is the sum of the four components with no fifth term hiding
    in it, and ``total_usd`` is that total converted once, through
    :mod:`backend.costs.units`, at the notional the cost is quoted on. Exact
    equality rather than a tolerance: both sides are the same arithmetic in the
    same order, so any difference at all is a different arithmetic.
    """
    cost = estimate_trade_cost(order, params)
    half, commission, impact, borrow = _components(cost)

    assert cost.total_bps == half + commission + impact + borrow
    assert cost.total_usd == bps_to_fraction(cost.total_bps) * abs(order.notional_usd)
    assert cost.notional_usd == order.notional_usd


@given(order=_orders(allow_zero=False), params=_parameter_sets())
def test_dollars_convert_back_to_the_basis_points_they_came_from(
    order: Order, params: CostModelParams
) -> None:
    """A cost quoted in bps and the same cost quoted in dollars are one number.

    If the two ever diverge, one of the two is being reported against a
    different notional than the other, and every "net of costs" figure built on
    them is off by that ratio.
    """
    cost = estimate_trade_cost(order, params)
    assert usd_to_bps_of_notional(cost.total_usd, order.notional_usd) == pytest.approx(
        cost.total_bps, rel=1e-12, abs=1e-12
    )


@given(order=_orders(allow_zero=False), params=_parameter_sets(), scale=_SCALE_FACTORS)
def test_basis_points_are_invariant_to_the_currency_scale_and_dollars_are_linear_in_it(
    order: Order, params: CostModelParams, scale: float
) -> None:
    """Dimensional analysis, as a property: bps are a rate, dollars are an amount.

    Scaling the order *and* the day's volume by the same factor leaves the
    order's share of the day's volume unchanged, so every rate quoted in basis
    points must be unchanged too, while the dollar cost must scale by exactly
    that factor. This is the property that would catch a fraction leaking into
    a ``*_bps`` field: such a term would be a *fixed* number of dollars rather
    than a rate, and would break the invariance in one direction or the
    linearity in the other.
    """
    assume(order.notional_usd * scale <= 1e15)
    assume(order.adv_usd * scale <= 1e15)
    scaled = replace(order, notional_usd=order.notional_usd * scale, adv_usd=order.adv_usd * scale)

    base = estimate_trade_cost(order, params)
    grown = estimate_trade_cost(scaled, params)

    assert grown.participation == pytest.approx(base.participation, rel=1e-9)
    for original, rescaled in zip(_components(base), _components(grown), strict=True):
        assert rescaled == pytest.approx(original, rel=1e-9, abs=1e-12)
    assert grown.total_bps == pytest.approx(base.total_bps, rel=1e-9, abs=1e-12)
    assert grown.total_usd == pytest.approx(base.total_usd * scale, rel=1e-9)


@given(notional=_NOTIONALS, adv=_ADVS)
def test_quoting_a_rate_as_a_fraction_instead_of_basis_points_is_a_ten_thousand_fold_error(
    notional: float, adv: float
) -> None:
    """The §8 failure mode, priced, through the composed model rather than a conversion.

    Impact and borrow are switched off so the total is exactly the two fixed
    rates. Handing the model those same rates as *fractions* — the mistake a
    caller with a ``0.0010`` spread makes — understates the cost by exactly
    ``BPS_PER_UNIT``. No validator can catch it: ``0.0010`` is an ordinary
    non-negative float. The defence is that every parameter name ends in its
    unit, and this is what the mistake costs when the naming fails.
    """
    order = Order(side=Side.BUY, notional_usd=notional, adv_usd=adv)
    correct = CostModelParams(half_spread_bps=5.0, commission_bps=1.0, impact_coefficient=0.0)
    mistaken = correct.with_parameters(
        half_spread_bps=bps_to_fraction(5.0), commission_bps=bps_to_fraction(1.0)
    )

    correct_cost = estimate_trade_cost(order, correct)
    mistaken_cost = estimate_trade_cost(order, mistaken)

    assert correct_cost.total_bps == pytest.approx(6.0)
    assert correct_cost.total_bps / mistaken_cost.total_bps == pytest.approx(BPS_PER_UNIT)
    assert correct_cost.total_usd / mistaken_cost.total_usd == pytest.approx(BPS_PER_UNIT)


@given(order=_orders(), params=_parameter_sets())
def test_every_estimate_states_the_unit_it_is_quoted_in(
    order: Order, params: CostModelParams
) -> None:
    """A number rendered without its unit is the bug class §8 is about.

    The summary is what a dashboard and a stored artifact read, so the unit
    statement travels with the number rather than living in a docstring the
    renderer never sees.
    """
    summary = estimate_trade_cost(order, params).summary()
    assert summary["units"] == ("basis points of traded notional (1 bp = 1e-4); total_usd in USD")


# --------------------------------------------------------------------------
# Monotonicity in order size
# --------------------------------------------------------------------------


@given(order=_orders(), params=_parameter_sets(), factor=_SCALE_FACTORS)
def test_cost_is_monotone_non_decreasing_in_order_size(
    order: Order, params: CostModelParams, factor: float
) -> None:
    """A bigger order never costs less, in either unit.

    Asserted without tolerance. Participation is a correctly-rounded division,
    the square root and the non-negative multiplications that follow it are
    monotone under round-to-nearest, and the fixed rates are identical between
    the two calls — so the inequality holds exactly, and a tolerance would only
    be hiding a model that had stopped being monotone.
    """
    assume(order.notional_usd * factor <= 1e15)
    grown = replace(order, notional_usd=order.notional_usd * factor)

    base_cost = estimate_trade_cost(order, params)
    grown_cost = estimate_trade_cost(grown, params)

    assert grown_cost.total_bps >= base_cost.total_bps
    assert grown_cost.total_usd >= base_cost.total_usd
    assert grown_cost.impact_bps >= base_cost.impact_bps


@given(order=_orders(allow_zero=False), params=_parameter_sets(), factor=_SCALE_FACTORS)
def test_cost_is_strictly_increasing_in_order_size_once_impact_bites(
    order: Order, params: CostModelParams, factor: float
) -> None:
    """Where the model has an impact term at all, growth is strict, not merely weak.

    Weak monotonicity alone is satisfied by a model that ignores order size
    entirely — which is precisely the model that makes a strategy look scalable
    when it is not. This pins the other side.

    **Why the factor floor is 1 + 1e-6 and not simply > 1.** Impact scales as
    the square root of participation, and a square root *halves* a relative
    difference. At ``factor = 1 + 2**-52`` — the next float above one, which
    ``> 1.0`` admits — the relative change entering the square root is ~1e-16
    and the change leaving it is ~5e-17, below the ~1.1e-16 resolution of a
    float64. The two impacts are then genuinely the same number and strictness
    is a claim about IEEE-754, not about the cost model. Hypothesis found this
    directly: ``notional_usd`` 1.0 vs 1.0000000000000002 against an ADV of
    100,001 produced impact 0.006324523697797326 twice over. A floor of 1e-6
    relative leaves ~5e-7 of relative change in the result, four thousand times
    the noise floor, so a failure here means the *model* ignored order size —
    which is what this test exists to detect. The weak inequality is asserted
    without any floor by
    :func:`test_cost_is_monotone_non_decreasing_in_order_size` above.
    """
    assume(factor >= 1.0 + 1e-6)
    assume(order.notional_usd * factor <= 1e15)
    assume(params.impact_coefficient > 1e-3)
    volatility = (
        params.default_daily_volatility_bps
        if order.daily_volatility_bps is None
        else order.daily_volatility_bps
    )
    assume(volatility > 1.0)
    grown = replace(order, notional_usd=order.notional_usd * factor)

    assert (
        estimate_trade_cost(grown, params).impact_bps
        > estimate_trade_cost(order, params).impact_bps
    )


@given(order=_orders(allow_zero=False), params=_parameter_sets(), factor=_SCALE_FACTORS)
def test_cost_is_monotone_non_increasing_in_available_volume(
    order: Order, params: CostModelParams, factor: float
) -> None:
    """The same order in a more liquid name never costs more.

    The other half of monotonicity: impact depends on the *ratio* of the two,
    so a model monotone in the numerator but not in the denominator would price
    a megacap and a microcap identically at the same dollar size.
    """
    assume(order.adv_usd * factor <= 1e15)
    deeper = replace(order, adv_usd=order.adv_usd * factor)

    assert estimate_trade_cost(deeper, params).total_bps <= (
        estimate_trade_cost(order, params).total_bps
    )


# --------------------------------------------------------------------------
# Monotonicity in every parameter
# --------------------------------------------------------------------------


@given(
    order=_orders(allow_zero=False),
    params=_parameter_sets(),
    field=st.sampled_from(
        [
            "half_spread_bps",
            "commission_bps",
            "impact_coefficient",
            "default_daily_volatility_bps",
            "borrow_rate_bps_per_year",
        ]
    ),
    increment=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_cost_is_monotone_non_decreasing_in_every_parameter(
    order: Order, params: CostModelParams, field: str, increment: float
) -> None:
    """Raising any single parameter never lowers the estimate.

    This is what makes the parameter set usable for the sensitivity analysis
    D-013 contemplates: an operator widening the assumed spread, or the assumed
    borrow rate, must see the cost move in the direction they moved it, at every
    order size, or the "conservative" direction is not well defined.

    The order is forced onto the model's default volatility so that raising
    ``default_daily_volatility_bps`` is actually reachable — an order carrying
    its own volatility ignores that parameter by design.
    """
    on_default_volatility = replace(order, daily_volatility_bps=None)
    raised = params.with_parameters(**{field: getattr(params, field) + increment})

    assert (
        estimate_trade_cost(on_default_volatility, raised).total_bps
        >= estimate_trade_cost(on_default_volatility, params).total_bps
    )


@given(
    rate=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    days=_HOLDING_PERIODS,
    extra=st.floats(min_value=0.0, max_value=250.0, allow_nan=False, allow_infinity=False),
    order=_orders(allow_zero=False),
)
def test_a_short_held_longer_never_costs_less_to_finance(
    rate: float, days: float, extra: float, order: Order
) -> None:
    """Borrow is a holding cost, so it accrues; it never unwinds.

    Checked through the composed model as well as through the component, since
    it is the composition that decides which orders the accrual is applied to.
    """
    params = CostModelParams(borrow_rate_bps_per_year=rate)
    short = replace(
        order, is_short_position=True, holding_period_days=days, daily_volatility_bps=None
    )
    longer = replace(short, holding_period_days=days + extra)

    assert borrow_cost_bps(
        borrow_rate_bps_per_year=rate, holding_period_days=days + extra
    ) >= borrow_cost_bps(borrow_rate_bps_per_year=rate, holding_period_days=days)
    assert (
        estimate_trade_cost(longer, params).total_bps
        >= estimate_trade_cost(short, params).total_bps
    )


# --------------------------------------------------------------------------
# Square-root scaling
# --------------------------------------------------------------------------


@given(
    order=_orders(allow_zero=False),
    params=_parameter_sets(),
    factor=st.floats(min_value=1.0001, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_impact_scales_as_the_square_root_of_order_size(
    order: Order, params: CostModelParams, factor: float
) -> None:
    """``impact(k * size) / impact(size) == sqrt(k)``, at every size and every ``k``.

    The ratio is the only observable that distinguishes the square-root law from
    its plausible neighbours: a linear, cube-root or three-halves law all
    produce non-negative, monotone, correctly-signed costs and would satisfy
    every other property in this file. Asserted against
    :data:`~backend.costs.IMPACT_EXPONENT` so the constant the module documents
    and the constant its arithmetic uses cannot drift apart.
    """
    assume(order.notional_usd * factor <= 1e15)
    assume(params.impact_coefficient > 1e-3)
    volatility = (
        params.default_daily_volatility_bps
        if order.daily_volatility_bps is None
        else order.daily_volatility_bps
    )
    assume(volatility > 1.0)

    base = estimate_trade_cost(order, params).impact_bps
    grown = estimate_trade_cost(
        replace(order, notional_usd=order.notional_usd * factor), params
    ).impact_bps

    assert grown / base == pytest.approx(factor**IMPACT_EXPONENT, rel=1e-9)


@given(order=_orders(allow_zero=False), params=_parameter_sets())
def test_the_impact_exponent_is_one_half_and_is_none_of_its_neighbours(
    order: Order, params: CostModelParams
) -> None:
    """A hundredfold size difference must move impact by exactly ten.

    Stated as an anchored ratio because that is the form in which a wrong
    exponent is visible: at 1% of ADV a linear law is ten times a square-root
    law, and at 100% they agree exactly. Every real order lives at the former.
    """
    assume(params.impact_coefficient > 1e-3)
    volatility = (
        params.default_daily_volatility_bps
        if order.daily_volatility_bps is None
        else order.daily_volatility_bps
    )
    assume(volatility > 1.0)

    small = estimate_trade_cost(replace(order, adv_usd=order.notional_usd * 100.0), params)
    full = estimate_trade_cost(replace(order, adv_usd=order.notional_usd), params)

    assert small.participation == pytest.approx(0.01, rel=1e-12)
    assert full.participation == pytest.approx(1.0, rel=1e-12)
    assert full.impact_bps / small.impact_bps == pytest.approx(10.0, rel=1e-9)
    for wrong_exponent in (1.0, 1.0 / 3.0, 1.5):
        assert not math.isclose(
            full.impact_bps / small.impact_bps, 100.0**wrong_exponent, rel_tol=1e-6
        )


@given(order=_orders(allow_zero=False), params=_parameter_sets())
def test_impact_at_one_full_day_of_volume_is_the_coefficient_times_the_volatility(
    order: Order, params: CostModelParams
) -> None:
    """The anchor the square root is measured from, at arbitrary parameters.

    At ``participation == 1`` the exponent drops out entirely, so the impact
    term is exactly ``impact_coefficient * daily_volatility_bps``. Any model
    whose coefficient has quietly absorbed a unit conversion fails here rather
    than at the sizes where the square root could absorb it.
    """
    volatility = (
        params.default_daily_volatility_bps
        if order.daily_volatility_bps is None
        else order.daily_volatility_bps
    )
    at_full_volume = estimate_trade_cost(replace(order, adv_usd=order.notional_usd), params)

    assert at_full_volume.impact_bps == pytest.approx(
        params.impact_coefficient * volatility, rel=1e-9, abs=1e-12
    )


# --------------------------------------------------------------------------
# Composition — the borrow term, the sides, and a basket
# --------------------------------------------------------------------------


@given(order=_orders(allow_zero=False), params=_parameter_sets(), days=_HOLDING_PERIODS)
def test_a_short_exceeds_the_identical_long_by_exactly_the_borrow_term(
    order: Order, params: CostModelParams, days: float
) -> None:
    """Borrow is additive and applies to shorts alone — nothing else moves with the flag.

    If the flag changed anything else, the short book's cost attribution in
    dashboard §6.7 would be describing a different order than the long book's.
    """
    long_order = replace(order, is_short_position=False, holding_period_days=0.0)
    short_order = replace(order, is_short_position=True, holding_period_days=days)

    long_cost = estimate_trade_cost(long_order, params)
    short_cost = estimate_trade_cost(short_order, params)

    assert long_cost.borrow_bps == 0.0
    assert short_cost.borrow_bps == pytest.approx(
        borrow_cost_bps(
            borrow_rate_bps_per_year=params.borrow_rate_bps_per_year, holding_period_days=days
        )
    )
    # The tolerance scales with the TOTALS being differenced, not with the borrow
    # term, because that is where the precision is actually lost. Hypothesis found
    # the case: $1.45bn against $100k of ADV is 14,513x participation, so impact is
    # ~65,556 bps while borrow at 1 bp/year for one day is 0.00278 bps. Two floats
    # near 65,556 are ~1.5e-11 apart at best, so the difference cannot resolve the
    # borrow term to 1e-12 no matter how correct the arithmetic is. Judging a
    # cancellation by the size of the small operand asserts something about IEEE-754
    # rather than about the cost model.
    difference = short_cost.total_bps - long_cost.total_bps
    scale = max(abs(short_cost.total_bps), abs(long_cost.total_bps), 1.0)
    assert abs(difference - short_cost.borrow_bps) <= 8 * sys.float_info.epsilon * scale
    assert (long_cost.half_spread_bps, long_cost.commission_bps, long_cost.impact_bps) == (
        short_cost.half_spread_bps,
        short_cost.commission_bps,
        short_cost.impact_bps,
    )


@given(order=_orders(), params=_parameter_sets())
def test_buying_and_selling_cost_the_same_when_no_borrow_is_involved(
    order: Order, params: CostModelParams
) -> None:
    """The side is carried for the audit trail and must not reach the arithmetic."""
    flat = replace(order, is_short_position=False, holding_period_days=0.0)
    buy = estimate_trade_cost(replace(flat, side=Side.BUY), params)
    sell = estimate_trade_cost(replace(flat, side=Side.SELL), params)

    assert _components(buy) == _components(sell)
    assert buy.total_bps == sell.total_bps


@given(orders=st.lists(_orders(), min_size=0, max_size=6), params=_parameter_sets())
def test_a_basket_costs_exactly_what_its_orders_cost_one_at_a_time(
    orders: list[Order], params: CostModelParams
) -> None:
    """No cross-impact term is hiding in the basket path, and none is missing from it.

    The model documents that it has no cross-impact term, so the basket must be
    exactly the sum. A basket that quietly differed from its legs would make the
    per-name attribution and the portfolio total two different numbers.
    """
    basket = estimate_trade_costs(orders, params)
    assert len(basket) == len(orders)
    for costed, order in zip(basket, orders, strict=True):
        assert costed.total_bps == estimate_trade_cost(order, params).total_bps
        assert costed.total_usd == estimate_trade_cost(order, params).total_usd


@given(order=_orders(), params=_parameter_sets())
def test_a_zero_sized_order_costs_zero_in_both_units(order: Order, params: CostModelParams) -> None:
    """Nothing traded, nothing paid — in bps as well as in dollars.

    The discontinuity in bps at zero is the honest shape (an arbitrarily small
    order still crosses the spread), and it is why both units are reported. What
    must not happen is a zero-sized order carrying a fixed rate into a total.
    """
    cost = estimate_trade_cost(replace(order, notional_usd=0.0), params)

    assert _components(cost) == (0.0, 0.0, 0.0, 0.0)
    assert cost.total_bps == 0.0
    assert cost.total_usd == 0.0
    assert cost.participation == 0.0
    assert cost.uncalibrated is params.uncalibrated


def test_a_sub_float_resolution_size_increase_leaves_impact_exactly_equal() -> None:
    """The falsifying case above, pinned as a regression so the floor is not "tuned away".

    A reader who later loosens the ``1 + 1e-6`` floor in
    :func:`test_cost_is_strictly_increasing_in_order_size_once_impact_bites`
    back to ``> 1.0`` needs to see *why* it is there rather than rediscover it
    through an intermittent CI failure. This is the exact input Hypothesis
    minimised to: one float step in notional, which the square root rounds
    away entirely.

    Equality here is correct behaviour, not a defect. It is the weak
    inequality that must hold universally, and it does.
    """
    params = CostModelParams(
        half_spread_bps=0.0,
        commission_bps=0.0,
        impact_coefficient=1.0,
        default_daily_volatility_bps=2.0,
        borrow_rate_bps_per_year=0.0,
    )
    order = Order(side=Side.BUY, notional_usd=1.0, adv_usd=100_001.0)
    stepped = replace(order, notional_usd=math.nextafter(1.0, math.inf))

    assert stepped.notional_usd > order.notional_usd
    assert estimate_trade_cost(stepped, params).impact_bps == (
        estimate_trade_cost(order, params).impact_bps
    )
