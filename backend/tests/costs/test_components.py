"""Cost components in isolation: spread, participation, impact, borrow (P9.3).

Each component is tested against hand-computed numbers and against the
structural properties the composed model relies on. The impact exponent gets
the most attention: a wrong exponent produces perfectly plausible costs at
every order size and is invisible in any output, so it is verified numerically
rather than read off the source.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from backend.costs import (
    BORROW_DAY_COUNT_BASIS,
    IMPACT_EXPONENT,
    CostParameterError,
    borrow_cost_bps,
    half_spread_bps_from_quoted_spread_bps,
    half_spread_bps_from_quotes,
    participation_rate,
    quoted_spread_bps_from_quotes,
    square_root_impact_bps,
)

_DAILY_VOL_BPS = 200.0
"""2% per day, the model's uncalibrated default, in basis points."""


# --------------------------------------------------------------------------
# Spread
# --------------------------------------------------------------------------


def test_quoted_spread_from_quotes_is_hand_computable() -> None:
    """A $0.01 spread on a $50.00 mid is 2 bps. Full stop."""
    assert quoted_spread_bps_from_quotes(bid=49.995, ask=50.005) == pytest.approx(2.0)
    assert half_spread_bps_from_quotes(bid=49.995, ask=50.005) == pytest.approx(1.0)


def test_the_half_spread_is_exactly_half_the_quoted_spread() -> None:
    """The factor of two that a round trip pays twice, isolated in one function."""
    assert half_spread_bps_from_quoted_spread_bps(10.0) == 5.0
    assert half_spread_bps_from_quoted_spread_bps(0.0) == 0.0


@given(
    quoted=st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False),
)
def test_two_half_spreads_reconstruct_the_quoted_spread(quoted: float) -> None:
    assert 2.0 * half_spread_bps_from_quoted_spread_bps(quoted) == pytest.approx(quoted)


def test_a_locked_market_has_zero_spread() -> None:
    assert quoted_spread_bps_from_quotes(bid=50.0, ask=50.0) == 0.0


def test_a_crossed_book_is_refused_rather_than_absolute_valued() -> None:
    """Taking |ask - bid| would hide a data error inside a plausible cost."""
    with pytest.raises(CostParameterError, match="crossed book"):
        quoted_spread_bps_from_quotes(bid=50.01, ask=49.99)


@pytest.mark.parametrize(("bid", "ask"), [(0.0, 50.0), (50.0, 0.0), (-1.0, 50.0)])
def test_non_positive_quotes_are_refused(bid: float, ask: float) -> None:
    with pytest.raises(CostParameterError):
        quoted_spread_bps_from_quotes(bid=bid, ask=ask)


def test_a_negative_quoted_spread_is_refused() -> None:
    with pytest.raises(CostParameterError, match="quoted_spread_bps"):
        half_spread_bps_from_quoted_spread_bps(-1.0)


# --------------------------------------------------------------------------
# Participation rate
# --------------------------------------------------------------------------


def test_participation_is_notional_over_adv() -> None:
    """1% of ADV is 0.01, dimensionless. The input to the square-root law."""
    assert participation_rate(notional_usd=1_000_000.0, adv_usd=100_000_000.0) == pytest.approx(
        0.01
    )
    assert participation_rate(notional_usd=100_000_000.0, adv_usd=100_000_000.0) == 1.0


def test_participation_ignores_the_sign_of_the_notional() -> None:
    """A $1m sell consumes as much of the day's volume as a $1m buy."""
    assert participation_rate(notional_usd=-1_000_000.0, adv_usd=1e8) == pytest.approx(0.01)


def test_participation_above_one_is_permitted_and_is_extrapolation() -> None:
    """Refusing it would be a policy; the docstring states it is beyond the fit."""
    assert participation_rate(notional_usd=2e8, adv_usd=1e8) == pytest.approx(2.0)


@pytest.mark.parametrize("adv", [0.0, -1.0])
def test_a_name_with_no_volume_has_no_participation_rate(adv: float) -> None:
    """Substituting zero or infinity would put an unmodellable order in at a finite cost."""
    with pytest.raises(CostParameterError, match="adv_usd"):
        participation_rate(notional_usd=1_000.0, adv_usd=adv)


def test_a_non_finite_notional_is_refused() -> None:
    with pytest.raises(CostParameterError, match="notional_usd"):
        participation_rate(notional_usd=math.nan, adv_usd=1e8)


# --------------------------------------------------------------------------
# Impact — the exponent is verified numerically, not read off the source
# --------------------------------------------------------------------------


def _impact(participation: float) -> float:
    return square_root_impact_bps(
        participation=participation,
        daily_volatility_bps=_DAILY_VOL_BPS,
        impact_coefficient=1.0,
    )


def test_the_impact_exponent_is_one_half_measured_from_the_output() -> None:
    """A 1%-of-ADV order against a 100%-of-ADV order: the ratio must be exactly 10.

    Participation differs by a factor of 100, so a square-root law differs by
    ``sqrt(100) == 10``. This is the check the exponent needs, because every
    plausible alternative also produces plausible-looking costs:

    - a **linear** law would give a factor of 100 — costs 10x too high at 1% of
      ADV once the coefficient is fitted at the top end, and vice versa;
    - a **cube-root** law would give ``100 ** (1/3) ≈ 4.64``;
    - a **3/2-power** law would give 1000.

    None of those would fail a monotonicity test, a zero-size test, or a units
    test. Only the ratio distinguishes them, so the ratio is asserted.
    """
    impact_at_one_percent = _impact(0.01)
    impact_at_full_adv = _impact(1.0)

    assert impact_at_full_adv / impact_at_one_percent == pytest.approx(10.0, rel=1e-12)

    # And explicitly not the plausible neighbours.
    for wrong_exponent in (1.0, 1.0 / 3.0, 1.5):
        wrong_ratio = 100.0**wrong_exponent
        assert not math.isclose(
            impact_at_full_adv / impact_at_one_percent, wrong_ratio, rel_tol=1e-6
        )


def test_the_named_exponent_matches_the_arithmetic_it_documents() -> None:
    """``IMPACT_EXPONENT`` is exported for the tests and the reader; it must not lie."""
    assert IMPACT_EXPONENT == 0.5
    assert _impact(0.04) / _impact(0.01) == pytest.approx(4.0**IMPACT_EXPONENT, rel=1e-12)


@given(
    smaller=st.floats(min_value=1e-6, max_value=0.5, allow_nan=False, allow_infinity=False),
    factor=st.floats(min_value=1.5, max_value=1e4, allow_nan=False, allow_infinity=False),
)
def test_the_exponent_holds_at_every_pair_of_sizes(smaller: float, factor: float) -> None:
    """Property form: scaling participation by ``k`` scales impact by ``sqrt(k)``."""
    assert _impact(smaller * factor) / _impact(smaller) == pytest.approx(
        factor**IMPACT_EXPONENT, rel=1e-9
    )


def test_impact_at_one_full_day_of_volume_is_the_coefficient_times_volatility() -> None:
    """The model's anchor: participation 1.0 removes the square root entirely."""
    assert square_root_impact_bps(
        participation=1.0, daily_volatility_bps=200.0, impact_coefficient=0.7
    ) == pytest.approx(140.0)


def test_impact_is_hand_computable_at_one_percent_of_adv() -> None:
    """1.0 * 200 bps * sqrt(0.01) = 20 bps. The number the defaults produce."""
    assert _impact(0.01) == pytest.approx(20.0)


def test_a_zero_sized_order_has_no_impact() -> None:
    assert _impact(0.0) == 0.0


def test_impact_is_zero_when_the_name_does_not_move() -> None:
    """Impact is quoted as a multiple of volatility, so zero volatility is zero impact."""
    assert square_root_impact_bps(
        participation=0.5, daily_volatility_bps=0.0, impact_coefficient=1.0
    ) == pytest.approx(0.0)


@given(
    participation=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    extra=st.floats(min_value=1e-9, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_impact_is_monotone_in_participation(participation: float, extra: float) -> None:
    assert _impact(participation + extra) >= _impact(participation)


@pytest.mark.parametrize(
    ("participation", "volatility", "coefficient"),
    [(-1e-9, 200.0, 1.0), (0.1, -1.0, 1.0), (0.1, 200.0, -0.1), (math.nan, 200.0, 1.0)],
)
def test_impact_refuses_negative_or_non_finite_inputs(
    participation: float, volatility: float, coefficient: float
) -> None:
    with pytest.raises(CostParameterError):
        square_root_impact_bps(
            participation=participation,
            daily_volatility_bps=volatility,
            impact_coefficient=coefficient,
        )


# --------------------------------------------------------------------------
# Borrow
# --------------------------------------------------------------------------


def test_borrow_accrues_on_an_act_360_basis() -> None:
    """100 bps/year held 36 days is 10 bps: rate * days / 360, hand-computed."""
    assert BORROW_DAY_COUNT_BASIS == 360.0
    assert borrow_cost_bps(
        borrow_rate_bps_per_year=100.0, holding_period_days=36.0
    ) == pytest.approx(10.0)


def test_a_full_calendar_year_accrues_slightly_more_than_the_quoted_rate() -> None:
    """ACT/360 over 365 days is 365/360 of the annual rate — the conservative convention."""
    assert borrow_cost_bps(
        borrow_rate_bps_per_year=100.0, holding_period_days=365.0
    ) == pytest.approx(100.0 * 365.0 / 360.0)


def test_a_typical_monthly_short_hold_costs_a_few_basis_points() -> None:
    """The default rate over a 21-day holding period, as it will appear in a backtest."""
    assert borrow_cost_bps(
        borrow_rate_bps_per_year=100.0, holding_period_days=21.0
    ) == pytest.approx(100.0 * 21.0 / 360.0)


@pytest.mark.parametrize(("rate", "days"), [(0.0, 30.0), (100.0, 0.0), (0.0, 0.0)])
def test_borrow_is_zero_when_either_factor_is_zero(rate: float, days: float) -> None:
    assert borrow_cost_bps(borrow_rate_bps_per_year=rate, holding_period_days=days) == 0.0


@given(
    rate=st.floats(min_value=0.0, max_value=5_000.0, allow_nan=False, allow_infinity=False),
    days=st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    extra_days=st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
)
def test_borrow_is_monotone_in_the_holding_period(
    rate: float, days: float, extra_days: float
) -> None:
    """A short held longer never costs less to finance."""
    assert borrow_cost_bps(
        borrow_rate_bps_per_year=rate, holding_period_days=days + extra_days
    ) >= borrow_cost_bps(borrow_rate_bps_per_year=rate, holding_period_days=days)


def test_a_negative_borrow_rate_is_refused() -> None:
    """Being paid to short exists, but this model does not represent it."""
    with pytest.raises(CostParameterError, match="borrow_rate_bps_per_year"):
        borrow_cost_bps(borrow_rate_bps_per_year=-1.0, holding_period_days=30.0)


def test_a_negative_holding_period_is_refused() -> None:
    with pytest.raises(CostParameterError, match="holding_period_days"):
        borrow_cost_bps(borrow_rate_bps_per_year=100.0, holding_period_days=-1.0)
