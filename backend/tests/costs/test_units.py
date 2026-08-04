"""Unit conversions for the cost model: bps, fractions, percent, dollars (P9.3).

Directive §8 names this the most common bug class in the domain and notes that
it is silent. These tests exist to make it loud. Every conversion is checked
against a hand-computed value rather than against another conversion, so a
consistently-wrong factor cannot satisfy the suite by agreeing with itself.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from backend.costs import (
    BPS_PER_PERCENT,
    BPS_PER_UNIT,
    PERCENT_PER_UNIT,
    bps_of_notional_to_usd,
    bps_to_fraction,
    bps_to_percent,
    fraction_to_bps,
    percent_to_bps,
    usd_to_bps_of_notional,
)

_finite_values = st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False)


def test_the_constants_are_the_textbook_ones() -> None:
    """1 bp = 1e-4 of notional; 1% = 100 bps. Pinned as literals, not derived."""
    assert BPS_PER_UNIT == 10_000.0
    assert PERCENT_PER_UNIT == 100.0
    assert BPS_PER_PERCENT == 100.0


@pytest.mark.parametrize(
    ("bps", "fraction"),
    [(0.0, 0.0), (1.0, 0.0001), (5.0, 0.0005), (100.0, 0.01), (10_000.0, 1.0)],
)
def test_bps_and_fractions_convert_by_hand_computed_values(bps: float, fraction: float) -> None:
    assert bps_to_fraction(bps) == pytest.approx(fraction, rel=1e-15, abs=1e-18)
    assert fraction_to_bps(fraction) == pytest.approx(bps, rel=1e-15, abs=1e-15)


@pytest.mark.parametrize(
    ("percent", "bps"),
    [(0.0, 0.0), (0.01, 1.0), (0.05, 5.0), (1.0, 100.0), (100.0, 10_000.0)],
)
def test_percent_and_bps_convert_by_hand_computed_values(percent: float, bps: float) -> None:
    assert percent_to_bps(percent) == pytest.approx(bps, rel=1e-15, abs=1e-15)
    assert bps_to_percent(bps) == pytest.approx(percent, rel=1e-15, abs=1e-15)


@given(value=_finite_values)
def test_bps_and_fraction_round_trip(value: float) -> None:
    assert fraction_to_bps(bps_to_fraction(value)) == pytest.approx(value, rel=1e-12, abs=1e-12)


@given(value=_finite_values)
def test_bps_and_percent_round_trip(value: float) -> None:
    assert percent_to_bps(bps_to_percent(value)) == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_a_cost_in_bps_becomes_the_right_number_of_dollars() -> None:
    """5 bps on a $1,000,000 order is $500. The arithmetic every report depends on."""
    assert bps_of_notional_to_usd(5.0, 1_000_000.0) == pytest.approx(500.0)
    assert bps_of_notional_to_usd(1.0, 1_000_000.0) == pytest.approx(100.0)
    assert bps_of_notional_to_usd(26.0, 250_000.0) == pytest.approx(650.0)


def test_the_dollar_conversion_ignores_the_sign_of_the_notional() -> None:
    """A cost is a cost on a sell as much as on a buy; it never comes back negative."""
    assert bps_of_notional_to_usd(5.0, -1_000_000.0) == pytest.approx(500.0)


def test_dollars_and_bps_round_trip_through_a_notional() -> None:
    assert usd_to_bps_of_notional(500.0, 1_000_000.0) == pytest.approx(5.0)
    assert usd_to_bps_of_notional(
        bps_of_notional_to_usd(37.5, 8_432.10), 8_432.10
    ) == pytest.approx(37.5)


def test_a_rate_per_unit_of_nothing_is_refused() -> None:
    """Returning 0.0 or inf here would push an undefined rate into a report."""
    with pytest.raises(ZeroDivisionError):
        usd_to_bps_of_notional(100.0, 0.0)


def test_confusing_a_fraction_for_basis_points_is_a_ten_thousand_fold_error() -> None:
    """The §8 failure mode, stated as a number rather than a warning.

    A caller who has a 0.001 (10 bps) spread as a *fraction* and passes it into
    a ``*_bps`` parameter understates that component by 10,000x. No function
    can detect the mistake — the value is a perfectly ordinary float — so the
    defence is that every parameter name ends in its unit, and the consequence
    is recorded here.
    """
    spread_as_fraction = 0.001
    spread_in_bps = fraction_to_bps(spread_as_fraction)

    correct_usd = bps_of_notional_to_usd(spread_in_bps, 1_000_000.0)
    mistaken_usd = bps_of_notional_to_usd(spread_as_fraction, 1_000_000.0)

    assert spread_in_bps == pytest.approx(10.0)
    assert correct_usd == pytest.approx(1_000.0)
    assert mistaken_usd == pytest.approx(0.10)
    assert correct_usd / mistaken_usd == pytest.approx(BPS_PER_UNIT)


def test_confusing_percent_for_basis_points_is_a_hundred_fold_error() -> None:
    """The other half of the same failure mode: 0.05% is 5 bps, not 0.05 bps."""
    assert percent_to_bps(0.05) / 0.05 == pytest.approx(BPS_PER_PERCENT)


@given(bps=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_every_conversion_preserves_finiteness_and_sign(bps: float) -> None:
    fraction = bps_to_fraction(bps)
    percent = bps_to_percent(bps)
    assert math.isfinite(fraction)
    assert math.isfinite(percent)
    assert fraction >= 0.0
    assert percent >= 0.0
