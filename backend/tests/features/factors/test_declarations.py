"""The declarations: names, units, lags, source tables, and the budget (P5.3).

The declaration is the part of a factor that no test of its arithmetic can
check. A momentum implementation with a wrong availability lag computes exactly
the right number from data it should not have had; the array looks healthy, the
type checker is satisfied, the distribution is plausible, and the backtest
improves. So the lag, the units and the source tables are asserted here as
literal values — not derived from the code under test, which would make the
assertion a restatement rather than a check.

Every expected number in :data:`EXPECTED` is written out by hand from the
justification in the factor's own module docstring. Changing a lag therefore
requires changing this table too, which is the point: it makes a lag change a
visible, deliberate edit rather than a one-character diff nobody reviews.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest

from backend.features.compute import resolve_as_of
from backend.features.errors import AvailabilityLagViolationError
from backend.features.factors import (
    BASELINE_FACTORS,
    FUNDAMENTALS_TABLE,
    PRICE_SOURCE_TABLE,
    SHORT_INTEREST_TABLE,
)
from backend.features.registry import default_registry
from backend.features.spec import MAX_FEATURES, FeatureSpec, compute_instant

COMPUTE_DATE = dt.date(2026, 3, 2)
"""An arbitrary Monday used wherever a rebalance date is needed."""

ZERO_LAG = dt.timedelta(0)
FUNDAMENTAL_LAG = dt.timedelta(days=7)
SHORT_INTEREST_LAG = dt.timedelta(days=17)
"""14 days of publication gap + 1 intraday + 2 vendor redistribution (P5.3 wave 2)."""


@dataclass(frozen=True, slots=True)
class Expectation:
    """What one factor's declaration must say, written out independently."""

    lag: dt.timedelta
    tables: frozenset[str]
    units_contains: str
    premium_sign: str


EXPECTED: dict[str, Expectation] = {
    # Price factors: lag zero. A daily close is knowable at that close, and any
    # vendor delivery delay is carried by the connector's knowledge_time under
    # D-011 rather than by a margin here.
    "momentum_12_1": Expectation(
        lag=ZERO_LAG,
        tables=frozenset({PRICE_SOURCE_TABLE}),
        units_contains="log return",
        premium_sign="POSITIVE",
    ),
    "short_term_reversal": Expectation(
        lag=ZERO_LAG,
        tables=frozenset({PRICE_SOURCE_TABLE}),
        units_contains="negative log return",
        premium_sign="POSITIVE",
    ),
    "low_volatility": Expectation(
        lag=ZERO_LAG,
        tables=frozenset({PRICE_SOURCE_TABLE}),
        units_contains="negative annualized standard deviation",
        premium_sign="POSITIVE",
    ),
    # Fundamental factors: 7 days. A margin on top of the store's knowledge_time
    # covering three B1-blocked uncertainties in the unwritten P3.5 connector —
    # date-only datekey resolved to the next trading day (4), filing date versus
    # acceptance instant (1), vendor delivery after filing (2).
    "book_to_price": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE}),
        units_contains="dimensionless ratio",
        premium_sign="POSITIVE",
    ),
    "earnings_yield": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE}),
        units_contains="dimensionless ratio",
        premium_sign="POSITIVE",
    ),
    "gross_profitability": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE}),
        units_contains="dimensionless ratio",
        premium_sign="POSITIVE",
    ),
    "roic": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE}),
        units_contains="dimensionless ratio",
        premium_sign="POSITIVE",
    ),
    "accruals": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE}),
        units_contains="dimensionless ratio",
        premium_sign="NEGATIVE",
    ),
    "asset_growth": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE}),
        units_contains="dimensionless fraction",
        premium_sign="NEGATIVE",
    ),
    # Added in P5.3 wave 2, completing the eleven factors PLAN.md names.
    "amihud_illiquidity": Expectation(
        lag=ZERO_LAG,
        tables=frozenset({PRICE_SOURCE_TABLE}),
        units_contains="1/USD",
        premium_sign="POSITIVE",
    ),
    "size": Expectation(
        lag=FUNDAMENTAL_LAG,
        tables=frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE}),
        units_contains="natural log of market capitalization",
        premium_sign="NEGATIVE",
    ),
    "short_interest": Expectation(
        lag=SHORT_INTEREST_LAG,
        tables=frozenset({SHORT_INTEREST_TABLE, FUNDAMENTALS_TABLE}),
        units_contains="dimensionless fraction",
        premium_sign="NEGATIVE",
    ),
}

BY_NAME: dict[str, FeatureSpec] = {spec.name: spec for spec in BASELINE_FACTORS}


def test_exactly_the_planned_baseline_factors_are_declared() -> None:
    """P5.3's twelve baseline factors, no more and no fewer."""
    assert sorted(BY_NAME) == sorted(EXPECTED)


def test_every_declaration_is_registered_in_the_process_wide_catalog() -> None:
    """Importing the package populates the registry the 30-feature cap is about."""
    registry = default_registry()
    for name, spec in BY_NAME.items():
        assert name in registry
        assert registry.spec(name) is spec


def test_the_catalog_tuple_is_sorted_by_name_like_the_registry() -> None:
    """``BASELINE_FACTORS`` enumerates the way ``FeatureRegistry.specs()`` does."""
    assert [spec.name for spec in BASELINE_FACTORS] == sorted(BY_NAME)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_declared_availability_lag_is_the_justified_one(name: str) -> None:
    """The lag is the number the module docstring argues for, written out here by hand."""
    assert BY_NAME[name].availability_lag == EXPECTED[name].lag


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_declared_source_tables_are_exactly_what_the_factor_reads(name: str) -> None:
    """Impact analysis depends on this: it says which features a connector change touches."""
    assert BY_NAME[name].source_tables == EXPECTED[name].tables


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_units_name_the_family_the_number_belongs_to(name: str) -> None:
    """Directive §8: a float64 column carries no unit, so the declaration must."""
    assert EXPECTED[name].units_contains in BY_NAME[name].units


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_units_are_not_percent_or_basis_points(name: str) -> None:
    """Every baseline factor is a fraction or a ratio; the other two forms are the bug class."""
    units = BY_NAME[name].units.lower()
    assert "percent" not in units
    assert "basis point" not in units


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_definition_states_the_expected_premium_sign(name: str) -> None:
    """P5.4 tests a premium against a stated expectation, not against a remembered one."""
    definition = BY_NAME[name].definition
    assert f"Expected premium sign: {EXPECTED[name].premium_sign}" in definition


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_knowledge_cutoff_is_the_compute_instant_minus_the_declared_lag(name: str) -> None:
    """The lag's entire operational content, checked on a concrete date."""
    spec = BY_NAME[name]
    expected = compute_instant(COMPUTE_DATE) - EXPECTED[name].lag
    assert spec.knowledge_cutoff(COMPUTE_DATE) == expected


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_caller_may_not_ask_for_an_instant_fresher_than_the_lag_permits(name: str) -> None:
    """The declared lag is enforced before any I/O, per factor.

    One microsecond past the cutoff is refused. That granularity matters: a lag
    is not a hint about staleness, it is the boundary of what the feature is
    entitled to know, and a boundary that tolerates "just a bit fresher" is not
    a boundary.
    """
    spec = BY_NAME[name]
    permitted = spec.knowledge_cutoff(COMPUTE_DATE)
    assert resolve_as_of(spec, COMPUTE_DATE) == permitted
    with pytest.raises(AvailabilityLagViolationError):
        resolve_as_of(spec, COMPUTE_DATE, requested_as_of=permitted + dt.timedelta(microseconds=1))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_caller_may_reconstruct_the_factor_at_an_older_instant(name: str) -> None:
    """Reading older data is always I1-safe: it is what a historical replay does."""
    spec = BY_NAME[name]
    older = spec.knowledge_cutoff(COMPUTE_DATE) - dt.timedelta(days=30)
    assert resolve_as_of(spec, COMPUTE_DATE, requested_as_of=older) == older


def test_the_factors_whose_slowest_leg_is_fundamentals_share_one_lag() -> None:
    """One table, one undecided connector policy, one margin — not many opinions.

    Scoped to factors whose sources are fundamentals and prices *only*, because
    those are the ones whose margin is entirely the P3.5 connector's unresolved
    policy. A factor reading a third, slower source is governed by that source
    instead; see the companion test below.
    """
    fundamental_regime = frozenset({FUNDAMENTALS_TABLE, PRICE_SOURCE_TABLE})
    lags = {
        BY_NAME[name].availability_lag
        for name, expectation in EXPECTED.items()
        if FUNDAMENTALS_TABLE in expectation.tables and expectation.tables <= fundamental_regime
    }
    assert lags == {FUNDAMENTAL_LAG}


def test_every_factor_reading_fundamentals_covers_at_least_the_fundamentals_lag() -> None:
    """A slower second source raises the margin; it must never lower it.

    ``short_interest`` reads the share count for its denominator, so whatever
    its publication gap demands, it must still cover the fundamentals leg. This
    is the max-not-sum rule stated as a check: 17 days is correct because it
    exceeds both regimes, and 24 (the sum) would be needless signal loss while
    anything under 7 would read a share count before it was knowable.
    """
    for name, expectation in EXPECTED.items():
        if FUNDAMENTALS_TABLE in expectation.tables:
            assert BY_NAME[name].availability_lag >= FUNDAMENTAL_LAG, name


def test_the_price_factors_carry_no_margin_of_their_own() -> None:
    """Zero, because the store's knowledge_time already carries any delivery delay."""
    price_only_lags = {
        BY_NAME[name].availability_lag
        for name, expectation in EXPECTED.items()
        if expectation.tables == frozenset({PRICE_SOURCE_TABLE})
    }
    assert price_only_lags == {ZERO_LAG}


def test_the_baseline_factors_fit_inside_the_thirty_feature_cap() -> None:
    """Directive §5: the cap includes Phase 7's LLM features, so headroom is the finding."""
    registry = default_registry()
    assert len(BASELINE_FACTORS) == 12
    assert set(BY_NAME) <= set(registry.names())
    assert len(registry) <= MAX_FEATURES
    assert registry.remaining_capacity() == MAX_FEATURES - len(registry)
    # Every factor PLAN.md P5.3 names is now declared; Phase 7 spends the rest.
    # Twelve factors must not have eaten the budget.
    assert registry.remaining_capacity() >= 18
