"""VWAP/TWAP slicing: exact summation, the I1 bound on the forecast, and what stops slicing.

Three claims carry this suite, and each is asserted directly rather than inferred
from the absence of a failure:

1. **A schedule sums to exactly the parent quantity.** Property-tested over
   random quantities, bucket counts and minimums, and re-asserted by the
   :class:`~backend.execution.slicing.SliceSchedule` constructor against a
   hand-built schedule that does not sum.
2. **The volume forecast carries no lookahead (I1).** Tested on all three routes
   that could admit one — the forecast's own constructor, the fitted constructor
   that derives its bound from the sessions it consumed, and the schedule that
   re-checks against the day it actually trades. Same-day is a violation, not a
   boundary case, so the ``>=`` is tested at equality specifically.
3. **The cost model alone never stops slicing.** Costed through
   :func:`~backend.costs.model.estimate_trade_cost` — the platform's model, not a
   second one written here (I4) — total modelled cost falls monotonically all the
   way to :data:`~backend.execution.slicing.MAX_SLICE_COUNT`, impact falling as
   ``1/sqrt(n)`` while spread and commission do not move at all. That is the
   evidence that the slice-count bound has to be exogenous, and the test says so.

Thresholds are asserted as **literals**, never derived from the constants they
check (D-037: a test that computes its expectation from the value under test
cannot detect that value being wrong).
"""

from __future__ import annotations

import datetime as dt
import inspect
import itertools
import math
import re
from pathlib import Path
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.costs.model import UNCALIBRATED_DEFAULTS
from backend.execution import slicing
from backend.execution.errors import (
    ForecastLookaheadError,
    OrderValidationError,
    SliceScheduleError,
    VolumeForecastError,
)
from backend.execution.idempotency import idempotency_key
from backend.execution.orders import OrderIntent
from backend.execution.slicing import (
    BUCKET_MINUTES,
    MAX_SLICE_COUNT,
    MINIMUM_SLICE_SHARES_FLOOR,
    REGULAR_SESSION_MINUTES,
    ROUND_LOT_SHARES,
    U_CURVE_EDGE_TO_MIDDLE_RATIO,
    ForecastBasis,
    ScheduleCost,
    SessionVolume,
    SliceAlgorithm,
    SliceSchedule,
    VolumeForecast,
    apportion_shares,
    feasible_slice_count,
    modelled_schedule_cost,
    observed_volume_forecast,
    plan_twap,
    plan_vwap,
    stylised_u_curve_forecast,
    stylised_u_curve_weights,
    uniform_volume_forecast,
)
from backend.tests.execution.fixtures import make_intent, make_stamp
from backend.tests.execution.test_paper_only import FORBIDDEN_PARAMETERS, MODULE_PATHS

TRADING_DATE: Final = dt.date(2026, 8, 3)
"""The rebalance date every fixture parent trades on."""

DIVISIBLE_QUANTITY: Final = 27_720
"""A parent quantity divisible by every integer from 1 to 12.

Used where a claim is about *equal* slices — the ``1/sqrt(n)`` impact law — so
the largest-remainder rule contributes nothing and the arithmetic can be asserted
exactly rather than approximately.
"""

quantities = st.integers(min_value=1, max_value=5_000_000)
bucket_counts = st.integers(min_value=1, max_value=MAX_SLICE_COUNT)
minimums = st.integers(min_value=1, max_value=500)
weight_lists = st.lists(
    st.integers(min_value=1, max_value=10_000),
    min_size=1,
    max_size=MAX_SLICE_COUNT,
)


def parent(quantity_shares: int = 10_000) -> OrderIntent:
    """Build an unsliced parent order of the given size."""
    return make_intent(quantity_shares=quantity_shares, rebalance_date=TRADING_DATE)


def session(day: dt.date, volumes: tuple[int, ...]) -> SessionVolume:
    """Build one observed session."""
    return SessionVolume(session_date=day, bucket_volume_shares=volumes)


# ---------------------------------------------------------------------------
# Constants: asserted as literals, per D-037.
# ---------------------------------------------------------------------------


def test_the_horizon_and_lot_constants_are_the_numbers_they_are_documented_as() -> None:
    # D-037's lesson: an aborted mutation run once left MAX_SNAPSHOT_SKEW_SECONDS
    # at 10**9 and the suite did not notice, because its tests were written
    # relative to the constant. These are literals for that reason.
    assert REGULAR_SESSION_MINUTES == 390
    assert BUCKET_MINUTES == 30
    assert MAX_SLICE_COUNT == 13
    assert ROUND_LOT_SHARES == 100
    assert MINIMUM_SLICE_SHARES_FLOOR == 1
    assert U_CURVE_EDGE_TO_MIDDLE_RATIO == 3
    # And the derivation is the derivation, not a coincidence of two literals.
    assert MAX_SLICE_COUNT == REGULAR_SESSION_MINUTES // BUCKET_MINUTES


# ---------------------------------------------------------------------------
# Exact summation.
# ---------------------------------------------------------------------------


@settings(max_examples=400)
@given(total=quantities, weights=weight_lists, minimum=minimums)
def test_apportionment_sums_to_exactly_the_total_or_refuses(
    total: int, weights: list[int], minimum: int
) -> None:
    if total < len(weights) * minimum:
        with pytest.raises(SliceScheduleError):
            apportion_shares(
                total_shares=total, weight_units=tuple(weights), minimum_shares=minimum
            )
        return
    allocation = apportion_shares(
        total_shares=total, weight_units=tuple(weights), minimum_shares=minimum
    )
    assert sum(allocation) == total
    assert len(allocation) == len(weights)
    assert all(share >= minimum for share in allocation)


@settings(max_examples=300)
@given(quantity=quantities, count=bucket_counts, minimum=minimums)
def test_every_twap_schedule_sums_to_exactly_the_parent_quantity(
    quantity: int, count: int, minimum: int
) -> None:
    schedule = plan_twap(parent(quantity), bucket_count=count, minimum_slice_shares=minimum)
    assert sum(schedule.quantities) == quantity
    assert schedule.parent_quantity_shares == quantity
    assert 1 <= schedule.slice_count <= MAX_SLICE_COUNT


@settings(max_examples=300)
@given(quantity=quantities, count=bucket_counts, minimum=minimums)
def test_every_vwap_schedule_sums_to_exactly_the_parent_quantity(
    quantity: int, count: int, minimum: int
) -> None:
    forecast = stylised_u_curve_forecast(bucket_count=count, as_of=TRADING_DATE)
    if quantity < count * minimum and count > 1:
        with pytest.raises(SliceScheduleError):
            plan_vwap(parent(quantity), forecast=forecast, minimum_slice_shares=minimum)
        return
    schedule = plan_vwap(parent(quantity), forecast=forecast, minimum_slice_shares=minimum)
    assert sum(schedule.quantities) == quantity
    assert schedule.slice_count == count


def test_a_schedule_whose_slices_do_not_sum_to_the_parent_cannot_be_constructed() -> None:
    # The property belongs to the type, not only to the function that builds it:
    # this schedule is assembled by hand and is refused on the way in.
    base = parent(1_000)
    children = tuple(
        make_intent(
            quantity_shares=quantity,
            rebalance_date=TRADING_DATE,
            slice_index=index,
            slice_count=2,
        )
        for index, quantity in enumerate((400, 500))
    )
    with pytest.raises(SliceScheduleError, match="sum to 900 shares but the parent is 1000"):
        SliceSchedule(
            algorithm=SliceAlgorithm.TWAP,
            parent_quantity_shares=base.quantity_shares,
            minimum_slice_shares=100,
            forecast=uniform_volume_forecast(bucket_count=2, as_of=TRADING_DATE),
            slices=children,
        )


def test_the_remainder_is_never_dropped_on_a_quantity_that_does_not_divide() -> None:
    # 1,003 over ten buckets is the canonical leak: naive floor division loses
    # three shares that no order ever trades.
    schedule = plan_twap(parent(1_003), bucket_count=10, minimum_slice_shares=1)
    assert sum(schedule.quantities) == 1_003
    assert schedule.slice_count == 10
    assert max(schedule.quantities) - min(schedule.quantities) == 1


# ---------------------------------------------------------------------------
# The minimum-slice rule and the horizon: what stops infinite slicing.
# ---------------------------------------------------------------------------


@settings(max_examples=300)
@given(quantity=quantities, count=bucket_counts, minimum=minimums)
def test_no_slice_is_ever_zero_negative_or_below_the_minimum(
    quantity: int, count: int, minimum: int
) -> None:
    schedule = plan_twap(parent(quantity), bucket_count=count, minimum_slice_shares=minimum)
    assert schedule.minimum_slice_shares >= MINIMUM_SLICE_SHARES_FLOOR
    for share_count in schedule.quantities:
        assert share_count > 0
        assert share_count >= schedule.minimum_slice_shares


@settings(max_examples=300)
@given(quantity=quantities, count=st.integers(min_value=1, max_value=200), minimum=minimums)
def test_the_slice_count_never_exceeds_the_horizon_or_what_the_quantity_supports(
    quantity: int, count: int, minimum: int
) -> None:
    feasible = feasible_slice_count(
        quantity_shares=quantity, requested_count=count, minimum_slice_shares=minimum
    )
    assert 1 <= feasible <= 13  # literal, per D-037
    assert feasible <= count
    assert feasible == 1 or feasible * minimum <= quantity


def test_the_round_lot_default_bounds_a_thousand_share_parent_at_ten_slices() -> None:
    # The minimum-slice rule doing its job with the shipped default, stated as a
    # worked number rather than as an inequality.
    assert feasible_slice_count(quantity_shares=1_000, requested_count=MAX_SLICE_COUNT) == 10
    assert plan_twap(parent(1_000), bucket_count=MAX_SLICE_COUNT).slice_count == 10


def test_a_parent_smaller_than_one_lot_is_traded_as_a_single_order() -> None:
    # Never zero slices: a parent too small to slice still has to be traded.
    schedule = plan_twap(parent(50), bucket_count=5)
    assert schedule.slice_count == 1
    assert schedule.quantities == (50,)
    # The minimum recorded is the one actually applied — a single slice *is* the
    # parent, so there is nothing for the round lot to constrain.
    assert schedule.minimum_slice_shares == 50


def test_a_minimum_of_zero_is_refused_because_it_reopens_unbounded_slicing() -> None:
    with pytest.raises(SliceScheduleError, match="at least 1"):
        feasible_slice_count(quantity_shares=1_000, requested_count=5, minimum_slice_shares=0)
    with pytest.raises(SliceScheduleError, match="at least 1"):
        apportion_shares(total_shares=100, weight_units=(1, 1), minimum_shares=0)


def test_a_forecast_wider_than_the_quantity_supports_is_refused_not_reshaped() -> None:
    # Truncating a 13-bucket U into 5 buckets is not a 5-bucket U, and silently
    # substituting one volume shape for another is the failure this refusal
    # exists for. The message names the count that would have worked.
    forecast = stylised_u_curve_forecast(bucket_count=MAX_SLICE_COUNT, as_of=TRADING_DATE)
    with pytest.raises(SliceScheduleError, match=r"at most 5 bucket\(s\)"):
        plan_vwap(parent(500), forecast=forecast, minimum_slice_shares=ROUND_LOT_SHARES)


def test_a_forecast_wider_than_the_session_cannot_be_built_at_all() -> None:
    with pytest.raises(VolumeForecastError, match="between 1 and 13 buckets"):
        uniform_volume_forecast(bucket_count=MAX_SLICE_COUNT + 1, as_of=TRADING_DATE)


def test_every_slice_carries_the_parents_rebalance_date_so_a_schedule_cannot_span_sessions() -> (
    None
):
    schedule = plan_twap(parent(10_000), bucket_count=MAX_SLICE_COUNT)
    assert {child.rebalance_date for child in schedule.slices} == {TRADING_DATE}
    assert schedule.rebalance_date == TRADING_DATE


# ---------------------------------------------------------------------------
# TWAP: uniform up to the remainder rule.
# ---------------------------------------------------------------------------


@settings(max_examples=300)
@given(quantity=quantities, count=bucket_counts, minimum=minimums)
def test_twap_is_uniform_up_to_the_remainder_rule(quantity: int, count: int, minimum: int) -> None:
    schedule = plan_twap(parent(quantity), bucket_count=count, minimum_slice_shares=minimum)
    allocation = schedule.quantities
    assert max(allocation) - min(allocation) <= 1
    # And the remainder goes to the *earliest* buckets, deterministically: a
    # schedule can be halted part-way, so quantity left for later may never trade.
    applied = schedule.minimum_slice_shares
    discretionary = quantity - schedule.slice_count * applied
    base, spare = divmod(discretionary, schedule.slice_count)
    for index, share_count in enumerate(allocation):
        assert share_count == applied + base + (1 if index < spare else 0)


def test_twap_is_planned_against_a_forecast_that_declares_no_shape() -> None:
    schedule = plan_twap(parent(10_000), bucket_count=5)
    assert schedule.algorithm is SliceAlgorithm.TWAP
    assert schedule.forecast.basis is ForecastBasis.UNIFORM
    assert schedule.forecast.weight_units == (1, 1, 1, 1, 1)
    assert schedule.forecast.fitted is False
    assert schedule.forecast.observed_through is None


def test_twap_and_a_vwap_on_a_uniform_forecast_allocate_identically() -> None:
    # TWAP is the degenerate VWAP; there is no second piece of arithmetic that
    # could disagree. The *label* still differs, because they are not the same
    # decision even when they are the same numbers.
    quantity = 7_777
    twap = plan_twap(parent(quantity), bucket_count=6)
    vwap = plan_vwap(
        parent(quantity),
        forecast=uniform_volume_forecast(bucket_count=6, as_of=TRADING_DATE),
    )
    assert twap.quantities == vwap.quantities
    assert twap.algorithm is SliceAlgorithm.TWAP
    assert vwap.algorithm is SliceAlgorithm.VWAP


# ---------------------------------------------------------------------------
# VWAP: the allocation tracks its forecast.
# ---------------------------------------------------------------------------


@settings(max_examples=400)
@given(total=quantities, weights=weight_lists, minimum=minimums)
def test_no_slice_differs_from_its_exact_quota_by_a_whole_share(
    total: int, weights: list[int], minimum: int
) -> None:
    # The largest-remainder guarantee, in integers so the assertion is exact:
    #   |(allocation_i - minimum) * W - D * w_i|  <  W
    # which is |allocation_i - quota_i| < 1 share with nothing divided.
    if total < len(weights) * minimum:
        return
    allocation = apportion_shares(
        total_shares=total, weight_units=tuple(weights), minimum_shares=minimum
    )
    total_weight = sum(weights)
    discretionary = total - len(weights) * minimum
    for share_count, weight in zip(allocation, weights, strict=True):
        deviation = (share_count - minimum) * total_weight - discretionary * weight
        assert abs(deviation) < total_weight


@settings(max_examples=400)
@given(total=quantities, weights=weight_lists, minimum=minimums)
def test_a_bucket_with_more_expected_volume_never_receives_fewer_shares(
    total: int, weights: list[int], minimum: int
) -> None:
    if total < len(weights) * minimum:
        return
    allocation = apportion_shares(
        total_shares=total, weight_units=tuple(weights), minimum_shares=minimum
    )
    for i, (share_i, weight_i) in enumerate(zip(allocation, weights, strict=True)):
        for j, (share_j, weight_j) in enumerate(zip(allocation, weights, strict=True)):
            if i == j:
                continue
            if weight_i > weight_j:
                assert share_i >= share_j
            elif weight_i == weight_j:
                # Equal weights may differ by the single share the index
                # tie-break hands to the earlier bucket, and by no more.
                assert abs(share_i - share_j) <= 1


def test_a_vwap_schedule_reproduces_the_shape_of_its_forecast() -> None:
    forecast = stylised_u_curve_forecast(bucket_count=13, as_of=TRADING_DATE)
    schedule = plan_vwap(parent(DIVISIBLE_QUANTITY), forecast=forecast, minimum_slice_shares=1)
    allocation = schedule.quantities
    # Heavier at both edges than in the middle, and the middle is the trough.
    assert allocation[0] > allocation[3] > allocation[6]
    assert allocation[-1] > allocation[-4] > allocation[6]
    assert min(allocation) == allocation[6]
    # And the edge/middle share ratio is the forecast's ratio, to within the
    # rounding the integer rule allows.
    assert allocation[0] / allocation[6] == pytest.approx(U_CURVE_EDGE_TO_MIDDLE_RATIO, rel=1e-3)


# ---------------------------------------------------------------------------
# The stylised U curve is an assumption (I3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [3, 5, 7, 9, 11, 13])
def test_the_u_curve_is_symmetric_and_carries_the_declared_edge_to_middle_ratio(
    count: int,
) -> None:
    weights = stylised_u_curve_weights(count)
    assert weights == tuple(reversed(weights))
    middle = weights[count // 2]
    assert weights[0] == U_CURVE_EDGE_TO_MIDDLE_RATIO * middle
    assert weights[-1] == U_CURVE_EDGE_TO_MIDDLE_RATIO * middle
    assert min(weights) == middle


def test_a_single_bucket_has_no_shape() -> None:
    assert stylised_u_curve_weights(1) == (1,)


def test_the_u_curve_forecast_labels_itself_an_assumption_rather_than_a_fit() -> None:
    # I3: the curve is a published stylised fact reproduced as a shape, and
    # nothing in this platform was fitted to produce it (B1). The admission is on
    # the object, so it reaches every summary rather than living in a document.
    forecast = stylised_u_curve_forecast(bucket_count=7, as_of=TRADING_DATE)
    assert forecast.basis is ForecastBasis.STYLISED_U_CURVE
    assert forecast.fitted is False
    assert forecast.observed_through is None
    assert "NOT FITTED" in forecast.basis_detail
    assert "B1" in forecast.basis_detail
    assert forecast.summary()["fitted"] is False
    # The weights are a denominator, never a share count: the total is whatever
    # the shape happens to sum to, and only ratios of it are ever used.
    assert forecast.total_weight_units == sum(forecast.weight_units) > 0
    assert forecast.bucket_count == 7


def test_only_the_observed_profile_basis_claims_to_be_fitted() -> None:
    assert ForecastBasis.OBSERVED_PROFILE.fitted is True
    assert ForecastBasis.UNIFORM.fitted is False
    assert ForecastBasis.STYLISED_U_CURVE.fitted is False


def test_a_forecast_with_no_stated_provenance_is_refused() -> None:
    with pytest.raises(VolumeForecastError, match="basis_detail must state"):
        VolumeForecast(
            weight_units=(1, 2, 1),
            basis=ForecastBasis.STYLISED_U_CURVE,
            basis_detail="   ",
            as_of=TRADING_DATE,
            observed_through=None,
        )


def test_a_basis_cannot_disagree_with_whether_it_observed_anything() -> None:
    with pytest.raises(VolumeForecastError, match="observed_through must be None"):
        VolumeForecast(
            weight_units=(1, 1),
            basis=ForecastBasis.UNIFORM,
            basis_detail="uniform",
            as_of=TRADING_DATE,
            observed_through=TRADING_DATE - dt.timedelta(days=1),
        )
    with pytest.raises(VolumeForecastError, match="observed_through must be present"):
        VolumeForecast(
            weight_units=(1, 1),
            basis=ForecastBasis.OBSERVED_PROFILE,
            basis_detail="claims a fit it cannot support",
            as_of=TRADING_DATE,
            observed_through=None,
        )


def test_a_bucket_expected_to_trade_nothing_is_refused_rather_than_allocated_zero() -> None:
    with pytest.raises(VolumeForecastError, match="strictly positive"):
        VolumeForecast(
            weight_units=(5, 0, 5),
            basis=ForecastBasis.UNIFORM,
            basis_detail="one dead bucket",
            as_of=TRADING_DATE,
            observed_through=None,
        )


# ---------------------------------------------------------------------------
# I1: the forecast cannot see the day it trades.
# ---------------------------------------------------------------------------


def test_a_forecast_may_not_be_built_from_the_session_it_forecasts() -> None:
    # The central trap of this task: a VWAP schedule weighted by the volume that
    # actually traded that day is not a forecast, it is the answer. Equality is
    # the violation, so it is tested at equality specifically.
    with pytest.raises(ForecastLookaheadError, match="on or after"):
        VolumeForecast(
            weight_units=(1, 2, 1),
            basis=ForecastBasis.OBSERVED_PROFILE,
            basis_detail="realised volume from the session being scheduled",
            as_of=TRADING_DATE,
            observed_through=TRADING_DATE,
        )


def test_a_forecast_may_not_be_built_from_a_future_session_either() -> None:
    with pytest.raises(ForecastLookaheadError):
        VolumeForecast(
            weight_units=(1, 2, 1),
            basis=ForecastBasis.OBSERVED_PROFILE,
            basis_detail="tomorrow's volume",
            as_of=TRADING_DATE,
            observed_through=TRADING_DATE + dt.timedelta(days=1),
        )


def test_the_strict_bound_admits_the_session_immediately_before() -> None:
    # The refusal must not be so wide that no forecast can be built: the previous
    # session is knowable and is admitted.
    forecast = VolumeForecast(
        weight_units=(1, 2, 1),
        basis=ForecastBasis.OBSERVED_PROFILE,
        basis_detail="yesterday",
        as_of=TRADING_DATE,
        observed_through=TRADING_DATE - dt.timedelta(days=1),
    )
    assert forecast.fitted is True
    assert forecast.observed_through == dt.date(2026, 8, 2)


def test_the_fitted_constructor_refuses_a_same_day_session() -> None:
    # The match is on the *constructor's* wording ("session <date> is on or
    # after"), not merely on the exception type. The value object would refuse
    # this too, from the derived maximum — so a type-only assertion would still
    # pass with this check deleted, and the redundancy would be untested rather
    # than defence in depth. It also names the offending session rather than the
    # maximum, which is what an operator needs.
    with pytest.raises(ForecastLookaheadError, match=r"^session 2026-08-03 is on or after"):
        observed_volume_forecast(
            [
                session(dt.date(2026, 7, 31), (10, 5, 10)),
                session(TRADING_DATE, (10, 5, 10)),
            ],
            as_of=TRADING_DATE,
        )


def test_the_fitted_constructor_derives_its_own_temporal_bound_from_what_it_consumed() -> None:
    # The load-bearing anti-lookahead property: `observed_through` is not a claim
    # the caller makes, it is computed from the sessions handed in, so a caller
    # cannot understate what the forecast saw.
    forecast = observed_volume_forecast(
        [
            session(dt.date(2026, 7, 29), (30, 10, 20)),
            session(dt.date(2026, 8, 1), (10, 10, 40)),
            session(dt.date(2026, 7, 30), (20, 10, 10)),
        ],
        as_of=TRADING_DATE,
    )
    assert forecast.observed_through == dt.date(2026, 8, 1)
    assert forecast.observed_through < TRADING_DATE
    assert forecast.weight_units == (60, 30, 70)
    assert forecast.basis is ForecastBasis.OBSERVED_PROFILE
    assert forecast.fitted is True
    assert "2026-07-29" in forecast.basis_detail
    assert "I1" in forecast.basis_detail


@settings(max_examples=400)
@given(
    offsets=st.lists(st.integers(min_value=-40, max_value=5), min_size=1, max_size=6),
)
def test_a_fitted_forecast_is_accepted_exactly_when_every_session_predates_it(
    offsets: list[int],
) -> None:
    days = [TRADING_DATE + dt.timedelta(days=offset) for offset in offsets]
    observations = [session(day, (7, 3, 5)) for day in dict.fromkeys(days)]
    if any(day >= TRADING_DATE for day in days):
        with pytest.raises(ForecastLookaheadError):
            observed_volume_forecast(observations, as_of=TRADING_DATE)
        return
    forecast = observed_volume_forecast(observations, as_of=TRADING_DATE)
    assert forecast.observed_through is not None
    assert forecast.observed_through < TRADING_DATE
    assert forecast.observed_through == max(days)


@settings(max_examples=200)
@given(quantity=quantities, count=bucket_counts, offset=st.integers(min_value=1, max_value=200))
def test_no_schedule_this_module_can_build_consumed_data_from_its_own_session(
    quantity: int, count: int, offset: int
) -> None:
    # The property restated over schedules rather than forecasts: whatever route
    # produced it, a schedule that exists has consumed nothing dated on or after
    # the day it trades.
    observations = [
        session(TRADING_DATE - dt.timedelta(days=offset + step), tuple(range(1, count + 1)))
        for step in range(3)
    ]
    forecast = observed_volume_forecast(observations, as_of=TRADING_DATE)
    # At least one share per bucket, or the schedule does not exist to check.
    size = max(quantity, count)
    schedule = plan_vwap(parent(size), forecast=forecast, minimum_slice_shares=1)
    observed = schedule.forecast.observed_through
    assert observed is not None
    assert observed < schedule.rebalance_date
    assert sum(schedule.quantities) == size


def test_planning_refuses_a_forecast_whose_observations_reach_past_the_trading_day() -> None:
    # A forecast built for a *later* session, handed to an earlier schedule. Its
    # own constructor cannot catch this — it only knows its own `as_of` — so the
    # planner checks against the day actually being traded, and checks the
    # lookahead before the date mismatch because one is an invariant violation
    # and the other is a filing error.
    late = observed_volume_forecast(
        [session(dt.date(2026, 8, 5), (4, 2, 4))],
        as_of=dt.date(2026, 8, 10),
    )
    with pytest.raises(ForecastLookaheadError, match="2026-08-05"):
        plan_vwap(parent(3_000), forecast=late, minimum_slice_shares=1)


def test_planning_refuses_a_forecast_that_reaches_exactly_the_trading_day() -> None:
    # The boundary, which is where this check earns its ``>=``. The forecast is
    # internally valid — 3 August is strictly before its own 10 August ``as_of``
    # — and it consumed the very session the schedule would trade. Same-day
    # realised volume arriving through a mis-dated forecast is the same I1
    # violation as arriving directly, and equality is the case that distinguishes
    # a real bound from one that only looks like it.
    same_day = observed_volume_forecast(
        [session(TRADING_DATE, (4, 2, 4))],
        as_of=TRADING_DATE + dt.timedelta(days=7),
    )
    assert same_day.observed_through == TRADING_DATE
    with pytest.raises(ForecastLookaheadError, match="on or after the session it would schedule"):
        plan_vwap(parent(3_000), forecast=same_day, minimum_slice_shares=1)


def test_a_schedule_refuses_the_same_lookahead_when_assembled_by_hand() -> None:
    late = observed_volume_forecast(
        [session(TRADING_DATE, (1, 1))],
        as_of=dt.date(2026, 8, 10),
    )
    children = tuple(
        make_intent(
            quantity_shares=500,
            rebalance_date=TRADING_DATE,
            slice_index=index,
            slice_count=2,
        )
        for index in range(2)
    )
    with pytest.raises(ForecastLookaheadError, match="on or after the session it would schedule"):
        SliceSchedule(
            algorithm=SliceAlgorithm.VWAP,
            parent_quantity_shares=1_000,
            minimum_slice_shares=100,
            forecast=late,
            slices=children,
        )


def test_a_forecast_for_another_session_is_refused() -> None:
    forecast = stylised_u_curve_forecast(bucket_count=4, as_of=dt.date(2026, 8, 4))
    with pytest.raises(SliceScheduleError, match="forecast is for 2026-08-04"):
        plan_vwap(parent(4_000), forecast=forecast)


def test_a_timestamp_is_refused_where_a_session_date_belongs() -> None:
    # datetime is a subclass of date, so this would pass an isinstance check
    # silently. An intraday timestamp is precision this platform does not have.
    moment = dt.datetime(2026, 8, 3, 9, 30, tzinfo=dt.UTC)
    with pytest.raises(VolumeForecastError, match="calendar date"):
        uniform_volume_forecast(bucket_count=3, as_of=moment)
    with pytest.raises(VolumeForecastError, match="calendar date"):
        SessionVolume(session_date=moment, bucket_volume_shares=(1, 2, 3))


def test_a_session_that_traded_nothing_is_not_evidence_about_a_session() -> None:
    with pytest.raises(VolumeForecastError, match="traded no shares"):
        session(dt.date(2026, 8, 1), (0, 0, 0))


def test_a_bucket_dead_across_every_observed_session_is_refused() -> None:
    with pytest.raises(VolumeForecastError, match="bucket 1 traded nothing"):
        observed_volume_forecast(
            [
                session(dt.date(2026, 8, 1), (5, 0, 5)),
                session(dt.date(2026, 7, 31), (7, 0, 3)),
            ],
            as_of=TRADING_DATE,
        )


def test_sessions_that_disagree_about_their_gridding_are_refused() -> None:
    with pytest.raises(VolumeForecastError, match="disagree about the bucket count"):
        observed_volume_forecast(
            [
                session(dt.date(2026, 8, 1), (5, 5, 5)),
                session(dt.date(2026, 7, 31), (7, 3)),
            ],
            as_of=TRADING_DATE,
        )


def test_a_forecast_fitted_to_nothing_is_refused() -> None:
    with pytest.raises(VolumeForecastError, match="at least one observed session"):
        observed_volume_forecast([], as_of=TRADING_DATE)


# ---------------------------------------------------------------------------
# Idempotency keys.
# ---------------------------------------------------------------------------


def test_two_runs_of_the_same_schedule_produce_the_same_keys_in_the_same_order() -> None:
    # The whole point of a content-derived key (D-033): re-running a rebalance is
    # absorbed by the unique constraint instead of doubling the book.
    forecast = stylised_u_curve_forecast(bucket_count=9, as_of=TRADING_DATE)
    first = plan_vwap(parent(12_345), forecast=forecast, minimum_slice_shares=100)
    second = plan_vwap(
        parent(12_345),
        forecast=stylised_u_curve_forecast(bucket_count=9, as_of=TRADING_DATE),
        minimum_slice_shares=100,
    )
    assert first.quantities == second.quantities
    assert first.idempotency_keys() == second.idempotency_keys()


@settings(max_examples=200)
@given(quantity=quantities, count=bucket_counts)
def test_every_child_of_one_parent_has_a_distinct_key(quantity: int, count: int) -> None:
    schedule = plan_twap(parent(quantity), bucket_count=count, minimum_slice_shares=1)
    keys = schedule.idempotency_keys()
    assert len(set(keys)) == len(keys) == schedule.slice_count
    assert keys == tuple(idempotency_key(child) for child in schedule.slices)


def test_changing_the_slice_count_changes_every_key() -> None:
    # slice_count is in the preimage, so a schedule re-planned at a different
    # width is a set of new orders rather than a partial duplicate of the old one.
    three = plan_twap(parent(9_000), bucket_count=3, minimum_slice_shares=100)
    four = plan_twap(parent(9_000), bucket_count=4, minimum_slice_shares=100)
    assert set(three.idempotency_keys()) & set(four.idempotency_keys()) == set()


def test_a_different_reproducibility_stamp_gives_different_keys() -> None:
    other = make_stamp(seed=99)
    base = plan_twap(parent(9_000), bucket_count=3, minimum_slice_shares=100)
    restamped = plan_twap(
        make_intent(quantity_shares=9_000, rebalance_date=TRADING_DATE, stamp=other),
        bucket_count=3,
        minimum_slice_shares=100,
    )
    assert base.quantities == restamped.quantities
    assert set(base.idempotency_keys()) & set(restamped.idempotency_keys()) == set()


# ---------------------------------------------------------------------------
# Cost: the model alone never stops slicing (I4).
# ---------------------------------------------------------------------------


def costed(count: int, quantity: int = DIVISIBLE_QUANTITY) -> ScheduleCost:
    """Cost a TWAP schedule of the given width through the platform's cost model."""
    return modelled_schedule_cost(
        plan_twap(parent(quantity), bucket_count=count, minimum_slice_shares=1),
        reference_price_usd=50.0,
        adv_usd=50_000_000.0,
        daily_volatility_bps=200.0,
    )


@pytest.mark.parametrize("count", list(range(2, 13)))
def test_total_modelled_impact_falls_as_one_over_the_square_root_of_the_slice_count(
    count: int,
) -> None:
    # 27,720 divides by every count here, so the slices are exactly equal and the
    # square-root law can be asserted exactly rather than approximately.
    unsliced = costed(1)
    sliced = costed(count)
    assert sliced.slice_count == count
    assert math.isclose(sliced.impact_usd * math.sqrt(count), unsliced.impact_usd, rel_tol=1e-9)


@pytest.mark.parametrize("count", list(range(2, 14)))
def test_spread_and_commission_dollars_do_not_move_with_the_slice_count(count: int) -> None:
    # The demonstration that the cost model has *no per-order term*: the same
    # shares are traded either way, both rates are flat on notional, and every
    # additional child is therefore free in this model. It is not free in
    # reality, which is why the slice-count bound is exogenous.
    assert math.isclose(
        costed(count).spread_and_commission_usd,
        costed(1).spread_and_commission_usd,
        rel_tol=1e-9,
    )
    assert math.isclose(costed(count).notional_usd, costed(1).notional_usd, rel_tol=1e-9)


def test_the_cost_model_alone_never_stops_slicing() -> None:
    # The central I4 point. Modelled cost decreases monotonically all the way to
    # the horizon ceiling, so a naive optimizer over the slice count does not
    # converge — it runs out of shares. That is a property of a model that
    # charges nothing for slicing, not a finding about markets, and it is why
    # MAX_SLICE_COUNT and ROUND_LOT_SHARES are stated constants rather than the
    # output of a search.
    totals = [costed(count).total_usd for count in range(1, MAX_SLICE_COUNT + 1)]
    assert all(later < earlier for earlier, later in itertools.pairwise(totals))
    # And the residual it is heading towards is the flat part, which slicing
    # never touches.
    assert totals[-1] > costed(1).spread_and_commission_usd


def test_a_cost_estimate_carries_the_calibration_and_forecast_provenance_forward() -> None:
    # I4: the number comes from the platform's cost model, and its "these are
    # assumptions" flags travel with it — as does whether the volume shape was
    # fitted, which for every forecast this platform can build today is False.
    schedule = plan_vwap(
        parent(20_000),
        forecast=stylised_u_curve_forecast(bucket_count=5, as_of=TRADING_DATE),
        minimum_slice_shares=100,
    )
    cost = modelled_schedule_cost(schedule, reference_price_usd=40.0, adv_usd=20_000_000.0)
    assert cost.uncalibrated is True
    assert cost.calibration_basis == UNCALIBRATED_DEFAULTS.calibration_basis
    assert cost.forecast_is_fitted is False
    assert "NOT FITTED" in cost.forecast_basis_detail
    summary = cost.summary()
    assert summary["units"] == "US dollars"
    assert "no per-order fixed cost" in str(summary["omits"])
    assert "no timing or volatility risk" in str(summary["omits"])
    assert math.isclose(
        cost.total_usd, cost.impact_usd + cost.spread_and_commission_usd, rel_tol=1e-9
    )


def test_a_schedule_cannot_be_costed_at_a_zero_price() -> None:
    schedule = plan_twap(parent(1_000), bucket_count=2, minimum_slice_shares=100)
    with pytest.raises(SliceScheduleError, match="strictly positive"):
        modelled_schedule_cost(schedule, reference_price_usd=0.0, adv_usd=1_000_000.0)


# ---------------------------------------------------------------------------
# Schedule integrity.
# ---------------------------------------------------------------------------


def test_slicing_an_order_that_is_already_a_slice_is_refused() -> None:
    child = make_intent(
        quantity_shares=5_000, rebalance_date=TRADING_DATE, slice_index=1, slice_count=4
    )
    with pytest.raises(SliceScheduleError, match="already slice 1 of 4"):
        plan_twap(child, bucket_count=3)
    with pytest.raises(SliceScheduleError, match="already slice 1 of 4"):
        plan_vwap(child, forecast=uniform_volume_forecast(bucket_count=3, as_of=TRADING_DATE))


def test_a_schedule_whose_children_are_different_orders_is_refused() -> None:
    children = (
        make_intent(quantity_shares=500, rebalance_date=TRADING_DATE, slice_index=0, slice_count=2),
        make_intent(
            quantity_shares=500,
            rebalance_date=TRADING_DATE,
            slice_index=1,
            slice_count=2,
            security_id=99,
        ),
    )
    with pytest.raises(SliceScheduleError, match="security_id"):
        SliceSchedule(
            algorithm=SliceAlgorithm.TWAP,
            parent_quantity_shares=1_000,
            minimum_slice_shares=100,
            forecast=uniform_volume_forecast(bucket_count=2, as_of=TRADING_DATE),
            slices=children,
        )


def test_a_schedule_with_wrong_slice_coordinates_is_refused() -> None:
    children = tuple(
        make_intent(quantity_shares=500, rebalance_date=TRADING_DATE, slice_index=0, slice_count=2)
        for _ in range(2)
    )
    with pytest.raises(SliceScheduleError, match="declares slice_index=0"):
        SliceSchedule(
            algorithm=SliceAlgorithm.TWAP,
            parent_quantity_shares=1_000,
            minimum_slice_shares=100,
            forecast=uniform_volume_forecast(bucket_count=2, as_of=TRADING_DATE),
            slices=children,
        )


def test_a_schedule_whose_forecast_covers_other_buckets_is_refused() -> None:
    children = tuple(
        make_intent(
            quantity_shares=500, rebalance_date=TRADING_DATE, slice_index=index, slice_count=2
        )
        for index in range(2)
    )
    with pytest.raises(SliceScheduleError, match="covers 3 buckets"):
        SliceSchedule(
            algorithm=SliceAlgorithm.TWAP,
            parent_quantity_shares=1_000,
            minimum_slice_shares=100,
            forecast=uniform_volume_forecast(bucket_count=3, as_of=TRADING_DATE),
            slices=children,
        )


def test_a_zero_share_slice_has_no_representation_at_all() -> None:
    # Two layers: the apportionment never produces one, and OrderIntent refuses
    # one outright, so a schedule containing one cannot be assembled either.
    with pytest.raises(OrderValidationError, match="strictly positive"):
        make_intent(quantity_shares=0, rebalance_date=TRADING_DATE)


def test_a_schedule_summary_states_its_units_and_its_forecast_provenance() -> None:
    schedule = plan_vwap(
        parent(10_000),
        forecast=stylised_u_curve_forecast(bucket_count=5, as_of=TRADING_DATE),
        minimum_slice_shares=100,
    )
    summary = schedule.summary()
    assert summary["algorithm"] == "vwap"
    assert summary["allocated_shares"] == summary["parent_quantity_shares"] == 10_000
    assert summary["rebalance_date"] == "2026-08-03"
    forecast_summary = summary["forecast"]
    assert isinstance(forecast_summary, dict)
    assert forecast_summary["fitted"] is False
    assert "dimensionless" in str(forecast_summary["units"])


# ---------------------------------------------------------------------------
# Paper-only, restated for this module.
# ---------------------------------------------------------------------------


def test_this_module_is_inside_the_paper_only_scan() -> None:
    # test_paper_only.py parametrises its token, import, URL, port and AST scans
    # over every module in the package; this asserts that the module-set literal
    # guarding that glob names this one, so the coverage cannot silently lapse.
    assert "slicing.py" in {path.name for path in MODULE_PATHS}


def test_no_public_callable_here_admits_a_venue_client_or_endpoint() -> None:
    # The runtime-signature half of the claim, in the same shape as
    # test_control_paper_only.py: the AST scan covers `def` statements, this
    # covers the constructed objects, where a decorator or re-export would hide.
    for name in slicing.__all__:
        candidate = getattr(slicing, name)
        if not callable(candidate) or isinstance(candidate, type):
            continue
        parameters = set(inspect.signature(candidate).parameters)
        assert parameters & FORBIDDEN_PARAMETERS == set(), (name, parameters)


def test_no_method_on_an_exported_type_here_admits_a_venue() -> None:
    for name in slicing.__all__:
        candidate = getattr(slicing, name)
        if not isinstance(candidate, type):
            continue
        for attribute, member in vars(candidate).items():
            if not callable(member):
                continue
            try:
                parameters = set(inspect.signature(member).parameters)
            except (TypeError, ValueError):
                continue
            assert parameters & FORBIDDEN_PARAMETERS == set(), (name, attribute)


def test_this_module_holds_no_transport_of_its_own() -> None:
    source = Path(str(slicing.__file__)).read_text(encoding="utf-8")
    assert "://" not in source
    for forbidden in ("import socket", "import httpx", "import requests", "ib_insync"):
        assert forbidden not in source, forbidden
    # It also reads no clock: a schedule is content, and a timestamp in it would
    # be both a precision this platform does not have and an input the
    # idempotency key would have to exclude.
    assert re.search(r"\bnow\s*\(", source) is None
    assert re.search(r"\butcnow\b", source) is None
