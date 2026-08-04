"""Triple-barrier engine on constructed paths whose answer is obvious by inspection.

Every case below states the arithmetic in the test itself. With the warm-up in
:mod:`backend.tests.labels.paths`, trailing volatility at the event bar is
``0.01 * sqrt(2) = 0.014142…`` and a 5-bar horizon with unit multiples puts the
barriers at ``+/- 0.01 * sqrt(2) * sqrt(5) = +/- 0.031623…`` cumulative log
return. Forward returns of ``+0.02`` per bar therefore cross the upper barrier
on the **second** forward bar (0.02 < 0.0316 <= 0.04), not the first — chosen
deliberately so that an off-by-one in the scan window fails the test.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from backend.labels.barriers import (
    BarrierSpec,
    IntrabarPolicy,
    LabelBasis,
    LabelSet,
    TripleBarrierOutcome,
    barrier_log_return_width,
    resolve_barrier_touch,
    triple_barrier_labels,
    usable_event_indices,
)
from backend.labels.errors import (
    AmbiguousLabelError,
    DegenerateVolatilityError,
    InsufficientHistoryError,
    LabelConfigurationError,
    LabelInputError,
)
from backend.tests.labels.paths import (
    FIRST_EVENT_BAR,
    WARMUP_VOLATILITY,
    WARMUP_WINDOW,
    barrier_width,
    hand_derived_path,
    prices_from_log_returns,
)

if TYPE_CHECKING:
    from backend.labels._arrays import FloatArray

HORIZON = 5
SPEC = BarrierSpec(horizon=HORIZON, volatility_window=WARMUP_WINDOW)
EXPECTED_BARRIER = barrier_width(WARMUP_VOLATILITY, HORIZON)

# A forward path carrying two labellable events. No two consecutive returns are
# equal, so the 2-bar trailing window is never flat and every event has a
# positive volatility to be sized by.
TWO_EVENT_FORWARD = (0.02, 0.021, 0.0, 0.001, 0.0, -0.02, -0.021, 0.0, 0.001, 0.0)

# Small alternating moves: large enough that no trailing window is flat, far too
# small to reach a barrier on their own.
QUIET_FORWARD = (0.001, -0.001) * 5


class TestBarrierWidth:
    def test_width_is_the_documented_formula(self) -> None:
        assert barrier_log_return_width(0.02, horizon=9, multiple=1.5) == pytest.approx(
            1.5 * 0.02 * 3.0
        )

    def test_doubling_volatility_doubles_the_barrier_distance(self) -> None:
        single = barrier_log_return_width(0.013, horizon=21, multiple=1.0)
        double = barrier_log_return_width(0.026, horizon=21, multiple=1.0)
        assert double == pytest.approx(2.0 * single)

    @pytest.mark.parametrize("factor", [0.5, 2.0, 7.5])
    def test_barrier_distance_is_linear_in_volatility(self, factor: float) -> None:
        base = barrier_log_return_width(0.011, horizon=63, multiple=1.3)
        scaled = barrier_log_return_width(0.011 * factor, horizon=63, multiple=1.3)
        assert scaled == pytest.approx(factor * base)

    def test_the_engine_uses_that_same_width(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([0.0] * HORIZON), [FIRST_EVENT_BAR], SPEC)
        assert labels.trailing_volatility[0] == pytest.approx(WARMUP_VOLATILITY)
        assert labels.upper_barrier_log_return[0] == pytest.approx(EXPECTED_BARRIER)
        assert labels.lower_barrier_log_return[0] == pytest.approx(-EXPECTED_BARRIER)

    def test_doubling_the_path_volatility_doubles_the_engine_barrier(self) -> None:
        quiet = triple_barrier_labels(hand_derived_path([0.0] * HORIZON), [FIRST_EVENT_BAR], SPEC)
        loud = triple_barrier_labels(
            prices_from_log_returns([0.02, -0.02, *([0.0] * HORIZON)]),
            [FIRST_EVENT_BAR],
            SPEC,
        )
        assert loud.trailing_volatility[0] == pytest.approx(2.0 * quiet.trailing_volatility[0])
        assert loud.upper_barrier_log_return[0] == pytest.approx(
            2.0 * quiet.upper_barrier_log_return[0]
        )

    def test_asymmetric_multiples_place_asymmetric_barriers(self) -> None:
        spec = BarrierSpec(
            horizon=HORIZON, upper_multiple=2.0, lower_multiple=0.5, volatility_window=WARMUP_WINDOW
        )
        labels = triple_barrier_labels(hand_derived_path([0.0] * HORIZON), [FIRST_EVENT_BAR], spec)
        assert labels.upper_barrier_log_return[0] == pytest.approx(2.0 * EXPECTED_BARRIER)
        assert labels.lower_barrier_log_return[0] == pytest.approx(-0.5 * EXPECTED_BARRIER)


class TestObviousPaths:
    def test_monotone_up_touches_the_upper_barrier(self) -> None:
        # +0.02 per bar: cumulative 0.02 at bar 3, 0.04 at bar 4. The barrier is
        # 0.0316, so the second forward bar resolves it.
        labels = triple_barrier_labels(hand_derived_path([0.02] * 8), [FIRST_EVENT_BAR], SPEC)
        assert labels.outcome[0] == TripleBarrierOutcome.UPPER_FIRST
        assert labels.resolution_index[0] == FIRST_EVENT_BAR + 2
        assert labels.realized_log_return[0] == pytest.approx(0.04)
        assert labels.basis is LabelBasis.PRICE

    def test_monotone_down_touches_the_lower_barrier(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([-0.02] * 8), [FIRST_EVENT_BAR], SPEC)
        assert labels.outcome[0] == TripleBarrierOutcome.LOWER_FIRST
        assert labels.resolution_index[0] == FIRST_EVENT_BAR + 2
        assert labels.realized_log_return[0] == pytest.approx(-0.04)

    def test_a_flat_path_runs_to_the_vertical_barrier(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([0.0] * 8), [FIRST_EVENT_BAR], SPEC)
        assert labels.outcome[0] == TripleBarrierOutcome.VERTICAL
        assert labels.resolution_index[0] == FIRST_EVENT_BAR + HORIZON
        assert labels.realized_log_return[0] == pytest.approx(0.0)

    def test_drift_too_small_to_reach_a_barrier_is_vertical_not_signed(self) -> None:
        # +0.004 per bar reaches +0.02 by the deadline: clearly positive, and
        # clearly short of the 0.0316 barrier. The documented choice is that
        # this is the VERTICAL class, not sign(+0.02) == UPPER.
        labels = triple_barrier_labels(hand_derived_path([0.004] * 8), [FIRST_EVENT_BAR], SPEC)
        assert labels.outcome[0] == TripleBarrierOutcome.VERTICAL
        assert labels.realized_log_return[0] == pytest.approx(0.02)
        assert labels.realized_log_return[0] > 0.0

    def test_upper_then_lower_resolves_at_the_upper_barrier(self) -> None:
        # Up 0.02, up 0.02 (upper touched at bar 4), then a collapse that would
        # have cleared the lower barrier had the scan continued.
        labels = triple_barrier_labels(
            hand_derived_path([0.02, 0.02, -0.10, -0.10, 0.0]), [FIRST_EVENT_BAR], SPEC
        )
        assert labels.outcome[0] == TripleBarrierOutcome.UPPER_FIRST
        assert labels.resolution_index[0] == FIRST_EVENT_BAR + 2

    def test_lower_then_upper_resolves_at_the_lower_barrier(self) -> None:
        labels = triple_barrier_labels(
            hand_derived_path([-0.02, -0.02, 0.10, 0.10, 0.0]), [FIRST_EVENT_BAR], SPEC
        )
        assert labels.outcome[0] == TripleBarrierOutcome.LOWER_FIRST
        assert labels.resolution_index[0] == FIRST_EVENT_BAR + 2

    def test_touching_the_barrier_exactly_counts_as_touching_it(self) -> None:
        labels = triple_barrier_labels(
            hand_derived_path([EXPECTED_BARRIER, 0.0, 0.0, 0.0, 0.0]), [FIRST_EVENT_BAR], SPEC
        )
        assert labels.outcome[0] == TripleBarrierOutcome.UPPER_FIRST
        assert labels.resolution_index[0] == FIRST_EVENT_BAR + 1

    def test_a_hair_below_the_barrier_does_not_count(self) -> None:
        labels = triple_barrier_labels(
            hand_derived_path([EXPECTED_BARRIER * (1 - 1e-9), 0.0, 0.0, 0.0, 0.0]),
            [FIRST_EVENT_BAR],
            SPEC,
        )
        assert labels.outcome[0] == TripleBarrierOutcome.VERTICAL

    def test_the_event_bar_itself_is_not_scanned(self) -> None:
        # A zero-width barrier is impossible by construction, so the only way
        # bar t0 could resolve a label is a scan that starts one bar early. A
        # violent move *into* the event bar must not label it.
        prices = prices_from_log_returns([0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
        labels = triple_barrier_labels(prices, [FIRST_EVENT_BAR], SPEC)
        assert labels.resolution_index[0] > FIRST_EVENT_BAR

    def test_several_events_are_labelled_independently(self) -> None:
        prices = hand_derived_path(TWO_EVENT_FORWARD)
        events = [2, 7]
        together = triple_barrier_labels(prices, events, SPEC)
        separate = [triple_barrier_labels(prices, [event], SPEC) for event in events]
        assert together.outcome.tolist() == [int(one.outcome[0]) for one in separate]
        assert together.resolution_index.tolist() == [
            int(one.resolution_index[0]) for one in separate
        ]

    def test_event_order_is_preserved_not_sorted(self) -> None:
        prices = hand_derived_path(TWO_EVENT_FORWARD)
        labels = triple_barrier_labels(prices, [7, 2], SPEC)
        assert labels.event_index.tolist() == [7, 2]


class _SpikeSeries(NamedTuple):
    """A flat close path with a barrier-piercing wick on one bar."""

    close: FloatArray
    high: FloatArray
    low: FloatArray
    spike_bar: int


def _spike_series(*, up_spike: bool, down_spike: bool) -> _SpikeSeries:
    """Build a path whose closes never move but whose wicks pierce the barriers."""
    closes = hand_derived_path([0.0] * 8)
    highs = closes.copy()
    lows = closes.copy()
    anchor = float(closes[FIRST_EVENT_BAR])
    spike_bar = FIRST_EVENT_BAR + 3
    if up_spike:
        highs[spike_bar] = anchor * math.exp(EXPECTED_BARRIER + 0.001)
    if down_spike:
        lows[spike_bar] = anchor * math.exp(-EXPECTED_BARRIER - 0.001)
    return _SpikeSeries(close=closes, high=highs, low=lows, spike_bar=spike_bar)


class TestIntrabarExtremes:
    """A barrier pierced inside a bar and given back by the close."""

    def test_an_intrabar_high_touches_the_upper_barrier(self) -> None:
        series = _spike_series(up_spike=True, down_spike=False)
        labels = triple_barrier_labels(
            series.close, [FIRST_EVENT_BAR], SPEC, high=series.high, low=series.low
        )
        assert labels.outcome[0] == TripleBarrierOutcome.UPPER_FIRST
        assert labels.resolution_index[0] == series.spike_bar
        # The close gave the move back: the realized close-to-close return is
        # zero even though the barrier was touched. Documented, and asserted so
        # nobody "fixes" realized_log_return into the barrier level.
        assert labels.realized_log_return[0] == pytest.approx(0.0)

    def test_without_high_and_low_the_same_spike_is_invisible(self) -> None:
        series = _spike_series(up_spike=True, down_spike=False)
        labels = triple_barrier_labels(series.close, [FIRST_EVENT_BAR], SPEC)
        assert labels.outcome[0] == TripleBarrierOutcome.VERTICAL

    def test_both_barriers_in_one_bar_is_flagged_by_default(self) -> None:
        series = _spike_series(up_spike=True, down_spike=True)
        labels = triple_barrier_labels(
            series.close, [FIRST_EVENT_BAR], SPEC, high=series.high, low=series.low
        )
        assert labels.outcome[0] == TripleBarrierOutcome.AMBIGUOUS
        assert labels.resolution_index[0] == series.spike_bar
        assert labels.n_ambiguous == 1

    def test_the_lower_first_policy_resolves_that_same_bar_downwards(self) -> None:
        series = _spike_series(up_spike=True, down_spike=True)
        spec = BarrierSpec(
            horizon=HORIZON,
            volatility_window=WARMUP_WINDOW,
            intrabar_policy=IntrabarPolicy.LOWER_FIRST,
        )
        labels = triple_barrier_labels(
            series.close, [FIRST_EVENT_BAR], spec, high=series.high, low=series.low
        )
        assert labels.outcome[0] == TripleBarrierOutcome.LOWER_FIRST
        assert labels.resolution_index[0] == series.spike_bar
        assert labels.n_ambiguous == 0

    def test_an_earlier_unambiguous_touch_beats_a_later_ambiguous_bar(self) -> None:
        series = _spike_series(up_spike=True, down_spike=True)
        lows = series.low.copy()
        # Clean lower touch one bar before the doubly-pierced bar.
        earlier = FIRST_EVENT_BAR + 2
        lows[earlier] = float(series.close[FIRST_EVENT_BAR]) * math.exp(-EXPECTED_BARRIER - 0.001)
        labels = triple_barrier_labels(
            series.close, [FIRST_EVENT_BAR], SPEC, high=series.high, low=lows
        )
        assert labels.outcome[0] == TripleBarrierOutcome.LOWER_FIRST
        assert labels.resolution_index[0] == earlier

    def test_supplying_only_one_of_high_and_low_is_refused(self) -> None:
        closes = hand_derived_path([0.0] * 8)
        with pytest.raises(LabelInputError, match="must be supplied together"):
            triple_barrier_labels(closes, [FIRST_EVENT_BAR], SPEC, high=closes)

    def test_high_below_low_is_refused(self) -> None:
        closes = hand_derived_path([0.0] * 8)
        with pytest.raises(LabelInputError, match="high must be >= low"):
            triple_barrier_labels(closes, [FIRST_EVENT_BAR], SPEC, high=closes * 0.9, low=closes)

    def test_a_close_outside_its_own_bar_range_is_refused(self) -> None:
        closes = hand_derived_path([0.0] * 8)
        highs = closes.copy()
        highs[4] = closes[4] * 0.5
        lows = closes * 0.5
        lows[4] = closes[4] * 0.25
        with pytest.raises(LabelInputError, match=r"close must lie within"):
            triple_barrier_labels(closes, [FIRST_EVENT_BAR], SPEC, high=highs, low=lows)


class TestResolveBarrierTouch:
    """The resolution rule on its own, without any price series around it."""

    def test_no_touch_returns_the_deadline(self) -> None:
        path = np.array([0.0, 0.01, -0.01, 0.0])
        outcome, offset = resolve_barrier_touch(
            path,
            path,
            upper_log_return=0.05,
            lower_log_return=-0.05,
            policy=IntrabarPolicy.FLAG_AMBIGUOUS,
        )
        assert outcome is TripleBarrierOutcome.VERTICAL
        assert offset == path.size - 1

    def test_the_earliest_touch_wins_even_when_a_later_one_is_larger(self) -> None:
        highs = np.array([0.06, 0.5, 0.5])
        lows = np.array([0.0, -0.9, -0.9])
        outcome, offset = resolve_barrier_touch(
            highs,
            lows,
            upper_log_return=0.05,
            lower_log_return=-0.05,
            policy=IntrabarPolicy.FLAG_AMBIGUOUS,
        )
        assert outcome is TripleBarrierOutcome.UPPER_FIRST
        assert offset == 0

    def test_a_single_bar_piercing_both_is_ambiguous(self) -> None:
        outcome, offset = resolve_barrier_touch(
            np.array([0.0, 0.2]),
            np.array([0.0, -0.2]),
            upper_log_return=0.05,
            lower_log_return=-0.05,
            policy=IntrabarPolicy.FLAG_AMBIGUOUS,
        )
        assert outcome is TripleBarrierOutcome.AMBIGUOUS
        assert offset == 1

    def test_asymmetric_barriers_are_honoured(self) -> None:
        path = np.array([-0.03, 0.0, 0.0])
        outcome, offset = resolve_barrier_touch(
            path,
            path,
            upper_log_return=0.10,
            lower_log_return=-0.02,
            policy=IntrabarPolicy.FLAG_AMBIGUOUS,
        )
        assert outcome is TripleBarrierOutcome.LOWER_FIRST
        assert offset == 0


class TestLabelSetSurface:
    def _mixed_labels(self) -> LabelSet:
        """One ambiguous label and one ordinary one, in a single set."""
        closes = hand_derived_path(QUIET_FORWARD)
        highs = closes.copy()
        lows = closes.copy()
        anchor = closes[FIRST_EVENT_BAR]
        highs[FIRST_EVENT_BAR + 1] = anchor * math.exp(EXPECTED_BARRIER + 0.001)
        lows[FIRST_EVENT_BAR + 1] = anchor * math.exp(-EXPECTED_BARRIER - 0.001)
        return triple_barrier_labels(
            closes, [FIRST_EVENT_BAR, FIRST_EVENT_BAR + 3], SPEC, high=highs, low=lows
        )

    def test_class_labels_refuse_to_invent_a_value_for_ambiguous_rows(self) -> None:
        labels = self._mixed_labels()
        with pytest.raises(AmbiguousLabelError, match="AMBIGUOUS"):
            labels.class_labels()

    def test_dropping_ambiguous_rows_leaves_a_usable_set(self) -> None:
        labels = self._mixed_labels().drop_ambiguous()
        assert labels.n_labels == 1
        assert labels.n_ambiguous == 0
        assert set(labels.class_labels().tolist()) <= {-1, 0, 1}

    def test_outcome_counts_report_every_class_including_empty_ones(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([0.0] * 8), [FIRST_EVENT_BAR], SPEC)
        counts = labels.outcome_counts()
        assert set(counts) == set(TripleBarrierOutcome)
        assert counts[TripleBarrierOutcome.VERTICAL] == 1
        assert counts[TripleBarrierOutcome.UPPER_FIRST] == 0
        assert sum(counts.values()) == labels.n_labels

    def test_the_information_span_starts_one_bar_after_the_event(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([0.0] * 8), [FIRST_EVENT_BAR], SPEC)
        assert labels.first_information_bar.tolist() == [FIRST_EVENT_BAR + 1]

    def test_the_spec_travels_with_the_labels(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([0.0] * 8), [FIRST_EVENT_BAR], SPEC)
        assert labels.spec == SPEC


class TestRefusals:
    def test_an_event_without_enough_forward_bars_is_refused(self) -> None:
        prices = hand_derived_path([0.0] * 3)  # 6 bars: 0..5
        with pytest.raises(InsufficientHistoryError, match="past the end of the series"):
            triple_barrier_labels(prices, [4], SPEC)

    def test_an_event_without_enough_trailing_bars_is_refused(self) -> None:
        prices = hand_derived_path([0.0] * 8)
        with pytest.raises(InsufficientHistoryError, match="bar\\(s\\) of history precede it"):
            triple_barrier_labels(prices, [1], SPEC)

    def test_a_flat_trailing_window_is_refused_rather_than_labelled(self) -> None:
        prices = prices_from_log_returns([0.0, 0.0, 0.02, 0.02, 0.02, 0.02, 0.02])
        with pytest.raises(DegenerateVolatilityError, match="trailing volatility"):
            triple_barrier_labels(prices, [FIRST_EVENT_BAR], SPEC)

    def test_an_event_index_past_the_end_is_refused(self) -> None:
        prices = hand_derived_path([0.0] * 8)
        with pytest.raises(LabelInputError, match="but the series has"):
            triple_barrier_labels(prices, [999], SPEC)

    def test_a_fractional_event_index_is_refused_rather_than_truncated(self) -> None:
        prices = hand_derived_path([0.0] * 8)
        with pytest.raises(LabelInputError, match="whole-numbered"):
            triple_barrier_labels(prices, [2.5], SPEC)

    def test_a_volatility_series_of_the_wrong_length_is_refused(self) -> None:
        prices = hand_derived_path([0.0] * 8)
        with pytest.raises(LabelInputError, match="same length"):
            triple_barrier_labels(prices, [FIRST_EVENT_BAR], SPEC, volatility=np.full(3, 0.02))

    def test_no_events_gives_an_empty_label_set_not_an_error(self) -> None:
        labels = triple_barrier_labels(hand_derived_path([0.0] * 8), [], SPEC)
        assert labels.n_labels == 0
        assert labels.outcome_counts() == dict.fromkeys(TripleBarrierOutcome, 0)


class TestBarrierSpecValidation:
    @pytest.mark.parametrize("horizon", [-1, 0])
    def test_a_horizon_below_one_is_refused(self, horizon: int) -> None:
        with pytest.raises(LabelConfigurationError, match="horizon must be >= 1"):
            BarrierSpec(horizon=horizon)

    @pytest.mark.parametrize("multiple", [0.0, -1.0, math.nan, math.inf])
    def test_a_non_positive_multiple_is_refused(self, multiple: float) -> None:
        with pytest.raises(LabelConfigurationError, match="upper_multiple"):
            BarrierSpec(horizon=5, upper_multiple=multiple)
        with pytest.raises(LabelConfigurationError, match="lower_multiple"):
            BarrierSpec(horizon=5, lower_multiple=multiple)

    def test_a_volatility_window_below_two_is_refused(self) -> None:
        with pytest.raises(LabelConfigurationError, match="volatility_window must be >= 2"):
            BarrierSpec(horizon=5, volatility_window=1)

    @pytest.mark.parametrize("horizon", [5, 21, 63])
    def test_the_project_horizons_are_all_constructible(self, horizon: int) -> None:
        assert BarrierSpec(horizon=horizon).horizon == horizon


class TestUsableEventIndices:
    def test_it_selects_exactly_the_labellable_bars(self) -> None:
        spec = BarrierSpec(horizon=3, volatility_window=4)
        usable = usable_event_indices(12, spec)
        assert usable.tolist() == [4, 5, 6, 7, 8]

    def test_every_selected_bar_actually_labels(self) -> None:
        spec = BarrierSpec(horizon=3, volatility_window=WARMUP_WINDOW)
        prices = hand_derived_path([0.01, -0.015, 0.02, -0.005, 0.012, -0.02, 0.008, 0.0])
        usable = usable_event_indices(prices.size, spec)
        assert usable.size > 0
        labels = triple_barrier_labels(prices, usable, spec)
        assert labels.n_labels == usable.size

    def test_the_bar_just_outside_the_selection_is_refused(self) -> None:
        spec = BarrierSpec(horizon=3, volatility_window=WARMUP_WINDOW)
        prices = hand_derived_path([0.01, -0.015, 0.02, -0.005, 0.012, -0.02, 0.008, 0.0])
        usable = usable_event_indices(prices.size, spec)
        with pytest.raises(InsufficientHistoryError):
            triple_barrier_labels(prices, [int(usable[-1]) + 1], spec)

    def test_a_series_too_short_to_label_gives_an_empty_selection(self) -> None:
        assert usable_event_indices(5, BarrierSpec(horizon=21)).size == 0

    def test_the_trailing_requirement_can_be_overridden(self) -> None:
        spec = BarrierSpec(horizon=2, volatility_window=2)
        assert usable_event_indices(10, spec, min_trailing_bars=6).tolist() == [6, 7]
