"""Sample-uniqueness weights and effective sample size, on spans done by hand.

Concurrency is small enough here to count on fingers, so every expected weight
below is arithmetic the reader can check: with two labels covering a bar, each
gets ``1/2`` of it; a label's weight is the mean of its bars' shares.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.labels.barriers import BarrierSpec, triple_barrier_labels
from backend.labels.errors import LabelInputError
from backend.labels.uniqueness import (
    MAX_CONCURRENCY_SPAN_BARS,
    effective_sample_size,
    sample_uniqueness,
    sample_uniqueness_from_labels,
)
from backend.tests.labels.paths import (
    FIRST_EVENT_BAR,
    WARMUP_WINDOW,
    hand_derived_path,
)


class TestHandCountedSpans:
    def test_non_overlapping_labels_all_weigh_exactly_one(self) -> None:
        result = sample_uniqueness([0, 5, 10], [4, 9, 14])
        assert result.average_uniqueness.tolist() == [1.0, 1.0, 1.0]
        assert result.effective_sample_size == 3.0
        assert result.uniqueness_ratio == 1.0
        assert result.max_concurrency == 1

    def test_adjacent_spans_do_not_overlap(self) -> None:
        # [0,4] and [5,9] touch but share no bar. An off-by-one in the
        # difference array would show up here as a weight below 1.
        result = sample_uniqueness([0, 5], [4, 9])
        assert result.average_uniqueness.tolist() == [1.0, 1.0]

    @pytest.mark.parametrize("n_labels", [2, 3, 7])
    def test_fully_overlapping_labels_each_weigh_one_over_n(self, n_labels: int) -> None:
        result = sample_uniqueness([3] * n_labels, [11] * n_labels)
        assert result.average_uniqueness == pytest.approx([1.0 / n_labels] * n_labels)
        assert result.effective_sample_size == pytest.approx(1.0)
        assert result.max_concurrency == n_labels

    def test_a_half_overlap_splits_the_shared_bars(self) -> None:
        # A covers 0..3, B covers 2..5. Bars 2 and 3 are shared.
        # u_A = (1 + 1 + 1/2 + 1/2) / 4 = 0.75, and B is symmetric.
        result = sample_uniqueness([0, 2], [3, 5])
        assert result.average_uniqueness == pytest.approx([0.75, 0.75])
        assert result.effective_sample_size == pytest.approx(1.5)
        assert result.concurrency.tolist() == [1, 1, 2, 2, 1, 1]

    def test_a_long_label_nesting_two_short_ones(self) -> None:
        # A covers 0..9; B covers 0..4; C covers 5..9. Every bar is claimed
        # twice, so all three weigh 1/2 and the sample is worth 1.5.
        result = sample_uniqueness([0, 0, 5], [9, 4, 9])
        assert result.average_uniqueness == pytest.approx([0.5, 0.5, 0.5])
        assert result.effective_sample_size == pytest.approx(1.5)

    def test_gaps_between_labels_are_allowed(self) -> None:
        result = sample_uniqueness([0, 10], [2, 12])
        assert result.average_uniqueness.tolist() == [1.0, 1.0]
        assert result.concurrency.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        assert result.first_bar == 0
        assert result.last_bar == 12

    def test_single_bar_spans_are_legal(self) -> None:
        result = sample_uniqueness([4, 4, 9], [4, 4, 9])
        assert result.average_uniqueness == pytest.approx([0.5, 0.5, 1.0])
        assert result.effective_sample_size == pytest.approx(2.0)

    def test_weights_times_span_lengths_count_the_covered_bars(self) -> None:
        # An exact identity: sum_i sum_{t in span_i} 1/c_t = sum_{t covered} 1.
        first = np.array([0, 2, 3, 11])
        last = np.array([3, 5, 8, 13])
        result = sample_uniqueness(first, last)
        lengths = last - first + 1
        covered = int(np.count_nonzero(result.concurrency > 0))
        assert float(np.sum(result.average_uniqueness * lengths)) == pytest.approx(covered)


class TestEffectiveSampleSize:
    def test_it_is_the_sum_of_the_weights(self) -> None:
        assert effective_sample_size([0.5, 0.25, 0.25]) == pytest.approx(1.0)

    def test_heavy_overlap_costs_most_of_the_nominal_sample(self) -> None:
        # 100 labels each 21 bars long, started one bar apart: the classic
        # daily-rebalance panel. The point of the whole module is that this is
        # worth far fewer than 100 independent observations.
        starts = np.arange(100)
        result = sample_uniqueness(starts, starts + 20)
        assert result.nominal_count == 100
        assert result.effective_sample_size < 10.0
        assert result.uniqueness_ratio < 0.10
        assert result.max_concurrency == 21

    def test_the_ratio_is_one_only_when_nothing_overlaps(self) -> None:
        starts = np.arange(0, 100, 21)
        assert sample_uniqueness(starts, starts + 20).uniqueness_ratio == 1.0

    def test_normalized_weights_average_one_and_keep_the_ordering(self) -> None:
        result = sample_uniqueness([0, 2, 20], [3, 5, 21])
        normalized = result.normalized_weights()
        assert float(np.sum(normalized)) == pytest.approx(result.nominal_count)
        assert np.argsort(normalized).tolist() == np.argsort(result.average_uniqueness).tolist()

    def test_the_report_states_the_effect_on_sample_size(self) -> None:
        starts = np.arange(50)
        report = sample_uniqueness(starts, starts + 20).report()
        assert "nominal observations   : 50" in report
        assert "effective sample size" in report
        assert "ESS / nominal" in report
        assert "peak concurrency       : 21" in report


class TestLabelSpanConvention:
    def test_a_label_span_starts_the_bar_after_its_event(self) -> None:
        # Two labels whose event bars are 5 apart with a 5-bar horizon: the
        # first consumes bars 3..7, the second bars 8..12. They abut and do not
        # overlap, so both weigh 1. Using [event, resolution] instead would have
        # them share bar 7 and both drop below 1.
        spec = BarrierSpec(horizon=5, volatility_window=WARMUP_WINDOW)
        prices = hand_derived_path([0.001, -0.002, 0.001, -0.001, 0.002] * 3)
        labels = triple_barrier_labels(prices, [FIRST_EVENT_BAR, FIRST_EVENT_BAR + 5], spec)
        assert labels.resolution_index.tolist() == [7, 12]

        result = sample_uniqueness_from_labels(labels)
        assert result.average_uniqueness.tolist() == [1.0, 1.0]
        assert result.first_bar == FIRST_EVENT_BAR + 1

    def test_overlapping_events_are_down_weighted(self) -> None:
        spec = BarrierSpec(horizon=5, volatility_window=WARMUP_WINDOW)
        prices = hand_derived_path([0.001, -0.002, 0.001, -0.001, 0.002] * 3)
        events = [FIRST_EVENT_BAR, FIRST_EVENT_BAR + 1, FIRST_EVENT_BAR + 2]
        result = sample_uniqueness_from_labels(triple_barrier_labels(prices, events, spec))
        assert np.all(result.average_uniqueness < 1.0)
        assert result.effective_sample_size < result.nominal_count

    def test_duplicate_events_are_treated_as_fully_overlapping(self) -> None:
        spec = BarrierSpec(horizon=5, volatility_window=WARMUP_WINDOW)
        prices = hand_derived_path([0.001, -0.002, 0.001, -0.001, 0.002] * 3)
        labels = triple_barrier_labels(prices, [FIRST_EVENT_BAR] * 3, spec)
        result = sample_uniqueness_from_labels(labels)
        assert result.average_uniqueness == pytest.approx([1 / 3, 1 / 3, 1 / 3])
        assert result.effective_sample_size == pytest.approx(1.0)


class TestRefusals:
    def test_zero_labels_is_refused(self) -> None:
        with pytest.raises(LabelInputError, match="zero labels"):
            sample_uniqueness([], [])

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(LabelInputError, match="equal length"):
            sample_uniqueness([0, 1], [5])

    def test_a_span_that_ends_before_it_starts_is_refused(self) -> None:
        with pytest.raises(LabelInputError, match="last_bar must be >="):
            sample_uniqueness([0, 9], [5, 3])

    def test_a_negative_bar_index_is_refused(self) -> None:
        with pytest.raises(LabelInputError, match="non-negative"):
            sample_uniqueness([-1], [5])

    def test_an_absurd_bar_range_is_refused_rather_than_allocated(self) -> None:
        with pytest.raises(LabelInputError, match="guard"):
            sample_uniqueness([0], [MAX_CONCURRENCY_SPAN_BARS + 1])
