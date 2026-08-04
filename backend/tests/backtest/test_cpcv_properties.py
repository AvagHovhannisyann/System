"""Property tests for CPCV purging and path assembly (P10.2).

**What this suite is designed against.** DECISIONS.md D-018 closes with a
standing lesson from the bitemporal layer: *"A property suite that randomizes
data while holding query shape fixed measures breadth, not correctness. 75,936
breadth cases passed while this leak was live; it took nesting depth as a
first-class strategy dimension to reach the shape."* The analogue here is
obvious once stated. Drawing forty random label timestamps with the same
uniform spacing and the same short horizons, over and over, exercises one
structural regime very thoroughly and every other regime not at all — and
purging is *entirely* a question of structure. A label that stays inside its own
group and a label that spans three groups reach different branches; an embargo
shorter than a group and an embargo longer than the whole sample reach different
branches; evenly spaced observations and a burst of simultaneous ones reach
different branches.

So the sampler below draws a *shape* first and values second. Three independent
structural dimensions, each with regimes chosen to straddle the group boundary,
because the group boundary is where this code decides things:

* **spacing** — evenly spaced, heavily tied (many observations sharing a
  timestamp), or bursty (dense clusters separated by long gaps);
* **horizon** — labels that resolve instantly, inside their own group, across
  several groups, or over the whole sample;
* **embargo** — none, a fraction of a group, a whole group, or several.

Crossed with the group count, the test-group count and the sample length, that
is the space the properties below are asserted over.

**The properties.**

1. **No leakage, in every combination.** No surviving training observation's
   label interval overlaps any test observation's. Not on average — in each of
   the ``C(N, k)`` combinations separately. This is the property the whole
   construction exists to provide.
2. **Agreement with an independent reference.** The implementation purges
   against each contiguous test block's hull. The reference here is López de
   Prado (2018) §7.4.1's literal three-clause rule applied to every individual
   test observation, with each observation's end extended by the embargo
   (§7.4.2). The two are equivalent when label starts are non-decreasing —
   which :meth:`CombinatorialPurgedCV.split` validates — so any disagreement is
   a bug in one of them.
3. **Exact accounting.** Test, purged, embargoed and kept partition the sample.
4. **Monotonicity in the embargo.** Lengthening the embargo can only ever remove
   training observations, never add them. A sign error in the embargo arithmetic
   survives every count-based check and fails this one.
5. **Complete paths.** Every assembled path covers every observation exactly
   once, whatever the structure.
"""

from __future__ import annotations

import itertools
from math import comb
from typing import Literal

import numpy as np
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.backtest.cpcv import CombinatorialPurgedCV, CPCVSplits, EmptyTrainingSetError
from backend.backtest.metrics import FloatArray

_Spacing = Literal["even", "tied", "bursty"]
_HorizonRegime = Literal["instant", "within_group", "across_groups", "whole_sample"]
_EmbargoRegime = Literal["none", "sub_group", "one_group", "several_groups"]

# Horizon and embargo caps as multiples of one group's width in time. The
# fractions straddle the group boundary deliberately: purging and embargoing
# behave differently either side of it.
_HORIZON_CAPS: dict[_HorizonRegime, float] = {
    "instant": 0.0,
    "within_group": 0.6,
    "across_groups": 1.5,
    "whole_sample": 3.5,
}
_EMBARGO_CAPS: dict[_EmbargoRegime, float] = {
    "none": 0.0,
    "sub_group": 0.4,
    "one_group": 1.0,
    "several_groups": 3.0,
}


class _Sample:
    """One randomly drawn CPCV problem, plus the structural regime that shaped it."""

    def __init__(
        self,
        *,
        n_groups: int,
        n_test_groups: int,
        embargo: float,
        starts: FloatArray,
        ends: FloatArray,
        spacing: _Spacing,
        horizon_regime: _HorizonRegime,
        embargo_regime: _EmbargoRegime,
    ) -> None:
        self.cv = CombinatorialPurgedCV(
            n_groups=n_groups, n_test_groups=n_test_groups, embargo=embargo
        )
        self.starts = starts
        self.ends = ends
        self.embargo = embargo
        self.spacing = spacing
        self.horizon_regime = horizon_regime
        self.embargo_regime = embargo_regime

    def __repr__(self) -> str:
        return (
            f"_Sample(n_groups={self.cv.n_groups}, n_test_groups={self.cv.n_test_groups}, "
            f"n_observations={self.starts.size}, spacing={self.spacing!r}, "
            f"horizon={self.horizon_regime!r}, embargo={self.embargo_regime!r} "
            f"({self.embargo:.4f}))"
        )

    def with_embargo(self, embargo: float) -> _Sample:
        """Return the same sample under a different embargo length."""
        return _Sample(
            n_groups=self.cv.n_groups,
            n_test_groups=self.cv.n_test_groups,
            embargo=embargo,
            starts=self.starts,
            ends=self.ends,
            spacing=self.spacing,
            horizon_regime=self.horizon_regime,
            embargo_regime=self.embargo_regime,
        )


def _gaps_for(spacing: _Spacing, raw: list[float]) -> list[float]:
    """Turn draws in ``[0, 1]`` into inter-observation gaps of the chosen shape."""
    if spacing == "even":
        return [0.5 + 1.5 * value for value in raw]
    if spacing == "tied":
        # Half the observations share their predecessor's timestamp exactly, so
        # group boundaries fall in the middle of simultaneous events.
        return [0.0 if value < 0.5 else 1.0 for value in raw]
    # Dense clusters separated by long silences: group widths in *time* then
    # differ by orders of magnitude even though group sizes differ by at most 1.
    return [0.02 if value < 0.85 else 8.0 for value in raw]


@st.composite
def _samples(draw: st.DrawFn) -> _Sample:
    """Draw a CPCV problem: structure first, values second."""
    n_groups = draw(st.integers(min_value=2, max_value=6))
    n_test_groups = draw(st.integers(min_value=1, max_value=n_groups - 1))
    n_observations = draw(st.integers(min_value=n_groups * 2, max_value=40))
    spacing: _Spacing = draw(st.sampled_from(["even", "tied", "bursty"]))
    horizon_regime: _HorizonRegime = draw(
        st.sampled_from(["instant", "within_group", "across_groups", "whole_sample"])
    )
    embargo_regime: _EmbargoRegime = draw(
        st.sampled_from(["none", "sub_group", "one_group", "several_groups"])
    )

    unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    spacing_draws = draw(st.lists(unit, min_size=n_observations, max_size=n_observations))
    horizon_draws = draw(st.lists(unit, min_size=n_observations, max_size=n_observations))
    embargo_draw = draw(unit)

    starts = np.cumsum(np.asarray(_gaps_for(spacing, spacing_draws), dtype=np.float64))
    span = float(starts[-1] - starts[0])
    # When every observation shares a timestamp the span is 0; fall back to a
    # unit width so the horizon and embargo regimes remain meaningful rather
    # than silently collapsing to zero.
    group_width = (span / n_groups) if span > 0.0 else 1.0

    horizon_cap = _HORIZON_CAPS[horizon_regime] * group_width
    ends = starts + np.asarray(horizon_draws, dtype=np.float64) * horizon_cap
    embargo = _EMBARGO_CAPS[embargo_regime] * group_width * embargo_draw

    return _Sample(
        n_groups=n_groups,
        n_test_groups=n_test_groups,
        embargo=embargo,
        starts=starts,
        ends=ends,
        spacing=spacing,
        horizon_regime=horizon_regime,
        embargo_regime=embargo_regime,
    )


def _reference_train_indices(
    starts: FloatArray,
    ends: FloatArray,
    bounds: tuple[tuple[int, int], ...],
    test_groups: tuple[int, ...],
    embargo: float,
) -> list[int]:
    """Purge by López de Prado (2018) §7.4.1's rule, one test observation at a time.

    A training observation is dropped when its label interval and a test
    observation's interval — the latter extended forward by the embargo — satisfy
    any of the three overlap clauses: the training label starts inside the test
    interval, ends inside it, or envelops it.
    """
    test: set[int] = set()
    for group in test_groups:
        start, stop = bounds[group]
        test.update(range(start, stop))

    keep: list[int] = []
    for candidate in range(starts.size):
        if candidate in test:
            continue
        train_start = float(starts[candidate])
        train_end = float(ends[candidate])
        dropped = False
        for observation in sorted(test):
            test_start = float(starts[observation])
            test_end = float(ends[observation]) + embargo
            starts_inside = test_start <= train_start <= test_end
            ends_inside = test_start <= train_end <= test_end
            envelops = train_start <= test_start and test_end <= train_end
            if starts_inside or ends_inside or envelops:
                dropped = True
                break
        if not dropped:
            keep.append(candidate)
    return keep


def _reference_splits(sample: _Sample) -> list[list[int]]:
    bounds = sample.cv.group_slices(int(sample.starts.size))
    return [
        _reference_train_indices(sample.starts, sample.ends, bounds, combination, sample.embargo)
        for combination in itertools.combinations(
            range(sample.cv.n_groups), sample.cv.n_test_groups
        )
    ]


def _has_an_empty_training_set(sample: _Sample) -> bool:
    """Whether some combination is purged down to nothing, per the reference."""
    return any(len(train) == 0 for train in _reference_splits(sample))


def _splits_or_confirmed_refusal(sample: _Sample) -> CPCVSplits | None:
    """Return the sample's splits, or confirm it was refused and return ``None``.

    Aggressive structures — a label spanning the whole sample, an embargo longer
    than the data — legitimately leave some combination with nothing to train on,
    and :meth:`CombinatorialPurgedCV.split` is required to refuse those rather
    than emit an undefined path. Routing every property through this helper means
    no drawn example ever asserts nothing: it either exercises the property or
    exercises the refusal.
    """
    if _has_an_empty_training_set(sample):
        with pytest.raises(EmptyTrainingSetError):
            sample.cv.split(sample.starts, sample.ends)
        return None
    return sample.cv.split(sample.starts, sample.ends)


_SETTINGS = hypothesis_settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)


@given(_samples())
@_SETTINGS
def test_no_training_label_overlaps_any_test_label_in_any_combination(sample: _Sample) -> None:
    splits = _splits_or_confirmed_refusal(sample)
    if splits is None:
        return

    for split in splits:
        train_starts = sample.starts[split.train_indices]
        train_ends = sample.ends[split.train_indices]
        test_starts = sample.starts[split.test_indices]
        test_ends = sample.ends[split.test_indices]
        # Intervals [a0, a1] and [b0, b1] overlap iff a0 <= b1 and a1 >= b0.
        overlaps = (train_starts[:, None] <= test_ends[None, :]) & (
            train_ends[:, None] >= test_starts[None, :]
        )
        assert not overlaps.any(), (
            f"split {split.split_id} (test groups {split.test_groups}) leaves "
            f"{int(overlaps.sum())} overlapping train/test label pairs"
        )


@given(_samples())
@_SETTINGS
def test_purging_agrees_with_the_reference_implementation(sample: _Sample) -> None:
    splits = _splits_or_confirmed_refusal(sample)
    if splits is None:
        return

    reference = _reference_splits(sample)
    assert len(splits) == len(reference)
    for split, expected in zip(splits, reference, strict=True):
        assert split.train_indices.tolist() == expected


@given(_samples())
@_SETTINGS
def test_test_purged_embargoed_and_kept_partition_the_sample(sample: _Sample) -> None:
    splits = _splits_or_confirmed_refusal(sample)
    if splits is None:
        return
    n_observations = int(sample.starts.size)
    for split in splits:
        total = (
            split.test_indices.size + split.n_purged + split.n_embargoed + split.train_indices.size
        )
        assert total == n_observations, (
            f"split {split.split_id} accounts for {total} of {n_observations} observations; "
            "purged and embargoed counts must never double-count or drop one"
        )
        assert split.n_purged >= 0
        assert split.n_embargoed >= 0
        if sample.embargo == 0.0:
            assert split.n_embargoed == 0


@given(_samples(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
@_SETTINGS
def test_lengthening_the_embargo_only_ever_removes_training_observations(
    sample: _Sample, extra: float
) -> None:
    # A sign error in the embargo arithmetic — dropping observations *before* the
    # test block instead of after it, say — keeps every count plausible and
    # every partition exact. It does not survive this.
    longer = sample.with_embargo(sample.embargo + extra * max(sample.embargo, 1.0))
    shorter_splits = _splits_or_confirmed_refusal(sample)
    longer_splits = _splits_or_confirmed_refusal(longer)
    if shorter_splits is None or longer_splits is None:
        return

    for short, long in zip(shorter_splits, longer_splits, strict=True):
        assert short.test_groups == long.test_groups
        assert set(long.train_indices.tolist()) <= set(short.train_indices.tolist())


@given(_samples())
@_SETTINGS
def test_the_structural_counts_hold_whatever_the_sample_looks_like(sample: _Sample) -> None:
    cv = sample.cv
    assert cv.n_splits == comb(cv.n_groups, cv.n_test_groups)
    assert cv.n_paths == comb(cv.n_groups - 1, cv.n_test_groups - 1)
    assert cv.n_paths * cv.n_groups == cv.n_test_groups * cv.n_splits

    bounds = cv.group_slices(int(sample.starts.size))
    assert bounds[0][0] == 0
    assert bounds[-1][1] == sample.starts.size
    for earlier, later in itertools.pairwise(bounds):
        assert earlier[1] == later[0]
        assert earlier[1] > earlier[0]


@given(_samples())
@_SETTINGS
def test_every_path_covers_every_observation_exactly_once(sample: _Sample) -> None:
    splits = _splits_or_confirmed_refusal(sample)
    if splits is None:
        return
    n_observations = int(sample.starts.size)

    # Each split forecasts the observation's own index, so a correctly assembled
    # path reads back as arange(n): gaps, duplicates and misalignments all show.
    per_split = [split.test_indices.astype(np.float64) for split in splits]
    paths = splits.assemble_paths(per_split)

    assert paths.shape == (sample.cv.n_paths, n_observations)
    expected = np.arange(n_observations, dtype=np.float64)
    for row in paths:
        assert np.array_equal(row, expected)


@given(_samples())
@_SETTINGS
def test_train_and_test_are_disjoint_and_test_is_the_union_of_its_groups(sample: _Sample) -> None:
    splits = _splits_or_confirmed_refusal(sample)
    if splits is None:
        return
    for split in splits:
        expected: list[int] = []
        for group in split.test_groups:
            start, stop = splits.group_slices[group]
            expected.extend(range(start, stop))
        assert split.test_indices.tolist() == expected
        assert set(split.train_indices.tolist()).isdisjoint(expected)


@given(_samples())
@_SETTINGS
def test_the_path_map_draws_each_group_from_a_split_that_tested_it(sample: _Sample) -> None:
    splits = _splits_or_confirmed_refusal(sample)
    if splits is None:
        return
    mapping = splits.path_split_map()

    assert mapping.shape == (sample.cv.n_paths, sample.cv.n_groups)
    for group in range(sample.cv.n_groups):
        supplying = mapping[:, group].tolist()
        # One distinct split per path, and each of them really did test the group.
        assert len(set(supplying)) == sample.cv.n_paths
        for split_id in supplying:
            assert group in splits[split_id].test_groups
