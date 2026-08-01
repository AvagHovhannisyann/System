"""Hypothesis property tests for PurgedKFold (P8.1, directive §8).

The unit tests next door pin exact index sets on hand-built samples. These pin
the *invariant the whole method exists to provide*, on samples nobody chose:

    for every fold, no training observation's label interval overlaps that
    fold's test span, and none begins inside the embargo window after it.

Two things make these tests worth their runtime rather than decoration.

**An independent oracle.** :func:`_reference_folds` re-implements the splitter
in the dumbest possible way — Python loops over :class:`datetime.datetime`
objects, comparing instants directly. It shares no code with the implementation
under test, which converts everything to int64 nanoseconds and works in
vectorized numpy. When the two agree on random input, the agreement is evidence;
if the fast path's integer conversion, its lexsort tie-breaking or its mask
arithmetic were wrong, the oracle would not follow it into the mistake.

**Failures are asserted, not skipped.** When a random configuration empties a
training set the implementation raises, and the test asserts the *oracle* also
produced an empty training set for the same fold. Swallowing the exception
would let a splitter that raises on everything pass every property here.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.models import EmptyTrainingSetError, PurgedKFold

if TYPE_CHECKING:
    from collections.abc import Sequence

_BASE = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
_UNIT = dt.timedelta(hours=1)
"""Sampling grid. Hours keep the integers small while leaving room for ties."""

Sample = tuple[list[dt.datetime], list[dt.datetime]]


# ---------------------------------------------------------------------------
# Independent reference implementation — deliberately naive
# ---------------------------------------------------------------------------


def _reference_folds(
    events: Sequence[dt.datetime],
    ends: Sequence[dt.datetime],
    *,
    n_splits: int,
    embargo: dt.timedelta,
) -> list[tuple[list[int], list[int]]]:
    """Compute purged, embargoed folds by brute force over datetimes.

    Args:
        events: event time per observation.
        ends: label end time per observation.
        n_splits: number of folds.
        embargo: wall-clock embargo applied after each test span.

    Returns:
        ``[(train positions, test positions), ...]`` in fold order, each sorted
        ascending. Training sets may be empty here: the reference has no
        opinion about that, which is what lets the tests check the real
        implementation's error against it.
    """
    n = len(events)
    order = sorted(range(n), key=lambda i: (events[i], ends[i], i))
    sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        sizes[i] += 1

    result: list[tuple[list[int], list[int]]] = []
    cursor = 0
    for size in sizes:
        test = sorted(order[cursor : cursor + size])
        cursor += size
        test_set = set(test)
        span_start = min(events[i] for i in test)
        span_end = max(ends[i] for i in test)
        train = [
            j
            for j in range(n)
            if j not in test_set
            and not (events[j] <= span_end and ends[j] >= span_start)  # not purged
            and not (span_end < events[j] <= span_end + embargo)  # not embargoed
        ]
        result.append((train, test))
    return result


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _samples(draw: st.DrawFn, *, min_size: int = 2, max_size: int = 40) -> Sample:
    """Draw label spans on an hourly grid, with ties and zero-length labels likely.

    The offset range is deliberately narrow relative to the sample size so
    duplicate timestamps occur often — a cross-sectional panel is nothing but
    duplicate timestamps, and that is where a tie-breaking bug would hide.
    """
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    starts = draw(st.lists(st.integers(min_value=0, max_value=120), min_size=n, max_size=n))
    horizons = draw(st.lists(st.integers(min_value=0, max_value=48), min_size=n, max_size=n))
    events = [_BASE + offset * _UNIT for offset in starts]
    ends = [event + horizon * _UNIT for event, horizon in zip(events, horizons, strict=True)]
    return events, ends


_embargoes = st.integers(min_value=0, max_value=72).map(lambda hours: hours * _UNIT)


@st.composite
def _configurations(draw: st.DrawFn) -> tuple[Sample, int, dt.timedelta]:
    """Draw a sample together with a legal ``n_splits`` and an embargo."""
    sample = draw(_samples())
    n_splits = draw(st.integers(min_value=2, max_value=len(sample[0])))
    return sample, n_splits, draw(_embargoes)


# ---------------------------------------------------------------------------
# The invariant the method exists to provide
# ---------------------------------------------------------------------------


@given(_configurations())
@hypothesis_settings(max_examples=600, deadline=None)
def test_no_training_observation_overlaps_its_test_span(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """Purging holds: no training label interval meets the test fold's span."""
    (events, ends), n_splits, embargo = configuration
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return  # covered by test_empty_training_set_error_agrees_with_the_reference

    for fold in folds:
        for j in fold.train.tolist():
            overlaps = events[j] <= fold.test_span_end and ends[j] >= fold.test_span_start
            assert not overlaps, (fold.index, j, events[j], ends[j])


@given(_configurations())
@hypothesis_settings(max_examples=600, deadline=None)
def test_no_training_observation_starts_inside_the_embargo(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """The embargo holds: no training observation begins in the forbidden window."""
    (events, ends), n_splits, embargo = configuration
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return

    for fold in folds:
        horizon = fold.test_span_end + embargo
        for j in fold.train.tolist():
            assert not (fold.test_span_end < events[j] <= horizon), (fold.index, j, events[j])


@given(_configurations())
@hypothesis_settings(max_examples=600, deadline=None)
def test_matches_the_brute_force_reference_exactly(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """Vectorized int64 splitter agrees with naive datetime loops, index for index."""
    (events, ends), n_splits, embargo = configuration
    expected = _reference_folds(events, ends, n_splits=n_splits, embargo=embargo)
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return

    actual = [(fold.train.tolist(), fold.test.tolist()) for fold in folds]
    assert actual == expected


@given(_configurations())
@hypothesis_settings(max_examples=600, deadline=None)
def test_empty_training_set_error_agrees_with_the_reference(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """The splitter raises exactly when a fold genuinely has nothing left to train on.

    Both directions matter. Raising when the reference kept rows would mean
    usable folds are being thrown away; *not* raising when the reference kept
    nothing would mean an empty training array reaching a caller.
    """
    (events, ends), n_splits, embargo = configuration
    expected = _reference_folds(events, ends, n_splits=n_splits, embargo=embargo)
    empty_folds = [index for index, (train, _) in enumerate(expected) if not train]
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)

    if empty_folds:
        with pytest.raises(EmptyTrainingSetError) as excinfo:
            cv.folds()
        assert excinfo.value.fold_index == empty_folds[0]
    else:
        assert all(len(fold.train) > 0 for fold in cv.folds())


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------


@given(_configurations())
@hypothesis_settings(max_examples=400, deadline=None)
def test_train_and_test_are_disjoint_and_test_sets_partition_the_sample(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """Train ∩ test = ∅ per fold, and the test sets cover every row exactly once."""
    (events, ends), n_splits, embargo = configuration
    n = len(events)
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return

    covered: list[int] = []
    for fold in folds:
        train = fold.train.tolist()
        test = fold.test.tolist()
        assert set(train).isdisjoint(test)
        assert all(0 <= i < n for i in train + test)
        covered.extend(test)

    assert sorted(covered) == list(range(n))


@given(_configurations())
@hypothesis_settings(max_examples=400, deadline=None)
def test_counts_partition_the_sample(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """Train + test + purged + embargoed = n, so the diagnostics never double-count."""
    (events, ends), n_splits, embargo = configuration
    n = len(events)
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return

    for fold in folds:
        assert len(fold.train) + len(fold.test) + fold.n_purged + fold.n_embargoed == n


@given(_configurations())
@hypothesis_settings(max_examples=400, deadline=None)
def test_index_arrays_are_sorted_ascending(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """Both halves come back in ascending original-position order, without duplicates."""
    (events, ends), n_splits, embargo = configuration
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        splits = list(cv.split())
    except EmptyTrainingSetError:
        return

    for train, test in splits:
        assert train.tolist() == sorted(set(train.tolist()))
        assert test.tolist() == sorted(set(test.tolist()))


# ---------------------------------------------------------------------------
# Monotonicity and permutation invariance
# ---------------------------------------------------------------------------


@given(_configurations(), _embargoes)
@hypothesis_settings(max_examples=400, deadline=None)
def test_increasing_the_embargo_never_grows_a_training_set(
    configuration: tuple[Sample, int, dt.timedelta],
    extra: dt.timedelta,
) -> None:
    """Training sets are nested as the embargo grows, and emptiness is absorbing.

    A larger embargo drops a superset of what a smaller one drops, so each
    fold's training set can only shrink — and once some fold is empty, adding
    embargo cannot bring it back.
    """
    (events, ends), n_splits, embargo = configuration
    smaller = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    larger = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo + extra)

    try:
        small_folds = [set(fold.train.tolist()) for fold in smaller.folds()]
    except EmptyTrainingSetError:
        with pytest.raises(EmptyTrainingSetError):
            larger.folds()
        return

    try:
        large_folds = [set(fold.train.tolist()) for fold in larger.folds()]
    except EmptyTrainingSetError:
        return

    for small, large in zip(small_folds, large_folds, strict=True):
        assert large <= small


@given(_configurations(), st.randoms(use_true_random=False))
@hypothesis_settings(max_examples=400, deadline=None)
def test_fold_membership_is_a_property_of_the_observation_not_its_row_number(
    configuration: tuple[Sample, int, dt.timedelta],
    rng: random.Random,
) -> None:
    """Permuting the input permutes the indices and changes nothing else.

    The tie-break on original position means a permutation *can* move rows that
    share an identical ``(event_time, label_end_time)`` between folds. The
    property that must hold regardless is the leakage one: whatever the row
    order, no training row in the permuted split overlaps its test span or sits
    in the embargo.
    """
    (events, ends), n_splits, embargo = configuration
    permutation = list(range(len(events)))
    rng.shuffle(permutation)
    shuffled_events = [events[i] for i in permutation]
    shuffled_ends = [ends[i] for i in permutation]

    cv = PurgedKFold(shuffled_events, shuffled_ends, n_splits=n_splits, embargo=embargo)
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return

    for fold in folds:
        horizon = fold.test_span_end + embargo
        for j in fold.train.tolist():
            event, end = shuffled_events[j], shuffled_ends[j]
            assert not (event <= fold.test_span_end and end >= fold.test_span_start)
            assert not (fold.test_span_end < event <= horizon)


@given(_configurations())
@hypothesis_settings(max_examples=200, deadline=None)
def test_zero_embargo_is_pure_purging(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """``embargo=0`` reports nothing embargoed, on any sample."""
    (events, ends), n_splits, _ = configuration
    cv = PurgedKFold(events, ends, n_splits=n_splits, embargo=dt.timedelta(0))
    try:
        folds = cv.folds()
    except EmptyTrainingSetError:
        return

    assert all(fold.n_embargoed == 0 for fold in folds)


@given(_configurations())
@hypothesis_settings(max_examples=200, deadline=None)
def test_splitting_is_deterministic(
    configuration: tuple[Sample, int, dt.timedelta],
) -> None:
    """Two identically-configured splitters produce identical arrays (invariant I2)."""
    (events, ends), n_splits, embargo = configuration
    first = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    second = PurgedKFold(events, ends, n_splits=n_splits, embargo=embargo)
    try:
        expected = [(a.tobytes(), b.tobytes()) for a, b in first.split()]
    except EmptyTrainingSetError:
        with pytest.raises(EmptyTrainingSetError):
            second.split()
        return

    assert [(a.tobytes(), b.tobytes()) for a, b in second.split()] == expected
