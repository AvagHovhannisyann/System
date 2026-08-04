"""Exact fold-boundary tests for PurgedKFold (P8.1, directive §5 Phase 8 gate).

The gate for Phase 8 is that fold boundaries are *unit-tested for correctness*,
not assumed. So every expectation in this file is a literal index set derived by
hand from the observation spans, with the derivation written next to it. Sizes
are never asserted in place of membership: a fold of the right size containing
the wrong rows is exactly the silent failure this module exists to prevent.

The reference sample most tests use (:data:`_CANONICAL`) is 10 daily
observations with a 2-day label horizon::

    obs   event_time   label_end_time
     0     Jan 1        Jan 3
     1     Jan 2        Jan 4
     2     Jan 3        Jan 5
     3     Jan 4        Jan 6
     4     Jan 5        Jan 7
     5     Jan 6        Jan 8
     6     Jan 7        Jan 9
     7     Jan 8        Jan 10
     8     Jan 9        Jan 11
     9     Jan 10       Jan 12

With ``n_splits=5`` the test folds are the contiguous pairs
``{0,1} {2,3} {4,5} {6,7} {8,9}``. All times are UTC.
"""

from __future__ import annotations

import datetime as dt
import itertools
import re
from typing import cast

import numpy as np
import pandas as pd
import pytest

from backend.models import EmptyTrainingSetError, Fold, PurgedCVError, PurgedKFold

DAY = dt.timedelta(days=1)
HOUR = dt.timedelta(hours=1)
JAN1 = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def _jan(day: int) -> dt.datetime:
    """Return "January ``day``" 2024 at midnight UTC, counted from Jan 1.

    Days past 31 roll into February so a test can express a long label horizon
    (``_jan(40)`` is Feb 9) without arithmetic noise at the call site.
    """
    return JAN1 + (day - 1) * DAY


def _daily(n: int, *, horizon_days: int) -> tuple[list[dt.datetime], list[dt.datetime]]:
    """Build ``n`` daily observations from Jan 1 with a fixed label horizon."""
    events = [JAN1 + i * DAY for i in range(n)]
    ends = [e + horizon_days * DAY for e in events]
    return events, ends


_CANONICAL = _daily(10, horizon_days=2)


def _index_sets(folds: list[Fold]) -> list[tuple[set[int], set[int]]]:
    """Reduce folds to ``(train set, test set)`` pairs of plain ints."""
    return [(set(fold.train.tolist()), set(fold.test.tolist())) for fold in folds]


# ---------------------------------------------------------------------------
# Exact fold boundaries: purging only (embargo = 0)
# ---------------------------------------------------------------------------


def test_purge_only_produces_exact_index_sets() -> None:
    """Every train/test set on the canonical sample, derived by hand.

    Test span of fold *k* is ``[min event_time, max label_end_time]`` over its
    test rows; a non-test row is purged iff its closed span
    ``[event, label_end]`` meets that interval.

    fold 0 test {0,1}: span [Jan 1, Jan 4].
        obs2 Jan3-Jan5 meets it, obs3 Jan4-Jan6 meets it (event == span end),
        obs4 Jan5-Jan7 starts after Jan 4 -> kept.
    fold 1 test {2,3}: span [Jan 3, Jan 6].
        obs0 Jan1-Jan3 ends exactly at the span start -> purged (boundary rule),
        obs1 Jan2-Jan4 meets it, obs4 Jan5-Jan7 and obs5 Jan6-Jan8 start inside,
        obs6 Jan7-Jan9 starts after Jan 6 -> kept.
    fold 2 test {4,5}: span [Jan 5, Jan 8].
        obs0 ends Jan 3 and obs1 ends Jan 4, both before Jan 5 -> kept,
        obs2 Jan3-Jan5 ends exactly at the span start -> purged,
        obs3 Jan4-Jan6 meets it, obs6 and obs7 start inside,
        obs8 Jan9-Jan11 and obs9 Jan10-Jan12 start after Jan 8 -> kept.
    fold 3 test {6,7}: span [Jan 7, Jan 10].
        obs0..obs3 end Jan 3..Jan 6, all before Jan 7 -> kept,
        obs4 Jan5-Jan7 ends exactly at the span start -> purged,
        obs5 Jan6-Jan8 meets it, obs8 and obs9 start on/before Jan 10 -> purged.
    fold 4 test {8,9}: span [Jan 9, Jan 12].
        obs0..obs5 end Jan 3..Jan 8, all before Jan 9 -> kept,
        obs6 Jan7-Jan9 ends exactly at the span start -> purged,
        obs7 Jan8-Jan10 meets it.
    """
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5, embargo=dt.timedelta(0))

    assert _index_sets(cv.folds()) == [
        ({4, 5, 6, 7, 8, 9}, {0, 1}),
        ({6, 7, 8, 9}, {2, 3}),
        ({0, 1, 8, 9}, {4, 5}),
        ({0, 1, 2, 3}, {6, 7}),
        ({0, 1, 2, 3, 4, 5}, {8, 9}),
    ]


def test_embargo_of_one_day_removes_exactly_the_next_observation() -> None:
    """One extra row leaves each early fold, and it is the identifiable one.

    Relative to the purge-only sets above, an embargo of one day additionally
    drops every non-test row whose ``event_time`` lies in
    ``(span end, span end + 1 day]``:

    fold 0 span ends Jan 4 -> window (Jan 4, Jan 5] -> obs4 (event Jan 5).
    fold 1 span ends Jan 6 -> window (Jan 6, Jan 7] -> obs6 (event Jan 7).
    fold 2 span ends Jan 8 -> window (Jan 8, Jan 9] -> obs8 (event Jan 9).
    fold 3 span ends Jan 10 -> window (Jan 10, Jan 11] -> no observation starts
        after Jan 10, so nothing is embargoed.
    fold 4 span ends Jan 12 -> nothing follows it at all.
    """
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5, embargo=DAY)

    assert _index_sets(cv.folds()) == [
        ({5, 6, 7, 8, 9}, {0, 1}),
        ({7, 8, 9}, {2, 3}),
        ({0, 1, 9}, {4, 5}),
        ({0, 1, 2, 3}, {6, 7}),
        ({0, 1, 2, 3, 4, 5}, {8, 9}),
    ]


def test_fold_spans_and_counts_are_reported_exactly() -> None:
    """Fold diagnostics match the hand derivation, not just the index sets."""
    events, ends = _CANONICAL
    folds = PurgedKFold(events, ends, n_splits=5, embargo=DAY).folds()

    assert [(f.test_span_start, f.test_span_end) for f in folds] == [
        (_jan(1), _jan(4)),
        (_jan(3), _jan(6)),
        (_jan(5), _jan(8)),
        (_jan(7), _jan(10)),
        (_jan(9), _jan(12)),
    ]
    assert [f.n_purged for f in folds] == [2, 4, 4, 4, 2]
    assert [f.n_embargoed for f in folds] == [1, 1, 1, 0, 0]
    assert [f.index for f in folds] == [0, 1, 2, 3, 4]
    assert {f.embargo for f in folds} == {DAY}


def test_purge_and_embargo_counts_account_for_every_observation() -> None:
    """Train + test + purged + embargoed = n, so no row is dropped twice or lost.

    Purge and embargo are mutually exclusive by construction (purging needs
    ``event_time <= span end``, the embargo needs ``event_time > span end``), so
    the four counts partition the sample. If they ever overlapped, the reported
    ``n_purged``/``n_embargoed`` would double-count and the fold-size diagnostic
    P8.5 renders would be wrong.
    """
    events, ends = _CANONICAL
    for embargo_days in (0, 1, 2, 3):
        folds = PurgedKFold(events, ends, n_splits=5, embargo=embargo_days * DAY).folds()
        for fold in folds:
            total = len(fold.train) + len(fold.test) + fold.n_purged + fold.n_embargoed
            assert total == 10, (embargo_days, fold.index)


# ---------------------------------------------------------------------------
# The boundary decision: a label ending exactly at the test start is purged
# ---------------------------------------------------------------------------


def test_label_ending_exactly_at_test_start_is_purged() -> None:
    """Touching at a single instant counts as overlap and is purged.

    Sample: obs0 Jan1-Jan5, obs1 Jan2-Jan4 (ends one day *before* the span
    start), obs2 Jan5-Jan6, obs3 Jan6-Jan7. With ``n_splits=2`` the second test
    fold is {2,3}, whose span starts Jan 5. obs0's label ends exactly at Jan 5
    and must be purged; obs1's ends Jan 4 and must survive.
    """
    events = [_jan(1), _jan(2), _jan(5), _jan(6)]
    ends = [_jan(5), _jan(4), _jan(6), _jan(7)]
    folds = PurgedKFold(events, ends, n_splits=2, embargo=dt.timedelta(0)).folds()

    assert folds[1].test_span_start == _jan(5)
    assert set(folds[1].train.tolist()) == {1}
    assert folds[1].n_purged == 1


def test_boundary_rule_is_exact_to_the_nanosecond() -> None:
    """One nanosecond earlier and the same observation survives.

    This is the units test for the boundary decision: the comparison is exact
    integer nanoseconds, not a rounded or floating-point date comparison, so an
    observation ending 1 ns before the test span opens is kept while one ending
    exactly on it is purged.
    """
    span_start = pd.Timestamp("2024-01-05 00:00:00", tz="UTC")
    events = [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
        span_start,
        pd.Timestamp("2024-01-06", tz="UTC"),
    ]
    ends = [
        span_start,  # ends exactly at the test span start -> purged
        span_start - pd.Timedelta(1, unit="ns"),  # 1 ns earlier -> kept
        pd.Timestamp("2024-01-06", tz="UTC"),
        pd.Timestamp("2024-01-07", tz="UTC"),
    ]
    folds = PurgedKFold(events, ends, n_splits=2, embargo=dt.timedelta(0)).folds()

    assert set(folds[1].test.tolist()) == {2, 3}
    assert set(folds[1].train.tolist()) == {1}


def test_embargo_window_is_closed_at_its_far_end() -> None:
    """An observation starting exactly ``embargo`` after the span end is dropped.

    Sample: obs0 and obs1 form test fold 0 with span [Jan 1, Jan 2]; obs2 starts
    Jan 4 (exactly 2 days after the span end) and obs3 starts Jan 5. An embargo
    of 2 days must drop obs2 and keep obs3.
    """
    events = [_jan(1), _jan(2), _jan(4), _jan(5)]
    ends = [_jan(1), _jan(2), _jan(4), _jan(5)]
    folds = PurgedKFold(events, ends, n_splits=2, embargo=2 * DAY).folds()

    assert folds[0].test_span_end == _jan(2)
    assert set(folds[0].train.tolist()) == {3}
    assert folds[0].n_embargoed == 1


# ---------------------------------------------------------------------------
# Overlap at both edges of the test fold
# ---------------------------------------------------------------------------


def test_overlap_is_purged_at_both_edges_and_when_it_straddles_the_fold() -> None:
    """Left straddle, right straddle, containment and both touching edges all purge.

    Ten observations already in time order; ``n_splits=5`` makes the pairs
    ``{0,1} {2,3} {4,5} {6,7} {8,9}``, and fold 2 — test rows {4,5}, span
    [Jan 10, Jan 12] — is surrounded by one of every overlap shape::

        obs0 Jan1-Jan2    closes before the span opens        -> kept
        obs1 Jan3-Jan11   crosses the LEFT edge               -> purged
        obs2 Jan4-Jan25   CONTAINS the whole span             -> purged
        obs3 Jan5-Jan10   closes exactly ON the span start    -> purged
        obs4 Jan10-Jan11  test
        obs5 Jan10-Jan12  test
        obs6 Jan11-Jan30  opens inside, crosses the RIGHT edge -> purged
        obs7 Jan12-Jan14  opens exactly ON the span end       -> purged
        obs8 Jan20-Jan21  opens after the span (embargo is 0) -> kept
        obs9 Jan28-Jan29  opens after the span                -> kept
    """
    events = [
        _jan(1), _jan(3), _jan(4), _jan(5), _jan(10),
        _jan(10), _jan(11), _jan(12), _jan(20), _jan(28),
    ]  # fmt: skip
    ends = [
        _jan(2), _jan(11), _jan(25), _jan(10), _jan(11),
        _jan(12), _jan(30), _jan(14), _jan(21), _jan(29),
    ]  # fmt: skip
    fold = PurgedKFold(events, ends, n_splits=5, embargo=dt.timedelta(0)).folds()[2]

    assert set(fold.test.tolist()) == {4, 5}
    assert (fold.test_span_start, fold.test_span_end) == (_jan(10), _jan(12))
    assert set(fold.train.tolist()) == {0, 8, 9}
    assert fold.n_purged == 5


def test_long_label_on_a_test_row_extends_the_span_it_protects() -> None:
    """A single long test label purges rows far past the fold's own rows.

    obs2 is in test fold 1 and its label runs to Feb 9 while every other label
    is at most a day. The fold's span is therefore [Jan 3, Feb 9], and every
    remaining row whose interval reaches into it must go — including obs4 and
    obs5, which sit weeks after the fold's own test rows. Only obs0 and obs1,
    which close on Jan 2, survive.
    """
    events = [_jan(1), _jan(2), _jan(3), _jan(4), _jan(20), _jan(31)]
    ends = [_jan(2), _jan(2), _jan(40), _jan(5), _jan(21), _jan(32)]
    folds = PurgedKFold(events, ends, n_splits=3, embargo=dt.timedelta(0)).folds()

    fold = folds[1]
    assert set(fold.test.tolist()) == {2, 3}
    assert fold.test_span_end == _jan(40)
    assert set(fold.train.tolist()) == {0, 1}
    assert fold.n_purged == 2  # obs4 and obs5, both weeks past the test rows


# ---------------------------------------------------------------------------
# Structural invariants across folds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_splits", [2, 3, 4, 5, 7, 10])
def test_train_and_test_are_disjoint_in_every_fold(n_splits: int) -> None:
    """No index appears in both halves of a split."""
    events, ends = _CANONICAL
    for fold in PurgedKFold(events, ends, n_splits=n_splits, embargo=DAY).folds():
        assert set(fold.train.tolist()).isdisjoint(set(fold.test.tolist()))


@pytest.mark.parametrize("n_splits", [2, 3, 4, 5, 7, 10])
def test_test_folds_cover_the_sample_exactly_once(n_splits: int) -> None:
    """The test sets partition the sample: every row once, no row twice."""
    events, ends = _CANONICAL
    folds = PurgedKFold(events, ends, n_splits=n_splits, embargo=DAY).folds()
    seen = [index for fold in folds for index in fold.test.tolist()]

    assert sorted(seen) == list(range(10))
    assert len(seen) == len(set(seen))


def test_fold_sizes_follow_the_sklearn_rule() -> None:
    """With 10 rows and 3 folds the sizes are 4, 3, 3 — first folds take the remainder."""
    events, ends = _CANONICAL
    folds = PurgedKFold(events, ends, n_splits=3).folds()

    assert [len(fold.test) for fold in folds] == [4, 3, 3]


def test_first_and_last_folds_sit_at_the_sample_edges() -> None:
    """The first fold has no training data before it; the last has none after it.

    The embargo can only bite on the first fold's side, and it must not silently
    do nothing on the last: the last fold's training set is purged only.
    """
    events, ends = _CANONICAL
    folds = PurgedKFold(events, ends, n_splits=5, embargo=3 * DAY).folds()

    assert min(folds[0].train.tolist()) > max(folds[0].test.tolist())
    assert max(folds[-1].train.tolist()) < min(folds[-1].test.tolist())
    assert folds[-1].n_embargoed == 0


def test_indices_returned_are_positions_into_the_original_arrays() -> None:
    """Index arrays are ``numpy.intp`` positions, ascending, usable for fancy indexing."""
    events, ends = _CANONICAL
    features = np.arange(10) * 10
    for train, test in PurgedKFold(events, ends, n_splits=5, embargo=DAY).split(features):
        assert train.dtype == np.intp
        assert test.dtype == np.intp
        assert list(train) == sorted(train.tolist())
        assert list(test) == sorted(test.tolist())
        assert features[test].tolist() == [10 * i for i in test.tolist()]


# ---------------------------------------------------------------------------
# Embargo semantics
# ---------------------------------------------------------------------------


def test_zero_embargo_equals_no_embargo_equals_pure_purging() -> None:
    """``embargo=0``, ``embargo=None`` and ``embargo_fraction=0`` all coincide.

    This is what the strict ``<`` on the left edge of the embargo window buys:
    a zero-length window drops nothing rather than catching the boundary row.
    """
    events, ends = _CANONICAL
    explicit_zero = _index_sets(
        PurgedKFold(events, ends, n_splits=5, embargo=dt.timedelta(0)).folds()
    )
    omitted = _index_sets(PurgedKFold(events, ends, n_splits=5).folds())
    zero_fraction = _index_sets(PurgedKFold(events, ends, n_splits=5, embargo_fraction=0.0).folds())

    assert omitted == explicit_zero
    assert zero_fraction == explicit_zero


def test_increasing_embargo_monotonically_shrinks_training_sets() -> None:
    """Training sets are nested as the embargo grows, and strictly shrink here.

    On the canonical sample fold 1's training set is {6,7,8,9} with no embargo
    and loses one row per extra day: {7,8,9}, {8,9}, {9}.
    """
    events, ends = _CANONICAL
    trains = [
        [
            set(fold.train.tolist())
            for fold in PurgedKFold(events, ends, n_splits=5, embargo=d * DAY).folds()
        ]
        for d in (0, 1, 2, 3)
    ]

    for smaller, larger in itertools.pairwise(trains):
        for a, b in zip(smaller, larger, strict=True):
            assert b <= a
    assert [t[1] for t in trains] == [{6, 7, 8, 9}, {7, 8, 9}, {8, 9}, {9}]


def test_embargo_only_applies_after_the_test_fold() -> None:
    """A symmetric embargo would drop obs5; a forward-only embargo must keep it.

    Canonical sample, fold 4: test rows {8,9}, span [Jan 9, Jan 12], embargo
    3 days. obs5 runs Jan 6 - Jan 8, so it closes *before* the span opens and is
    clean. A symmetric embargo — one that also blanked
    ``[span start - embargo, span start)`` = [Jan 6, Jan 9) — would discard it
    for nothing, since information only flows forward and purging already
    removed everything reaching into the window.
    """
    events, ends = _CANONICAL
    fold = PurgedKFold(events, ends, n_splits=5, embargo=3 * DAY).folds()[4]

    assert (fold.test_span_start, fold.test_span_end) == (_jan(9), _jan(12))
    assert 5 in fold.train.tolist()
    assert set(fold.train.tolist()) == {0, 1, 2, 3, 4, 5}
    assert fold.n_embargoed == 0


def test_embargo_fraction_resolves_against_the_total_time_span() -> None:
    """A fraction is a fraction of elapsed time, not of the row count.

    Ten daily point-in-time observations span Jan 1 to Jan 10, i.e. 9 days, so
    ``embargo_fraction=1/9`` resolves to exactly one day — and produces the same
    folds as passing that day directly. Under a row-count reading (López de
    Prado's ``pctEmbargo``) 1/9 of 10 rows would be ~1 *row*, which is only the
    same thing because this sample happens to be evenly spaced.
    """
    events = [JAN1 + i * DAY for i in range(10)]
    ends = list(events)
    fractional = PurgedKFold(events, ends, n_splits=5, embargo_fraction=1 / 9)

    assert fractional.embargo == DAY
    assert fractional.embargo_fraction == pytest.approx(1 / 9)
    explicit = PurgedKFold(events, ends, n_splits=5, embargo=DAY)
    assert _index_sets(fractional.folds()) == _index_sets(explicit.folds())


def test_embargo_fraction_reports_the_value_it_applies() -> None:
    """The resolved timedelta is the applied timedelta — no hidden rounding drift.

    The fraction is rounded to whole microseconds once, at construction, and
    that rounded value both drives the split and is what :attr:`embargo`
    returns, so a caller logging the configuration (I2) logs the truth.
    """
    events = [JAN1, JAN1 + dt.timedelta(microseconds=7)]
    ends = list(events)
    cv = PurgedKFold(events, ends, n_splits=2, embargo_fraction=0.5)

    assert cv.embargo == dt.timedelta(microseconds=4)  # 3.5 us rounds to 4 us
    assert cv.folds()[0].embargo == cv.embargo


def test_embargo_fraction_of_a_zero_length_sample_is_zero() -> None:
    """When every observation shares one instant the total span is zero, so is the embargo."""
    events = [JAN1] * 4
    cv = PurgedKFold(events, list(events), n_splits=2, embargo_fraction=1.0)

    assert cv.embargo == dt.timedelta(0)


def test_absurdly_large_embargo_does_not_overflow_the_time_arithmetic() -> None:
    """``timedelta.max`` exceeds int64 nanoseconds; the horizon is clamped, not wrapped.

    Without the clamp the nanosecond horizon would not fit in ``int64`` and the
    comparison would raise ``OverflowError`` from numpy — a confusing failure
    for a caller who merely asked for a very long embargo. The honest failure is
    the one that says the training set is empty.
    """
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5, embargo=dt.timedelta.max)

    with pytest.raises(EmptyTrainingSetError, match="fold 0 has no training observations"):
        cv.folds()


# ---------------------------------------------------------------------------
# Input handling: unsorted, duplicated, mixed offsets
# ---------------------------------------------------------------------------


def test_unsorted_input_yields_the_same_folds_mapped_back_to_input_positions() -> None:
    """Shuffling the rows permutes the returned indices and nothing else.

    The splitter sorts internally; the guarantee is that fold membership is a
    property of the *observation*, not of where the caller happened to put it.
    """
    events, ends = _CANONICAL
    permutation = [7, 0, 3, 9, 1, 6, 2, 8, 5, 4]
    shuffled_events = [events[i] for i in permutation]
    shuffled_ends = [ends[i] for i in permutation]

    ordered = _index_sets(PurgedKFold(events, ends, n_splits=5, embargo=DAY).folds())
    shuffled = _index_sets(
        PurgedKFold(shuffled_events, shuffled_ends, n_splits=5, embargo=DAY).folds()
    )

    remapped = [
        ({permutation[i] for i in train}, {permutation[i] for i in test})
        for train, test in shuffled
    ]
    assert remapped == ordered


def test_duplicate_timestamps_split_deterministically_by_input_position() -> None:
    """Rows sharing a timestamp are ordered by their original position, always.

    Four observations on Jan 1 and four on Jan 2, all zero-length labels, with
    ``n_splits=2``. The tie-break is the input position, so the first fold is
    exactly the first four rows — reproducible run to run (invariant I2).
    """
    events = [_jan(1)] * 4 + [_jan(2)] * 4
    cv = PurgedKFold(events, list(events), n_splits=2, embargo=dt.timedelta(0))
    folds = cv.folds()

    assert set(folds[0].test.tolist()) == {0, 1, 2, 3}
    assert set(folds[1].test.tolist()) == {4, 5, 6, 7}
    # Every row sharing the test fold's instant overlaps its span and is purged,
    # so no same-day row leaks into training.
    assert set(folds[0].train.tolist()) == {4, 5, 6, 7}
    assert _index_sets(cv.folds()) == _index_sets(
        PurgedKFold(events, list(events), n_splits=2).folds()
    )


def test_duplicate_timestamps_never_leak_across_the_fold_boundary() -> None:
    """A boundary cutting through one timestamp group purges the rest of the group.

    Three cross-sections of three names each — Jan 1 (obs0-2), Jan 5 (obs3-5)
    and Jan 20 (obs6-8), every label one day long. Nine rows over two folds
    gives sizes 5 and 4, so the boundary falls *inside* the Jan 5 cross-section:
    obs3 and obs4 are in fold 0's test set while obs5, its same-day twin, is
    not. obs5 shares the test span exactly and must be purged — if it were not,
    a model would train on a row indistinguishable from one it is scored on.
    """
    events = [_jan(1)] * 3 + [_jan(5)] * 3 + [_jan(20)] * 3
    ends = [_jan(2)] * 3 + [_jan(6)] * 3 + [_jan(21)] * 3
    folds = PurgedKFold(events, ends, n_splits=2, embargo=dt.timedelta(0)).folds()

    assert set(folds[0].test.tolist()) == {0, 1, 2, 3, 4}
    assert set(folds[0].train.tolist()) == {6, 7, 8}
    assert folds[0].n_purged == 1  # obs5, the Jan 5 row left out of the test fold

    assert set(folds[1].test.tolist()) == {5, 6, 7, 8}
    assert set(folds[1].train.tolist()) == {0, 1, 2}
    assert folds[1].n_purged == 2  # obs3 and obs4, the Jan 5 rows now outside


def test_zero_length_labels_are_accepted() -> None:
    """``label_end_time == event_time`` is a legal point-in-time label."""
    events = [JAN1 + i * DAY for i in range(6)]
    folds = PurgedKFold(events, list(events), n_splits=3, embargo=dt.timedelta(0)).folds()

    assert set(folds[0].test.tolist()) == {0, 1}
    assert set(folds[0].train.tolist()) == {2, 3, 4, 5}


def test_non_utc_offsets_are_converted_not_reinterpreted() -> None:
    """Aware datetimes at a non-UTC offset give identical folds to their UTC equivalents.

    D-012 records an hours-scale silent temporal error caused by reinterpreting
    a datetime's wall clock instead of converting it. The same instants
    expressed at UTC-05:00 must therefore split identically.
    """
    eastern = dt.timezone(-5 * HOUR)
    events, ends = _CANONICAL
    shifted_events = [e.astimezone(eastern) for e in events]
    shifted_ends = [e.astimezone(eastern) for e in ends]

    assert _index_sets(
        PurgedKFold(shifted_events, shifted_ends, n_splits=5, embargo=DAY).folds()
    ) == (_index_sets(PurgedKFold(events, ends, n_splits=5, embargo=DAY).folds()))


def test_pandas_containers_are_accepted() -> None:
    """A ``DatetimeIndex`` or tz-aware ``Series`` works as directly as a list."""
    events, ends = _CANONICAL
    from_index = PurgedKFold(
        pd.DatetimeIndex(events), pd.DatetimeIndex(ends), n_splits=5, embargo=DAY
    )
    from_series = PurgedKFold(
        pd.Series(pd.DatetimeIndex(events)),
        pd.Series(pd.DatetimeIndex(ends)),
        n_splits=5,
        embargo=DAY,
    )
    from_list = PurgedKFold(events, ends, n_splits=5, embargo=DAY)

    assert _index_sets(from_index.folds()) == _index_sets(from_list.folds())
    assert _index_sets(from_series.folds()) == _index_sets(from_list.folds())


def test_generator_input_is_accepted() -> None:
    """Spans may arrive as any iterable, materialized once."""
    events, ends = _CANONICAL
    cv = PurgedKFold((e for e in events), (e for e in ends), n_splits=5, embargo=DAY)

    assert cv.n_observations == 10


# ---------------------------------------------------------------------------
# Empty training sets are an error, never a silent empty array
# ---------------------------------------------------------------------------


def test_purging_alone_can_empty_the_training_set_and_raises() -> None:
    """Four identical spans leave nothing to train on, and that is reported precisely."""
    events = [JAN1] * 4
    ends = [JAN1 + 5 * DAY] * 4
    cv = PurgedKFold(events, ends, n_splits=2, embargo=dt.timedelta(0))

    with pytest.raises(EmptyTrainingSetError) as excinfo:
        cv.folds()

    error = excinfo.value
    assert error.fold_index == 0
    assert error.n_observations == 4
    assert error.n_test == 2
    assert error.n_purged == 2
    assert error.n_embargoed == 0
    assert error.embargo == dt.timedelta(0)


def test_embargo_can_empty_the_training_set_and_raises() -> None:
    """An embargo wide enough to swallow the remaining rows is an error, not an empty split.

    Four point-in-time observations on Jan 1..4, ``n_splits=2``. Fold 0's span
    ends Jan 2; a 100-day embargo covers obs2 and obs3, leaving nothing.
    """
    events = [_jan(1), _jan(2), _jan(3), _jan(4)]
    cv = PurgedKFold(events, list(events), n_splits=2, embargo=100 * DAY)

    with pytest.raises(EmptyTrainingSetError) as excinfo:
        cv.folds()

    error = excinfo.value
    assert (error.n_purged, error.n_embargoed) == (0, 2)
    assert "4 total - 2 test - 0 purged - 2 embargoed = 0" in str(error)
    assert "datetime.timedelta(days=100)" in str(error)


def test_split_raises_before_yielding_anything() -> None:
    """``split`` fails at the call site, not partway through a training loop.

    Constructed so that fold 0 is *fine* and fold 1 is the one that empties::

        obs0 Jan1-Jan4   obs1 Jan2-Jan4   obs2 Jan3-Jan5   obs3 Jan4-Jan5
        obs4..obs9  Jan5..Jan10, each label one day long;  embargo 5 days

    Fold 0 (test {0,1}, span [Jan 1, Jan 4]) keeps obs9, whose Jan 10 start
    clears the Jan 9 embargo horizon. Fold 1 (test {2,3}, span [Jan 3, Jan 5])
    purges obs0, obs1, obs4 and embargoes obs5..obs9, leaving nothing.

    A lazily-evaluated ``split`` would hand back fold 0, let the caller fit a
    model on it, and only fail on the second iteration; worse, merely calling
    ``split()`` would do nothing at all. Asserting the raise happens here —
    with no iteration — is what pins the eager behaviour.
    """
    events = [_jan(i + 1) for i in range(10)]
    ends = [_jan(4), _jan(4), _jan(5), _jan(5), *[_jan(i + 2) for i in range(4, 10)]]
    cv = PurgedKFold(events, ends, n_splits=5, embargo=5 * DAY)

    with pytest.raises(EmptyTrainingSetError) as excinfo:
        cv.split()  # deliberately not iterated

    # fold_index == 1 is itself the proof that fold 0 was built successfully
    # first: folds are constructed in order and the first failure raises.
    assert excinfo.value.fold_index == 1


def test_empty_training_set_error_is_a_purged_cv_error() -> None:
    """The precise error is catchable through the package's base class."""
    assert issubclass(EmptyTrainingSetError, PurgedCVError)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_naive_datetimes_are_refused() -> None:
    """Naive input raises rather than being assumed to be UTC."""
    events = [dt.datetime(2024, 1, i + 1) for i in range(4)]  # noqa: DTZ001 — naive on purpose
    with pytest.raises(TypeError, match="event_time must be timezone-aware"):
        PurgedKFold(events, events, n_splits=2)


def test_naive_label_end_times_are_refused() -> None:
    """The same rule applies to the label end, named in the message."""
    events = [JAN1 + i * DAY for i in range(4)]
    naive = [dt.datetime(2024, 1, i + 1) for i in range(4)]  # noqa: DTZ001 — naive on purpose
    with pytest.raises(TypeError, match="label_end_time must be timezone-aware"):
        PurgedKFold(events, naive, n_splits=2)


def test_mixed_utc_offsets_are_refused() -> None:
    """A sequence that does not resolve to one timezone is refused, not coerced."""
    events = [JAN1, JAN1.astimezone(dt.timezone(2 * HOUR)) + DAY]
    with pytest.raises(TypeError, match="sharing one UTC offset"):
        PurgedKFold(events, events, n_splits=2)


def test_missing_times_are_refused() -> None:
    """``NaT`` has no defensible default for an event or label-end time.

    The ``cast`` is the point: static typing already forbids this, and the
    runtime check exists for callers who arrive through untyped data (a
    DataFrame column with a gap in it), where the type checker never looks.
    """
    events = cast("list[dt.datetime]", [JAN1, JAN1 + DAY, pd.NaT, JAN1 + 3 * DAY])
    with pytest.raises(ValueError, match="event_time contains missing values"):
        PurgedKFold(events, events, n_splits=2)


def test_non_datetime_input_is_refused() -> None:
    """Numbers are not instants; the error says so instead of silently succeeding."""
    numbers = cast("list[dt.datetime]", [1, 2, 3, 4])
    with pytest.raises(TypeError, match="event_time must be timezone-aware"):
        PurgedKFold(numbers, numbers, n_splits=2)


def test_mismatched_span_lengths_are_refused() -> None:
    """Every observation needs both of its times."""
    events = [JAN1 + i * DAY for i in range(4)]
    with pytest.raises(ValueError, match="must have equal length; got 4 and 3"):
        PurgedKFold(events, events[:3], n_splits=2)


def test_empty_sample_is_refused() -> None:
    """There is nothing to cross-validate."""
    with pytest.raises(ValueError, match="cannot cross-validate an empty sample"):
        PurgedKFold([], [], n_splits=2)


def test_label_ending_before_its_event_is_refused() -> None:
    """A backwards span is a data defect, reported with the offending position."""
    events = [JAN1 + i * DAY for i in range(4)]
    ends = list(events)
    ends[2] = events[2] - DAY
    with pytest.raises(ValueError, match=r"first at position 2"):
        PurgedKFold(events, ends, n_splits=2)


@pytest.mark.parametrize("n_splits", [-1, 0, 1])
def test_fewer_than_two_splits_is_refused(n_splits: int) -> None:
    """A single fold is not cross-validation."""
    events = [JAN1 + i * DAY for i in range(4)]
    with pytest.raises(ValueError, match=r"n_splits must be >= 2"):
        PurgedKFold(events, events, n_splits=n_splits)


def test_more_splits_than_observations_is_refused() -> None:
    """Some fold would be empty, which is not a fold."""
    events = [JAN1 + i * DAY for i in range(4)]
    with pytest.raises(ValueError, match=r"n_splits \(5\) cannot exceed"):
        PurgedKFold(events, events, n_splits=5)


def test_supplying_both_embargo_forms_is_refused() -> None:
    """Two sources of truth for one duration is how the wrong one gets applied."""
    events = [JAN1 + i * DAY for i in range(4)]
    with pytest.raises(ValueError, match="not both"):
        PurgedKFold(events, events, n_splits=2, embargo=DAY, embargo_fraction=0.1)


def test_negative_embargo_is_refused() -> None:
    """A negative embargo would mean embargoing backwards in time."""
    events = [JAN1 + i * DAY for i in range(4)]
    with pytest.raises(ValueError, match=r"embargo must be >= 0"):
        PurgedKFold(events, events, n_splits=2, embargo=-DAY)


@pytest.mark.parametrize("fraction", [-0.1, 1.5, float("nan"), float("inf")])
def test_embargo_fraction_outside_the_unit_interval_is_refused(fraction: float) -> None:
    """The fraction is of the sample's total span, so it lives in [0, 1]."""
    events = [JAN1 + i * DAY for i in range(4)]
    with pytest.raises(ValueError, match=r"finite fraction in \[0, 1\]"):
        PurgedKFold(events, events, n_splits=2, embargo_fraction=fraction)


def test_x_of_the_wrong_length_is_refused() -> None:
    """Spans that do not describe the supplied rows is a silent disaster if allowed."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5)
    with pytest.raises(ValueError, match="X has 7 rows but the splitter was built over 10"):
        cv.split(np.zeros(7))


def test_y_of_the_wrong_length_is_refused() -> None:
    """The label vector is length-checked the same way as the feature matrix."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5)
    with pytest.raises(ValueError, match="y has 3 rows"):
        cv.split(np.zeros(10), np.zeros(3))


def test_groups_are_refused_rather_than_ignored() -> None:
    """Silently ignoring ``groups`` would let a caller believe it took effect."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5)
    with pytest.raises(ValueError, match="does not use `groups`"):
        cv.split(np.zeros(10), None, np.zeros(10))


def test_unsized_x_is_accepted() -> None:
    """An ``X`` with no length (a lazy loader, say) is passed over rather than rejected."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5)

    assert len(list(cv.split(object()))) == 5


# ---------------------------------------------------------------------------
# scikit-learn shaped surface and reproducibility
# ---------------------------------------------------------------------------


def test_get_n_splits_matches_the_configuration() -> None:
    """``get_n_splits`` answers without consulting the data, as sklearn expects."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=4)

    assert cv.get_n_splits() == 4
    assert cv.get_n_splits(np.zeros(10), np.zeros(10), None) == 4


def test_split_yields_pairs_of_arrays() -> None:
    """The iteration protocol is the sklearn one: ``for train, test in cv.split(X)``."""
    events, ends = _CANONICAL
    splits = list(PurgedKFold(events, ends, n_splits=5, embargo=DAY).split(np.zeros(10)))

    assert len(splits) == 5
    assert all(isinstance(part, np.ndarray) for pair in splits for part in pair)


def test_repeated_calls_are_byte_identical(recwarn: pytest.WarningsRecorder) -> None:
    """No shuffling, no seed, no state: the same splitter always answers the same (I2)."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5, embargo=DAY)
    first = [(train.tobytes(), test.tobytes()) for train, test in cv.split()]
    second = [(train.tobytes(), test.tobytes()) for train, test in cv.split()]

    assert first == second
    assert not recwarn.list


def test_split_returns_fresh_arrays_each_call() -> None:
    """Mutating a returned index array cannot corrupt a later split."""
    events, ends = _CANONICAL
    cv = PurgedKFold(events, ends, n_splits=5, embargo=DAY)
    train, _ = next(iter(cv.split()))
    train[:] = -1

    fresh_train, _ = next(iter(cv.split()))
    assert (fresh_train >= 0).all()


def test_repr_states_the_full_configuration() -> None:
    """A run is regenerable from its configuration (I2), so the repr carries all of it."""
    events, ends = _CANONICAL
    text = repr(PurgedKFold(events, ends, n_splits=5, embargo=DAY))

    assert re.fullmatch(
        r"PurgedKFold\(n_splits=5, embargo=datetime\.timedelta\(days=1\), "
        r"embargo_fraction=None, n_observations=10\)",
        text,
    )
