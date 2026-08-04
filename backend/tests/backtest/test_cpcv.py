"""Exact-answer tests for Combinatorial Purged Cross-Validation (P10.2).

The counts asserted here are theorems, not observations: ``C(N, k)`` splits and
``C(N-1, k-1)`` paths. If the implementation ever disagrees with the arithmetic,
the CPCV distribution is being built from the wrong number of paths and every
confidence band drawn from it is the wrong width.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from backend.backtest.cpcv import (
    CombinatorialPurgedCV,
    EmptyTrainingSetError,
    PathDistribution,
    path_sharpe_ratios,
)

# (n_groups, n_test_groups, expected splits, expected paths)
_COUNT_TABLE = [
    (2, 1, 2, 1),
    (5, 1, 5, 1),
    (6, 2, 15, 5),
    (8, 2, 28, 7),
    (8, 3, 56, 21),
    (10, 2, 45, 9),
    (12, 6, 924, 462),
]


@pytest.mark.parametrize(("n_groups", "n_test", "n_splits", "n_paths"), _COUNT_TABLE)
def test_split_and_path_counts_match_the_closed_form(
    n_groups: int, n_test: int, n_splits: int, n_paths: int
) -> None:
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test)
    assert cv.n_splits == n_splits == comb(n_groups, n_test)
    # The path count is k * C(N, k) / N, which is the same as C(N-1, k-1):
    # every split contributes k tested groups, those k * C(N, k) tested-group
    # slots spread evenly over N groups, and each path consumes one per group.
    assert cv.n_paths == n_paths == comb(n_groups - 1, n_test - 1)
    assert cv.n_paths * n_groups == n_test * cv.n_splits


@pytest.mark.parametrize(("n_groups", "n_test", "n_splits", "n_paths"), _COUNT_TABLE)
def test_the_enumeration_actually_produces_those_counts(
    n_groups: int, n_test: int, n_splits: int, n_paths: int
) -> None:
    cv = CombinatorialPurgedCV(n_groups=n_groups, n_test_groups=n_test)
    combinations = cv.combinations()
    assert len(combinations) == n_splits
    assert len(set(combinations)) == n_splits
    assert all(len(c) == n_test for c in combinations)
    assert all(tuple(sorted(c)) == c for c in combinations)

    # Every group is tested in exactly C(N-1, k-1) splits, and that count is
    # also the number of paths — the equality is why the per-group forecasts
    # assemble into complete paths with none left over.
    for group in range(n_groups):
        appearances = sum(1 for c in combinations if group in c)
        assert appearances == n_paths == cv.splits_per_group


def test_combination_order_is_deterministic_across_calls() -> None:
    cv = CombinatorialPurgedCV(n_groups=7, n_test_groups=3)
    assert cv.combinations() == cv.combinations()
    assert cv.combinations()[0] == (0, 1, 2)
    assert cv.combinations()[-1] == (4, 5, 6)


def test_group_slices_partition_the_sample_with_sizes_differing_by_at_most_one() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    bounds = cv.group_slices(20)
    assert bounds == ((0, 4), (4, 8), (8, 11), (11, 14), (14, 17), (17, 20))
    sizes = [stop - start for start, stop in bounds]
    assert sum(sizes) == 20
    assert max(sizes) - min(sizes) <= 1


def test_a_sample_shorter_than_the_group_count_is_refused() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    with pytest.raises(ValueError, match="smaller than n_groups"):
        cv.group_slices(5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_groups": 1, "n_test_groups": 1}, "n_groups must be at least 2"),
        ({"n_groups": 5, "n_test_groups": 0}, "1 <= k < n_groups"),
        ({"n_groups": 5, "n_test_groups": 5}, "1 <= k < n_groups"),
        ({"n_groups": 5, "n_test_groups": 2, "embargo": -1.0}, "embargo must be finite"),
        ({"n_groups": 5, "n_test_groups": 2, "embargo": float("nan")}, "embargo must be finite"),
    ],
)
def test_malformed_configurations_are_refused(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CombinatorialPurgedCV(**kwargs)  # type: ignore[arg-type]


def _point_events(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return event times for ``n`` observations whose labels resolve instantly."""
    starts = np.arange(n, dtype=np.float64)
    return starts, starts.copy()


def test_with_instantaneous_labels_and_no_embargo_nothing_is_purged() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    starts, ends = _point_events(8)
    splits = cv.split(starts, ends)

    assert len(splits) == 4
    for split in splits:
        assert split.n_purged == 0
        assert split.n_embargoed == 0
        assert sorted(split.train_indices.tolist() + split.test_indices.tolist()) == list(range(8))


def test_purging_removes_exactly_the_labels_that_overlap_the_test_block() -> None:
    # 12 observations at times 0..11, each label resolving 3 periods later.
    # Groups of 3: [0,1,2] [3,4,5] [6,7,8] [9,10,11].
    # Test group 1 spans starts 3..5 and its labels end at 8, so the test block
    # interval is [3, 8].
    #   - observation 0 has [0, 3]: touches 3          -> purged
    #   - observation 1 has [1, 4], observation 2 [2, 5] -> purged
    #   - observations 6, 7, 8 start at 6, 7, 8 <= 8   -> purged
    #   - observation 9 starts at 9 > 8                -> kept
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    starts = np.arange(12, dtype=np.float64)
    ends = starts + 3.0
    splits = cv.split(starts, ends)

    split = splits[1]
    assert split.test_groups == (1,)
    assert split.test_indices.tolist() == [3, 4, 5]
    assert split.train_indices.tolist() == [9, 10, 11]
    assert split.n_purged == 6
    assert split.n_embargoed == 0


def test_the_embargo_applies_forward_only() -> None:
    # Instantaneous labels so purging removes nothing; only the embargo acts.
    # Test group 1 covers positions 3..5, so its block ends at time 5 and an
    # embargo of 2 removes observations starting in (5, 7] — positions 6 and 7.
    # Positions 0..2, which precede the block, must survive: the leak an embargo
    # addresses runs forward in time, and purging backwards would throw away
    # legitimate history.
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1, embargo=2.0)
    starts, ends = _point_events(12)
    splits = cv.split(starts, ends)

    split = splits[1]
    assert split.test_indices.tolist() == [3, 4, 5]
    assert split.train_indices.tolist() == [0, 1, 2, 8, 9, 10, 11]
    assert split.n_purged == 0
    assert split.n_embargoed == 2


def test_purged_and_embargoed_counts_never_double_count() -> None:
    # Labels of length 2 and an embargo of 2. Test group 1 covers positions 3..5,
    # block interval [3, 7]. Purging removes 1, 2 (labels reach 3 and 4) and
    # 6, 7 (start at or before 7). The embargo then covers (7, 9] = positions
    # 8, 9, neither of which was purged, so it adds exactly 2.
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1, embargo=2.0)
    starts = np.arange(12, dtype=np.float64)
    ends = starts + 2.0
    split = cv.split(starts, ends)[1]

    assert split.test_indices.tolist() == [3, 4, 5]
    assert split.train_indices.tolist() == [0, 10, 11]
    assert split.n_purged == 4
    assert split.n_embargoed == 2
    # 12 observations = 3 test + 4 purged + 2 embargoed + 3 kept.
    assert 3 + split.n_purged + split.n_embargoed + len(split.train_indices) == 12


def test_disjoint_test_groups_do_not_purge_the_training_data_between_them() -> None:
    # Groups of 3 over 15 observations, instantaneous labels, no embargo.
    # Testing groups 0 and 4 leaves group 1..3 (positions 3..11) untouched:
    # treating {0, 4} as one span from time 0 to time 14 would purge the whole
    # sample for no reason.
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
    starts, ends = _point_events(15)
    splits = cv.split(starts, ends)

    split = next(s for s in splits if s.test_groups == (0, 4))
    assert split.test_indices.tolist() == [0, 1, 2, 12, 13, 14]
    assert split.train_indices.tolist() == [3, 4, 5, 6, 7, 8, 9, 10, 11]
    assert split.n_purged == 0


def test_adjacent_test_groups_are_treated_as_one_block() -> None:
    # Groups 1 and 2 are adjacent, so their embargo starts after group 2 ends
    # (time 8), not after group 1 ends. With an embargo of 2 that removes
    # positions 9 and 10 only.
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=2, embargo=2.0)
    starts, ends = _point_events(12)
    split = next(s for s in cv.split(starts, ends) if s.test_groups == (1, 2))

    assert split.test_indices.tolist() == [3, 4, 5, 6, 7, 8]
    assert split.train_indices.tolist() == [0, 1, 2, 11]
    assert split.n_embargoed == 2


def test_test_indices_are_exactly_the_union_of_the_selected_groups() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=1.0)
    starts = np.arange(30, dtype=np.float64)
    splits = cv.split(starts, starts + 2.0)
    bounds = splits.group_slices

    for split in splits:
        expected: list[int] = []
        for group in split.test_groups:
            start, stop = bounds[group]
            expected.extend(range(start, stop))
        assert split.test_indices.tolist() == sorted(expected)


def test_training_and_test_sets_never_intersect() -> None:
    cv = CombinatorialPurgedCV(n_groups=7, n_test_groups=3, embargo=1.5)
    starts = np.arange(70, dtype=np.float64)
    for split in cv.split(starts, starts + 4.0):
        assert set(split.train_indices.tolist()).isdisjoint(split.test_indices.tolist())


def test_an_embargo_that_swallows_the_sample_raises_instead_of_training_on_nothing() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1, embargo=1_000.0)
    starts, ends = _point_events(12)
    with pytest.raises(EmptyTrainingSetError, match="empty training set"):
        cv.split(starts, ends)


@pytest.mark.parametrize(
    ("starts", "ends", "match"),
    [
        ([0.0, 1.0, 2.0], [0.0, 1.0], "same length"),
        ([0.0, 2.0, 1.0, 3.0], [0.0, 2.0, 1.0, 3.0], "non-decreasing"),
        ([0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 2.0, 3.0], "at least its event_start"),
    ],
)
def test_malformed_event_times_are_refused(
    starts: list[float], ends: list[float], match: str
) -> None:
    cv = CombinatorialPurgedCV(n_groups=2, n_test_groups=1)
    with pytest.raises(ValueError, match=match):
        cv.split(starts, ends)


def test_the_path_map_lists_each_group_once_per_path() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    starts, ends = _point_events(24)
    splits = cv.split(starts, ends)
    mapping = splits.path_split_map()

    assert mapping.shape == (cv.n_paths, cv.n_groups) == (5, 6)
    for group in range(cv.n_groups):
        supplying = mapping[:, group].tolist()
        # Each of the 5 paths draws group `group` from a different split, and
        # those are precisely the splits that tested it.
        assert len(set(supplying)) == cv.n_paths
        for split_id in supplying:
            assert group in splits[split_id].test_groups


def test_assembled_paths_cover_the_whole_sample_once_per_path() -> None:
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    starts, ends = _point_events(24)
    splits = cv.split(starts, ends)

    # Let each split "forecast" the observation's own index. Every path must
    # then reproduce arange(24) exactly: any gap, duplicate or misalignment in
    # the assembly shows up immediately.
    per_split = [split.test_indices.astype(np.float64) for split in splits]
    paths = splits.assemble_paths(per_split)

    assert paths.shape == (cv.n_paths, 24)
    for row in paths:
        assert row.tolist() == list(range(24))


def test_assembly_draws_each_group_from_the_split_the_map_names() -> None:
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
    starts, ends = _point_events(20)
    splits = cv.split(starts, ends)

    # Each split now stamps its own id on every observation it forecasts, so an
    # assembled path reads back as the split provenance of each group.
    per_split = [np.full(split.test_indices.size, float(split.split_id)) for split in splits]
    paths = splits.assemble_paths(per_split)
    mapping = splits.path_split_map()

    for path in range(cv.n_paths):
        for group, (start, stop) in enumerate(splits.group_slices):
            assert set(paths[path, start:stop].tolist()) == {float(mapping[path, group])}


def test_assembly_refuses_misaligned_input() -> None:
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    starts, ends = _point_events(12)
    splits = cv.split(starts, ends)
    good = [split.test_indices.astype(np.float64) for split in splits]

    with pytest.raises(ValueError, match="one entry per split"):
        splits.assemble_paths(good[:-1])

    truncated = [*good[:-1], good[-1][:-1]]
    with pytest.raises(ValueError, match="test observations"):
        splits.assemble_paths(truncated)


def test_path_sharpe_ratios_produce_one_value_per_path() -> None:
    rng = np.random.default_rng(3)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
    starts, ends = _point_events(120)
    splits = cv.split(starts, ends)
    per_split = [rng.normal(0.0, 0.01, size=split.test_indices.size) for split in splits]

    distribution = path_sharpe_ratios(splits.assemble_paths(per_split))
    assert len(distribution) == cv.n_paths == 5
    assert distribution.values.shape == (5,)


def test_path_sharpe_ratios_reject_a_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        path_sharpe_ratios(np.zeros(10))


def test_path_sharpe_ratios_reject_an_empty_distribution() -> None:
    # A distribution with no paths in it is not a distribution, and summarizing
    # one would produce a NaN mean rather than an error.
    with pytest.raises(ValueError, match="at least one path"):
        path_sharpe_ratios(np.zeros((0, 12)))


def test_assembly_refuses_per_split_values_that_leave_an_observation_uncovered() -> None:
    # The final finiteness check in assemble_paths is an internal invariant: if
    # the path map ever stopped covering every group, some observation would
    # keep the NaN it was initialized with, and a NaN return silently poisons
    # every metric downstream of it. Forcing a NaN into the input proves the
    # guard is wired up rather than merely written down.
    cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
    starts, ends = _point_events(12)
    splits = cv.split(starts, ends)
    per_split = [split.test_indices.astype(np.float64) for split in splits]
    per_split[2] = per_split[2].copy()
    per_split[2][0] = np.nan

    with pytest.raises(ValueError, match="must be finite"):
        splits.assemble_paths(per_split)


def test_the_path_distribution_reports_an_interval_alongside_every_summary() -> None:
    distribution = PathDistribution(values=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    assert distribution.mean == pytest.approx(0.3)
    assert distribution.median == pytest.approx(0.3)
    assert distribution.minimum == pytest.approx(0.1)
    assert distribution.maximum == pytest.approx(0.5)
    assert distribution.quantile(0.5) == pytest.approx(0.3)

    lower, upper = distribution.confidence_interval(0.5)
    assert (lower, upper) == pytest.approx((0.2, 0.4))

    summary = distribution.describe()
    assert summary["n_paths"] == 5.0
    assert "ci_lower" in summary
    assert "ci_upper" in summary
    assert summary["std"] == pytest.approx(float(np.std([0.1, 0.2, 0.3, 0.4, 0.5], ddof=1)))


def test_a_single_path_reports_no_spread_rather_than_a_spread_of_zero() -> None:
    distribution = PathDistribution(values=np.array([0.25]))
    with pytest.raises(ValueError, match="at least 2 paths"):
        _ = distribution.std
    # describe() omits the key instead of claiming certainty the data lacks.
    assert "std" not in distribution.describe()


@pytest.mark.parametrize(("level", "match"), [(0.0, r"\(0, 1\)"), (1.0, r"\(0, 1\)")])
def test_confidence_interval_rejects_degenerate_levels(level: float, match: str) -> None:
    distribution = PathDistribution(values=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match=match):
        distribution.confidence_interval(level)


def test_quantile_rejects_a_level_outside_the_unit_interval() -> None:
    distribution = PathDistribution(values=np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        distribution.quantile(1.5)
