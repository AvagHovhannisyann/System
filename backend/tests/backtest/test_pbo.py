"""Tests for the Probability of Backtest Overfitting via CSCV (P10.4).

Three kinds of check, in increasing order of how much they would hurt if they
failed:

1. **Counts.** ``C(S, S/2)`` combinations, exactly, asserted against
   :func:`math.comb`. The logit distribution has one entry per combination.
2. **Constructed extremes.** A ranking that generalizes perfectly gives
   ``PBO = 0`` with every logit equal to ``log(N)``; a ranking that inverts
   perfectly gives ``PBO = 1`` with every logit equal to ``log(1/2)``. Both are
   arithmetic, not approximations, and both are derived in the test bodies.
3. **Equivalence of the two code paths.** The default metric evaluates all
   combinations from partition sufficient statistics — two matrix products
   instead of ``C(S, S/2)`` passes over the data. That is exactly the sort of
   optimization that silently disagrees with the thing it replaces, so it is
   checked against a naive per-combination Sharpe ratio computed with
   :func:`backend.backtest.metrics.sharpe_ratio`.

The behaviour on pure noise — ``PBO ~ 0.5``, the gate the directive calls the
critical one — lives in ``test_synthetic_truth.py`` with the rest of the
noise-versus-signal harness.

Source: Bailey, Borwein, López de Prado & Zhu (2017), "The Probability of
Backtest Overfitting", *Journal of Computational Finance* 20(4), 39-69.
"""

from __future__ import annotations

import itertools
import math
from math import comb

import numpy as np
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.backtest.metrics import FloatArray, sharpe_ratio
from backend.backtest.pbo import probability_of_backtest_overfitting


def _naive_column_sharpes(block: FloatArray) -> FloatArray:
    """Sharpe ratio of each column, computed one column at a time."""
    return np.array([sharpe_ratio(block[:, j]) for j in range(block.shape[1])], dtype=np.float64)


def _column_means(block: FloatArray) -> FloatArray:
    """Mean of each column — a performance metric simple enough to reason about."""
    return np.asarray(block.mean(axis=0), dtype=np.float64)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_partitions", "expected"), [(2, 2), (4, 6), (6, 20), (8, 70), (10, 252), (16, 12870)]
)
def test_the_combination_count_is_exactly_c_s_half_s(n_partitions: int, expected: int) -> None:
    matrix = np.random.default_rng(n_partitions).normal(0.0, 0.01, size=(120, 4))
    result = probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)

    assert result.n_combinations == expected == comb(n_partitions, n_partitions // 2)
    assert result.logits.shape == (expected,)
    assert result.relative_ranks.shape == (expected,)
    assert result.best_configuration.shape == (expected,)
    assert result.in_sample_performance_of_best.shape == (expected,)
    assert result.out_of_sample_performance_of_best.shape == (expected,)
    assert result.n_partitions == n_partitions
    assert result.n_configurations == 4


def test_the_scalar_is_the_fraction_of_non_positive_logits() -> None:
    matrix = np.random.default_rng(17).normal(0.0, 0.01, size=(400, 6))
    result = probability_of_backtest_overfitting(matrix, n_partitions=8)
    assert result.pbo == pytest.approx(
        float(np.count_nonzero(result.logits <= 0.0)) / result.n_combinations
    )


def test_relative_ranks_stay_strictly_inside_the_unit_interval() -> None:
    # The (N + 1) denominator is what keeps the logit finite at the extremes: a
    # rank of N would otherwise give w = 1 and log(w / (1 - w)) = +infinity.
    matrix = np.random.default_rng(23).normal(0.0, 0.01, size=(300, 5))
    result = probability_of_backtest_overfitting(matrix, n_partitions=6)
    assert np.all(result.relative_ranks > 0.0)
    assert np.all(result.relative_ranks < 1.0)
    assert np.all(np.isfinite(result.logits))


# ---------------------------------------------------------------------------
# Constructed extremes
# ---------------------------------------------------------------------------


def test_a_ranking_that_generalizes_perfectly_gives_pbo_zero() -> None:
    # Four configurations with constant, strictly ordered edges (0, 1%, 2%, 3%
    # per period) and a little common noise. Configuration 3 is best in every
    # subsample, so it wins in-sample and is still best out-of-sample in every
    # one of the C(4, 2) = 6 combinations.
    #   rank of the winner = N = 4, w = 4 / (4 + 1) = 0.8
    #   logit = log(0.8 / 0.2) = log(4) = 1.3862943611198906
    edges = np.arange(4, dtype=np.float64) * 0.01
    matrix = np.tile(edges, (40, 1)) + np.random.default_rng(1).normal(0.0, 0.0005, size=(40, 4))

    result = probability_of_backtest_overfitting(matrix, n_partitions=4)
    assert result.pbo == 0.0
    assert np.all(result.best_configuration == 3)
    assert result.relative_ranks == pytest.approx(np.full(6, 0.8))
    assert result.logits == pytest.approx(np.full(6, math.log(4.0)))


def test_a_ranking_that_inverts_perfectly_gives_pbo_one() -> None:
    # Two configurations, four partitions of ten rows, performance = column mean.
    #
    # Configuration A takes the per-partition values (1/4, 1/16, -1/8, -3/16) —
    # all exactly representable in binary, and summing to exactly zero.
    # Configuration B is A negated, so B's mean is always minus A's mean.
    #
    # Because A's total is exactly zero, A's out-of-sample sum is exactly minus
    # its in-sample sum, and the two halves hold the same number of rows, so
    #     OOS_mean(A) = -IS_mean(A)      and      OOS_mean(B) = -IS_mean(B).
    # Whichever configuration wins in-sample therefore loses out-of-sample, in
    # every combination. No two-partition subset sums to zero (the pairwise sums
    # are ±5/8, ±1/4, ±1/8), so no combination ties.
    #   rank of the winner = 1, w = 1 / (2 + 1) = 1/3
    #   logit = log((1/3) / (2/3)) = log(1/2) = -0.6931471805599453
    per_partition = [0.25, 0.0625, -0.125, -0.1875]
    column_a = np.repeat(per_partition, 10)
    assert column_a.sum() == 0.0  # exact in binary floating point
    matrix = np.column_stack([column_a, -column_a])

    result = probability_of_backtest_overfitting(matrix, n_partitions=4, performance=_column_means)
    assert result.n_combinations == 6
    assert result.pbo == 1.0
    assert result.relative_ranks == pytest.approx(np.full(6, 1.0 / 3.0))
    assert result.logits == pytest.approx(np.full(6, math.log(0.5)))
    # The degradation scatter of Bailey et al. §4: every in-sample winner has a
    # strictly negative out-of-sample performance.
    assert np.all(result.in_sample_performance_of_best > 0.0)
    assert np.all(result.out_of_sample_performance_of_best < 0.0)


def test_identical_configurations_land_at_the_middle_of_the_rank_distribution() -> None:
    # Every column identical: selection is meaningless, every out-of-sample value
    # ties, and the average rank of the winner is (N + 1) / 2 out of N. With
    # N = 4 that is 2.5, so w = 2.5 / 5 = 0.5 and the logit is exactly 0 — which
    # counts as "did not beat the median", so PBO = 1.
    column = np.random.default_rng(31).normal(0.0, 0.01, size=200)
    matrix = np.tile(column[:, None], (1, 4))
    result = probability_of_backtest_overfitting(matrix, n_partitions=6)

    assert result.relative_ranks == pytest.approx(np.full(result.n_combinations, 0.5))
    assert result.logits == pytest.approx(np.zeros(result.n_combinations), abs=1e-15)
    assert result.pbo == 1.0


# ---------------------------------------------------------------------------
# The fast path must equal the slow one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_rows", "n_configurations", "n_partitions", "seed"),
    [(60, 4, 6, 0), (97, 3, 4, 1), (240, 8, 8, 2), (53, 5, 4, 3)],
)
def test_the_sufficient_statistic_sharpe_equals_a_naive_per_combination_sharpe(
    n_rows: int, n_configurations: int, n_partitions: int, seed: int
) -> None:
    # The default metric never materializes a combination's rows: it accumulates
    # each partition's count, sum and sum of squares once and combines them with
    # two matrix products. Sharpe ratios of a union of partitions depend on the
    # rows only through those three numbers, so the shortcut is exact — but only
    # if the algebra is right, which is what this checks. n_rows is deliberately
    # not always divisible by n_partitions, so the partitions differ in size.
    matrix = np.random.default_rng(seed).normal(0.0005, 0.01, size=(n_rows, n_configurations))

    fast = probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)
    slow = probability_of_backtest_overfitting(
        matrix, n_partitions=n_partitions, performance=_naive_column_sharpes
    )

    assert fast.pbo == slow.pbo
    assert np.array_equal(fast.best_configuration, slow.best_configuration)
    assert fast.logits == pytest.approx(slow.logits, rel=1e-9, abs=1e-12)
    assert fast.in_sample_performance_of_best == pytest.approx(
        slow.in_sample_performance_of_best, rel=1e-9, abs=1e-12
    )
    assert fast.out_of_sample_performance_of_best == pytest.approx(
        slow.out_of_sample_performance_of_best, rel=1e-9, abs=1e-12
    )


def test_the_in_sample_winner_really_is_the_best_column_of_its_selection() -> None:
    # Rebuild the in-sample selections by hand and confirm the reported winner is
    # the argmax of an independently computed Sharpe ratio over exactly those
    # rows, and that the reported out-of-sample value belongs to the complement.
    n_rows, n_partitions = 48, 4
    matrix = np.random.default_rng(9).normal(0.0002, 0.01, size=(n_rows, 5))
    result = probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)

    size = n_rows // n_partitions
    partitions = [np.arange(index * size, (index + 1) * size) for index in range(n_partitions)]
    combinations = list(itertools.combinations(range(n_partitions), n_partitions // 2))
    assert len(combinations) == result.n_combinations

    for index, chosen in enumerate(combinations):
        inside = np.concatenate([partitions[p] for p in chosen])
        outside = np.concatenate([partitions[p] for p in range(n_partitions) if p not in chosen])
        in_sample = _naive_column_sharpes(matrix[inside])
        out_of_sample = _naive_column_sharpes(matrix[outside])

        winner = int(np.argmax(in_sample))
        assert int(result.best_configuration[index]) == winner
        assert result.in_sample_performance_of_best[index] == pytest.approx(in_sample[winner])
        assert result.out_of_sample_performance_of_best[index] == pytest.approx(
            out_of_sample[winner]
        )
        # And the logit is the winner's out-of-sample standing, one-based
        # ascending over N candidates, mapped through log(w / (1 - w)).
        rank = 1 + int(np.count_nonzero(out_of_sample < out_of_sample[winner]))
        relative = rank / (matrix.shape[1] + 1.0)
        assert result.relative_ranks[index] == pytest.approx(relative)
        assert result.logits[index] == pytest.approx(math.log(relative / (1.0 - relative)))


def test_a_custom_metric_is_used_in_place_of_the_default_sharpe_ratio() -> None:
    # Column 0 has the best mean and column 2 the best Sharpe ratio, so the two
    # metrics must disagree about the winner. If the custom function were
    # ignored, this would not be visible.
    rng = np.random.default_rng(41)
    steady = rng.normal(0.001, 0.001, size=200)  # small mean, tiny variance
    lumpy = rng.normal(0.004, 0.05, size=200)  # large mean, large variance
    matrix = np.column_stack([lumpy, rng.normal(0.0, 0.01, size=200), steady])

    by_mean = probability_of_backtest_overfitting(matrix, n_partitions=4, performance=_column_means)
    by_sharpe = probability_of_backtest_overfitting(matrix, n_partitions=4)
    assert np.all(by_mean.best_configuration == 0)
    assert np.all(by_sharpe.best_configuration == 2)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_partitions", "match"),
    [
        (5, "even number of at least 2"),
        (0, "even number of at least 2"),
        (-2, "even number of at least 2"),
        (400, "exceeds the 100 observations"),
    ],
)
def test_malformed_partition_counts_are_refused(n_partitions: int, match: str) -> None:
    matrix = np.random.default_rng(0).normal(0.0, 0.01, size=(100, 3))
    with pytest.raises(ValueError, match=match):
        probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)


def test_a_single_configuration_cannot_be_ranked() -> None:
    matrix = np.random.default_rng(0).normal(0.0, 0.01, size=(100, 1))
    with pytest.raises(ValueError, match="at least 2 configurations"):
        probability_of_backtest_overfitting(matrix)


def test_a_non_matrix_or_non_finite_input_is_refused() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        probability_of_backtest_overfitting(np.zeros(10))
    bad = np.zeros((40, 2))
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        probability_of_backtest_overfitting(bad, n_partitions=4)


def test_a_constant_configuration_cannot_take_part() -> None:
    # A column with zero variance has no Sharpe ratio in any selection, so the
    # comparison is undefined rather than infinite.
    matrix = np.column_stack(
        [np.random.default_rng(0).normal(0.0, 0.01, size=40), np.full(40, 0.001)]
    )
    with pytest.raises(ValueError, match="non-positive return variance"):
        probability_of_backtest_overfitting(matrix, n_partitions=4)


def test_too_few_rows_per_half_for_the_default_metric_is_refused() -> None:
    # ddof = 1 needs more than one row in each half. Three rows split into two
    # partitions gives sizes 2 and 1, so the smaller half holds a single row and
    # its standard deviation — hence its Sharpe ratio — is undefined.
    matrix = np.random.default_rng(0).normal(0.0, 0.01, size=(3, 3))
    with pytest.raises(ValueError, match="more than ddof"):
        probability_of_backtest_overfitting(matrix, n_partitions=2)


def test_a_custom_metric_returning_the_wrong_shape_is_refused() -> None:
    matrix = np.random.default_rng(0).normal(0.0, 0.01, size=(40, 3))
    with pytest.raises(ValueError, match="one value per configuration"):
        probability_of_backtest_overfitting(
            matrix, n_partitions=4, performance=lambda block: block.mean(axis=0)[:2]
        )
    with pytest.raises(ValueError, match="non-finite value"):
        probability_of_backtest_overfitting(
            matrix, n_partitions=4, performance=lambda block: np.full(block.shape[1], np.nan)
        )


# ---------------------------------------------------------------------------
# Properties — randomized structure, not one fixed shape
# ---------------------------------------------------------------------------


@st.composite
def _performance_matrices(draw: st.DrawFn) -> tuple[FloatArray, int]:
    """Draw a performance matrix together with a partition count that fits it."""
    n_configurations = draw(st.integers(min_value=2, max_value=6))
    n_partitions = draw(st.sampled_from([2, 4, 6, 8]))
    n_rows = draw(st.integers(min_value=n_partitions * 2, max_value=n_partitions * 12))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    # Columns differ in both drift and volatility, so the ranking is not
    # degenerate and the partitions are not interchangeable.
    drifts = rng.uniform(-0.002, 0.002, size=n_configurations)
    volatilities = rng.uniform(0.005, 0.03, size=n_configurations)
    matrix = rng.normal(drifts, volatilities, size=(n_rows, n_configurations))
    return matrix, n_partitions


@given(_performance_matrices())
@hypothesis_settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.data_too_large]
)
def test_pbo_is_invariant_to_the_order_of_the_configurations(
    problem: tuple[FloatArray, int],
) -> None:
    """Relabelling the candidates cannot change how often selection fails."""
    matrix, n_partitions = problem
    rng = np.random.default_rng(0)
    order = rng.permutation(matrix.shape[1])

    original = probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)
    permuted = probability_of_backtest_overfitting(matrix[:, order], n_partitions=n_partitions)

    assert permuted.pbo == pytest.approx(original.pbo)
    assert np.sort(permuted.logits) == pytest.approx(np.sort(original.logits))


@given(_performance_matrices())
@hypothesis_settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.data_too_large]
)
def test_the_reported_distribution_is_internally_consistent(
    problem: tuple[FloatArray, int],
) -> None:
    """Scalar, ranks and logits must always describe the same thing."""
    matrix, n_partitions = problem
    result = probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)

    assert result.n_combinations == comb(n_partitions, n_partitions // 2)
    assert result.logits.shape == (result.n_combinations,)
    assert np.all(np.isfinite(result.logits))
    assert np.all(result.relative_ranks > 0.0)
    assert np.all(result.relative_ranks < 1.0)
    assert 0.0 <= result.pbo <= 1.0
    assert result.pbo == pytest.approx(float(np.mean(result.logits <= 0.0)))
    # A logit is non-positive exactly when the relative rank is at or below 0.5.
    assert np.array_equal(result.logits <= 0.0, result.relative_ranks <= 0.5)
    assert np.all(result.best_configuration >= 0)
    assert np.all(result.best_configuration < matrix.shape[1])


@given(_performance_matrices())
@hypothesis_settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.data_too_large]
)
def test_pbo_is_invariant_to_a_positive_rescaling_of_every_return(
    problem: tuple[FloatArray, int],
) -> None:
    """The Sharpe ratio is scale-free, so doubling every return changes nothing."""
    matrix, n_partitions = problem
    original = probability_of_backtest_overfitting(matrix, n_partitions=n_partitions)
    scaled = probability_of_backtest_overfitting(matrix * 7.5, n_partitions=n_partitions)
    assert scaled.pbo == original.pbo
    assert scaled.logits == pytest.approx(original.logits, rel=1e-9, abs=1e-12)
