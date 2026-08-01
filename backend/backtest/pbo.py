"""Probability of Backtest Overfitting via CSCV — P10.4.

What the number means
---------------------

The Deflated Sharpe Ratio asks "is this one number too good given how hard I
searched". PBO asks a different and complementary question: **does the act of
selecting the best configuration in-sample generalize at all?** It is estimated
by Combinatorially Symmetric Cross-Validation (CSCV):

1. Build a performance matrix ``M`` of shape ``(T, N)``: ``T`` per-period
   observations for each of ``N`` candidate configurations.
2. Cut the ``T`` rows into ``S`` disjoint contiguous partitions (``S`` even).
3. For each of the ``C(S, S/2)`` ways of choosing half the partitions, call the
   chosen half in-sample (IS) and the complement out-of-sample (OOS).
4. Pick ``n*``, the configuration with the best IS performance. Find its rank
   ``r`` among the ``N`` OOS performances (ascending, so ``N`` is best) and form
   the relative rank ``w = r / (N + 1)``.
5. The logit ``lambda = log(w / (1 - w))`` is positive when the IS-best beats
   the OOS median and negative when it does not.

``PBO = P(lambda <= 0)``, estimated as the fraction of combinations in which the
in-sample winner landed at or below the out-of-sample median.

Reading the result
------------------

* ``PBO ~ 0.5`` means selection carries no information: the configuration that
  won in-sample is a coin flip out-of-sample. This is what pure noise gives, and
  it is the value this implementation must produce on random data or everything
  downstream is fiction (directive §5, gate G10).
* ``PBO ~ 0`` means the in-sample winner reliably stays a winner out-of-sample.
* ``PBO ~ 1`` is worse than useless: the in-sample winner is reliably an
  out-of-sample loser, the signature of fitting a pattern that inverts.

Note carefully what PBO is *not*. It measures whether **selection among the
candidates** generalizes, not whether any of them has alpha. ``N`` configurations
that all share the same genuine edge, differing only by noise, will produce
``PBO ~ 0.5`` — correctly, because choosing between them is indeed a coin flip.
Low PBO requires the candidates to differ in true skill.

The ``(N + 1)`` denominator in the relative rank is not cosmetic: it keeps ``w``
strictly inside ``(0, 1)`` so the logit is always finite, whatever happens at the
extremes.

Source: Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017).
"The Probability of Backtest Overfitting." *Journal of Computational Finance*
20(4), 39-69. Also López de Prado, M. (2018), *Advances in Financial Machine
Learning*, chapter 11.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.backtest.metrics import FloatArray, IntArray

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "PBOResult",
    "PerformanceFunction",
    "probability_of_backtest_overfitting",
]

type PerformanceFunction = Callable[[FloatArray], FloatArray]
"""A custom performance metric: ``(rows, n_configurations) -> (n_configurations,)``."""


@dataclass(frozen=True, slots=True, eq=False)
class PBOResult:
    """The full CSCV output: the scalar and the distribution behind it.

    Attributes:
        pbo: the Probability of Backtest Overfitting — the fraction of
            combinations whose in-sample winner ranked at or below the
            out-of-sample median. A probability in ``[0, 1]``.
        logits: ``lambda_c`` for every combination, in combination order. This
            is the distribution the scalar summarizes; directive §10 forbids
            reporting the point estimate on its own, and a PBO of 0.4 built from
            logits piled at zero says something very different from a PBO of 0.4
            built from a wide bimodal spread.
        relative_ranks: ``w_c = rank / (N + 1)`` for every combination, strictly
            inside ``(0, 1)``.
        best_configuration: index of the in-sample winner in each combination.
        in_sample_performance_of_best: that winner's in-sample performance.
        out_of_sample_performance_of_best: that winner's out-of-sample
            performance. Paired with the previous field this is the
            performance-degradation scatter of Bailey et al. (2017) §4.
        n_combinations: ``C(S, S/2)``.
        n_configurations: ``N``.
        n_partitions: ``S``.
    """

    pbo: float
    logits: FloatArray
    relative_ranks: FloatArray
    best_configuration: IntArray
    in_sample_performance_of_best: FloatArray
    out_of_sample_performance_of_best: FloatArray
    n_combinations: int
    n_configurations: int
    n_partitions: int


def _partition_bounds(n_rows: int, n_partitions: int) -> tuple[tuple[int, int], ...]:
    """Cut ``n_rows`` observations into ``n_partitions`` contiguous blocks.

    The first ``n_rows % n_partitions`` blocks receive one extra row. Bailey et
    al. specify equal-size partitions; when ``T`` is not divisible by ``S`` the
    sizes differ by at most one, which is the closest the data allows.
    """
    base, remainder = divmod(n_rows, n_partitions)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(n_partitions):
        stop = start + base + (1 if index < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)


def _sharpe_from_sufficient_statistics(
    *,
    counts: FloatArray,
    sums: FloatArray,
    sums_of_squares: FloatArray,
    shift: FloatArray,
    ddof: int,
) -> FloatArray:
    """Compute per-configuration Sharpe ratios from partition sums.

    The Sharpe ratio of a union of partitions depends on the rows only through
    their count, sum and sum of squares, so all ``C(S, S/2)`` combinations can be
    evaluated with two matrix products instead of ``C(S, S/2)`` passes over the
    data. Sums are accumulated on values shifted by each configuration's global
    mean, which keeps ``sum(x**2) - n * mean(x)**2`` away from catastrophic
    cancellation.

    Args:
        counts: shape ``(n_combinations,)``, rows selected per combination.
        sums: shape ``(n_combinations, n_configurations)``, sums of shifted values.
        sums_of_squares: same shape, sums of squared shifted values.
        shift: shape ``(n_configurations,)``, the amount subtracted per column.
        ddof: delta degrees of freedom for the standard deviation.

    Returns:
        Shape ``(n_combinations, n_configurations)`` of Sharpe ratios, per
        observation period.

    Raises:
        ValueError: if any selection has a non-positive variance, which leaves
            its Sharpe ratio undefined.
    """
    rows = counts[:, None]
    shifted_mean = sums / rows
    variance = (sums_of_squares - rows * shifted_mean**2) / (rows - ddof)
    if not np.all(variance > 0.0):
        msg = (
            "a configuration has non-positive return variance within some in-sample or "
            "out-of-sample selection, so its Sharpe ratio is undefined; constant return "
            "series cannot take part in CSCV"
        )
        raise ValueError(msg)
    return (shifted_mean + shift[None, :]) / np.sqrt(variance)


def _default_performance_matrix(
    matrix: FloatArray,
    bounds: tuple[tuple[int, int], ...],
    membership: FloatArray,
    *,
    ddof: int,
) -> tuple[FloatArray, FloatArray]:
    """Compute IS and OOS Sharpe ratios for every combination, vectorized."""
    shift = matrix.mean(axis=0)
    centred = matrix - shift[None, :]
    counts = np.array([stop - start for start, stop in bounds], dtype=np.float64)
    sums = np.stack([centred[start:stop].sum(axis=0) for start, stop in bounds])
    sums_of_squares = np.stack([(centred[start:stop] ** 2).sum(axis=0) for start, stop in bounds])

    complement = 1.0 - membership
    in_sample = _sharpe_from_sufficient_statistics(
        counts=membership @ counts,
        sums=membership @ sums,
        sums_of_squares=membership @ sums_of_squares,
        shift=shift,
        ddof=ddof,
    )
    out_of_sample = _sharpe_from_sufficient_statistics(
        counts=complement @ counts,
        sums=complement @ sums,
        sums_of_squares=complement @ sums_of_squares,
        shift=shift,
        ddof=ddof,
    )
    return in_sample, out_of_sample


def _custom_performance_matrix(
    matrix: FloatArray,
    bounds: tuple[tuple[int, int], ...],
    combinations: tuple[tuple[int, ...], ...],
    performance: Callable[[FloatArray], FloatArray],
) -> tuple[FloatArray, FloatArray]:
    """Compute IS and OOS performance for every combination with a user function."""
    n_partitions = len(bounds)
    n_configurations = matrix.shape[1]
    in_sample = np.empty((len(combinations), n_configurations), dtype=np.float64)
    out_of_sample = np.empty_like(in_sample)
    partition_rows = [np.arange(start, stop, dtype=np.int64) for start, stop in bounds]
    for index, chosen in enumerate(combinations):
        chosen_set = set(chosen)
        is_rows = np.concatenate(
            [partition_rows[p] for p in range(n_partitions) if p in chosen_set]
        )
        oos_rows = np.concatenate(
            [partition_rows[p] for p in range(n_partitions) if p not in chosen_set]
        )
        in_sample[index] = _checked_performance(performance, matrix[is_rows], n_configurations)
        out_of_sample[index] = _checked_performance(performance, matrix[oos_rows], n_configurations)
    return in_sample, out_of_sample


def _checked_performance(
    performance: Callable[[FloatArray], FloatArray],
    block: FloatArray,
    n_configurations: int,
) -> FloatArray:
    """Call a user performance function and validate its output shape."""
    result = np.asarray(performance(block), dtype=np.float64)
    if result.shape != (n_configurations,):
        msg = (
            f"performance function must return one value per configuration, shape "
            f"({n_configurations},); got {result.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(result)):
        msg = "performance function returned a non-finite value"
        raise ValueError(msg)
    return result


def probability_of_backtest_overfitting(
    performance_matrix: Sequence[Sequence[float]] | npt.ArrayLike,
    *,
    n_partitions: int = 16,
    performance: Callable[[FloatArray], FloatArray] | None = None,
    ddof: int = 1,
) -> PBOResult:
    """Estimate the Probability of Backtest Overfitting by CSCV.

    Args:
        performance_matrix: shape ``(T, N)``. Column ``n`` holds the per-period
            **net-of-cost** returns (invariant I4) of candidate configuration
            ``n``, and row ``t`` is one observation period, in chronological
            order. Every candidate must be evaluated over the same periods —
            CSCV compares columns within a row set, so a ragged matrix has no
            meaning.
        n_partitions: ``S``, the number of contiguous partitions. Must be even
            (the "symmetric" in combinatorially symmetric), at least 2, and no
            larger than ``T``. Default 16, as used in Bailey et al. (2017);
            ``C(16, 8) = 12870`` combinations. Larger ``S`` gives a finer logit
            distribution at combinatorial cost.
        performance: optional custom metric mapping an ``(rows, N)`` block to
            ``N`` performance values. Defaults to the per-period Sharpe ratio
            with a zero risk-free rate, computed from partition sufficient
            statistics so that all combinations cost two matrix products rather
            than ``C(S, S/2)`` passes over the data.
        ddof: delta degrees of freedom for the default Sharpe metric's standard
            deviation. Default 1. Ignored when ``performance`` is supplied.

    Returns:
        A :class:`PBOResult` carrying the scalar and the full logit
        distribution.

    Raises:
        ValueError: if the matrix is not two-dimensional, holds fewer than 2
            configurations (a rank among one candidate is meaningless), holds
            non-finite values, or if ``n_partitions`` is odd, below 2, above
            ``T``, or leaves an in-sample or out-of-sample selection with too
            few rows for the metric.
    """
    matrix = np.asarray(performance_matrix, dtype=np.float64)
    if matrix.ndim != 2:
        msg = f"performance_matrix must be two-dimensional (T, N); got shape {matrix.shape}"
        raise ValueError(msg)
    n_rows, n_configurations = matrix.shape
    if n_configurations < 2:
        msg = (
            f"performance_matrix must hold at least 2 configurations to rank; got "
            f"{n_configurations}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "performance_matrix must be finite; found NaN or infinity"
        raise ValueError(msg)
    if n_partitions < 2 or n_partitions % 2 != 0:
        msg = f"n_partitions must be an even number of at least 2; got {n_partitions}"
        raise ValueError(msg)
    if n_partitions > n_rows:
        msg = (
            f"n_partitions={n_partitions} exceeds the {n_rows} observations available; "
            "every partition must hold at least one row"
        )
        raise ValueError(msg)

    bounds = _partition_bounds(n_rows, n_partitions)
    combinations = tuple(itertools.combinations(range(n_partitions), n_partitions // 2))

    if performance is None:
        membership = np.zeros((len(combinations), n_partitions), dtype=np.float64)
        for index, chosen in enumerate(combinations):
            membership[index, list(chosen)] = 1.0
        smallest_half = min(
            sum(stop - start for start, stop in bounds[: n_partitions // 2]),
            sum(stop - start for start, stop in bounds[n_partitions // 2 :]),
        )
        if smallest_half <= ddof:
            msg = (
                f"each half of the sample must hold more than ddof={ddof} observations; "
                f"the smallest holds {smallest_half}"
            )
            raise ValueError(msg)
        in_sample, out_of_sample = _default_performance_matrix(
            matrix, bounds, membership, ddof=ddof
        )
    else:
        in_sample, out_of_sample = _custom_performance_matrix(
            matrix, bounds, combinations, performance
        )

    best = np.argmax(in_sample, axis=1).astype(np.int64)
    rows = np.arange(len(combinations))
    best_out_of_sample = out_of_sample[rows, best][:, None]
    # Average rank of the in-sample winner among the N out-of-sample values,
    # ascending and one-based, ties shared: (#strictly worse) + (#tied + 1) / 2.
    strictly_worse = np.count_nonzero(out_of_sample < best_out_of_sample, axis=1)
    tied = np.count_nonzero(out_of_sample == best_out_of_sample, axis=1)
    ranks = strictly_worse + (tied + 1.0) / 2.0
    relative_ranks = ranks / (n_configurations + 1.0)
    logits = np.log(relative_ranks / (1.0 - relative_ranks))

    return PBOResult(
        pbo=float(np.count_nonzero(logits <= 0.0)) / len(combinations),
        logits=logits,
        relative_ranks=relative_ranks,
        best_configuration=best,
        in_sample_performance_of_best=in_sample[rows, best],
        out_of_sample_performance_of_best=out_of_sample[rows, best],
        n_combinations=len(combinations),
        n_configurations=int(n_configurations),
        n_partitions=n_partitions,
    )
