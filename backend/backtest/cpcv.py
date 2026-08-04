"""Combinatorial Purged Cross-Validation — P10.2.

Why this exists
---------------

A walk-forward backtest yields exactly **one** out-of-sample path. One path is
one draw. Reporting its Sharpe ratio as "the" out-of-sample Sharpe is reporting
a single sample from an unknown distribution as though it were the mean of that
distribution, and it is how a lucky sequence becomes a strategy.

CPCV replaces the single path with a *distribution* of paths. The sample is cut
into ``N`` contiguous groups; every ``k``-subset of those groups is used as a
test set in turn, so there are ``C(N, k)`` train/test splits rather than one
sequence. Because each group is tested in ``C(N-1, k-1)`` of those splits, the
per-group out-of-sample forecasts can be reassembled into exactly
``C(N-1, k-1)`` complete backtest paths, each covering the whole sample once.
That count is a theorem, not a tuning knob, and it is asserted as one in the
tests:

``n_paths = k * C(N, k) / N = C(N-1, k-1)``

Purging and embargoing
----------------------

Financial labels overlap: a label observed at ``t`` is typically resolved over
``[t, t + h]``. If a training observation's label interval overlaps the test
interval, the training set has seen the test period's outcome and the split
leaks. Two corrections, both from López de Prado (2018) chapter 7:

* **Purging** (§7.4.1): drop from the training set every observation whose
  label interval ``[t0, t1]`` overlaps the test set's interval.
* **Embargo** (§7.4.2): additionally drop training observations that *begin*
  within ``embargo`` of the end of a test block. Purging alone does not remove
  leakage carried by serial correlation into the period immediately following
  the test set. The embargo is one-sided — forward only — because the leak it
  addresses is one-sided.

With ``k > 1`` the test groups need not be adjacent, so purging and embargoing
are applied per **contiguous run** of selected groups rather than to the span
between the first and last of them; treating a disjoint selection as one span
would purge the gap between them for no reason.

Duplication note
----------------

P8.1 builds purged K-fold with embargo in ``backend/models``. The purge and
embargo rules implemented here are the same rules. They are implemented
independently on purpose — that package was unfinished when this was written, and
importing across an unfinished boundary is worse than duplicating forty lines of
interval arithmetic — and the two should be consolidated once both sides are
stable. Whichever survives must keep the property suite in
``backend/tests/backtest/test_cpcv_properties.py``, which checks purging against
an independent transcription of the textbook rule rather than against itself.

Units
-----

``event_starts``, ``event_ends`` and ``embargo`` are plain floats **in one
consistent unit chosen by the caller** — days, seconds, or index positions. The
module never guesses. A caller working from timestamps must convert once, at the
boundary, and pass the embargo in the same unit; mixing days into a
seconds-based series is a silent, plausible-looking error.

Source: López de Prado, M. (2018). *Advances in Financial Machine Learning*,
Wiley. Chapter 7 (purging, embargo) and chapter 12 (combinatorial purged
cross-validation, §12.4).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb
from typing import TYPE_CHECKING

import numpy as np

from backend.backtest.metrics import (
    BoolArray,
    FloatArray,
    IntArray,
    as_float_array,
    sharpe_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt

__all__ = [
    "CPCVSplit",
    "CPCVSplits",
    "CombinatorialPurgedCV",
    "EmptyTrainingSetError",
    "PathDistribution",
    "path_sharpe_ratios",
]


class EmptyTrainingSetError(ValueError):
    """Raised when purging and embargoing leave a split with no training data.

    This is deliberately fatal rather than a warning. A split with an empty
    training set cannot produce an out-of-sample forecast, so every path that
    would draw a group from it is undefined, and a CPCV distribution assembled
    from undefined paths is worse than no distribution at all. The usual causes
    are too few groups, labels longer than a group, or an embargo out of scale
    with the sample.
    """


@dataclass(frozen=True, slots=True, eq=False)
class CPCVSplit:
    """One train/test split of a combinatorial purged cross-validation.

    Attributes:
        split_id: index of this split in the lexicographic enumeration of
            ``C(N, k)`` group combinations. Stable across runs, which is what
            makes a CPCV result reproducible (invariant I2).
        test_groups: the group indices used as the test set, ascending.
        train_indices: positional indices of the training observations, ascending
            and already purged and embargoed.
        test_indices: positional indices of the test observations, ascending.
            These are the union of ``test_groups`` and are never purged —
            purging removes training observations, never test ones.
        n_purged: how many training candidates were removed because their label
            interval overlapped a test block.
        n_embargoed: how many *additional* training candidates were removed by
            the embargo alone (observations removed by both purging and the
            embargo are counted only under ``n_purged``, so the two never
            double-count).
    """

    split_id: int
    test_groups: tuple[int, ...]
    train_indices: IntArray
    test_indices: IntArray
    n_purged: int
    n_embargoed: int


@dataclass(frozen=True, slots=True, eq=False)
class PathDistribution:
    """The distribution of a metric across CPCV backtest paths.

    Directive §10 forbids displaying a point estimate alone. This object is the
    shape that requirement takes for backtest metrics: it carries every path's
    value, and its summaries always come with an interval.

    Attributes:
        values: one metric value per backtest path (for example one Sharpe ratio
            per path). Length equals ``CombinatorialPurgedCV.n_paths``.
    """

    values: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize the stored values."""
        object.__setattr__(self, "values", as_float_array(self.values, name="values"))

    def __len__(self) -> int:
        """Return the number of paths."""
        return int(self.values.size)

    @property
    def mean(self) -> float:
        """Return the arithmetic mean across paths."""
        return float(np.mean(self.values))

    @property
    def median(self) -> float:
        """Return the median across paths."""
        return float(np.median(self.values))

    @property
    def std(self) -> float:
        """Return the sample standard deviation across paths (``ddof=1``).

        Raises:
            ValueError: if there is only one path, where a spread is undefined
                and reporting 0 would claim certainty the data does not support.
        """
        if self.values.size < 2:
            msg = "standard deviation across paths requires at least 2 paths"
            raise ValueError(msg)
        return float(np.std(self.values, ddof=1))

    @property
    def minimum(self) -> float:
        """Return the worst path value."""
        return float(np.min(self.values))

    @property
    def maximum(self) -> float:
        """Return the best path value."""
        return float(np.max(self.values))

    def quantile(self, q: float) -> float:
        """Return the empirical ``q``-quantile across paths.

        Args:
            q: quantile level in ``[0, 1]``.

        Returns:
            The linearly interpolated empirical quantile.

        Raises:
            ValueError: if ``q`` is outside ``[0, 1]``.
        """
        if not 0.0 <= q <= 1.0:
            msg = f"quantile level must lie in [0, 1]; got {q!r}"
            raise ValueError(msg)
        return float(np.quantile(self.values, q))

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the empirical percentile interval across paths.

        This is the spread of the CPCV path distribution, not a parametric
        confidence interval for a population mean, and its resolution is limited
        by the number of paths: with 5 paths a "95% interval" is little more than
        the observed range. Report the path count alongside it.

        Args:
            level: central mass to cover, in ``(0, 1)``. Default 0.95.

        Returns:
            ``(lower, upper)`` empirical quantiles at
            ``(1 - level) / 2`` and ``1 - (1 - level) / 2``.

        Raises:
            ValueError: if ``level`` is not strictly inside ``(0, 1)``.
        """
        if not 0.0 < level < 1.0:
            msg = f"level must lie strictly in (0, 1); got {level!r}"
            raise ValueError(msg)
        tail = (1.0 - level) / 2.0
        return self.quantile(tail), self.quantile(1.0 - tail)

    def describe(self, *, level: float = 0.95) -> dict[str, float]:
        """Summarize the path distribution as a flat dictionary.

        Args:
            level: central mass for the reported interval. Default 0.95.

        Returns:
            ``n_paths``, ``mean``, ``median``, ``std``, ``min``, ``max`` and the
            interval bounds ``ci_lower``/``ci_upper``. ``std`` is omitted when
            there is a single path rather than reported as 0.
        """
        lower, upper = self.confidence_interval(level)
        summary = {
            "n_paths": float(self.values.size),
            "mean": self.mean,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "ci_lower": lower,
            "ci_upper": upper,
        }
        if self.values.size >= 2:
            summary["std"] = self.std
        return summary


@dataclass(frozen=True, slots=True, eq=False)
class CPCVSplits:
    """The complete set of splits produced for one sample, plus path assembly.

    Attributes:
        cv: the configuration that produced these splits.
        n_observations: number of observations in the sample the splits index.
        group_slices: ``(start, stop)`` positional bounds of each group, in
            group order, partitioning ``range(n_observations)`` exactly.
        splits: every split, in lexicographic combination order.
    """

    cv: CombinatorialPurgedCV
    n_observations: int
    group_slices: tuple[tuple[int, int], ...]
    splits: tuple[CPCVSplit, ...]

    def __len__(self) -> int:
        """Return the number of splits, ``C(N, k)``."""
        return len(self.splits)

    def __iter__(self) -> Iterator[CPCVSplit]:
        """Iterate over the splits in lexicographic combination order."""
        return iter(self.splits)

    def __getitem__(self, index: int) -> CPCVSplit:
        """Return the split with the given ``split_id``."""
        return self.splits[index]

    @property
    def n_paths(self) -> int:
        """Return the number of complete backtest paths, ``C(N-1, k-1)``."""
        return self.cv.n_paths

    def path_split_map(self) -> IntArray:
        """Map every ``(path, group)`` pair to the split that supplies it.

        Group ``g`` is tested in exactly ``C(N-1, k-1)`` splits, which is also
        the number of paths. Path ``j`` therefore takes group ``g``'s
        out-of-sample forecast from the ``j``-th split (in ascending split
        order) that tested ``g``. Every path covers every group exactly once, so
        every path spans the whole sample.

        Returns:
            An ``int64`` array of shape ``(n_paths, n_groups)`` holding split
            ids.
        """
        per_group: list[list[int]] = [[] for _ in range(self.cv.n_groups)]
        for split in self.splits:
            for group in split.test_groups:
                per_group[group].append(split.split_id)
        mapping = np.empty((self.n_paths, self.cv.n_groups), dtype=np.int64)
        for group, split_ids in enumerate(per_group):
            mapping[:, group] = split_ids
        return mapping

    def assemble_paths(self, per_split_values: Sequence[npt.ArrayLike]) -> FloatArray:
        """Reassemble per-split out-of-sample values into complete paths.

        Args:
            per_split_values: one entry per split, in ``split_id`` order. Entry
                ``i`` holds one value per observation in ``splits[i].test_indices``,
                aligned to that array's order. In a backtest these are the
                strategy's **net-of-cost** per-period returns (invariant I4) for
                the observations the model trained on split ``i`` was asked to
                forecast.

        Returns:
            A ``float64`` array of shape ``(n_paths, n_observations)``. Row ``j``
            is one complete out-of-sample path over the entire sample; there are
            no gaps, because every path covers every group.

        Raises:
            ValueError: if the number of entries does not equal the number of
                splits, if any entry's length does not match its split's test
                set, or if any value is not finite.
        """
        if len(per_split_values) != len(self.splits):
            msg = (
                f"per_split_values must have one entry per split; expected "
                f"{len(self.splits)}, got {len(per_split_values)}"
            )
            raise ValueError(msg)
        values: list[FloatArray] = []
        for split, raw in zip(self.splits, per_split_values, strict=True):
            array = as_float_array(raw, name=f"per_split_values[{split.split_id}]")
            if array.size != split.test_indices.size:
                msg = (
                    f"per_split_values[{split.split_id}] has {array.size} values but split "
                    f"{split.split_id} has {split.test_indices.size} test observations"
                )
                raise ValueError(msg)
            values.append(array)

        mapping = self.path_split_map()
        paths = np.full((self.n_paths, self.n_observations), np.nan, dtype=np.float64)
        for group, (start, stop) in enumerate(self.group_slices):
            for path in range(self.n_paths):
                split_id = int(mapping[path, group])
                split = self.splits[split_id]
                selector = (split.test_indices >= start) & (split.test_indices < stop)
                paths[path, start:stop] = values[split_id][selector]
        if not np.all(np.isfinite(paths)):
            msg = "internal error: assembled paths contain uncovered observations"
            raise ValueError(msg)
        return paths


@dataclass(frozen=True, slots=True)
class CombinatorialPurgedCV:
    """Configuration for combinatorial purged cross-validation.

    Attributes:
        n_groups: ``N``, the number of contiguous groups the sample is cut into.
            Must be at least 2.
        n_test_groups: ``k``, how many groups form each test set. Must satisfy
            ``1 <= k < N`` so that at least one group is always available to
            train on.
        embargo: length of the one-sided embargo applied *after* every
            contiguous test block, in the **same unit as the event times passed
            to** :meth:`split`. Must be non-negative. Defaults to 0, which
            disables the embargo and leaves only purging.
    """

    n_groups: int
    n_test_groups: int
    embargo: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configuration; a malformed scheme is a defect, not a warning.

        Raises:
            ValueError: if ``n_groups < 2``, if ``n_test_groups`` is outside
                ``[1, n_groups - 1]``, or if ``embargo`` is negative or not
                finite.
        """
        if self.n_groups < 2:
            msg = f"n_groups must be at least 2; got {self.n_groups}"
            raise ValueError(msg)
        if not 1 <= self.n_test_groups < self.n_groups:
            msg = (
                f"n_test_groups must satisfy 1 <= k < n_groups={self.n_groups}; "
                f"got {self.n_test_groups}"
            )
            raise ValueError(msg)
        if not np.isfinite(self.embargo) or self.embargo < 0.0:
            msg = f"embargo must be finite and non-negative; got {self.embargo!r}"
            raise ValueError(msg)

    @property
    def n_splits(self) -> int:
        """Return the number of train/test splits, ``C(N, k)``."""
        return comb(self.n_groups, self.n_test_groups)

    @property
    def n_paths(self) -> int:
        """Return the number of complete backtest paths.

        ``n_paths = k * C(N, k) / N = C(N-1, k-1)``: each of the ``C(N, k)``
        splits contributes ``k`` tested groups, the ``k * C(N, k)`` tested-group
        slots spread evenly over ``N`` groups, and one path consumes one slot
        per group. This is López de Prado (2018) §12.4.
        """
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    @property
    def splits_per_group(self) -> int:
        """Return how many splits test any given group, ``C(N-1, k-1)``.

        Equal to :attr:`n_paths` — that equality is exactly why the per-group
        forecasts assemble into complete paths with none left over.
        """
        return comb(self.n_groups - 1, self.n_test_groups - 1)

    def combinations(self) -> tuple[tuple[int, ...], ...]:
        """Return every ``k``-subset of the groups, in lexicographic order.

        Returns:
            ``C(N, k)`` ascending tuples of group indices. The order is fixed by
            :func:`itertools.combinations` and is therefore reproducible, which
            is what lets a stored ``split_id`` mean the same thing on a re-run
            (invariant I2).
        """
        return tuple(itertools.combinations(range(self.n_groups), self.n_test_groups))

    def group_slices(self, n_observations: int) -> tuple[tuple[int, int], ...]:
        """Cut ``n_observations`` observations into ``N`` contiguous groups.

        The first ``n_observations % N`` groups receive one extra observation;
        every group is non-empty and the groups partition the sample exactly.

        Args:
            n_observations: sample length. Must be at least ``n_groups``.

        Returns:
            ``N`` ``(start, stop)`` half-open positional bounds in group order.

        Raises:
            ValueError: if there are fewer observations than groups, which would
                produce an empty group and therefore an undefined path.
        """
        if n_observations < self.n_groups:
            msg = (
                f"n_observations={n_observations} is smaller than n_groups={self.n_groups}; "
                "every group must hold at least one observation"
            )
            raise ValueError(msg)
        base, remainder = divmod(n_observations, self.n_groups)
        bounds: list[tuple[int, int]] = []
        start = 0
        for group in range(self.n_groups):
            stop = start + base + (1 if group < remainder else 0)
            bounds.append((start, stop))
            start = stop
        return tuple(bounds)

    def split(
        self,
        event_starts: Sequence[float] | npt.ArrayLike,
        event_ends: Sequence[float] | npt.ArrayLike,
    ) -> CPCVSplits:
        """Produce every purged and embargoed train/test split.

        Args:
            event_starts: ``t0`` for each observation — when its label begins.
                Must be non-decreasing, because the groups are contiguous
                *positional* blocks and only a time-ordered sample makes them
                contiguous *in time* as well. Units are the caller's choice and
                must match ``event_ends`` and ``embargo``.
            event_ends: ``t1`` for each observation — when its label is
                resolved. For a point-in-time observation with no horizon, pass
                the same values as ``event_starts``. Must be elementwise at
                least ``event_starts``.

        Returns:
            A :class:`CPCVSplits` holding ``C(N, k)`` splits in lexicographic
            combination order.

        Raises:
            ValueError: if the two arrays differ in length, if ``event_starts``
                is not non-decreasing, or if any ``event_end < event_start``.
            EmptyTrainingSetError: if purging and embargoing empty any split's
                training set.
        """
        starts = as_float_array(event_starts, name="event_starts")
        ends = as_float_array(event_ends, name="event_ends")
        if starts.size != ends.size:
            msg = (
                "event_starts and event_ends must be the same length; got "
                f"{starts.size} and {ends.size}"
            )
            raise ValueError(msg)
        if np.any(np.diff(starts) < 0.0):
            msg = "event_starts must be non-decreasing; sort the sample by label start time first"
            raise ValueError(msg)
        if np.any(ends < starts):
            msg = "every event_end must be at least its event_start"
            raise ValueError(msg)

        n_observations = int(starts.size)
        bounds = self.group_slices(n_observations)
        positions = np.arange(n_observations, dtype=np.int64)

        splits: list[CPCVSplit] = []
        for split_id, test_groups in enumerate(self.combinations()):
            test_mask = np.zeros(n_observations, dtype=np.bool_)
            for group in test_groups:
                start, stop = bounds[group]
                test_mask[start:stop] = True
            train_mask = ~test_mask

            kept, n_purged, n_embargoed = self._purge_and_embargo(
                train_mask=train_mask,
                starts=starts,
                ends=ends,
                test_runs=self._contiguous_runs(test_groups, bounds),
            )
            if not np.any(kept):
                msg = (
                    f"split {split_id} (test groups {test_groups}) has an empty training set "
                    f"after purging {n_purged} and embargoing {n_embargoed} observations"
                )
                raise EmptyTrainingSetError(msg)
            splits.append(
                CPCVSplit(
                    split_id=split_id,
                    test_groups=test_groups,
                    train_indices=positions[kept],
                    test_indices=positions[test_mask],
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
            )
        return CPCVSplits(
            cv=self,
            n_observations=n_observations,
            group_slices=bounds,
            splits=tuple(splits),
        )

    @staticmethod
    def _contiguous_runs(
        test_groups: tuple[int, ...],
        bounds: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Merge adjacent selected groups into contiguous positional runs.

        Adjacent test groups form one uninterrupted test block, so they are
        purged and embargoed as one. Non-adjacent groups are kept separate: the
        training observations sitting between two disjoint test blocks are
        legitimate training data and must not be purged as if they lay inside a
        single span.
        """
        runs: list[tuple[int, int]] = []
        for group in test_groups:
            start, stop = bounds[group]
            if runs and runs[-1][1] == start:
                runs[-1] = (runs[-1][0], stop)
            else:
                runs.append((start, stop))
        return tuple(runs)

    def _purge_and_embargo(
        self,
        *,
        train_mask: BoolArray,
        starts: FloatArray,
        ends: FloatArray,
        test_runs: tuple[tuple[int, int], ...],
    ) -> tuple[BoolArray, int, int]:
        """Apply purging and the forward embargo to the training candidates.

        Purging drops any training observation whose label interval
        ``[t0, t1]`` overlaps a test block's interval — the standard interval
        overlap test ``t0 <= block_end and t1 >= block_start``. The embargo
        additionally drops training observations whose label *begins* in
        ``(block_end, block_end + embargo]``.

        Returns:
            ``(kept_mask, n_purged, n_embargoed)``. ``n_purged`` counts training
            candidates removed by overlap; ``n_embargoed`` counts those removed
            by the embargo *and not already purged*, so the two are disjoint.
        """
        purged = np.zeros(train_mask.size, dtype=np.bool_)
        embargoed = np.zeros(train_mask.size, dtype=np.bool_)
        for start, stop in test_runs:
            block_start = float(starts[start])
            block_end = float(np.max(ends[start:stop]))
            purged |= (starts <= block_end) & (ends >= block_start)
            if self.embargo > 0.0:
                embargoed |= (starts > block_end) & (starts <= block_end + self.embargo)
        kept = train_mask & ~(purged | embargoed)
        n_purged = int(np.count_nonzero(train_mask & purged))
        n_embargoed = int(np.count_nonzero(train_mask & embargoed & ~purged))
        return kept, n_purged, n_embargoed


def path_sharpe_ratios(
    paths: FloatArray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> PathDistribution:
    """Compute one Sharpe ratio per CPCV backtest path.

    Args:
        paths: shape ``(n_paths, n_observations)``, as returned by
            :meth:`CPCVSplits.assemble_paths`. Values are simple per-period
            returns as fractions and **net of modelled costs** (invariant I4).
        risk_free_rate: per-period risk-free rate as a fraction, same
            periodicity as the returns. Default 0.
        periods_per_year: if given, annualize each path's Sharpe by
            ``sqrt(periods_per_year)``. If ``None``, the Sharpe ratios are per
            observation period — which is the unit
            :mod:`backend.backtest.dsr` requires, so leave it ``None`` when the
            distribution feeds the Deflated Sharpe Ratio.
        ddof: delta degrees of freedom for each path's standard deviation.
            Default 1.

    Returns:
        A :class:`PathDistribution` with one value per path.

    Raises:
        ValueError: if ``paths`` is not two-dimensional or has no rows, or if
            any individual path fails :func:`~backend.backtest.metrics.sharpe_ratio`
            validation.
    """
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 2:
        msg = f"paths must be two-dimensional (n_paths, n_observations); got shape {array.shape}"
        raise ValueError(msg)
    if array.shape[0] == 0:
        msg = "paths must contain at least one path"
        raise ValueError(msg)
    values = np.array(
        [
            sharpe_ratio(
                row,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
                ddof=ddof,
            )
            for row in array
        ],
        dtype=np.float64,
    )
    return PathDistribution(values=values)
