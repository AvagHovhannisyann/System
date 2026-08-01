"""Sample-uniqueness weights and effective sample size (P6.3).

Financial labels are **spans, not points**. A 21-day triple-barrier label
observed on Monday and another observed on Tuesday are computed from almost the
same 21 days of price movement. Counting them as two independent observations
inflates every statistic that divides by ``n``: standard errors shrink, ``t``
statistics grow, and a model trained with equal weights is effectively fitting
the same information many times over while believing it has seen a large sample.

The correction is López de Prado, *Advances in Financial Machine Learning*,
ch. 4:

**Concurrency.** ``c_t`` is the number of labels whose span covers bar ``t``.

**Uniqueness at a bar.** A label's share of bar ``t`` is ``1 / c_t`` — one bar's
information split evenly among the labels claiming it.

**Average uniqueness.** A label's weight is the mean of ``1 / c_t`` over the
bars in its own span::

    u_i = (1 / |span_i|) * sum_{t in span_i} 1 / c_t

**Effective sample size.** ``ESS = sum_i u_i``. Ten thousand heavily overlapping
21-day labels can carry the information of a few hundred independent ones, and
the directive requires that effect to be *reported*, not merely applied — see
:meth:`UniquenessResult.report`.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**Spans are inclusive bar-index intervals** ``[first_bar, last_bar]``, both ends
included. A single-bar span is legal (``first_bar == last_bar``) and has length
1. Indices are dimensionless bar positions, not timestamps: bar arithmetic is
exact integer arithmetic, whereas wall-clock arithmetic would have to take a
position on weekends and holidays that this module has no business taking.

**The span of a triple-barrier label is** ``[event_index + 1,
resolution_index]`` — the bars whose returns the label actually consumes. The
event bar itself is excluded: the label is stamped at that bar's close and
depends on nothing that happens after it within that bar.
:func:`sample_uniqueness_from_labels` applies this mapping so callers do not
have to remember it.

Note the deliberate half-bar difference from
:class:`backend.models.purged_cv.PurgedKFold`, which treats the span as
``[event_time, label_end_time]`` closed at both ends. There, including the event
bar is the *conservative* direction — it purges one extra observation and can
only remove leakage. Here the same choice would be an *error* in the opposite
direction: it would invent an overlap between a label and its own predecessor
where none exists, and inflate the down-weighting.

**Weights lie in ``(0, 1]``.** A weight of 1.0 means the label had every bar of
its span to itself. A weight of ``1/n`` means ``n`` labels covered exactly the
same span. Weights are never zero: a label always covers its own span, so
``c_t >= 1`` there.

**Weights are not normalized.** They are returned as raw average uniqueness.
scikit-learn's ``sample_weight`` is scale-invariant for fitting, and the raw
scale is the one that carries meaning — it sums to the effective sample size.
:meth:`UniquenessResult.normalized_weights` rescales to mean 1 for callers that
want the conventional form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.labels._arrays import as_index_1d
from backend.labels.errors import LabelInputError

if TYPE_CHECKING:
    import numpy.typing as npt

    from backend.labels._arrays import FloatArray, IntArray
    from backend.labels.barriers import LabelSet

__all__ = [
    "MAX_CONCURRENCY_SPAN_BARS",
    "UniquenessResult",
    "effective_sample_size",
    "sample_uniqueness",
    "sample_uniqueness_from_labels",
]

MAX_CONCURRENCY_SPAN_BARS: Final = 100_000_000
"""Largest bar range the concurrency array may cover, as a memory guard."""


@dataclass(frozen=True, slots=True)
class UniquenessResult:
    """Sample-uniqueness weights for a set of label spans.

    Attributes:
        average_uniqueness: one weight per label, in the caller's order, each in
            ``(0, 1]`` (dimensionless). Sums to
            :attr:`effective_sample_size`.
        concurrency: number of labels covering each bar, for bars
            :attr:`first_bar` … :attr:`last_bar` inclusive (counts). Zero at
            bars no label covers — gaps between labels are possible and are not
            an error.
        first_bar: lowest bar index covered by any label (dimensionless).
        last_bar: highest bar index covered by any label (dimensionless).
        effective_sample_size: sum of the weights — the number of *independent*
            observations the sample is worth (a count, generally not an
            integer). Never exceeds :attr:`nominal_count`, and equals it exactly
            when no two labels overlap.
        nominal_count: number of labels supplied (count).
    """

    average_uniqueness: FloatArray
    concurrency: IntArray
    first_bar: int
    last_bar: int
    effective_sample_size: float
    nominal_count: int

    @property
    def uniqueness_ratio(self) -> float:
        """Effective sample size as a fraction of the nominal count.

        Returns:
            ``effective_sample_size / nominal_count``, in ``(0, 1]``
            (dimensionless). 1.0 means no overlap at all; 0.08 means ten
            thousand labels carry the information of eight hundred.
        """
        return self.effective_sample_size / self.nominal_count

    @property
    def max_concurrency(self) -> int:
        """Largest number of labels covering any single bar (count)."""
        return int(self.concurrency.max()) if self.concurrency.size else 0

    def normalized_weights(self) -> FloatArray:
        """Return the weights rescaled to mean 1.

        Returns:
            ``average_uniqueness * nominal_count / effective_sample_size``
            (dimensionless), summing to :attr:`nominal_count`. This is the
            conventional ``sample_weight`` scaling; the relative weighting is
            identical to :attr:`average_uniqueness`, which is what actually
            matters to a fitter, so the choice between them is presentational.
        """
        return np.asarray(
            self.average_uniqueness * (self.nominal_count / self.effective_sample_size),
            dtype=np.float64,
        )

    def report(self) -> str:
        """Render the overlap's effect on sample size as readable text.

        The directive's Phase 6 gate requires the effect on effective sample
        size to be *reported*, not merely computed: a model trained on ten
        thousand labels worth eight hundred is not obviously broken from any
        other artefact, and this is the number that says so.

        Returns:
            A multi-line summary: nominal count, effective sample size, the
            ratio between them, peak concurrency, and the weight distribution.
        """
        weights = self.average_uniqueness
        return "\n".join(
            (
                "Sample uniqueness (label overlap → effective sample size)",
                f"  nominal observations   : {self.nominal_count}",
                f"  effective sample size  : {self.effective_sample_size:.2f}",
                f"  ESS / nominal          : {self.uniqueness_ratio:.4f}",
                f"  bars covered           : {self.first_bar}..{self.last_bar}",
                f"  peak concurrency       : {self.max_concurrency} labels on one bar",
                (
                    f"  average uniqueness     : min {weights.min():.4f}, "
                    f"median {float(np.median(weights)):.4f}, max {weights.max():.4f}"
                ),
            )
        )


def sample_uniqueness(first_bar: npt.ArrayLike, last_bar: npt.ArrayLike) -> UniquenessResult:
    """Compute average uniqueness and effective sample size from label spans.

    Args:
        first_bar: first bar each label's information comes from (inclusive,
            dimensionless bar index).
        last_bar: last bar each label's information comes from (inclusive).
            Must be at least ``first_bar`` for every label; a zero-length span
            is not representable and is refused rather than given a weight
            computed from an empty mean.

    Returns:
        A :class:`UniquenessResult` whose weights are in the caller's label
        order.

    Raises:
        LabelInputError: if the two arrays differ in length, are empty, contain
            a span with ``last_bar < first_bar``, or cover more than
            :data:`MAX_CONCURRENCY_SPAN_BARS` bars (a guard against allocating
            an enormous concurrency array from a typo'd index).

    Example:
        >>> result = sample_uniqueness([0, 5], [4, 9])   # adjacent, no overlap
        >>> result.average_uniqueness
        array([1., 1.])
        >>> result.effective_sample_size
        2.0
        >>> overlapping = sample_uniqueness([0, 0, 0], [9, 9, 9])
        >>> overlapping.effective_sample_size
        1.0
    """
    starts = as_index_1d(first_bar, name="first_bar")
    ends = as_index_1d(last_bar, name="last_bar")
    if starts.shape[0] != ends.shape[0]:
        msg = (
            f"first_bar and last_bar must have equal length; got {starts.shape[0]} "
            f"and {ends.shape[0]}"
        )
        raise LabelInputError(msg)
    n_labels = int(starts.shape[0])
    if n_labels == 0:
        msg = "cannot compute sample uniqueness for zero labels"
        raise LabelInputError(msg)

    inverted = np.flatnonzero(ends < starts)
    if inverted.size:
        position = int(inverted[0])
        msg = (
            f"last_bar must be >= first_bar for every label; {inverted.size} span(s) "
            f"violate this, first at position {position} "
            f"({int(ends[position])} < {int(starts[position])})"
        )
        raise LabelInputError(msg)

    lowest = int(starts.min())
    highest = int(ends.max())
    n_bars = highest - lowest + 1
    if n_bars > MAX_CONCURRENCY_SPAN_BARS:
        msg = (
            f"label spans cover {n_bars} bars ({lowest}..{highest}), above the "
            f"{MAX_CONCURRENCY_SPAN_BARS}-bar guard. Concurrency is computed on a dense "
            f"array over the covered range; a range this large is far more likely to be "
            f"a bad index than a real sample."
        )
        raise LabelInputError(msg)

    # Difference array: +1 where a span opens, -1 one bar past where it closes.
    # Cumulative sum then gives the number of spans covering each bar in one pass.
    deltas = np.zeros(n_bars + 1, dtype=np.int64)
    np.add.at(deltas, starts - lowest, 1)
    np.add.at(deltas, ends - lowest + 1, -1)
    concurrency = np.cumsum(deltas)[:n_bars]

    # 1/c_t on covered bars, 0 elsewhere. Uncovered bars contribute to no span,
    # so their value is never read; setting it to 0 keeps the prefix sum finite.
    covered = concurrency > 0
    share = np.zeros(n_bars, dtype=np.float64)
    share[covered] = 1.0 / concurrency[covered]

    # Prefix sums make each label's mean an O(1) lookup rather than a slice sum,
    # so the whole computation is O(n_labels + n_bars) regardless of span length.
    prefix = np.concatenate(([0.0], np.cumsum(share)))
    span_lengths = (ends - starts + 1).astype(np.float64)
    totals = prefix[ends - lowest + 1] - prefix[starts - lowest]
    average_uniqueness = np.asarray(totals / span_lengths, dtype=np.float64)

    return UniquenessResult(
        average_uniqueness=average_uniqueness,
        concurrency=np.asarray(concurrency, dtype=np.intp),
        first_bar=lowest,
        last_bar=highest,
        effective_sample_size=effective_sample_size(average_uniqueness),
        nominal_count=n_labels,
    )


def sample_uniqueness_from_labels(labels: LabelSet) -> UniquenessResult:
    """Compute sample uniqueness for a triple-barrier label set.

    Applies the span convention documented in this module: a label's information
    span is ``[event_index + 1, resolution_index]``, the bars whose returns it
    consumes.

    Args:
        labels: the label set. Ambiguous labels, if any, are included with the
            spans they carry — this function weights whatever it is given. Call
            :meth:`~backend.labels.barriers.LabelSet.drop_ambiguous` first if
            they are to be excluded from training, so that the weights describe
            the sample actually used.

    Returns:
        A :class:`UniquenessResult` parallel to ``labels.event_index``.

    Raises:
        LabelInputError: if the label set is empty.
    """
    return sample_uniqueness(labels.first_information_bar, labels.resolution_index)


def effective_sample_size(average_uniqueness: npt.ArrayLike) -> float:
    """Sum uniqueness weights into an effective sample size.

    Args:
        average_uniqueness: per-label average uniqueness weights, each in
            ``(0, 1]`` (dimensionless).

    Returns:
        The effective sample size (a count, generally not an integer): the
        number of non-overlapping observations carrying the same information as
        the supplied sample. Bounded above by the number of weights, with
        equality exactly when every weight is 1.0.
    """
    return float(np.sum(np.asarray(average_uniqueness, dtype=np.float64)))
