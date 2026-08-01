"""Purged K-fold cross-validation with embargo (P8.1, directive §5 Phase 8).

Financial ML labels are **spans, not points**. A 21-day triple-barrier label
observed at ``t`` is only resolved by price information through ``t + 21d``.
Naive K-fold shuffles observations as if each were a point, so a training
observation whose label window overlaps the test window carries the test
period's answer into training. The model then scores well on data it has
already seen through the back door, and every downstream number — IC,
t-statistic, Sharpe, the Deflated Sharpe of Phase 10 — is fiction.

Two corrections, both from López de Prado, *Advances in Financial Machine
Learning*, ch. 7, are implemented here:

**Purging.** Drop from the training set every observation whose label interval
overlaps the test fold's span. This removes the direct overlap.

**Embargo.** Additionally drop training observations that *begin* shortly after
the test fold ends. Purging alone is insufficient because serial correlation
makes an observation starting just after the test window a near-duplicate of
one inside it: the features are built from overlapping trailing windows and the
returns are autocorrelated, so it leaks by proximity rather than by literal
interval overlap.

--------------------------------------------------------------------------
Units and conventions — read before using
--------------------------------------------------------------------------

**Times.** ``event_time`` and ``label_end_time`` are timezone-aware instants.
Naive datetimes are rejected with :class:`TypeError`, never silently localized:
D-012 records an hours-scale silent temporal error in this codebase caused by
exactly that reinterpretation. Internally everything is nanoseconds since the
Unix epoch in UTC (``int64``), so comparisons are exact integer comparisons and
no floating-point date arithmetic is involved anywhere.

**Label spans are closed intervals** ``[event_time, label_end_time]``. Both ends
are inclusive. ``label_end_time == event_time`` is a legal zero-length span (a
point-in-time label).

**Overlap is inclusive at the boundary.** A training observation whose label
ends *exactly* at the test fold's start instant is treated as overlapping and is
purged. Rationale: ``label_end_time`` is the last instant whose information the
label consumes, and ``event_time`` is the first instant the test observation's
own window consumes — touching at a single instant means they share that
instant's information. The error costs here are wildly asymmetric. Wrongly
keeping a leaking observation inflates every downstream metric *silently*.
Wrongly dropping a clean one costs a little statistical power, and that cost is
visible in the fold sizes this module reports. When the two are indistinguishable
on principle, take the loud error. This is a fixed decision, not a flag: a knob
that turns leak protection off is a knob that will eventually be turned.

**Embargo is a `datetime.timedelta` — wall-clock calendar time.** Not trading
days, not bars, not a number of observations. ``timedelta(days=2)`` applied
across a weekend embargoes Saturday and Sunday and expires Monday; if you want
two *trading* days you must convert before constructing the splitter, because
this class knows nothing about exchange calendars. The embargo window is
half-open at the left and closed at the right:

    embargoed  ⟺  test_span_end < event_time <= test_span_end + embargo

The strict ``<`` on the left is what makes ``embargo=timedelta(0)`` reduce
*exactly* to pure purging rather than dropping one extra boundary observation.

**Embargo is applied only after the test fold, never before.** Information
flows forward. A training observation whose entire label interval closes before
the test window opens cannot contain test-period information; purging already
removes anything that reaches into the window. Applying a symmetric embargo
would discard clean data for no leakage reason.

**Why a timedelta, and not a fraction, is the canonical form.** "Fraction of
total sample length" is ambiguous in a way that hides bugs: López de Prado's
``pctEmbargo`` is a fraction of the *number of observations*, which is a time
window only when observations are evenly spaced in time. Event-based sampling —
which is what triple-barrier labelling produces, and what Phase 6 will feed this
— is emphatically not evenly spaced, and a cross-sectional panel puts hundreds
of observations on a single day. A row-count fraction is then not a duration at
all. Serial correlation decays in *time*, so the embargo that suppresses it must
be stated in time. ``embargo_fraction`` is offered as convenience, and it is a
fraction of the sample's total **time span** (``max(label_end_time) -
min(event_time)``), not of the row count. It is resolved to a
:class:`datetime.timedelta` **once, at construction**, rounded to the nearest
microsecond; :attr:`PurgedKFold.embargo` returns that resolved value, so the
number reported is by construction the number applied.

**Index arrays are positions into the caller's arrays** (``numpy.intp``), in the
original input order, returned sorted ascending. Input need not be sorted by
time; the splitter sorts internally and maps back.

**Determinism (invariant I2).** No shuffling, no randomness, no seed. Ties among
identical ``(event_time, label_end_time)`` pairs are broken by original position
via a lexicographic sort, so the same spans and parameters always produce
byte-identical index arrays.

--------------------------------------------------------------------------
Known limitation, stated rather than hidden
--------------------------------------------------------------------------

Fold boundaries are drawn between *observations*, not between *timestamps*. In a
cross-sectional panel many observations share one ``event_time``, so a boundary
can cut a single day's cross-section between two test folds. This is not a
leakage hole — any training observation sharing that timestamp overlaps the test
span and is purged — but it does mean fold sizes are equal in rows rather than
aligned to dates. Whether the predictor should instead split on distinct event
times is a modelling decision for P8.2, not a correctness question here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import numpy.typing as npt

__all__ = [
    "EmptyTrainingSetError",
    "Fold",
    "PurgedCVError",
    "PurgedKFold",
]

_NS_PER_SECOND: Final = 1_000_000_000
_SECONDS_PER_DAY: Final = 86_400
_NS_PER_MICROSECOND: Final = 1_000
_INT64_MAX: Final = int(np.iinfo(np.int64).max)


class PurgedCVError(Exception):
    """Base class for purged cross-validation failures."""


class EmptyTrainingSetError(PurgedCVError):
    """Raised when purging and embargo leave a fold with no training rows.

    Returning an empty training split would be the silent version of this
    failure: the caller would fit on nothing, or scikit-learn would raise
    somewhere far from the cause, and the fold configuration that produced it
    would never be identified. The message names the fold and the exact
    accounting that emptied it.

    Attributes:
        fold_index: zero-based position of the offending fold.
        n_observations: total observations in the sample (count).
        n_test: observations in this fold's test set (count).
        n_purged: non-test observations dropped for label-interval overlap
            with the test span (count).
        n_embargoed: non-test observations dropped by the embargo window
            (count).
        embargo: the resolved embargo actually applied (wall-clock duration).
    """

    def __init__(
        self,
        *,
        fold_index: int,
        n_observations: int,
        n_test: int,
        n_purged: int,
        n_embargoed: int,
        embargo: dt.timedelta,
    ) -> None:
        """Build the error and its message from the fold's accounting."""
        self.fold_index = fold_index
        self.n_observations = n_observations
        self.n_test = n_test
        self.n_purged = n_purged
        self.n_embargoed = n_embargoed
        self.embargo = embargo
        super().__init__(
            f"fold {fold_index} has no training observations: "
            f"{n_observations} total - {n_test} test - {n_purged} purged "
            f"- {n_embargoed} embargoed = 0. "
            f"Resolved embargo was {embargo!r}. "
            f"Reduce the embargo, reduce n_splits, or shorten the label horizon; "
            f"an empty training split is never returned."
        )


@dataclass(frozen=True, slots=True)
class Fold:
    """One purged, embargoed train/test split, with its accounting.

    The diagnostic counts are part of the public surface on purpose: a purged
    fold silently throwing away most of the sample is a real and easily missed
    failure mode, and P8.5's fold visualization needs the same numbers.

    Attributes:
        index: zero-based fold number.
        train: positions of the training observations, ascending, into the
            caller's original (unsorted) arrays. Dimensionless indices.
        test: positions of the test observations, ascending, into the caller's
            original arrays. Dimensionless indices.
        test_span_start: earliest ``event_time`` among the test observations
            (timezone-aware UTC instant). The left end of the interval that
            purging protects.
        test_span_end: latest ``label_end_time`` among the test observations
            (timezone-aware UTC instant). The right end of that interval, and
            the anchor the embargo window starts from.
        n_purged: non-test observations dropped because their label interval
            overlapped ``[test_span_start, test_span_end]`` (count).
        n_embargoed: non-test observations dropped because their ``event_time``
            fell in ``(test_span_end, test_span_end + embargo]`` (count).
        embargo: the resolved embargo applied to this fold (wall-clock
            duration; identical across folds, carried here so a fold is
            self-describing).
    """

    index: int
    train: npt.NDArray[np.intp]
    test: npt.NDArray[np.intp]
    test_span_start: dt.datetime
    test_span_end: dt.datetime
    n_purged: int
    n_embargoed: int
    embargo: dt.timedelta


def _epoch_ns(values: Iterable[dt.datetime], *, name: str) -> npt.NDArray[np.int64]:
    """Convert timezone-aware instants to nanoseconds since the Unix epoch.

    Args:
        values: the instants. Every element must be timezone-aware; a
            :class:`pandas.DatetimeIndex` or ``Series`` with a ``tz`` is
            accepted directly, as is any iterable of aware
            :class:`datetime.datetime`.
        name: parameter name, used in error messages.

    Returns:
        A ``int64`` array of nanoseconds since 1970-01-01T00:00:00Z, in UTC.
        Nanosecond resolution is the storage unit, not a claim about the
        precision of the inputs.

    Raises:
        TypeError: if the values are not datetimes, are timezone-naive, or
            carry mixed UTC offsets that pandas cannot resolve to a single
            timezone. Naive input is refused rather than assumed to be UTC:
            silently localizing is how a temporal error becomes invisible.
        ValueError: if any value is missing (``NaT``/``None``). A missing
            event or label-end time has no defensible default here.
    """
    materialized = (
        values
        if isinstance(values, list | tuple | np.ndarray | pd.Series | pd.Index)
        else list(values)
    )
    if len(materialized) == 0:
        # An empty sample carries no timezone to check. The caller-facing error
        # belongs to the splitter ("cannot cross-validate an empty sample"), not
        # here, so report emptiness rather than a spurious naive-datetime error.
        return np.empty(0, dtype=np.int64)
    try:
        index = pd.DatetimeIndex(materialized)
    except (TypeError, ValueError) as exc:
        msg = (
            f"{name} must be timezone-aware datetimes sharing one UTC offset; "
            f"pandas could not build a DatetimeIndex from them ({exc})"
        )
        raise TypeError(msg) from exc
    if index.tz is None:
        msg = (
            f"{name} must be timezone-aware; got naive datetimes. "
            f"Naive timestamps are refused rather than assumed to be UTC."
        )
        raise TypeError(msg)
    if index.hasnans:
        msg = f"{name} contains missing values (NaT); every observation must carry both times"
        raise ValueError(msg)
    utc = index.tz_convert("UTC").to_numpy(dtype="datetime64[ns]")
    return np.asarray(utc, dtype="datetime64[ns]").astype(np.int64, copy=True)


def _timedelta_to_ns(value: dt.timedelta) -> int:
    """Convert a timedelta to whole nanoseconds, exactly, as a Python int.

    Args:
        value: any duration. :class:`datetime.timedelta` has microsecond
            resolution, so the conversion is exact and lossless.

    Returns:
        The duration in nanoseconds. A Python ``int`` (unbounded) rather than
        ``int64``, because ``timedelta.max`` overflows ``int64`` nanoseconds
        and callers are allowed to pass absurd embargoes; the overflow is
        handled where the value is used, by clamping the comparison threshold.
    """
    seconds = value.days * _SECONDS_PER_DAY + value.seconds
    return seconds * _NS_PER_SECOND + value.microseconds * _NS_PER_MICROSECOND


def _ns_to_datetime(value: int) -> dt.datetime:
    """Render nanoseconds since the Unix epoch as an aware UTC datetime.

    Args:
        value: nanoseconds since 1970-01-01T00:00:00Z.

    Returns:
        The instant as a timezone-aware :class:`datetime.datetime` in UTC.
        Sub-microsecond components are preserved by returning a
        :class:`pandas.Timestamp`, which is a ``datetime`` subclass carrying
        nanosecond precision.
    """
    return pd.Timestamp(value, unit="ns", tz="UTC")


class PurgedKFold:
    """K-fold splitter for observations whose labels span time.

    Compatible in spirit with scikit-learn's splitters — ``split(X, y=None,
    groups=None)`` produces train/test index arrays and ``get_n_splits`` reports
    the fold count — but *not* interchangeable with them, deliberately. The
    label spans are constructor arguments rather than something inferred from
    ``X``, because the spans are what the splitter is about; passing them at
    ``split`` time through ``groups`` (the only sklearn-shaped alternative)
    would mean a caller who forgets them gets a silently leaking naive K-fold
    instead of an error.

    Folds are contiguous blocks of the time-ordered sample, sized as evenly as
    possible with the first ``n_observations % n_splits`` folds taking one extra
    row — the same rule scikit-learn's ``KFold`` uses. There is no shuffling:
    shuffling is precisely the thing that breaks temporal structure.

    Example:
        >>> import datetime as dt
        >>> day = dt.timedelta(days=1)
        >>> t0 = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
        >>> events = [t0 + i * day for i in range(10)]
        >>> ends = [e + 2 * day for e in events]
        >>> cv = PurgedKFold(events, ends, n_splits=5, embargo=day)
        >>> [(len(tr), len(te)) for tr, te in cv.split()]
        [(5, 2), (3, 2), (3, 2), (4, 2), (6, 2)]

    Attributes:
        n_splits: number of folds (count).
        embargo: the resolved embargo, always a concrete
            :class:`datetime.timedelta` in wall-clock time, even when it was
            specified as a fraction.
        embargo_fraction: the fraction supplied by the caller, or ``None`` when
            the embargo was given directly as a duration. Kept for provenance
            (I2: a result must be regenerable from its configuration).
        n_observations: number of observations the splitter was built over
            (count).
    """

    def __init__(
        self,
        event_time: Iterable[dt.datetime],
        label_end_time: Iterable[dt.datetime],
        *,
        n_splits: int = 5,
        embargo: dt.timedelta | None = None,
        embargo_fraction: float | None = None,
    ) -> None:
        """Validate the spans and resolve the embargo, once, up front.

        Everything that can be checked without folds is checked here, so a
        misconfigured splitter fails at construction rather than midway through
        a training loop.

        Args:
            event_time: for each observation, the instant its feature window
                closes and its label begins — the first instant the observation
                consumes information from. Timezone-aware.
            label_end_time: for each observation, the last instant whose
                information the label consumes (e.g. the triple-barrier touch
                time, or the horizon end). Timezone-aware, and never earlier
                than the matching ``event_time``.
            n_splits: number of folds. At least 2, at most one per observation.
            embargo: wall-clock duration to embargo after each test fold's end.
                ``None`` and ``timedelta(0)`` both mean "purge only". Not
                trading days — see the module docstring.
            embargo_fraction: alternative specification, as a fraction of the
                sample's **total time span** ``max(label_end_time) -
                min(event_time)``. Must be in ``[0, 1]``. Mutually exclusive
                with ``embargo``. Resolved to a ``timedelta`` here, rounded to
                the nearest microsecond, and that resolved value is what is
                applied and what :attr:`embargo` reports.

        Raises:
            TypeError: if any time is naive or is not a datetime.
            ValueError: if the two time arrays differ in length; if the sample
                is empty; if any ``label_end_time`` precedes its
                ``event_time``; if ``n_splits`` is below 2 or exceeds the
                number of observations; if both or neither embargo form is
                usable (both given); if ``embargo`` is negative; or if
                ``embargo_fraction`` is outside ``[0, 1]`` or not finite.
        """
        if embargo is not None and embargo_fraction is not None:
            msg = (
                "specify embargo or embargo_fraction, not both — two sources of truth for "
                "one duration is exactly how the wrong one gets applied"
            )
            raise ValueError(msg)

        self._event_ns = _epoch_ns(event_time, name="event_time")
        self._label_end_ns = _epoch_ns(label_end_time, name="label_end_time")

        if self._event_ns.shape[0] != self._label_end_ns.shape[0]:
            msg = (
                f"event_time and label_end_time must have equal length; got "
                f"{self._event_ns.shape[0]} and {self._label_end_ns.shape[0]}"
            )
            raise ValueError(msg)

        n_observations = int(self._event_ns.shape[0])
        if n_observations == 0:
            msg = "cannot cross-validate an empty sample: event_time is empty"
            raise ValueError(msg)

        inverted = np.flatnonzero(self._label_end_ns < self._event_ns)
        if inverted.size:
            first = int(inverted[0])
            msg = (
                f"label_end_time must be >= event_time for every observation; "
                f"{inverted.size} violate this, first at position {first} "
                f"({_ns_to_datetime(int(self._label_end_ns[first]))} < "
                f"{_ns_to_datetime(int(self._event_ns[first]))})"
            )
            raise ValueError(msg)

        if n_splits < 2:
            msg = f"n_splits must be >= 2; got {n_splits}"
            raise ValueError(msg)
        if n_splits > n_observations:
            msg = (
                f"n_splits ({n_splits}) cannot exceed the number of observations "
                f"({n_observations}); some fold would be empty"
            )
            raise ValueError(msg)

        self.n_splits = int(n_splits)
        self.n_observations = n_observations
        self.embargo_fraction = None if embargo_fraction is None else float(embargo_fraction)
        self.embargo = self._resolve_embargo(embargo, embargo_fraction)
        self._embargo_ns = _timedelta_to_ns(self.embargo)

        positions = np.arange(n_observations, dtype=np.intp)
        # lexsort's last key is primary: event_time, then label_end_time, then
        # original position. The position tiebreak is what makes duplicate
        # timestamps land in deterministic folds (I2).
        self._order: npt.NDArray[np.intp] = np.lexsort(
            (positions, self._label_end_ns, self._event_ns)
        ).astype(np.intp, copy=False)

    def _resolve_embargo(
        self, embargo: dt.timedelta | None, embargo_fraction: float | None
    ) -> dt.timedelta:
        """Turn whichever embargo form was supplied into one wall-clock duration.

        Args:
            embargo: the duration form, or ``None``.
            embargo_fraction: the fraction-of-total-time-span form, or ``None``.

        Returns:
            The embargo to apply. ``timedelta(0)`` when neither form was given.
            A fraction is multiplied by ``max(label_end_time) -
            min(event_time)`` and rounded to the nearest microsecond, so the
            returned value is exactly what the splitter applies.

        Raises:
            ValueError: if ``embargo`` is negative, or ``embargo_fraction`` is
                not a finite number in ``[0, 1]``.
        """
        if embargo is not None:
            if embargo < dt.timedelta(0):
                msg = f"embargo must be >= 0; got {embargo!r}"
                raise ValueError(msg)
            return embargo
        if embargo_fraction is None:
            return dt.timedelta(0)
        fraction = float(embargo_fraction)
        if not np.isfinite(fraction) or not (0.0 <= fraction <= 1.0):
            msg = (
                f"embargo_fraction must be a finite fraction in [0, 1] of the sample's total "
                f"time span; got {embargo_fraction!r}"
            )
            raise ValueError(msg)
        total_span_ns = int(self._label_end_ns.max()) - int(self._event_ns.min())
        microseconds = round(fraction * total_span_ns / _NS_PER_MICROSECOND)
        return dt.timedelta(microseconds=microseconds)

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803 — sklearn's splitter signature is capital-X
        y: object = None,
        groups: object = None,
    ) -> int:
        """Return the number of folds, ignoring the arguments.

        Args:
            X: accepted and unused; present for scikit-learn signature
                compatibility.
            y: accepted and unused.
            groups: accepted and unused.

        Returns:
            :attr:`n_splits` (count).
        """
        del X, y, groups
        return self.n_splits

    def folds(
        self,
        X: object = None,  # noqa: N803 — sklearn's splitter signature is capital-X
        y: object = None,
        groups: object = None,
    ) -> list[Fold]:
        """Compute every fold, with its purge/embargo accounting.

        All folds are built before any is returned, so a configuration that
        empties some later fold's training set fails immediately instead of
        after several folds have already been fitted.

        Args:
            X: optional feature container. Only its length is used, and only to
                check it matches the number of observations the splitter was
                built over — a mismatch means the spans do not describe these
                rows, which is a silent disaster if allowed through.
            y: optional label container, length-checked the same way.
            groups: must be ``None``. Grouping is expressed through the label
                spans, so a supplied ``groups`` would be silently ignored;
                refusing is honest.

        Returns:
            One :class:`Fold` per split, in time order.

        Raises:
            ValueError: if ``X`` or ``y`` has a length other than
                :attr:`n_observations`, or if ``groups`` is not ``None``.
            EmptyTrainingSetError: if purging and embargo leave any fold with
                no training observations.
        """
        self._check_split_arguments(X, y, groups)

        fold_sizes = np.full(self.n_splits, self.n_observations // self.n_splits, dtype=np.intp)
        fold_sizes[: self.n_observations % self.n_splits] += 1
        boundaries = np.concatenate(([0], np.cumsum(fold_sizes))).astype(np.intp, copy=False)

        return [
            self._build_fold(
                index=fold_index,
                test_positions=self._order[
                    int(boundaries[fold_index]) : int(boundaries[fold_index + 1])
                ],
            )
            for fold_index in range(self.n_splits)
        ]

    def split(
        self,
        X: object = None,  # noqa: N803 — sklearn's splitter signature is capital-X
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]]:
        """Yield ``(train, test)`` position arrays for each fold.

        Deliberately *not* a generator function: the folds are computed eagerly
        and an iterator over the finished list is returned, so a bad
        configuration raises at the call site rather than partway through the
        caller's training loop.

        Args:
            X: optional feature container; see :meth:`folds`.
            y: optional label container; see :meth:`folds`.
            groups: must be ``None``; see :meth:`folds`.

        Returns:
            An iterator of ``(train, test)`` pairs of ``numpy.intp`` position
            arrays, each sorted ascending, indexing into the caller's original
            arrays. Fresh arrays on every call; callers may mutate them.

        Raises:
            ValueError: as :meth:`folds`.
            EmptyTrainingSetError: as :meth:`folds`.
        """
        return iter([(fold.train, fold.test) for fold in self.folds(X, y, groups)])

    def _check_split_arguments(self, X: object, y: object, groups: object) -> None:  # noqa: N803
        """Reject arguments that would make the split describe the wrong rows.

        Args:
            X: candidate feature container, or ``None``.
            y: candidate label container, or ``None``.
            groups: must be ``None``.

        Raises:
            ValueError: on a length mismatch or a non-``None`` ``groups``.
        """
        if groups is not None:
            msg = (
                "PurgedKFold does not use `groups`: grouping is expressed by the label spans "
                "passed to the constructor. Refusing rather than ignoring it silently."
            )
            raise ValueError(msg)
        for name, container in (("X", X), ("y", y)):
            if container is None:
                continue
            try:
                length = len(container)  # type: ignore[arg-type]
            except TypeError:
                continue
            if length != self.n_observations:
                msg = (
                    f"{name} has {length} rows but the splitter was built over "
                    f"{self.n_observations} label spans; the spans do not describe these rows"
                )
                raise ValueError(msg)

    def _build_fold(self, *, index: int, test_positions: npt.NDArray[np.intp]) -> Fold:
        """Purge and embargo one fold's training set.

        Args:
            index: zero-based fold number.
            test_positions: positions of this fold's test observations, in
                time order.

        Returns:
            The completed :class:`Fold`.

        Raises:
            EmptyTrainingSetError: if nothing survives purging and embargo.
        """
        is_test = np.zeros(self.n_observations, dtype=bool)
        is_test[test_positions] = True

        span_start_ns = int(self._event_ns[test_positions].min())
        span_end_ns = int(self._label_end_ns[test_positions].max())

        # Closed-interval overlap: [event, label_end] meets [span_start, span_end].
        overlaps = (self._event_ns <= span_end_ns) & (self._label_end_ns >= span_start_ns)
        # Embargo: strictly after the span end, up to and including the horizon.
        # Clamped because a caller may legitimately pass an embargo whose
        # nanosecond horizon overflows int64.
        horizon_ns = min(span_end_ns + self._embargo_ns, _INT64_MAX)
        embargoed = (self._event_ns > span_end_ns) & (self._event_ns <= horizon_ns)

        candidate = ~is_test
        purged_mask = overlaps & candidate
        embargoed_mask = embargoed & candidate
        train_mask = candidate & ~purged_mask & ~embargoed_mask

        n_purged = int(np.count_nonzero(purged_mask))
        n_embargoed = int(np.count_nonzero(embargoed_mask))
        if not train_mask.any():
            raise EmptyTrainingSetError(
                fold_index=index,
                n_observations=self.n_observations,
                n_test=int(test_positions.size),
                n_purged=n_purged,
                n_embargoed=n_embargoed,
                embargo=self.embargo,
            )

        return Fold(
            index=index,
            train=np.flatnonzero(train_mask).astype(np.intp, copy=False),
            test=np.sort(test_positions).astype(np.intp, copy=True),
            test_span_start=_ns_to_datetime(span_start_ns),
            test_span_end=_ns_to_datetime(span_end_ns),
            n_purged=n_purged,
            n_embargoed=n_embargoed,
            embargo=self.embargo,
        )

    def __repr__(self) -> str:
        """Return a reproducible one-line summary of the configuration (I2)."""
        return (
            f"{type(self).__name__}(n_splits={self.n_splits}, embargo={self.embargo!r}, "
            f"embargo_fraction={self.embargo_fraction!r}, "
            f"n_observations={self.n_observations})"
        )
