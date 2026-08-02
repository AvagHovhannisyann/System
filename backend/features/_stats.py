"""Cross-sectional statistics with an explicit "not available" policy (P5.2).

Private to :mod:`backend.features`. Every public transform funnels its inputs
through these helpers so that coercion, shape checking, the treatment of missing
values, and the definition of "this cross-section has no dispersion" are decided
once, in one place, with one wording.

--------------------------------------------------------------------------
The two failure channels, and why they are different
--------------------------------------------------------------------------

This package distinguishes **caller errors** from **data conditions**, and they
do not share an outcome:

- A *caller error* is a fact about the code: arrays of different lengths, a
  two-dimensional panel where a cross-section was expected, a percentile of
  ``120``, a feature value of ``+inf``. These raise :class:`ValueError`. They
  cannot be right on any date, so failing loudly on the first one is the only
  useful behaviour.
- A *data condition* is a fact about one date: every name in a sector missing,
  a cross-section with no dispersion, three names where a regression needs
  four. These produce ``NaN`` — "not available" — for the affected entries.
  A pipeline sweeping twenty years of rebalance dates must not abort on the one
  date whose small-cap sector had a single surviving member; it must record that
  the feature is unavailable there, which is exactly what ``NaN`` says.

``NaN`` is never *imputed*. It is never replaced by a mean, a zero, or a
neighbouring date's value. Under directive invariant I3 a filled ``NaN`` is
fabricated data entering the feature matrix, indistinguishable downstream from
a measurement.

There is a third thing that is neither: an **infinity**. It is refused on input
(:func:`reject_infinities`) and it is never produced on output. Every finite
cross-section has a magnitude at which ``float64`` arithmetic overflows — a
sum of values near ``1e308``, a sum of squares of values above ``1e154`` — and
an overflow is a fact about the arithmetic, not about the securities. Where one
occurs the affected entries are ``NaN``: not available, which is true, rather
than ``inf``, which is not a number a downstream model can consume, or ``0.0``,
which is a fabricated measurement. :func:`statistics_are_representable` guards
the statistics and :func:`nan_where_overflowed` guards the results.

--------------------------------------------------------------------------
Units
--------------------------------------------------------------------------

Nothing here has units of its own. Every function is unit-agnostic: it consumes
whatever the caller's feature is denominated in (USD, a ratio, a dimensionless
z-score) and returns statistics in those same units, or a dimensionless count or
flag. The one exception is documented on
:func:`dispersion_is_degenerate`, which compares a standard deviation against a
magnitude in the same units so that the comparison is dimensionless.

--------------------------------------------------------------------------
Assumptions
--------------------------------------------------------------------------

Inputs are **one date's cross-section**: one element per security, in the
caller's security order, which is preserved in every output. Nothing here sorts,
reindexes, or looks at more than the single array it was handed — the property
that makes cross-date leakage structurally impossible in
:mod:`backend.features.transforms` starts here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "BoolArray",
    "FloatArray",
    "GroupArray",
    "as_float_1d",
    "as_group_1d",
    "dispersion_is_degenerate",
    "nan_where_overflowed",
    "observed_mask",
    "order_statistic_bounds",
    "reject_infinities",
    "require_matching_length",
    "require_percentile_pair",
    "statistics_are_representable",
]

type FloatArray = npt.NDArray[np.float64]
"""One-dimensional ``float64`` cross-section. Units are stated by each caller."""

type GroupArray = npt.NDArray[np.int64]
"""One-dimensional ``int64`` array of group labels (dimensionless identifiers)."""

type BoolArray = npt.NDArray[np.bool_]
"""One-dimensional boolean mask, parallel to the cross-section it describes."""


def as_float_1d(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Coerce a cross-section to a one-dimensional ``float64`` array.

    Args:
        values: any array-like of numbers — list, tuple, :class:`numpy.ndarray`,
            :class:`pandas.Series`. ``NaN`` entries are allowed and meaningful;
            see the module docstring.
        name: parameter name, used verbatim in error messages.

    Returns:
        A fresh one-dimensional ``float64`` array (always a copy, so the caller
        may mutate its input afterwards without disturbing a result already
        computed). Element order is the caller's security order and is
        preserved.

    Raises:
        ValueError: if the values are not numeric or are not one-dimensional.
            A two-dimensional panel is refused rather than flattened: a
            flattened panel would be winsorized and standardized *across dates*,
            which is precisely the leakage this package exists to prevent, and
            it would not look wrong in any output.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric; could not convert to float64 ({exc})"
        raise ValueError(msg) from exc
    array = array.reshape(-1) if array.ndim == 2 else array
    if array.ndim != 1:
        msg = (
            f"{name} must be a one-dimensional cross-section (one element per "
            f"security, one date); got shape {array.shape}. Transform each date "
            f"separately — flattening a panel would compute percentiles and means "
            f"across dates, which is look-ahead leakage that produces no visible "
            f"symptom."
        )
        raise ValueError(msg)
    return np.array(array, dtype=np.float64, copy=True)


def as_group_1d(groups: npt.ArrayLike, *, name: str) -> GroupArray:
    """Coerce group labels to a one-dimensional ``int64`` array.

    Args:
        groups: any array-like of integers — sector codes, industry codes, or
            any other exhaustive partition of the cross-section. Labels are
            arbitrary identifiers: only equality between them is used, never
            their order or magnitude.
        name: parameter name, used verbatim in error messages.

    Returns:
        A fresh one-dimensional ``int64`` array in the caller's order.

    Raises:
        ValueError: if the labels are not one-dimensional, are not integral, or
            are floats that are not exactly whole numbers. Floats are refused
            rather than truncated — a sector code of ``4.999`` is an upstream
            bug, not a request for sector 4 — and ``NaN`` is refused outright:
            an unknown sector is not a sector, and silently pooling every
            unknown into one bucket would neutralize those names against each
            other as if they shared an industry.
    """
    array = np.asarray(groups)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional (one label per security); got shape {array.shape}"
        raise ValueError(msg)
    if array.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.floating):
            msg = f"{name} must contain integer group labels; got dtype {array.dtype}"
            raise ValueError(msg)
        if not bool(np.all(np.isfinite(array))) or not bool(np.all(array == np.floor(array))):
            msg = (
                f"{name} must contain whole-numbered group labels; got non-integral, "
                f"infinite or missing values. A security whose sector is unknown must be "
                f"excluded by the caller (pass NaN for its feature value) rather than "
                f"pooled into a catch-all group, which would neutralize unrelated names "
                f"against each other."
            )
            raise ValueError(msg)
    return np.array(array, dtype=np.int64, copy=True)


def reject_infinities(values: FloatArray, *, name: str) -> None:
    """Refuse ``+inf`` / ``-inf`` in a feature cross-section.

    ``NaN`` is a statement ("not available") and is handled everywhere in this
    package. An infinity is not a statement — it is the residue of a division by
    zero upstream. It must not reach a transform: winsorization would clip it to
    the 99th percentile and it would enter the feature matrix as a plausible
    extreme value, which is fabrication under invariant I3, and standardization
    would turn the whole cross-section into ``NaN`` with no indication of why.

    Args:
        values: the cross-section to check. ``NaN`` entries are permitted.
        name: parameter name, used verbatim in error messages.

    Raises:
        ValueError: if any element is ``+inf`` or ``-inf``, naming the first
            offending position.
    """
    infinite = np.flatnonzero(np.isinf(values))
    if infinite.size:
        first = int(infinite[0])
        msg = (
            f"{name} contains {infinite.size} infinite value(s), first at position "
            f"{first} ({values[first]!r}). Infinities are refused, not clipped: "
            f"winsorizing one to the 99th percentile would launder a division by zero "
            f"into a plausible number. Emit NaN upstream where the value is genuinely "
            f"unavailable."
        )
        raise ValueError(msg)


def require_matching_length(arrays: dict[str, FloatArray | GroupArray]) -> int:
    """Check that every named array describes the same cross-section.

    Args:
        arrays: mapping of parameter name to array. Must be non-empty.

    Returns:
        The common length — the number of securities in the cross-section
        (a count).

    Raises:
        ValueError: if two arrays differ in length. Lengths are not broadcast:
            a betas array one element short would silently pair every security
            with the wrong beta from that point on.
    """
    lengths = {name: int(array.shape[0]) for name, array in arrays.items()}
    if len(set(lengths.values())) > 1:
        rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
        msg = (
            f"all cross-section arrays must describe the same securities in the same "
            f"order and therefore have equal length; got {rendered}"
        )
        raise ValueError(msg)
    return next(iter(lengths.values()))


def require_percentile_pair(lower_pct: float, upper_pct: float) -> None:
    """Validate a winsorization percentile pair.

    Args:
        lower_pct: lower cut point, in percent of the observed distribution
            (``1.0`` means the 1st percentile), in ``[0, 100]``.
        upper_pct: upper cut point, same units, at least ``lower_pct``.

    Raises:
        ValueError: if either bound is non-finite, outside ``[0, 100]``, or if
            ``upper_pct < lower_pct``. These are caller errors: a percentile
            pair does not depend on the data, so it is wrong on every date or
            none.
    """
    for label, value in (("lower_pct", lower_pct), ("upper_pct", upper_pct)):
        if not np.isfinite(value):
            msg = f"{label} must be finite; got {value!r}"
            raise ValueError(msg)
        if not 0.0 <= value <= 100.0:
            msg = f"{label} must lie in [0, 100] (percent of the distribution); got {value!r}"
            raise ValueError(msg)
    if upper_pct < lower_pct:
        msg = (
            f"upper_pct must be >= lower_pct; got lower_pct={lower_pct!r}, upper_pct={upper_pct!r}"
        )
        raise ValueError(msg)


def observed_mask(values: FloatArray) -> BoolArray:
    """Return the mask of entries that carry a value.

    Args:
        values: the cross-section. ``NaN`` marks "not available".

    Returns:
        A boolean array, ``True`` where the value is present. This is the mask
        every statistic in this package is computed over, and — for the
        transforms that cannot fabricate a result for an absent input — the mask
        of entries that can be non-``NaN`` in the output.
    """
    return np.asarray(~np.isnan(values), dtype=np.bool_)


def order_statistic_bounds(
    observed: FloatArray, *, lower_pct: float, upper_pct: float
) -> tuple[float, float]:
    """Return winsorization cut points as actual order statistics of the data.

    The cut points are **elements of the sample**, not interpolated between
    them: the lower bound is the largest order statistic at or below the
    requested percentile (:func:`numpy.percentile` with ``method="lower"``) and
    the upper bound the smallest at or above it (``method="higher"``).

    Two consequences, both deliberate:

    1. **Winsorization becomes exactly idempotent.** Clipping to an order
       statistic leaves that order statistic in place at the same rank — the
       values below it collapse onto it and the sample size is unchanged — so
       the second application recomputes the identical bound and changes
       nothing, bit for bit. Linear interpolation (NumPy's default) does not
       have this property: the interpolated 1st percentile of an
       already-winsorized sample sits strictly above the previous one, and
       repeated application keeps eating into the distribution.
    2. **The cut is conservative.** Rounding outward means at most the
       requested tail fraction is modified, never more. Winsorization edits real
       observations; when the rank falls between two names, the tie is broken in
       favour of leaving the data alone.

    Args:
        observed: the present values only, with ``NaN`` already removed. Must be
            non-empty.
        lower_pct: lower cut point in percent, in ``[0, upper_pct]``.
        upper_pct: upper cut point in percent, in ``[lower_pct, 100]``.

    Returns:
        ``(lower_bound, upper_bound)`` in the input's units, with
        ``lower_bound <= upper_bound``. For a single observation both equal it,
        and winsorization is a no-op.
    """
    lower = float(np.percentile(observed, lower_pct, method="lower"))
    upper = float(np.percentile(observed, upper_pct, method="higher"))
    return lower, upper


def dispersion_is_degenerate(
    standard_deviation: float, *, scale: float, relative_tolerance: float
) -> bool:
    """Decide whether a cross-section has any dispersion worth dividing by.

    A cross-section in which every name carries the same value has zero
    dispersion, and dividing by it is the classic source of a silent ``inf``
    (or, once ``0 / 0`` appears, of a ``NaN`` that looks like missing data
    rather than like a degenerate date). The test is not ``std == 0.0``, because
    a *constant* cross-section rarely has exactly zero sample standard
    deviation: subtracting a mean that is itself a rounded sum leaves residue of
    order ``eps * |value|``, and dividing by that residue amplifies pure
    rounding noise into full-scale z-scores — the worst possible outcome, since
    the result looks like a well-behaved standardized feature.

    The comparison is therefore relative: dispersion is degenerate when

    ::

        standard_deviation <= relative_tolerance * scale

    Both sides are in the input's units, so the tolerance itself is
    dimensionless. ``scale`` is the largest absolute value in the cross-section,
    which is the magnitude the rounding residue is proportional to. A
    cross-section of exact zeros has ``scale == 0`` and is caught by the
    equality.

    Args:
        standard_deviation: sample standard deviation of the observed values,
            in the input's units. Must be non-negative.
        scale: magnitude reference in the same units — the maximum absolute
            observed value.
        relative_tolerance: dimensionless floor, well above the floating-point
            noise level and well below the relative dispersion of any real
            cross-sectional feature.

    Returns:
        ``True`` when the cross-section carries no usable dispersion, in which
        case the caller must return ``NaN`` rather than divide.
    """
    return bool(standard_deviation <= relative_tolerance * scale)


def statistics_are_representable(*statistics: float) -> bool:
    """Whether every summary statistic came back as a finite ``float64``.

    :func:`dispersion_is_degenerate` guards the *bottom* of the range — a
    standard deviation too small to divide by. This guards the *top*. A
    cross-section may consist entirely of finite, admissible values and still
    overflow the arithmetic that summarizes it: a sum of values near ``1e308``
    overflows to ``inf``, and ``numpy``'s variance takes a mean of *squared*
    deviations, so any cross-section whose spread exceeds roughly ``1.3e154``
    overflows too.

    Neither outcome may be used. An infinite mean turns every residual into
    ``-inf``; an infinite standard deviation turns every z-score into exactly
    ``0.0``, which reads downstream as "every name is precisely average" — a
    fabricated measurement of the worst kind, because it is finite, plausible
    and silent. The transform must instead report the date as unavailable.

    Args:
        *statistics: summary statistics just computed from a cross-section, in
            the cross-section's own units (a mean, a standard deviation, a
            regression slope, a magnitude). ``NaN`` counts as unrepresentable:
            it can only arrive here as ``inf - inf`` from a partial sum that
            overflowed in both directions.

    Returns:
        ``True`` when every statistic is finite and may be divided by,
        subtracted or multiplied; ``False`` when the caller must return ``NaN``.
    """
    return all(math.isfinite(statistic) for statistic in statistics)


def nan_where_overflowed(values: FloatArray) -> FloatArray:
    """Replace any infinity the arithmetic produced with ``NaN``.

    Every transform in this package rejects infinite *inputs*
    (:func:`reject_infinities`), so an infinity in a *result* can only be a
    ``float64`` overflow — most reachably a residual ``x - group_mean`` whose
    two finite terms are near opposite ends of the range and whose difference
    is not representable. That is a fact about the arithmetic on one date, so
    it is a data condition: the affected entries become ``NaN``, "not
    available", rather than travelling on as ``inf``.

    Emitting the ``inf`` instead would be worse than useless. Downstream it
    would be refused by the very next transform's :func:`reject_infinities`,
    turning a data condition into a :class:`ValueError` that blames a caller
    who passed nothing infinite.

    Args:
        values: a freshly computed result array, in the caller's units.
            Existing ``NaN`` entries are left as they are.

    Returns:
        A new array, identical except that ``+inf`` and ``-inf`` have become
        ``NaN``.
    """
    return np.asarray(np.where(np.isinf(values), np.nan, values), dtype=np.float64)
