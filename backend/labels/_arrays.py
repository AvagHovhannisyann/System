"""Array coercion and validation shared across :mod:`backend.labels`.

Private to the package. Every public entry point funnels its inputs through
these helpers so that shape, dtype and finiteness are checked in one place with
one wording, and so no downstream function has to guess whether it was handed a
list, a :class:`pandas.Series` or an already-validated array.

Nothing here is clever. It exists because the alternative — each module doing
its own ad-hoc ``np.asarray`` — is how a two-dimensional input silently
broadcasts into a wrong answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from backend.labels.errors import LabelInputError

if TYPE_CHECKING:
    import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]
"""One-dimensional ``float64`` array. Units are stated by each caller."""

type IntArray = npt.NDArray[np.intp]
"""One-dimensional ``intp`` array of bar positions (dimensionless indices)."""


def as_float_1d(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Coerce input to a one-dimensional ``float64`` array.

    Args:
        values: any array-like of numbers — list, tuple, ``numpy`` array,
            :class:`pandas.Series`.
        name: parameter name, used verbatim in error messages.

    Returns:
        A fresh, contiguous, one-dimensional ``float64`` array. Always a copy,
        so the caller may mutate its input afterwards without affecting a
        result already computed.

    Raises:
        LabelInputError: if the values are not numeric, or are not
            one-dimensional. Two-dimensional input is refused rather than
            flattened: a silently flattened panel produces a plausible,
            entirely wrong answer.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be numeric; could not convert to float64 ({exc})"
        raise LabelInputError(msg) from exc
    if array.ndim != 1:
        msg = (
            f"{name} must be one-dimensional; got shape {array.shape}. "
            f"Label one security at a time and stack the results — a flattened "
            f"panel would be labelled as if it were a single price path."
        )
        raise LabelInputError(msg)
    return np.array(array, dtype=np.float64, copy=True)


def as_index_1d(values: npt.ArrayLike, *, name: str) -> IntArray:
    """Coerce input to a one-dimensional array of non-negative bar indices.

    Args:
        values: any array-like of integers.
        name: parameter name, used verbatim in error messages.

    Returns:
        A fresh one-dimensional ``intp`` array (dimensionless bar positions),
        in the order given. Order is *not* normalized: the caller's event order
        is the order of the returned labels, so results line up with whatever
        else the caller is carrying alongside.

    Raises:
        LabelInputError: if the values are not integral, are not
            one-dimensional, or contain a negative index. Floats that are not
            exactly integral are refused rather than truncated — a bar index of
            ``4.999`` is a bug upstream, not a request for bar 4.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional; got shape {array.shape}"
        raise LabelInputError(msg)
    if array.size == 0:
        return np.empty(0, dtype=np.intp)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.floating):
            msg = f"{name} must contain integer bar indices; got dtype {array.dtype}"
            raise LabelInputError(msg)
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            msg = (
                f"{name} must contain whole-numbered bar indices; got non-integral values. "
                f"Refusing to truncate — a fractional bar index is an upstream bug."
            )
            raise LabelInputError(msg)
    result = np.asarray(array, dtype=np.intp)
    if np.any(result < 0):
        msg = f"{name} must contain non-negative bar indices; got a negative value"
        raise LabelInputError(msg)
    return np.array(result, dtype=np.intp, copy=True)


def require_same_length(arrays: dict[str, FloatArray]) -> int:
    """Check that every named array has the same length, and return it.

    Args:
        arrays: mapping of parameter name to array. Must be non-empty.

    Returns:
        The common length (count of elements).

    Raises:
        LabelInputError: if two arrays differ in length.
    """
    lengths = {name: int(array.shape[0]) for name, array in arrays.items()}
    distinct = set(lengths.values())
    if len(distinct) > 1:
        rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
        msg = f"all series must have the same length; got {rendered}"
        raise LabelInputError(msg)
    return next(iter(distinct))


def require_positive(array: FloatArray, *, name: str) -> None:
    """Check that every element is finite and strictly positive.

    Used for price series, whose logarithm the barrier engine takes. A
    non-positive or missing price is refused rather than dropped or forward
    filled: both repairs are data decisions belonging to the ingestion layer,
    where they are visible in a quality report, not to the labeller.

    Args:
        array: the values to check.
        name: parameter name, used verbatim in error messages.

    Raises:
        LabelInputError: if any element is non-finite or not strictly positive,
            naming the first offending position.
    """
    bad = np.flatnonzero(~np.isfinite(array) | (array <= 0.0))
    if bad.size:
        first = int(bad[0])
        msg = (
            f"{name} must be finite and strictly positive at every bar; "
            f"{bad.size} value(s) are not, first at bar {first} ({array[first]!r}). "
            f"Gaps and bad prints are an ingestion-layer concern — repairing them here "
            f"would hide them from the data-quality report."
        )
        raise LabelInputError(msg)
