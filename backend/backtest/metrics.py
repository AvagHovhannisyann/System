"""Return statistics for the Phase 10 validation framework (P10.2-P10.4).

Everything here is small, pure and free of I/O so that the mathematics can be
checked against hand computation rather than against another implementation of
itself.

**Units — stated once, repeated on every function.** Directive §8 is right that
the silent bug class in this domain is a unit mistake, so:

* a *return* is a simple (arithmetic) return over one observation period,
  expressed as a **fraction**: ``0.01`` is one percent, never ``1``;
* a *Sharpe ratio* is **per observation period** unless a function is
  explicitly asked to annualize. Feeding an annualized Sharpe into a formula
  that expects a per-period one inflates the result by ``sqrt(periods_per_year)``
  and nothing downstream will complain;
* returns handed to this module must already be **net of modelled costs**
  (invariant I4). No function here can verify that — which is precisely why it
  is written down at every entry point.

Estimator conventions, fixed here so that :mod:`backend.backtest.dsr` and
:mod:`backend.backtest.pbo` cannot disagree with each other:

* standard deviation uses ``ddof=1`` (the sample estimator) by default;
* :func:`skewness` and :func:`kurtosis` are the **population** (biased) moment
  estimators, matching ``scipy.stats.skew(x)`` and
  ``scipy.stats.kurtosis(x, fisher=False)``. Bailey & López de Prado's
  Probabilistic Sharpe Ratio is written in terms of those, and ``pandas``'
  ``.skew()``/``.kurt()`` are *not* them (``pandas`` returns bias-corrected
  skew and **excess** kurtosis) — swapping the two silently shifts the answer.

The normal distribution functions come from :class:`statistics.NormalDist` in
the standard library (Wichura's AS241 for the inverse), so this package pulls
in no numerical dependency beyond numpy.

Sources:

* Sharpe, W. F. (1994). "The Sharpe Ratio." *Journal of Portfolio Management*
  21(1), 49-58.
* Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient
  Frontier." *Journal of Risk* 15(2), 3-44.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BoolArray",
    "FloatArray",
    "IntArray",
    "as_float_array",
    "average_ranks",
    "kurtosis",
    "sharpe_ratio",
    "skewness",
    "standard_normal_cdf",
    "standard_normal_ppf",
]

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int64]
type BoolArray = npt.NDArray[np.bool_]

_STANDARD_NORMAL = NormalDist(0.0, 1.0)


def standard_normal_cdf(x: float) -> float:
    """Return ``Z[x]``, the standard normal cumulative distribution function.

    Args:
        x: the point at which to evaluate. Dimensionless.

    Returns:
        ``P(X <= x)`` for ``X ~ N(0, 1)``, in ``[0, 1]``.
    """
    return _STANDARD_NORMAL.cdf(x)


def standard_normal_ppf(p: float) -> float:
    """Return ``Z^-1[p]``, the standard normal inverse CDF (the probit).

    Args:
        p: a probability, strictly inside ``(0, 1)``.

    Returns:
        The ``x`` with ``standard_normal_cdf(x) == p``. Dimensionless.

    Raises:
        ValueError: if ``p`` is not strictly between 0 and 1. The endpoints map
            to infinities; a caller that reaches them has a modelling error
            (typically a trial count of 1 — see
            :func:`backend.backtest.dsr.expected_maximum_sharpe_ratio`) and is
            better served by an exception than by ``-inf``.
    """
    if not 0.0 < p < 1.0:
        msg = f"standard_normal_ppf requires 0 < p < 1; got {p!r}"
        raise ValueError(msg)
    return _STANDARD_NORMAL.inv_cdf(p)


def as_float_array(values: Sequence[float] | npt.ArrayLike, *, name: str) -> FloatArray:
    """Coerce ``values`` to a validated one-dimensional float64 array.

    Args:
        values: anything numpy can turn into a float array.
        name: the parameter name, used in error messages so a failure names the
            argument the caller actually passed.

    Returns:
        A one-dimensional ``float64`` array (a copy is not guaranteed; treat the
        result as read-only).

    Raises:
        ValueError: if the input is not one-dimensional, is empty, or contains
            a NaN or infinity. Silently propagating a NaN through a Sharpe ratio
            produces a NaN metric that renders as a blank cell rather than as an
            error, which is exactly the failure mode this project cannot afford.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional; got shape {array.shape}"
        raise ValueError(msg)
    if array.size == 0:
        msg = f"{name} must be non-empty"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite; found NaN or infinity"
        raise ValueError(msg)
    return array


def sharpe_ratio(
    returns: Sequence[float] | npt.ArrayLike,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: float | None = None,
    ddof: int = 1,
) -> float:
    """Compute the Sharpe ratio of a return series.

    ``SR = mean(r - rf) / stdev(r - rf)``.

    Args:
        returns: simple per-period returns as fractions (``0.01`` is 1%), and
            **net of modelled costs** (invariant I4). Nothing here can check
            that; a gross series produces a real number that is a lie.
        risk_free_rate: the per-period risk-free rate, as a fraction, in the
            same periodicity as ``returns``. Defaults to 0.
        periods_per_year: if given, the result is multiplied by
            ``sqrt(periods_per_year)`` to annualize (252 for daily, 12 for
            monthly). If ``None`` (the default) the result is **per observation
            period**, which is the unit every function in
            :mod:`backend.backtest.dsr` expects. Annualization by ``sqrt(T)``
            assumes serially independent returns; it is a reporting convention,
            not a statistical correction.
        ddof: delta degrees of freedom for the standard deviation. Default 1
            (sample estimator).

    Returns:
        The Sharpe ratio, dimensionless.

    Raises:
        ValueError: if ``returns`` fails :func:`as_float_array` validation, if
            there are not more than ``ddof`` observations, if
            ``periods_per_year`` is non-positive, if the excess-return standard
            deviation is zero (an undefined Sharpe ratio, which must never be
            reported as ``inf``), or if the series overflows the double range so
            that the ratio would be a silent 0 or a NaN.
    """
    excess = as_float_array(returns, name="returns") - risk_free_rate
    if excess.size <= ddof:
        msg = f"returns must contain more than ddof={ddof} observations; got {excess.size}"
        raise ValueError(msg)
    deviation = float(np.std(excess, ddof=ddof))
    if deviation == 0.0:
        msg = "Sharpe ratio is undefined: excess returns have zero standard deviation"
        raise ValueError(msg)
    if not math.isfinite(deviation):
        # The variance overflowed. Left alone this returns mean / inf == 0.0 for
        # a symmetric series — a silent, plausible-looking zero rather than an
        # error, which is the failure mode this project cannot afford.
        msg = (
            "Sharpe ratio is not finite: the excess returns overflow the double "
            "range, so their standard deviation cannot be evaluated"
        )
        raise ValueError(msg)
    ratio = float(np.mean(excess)) / deviation
    if periods_per_year is None:
        return ratio
    if periods_per_year <= 0.0:
        msg = f"periods_per_year must be positive; got {periods_per_year!r}"
        raise ValueError(msg)
    return ratio * math.sqrt(periods_per_year)


def _standardized_central_moments(values: FloatArray) -> tuple[float, float, float]:
    """Return the 2nd, 3rd and 4th central moments of ``values / max|deviation|``.

    Standardized moments are scale-free: dividing every deviation by a positive
    constant ``s`` divides ``mk`` by ``s ** k``, and both ``m3 / m2 ** 1.5`` and
    ``m4 / m2 ** 2`` are invariant under that. Rescaling the deviations onto
    ``[-1, 1]`` before taking powers therefore changes no answer while removing
    the two ways the naive form fails silently:

    * **underflow.** A series with deviations of order ``1e-100`` has
      ``m2 ~ 1e-200``, and ``m2 ** 2`` underflows to exactly 0 — a division by
      zero in a function whose zero-variance guard has already passed.
    * **overflow.** Deviations of order ``1e200`` square to infinity, and the
      ratio of two infinities is a NaN.

    After rescaling, ``max|u| = 1`` so ``m2 >= 1 / n`` and every moment is bounded
    by 1. Neither failure is reachable.

    Args:
        values: the sample, already validated finite and non-empty.

    Returns:
        ``(m2, m3, m4)`` of the rescaled deviations.

    Raises:
        ValueError: if every value equals the sample mean, which leaves the
            standardized moments undefined.
    """
    centred = values - np.mean(values)
    scale = float(np.max(np.abs(centred)))
    if scale == 0.0:
        msg = "returns have zero variance"
        raise ValueError(msg)
    unit = centred / scale
    return (
        float(np.mean(unit**2)),
        float(np.mean(unit**3)),
        float(np.mean(unit**4)),
    )


def skewness(returns: Sequence[float] | npt.ArrayLike) -> float:
    """Compute the population (biased) skewness ``g3`` of a return series.

    ``g3 = m3 / m2 ** 1.5`` with ``mk`` the k-th central moment. This is
    ``scipy.stats.skew(x)`` and is the estimator the Probabilistic Sharpe Ratio
    is written in terms of. It is **not** ``pandas.Series.skew()``, which
    applies a bias correction.

    Args:
        returns: simple per-period returns as fractions.

    Returns:
        The skewness, dimensionless. Zero for any symmetric sample.

    Raises:
        ValueError: if the sample has zero variance, which leaves skewness
            undefined.
    """
    values = as_float_array(returns, name="returns")
    try:
        second, third, _ = _standardized_central_moments(values)
    except ValueError as error:
        msg = "skewness is undefined: returns have zero variance"
        raise ValueError(msg) from error
    return third / math.pow(second, 1.5)


def kurtosis(returns: Sequence[float] | npt.ArrayLike) -> float:
    """Compute the population (biased) **non-excess** kurtosis ``g4``.

    ``g4 = m4 / m2 ** 2``. A Gaussian sample gives approximately 3, not 0. This
    is ``scipy.stats.kurtosis(x, fisher=False)``. Passing an *excess* kurtosis
    where a non-excess one is expected shifts the Probabilistic Sharpe Ratio's
    variance term by ``3/4 * SR ** 2`` and produces a confidently wrong
    probability, so the distinction is enforced by naming rather than by
    comment.

    Args:
        returns: simple per-period returns as fractions.

    Returns:
        The non-excess kurtosis, dimensionless and never below 1.

    Raises:
        ValueError: if the sample has zero variance, which leaves kurtosis
            undefined.
    """
    values = as_float_array(returns, name="returns")
    try:
        second, _, fourth = _standardized_central_moments(values)
    except ValueError as error:
        msg = "kurtosis is undefined: returns have zero variance"
        raise ValueError(msg) from error
    return fourth / second**2


def average_ranks(values: Sequence[float] | npt.ArrayLike) -> FloatArray:
    """Rank ``values`` ascending, one-based, averaging ranks within ties.

    Ascending means the largest value receives the largest rank, so "higher
    rank" reads as "better performance" wherever this is used.

    Args:
        values: the values to rank.

    Returns:
        A float array of the same length holding ranks in ``[1, n]``. Tied
        values share the arithmetic mean of the ranks they span, so the ranks
        always sum to ``n * (n + 1) / 2``.

    Raises:
        ValueError: if ``values`` fails :func:`as_float_array` validation.
    """
    array = as_float_array(values, name="values")
    size = array.size
    order = np.argsort(array, kind="stable")
    ordered = array[order]
    ranks = np.empty(size, dtype=np.float64)
    start = 0
    while start < size:
        stop = start + 1
        while stop < size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks
