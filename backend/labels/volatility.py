"""Trailing volatility — the quantity that sizes every barrier (P6.1).

The single most important word in this module is **trailing**. A barrier sized
by the volatility realized *over the labelling window* is a lookahead bug of the
worst kind: it is not detectable in any distributional sanity check, it makes
the labels mildly prescient in exactly the periods where prediction is hard, and
the resulting backtest looks good. Every estimator here is a function of bars at
or before its own index and nothing else, and
:mod:`backend.tests.labels.test_lookahead` proves it by recomputation on
truncated series rather than by inspection.

--------------------------------------------------------------------------
Units and conventions
--------------------------------------------------------------------------

**Returns are natural-log returns, per bar, as a fraction.** ``r_t = ln(P_t) -
ln(P_{t-1})``. A 1% up-move is ``0.00995``, not ``1.0`` and not ``100``. Log
space is used throughout the package because barriers are then symmetric in the
quantity that actually behaves symmetrically: a ``+x`` and a ``-x`` log move are
exact reciprocals in price, whereas symmetric *simple*-return barriers are
asymmetric in log space and give the upper barrier a systematically higher
touch probability under a driftless random walk. That asymmetry would show up as
a spurious upward tilt in the label distribution.

**Volatility is a standard deviation of those log returns, per bar, as a
fraction.** ``sigma[t] = 0.02`` means a 2% daily standard deviation. It is
*not* annualized anywhere in this package; nothing here multiplies by
``sqrt(252)``.

**The estimator is a simple rolling sample standard deviation** over a fixed
window, about the sample mean, with ``ddof=1``. Two deliberate choices:

- *Fixed window rather than exponential weighting.* An EWMA (López de Prado uses
  ``ewm(span=100).std()``) has infinite memory, so "which bars went into this
  number" has no finite answer — inconvenient for a feature-availability audit
  and impossible to state in a docstring. A fixed window has an exact answer:
  bars ``t-window+1`` through ``t`` inclusive.
- *Full window required.* ``sigma[t]`` is ``NaN`` unless all ``window`` returns
  in the window are present and finite. No partial-window estimate is produced,
  because a 3-observation standard deviation and a 20-observation one are not
  the same statistic, and mixing them makes the barriers of early events
  incomparable with those of later ones — silently.

**Alignment.** ``sigma[t]`` uses returns ``r[t-window+1] … r[t]`` inclusive.
``r[t]`` is the return realized *into* bar ``t``, and is therefore known at
bar ``t``'s close, which is when an event at bar ``t`` is stamped. Including it
is not lookahead; excluding it would throw away a bar of information for no
reason. Since ``r[0]`` is undefined (there is no bar before the first), the
first finite ``sigma`` is at index ``window``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from backend.labels._arrays import as_float_1d, require_positive
from backend.labels.errors import LabelConfigurationError

if TYPE_CHECKING:
    import numpy.typing as npt

    from backend.labels._arrays import FloatArray

__all__ = [
    "daily_log_returns",
    "trailing_volatility",
]


def daily_log_returns(close: npt.ArrayLike) -> FloatArray:
    """Convert a close-price series to per-bar natural-log returns.

    Args:
        close: close prices, one per bar, in any currency. Must be finite and
            strictly positive at every bar; the logarithm of a zero or negative
            print is undefined and there is no defensible repair for it here.

    Returns:
        An array of the same length as ``close``. Element ``t`` is
        ``ln(close[t]) - ln(close[t-1])``, a dimensionless fraction (0.01 ≈ a
        1% move). Element ``0`` is ``NaN``: there is no bar before the first,
        and returning ``0.0`` there would be a fabricated observation that
        would then bias every volatility window overlapping it downward.

    Raises:
        LabelInputError: if ``close`` is not one-dimensional, not numeric, or
            contains a non-positive or non-finite price.
    """
    prices = as_float_1d(close, name="close")
    require_positive(prices, name="close")
    returns = np.empty_like(prices)
    returns[0] = np.nan
    if prices.size > 1:
        returns[1:] = np.diff(np.log(prices))
    return returns


def trailing_volatility(log_returns: npt.ArrayLike, *, window: int) -> FloatArray:
    """Rolling sample standard deviation of log returns, using trailing bars only.

    Args:
        log_returns: per-bar natural-log returns as fractions, aligned to the
            price series (element ``t`` is the return realized into bar ``t``).
            Typically the output of :func:`daily_log_returns`, whose first
            element is ``NaN``.
        window: number of trailing returns in each estimate (count of bars).
            Must be at least 2; a sample standard deviation with ``ddof=1``
            over one observation is ``0/0``.

    Returns:
        An array of the same length as ``log_returns``. Element ``t`` is the
        sample standard deviation (``ddof=1``, about the window's own mean) of
        ``log_returns[t-window+1 … t]`` inclusive — a per-bar volatility as a
        fraction, *not* annualized. Element ``t`` is ``NaN`` when fewer than
        ``window`` returns precede it or when any return in the window is
        ``NaN``; no partial-window estimate is ever produced.

    Raises:
        LabelConfigurationError: if ``window`` is below 2.
        LabelInputError: if ``log_returns`` is not a one-dimensional numeric
            array.

    Example:
        >>> import numpy as np
        >>> returns = np.array([np.nan, 0.01, -0.01, 0.01, -0.01])
        >>> np.round(trailing_volatility(returns, window=2), 6)
        array([     nan,      nan, 0.014142, 0.014142, 0.014142])
    """
    if window < 2:
        msg = (
            f"window must be >= 2; got {window}. A sample standard deviation with "
            f"ddof=1 over a single observation is undefined."
        )
        raise LabelConfigurationError(msg)

    returns = as_float_1d(log_returns, name="log_returns")
    volatility = np.full(returns.shape[0], np.nan, dtype=np.float64)
    if returns.shape[0] < window:
        return volatility

    # sliding_window_view is a stride view (no copy of the data); np.std then
    # reduces along the window axis with the stable two-pass algorithm.
    windows = np.lib.stride_tricks.sliding_window_view(returns, window)
    volatility[window - 1 :] = np.std(windows, axis=-1, ddof=1)
    return volatility
