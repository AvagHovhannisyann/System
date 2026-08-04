"""Constructed price paths whose correct label is derivable by hand.

**These are test scaffolding, not a data source.** Invariant I3 forbids
presenting generated numbers as real market data or stubbing a function to
return plausible values; it does not forbid feeding a function a path whose
answer is known in advance, which is the only way to prove that a barrier
engine touches the barrier it should. Nothing here is importable from
production code, nothing is written to a store, and no function in this module
claims to be a quote.

Every path is built from an explicit list of per-bar log returns, so the
cumulative return at any bar is a sum the reader can do in their head:
``prices_from_log_returns([0.02, 0.02])`` reaches ``+0.04`` cumulative log
return at bar 2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from backend.labels._arrays import FloatArray

# Two returns of +/-0.01 have sample standard deviation |r1 - r2| / sqrt(2)
# = 0.01 * sqrt(2). Used as the warm-up for every hand-derived case so that the
# barrier width is a number the test can state in closed form.
WARMUP_RETURNS: tuple[float, ...] = (0.01, -0.01)
WARMUP_VOLATILITY: float = 0.01 * np.sqrt(2.0)
WARMUP_WINDOW: int = 2
FIRST_EVENT_BAR: int = 2
"""Earliest labellable bar for ``WARMUP_RETURNS``: two returns need two bars."""


def prices_from_log_returns(
    log_returns: npt.ArrayLike, *, initial_price: float = 100.0
) -> FloatArray:
    """Build a close-price series from per-bar log returns.

    Args:
        log_returns: log return of each bar after the first (fractions).
        initial_price: price of bar 0, in any currency.

    Returns:
        ``len(log_returns) + 1`` prices. Bar 0 is ``initial_price``; bar ``t``
        is ``initial_price * exp(sum(log_returns[:t]))``.
    """
    cumulative = np.concatenate(([0.0], np.cumsum(np.asarray(log_returns, dtype=np.float64))))
    return np.asarray(initial_price * np.exp(cumulative), dtype=np.float64)


def hand_derived_path(forward_returns: npt.ArrayLike) -> FloatArray:
    """Build a warm-up plus a chosen forward path.

    The warm-up is :data:`WARMUP_RETURNS`, so an event placed at
    :data:`FIRST_EVENT_BAR` has trailing volatility exactly
    :data:`WARMUP_VOLATILITY` under a 2-bar volatility window, and the forward
    returns are the cumulative path the barriers are tested against.

    Args:
        forward_returns: per-bar log returns after the event (fractions).

    Returns:
        The close-price series (length ``2 + len(forward_returns) + 1``).
    """
    forward = np.asarray(forward_returns, dtype=np.float64).ravel()
    return prices_from_log_returns(np.concatenate((np.asarray(WARMUP_RETURNS), forward)))


def wave_log_returns(
    n_bars: int,
    *,
    scale: float = 1.0,
    phase: float = 0.0,
    frequency: float = 0.7,
    second_frequency: float = 0.31,
) -> FloatArray:
    """A deterministic, varied return path with no flat trailing window.

    Two incommensurate sinusoids, so consecutive returns are never equal (no
    zero-volatility window), the path has no drift, and the whole series is
    reproducible from its arguments with no random state anywhere — which
    matters for a lookahead test, where the *same* series must be rebuilt after
    truncation.

    **Vary the frequencies, not only the phase, when you need independent
    series.** A phase shift of a fixed pair of sinusoids stays inside the same
    two-dimensional function space, so three phase-shifted waves are linearly
    *dependent*: an "idiosyncratic" term built that way is an exact linear
    combination of the two factors, the regression residual is zero, and a test
    using it measures nothing. (This is not hypothetical — the first draft of
    the residualization tests did exactly that, and it was
    :class:`~backend.labels.errors.DegenerateVolatilityError` that caught it.)

    Args:
        n_bars: number of per-bar returns to generate (count). Callers feed the
            result to :func:`prices_from_log_returns`, which prepends bar 0.
        scale: multiplies every return (dimensionless).
        phase: phase offset in radians.
        frequency: angular frequency of the dominant sinusoid (radians per bar).
        second_frequency: angular frequency of the smaller sinusoid.

    Returns:
        ``n_bars`` per-bar log returns (fractions).
    """
    index = np.arange(n_bars, dtype=np.float64)
    wave = 0.012 * np.sin(frequency * index + phase) + 0.006 * np.cos(
        second_frequency * index + 1.0 + phase
    )
    return np.asarray(scale * wave, dtype=np.float64)


def factor_series(
    n_bars: int,
    *,
    beta_market: float = 1.4,
    beta_sector: float = 0.6,
    idiosyncratic_scale: float = 0.35,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Asset, market and sector closes with a genuine two-factor structure.

    The three underlying return series use **disjoint frequency pairs**, so they
    are linearly independent and the asset's idiosyncratic component genuinely
    survives the regression.

    Args:
        n_bars: number of per-bar returns (the price series is one bar longer).
        beta_market: the asset's loading on the market factor (dimensionless).
        beta_sector: the asset's loading on the sector factor (dimensionless).
        idiosyncratic_scale: scale of the asset's own component
            (dimensionless multiplier on the wave amplitude).

    Returns:
        ``(asset_close, market_close, sector_close)``.
    """
    market = wave_log_returns(n_bars, scale=0.8, frequency=0.7, second_frequency=0.31)
    sector = wave_log_returns(n_bars, scale=0.6, phase=1.7, frequency=0.23, second_frequency=1.10)
    idiosyncratic = wave_log_returns(
        n_bars, scale=idiosyncratic_scale, phase=3.9, frequency=0.41, second_frequency=1.90
    )
    asset = beta_market * market + beta_sector * sector + idiosyncratic
    return (
        prices_from_log_returns(asset),
        prices_from_log_returns(market),
        prices_from_log_returns(sector),
    )


def barrier_width(volatility: float, horizon: int, multiple: float = 1.0) -> float:
    """Restate the barrier formula independently of the implementation.

    Deliberately a second copy of ``multiple * sigma * sqrt(horizon)``: a test
    that asks the implementation for the barrier it used cannot detect the
    implementation changing the formula.

    Args:
        volatility: per-bar log-return standard deviation (a fraction).
        horizon: bars to the vertical barrier (count).
        multiple: barrier width in units of ``sigma * sqrt(horizon)``.

    Returns:
        The barrier distance as a non-negative cumulative log return.
    """
    return multiple * volatility * float(np.sqrt(horizon))
