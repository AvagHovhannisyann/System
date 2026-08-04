"""Unit conversions for the cost model — the one place bps meets fractions (P9.3).

Directive §8: *"basis points versus percent versus fraction is the most common
bug class in this domain and it is silent"*. This module exists so that every
such conversion happens in exactly one named, tested place, and so that no
other module in :mod:`backend.costs` ever writes a bare ``10_000`` or ``/ 100``.

**The internal unit of the whole cost package is basis points of traded
notional.** One basis point is ``1e-4`` of notional: a 5 bps cost on a
$1,000,000 order is $500. Conversions to and from fractions, percent and
dollars happen at the package boundary and nowhere else.

Every function name states both the source and the destination unit. There is
deliberately no function called ``convert``.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "BPS_PER_PERCENT",
    "BPS_PER_UNIT",
    "PERCENT_PER_UNIT",
    "bps_of_notional_to_usd",
    "bps_to_fraction",
    "bps_to_percent",
    "fraction_to_bps",
    "percent_to_bps",
    "usd_to_bps_of_notional",
]

BPS_PER_UNIT: Final = 10_000.0
"""Basis points in one whole unit (100%, i.e. a fraction of 1.0)."""

PERCENT_PER_UNIT: Final = 100.0
"""Percent in one whole unit (a fraction of 1.0)."""

BPS_PER_PERCENT: Final = BPS_PER_UNIT / PERCENT_PER_UNIT
"""Basis points in one percent. Equals 100."""


def bps_to_fraction(bps: float) -> float:
    """Convert basis points to a dimensionless fraction.

    Args:
        bps: value in basis points (``5.0`` means 5 bps).

    Returns:
        The same value as a fraction (``5.0`` bps returns ``0.0005``).
    """
    return bps / BPS_PER_UNIT


def fraction_to_bps(fraction: float) -> float:
    """Convert a dimensionless fraction to basis points.

    Args:
        fraction: value as a fraction (``0.0005`` means 5 bps, ``1.0`` means
            100%).

    Returns:
        The same value in basis points (``0.0005`` returns ``5.0``).
    """
    return fraction * BPS_PER_UNIT


def percent_to_bps(percent: float) -> float:
    """Convert percent to basis points.

    Args:
        percent: value in percent (``0.05`` means 0.05%, i.e. 5 bps).

    Returns:
        The same value in basis points (``0.05`` returns ``5.0``).
    """
    return percent * BPS_PER_PERCENT


def bps_to_percent(bps: float) -> float:
    """Convert basis points to percent.

    Args:
        bps: value in basis points (``5.0`` means 5 bps).

    Returns:
        The same value in percent (``5.0`` returns ``0.05``).
    """
    return bps / BPS_PER_PERCENT


def bps_of_notional_to_usd(bps: float, notional_usd: float) -> float:
    """Convert a cost quoted in basis points of notional into US dollars.

    Args:
        bps: cost in basis points of the traded notional.
        notional_usd: absolute traded notional in US dollars. Sign is ignored
            — a cost is a cost whether the order buys or sells — so the caller
            may pass a signed notional without flipping the cost's sign.

    Returns:
        The cost in US dollars, always non-negative for a non-negative ``bps``.
    """
    return bps_to_fraction(bps) * abs(notional_usd)


def usd_to_bps_of_notional(cost_usd: float, notional_usd: float) -> float:
    """Convert a dollar cost into basis points of the notional it was paid on.

    Args:
        cost_usd: cost in US dollars.
        notional_usd: absolute traded notional in US dollars. Must be non-zero;
            a rate per unit of nothing is undefined, and returning ``0.0`` or
            ``inf`` would push that undefinedness downstream.

    Returns:
        The cost in basis points of notional.

    Raises:
        ZeroDivisionError: if ``notional_usd`` is zero.
    """
    denominator = abs(notional_usd)
    if denominator == 0.0:
        msg = "cannot express a cost in basis points of a zero notional"
        raise ZeroDivisionError(msg)
    return fraction_to_bps(cost_usd / denominator)
