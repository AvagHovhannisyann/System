"""Quoted spread and half-spread, in basis points of notional (P9.3).

The half-spread is the cost of crossing from the mid to the touch on one side.
A round trip crosses twice and therefore pays the full quoted spread — which
is why converting *quoted* spread to *half* spread happens through an
explicitly named function here rather than as a bare ``/ 2`` buried in the cost
arithmetic. A missing or doubled factor of two is a 100% error in the largest
fixed component of the model, and it is invisible in the output.

**Modelling stance.** Every fill is assumed to cross the spread. No passive
execution, no midpoint fills, no price improvement. That is deliberately
pessimistic and consistent with DECISIONS.md D-013: a cost model wrong in the
conservative direction loses money on paper, one wrong in the optimistic
direction loses it for real.

All returned values are in basis points **of the mid price**, which for an
order executed at the touch equals basis points of traded notional to first
order.
"""

from __future__ import annotations

from backend.costs._validate import require_non_negative, require_positive
from backend.costs.errors import CostParameterError
from backend.costs.units import fraction_to_bps

__all__ = [
    "half_spread_bps_from_quoted_spread_bps",
    "half_spread_bps_from_quotes",
    "quoted_spread_bps_from_quotes",
]


def quoted_spread_bps_from_quotes(*, bid: float, ask: float) -> float:
    """Quoted bid-ask spread in basis points of the mid price.

    Args:
        bid: best bid price, in any currency unit, strictly positive.
        ask: best ask price, same currency unit, at least ``bid``.

    Returns:
        ``(ask - bid) / mid`` in basis points, where ``mid = (bid + ask) / 2``.
        A $0.01 spread on a $50.00 mid returns ``2.0`` bps.

    Raises:
        CostParameterError: if either quote is non-finite or non-positive, or
            if the book is crossed (``ask < bid``). A crossed book is a data
            error; taking its absolute value would hide it inside a plausible
            cost.
    """
    require_positive("bid", bid)
    require_positive("ask", ask)
    if ask < bid:
        msg = f"crossed book: ask={ask!r} is below bid={bid!r}"
        raise CostParameterError(msg)
    mid = (bid + ask) / 2.0
    return fraction_to_bps((ask - bid) / mid)


def half_spread_bps_from_quotes(*, bid: float, ask: float) -> float:
    """One-way cost of crossing to the touch, in basis points of the mid price.

    Args:
        bid: best bid price, strictly positive.
        ask: best ask price, at least ``bid``.

    Returns:
        Half the quoted spread in basis points. A $0.01 spread on a $50.00 mid
        returns ``1.0`` bps: that is what one trade pays, and a round trip pays
        ``2.0``.

    Raises:
        CostParameterError: if the quotes are not usable (see
            :func:`quoted_spread_bps_from_quotes`).
    """
    return half_spread_bps_from_quoted_spread_bps(quoted_spread_bps_from_quotes(bid=bid, ask=ask))


def half_spread_bps_from_quoted_spread_bps(quoted_spread_bps: float) -> float:
    """Halve a quoted spread, in basis points.

    Args:
        quoted_spread_bps: full quoted bid-ask spread in basis points of the
            mid price. Must be finite and non-negative.

    Returns:
        The one-way half-spread in basis points.

    Raises:
        CostParameterError: if the quoted spread is non-finite or negative.
    """
    require_non_negative("quoted_spread_bps", quoted_spread_bps)
    return quoted_spread_bps / 2.0
