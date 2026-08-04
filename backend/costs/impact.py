"""Square-root market impact, in basis points of notional (P9.3).

**The model.** Impact is the price the order moves against itself while it
executes. The empirically robust form — Almgren et al. (2005), Grinold & Kahn,
and the wider "square-root law" literature — is

::

    impact_bps = impact_coefficient * daily_volatility_bps * participation ** 0.5

where ``participation = order_notional / average_daily_dollar_volume`` is
dimensionless. Each factor is named for its unit:

- ``impact_coefficient`` is **dimensionless**. It is the multiplier on
  volatility, commonly estimated in the 0.5-1.0 range.
- ``daily_volatility_bps`` is the name's **daily return standard deviation in
  basis points** — 200 means 2% per day, not 0.02 and not 200%.
- ``participation`` is a **fraction**: ``0.01`` is 1% of ADV, ``1.0`` is the
  entire day's volume.

So the coefficient times the daily volatility *is* the modelled impact of
trading one full day's volume, and everything smaller scales as the square
root of participation.

**Why the exponent is a named constant.** Getting the exponent wrong is the
silent failure of this component: a linear model and a square-root model agree
exactly at 100% of ADV and disagree by a factor of ten at 1%, which is the
regime every real order lives in. :data:`IMPACT_EXPONENT` is exported so the
property tests assert against the same constant the arithmetic uses, and so a
reader can see that it is 0.5 without reading the formula.
"""

from __future__ import annotations

import math
from typing import Final

from backend.costs._validate import require_finite, require_non_negative, require_positive

__all__ = [
    "IMPACT_EXPONENT",
    "participation_rate",
    "square_root_impact_bps",
]

IMPACT_EXPONENT: Final = 0.5
"""Exponent on the participation rate in the impact law. Dimensionless.

0.5 is the square-root law. It is a constant rather than a parameter because
changing it is a change of model, not a recalibration, and it should be a
visible edit with a decision behind it.
"""


def participation_rate(*, notional_usd: float, adv_usd: float) -> float:
    """Order size as a fraction of average daily dollar volume.

    Args:
        notional_usd: absolute traded notional in US dollars. Sign is ignored:
            a sell of $1m participates in the day's volume exactly as much as
            a buy of $1m.
        adv_usd: average daily dollar volume for the same name, in US dollars.
            Must be strictly positive — a name with no volume has no defined
            participation rate, and substituting zero or infinity would put an
            unmodellable order into a backtest at a finite cost.

    Returns:
        ``|notional_usd| / adv_usd``, dimensionless. ``0.01`` means the order
        is 1% of a day's volume; ``1.0`` means it is the whole day. Values
        above ``1.0`` are permitted and are extrapolation — the square-root law
        is estimated on participation rates well below 1, and the model does
        not pretend otherwise.

    Raises:
        CostParameterError: if ``notional_usd`` is non-finite, or ``adv_usd``
            is non-finite or non-positive.
    """
    # Finiteness only, not sign: the docstring says the sign is ignored, and
    # `require_non_negative` on an absolute value could never fail its sign
    # check, which would read as a guard that is not one.
    require_finite("notional_usd", notional_usd)
    require_positive("adv_usd", adv_usd)
    return abs(notional_usd) / adv_usd


def square_root_impact_bps(
    *,
    participation: float,
    daily_volatility_bps: float,
    impact_coefficient: float,
) -> float:
    """Market impact of an order, in basis points of its own notional.

    Computes ``impact_coefficient * daily_volatility_bps * participation ** 0.5``
    (see :data:`IMPACT_EXPONENT`).

    Args:
        participation: order notional divided by average daily dollar volume.
            **Dimensionless fraction** — ``0.01`` is 1% of ADV. Must be finite
            and non-negative.
        daily_volatility_bps: the name's daily return standard deviation in
            **basis points** — ``200.0`` means 2% per day. Must be finite and
            non-negative.
        impact_coefficient: **dimensionless** multiplier on volatility. Must be
            finite and non-negative.

    Returns:
        Impact in basis points of the order's notional. Zero when the order is
        zero-sized, and monotonically increasing in participation thereafter.

    Raises:
        CostParameterError: if any argument is non-finite or negative.
    """
    require_non_negative("participation", participation)
    require_non_negative("daily_volatility_bps", daily_volatility_bps)
    require_non_negative("impact_coefficient", impact_coefficient)
    return impact_coefficient * daily_volatility_bps * math.pow(participation, IMPACT_EXPONENT)
