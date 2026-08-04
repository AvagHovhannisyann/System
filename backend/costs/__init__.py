"""Transaction cost model: spread, commission, impact, borrow (P9.3).

Invariant I4 says no backtest result is ever reported gross. This package is
what "net" means: an all-in per-order cost estimate composed of a half-spread
crossed on every fill, broker commission and fees, square-root market impact
scaled by the order's share of average daily volume, and stock borrow accrued
on short positions.

**Units.** The internal unit throughout is **basis points of traded notional**
(1 bp = 1e-4). Dollars enter and leave only through
:mod:`backend.costs.units`, and every parameter and field name states its unit
— ``half_spread_bps``, ``adv_usd``, ``borrow_rate_bps_per_year``,
``daily_volatility_bps``. Directive §8 calls unit confusion the most common and
most silent bug class in this domain; naming is the defence.

**Calibration status is part of the answer.** The shipped defaults
(:data:`~backend.costs.model.UNCALIBRATED_DEFAULTS`) are conservative
literature values, not measurements, and every :class:`~backend.costs.model.TradeCost`
they produce carries ``uncalibrated=True`` and a ``calibration_basis`` string.
A backtest can therefore declare, from its own artifacts, that its costs are
assumed. Per DECISIONS.md **D-013**, paper fills are a *lower bound* on
slippage rather than an estimate of it, so a parameter set may not be marked
calibrated while any component is cheaper than the conservative default —
:class:`~backend.costs.errors.CostCalibrationError` enforces that at
construction. P11.8 owns calibration and its documented haircut; nothing here
fits anything to anything.

Modules:

- :mod:`backend.costs.units` — the only place bps, fractions, percent and
  dollars meet;
- :mod:`backend.costs.spread` — quoted spread and the half of it one trade
  pays;
- :mod:`backend.costs.impact` — the square-root law and the participation rate
  it takes;
- :mod:`backend.costs.borrow` — annualized borrow accrued ACT/360;
- :mod:`backend.costs.model` — the parameters, the order, and the composed
  estimate;
- :mod:`backend.costs.errors` — the failure taxonomy, including the D-013
  calibration fence.
"""

from __future__ import annotations

from backend.costs.borrow import BORROW_DAY_COUNT_BASIS, borrow_cost_bps
from backend.costs.errors import CostCalibrationError, CostModelError, CostParameterError
from backend.costs.impact import IMPACT_EXPONENT, participation_rate, square_root_impact_bps
from backend.costs.model import (
    UNCALIBRATED_BASIS,
    UNCALIBRATED_DEFAULTS,
    CostModelParams,
    Order,
    Side,
    TradeCost,
    estimate_trade_cost,
    estimate_trade_costs,
)
from backend.costs.spread import (
    half_spread_bps_from_quoted_spread_bps,
    half_spread_bps_from_quotes,
    quoted_spread_bps_from_quotes,
)
from backend.costs.units import (
    BPS_PER_PERCENT,
    BPS_PER_UNIT,
    PERCENT_PER_UNIT,
    bps_of_notional_to_usd,
    bps_to_fraction,
    bps_to_percent,
    fraction_to_bps,
    percent_to_bps,
    usd_to_bps_of_notional,
)

__all__ = [
    "BORROW_DAY_COUNT_BASIS",
    "BPS_PER_PERCENT",
    "BPS_PER_UNIT",
    "IMPACT_EXPONENT",
    "PERCENT_PER_UNIT",
    "UNCALIBRATED_BASIS",
    "UNCALIBRATED_DEFAULTS",
    "CostCalibrationError",
    "CostModelError",
    "CostModelParams",
    "CostParameterError",
    "Order",
    "Side",
    "TradeCost",
    "borrow_cost_bps",
    "bps_of_notional_to_usd",
    "bps_to_fraction",
    "bps_to_percent",
    "estimate_trade_cost",
    "estimate_trade_costs",
    "fraction_to_bps",
    "half_spread_bps_from_quoted_spread_bps",
    "half_spread_bps_from_quotes",
    "participation_rate",
    "percent_to_bps",
    "quoted_spread_bps_from_quotes",
    "square_root_impact_bps",
    "usd_to_bps_of_notional",
]
