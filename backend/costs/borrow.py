"""Stock borrow cost on short positions, in basis points of notional (P9.3).

A short position pays a borrow fee for as long as it is held. Unlike spread,
commission and impact — which are one-off execution costs paid when the order
trades — borrow is a **holding** cost that accrues over time. It is included in
the trade-cost total because the directive's cost realism invariant (I4)
requires every reported number to be net of borrow, and because a short book
whose financing is omitted looks systematically better than it is.

**Units and day count.** The rate is quoted the way securities-lending desks
quote it: an **annualized rate in basis points of the position's market
value**, accrued daily. ``100.0`` means 1% per year, not 1 bp per year and not
100% per year. Accrual uses an **ACT/360** basis — the money-market convention
for US securities lending — so a rate held for a full 365 days accrues
``rate * 365 / 360``, slightly more than the quoted annual rate. Using 360
rather than 365 is both the market convention and the marginally more
conservative choice.
"""

from __future__ import annotations

from typing import Final

from backend.costs._validate import require_non_negative

__all__ = [
    "BORROW_DAY_COUNT_BASIS",
    "borrow_cost_bps",
]

BORROW_DAY_COUNT_BASIS: Final = 360.0
"""Days per year used to accrue a borrow rate. ACT/360, the US securities-lending convention."""


def borrow_cost_bps(
    *,
    borrow_rate_bps_per_year: float,
    holding_period_days: float,
) -> float:
    """Borrow cost accrued over a holding period, in basis points of notional.

    Computes ``borrow_rate_bps_per_year * holding_period_days /
    BORROW_DAY_COUNT_BASIS``.

    Args:
        borrow_rate_bps_per_year: annualized stock borrow fee in **basis
            points of position market value** — ``100.0`` is 1% per year.
            General-collateral US large caps typically run 25-50; hard-to-borrow
            names run from several hundred to several thousand. Must be finite
            and non-negative.
        holding_period_days: calendar days the short is held, in **days**
            (fractional days allowed). Must be finite and non-negative. Zero
            means the position is opened and closed within the accrual window
            and no fee accrues, which is the caller's assertion, not a default.

    Returns:
        Borrow cost in basis points of the position's notional. Zero if either
        input is zero.

    Raises:
        CostParameterError: if either argument is non-finite or negative. A
            negative borrow rate would model being *paid* to short, which
            happens only in special situations this model does not represent.
    """
    require_non_negative("borrow_rate_bps_per_year", borrow_rate_bps_per_year)
    require_non_negative("holding_period_days", holding_period_days)
    return borrow_rate_bps_per_year * holding_period_days / BORROW_DAY_COUNT_BASIS
