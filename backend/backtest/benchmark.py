"""Buy-and-hold benchmark, and the portfolio accounting both sides share (P10.5).

Directive §5-P10 requires a "benchmark comparison against buy-and-hold index,
always shown alongside every strategy result". Two things follow from taking
that seriously.

**The benchmark pays costs too.** A buy-and-hold that gets invested for free is
not a baseline, it is a head start awarded to the strategy. The benchmark here
crosses the spread and pays commission and impact on its entry trade, and
accrues borrow on any short leg, using the same
:class:`~backend.costs.model.CostModelParams` the strategy used. It is then
never rebalanced — that is the whole of the strategy it represents — so its
weights drift with the market and it pays nothing further.

**The strategy and the benchmark share one accounting kernel.** The costing,
borrow accrual and weight-drift functions in this module are the same objects
:mod:`backend.backtest.engine` calls in its daily loop. That is deliberate and
it is why they live here rather than in the engine: if the two sides were
costed by two implementations, the first divergence between them would show up
as alpha. The engine is the active driver; this module is the passive baseline
plus the arithmetic they have in common, so it imports nothing from the engine
and the dependency runs one way.

Units, stated once:

* weights are **fractions of portfolio value** — ``0.05`` is 5% of the book,
  negative is short, and the unallocated remainder is cash earning **zero**
  (no cash rate is modelled; that omission makes results marginally worse in a
  positive-rate world, which is the direction to be wrong in);
* returns are simple per-period **fractions**;
* every ``*_usd`` quantity is **US dollars**;
* holding periods for borrow accrual are **calendar days**, ACT/360, matching
  :func:`backend.costs.borrow.borrow_cost_bps`.

**Borrow is charged as a holding cost, not on the trade.** The P9.3 cost model
can charge borrow on an order's notional over a stated horizon
(:attr:`~backend.costs.model.Order.is_short_position`). This module does not
use that path: a short established once and held for a year would then pay
borrow only on the notional it traded, and a position held without trading
would pay none at all — an understatement, which is the direction D-013
forbids. Instead every simulated period accrues borrow on the **short exposure
actually held** over that period, and orders are costed with
``is_short_position=False`` so nothing is counted twice.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.backtest.artifact import EquityCurve, RealizedAccounting
from backend.costs.borrow import borrow_cost_bps
from backend.costs.model import Order, Side, TradeCost, estimate_trade_cost
from backend.costs.units import bps_of_notional_to_usd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from backend.costs.model import CostModelParams

__all__ = [
    "SECONDS_PER_DAY",
    "BuyAndHoldSpec",
    "ExecutionCosts",
    "PortfolioError",
    "PortfolioWipeoutError",
    "WeightError",
    "accrue_borrow_usd",
    "buy_and_hold",
    "cost_weight_changes",
    "drift_weights",
    "gross_exposure",
    "validate_weights",
]

SECONDS_PER_DAY: Final = 86_400.0
"""Seconds in a calendar day, for converting an interval to the **days** that
:func:`backend.costs.borrow.borrow_cost_bps` accrues over. Defined once and
imported by :mod:`backend.backtest.engine` so the strategy and the benchmark
cannot come to disagree about the length of a holding period."""


class PortfolioError(Exception):
    """Base class for portfolio accounting failures."""


class WeightError(PortfolioError, ValueError):
    """Raised when a set of target weights is not usable.

    Covers non-finite weights, an unknown asset, and gross exposure above the
    configured cap. The cap defaults to 1.0 because leverage is an explicit
    non-goal (directive §1.1); a strategy that asks for more is refused rather
    than quietly simulated.
    """


class PortfolioWipeoutError(PortfolioError, ArithmeticError):
    """Raised when a simulated period wipes out the portfolio.

    A period return of ``-1`` or worse leaves no defined return afterwards.
    Continuing would produce an equity path through zero and a drawdown of
    100% followed by meaningless numbers, so the run stops and says so.
    """


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    """Execution costs of one basket of trades, broken out for a cost waterfall.

    Borrow is deliberately absent: it accrues on held exposure, not on traded
    notional, and is accumulated separately by :func:`accrue_borrow_usd`.

    Attributes:
        half_spread_usd: total paid crossing to the touch, US dollars.
        commission_usd: total broker commission and fees, US dollars.
        impact_usd: total modelled square-root market impact, US dollars.
        traded_notional_usd: total absolute notional traded, US dollars.
        n_orders: number of non-zero orders in the basket.
    """

    half_spread_usd: float = 0.0
    commission_usd: float = 0.0
    impact_usd: float = 0.0
    traded_notional_usd: float = 0.0
    n_orders: int = 0

    @property
    def total_usd(self) -> float:
        """Return the sum of the three execution components, in US dollars."""
        return self.half_spread_usd + self.commission_usd + self.impact_usd

    def __add__(self, other: ExecutionCosts) -> ExecutionCosts:
        """Return the componentwise sum of two cost baskets."""
        return ExecutionCosts(
            half_spread_usd=self.half_spread_usd + other.half_spread_usd,
            commission_usd=self.commission_usd + other.commission_usd,
            impact_usd=self.impact_usd + other.impact_usd,
            traded_notional_usd=self.traded_notional_usd + other.traded_notional_usd,
            n_orders=self.n_orders + other.n_orders,
        )


def gross_exposure(weights: Mapping[str, float]) -> float:
    """Return the sum of absolute weights, a dimensionless fraction of portfolio value.

    Args:
        weights: asset to weight, as fractions of portfolio value.

    Returns:
        ``sum(|w|)``. ``1.0`` is a fully invested unlevered book; ``2.0`` is a
        100/100 long-short book, which is leverage.
    """
    return float(sum(abs(weight) for weight in weights.values()))


def validate_weights(
    weights: Mapping[str, float],
    *,
    known_assets: frozenset[str],
    max_gross_exposure: float,
    context: str,
) -> dict[str, float]:
    """Validate target weights against the known universe and the leverage cap.

    Args:
        weights: asset to target weight, as fractions of portfolio value.
        known_assets: assets the caller could legitimately have observed at the
            decision instant. An asset outside this set is a lookahead: the
            decision references something that was not knowable.
        max_gross_exposure: cap on ``sum(|w|)``. 1.0 forbids leverage.
        context: short description of the caller, used in error messages.

    Returns:
        A plain ``dict`` copy of the weights with zero-weight entries dropped,
        so downstream arithmetic never carries positions that do not exist.

    Raises:
        WeightError: if a weight is non-finite, if an asset is not in
            ``known_assets``, or if gross exposure exceeds the cap.
    """
    cleaned: dict[str, float] = {}
    for asset, raw in weights.items():
        weight = float(raw)
        if not np.isfinite(weight):
            msg = f"{context}: weight for {asset!r} is not finite ({raw!r})"
            raise WeightError(msg)
        if asset not in known_assets:
            msg = (
                f"{context}: target weight references {asset!r}, which has no observation "
                "knowable at this instant. A position in an asset that cannot be seen yet "
                "is lookahead, so it is refused rather than simulated."
            )
            raise WeightError(msg)
        if weight != 0.0:
            cleaned[asset] = weight
    gross = gross_exposure(cleaned)
    if gross > max_gross_exposure + 1e-9:
        msg = (
            f"{context}: gross exposure {gross:.6f} exceeds the cap {max_gross_exposure:.6f}. "
            "Leverage is an explicit non-goal (directive §1.1); raise the cap deliberately "
            "or fix the strategy."
        )
        raise WeightError(msg)
    return cleaned


def cost_weight_changes(
    *,
    weight_changes: Mapping[str, float],
    portfolio_value_usd: float,
    adv_usd: Mapping[str, float],
    daily_volatility_bps: Mapping[str, float | None],
    params: CostModelParams,
) -> ExecutionCosts:
    """Cost a basket of weight changes with the P9.3 model.

    Every non-zero change becomes one :class:`~backend.costs.model.Order` whose
    notional is ``|delta| * portfolio_value_usd``. Orders are costed with
    ``is_short_position=False``: borrow is a holding cost here and is accrued by
    :func:`accrue_borrow_usd`, so charging it on the trade as well would double
    count. Arbitrarily small changes are costed rather than filtered by a
    minimum order size — filtering would understate costs, and understated
    costs are the failure mode invariant I4 exists to prevent.

    Args:
        weight_changes: asset to change in weight, as a fraction of portfolio
            value. Sign gives the side; magnitude gives the notional.
        portfolio_value_usd: the book's value the weights are fractions of, in
            US dollars. Must be positive.
        adv_usd: asset to average daily dollar volume, US dollars, strictly
            positive. Every traded asset must be present.
        daily_volatility_bps: asset to daily return standard deviation in basis
            points, or ``None`` to fall back to the model's default.
        params: the cost parameters to charge.

    Returns:
        The componentwise :class:`ExecutionCosts` of the basket.

    Raises:
        KeyError: if a traded asset has no ADV. Substituting a default would put
            an unmodellable order into the backtest at a finite cost.
        CostParameterError: propagated from the cost model for a non-positive
            ADV or a malformed order.
    """
    costs = ExecutionCosts()
    for asset, delta in sorted(weight_changes.items()):
        if delta == 0.0:
            continue
        notional = abs(float(delta)) * portfolio_value_usd
        if asset not in adv_usd:
            msg = (
                f"no average daily volume for {asset!r}; an order cannot be costed without "
                "it, and defaulting one would price an unmodellable trade"
            )
            raise KeyError(msg)
        estimate: TradeCost = estimate_trade_cost(
            Order(
                side=Side.BUY if delta > 0.0 else Side.SELL,
                notional_usd=notional,
                adv_usd=adv_usd[asset],
                daily_volatility_bps=daily_volatility_bps.get(asset),
                is_short_position=False,
            ),
            params,
        )
        costs = costs + ExecutionCosts(
            half_spread_usd=bps_of_notional_to_usd(estimate.half_spread_bps, notional),
            commission_usd=bps_of_notional_to_usd(estimate.commission_bps, notional),
            impact_usd=bps_of_notional_to_usd(estimate.impact_bps, notional),
            traded_notional_usd=notional,
            n_orders=1,
        )
    return costs


def accrue_borrow_usd(
    *,
    weights: Mapping[str, float],
    portfolio_value_usd: float,
    holding_period_days: float,
    params: CostModelParams,
) -> float:
    """Accrue borrow on the short exposure held over one period.

    Computes ``short_exposure * portfolio_value_usd`` times the ACT/360 accrual
    of :func:`backend.costs.borrow.borrow_cost_bps`. A single universe-wide rate
    is used because per-name borrow data does not exist in this system yet
    (BLOCKERS.md B1 rescoped the borrow feed onto IBKR in Phase 11); the model's
    default rate is several times general collateral but far below any
    hard-to-borrow name, and that limitation is documented on
    :class:`~backend.costs.model.CostModelParams`.

    Args:
        weights: asset to weight held over the period, as fractions of
            portfolio value. Only negative weights accrue borrow.
        portfolio_value_usd: the book's value over the period, US dollars.
        holding_period_days: length of the period in **calendar days**.
        params: the cost parameters supplying the borrow rate.

    Returns:
        Borrow cost for the period in US dollars, never negative.
    """
    short_exposure = float(sum(-weight for weight in weights.values() if weight < 0.0))
    if short_exposure == 0.0 or holding_period_days == 0.0:
        return 0.0
    accrued_bps = borrow_cost_bps(
        borrow_rate_bps_per_year=params.borrow_rate_bps_per_year,
        holding_period_days=holding_period_days,
    )
    return bps_of_notional_to_usd(accrued_bps, short_exposure * portfolio_value_usd)


def drift_weights(
    weights: Mapping[str, float],
    asset_returns: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    """Advance held weights by one period of asset returns.

    The portfolio return is ``sum(w_j * r_j)``; the remainder of the book is
    cash and earns zero. Each weight becomes ``w_j * (1 + r_j) / (1 + r_p)``,
    which is the position's share of the *new* portfolio value — this is what
    makes a fully invested single-asset book require no rebalancing trade, and
    getting it wrong manufactures turnover that costs money in the simulation
    and never happened.

    Args:
        weights: asset to weight held at the start of the period, as fractions
            of portfolio value.
        asset_returns: asset to simple return over the period, as a fraction.
            Must cover every held asset.

    Returns:
        ``(drifted_weights, portfolio_return)``. The portfolio return is a
        fraction of the pre-period value.

    Raises:
        KeyError: if a held asset has no return for the period.
        PortfolioWipeoutError: if the period return is ``-1`` or worse.
    """
    portfolio_return = 0.0
    for asset, weight in weights.items():
        if asset not in asset_returns:
            msg = f"no return for held asset {asset!r} over this period"
            raise KeyError(msg)
        portfolio_return += weight * asset_returns[asset]
    growth = 1.0 + portfolio_return
    if growth <= 0.0:
        msg = (
            f"portfolio return {portfolio_return:.6f} wipes out the book; there is no "
            "defined return after that point, so the simulation stops here"
        )
        raise PortfolioWipeoutError(msg)
    drifted = {
        asset: weight * (1.0 + asset_returns[asset]) / growth for asset, weight in weights.items()
    }
    return drifted, portfolio_return


@dataclass(frozen=True, slots=True)
class BuyAndHoldSpec:
    """The benchmark: what to buy on the first date, and never trade again.

    Attributes:
        weights: asset to weight, as fractions of initial capital. ``{"SPY":
            1.0}`` is a fully invested index position. The unallocated remainder
            is cash earning zero.
        description: what the benchmark is, in words, carried onto
            :class:`~backend.backtest.artifact.BenchmarkComparison` so a reader
            is never left guessing what "beat the benchmark" was measured
            against. Generated from the weights when left blank.
    """

    weights: Mapping[str, float]
    description: str = ""

    def __post_init__(self) -> None:
        """Freeze the weights and fill in a description if none was given.

        Raises:
            WeightError: if the weights are empty — a benchmark of nothing
                compares a strategy against cash without saying so — or if any
                weight is not finite. The leverage of the benchmark is checked
                where the run's cap is known, in
                :class:`~backend.backtest.engine.BacktestConfig`.
        """
        object.__setattr__(self, "weights", {asset: float(w) for asset, w in self.weights.items()})
        if not self.weights:
            msg = "a buy-and-hold benchmark must hold something; got no weights"
            raise WeightError(msg)
        for asset, weight in self.weights.items():
            if not np.isfinite(weight):
                msg = f"benchmark weight for {asset!r} is not finite ({weight!r})"
                raise WeightError(msg)
        if not self.description.strip():
            holdings = ", ".join(
                f"{asset} {weight:.1%}" for asset, weight in sorted(self.weights.items())
            )
            object.__setattr__(
                self,
                "description",
                (
                    f"buy-and-hold ({holdings}), purchased at the first date and never "
                    "rebalanced, net of its own entry cost and of borrow on any short leg"
                ),
            )


def buy_and_hold(
    *,
    spec: BuyAndHoldSpec,
    dates: Sequence[dt.datetime],
    asset_returns: Mapping[str, Sequence[float]],
    adv_usd: Mapping[str, float],
    daily_volatility_bps: Mapping[str, float | None],
    initial_capital_usd: float,
    params: CostModelParams,
) -> tuple[EquityCurve, RealizedAccounting]:
    """Simulate the buy-and-hold benchmark over the strategy's own dates.

    One entry trade at ``dates[0]``, costed exactly as the strategy's trades
    are. Thereafter the weights drift and no further execution cost is incurred;
    borrow accrues each period on whatever short exposure the drifted book
    holds.

    Args:
        spec: what to buy and hold.
        dates: the strategy's dates, timezone-aware UTC and strictly
            increasing. Length ``T + 1``. Passing the strategy's own calendar is
            what makes the comparison a comparison.
        asset_returns: asset to its ``T`` per-period simple returns, aligned so
            that entry ``i`` is the return over ``(dates[i], dates[i + 1]]``.
            Must cover every asset in ``spec.weights``.
        adv_usd: asset to average daily dollar volume at the entry date, US
            dollars.
        daily_volatility_bps: asset to daily volatility in basis points at the
            entry date, or ``None`` for the model default.
        initial_capital_usd: starting capital in US dollars, strictly positive
            and equal to the strategy's.
        params: the same cost parameters the strategy was charged.

    Returns:
        ``(equity_curve, accounting)`` — the net-of-cost equity path on
        ``dates`` and the exact realized quantities behind it.

    Raises:
        ValueError: if ``initial_capital_usd`` is not positive, or if a return
            series length disagrees with the calendar.
        KeyError: if a held asset lacks returns or ADV.
        PortfolioWipeoutError: if the entry cost exhausts the capital, or if a
            period wipes out the book.
    """
    if initial_capital_usd <= 0.0:
        msg = f"initial_capital_usd must be positive; got {initial_capital_usd!r}"
        raise ValueError(msg)
    n_periods = len(dates) - 1
    for asset in spec.weights:
        if asset not in asset_returns:
            msg = f"no return series for benchmark asset {asset!r}"
            raise KeyError(msg)
        if len(asset_returns[asset]) != n_periods:
            msg = (
                f"benchmark asset {asset!r} has {len(asset_returns[asset])} returns but the "
                f"calendar has {n_periods} periods"
            )
            raise ValueError(msg)

    weights = dict(spec.weights)
    entry = cost_weight_changes(
        weight_changes=weights,
        portfolio_value_usd=initial_capital_usd,
        adv_usd=adv_usd,
        daily_volatility_bps=daily_volatility_bps,
        params=params,
    )
    equity = [initial_capital_usd]
    value = initial_capital_usd - entry.total_usd
    if value <= 0.0:
        msg = (
            f"the benchmark's entry cost {entry.total_usd!r} exhausts its capital "
            f"{initial_capital_usd!r}; there is no equity path to report"
        )
        raise PortfolioWipeoutError(msg)
    borrow_total = 0.0
    for index in range(n_periods):
        holding_days = (dates[index + 1] - dates[index]).total_seconds() / SECONDS_PER_DAY
        borrow = accrue_borrow_usd(
            weights=weights,
            portfolio_value_usd=value,
            holding_period_days=holding_days,
            params=params,
        )
        borrow_total += borrow
        value -= borrow
        period_returns = {asset: float(series[index]) for asset, series in asset_returns.items()}
        weights, portfolio_return = drift_weights(weights, period_returns)
        value *= 1.0 + portfolio_return
        if value <= 0.0:
            msg = f"benchmark equity reached {value!r} at {dates[index + 1].isoformat()}"
            raise PortfolioWipeoutError(msg)
        equity.append(value)

    accounting = RealizedAccounting(
        initial_capital_usd=initial_capital_usd,
        final_equity_usd=equity[-1],
        half_spread_usd=entry.half_spread_usd,
        commission_usd=entry.commission_usd,
        impact_usd=entry.impact_usd,
        borrow_usd=borrow_total,
        total_cost_usd=entry.total_usd + borrow_total,
        traded_notional_usd=entry.traded_notional_usd,
        n_orders=entry.n_orders,
        n_rebalances=1,
    )
    curve = EquityCurve(
        dates=tuple(dates),
        equity_usd=np.asarray(equity, dtype=np.float64),
    )
    return curve, accounting
