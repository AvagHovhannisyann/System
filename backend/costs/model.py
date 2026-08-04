"""The composed transaction cost model and its conservative defaults (P9.3).

::

    total_bps = half_spread_bps          # cross to the touch, one way
              + commission_bps           # broker commission and fees
              + impact_bps               # coefficient * daily_vol_bps * sqrt(size/ADV)
              + borrow_bps               # short financing, accrued over the holding period

**One internal unit.** Every quantity in this module is in **basis points of
traded notional** (1 bp = 1e-4 of notional). Dollars appear only at the
boundary, through :mod:`backend.costs.units`. Every field and argument name
carries its unit — ``half_spread_bps``, ``notional_usd``,
``borrow_rate_bps_per_year`` — because directive §8 identifies unit confusion
as the most common and most silent bug class in this domain, and a parameter
called ``spread`` is an invitation to it.

**The defaults are UNCALIBRATED and say so.** :data:`UNCALIBRATED_DEFAULTS`
carries ``uncalibrated=True`` and a ``calibration_basis`` string, and every
:class:`TradeCost` it produces carries both forward. A backtest built on these
numbers can therefore state, from the artifact itself and without consulting
prose, that its costs are assumptions rather than measurements. The values are
chosen at or above the pessimistic end of the published ranges — see each
field's documentation on :class:`CostModelParams` for the number and its
justification.

**Calibration is fenced, per DECISIONS.md D-013.** Paper fills are a *lower
bound* on slippage, never an estimate: IBKR paper fills at the touch far more
readily than reality and models no queue position. This module therefore
refuses to construct a parameter set marked ``uncalibrated=False`` whose
components are cheaper than the conservative defaults
(:class:`~backend.costs.errors.CostCalibrationError`). Optimistic fills cannot
silently tighten the model. P11.8 owns the calibration itself, including the
documented haircut; nothing here fits anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.costs._validate import require_non_negative, require_positive
from backend.costs.borrow import borrow_cost_bps
from backend.costs.errors import CostCalibrationError, CostParameterError
from backend.costs.impact import participation_rate, square_root_impact_bps
from backend.costs.units import bps_of_notional_to_usd

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "UNCALIBRATED_BASIS",
    "UNCALIBRATED_DEFAULTS",
    "CostModelParams",
    "Order",
    "Side",
    "TradeCost",
    "estimate_trade_cost",
    "estimate_trade_costs",
]

UNCALIBRATED_BASIS: Final = (
    "UNCALIBRATED — conservative literature defaults for US large/mid-cap equities; "
    "no parameter has been fitted to any observed fill (DECISIONS.md D-013)"
)
"""The ``calibration_basis`` string carried by the shipped defaults.

Displayed verbatim wherever a cost-derived number is reported, so "net of
costs" never reads as "net of *measured* costs".
"""

_COST_FIELDS: Final = (
    "half_spread_bps",
    "commission_bps",
    "impact_coefficient",
    "default_daily_volatility_bps",
    "borrow_rate_bps_per_year",
)


class Side(StrEnum):
    """Direction of an order.

    The side does not enter the cost arithmetic at all: spread, commission and
    impact are symmetric between buying and selling, and only the borrow term
    distinguishes a short. It is carried on :class:`Order` so a cost estimate
    is self-describing in an audit trail, and so the buy/sell symmetry is a
    property that can be tested rather than an absence that can be assumed.
    """

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class CostModelParams:
    """Parameters of the transaction cost model, with units in every name.

    Defaults are conservative and **uncalibrated**. Each justification below is
    for a US large/mid-cap equity universe, which is the universe the directive
    describes.

    Attributes:
        half_spread_bps: one-way cost of crossing from the mid to the touch, in
            basis points of notional. Default ``5.0`` — a 10 bps full quoted
            spread. S&P 500 megacaps quote 1-3 bps; mid caps commonly quote
            10-30 bps. 10 bps sits at the wide end for a liquid universe and
            assumes every fill crosses (no passive execution, no price
            improvement), which is the D-013 direction of error.
        commission_bps: broker commission plus exchange and regulatory fees, in
            basis points of notional. Default ``1.0``. IBKR tiered pricing is
            about $0.0035/share, fixed about $0.005/share; on a $50 share that
            is 0.7-1.0 bps. The SEC Section 31 fee adds roughly 0.28 bps on
            sales, and FINRA TAF a little more. 1.0 bp covers commission and
            fees for typical share prices. **Known limitation:** a per-share
            commission expressed in bps rises as the share price falls, so this
            understates costs for low-priced names; modelling commission
            per-share requires share counts this model does not take.
        impact_coefficient: dimensionless multiplier on daily volatility in the
            square-root impact law. Default ``1.0``. Published estimates of
            this coefficient cluster in 0.5-1.0 (Almgren et al. 2005;
            Grinold & Kahn); 1.0 is the top of that range.
        default_daily_volatility_bps: daily return standard deviation used when
            an order does not carry its own, in basis points. Default ``200.0``
            (2% per day, about 32% annualized). Typical US single-name daily
            vol is 1.5-2.5%. Because volatility multiplies impact, a higher
            value is the conservative one; per-name volatility should be passed
            on the order whenever it is available.
        borrow_rate_bps_per_year: annualized stock borrow fee applied to short
            positions, in basis points of position value. Default ``100.0``
            (1% per year). General-collateral US large caps run 25-50 bps, so
            this is 2-4x the typical GC rate. **Known limitation:** it is far
            below the several hundred to several thousand basis points a
            hard-to-borrow name costs; a short book that can hold HTB names
            needs per-name borrow data, and this default will understate it.
        uncalibrated: ``True`` while no parameter has been fitted to observed
            fills. Carried onto every :class:`TradeCost` so any consumer can
            surface it. Setting it to ``False`` is fenced — see
            :class:`~backend.costs.errors.CostCalibrationError`.
        calibration_basis: human-readable statement of where these numbers came
            from, displayed alongside any cost-derived result. Must be
            non-empty.
    """

    half_spread_bps: float = 5.0
    commission_bps: float = 1.0
    impact_coefficient: float = 1.0
    default_daily_volatility_bps: float = 200.0
    borrow_rate_bps_per_year: float = 100.0
    uncalibrated: bool = True
    calibration_basis: str = UNCALIBRATED_BASIS

    def __post_init__(self) -> None:
        """Validate units, signs, and — for calibrated sets — the D-013 floor.

        Raises:
            CostParameterError: if any parameter is non-finite or negative, or
                if ``calibration_basis`` is blank.
            CostCalibrationError: if ``uncalibrated`` is ``False`` and any
                parameter is cheaper than the conservative default. Per D-013,
                paper fills bound slippage from below, so a calibration may
                only make costs more expensive.
        """
        for field_name in _COST_FIELDS:
            require_non_negative(field_name, float(getattr(self, field_name)))
        if not self.calibration_basis.strip():
            msg = (
                "calibration_basis must state where these parameters came from; "
                "an empty basis makes an uncalibrated cost indistinguishable from "
                "a measured one"
            )
            raise CostParameterError(msg)
        if not self.uncalibrated:
            self._enforce_conservative_floor()

    def _enforce_conservative_floor(self) -> None:
        """Refuse a calibrated parameter set that is cheaper than the defaults."""
        floor = UNCALIBRATED_DEFAULTS
        for field_name in _COST_FIELDS:
            value = float(getattr(self, field_name))
            limit = float(getattr(floor, field_name))
            if value < limit:
                raise CostCalibrationError(parameter=field_name, value=value, floor=limit)
        if self.calibration_basis == UNCALIBRATED_BASIS:
            msg = (
                "a parameter set marked calibrated must state its own calibration "
                "basis, not the uncalibrated default string"
            )
            raise CostParameterError(msg)

    def with_parameters(self, **overrides: float) -> CostModelParams:
        """Return a copy with numeric parameters overridden, keeping the flags.

        The only sanctioned way to adjust a parameter set. It deliberately
        cannot change ``uncalibrated`` or ``calibration_basis``: marking a set
        calibrated goes through the fenced constructor (P11.8's job), so no
        code path can widen the calibration claim as a side effect of adjusting
        a number. Sensitivity analysis on an uncalibrated set stays
        uncalibrated, however the numbers move.

        Args:
            **overrides: any of the numeric parameter fields, in that field's
                documented units.

        Returns:
            A new :class:`CostModelParams`, validated the same way — including
            the D-013 floor if this set is marked calibrated.

        Raises:
            CostParameterError: if an unknown field is named, or a value fails
                validation.
            CostCalibrationError: if this set is calibrated and an override
                takes a parameter below the conservative floor.
        """
        unknown = set(overrides) - set(_COST_FIELDS)
        if unknown:
            msg = (
                f"unknown cost parameter(s) {sorted(unknown)}; "
                f"expected any of {sorted(_COST_FIELDS)}"
            )
            raise CostParameterError(msg)
        values = {name: float(getattr(self, name)) for name in _COST_FIELDS}
        values.update(overrides)
        return CostModelParams(
            half_spread_bps=values["half_spread_bps"],
            commission_bps=values["commission_bps"],
            impact_coefficient=values["impact_coefficient"],
            default_daily_volatility_bps=values["default_daily_volatility_bps"],
            borrow_rate_bps_per_year=values["borrow_rate_bps_per_year"],
            uncalibrated=self.uncalibrated,
            calibration_basis=self.calibration_basis,
        )


UNCALIBRATED_DEFAULTS: Final = CostModelParams()
"""The shipped conservative defaults, and the floor a calibration may not go below.

Also the default argument of :func:`estimate_trade_cost`, so a caller who
supplies nothing gets pessimistic, self-labelling costs rather than free ones.
"""


@dataclass(frozen=True, slots=True)
class Order:
    """One order to be costed.

    Attributes:
        side: :class:`Side.BUY` or :class:`Side.SELL`. Does not affect the
            cost; see :class:`Side`.
        notional_usd: traded notional in **US dollars**, non-negative. Pass the
            absolute value; direction lives in ``side``.
        adv_usd: average daily dollar volume for the name, in **US dollars**,
            strictly positive. The denominator of the participation rate.
        daily_volatility_bps: the name's daily return standard deviation in
            **basis points** (``200.0`` is 2% per day). ``None`` falls back to
            :attr:`CostModelParams.default_daily_volatility_bps`; pass the real
            per-name figure whenever it is available.
        is_short_position: ``True`` if this order establishes or maintains a
            **short** position, so borrow accrues on it. The closing buy of a
            short passes ``False``: borrow is charged once, on the leg that
            holds the borrow, not on both legs of the round trip.
        holding_period_days: days the short position is held, in **days**
            (fractional allowed). Only used when ``is_short_position`` is
            ``True``, and **required** to be strictly positive there. A short
            with an unstated horizon would silently drop its financing cost,
            which is an understatement in the one direction D-013 forbids;
            since the platform is daily-rebalanced and intraday strategies are
            out of scope (directive §1.1), every short in this system is held
            at least overnight, so a zero-day short is a caller mistake rather
            than a real trade. The horizon is an assumption either way — this
            makes it a stated one.
    """

    side: Side
    notional_usd: float
    adv_usd: float
    daily_volatility_bps: float | None = None
    is_short_position: bool = False
    holding_period_days: float = 0.0

    def __post_init__(self) -> None:
        """Validate sizes, volumes and horizons.

        Raises:
            CostParameterError: if ``notional_usd`` or ``holding_period_days``
                is non-finite or negative, if ``adv_usd`` is non-finite or
                non-positive, if ``daily_volatility_bps`` is supplied and is
                non-finite or negative, or if a short position does not state a
                strictly positive holding period.
        """
        require_non_negative("notional_usd", self.notional_usd)
        require_positive("adv_usd", self.adv_usd)
        require_non_negative("holding_period_days", self.holding_period_days)
        if self.daily_volatility_bps is not None:
            require_non_negative("daily_volatility_bps", self.daily_volatility_bps)
        if self.is_short_position and self.holding_period_days <= 0.0:
            msg = (
                "a short position must state its holding_period_days (> 0): borrow is a "
                "holding cost, and defaulting the horizon to zero would report the short "
                "book net of everything except its financing. Pass the intended horizon "
                "— the rebalance interval is the usual choice — so the assumption is "
                "visible in the estimate rather than implied by its absence."
            )
            raise CostParameterError(msg)


@dataclass(frozen=True, slots=True)
class TradeCost:
    """The modelled cost of one order, broken out by component.

    Every ``*_bps`` field is in **basis points of the order's own traded
    notional**; ``total_usd`` is the same total expressed in **US dollars**.
    The breakdown is kept rather than collapsed so that the operator UI (P9.5)
    and any backtest attribution can show *which* component dominates — a
    strategy killed by impact needs smaller orders, one killed by borrow needs
    a different short book, and a single total cannot tell them apart.

    Attributes:
        half_spread_bps: cost of crossing to the touch (bps of notional).
        commission_bps: broker commission and fees (bps of notional).
        impact_bps: square-root market impact (bps of notional).
        borrow_bps: short financing accrued over the holding period (bps of
            notional). Zero for long orders and for the closing leg of a short.
        total_bps: sum of the four components (bps of notional).
        total_usd: ``total_bps`` applied to ``notional_usd`` (US dollars).
        participation: order notional over average daily dollar volume,
            dimensionless (``0.01`` is 1% of ADV).
        notional_usd: the traded notional the costs are quoted on (US dollars).
        side: the order's side, carried through for audit.
        uncalibrated: ``True`` if the parameters used have never been fitted to
            observed fills. Propagated from
            :attr:`CostModelParams.uncalibrated` so a consumer never has to
            reach back to the parameter set to find out.
        calibration_basis: the parameter set's statement of provenance.
    """

    half_spread_bps: float
    commission_bps: float
    impact_bps: float
    borrow_bps: float
    total_bps: float
    total_usd: float
    participation: float
    notional_usd: float
    side: Side
    uncalibrated: bool
    calibration_basis: str

    def summary(self) -> dict[str, float | str | bool]:
        """Return a JSON-safe breakdown for operator display and run artifacts.

        Returns:
            Mapping of every component (basis points of notional), the dollar
            total, the participation rate, and the calibration status. The
            ``units`` key states the unit explicitly so a consumer rendering
            the number cannot guess wrong.
        """
        return {
            "half_spread_bps": self.half_spread_bps,
            "commission_bps": self.commission_bps,
            "impact_bps": self.impact_bps,
            "borrow_bps": self.borrow_bps,
            "total_bps": self.total_bps,
            "total_usd": self.total_usd,
            "participation": self.participation,
            "notional_usd": self.notional_usd,
            "side": self.side.value,
            "uncalibrated": self.uncalibrated,
            "calibration_basis": self.calibration_basis,
            "units": "basis points of traded notional (1 bp = 1e-4); total_usd in USD",
        }


def estimate_trade_cost(
    order: Order,
    params: CostModelParams = UNCALIBRATED_DEFAULTS,
) -> TradeCost:
    """Estimate the all-in cost of one order.

    ``total_bps = half_spread_bps + commission_bps + impact_bps + borrow_bps``,
    where impact is ``impact_coefficient * daily_volatility_bps *
    (notional_usd / adv_usd) ** 0.5`` and borrow is
    ``borrow_rate_bps_per_year * holding_period_days / 360`` on short positions
    only.

    A zero-notional order costs zero in **both** units: nothing traded, so no
    spread was crossed and no commission was charged. Cost in basis points is
    consequently discontinuous at zero — arbitrarily small orders pay the fixed
    rates — while cost in dollars is continuous. That is the honest shape, and
    it is why both are reported.

    **What this model omits**, all of it in the optimistic direction and none of
    it hidden: no cross-impact between simultaneously traded correlated names
    (see :func:`estimate_trade_costs`); a single universe-wide half-spread
    rather than a per-name quoted spread; commission in basis points rather than
    per share, which understates low-priced names; and a single borrow rate
    rather than per-name rates, which understates hard-to-borrow names. Each of
    these pushes the estimate *down*; the conservative defaults push it up. The
    two do not cancel in any principled way, so the total is neither a bound nor
    a measurement — it is an assumption, which is what ``uncalibrated`` says.

    Args:
        order: the order to cost. Units are documented on :class:`Order`.
        params: the cost parameters. Defaults to
            :data:`UNCALIBRATED_DEFAULTS`, whose ``uncalibrated`` flag
            propagates to the result.

    Returns:
        A :class:`TradeCost` with the component breakdown in basis points of
        the order's notional, the dollar total, and the calibration status of
        the parameters used.

    Raises:
        CostParameterError: propagated from :class:`Order` or
            :class:`CostModelParams` validation.
    """
    if order.notional_usd == 0.0:
        return TradeCost(
            half_spread_bps=0.0,
            commission_bps=0.0,
            impact_bps=0.0,
            borrow_bps=0.0,
            total_bps=0.0,
            total_usd=0.0,
            participation=0.0,
            notional_usd=0.0,
            side=order.side,
            uncalibrated=params.uncalibrated,
            calibration_basis=params.calibration_basis,
        )

    participation = participation_rate(notional_usd=order.notional_usd, adv_usd=order.adv_usd)
    volatility_bps = (
        params.default_daily_volatility_bps
        if order.daily_volatility_bps is None
        else order.daily_volatility_bps
    )
    impact_bps = square_root_impact_bps(
        participation=participation,
        daily_volatility_bps=volatility_bps,
        impact_coefficient=params.impact_coefficient,
    )
    borrow_bps = (
        borrow_cost_bps(
            borrow_rate_bps_per_year=params.borrow_rate_bps_per_year,
            holding_period_days=order.holding_period_days,
        )
        if order.is_short_position
        else 0.0
    )
    total_bps = params.half_spread_bps + params.commission_bps + impact_bps + borrow_bps

    return TradeCost(
        half_spread_bps=params.half_spread_bps,
        commission_bps=params.commission_bps,
        impact_bps=impact_bps,
        borrow_bps=borrow_bps,
        total_bps=total_bps,
        total_usd=bps_of_notional_to_usd(total_bps, order.notional_usd),
        participation=participation,
        notional_usd=order.notional_usd,
        side=order.side,
        uncalibrated=params.uncalibrated,
        calibration_basis=params.calibration_basis,
    )


def estimate_trade_costs(
    orders: Iterable[Order],
    params: CostModelParams = UNCALIBRATED_DEFAULTS,
) -> Sequence[TradeCost]:
    """Estimate costs for a basket of orders, one :class:`TradeCost` each.

    Costs are independent across names: this model has no cross-impact term,
    and a basket traded simultaneously in correlated names will cost more than
    the sum reported here. That omission is in the optimistic direction, so it
    is stated rather than hidden.

    Args:
        orders: the orders to cost.
        params: the cost parameters. Defaults to
            :data:`UNCALIBRATED_DEFAULTS`.

    Returns:
        A tuple of :class:`TradeCost`, in the order the orders were given.
    """
    return tuple(estimate_trade_cost(order, params) for order in orders)
