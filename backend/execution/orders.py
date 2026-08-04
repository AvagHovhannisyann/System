"""Order value objects, and the reason a live order has no representation here.

Paper-only, structurally
------------------------

The platform is paper-trading only and permanently so (directive §1.1, §9.5).
This package makes that a property of the *types and the schema* rather than a
setting that happens to be off:

1. :class:`ExecutionVenue` has exactly **one** member. "Live" is not a disabled
   option, it is an absent one — there is no value to select, so no
   configuration, environment variable or feature flag can name it.
2. No constructor, function or persistence call in this package takes a venue
   argument. :class:`OrderIntent` does not accept one;
   :func:`backend.execution.store.record_order` does not accept one. The column
   carries a server default and a ``CHECK (venue = 'paper')`` constraint
   (migration 0014), so the value cannot be supplied by any writer at all,
   including raw SQL.
3. :class:`FillSource` has no live member either, and the same column-level
   CHECK restricts the persisted value to the two non-live literals. A fill
   originating anywhere other than a simulation or a paper account has no
   representation, so it cannot be recorded even by a writer that bypasses this
   code.
4. **There is no transport.** This package opens no socket, builds no client,
   reads no setting, and defines no adapter interface, registry or injection
   seam that an implementation could be selected into. Orders are persisted;
   venue messages arrive as events a caller applies. Nothing here can reach
   anything, so there is no slot for an endpoint — live or otherwise — to be
   dropped into. That is the structural claim, and
   ``backend/tests/execution/test_paper_only.py`` proves it by tokenizing this
   package's source.

Units
-----

Quantities are **whole shares** (integers; fractional shares are out of scope).
Prices are **US dollars per share**, carried as :class:`decimal.Decimal` with at
most :data:`PRICE_SCALE` decimal places, never as ``float`` — a binary float
cannot represent a limit price exactly, and an order's identity is hashed from
its text.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.costs.model import Side
from backend.execution.errors import OrderValidationError

if TYPE_CHECKING:
    from backend.tracking.stamp import ReproducibilityStamp

__all__ = [
    "PAPER_FILL_COST_BASIS",
    "PRICE_SCALE",
    "ExecutionVenue",
    "FillReport",
    "FillSource",
    "OrderIntent",
    "OrderType",
    "Side",
    "TimeInForce",
]

PRICE_SCALE: Final = 6
"""Decimal places a price may carry, matching ``Numeric(18, 6)`` in the schema.

A limit price with more precision than the column can hold would be rounded on
the way in, and the stored order would then disagree with the idempotency key
computed from the submitted one. Refused at construction instead.
"""

PAPER_FILL_COST_BASIS: Final = "lower_bound"
"""The only basis a fill recorded by this platform may claim (D-013).

IBKR paper fills are optimistic: they fill at the touch far more readily than
reality and model no queue position. Realised slippage measured from them is a
**lower bound** on the true cost, never an estimate of it. Every persisted fill
carries this string, and migration 0014 constrains the column to it, so a later
calibration (P11.8) cannot quietly relabel a paper fill as a measured estimate
without a migration and the review that comes with one.
"""


class ExecutionVenue(StrEnum):
    """Where an order is executed. Exactly one member, permanently.

    A single-member enum is the point, not an accident of the current phase.
    Directive §1.1 and §9.5 place live trading permanently out of scope, so this
    type exists to make "live" unrepresentable rather than merely unselected: a
    second member would have to be added by a code change and a migration, not
    by configuration.

    Nothing takes a venue as an argument. The value reaches the database as a
    column default under a ``CHECK (venue = 'paper')`` constraint, and the read
    side refuses a row carrying anything else
    (:class:`~backend.execution.errors.NotPaperOrderError`) rather than trusting
    the write side.
    """

    PAPER = "paper"


class OrderType(StrEnum):
    """How the order is priced at the venue.

    ``MARKET`` carries no price and takes whatever the book offers. ``LIMIT``
    carries a price and trades at it or better. The two fields together are the
    instruction, so a market order with a price and a limit order without one
    are both refused (:class:`~backend.execution.errors.OrderValidationError`) —
    either half alone is ambiguous.
    """

    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    """How long the order stays working.

    ``DAY`` expires at the close of the session it was released in; ``GTC``
    persists across sessions until filled or cancelled. Immediate-or-cancel and
    fill-or-kill are deliberately absent: they are intraday instruments, and
    intraday strategies are out of scope (directive §1.1).
    """

    DAY = "day"
    GTC = "gtc"


class FillSource(StrEnum):
    """Where a reported fill came from. No member denotes a live execution.

    Invariant I3 is that no simulated fill may be mistakable for a broker fill,
    so the two are distinct values on every fill row rather than a distinction
    left to context:

    - ``SIMULATED`` — produced in this process against a modelled book. Nothing
      in this package produces one; the value exists so that anything which does
      must say so.
    - ``PAPER_BROKER`` — reported by a paper brokerage account (P11.1, blocked
      on B2). Still not a real execution: no capital moves, and per D-013 the
      slippage it implies is a lower bound.

    There is deliberately no member for a live broker, and the schema's CHECK
    admits only these two literals. This is a provenance label, not a router: it
    records where a message came from and selects nothing.
    """

    SIMULATED = "simulated"
    PAPER_BROKER = "paper_broker"


def _require(condition: bool, message: str) -> None:
    """Raise :class:`OrderValidationError` when ``condition`` is false.

    Args:
        condition: the requirement that must hold.
        message: what the caller got wrong, quoted verbatim in the error.

    Raises:
        OrderValidationError: when ``condition`` is false.
    """
    if not condition:
        raise OrderValidationError(message)


def _require_whole(name: str, value: int) -> None:
    """Refuse a quantity that is not a plain ``int``.

    ``bool`` is refused explicitly: ``True`` is an ``int`` and would silently
    record as a quantity of one share.

    Args:
        name: field name, for the message.
        value: the value to check.

    Raises:
        OrderValidationError: if ``value`` is a ``bool`` or not an ``int``.
    """
    checked: object = value
    if isinstance(checked, bool) or not isinstance(checked, int):
        msg = f"{name} must be a whole number of shares (bool is refused), got {checked!r}"
        raise OrderValidationError(msg)


def price_text(price: Decimal) -> str:
    """Render a price at the fixed schema scale, for hashing and display.

    ``Decimal("1.5")`` and ``Decimal("1.50")`` are the same price and must
    produce the same text, or two submissions of one order would hash to two
    idempotency keys and the retry would be sent twice. Normalising to a fixed
    scale is what makes the rendering a function of the *value* rather than of
    how the caller typed it.

    Args:
        price: a finite, non-negative price in **US dollars per share** with at
            most :data:`PRICE_SCALE` decimal places.

    Returns:
        The price in plain decimal notation at exactly :data:`PRICE_SCALE`
        decimal places, e.g. ``"123.450000"``.

    Raises:
        OrderValidationError: if the price is non-finite or carries more than
            :data:`PRICE_SCALE` decimal places (quantising it here would round a
            price the caller stated).
    """
    _require(price.is_finite(), f"price {price!r} is not finite")
    exponent = price.as_tuple().exponent
    _require(
        isinstance(exponent, int) and exponent >= -PRICE_SCALE,
        f"price {price!r} carries more than {PRICE_SCALE} decimal places, which the "
        f"Numeric(18, {PRICE_SCALE}) price column cannot store; the stored order would then "
        f"disagree with the idempotency key computed from the submitted one",
    )
    return f"{price.quantize(Decimal(1).scaleb(-PRICE_SCALE)):f}"


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """A fully-specified instruction to trade, and the identity of the decision.

    An intent is *content*, not an event: it says what to trade and which
    computation asked for it, and it carries no timestamp, sequence number or
    identifier assigned by anything. That is what lets
    :func:`backend.execution.idempotency.idempotency_key` derive a stable key
    from it, and what makes a retry recompute the same key without having to
    remember anything.

    The reproducibility stamp (I2) is part of the intent rather than metadata
    attached to it. Two consequences, both wanted:

    - A fill traces back through its transition to the order and from there to
      the commit, config hash, data version and seed that produced the decision.
    - Re-running an identical plan — same config, same data, same seed, same
      commit, same rebalance date — produces the *same* keys, so the second run
      is absorbed as a duplicate instead of doubling the position. Changing any
      input changes the config hash or the data version, and the orders it
      produces are correctly new.

    Units: ``quantity_shares`` is **whole shares**; ``limit_price_usd`` is **US
    dollars per share**.

    Attributes:
        security_id: identity-anchor key of the security to trade
            (``security.security_id``), strictly positive.
        side: buy or sell.
        quantity_shares: shares to trade, strictly positive. Direction lives in
            ``side``, never in the sign of the quantity.
        order_type: market or limit.
        time_in_force: how long the order works.
        limit_price_usd: the limit price, present exactly when ``order_type`` is
            ``LIMIT``.
        rebalance_date: the rebalance this order implements. Part of the
            identity: the same target position on two dates is two orders.
        slice_index: zero-based index of this slice within a parent order, ``0``
            for an unsliced order. Part of the identity, so the ten slices of one
            TWAP are ten distinct orders rather than one order submitted ten
            times (P11.4).
        slice_count: number of slices the parent was divided into, ``1`` when
            unsliced.
        stamp: the I2 reproducibility stamp of the computation that produced
            this order.
    """

    security_id: int
    side: Side
    quantity_shares: int
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price_usd: Decimal | None
    rebalance_date: dt.date
    slice_index: int
    slice_count: int
    stamp: ReproducibilityStamp

    @property
    def venue(self) -> ExecutionVenue:
        """The venue this order executes at: always the paper venue.

        A property with no setter and no argument, not a field. There is nothing
        to pass and nothing to configure — see :class:`ExecutionVenue`.
        """
        return ExecutionVenue.PAPER

    def __post_init__(self) -> None:
        """Validate that the fields together describe a tradeable instruction.

        Raises:
            OrderValidationError: on a non-positive or non-integral quantity, a
                non-positive security id, a limit order without a price or a
                market order with one, a non-positive or over-precise limit
                price, or a slice index outside its slice count. Each of these
                would otherwise reach the database and fail there against a
                CHECK constraint — a real backstop that stays, but one whose
                message names a constraint rather than a field, and by then the
                caller has a transaction to unwind.
        """
        _require_whole("security_id", self.security_id)
        _require(self.security_id > 0, f"security_id must be positive, got {self.security_id}")
        _require_whole("quantity_shares", self.quantity_shares)
        _require(
            self.quantity_shares > 0,
            f"quantity_shares must be strictly positive, got {self.quantity_shares}; an "
            f"order for no shares is not an instruction to trade",
        )
        self._validate_price()
        _require_whole("slice_index", self.slice_index)
        _require_whole("slice_count", self.slice_count)
        _require(self.slice_count >= 1, f"slice_count must be at least 1, got {self.slice_count}")
        _require(
            0 <= self.slice_index < self.slice_count,
            f"slice_index={self.slice_index} is outside its slice_count={self.slice_count}; "
            f"slices are indexed 0..slice_count-1",
        )
        stamp: object = self.stamp
        _require(
            hasattr(stamp, "git_commit")
            and hasattr(stamp, "config_hash")
            and hasattr(stamp, "data_version")
            and hasattr(stamp, "seed"),
            f"stamp must be a ReproducibilityStamp carrying all four I2 components, got "
            f"{type(stamp).__name__}; an order that cannot name the commit and config that "
            f"produced it makes its own fills untraceable",
        )

    def _validate_price(self) -> None:
        """Check the limit price against the order type.

        Raises:
            OrderValidationError: if the price and the type disagree, or if the
                price is not a positive, finite, representable decimal.
        """
        if self.order_type is OrderType.LIMIT:
            _require(
                self.limit_price_usd is not None,
                "a limit order must carry limit_price_usd: the type and the price together "
                "are the instruction, and either half alone is ambiguous",
            )
            price = self.limit_price_usd
            if price is not None:
                # Finiteness first: comparing a NaN raises InvalidOperation
                # rather than returning False, and the caller would get a
                # decimal exception instead of a message naming the field.
                price_text(price)
                _require(
                    price > 0,
                    f"limit_price_usd must be strictly positive, got {price}",
                )
            return
        _require(
            self.limit_price_usd is None,
            f"a {self.order_type.value} order must not carry limit_price_usd (got "
            f"{self.limit_price_usd}): a price on an order that ignores prices is a "
            f"contradiction the venue would silently drop",
        )


@dataclass(frozen=True, slots=True)
class FillReport:
    """One reported execution against an order.

    Every field is something a venue told us; nothing here is inferred. The
    ``source`` is what keeps invariant I3 checkable — a simulated fill and a
    paper-broker fill are different values on the row, so no reader has to
    reconstruct which it is holding.

    Units: ``quantity_shares`` is **whole shares**; ``price_usd`` is **US
    dollars per share**.

    Attributes:
        quantity_shares: shares traded by this report, strictly positive.
        price_usd: the price traded at, strictly positive.
        source: where the report came from. No value denotes a live execution.
        venue_fill_id: the venue's identifier for this execution, when it gave
            one. ``None`` when it did not — absent rather than invented (I3).
    """

    quantity_shares: int
    price_usd: Decimal
    source: FillSource
    venue_fill_id: str | None = None

    @property
    def cost_basis(self) -> str:
        """The basis any cost derived from this fill may claim: a lower bound.

        Constant by construction, because every source this type admits is a
        simulation or a paper account, and both are optimistic (D-013). Carried
        onto the persisted row so the qualification travels with the number
        rather than living in a document nobody reads at query time.
        """
        return PAPER_FILL_COST_BASIS

    def __post_init__(self) -> None:
        """Validate the traded quantity and price.

        Raises:
            OrderValidationError: if the quantity is not a strictly positive
                whole number of shares, if the price is not a finite, strictly
                positive, representable decimal, or if ``venue_fill_id`` is
                present but blank (a blank identifier is an absent one wearing a
                value's clothes).
        """
        _require_whole("quantity_shares", self.quantity_shares)
        _require(
            self.quantity_shares > 0,
            f"quantity_shares must be strictly positive, got {self.quantity_shares}; a "
            f"zero-quantity report is not a fill",
        )
        # Finiteness first, for the reason given in OrderIntent._validate_price.
        price_text(self.price_usd)
        _require(
            self.price_usd > 0,
            f"price_usd must be strictly positive, got {self.price_usd}",
        )
        if self.venue_fill_id is not None:
            _require(
                self.venue_fill_id.strip() != "",
                "venue_fill_id is blank; pass None when the venue gave no identifier "
                "rather than an empty string that reads as one",
            )
