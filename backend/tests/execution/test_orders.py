"""Order and fill value objects: what they refuse, and why the refusal is the point."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.execution.errors import OrderValidationError
from backend.execution.orders import (
    PAPER_FILL_COST_BASIS,
    PRICE_SCALE,
    ExecutionVenue,
    FillReport,
    FillSource,
    OrderType,
    Side,
    TimeInForce,
    price_text,
)
from backend.tests.execution.fixtures import make_intent, make_stamp


def test_a_valid_intent_reports_the_paper_venue() -> None:
    intent = make_intent()
    assert intent.venue is ExecutionVenue.PAPER
    assert intent.venue.value == "paper"


def test_an_intent_takes_no_venue_argument() -> None:
    # The structural half of paper-only: there is nothing to pass, so there is
    # nothing a configuration could set.
    with pytest.raises(TypeError):
        make_intent(venue=ExecutionVenue.PAPER)  # type: ignore[call-arg]


@pytest.mark.parametrize("quantity", [0, -1, -100])
def test_a_non_positive_quantity_is_refused(quantity: int) -> None:
    with pytest.raises(OrderValidationError, match="strictly positive"):
        make_intent(quantity_shares=quantity)


def test_a_boolean_quantity_is_refused() -> None:
    # True is an int and would silently record as one share.
    with pytest.raises(OrderValidationError, match="bool is refused"):
        make_intent(quantity_shares=True)


def test_a_limit_order_without_a_price_is_refused() -> None:
    with pytest.raises(OrderValidationError, match="must carry limit_price_usd"):
        make_intent(order_type=OrderType.LIMIT, limit_price_usd=None)


def test_a_market_order_with_a_price_is_refused() -> None:
    with pytest.raises(OrderValidationError, match="must not carry limit_price_usd"):
        make_intent(order_type=OrderType.MARKET, limit_price_usd=Decimal("10"))


def test_a_market_order_without_a_price_is_accepted() -> None:
    intent = make_intent(order_type=OrderType.MARKET, limit_price_usd=None)
    assert intent.limit_price_usd is None


@pytest.mark.parametrize("price", [Decimal("0"), Decimal("-1.50")])
def test_a_non_positive_limit_price_is_refused(price: Decimal) -> None:
    with pytest.raises(OrderValidationError, match="strictly positive"):
        make_intent(limit_price_usd=price)


def test_a_price_finer_than_the_column_is_refused_rather_than_rounded() -> None:
    # Rounding here would store a price the caller did not state, and the stored
    # order would then disagree with the key computed from the submitted one.
    with pytest.raises(OrderValidationError, match="decimal places"):
        make_intent(limit_price_usd=Decimal("1.1234567"))


def test_a_non_finite_price_is_refused() -> None:
    with pytest.raises(OrderValidationError, match="not finite"):
        make_intent(limit_price_usd=Decimal("NaN"))


def test_price_text_normalises_to_the_schema_scale() -> None:
    assert price_text(Decimal("123.45")) == "123.450000"
    assert price_text(Decimal("123.450000")) == "123.450000"
    assert len(price_text(Decimal("1")).split(".")[1]) == PRICE_SCALE


@pytest.mark.parametrize(
    ("index", "count"),
    [(1, 1), (5, 5), (-1, 3), (3, 0)],
)
def test_a_slice_outside_its_count_is_refused(index: int, count: int) -> None:
    with pytest.raises(OrderValidationError):
        make_intent(slice_index=index, slice_count=count)


def test_a_valid_slice_is_accepted() -> None:
    intent = make_intent(slice_index=4, slice_count=5)
    assert (intent.slice_index, intent.slice_count) == (4, 5)


@pytest.mark.parametrize("security_id", [0, -3])
def test_a_non_positive_security_id_is_refused(security_id: int) -> None:
    with pytest.raises(OrderValidationError, match="security_id"):
        make_intent(security_id=security_id)


def test_an_order_without_a_stamp_is_refused() -> None:
    # I2: an order that cannot name the commit and config that produced it makes
    # its own fills untraceable.
    with pytest.raises(OrderValidationError, match="ReproducibilityStamp"):
        make_intent(stamp="a" * 40)  # type: ignore[arg-type]


def test_the_intent_is_immutable() -> None:
    intent = make_intent()
    with pytest.raises(AttributeError):
        intent.quantity_shares = 5  # type: ignore[misc]


def test_the_intent_carries_the_stamp_it_was_built_with() -> None:
    stamp = make_stamp(seed=99)
    assert make_intent(stamp=stamp).stamp is stamp


def test_the_defaults_describe_a_tradeable_instruction() -> None:
    intent = make_intent()
    assert intent.side is Side.BUY
    assert intent.time_in_force is TimeInForce.DAY
    assert intent.rebalance_date == dt.date(2026, 8, 3)


def test_a_fill_report_labels_its_cost_basis_as_a_lower_bound() -> None:
    # D-013: paper and simulated fills are optimistic, so slippage measured from
    # them bounds the true cost from below and is never an estimate of it.
    report = FillReport(quantity_shares=10, price_usd=Decimal("12.34"), source=FillSource.SIMULATED)
    assert report.cost_basis == PAPER_FILL_COST_BASIS == "lower_bound"


def test_a_fill_report_states_where_it_came_from() -> None:
    # I3: a simulated fill and a paper-broker fill are different values on the
    # row, so neither can be mistaken for the other.
    simulated = FillReport(
        quantity_shares=10, price_usd=Decimal("12.34"), source=FillSource.SIMULATED
    )
    broker = FillReport(
        quantity_shares=10, price_usd=Decimal("12.34"), source=FillSource.PAPER_BROKER
    )
    assert simulated.source is not broker.source
    assert {member.value for member in FillSource} == {"simulated", "paper_broker"}


@pytest.mark.parametrize("quantity", [0, -1])
def test_a_fill_report_refuses_a_non_positive_quantity(quantity: int) -> None:
    with pytest.raises(OrderValidationError, match="strictly positive"):
        FillReport(quantity_shares=quantity, price_usd=Decimal("1"), source=FillSource.SIMULATED)


def test_a_fill_report_refuses_a_non_positive_price() -> None:
    with pytest.raises(OrderValidationError, match="price_usd"):
        FillReport(quantity_shares=1, price_usd=Decimal("0"), source=FillSource.SIMULATED)


def test_a_fill_report_refuses_a_blank_venue_identifier() -> None:
    # A blank identifier is an absent one wearing a value's clothes.
    with pytest.raises(OrderValidationError, match="venue_fill_id"):
        FillReport(
            quantity_shares=1,
            price_usd=Decimal("1"),
            source=FillSource.SIMULATED,
            venue_fill_id="  ",
        )


def test_a_fill_report_accepts_an_absent_venue_identifier() -> None:
    report = FillReport(quantity_shares=1, price_usd=Decimal("1"), source=FillSource.SIMULATED)
    assert report.venue_fill_id is None
