"""The store: idempotent submission, the append-only trail, and behaviour under contention.

The uniqueness that makes a retry safe belongs to the database, so these tests
run against an in-memory double that models the two constraints atomically (see
``fixtures.PaperOrderStoreDouble``). They prove the store's *control flow* is
correct when writers race. That Postgres actually enforces the constraints is
proved in ``backend/tests/integration/test_order_lifecycle.py``, which needs a
container and cannot run while Docker is down.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal

import pytest

from backend.execution.errors import (
    ConcurrentTransitionError,
    DuplicateOrderError,
    FillAccountingError,
    IdempotencyCollisionError,
    IllegalTransitionError,
    NotPaperOrderError,
    OrderNotFoundError,
    OrderValidationError,
    TerminalOrderError,
)
from backend.execution.idempotency import (
    IDEMPOTENCY_SCHEMA,
    idempotency_key,
    idempotency_preimage,
)
from backend.execution.lifecycle import INITIAL_STATE, OrderEvent, OrderState
from backend.execution.orders import FillReport, FillSource
from backend.execution.store import (
    append_transition,
    load_order,
    load_order_by_key,
    record_order,
)
from backend.tests.execution.fixtures import (
    MOMENT,
    PaperOrderStoreDouble,
    as_session,
    make_intent,
    order_columns,
)

CONCURRENT_WRITERS = 16


async def _release_and_acknowledge(double: PaperOrderStoreDouble, order_id: int) -> None:
    """Walk an order to ACKNOWLEDGED, the state most tests start from."""
    session = as_session(double)
    await append_transition(
        session, order_id=order_id, event=OrderEvent.RELEASE, occurred_at=MOMENT
    )
    await append_transition(
        session, order_id=order_id, event=OrderEvent.ACKNOWLEDGE, occurred_at=MOMENT
    )


async def test_a_recorded_order_starts_as_a_draft_with_no_transitions() -> None:
    double = PaperOrderStoreDouble()
    intent = make_intent()
    recorded = await record_order(as_session(double), intent)
    assert recorded.was_already_recorded is False
    assert recorded.idempotency_key == idempotency_key(intent)
    loaded = await load_order(as_session(double), recorded.order_id)
    assert loaded.state is INITIAL_STATE
    assert loaded.filled_quantity_shares == 0
    assert loaded.transitions == ()
    assert loaded.sequence_number == 0


async def test_the_order_row_carries_the_key_the_schema_and_the_preimage() -> None:
    double = PaperOrderStoreDouble()
    intent = make_intent()
    recorded = await record_order(as_session(double), intent)
    row = double.orders[recorded.order_id]
    assert row["idempotency_key"] == idempotency_key(intent)
    assert row["idempotency_schema"] == IDEMPOTENCY_SCHEMA
    assert row["idempotency_preimage"] == idempotency_preimage(intent)


async def test_the_order_row_carries_all_four_stamp_components() -> None:
    # I2: a fill traces through its transition to the order, and from there to
    # the commit and config that produced the decision.
    double = PaperOrderStoreDouble()
    intent = make_intent()
    recorded = await record_order(as_session(double), intent)
    row = double.orders[recorded.order_id]
    assert row["git_commit"] == intent.stamp.git_commit
    assert row["git_dirty"] == intent.stamp.git_dirty
    assert row["data_version"] == intent.stamp.data_version
    assert row["config_hash"] == intent.stamp.config_hash
    assert row["seed"] == intent.stamp.seed


async def test_the_store_never_supplies_a_venue() -> None:
    # The column is filled by the server default; nothing here chooses it.
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    assert double.orders[recorded.order_id]["venue"] == "paper"
    assert double.order_insert_attempts == 1


async def test_a_repeated_submission_is_absorbed_rather_than_written_twice() -> None:
    double = PaperOrderStoreDouble()
    intent = make_intent()
    first = await record_order(as_session(double), intent)
    second = await record_order(as_session(double), intent)
    assert second.order_id == first.order_id
    assert second.was_already_recorded is True
    assert first.was_already_recorded is False
    assert len(double.orders) == 1


async def test_a_repeated_submission_can_be_refused_instead() -> None:
    double = PaperOrderStoreDouble()
    intent = make_intent()
    first = await record_order(as_session(double), intent)
    with pytest.raises(DuplicateOrderError) as raised:
        await record_order(as_session(double), intent, if_exists="refuse")
    assert raised.value.order_id == first.order_id
    assert raised.value.idempotency_key == first.idempotency_key


async def test_a_key_held_by_different_content_is_a_collision_not_a_duplicate() -> None:
    # Absorbing this would substitute one trade for another: the caller would
    # believe its order was already recorded when a different one was.
    double = PaperOrderStoreDouble()
    intent = make_intent()
    other = make_intent(quantity_shares=999)
    double.seed_order(
        **order_columns(
            other,
            idempotency_key=idempotency_key(intent),
            preimage=idempotency_preimage(other),
        )
    )
    with pytest.raises(IdempotencyCollisionError) as raised:
        await record_order(as_session(double), intent)
    assert raised.value.stored_preimage == idempotency_preimage(other)
    assert raised.value.computed_preimage == idempotency_preimage(intent)
    assert "IDEMPOTENCY_SCHEMA" in str(raised.value)


async def test_concurrent_duplicate_submissions_produce_exactly_one_order() -> None:
    # The property that matters most, and the one a sequential test cannot show.
    # All sixteen writers are held at the insert boundary by a barrier, so they
    # contend simultaneously rather than by scheduler luck; the database's
    # uniqueness picks one and the other fifteen absorb it.
    double = PaperOrderStoreDouble()
    double.insert_barrier = asyncio.Barrier(CONCURRENT_WRITERS)
    intent = make_intent()
    results = await asyncio.gather(
        *(record_order(as_session(double), intent) for _ in range(CONCURRENT_WRITERS))
    )
    assert double.order_insert_attempts == CONCURRENT_WRITERS
    assert len(double.orders) == 1
    assert len({result.order_id for result in results}) == 1
    assert sum(not result.was_already_recorded for result in results) == 1
    assert sum(result.was_already_recorded for result in results) == CONCURRENT_WRITERS - 1
    assert {result.idempotency_key for result in results} == {idempotency_key(intent)}


async def test_concurrent_submissions_of_distinct_orders_all_land() -> None:
    # The other half of the claim: the mechanism refuses duplicates, not writes.
    double = PaperOrderStoreDouble()
    double.insert_barrier = asyncio.Barrier(CONCURRENT_WRITERS)
    intents = [make_intent(security_id=index + 1) for index in range(CONCURRENT_WRITERS)]
    results = await asyncio.gather(
        *(record_order(as_session(double), intent) for intent in intents)
    )
    assert len(double.orders) == CONCURRENT_WRITERS
    assert len({result.order_id for result in results}) == CONCURRENT_WRITERS
    assert all(not result.was_already_recorded for result in results)


async def test_concurrent_appends_of_one_event_leave_one_transition() -> None:
    # Two workers reading the same tail both compute last+1; the primary key
    # picks one and the losers are told to re-read.
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    double.insert_barrier = asyncio.Barrier(CONCURRENT_WRITERS)
    outcomes = await asyncio.gather(
        *(
            append_transition(
                as_session(double),
                order_id=recorded.order_id,
                event=OrderEvent.RELEASE,
                occurred_at=MOMENT,
            )
            for _ in range(CONCURRENT_WRITERS)
        ),
        return_exceptions=True,
    )
    written = [item for item in outcomes if not isinstance(item, BaseException)]
    refused = [item for item in outcomes if isinstance(item, ConcurrentTransitionError)]
    assert len(written) == 1
    assert len(refused) == CONCURRENT_WRITERS - 1
    assert len(outcomes) == len(written) + len(refused)
    assert len(double.transitions[recorded.order_id]) == 1
    loaded = await load_order(as_session(double), recorded.order_id)
    assert loaded.state is OrderState.PENDING_NEW
    assert loaded.sequence_number == 1


async def test_the_full_lifecycle_is_recorded_step_by_step() -> None:
    double = PaperOrderStoreDouble()
    session = as_session(double)
    recorded = await record_order(session, make_intent(quantity_shares=100))
    await _release_and_acknowledge(double, recorded.order_id)
    await append_transition(
        session,
        order_id=recorded.order_id,
        event=OrderEvent.PARTIAL_FILL,
        occurred_at=MOMENT,
        fill=FillReport(
            quantity_shares=40, price_usd=Decimal("123.45"), source=FillSource.SIMULATED
        ),
    )
    midway = await load_order(session, recorded.order_id)
    assert midway.state is OrderState.PARTIALLY_FILLED
    assert midway.filled_quantity_shares == 40
    await append_transition(
        session,
        order_id=recorded.order_id,
        event=OrderEvent.FILL_COMPLETE,
        occurred_at=MOMENT,
        fill=FillReport(
            quantity_shares=60, price_usd=Decimal("123.50"), source=FillSource.SIMULATED
        ),
    )
    final = await load_order(session, recorded.order_id)
    assert final.state is OrderState.FILLED
    assert final.filled_quantity_shares == 100
    assert [item.sequence_number for item in final.transitions] == [1, 2, 3, 4]
    assert [item.event for item in final.transitions] == [
        OrderEvent.RELEASE,
        OrderEvent.ACKNOWLEDGE,
        OrderEvent.PARTIAL_FILL,
        OrderEvent.FILL_COMPLETE,
    ]


async def test_a_recorded_fill_states_its_source_and_its_lower_bound_basis() -> None:
    # I3 and D-013 on the persisted row, not only on the value object.
    double = PaperOrderStoreDouble()
    session = as_session(double)
    recorded = await record_order(session, make_intent(quantity_shares=10))
    await _release_and_acknowledge(double, recorded.order_id)
    await append_transition(
        session,
        order_id=recorded.order_id,
        event=OrderEvent.FILL_COMPLETE,
        occurred_at=MOMENT,
        fill=FillReport(
            quantity_shares=10,
            price_usd=Decimal("1.25"),
            source=FillSource.SIMULATED,
            venue_fill_id="sim-1",
        ),
    )
    row = double.transitions[recorded.order_id][-1]
    assert row["fill_source"] == "simulated"
    assert row["fill_cost_basis"] == "lower_bound"
    assert row["venue_fill_id"] == "sim-1"
    assert row["fill_price_usd"] == Decimal("1.25")


async def test_a_non_fill_transition_stores_no_fill_payload() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    await append_transition(
        as_session(double),
        order_id=recorded.order_id,
        event=OrderEvent.RELEASE,
        occurred_at=MOMENT,
    )
    row = double.transitions[recorded.order_id][0]
    for column in ("fill_quantity_shares", "fill_price_usd", "fill_source", "fill_cost_basis"):
        assert row[column] is None, column


async def test_an_illegal_event_is_refused_before_it_reaches_sql() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    with pytest.raises(IllegalTransitionError):
        await append_transition(
            as_session(double),
            order_id=recorded.order_id,
            event=OrderEvent.ACKNOWLEDGE,
            occurred_at=MOMENT,
        )
    assert double.transitions[recorded.order_id] == []


async def test_a_finished_order_refuses_every_further_event() -> None:
    double = PaperOrderStoreDouble()
    session = as_session(double)
    recorded = await record_order(session, make_intent(quantity_shares=10))
    await _release_and_acknowledge(double, recorded.order_id)
    await append_transition(
        session,
        order_id=recorded.order_id,
        event=OrderEvent.FILL_COMPLETE,
        occurred_at=MOMENT,
        fill=FillReport(quantity_shares=10, price_usd=Decimal("1"), source=FillSource.SIMULATED),
    )
    for event in (OrderEvent.RELEASE, OrderEvent.PARTIAL_FILL, OrderEvent.REQUEST_CANCEL):
        with pytest.raises(TerminalOrderError):
            await append_transition(
                session, order_id=recorded.order_id, event=event, occurred_at=MOMENT
            )
    assert len(double.transitions[recorded.order_id]) == 3


async def test_a_fill_event_without_a_report_is_refused() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    await _release_and_acknowledge(double, recorded.order_id)
    with pytest.raises(FillAccountingError, match="must carry a fill report"):
        await append_transition(
            as_session(double),
            order_id=recorded.order_id,
            event=OrderEvent.PARTIAL_FILL,
            occurred_at=MOMENT,
        )


async def test_a_non_fill_event_carrying_a_report_is_refused() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    with pytest.raises(FillAccountingError, match="is not a fill but carries"):
        await append_transition(
            as_session(double),
            order_id=recorded.order_id,
            event=OrderEvent.RELEASE,
            occurred_at=MOMENT,
            fill=FillReport(quantity_shares=1, price_usd=Decimal("1"), source=FillSource.SIMULATED),
        )


async def test_the_event_must_agree_with_the_fill_arithmetic() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent(quantity_shares=100))
    await _release_and_acknowledge(double, recorded.order_id)
    with pytest.raises(FillAccountingError, match="the arithmetic makes that"):
        await append_transition(
            as_session(double),
            order_id=recorded.order_id,
            event=OrderEvent.FILL_COMPLETE,
            occurred_at=MOMENT,
            fill=FillReport(
                quantity_shares=40, price_usd=Decimal("1"), source=FillSource.SIMULATED
            ),
        )


async def test_an_overfill_is_refused_rather_than_clamped() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent(quantity_shares=100))
    await _release_and_acknowledge(double, recorded.order_id)
    with pytest.raises(FillAccountingError, match="past the ordered 100"):
        await append_transition(
            as_session(double),
            order_id=recorded.order_id,
            event=OrderEvent.PARTIAL_FILL,
            occurred_at=MOMENT,
            fill=FillReport(
                quantity_shares=150, price_usd=Decimal("1"), source=FillSource.SIMULATED
            ),
        )
    assert len(double.transitions[recorded.order_id]) == 2


async def test_a_naive_timestamp_is_refused() -> None:
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    with pytest.raises(OrderValidationError, match="timezone-naive"):
        await append_transition(
            as_session(double),
            order_id=recorded.order_id,
            event=OrderEvent.RELEASE,
            occurred_at=dt.datetime(2026, 8, 3, 14, 30),  # noqa: DTZ001 - the point of the test
        )


async def test_a_missing_order_is_an_error_not_a_none() -> None:
    double = PaperOrderStoreDouble()
    with pytest.raises(OrderNotFoundError):
        await load_order(as_session(double), 999)
    with pytest.raises(OrderNotFoundError):
        await load_order_by_key(as_session(double), "f" * 64)


async def test_an_order_can_be_found_by_its_content_key() -> None:
    # What a retrying caller has: the intent, not the id.
    double = PaperOrderStoreDouble()
    intent = make_intent()
    recorded = await record_order(as_session(double), intent)
    loaded = await load_order_by_key(as_session(double), idempotency_key(intent))
    assert loaded.order_id == recorded.order_id


async def test_a_row_whose_venue_is_not_paper_is_refused_on_read() -> None:
    # Unreachable through this package; checked anyway, because the single claim
    # it makes is worth verifying on both sides of the database.
    double = PaperOrderStoreDouble()
    intent = make_intent()
    order_id = double.seed_order(
        **order_columns(
            intent,
            idempotency_key=idempotency_key(intent),
            preimage=idempotency_preimage(intent),
        )
    )
    double.orders[order_id]["venue"] = "live"
    with pytest.raises(NotPaperOrderError) as raised:
        await load_order(as_session(double), order_id)
    assert raised.value.stored_venue == "live"
    assert "paper-only and permanently so" in str(raised.value)


async def test_a_cancel_that_races_a_fill_is_representable() -> None:
    # The reason PENDING_CANCEL accepts fills: the race is real, and refusing to
    # record it would leave a traded position with no transition to explain it.
    double = PaperOrderStoreDouble()
    session = as_session(double)
    recorded = await record_order(session, make_intent(quantity_shares=100))
    await _release_and_acknowledge(double, recorded.order_id)
    await append_transition(
        session, order_id=recorded.order_id, event=OrderEvent.REQUEST_CANCEL, occurred_at=MOMENT
    )
    await append_transition(
        session,
        order_id=recorded.order_id,
        event=OrderEvent.PARTIAL_FILL,
        occurred_at=MOMENT,
        fill=FillReport(
            quantity_shares=30, price_usd=Decimal("10"), source=FillSource.PAPER_BROKER
        ),
    )
    await append_transition(
        session,
        order_id=recorded.order_id,
        event=OrderEvent.CANCEL_REJECTED,
        occurred_at=MOMENT,
    )
    loaded = await load_order(session, recorded.order_id)
    assert loaded.state is OrderState.PARTIALLY_FILLED
    assert loaded.filled_quantity_shares == 30
