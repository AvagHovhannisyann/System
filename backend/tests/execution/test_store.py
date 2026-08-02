"""The store: idempotent submission, the append-only trail, and behaviour under contention.

The uniqueness that makes a retry safe belongs to the database, so these tests
run against an in-memory double that models its refusals atomically (see
``fixtures.PaperOrderStoreDouble``). They prove the store's *control flow* is
correct when writers race. That Postgres actually enforces the constraints is
proved in ``backend/tests/integration/test_order_lifecycle.py``, which needs a
container.

The double's first version modelled a concurrent transition append as the primary
key refusing the loser. Real Postgres refuses it through the ``BEFORE INSERT``
chain guard instead, which fires ahead of every constraint — so this whole file
was green while the integration suite failed. The double now refuses in the
database's order and with the database's SQLSTATEs, and the tests at the end of
this file pin the retryable/malformed distinction the store decides on.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib
from decimal import Decimal
from types import ModuleType
from typing import TYPE_CHECKING, Any, NoReturn, cast

import pytest
from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_dbapi
from sqlalchemy.exc import DBAPIError, IntegrityError

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
    UNIQUE_VIOLATION_SQLSTATE,
    _is_position_taken,
    _sqlstate,
    append_transition,
    load_order,
    load_order_by_key,
    record_order,
)
from backend.tests.execution.fixtures import (
    MOMENT,
    RAISE_EXCEPTION_SQLSTATE,
    DriverError,
    PaperOrderStoreDouble,
    as_session,
    make_intent,
    order_columns,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CONCURRENT_WRITERS = 16


def _asyncpg_exception(name: str) -> Any:  # noqa: ANN401 - asyncpg ships no type information
    """Return one of asyncpg's generated exception classes by name."""
    return getattr(importlib.import_module("asyncpg.exceptions"), name)


def _asyncpg_dbapi() -> Any:  # noqa: ANN401 - the shim's constructor is untyped
    """Return SQLAlchemy's asyncpg DBAPI shim, which owns the error-class mapping."""
    factory = cast("Callable[[ModuleType], Any]", AsyncAdapt_asyncpg_dbapi)
    return factory(importlib.import_module("asyncpg"))


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


async def test_a_position_taken_refusal_is_retryable_whichever_route_it_arrives_by() -> None:
    # Both of the database's refusal paths — the primary key and the chain
    # guard's `USING ERRCODE = 'unique_violation'` — carry SQLSTATE 23505, and
    # the store must translate on the code rather than on the exception class.
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    await append_transition(
        as_session(double),
        order_id=recorded.order_id,
        event=OrderEvent.RELEASE,
        occurred_at=MOMENT,
    )
    for original in (
        IntegrityError("INSERT", {}, DriverError(UNIQUE_VIOLATION_SQLSTATE, "pk conflict")),
        DBAPIError("INSERT", {}, DriverError(UNIQUE_VIOLATION_SQLSTATE, "trigger conflict")),
    ):
        assert _is_position_taken(original) is True


async def test_a_malformed_row_is_not_reported_as_a_retryable_conflict() -> None:
    # The failure this guard exists for: telling a caller to retry a row the
    # database will refuse every time is worse than failing once. A chain error
    # (P0001) and a CHECK violation (23514, still an IntegrityError) must both
    # pass through untranslated.
    chain_error = DBAPIError("INSERT", {}, DriverError(RAISE_EXCEPTION_SQLSTATE, "gap"))
    check_error = IntegrityError("INSERT", {}, DriverError("23514", "check violation"))
    assert _is_position_taken(chain_error) is False
    assert _is_position_taken(check_error) is False


async def test_the_class_is_the_fallback_only_when_no_sqlstate_is_exposed() -> None:
    # A driver that will not name a code gets the best available answer; one
    # that will is never second-guessed.
    silent = IntegrityError("INSERT", {}, Exception("driver exposes no code"))
    assert _is_position_taken(silent) is True
    assert _is_position_taken(DBAPIError("INSERT", {}, Exception("no code"))) is False


def _refuse_as_chain_error(_double: PaperOrderStoreDouble, values: dict[str, object]) -> NoReturn:
    """Stand in for the chain guard refusing a malformed row under ``P0001``."""
    raise DBAPIError(
        "INSERT INTO execution_order_transition",
        values,
        DriverError(RAISE_EXCEPTION_SQLSTATE, "order 1 is at sequence 3, so the next is 4, not 9"),
    )


async def test_a_chain_refusal_reaches_the_caller_unchanged() -> None:
    # End to end: the store must let a non-retryable refusal through. Before CI
    # the double could only produce one kind of error, so this distinction had
    # no test that could fail.
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PaperOrderStoreDouble, "_insert_transition", _refuse_as_chain_error)
        with pytest.raises(DBAPIError) as raised:
            await append_transition(
                as_session(double),
                order_id=recorded.order_id,
                event=OrderEvent.RELEASE,
                occurred_at=MOMENT,
            )
    assert not isinstance(raised.value, ConcurrentTransitionError)
    assert _sqlstate(raised.value) == RAISE_EXCEPTION_SQLSTATE


def test_the_driver_maps_the_two_refusals_to_different_exception_classes() -> None:
    """Pin the driver behaviour the SQLSTATE decision rests on.

    This is the fact that made the first CI run fail, checked here without a
    database so it cannot drift silently:

    - a plpgsql ``RAISE EXCEPTION`` with no code of its own (``P0001``) arrives
      as asyncpg's ``RaiseError``, whose nearest mapped ancestor is the *generic*
      DBAPI ``Error`` — so it is a ``DBAPIError`` and **not** an
      ``IntegrityError``. An ``except IntegrityError`` around the transition
      insert therefore never saw the chain guard's refusals at all;
    - ``unique_violation`` (``23505``) arrives as ``IntegrityError``, which is
      why migration 0014 gives the trigger's position-taken branch that code;
    - ``check_violation`` (``23514``) *also* arrives as ``IntegrityError``, which
      is why the class alone is not a safe test for "retryable".
    """
    dbapi = _asyncpg_dbapi()
    mapping = dbapi._asyncpg_error_translate

    def resolved(error: Any) -> Any:  # noqa: ANN401 - asyncpg ships no type information
        return next(mapping[base] for base in error.__mro__ if base in mapping)

    unique = _asyncpg_exception("UniqueViolationError")
    check = _asyncpg_exception("CheckViolationError")
    raised = _asyncpg_exception("RaiseError")
    assert unique.sqlstate == UNIQUE_VIOLATION_SQLSTATE
    assert raised.sqlstate == RAISE_EXCEPTION_SQLSTATE
    assert resolved(unique) is dbapi.IntegrityError
    assert resolved(check) is dbapi.IntegrityError
    assert resolved(raised) is dbapi.Error
    assert not issubclass(dbapi.Error, dbapi.IntegrityError)


def test_the_sqlstate_decision_separates_what_the_class_cannot() -> None:
    # The consequence of the mapping above: a CHECK violation is an
    # IntegrityError too, so deciding on the class would send a caller round a
    # retry loop for a row the database refuses every time.
    check_violation = str(_asyncpg_exception("CheckViolationError").sqlstate)
    for sqlstate, retryable in (
        (UNIQUE_VIOLATION_SQLSTATE, True),
        (check_violation, False),
        (RAISE_EXCEPTION_SQLSTATE, False),
    ):
        error = IntegrityError("INSERT", {}, DriverError(sqlstate, "refused"))
        assert _is_position_taken(error) is retryable, sqlstate


def _refuse_as_taken_position_without_integrity_error(
    _double: PaperOrderStoreDouble, values: dict[str, object]
) -> NoReturn:
    """Deliver a taken position as a *generic* ``DBAPIError``, not an ``IntegrityError``."""
    raise DBAPIError(
        "INSERT INTO execution_order_transition",
        values,
        DriverError(UNIQUE_VIOLATION_SQLSTATE, "position taken"),
    )


async def test_the_translation_follows_the_sqlstate_not_the_exception_class() -> None:
    """A retryable refusal is recognised however the driver classes it.

    ``except IntegrityError`` is what missed the chain guard's refusals in the
    first CI run — asyncpg maps a plpgsql ``RAISE`` to the *generic* DBAPI error
    class. Catching ``DBAPIError`` and deciding on the SQLSTATE is the fix, and
    this is the case that tells the two apart: ``23505`` arriving on a class that
    is not ``IntegrityError``. Narrowing the catch again makes this fail.
    """
    double = PaperOrderStoreDouble()
    recorded = await record_order(as_session(double), make_intent())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            PaperOrderStoreDouble,
            "_insert_transition",
            _refuse_as_taken_position_without_integrity_error,
        )
        with pytest.raises(ConcurrentTransitionError):
            await append_transition(
                as_session(double),
                order_id=recorded.order_id,
                event=OrderEvent.RELEASE,
                occurred_at=MOMENT,
            )


def test_the_double_refuses_positions_the_way_the_database_refuses_them() -> None:
    """The double's own contract, asserted directly rather than left implicit.

    CI proved that a double modelling the wrong layer hides a bug in the code it
    stands in for, and a double nothing tests is free to drift back. Its three
    rules are pinned here: at or below the tail is a taken position (``23505``,
    retryable), above the next one is a gap (``P0001``, not retryable), and the
    next one lands.

    The ``below`` case cannot be produced through :func:`record_order` — the
    store always computes ``tail + 1`` from its own read — but the database
    produces it whenever a competing writer appended more than one row, so the
    double has to model it.
    """
    double = PaperOrderStoreDouble()
    double.transitions[1] = [{"sequence_number": index} for index in (1, 2, 3)]
    for taken in (1, 2, 3):
        with pytest.raises(IntegrityError) as conflict:
            double._insert_transition({"order_id": 1, "sequence_number": taken})
        assert _sqlstate(conflict.value) == UNIQUE_VIOLATION_SQLSTATE, taken
        assert _is_position_taken(conflict.value) is True, taken
    with pytest.raises(DBAPIError) as gap:
        double._insert_transition({"order_id": 1, "sequence_number": 5})
    assert _sqlstate(gap.value) == RAISE_EXCEPTION_SQLSTATE
    assert _is_position_taken(gap.value) is False
    double._insert_transition({"order_id": 1, "sequence_number": 4})
    assert len(double.transitions[1]) == 4
