"""P11.2 against a real TimescaleDB: the constraints, the triggers, and true concurrency.

The unit suite in ``backend/tests/execution`` covers the state machine, the key
derivation and the store's control flow — the last of those against an in-memory
double that models the two constraints atomically. What a double cannot cover is
the half the design actually rests on:

- **the unique index doing the deciding.** ``record_order`` deliberately does not
  look before it inserts, because a look-then-insert has a window two workers can
  both walk through. Whether the *database* closes that window is a property of
  Postgres and the constraint, and only Postgres can be asked. Here the duplicate
  submissions run in genuinely separate transactions on separate connections, so
  one of them really does block on the index and really does fail when the other
  commits.
- **the CHECK constraints refusing a writer that skips Python.** The point of
  restating the transition table, the paper venue and the non-live fill sources in
  SQL is that they bind raw ``INSERT``s. That claim is only testable by issuing
  raw ``INSERT``s.
- **the chain-guard trigger**, whose conditions are relations *between* rows and
  therefore cannot be a CHECK at all.
- **the append-only triggers** rejecting ``UPDATE`` and ``DELETE``.

**Not executed in this environment.** The Docker daemon is unavailable where this
was written, so the Testcontainers fixture cannot start and every test in this
module errors on the environment rather than on its assertions. They are
deliberately **not** skipped or xfailed (I6): a skipped test reports success it
did not earn, and the next environment with a working daemon must see these run
and either pass or fail on their merits.

Migration 0014 declares ``down_revision = "0013"`` and depends on that revision by
identifier alone. Revision 0013 has since landed from its own track, so the chain
is linear with a single head at 0014 — but the dependency was written before it
existed and would have failed at the fixture rather than at an assertion, which
is the correct shape for a cross-track dependency and is also not skipped.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import ingest_writer_session
from backend.execution.errors import ConcurrentTransitionError
from backend.execution.idempotency import idempotency_key, idempotency_preimage
from backend.execution.lifecycle import INITIAL_STATE, OrderEvent, OrderState
from backend.execution.orders import (
    FillReport,
    FillSource,
    OrderIntent,
    OrderType,
    Side,
    TimeInForce,
)
from backend.execution.store import append_transition, load_order, record_order
from backend.tests.execution.fixtures import make_stamp
from backend.tests.integration.factories import create_security

if TYPE_CHECKING:
    from backend.execution.store import RecordedOrder

MOMENT = dt.datetime(2026, 8, 3, 14, 30, tzinfo=dt.UTC)
REBALANCE_DATE = dt.date(2026, 8, 3)
CONCURRENT_WRITERS = 6
"""Kept modest: each writer holds its own connection from the pool for the whole
duration of the contention, and the property under test does not need more."""


def intent_for(security_id: int, *, quantity_shares: int = 100) -> OrderIntent:
    """Build a valid paper order intent against a real security anchor."""
    return OrderIntent(
        security_id=security_id,
        side=Side.BUY,
        quantity_shares=quantity_shares,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price_usd=Decimal("123.45"),
        rebalance_date=REBALANCE_DATE,
        slice_index=0,
        slice_count=1,
        stamp=make_stamp(),
    )


async def submit_and_commit(intent: OrderIntent, gate: asyncio.Barrier) -> RecordedOrder:
    """Record one order in its own transaction, arriving with every other writer."""
    async with ingest_writer_session() as session:
        await gate.wait()
        recorded = await record_order(session, intent)
        await session.commit()
    return recorded


async def append_and_commit(order_id: int, event: OrderEvent, gate: asyncio.Barrier) -> object:
    """Append one transition in its own transaction, arriving with every other writer."""
    async with ingest_writer_session() as session:
        await gate.wait()
        written = await append_transition(
            session, order_id=order_id, event=event, occurred_at=MOMENT
        )
        await session.commit()
    return written


async def raw_insert(statement: str, **parameters: object) -> None:
    """Execute one textual INSERT — the writer that skips this package entirely."""
    async with ingest_writer_session() as session:
        await session.execute(sa.text(statement), parameters)
        await session.commit()


async def mutate(statement: str, order_id: int) -> None:
    """Execute one textual UPDATE or DELETE, which the append-only triggers refuse."""
    async with ingest_writer_session() as session:
        await session.execute(sa.text(statement), {"order_id": order_id})
        await session.commit()


async def seed_order(security_id: int, *, quantity_shares: int = 100) -> int:
    """Record one order through the store and commit it."""
    intent = intent_for(security_id, quantity_shares=quantity_shares)
    async with ingest_writer_session() as session:
        recorded = await record_order(session, intent)
        await session.commit()
    return recorded.order_id


async def walk_to_acknowledged(order_id: int) -> None:
    """Advance an order to ACKNOWLEDGED and commit."""
    async with ingest_writer_session() as session:
        await append_transition(
            session, order_id=order_id, event=OrderEvent.RELEASE, occurred_at=MOMENT
        )
        await append_transition(
            session, order_id=order_id, event=OrderEvent.ACKNOWLEDGE, occurred_at=MOMENT
        )
        await session.commit()


TRANSITION_INSERT = (
    "INSERT INTO execution_order_transition ("
    "order_id, sequence_number, from_state, event, to_state, "
    "filled_quantity_after_shares, fill_quantity_shares, fill_price_usd, "
    "fill_source, fill_cost_basis, venue_fill_id, occurred_at) VALUES ("
    ":order_id, :sequence_number, :from_state, :event, :to_state, "
    ":filled_quantity_after_shares, :fill_quantity_shares, :fill_price_usd, "
    ":fill_source, :fill_cost_basis, :venue_fill_id, :occurred_at)"
)


def transition_values(**overrides: object) -> dict[str, object]:
    """Default column values for a raw transition INSERT, overridden per test."""
    values: dict[str, object] = {
        "order_id": 0,
        "sequence_number": 1,
        "from_state": OrderState.DRAFT.value,
        "event": OrderEvent.RELEASE.value,
        "to_state": OrderState.PENDING_NEW.value,
        "filled_quantity_after_shares": 0,
        "fill_quantity_shares": None,
        "fill_price_usd": None,
        "fill_source": None,
        "fill_cost_basis": None,
        "venue_fill_id": None,
        "occurred_at": MOMENT,
    }
    values.update(overrides)
    return values


async def test_an_order_round_trips_through_the_whole_lifecycle() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=100)
    async with ingest_writer_session() as session:
        assert (await load_order(session, order_id)).state is INITIAL_STATE
    await walk_to_acknowledged(order_id)
    async with ingest_writer_session() as session:
        await append_transition(
            session,
            order_id=order_id,
            event=OrderEvent.PARTIAL_FILL,
            occurred_at=MOMENT,
            fill=FillReport(
                quantity_shares=40,
                price_usd=Decimal("123.45"),
                source=FillSource.PAPER_BROKER,
                venue_fill_id="ib-1",
            ),
        )
        await append_transition(
            session,
            order_id=order_id,
            event=OrderEvent.FILL_COMPLETE,
            occurred_at=MOMENT,
            fill=FillReport(
                quantity_shares=60,
                price_usd=Decimal("123.50"),
                source=FillSource.PAPER_BROKER,
                venue_fill_id="ib-2",
            ),
        )
        await session.commit()
        loaded = await load_order(session, order_id)
    assert loaded.state is OrderState.FILLED
    assert loaded.filled_quantity_shares == 100
    assert [item.sequence_number for item in loaded.transitions] == [1, 2, 3, 4]


async def test_every_persisted_fill_is_labelled_a_lower_bound() -> None:
    # D-013 on the row itself: the qualification reaches every query and export
    # rather than living in a document nobody reads at query time.
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=10)
    await walk_to_acknowledged(order_id)
    async with ingest_writer_session() as session:
        await append_transition(
            session,
            order_id=order_id,
            event=OrderEvent.FILL_COMPLETE,
            occurred_at=MOMENT,
            fill=FillReport(
                quantity_shares=10, price_usd=Decimal("5"), source=FillSource.SIMULATED
            ),
        )
        await session.commit()
        rows = (
            await session.execute(
                sa.text(
                    "SELECT fill_source, fill_cost_basis FROM execution_order_transition "
                    "WHERE order_id = :order_id AND fill_quantity_shares IS NOT NULL"
                ),
                {"order_id": order_id},
            )
        ).all()
    # Compared as plain tuples: a Row is tuple-like at runtime but is not a
    # tuple to the type checker, so the direct comparison is one mypy --strict
    # rejects as non-overlapping. Since these tests cannot run without a Docker
    # daemon, a silently always-false assertion here would go unnoticed.
    assert [tuple(row) for row in rows] == [("simulated", "lower_bound")]


async def test_the_order_row_records_the_reproducibility_stamp() -> None:
    # I2: a fill traces to the config and the commit that produced the decision.
    security_id = await create_security()
    order_id = await seed_order(security_id)
    stamp = make_stamp()
    async with ingest_writer_session() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT git_commit, git_dirty, data_version, config_hash, seed, venue "
                    "FROM execution_order WHERE order_id = :order_id"
                ),
                {"order_id": order_id},
            )
        ).one()
    assert row == (
        stamp.git_commit,
        stamp.git_dirty,
        stamp.data_version,
        stamp.config_hash,
        stamp.seed,
        "paper",
    )


async def test_concurrent_duplicate_submissions_leave_exactly_one_order() -> None:
    # The property the whole mechanism exists for, proved where it is actually
    # enforced. Six writers in six separate transactions on six connections
    # arrive together; the unique index picks one, and the other five discover
    # the incumbent and absorb it.
    security_id = await create_security()
    intent = intent_for(security_id)
    gate = asyncio.Barrier(CONCURRENT_WRITERS)
    results = await asyncio.gather(
        *(submit_and_commit(intent, gate) for _ in range(CONCURRENT_WRITERS))
    )
    assert len({result.order_id for result in results}) == 1
    assert sum(not result.was_already_recorded for result in results) == 1
    assert {result.idempotency_key for result in results} == {idempotency_key(intent)}
    async with ingest_writer_session() as session:
        count = (
            await session.execute(
                sa.text("SELECT count(*) FROM execution_order WHERE idempotency_key = :key"),
                {"key": idempotency_key(intent)},
            )
        ).scalar_one()
    assert count == 1


async def test_concurrent_distinct_submissions_all_land() -> None:
    # The mechanism refuses duplicates, not writes.
    security_id = await create_security()
    intents = [
        intent_for(security_id, quantity_shares=index + 1) for index in range(CONCURRENT_WRITERS)
    ]
    gate = asyncio.Barrier(CONCURRENT_WRITERS)
    results = await asyncio.gather(*(submit_and_commit(intent, gate) for intent in intents))
    assert len({result.order_id for result in results}) == CONCURRENT_WRITERS
    assert all(not result.was_already_recorded for result in results)


async def test_concurrent_appends_of_one_event_leave_one_transition() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    gate = asyncio.Barrier(CONCURRENT_WRITERS)
    outcomes = await asyncio.gather(
        *(append_and_commit(order_id, OrderEvent.RELEASE, gate) for _ in range(CONCURRENT_WRITERS)),
        return_exceptions=True,
    )
    refused = [item for item in outcomes if isinstance(item, ConcurrentTransitionError)]
    written = [item for item in outcomes if not isinstance(item, BaseException)]
    assert len(written) == 1
    assert len(refused) == CONCURRENT_WRITERS - 1
    async with ingest_writer_session() as session:
        loaded = await load_order(session, order_id)
    assert loaded.sequence_number == 1
    assert loaded.state is OrderState.PENDING_NEW


async def test_a_duplicate_key_is_refused_by_the_database_not_by_python() -> None:
    # The raw INSERT proves the constraint, not the code path around it.
    security_id = await create_security()
    intent = intent_for(security_id)
    order_id = await seed_order(security_id)
    assert order_id > 0
    with pytest.raises(IntegrityError):
        await raw_insert(
            "INSERT INTO execution_order (idempotency_key, idempotency_schema, "
            "idempotency_preimage, security_id, side, quantity_shares, order_type, "
            "time_in_force, limit_price_usd, rebalance_date, git_commit, git_dirty, "
            "data_version, config_hash, seed) VALUES (:key, 'x', :preimage, "
            ":security_id, 'buy', 1, 'market', 'day', NULL, :date, :commit, false, "
            "'v', :config_hash, 0)",
            key=idempotency_key(intent),
            preimage=idempotency_preimage(intent),
            security_id=security_id,
            date=REBALANCE_DATE,
            commit="a" * 40,
            config_hash="1" * 64,
        )


async def test_a_key_that_is_not_a_digest_is_refused() -> None:
    # A counter wearing a digest's clothes cannot survive the restart the key
    # exists for, so the schema refuses one.
    security_id = await create_security()
    with pytest.raises(IntegrityError):
        await raw_insert(
            "INSERT INTO execution_order (idempotency_key, idempotency_schema, "
            "idempotency_preimage, security_id, side, quantity_shares, order_type, "
            "time_in_force, limit_price_usd, rebalance_date, git_commit, git_dirty, "
            "data_version, config_hash, seed) VALUES ('order-00001', 'x', '{}', "
            ":security_id, 'buy', 1, 'market', 'day', NULL, :date, :commit, false, "
            "'v', :config_hash, 0)",
            security_id=security_id,
            date=REBALANCE_DATE,
            commit="a" * 40,
            config_hash="1" * 64,
        )


async def test_a_non_paper_venue_is_refused_by_the_database() -> None:
    # Directive §1.1, §9.5. No writer supplies this column; one that tries is
    # refused by the CHECK regardless of what Python believes.
    security_id = await create_security()
    with pytest.raises(IntegrityError):
        await raw_insert(
            "INSERT INTO execution_order (idempotency_key, idempotency_schema, "
            "idempotency_preimage, venue, security_id, side, quantity_shares, "
            "order_type, time_in_force, limit_price_usd, rebalance_date, git_commit, "
            "git_dirty, data_version, config_hash, seed) VALUES (:key, 'x', '{}', "
            "'live', :security_id, 'buy', 1, 'market', 'day', NULL, :date, :commit, "
            "false, 'v', :config_hash, 0)",
            key="b" * 64,
            security_id=security_id,
            date=REBALANCE_DATE,
            commit="a" * 40,
            config_hash="1" * 64,
        )


async def test_a_limit_order_without_a_price_is_refused_by_the_database() -> None:
    security_id = await create_security()
    with pytest.raises(IntegrityError):
        await raw_insert(
            "INSERT INTO execution_order (idempotency_key, idempotency_schema, "
            "idempotency_preimage, security_id, side, quantity_shares, order_type, "
            "time_in_force, limit_price_usd, rebalance_date, git_commit, git_dirty, "
            "data_version, config_hash, seed) VALUES (:key, 'x', '{}', :security_id, "
            "'buy', 1, 'limit', 'day', NULL, :date, :commit, false, 'v', "
            ":config_hash, 0)",
            key="c" * 64,
            security_id=security_id,
            date=REBALANCE_DATE,
            commit="a" * 40,
            config_hash="1" * 64,
        )


async def test_a_transition_out_of_a_terminal_state_is_refused_by_the_database() -> None:
    # FILLED -> PENDING_NEW, issued as raw SQL. The CHECK is what makes it
    # unrepresentable rather than merely unreachable through this package.
    security_id = await create_security()
    order_id = await seed_order(security_id)
    with pytest.raises(IntegrityError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                from_state=OrderState.FILLED.value,
                event=OrderEvent.RELEASE.value,
                to_state=OrderState.PENDING_NEW.value,
            ),
        )


async def test_a_transition_outside_the_enumerated_table_is_refused() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    with pytest.raises(IntegrityError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.RELEASE.value,
                to_state=OrderState.PENDING_NEW.value,
            ),
        )


async def test_a_non_fill_event_carrying_a_payload_is_refused() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    with pytest.raises(IntegrityError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                fill_quantity_shares=10,
                fill_price_usd=Decimal("1"),
                fill_source=FillSource.SIMULATED.value,
                fill_cost_basis="lower_bound",
            ),
        )


async def test_a_fill_event_missing_its_payload_is_refused() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=10)
    await walk_to_acknowledged(order_id)
    with pytest.raises(IntegrityError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.FILL_COMPLETE.value,
                to_state=OrderState.FILLED.value,
                filled_quantity_after_shares=10,
            ),
        )


async def test_a_live_fill_source_is_refused_by_the_database() -> None:
    # I3: there is no value denoting a live execution, and the CHECK is what
    # makes that true for a writer that never imported this package.
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=10)
    await walk_to_acknowledged(order_id)
    with pytest.raises(IntegrityError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.FILL_COMPLETE.value,
                to_state=OrderState.FILLED.value,
                filled_quantity_after_shares=10,
                fill_quantity_shares=10,
                fill_price_usd=Decimal("1"),
                fill_source="live_broker",
                fill_cost_basis="lower_bound",
            ),
        )


async def test_a_fill_claiming_a_calibrated_cost_basis_is_refused() -> None:
    # D-013: a paper fill may not be relabelled as a measured estimate without a
    # migration and the review that comes with one.
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=10)
    await walk_to_acknowledged(order_id)
    with pytest.raises(IntegrityError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.FILL_COMPLETE.value,
                to_state=OrderState.FILLED.value,
                filled_quantity_after_shares=10,
                fill_quantity_shares=10,
                fill_price_usd=Decimal("1"),
                fill_source=FillSource.SIMULATED.value,
                fill_cost_basis="calibrated",
            ),
        )


async def test_the_chain_guard_refuses_a_gap_in_the_sequence() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    with pytest.raises(DBAPIError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(order_id=order_id, sequence_number=2),
        )


async def test_the_chain_guard_refuses_a_history_that_does_not_connect() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await raw_insert(TRANSITION_INSERT, **transition_values(order_id=order_id))
    with pytest.raises(DBAPIError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=2,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.EXPIRE.value,
                to_state=OrderState.EXPIRED.value,
            ),
        )


async def test_the_chain_guard_refuses_a_cumulative_quantity_that_is_not_the_running_sum() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=100)
    await walk_to_acknowledged(order_id)
    with pytest.raises(DBAPIError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.PARTIAL_FILL.value,
                to_state=OrderState.PARTIALLY_FILLED.value,
                filled_quantity_after_shares=55,
                fill_quantity_shares=40,
                fill_price_usd=Decimal("1"),
                fill_source=FillSource.SIMULATED.value,
                fill_cost_basis="lower_bound",
            ),
        )


async def test_the_chain_guard_refuses_an_overfill() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=100)
    await walk_to_acknowledged(order_id)
    with pytest.raises(DBAPIError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.PARTIAL_FILL.value,
                to_state=OrderState.PARTIALLY_FILLED.value,
                filled_quantity_after_shares=150,
                fill_quantity_shares=150,
                fill_price_usd=Decimal("1"),
                fill_source=FillSource.SIMULATED.value,
                fill_cost_basis="lower_bound",
            ),
        )


async def test_the_chain_guard_refuses_a_filled_order_that_did_not_trade_in_full() -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=100)
    await walk_to_acknowledged(order_id)
    with pytest.raises(DBAPIError):
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.FILL_COMPLETE.value,
                to_state=OrderState.FILLED.value,
                filled_quantity_after_shares=60,
                fill_quantity_shares=60,
                fill_price_usd=Decimal("1"),
                fill_source=FillSource.SIMULATED.value,
                fill_cost_basis="lower_bound",
            ),
        )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE execution_order SET quantity_shares = 1 WHERE order_id = :order_id",
        "DELETE FROM execution_order WHERE order_id = :order_id",
    ],
)
async def test_the_order_table_is_append_only(statement: str) -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    with pytest.raises(DBAPIError):
        await mutate(statement, order_id)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE execution_order_transition SET to_state = 'filled' WHERE order_id = :order_id",
        "DELETE FROM execution_order_transition WHERE order_id = :order_id",
    ],
)
async def test_the_transition_table_is_append_only(statement: str) -> None:
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await walk_to_acknowledged(order_id)
    with pytest.raises(DBAPIError):
        await mutate(statement, order_id)
