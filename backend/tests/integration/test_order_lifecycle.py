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

What the first CI run found
--------------------------

This module was written where no Docker daemon was available, so it had never
executed. Its first run failed five tests, and none of the five was a typo — each
was a claim the local unit suite structurally could not check:

- **A ``BEFORE INSERT`` trigger fires ahead of every CHECK constraint.** Four
  tests aimed a raw ``INSERT`` at a named CHECK while also breaking a chain
  property, so the chain guard refused the row first and the CHECK was never
  consulted. Every one of them now builds a valid chain prefix (see
  :func:`walk_to_acknowledged`) so the constraint under test is the only thing
  violated, and asserts the constraint's name in the message rather than merely
  that *something* refused it.
- **A concurrent append is usually refused by that trigger, not by the primary
  key.** Under ``READ COMMITTED`` a losing writer's ``INSERT`` takes a fresh
  snapshot, so the trigger sees the winner's committed row and refuses before the
  index is consulted. The in-memory double modelled only the index, which is
  exactly the gap a double is capable of hiding. Migration 0014 now raises that
  branch ``USING ERRCODE = 'unique_violation'`` so both routes reach the store as
  one SQLSTATE, and the store decides on the code rather than the class.

Migration 0014 declares ``down_revision = "0013"`` and depends on that revision by
identifier alone; 0013 has since landed from its own track, so the chain is linear
with a single head at 0014.

Still not skipped, not xfailed (I6) — where the daemon is missing these error on
the environment rather than reporting a success they did not earn.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import re
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import ingest_writer_session
from backend.execution.errors import ConcurrentTransitionError, IllegalTransitionError
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
from backend.execution.store import (
    UNIQUE_VIOLATION_SQLSTATE,
    _is_position_taken,
    _sqlstate,
    append_transition,
    load_order,
    record_order,
)
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
    """Advance an order to ACKNOWLEDGED (sequence 2) and commit.

    Every test that targets a CHECK constraint on a transition row starts here.
    The ``BEFORE INSERT`` chain guard runs ahead of every constraint, so a raw
    row that also breaks a chain property is refused by the *trigger* and the
    CHECK is never consulted — which is how four tests in this module came to
    name constraints they never reached.
    """
    async with ingest_writer_session() as session:
        await append_transition(
            session, order_id=order_id, event=OrderEvent.RELEASE, occurred_at=MOMENT
        )
        await append_transition(
            session, order_id=order_id, event=OrderEvent.ACKNOWLEDGE, occurred_at=MOMENT
        )
        await session.commit()


async def walk_to_filled(order_id: int, quantity_shares: int) -> None:
    """Advance an order all the way to FILLED (sequence 3) and commit."""
    await walk_to_acknowledged(order_id)
    async with ingest_writer_session() as session:
        await append_transition(
            session,
            order_id=order_id,
            event=OrderEvent.FILL_COMPLETE,
            occurred_at=MOMENT,
            fill=FillReport(
                quantity_shares=quantity_shares,
                price_usd=Decimal("123.45"),
                source=FillSource.PAPER_BROKER,
            ),
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
    """Six writers claim sequence 1; one lands and five are refused, three ways.

    There are **three** legitimate refusals here, not two, and the third only
    became visible once CI ran this against a real database:

    1. ``ConcurrentTransitionError`` from the **primary key** — the loser's
       ``INSERT`` reached the index first and blocked until the winner committed.
    2. ``ConcurrentTransitionError`` from the **chain guard** — the loser's
       statement started afterwards, so its fresh ``READ COMMITTED`` snapshot
       already showed the winner's row. Migration 0014 gives that branch SQLSTATE
       ``23505`` deliberately, so both database routes become one retryable error
       and the caller never has to know which fired.
    3. ``IllegalTransitionError`` from the **state machine, in Python, before any
       insert is attempted** — the loser's ``load_order`` ran after the winner
       committed, so the order already read ``PENDING_NEW`` and ``release`` is no
       longer a legal event from there.

    The third is not a lesser outcome. It is the *same* refusal arriving earlier
    and cheaper, and it is exactly what ``ConcurrentTransitionError``'s own
    message tells a caller to expect: re-read the history, because the event may
    no longer be legal from the state that now holds. A retrying caller reaches
    it on the next attempt regardless.

    This test's history is the argument for the accounting assertion below. Its
    first version counted only ``ConcurrentTransitionError`` and passed against a
    double that modelled only the index; CI showed five refusals, none of that
    type. Its second version fixed the type and added the total, and CI showed
    ``1 + 3 == 6`` — two writers refused by a path nobody had named. Both times
    the invariant that actually matters held: exactly one transition on disk.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id)
    gate = asyncio.Barrier(CONCURRENT_WRITERS)
    outcomes = await asyncio.gather(
        *(append_and_commit(order_id, OrderEvent.RELEASE, gate) for _ in range(CONCURRENT_WRITERS)),
        return_exceptions=True,
    )
    written = [item for item in outcomes if not isinstance(item, BaseException)]
    lost_at_the_database = [
        item for item in outcomes if isinstance(item, ConcurrentTransitionError)
    ]
    lost_in_the_machine = [item for item in outcomes if isinstance(item, IllegalTransitionError)]
    refused = lost_at_the_database + lost_in_the_machine
    # Every outcome must be one of the three named above. Kept as a total rather
    # than a per-type count because which mechanism catches a given writer is a
    # scheduling accident, while "nothing was refused by a path we have not
    # thought about" is the property. A stray exception type cannot hide inside
    # "not written".
    assert len(written) + len(refused) == CONCURRENT_WRITERS, outcomes
    assert len(written) == 1
    assert len(refused) == CONCURRENT_WRITERS - 1
    # At least one loser must reach the database, or this test has stopped
    # exercising the constraint it exists for and become a state-machine test.
    assert lost_at_the_database, outcomes
    async with ingest_writer_session() as session:
        loaded = await load_order(session, order_id)
    assert loaded.sequence_number == 1
    assert loaded.state is OrderState.PENDING_NEW


async def test_the_chain_guard_refuses_a_taken_position_as_a_unique_violation() -> None:
    """The dominant concurrent refusal, proved deterministically instead of by racing.

    A writer whose statement starts after the winner committed never reaches the
    index: the ``BEFORE INSERT`` trigger sees the winner's row first and refuses
    it. That is reproduced here without a race by committing sequence 3 and then
    re-issuing it, which puts the trigger in exactly the state a losing writer
    puts it in.

    The assertions are the whole chain the fix depends on: Postgres raises it as
    ``23505``, and :func:`backend.execution.store._is_position_taken` — the
    predicate that decides retryable from malformed — says yes to it.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await walk_to_acknowledged(order_id)
    taken = transition_values(
        order_id=order_id,
        sequence_number=3,
        from_state=OrderState.ACKNOWLEDGED.value,
        event=OrderEvent.REQUEST_CANCEL.value,
        to_state=OrderState.PENDING_CANCEL.value,
    )
    await raw_insert(TRANSITION_INSERT, **taken)
    with pytest.raises(IntegrityError) as raised:
        await raw_insert(TRANSITION_INSERT, **taken)
    assert _sqlstate(raised.value) == UNIQUE_VIOLATION_SQLSTATE
    assert _is_position_taken(raised.value) is True
    assert "another writer claimed it first" in str(raised.value)


async def test_the_chain_guard_keeps_a_gap_out_of_the_retryable_code() -> None:
    """A gap is malformed, not contended, and must not arrive as ``23505``.

    Retrying it would loop forever, so the trigger leaves this branch under the
    default ``raise_exception`` code and the store lets it through unchanged.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await walk_to_acknowledged(order_id)
    with pytest.raises(DBAPIError) as raised:
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=9,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.REQUEST_CANCEL.value,
                to_state=OrderState.PENDING_CANCEL.value,
            ),
        )
    assert _sqlstate(raised.value) != UNIQUE_VIOLATION_SQLSTATE
    assert _is_position_taken(raised.value) is False
    assert "the next transition is 3, not 9" in str(raised.value)


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
    """``FILLED -> PENDING_NEW`` as raw SQL, on a chain that is otherwise valid.

    The order is walked to ``FILLED`` first so the appended row satisfies every
    chain property the ``BEFORE INSERT`` trigger checks — it is sequence 4 after
    sequence 3, it starts where the previous row ended, and it trades nothing.
    The trigger therefore passes and a CHECK is the only thing left to refuse it.

    **Two CHECKs necessarily refuse it together**, and that is a property of the
    schema rather than a looseness in the test: every triple with a terminal
    ``from_state`` is also absent from the enumerated table, so no input can
    isolate ``from_state_not_terminal``. The assertion names both and requires
    one of them, which is the strongest true claim available.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id, quantity_shares=10)
    await walk_to_filled(order_id, 10)
    with pytest.raises(IntegrityError) as raised:
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=4,
                from_state=OrderState.FILLED.value,
                event=OrderEvent.RELEASE.value,
                to_state=OrderState.PENDING_NEW.value,
                filled_quantity_after_shares=10,
            ),
        )
    assert re.search(r"from_state_not_terminal|legal_transition", str(raised.value))


async def test_a_transition_outside_the_enumerated_table_is_refused() -> None:
    """A non-terminal triple absent from the 25, isolating ``legal_transition``.

    ``acknowledged + acknowledge -> acknowledged`` is a self-loop the machine
    does not have. It passes the chain guard (sequence 3 after 2, starts where
    the previous row ended, trades nothing), and ``from_state`` is not terminal,
    so ``legal_transition`` is the only constraint it violates.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await walk_to_acknowledged(order_id)
    with pytest.raises(IntegrityError) as raised:
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.ACKNOWLEDGE.value,
                to_state=OrderState.ACKNOWLEDGED.value,
            ),
        )
    assert "legal_transition" in str(raised.value)


async def test_a_non_fill_event_carrying_a_payload_is_refused() -> None:
    """A payload on a non-fill event, with the chain arithmetic left consistent.

    The trigger sums ``COALESCE(fill_quantity_shares, 0)`` whatever the event is,
    so a payload of 10 shares must be matched by a cumulative of 10 or the
    trigger refuses the row before ``fill_payload_absent`` is reached. With the
    arithmetic satisfied and the triple legal, that CHECK is the only violation
    left — a quantity on a row that reports no trade.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await walk_to_acknowledged(order_id)
    with pytest.raises(IntegrityError) as raised:
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.REQUEST_CANCEL.value,
                to_state=OrderState.PENDING_CANCEL.value,
                filled_quantity_after_shares=10,
                fill_quantity_shares=10,
                fill_price_usd=Decimal("1"),
                fill_source=FillSource.SIMULATED.value,
                fill_cost_basis="lower_bound",
            ),
        )
    assert "fill_payload_absent" in str(raised.value)


@pytest.mark.parametrize(
    ("missing", "quantity", "cumulative"),
    [
        ("fill_price_usd", 40, 40),
        ("fill_source", 40, 40),
        ("fill_cost_basis", 40, 40),
        ("fill_quantity_shares", None, 0),
    ],
    ids=["price", "source", "cost_basis", "quantity"],
)
async def test_a_fill_event_missing_part_of_its_payload_is_refused(
    missing: str, quantity: int | None, cumulative: int
) -> None:
    """``fill_payload_present`` refuses a fill missing any one of its four columns.

    The original version of this test aimed at ``fill_complete`` with a NULL
    quantity and never reached the CHECK: with no quantity the trigger's running
    sum stays where it was, while ``fill_complete -> filled`` requires the order
    to be fully traded, so the trigger refused it first. That combination is in
    fact **unreachable** — reaching ``filled`` always means the cumulative
    quantity equals the ordered quantity, which cannot happen with nothing
    traded.

    ``partial_fill`` is reachable, and so the CHECK is not subsumed by the
    trigger. Each of the four columns is dropped in turn; a NULL quantity keeps
    the cumulative where it was so the trigger still passes, and the other three
    leave the arithmetic untouched. Each row therefore violates
    ``fill_payload_present`` and nothing else — the ``IS NULL OR ...`` shape of
    the sibling CHECKs means an absent value satisfies them.
    """
    security_id = await create_security()
    order_id = await seed_order(security_id)
    await walk_to_acknowledged(order_id)
    payload: dict[str, object] = {
        "fill_quantity_shares": quantity,
        "fill_price_usd": Decimal("123.45"),
        "fill_source": FillSource.SIMULATED.value,
        "fill_cost_basis": "lower_bound",
    }
    payload[missing] = None
    with pytest.raises(IntegrityError) as raised:
        await raw_insert(
            TRANSITION_INSERT,
            **transition_values(
                order_id=order_id,
                sequence_number=3,
                from_state=OrderState.ACKNOWLEDGED.value,
                event=OrderEvent.PARTIAL_FILL.value,
                to_state=OrderState.PARTIALLY_FILLED.value,
                filled_quantity_after_shares=cumulative,
                **payload,
            ),
        )
    assert "fill_payload_present" in str(raised.value)


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
