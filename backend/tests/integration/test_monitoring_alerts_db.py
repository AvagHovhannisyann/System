"""P12.1/P12.4 against a real TimescaleDB: the constraints, the triggers, the halt gate.

**Not executed where this was written — no Docker daemon is available here — and
deliberately neither skipped nor xfailed (I6).** Where the daemon is missing
these error on the environment rather than reporting a success they did not
earn. Migration 0017 declares ``down_revision = "0016"`` and depends on that
revision by identifier alone; until 0016 lands from its own track ``alembic
upgrade head`` cannot resolve the chain, so this module cannot run here for two
independent reasons.

Why it lives here rather than in ``backend/tests/monitoring/``
--------------------------------------------------------------

This directory's ``conftest.py`` already owns the container, the migrated
schema and the per-test engine disposal. A second copy of that wiring under
``backend/tests/monitoring/`` would be a second thing to keep in step, and it
would also have needed the module-private migration engine for its own
``TRUNCATE`` — an import the D-011 contract bans outside ``backend/db`` and
sanctions only by an explicit allowlist. The rest of the P12 suite needs no
database at all and stays where it is.

**Isolation is by rollback, not by TRUNCATE.** Every test below runs inside one
transaction and never commits, so the monitoring tables are clean again the
moment the session closes — no reset path, no privileged role, nothing to
allowlist. That works because ``PostgresAlertStore`` performs its
constraint-deciding inserts inside *savepoints*
(:meth:`~sqlalchemy.ext.asyncio.AsyncSession.begin_nested`), so a refused
duplicate leaves the surrounding transaction usable rather than poisoned.

The one exception is the concurrency test, which has to commit — a writer that
rolls back releases the unique index and the loser then *succeeds*, which is the
opposite of the property under test. Its residue is made inert two ways: the
condition carries a per-run UUID so it can never collide with anything, and the
test acknowledges its own alert so it does not appear in any later
``open_alerts()`` result.

What only a database can be asked
---------------------------------

- **``UNIQUE (dedup_key)`` doing the deciding.** ``PostgresAlertStore.record``
  deliberately does not look before it inserts. Whether that window is closed is
  a property of Postgres and the constraint. Here monitoring workers race
  through it on separate connections in separate transactions, and exactly one
  row survives.
- **The CHECK constraints refusing a writer that skips Python.** The point of
  restating the severity vocabulary, the digest shape and the two halt-row
  shapes in SQL is that they bind raw ``INSERT``s. That claim is only testable
  by issuing raw ``INSERT``s.
- **The append-only triggers** rejecting UPDATE and DELETE — including, most
  importantly, an UPDATE of an acknowledgement, which is how a signature would
  otherwise be quietly reassigned.
- **The derived halt state** — the earliest halt no resume points at — and the
  acknowledgement gate on resuming, both of which are queries across tables.

The seam the execution-side kill switch reads is exercised here too:
:func:`~backend.monitoring.history.require_not_halted` raises while a halt is
open and returns once it is resumed.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError

from backend.db import ingest_writer_session
from backend.monitoring.alerts import (
    DeliveryOutcome,
    PostgresAlertStore,
    dispatch,
    pending_escalations,
)
from backend.monitoring.errors import (
    AlertUndeliverableError,
    HaltHistoryError,
    SystemHaltedError,
    UnacknowledgedHaltError,
)
from backend.monitoring.expectation import HaltAction, HaltCause, HaltDecision, decide
from backend.monitoring.history import (
    AUTOMATIC_ACTOR,
    HaltEventKind,
    active_halt,
    halt_history,
    record_halt,
    record_resume,
    require_not_halted,
)
from backend.tests.monitoring import store_contract
from backend.tests.monitoring.alert_fixtures import LATER, RAISED_AT, fixture_alert
from backend.tests.monitoring.doubles import FailingChannel, RecordingChannel
from backend.tests.monitoring.expectation_fixtures import (
    DECISION_AT,
    fixture_band,
    live_window,
    series_with_sharpe,
)
from backend.tests.monitoring.fixtures import fixture_stamp

_OPERATOR = "operator@example.invalid"


def _halt_decision() -> HaltDecision:
    """Build a halting decision from an injected performance deviation."""
    band = fixture_band()
    live = live_window(
        series_with_sharpe(band.lower - 0.2 * band.dispersion, n_periods=band.window_periods)
    )
    decision = decide(band=band, live=live, stamp=fixture_stamp(), now=DECISION_AT)
    assert decision.action is HaltAction.HALT
    assert decision.cause is HaltCause.BELOW_EXPECTED_BAND
    return decision


# ---------------------------------------------------------------------------
# 1. The shared contract, against the real store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("contract", store_contract.CONTRACT, ids=lambda body: body.__name__)
async def test_contract_holds_for_the_postgres_store(contract: object) -> None:
    assert callable(contract)
    async with ingest_writer_session() as session:
        await contract(PostgresAlertStore(session=session))
        await session.rollback()


def test_every_contract_body_is_wired_into_this_runner() -> None:
    """The same guard the in-memory runner carries: both backends or neither."""
    marks = getattr(test_contract_holds_for_the_postgres_store, "pytestmark", [])
    parametrized = [mark for mark in marks if mark.name == "parametrize"]
    assert len(parametrized) == 1
    assert len(parametrized[0].args[1]) == len(store_contract.CONTRACT)


# ---------------------------------------------------------------------------
# 2. What only the database enforces
# ---------------------------------------------------------------------------


async def test_two_workers_racing_one_condition_leave_exactly_one_alert() -> None:
    """The window a look-then-insert would leave open, closed by the constraint.

    The only committing test in the module (see the module docstring): a writer
    that rolled back would release the index and let the loser succeed, which is
    the opposite of the property under test. The condition carries a per-run
    UUID and the alert is acknowledged at the end, so the committed row cannot
    collide with, or appear in, anything else.
    """
    alert = fixture_alert(condition=f"race-{uuid.uuid4().hex}")

    async def record() -> bool:
        async with ingest_writer_session() as session:
            stored = await PostgresAlertStore(session=session).record(alert, now=RAISED_AT)
            await session.commit()
            return stored.is_new

    results = await asyncio.gather(record(), record(), record())
    assert sum(results) == 1, "exactly one writer may create the row"
    async with ingest_writer_session() as session:
        count = await session.execute(
            sa.text("SELECT count(*) FROM monitoring_alert WHERE dedup_key = :key"),
            {"key": alert.dedup_key},
        )
        assert count.scalar_one() == 1
        await PostgresAlertStore(session=session).acknowledge(
            dedup_key=alert.dedup_key,
            acknowledged_by=_OPERATOR,
            note="race fixture; closed so it cannot appear in a later open-alert query",
            now=LATER,
        )
        await session.commit()


@pytest.mark.parametrize(
    ("column", "value", "constraint"),
    [
        ("severity", "urgent", "ck_monitoring_alert_severity_known"),
        ("dedup_key", "not-a-digest", "ck_monitoring_alert_dedup_key_is_digest"),
        ("config_hash", "abc", "ck_monitoring_alert_config_hash_is_sha256"),
    ],
)
async def test_a_raw_insert_that_skips_python_is_still_refused(
    column: str, value: str, constraint: str
) -> None:
    """The CHECKs bind a writer that never imports this package."""
    row = {
        "dedup_key": "a" * 64,
        "rule_id": "raw",
        "severity": "critical",
        "subject": "s",
        "detail": "d",
        "payload": "{}",
        "raised_at": RAISED_AT,
        "git_commit": "0" * 40,
        "git_dirty": False,
        "data_version": "v",
        "config_hash": "0" * 64,
        "seed": 0,
    }
    row[column] = value
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError) as raised:
            await session.execute(
                sa.text(
                    "INSERT INTO monitoring_alert (dedup_key, rule_id, severity, subject, "
                    "detail, payload, raised_at, git_commit, git_dirty, data_version, "
                    "config_hash, seed) VALUES (:dedup_key, :rule_id, :severity, :subject, "
                    ":detail, CAST(:payload AS jsonb), :raised_at, :git_commit, :git_dirty, "
                    ":data_version, :config_hash, :seed)"
                ),
                row,
            )
        assert constraint in str(raised.value)
        await session.rollback()


@pytest.mark.parametrize(
    ("table", "key_column"),
    [
        ("monitoring_alert", "alert_id"),
        ("monitoring_alert_delivery", "delivery_id"),
        ("monitoring_alert_acknowledgement", "acknowledgement_id"),
        ("monitoring_halt_event", "halt_event_id"),
    ],
)
async def test_every_monitoring_table_refuses_update_and_delete(
    table: str, key_column: str
) -> None:
    """Append-only, including the acknowledgement — a signature is never reassigned."""
    alert = fixture_alert(condition=f"append-only-{table}")
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        await store.record(alert, now=RAISED_AT)
        await store.record_attempt(
            dedup_key=alert.dedup_key,
            channel_id="pager",
            outcome=DeliveryOutcome.FAILED,
            detail="simulated outage",
            now=RAISED_AT,
        )
        await store.acknowledge(
            dedup_key=alert.dedup_key, acknowledged_by=_OPERATOR, note="seen", now=LATER
        )
        await record_halt(session, _halt_decision(), alert_dedup_key=alert.dedup_key)
        await session.flush()

        # Both refusals run inside savepoints so the transaction survives to be
        # rolled back as a whole, which is this module's isolation mechanism.
        for statement in (
            # Identifiers cannot be bind parameters; both values come from this
            # module's own parametrize list, never from any input.
            f"UPDATE {table} SET {key_column} = {key_column}",  # noqa: S608
            f"DELETE FROM {table}",  # noqa: S608
        ):
            with pytest.raises(DBAPIError) as refused:
                async with session.begin_nested():
                    await session.execute(sa.text(statement))
            assert "append-only" in str(refused.value)
        await session.rollback()


@pytest.mark.parametrize(
    ("kind", "cause", "resolves"),
    [("halt", None, None), ("resume", "internal_error", 1), ("resume", None, None)],
)
async def test_a_malformed_halt_row_is_refused_by_shape(
    kind: str, cause: str | None, resolves: int | None
) -> None:
    """A halt without a cause, or a resume that resolves nothing, is not representable."""
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError) as raised:
            await session.execute(
                sa.text(
                    "INSERT INTO monitoring_halt_event (kind, cause, detail, actor, "
                    "occurred_at, resolves_halt_event_id, git_commit, git_dirty, "
                    "data_version, config_hash, seed) VALUES (:kind, :cause, 'd', 'a', "
                    ":occurred_at, :resolves, :commit, false, 'v', :config, 0)"
                ),
                {
                    "kind": kind,
                    "cause": cause,
                    "occurred_at": RAISED_AT,
                    "resolves": resolves,
                    "commit": "0" * 40,
                    "config": "0" * 64,
                },
            )
        assert "shape_matches_kind" in str(raised.value) or "fk_monitoring_halt_event" in str(
            raised.value
        )
        await session.rollback()


# ---------------------------------------------------------------------------
# 3. The halt history and the kill-switch seam
# ---------------------------------------------------------------------------


async def test_a_halt_is_derived_from_the_history_and_gates_trading() -> None:
    """The whole round trip: halt, gate closed, acknowledge, resume, gate open."""
    alert = fixture_alert(condition="halt-round-trip")
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        await dispatch(alert, store=store, channels=[RecordingChannel()], now=RAISED_AT)
        event = await record_halt(session, _halt_decision(), alert_dedup_key=alert.dedup_key)
        assert event.kind is HaltEventKind.HALT
        assert event.actor == AUTOMATIC_ACTOR

        current = await active_halt(session)
        assert current is not None
        assert current.halt_event_id == event.halt_event_id
        assert current.cause is HaltCause.BELOW_EXPECTED_BAND
        # The decision payload travels onto the row, band and disclosures included.
        assert current.decision is not None
        assert current.decision["cause"] == "below_expected_band"

        with pytest.raises(SystemHaltedError) as gated:
            await require_not_halted(session)
        assert gated.value.halt_event_id == event.halt_event_id

        # Nobody has signed for the alert, so trading does not restart.
        with pytest.raises(UnacknowledgedHaltError):
            await record_resume(
                session,
                halt_event_id=event.halt_event_id,
                actor=_OPERATOR,
                reason="looks fine to me",
                stamp=fixture_stamp(),
            )

        await store.acknowledge(
            dedup_key=alert.dedup_key,
            acknowledged_by=_OPERATOR,
            note="reviewed the CPCV band and the cost calibration",
            now=LATER,
        )
        resume = await record_resume(
            session,
            halt_event_id=event.halt_event_id,
            actor=_OPERATOR,
            reason="cost model recalibrated; deviation explained",
            stamp=fixture_stamp(),
            now=LATER,
        )
        assert resume.kind is HaltEventKind.RESUME
        assert resume.resolves_halt_event_id == event.halt_event_id

        assert await active_halt(session) is None
        # Returns None; the seam's whole contract is that it RAISES when halted.
        await require_not_halted(session)
        history = await halt_history(session)
        assert [entry.kind for entry in history] == [HaltEventKind.RESUME, HaltEventKind.HALT]
        await session.rollback()


async def test_a_halt_cannot_be_resumed_twice() -> None:
    """One resume per halt, so the derived state is never ambiguous."""
    alert = fixture_alert(condition="double-resume")
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        await store.record(alert, now=RAISED_AT)
        await store.acknowledge(
            dedup_key=alert.dedup_key, acknowledged_by=_OPERATOR, note="seen", now=LATER
        )
        event = await record_halt(session, _halt_decision(), alert_dedup_key=alert.dedup_key)
        await record_resume(
            session,
            halt_event_id=event.halt_event_id,
            actor=_OPERATOR,
            reason="investigated",
            stamp=fixture_stamp(),
            now=LATER,
        )
        with pytest.raises(HaltHistoryError, match="already resumed"):
            await record_resume(
                session,
                halt_event_id=event.halt_event_id,
                actor=_OPERATOR,
                reason="again",
                stamp=fixture_stamp(),
                now=LATER + dt.timedelta(hours=1),
            )
        await session.rollback()


async def test_a_halt_with_no_alert_can_still_be_resumed() -> None:
    """The acknowledgement gate applies to halts that raised an alert, not to all halts."""
    async with ingest_writer_session() as session:
        event = await record_halt(session, _halt_decision())
        await record_resume(
            session,
            halt_event_id=event.halt_event_id,
            actor=_OPERATOR,
            reason="manual halt cleared",
            stamp=fixture_stamp(),
            now=LATER,
        )
        assert await active_halt(session) is None
        await session.rollback()


async def test_the_earliest_open_halt_is_the_one_reported() -> None:
    """Two halts, one resumed: the outage that is still running is the answer."""
    async with ingest_writer_session() as session:
        assert await active_halt(session) is None, "the suite must start with no open halt"
        first = await record_halt(session, _halt_decision())
        second = await record_halt(session, _halt_decision())
        assert second.halt_event_id != first.halt_event_id
        current = await active_halt(session)
        assert current is not None
        assert current.halt_event_id == first.halt_event_id
        await session.rollback()


async def test_an_undeliverable_alert_is_queryable_afterwards() -> None:
    """Which alerts reached nobody is a query — that is what "not dropped" means."""
    alert = fixture_alert(condition="undeliverable-db")
    async with ingest_writer_session() as session:
        store = PostgresAlertStore(session=session)
        with pytest.raises(AlertUndeliverableError):
            await dispatch(alert, store=store, channels=[FailingChannel()], now=RAISED_AT)

        stored = await store.get(alert.dedup_key)
        assert stored.escalated
        assert not stored.delivered
        pending = await pending_escalations(store, now=RAISED_AT + dt.timedelta(minutes=1))
        assert alert.dedup_key in {item.alert.dedup_key for item in pending}
        await session.rollback()
