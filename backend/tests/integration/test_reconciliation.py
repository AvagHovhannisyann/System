"""P11.3/P11.5 against a real TimescaleDB: the constraints, the guards, and a restart.

The unit suite in ``backend/tests/execution`` covers the comparison, the
fail-closed evaluation and the stores' control flow — the last of those against
an in-memory double. What a double structurally cannot cover is the half the
design rests on:

- **the CHECK constraints refusing a writer that skips Python.** The point of
  restating the tolerance ceiling, the non-real ``reported_origin``, the halt
  trigger vocabulary and the ``matched = (break_count = 0)`` identity in SQL is
  that they bind raw ``INSERT``s. That claim is only testable by issuing raw
  ``INSERT``s.
- **the clearance guard**, whose conditions are relations *between* rows and
  therefore cannot be a CHECK at all — and whose refusals split by SQLSTATE
  (D-034), which only Postgres can produce.
- **the guard's silence on engagements.** That nothing can refuse a halt is the
  property the whole kill switch depends on, and it is worth proving against the
  real trigger rather than against a double written by the same person.
- **the append-only triggers** rejecting ``UPDATE`` and ``DELETE``.
- **a halt surviving a genuine restart**: the engine is disposed and a new session
  is built, so the halt is read back over a fresh connection pool by code that
  kept nothing.

Migration 0016 declares ``down_revision = "0015"`` and depends on that revision by
identifier alone; 0015 is another track's file and is never read here.

**Unrun in this environment: there is no Docker daemon available**, so every test
in this module errors on the missing container rather than reporting a pass it did
not earn. Not skipped and not xfailed (I6) — see the report accompanying this
change for the exact list.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db import dispose_database, ingest_writer_session
from backend.execution.errors import (
    HaltAlreadyClearedError,
    HaltClearanceError,
    SystemHaltedError,
)
from backend.execution.halt import (
    HaltReason,
    HaltTrigger,
    assert_not_halted,
    clear_halt,
    engage_halt,
    open_halts,
)
from backend.execution.killswitch import (
    DataFreshnessObservation,
    DrawdownObservation,
    ManualHaltRequest,
    guard_before_release,
    run_cycle,
)
from backend.execution.reconciliation import (
    MAX_CASH_TOLERANCE_USD,
    SnapshotOrigin,
    load_reconciliation,
    reconcile,
    record_reconciliation,
    rerun,
)
from backend.tests.execution.control_fixtures import (
    CYCLE_ID,
    OBSERVED_AT,
    broken_result,
    clean_result,
    internal_snapshot,
    observation,
    reported_snapshot,
)
from backend.tests.execution.fixtures import COMMIT_B, CONFIG_HASH_B, make_stamp

if TYPE_CHECKING:
    from collections.abc import Mapping

CONCURRENT_CLEARERS = 5
"""Kept modest: each clearer holds its own connection for the whole contention."""

_STAMP_COLUMNS: Mapping[str, object] = {
    "git_commit": "a" * 40,
    "git_dirty": False,
    "data_version": "sharadar-2026-08-01",
    "config_hash": "1" * 64,
    "seed": 7,
}


def _insert_sql(table: str, values: Mapping[str, object]) -> str:
    """Render an INSERT whose values are all bound and whose table is a constant.

    The point of restating the constraints in SQL is that they bind a writer that
    skips this codebase, and that claim is only testable by being such a writer.
    Table and column names are interpolated because they cannot be bound
    parameters; every *value* is bound. Both table names are module constants,
    never caller input.
    """
    columns = ", ".join(values)
    binds = ", ".join(f":{name}" for name in values)
    return f"INSERT INTO {table} ({columns}) VALUES ({binds})"  # noqa: S608 - see docstring


async def raw_insert(table: str, values: Mapping[str, object]) -> None:
    """Insert one row with raw SQL, bypassing every Python-side guard."""
    async with ingest_writer_session() as session:
        await session.execute(sa.text(_insert_sql(table, values)), values)
        await session.commit()


async def insert_halt_row(**overrides: object) -> int:
    """Insert one halt row raw and return its generated ``halt_id``.

    Needed by the probes that must name a real foreign-key target: a dangling
    ``clears_halt_id`` would let the foreign key fire instead of the CHECK under
    test, which is the same "the row breaks more than one thing" failure these
    probes exist to avoid.
    """
    values = halt_row(**overrides)
    statement = f"{_insert_sql('execution_halt', values)} RETURNING halt_id"
    async with ingest_writer_session() as session:
        halt_id = int((await session.execute(sa.text(statement), values)).scalar_one())
        await session.commit()
    return halt_id


def halt_row(**overrides: object) -> dict[str, object]:
    """Build a valid ``execution_halt`` engagement row for a raw insert."""
    row: dict[str, object] = {
        "event": "engaged",
        "halt_trigger": "manual",
        "cycle_id": CYCLE_ID,
        "detail": "engaged by a raw insert",
        "evidence": "{}",
        "occurred_at": OBSERVED_AT,
        **_STAMP_COLUMNS,
    }
    row.update(overrides)
    return row


def reconciliation_row(**overrides: object) -> dict[str, object]:
    """Build a valid ``execution_reconciliation`` row for a raw insert."""
    result = clean_result()
    row: dict[str, object] = {
        "cycle_id": CYCLE_ID,
        "internal_origin": "internal_ledger",
        "reported_origin": "simulated",
        "internal_snapshot": "{}",
        "internal_digest": "0" * 64,
        "reported_snapshot": "{}",
        "reported_digest": "0" * 64,
        "internal_observed_at": OBSERVED_AT,
        "reported_observed_at": OBSERVED_AT,
        "cash_tolerance_usd": result.cash_tolerance_usd,
        "findings": "[]",
        "finding_count": 0,
        "break_count": 0,
        "matched": True,
        "result_digest": "0" * 64,
        **_STAMP_COLUMNS,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# The gate, both directions, through a real database.
# ---------------------------------------------------------------------------


async def test_an_injected_mismatch_is_caught_and_halts_within_the_cycle() -> None:
    # Gate G11's first two clauses in one test: the injected break is found, and
    # the halt it causes lands in the observing cycle and stops the release.
    async with ingest_writer_session() as session:
        result = broken_result()
        assert not result.matched
        reconciliation_id = await record_reconciliation(session, result)
        outcome = await run_cycle(session, observation(reconciliation=result))
        await session.commit()

    assert outcome.halted
    assert not outcome.release_permitted
    assert [reason.trigger for reason in outcome.decision.reasons] == [
        HaltTrigger.RECONCILIATION_MISMATCH
    ]

    async with ingest_writer_session() as session:
        halts = await open_halts(session)
        assert len(halts) == 1
        assert halts[0].cycle_id == CYCLE_ID
        assert halts[0].trigger is HaltTrigger.RECONCILIATION_MISMATCH
        with pytest.raises(SystemHaltedError):
            await guard_before_release(session)
        stored = await load_reconciliation(session, reconciliation_id)
        assert stored.result_digest == result.result_digest


async def test_a_clean_reconciliation_does_not_halt_anything() -> None:
    # The false-positive direction. A control that fires on a healthy book is
    # ignored within a week, which is the same outcome as one that never fires.
    async with ingest_writer_session() as session:
        result = clean_result()
        assert result.matched
        await record_reconciliation(session, result)
        outcome = await run_cycle(session, observation(reconciliation=result))
        await session.commit()
    assert outcome.release_permitted
    async with ingest_writer_session() as session:
        assert await open_halts(session) == ()
        await guard_before_release(session)


async def test_a_stored_verdict_reruns_identically_from_the_database() -> None:
    async with ingest_writer_session() as session:
        result = broken_result()
        reconciliation_id = await record_reconciliation(session, result)
        await session.commit()
    await dispose_database()
    async with ingest_writer_session() as session:
        stored = await load_reconciliation(session, reconciliation_id)
    replayed = rerun(stored, stamp=make_stamp(git_commit=COMMIT_B, config_hash=CONFIG_HASH_B))
    assert replayed.result_digest == result.result_digest
    assert [finding.detail for finding in replayed.findings] == [
        finding.detail for finding in result.findings
    ]


@pytest.mark.parametrize(
    ("overrides", "trigger"),
    [
        (
            {
                "drawdown": DrawdownObservation(
                    equity_usd=Decimal("80000.00"),
                    peak_equity_usd=Decimal("100000.00"),
                    limit_fraction=Decimal("0.10"),
                )
            },
            HaltTrigger.DRAWDOWN_BREACH,
        ),
        (
            {
                "freshness": DataFreshnessObservation(
                    age_seconds=Decimal("7200"), max_age_seconds=Decimal("3600")
                )
            },
            HaltTrigger.STALE_DATA,
        ),
        ({"reconciliation": broken_result()}, HaltTrigger.RECONCILIATION_MISMATCH),
        (
            {"manual": ManualHaltRequest(requested_by="risk-desk", reason="stand down")},
            HaltTrigger.MANUAL,
        ),
        # The fifth: not one of the directive's four, and the one a kill switch
        # with an exhaustive trigger list would fail open on.
        ({"freshness": None}, HaltTrigger.UNKNOWN_CONDITION),
    ],
    ids=["drawdown", "stale_data", "reconciliation", "manual", "unknown_condition"],
)
async def test_each_trigger_halts_within_one_cycle_against_the_real_log(
    overrides: dict[str, Any], trigger: HaltTrigger
) -> None:
    async with ingest_writer_session() as session:
        outcome = await run_cycle(session, observation(**overrides))
        await session.commit()
    assert outcome.halted
    assert [reason.trigger for reason in outcome.decision.reasons] == [trigger]
    async with ingest_writer_session() as session:
        halts = await open_halts(session)
        assert len(halts) == len(outcome.engaged_halt_ids)
        assert [halt.trigger for halt in halts] == [trigger]
        # "Within one cycle", checkable from the log alone: the halt the cycle
        # wrote carries that cycle's own id, so no cycle elapsed in between.
        assert {halt.cycle_id for halt in halts} == {CYCLE_ID}


async def test_a_halt_survives_disposing_the_engine_and_starting_over() -> None:
    # A genuine restart: the pool is torn down and everything in this process is
    # discarded. If the halt were a flag it would be gone.
    async with ingest_writer_session() as session:
        await engage_halt(
            session,
            cycle_id=CYCLE_ID,
            reason=HaltReason(
                trigger=HaltTrigger.DRAWDOWN_BREACH, detail="breached", evidence={"x": "1"}
            ),
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        await session.commit()
    await dispose_database()
    async with ingest_writer_session() as session:
        assert len(await open_halts(session)) == 1
        with pytest.raises(SystemHaltedError):
            await assert_not_halted(session)


# ---------------------------------------------------------------------------
# The clearance guard, against the real trigger.
# ---------------------------------------------------------------------------


async def test_clearing_a_halt_permits_trading_again() -> None:
    async with ingest_writer_session() as session:
        halt_id = await engage_halt(
            session,
            cycle_id=CYCLE_ID,
            reason=HaltReason(trigger=HaltTrigger.MANUAL, detail="pause", evidence={}),
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        await session.commit()
    async with ingest_writer_session() as session:
        await clear_halt(
            session,
            halt_id=halt_id,
            cleared_by="operator-on-call",
            clearance_reason="investigated and resolved",
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        await session.commit()
    async with ingest_writer_session() as session:
        await assert_not_halted(session)


async def test_a_second_clearance_is_refused_by_the_database() -> None:
    async with ingest_writer_session() as session:
        halt_id = await engage_halt(
            session,
            cycle_id=CYCLE_ID,
            reason=HaltReason(trigger=HaltTrigger.MANUAL, detail="pause", evidence={}),
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        await session.commit()

    async def clear_once() -> int:
        async with ingest_writer_session() as session:
            clearance_id = await clear_halt(
                session,
                halt_id=halt_id,
                cleared_by="operator-on-call",
                clearance_reason="investigated",
                occurred_at=OBSERVED_AT,
                stamp=make_stamp(),
            )
            await session.commit()
            return clearance_id

    await clear_once()
    with pytest.raises(HaltAlreadyClearedError):
        await clear_once()


async def test_concurrent_clearances_leave_exactly_one_winner() -> None:
    async with ingest_writer_session() as session:
        halt_id = await engage_halt(
            session,
            cycle_id=CYCLE_ID,
            reason=HaltReason(trigger=HaltTrigger.MANUAL, detail="pause", evidence={}),
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        await session.commit()
    gate = asyncio.Barrier(CONCURRENT_CLEARERS)

    async def racer() -> int:
        async with ingest_writer_session() as session:
            await gate.wait()
            clearance_id = await clear_halt(
                session,
                halt_id=halt_id,
                cleared_by="operator-on-call",
                clearance_reason="investigated",
                occurred_at=OBSERVED_AT,
                stamp=make_stamp(),
            )
            await session.commit()
            return clearance_id

    outcomes = await asyncio.gather(
        *(racer() for _ in range(CONCURRENT_CLEARERS)), return_exceptions=True
    )
    winners = [item for item in outcomes if isinstance(item, int)]
    losers = [item for item in outcomes if isinstance(item, HaltAlreadyClearedError)]
    assert len(winners) == 1
    assert len(losers) == CONCURRENT_CLEARERS - 1
    async with ingest_writer_session() as session:
        await assert_not_halted(session)


async def test_clearing_a_row_that_is_not_an_engagement_is_refused() -> None:
    async with ingest_writer_session() as session:
        halt_id = await engage_halt(
            session,
            cycle_id=CYCLE_ID,
            reason=HaltReason(trigger=HaltTrigger.MANUAL, detail="pause", evidence={}),
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        clearance_id = await clear_halt(
            session,
            halt_id=halt_id,
            cleared_by="operator-on-call",
            clearance_reason="investigated",
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
        await session.commit()
    async with ingest_writer_session() as session:
        with pytest.raises(HaltClearanceError, match="not an open engagement"):
            await clear_halt(
                session,
                halt_id=clearance_id,
                cleared_by="operator-on-call",
                clearance_reason="clearing a clearance",
                occurred_at=OBSERVED_AT,
                stamp=make_stamp(),
            )


async def test_clearing_a_halt_that_does_not_exist_is_refused() -> None:
    async with ingest_writer_session() as session:
        with pytest.raises(HaltClearanceError):
            await clear_halt(
                session,
                halt_id=987654,
                cleared_by="operator-on-call",
                clearance_reason="nothing to clear",
                occurred_at=OBSERVED_AT,
                stamp=make_stamp(),
            )


async def test_the_guard_never_refuses_an_engagement_however_many_arrive() -> None:
    # The property the whole kill switch depends on, proved against the real
    # trigger: nothing can stop a halt from being recorded.
    async with ingest_writer_session() as session:
        for _ in range(10):
            await engage_halt(
                session,
                cycle_id=CYCLE_ID,
                reason=HaltReason(
                    trigger=HaltTrigger.UNKNOWN_CONDITION, detail="again", evidence={}
                ),
                occurred_at=OBSERVED_AT,
                stamp=make_stamp(),
            )
        await session.commit()
    async with ingest_writer_session() as session:
        assert len(await open_halts(session)) == 10


async def test_concurrent_engagements_from_separate_transactions_all_land() -> None:
    gate = asyncio.Barrier(CONCURRENT_CLEARERS)

    async def engage() -> int:
        async with ingest_writer_session() as session:
            await gate.wait()
            halt_id = await engage_halt(
                session,
                cycle_id=CYCLE_ID,
                reason=HaltReason(trigger=HaltTrigger.MANUAL, detail="race", evidence={}),
                occurred_at=OBSERVED_AT,
                stamp=make_stamp(),
            )
            await session.commit()
            return halt_id

    ids = await asyncio.gather(*(engage() for _ in range(CONCURRENT_CLEARERS)))
    assert len(set(ids)) == CONCURRENT_CLEARERS
    async with ingest_writer_session() as session:
        assert len(await open_halts(session)) == CONCURRENT_CLEARERS


# ---------------------------------------------------------------------------
# The CHECK constraints, against writers that skip Python entirely.
#
# **A probe row must violate exactly the constraint it names.** CHECK evaluation
# order is unspecified in Postgres, so a row that breaks two constraints reports
# whichever the server reaches first — and the test then either fails for the
# wrong reason or passes by luck. P11.2 lost four tests to this and P11.3 lost
# one; a third, ``{"event": "paused"}``, was passing on luck alone (it also broke
# ``trigger_iff_engaged``, because an unknown event makes ``event = 'engaged'``
# false while ``halt_trigger`` is still set).
#
# So each probe declares ``also_mentions``: the other constraints whose SQL names
# a column this probe overrides, and which its values are chosen to keep
# satisfied. ``test_control_migration.py`` checks that declaration against the
# ORM's constraint texts without needing a database, so an override that quietly
# starts touching a second constraint fails in the unit suite rather than in CI.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstraintProbe:
    """One raw-insert row aimed at exactly one CHECK constraint.

    Attributes:
        constraint: the unprefixed name the refusal must quote.
        overrides: the deviations from the valid base row.
        also_mentions: other constraints on the table whose SQL names one of the
            overridden columns. Declaring them is what forces the isolation
            analysis to be done rather than assumed; the probe's values must
            leave every one of them satisfied.
    """

    constraint: str
    overrides: dict[str, object]
    also_mentions: frozenset[str] = frozenset()


RECONCILIATION_PROBES: tuple[ConstraintProbe, ...] = (
    ConstraintProbe("reported_origin_is_not_live", {"reported_origin": "live_broker"}),
    ConstraintProbe("reported_origin_is_not_live", {"reported_origin": "internal_ledger"}),
    ConstraintProbe("internal_origin_is_ledger", {"internal_origin": "paper_broker"}),
    ConstraintProbe("tolerance_within_ceiling", {"cash_tolerance_usd": Decimal("1.00")}),
    ConstraintProbe("tolerance_within_ceiling", {"cash_tolerance_usd": Decimal("-0.01")}),
    ConstraintProbe("matched_iff_no_breaks", {"matched": False}),
    # finding_count is raised alongside break_count so counts_consistent still
    # holds. Without it the row broke both, and CI reported counts_consistent.
    ConstraintProbe(
        "matched_iff_no_breaks",
        {"break_count": 1, "finding_count": 1, "matched": True},
        frozenset({"counts_consistent"}),
    ),
    ConstraintProbe(
        "counts_consistent",
        {"break_count": 1, "finding_count": 0, "matched": False},
        frozenset({"matched_iff_no_breaks"}),
    ),
    ConstraintProbe("cycle_id_present", {"cycle_id": ""}),
    ConstraintProbe("result_digest_is_sha256", {"result_digest": "not-a-digest"}),
    ConstraintProbe("git_commit_is_sha", {"git_commit": "abc"}),
    ConstraintProbe("seed_non_negative", {"seed": -1}),
)

HALT_PROBES: tuple[ConstraintProbe, ...] = (
    # halt_trigger is cleared too: an unknown event makes (event = 'engaged')
    # false, so a row keeping halt_trigger would also break trigger_iff_engaged.
    ConstraintProbe(
        "event_is_known",
        {"event": "paused", "halt_trigger": None},
        frozenset({"trigger_iff_engaged", "trigger_is_known", "clearance_fields_iff_cleared"}),
    ),
    ConstraintProbe(
        "trigger_is_known",
        {"halt_trigger": "because_i_said_so"},
        frozenset({"trigger_iff_engaged"}),
    ),
    ConstraintProbe("trigger_iff_engaged", {"halt_trigger": None}, frozenset({"trigger_is_known"})),
    ConstraintProbe("cycle_id_present", {"cycle_id": ""}),
    ConstraintProbe("detail_present", {"detail": ""}),
    ConstraintProbe(
        "clearance_fields_iff_cleared",
        {"cleared_by": "someone"},
        frozenset({"cleared_by_present"}),
    ),
    ConstraintProbe(
        "clearance_fields_iff_cleared",
        {"clearance_reason": "because I felt like it"},
        frozenset({"clearance_reason_present"}),
    ),
    ConstraintProbe("git_commit_is_sha", {"git_commit": "abc"}),
    ConstraintProbe("seed_non_negative", {"seed": -1}),
)


@pytest.mark.parametrize("probe", RECONCILIATION_PROBES, ids=lambda probe: probe.constraint)
async def test_a_raw_insert_cannot_bypass_the_reconciliation_constraints(
    probe: ConstraintProbe,
) -> None:
    with pytest.raises(IntegrityError) as caught:
        await raw_insert("execution_reconciliation", reconciliation_row(**probe.overrides))
    # The constraint *name*, not merely "something refused it". Asserting the name
    # is what caught the un-isolated probe this list used to contain.
    assert probe.constraint in str(caught.value)


@pytest.mark.parametrize("probe", HALT_PROBES, ids=lambda probe: probe.constraint)
async def test_a_raw_insert_cannot_bypass_the_halt_constraints(probe: ConstraintProbe) -> None:
    with pytest.raises(IntegrityError) as caught:
        await raw_insert("execution_halt", halt_row(**probe.overrides))
    assert probe.constraint in str(caught.value)


async def test_an_engagement_carrying_a_clearance_id_is_refused() -> None:
    """The stray that made a halt permanently un-clearable.

    ``uq_execution_halt_clears_halt_id`` is on the column unconditionally, so an
    engagement carrying a ``clears_halt_id`` consumed the unique slot for that
    halt: the genuine clearance was then refused with ``23505`` and reported as
    ``HaltAlreadyClearedError`` while ``open_halts`` — which counts clearances
    only where ``event = 'cleared'`` — kept reporting the halt open.

    Given its own test rather than a parametrised row because it needs a real
    foreign-key target, and a dangling id would let the FK fire instead of the
    CHECK.
    """
    target = await insert_halt_row()
    with pytest.raises(IntegrityError) as caught:
        await raw_insert("execution_halt", halt_row(clears_halt_id=target))
    assert "clearance_fields_iff_cleared" in str(caught.value)


async def test_a_well_formed_clearance_inserted_raw_is_accepted() -> None:
    # The positive control the three clearance refusals need. A constraint that
    # rejects every shape proves nothing about the shapes it means to reject, and
    # the per-column form is a widening — so what it still admits has to be shown.
    target = await insert_halt_row()
    await raw_insert(
        "execution_halt",
        halt_row(
            event="cleared",
            halt_trigger=None,
            clears_halt_id=target,
            cleared_by="operator-on-call",
            clearance_reason="investigated and resolved",
        ),
    )


async def test_the_tolerance_ceiling_admits_its_own_boundary() -> None:
    # The refusals above are only meaningful if the permitted value is permitted:
    # a constraint that rejects everything proves nothing about what it rejects.
    await raw_insert(
        "execution_reconciliation",
        reconciliation_row(cash_tolerance_usd=MAX_CASH_TOLERANCE_USD),
    )
    await raw_insert(
        "execution_reconciliation", reconciliation_row(cash_tolerance_usd=Decimal("0"))
    )


@pytest.mark.parametrize("trigger", [member.value for member in HaltTrigger])
async def test_every_declared_trigger_is_accepted_by_the_schema(trigger: str) -> None:
    # A trigger Python can engage and the schema refuses is a kill switch that
    # cannot fire, so every member is inserted rather than only the vocabulary
    # being compared as text.
    await raw_insert("execution_halt", halt_row(halt_trigger=trigger))


@pytest.mark.parametrize("origin", [member.value for member in SnapshotOrigin])
async def test_the_reported_origin_check_admits_exactly_the_non_internal_origins(
    origin: str,
) -> None:
    row = reconciliation_row(reported_origin=origin)
    if origin == SnapshotOrigin.INTERNAL_LEDGER.value:
        with pytest.raises(IntegrityError):
            await raw_insert("execution_reconciliation", row)
        return
    await raw_insert("execution_reconciliation", row)


# ---------------------------------------------------------------------------
# Append-only.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", ["execution_reconciliation", "execution_halt"])
@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
async def test_neither_control_table_can_be_edited_or_erased(table: str, operation: str) -> None:
    if table == "execution_halt":
        await raw_insert(table, halt_row())
        statement = (
            "UPDATE execution_halt SET detail = 'rewritten'"
            if operation == "UPDATE"
            else "DELETE FROM execution_halt"
        )
    else:
        await raw_insert(table, reconciliation_row())
        statement = (
            "UPDATE execution_reconciliation SET matched = false"
            if operation == "UPDATE"
            else "DELETE FROM execution_reconciliation"
        )
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError) as caught:
            await session.execute(sa.text(statement))
        assert "append-only" in str(caught.value)


async def test_a_recorded_verdict_cannot_be_relabelled_as_matched() -> None:
    # The specific edit the append-only guard exists to prevent: a break quietly
    # becoming a pass after somebody has looked at it.
    async with ingest_writer_session() as session:
        await record_reconciliation(session, broken_result())
        await session.commit()
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError):
            await session.execute(
                sa.text("UPDATE execution_reconciliation SET matched = true, break_count = 0")
            )


# ---------------------------------------------------------------------------
# The comparison itself, over values that round-trip through Postgres.
# ---------------------------------------------------------------------------


async def test_the_snapshot_payload_round_trips_through_jsonb_unchanged() -> None:
    # The digest is taken over the payload, and the payload is stored as JSONB:
    # if the database normalised it, the stored digest would stop verifying.
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(
            positions={11: 100, 2: -30, 7: 0}, cash_usd=Decimal("12345.678900")
        ),
        reported=reported_snapshot(positions={11: 100, 2: -30}, cash_usd=Decimal("12345.678901")),
        stamp=make_stamp(),
    )
    async with ingest_writer_session() as session:
        reconciliation_id = await record_reconciliation(session, result)
        await session.commit()
    async with ingest_writer_session() as session:
        stored = await load_reconciliation(session, reconciliation_id)
    assert stored.internal_payload == result.internal.as_json()
    assert stored.reported_payload == result.reported.as_json()
    assert rerun(stored, stamp=make_stamp()).result_digest == result.result_digest


async def test_a_cash_difference_of_one_ulp_past_the_tolerance_survives_the_round_trip() -> None:
    # Numeric(18, 6) has to hold the boundary exactly, or the tolerance stored on
    # the row would not be the tolerance the verdict used.
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(cash_usd=Decimal("100000.000000")),
        reported=reported_snapshot(cash_usd=Decimal("99999.989999")),
        stamp=make_stamp(),
    )
    assert not result.matched
    async with ingest_writer_session() as session:
        reconciliation_id = await record_reconciliation(session, result)
        await session.commit()
    async with ingest_writer_session() as session:
        stored = await load_reconciliation(session, reconciliation_id)
    assert stored.cash_tolerance_usd == result.cash_tolerance_usd
    assert rerun(stored, stamp=make_stamp()).result_digest == result.result_digest


async def test_the_observation_instants_are_stored_as_aware_utc() -> None:
    async with ingest_writer_session() as session:
        await record_reconciliation(session, clean_result())
        await session.commit()
    async with ingest_writer_session() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT internal_observed_at, reported_observed_at "
                    "FROM execution_reconciliation"
                )
            )
        ).first()
    assert row is not None
    for value in row:
        assert isinstance(value, dt.datetime)
        assert value.tzinfo is not None
        assert value == OBSERVED_AT
