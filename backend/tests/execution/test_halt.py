"""P11.5: the halt log — a halt is a row, nothing refuses one, and clearing is an act.

Three claims are load-bearing here and each has its own group of tests:

1. **Nothing can refuse an engagement.** Duplicates, concurrent writers, a cycle
   that already halted — all accepted. A guard that could reject a halt-engage row
   is a guard that can stop the kill switch from firing.
2. **A halt survives a restart**, because it is a row and the current state is a
   fold over rows computed on every read. The restart is modelled by discarding
   the session and building a new one over the same storage, which is what a
   restarted process does; and a structural test asserts there is no module-level
   mutable state that *could* have cached it.
3. **Clearing is explicit, attributed, and happens at most once** — refused in the
   database by two routes carrying one SQLSTATE, and classified on the code rather
   than on the exception class (D-034).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy.exc import DBAPIError

from backend.execution import halt as halt_module
from backend.execution import killswitch as killswitch_module
from backend.execution.errors import (
    HaltAlreadyClearedError,
    HaltClearanceError,
    HaltStateUnavailableError,
    SystemHaltedError,
)
from backend.execution.halt import (
    HaltEventKind,
    HaltReason,
    HaltTrigger,
    assert_not_halted,
    clear_halt,
    engage_halt,
    open_halts,
)
from backend.tests.execution.control_fixtures import (
    CYCLE_ID,
    OBSERVED_AT,
    ControlRows,
    ControlSessionDouble,
    DriverError,
    as_session,
    restart,
)
from backend.tests.execution.fixtures import make_stamp

if TYPE_CHECKING:
    from collections.abc import Sequence

CLEARED_BY = "operator-on-call"
CLEARANCE_REASON = "statement was stale; re-pulled and reconciled clean"


def a_reason(trigger: HaltTrigger = HaltTrigger.MANUAL) -> HaltReason:
    """Build a halt reason with structured evidence."""
    return HaltReason(
        trigger=trigger,
        detail=f"{trigger.value} fired in a test",
        evidence={"measured": "1.0", "limit": "0.5"},
    )


async def engage(session: Any, *, trigger: HaltTrigger = HaltTrigger.MANUAL) -> int:  # noqa: ANN401
    """Engage one halt through the store."""
    return await engage_halt(
        session,
        cycle_id=CYCLE_ID,
        reason=a_reason(trigger),
        occurred_at=OBSERVED_AT,
        stamp=make_stamp(),
    )


async def clear(session: Any, halt_id: int) -> int:  # noqa: ANN401 - the double
    """Clear one halt through the store."""
    return await clear_halt(
        session,
        halt_id=halt_id,
        cleared_by=CLEARED_BY,
        clearance_reason=CLEARANCE_REASON,
        occurred_at=OBSERVED_AT,
        stamp=make_stamp(),
    )


# ---------------------------------------------------------------------------
# Engaging: nothing refuses a halt.
# ---------------------------------------------------------------------------


async def test_engaging_a_halt_writes_the_trigger_cycle_evidence_and_stamp() -> None:
    rows = ControlRows()
    halt_id = await engage(as_session(ControlSessionDouble(rows)))
    row = rows.halt(halt_id)
    assert row["event"] == HaltEventKind.ENGAGED.value
    assert row["halt_trigger"] == HaltTrigger.MANUAL.value
    assert row["cycle_id"] == CYCLE_ID
    assert row["evidence"] == {"measured": "1.0", "limit": "0.5"}
    assert row["clears_halt_id"] is None
    # I2: the halt names the commit and config that observed the condition.
    stamp = make_stamp()
    assert row["git_commit"] == stamp.git_commit
    assert row["config_hash"] == stamp.config_hash
    assert row["data_version"] == stamp.data_version
    assert row["seed"] == stamp.seed


async def test_a_duplicate_engagement_is_accepted_rather_than_refused() -> None:
    # A redundant halt row costs nothing; a refused one costs everything.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    first = await engage(session)
    second = await engage(session)
    assert first != second
    assert rows.open_halt_ids() == [first, second]


async def test_concurrent_engagements_all_land() -> None:
    rows = ControlRows()
    writers = 6
    rows.insert_barrier = asyncio.Barrier(writers)
    sessions = [as_session(ControlSessionDouble(rows)) for _ in range(writers)]
    ids = await asyncio.gather(*(engage(session) for session in sessions))
    assert len(set(ids)) == writers
    assert rows.open_halt_ids() == sorted(ids)


async def test_a_blank_detail_is_normalised_rather_than_refusing_the_halt() -> None:
    reason = HaltReason(trigger=HaltTrigger.STALE_DATA, detail="   ", evidence={})
    assert reason.detail == "stale_data: no detail supplied by the caller"


async def test_halt_evidence_is_read_only_once_constructed() -> None:
    reason = a_reason()
    assert isinstance(reason.evidence, MappingProxyType)


# ---------------------------------------------------------------------------
# Reading: the fold, and failing closed.
# ---------------------------------------------------------------------------


async def test_open_halts_is_empty_before_anything_is_engaged() -> None:
    assert await open_halts(as_session(ControlSessionDouble(ControlRows()))) == ()


async def test_open_halts_reports_the_trigger_cycle_and_detail() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    halt_id = await engage(session, trigger=HaltTrigger.DRAWDOWN_BREACH)
    halts = await open_halts(session)
    assert len(halts) == 1
    assert halts[0].halt_id == halt_id
    assert halts[0].trigger is HaltTrigger.DRAWDOWN_BREACH
    assert halts[0].cycle_id == CYCLE_ID
    assert halts[0].occurred_at == OBSERVED_AT


async def test_assert_not_halted_raises_and_names_every_open_halt() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    first = await engage(session, trigger=HaltTrigger.DRAWDOWN_BREACH)
    second = await engage(session, trigger=HaltTrigger.STALE_DATA)
    with pytest.raises(SystemHaltedError) as caught:
        await assert_not_halted(session)
    assert caught.value.halt_ids == (first, second)
    assert caught.value.triggers == ("drawdown_breach", "stale_data")


async def test_an_unreadable_halt_log_is_treated_as_halted_not_as_empty() -> None:
    # The fail-closed path. Reporting "no halts" here would let a database outage
    # do what no operator is allowed to do.
    rows = ControlRows()
    rows.read_failure = RuntimeError("connection reset")
    session = as_session(ControlSessionDouble(rows))
    with pytest.raises(HaltStateUnavailableError, match="cannot determine whether it is halted"):
        await open_halts(session)
    with pytest.raises(HaltStateUnavailableError):
        await assert_not_halted(session)


async def test_a_halt_whose_trigger_is_unreadable_is_refused_not_guessed() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    await engage(session)
    rows.halts[0]["halt_trigger"] = "something_nobody_declared"
    with pytest.raises(HaltStateUnavailableError, match="names no known condition"):
        await open_halts(session)


class _StatementCapturingSession(ControlSessionDouble):
    """A session that keeps the last statement it was handed, compiled to SQL."""

    def __init__(self, rows: ControlRows) -> None:
        super().__init__(rows)
        self.last_sql = ""

    async def execute(self, statement: Any) -> Any:  # noqa: ANN401 - SQLAlchemy statements
        """Record the compiled SQL, then dispatch as usual."""
        self.last_sql = str(statement)
        return await super().execute(statement)


async def test_the_open_halt_query_excludes_cleared_ids_and_survives_null_clearances() -> None:
    # The double folds the log itself, so it cannot catch a wrong WHERE clause —
    # mutation testing confirmed that dropping `not_in(clearances)` left every
    # unit test green. The query text is asserted instead, and the behaviour is
    # proved for real in backend/tests/integration/test_reconciliation.py.
    #
    # The `IS NOT NULL` inside the subquery is the load-bearing half. Every
    # *engagement* carries clears_halt_id = NULL, and in SQL `x NOT IN (…, NULL)`
    # is never true — so without that filter the first clearance ever written
    # would make every open halt vanish from this query. A halt that silently
    # stops being reported is the worst failure this module has.
    session = _StatementCapturingSession(ControlRows())
    await open_halts(as_session(session))
    sql = " ".join(session.last_sql.split())
    assert "execution_halt.event = " in sql
    assert "(execution_halt.halt_id NOT IN (SELECT execution_halt.clears_halt_id" in sql
    assert "execution_halt.clears_halt_id IS NOT NULL" in sql
    assert "ORDER BY execution_halt.halt_id" in sql


# ---------------------------------------------------------------------------
# Surviving a restart.
# ---------------------------------------------------------------------------


async def test_a_halt_survives_a_simulated_restart() -> None:
    rows = ControlRows()
    halt_id = await engage(as_session(ControlSessionDouble(rows)))
    # Everything the first "process" held is discarded; only the rows remain.
    reborn = as_session(restart(rows))
    halts = await open_halts(reborn)
    assert [item.halt_id for item in halts] == [halt_id]
    with pytest.raises(SystemHaltedError):
        await assert_not_halted(reborn)


async def test_a_cleared_halt_stays_cleared_across_a_restart() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    halt_id = await engage(session)
    await clear(session, halt_id)
    await assert_not_halted(as_session(restart(rows)))


def test_no_module_level_mutable_state_could_cache_the_halt() -> None:
    # The structural half of "a halt is a row, not a flag": if either module held
    # a list, dict or set at module scope, a halt *could* be cached there and the
    # restart test above would be proving something about object lifetimes.
    for module in (halt_module, killswitch_module):
        for name, value in vars(module).items():
            if name.startswith("__"):
                continue
            assert not isinstance(value, list | dict | set), (module.__name__, name)


# ---------------------------------------------------------------------------
# Clearing: explicit, attributed, once.
# ---------------------------------------------------------------------------


async def test_clearing_a_halt_permits_trading_again() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    halt_id = await engage(session)
    clearance_id = await clear(session, halt_id)
    assert rows.halt(clearance_id)["event"] == HaltEventKind.CLEARED.value
    assert rows.halt(clearance_id)["cleared_by"] == CLEARED_BY
    assert rows.halt(clearance_id)["clearance_reason"] == CLEARANCE_REASON
    assert rows.halt(clearance_id)["halt_trigger"] is None
    await assert_not_halted(session)


async def test_two_open_halts_need_two_clearances() -> None:
    # There is no "clear all": clearing a halt you have not read is
    # indistinguishable from clearing one you have.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    first = await engage(session, trigger=HaltTrigger.DRAWDOWN_BREACH)
    second = await engage(session, trigger=HaltTrigger.MANUAL)
    await clear(session, first)
    with pytest.raises(SystemHaltedError) as caught:
        await assert_not_halted(session)
    assert caught.value.halt_ids == (second,)
    await clear(session, second)
    await assert_not_halted(session)


@pytest.mark.parametrize(
    ("cleared_by", "clearance_reason"),
    [("", CLEARANCE_REASON), ("   ", CLEARANCE_REASON), (CLEARED_BY, ""), (CLEARED_BY, "  ")],
)
async def test_an_unattributed_clearance_is_refused_and_writes_nothing(
    cleared_by: str, clearance_reason: str
) -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    halt_id = await engage(session)
    with pytest.raises(HaltClearanceError):
        await clear_halt(
            session,
            halt_id=halt_id,
            cleared_by=cleared_by,
            clearance_reason=clearance_reason,
            occurred_at=OBSERVED_AT,
            stamp=make_stamp(),
        )
    assert rows.open_halt_ids() == [halt_id]


async def test_clearing_a_halt_that_does_not_exist_is_refused() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    with pytest.raises(HaltClearanceError, match="not an open engagement"):
        await clear(session, 4242)


async def test_clearing_a_clearance_is_refused() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    halt_id = await engage(session)
    clearance_id = await clear(session, halt_id)
    with pytest.raises(HaltClearanceError, match="not an open engagement"):
        await clear(session, clearance_id)


@pytest.mark.parametrize("route", ["trigger", "index"])
async def test_a_second_clearance_is_refused_identically_by_either_route(route: str) -> None:
    # D-034: the BEFORE INSERT guard fires ahead of the unique index, so the two
    # refusals must be one condition to the caller. Both are exercised.
    rows = ControlRows()
    rows.clearance_refusal_route = route
    session = as_session(ControlSessionDouble(rows))
    halt_id = await engage(session)
    await clear(session, halt_id)
    with pytest.raises(HaltAlreadyClearedError) as caught:
        await clear(session, halt_id)
    assert caught.value.halt_id == halt_id


async def test_concurrent_clearances_leave_exactly_one_winner() -> None:
    rows = ControlRows()
    setup = as_session(ControlSessionDouble(rows))
    halt_id = await engage(setup)
    racers = 5
    rows.insert_barrier = asyncio.Barrier(racers)
    outcomes = await asyncio.gather(
        *(clear(as_session(ControlSessionDouble(rows)), halt_id) for _ in range(racers)),
        return_exceptions=True,
    )
    winners = [item for item in outcomes if isinstance(item, int)]
    losers = [item for item in outcomes if isinstance(item, HaltAlreadyClearedError)]
    assert len(winners) == 1
    assert len(losers) == racers - 1
    assert rows.open_halt_ids() == []


class _RefusingSession(ControlSessionDouble):
    """A session whose halt inserts fail with a chosen driver error."""

    def __init__(self, rows: ControlRows, error: Exception) -> None:
        super().__init__(rows)
        self._error = error

    async def execute(self, statement: Any) -> Any:  # noqa: ANN401, ARG002 - see docstring
        """Raise the configured error instead of inserting, whatever the statement."""
        raise self._error


@pytest.mark.parametrize(
    "original",
    [
        DriverError("23514", "new row violates check constraint"),
        DriverError("23503", "insert violates foreign key constraint"),
    ],
)
async def test_an_unrelated_refusal_is_re_raised_rather_than_reinterpreted(
    original: Exception,
) -> None:
    # The store interprets exactly two SQLSTATEs and refuses to guess about the
    # rest. Translating a CHECK violation into "already cleared" would report a
    # cleared halt where there is a malformed row.
    session: Any = _RefusingSession(
        ControlRows(), DBAPIError("INSERT INTO execution_halt", {}, original)
    )
    with pytest.raises(DBAPIError):
        await clear(session, 1)


async def test_a_driver_that_exposes_no_sqlstate_is_not_guessed_at() -> None:
    session: Any = _RefusingSession(
        ControlRows(), DBAPIError("INSERT INTO execution_halt", {}, Exception("no code"))
    )
    with pytest.raises(DBAPIError):
        await clear(session, 1)


# ---------------------------------------------------------------------------
# The trigger vocabulary.
# ---------------------------------------------------------------------------


def test_the_four_directive_triggers_exist_plus_the_fail_closed_one() -> None:
    values: Sequence[str] = [member.value for member in HaltTrigger]
    assert set(values) == {
        "drawdown_breach",
        "stale_data",
        "reconciliation_mismatch",
        "manual",
        "unknown_condition",
    }


def test_a_halt_event_has_exactly_two_kinds() -> None:
    assert {member.value for member in HaltEventKind} == {"engaged", "cleared"}


def test_the_clearance_sqlstates_are_distinct() -> None:
    # Retrying a "not an engagement" refusal would loop forever; retrying an
    # "already cleared" one is at least a no-op. They must not share a code.
    codes: set[str] = {
        halt_module.UNIQUE_VIOLATION_SQLSTATE,
        halt_module.HALT_CLEARANCE_REFUSED_SQLSTATE,
    }
    assert len(codes) == 2


async def test_occurred_at_is_stored_as_given_and_never_reinterpreted() -> None:
    rows = ControlRows()
    moment = dt.datetime(2026, 8, 3, 15, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    halt_id = await engage_halt(
        as_session(ControlSessionDouble(rows)),
        cycle_id=CYCLE_ID,
        reason=a_reason(),
        occurred_at=moment,
        stamp=make_stamp(),
    )
    assert rows.halt(halt_id)["occurred_at"] == moment
