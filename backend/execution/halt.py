"""The halt log: a halt is a row, not a flag, and clearing one is an act (P11.5).

Why a halt is persisted rather than held in memory
--------------------------------------------------

A halt that evaporates on restart is not a halt. It is a pause with a failure
mode that fires exactly when it matters: the process that halted crashes, the
supervisor restarts it, and the restarted worker begins its next cycle believing
nothing is wrong — with the condition that caused the halt still true and now
unobserved, because whatever detected it ran in the previous process.

So there is no in-process halt flag anywhere in this package. There is an
append-only table (:class:`backend.db.models.ExecutionHalt`, migration 0016) and
a fold over it. :func:`open_halts` derives the current state by reading the log
every time it is asked; :func:`assert_not_halted` is the gate every release path
calls. A restart changes nothing because there was nothing in memory to lose.

Clearing is explicit, attributed, and cannot be implicit
--------------------------------------------------------

An ``engaged`` row is open until a ``cleared`` row names it by id. There is no
timeout, no automatic re-arm, and no "the condition went away" path — a halt
survives the condition that caused it, because the whole point is that a human
looks. A clearance carries who cleared it and why, both non-blank by CHECK, so
the halt history answers "who turned this back on" without anyone having to
remember.

Two refusals are enforced in the database rather than here: a halt may be cleared
at most once (``UNIQUE (clears_halt_id)``), and a clearance may only name a row
that is an engagement (the ``execution_halt_clearance_guard`` trigger).

**The guard never refuses an engagement.** That asymmetry is deliberate and is
the single most important line in the trigger: a constraint that could reject a
halt-engage row is a constraint that can stop the kill switch from firing.
Concurrent engagements, duplicate engagements, engagements for a cycle that
already halted — all are accepted and recorded. Redundant halt rows cost nothing;
a refused one costs everything.

Deciding on SQLSTATE, never on the exception class (D-034)
-----------------------------------------------------------

A ``BEFORE INSERT`` trigger fires ahead of every CHECK and every index, and a
plpgsql ``RAISE`` reaches SQLAlchemy as a generic ``DBAPIError`` rather than an
``IntegrityError``. So both refusals are classified on the five-character
SQLSTATE the driver carries:

- ``23505`` — the halt is already cleared. Raised natively by the unique index
  when the loser of a race reaches it, and raised by the trigger
  ``USING ERRCODE = 'unique_violation'`` when the loser's statement starts after
  the winner committed and the trigger sees a clearance the loser's own earlier
  read did not. One condition, two routes, one code — exactly the shape migration
  0014 established for the transition chain.
- ``P0001`` — the clearance names a row that is not an open engagement. Not
  retryable, so it keeps the default code and reaches the caller as a permanent
  refusal rather than an invitation to loop.

Anything else is re-raised unchanged: this module interprets exactly two
conditions and refuses to guess about the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError

from backend.db.models import ExecutionHalt as ExecutionHaltRow
from backend.execution.errors import (
    HaltAlreadyClearedError,
    HaltClearanceError,
    HaltStateUnavailableError,
    SystemHaltedError,
)

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.tracking.stamp import ReproducibilityStamp

__all__ = [
    "HALT_CLEARANCE_REFUSED_SQLSTATE",
    "UNIQUE_VIOLATION_SQLSTATE",
    "HaltEventKind",
    "HaltReason",
    "HaltTrigger",
    "OpenHalt",
    "assert_not_halted",
    "clear_halt",
    "engage_halt",
    "open_halts",
]

UNIQUE_VIOLATION_SQLSTATE: Final = "23505"
"""SQLSTATE ``unique_violation`` — "this halt has already been cleared".

Carried by both routes: the unique index on ``clears_halt_id``, and the clearance
guard's own ``USING ERRCODE = 'unique_violation'`` for the writer whose statement
starts after the winner committed and is refused by the trigger before the index
is consulted. Under ``READ COMMITTED`` that second route is the likelier one, and
it is the one an in-memory double modelling only the index will miss (D-034).
"""

HALT_CLEARANCE_REFUSED_SQLSTATE: Final = "P0001"
"""SQLSTATE of a plpgsql ``RAISE EXCEPTION`` with no code of its own.

The clearance guard uses it for the non-retryable refusal: the row this clearance
names does not exist, or is itself a clearance rather than an engagement. Kept
distinct from ``23505`` because retrying it would loop forever — the same split
migration 0014 makes between a taken sequence position and a gap.
"""


class HaltTrigger(StrEnum):
    """Why trading is halted. Four named conditions, and the one that is not.

    The four the directive names (§5 Phase 11): a drawdown breach, stale data, a
    reconciliation mismatch, and an operator pulling the switch.

    ``UNKNOWN_CONDITION`` is the fifth, and it exists because a kill switch whose
    trigger list is exhaustive fails open on everything not on the list. A
    measurement that is missing, non-finite, negative where it cannot be, or
    produced by a probe that raised is not evidence of safety — it is the absence
    of evidence, and this system halts on it. See
    :mod:`backend.execution.killswitch`, where every fail-closed path lands here.
    """

    DRAWDOWN_BREACH = "drawdown_breach"
    STALE_DATA = "stale_data"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"
    MANUAL = "manual"
    UNKNOWN_CONDITION = "unknown_condition"


class HaltEventKind(StrEnum):
    """The two things that can happen to a halt.

    ``ENGAGED`` opens one. ``CLEARED`` closes exactly one open engagement, naming
    it by id. There is no third event: a halt is not amended, re-scoped or
    escalated in place, because the table is append-only and a mutable halt is a
    halt whose history cannot be read back.
    """

    ENGAGED = "engaged"
    CLEARED = "cleared"


@dataclass(frozen=True, slots=True)
class HaltReason:
    """One cause of a halt, with the evidence that justified it.

    Evidence is structured rather than prose because the operator's first question
    is always "what was the number" — the drawdown that breached and the limit it
    breached, the age of the data and the age allowed, the digest of the
    reconciliation that failed. A halt whose evidence is a sentence cannot be
    audited without re-running the thing that produced it.

    Attributes:
        trigger: which condition fired.
        detail: prose naming the condition, non-blank.
        evidence: JSON-serialisable measurements behind the decision.
    """

    trigger: HaltTrigger
    detail: str
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        """Normalise a blank detail rather than refusing it.

        Deliberately does **not** raise. Every construction path for this object
        ends in a halt being recorded, so a validation error here would be a
        refusal to record a halt — the one failure this whole module exists to
        prevent. A missing detail is replaced with a marker that says so.
        """
        if not self.detail.strip():
            object.__setattr__(
                self,
                "detail",
                f"{self.trigger.value}: no detail supplied by the caller",
            )
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True, slots=True)
class OpenHalt:
    """One engagement that has not been cleared.

    Attributes:
        halt_id: the engagement row's database key — the id a clearance must name.
        trigger: which condition fired.
        cycle_id: the execution cycle the halt was engaged in. This is what makes
            "within one cycle" auditable after the fact rather than only at the
            moment it happened.
        detail: the prose recorded with the engagement.
        occurred_at: when the condition was observed, timezone-aware.
    """

    halt_id: int
    trigger: HaltTrigger
    cycle_id: str
    detail: str
    occurred_at: dt.datetime


def _sqlstate(exc: DBAPIError) -> str | None:
    """Return the five-character SQLSTATE a driver exception carries, if any.

    SQLAlchemy exposes no portable accessor, so the driver's own attribute is
    read: asyncpg names it ``sqlstate``, psycopg names it ``pgcode``.

    Args:
        exc: the wrapped database error.

    Returns:
        The SQLSTATE, or ``None`` when the driver does not expose one.
    """
    original: object = exc.orig
    for attribute in ("sqlstate", "pgcode"):
        code: object = getattr(original, attribute, None)
        if isinstance(code, str) and len(code) == 5:
            return code
    return None


async def engage_halt(
    session: AsyncSession,
    *,
    cycle_id: str,
    reason: HaltReason,
    occurred_at: dt.datetime,
    stamp: ReproducibilityStamp,
) -> int:
    """Record one halt engagement. Nothing refuses this row.

    No savepoint, no pre-read, no uniqueness check: the write path for engaging a
    halt is as short as it can be made, because every branch in it is a branch
    that could fail and leave trading un-halted. Duplicate and concurrent
    engagements are accepted; a redundant halt row is free and a missing one is
    not.

    Does **not** commit — the caller owns the transaction, matching
    :mod:`backend.execution.store`. The caller must commit for the halt to survive
    the restart it exists for.

    Args:
        session: any writable ``AsyncSession``.
        cycle_id: the execution cycle in which the condition was observed.
        reason: the trigger, its prose and its evidence.
        occurred_at: when the condition was observed, timezone-aware.
        stamp: the I2 stamp of the run that observed it, so a halt traces to the
            commit, config, data version and seed that produced the decision.

    Returns:
        The new ``halt_id`` — the id a later clearance must name.
    """
    statement = (
        sa.insert(ExecutionHaltRow)
        .values(
            event=HaltEventKind.ENGAGED.value,
            halt_trigger=reason.trigger.value,
            cycle_id=cycle_id,
            detail=reason.detail,
            evidence=dict(reason.evidence),
            clears_halt_id=None,
            cleared_by=None,
            clearance_reason=None,
            occurred_at=occurred_at,
            git_commit=stamp.git_commit,
            git_dirty=stamp.git_dirty,
            data_version=stamp.data_version,
            config_hash=stamp.config_hash,
            seed=stamp.seed,
        )
        .returning(ExecutionHaltRow.halt_id)
    )
    inserted = await session.execute(statement)
    return int(inserted.scalar_one())


async def open_halts(session: AsyncSession) -> tuple[OpenHalt, ...]:
    """Return every engagement that has not been cleared, oldest first.

    Derived from the log on every call. There is no cached answer and no
    in-process flag, which is precisely why a halt survives a restart: the state
    lives in rows, and a new process reads the same rows.

    Args:
        session: any readable ``AsyncSession``.

    Returns:
        The open halts in ascending ``halt_id`` order — a total order, so two
        readers of the same log agree on the sequence.

    Raises:
        HaltStateUnavailableError: if the log cannot be read, or holds a trigger
            that names no known condition. Both are refusals to guess: see
            :func:`assert_not_halted` for why an unreadable halt log is treated as
            a halt rather than as an absence of one.
    """
    clearances = sa.select(ExecutionHaltRow.clears_halt_id).where(
        ExecutionHaltRow.event == HaltEventKind.CLEARED.value,
        ExecutionHaltRow.clears_halt_id.is_not(None),
    )
    statement = (
        sa.select(
            ExecutionHaltRow.halt_id,
            ExecutionHaltRow.halt_trigger,
            ExecutionHaltRow.cycle_id,
            ExecutionHaltRow.detail,
            ExecutionHaltRow.occurred_at,
        )
        .where(
            ExecutionHaltRow.event == HaltEventKind.ENGAGED.value,
            ExecutionHaltRow.halt_id.not_in(clearances),
        )
        .order_by(ExecutionHaltRow.halt_id)
    )
    try:
        rows = (await session.execute(statement)).all()
    # Deliberately broad. Anything at all that stops this read — a dropped
    # connection, a permission change, a driver bug — must produce a refusal to
    # trade rather than an exception the caller might treat as "no halts found".
    except Exception as exc:
        msg = (
            f"the halt log could not be read ({type(exc).__name__}: {exc}). A system that "
            f"cannot determine whether it is halted is halted: reporting 'not halted' here "
            f"would let a database outage do what no operator is allowed to do"
        )
        raise HaltStateUnavailableError(msg) from exc
    halts: list[OpenHalt] = []
    for row in rows:
        trigger_value = str(row[1])
        try:
            trigger = HaltTrigger(trigger_value)
        except ValueError as exc:
            msg = (
                f"halt_id={row[0]} carries halt_trigger={trigger_value!r}, which names no known "
                f"condition. Refusing to interpret it: a halt whose cause cannot be read is "
                f"still a halt, and guessing its cause is how one gets cleared by mistake"
            )
            raise HaltStateUnavailableError(msg) from exc
        halts.append(
            OpenHalt(
                halt_id=int(row[0]),
                trigger=trigger,
                cycle_id=str(row[2]),
                detail=str(row[3]),
                occurred_at=row[4],
            )
        )
    return tuple(halts)


async def assert_not_halted(session: AsyncSession) -> None:
    """Raise unless trading is permitted. The gate every release path calls.

    Fail-closed in both directions that matter:

    - an open halt raises :class:`~backend.execution.errors.SystemHaltedError`;
    - a halt log that cannot be *read* raises
      :class:`~backend.execution.errors.HaltStateUnavailableError`, because "I do
      not know whether I am halted" and "I am not halted" are different facts and
      only one of them permits trading.

    There is no return value to ignore and no boolean to invert. A caller that
    forgets to call this does not silently trade through a halt — it fails the
    test that asserts the release path calls it.

    Args:
        session: any readable ``AsyncSession``.

    Raises:
        SystemHaltedError: if any engagement is open.
        HaltStateUnavailableError: if the halt log cannot be read.
    """
    halts = await open_halts(session)
    if halts:
        raise SystemHaltedError(
            halt_ids=tuple(halt.halt_id for halt in halts),
            triggers=tuple(halt.trigger.value for halt in halts),
            detail="; ".join(f"halt_id={halt.halt_id}: {halt.detail}" for halt in halts),
        )


async def clear_halt(
    session: AsyncSession,
    *,
    halt_id: int,
    cleared_by: str,
    clearance_reason: str,
    occurred_at: dt.datetime,
    stamp: ReproducibilityStamp,
) -> int:
    """Clear one open halt, attributing the decision to a person and a reason.

    Deliberately requires the ``halt_id``: there is no "clear all", because
    clearing a halt you have not read is indistinguishable from clearing one you
    have, and the whole value of a persistent halt is that somebody had to look at
    it. Two open halts take two clearances.

    Does **not** commit — the caller owns the transaction.

    Args:
        session: any writable ``AsyncSession``.
        halt_id: the engagement being cleared.
        cleared_by: who cleared it, non-blank.
        clearance_reason: why, non-blank.
        occurred_at: when, timezone-aware.
        stamp: the I2 stamp of the run recording the clearance.

    Returns:
        The new row's ``halt_id`` (the clearance's own key, not the cleared one).

    Raises:
        HaltClearanceError: if ``cleared_by`` or ``clearance_reason`` is blank, or
            if ``halt_id`` names a row that is not an open engagement. An
            unattributed clearance is refused here even though a blank *halt* is
            never refused — the asymmetry is the point: recording a halt must
            never fail, and removing one must never be casual.
        HaltAlreadyClearedError: if the halt is already cleared, by either route
            (the unique index, or the guard seeing a committed clearance the
            caller's own read did not).
    """
    if not cleared_by.strip():
        msg = (
            "cleared_by is blank; a halt is cleared by a person and the halt history has to "
            "say which one"
        )
        raise HaltClearanceError(msg)
    if not clearance_reason.strip():
        msg = (
            "clearance_reason is blank; 'the alarm was inconvenient' is a reason and so is "
            "'the break was a stale statement', and the history is worthless if it cannot "
            "tell them apart"
        )
        raise HaltClearanceError(msg)
    statement = (
        sa.insert(ExecutionHaltRow)
        .values(
            event=HaltEventKind.CLEARED.value,
            halt_trigger=None,
            cycle_id=f"clearance-of-{halt_id}",
            detail=f"cleared halt_id={halt_id}: {clearance_reason}",
            evidence={"clears_halt_id": halt_id, "cleared_by": cleared_by},
            clears_halt_id=halt_id,
            cleared_by=cleared_by,
            clearance_reason=clearance_reason,
            occurred_at=occurred_at,
            git_commit=stamp.git_commit,
            git_dirty=stamp.git_dirty,
            data_version=stamp.data_version,
            config_hash=stamp.config_hash,
            seed=stamp.seed,
        )
        .returning(ExecutionHaltRow.halt_id)
    )
    try:
        async with session.begin_nested():
            inserted = await session.execute(statement)
            return int(inserted.scalar_one())
    except DBAPIError as exc:
        sqlstate = _sqlstate(exc)
        if sqlstate == UNIQUE_VIOLATION_SQLSTATE:
            raise HaltAlreadyClearedError(halt_id=halt_id) from exc
        if sqlstate == HALT_CLEARANCE_REFUSED_SQLSTATE:
            msg = (
                f"halt_id={halt_id} is not an open engagement, so there is nothing to clear: {exc}"
            )
            raise HaltClearanceError(msg) from exc
        # A foreign-key failure, a CHECK violation, a driver that exposed no
        # code: all real refusals, none of them one of the two conditions this
        # function interprets. Re-raised rather than guessed at.
        raise
