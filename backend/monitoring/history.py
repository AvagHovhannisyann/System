"""Halt history: an append-only event log, and the seam the kill switch reads (P12.4).

Directive §6.10 asks for "halt history with cause". This module is that history,
and the shape it takes is the shape :mod:`backend.execution.store` uses for
orders, for the same reasons:

**Halts and resumes are events, not a status column.** ``monitoring_halt_event``
is append-only, and the current state is *derived* by asking which halt has no
resume pointing at it (:func:`active_halt`). A denormalised ``is_halted`` flag
would be the one thing an operator most needs to trust and the one thing most
able to drift from the history that explains it. A resume references the halt it
clears through a ``UNIQUE`` foreign key, so "resume the same halt twice" is
refused by the database rather than by a check somebody has to remember.

**Every halt names its cause.** The cause is
:class:`~backend.monitoring.expectation.HaltCause`, so "halted because the
comparison was unavailable" and "halted because live performance was below the
band" are different rows an operator can filter on — not one row whose meaning
lives in a message.

**A resume names a human.** The decision to halt is a machine's; the decision to
resume is a person's, and the history records which person, when, and why. A
resume with no actor or no reason is refused.

**A resume requires the halt's alert to be acknowledged.** This is what stops
P12.4 from being a log nobody reads: if the halt raised an alert and nobody has
signed for it, :func:`record_resume` refuses
(:class:`~backend.monitoring.errors.UnacknowledgedHaltError`). Acknowledgement
becomes the gate on restarting trading rather than a checkbox on a dashboard.

--------------------------------------------------------------------------
The seam with the execution-side kill switch
--------------------------------------------------------------------------

Two different things are called "the kill switch" and they belong to different
packages:

* **The monitoring-side decision to halt** — this package. It decides *whether*
  the live-versus-expected comparison, the drift monitors and the freshness
  checks permit trading, and it records that decision with its cause and its I2
  stamp.
* **The execution-side kill switch** (``backend/execution/killswitch.py``, owned
  elsewhere) — the thing that actually stops orders leaving, cancels working
  ones and refuses new intents.

The seam between them is deliberately one function and one exception, and it is
**pull, not push**:

.. code-block:: python

    from backend.monitoring.history import require_not_halted

    await require_not_halted(session)   # raises SystemHaltedError, or returns None

The contract, stated so the execution side can be written against it without
reading this module:

1. :func:`require_not_halted` raises
   :class:`~backend.monitoring.errors.SystemHaltedError` when a halt is in force
   and returns ``None`` otherwise. It is an exception rather than a boolean on
   purpose (D-032's reasoning): a boolean has a value a caller can forget to
   check, and the forgotten check leaves orders flowing during a halt.
2. **Any exception is a halt.** If the query itself fails — the database is
   unreachable, the table is missing, the transaction is poisoned — the caller
   must treat that as halted. Monitoring cannot prove trading is safe from a
   database it cannot read, and "the check failed" must never be softer than
   "the check said stop".
3. Monitoring never calls into ``backend.execution``. The dependency runs one
   way, so the execution package can be tested without a monitoring database and
   monitoring has no opinion about how orders are stopped.
4. The execution side may *also* halt for its own reasons — drawdown, stale
   market data, reconciliation mismatch, a manual operator trigger (§5 Phase 11).
   Those are its rows to write, not this module's. If it wants them in this
   history it calls :func:`record_halt` with its own cause; nothing here
   requires that.

Note that :func:`require_not_halted` deliberately does **not** cache. A cached
"not halted" is exactly as dangerous as a stale monitor, and a halt that takes
effect on the next cycle is the requirement in §5 Phase 11 ("halts within one
cycle").
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.db.models import MonitoringAlert as MonitoringAlertRow
from backend.db.models import MonitoringAlertAcknowledgement as AcknowledgementRow
from backend.db.models import MonitoringHaltEvent as HaltEventRow
from backend.monitoring.errors import (
    HaltHistoryError,
    SystemHaltedError,
    UnacknowledgedHaltError,
)
from backend.monitoring.expectation import HaltAction, HaltCause, HaltDecision
from backend.monitoring.psi import JsonValue
from backend.tracking.stamp import ReproducibilityStamp

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "AUTOMATIC_ACTOR",
    "HaltEvent",
    "HaltEventKind",
    "active_halt",
    "halt_history",
    "record_halt",
    "record_resume",
    "require_not_halted",
]

AUTOMATIC_ACTOR: Final = "auto:monitoring"
"""Actor recorded on a halt raised by the monitor rather than by a person.

Spelled out rather than left NULL: "who halted trading" is a question with an
answer even when the answer is a scheduled job, and a NULL there reads as
"nobody knows", which is a different and more alarming fact.
"""


class HaltEventKind(StrEnum):
    """What kind of event a history row records.

    Attributes:
        HALT: trading stopped.
        RESUME: an operator cleared a specific halt.
    """

    HALT = "halt"
    RESUME = "resume"


@dataclass(frozen=True, slots=True)
class HaltEvent:
    """One row of the halt history.

    Attributes:
        halt_event_id: database key, ascending with time.
        kind: halt or resume.
        cause: why trading stopped, on a halt row. ``None`` on a resume.
        detail: the decision's own words on a halt; the operator's reason on a
            resume.
        actor: who — :data:`AUTOMATIC_ACTOR` for a monitor-raised halt, a person
            for a resume.
        occurred_at: when (UTC).
        resolves_halt_event_id: on a resume, the halt it clears. ``None`` on a
            halt.
        alert_dedup_key: the alert this halt raised, when it raised one. The
            link that makes acknowledgement a precondition for resuming.
        decision: the :meth:`~backend.monitoring.expectation.HaltDecision.to_dict`
            payload that caused the halt, when there was one.
        stamp: the I2 stamp of the run that wrote the row.
    """

    halt_event_id: int
    kind: HaltEventKind
    cause: HaltCause | None
    detail: str
    actor: str
    occurred_at: dt.datetime
    resolves_halt_event_id: int | None
    alert_dedup_key: str | None
    decision: dict[str, JsonValue] | None
    stamp: ReproducibilityStamp

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the event as a JSON-safe mapping."""
        return {
            "halt_event_id": self.halt_event_id,
            "kind": str(self.kind),
            "cause": None if self.cause is None else str(self.cause),
            "detail": self.detail,
            "actor": self.actor,
            "occurred_at": self.occurred_at.isoformat(),
            "resolves_halt_event_id": self.resolves_halt_event_id,
            "alert_dedup_key": self.alert_dedup_key,
            "git_reference": self.stamp.git_reference,
            "data_version": self.stamp.data_version,
            "config_hash": self.stamp.config_hash,
            "seed": self.stamp.seed,
            "decision": None if self.decision is None else dict(self.decision),
        }


def _row_to_event(row: HaltEventRow) -> HaltEvent:
    """Build a :class:`HaltEvent` from its ORM row.

    Args:
        row: the persisted row.

    Returns:
        The event.
    """
    payload: object = row.decision
    return HaltEvent(
        halt_event_id=row.halt_event_id,
        kind=HaltEventKind(row.kind),
        cause=None if row.cause is None else HaltCause(row.cause),
        detail=row.detail,
        actor=row.actor,
        occurred_at=row.occurred_at,
        resolves_halt_event_id=row.resolves_halt_event_id,
        alert_dedup_key=row.alert_dedup_key,
        decision=dict(payload) if isinstance(payload, dict) else None,
        stamp=ReproducibilityStamp(
            git_commit=row.git_commit,
            git_dirty=row.git_dirty,
            data_version=row.data_version,
            config_hash=row.config_hash,
            seed=row.seed,
        ),
    )


async def record_halt(
    session: AsyncSession,
    decision: HaltDecision,
    *,
    alert_dedup_key: str | None = None,
    actor: str = AUTOMATIC_ACTOR,
) -> HaltEvent:
    """Record that trading has stopped, with the decision that stopped it.

    Does not commit: the caller owns the transaction, so a cycle's alert and its
    halt land together or not at all.

    Args:
        session: a writable ``AsyncSession``. This table is not bitemporal — a
            halt is something *we* decided, not a fact about the market — so no
            as-of scoping applies.
        decision: the halting decision. Its cause, detail, stamp and full
            payload are copied onto the row, so the history stands alone when
            the objects that produced it are gone.
        alert_dedup_key: the alert this halt raised, when one was raised.
            Recording it is what lets :func:`record_resume` require an
            acknowledgement.
        actor: who halted. Defaults to :data:`AUTOMATIC_ACTOR`.

    Returns:
        The recorded :class:`HaltEvent`.

    Raises:
        HaltHistoryError: if ``decision`` is not a halt, or carries no cause. A
            continue decision has nothing to record here, and recording one
            would put a row in the history that reads as an outage.
    """
    supplied: object = decision
    if not isinstance(supplied, HaltDecision):
        msg = f"decision must be a HaltDecision, got {type(supplied).__name__}"
        raise HaltHistoryError(msg)
    if decision.action is not HaltAction.HALT or decision.cause is None:
        msg = (
            f"record_halt requires a halting decision with a cause; got action="
            f"{decision.action!r}, cause={decision.cause!r}. A continue decision recorded in "
            f"the halt history would read as an outage that never happened."
        )
        raise HaltHistoryError(msg)
    result = await session.execute(
        sa.insert(HaltEventRow)
        .values(
            kind=str(HaltEventKind.HALT),
            cause=str(decision.cause),
            detail=decision.detail,
            actor=actor,
            occurred_at=decision.decided_at,
            resolves_halt_event_id=None,
            alert_dedup_key=alert_dedup_key,
            decision=decision.to_dict(),
            git_commit=decision.stamp.git_commit,
            git_dirty=decision.stamp.git_dirty,
            data_version=decision.stamp.data_version,
            config_hash=decision.stamp.config_hash,
            seed=decision.stamp.seed,
        )
        .returning(HaltEventRow.halt_event_id)
    )
    halt_event_id = int(result.scalar_one())
    return HaltEvent(
        halt_event_id=halt_event_id,
        kind=HaltEventKind.HALT,
        cause=decision.cause,
        detail=decision.detail,
        actor=actor,
        occurred_at=decision.decided_at,
        resolves_halt_event_id=None,
        alert_dedup_key=alert_dedup_key,
        decision=decision.to_dict(),
        stamp=decision.stamp,
    )


async def record_resume(
    session: AsyncSession,
    *,
    halt_event_id: int,
    actor: str,
    reason: str,
    stamp: ReproducibilityStamp,
    now: dt.datetime | None = None,
) -> HaltEvent:
    """Record an operator clearing a specific halt.

    Refuses three ways, each of which would make the history a worse record than
    no history:

    * the referenced event is not an unresolved halt — resuming something that
      is not halted, or resuming twice, would leave the derived state ambiguous;
    * the actor or the reason is blank — the resume is the human decision in
      this pipeline and the history records which human made it;
    * the halt's alert has no acknowledgement — nobody has stated they read it,
      so nothing has been reviewed and trading does not restart.

    Args:
        session: a writable ``AsyncSession``.
        halt_event_id: the halt being cleared.
        actor: the operator resuming. Never blank.
        reason: what they concluded and what changed. Never blank.
        stamp: the I2 stamp of the run recording the resume.
        now: the resume instant (UTC). Defaults to the current instant.

    Returns:
        The recorded resume :class:`HaltEvent`.

    Raises:
        HaltHistoryError: if the event is unknown, is not a halt, is already
            resumed, or the actor or reason is blank.
        UnacknowledgedHaltError: if the halt's alert carries no acknowledgement.
    """
    for name, value in (("actor", actor), ("reason", reason)):
        supplied: object = value
        if not isinstance(supplied, str) or not supplied.strip():
            msg = (
                f"{name} must be a non-empty string: a resume is the human decision in this "
                f"pipeline and the history records which human made it and why"
            )
            raise HaltHistoryError(msg)
    moment = dt.datetime.now(dt.UTC) if now is None else now
    row = (
        await session.execute(
            sa.select(HaltEventRow).where(HaltEventRow.halt_event_id == halt_event_id)
        )
    ).scalar_one_or_none()
    if row is None:
        msg = f"no halt event {halt_event_id} exists"
        raise HaltHistoryError(msg)
    if row.kind != str(HaltEventKind.HALT):
        msg = f"halt event {halt_event_id} is a {row.kind!r} row, not a halt"
        raise HaltHistoryError(msg)
    existing = (
        await session.execute(
            sa.select(HaltEventRow.halt_event_id).where(
                HaltEventRow.resolves_halt_event_id == halt_event_id
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        msg = (
            f"halt event {halt_event_id} was already resumed by event {existing}; resuming it "
            f"again would leave the derived halt state ambiguous"
        )
        raise HaltHistoryError(msg)
    if row.alert_dedup_key is not None:
        acknowledged = (
            await session.execute(
                sa.select(AcknowledgementRow.acknowledgement_id)
                .join(
                    MonitoringAlertRow,
                    MonitoringAlertRow.alert_id == AcknowledgementRow.alert_id,
                )
                .where(MonitoringAlertRow.dedup_key == row.alert_dedup_key)
            )
        ).scalar_one_or_none()
        if acknowledged is None:
            raise UnacknowledgedHaltError(
                halt_event_id=halt_event_id, dedup_key=row.alert_dedup_key
            )
    result = await session.execute(
        sa.insert(HaltEventRow)
        .values(
            kind=str(HaltEventKind.RESUME),
            cause=None,
            detail=reason.strip(),
            actor=actor.strip(),
            occurred_at=moment,
            resolves_halt_event_id=halt_event_id,
            alert_dedup_key=row.alert_dedup_key,
            decision=None,
            git_commit=stamp.git_commit,
            git_dirty=stamp.git_dirty,
            data_version=stamp.data_version,
            config_hash=stamp.config_hash,
            seed=stamp.seed,
        )
        .returning(HaltEventRow.halt_event_id)
    )
    return HaltEvent(
        halt_event_id=int(result.scalar_one()),
        kind=HaltEventKind.RESUME,
        cause=None,
        detail=reason.strip(),
        actor=actor.strip(),
        occurred_at=moment,
        resolves_halt_event_id=halt_event_id,
        alert_dedup_key=row.alert_dedup_key,
        decision=None,
        stamp=stamp,
    )


async def active_halt(session: AsyncSession) -> HaltEvent | None:
    """Return the halt currently in force, or ``None``.

    Derived, never stored: the earliest halt row with no resume pointing at it.
    Earliest rather than latest, because that is the event that started the
    current outage and the one an operator needs to read.

    Args:
        session: any readable ``AsyncSession``.

    Returns:
        The active :class:`HaltEvent`, or ``None`` when trading is permitted.
    """
    resolved = sa.select(HaltEventRow.resolves_halt_event_id).where(
        HaltEventRow.resolves_halt_event_id.is_not(None)
    )
    row = (
        await session.execute(
            sa.select(HaltEventRow)
            .where(
                HaltEventRow.kind == str(HaltEventKind.HALT),
                HaltEventRow.halt_event_id.notin_(resolved),
            )
            .order_by(HaltEventRow.halt_event_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    return None if row is None else _row_to_event(row)


async def require_not_halted(session: AsyncSession) -> None:
    """Raise if a halt is in force — the seam the execution kill switch reads.

    Returns ``None`` when trading is permitted and raises otherwise, so there is
    no return value a caller can forget to inspect (module docstring). It does
    not cache: a cached permission is a stale monitor by another name.

    Args:
        session: any readable ``AsyncSession``.

    Raises:
        SystemHaltedError: when a halt is in force.

    Note:
        Callers must treat **any** exception from this function as halted, not
        only :class:`~backend.monitoring.errors.SystemHaltedError`. A database
        this call cannot read is a monitor that cannot see, and monitoring
        cannot certify that trading is safe from data it does not have.
    """
    halt = await active_halt(session)
    if halt is None:
        return
    raise SystemHaltedError(
        halt_event_id=halt.halt_event_id,
        cause="unknown" if halt.cause is None else str(halt.cause),
        occurred_at_iso=halt.occurred_at.isoformat(),
    )


async def halt_history(session: AsyncSession, *, limit: int = 100) -> tuple[HaltEvent, ...]:
    """Return the halt history, newest first.

    Args:
        session: any readable ``AsyncSession``.
        limit: maximum rows to return. Must be positive.

    Returns:
        Halt and resume events, newest first.

    Raises:
        HaltHistoryError: if ``limit`` is not positive.
    """
    if limit < 1:
        msg = f"limit must be at least 1; got {limit}"
        raise HaltHistoryError(msg)
    rows = (
        (
            await session.execute(
                sa.select(HaltEventRow).order_by(HaltEventRow.halt_event_id.desc()).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return tuple(_row_to_event(row) for row in rows)
