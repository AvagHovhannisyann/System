"""Persistence for orders and their transitions: append-only, with the audit trail.

What this module is, and is not
-------------------------------

It reads and writes two tables and it does nothing else. It opens no connection
of its own — every function takes the ``AsyncSession`` the caller is already
holding and never commits, because a rebalance's orders should land together or
not at all. It contacts no venue: there is no client here, no endpoint, no
setting, and no interface an adapter could be selected into (see
:mod:`backend.execution.orders` for why that absence is the design rather than an
omission). Venue messages arrive as :class:`~backend.execution.lifecycle.OrderEvent`
values a caller applies.

Where the guarantees actually live
----------------------------------

**Idempotency is the database's job.** :func:`record_order` does not look before
it inserts. It inserts inside a savepoint and lets ``UNIQUE
(idempotency_key)`` decide; only after the database has refused does it read the
incumbent, and only to decide what to tell the caller. A look-then-insert would
have a window between the two, and two workers racing through that window both
find the key free — which is precisely the duplicate this whole mechanism exists
to prevent. The Python here cannot be the enforcement point, and is written so
that it visibly is not.

**Legality is the state machine's job, checked twice.**
:func:`append_transition` computes the next state through
:func:`~backend.execution.lifecycle.apply_event`, so an illegal event never
reaches SQL; migration 0014's ``legal_transition`` and ``from_state_not_terminal``
CHECKs and the ``execution_transition_chain_guard`` trigger refuse it again for
any writer that skips this module.

**Current state is derived, never stored.** ``execution_order`` has no state
column: the tables are append-only, so a stored state could not be updated, and a
denormalized value that drifts from the history is the condition the transition
log exists to make impossible. :func:`load_order` replays the log
(:func:`~backend.execution.lifecycle.replay`), which validates the whole chain on
the way through.

Transaction isolation
---------------------

Written for Postgres' default ``READ COMMITTED``. Under it, a concurrent
duplicate insert *blocks* on the unique index until the first transaction
commits or rolls back, then either fails (committed) or succeeds (rolled back),
and the losing transaction can immediately read the winner's row. Under
``REPEATABLE READ`` the loser would instead get a serialization failure, which is
also correct behaviour — a retry re-runs and is absorbed — but the caller sees a
different exception, so a caller choosing a stricter isolation level should be
prepared to retry the transaction as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError, IntegrityError

from backend.db.models import ExecutionOrder as ExecutionOrderRow
from backend.db.models import ExecutionOrderTransition as ExecutionOrderTransitionRow
from backend.execution.errors import (
    ConcurrentTransitionError,
    DuplicateOrderError,
    FillAccountingError,
    IdempotencyCollisionError,
    NotPaperOrderError,
    OrderNotFoundError,
    OrderValidationError,
    TransitionChainError,
)
from backend.execution.idempotency import (
    IDEMPOTENCY_SCHEMA,
    idempotency_preimage,
    key_of_preimage,
)
from backend.execution.lifecycle import (
    FILL_EVENTS,
    OrderEvent,
    OrderState,
    Transition,
    apply_event,
    fill_event,
    replay,
)
from backend.execution.orders import ExecutionVenue

if TYPE_CHECKING:
    import datetime as dt

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.execution.orders import FillReport, OrderIntent

__all__ = [
    "LoadedOrder",
    "RecordedOrder",
    "append_transition",
    "load_order",
    "load_order_by_key",
    "record_order",
]

UNIQUE_VIOLATION_SQLSTATE: Final = "23505"
"""SQLSTATE ``unique_violation`` — the code both sequence-conflict paths carry.

A concurrent append is refused by the database in one of **two** ways, depending
on where the loser is when the winner commits, and this module has to translate
both into the same retryable :class:`~backend.execution.errors.ConcurrentTransitionError`:

1. **The primary key.** The loser's ``INSERT`` reaches the index while the
   winner's row is still uncommitted, blocks on it, and fails when the winner
   commits. Postgres raises ``unique_violation`` natively.
2. **The chain-guard trigger.** The loser's ``INSERT`` starts *after* the winner
   committed. Under ``READ COMMITTED`` each statement takes a fresh snapshot, so
   the ``BEFORE INSERT`` trigger sees the winner's row even though the loser's
   earlier ``SELECT`` did not — and refuses the row before the index is ever
   consulted. Migration 0014 raises that specific case ``USING ERRCODE =
   'unique_violation'`` so it arrives here as the same SQLSTATE rather than as a
   generic ``P0001`` the caller would have to match on message text.

Path 2 is the likelier one under real contention, and it is the one an in-memory
double that models only the index will miss entirely.
"""

DuplicatePolicy = Literal["absorb", "refuse"]
"""What :func:`record_order` does when the key is already present.

``"absorb"`` returns the incumbent (the behaviour that makes a retry safe);
``"refuse"`` raises :class:`~backend.execution.errors.DuplicateOrderError`, for
the operator flow where recording an order that already exists means the
caller's own bookkeeping is wrong.
"""


@dataclass(frozen=True, slots=True)
class RecordedOrder:
    """The outcome of a submission.

    Attributes:
        order_id: the order's database key — the incumbent's when a duplicate
            was absorbed, so a retry addresses the same order the first attempt
            created.
        idempotency_key: the content-derived key, 64 lowercase hex characters.
        was_already_recorded: ``True`` when this submission was absorbed as a
            duplicate rather than written. The whole point of the mechanism: a
            caller can retry blindly and read this to find out whether it was the
            one that landed. Read the order's state with :func:`load_order`.
    """

    order_id: int
    idempotency_key: str
    was_already_recorded: bool


@dataclass(frozen=True, slots=True)
class LoadedOrder:
    """A persisted order and the state its recorded history folds to.

    Attributes:
        order_id: database key.
        idempotency_key: the content-derived key it was recorded under.
        quantity_shares: the ordered quantity, in **whole shares**.
        state: current state, replayed from ``transitions``.
        filled_quantity_shares: cumulative shares filled, in **whole shares**.
        transitions: the whole audit trail in sequence order, oldest first.
    """

    order_id: int
    idempotency_key: str
    quantity_shares: int
    state: OrderState
    filled_quantity_shares: int
    transitions: tuple[Transition, ...]

    @property
    def sequence_number(self) -> int:
        """The last recorded sequence number; ``0`` when nothing has happened yet."""
        return len(self.transitions)


def _require_paper(order_id: int, stored_venue: str) -> None:
    """Refuse a persisted order whose venue is not the paper venue.

    Checked on the read side even though the write side cannot produce one: the
    single claim this package makes is that nothing here can describe a live
    order, and it is worth checking on both sides of the database.

    Args:
        order_id: the row's key, for the message.
        stored_venue: the venue string found on the row.

    Raises:
        NotPaperOrderError: if ``stored_venue`` is not
            :attr:`~backend.execution.orders.ExecutionVenue.PAPER`.
    """
    if stored_venue != ExecutionVenue.PAPER.value:
        raise NotPaperOrderError(order_id=order_id, stored_venue=stored_venue)


def _sqlstate(exc: DBAPIError) -> str | None:
    """Return the five-character SQLSTATE a driver exception carries, if any.

    SQLAlchemy exposes no portable accessor, so the driver's own attribute is
    read: asyncpg names it ``sqlstate``, psycopg names it ``pgcode``. A driver
    that exposes neither returns ``None``, and the caller falls back to the
    exception class.

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


def _is_position_taken(exc: DBAPIError) -> bool:
    """Return whether ``exc`` means another writer already holds this sequence number.

    Decided on :data:`UNIQUE_VIOLATION_SQLSTATE`, not on the exception class.
    The class is too coarse: a CHECK violation and a foreign-key failure are
    ``IntegrityError`` too, and translating either into a *retryable*
    ``ConcurrentTransitionError`` would send the caller round a loop that can
    never succeed. Only ``23505`` means "that position is taken", and both of
    the database's two refusal paths carry it (see
    :data:`UNIQUE_VIOLATION_SQLSTATE`).

    Args:
        exc: the wrapped database error raised by the transition insert.

    Returns:
        ``True`` when the row was refused because its position is already held.

    Note:
        When the driver exposes no SQLSTATE the class is used as a fallback,
        which is the best available answer for a driver that will not say —
        never a silent widening for one that will.
    """
    sqlstate = _sqlstate(exc)
    if sqlstate is not None:
        return sqlstate == UNIQUE_VIOLATION_SQLSTATE
    return isinstance(exc, IntegrityError)


def _require_aware(name: str, moment: dt.datetime) -> None:
    """Refuse a naive timestamp.

    Args:
        name: field name, for the message.
        moment: the timestamp to check.

    Raises:
        OrderValidationError: if ``moment`` carries no timezone. A naive instant
            in an audit trail is a number whose meaning depends on the machine
            that wrote it, and reconciliation compares our timestamps with a
            venue's.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        msg = (
            f"{name}={moment!r} is timezone-naive; an audit trail compared against a venue's "
            f"clock cannot carry an instant whose meaning depends on the writer's locale"
        )
        raise OrderValidationError(msg)


async def record_order(
    session: AsyncSession,
    intent: OrderIntent,
    *,
    if_exists: DuplicatePolicy = "absorb",
) -> RecordedOrder:
    """Record an order, absorbing a duplicate submission instead of doubling it.

    The insert is attempted first and the ``UNIQUE (idempotency_key)`` constraint
    decides. Only after the database refuses is the incumbent read, and then only
    to choose between three outcomes:

    - the stored preimage matches, and ``if_exists="absorb"`` — the incumbent's
      ``order_id`` is returned with ``was_already_recorded=True``, which is what
      makes an unconditional retry safe;
    - the stored preimage matches, and ``if_exists="refuse"`` —
      :class:`~backend.execution.errors.DuplicateOrderError`;
    - the stored preimage **differs** —
      :class:`~backend.execution.errors.IdempotencyCollisionError`, because
      absorbing it would substitute one trade for another.

    Does **not** commit: the caller owns the transaction. The insert runs inside a
    savepoint so a refused duplicate leaves the caller's transaction usable.

    Args:
        session: any writable ``AsyncSession``. Neither table is bitemporal, so
            no as-of scoping applies.
        intent: the order to record, already validated by its own constructor.
        if_exists: see :data:`DuplicatePolicy`.

    Returns:
        A :class:`RecordedOrder`. The order is in
        :data:`~backend.execution.lifecycle.INITIAL_STATE` when freshly written;
        read :func:`load_order` for an absorbed one.

    Raises:
        DuplicateOrderError: the key exists and ``if_exists="refuse"``.
        IdempotencyCollisionError: the key exists on an order with different
            content.
        NotPaperOrderError: the incumbent row is not a paper order.
        sqlalchemy.exc.IntegrityError: any other constraint violation, re-raised
            unchanged — this function interprets exactly one constraint and
            refuses to guess about the rest.
    """
    preimage = idempotency_preimage(intent)
    key = key_of_preimage(preimage)
    statement = (
        sa.insert(ExecutionOrderRow)
        .values(
            idempotency_key=key,
            idempotency_schema=IDEMPOTENCY_SCHEMA,
            idempotency_preimage=preimage,
            # `venue` is deliberately absent: no writer supplies it, the server
            # default does, and the CHECK refuses anything else. Passing it here
            # would create the one thing this package must not have — a place
            # where a venue is chosen.
            security_id=intent.security_id,
            side=intent.side.value,
            quantity_shares=intent.quantity_shares,
            order_type=intent.order_type.value,
            time_in_force=intent.time_in_force.value,
            limit_price_usd=intent.limit_price_usd,
            rebalance_date=intent.rebalance_date,
            slice_index=intent.slice_index,
            slice_count=intent.slice_count,
            git_commit=intent.stamp.git_commit,
            git_dirty=intent.stamp.git_dirty,
            data_version=intent.stamp.data_version,
            config_hash=intent.stamp.config_hash,
            seed=intent.stamp.seed,
        )
        .returning(ExecutionOrderRow.order_id)
    )
    try:
        async with session.begin_nested():
            inserted = await session.execute(statement)
            order_id = int(inserted.scalar_one())
    except IntegrityError:
        incumbent = await _incumbent(session, key)
        if incumbent is None:
            # Some other constraint failed. Reinterpreting it as a duplicate
            # would report a doubled order where there is a malformed one.
            raise
        existing_id, stored_venue, stored_preimage = incumbent
        _require_paper(existing_id, stored_venue)
        if stored_preimage != preimage:
            raise IdempotencyCollisionError(
                idempotency_key=key,
                order_id=existing_id,
                stored_preimage=stored_preimage,
                computed_preimage=preimage,
            ) from None
        if if_exists == "refuse":
            raise DuplicateOrderError(idempotency_key=key, order_id=existing_id) from None
        return RecordedOrder(order_id=existing_id, idempotency_key=key, was_already_recorded=True)
    return RecordedOrder(order_id=order_id, idempotency_key=key, was_already_recorded=False)


async def _incumbent(session: AsyncSession, key: str) -> tuple[int, str, str] | None:
    """Read the order already holding ``key``, if any.

    Args:
        session: the session to read through.
        key: the idempotency key to look up.

    Returns:
        ``(order_id, venue, idempotency_preimage)``, or ``None`` when the key is
        not present — which, on the duplicate path, means some other constraint
        was the one that failed.
    """
    result = await session.execute(
        sa.select(
            ExecutionOrderRow.order_id,
            ExecutionOrderRow.venue,
            ExecutionOrderRow.idempotency_preimage,
        ).where(ExecutionOrderRow.idempotency_key == key)
    )
    row = result.first()
    if row is None:
        return None
    return int(row[0]), str(row[1]), str(row[2])


def _transition_of_row(
    order_id: int,
    sequence_number: int,
    from_state: str,
    event: str,
    to_state: str,
    fill_quantity_shares: int | None,
    filled_quantity_after_shares: int,
) -> Transition:
    """Build a :class:`~backend.execution.lifecycle.Transition` from stored columns.

    Args:
        order_id: the owning order, quoted in the error.
        sequence_number: the row's position in the history.
        from_state: stored state string.
        event: stored event string.
        to_state: stored state string.
        fill_quantity_shares: stored fill quantity, or ``None``.
        filled_quantity_after_shares: stored cumulative quantity.

    Returns:
        The in-memory transition.

    Raises:
        TransitionChainError: if a stored string names no member of the state or
            event enum. Such a row cannot have been written by this package, and
            guessing what it meant is how a blotter starts lying.
    """
    try:
        return Transition(
            sequence_number=sequence_number,
            from_state=OrderState(from_state),
            event=OrderEvent(event),
            to_state=OrderState(to_state),
            fill_quantity_shares=fill_quantity_shares,
            filled_quantity_after_shares=filled_quantity_after_shares,
        )
    except ValueError as exc:
        msg = (
            f"transition {sequence_number} of order_id={order_id} stores "
            f"({from_state!r}, {event!r}, {to_state!r}), which names no member of the "
            f"lifecycle enums: {exc}"
        )
        raise TransitionChainError(msg) from exc


async def _load_transitions(session: AsyncSession, order_id: int) -> tuple[Transition, ...]:
    """Read one order's whole history in sequence order.

    Args:
        session: the session to read through.
        order_id: the order whose history to read.

    Returns:
        The transitions, oldest first.

    Raises:
        TransitionChainError: on a row whose state or event is not a known enum
            member.
    """
    result = await session.execute(
        sa.select(
            ExecutionOrderTransitionRow.sequence_number,
            ExecutionOrderTransitionRow.from_state,
            ExecutionOrderTransitionRow.event,
            ExecutionOrderTransitionRow.to_state,
            ExecutionOrderTransitionRow.fill_quantity_shares,
            ExecutionOrderTransitionRow.filled_quantity_after_shares,
        )
        .where(ExecutionOrderTransitionRow.order_id == order_id)
        .order_by(ExecutionOrderTransitionRow.sequence_number)
    )
    return tuple(
        _transition_of_row(
            order_id=order_id,
            sequence_number=int(row[0]),
            from_state=str(row[1]),
            event=str(row[2]),
            to_state=str(row[3]),
            fill_quantity_shares=None if row[4] is None else int(row[4]),
            filled_quantity_after_shares=int(row[5]),
        )
        for row in result.all()
    )


async def _order_header(
    session: AsyncSession, *, order_id: int | None, idempotency_key: str | None
) -> tuple[int, str, str, int]:
    """Read one order's immutable header by id or by key.

    Args:
        session: the session to read through.
        order_id: the order's database key, or ``None``.
        idempotency_key: the order's content key, or ``None``. Exactly one of
            the two must be given.

    Returns:
        ``(order_id, idempotency_key, venue, quantity_shares)``.

    Raises:
        OrderValidationError: if both or neither selector is given.
        OrderNotFoundError: if no order matches.
    """
    if (order_id is None) == (idempotency_key is None):
        msg = "exactly one of order_id and idempotency_key must be given"
        raise OrderValidationError(msg)
    statement = sa.select(
        ExecutionOrderRow.order_id,
        ExecutionOrderRow.idempotency_key,
        ExecutionOrderRow.venue,
        ExecutionOrderRow.quantity_shares,
    )
    if order_id is not None:
        statement = statement.where(ExecutionOrderRow.order_id == order_id)
    else:
        statement = statement.where(ExecutionOrderRow.idempotency_key == idempotency_key)
    row = (await session.execute(statement)).first()
    if row is None:
        selector = f"order_id={order_id}" if order_id is not None else f"key={idempotency_key}"
        msg = (
            f"no order matches {selector}. Every caller here is acting on an order it "
            f"believes exists, so a miss is an error rather than a None that turns into an "
            f"attribute error three frames away"
        )
        raise OrderNotFoundError(msg)
    return int(row[0]), str(row[1]), str(row[2]), int(row[3])


async def load_order(session: AsyncSession, order_id: int) -> LoadedOrder:
    """Load one order and replay its recorded history into a current state.

    Args:
        session: any readable ``AsyncSession``.
        order_id: the order's database key.

    Returns:
        A :class:`LoadedOrder`.

    Raises:
        OrderNotFoundError: if no order has that id.
        NotPaperOrderError: if the row's venue is not the paper venue.
        TransitionChainError: if the recorded history is not internally
            consistent — see :func:`~backend.execution.lifecycle.replay`.
    """
    found_id, key, stored_venue, quantity = await _order_header(
        session, order_id=order_id, idempotency_key=None
    )
    return await _loaded(session, found_id, key, stored_venue, quantity)


async def load_order_by_key(session: AsyncSession, idempotency_key: str) -> LoadedOrder:
    """Load one order by its content-derived idempotency key.

    The lookup a retrying caller uses when it has the intent but not the id: the
    key is recomputable from the order's own fields, so it needs nothing
    remembered from the first attempt.

    Args:
        session: any readable ``AsyncSession``.
        idempotency_key: 64 lowercase hex characters.

    Returns:
        A :class:`LoadedOrder`.

    Raises:
        OrderNotFoundError: if no order holds that key.
        NotPaperOrderError: if the row's venue is not the paper venue.
        TransitionChainError: if the recorded history is not internally
            consistent.
    """
    found_id, key, stored_venue, quantity = await _order_header(
        session, order_id=None, idempotency_key=idempotency_key
    )
    return await _loaded(session, found_id, key, stored_venue, quantity)


async def _loaded(
    session: AsyncSession, order_id: int, key: str, stored_venue: str, quantity: int
) -> LoadedOrder:
    """Assemble a :class:`LoadedOrder` from a header and the order's history.

    Args:
        session: the session to read through.
        order_id: the order's database key.
        key: its idempotency key.
        stored_venue: its stored venue string.
        quantity: its ordered quantity in whole shares.

    Returns:
        A :class:`LoadedOrder`.

    Raises:
        NotPaperOrderError: if ``stored_venue`` is not the paper venue.
        TransitionChainError: if the history does not replay consistently.
    """
    _require_paper(order_id, stored_venue)
    transitions = await _load_transitions(session, order_id)
    folded = replay(transitions, ordered_quantity_shares=quantity)
    return LoadedOrder(
        order_id=order_id,
        idempotency_key=key,
        quantity_shares=quantity,
        state=folded.state,
        filled_quantity_shares=folded.filled_quantity_shares,
        transitions=transitions,
    )


def _fill_quantity(
    *,
    event: OrderEvent,
    fill: FillReport | None,
    ordered_quantity_shares: int,
    filled_before_shares: int,
) -> int:
    """Validate the fill payload against the event and return the traded quantity.

    Which of ``PARTIAL_FILL`` and ``FILL_COMPLETE`` applies is a function of the
    arithmetic, never of the caller's opinion, so the caller's event is checked
    against :func:`~backend.execution.lifecycle.fill_event` and a disagreement is
    refused: one of the two is wrong and neither may be trusted.

    Args:
        event: the event being appended.
        fill: the fill report, or ``None``.
        ordered_quantity_shares: the order's quantity, in whole shares.
        filled_before_shares: cumulative filled quantity before this event.

    Returns:
        The quantity this event trades, in whole shares — ``0`` for a non-fill.

    Raises:
        FillAccountingError: if a fill event carries no report, a non-fill event
            carries one, the arithmetic overfills the order, or the caller's
            event disagrees with the arithmetic.
    """
    if event not in FILL_EVENTS:
        if fill is not None:
            msg = (
                f"event {event.value!r} is not a fill but carries a fill report. The payload "
                f"columns are present exactly on fill events and absent on every other "
                f"(migration 0014, both directions): a quantity on a non-fill row is a trade "
                f"nobody reported"
            )
            raise FillAccountingError(msg)
        return 0
    if fill is None:
        msg = (
            f"event {event.value!r} is a fill and must carry a fill report; a fill whose "
            f"quantity, price and source are absent is a trade whose size is a gap"
        )
        raise FillAccountingError(msg)
    implied = fill_event(
        ordered_quantity_shares=ordered_quantity_shares,
        filled_before_shares=filled_before_shares,
        fill_quantity_shares=fill.quantity_shares,
    )
    if implied is not event:
        msg = (
            f"caller supplied {event.value!r} for a fill of {fill.quantity_shares} shares "
            f"taking the cumulative quantity from {filled_before_shares} to "
            f"{filled_before_shares + fill.quantity_shares} of {ordered_quantity_shares}, "
            f"but the arithmetic makes that {implied.value!r}. Which event applies is a "
            f"function of the quantities, so a disagreement means one of the two is wrong "
            f"and neither may be trusted"
        )
        raise FillAccountingError(msg)
    return fill.quantity_shares


async def append_transition(
    session: AsyncSession,
    *,
    order_id: int,
    event: OrderEvent,
    occurred_at: dt.datetime,
    fill: FillReport | None = None,
    note: str | None = None,
) -> Transition:
    """Append one state change to an order's history.

    Reads the order's current state by replaying its log, checks the event
    against the transition table, checks the fill arithmetic, and inserts the
    next row in the per-order sequence. Does **not** commit.

    Args:
        session: any writable ``AsyncSession``.
        order_id: the order to extend.
        event: the event to apply.
        occurred_at: when the event happened at its origin — the venue's
            timestamp for a venue-reported event, ours for a locally-originated
            one. Must be timezone-aware. Never used for ordering.
        fill: the execution report, required exactly when ``event`` is a fill.
        note: free text (a venue reject reason, an operator's justification).

    Returns:
        The :class:`~backend.execution.lifecycle.Transition` that was written.

    Raises:
        OrderNotFoundError: if no order has that id.
        NotPaperOrderError: if the stored order is not a paper order.
        TerminalOrderError: if the order is already finished.
        IllegalTransitionError: if the event has no transition from the current
            state.
        FillAccountingError: if the fill payload and the event disagree, or the
            fill would overfill the order.
        OrderValidationError: if ``occurred_at`` is timezone-naive.
        ConcurrentTransitionError: if another writer claimed the same sequence
            number first. Retryable: re-read and decide again, because the event
            may no longer be legal from the state that now holds.
        TransitionChainError: if the existing history does not replay
            consistently.
    """
    _require_aware("occurred_at", occurred_at)
    order = await load_order(session, order_id)
    to_state = apply_event(order.state, event)
    traded = _fill_quantity(
        event=event,
        fill=fill,
        ordered_quantity_shares=order.quantity_shares,
        filled_before_shares=order.filled_quantity_shares,
    )
    sequence_number = order.sequence_number + 1
    row = Transition(
        sequence_number=sequence_number,
        from_state=order.state,
        event=event,
        to_state=to_state,
        fill_quantity_shares=None if fill is None else fill.quantity_shares,
        filled_quantity_after_shares=order.filled_quantity_shares + traded,
    )
    statement = sa.insert(ExecutionOrderTransitionRow).values(
        order_id=order_id,
        sequence_number=sequence_number,
        from_state=order.state.value,
        event=event.value,
        to_state=to_state.value,
        filled_quantity_after_shares=row.filled_quantity_after_shares,
        fill_quantity_shares=None if fill is None else fill.quantity_shares,
        fill_price_usd=None if fill is None else fill.price_usd,
        fill_source=None if fill is None else fill.source.value,
        fill_cost_basis=None if fill is None else fill.cost_basis,
        venue_fill_id=None if fill is None else fill.venue_fill_id,
        occurred_at=occurred_at,
        note=note,
    )
    try:
        async with session.begin_nested():
            await session.execute(statement)
    except DBAPIError as exc:
        if not _is_position_taken(exc):
            # A chain error, a CHECK violation, a foreign-key failure: all real
            # refusals of a malformed row, none of them retryable. Translating
            # them into ConcurrentTransitionError would tell the caller to try
            # again forever.
            raise
        raise ConcurrentTransitionError(order_id=order_id, sequence_number=sequence_number) from exc
    return row
