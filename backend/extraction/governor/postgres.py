"""The durable spend ledger: PostgreSQL, append-only, correct across processes (P7.7).

Same surface as :class:`~backend.extraction.governor.ledger.InMemorySpendLedger`
and the same event algebra — reserved, released, settled, with committed spend
as the sum of signed deltas. What this adds is the two properties a deployment
needs and a process-local list cannot have:

**Atomicity across processes.** The in-memory ledger's ``asyncio.Lock`` protects
the coroutines of one event loop. Two Celery workers each holding their own
instance would each admit a full cap's worth of spend, and the cap would be
worth as many multiples of itself as there are workers. Here the critical
section — read committed spend, compare, insert the reservation — runs inside
one transaction that first takes ``pg_advisory_xact_lock`` on a key derived from
the provider (:func:`_provider_lock_key`, the same construction
:mod:`backend.extraction.providers.assignments` and :mod:`backend.db.audit` use).
A second reserver for the same provider blocks until the first commits, and then
reads a total that includes it. Different providers never contend, so one
provider's burst cannot serialize another's.

The lock is taken **before** the read, not around the insert. That ordering is
the entire correctness argument: a lock held only over the write would still let
two transactions read the same headroom.

**Durability of the audit trail.** Rows survive the process, so "what did we
spend, on what, under whose cap, and which model actually answered" is
answerable after a restart, and §6.5's spend gauges and month-end projection
read committed rows rather than a total held in somebody's memory. Migration
0013 installs the same ``BEFORE UPDATE OR DELETE`` trigger shape the rest of this
schema uses: a spend event is an observation, and an observation that can be
edited afterwards is not evidence.

**This module's runtime behaviour is unexercised.** There is no Docker daemon in
this environment, so no statement below has ever reached a server. Its SQL
construction is a pure function and is tested as one — the compiled statements
are asserted to take the advisory lock and to scope the sum to the right
provider and window — and migration 0013 is asserted to match the ORM model
column for column. Treat the round trip itself as unverified until the
integration suite runs; nothing here is presented as evidence that it works
(I3). Same honesty as
:class:`~backend.extraction.cache.RedisCacheStore`, and for the same reason.

Units: amounts are ``NUMERIC(20, 10)`` in the currency named by the row's
``currency`` column, in the currency's major unit (I4). Token counts are
dimensionless.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from decimal import Decimal
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.models import LlmSpendLedger
from backend.extraction.governor.caps import CapPolicy, SpendWindow
from backend.extraction.governor.errors import (
    LedgerIntegrityError,
    SpendCapExceededError,
)
from backend.extraction.governor.estimate import Money
from backend.extraction.governor.ledger import (
    CallOutcome,
    Reservation,
    SpendEvent,
    SpendRecord,
    new_reservation_id,
)
from backend.extraction.providers.catalog import Provider

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.extraction.governor.estimate import CallEstimate

__all__ = ["PostgresSpendLedger", "committed_statement"]

_logger = get_logger(__name__)

_LOCK_KEY_BYTES: Final = 8
"""Digest width of the advisory-lock key: 8 bytes == one PostgreSQL ``bigint``."""

_LOCK_NAMESPACE: Final = "llm_spend_ledger"
"""Prefix mixed into the advisory-lock digest so this module's locks cannot
collide with another module's locks over the same provider name."""


def _provider_lock_key(provider: Provider) -> int:
    """Return the advisory-lock key serializing reservations for one provider.

    Args:
        provider: whose budget is about to be read and written.

    Returns:
        A signed 64-bit integer for ``pg_advisory_xact_lock(bigint)``, derived
        from a BLAKE2b digest of this module's namespace and the provider name
        joined by a NUL byte (which cannot occur inside either, so no two
        distinct keys collide by concatenation). Hashed in Python rather than
        with PostgreSQL's ``hashtext`` so the mapping is deterministic,
        documented and testable instead of resting on an internal server
        function — the same construction
        :mod:`backend.extraction.providers.assignments` uses.
    """
    material = "\x00".join((_LOCK_NAMESPACE, provider.value)).encode()
    digest = hashlib.blake2b(material, digest_size=_LOCK_KEY_BYTES).digest()
    return int.from_bytes(digest, "big", signed=True)


def committed_statement(
    provider: Provider, window: SpendWindow, window_key: str
) -> sa.Select[tuple[Decimal]]:
    """Return the statement totalling committed spend in one window.

    ``SUM(delta_amount)`` over every row booked to that provider and window —
    reservations still outstanding plus the deltas their settlements booked. A
    module-level function rather than a method so the SQL can be compiled and
    asserted without a database, which is the only verification available while
    Docker is down.

    Args:
        provider: whose budget to total.
        window: daily or monthly.
        window_key: the UTC key, ``YYYY-MM-DD`` or ``YYYY-MM``.

    Returns:
        A ``SELECT`` yielding one ``NUMERIC``, zero when the window has no rows.
    """
    column = (
        LlmSpendLedger.daily_window
        if window is SpendWindow.DAILY
        else LlmSpendLedger.monthly_window
    )
    return sa.select(sa.func.coalesce(sa.func.sum(LlmSpendLedger.delta_amount), 0)).where(
        LlmSpendLedger.provider == provider.value,
        column == window_key,
    )


class PostgresSpendLedger:
    """The durable, cross-process spend ledger.

    Satisfies :class:`~backend.extraction.governor.ledger.SpendLedger`. Each
    method opens its own ``ingest_writer_session`` — the sanctioned append-only
    write path — because a reservation must commit before the call it authorizes
    is made, and holding a caller's transaction open across a provider round trip
    would pin a connection for the duration of an LLM call.
    """

    async def reserve(
        self,
        *,
        provider: Provider,
        requested_model: str,
        served_model: str,
        estimate: CallEstimate,
        windows: Mapping[SpendWindow, str],
        limits: Mapping[SpendWindow, Money],
        policy: CapPolicy,
        correlation_id: str | None,
        occurred_at: dt.datetime,
    ) -> Reservation:
        """Atomically admit *estimate* against every window, or refuse.

        The transaction takes the provider's advisory lock **first**, then reads
        committed spend, then inserts. A concurrent reserver for the same
        provider blocks on the lock and, once it proceeds, reads a total that
        already includes this reservation. That is the whole cross-process
        answer; see the module docstring for why a lock around the write alone
        would not be one.

        Args:
            provider: whose budget is being spent.
            requested_model: qualified model the caller asked for.
            served_model: qualified model that will actually be called.
            estimate: the upper bound to reserve.
            windows: window to UTC key.
            limits: window to limit in force.
            policy: the provider's ceiling behaviour, recorded on the row.
            correlation_id: request id (D-003), or ``None``.
            occurred_at: UTC instant the windows were computed from.

        Returns:
            The :class:`~backend.extraction.governor.ledger.Reservation`.

        Raises:
            backend.extraction.governor.errors.SpendCapExceededError: a window
                would be breached. The transaction is abandoned, nothing is
                written, and **no request has been made**.
        """
        currency = estimate.cost.currency
        reservation = Reservation(
            reservation_id=new_reservation_id(),
            provider=provider,
            requested_model=requested_model,
            served_model=served_model,
            estimate=estimate,
            windows=dict(windows),
            limits=dict(limits),
            policy=policy,
            correlation_id=correlation_id,
            occurred_at=occurred_at,
        )
        lock_key = sa.literal(_provider_lock_key(provider), sa.BigInteger)
        async with ingest_writer_session() as session:
            await session.execute(sa.select(sa.func.pg_advisory_xact_lock(lock_key)))
            for window, limit in limits.items():
                window_key = windows[window]
                committed = await self._committed(session, provider, window, window_key, currency)
                if committed + estimate.cost > limit:
                    raise SpendCapExceededError(
                        provider=provider,
                        window=window,
                        window_key=window_key,
                        limit=limit,
                        committed=committed,
                        estimate=estimate.cost,
                        requested_model=requested_model,
                    )
            session.add(
                _row(
                    reservation,
                    event=SpendEvent.RESERVED,
                    actual=None,
                    delta=estimate.cost,
                    input_tokens=None,
                    output_tokens=None,
                    outcome=None,
                    occurred_at=occurred_at,
                )
            )
            await session.commit()
        _logger.info(
            "llm_spend_reserved",
            provider=provider.value,
            requested_model=requested_model,
            served_model=served_model,
            degraded=reservation.degraded,
            estimated_cost=str(estimate.cost),
            reservation_id=reservation.reservation_id,
        )
        return reservation

    async def settle(
        self,
        reservation: Reservation,
        *,
        actual: Money | None,
        input_tokens: int | None,
        output_tokens: int | None,
        outcome: CallOutcome,
    ) -> SpendRecord:
        """Reconcile a reservation against what the call really cost.

        Args:
            reservation: the reservation taken before the call.
            actual: measured cost from provider-reported token counts, or
                ``None`` when the provider reported none — the reservation then
                settles at its bound, which over-counts on purpose.
            input_tokens: prompt tokens as reported, or ``None``.
            output_tokens: response tokens as reported, or ``None``.
            outcome: whether the call succeeded or failed.

        Returns:
            The ``settled`` row.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: this
                reservation was already settled or released, or the ledger has
                no reservation row for it. The unique index on
                ``(reservation_id, event)`` is the database's half of the same
                guarantee.
        """
        settled_amount = actual if actual is not None else reservation.amount
        delta = settled_amount - reservation.amount
        row = _row(
            reservation,
            event=SpendEvent.SETTLED,
            actual=actual,
            delta=delta,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            outcome=outcome,
            occurred_at=dt.datetime.now(tz=dt.UTC),
        )
        async with ingest_writer_session() as session:
            await self._require_open(session, reservation)
            session.add(row)
            await session.commit()
        if actual is not None and actual > reservation.amount:
            _logger.warning(
                "llm_spend_exceeded_bound",
                provider=reservation.provider.value,
                served_model=reservation.served_model,
                estimated_cost=str(reservation.amount),
                actual_cost=str(actual),
                reservation_id=reservation.reservation_id,
            )
        return _record_of(row)

    async def release(self, reservation: Reservation) -> SpendRecord:
        """Give back a reservation whose call the caller can prove was never sent.

        Args:
            reservation: the reservation to undo.

        Returns:
            The ``released`` row, whose delta cancels the reservation exactly.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: already
                settled or released, or never reserved.
        """
        row = _row(
            reservation,
            event=SpendEvent.RELEASED,
            actual=None,
            delta=-reservation.amount,
            input_tokens=None,
            output_tokens=None,
            outcome=None,
            occurred_at=dt.datetime.now(tz=dt.UTC),
        )
        async with ingest_writer_session() as session:
            await self._require_open(session, reservation)
            session.add(row)
            await session.commit()
        return _record_of(row)

    async def committed(
        self,
        *,
        provider: Provider,
        window: SpendWindow,
        window_key: str,
        currency: str,
    ) -> Money:
        """Return spend settled or reserved in one window.

        Args:
            provider: whose budget to total.
            window: daily or monthly.
            window_key: the UTC key.
            currency: the currency of the answer, needed because an empty window
                has no row to take one from.

        Returns:
            The sum of every row's delta in that window.
        """
        async with ingest_writer_session() as session:
            return await self._committed(session, provider, window, window_key, currency)

    async def records(self) -> tuple[SpendRecord, ...]:
        """Return every recorded event, oldest first — the durable audit trail."""
        statement = sa.select(LlmSpendLedger).order_by(LlmSpendLedger.ledger_id)
        async with ingest_writer_session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(_record_of(row) for row in rows)

    async def _committed(
        self,
        session: AsyncSession,
        provider: Provider,
        window: SpendWindow,
        window_key: str,
        currency: str,
    ) -> Money:
        """Total the deltas booked to one window on an open session."""
        amount = (await session.scalars(committed_statement(provider, window, window_key))).one()
        return Money(Decimal(amount), currency)

    async def _require_open(self, session: AsyncSession, reservation: Reservation) -> None:
        """Refuse to close a reservation that was never opened or is already closed.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: no
                ``reserved`` row exists for this identifier, or a ``settled`` or
                ``released`` row already does. Both would corrupt the running
                total every later cap check reads.
        """
        statement = sa.select(LlmSpendLedger.event).where(
            LlmSpendLedger.reservation_id == reservation.reservation_id
        )
        events = set((await session.scalars(statement)).all())
        if SpendEvent.RESERVED.value not in events:
            msg = (
                f"reservation {reservation.reservation_id!r} has no reserved row in the spend "
                "ledger, so there is no headroom of its to reconcile"
            )
            raise LedgerIntegrityError(msg)
        for closing in (SpendEvent.SETTLED, SpendEvent.RELEASED):
            if closing.value in events:
                msg = (
                    f"reservation {reservation.reservation_id!r} was already {closing.value}; "
                    "closing it twice would double-count its delta and corrupt every later cap "
                    "check"
                )
                raise LedgerIntegrityError(msg)


def _row(
    reservation: Reservation,
    *,
    event: SpendEvent,
    actual: Money | None,
    delta: Money,
    input_tokens: int | None,
    output_tokens: int | None,
    outcome: CallOutcome | None,
    occurred_at: dt.datetime,
) -> LlmSpendLedger:
    """Build the ORM row for one ledger event.

    Args:
        reservation: the reservation this row belongs to.
        event: reserved, settled or released.
        actual: measured cost, or ``None``.
        delta: signed contribution to committed spend.
        input_tokens: prompt tokens as reported, or ``None``.
        output_tokens: response tokens as reported, or ``None``.
        outcome: how the call ended, on a settled row.
        occurred_at: UTC instant of the event.

    Returns:
        An unflushed :class:`~backend.db.models.LlmSpendLedger`.
    """
    return LlmSpendLedger(
        reservation_id=reservation.reservation_id,
        event=event.value,
        provider=reservation.provider.value,
        requested_model=reservation.requested_model,
        served_model=reservation.served_model,
        currency=reservation.amount.currency,
        estimated_cost=reservation.amount.amount,
        actual_cost=None if actual is None else actual.amount,
        delta_amount=delta.amount,
        input_tokens_bound=reservation.estimate.input_tokens_bound,
        output_tokens_bound=reservation.estimate.output_tokens_bound,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        daily_window=reservation.windows[SpendWindow.DAILY],
        monthly_window=reservation.windows[SpendWindow.MONTHLY],
        daily_limit=reservation.limits[SpendWindow.DAILY].amount,
        monthly_limit=reservation.limits[SpendWindow.MONTHLY].amount,
        policy=reservation.policy.value,
        outcome=None if outcome is None else outcome.value,
        reconciled=actual is not None,
        correlation_id=reservation.correlation_id,
        occurred_at=occurred_at,
    )


def _record_of(row: LlmSpendLedger) -> SpendRecord:
    """Convert a stored row into the read model.

    Monetary columns come back as :class:`~decimal.Decimal` and are rewrapped in
    :class:`~backend.extraction.governor.estimate.Money` with the row's own
    ``currency``, so no amount leaves this module as a bare number (I4).
    """
    currency = row.currency
    return SpendRecord(
        reservation_id=row.reservation_id,
        event=SpendEvent(row.event),
        provider=Provider(row.provider),
        requested_model=row.requested_model,
        served_model=row.served_model,
        estimated_cost=Money(row.estimated_cost, currency),
        actual_cost=None if row.actual_cost is None else Money(row.actual_cost, currency),
        delta=Money(row.delta_amount, currency),
        input_tokens_bound=row.input_tokens_bound,
        output_tokens_bound=row.output_tokens_bound,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        windows={
            SpendWindow.DAILY: row.daily_window,
            SpendWindow.MONTHLY: row.monthly_window,
        },
        limits={
            SpendWindow.DAILY: Money(row.daily_limit, currency),
            SpendWindow.MONTHLY: Money(row.monthly_limit, currency),
        },
        policy=CapPolicy(row.policy),
        outcome=None if row.outcome is None else CallOutcome(row.outcome),
        reconciled=row.reconciled,
        correlation_id=row.correlation_id,
        occurred_at=row.occurred_at,
    )
