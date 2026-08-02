"""The spend ledger: reserve before the call, settle after it (P7.7, I2, I4).

This is where the cap is actually enforced, and the shape of the interface is
the whole argument.

Check-then-act is a race, and this one spends money
----------------------------------------------------

The obvious implementation reads the day's committed spend, compares it to the
limit, and — if it fits — makes the call. Two calls running that sequence at the
same time both read the same total, both find room, and both spend. Nothing in
the code is wrong line by line; the window between the read and the spend is
the bug, and it widens with concurrency, which is exactly the condition a
backfill runs under. A cap enforced this way holds under load precisely as well
as it is not needed.

So there is no "check" operation on this interface. There is
:meth:`SpendLedger.reserve`, which **checks and commits the reservation as one
atomic step**, and either returns a :class:`Reservation` or raises
:class:`~backend.extraction.governor.errors.SpendCapExceededError`. A caller
cannot use it wrongly by forgetting to hold something, because there is nothing
to forget: obtaining permission and consuming headroom are the same call.

Atomicity is a property of the implementation, and both implementations state
theirs:

* :class:`InMemorySpendLedger` — one ``asyncio.Lock``, correct for every
  coroutine in one event loop and **not** across processes. It is what a
  single-process backfill and the tests use.
* :class:`~backend.extraction.governor.postgres.PostgresSpendLedger` — a
  transaction-scoped PostgreSQL advisory lock per provider, which is correct
  across connections, processes and hosts. It is what a deployment uses.

Reserve at the bound, settle at the truth
------------------------------------------

A reservation consumes the call's **upper bound**
(:func:`~backend.extraction.governor.estimate.estimate_call_cost`), because at
reservation time the response does not exist. Reserving the expected cost would
re-introduce the leak the bound exists to close.

Settlement then books the difference. The ledger is an **append-only event log**
— reserved, released, settled — and each event carries a signed ``delta`` that
sums to the window's committed spend:

===========  ==========================================
event        delta
===========  ==========================================
reserved     ``+estimate``
released     ``-estimate``
settled      ``actual - estimate`` (usually negative)
===========  ==========================================

so ``committed(window) = Σ delta``. Nothing is ever updated, which is what makes
the trail an audit trail rather than a running total whose history has been
overwritten (§6.11: config changes are events, not mutations; the same reasoning
applies to spend). It is also what makes the arithmetic checkable after the
fact: the reservation and its settlement are both still there, so "how wrong was
the bound" is a question the rows answer.

Failure settles at the bound, and does not release
---------------------------------------------------

When the provider call raises, this system cannot tell whether the request never
left the host or died after the model had generated most of a response. Both
raise the same
:class:`~backend.extraction.tasks.client.ModelCallError`-shaped
failure. Releasing would assume nothing was spent; settling at the bound assumes
the worst. The worst is the correct assumption for a control whose purpose is to
prevent overspend: over-counting refuses calls that would have fit, under-
counting lets real money out.

:meth:`SpendLedger.release` therefore exists for one situation only — a caller
that **knows** no request was sent, because it abandoned the call after
reserving. The governed client never uses it, because it never knows; that
asymmetry is documented on
:class:`~backend.extraction.governor.guard.GovernedModelClient` rather than left
for a reader to infer.

Units: every amount is a :class:`~backend.extraction.governor.estimate.Money`
carrying its own currency (I4). Token counts are dimensionless and, where they
describe what actually happened, are **as reported by the provider** — never
estimated (I3).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import uuid
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from backend.core.logging import get_logger
from backend.extraction.governor.caps import CapPolicy, SpendWindow
from backend.extraction.governor.errors import (
    LedgerIntegrityError,
    SpendCapExceededError,
)
from backend.extraction.governor.estimate import Money, total

if TYPE_CHECKING:
    from collections.abc import Mapping

    from backend.extraction.governor.estimate import CallEstimate
    from backend.extraction.providers.catalog import Provider

__all__ = [
    "CallOutcome",
    "InMemorySpendLedger",
    "Reservation",
    "SpendEvent",
    "SpendLedger",
    "SpendRecord",
    "new_reservation_id",
]

_logger = get_logger(__name__)


class SpendEvent(StrEnum):
    """What one ledger row records.

    Attributes:
        RESERVED: headroom was consumed before a call. Written by the only
            operation that may admit new spend.
        RELEASED: a reservation was given back because the caller could prove no
            request was sent.
        SETTLED: the call finished — successfully or not — and the reservation
            was reconciled against what it actually cost.
    """

    RESERVED = "reserved"
    RELEASED = "released"
    SETTLED = "settled"


class CallOutcome(StrEnum):
    """How a settled call ended.

    Attributes:
        SUCCEEDED: a response came back.
        FAILED: the provider call raised. The reservation still settles at its
            bound — see the module docstring for why the worst case is the
            correct assumption here.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"


def new_reservation_id() -> str:
    """Return a fresh reservation identifier.

    Returns:
        A 32-character lowercase hex UUID4. Generated in the process rather than
        by the database so a reservation can be referred to before its row is
        committed, and so the in-memory and PostgreSQL ledgers issue identifiers
        of the same shape.
    """
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class Reservation:
    """Headroom consumed on behalf of one not-yet-made call.

    Attributes:
        reservation_id: the identifier settlement and release quote.
        provider: whose budget it consumes.
        requested_model: the qualified model the caller asked for.
        served_model: the qualified model that is actually going to be called.
            Differs from *requested_model* exactly when the governor degraded.
        estimate: the **upper bound** that was reserved, with the token bounds
            it was derived from.
        windows: window to its UTC key, as of the reservation instant.
        limits: window to the limit in force when the reservation was taken.
            Carried on the reservation, not looked up later, so a cap raised
            this afternoon cannot rewrite this morning's audit trail.
        policy: the ceiling behaviour in force for this provider.
        correlation_id: the request id (D-003) the call belongs to, or ``None``
            outside a request.
        occurred_at: the UTC instant the windows were computed from.
    """

    reservation_id: str
    provider: Provider
    requested_model: str
    served_model: str
    estimate: CallEstimate
    windows: Mapping[SpendWindow, str]
    limits: Mapping[SpendWindow, Money]
    policy: CapPolicy
    correlation_id: str | None
    occurred_at: dt.datetime

    @property
    def amount(self) -> Money:
        """The reserved upper bound, in the cap's currency."""
        return self.estimate.cost

    @property
    def degraded(self) -> bool:
        """Whether a cheaper model was substituted for the requested one."""
        return self.served_model != self.requested_model


@dataclass(frozen=True, slots=True)
class SpendRecord:
    """One append-only ledger row: the audit trail of a single spend event.

    Attributes:
        reservation_id: ties the reserved / settled / released rows of one call
            together.
        event: which of the three this row is.
        provider: whose budget moved.
        requested_model: what the caller asked for.
        served_model: what actually answered — the substituted model when the
            governor degraded. **This is the field that makes a degraded result
            reproducible (I2)**: without it, an artefact says a call was made to
            a model that never saw the document.
        estimated_cost: the upper bound reserved for this call.
        actual_cost: what it really cost, from provider-reported token counts,
            or ``None`` when the provider reported none. Never re-estimated.
        delta: this row's signed contribution to its windows' committed spend.
            Committed spend is the sum of deltas — see the module docstring.
        input_tokens_bound: the prompt-token upper bound the estimate used.
        output_tokens_bound: the response-token upper bound, i.e. ``max_tokens``.
        input_tokens: prompt tokens as reported by the provider, or ``None``.
        output_tokens: response tokens as reported, or ``None``.
        windows: window to UTC key this row is booked in.
        limits: the limits in force when the row was written.
        policy: the ceiling behaviour in force.
        outcome: how the call ended, on a ``settled`` row; ``None`` otherwise.
        reconciled: whether *actual_cost* is a real measurement. ``False`` means
            the row settled at its bound because the provider reported no token
            counts — which over-counts, deliberately.
        correlation_id: request id (D-003), or ``None``.
        occurred_at: UTC instant of the event.
    """

    reservation_id: str
    event: SpendEvent
    provider: Provider
    requested_model: str
    served_model: str
    estimated_cost: Money
    actual_cost: Money | None
    delta: Money
    input_tokens_bound: int
    output_tokens_bound: int
    input_tokens: int | None
    output_tokens: int | None
    windows: Mapping[SpendWindow, str]
    limits: Mapping[SpendWindow, Money]
    policy: CapPolicy
    outcome: CallOutcome | None
    reconciled: bool
    correlation_id: str | None
    occurred_at: dt.datetime

    @property
    def degraded(self) -> bool:
        """Whether this call was served by a substituted model."""
        return self.served_model != self.requested_model


@runtime_checkable
class SpendLedger(Protocol):
    """Somewhere spend is atomically admitted and durably recorded.

    Deliberately has no ``check`` and no ``add``: admitting spend and recording
    it are one operation, because separating them is the race the module
    docstring describes.
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

        Raises:
            backend.extraction.governor.errors.SpendCapExceededError: the call
                would breach a limit. **No request has been made** — this is
                raised before the caller ever reaches the provider.
        """
        ...

    async def settle(
        self,
        reservation: Reservation,
        *,
        actual: Money | None,
        input_tokens: int | None,
        output_tokens: int | None,
        outcome: CallOutcome,
    ) -> SpendRecord:
        """Reconcile a reservation against what the call really cost."""
        ...

    async def release(self, reservation: Reservation) -> SpendRecord:
        """Give back a reservation whose call the caller can prove was never sent."""
        ...

    async def committed(
        self,
        *,
        provider: Provider,
        window: SpendWindow,
        window_key: str,
        currency: str,
    ) -> Money:
        """Return spend settled or reserved in one window."""
        ...

    async def records(self) -> tuple[SpendRecord, ...]:
        """Return every recorded event, oldest first."""
        ...


class InMemorySpendLedger:
    """A process-local spend ledger, atomic within one event loop.

    Correct for every coroutine sharing this object's ``asyncio.Lock``, and
    **not** correct across processes: two workers each holding their own
    instance would each admit a full cap's worth of spend. That is not a defect
    to be fixed here — it is what
    :class:`~backend.extraction.governor.postgres.PostgresSpendLedger` is for,
    and it is stated so that nobody wires this one into a multi-worker backfill
    on the assumption that a lock is a lock.

    Rows live in a list and are never mutated, mirroring the append-only table
    in migration 0013, so the two implementations answer ``committed`` with the
    same arithmetic over the same event shapes.
    """

    def __init__(self) -> None:
        """Build an empty ledger with its own lock."""
        self._events: list[SpendRecord] = []
        self._open: dict[str, Reservation] = {}
        self._closed: dict[str, SpendEvent] = {}
        self._lock = asyncio.Lock()

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

        The lock spans the read of committed spend, the comparison, and the
        write of the reservation row. That span **is** the enforcement: without
        it two coroutines both read the same headroom and both consume it.

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
            The :class:`Reservation`, with headroom already consumed.

        Raises:
            backend.extraction.governor.errors.SpendCapExceededError: a window
                would be breached. Nothing was reserved and **no request has
                been made**.
            backend.extraction.governor.errors.CurrencyMismatchError: the
                estimate and a limit are in different currencies.
        """
        currency = estimate.cost.currency
        async with self._lock:
            for window, limit in limits.items():
                window_key = windows[window]
                committed = self._committed(provider, window, window_key, currency)
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
            # A cooperative yield inside the critical section, at the point the
            # PostgreSQL ledger performs a round trip. It is here so the section
            # is a real critical section under test: a coroutine containing no
            # await cannot be interleaved, so without this the lock would be
            # untestable and deleting it would pass every test. See
            # backend/tests/extraction/governor/test_ledger.py, which admits
            # exactly the number of concurrent calls the cap allows and fails
            # when the lock is removed.
            await asyncio.sleep(0)
            reservation = Reservation(
                reservation_id=new_reservation_id(),
                provider=provider,
                requested_model=requested_model,
                served_model=served_model,
                estimate=estimate,
                windows=MappingProxyType(dict(windows)),
                limits=MappingProxyType(dict(limits)),
                policy=policy,
                correlation_id=correlation_id,
                occurred_at=occurred_at,
            )
            self._open[reservation.reservation_id] = reservation
            self._events.append(
                _record(
                    reservation,
                    event=SpendEvent.RESERVED,
                    actual=None,
                    delta=estimate.cost,
                    input_tokens=None,
                    output_tokens=None,
                    outcome=None,
                    reconciled=False,
                    occurred_at=occurred_at,
                )
            )
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
        """Reconcile *reservation* against what the call really cost.

        Args:
            reservation: the reservation taken before the call.
            actual: the measured cost from provider-reported token counts, or
                ``None`` when the provider reported none — in which case the
                reservation settles at its bound, which over-counts on purpose.
            input_tokens: prompt tokens as reported, or ``None``.
            output_tokens: response tokens as reported, or ``None``.
            outcome: whether the call succeeded or failed.

        Returns:
            The ``settled`` row.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: this
                reservation was already settled or released, or the ledger never
                issued it. Each would corrupt the running total every later cap
                check reads.
        """
        settled_amount = actual if actual is not None else reservation.amount
        async with self._lock:
            self._require_open(reservation)
            delta = settled_amount - reservation.amount
            record = _record(
                reservation,
                event=SpendEvent.SETTLED,
                actual=actual,
                delta=delta,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                outcome=outcome,
                reconciled=actual is not None,
                occurred_at=dt.datetime.now(tz=dt.UTC),
            )
            self._events.append(record)
            del self._open[reservation.reservation_id]
            self._closed[reservation.reservation_id] = SpendEvent.SETTLED
        if actual is not None and actual > reservation.amount:
            # The bound was not a bound. Recorded rather than clamped: clamping
            # would hide the one observation that falsifies the estimator, and
            # the cap can only ever be exceeded by this difference.
            _logger.warning(
                "llm_spend_exceeded_bound",
                provider=reservation.provider.value,
                served_model=reservation.served_model,
                estimated_cost=str(reservation.amount),
                actual_cost=str(actual),
                reservation_id=reservation.reservation_id,
            )
        return record

    async def release(self, reservation: Reservation) -> SpendRecord:
        """Give back *reservation*, for a caller that can prove no request was sent.

        Args:
            reservation: the reservation to undo.

        Returns:
            The ``released`` row, whose delta cancels the reservation exactly.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: this
                reservation was already settled or released, or was never
                issued.
        """
        async with self._lock:
            self._require_open(reservation)
            record = _record(
                reservation,
                event=SpendEvent.RELEASED,
                actual=None,
                delta=-reservation.amount,
                input_tokens=None,
                output_tokens=None,
                outcome=None,
                reconciled=False,
                occurred_at=dt.datetime.now(tz=dt.UTC),
            )
            self._events.append(record)
            del self._open[reservation.reservation_id]
            self._closed[reservation.reservation_id] = SpendEvent.RELEASED
        return record

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
            window_key: the UTC key, ``YYYY-MM-DD`` or ``YYYY-MM``.
            currency: the currency of the answer, needed because an empty window
                has no row to take one from — and a zero without a currency is
                the bare number I4 forbids.

        Returns:
            The sum of every row's delta in that window: settled spend plus
            reservations still outstanding.
        """
        async with self._lock:
            return self._committed(provider, window, window_key, currency)

    async def records(self) -> tuple[SpendRecord, ...]:
        """Return every recorded event, oldest first — the audit trail."""
        async with self._lock:
            return tuple(self._events)

    def _committed(
        self, provider: Provider, window: SpendWindow, window_key: str, currency: str
    ) -> Money:
        """Sum the deltas booked to one window. Caller holds the lock."""
        return total(
            (
                event.delta
                for event in self._events
                if event.provider is provider and event.windows.get(window) == window_key
            ),
            currency=currency,
        )

    def _require_open(self, reservation: Reservation) -> None:
        """Refuse to close a reservation that is not open. Caller holds the lock.

        Raises:
            backend.extraction.governor.errors.LedgerIntegrityError: already
                closed, or never issued.
        """
        closed = self._closed.get(reservation.reservation_id)
        if closed is not None:
            msg = (
                f"reservation {reservation.reservation_id!r} was already {closed.value}; "
                "closing it twice would double-count its delta and corrupt every later cap check"
            )
            raise LedgerIntegrityError(msg)
        if reservation.reservation_id not in self._open:
            msg = (
                f"reservation {reservation.reservation_id!r} was never issued by this ledger, "
                "so there is no headroom of its to reconcile"
            )
            raise LedgerIntegrityError(msg)


def _record(
    reservation: Reservation,
    *,
    event: SpendEvent,
    actual: Money | None,
    delta: Money,
    input_tokens: int | None,
    output_tokens: int | None,
    outcome: CallOutcome | None,
    reconciled: bool,
    occurred_at: dt.datetime,
) -> SpendRecord:
    """Build one ledger row from its reservation and the event that closed it.

    Shared by both ledger implementations so the row shape cannot drift between
    the process-local and the durable trail.

    Args:
        reservation: the reservation this row belongs to.
        event: reserved, settled or released.
        actual: measured cost, or ``None``.
        delta: this row's signed contribution to committed spend.
        input_tokens: prompt tokens as reported, or ``None``.
        output_tokens: response tokens as reported, or ``None``.
        outcome: how the call ended, on a settled row.
        reconciled: whether *actual* is a real measurement.
        occurred_at: UTC instant of the event.

    Returns:
        The :class:`SpendRecord`.
    """
    return SpendRecord(
        reservation_id=reservation.reservation_id,
        event=event,
        provider=reservation.provider,
        requested_model=reservation.requested_model,
        served_model=reservation.served_model,
        estimated_cost=reservation.amount,
        actual_cost=actual,
        delta=delta,
        input_tokens_bound=reservation.estimate.input_tokens_bound,
        output_tokens_bound=reservation.estimate.output_tokens_bound,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        windows=reservation.windows,
        limits=reservation.limits,
        policy=reservation.policy,
        outcome=outcome,
        reconciled=reconciled,
        correlation_id=reservation.correlation_id,
        occurred_at=occurred_at,
    )
