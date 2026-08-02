"""Failure taxonomy for the order management system (P11.2, directive §5 Phase 11).

Every failure here is loud, and the reason is specific to execution rather than
general hygiene. An order management system fails in two characteristic ways,
and both of them are silent by default:

**It sends the same order twice.** A retry after a timeout, a worker restarted
mid-cycle, a broker event stream replayed from an offset — each of these
duplicates an order unless something refuses the second copy. The duplicate is
not a logged warning: it is a doubled position, and in the paper case it is a
corrupted experiment whose realised slippage no longer corresponds to the
intended trade. :class:`DuplicateOrderError` and
:class:`IdempotencyCollisionError` exist so that a duplicate is either absorbed
(the caller gets the incumbent order back) or refused loudly (the key matched
but the *content* did not) — never written.

**It records a state its own history cannot justify.** An order that moves from
``FILLED`` back to ``PENDING_NEW`` reconciles against nothing: the position
implied by the fills and the position implied by the state disagree, and the
disagreement surfaces days later as a phantom holding.
:class:`IllegalTransitionError`, :class:`TerminalOrderError` and
:class:`TransitionChainError` make that unrepresentable rather than unlikely.

The remaining errors guard the arithmetic around those two decisions: an order
whose fields do not describe a tradeable instruction
(:class:`OrderValidationError`), a fill that would take the cumulative quantity
past the ordered quantity (:class:`FillAccountingError`), two workers appending
to one order's history at once (:class:`ConcurrentTransitionError`), a lookup of
an order that was never recorded (:class:`OrderNotFoundError`), and a persisted
row whose venue is not the paper venue (:class:`NotPaperOrderError` — see
:mod:`backend.execution` for why that is a structural impossibility rather than
a configuration mistake).

Reconciliation and the halt log (P11.3, P11.5)
-----------------------------------------------

The second group of failures belongs to the controls rather than to the orders,
and they fail in a third characteristic way: **silently deciding that everything
is fine.** A reconciliation that cannot parse a snapshot, a tolerance widened
until nothing trips it, a halt log that cannot be read — each of those has an
obvious, comfortable, wrong answer ("nothing to report"), and each raises here
instead. :class:`SnapshotValidationError`, :class:`ReconciliationMismatchError`
and :class:`ReconciliationReplayError` cover the comparison;
:class:`SystemHaltedError`, :class:`HaltStateUnavailableError`,
:class:`HaltAlreadyClearedError` and :class:`HaltClearanceError` cover the halt.

:class:`HaltStateUnavailableError` is the one worth reading twice. It exists so
that "I could not determine whether I am halted" is a *refusal to trade* rather
than an exception a caller might mistake for an empty result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.execution.lifecycle import OrderEvent, OrderState

__all__ = [
    "ConcurrentTransitionError",
    "DuplicateOrderError",
    "ExecutionError",
    "FillAccountingError",
    "HaltAlreadyClearedError",
    "HaltClearanceError",
    "HaltStateUnavailableError",
    "IdempotencyCollisionError",
    "IllegalTransitionError",
    "NotPaperOrderError",
    "OrderNotFoundError",
    "OrderValidationError",
    "ReconciliationMismatchError",
    "ReconciliationReplayError",
    "SnapshotValidationError",
    "SystemHaltedError",
    "TerminalOrderError",
    "TransitionChainError",
]


class ExecutionError(Exception):
    """Base class for every failure raised by :mod:`backend.execution`."""


class OrderValidationError(ExecutionError, ValueError):
    """Raised when an order's fields do not describe a tradeable instruction.

    Examples: a non-positive share quantity (nothing to trade), a limit order
    with no limit price or a market order carrying one (the two fields together
    are the instruction, and either half alone is ambiguous), a limit price with
    more precision than the price scale can represent, a slice index outside its
    slice count.

    Every one of these would otherwise reach the database and fail there against
    a CHECK constraint. That is a real backstop and it stays, but the message it
    produces names a constraint rather than a field, and by then the caller has
    an open transaction to unwind. Validating in the value object means the
    refusal names what the caller got wrong.

    Subclasses :class:`ValueError` so ordinary caller-side validation and
    ``pytest.raises(ValueError)`` keep working.
    """


class IllegalTransitionError(ExecutionError, ValueError):
    """Raised when an event has no legal transition from the order's current state.

    The transition table in :mod:`backend.execution.lifecycle` is a partial
    function: 25 of the 110 ``(state, event)`` pairs are legal, and the other 85
    are not oversights — each carries a named
    :class:`~backend.execution.lifecycle.TransitionRefusal`. A venue
    acknowledgement for an order that was never released, a cancel request for
    an order the venue has not accepted, a rejection of an order that has
    already traded — each names a disagreement between our record and the
    venue's, and the honest response is to refuse the transition and let
    reconciliation (P11.3) see the mismatch. Applying it anyway would produce a
    state whose own history does not support it, which is the failure that makes
    a blotter untrustworthy.

    ``refusal`` and ``reason`` are plain strings rather than the enum itself so
    this module stays free of a runtime import of ``lifecycle`` (which imports
    *this* module). :class:`~backend.execution.lifecycle.TransitionRefusal` is a
    ``StrEnum``, so passing a member satisfies the annotation unchanged.

    Attributes:
        state: the state the order was in.
        event: the event that has no transition from it.
        refusal: the named refusal classifying the pair.
        reason: the prose for that refusal.
    """

    def __init__(self, *, state: OrderState, event: OrderEvent, refusal: str, reason: str) -> None:
        """Build the error from the current state, the refused event and its reason."""
        self.state = state
        self.event = event
        self.refusal = refusal
        self.reason = reason
        super().__init__(
            f"event {event.value!r} has no legal transition from state {state.value!r} "
            f"({refusal}): {reason}. The order lifecycle is a partial function on "
            f"(state, event) by design (backend.execution.lifecycle.TRANSITIONS): an event "
            f"the current state cannot accept means our record and the venue's disagree, "
            f"and recording it would produce a state the order's own history does not "
            f"support"
        )


class TerminalOrderError(IllegalTransitionError):
    """Raised when any event is applied to an order in a terminal state.

    A subclass rather than a separate error, because it *is* an illegal
    transition — the specialisation exists so a caller can distinguish "this
    order is finished" from "this event is wrong for this state", which are
    different operational situations: the first usually means a duplicate or
    late venue message, the second means a genuine disagreement.

    Terminal states (``FILLED``, ``CANCELLED``, ``REJECTED``, ``EXPIRED``) have
    **zero** outgoing edges in the transition table. That is what makes
    ``FILLED -> PENDING_NEW`` unrepresentable rather than merely unlikely, and
    migration 0014 restates it as two CHECK constraints — one banning any
    terminal ``from_state``, one enumerating the legal
    ``(from_state, event, to_state)`` triples — so a writer that bypasses this
    module entirely still cannot record one.
    """

    def __init__(self, *, state: OrderState, event: OrderEvent, refusal: str, reason: str) -> None:
        """Build the error from the terminal state, the refused event and its reason."""
        self.state = state
        self.event = event
        self.refusal = refusal
        self.reason = reason
        ExecutionError.__init__(
            self,
            f"order is in terminal state {state.value!r}; event {event.value!r} is refused "
            f"({refusal}): {reason}. Terminal states have no outgoing transitions at all — "
            f"an order that could leave FILLED would let the position implied by its fills "
            f"and the position implied by its state disagree, which is the reconciliation "
            f"bug this machine exists to prevent",
        )


class TransitionChainError(ExecutionError, ValueError):
    """Raised when a persisted transition history does not replay consistently.

    Raised by :func:`backend.execution.lifecycle.replay`, which folds a stored
    history back into a state. It refuses a chain whose sequence numbers are not
    ``1..n`` without gaps, whose ``from_state`` does not equal the previous
    row's ``to_state``, whose transitions are not all legal, or whose cumulative
    fill quantities do not equal the running sum of the per-event quantities.

    These are assertions about this system's own bookkeeping rather than about a
    venue's behaviour, which is why they raise instead of returning a flag: an
    order whose history cannot reconcile with itself has no honest state to
    report, and reporting one anyway is how a phantom position is born.
    """


class FillAccountingError(ExecutionError, ValueError):
    """Raised when a fill's arithmetic does not fit the order it belongs to.

    Two distinct cases, both refusals rather than clamps:

    - **Overfill.** The cumulative filled quantity would exceed the ordered
      quantity. Clamping to the ordered quantity would silently discard shares
      the venue says it traded, leaving the position and the blotter permanently
      out of step; refusing surfaces it while the message is still in hand.
    - **Event/arithmetic mismatch.** The caller supplied ``PARTIAL_FILL`` for a
      fill that completes the order, or ``FILL_COMPLETE`` for one that does not.
      Which of the two events applies is a function of the arithmetic
      (:func:`backend.execution.lifecycle.fill_event`), never of the caller's
      opinion, so a disagreement means one of the two is wrong and neither may
      be trusted.

    Units: every quantity in the message is in **whole shares**.
    """


class DuplicateOrderError(ExecutionError, RuntimeError):
    """Raised when a duplicate submission is refused rather than absorbed.

    The normal path does **not** raise: :func:`backend.execution.store.record_order`
    absorbs a duplicate and returns the incumbent order with
    ``was_already_recorded=True``, which is what makes a retry safe. This error
    exists for the caller who explicitly asks for a duplicate to be an error
    (``if_exists="refuse"``) — the operator flow where recording an order that
    already exists means the caller's own bookkeeping is wrong.

    Attributes:
        idempotency_key: the key already present in the store.
        order_id: the existing order's database key.
    """

    def __init__(self, *, idempotency_key: str, order_id: int) -> None:
        """Build the error from the key and the incumbent order's id."""
        self.idempotency_key = idempotency_key
        self.order_id = order_id
        super().__init__(
            f"an order with idempotency key {idempotency_key} is already recorded as "
            f"order_id={order_id}; refusing to record it a second time"
        )


class IdempotencyCollisionError(ExecutionError, RuntimeError):
    """Raised when a stored order shares a key with a *different* order content.

    The idempotency key is a SHA-256 digest of the order's canonical content, so
    two different orders sharing a key is either a hash collision (which nobody
    should expect to see) or, far more likely, a change to the preimage recipe
    that was not accompanied by a version bump on
    :data:`backend.execution.idempotency.IDEMPOTENCY_SCHEMA`.

    Either way the store refuses. Treating the incumbent as "the same order"
    would silently substitute one trade for another — the caller would believe
    its order was already recorded when a different one was. The stored preimage
    exists precisely so this is detectable rather than assumed: the digest is
    verifiable against the text that produced it.

    Attributes:
        idempotency_key: the shared key.
        order_id: the incumbent order's database key.
        stored_preimage: the canonical JSON recorded with the incumbent.
        computed_preimage: the canonical JSON of the order being recorded.
    """

    def __init__(
        self,
        *,
        idempotency_key: str,
        order_id: int,
        stored_preimage: str,
        computed_preimage: str,
    ) -> None:
        """Build the error from the key, the incumbent, and both preimages."""
        self.idempotency_key = idempotency_key
        self.order_id = order_id
        self.stored_preimage = stored_preimage
        self.computed_preimage = computed_preimage
        super().__init__(
            f"idempotency key {idempotency_key} is already held by order_id={order_id}, "
            f"whose recorded content differs from the order being submitted. Stored "
            f"preimage: {stored_preimage}. Computed preimage: {computed_preimage}. "
            f"Two distinct orders cannot share one key: absorbing this submission as a "
            f"duplicate would substitute one trade for another. If the preimage recipe "
            f"changed, bump backend.execution.idempotency.IDEMPOTENCY_SCHEMA so old and "
            f"new keys cannot collide"
        )


class ConcurrentTransitionError(ExecutionError, RuntimeError):
    """Raised when two writers append to one order's history at the same time.

    Transition rows carry a per-order ``sequence_number``, unique within the
    order, and each writer computes ``last + 1`` from what it read. Two writers
    that read the same tail both try to write the same sequence number, and the
    database rejects the loser. That rejection is this error.

    **The rejection arrives by one of two routes**, and which one depends only
    on where the loser is when the winner commits. If the loser's ``INSERT``
    reached the index first it blocks there and fails on the primary key; if it
    starts after the winner committed, the ``BEFORE INSERT`` chain guard sees
    the winner's row — ``READ COMMITTED`` gives every statement a fresh snapshot,
    so the trigger sees a tail the loser's own earlier ``SELECT`` did not — and
    refuses it before the index is consulted. Migration 0014 raises that case
    with SQLSTATE ``unique_violation`` so both routes are one condition here
    rather than two, one of which nobody would have thought to catch.

    It is a **retryable** condition, not corruption: the loser re-reads the tail
    (which now includes the winner's row) and decides again — its event may or
    may not still be legal from the new state, and that is the point. The
    alternative, letting both writes land, would produce two rows claiming the
    same position in one order's history with no way to order them.

    Optimistic control rather than a lock because the contended case is rare
    (one reconciliation cycle at a time per order) and a lock held across a
    broker round trip is how an OMS deadlocks itself.

    Attributes:
        order_id: the order whose history was contended.
        sequence_number: the position both writers tried to claim.
    """

    def __init__(self, *, order_id: int, sequence_number: int) -> None:
        """Build the error from the order and the contended sequence number."""
        self.order_id = order_id
        self.sequence_number = sequence_number
        super().__init__(
            f"transition {sequence_number} of order_id={order_id} was written by another "
            f"writer first. Re-read the order's history and decide again: the event may no "
            f"longer be legal from the state that now holds"
        )


class OrderNotFoundError(ExecutionError, LookupError):
    """Raised when an order id or idempotency key names no recorded order.

    A lookup miss is an error rather than ``None`` because every caller in this
    package is acting on an order it believes exists — appending a venue message
    to it, reading its state for reconciliation. ``None`` at those call sites
    turns into an attribute error three frames away from the cause.
    """


class NotPaperOrderError(ExecutionError, RuntimeError):
    """Raised when a persisted order's venue is not the paper venue.

    This should be unreachable, and saying so precisely is the point of the
    class. The venue column has a ``CHECK (venue = 'paper')`` constraint
    (migration 0014), the Python value object takes no venue argument at all,
    and :class:`backend.execution.orders.ExecutionVenue` has exactly one member.
    A row that violates all three arrived by a path outside this system.

    The read side refuses it rather than trusting the write side, because the
    single claim this package makes — that nothing here can describe a live
    order — is worth checking on both sides of the database. Directive §1.1 and
    §9.5: live trading is not configurable, so a non-paper order is not a
    configuration to honour but a corruption to refuse.

    The constructor parameter is named ``stored_venue`` rather than ``venue``
    deliberately: it is a value *read out of a row and refused*, never one a
    caller chooses. Nothing in this package accepts a venue to act on, and the
    paper-only test suite enforces that by scanning every parameter name.

    Attributes:
        order_id: the offending order's database key.
        stored_venue: the venue string found on the row.
    """

    def __init__(self, *, order_id: int, stored_venue: str) -> None:
        """Build the error from the order id and the unexpected venue string."""
        self.order_id = order_id
        self.stored_venue = stored_venue
        super().__init__(
            f"order_id={order_id} carries venue={stored_venue!r}, which is not the paper "
            f"venue. "
            f"This platform is paper-only and permanently so (directive §1.1, §9.5): there "
            f"is no configuration that produces a non-paper order, so this row did not come "
            f"from this system. Refusing to load it"
        )


class SnapshotValidationError(ExecutionError, ValueError):
    """Raised when a position/cash snapshot cannot be trusted as an observation.

    Covers a snapshot whose cash is non-finite or carries more precision than the
    schema can hold, a share count that is not a whole number, a naive timestamp,
    an origin on the wrong side of the comparison, a tolerance outside its bounds,
    two snapshots too far apart in time to describe one book, and a stored payload
    that cannot be rebuilt.

    Every one of those has a tempting silent alternative — coerce the ``NaN``,
    round the cash, assume UTC, widen the tolerance, interpret half the payload —
    and each of them produces a *clean* reconciliation over a book that was never
    checked. Raising means the cycle gets no verdict, and a cycle with no verdict
    is an unknown condition the kill switch halts on
    (:mod:`backend.execution.killswitch`). Failing towards a halt is the whole
    design.

    Subclasses :class:`ValueError` so ordinary caller-side validation keeps
    working.
    """


class ReconciliationMismatchError(ExecutionError, RuntimeError):
    """Raised by :func:`backend.execution.reconciliation.require_matched` on a break.

    The explicit form of "a mismatch halts", for a caller that wants the cycle to
    stop at the reconciliation step rather than continue and let the kill switch
    record it. The kill switch itself does **not** use this: a halt has to be
    persisted, and an exception is not a halt — it does not survive the process
    that raised it.

    Attributes:
        cycle_id: the cycle whose reconciliation failed.
        break_count: how many findings were breaks.
        result_digest: the verdict's digest, so the stored row that carries the
            same digest can be found and re-run.
        detail: the concatenated prose of every break.
    """

    def __init__(self, *, cycle_id: str, break_count: int, result_digest: str, detail: str) -> None:
        """Build the error from the cycle, the break count, the digest and the prose."""
        self.cycle_id = cycle_id
        self.break_count = break_count
        self.result_digest = result_digest
        self.detail = detail
        super().__init__(
            f"reconciliation for cycle {cycle_id!r} found {break_count} break(s) "
            f"(result_digest={result_digest}): {detail}"
        )


class ReconciliationReplayError(ExecutionError, ValueError):
    """Raised when a stored reconciliation cannot be found or does not re-derive.

    A break that cannot be re-examined afterwards cannot be investigated, so the
    stored row holds both snapshots and the verdict's digest, and
    :func:`backend.execution.reconciliation.rerun` recomputes the second from the
    first. A digest that moves means either the stored payload or the comparison
    changed since the verdict was recorded, and neither of those may be presented
    as the original finding — the re-derived answer would look like history and
    would not be.
    """


class SystemHaltedError(ExecutionError, RuntimeError):
    """Raised by the release guard when any halt is open.

    Trading is stopped and stays stopped until a human clears the halt by id
    (:func:`backend.execution.halt.clear_halt`). There is no timeout and no
    automatic re-arm: a halt survives the condition that caused it, because the
    point of halting is that somebody looks.

    Attributes:
        halt_ids: the open engagements' ids, ascending. These are the ids a
            clearance must name — the error carries them so an operator does not
            have to go looking.
        triggers: the trigger of each open halt, in the same order.
        detail: the prose recorded with each.
    """

    def __init__(self, *, halt_ids: Sequence[int], triggers: Sequence[str], detail: str) -> None:
        """Build the error from the open halts' ids, triggers and prose."""
        self.halt_ids = tuple(halt_ids)
        self.triggers = tuple(triggers)
        self.detail = detail
        super().__init__(
            f"trading is halted by {len(self.halt_ids)} open halt(s) "
            f"{list(self.halt_ids)} ({', '.join(self.triggers)}): {detail}. A halt is cleared "
            f"only by an explicit, attributed clearance naming the halt id"
        )


class HaltStateUnavailableError(ExecutionError, RuntimeError):
    """Raised when whether the system is halted cannot be determined.

    This is the fail-closed error, and it is deliberately not an empty result. "I
    could not read the halt log" and "there are no halts" are different facts, and
    only the second permits trading; returning the first as the second would let a
    database outage do what no operator is allowed to do — silently re-enable a
    halted system.

    Also raised when the log holds a trigger that names no known condition.
    Guessing what it meant is how a halt gets cleared by mistake.
    """


class HaltAlreadyClearedError(ExecutionError, RuntimeError):
    """Raised when a halt has already been cleared by someone else.

    Not corruption and not retryable — the halt *is* cleared, and the caller's
    clearance is simply not the one that did it. The condition arrives by either
    of two routes carrying one SQLSTATE (``23505``): the unique index on
    ``clears_halt_id``, or the clearance guard seeing a committed clearance the
    caller's own earlier read did not (D-034 — under ``READ COMMITTED`` a
    ``BEFORE INSERT`` trigger takes a fresh snapshot and fires ahead of the index).

    Attributes:
        halt_id: the engagement the caller tried to clear.
    """

    def __init__(self, *, halt_id: int) -> None:
        """Build the error from the halt that was already cleared."""
        self.halt_id = halt_id
        super().__init__(
            f"halt_id={halt_id} has already been cleared by another clearance. The halt is "
            f"cleared; this attempt is not the one that cleared it, and a second clearance "
            f"row would leave the log with two answers to 'who turned it back on'"
        )


class HaltClearanceError(ExecutionError, ValueError):
    """Raised when a clearance is unattributed or names something that is not a halt.

    Two cases, both refusals rather than normalisations:

    - **Unattributed.** A blank ``cleared_by`` or ``clearance_reason``. The
      asymmetry with :class:`~backend.execution.killswitch.ManualHaltRequest` —
      which normalises blanks rather than raising, so an operator's halt can never
      be refused over a format check — is the design: recording a halt must never
      fail, and removing one must never be casual.
    - **Not an open engagement.** The named row does not exist, or is itself a
      clearance. Refused by the ``execution_halt_clearance_guard`` trigger under
      SQLSTATE ``P0001``, which is kept distinct from ``23505`` because retrying
      it would loop forever (D-034, and the same split migration 0014 makes
      between a taken sequence position and a gap).
    """
