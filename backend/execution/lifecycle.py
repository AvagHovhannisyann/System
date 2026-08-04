"""Order lifecycle: states, events, the legal transition table, and its complement.

This module is the whole of the order state machine. It holds no connection, no
session and no clock: :func:`apply_event` is a pure function on
``(state, event)``, and everything that persists its results lives in
:mod:`backend.execution.store`.

Why the illegal transitions are enumerated rather than left implicit
-------------------------------------------------------------------

:data:`TRANSITIONS` is a **partial** function: 25 of the 110 ``(state, event)``
pairs are legal. The other 85 are refused, and each one is classified by a named
:class:`TransitionRefusal` in :data:`ILLEGAL_TRANSITIONS`. The classification is
built at import time and the build fails if any illegal pair is left
unclassified, so an event or state added later without a refusal rule breaks the
import rather than acquiring a silent default.

The refusal that matters most is :attr:`TransitionRefusal.TERMINAL_STATE`. The
four terminal states — ``FILLED``, ``CANCELLED``, ``REJECTED``, ``EXPIRED`` —
have **zero** outgoing edges. An order that could go ``FILLED -> PENDING_NEW``
would let the position implied by its fills and the position implied by its
state disagree; the disagreement does not surface at the moment it is written,
it surfaces days later as a phantom holding that reconciliation (P11.3) cannot
attribute. Migration 0014 restates the same table as a CHECK constraint over
``(from_state, event, to_state)``, so a writer that bypasses this module
entirely still cannot record one.

Why there are two pending-cancel states
---------------------------------------

``PENDING_CANCEL`` and ``PENDING_CANCEL_PARTIAL`` differ only in whether the
order had traded before the cancel was requested, and they exist because a
venue's cancel *rejection* has to land somewhere. In FIX the reject message
carries the resulting order status, which is the venue telling us what we would
otherwise have to remember; remembering it in the state keeps
:data:`TRANSITIONS` a function of ``(state, event)`` alone. The alternative —
one ``PENDING_CANCEL`` state whose cancel-reject target depends on the
cumulative filled quantity — makes the target a function of the order's
arithmetic, which is exactly the coupling that turns a state machine into a
sequence of special cases. The distinction is also operationally real: a pending
cancel on an order that has already traded leaves a position to reconcile
whatever the venue answers.

Fill accounting
---------------

Which of ``PARTIAL_FILL`` and ``FILL_COMPLETE`` applies is a function of the
arithmetic (:func:`fill_event`), never of the caller's opinion. Quantities are
in **whole shares** throughout this module; no field here is a price or a cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from backend.execution.errors import (
    ExecutionError,
    FillAccountingError,
    IllegalTransitionError,
    TerminalOrderError,
    TransitionChainError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "FILL_EVENTS",
    "ILLEGAL_TRANSITIONS",
    "INITIAL_STATE",
    "REFUSAL_REASONS",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "OrderEvent",
    "OrderState",
    "ReplayResult",
    "Transition",
    "TransitionRefusal",
    "apply_event",
    "fill_event",
    "legal_events",
    "reachable_states",
    "replay",
]


class OrderState(StrEnum):
    """Every state an order can be in. Values are the strings persisted.

    ``DRAFT`` is where every order starts: recorded, stamped, addressable, and
    not yet released. ``PENDING_NEW`` means released and awaiting the venue's
    acknowledgement. ``ACKNOWLEDGED`` means working at the venue with nothing
    traded; ``PARTIALLY_FILLED`` means working with some quantity traded. The
    two ``PENDING_CANCEL*`` states mean a cancel is in flight, distinguished by
    whether anything had traded (see the module docstring). The last four are
    terminal.
    """

    DRAFT = "draft"
    PENDING_NEW = "pending_new"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    PENDING_CANCEL = "pending_cancel"
    PENDING_CANCEL_PARTIAL = "pending_cancel_partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderEvent(StrEnum):
    """Every event that can be applied to an order. Values are persisted.

    Three origins, and the distinction matters when reading a blotter.
    ``RELEASE``, ``ABANDON`` and ``REQUEST_CANCEL`` originate here — they are
    decisions this system made. ``ACKNOWLEDGE``, ``REJECT``, ``PARTIAL_FILL``,
    ``FILL_COMPLETE``, ``CANCEL_CONFIRMED``, ``CANCEL_REJECTED``, ``EXPIRE`` and
    ``VENUE_CANCEL`` originate at the venue — they are things reported to us.
    No event in either group causes this package to contact anything; an event
    is applied by a caller that already holds the message.
    """

    RELEASE = "release"
    ABANDON = "abandon"
    ACKNOWLEDGE = "acknowledge"
    REJECT = "reject"
    PARTIAL_FILL = "partial_fill"
    FILL_COMPLETE = "fill_complete"
    REQUEST_CANCEL = "request_cancel"
    CANCEL_CONFIRMED = "cancel_confirmed"
    CANCEL_REJECTED = "cancel_rejected"
    EXPIRE = "expire"
    VENUE_CANCEL = "venue_cancel"


class TransitionRefusal(StrEnum):
    """Named reasons an ``(state, event)`` pair has no transition.

    Every one of the 85 illegal pairs maps to exactly one of these. The names
    are part of the audit surface: an operator reading a refused event wants to
    know *which* disagreement it represents, because the operational response
    differs — a late duplicate message is routine, a venue reporting a trade on
    an order it never acknowledged is not.
    """

    TERMINAL_STATE = "terminal_state_has_no_outgoing_transitions"
    NOT_YET_RELEASED = "no_event_applies_to_an_unreleased_draft"
    ALREADY_RELEASED = "release_applies_only_to_a_draft"
    ALREADY_ACKNOWLEDGED = "acknowledge_applies_only_to_a_released_order"
    VENUE_MESSAGE_BEFORE_ACKNOWLEDGEMENT = "venue_reported_activity_before_acknowledging"
    CANCEL_WITHOUT_VENUE_ORDER = "cancel_requires_an_acknowledged_order_to_cancel"
    NO_CANCEL_IN_FLIGHT = "cancel_resolution_requires_a_cancel_in_flight"
    CANCEL_ALREADY_IN_FLIGHT = "a_second_cancel_request_while_one_is_in_flight"
    VENUE_CANCEL_IS_THE_CONFIRMATION = "unsolicited_cancel_while_our_cancel_is_in_flight"
    REJECT_AFTER_ACKNOWLEDGEMENT = "rejection_is_the_alternative_to_acknowledgement"
    REJECT_AFTER_TRADE = "an_order_that_has_traded_cannot_be_rejected"
    ABANDON_AFTER_RELEASE = "abandon_would_orphan_an_order_held_at_the_venue"


REFUSAL_REASONS: Final[Mapping[TransitionRefusal, str]] = MappingProxyType(
    {
        TransitionRefusal.TERMINAL_STATE: (
            "the order is finished. Terminal states have no outgoing transitions at all, "
            "because an order that could leave one would let the position implied by its "
            "fills and the position implied by its state disagree"
        ),
        TransitionRefusal.NOT_YET_RELEASED: (
            "the order is a draft that has never been released, so no venue has heard of it "
            "and nothing but release or abandon can apply to it"
        ),
        TransitionRefusal.ALREADY_RELEASED: (
            "release is what takes an order out of draft; applying it again would restart a "
            "lifecycle that is already in progress"
        ),
        TransitionRefusal.ALREADY_ACKNOWLEDGED: (
            "the venue has already acknowledged this order; a second acknowledgement is a "
            "replayed message rather than a state change"
        ),
        TransitionRefusal.VENUE_MESSAGE_BEFORE_ACKNOWLEDGEMENT: (
            "the venue reported a trade or a cancellation for an order it has not "
            "acknowledged. Our record and the venue's disagree, and recording the message "
            "would hide the disagreement that reconciliation exists to find"
        ),
        TransitionRefusal.CANCEL_WITHOUT_VENUE_ORDER: (
            "there is no acknowledged order to cancel: nothing has been accepted at the "
            "venue that a cancel request could name"
        ),
        TransitionRefusal.NO_CANCEL_IN_FLIGHT: (
            "a cancel confirmation or rejection resolves a cancel request, and no cancel "
            "request is outstanding on this order"
        ),
        TransitionRefusal.CANCEL_ALREADY_IN_FLIGHT: (
            "a cancel request is already outstanding; a second one produces two resolutions "
            "for one request and no way to tell which resolution belongs to which"
        ),
        TransitionRefusal.VENUE_CANCEL_IS_THE_CONFIRMATION: (
            "the venue cancelling an order whose cancel we requested *is* the confirmation; "
            "record cancel_confirmed so the request and its resolution stay paired"
        ),
        TransitionRefusal.REJECT_AFTER_ACKNOWLEDGEMENT: (
            "rejection is the alternative to acknowledgement, not a sequel to it. A venue "
            "that acknowledged an order and later rejects it is describing a different "
            "order than the one we hold"
        ),
        TransitionRefusal.REJECT_AFTER_TRADE: (
            "shares have already changed hands on this order. A rejection would erase a "
            "position that exists, which is the most expensive form of the reconciliation "
            "bug this machine prevents"
        ),
        TransitionRefusal.ABANDON_AFTER_RELEASE: (
            "abandon discards a draft nobody has seen. Applying it after release would drop "
            "our record of an order the venue still holds"
        ),
    }
)
"""Prose for each :class:`TransitionRefusal`, quoted in the raised error."""


INITIAL_STATE: Final = OrderState.DRAFT
"""Every order begins here. :func:`replay` folds from this state."""

TERMINAL_STATES: Final[frozenset[OrderState]] = frozenset(
    {
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    }
)
"""States with no outgoing transitions. The set is asserted against
:data:`TRANSITIONS` at import, so the two cannot drift apart."""

FILL_EVENTS: Final[frozenset[OrderEvent]] = frozenset(
    {OrderEvent.PARTIAL_FILL, OrderEvent.FILL_COMPLETE}
)
"""The two events that carry a traded quantity and price."""


TRANSITIONS: Final[Mapping[tuple[OrderState, OrderEvent], OrderState]] = MappingProxyType(
    {
        # A draft is ours alone: it can be released, or discarded before anyone
        # else has heard of it.
        (OrderState.DRAFT, OrderEvent.RELEASE): OrderState.PENDING_NEW,
        (OrderState.DRAFT, OrderEvent.ABANDON): OrderState.CANCELLED,
        # Released, awaiting the venue's answer. The venue either takes it,
        # refuses it, or the session ends under it.
        (OrderState.PENDING_NEW, OrderEvent.ACKNOWLEDGE): OrderState.ACKNOWLEDGED,
        (OrderState.PENDING_NEW, OrderEvent.REJECT): OrderState.REJECTED,
        (OrderState.PENDING_NEW, OrderEvent.EXPIRE): OrderState.EXPIRED,
        # Working, nothing traded.
        (OrderState.ACKNOWLEDGED, OrderEvent.PARTIAL_FILL): OrderState.PARTIALLY_FILLED,
        (OrderState.ACKNOWLEDGED, OrderEvent.FILL_COMPLETE): OrderState.FILLED,
        (OrderState.ACKNOWLEDGED, OrderEvent.REQUEST_CANCEL): OrderState.PENDING_CANCEL,
        (OrderState.ACKNOWLEDGED, OrderEvent.EXPIRE): OrderState.EXPIRED,
        (OrderState.ACKNOWLEDGED, OrderEvent.VENUE_CANCEL): OrderState.CANCELLED,
        # Working, some quantity traded. Partial fills accumulate in place.
        (OrderState.PARTIALLY_FILLED, OrderEvent.PARTIAL_FILL): OrderState.PARTIALLY_FILLED,
        (OrderState.PARTIALLY_FILLED, OrderEvent.FILL_COMPLETE): OrderState.FILLED,
        (
            OrderState.PARTIALLY_FILLED,
            OrderEvent.REQUEST_CANCEL,
        ): OrderState.PENDING_CANCEL_PARTIAL,
        (OrderState.PARTIALLY_FILLED, OrderEvent.EXPIRE): OrderState.EXPIRED,
        (OrderState.PARTIALLY_FILLED, OrderEvent.VENUE_CANCEL): OrderState.CANCELLED,
        # Cancel in flight, nothing traded yet. A fill can still race the cancel
        # — that race is the reason these edges exist rather than being treated
        # as impossible.
        (OrderState.PENDING_CANCEL, OrderEvent.CANCEL_CONFIRMED): OrderState.CANCELLED,
        (OrderState.PENDING_CANCEL, OrderEvent.CANCEL_REJECTED): OrderState.ACKNOWLEDGED,
        (OrderState.PENDING_CANCEL, OrderEvent.PARTIAL_FILL): OrderState.PENDING_CANCEL_PARTIAL,
        (OrderState.PENDING_CANCEL, OrderEvent.FILL_COMPLETE): OrderState.FILLED,
        (OrderState.PENDING_CANCEL, OrderEvent.EXPIRE): OrderState.EXPIRED,
        # Cancel in flight on an order that has traded. Identical shape; the
        # cancel rejection returns to PARTIALLY_FILLED rather than ACKNOWLEDGED,
        # which is the whole reason the two states are distinct.
        (OrderState.PENDING_CANCEL_PARTIAL, OrderEvent.CANCEL_CONFIRMED): OrderState.CANCELLED,
        (
            OrderState.PENDING_CANCEL_PARTIAL,
            OrderEvent.CANCEL_REJECTED,
        ): OrderState.PARTIALLY_FILLED,
        (
            OrderState.PENDING_CANCEL_PARTIAL,
            OrderEvent.PARTIAL_FILL,
        ): OrderState.PENDING_CANCEL_PARTIAL,
        (OrderState.PENDING_CANCEL_PARTIAL, OrderEvent.FILL_COMPLETE): OrderState.FILLED,
        (OrderState.PENDING_CANCEL_PARTIAL, OrderEvent.EXPIRE): OrderState.EXPIRED,
    }
)
"""The legal transitions, and the only ones. 25 pairs of the 110 possible."""


_RELEASED_UNACKNOWLEDGED: Final[frozenset[OrderState]] = frozenset({OrderState.PENDING_NEW})
_TRADED_STATES: Final[frozenset[OrderState]] = frozenset(
    {OrderState.PARTIALLY_FILLED, OrderState.PENDING_CANCEL_PARTIAL}
)
_CANCEL_IN_FLIGHT: Final[frozenset[OrderState]] = frozenset(
    {OrderState.PENDING_CANCEL, OrderState.PENDING_CANCEL_PARTIAL}
)
_CANCEL_RESOLUTIONS: Final[frozenset[OrderEvent]] = frozenset(
    {OrderEvent.CANCEL_CONFIRMED, OrderEvent.CANCEL_REJECTED}
)
_VENUE_ACTIVITY: Final[frozenset[OrderEvent]] = frozenset(
    {OrderEvent.PARTIAL_FILL, OrderEvent.FILL_COMPLETE, OrderEvent.VENUE_CANCEL}
)


def _refusal(state: OrderState, event: OrderEvent) -> TransitionRefusal | None:
    """Classify one illegal ``(state, event)`` pair, or return ``None``.

    Membership tests only, never identity narrowing: the classification must
    stay total over a set the type checker is not allowed to prove exhausted,
    because :func:`_build_illegal_transitions` relies on ``None`` meaning
    "unclassified" and raising at import time.

    Args:
        state: the state the order is in.
        event: the event with no transition from it.

    Returns:
        The named refusal, or ``None`` if no rule classifies the pair.
    """
    if state in TERMINAL_STATES:
        return TransitionRefusal.TERMINAL_STATE
    if state in {OrderState.DRAFT}:
        return TransitionRefusal.NOT_YET_RELEASED
    if event in {OrderEvent.RELEASE}:
        return TransitionRefusal.ALREADY_RELEASED
    if event in {OrderEvent.ABANDON}:
        return TransitionRefusal.ABANDON_AFTER_RELEASE
    if event in {OrderEvent.ACKNOWLEDGE}:
        return TransitionRefusal.ALREADY_ACKNOWLEDGED
    if event in {OrderEvent.REJECT}:
        if state in _TRADED_STATES:
            return TransitionRefusal.REJECT_AFTER_TRADE
        return TransitionRefusal.REJECT_AFTER_ACKNOWLEDGEMENT
    if state in _RELEASED_UNACKNOWLEDGED:
        if event in _VENUE_ACTIVITY:
            return TransitionRefusal.VENUE_MESSAGE_BEFORE_ACKNOWLEDGEMENT
        if event in {OrderEvent.REQUEST_CANCEL}:
            return TransitionRefusal.CANCEL_WITHOUT_VENUE_ORDER
        if event in _CANCEL_RESOLUTIONS:
            return TransitionRefusal.NO_CANCEL_IN_FLIGHT
        return None
    if state in _CANCEL_IN_FLIGHT:
        if event in {OrderEvent.REQUEST_CANCEL}:
            return TransitionRefusal.CANCEL_ALREADY_IN_FLIGHT
        if event in {OrderEvent.VENUE_CANCEL}:
            return TransitionRefusal.VENUE_CANCEL_IS_THE_CONFIRMATION
        return None
    if event in _CANCEL_RESOLUTIONS:
        return TransitionRefusal.NO_CANCEL_IN_FLIGHT
    return None


def _build_illegal_transitions() -> Mapping[tuple[OrderState, OrderEvent], TransitionRefusal]:
    """Classify every ``(state, event)`` pair absent from :data:`TRANSITIONS`.

    Returns:
        A read-only mapping from illegal pair to its named refusal.

    Raises:
        ExecutionError: if any illegal pair is unclassified. Raised at import,
            so a state or event added without a refusal rule breaks the build
            rather than acquiring a silent default.
    """
    classified: dict[tuple[OrderState, OrderEvent], TransitionRefusal] = {}
    for state in OrderState:
        for event in OrderEvent:
            if (state, event) in TRANSITIONS:
                continue
            refusal = _refusal(state, event)
            if refusal is None:
                msg = (
                    f"({state.value}, {event.value}) has no legal transition and no refusal "
                    f"rule classifies it. Every illegal pair must be named: an unclassified "
                    f"pair is a transition nobody decided about"
                )
                raise ExecutionError(msg)
            classified[state, event] = refusal
    return MappingProxyType(classified)


ILLEGAL_TRANSITIONS: Final[Mapping[tuple[OrderState, OrderEvent], TransitionRefusal]] = (
    _build_illegal_transitions()
)
"""Every refused ``(state, event)`` pair with the named reason it is refused."""


def _verify_table_consistency() -> None:
    """Check the invariants relating the legal table, its complement and terminality.

    Raises:
        ExecutionError: if the two tables overlap or fail to cover every pair,
            or if a state in :data:`TERMINAL_STATES` has an outgoing transition,
            or if a state outside it has none.
    """
    every_pair = {(state, event) for state in OrderState for event in OrderEvent}
    legal = set(TRANSITIONS)
    illegal = set(ILLEGAL_TRANSITIONS)
    if legal & illegal:
        msg = f"pairs are both legal and illegal: {sorted(legal & illegal)}"
        raise ExecutionError(msg)
    if legal | illegal != every_pair:
        unclassified = sorted(every_pair - legal - illegal)
        msg = f"pairs classified as neither legal nor illegal: {unclassified}"
        raise ExecutionError(msg)
    with_outgoing = {state for state, _ in legal}
    if with_outgoing & TERMINAL_STATES:
        msg = (
            f"terminal states have outgoing transitions: "
            f"{sorted(with_outgoing & TERMINAL_STATES)}. A terminal state with an exit is "
            f"the FILLED -> PENDING_NEW bug"
        )
        raise ExecutionError(msg)
    stranded = set(OrderState) - with_outgoing - TERMINAL_STATES
    if stranded:
        msg = f"non-terminal states with no outgoing transition: {sorted(stranded)}"
        raise ExecutionError(msg)


_verify_table_consistency()


def legal_events(state: OrderState) -> frozenset[OrderEvent]:
    """Return the events that have a transition from ``state``.

    Args:
        state: the state to look up.

    Returns:
        The events accepted in ``state``; empty for every terminal state.
    """
    return frozenset(event for existing, event in TRANSITIONS if existing is state)


def reachable_states() -> frozenset[OrderState]:
    """Return the states reachable from :data:`INITIAL_STATE` by legal transitions.

    Breadth-first over :data:`TRANSITIONS`. Used by the property suite to prove
    that every declared state is reachable — a state nothing can reach is either
    a typo or a lifecycle nobody implemented, and both look identical in a
    blotter that never shows the state.

    Returns:
        The reachable states, always including :data:`INITIAL_STATE`.
    """
    seen = {INITIAL_STATE}
    frontier = [INITIAL_STATE]
    while frontier:
        state = frontier.pop()
        for event in legal_events(state):
            target = TRANSITIONS[state, event]
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return frozenset(seen)


def apply_event(state: OrderState, event: OrderEvent) -> OrderState:
    """Return the state ``event`` leads to from ``state``.

    Args:
        state: the order's current state.
        event: the event to apply.

    Returns:
        The resulting state.

    Raises:
        TerminalOrderError: if ``state`` is terminal. A subclass of
            ``IllegalTransitionError``, separated so a caller can tell "this
            order is finished" (usually a late or duplicated venue message) from
            "this event is wrong for this state" (a genuine disagreement).
        IllegalTransitionError: if the pair has no transition. The message
            quotes the named :class:`TransitionRefusal`.
    """
    target = TRANSITIONS.get((state, event))
    if target is not None:
        return target
    refusal = ILLEGAL_TRANSITIONS[state, event]
    if state in TERMINAL_STATES:
        raise TerminalOrderError(
            state=state, event=event, refusal=refusal, reason=REFUSAL_REASONS[refusal]
        )
    raise IllegalTransitionError(
        state=state, event=event, refusal=refusal, reason=REFUSAL_REASONS[refusal]
    )


def fill_event(
    *,
    ordered_quantity_shares: int,
    filled_before_shares: int,
    fill_quantity_shares: int,
) -> OrderEvent:
    """Return which fill event a reported quantity constitutes.

    The choice between ``PARTIAL_FILL`` and ``FILL_COMPLETE`` is arithmetic, not
    judgement: a fill that takes the cumulative quantity to the ordered quantity
    completes the order, and any smaller one does not. Callers pass their own
    event to :func:`backend.execution.store.append_transition`, which checks it
    against this function and refuses a disagreement — one of the two is wrong
    and neither may be trusted.

    Units: every argument is in **whole shares**.

    Args:
        ordered_quantity_shares: the order's total quantity, strictly positive.
        filled_before_shares: cumulative quantity filled before this report,
            non-negative and not exceeding ``ordered_quantity_shares``.
        fill_quantity_shares: the quantity this report trades, strictly
            positive.

    Returns:
        ``OrderEvent.FILL_COMPLETE`` when the cumulative quantity reaches the
        ordered quantity exactly, ``OrderEvent.PARTIAL_FILL`` otherwise.

    Raises:
        FillAccountingError: if any quantity is out of range, or if the fill
            would take the cumulative quantity past the ordered quantity.
            Refused rather than clamped: clamping silently discards shares the
            venue says it traded, leaving the position and the blotter
            permanently out of step.
    """
    if ordered_quantity_shares <= 0:
        msg = f"ordered_quantity_shares must be strictly positive, got {ordered_quantity_shares}"
        raise FillAccountingError(msg)
    if filled_before_shares < 0:
        msg = f"filled_before_shares must be non-negative, got {filled_before_shares}"
        raise FillAccountingError(msg)
    if filled_before_shares > ordered_quantity_shares:
        msg = (
            f"filled_before_shares={filled_before_shares} already exceeds "
            f"ordered_quantity_shares={ordered_quantity_shares}"
        )
        raise FillAccountingError(msg)
    if fill_quantity_shares <= 0:
        msg = (
            f"fill_quantity_shares must be strictly positive, got {fill_quantity_shares}. "
            f"A zero-quantity fill is not a trade, and a negative one is not a fill"
        )
        raise FillAccountingError(msg)
    cumulative = filled_before_shares + fill_quantity_shares
    if cumulative > ordered_quantity_shares:
        msg = (
            f"fill of {fill_quantity_shares} shares would take the cumulative filled "
            f"quantity to {cumulative}, past the ordered {ordered_quantity_shares}. "
            f"Refused rather than clamped: clamping discards shares the venue says it "
            f"traded and leaves the position and the blotter permanently out of step"
        )
        raise FillAccountingError(msg)
    if cumulative == ordered_quantity_shares:
        return OrderEvent.FILL_COMPLETE
    return OrderEvent.PARTIAL_FILL


@dataclass(frozen=True, slots=True)
class Transition:
    """One recorded step of an order's history.

    The in-memory form of a ``execution_order_transition`` row, and the unit
    :func:`replay` folds. Immutable, matching the append-only table it mirrors.

    Attributes:
        sequence_number: position in this order's history, ``1``-based and
            gapless. The ordering key — never a timestamp.
        from_state: the state the order was in.
        event: the event applied.
        to_state: the resulting state.
        fill_quantity_shares: shares traded by this event (**whole shares**),
            or ``None`` for every event that is not a fill. Absent rather than
            zero: a zero would be a quantity nobody reported (D-030).
        filled_quantity_after_shares: cumulative shares filled after this event
            (**whole shares**). Stored rather than recomputed so a truncated
            history is detectable instead of merely shorter.
    """

    sequence_number: int
    from_state: OrderState
    event: OrderEvent
    to_state: OrderState
    fill_quantity_shares: int | None
    filled_quantity_after_shares: int


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """The state an order's recorded history folds to.

    Attributes:
        state: the order's current state.
        filled_quantity_shares: cumulative shares filled (**whole shares**).
        sequence_number: the last sequence number in the history, ``0`` for an
            order with no recorded transitions.
    """

    state: OrderState
    filled_quantity_shares: int
    sequence_number: int


def _require_chain(condition: bool, message: str) -> None:
    """Raise :class:`TransitionChainError` when ``condition`` is false.

    Args:
        condition: the invariant that must hold.
        message: what was violated, quoted verbatim in the error.

    Raises:
        TransitionChainError: when ``condition`` is false.
    """
    if not condition:
        raise TransitionChainError(message)


def replay(
    transitions: Sequence[Transition],
    *,
    ordered_quantity_shares: int,
) -> ReplayResult:
    """Fold a recorded history into the order's current state.

    Every consistency claim the store makes about a history is checked here,
    because this is the function every reader goes through: sequence numbers are
    exactly ``1..n`` with no gaps and no reordering, each step starts where the
    previous one ended, the first starts at :data:`INITIAL_STATE`, every step is
    a legal transition, fill payloads are present exactly on fill events, and
    the cumulative quantities are the running sum of the per-event quantities
    and never exceed the ordered quantity.

    These are assertions about this system's own bookkeeping rather than about a
    venue's behaviour, which is why they raise instead of returning a flag: an
    order whose history cannot reconcile with itself has no honest state to
    report, and reporting one anyway is how a phantom position is born.

    Args:
        transitions: the order's transitions in sequence order. Empty means a
            draft that has never had an event applied.
        ordered_quantity_shares: the order's quantity in **whole shares**.

    Returns:
        A :class:`ReplayResult`.

    Raises:
        TransitionChainError: if the history is not internally consistent.
        FillAccountingError: never raised here; fill arithmetic violations in a
            *stored* history are chain errors, since the store refused them on
            the way in and their presence means the row was written by
            something else.
    """
    _require_chain(
        ordered_quantity_shares > 0,
        f"ordered_quantity_shares must be strictly positive, got {ordered_quantity_shares}",
    )
    state = INITIAL_STATE
    filled = 0
    for index, transition in enumerate(transitions, start=1):
        _require_chain(
            transition.sequence_number == index,
            f"transition at position {index} carries sequence_number "
            f"{transition.sequence_number}; a history's sequence numbers are 1..n with no "
            f"gaps, so a mismatch means a row is missing, duplicated or out of order",
        )
        _require_chain(
            transition.from_state is state,
            f"transition {index} starts in {transition.from_state.value!r} but the history "
            f"had reached {state.value!r}; the chain does not connect",
        )
        expected = TRANSITIONS.get((transition.from_state, transition.event))
        if expected is None:
            msg = (
                f"transition {index} records {transition.event.value!r} from "
                f"{transition.from_state.value!r}, which has no legal transition: "
                f"{REFUSAL_REASONS[ILLEGAL_TRANSITIONS[transition.from_state, transition.event]]}"
            )
            raise TransitionChainError(msg)
        _require_chain(
            transition.to_state is expected,
            f"transition {index} records {transition.from_state.value!r} + "
            f"{transition.event.value!r} -> {transition.to_state.value!r}, but that pair "
            f"leads to {expected.value!r}",
        )
        is_fill = transition.event in FILL_EVENTS
        _require_chain(
            is_fill == (transition.fill_quantity_shares is not None),
            f"transition {index} records event {transition.event.value!r} with "
            f"fill_quantity_shares={transition.fill_quantity_shares}; a fill quantity is "
            f"present exactly on fill events and absent on every other",
        )
        traded = transition.fill_quantity_shares or 0
        if is_fill:
            _require_chain(
                traded > 0,
                f"transition {index} is a fill of {traded} shares; a fill trades a strictly "
                f"positive quantity",
            )
        filled += traded
        _require_chain(
            transition.filled_quantity_after_shares == filled,
            f"transition {index} records filled_quantity_after_shares="
            f"{transition.filled_quantity_after_shares}, but the running sum of the "
            f"per-event quantities is {filled}",
        )
        _require_chain(
            filled <= ordered_quantity_shares,
            f"transition {index} takes the cumulative filled quantity to {filled}, past the "
            f"ordered {ordered_quantity_shares}",
        )
        state = transition.to_state
    _require_chain(
        state is not OrderState.FILLED or filled == ordered_quantity_shares,
        f"the history ends in {OrderState.FILLED.value!r} with {filled} of "
        f"{ordered_quantity_shares} shares filled; a filled order has traded its whole "
        f"quantity, and a shortfall here is a position the blotter would not show",
    )
    return ReplayResult(
        state=state,
        filled_quantity_shares=filled,
        sequence_number=len(transitions),
    )
