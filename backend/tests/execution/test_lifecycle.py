"""The transition table, its named complement, and the fill arithmetic."""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.execution.errors import (
    FillAccountingError,
    IllegalTransitionError,
    TerminalOrderError,
    TransitionChainError,
)
from backend.execution.lifecycle import (
    FILL_EVENTS,
    ILLEGAL_TRANSITIONS,
    INITIAL_STATE,
    REFUSAL_REASONS,
    TERMINAL_STATES,
    TRANSITIONS,
    OrderEvent,
    OrderState,
    Transition,
    TransitionRefusal,
    apply_event,
    fill_event,
    legal_events,
    reachable_states,
    replay,
)

TERMINAL = (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED, OrderState.EXPIRED)


def test_the_table_is_a_partial_function_of_the_declared_size() -> None:
    assert len(OrderState) == 10
    assert len(OrderEvent) == 11
    assert len(TRANSITIONS) == 25
    assert len(ILLEGAL_TRANSITIONS) == 85
    assert len(TRANSITIONS) + len(ILLEGAL_TRANSITIONS) == len(OrderState) * len(OrderEvent)


def test_legal_and_illegal_partition_every_pair() -> None:
    every = {(state, event) for state in OrderState for event in OrderEvent}
    assert set(TRANSITIONS).isdisjoint(ILLEGAL_TRANSITIONS)
    assert set(TRANSITIONS) | set(ILLEGAL_TRANSITIONS) == every


def test_every_illegal_pair_carries_a_named_refusal_with_prose() -> None:
    # "Named explicitly" means each of the 85 says *why*, not that the table has
    # a hole where a transition would be.
    for pair, refusal in ILLEGAL_TRANSITIONS.items():
        assert isinstance(refusal, TransitionRefusal), pair
        assert REFUSAL_REASONS[refusal].strip() != "", pair


def test_every_refusal_reason_is_used() -> None:
    # A reason nobody reaches is a rule that was rewritten and left behind.
    assert set(ILLEGAL_TRANSITIONS.values()) == set(TransitionRefusal)
    assert set(REFUSAL_REASONS) == set(TransitionRefusal)


@pytest.mark.parametrize("state", TERMINAL)
def test_terminal_states_have_no_outgoing_transitions(state: OrderState) -> None:
    assert state in TERMINAL_STATES
    assert legal_events(state) == frozenset()
    for event in OrderEvent:
        assert ILLEGAL_TRANSITIONS[state, event] is TransitionRefusal.TERMINAL_STATE


def test_filled_cannot_return_to_pending_new() -> None:
    # The bug this machine exists to prevent, named as its own test so a
    # regression reads as what it is rather than as an arithmetic failure.
    assert (OrderState.FILLED, OrderEvent.RELEASE) not in TRANSITIONS
    assert OrderState.PENDING_NEW not in {
        target for (state, _), target in TRANSITIONS.items() if state is OrderState.FILLED
    }
    with pytest.raises(TerminalOrderError) as raised:
        apply_event(OrderState.FILLED, OrderEvent.RELEASE)
    assert raised.value.refusal == TransitionRefusal.TERMINAL_STATE
    assert "position implied by its fills" in str(raised.value)


def test_terminal_order_error_is_an_illegal_transition_error() -> None:
    # The specialisation must stay catchable as the general case: a caller that
    # only cares "this event was refused" writes one except clause.
    assert issubclass(TerminalOrderError, IllegalTransitionError)


@pytest.mark.parametrize(
    ("state", "event", "refusal"),
    [
        (OrderState.DRAFT, OrderEvent.ACKNOWLEDGE, TransitionRefusal.NOT_YET_RELEASED),
        (OrderState.DRAFT, OrderEvent.PARTIAL_FILL, TransitionRefusal.NOT_YET_RELEASED),
        (
            OrderState.PENDING_NEW,
            OrderEvent.PARTIAL_FILL,
            TransitionRefusal.VENUE_MESSAGE_BEFORE_ACKNOWLEDGEMENT,
        ),
        (
            OrderState.PENDING_NEW,
            OrderEvent.REQUEST_CANCEL,
            TransitionRefusal.CANCEL_WITHOUT_VENUE_ORDER,
        ),
        (
            OrderState.PENDING_NEW,
            OrderEvent.CANCEL_CONFIRMED,
            TransitionRefusal.NO_CANCEL_IN_FLIGHT,
        ),
        (OrderState.PENDING_NEW, OrderEvent.ABANDON, TransitionRefusal.ABANDON_AFTER_RELEASE),
        (
            OrderState.ACKNOWLEDGED,
            OrderEvent.REJECT,
            TransitionRefusal.REJECT_AFTER_ACKNOWLEDGEMENT,
        ),
        (OrderState.ACKNOWLEDGED, OrderEvent.RELEASE, TransitionRefusal.ALREADY_RELEASED),
        (
            OrderState.ACKNOWLEDGED,
            OrderEvent.ACKNOWLEDGE,
            TransitionRefusal.ALREADY_ACKNOWLEDGED,
        ),
        (OrderState.PARTIALLY_FILLED, OrderEvent.REJECT, TransitionRefusal.REJECT_AFTER_TRADE),
        (
            OrderState.PENDING_CANCEL,
            OrderEvent.REQUEST_CANCEL,
            TransitionRefusal.CANCEL_ALREADY_IN_FLIGHT,
        ),
        (
            OrderState.PENDING_CANCEL,
            OrderEvent.VENUE_CANCEL,
            TransitionRefusal.VENUE_CANCEL_IS_THE_CONFIRMATION,
        ),
        (
            OrderState.PENDING_CANCEL_PARTIAL,
            OrderEvent.REJECT,
            TransitionRefusal.REJECT_AFTER_TRADE,
        ),
    ],
)
def test_named_illegal_transitions(
    state: OrderState, event: OrderEvent, refusal: TransitionRefusal
) -> None:
    # Spot-checks of the classification, one per rule, written out so the
    # *reasons* are reviewable rather than only the counts.
    assert (state, event) not in TRANSITIONS
    assert ILLEGAL_TRANSITIONS[state, event] is refusal
    with pytest.raises(IllegalTransitionError) as raised:
        apply_event(state, event)
    assert raised.value.refusal == refusal
    assert REFUSAL_REASONS[refusal] in str(raised.value)


def test_a_cancel_rejection_returns_to_where_the_order_actually_was() -> None:
    # The reason the two pending-cancel states exist: an order that had traded
    # must not come back as one that had not.
    assert TRANSITIONS[OrderState.PENDING_CANCEL, OrderEvent.CANCEL_REJECTED] is (
        OrderState.ACKNOWLEDGED
    )
    assert TRANSITIONS[OrderState.PENDING_CANCEL_PARTIAL, OrderEvent.CANCEL_REJECTED] is (
        OrderState.PARTIALLY_FILLED
    )


def test_every_declared_state_is_reachable_from_draft() -> None:
    assert reachable_states() == frozenset(OrderState)
    assert INITIAL_STATE is OrderState.DRAFT


def test_every_non_terminal_state_has_a_way_out() -> None:
    for state in OrderState:
        if state in TERMINAL_STATES:
            continue
        assert legal_events(state), state


def test_fill_events_are_exactly_the_two_that_carry_a_quantity() -> None:
    assert FILL_EVENTS == frozenset({OrderEvent.PARTIAL_FILL, OrderEvent.FILL_COMPLETE})


def test_fill_event_is_decided_by_arithmetic() -> None:
    assert (
        fill_event(
            ordered_quantity_shares=100, filled_before_shares=0, fill_quantity_shares=40
        )
        is OrderEvent.PARTIAL_FILL
    )
    assert (
        fill_event(
            ordered_quantity_shares=100, filled_before_shares=60, fill_quantity_shares=40
        )
        is OrderEvent.FILL_COMPLETE
    )


def test_fill_event_refuses_an_overfill_rather_than_clamping() -> None:
    with pytest.raises(FillAccountingError) as raised:
        fill_event(ordered_quantity_shares=100, filled_before_shares=60, fill_quantity_shares=41)
    assert "past the ordered 100" in str(raised.value)
    assert "clamping" in str(raised.value)


@pytest.mark.parametrize(
    ("ordered", "before", "quantity"),
    [(0, 0, 1), (100, -1, 1), (100, 101, 1), (100, 0, 0), (100, 0, -5)],
)
def test_fill_event_refuses_quantities_outside_range(
    ordered: int, before: int, quantity: int
) -> None:
    with pytest.raises(FillAccountingError):
        fill_event(
            ordered_quantity_shares=ordered,
            filled_before_shares=before,
            fill_quantity_shares=quantity,
        )


def _chain(*steps: tuple[OrderEvent, int | None]) -> list[Transition]:
    """Build a well-formed history by applying ``steps`` from DRAFT."""
    state = INITIAL_STATE
    filled = 0
    history: list[Transition] = []
    for index, (event, quantity) in enumerate(steps, start=1):
        target = apply_event(state, event)
        filled += quantity or 0
        history.append(
            Transition(
                sequence_number=index,
                from_state=state,
                event=event,
                to_state=target,
                fill_quantity_shares=quantity,
                filled_quantity_after_shares=filled,
            )
        )
        state = target
    return history


def test_replay_folds_a_well_formed_history() -> None:
    history = _chain(
        (OrderEvent.RELEASE, None),
        (OrderEvent.ACKNOWLEDGE, None),
        (OrderEvent.PARTIAL_FILL, 40),
        (OrderEvent.FILL_COMPLETE, 60),
    )
    result = replay(history, ordered_quantity_shares=100)
    assert result.state is OrderState.FILLED
    assert result.filled_quantity_shares == 100
    assert result.sequence_number == 4


def test_replay_of_an_empty_history_is_a_draft() -> None:
    result = replay([], ordered_quantity_shares=100)
    assert result.state is INITIAL_STATE
    assert result.filled_quantity_shares == 0
    assert result.sequence_number == 0


def test_replay_refuses_a_gap_in_the_sequence() -> None:
    history = _chain((OrderEvent.RELEASE, None), (OrderEvent.ACKNOWLEDGE, None))
    broken = [history[0], replace(history[1], sequence_number=3)]
    with pytest.raises(TransitionChainError, match="1..n with no"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_a_chain_that_does_not_connect() -> None:
    history = _chain((OrderEvent.RELEASE, None), (OrderEvent.ACKNOWLEDGE, None))
    broken = [history[0], replace(history[1], from_state=OrderState.ACKNOWLEDGED)]
    with pytest.raises(TransitionChainError, match="does not connect"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_an_illegal_recorded_transition() -> None:
    broken = [
        Transition(
            sequence_number=1,
            from_state=OrderState.DRAFT,
            event=OrderEvent.FILL_COMPLETE,
            to_state=OrderState.FILLED,
            fill_quantity_shares=100,
            filled_quantity_after_shares=100,
        )
    ]
    with pytest.raises(TransitionChainError, match="no legal transition"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_a_wrong_target_state() -> None:
    broken = [
        Transition(
            sequence_number=1,
            from_state=OrderState.DRAFT,
            event=OrderEvent.RELEASE,
            to_state=OrderState.ACKNOWLEDGED,
            fill_quantity_shares=None,
            filled_quantity_after_shares=0,
        )
    ]
    with pytest.raises(TransitionChainError, match="leads to 'pending_new'"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_a_cumulative_quantity_that_is_not_the_running_sum() -> None:
    history = _chain(
        (OrderEvent.RELEASE, None),
        (OrderEvent.ACKNOWLEDGE, None),
        (OrderEvent.PARTIAL_FILL, 40),
    )
    broken = [*history[:2], replace(history[2], filled_quantity_after_shares=55)]
    with pytest.raises(TransitionChainError, match="running sum"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_a_fill_payload_on_a_non_fill_event() -> None:
    history = _chain((OrderEvent.RELEASE, None))
    broken = [replace(history[0], fill_quantity_shares=10)]
    with pytest.raises(TransitionChainError, match="present exactly on fill events"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_a_filled_order_that_did_not_trade_its_whole_quantity() -> None:
    broken = [
        Transition(
            sequence_number=1,
            from_state=OrderState.DRAFT,
            event=OrderEvent.RELEASE,
            to_state=OrderState.PENDING_NEW,
            fill_quantity_shares=None,
            filled_quantity_after_shares=0,
        ),
        Transition(
            sequence_number=2,
            from_state=OrderState.PENDING_NEW,
            event=OrderEvent.ACKNOWLEDGE,
            to_state=OrderState.ACKNOWLEDGED,
            fill_quantity_shares=None,
            filled_quantity_after_shares=0,
        ),
        Transition(
            sequence_number=3,
            from_state=OrderState.ACKNOWLEDGED,
            event=OrderEvent.FILL_COMPLETE,
            to_state=OrderState.FILLED,
            fill_quantity_shares=60,
            filled_quantity_after_shares=60,
        ),
    ]
    with pytest.raises(TransitionChainError, match="has traded its whole quantity"):
        replay(broken, ordered_quantity_shares=100)


def test_replay_refuses_an_overfilled_history() -> None:
    broken = [
        Transition(
            sequence_number=1,
            from_state=OrderState.DRAFT,
            event=OrderEvent.RELEASE,
            to_state=OrderState.PENDING_NEW,
            fill_quantity_shares=None,
            filled_quantity_after_shares=0,
        ),
        Transition(
            sequence_number=2,
            from_state=OrderState.PENDING_NEW,
            event=OrderEvent.ACKNOWLEDGE,
            to_state=OrderState.ACKNOWLEDGED,
            fill_quantity_shares=None,
            filled_quantity_after_shares=0,
        ),
        Transition(
            sequence_number=3,
            from_state=OrderState.ACKNOWLEDGED,
            event=OrderEvent.PARTIAL_FILL,
            to_state=OrderState.PARTIALLY_FILLED,
            fill_quantity_shares=150,
            filled_quantity_after_shares=150,
        ),
    ]
    with pytest.raises(TransitionChainError, match="past the ordered 100"):
        replay(broken, ordered_quantity_shares=100)
