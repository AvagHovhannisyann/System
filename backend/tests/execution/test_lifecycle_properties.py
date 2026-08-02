"""Property tests: no event sequence reaches an illegal state; every state is reachable.

The unit tests assert the shape of the transition table. These assert that the
*machine* behaves as the table says, over event sequences nobody wrote down —
which is the only way to catch a guard that is right in the cases someone thought
of.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.execution.errors import (
    FillAccountingError,
    IllegalTransitionError,
    TerminalOrderError,
)
from backend.execution.lifecycle import (
    FILL_EVENTS,
    ILLEGAL_TRANSITIONS,
    INITIAL_STATE,
    TERMINAL_STATES,
    TRANSITIONS,
    OrderEvent,
    OrderState,
    Transition,
    apply_event,
    fill_event,
    legal_events,
    reachable_states,
    replay,
)

events = st.sampled_from(list(OrderEvent))
states = st.sampled_from(list(OrderState))
event_sequences = st.lists(events, min_size=0, max_size=40)


@settings(max_examples=500)
@given(sequence=event_sequences)
def test_no_event_sequence_ever_reaches_a_state_outside_the_declared_set(
    sequence: list[OrderEvent],
) -> None:
    # Applying arbitrary events, keeping the state when one is refused: the
    # state must always be a declared member, and every step that *was* applied
    # must have been in the legal table.
    state = INITIAL_STATE
    for event in sequence:
        try:
            target = apply_event(state, event)
        except IllegalTransitionError:
            assert (state, event) in ILLEGAL_TRANSITIONS
            continue
        assert TRANSITIONS[state, event] is target
        state = target
        assert state in set(OrderState)


@settings(max_examples=500)
@given(sequence=event_sequences)
def test_a_terminal_state_is_absorbing_for_every_subsequent_event(
    sequence: list[OrderEvent],
) -> None:
    state = INITIAL_STATE
    finished = False
    for event in sequence:
        if finished:
            # Once terminal, *every* event must be refused as terminal — not
            # merely the ones that would obviously be wrong.
            try:
                apply_event(state, event)
            except TerminalOrderError:
                continue
            raise AssertionError(f"{state} accepted {event} after becoming terminal")
        try:
            state = apply_event(state, event)
        except IllegalTransitionError:
            continue
        finished = state in TERMINAL_STATES


@settings(max_examples=300)
@given(sequence=event_sequences)
def test_the_filled_state_is_only_ever_entered_with_a_completing_fill(
    sequence: list[OrderEvent],
) -> None:
    state = INITIAL_STATE
    for event in sequence:
        try:
            target = apply_event(state, event)
        except IllegalTransitionError:
            continue
        if target is OrderState.FILLED:
            assert event is OrderEvent.FILL_COMPLETE
        state = target


def _outcome(state: OrderState, event: OrderEvent) -> OrderState | IllegalTransitionError:
    """Return the resulting state, or the refusal, without raising through the test."""
    try:
        return apply_event(state, event)
    except IllegalTransitionError as refused:
        return refused


@given(state=states, event=events)
def test_every_pair_is_either_applied_or_refused_with_a_named_reason(
    state: OrderState, event: OrderEvent
) -> None:
    outcome = _outcome(state, event)
    if isinstance(outcome, IllegalTransitionError):
        assert (state, event) in ILLEGAL_TRANSITIONS
        assert outcome.refusal == ILLEGAL_TRANSITIONS[state, event]
        assert isinstance(outcome, TerminalOrderError) == (state in TERMINAL_STATES)
        return
    assert TRANSITIONS[state, event] is outcome


def test_every_legal_state_is_reachable_by_some_event_sequence() -> None:
    # reachable_states() is a breadth-first search; this walks it back into an
    # actual sequence per state, so "reachable" means "there is a path an order
    # can really take", not "there is an edge in a dict".
    paths: dict[OrderState, list[OrderEvent]] = {INITIAL_STATE: []}
    frontier = [INITIAL_STATE]
    while frontier:
        state = frontier.pop(0)
        for event in sorted(legal_events(state)):
            target = TRANSITIONS[state, event]
            if target not in paths:
                paths[target] = [*paths[state], event]
                frontier.append(target)
    assert set(paths) == set(OrderState)
    for state, path in paths.items():
        walked = INITIAL_STATE
        for event in path:
            walked = apply_event(walked, event)
        assert walked is state
    assert reachable_states() == set(paths)


@settings(max_examples=400)
@given(
    ordered=st.integers(min_value=1, max_value=10_000),
    quantities=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=12),
)
def test_fill_event_never_admits_a_cumulative_quantity_past_the_order(
    ordered: int, quantities: list[int]
) -> None:
    filled = 0
    for quantity in quantities:
        try:
            event = fill_event(
                ordered_quantity_shares=ordered,
                filled_before_shares=filled,
                fill_quantity_shares=quantity,
            )
        except FillAccountingError:
            # Either the order is already complete or this fill would overfill.
            assert filled == ordered or filled + quantity > ordered
            continue
        filled += quantity
        assert filled <= ordered
        assert (event is OrderEvent.FILL_COMPLETE) == (filled == ordered)


@settings(max_examples=400)
@given(
    ordered=st.integers(min_value=1, max_value=1_000),
    sequence=event_sequences,
    quantities=st.lists(st.integers(min_value=1, max_value=1_000), min_size=0, max_size=40),
)
def test_replay_reproduces_the_state_the_machine_walked_to(
    ordered: int, sequence: list[OrderEvent], quantities: list[int]
) -> None:
    # Build a history the way the store does — only legal events, only
    # arithmetically consistent fills — then prove replay lands where the walk
    # did. If replay and apply_event can disagree, a blotter and a reconciler
    # can disagree.
    state = INITIAL_STATE
    filled = 0
    history: list[Transition] = []
    supply = iter(quantities)
    for event in sequence:
        if (state, event) not in TRANSITIONS:
            continue
        traded: int | None = None
        if event in FILL_EVENTS:
            quantity = next(supply, None)
            if quantity is None:
                break
            try:
                implied = fill_event(
                    ordered_quantity_shares=ordered,
                    filled_before_shares=filled,
                    fill_quantity_shares=quantity,
                )
            except FillAccountingError:
                continue
            if implied is not event:
                continue
            traded = quantity
        target = apply_event(state, event)
        filled += traded or 0
        history.append(
            Transition(
                sequence_number=len(history) + 1,
                from_state=state,
                event=event,
                to_state=target,
                fill_quantity_shares=traded,
                filled_quantity_after_shares=filled,
            )
        )
        state = target
        if state in TERMINAL_STATES:
            break
    result = replay(history, ordered_quantity_shares=ordered)
    assert result.state is state
    assert result.filled_quantity_shares == filled
    assert result.sequence_number == len(history)
