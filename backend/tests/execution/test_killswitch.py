"""P11.5: the kill switch — four triggers, fail closed, and halted within the cycle.

The within-one-cycle proof
--------------------------

The directive's guarantee is a *timing* claim, so it is stated here as an
equality that a log can be checked against rather than as a duration nobody can
measure after the fact. For each of the four triggers,
:func:`~backend.execution.killswitch.run_cycle` is called **once** with an
observation carrying that condition, and three things are asserted together:

1. a halt row exists after that one call — zero further cycles elapsed;
2. the halt's ``cycle_id`` is *the observing cycle's own*, so the log itself shows
   that the halt and the observation belong to the same cycle;
3. the release gate refuses from that same cycle onward — the halt actually stops
   something rather than merely being recorded.

``test_the_halt_lands_in_the_observing_cycle_and_not_a_later_one`` makes the
"zero cycles elapsed" arithmetic explicit by running numbered cycles and
subtracting.

The fail-closed proof
---------------------

Every named way a measurement can fail to be a measurement gets its own test, and
a Hypothesis property sweeps the combinations: **any** observation with a missing
limit, a missing measurement, a non-finite number, an impossible number, a probe
that raises, or a verdict from another cycle halts. The mirror property — a
fully-measured, fully-inside-limits observation does *not* halt — is what stops
"halt on everything" from passing this file trivially.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.execution.errors import HaltStateUnavailableError, SystemHaltedError
from backend.execution.halt import HaltTrigger, clear_halt, open_halts
from backend.execution.killswitch import (
    CycleObservation,
    DataFreshnessObservation,
    DrawdownObservation,
    ManualHaltRequest,
    evaluate,
    guard_before_release,
    run_cycle,
)
from backend.tests.execution.control_fixtures import (
    CYCLE_ID,
    OBSERVED_AT,
    ControlRows,
    ControlSessionDouble,
    ExplodingDecimal,
    ExplodingResult,
    as_session,
    broken_result,
    clean_result,
    fresh_data,
    healthy_drawdown,
    internal_snapshot,
    observation,
    reported_snapshot,
    restart,
)
from backend.tests.execution.fixtures import make_stamp

if TYPE_CHECKING:
    from backend.execution.killswitch import CycleOutcome

BREACHING_DRAWDOWN = DrawdownObservation(
    equity_usd=Decimal("80000.00"),
    peak_equity_usd=Decimal("100000.00"),
    limit_fraction=Decimal("0.10"),
)
STALE = DataFreshnessObservation(age_seconds=Decimal("7200"), max_age_seconds=Decimal("3600"))
MANUAL = ManualHaltRequest(requested_by="risk-desk", reason="pausing before the open")

TRIGGERING_OBSERVATIONS: dict[HaltTrigger, dict[str, Any]] = {
    HaltTrigger.DRAWDOWN_BREACH: {"drawdown": BREACHING_DRAWDOWN},
    HaltTrigger.STALE_DATA: {"freshness": STALE},
    HaltTrigger.RECONCILIATION_MISMATCH: {"reconciliation": broken_result()},
    HaltTrigger.MANUAL: {"manual": MANUAL},
}
"""One minimal deviation from a safe observation per directive trigger."""


async def cycle(session: Any, **overrides: Any) -> CycleOutcome:  # noqa: ANN401 - the double
    """Run one cycle over an observation that is safe apart from ``overrides``."""
    return await run_cycle(session, observation(**overrides))


# ---------------------------------------------------------------------------
# The clean direction: a safe cycle trades.
# ---------------------------------------------------------------------------


async def test_a_fully_measured_cycle_inside_every_limit_does_not_halt() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    outcome = await cycle(session)
    assert outcome.decision.reasons == ()
    assert outcome.release_permitted
    assert not outcome.halted
    assert outcome.engaged_halt_ids == ()
    assert rows.halts == []
    await guard_before_release(session)


def test_a_safe_observation_produces_no_reasons() -> None:
    assert evaluate(observation()).should_halt is False


# ---------------------------------------------------------------------------
# The four triggers, each halting within the cycle that observed it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", list(TRIGGERING_OBSERVATIONS))
async def test_each_directive_trigger_halts_within_the_observing_cycle(
    trigger: HaltTrigger,
) -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    outcome = await cycle(session, **TRIGGERING_OBSERVATIONS[trigger])

    # 1. Halted after exactly one cycle.
    assert outcome.halted
    assert not outcome.release_permitted
    assert len(outcome.engaged_halt_ids) == 1
    assert [reason.trigger for reason in outcome.decision.reasons] == [trigger]

    # 2. The halt row names the observing cycle — checkable from the log alone.
    row = rows.halt(outcome.engaged_halt_ids[0])
    assert row["halt_trigger"] == trigger.value
    assert row["cycle_id"] == CYCLE_ID == outcome.cycle_id

    # 3. The release gate refuses from this same cycle onward.
    with pytest.raises(SystemHaltedError):
        await guard_before_release(session)


async def test_the_halt_lands_in_the_observing_cycle_and_not_a_later_one() -> None:
    # "Within one cycle", as arithmetic: run numbered cycles, put the condition
    # into cycle 3, and assert the halt carries cycle 3's id — zero cycles
    # elapsed between observing and halting.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    triggering_index = 3
    for index in range(1, 6):
        overrides: dict[str, Any] = (
            {"drawdown": BREACHING_DRAWDOWN} if index == triggering_index else {}
        )
        cycle_id = f"cycle-{index}"
        outcome = await run_cycle(
            session,
            observation(
                cycle_id=cycle_id,
                reconciliation=clean_result(cycle_id=cycle_id),
                **overrides,
            ),
        )
        if index < triggering_index:
            assert outcome.release_permitted, index
    engagements = [row for row in rows.halts if row["halt_trigger"] is not None]
    assert len(engagements) == 1
    assert engagements[0]["cycle_id"] == f"cycle-{triggering_index}"
    observed_in = triggering_index
    halted_in = int(str(engagements[0]["cycle_id"]).removeprefix("cycle-"))
    assert halted_in - observed_in == 0


async def test_a_reconciliation_mismatch_carries_the_verdict_digest_into_the_halt() -> None:
    # The halt has to be traceable back to the exact comparison that caused it,
    # or an investigation starts by guessing which reconciliation it meant.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    result = broken_result()
    outcome = await cycle(session, reconciliation=result)
    evidence = rows.halt(outcome.engaged_halt_ids[0])["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["result_digest"] == result.result_digest
    assert evidence["internal_digest"] == result.internal.digest
    assert evidence["reported_digest"] == result.reported.digest
    assert evidence["break_count"] == 1


async def test_all_firing_reasons_are_recorded_not_only_the_first() -> None:
    # One halt caused by four conditions and one caused by a single condition are
    # different situations, and an operator has to be able to tell them apart.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    outcome = await cycle(
        session,
        drawdown=BREACHING_DRAWDOWN,
        freshness=STALE,
        reconciliation=broken_result(),
        manual=MANUAL,
    )
    assert [reason.trigger for reason in outcome.decision.reasons] == [
        HaltTrigger.DRAWDOWN_BREACH,
        HaltTrigger.STALE_DATA,
        HaltTrigger.RECONCILIATION_MISMATCH,
        HaltTrigger.MANUAL,
    ]
    assert len(outcome.engaged_halt_ids) == 4


# ---------------------------------------------------------------------------
# Boundaries: at the limit is not past it.
# ---------------------------------------------------------------------------


async def test_a_drawdown_exactly_at_the_limit_does_not_halt() -> None:
    at_limit = DrawdownObservation(
        equity_usd=Decimal("90000.00"),
        peak_equity_usd=Decimal("100000.00"),
        limit_fraction=Decimal("0.10"),
    )
    assert not evaluate(observation(drawdown=at_limit)).should_halt


async def test_one_cent_past_the_drawdown_limit_halts() -> None:
    past_limit = DrawdownObservation(
        equity_usd=Decimal("89999.99"),
        peak_equity_usd=Decimal("100000.00"),
        limit_fraction=Decimal("0.10"),
    )
    decision = evaluate(observation(drawdown=past_limit))
    assert [reason.trigger for reason in decision.reasons] == [HaltTrigger.DRAWDOWN_BREACH]


async def test_data_exactly_at_the_maximum_age_does_not_halt() -> None:
    at_limit = DataFreshnessObservation(
        age_seconds=Decimal("3600"), max_age_seconds=Decimal("3600")
    )
    assert not evaluate(observation(freshness=at_limit)).should_halt


async def test_one_second_past_the_maximum_age_halts() -> None:
    past_limit = DataFreshnessObservation(
        age_seconds=Decimal("3601"), max_age_seconds=Decimal("3600")
    )
    decision = evaluate(observation(freshness=past_limit))
    assert [reason.trigger for reason in decision.reasons] == [HaltTrigger.STALE_DATA]


def test_the_drawdown_verdict_does_not_depend_on_the_reported_fraction_rounding() -> None:
    # The decision is taken on an exact multiplication; the fraction in the
    # evidence is divided out only for an operator to read. A third that rounds
    # either way must not move the verdict.
    exactly_a_third = DrawdownObservation(
        equity_usd=Decimal("2"),
        peak_equity_usd=Decimal("3"),
        limit_fraction=Decimal("0.333333"),
    )
    decision = evaluate(observation(drawdown=exactly_a_third))
    assert [reason.trigger for reason in decision.reasons] == [HaltTrigger.DRAWDOWN_BREACH]


# ---------------------------------------------------------------------------
# Fail closed.
# ---------------------------------------------------------------------------

FAIL_CLOSED_CASES: dict[str, tuple[dict[str, Any], str]] = {
    # Each case pairs the deviation with the phrase the halt must name.
    #
    # The phrase is not decoration. Mutation testing found that deleting three of
    # these guards outright left the suite green: with the guard gone the value
    # falls through to a comparison against `None`, that raises a TypeError, and
    # `evaluate`'s catch-all converts it into an unknown-condition halt anyway. The
    # system still fails closed — but by accident, through a path one refactor
    # away from changing, and with a message that tells the operator nothing.
    # Asserting *which* condition was named distinguishes the guard from the
    # accident.
    "no drawdown measurement": ({"drawdown": None}, "no drawdown measurement was taken"),
    "no equity": (
        {"drawdown": healthy_drawdown(equity_usd=None)},
        "equity or peak equity is missing",
    ),
    "no peak equity": (
        {"drawdown": healthy_drawdown(peak_equity_usd=None)},
        "equity or peak equity is missing",
    ),
    "no drawdown limit configured": (
        {"drawdown": healthy_drawdown(limit_fraction=None)},
        "no drawdown limit is configured",
    ),
    "equity is NaN": (
        {"drawdown": healthy_drawdown(equity_usd=Decimal("NaN"))},
        "equity or peak equity is missing",
    ),
    "peak is infinite": (
        {"drawdown": healthy_drawdown(peak_equity_usd=Decimal("Infinity"))},
        "equity or peak equity is missing",
    ),
    "limit is NaN": (
        {"drawdown": healthy_drawdown(limit_fraction=Decimal("NaN"))},
        "no drawdown limit is configured",
    ),
    "peak equity is zero": (
        {"drawdown": healthy_drawdown(peak_equity_usd=Decimal("0"))},
        "drawdown fraction is undefined",
    ),
    "peak equity is negative": (
        {"drawdown": healthy_drawdown(peak_equity_usd=Decimal("-1"))},
        "drawdown fraction is undefined",
    ),
    "limit is zero": (
        {"drawdown": healthy_drawdown(limit_fraction=Decimal("0"))},
        "outside (0, 1]",
    ),
    "limit is negative": (
        {"drawdown": healthy_drawdown(limit_fraction=Decimal("-0.1"))},
        "outside (0, 1]",
    ),
    "limit exceeds a total loss": (
        {"drawdown": healthy_drawdown(limit_fraction=Decimal("1.5"))},
        "outside (0, 1]",
    ),
    "no freshness measurement": ({"freshness": None}, "no data-age measurement was taken"),
    "no data age": (
        {"freshness": fresh_data(age_seconds=None)},
        "data age is missing or not a finite number",
    ),
    "no maximum age configured": (
        {"freshness": fresh_data(max_age_seconds=None)},
        "no maximum data age is configured",
    ),
    "data age is NaN": (
        {"freshness": fresh_data(age_seconds=Decimal("NaN"))},
        "data age is missing or not a finite number",
    ),
    "maximum age is NaN": (
        {"freshness": fresh_data(max_age_seconds=Decimal("NaN"))},
        "no maximum data age is configured",
    ),
    "data age is negative": (
        {"freshness": fresh_data(age_seconds=Decimal("-1"))},
        "which is negative",
    ),
    "maximum age is zero": (
        {"freshness": fresh_data(max_age_seconds=Decimal("0"))},
        "which is not positive",
    ),
    "no reconciliation": ({"reconciliation": None}, "no reconciliation was produced"),
    "reconciliation from another cycle": (
        {"reconciliation": clean_result(cycle_id="yesterday")},
        "belongs to cycle",
    ),
    "drawdown probe raises": (
        {"drawdown": healthy_drawdown(equity_usd=ExplodingDecimal("95000.00"))},
        "the drawdown check raised RuntimeError",
    ),
    "freshness probe raises": (
        {"freshness": fresh_data(age_seconds=ExplodingDecimal("60"))},
        "the data_freshness check raised RuntimeError",
    ),
}


@pytest.mark.parametrize("case", list(FAIL_CLOSED_CASES))
async def test_every_unknown_condition_halts(case: str) -> None:
    overrides, phrase = FAIL_CLOSED_CASES[case]
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    outcome = await cycle(session, **overrides)
    assert outcome.halted, case
    reasons = [
        reason
        for reason in outcome.decision.reasons
        if reason.trigger is HaltTrigger.UNKNOWN_CONDITION
    ]
    assert reasons, case
    # The halt must name the condition it could not rule out, not merely fire.
    assert any(phrase in reason.detail for reason in reasons), (case, [r.detail for r in reasons])
    assert rows.halts, case
    assert any(phrase in str(row["detail"]) for row in rows.halts), case
    with pytest.raises(SystemHaltedError):
        await guard_before_release(session)


async def test_a_verdict_that_raises_when_read_halts_rather_than_propagating() -> None:
    exploding = ExplodingResult(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(),
        reported=reported_snapshot(),
        cash_tolerance_usd=Decimal("0.01"),
        findings=(),
        stamp=make_stamp(),
    )
    decision = evaluate(observation(reconciliation=exploding))
    assert [reason.trigger for reason in decision.reasons] == [HaltTrigger.UNKNOWN_CONDITION]
    assert "could not be evaluated" in decision.reasons[0].detail


async def test_a_reconciliation_of_the_wrong_type_halts_rather_than_raising() -> None:
    decision = evaluate(observation(reconciliation="clean, honest"))  # type: ignore[arg-type]
    assert [reason.trigger for reason in decision.reasons] == [HaltTrigger.UNKNOWN_CONDITION]
    assert "not a verdict this module can read" in decision.reasons[0].detail


def test_evaluate_never_raises_whatever_it_is_handed() -> None:
    # A safety check that can propagate an exception is a safety check the caller
    # can lose: the exception is indistinguishable from the check never running.
    broken = CycleObservation(
        cycle_id=CYCLE_ID,
        observed_at=OBSERVED_AT,
        drawdown="not an observation",  # type: ignore[arg-type]
        freshness="also not one",  # type: ignore[arg-type]
        reconciliation=None,
        manual=None,
        stamp=make_stamp(),
    )
    decision = evaluate(broken)
    assert decision.should_halt
    assert {reason.trigger for reason in decision.reasons} == {HaltTrigger.UNKNOWN_CONDITION}


def test_a_blank_cycle_id_still_produces_a_writable_halt() -> None:
    # A malformed observation must not be the one thing that cannot be halted on.
    # A blank cycle id is itself refused upstream (reconcile will not produce a
    # verdict for one), so this observation is unknown-condition twice over — and
    # the halt it produces still carries a cycle_id the CHECK will accept.
    decision = evaluate(observation(cycle_id="   ", reconciliation=None))
    assert decision.cycle_id == "unnamed-cycle"
    assert decision.should_halt


# ---------------------------------------------------------------------------
# The manual trigger's deliberate asymmetry.
# ---------------------------------------------------------------------------


def test_an_absent_manual_request_is_the_only_none_that_does_not_halt() -> None:
    assert not evaluate(observation(manual=None)).should_halt


def test_a_manual_request_is_never_refused_for_being_unattributed() -> None:
    # Refusing to construct an operator's halt over a format check would be a
    # fail-open path through the one trigger a human reaches for in an emergency.
    request = ManualHaltRequest(requested_by="", reason="   ")
    assert request.requested_by == "unattributed"
    assert request.reason == "no reason given"
    decision = evaluate(observation(manual=request))
    assert [reason.trigger for reason in decision.reasons] == [HaltTrigger.MANUAL]


async def test_a_manual_halt_records_who_asked_and_why() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    outcome = await cycle(session, manual=MANUAL)
    evidence = rows.halt(outcome.engaged_halt_ids[0])["evidence"]
    assert isinstance(evidence, dict)
    assert evidence["requested_by"] == "risk-desk"
    assert evidence["reason"] == "pausing before the open"


# ---------------------------------------------------------------------------
# The cycle's relationship with halts that are already open.
# ---------------------------------------------------------------------------


async def test_a_cycle_that_begins_halted_stays_halted_however_clean_it_measures() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    first = await cycle(session, manual=MANUAL)
    second = await cycle(session)
    assert second.decision.reasons == ()
    assert second.engaged_halt_ids == ()
    assert second.pre_existing_halt_ids == first.engaged_halt_ids
    assert not second.release_permitted


async def test_a_clean_measurement_never_clears_a_halt_as_a_side_effect() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    await cycle(session, manual=MANUAL)
    await cycle(session)
    await cycle(session)
    assert len(rows.open_halt_ids()) == 1


async def test_clearing_lets_the_next_cycle_trade_again() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    engaged = await cycle(session, manual=MANUAL)
    await clear_halt(
        session,
        halt_id=engaged.engaged_halt_ids[0],
        cleared_by="risk-desk",
        clearance_reason="the pause is over",
        occurred_at=OBSERVED_AT,
        stamp=make_stamp(),
    )
    assert (await cycle(session)).release_permitted


async def test_a_cleared_halt_does_not_immunise_against_the_same_condition() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    engaged = await cycle(session, drawdown=BREACHING_DRAWDOWN)
    await clear_halt(
        session,
        halt_id=engaged.engaged_halt_ids[0],
        cleared_by="risk-desk",
        clearance_reason="investigated",
        occurred_at=OBSERVED_AT,
        stamp=make_stamp(),
    )
    again = await cycle(session, drawdown=BREACHING_DRAWDOWN)
    assert again.halted
    assert len(again.engaged_halt_ids) == 1


async def test_a_halt_engaged_by_a_cycle_survives_a_restart() -> None:
    rows = ControlRows()
    await cycle(as_session(ControlSessionDouble(rows)), drawdown=BREACHING_DRAWDOWN)
    reborn = as_session(restart(rows))
    assert len(await open_halts(reborn)) == 1
    with pytest.raises(SystemHaltedError):
        await guard_before_release(reborn)
    assert not (await cycle(reborn)).release_permitted


async def test_a_cycle_that_cannot_read_its_halt_state_refuses_to_proceed() -> None:
    # Propagated rather than summarised into an outcome the caller might ignore.
    rows = ControlRows()
    rows.read_failure = RuntimeError("connection reset")
    with pytest.raises(HaltStateUnavailableError):
        await cycle(as_session(ControlSessionDouble(rows)))


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------

_UNUSABLE = st.sampled_from([None, Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])


@given(
    equity=_UNUSABLE,
    peak=_UNUSABLE,
    limit=_UNUSABLE,
)
@settings(max_examples=100, deadline=None)
def test_no_combination_of_unusable_drawdown_inputs_permits_trading(
    equity: Decimal | None, peak: Decimal | None, limit: Decimal | None
) -> None:
    decision = evaluate(
        observation(
            drawdown=DrawdownObservation(
                equity_usd=equity, peak_equity_usd=peak, limit_fraction=limit
            )
        )
    )
    assert decision.should_halt
    assert decision.reasons[0].trigger is HaltTrigger.UNKNOWN_CONDITION


@given(age=_UNUSABLE, maximum=_UNUSABLE)
@settings(max_examples=60, deadline=None)
def test_no_combination_of_unusable_freshness_inputs_permits_trading(
    age: Decimal | None, maximum: Decimal | None
) -> None:
    decision = evaluate(
        observation(freshness=DataFreshnessObservation(age_seconds=age, max_age_seconds=maximum))
    )
    assert decision.should_halt


@given(
    shortfall=st.decimals(
        min_value=Decimal("0"), max_value=Decimal("100000"), allow_nan=False, places=2
    ),
    limit=st.decimals(min_value=Decimal("0.01"), max_value=Decimal("1"), allow_nan=False, places=2),
)
@settings(max_examples=200, deadline=None)
def test_the_drawdown_verdict_agrees_with_the_arithmetic_everywhere(
    shortfall: Decimal, limit: Decimal
) -> None:
    peak = Decimal("100000.00")
    decision = evaluate(
        observation(
            drawdown=DrawdownObservation(
                equity_usd=peak - shortfall, peak_equity_usd=peak, limit_fraction=limit
            )
        )
    )
    breached = {reason.trigger for reason in decision.reasons} == {HaltTrigger.DRAWDOWN_BREACH}
    assert breached == (shortfall > limit * peak)


@given(
    age=st.decimals(min_value=Decimal("0"), max_value=Decimal("100000"), allow_nan=False, places=1),
    maximum=st.decimals(
        min_value=Decimal("0.1"), max_value=Decimal("100000"), allow_nan=False, places=1
    ),
)
@settings(max_examples=200, deadline=None)
def test_the_staleness_verdict_agrees_with_the_arithmetic_everywhere(
    age: Decimal, maximum: Decimal
) -> None:
    decision = evaluate(
        observation(freshness=DataFreshnessObservation(age_seconds=age, max_age_seconds=maximum))
    )
    stale = {reason.trigger for reason in decision.reasons} == {HaltTrigger.STALE_DATA}
    assert stale == (age > maximum)
