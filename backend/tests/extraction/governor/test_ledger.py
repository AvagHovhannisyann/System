"""P7.7: reserving is the check, and it holds under genuine concurrency.

The concurrency test in this file is the one that decides whether the cap is a
control or a comment. Two calls checking a cap at the same time can both pass
and jointly breach it; the fix is that there is no separable "check" to lose the
race with — :meth:`~backend.extraction.governor.ledger.InMemorySpendLedger.reserve`
reads committed spend, compares, and writes the reservation inside one lock.

``test_concurrent_reservations_admit_exactly_the_number_the_cap_allows`` runs
genuinely concurrent tasks through ``asyncio.gather`` and asserts an exact
admitted count. It is not vacuous: the ledger yields inside its critical section
(documented at the yield), so removing the lock lets the interleaving happen and
the test fails. The mutation evidence is recorded in the task report.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from backend.extraction.governor.caps import CapPolicy, SpendWindow
from backend.extraction.governor.errors import LedgerIntegrityError, SpendCapExceededError
from backend.extraction.governor.estimate import CallEstimate, Money
from backend.extraction.governor.ledger import (
    CallOutcome,
    InMemorySpendLedger,
    Reservation,
    SpendEvent,
)
from backend.extraction.providers.catalog import Provider
from backend.tests.extraction.governor.doubles import AT, CURRENCY, PRIMARY_MODEL, usd

_WINDOWS = {SpendWindow.DAILY: "2026-08-02", SpendWindow.MONTHLY: "2026-08"}


def _estimate(cost: str, *, model: str = PRIMARY_MODEL) -> CallEstimate:
    """Return an estimate of *cost* USD with plausible token bounds."""
    return CallEstimate(
        model=model, input_tokens_bound=1124, output_tokens_bound=1000, cost=usd(cost)
    )


async def _reserve(
    ledger: InMemorySpendLedger,
    cost: str,
    *,
    daily: str = "1.00",
    monthly: str = "10.00",
    served_model: str = PRIMARY_MODEL,
    at: dt.datetime = AT,
) -> Reservation:
    """Reserve *cost* against the given limits."""
    return await ledger.reserve(
        provider=Provider.ANTHROPIC,
        requested_model=PRIMARY_MODEL,
        served_model=served_model,
        estimate=_estimate(cost, model=served_model),
        windows={window: window.key(at) for window in SpendWindow},
        limits={SpendWindow.DAILY: usd(daily), SpendWindow.MONTHLY: usd(monthly)},
        policy=CapPolicy.HALT,
        correlation_id="corr-1",
        occurred_at=at,
    )


async def _committed(ledger: InMemorySpendLedger, window: SpendWindow = SpendWindow.DAILY) -> Money:
    """Return committed spend in *window* for the test provider."""
    return await ledger.committed(
        provider=Provider.ANTHROPIC,
        window=window,
        window_key=_WINDOWS[window],
        currency=CURRENCY,
    )


# --------------------------------------------------------------------------
# Reserving is checking
# --------------------------------------------------------------------------


async def test_a_reservation_consumes_headroom_immediately() -> None:
    """Not at settlement — a reservation that did not consume would not be one."""
    ledger = InMemorySpendLedger()
    await _reserve(ledger, "0.40")

    assert await _committed(ledger) == usd("0.40")
    assert await _committed(ledger, SpendWindow.MONTHLY) == usd("0.40")


async def test_a_reservation_that_would_breach_the_daily_cap_is_refused() -> None:
    ledger = InMemorySpendLedger()
    await _reserve(ledger, "0.80")

    with pytest.raises(SpendCapExceededError) as raised:
        await _reserve(ledger, "0.30")

    breach = raised.value
    assert breach.window is SpendWindow.DAILY
    assert breach.window_key == "2026-08-02"
    assert breach.limit == usd("1.00")
    assert breach.committed == usd("0.80")
    assert breach.estimate == usd("0.30")
    assert "No request was made." in str(breach)
    # And the refusal consumed nothing.
    assert await _committed(ledger) == usd("0.80")


async def test_the_monthly_cap_binds_even_when_the_day_has_room() -> None:
    """Both windows are checked; a generous day cannot outspend a tight month."""
    ledger = InMemorySpendLedger()
    with pytest.raises(SpendCapExceededError) as raised:
        await _reserve(ledger, "0.50", daily="10.00", monthly="0.20")

    assert raised.value.window is SpendWindow.MONTHLY


async def test_a_call_exactly_filling_the_cap_is_admitted() -> None:
    """The limit is inclusive: spending the whole budget is what a budget is for."""
    ledger = InMemorySpendLedger()
    await _reserve(ledger, "1.00")
    assert await _committed(ledger) == usd("1.00")

    with pytest.raises(SpendCapExceededError):
        await _reserve(ledger, "0.0000000001")


async def test_a_zero_cap_refuses_any_call_that_costs_anything() -> None:
    ledger = InMemorySpendLedger()
    with pytest.raises(SpendCapExceededError):
        await _reserve(ledger, "0.0000000001", daily="0", monthly="0")


async def test_windows_are_independent_so_a_new_day_restores_headroom() -> None:
    ledger = InMemorySpendLedger()
    await _reserve(ledger, "1.00")
    tomorrow = AT + dt.timedelta(days=1)

    await _reserve(ledger, "1.00", at=tomorrow)

    assert await _committed(ledger) == usd("1.00")
    assert await _committed(ledger, SpendWindow.MONTHLY) == usd("2.00")


# --------------------------------------------------------------------------
# Concurrency: the answer, and the proof it is real
# --------------------------------------------------------------------------


async def test_concurrent_reservations_admit_exactly_the_number_the_cap_allows() -> None:
    """Ten simultaneous calls against a cap with room for four admit exactly four.

    This is the check-then-act race, run for real: the tasks are started
    together and interleave inside the ledger's critical section. Without the
    reservation being part of the check, every task would read zero committed
    spend, all ten would pass, and the cap would be breached by 150%.
    """
    ledger = InMemorySpendLedger()
    admitted = 0
    refused = 0

    async def attempt() -> None:
        nonlocal admitted, refused
        try:
            await _reserve(ledger, "0.25", daily="1.00", monthly="100.00")
        except SpendCapExceededError:
            refused += 1
        else:
            admitted += 1

    await asyncio.gather(*(attempt() for _ in range(10)))

    assert admitted == 4
    assert refused == 6
    assert await _committed(ledger) == usd("1.00")
    reserved_rows = [row for row in await ledger.records() if row.event is SpendEvent.RESERVED]
    assert len(reserved_rows) == 4


async def test_concurrent_reservations_across_providers_do_not_share_a_budget() -> None:
    """One provider's burst must not consume another's cap."""
    ledger = InMemorySpendLedger()

    async def attempt(provider: Provider) -> bool:
        try:
            await ledger.reserve(
                provider=provider,
                requested_model=f"{provider.value}:m",
                served_model=f"{provider.value}:m",
                estimate=_estimate("1.00", model=f"{provider.value}:m"),
                windows={window: window.key(AT) for window in SpendWindow},
                limits={SpendWindow.DAILY: usd("1.00"), SpendWindow.MONTHLY: usd("10.00")},
                policy=CapPolicy.HALT,
                correlation_id=None,
                occurred_at=AT,
            )
        except SpendCapExceededError:
            return False
        return True

    results = await asyncio.gather(attempt(Provider.ANTHROPIC), attempt(Provider.OPENAI))
    assert list(results) == [True, True]


# --------------------------------------------------------------------------
# Settlement: reserve at the bound, book the truth
# --------------------------------------------------------------------------


async def test_settlement_returns_the_difference_between_the_bound_and_the_truth() -> None:
    """Otherwise the cap would be consumed by pessimism rather than by spending."""
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.40")

    record = await ledger.settle(
        reservation,
        actual=usd("0.05"),
        input_tokens=120,
        output_tokens=40,
        outcome=CallOutcome.SUCCEEDED,
    )

    assert record.delta == usd("-0.35")
    assert record.reconciled is True
    assert await _committed(ledger) == usd("0.05")


async def test_settlement_without_reported_tokens_keeps_the_bound() -> None:
    """I3: an estimated actual is a fabricated measurement, so the bound stands."""
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.40")

    record = await ledger.settle(
        reservation,
        actual=None,
        input_tokens=None,
        output_tokens=None,
        outcome=CallOutcome.SUCCEEDED,
    )

    assert record.actual_cost is None
    assert record.reconciled is False
    assert record.delta == usd("0")
    assert await _committed(ledger) == usd("0.40")


async def test_an_actual_above_the_bound_is_recorded_rather_than_clamped() -> None:
    """If the bound was not a bound, the row is the evidence — hiding it hides the defect."""
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.10")

    record = await ledger.settle(
        reservation,
        actual=usd("0.30"),
        input_tokens=10_000,
        output_tokens=5_000,
        outcome=CallOutcome.SUCCEEDED,
    )

    assert record.delta == usd("0.20")
    assert await _committed(ledger) == usd("0.30")


async def test_a_release_hands_the_whole_reservation_back() -> None:
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.40")

    record = await ledger.release(reservation)

    assert record.delta == usd("-0.40")
    assert await _committed(ledger) == usd("0")


# --------------------------------------------------------------------------
# Integrity: a delta may be booked once
# --------------------------------------------------------------------------


async def test_settling_twice_is_refused() -> None:
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.40")
    await ledger.settle(
        reservation,
        actual=usd("0.05"),
        input_tokens=1,
        output_tokens=1,
        outcome=CallOutcome.SUCCEEDED,
    )

    with pytest.raises(LedgerIntegrityError, match="already settled"):
        await ledger.settle(
            reservation,
            actual=usd("0.05"),
            input_tokens=1,
            output_tokens=1,
            outcome=CallOutcome.SUCCEEDED,
        )


async def test_settling_a_released_reservation_is_refused() -> None:
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.40")
    await ledger.release(reservation)

    with pytest.raises(LedgerIntegrityError, match="already released"):
        await ledger.settle(
            reservation,
            actual=None,
            input_tokens=None,
            output_tokens=None,
            outcome=CallOutcome.FAILED,
        )


async def test_settling_a_reservation_this_ledger_never_issued_is_refused() -> None:
    ledger = InMemorySpendLedger()
    foreign = Reservation(
        reservation_id="deadbeef" * 4,
        provider=Provider.ANTHROPIC,
        requested_model=PRIMARY_MODEL,
        served_model=PRIMARY_MODEL,
        estimate=_estimate("0.10"),
        windows=_WINDOWS,
        limits={SpendWindow.DAILY: usd("1.00"), SpendWindow.MONTHLY: usd("10.00")},
        policy=CapPolicy.HALT,
        correlation_id=None,
        occurred_at=AT,
    )

    with pytest.raises(LedgerIntegrityError, match="never issued"):
        await ledger.release(foreign)


# --------------------------------------------------------------------------
# The audit trail
# --------------------------------------------------------------------------


async def test_every_event_is_recorded_and_nothing_is_edited() -> None:
    """The trail is append-only: the reservation row survives its own settlement."""
    ledger = InMemorySpendLedger()
    reservation = await _reserve(ledger, "0.40", served_model="anthropic:test-cheap")
    await ledger.settle(
        reservation,
        actual=usd("0.05"),
        input_tokens=120,
        output_tokens=40,
        outcome=CallOutcome.SUCCEEDED,
    )

    rows = await ledger.records()
    assert [row.event for row in rows] == [SpendEvent.RESERVED, SpendEvent.SETTLED]
    assert {row.reservation_id for row in rows} == {reservation.reservation_id}
    # Both models on every row: the substitution is a recorded fact, not an inference (I2).
    for row in rows:
        assert row.requested_model == PRIMARY_MODEL
        assert row.served_model == "anthropic:test-cheap"
        assert row.degraded is True
        assert row.correlation_id == "corr-1"
    # The bound is still readable next to the truth, so "how wrong was it" is answerable.
    assert rows[0].estimated_cost == usd("0.40")
    assert rows[1].actual_cost == usd("0.05")


async def test_the_limits_in_force_are_copied_onto_the_row() -> None:
    """A cap raised this afternoon must not rewrite this morning's audit trail."""
    ledger = InMemorySpendLedger()
    await _reserve(ledger, "0.40", daily="1.00", monthly="10.00")

    row = (await ledger.records())[0]
    assert row.limits[SpendWindow.DAILY] == usd("1.00")
    assert row.limits[SpendWindow.MONTHLY] == usd("10.00")
    assert row.policy is CapPolicy.HALT
