"""P7.7: the cap **blocks** — asserted on the client, not on the return value.

The central test in this file is
``test_a_breaching_call_never_reaches_the_provider``. It asserts on
:class:`~backend.tests.extraction.governor.doubles.NeverCalledClient`, which
fails if it is touched at all, because a test that only checked the raised
exception would pass just as happily if the request had been sent and the
response discarded — and that is precisely the failure this task exists to
prevent. "Refused" and "refused before the money left" are different claims, and
only one of them is a cost control.

The concurrency test runs genuinely concurrent governed calls through
``asyncio.gather`` and asserts an exact number of requests **reached the
provider**, which is the property a burst-time cap is actually about.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.extraction.governor.caps import CapPolicy, SpendWindow
from backend.extraction.governor.errors import (
    CapNotConfiguredError,
    ModelPriceUnknownError,
    SpendCapExceededError,
)
from backend.extraction.governor.estimate import PriceBook
from backend.extraction.governor.governor import CostGovernor
from backend.extraction.governor.guard import GovernedModelClient
from backend.extraction.governor.ledger import CallOutcome, InMemorySpendLedger, SpendEvent
from backend.extraction.providers.catalog import Provider
from backend.extraction.tasks.client import ModelCallError, ModelClient
from backend.tests.extraction.governor.doubles import (
    AT,
    CHEAP_MODEL,
    OTHER_PROVIDER_MODEL,
    PRIMARY_MODEL,
    FailingClient,
    NeverCalledClient,
    RecordingClient,
    cap,
    cap_book,
    clock,
    price_book,
    request,
    usd,
)


def _governed(
    inner: ModelClient,
    *,
    daily: str = "1.00",
    monthly: str = "10.00",
    policy: CapPolicy = CapPolicy.HALT,
    degrade_to: str | None = None,
    ledger: InMemorySpendLedger | None = None,
    prices: PriceBook | None = None,
) -> tuple[GovernedModelClient, CostGovernor, InMemorySpendLedger]:
    """Wire a governed client over *inner* with one anthropic cap."""
    spend_ledger = ledger if ledger is not None else InMemorySpendLedger()
    governor = CostGovernor(
        caps=cap_book(cap(daily=daily, monthly=monthly, policy=policy, degrade_to=degrade_to)),
        prices=prices if prices is not None else price_book(),
        ledger=spend_ledger,
        clock=clock(),
    )
    return GovernedModelClient(inner=inner, governor=governor), governor, spend_ledger


# --------------------------------------------------------------------------
# The cap blocks, and it blocks *before* the request
# --------------------------------------------------------------------------


async def test_a_breaching_call_never_reaches_the_provider() -> None:
    """The assertion lives on the client: nothing was sent, not merely nothing returned."""
    inner = NeverCalledClient()
    client, _, ledger = _governed(inner, daily="0.0000000001")

    with pytest.raises(SpendCapExceededError, match="No request was made"):
        await client.complete(request())

    assert await ledger.records() == ()


async def test_an_uncapped_provider_never_reaches_the_provider() -> None:
    """B4: ungoverned is refused, and refused before the call."""
    inner = RecordingClient()
    prices = price_book({OTHER_PROVIDER_MODEL: price_book().price_for(CHEAP_MODEL)})
    client, _, _ = _governed(inner, prices=prices)

    with pytest.raises(CapNotConfiguredError):
        await client.complete(request(model=OTHER_PROVIDER_MODEL))

    assert inner.calls == 0


async def test_an_unpriced_model_never_reaches_the_provider() -> None:
    """I3: with no price there is no bound, so there is no authorization."""
    inner = RecordingClient()
    client, _, _ = _governed(inner, prices=PriceBook({}))

    with pytest.raises(ModelPriceUnknownError):
        await client.complete(request())

    assert inner.calls == 0


async def test_the_cap_blocks_the_call_after_the_budget_is_spent() -> None:
    """The first call goes through, the second does not, and the second sends nothing."""
    inner = RecordingClient(input_tokens=None, output_tokens=None)
    client, _, _ = _governed(inner, daily="0.15", monthly="10.00")
    call = request(prompt="p" * 100, system="", max_tokens=1000)

    await client.complete(call)
    assert inner.calls == 1

    with pytest.raises(SpendCapExceededError):
        await client.complete(call)
    assert inner.calls == 1


# --------------------------------------------------------------------------
# Concurrency, end to end
# --------------------------------------------------------------------------


async def test_concurrent_governed_calls_send_exactly_the_number_the_cap_allows() -> None:
    """Twelve simultaneous calls against a cap with room for three send three requests.

    ``delay=True`` makes the inner client yield, so the tasks genuinely
    interleave around the provider call. Without check-and-reserve being one
    operation every task would find the budget empty and all twelve requests
    would go out.
    """
    inner = RecordingClient(delay=True, input_tokens=None, output_tokens=None)
    call = request(prompt="p" * 100, system="", max_tokens=1000)
    # One call's upper bound is USD 0.11124; three fit under USD 0.35, four do not.
    client, governor, ledger = _governed(inner, daily="0.35", monthly="100.00")

    async def attempt() -> bool:
        try:
            await client.complete(call)
        except SpendCapExceededError:
            return False
        return True

    outcomes = await asyncio.gather(*(attempt() for _ in range(12)))

    assert sum(outcomes) == 3
    assert inner.calls == 3, "a refused call must not reach the provider"
    committed = await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert committed <= usd("0.35")
    reserved = [row for row in await ledger.records() if row.event is SpendEvent.RESERVED]
    assert len(reserved) == 3


# --------------------------------------------------------------------------
# Degradation is recorded
# --------------------------------------------------------------------------


async def test_degradation_sends_the_substitute_and_records_which_model_answered() -> None:
    """I2: the artefact names the model that actually read the document."""
    inner = RecordingClient()
    client, _, ledger = _governed(
        inner,
        daily="0.05",
        monthly="10.00",
        policy=CapPolicy.DEGRADE,
        degrade_to=CHEAP_MODEL,
    )

    completion = await client.complete_governed(request())

    # What was actually sent.
    assert inner.models == (CHEAP_MODEL,)
    # What the completion says.
    assert completion.degraded is True
    assert completion.requested_model == PRIMARY_MODEL
    assert completion.served_model == CHEAP_MODEL
    # What the durable trail says — both models, on both rows.
    rows = await ledger.records()
    assert [row.event for row in rows] == [SpendEvent.RESERVED, SpendEvent.SETTLED]
    for row in rows:
        assert row.requested_model == PRIMARY_MODEL
        assert row.served_model == CHEAP_MODEL
        assert row.degraded is True
    assert rows[-1].outcome is CallOutcome.SUCCEEDED


async def test_a_call_that_fits_is_not_degraded() -> None:
    """Degradation is a ceiling behaviour, not a default."""
    inner = RecordingClient()
    client, _, ledger = _governed(
        inner, daily="10.00", monthly="100.00", policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL
    )

    completion = await client.complete_governed(request())

    assert inner.models == (PRIMARY_MODEL,)
    assert completion.degraded is False
    assert all(row.degraded is False for row in await ledger.records())


# --------------------------------------------------------------------------
# Reconciliation through the client
# --------------------------------------------------------------------------


async def test_a_completed_call_is_reconciled_against_reported_tokens() -> None:
    inner = RecordingClient(input_tokens=1000, output_tokens=100)
    client, governor, ledger = _governed(inner)

    completion = await client.complete_governed(request())

    assert completion.actual_cost is not None
    assert completion.actual_cost < completion.estimated_cost
    committed = await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert committed == completion.actual_cost
    assert [row.event for row in await ledger.records()] == [
        SpendEvent.RESERVED,
        SpendEvent.SETTLED,
    ]


async def test_a_provider_that_reports_no_tokens_settles_at_the_bound() -> None:
    inner = RecordingClient(input_tokens=None, output_tokens=None)
    client, governor, _ = _governed(inner)

    completion = await client.complete_governed(request())

    assert completion.actual_cost is None
    assert completion.record.reconciled is False
    committed = await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert committed == completion.estimated_cost


async def test_a_failed_call_is_charged_at_its_bound_and_the_error_propagates_unchanged() -> None:
    """The retry loop still sees a ModelCallError — and its retries are themselves capped."""
    inner = FailingClient()
    client, governor, ledger = _governed(inner, daily="0.15", monthly="10.00")
    call = request(prompt="p" * 100, system="", max_tokens=1000)

    with pytest.raises(ModelCallError, match="connection reset"):
        await client.complete(call)

    assert inner.calls == 1
    rows = await ledger.records()
    assert [row.event for row in rows] == [SpendEvent.RESERVED, SpendEvent.SETTLED]
    assert rows[-1].outcome is CallOutcome.FAILED
    assert rows[-1].reconciled is False
    committed = await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert committed == rows[0].estimated_cost

    # A retry loop cannot spend past the cap either: the second attempt is refused
    # before the client is touched again.
    with pytest.raises(SpendCapExceededError):
        await client.complete(call)
    assert inner.calls == 1


# --------------------------------------------------------------------------
# The seam itself
# --------------------------------------------------------------------------


async def test_the_governed_client_satisfies_the_model_client_protocol() -> None:
    """It drops into the P7.3 pipeline in place of the client it governs."""
    client, _, _ = _governed(RecordingClient())
    assert isinstance(client, ModelClient)


async def test_there_is_no_accessor_that_hands_out_the_inner_client() -> None:
    """An accessor would be a supported way to bypass the cap."""
    client, _, _ = _governed(RecordingClient())
    public = {name for name in dir(client) if not name.startswith("_")}
    assert public == {"complete", "complete_governed"}
