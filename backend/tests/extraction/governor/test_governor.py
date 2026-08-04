"""P7.7: the authorization decision — halt, degrade, or refuse to guess.

What is proved here:

* an unpriced model and an uncapped provider both **refuse**, and refuse
  differently, so an operator can tell "you have work to do" from "the control
  worked";
* halt is per provider: one provider hitting its ceiling leaves another's
  untouched;
* degrade substitutes a model that is cheaper **for this call**, and refuses
  when it is not — including when the fallback is unpriced, when it is the model
  that just breached, and when it does not fit either (halt is the floor of
  degrade);
* settlement reconciles at the price of the model that *actually answered*, not
  the one that was asked for.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from backend.extraction.governor.caps import CapBook, CapPolicy, SpendWindow
from backend.extraction.governor.errors import (
    CapNotConfiguredError,
    CurrencyMismatchError,
    DegradationUnavailableError,
    ModelPriceUnknownError,
    SpendCapExceededError,
)
from backend.extraction.governor.estimate import Money, PriceBook, TokenPrice
from backend.extraction.governor.governor import CostGovernor
from backend.extraction.governor.ledger import InMemorySpendLedger, SpendEvent
from backend.extraction.providers.catalog import Provider
from backend.extraction.tasks.client import ModelResponse
from backend.tests.extraction.governor.doubles import (
    AT,
    CHEAP_MODEL,
    OTHER_PROVIDER_MODEL,
    PRIMARY_MODEL,
    cap,
    cap_book,
    clock,
    price,
    price_book,
    request,
    usd,
)


def _governor(
    *,
    caps: CapBook | None = None,
    prices: PriceBook | None = None,
    ledger: InMemorySpendLedger | None = None,
) -> CostGovernor:
    """Build a governor over an in-memory ledger with the test price book."""
    return CostGovernor(
        caps=caps if caps is not None else cap_book(),
        prices=prices if prices is not None else price_book(),
        ledger=ledger if ledger is not None else InMemorySpendLedger(),
        clock=clock(),
    )


# --------------------------------------------------------------------------
# Refusing to guess
# --------------------------------------------------------------------------


async def test_an_uncapped_provider_is_refused_before_anything_is_priced() -> None:
    governor = _governor(caps=cap_book(cap(provider=Provider.ANTHROPIC)))
    with pytest.raises(CapNotConfiguredError):
        await governor.authorize(request(model=OTHER_PROVIDER_MODEL))


async def test_an_unpriced_model_is_refused_rather_than_bounded_at_a_guess() -> None:
    """I3: a cap enforced against a made-up price enforces nothing."""
    governor = _governor(prices=PriceBook({}))
    with pytest.raises(ModelPriceUnknownError):
        await governor.authorize(request())


async def test_a_price_in_another_currency_is_refused_naming_the_pair() -> None:
    prices = PriceBook(
        {
            PRIMARY_MODEL: TokenPrice(
                input_per_million_tokens=Money(Decimal("1"), "EUR"),
                output_per_million_tokens=Money(Decimal("1"), "EUR"),
            )
        }
    )
    governor = _governor(prices=prices)
    with pytest.raises(CurrencyMismatchError, match="unstated conversion"):
        await governor.authorize(request())


async def test_an_unqualified_model_is_refused_because_the_qualifier_names_the_budget() -> None:
    governor = _governor()
    with pytest.raises(ValueError, match="not qualified"):
        await governor.authorize(request(model="bare-model-name"))


async def test_a_naive_clock_is_refused() -> None:
    governor = CostGovernor(
        caps=cap_book(),
        prices=price_book(),
        ledger=InMemorySpendLedger(),
        clock=lambda: dt.datetime(2026, 8, 2, 12, 0),  # noqa: DTZ001 — the point
    )
    with pytest.raises(ValueError, match="naive"):
        await governor.authorize(request())


# --------------------------------------------------------------------------
# Authorization consumes headroom, and the bound is what is consumed
# --------------------------------------------------------------------------


async def test_authorization_reserves_the_upper_bound_not_an_expectation() -> None:
    ledger = InMemorySpendLedger()
    governor = _governor(ledger=ledger)

    authorization = await governor.authorize(request())

    assert authorization.estimate.output_tokens_bound == 1000
    assert authorization.served_model == PRIMARY_MODEL
    assert authorization.degraded is False
    committed = await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert committed == authorization.estimate.cost


async def test_headroom_reports_what_is_left_and_never_a_negative() -> None:
    governor = _governor(caps=cap_book(cap(daily="1.00", monthly="10.00")))
    before = await governor.headroom(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert before == usd("1.00")

    await governor.authorize(request())
    after = await governor.headroom(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT)
    assert after < before
    assert after.amount >= 0


# --------------------------------------------------------------------------
# Halt
# --------------------------------------------------------------------------


async def test_a_breaching_call_halts_and_reserves_nothing() -> None:
    ledger = InMemorySpendLedger()
    governor = _governor(caps=cap_book(cap(daily="0.05", monthly="10.00")), ledger=ledger)

    with pytest.raises(SpendCapExceededError) as raised:
        await governor.authorize(request())

    assert raised.value.requested_model == PRIMARY_MODEL
    assert await ledger.records() == ()


async def test_halting_one_provider_leaves_another_providers_budget_alone() -> None:
    """Halt versus degrade — and the cap itself — is per provider, not global."""
    ledger = InMemorySpendLedger()
    prices = price_book({OTHER_PROVIDER_MODEL: price(input_rate="1.00", output_rate="1.00")})
    governor = _governor(
        caps=cap_book(
            cap(provider=Provider.ANTHROPIC, daily="0.05"),
            cap(provider=Provider.OPENAI, daily="1.00"),
        ),
        prices=prices,
        ledger=ledger,
    )

    with pytest.raises(SpendCapExceededError):
        await governor.authorize(request())

    authorization = await governor.authorize(request(model=OTHER_PROVIDER_MODEL))
    assert authorization.served_model == OTHER_PROVIDER_MODEL


# --------------------------------------------------------------------------
# Degrade
# --------------------------------------------------------------------------


async def test_degrading_substitutes_the_cheaper_model_and_records_both() -> None:
    """I2: the artefact must say which model actually answered."""
    ledger = InMemorySpendLedger()
    governor = _governor(
        caps=cap_book(
            cap(daily="0.05", monthly="10.00", policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL)
        ),
        ledger=ledger,
    )

    authorization = await governor.authorize(request())

    assert authorization.degraded is True
    assert authorization.requested_model == PRIMARY_MODEL
    assert authorization.served_model == CHEAP_MODEL
    # The outbound request names the substitute: what is authorized is what is sent.
    assert authorization.request.model == CHEAP_MODEL
    assert authorization.request.prompt == request().prompt
    row = (await ledger.records())[0]
    assert row.event is SpendEvent.RESERVED
    assert (row.requested_model, row.served_model) == (PRIMARY_MODEL, CHEAP_MODEL)
    assert row.degraded is True


async def test_degrading_reserves_the_cheaper_bound() -> None:
    ledger = InMemorySpendLedger()
    governor = _governor(
        caps=cap_book(
            cap(daily="0.05", monthly="10.00", policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL)
        ),
        ledger=ledger,
    )
    primary_only = _governor(caps=cap_book(cap(daily="100", monthly="100")))
    primary_cost = (await primary_only.authorize(request())).estimate.cost

    degraded = await governor.authorize(request())

    assert degraded.estimate.cost < primary_cost
    assert degraded.estimate.model == CHEAP_MODEL


async def test_degrading_to_an_unpriced_model_refuses_rather_than_guessing() -> None:
    governor = _governor(
        caps=cap_book(
            cap(
                daily="0.05",
                monthly="10.00",
                policy=CapPolicy.DEGRADE,
                degrade_to="anthropic:never-priced",
            )
        )
    )
    with pytest.raises(ModelPriceUnknownError):
        await governor.authorize(request())


async def test_degrading_to_the_model_that_just_breached_is_refused() -> None:
    governor = _governor(
        caps=cap_book(
            cap(daily="0.05", monthly="10.00", policy=CapPolicy.DEGRADE, degrade_to=PRIMARY_MODEL)
        )
    )
    with pytest.raises(DegradationUnavailableError, match="fallback to itself"):
        await governor.authorize(request())


async def test_degrading_to_a_model_that_is_not_cheaper_for_this_call_is_refused() -> None:
    """Cheapness is decided per call, not by a headline rate."""
    expensive_fallback = price_book(
        {"anthropic:not-actually-cheaper": price(input_rate="10.00", output_rate="100.00")}
    )
    governor = _governor(
        caps=cap_book(
            cap(
                daily="0.05",
                monthly="10.00",
                policy=CapPolicy.DEGRADE,
                degrade_to="anthropic:not-actually-cheaper",
            )
        ),
        prices=expensive_fallback,
    )
    with pytest.raises(DegradationUnavailableError, match="the substitution does not spend"):
        await governor.authorize(request())


async def test_degrading_still_halts_when_the_cheaper_model_does_not_fit_either() -> None:
    """Halt is the floor of degrade, not an alternative to it."""
    ledger = InMemorySpendLedger()
    governor = _governor(
        caps=cap_book(
            cap(daily="0", monthly="10.00", policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL)
        ),
        ledger=ledger,
    )

    with pytest.raises(SpendCapExceededError):
        await governor.authorize(request())

    assert await ledger.records() == ()


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


async def test_settlement_reconciles_at_the_price_of_the_model_that_answered() -> None:
    """Settling a degraded call at the primary's rate would book a cheap call expensively."""
    ledger = InMemorySpendLedger()
    governor = _governor(
        caps=cap_book(
            cap(daily="0.05", monthly="10.00", policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL)
        ),
        ledger=ledger,
    )
    authorization = await governor.authorize(request())
    response = ModelResponse(text="{}", model=CHEAP_MODEL, input_tokens=1000, output_tokens=1000)

    record = await governor.settle(authorization, response)

    # 1000 tokens at USD 1.00/Mtok plus 1000 at USD 10.00/Mtok = USD 0.011.
    assert record.actual_cost == usd("0.0110000000")
    assert record.reconciled is True
    assert record.served_model == CHEAP_MODEL


async def test_a_failed_call_settles_at_its_bound_rather_than_being_released() -> None:
    """Nothing observable says whether the request left the host; a control assumes the worst."""
    ledger = InMemorySpendLedger()
    governor = _governor(ledger=ledger)
    authorization = await governor.authorize(request())

    record = await governor.settle_failed(authorization)

    assert record.actual_cost is None
    assert record.reconciled is False
    assert record.delta == usd("0")
    assert await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT) == (
        authorization.estimate.cost
    )


async def test_abandoning_gives_headroom_back_for_a_call_that_was_never_sent() -> None:
    ledger = InMemorySpendLedger()
    governor = _governor(ledger=ledger)
    authorization = await governor.authorize(request())

    await governor.abandon(authorization)

    assert await governor.spend_to_date(Provider.ANTHROPIC, SpendWindow.DAILY, at=AT) == usd("0")


async def test_reconciliation_frees_the_pessimism_so_the_next_call_fits() -> None:
    """The whole reason a bound is reconciled: otherwise a cap dies of arithmetic."""
    ledger = InMemorySpendLedger()
    governor = _governor(caps=cap_book(cap(daily="0.20", monthly="10.00")), ledger=ledger)
    call = request(prompt="p" * 100, system="", max_tokens=1000)

    first = await governor.authorize(call)
    # The bound is ~USD 0.111; a second call would not fit against a USD 0.20 cap.
    with pytest.raises(SpendCapExceededError):
        await governor.authorize(call)

    await governor.settle(
        first,
        ModelResponse(text="{}", model=PRIMARY_MODEL, input_tokens=30, output_tokens=20),
    )
    second = await governor.authorize(call)
    assert second.served_model == PRIMARY_MODEL
