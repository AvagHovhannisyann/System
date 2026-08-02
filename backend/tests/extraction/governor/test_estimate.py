"""P7.7: the estimate is an **upper bound**, and prices are never invented.

The properties worth stating, because they are the ones that make a cap real:

* the output term is ``max_tokens``, not an expected length — so no response the
  provider is permitted to return can cost more than was reserved;
* the input term is the UTF-8 byte length plus a declared framing allowance, and
  it is asserted here to dominate any plausible tokenization, including on
  multi-byte text where a naive character count would under-count;
* an unpriced model **raises**. The provider catalog ships no prices, so under
  B4 that refusal is every model's outcome, and a test pins it — if
  ``CATALOG_PRICES`` ever gains a figure that nobody sourced, this fails.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.extraction.governor.errors import CurrencyMismatchError, ModelPriceUnknownError
from backend.extraction.governor.estimate import (
    CATALOG_PRICES,
    MONEY_DECIMAL_PLACES,
    REQUEST_FRAMING_TOKENS,
    TOKENS_PER_PRICE_UNIT,
    Money,
    PriceBook,
    TokenPrice,
    actual_call_cost,
    bound_input_tokens,
    estimate_call_cost,
    provider_of,
    total,
)
from backend.extraction.tasks.client import ModelResponse
from backend.tests.extraction.governor.doubles import (
    CHEAP_MODEL,
    PRIMARY_MODEL,
    price,
    price_book,
    request,
    usd,
)

# --------------------------------------------------------------------------
# Money: exact, currency-carrying, and refusing to convert
# --------------------------------------------------------------------------


def test_money_always_renders_its_currency() -> None:
    """I4: no monetary figure leaves this system as a bare number."""
    assert str(usd("0.0001234")) == "USD 0.0001234"
    assert str(usd("50")) == "USD 50"


def test_money_refuses_a_currency_that_is_not_an_iso_alpha_code() -> None:
    for bad in ("usd", "US$", "", "USDX", "US"):
        with pytest.raises(ValueError, match="ISO 4217"):
            Money(Decimal("1"), bad)


def test_money_refuses_more_precision_than_it_can_store() -> None:
    """Refused rather than rounded: only the caller knows which way to err."""
    too_precise = Decimal(1).scaleb(-(MONEY_DECIMAL_PLACES + 1))
    with pytest.raises(ValueError, match="decimal places"):
        Money(too_precise, "USD")


def test_money_refuses_non_finite_amounts() -> None:
    for bad in (Decimal("NaN"), Decimal("Infinity")):
        with pytest.raises(ValueError, match="finite"):
            Money(bad, "USD")


def test_money_refuses_to_mix_currencies_rather_than_converting() -> None:
    """There is no exchange rate in this system, so there is no conversion."""
    with pytest.raises(CurrencyMismatchError, match="no exchange rate"):
        _ = usd("1") + Money(Decimal("1"), "EUR")
    with pytest.raises(CurrencyMismatchError):
        _ = usd("1") >= Money(Decimal("1"), "EUR")


def test_total_of_nothing_is_zero_in_the_stated_currency() -> None:
    assert total([], currency="USD") == usd("0")
    assert total([usd("0.25"), usd("0.75")], currency="USD") == usd("1.00")


# --------------------------------------------------------------------------
# The bound
# --------------------------------------------------------------------------


def test_input_bound_is_byte_length_plus_the_declared_framing_allowance() -> None:
    assert bound_input_tokens("abcd") == 4 + REQUEST_FRAMING_TOKENS
    assert bound_input_tokens("abcd", framing_tokens=0) == 4


def test_input_bound_counts_bytes_not_characters() -> None:
    """A character count would under-count multi-byte text — and under-counting is the leak.

    ``"€"`` is one character and three UTF-8 bytes. A tokenizer can spend up to
    one token per byte, so only the byte count is a bound.
    """
    assert bound_input_tokens("€" * 10, framing_tokens=0) == 30


@settings(max_examples=200)
@given(st.text(min_size=0, max_size=500))
def test_input_bound_never_falls_below_any_possible_token_count(text: str) -> None:
    """The bound dominates the best case a byte-level BPE tokenizer could achieve.

    No such tokenizer can emit more tokens than the input has bytes, because
    every token covers at least one byte. This asserts the bound is at least the
    byte count for arbitrary text, which is the property the argument rests on.
    """
    assert bound_input_tokens(text, framing_tokens=0) >= len(text.encode("utf-8"))


def test_bound_refuses_a_negative_framing_allowance() -> None:
    """An allowance that subtracts is a discount, and a discount is a leak."""
    with pytest.raises(ValueError, match="not be negative"):
        bound_input_tokens("abc", framing_tokens=-1)


def test_estimate_uses_max_tokens_for_output_not_an_expectation() -> None:
    """The output term is exactly what the provider is permitted to return."""
    call = request(prompt="p" * 90, system="s" * 10, max_tokens=1000)
    estimate = estimate_call_cost(call, price(input_rate="10.00", output_rate="100.00"))

    assert estimate.output_tokens_bound == 1000
    assert estimate.input_tokens_bound == 100 + REQUEST_FRAMING_TOKENS


def test_estimate_arithmetic_is_checkable_by_hand() -> None:
    """input_bound * input_rate / 1e6 + max_tokens * output_rate / 1e6, rounded up."""
    call = request(prompt="p" * 90, system="s" * 10, max_tokens=1000)
    estimate = estimate_call_cost(
        call, price(input_rate="10.00", output_rate="100.00"), framing_tokens=0
    )

    expected_input = Decimal(100) * Decimal("10.00") / Decimal(TOKENS_PER_PRICE_UNIT)
    expected_output = Decimal(1000) * Decimal("100.00") / Decimal(TOKENS_PER_PRICE_UNIT)
    assert estimate.cost == Money(
        (expected_input + expected_output).quantize(Decimal(1).scaleb(-MONEY_DECIMAL_PLACES)),
        "USD",
    )


def test_estimate_rounds_up_so_the_bound_is_never_undercut() -> None:
    """A sub-quantum cost rounds to the quantum, not to zero.

    One token at a rate that produces less than 1e-10 must still cost something:
    a call that rounds to free is a call a cap cannot see.
    """
    call = request(prompt="x", system="", max_tokens=1)
    tiny = price(input_rate="0.000001", output_rate="0.000001")
    estimate = estimate_call_cost(call, tiny, framing_tokens=0)

    assert estimate.cost.amount > 0


def test_estimate_can_be_recorded_against_a_model_other_than_the_requests() -> None:
    """The degrade path prices a substitute before it has built the substituted request."""
    estimate = estimate_call_cost(
        request(), price(input_rate="1.00", output_rate="1.00"), model=CHEAP_MODEL
    )
    assert estimate.model == CHEAP_MODEL


# --------------------------------------------------------------------------
# Reconciliation
# --------------------------------------------------------------------------


def test_actual_cost_uses_provider_reported_counts() -> None:
    response = ModelResponse(text="{}", model=PRIMARY_MODEL, input_tokens=100, output_tokens=50)
    actual = actual_call_cost(response, price(input_rate="10.00", output_rate="100.00"))

    expected = (
        Decimal(100) * Decimal("10.00") + Decimal(50) * Decimal("100.00")
    ) / Decimal(TOKENS_PER_PRICE_UNIT)
    assert actual == Money(expected.quantize(Decimal(1).scaleb(-MONEY_DECIMAL_PLACES)), "USD")


def test_actual_cost_is_unknown_rather_than_estimated_when_the_provider_reports_nothing() -> None:
    """I3: an estimated actual is a fabricated measurement."""
    for input_tokens, output_tokens in ((None, 50), (100, None), (None, None)):
        response = ModelResponse(
            text="{}",
            model=PRIMARY_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        assert actual_call_cost(response, price(input_rate="1", output_rate="1")) is None


def test_the_actual_cost_of_a_real_response_is_below_its_bound() -> None:
    """The bound is loose in the safe direction — that is what makes it a bound."""
    call = request(prompt="p" * 400, system="s" * 100, max_tokens=1000)
    rates = price(input_rate="10.00", output_rate="100.00")
    estimate = estimate_call_cost(call, rates)
    # A realistic outcome: ~4 bytes per token on the prompt, a short structured
    # answer well under the response cap.
    response = ModelResponse(text="{}", model=PRIMARY_MODEL, input_tokens=125, output_tokens=40)

    actual = actual_call_cost(response, rates)
    assert actual is not None
    assert actual < estimate.cost


# --------------------------------------------------------------------------
# Prices come from the catalog, and the catalog has none (I3, B4)
# --------------------------------------------------------------------------


def test_the_catalog_ships_no_prices() -> None:
    """B4: rates are a human's decision, and an invented one is fabricated data."""
    assert dict(CATALOG_PRICES) == {}


def test_the_catalog_price_book_refuses_every_model() -> None:
    book = PriceBook.from_catalog()
    with pytest.raises(ModelPriceUnknownError, match="the catalog ships no prices"):
        book.price_for(PRIMARY_MODEL)


def test_an_unpriced_model_raises_rather_than_defaulting() -> None:
    """No average, no cheapest, no zero: each would authorize against a number nobody chose."""
    with pytest.raises(ModelPriceUnknownError, match="cannot be bounded"):
        price_book().price_for("anthropic:never-configured")


def test_a_price_may_not_mix_currencies_or_go_negative() -> None:
    with pytest.raises(CurrencyMismatchError):
        TokenPrice(
            input_per_million_tokens=usd("1"),
            output_per_million_tokens=Money(Decimal("1"), "EUR"),
        )
    with pytest.raises(ValueError, match="must not be negative"):
        TokenPrice(
            input_per_million_tokens=usd("-1"),
            output_per_million_tokens=usd("1"),
        )


def test_cheaper_of_compares_output_rates() -> None:
    assert price_book().cheaper_of(PRIMARY_MODEL, CHEAP_MODEL) == CHEAP_MODEL
    assert price_book().cheaper_of(CHEAP_MODEL, PRIMARY_MODEL) == CHEAP_MODEL


# --------------------------------------------------------------------------
# Qualified model identifiers
# --------------------------------------------------------------------------


def test_provider_is_read_from_the_qualifier_never_guessed() -> None:
    assert provider_of("anthropic:some-model") == "anthropic"
    for unqualified in ("some-model", ":some-model", ""):
        with pytest.raises(ValueError, match="not qualified"):
            provider_of(unqualified)
