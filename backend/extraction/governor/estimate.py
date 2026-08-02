"""Money, prices, and the **upper bound** on what a call will cost (P7.7, I3, I4).

The governor enforces a cap *before* the call, and that ordering decides
everything in this module. At the moment of enforcement the response does not
exist: nobody knows how many tokens the model will emit, and nobody knows how
the provider will count the prompt. So the number the cap is compared against
cannot be an expectation. It has to be a **bound the call cannot exceed**,
because a cap enforced against an expectation leaks by exactly the amount the
expectation was optimistic, on exactly the calls that were unusually expensive
— which is to say, on the calls the cap existed for.

Everything here is therefore biased in one direction, deliberately:

* **Output tokens are bounded by ``max_tokens``, not by an expected length.**
  A provider cannot return more than the cap the request carries, so this bound
  is exact rather than pessimistic — it is the one place where "upper bound" and
  "the truth" coincide. Extraction returns a small schema-validated object, so
  the realised output will normally be far below it; that gap is recovered by
  reconciliation (see below), not by guessing a smaller number here.
* **Input tokens are bounded by the UTF-8 byte length of what is sent**, plus a
  declared framing allowance. See :data:`REQUEST_FRAMING_TOKENS` and
  :func:`bound_input_tokens` for why the byte length is a bound and not an
  estimate, and what it assumes about the tokenizer.
* **Rounding is upward.** :func:`_cost_of_tokens` quantizes with
  ``ROUND_CEILING``. At 10 decimal places this changes nothing anyone can see;
  it is written that way so that no future reader has to wonder which direction
  the rounding error goes.

Reconciliation, and why the bound is not the end of the story
--------------------------------------------------------------

An upper bound that is never corrected is a cap that shrinks: a day of calls
each reserving 1024 output tokens and using 40 would exhaust a budget on
arithmetic rather than on spending. So the bound is a *reservation*, and after
the response arrives the reservation is settled against
:func:`actual_call_cost`, computed from the token counts **the provider
reported**. The difference returns to the window's headroom.

When the provider reports no token counts, :func:`actual_call_cost` returns
``None`` and the reservation settles at the bound. It does not fall back to an
estimate: an estimated actual is a fabricated measurement (I3), and
:attr:`~backend.extraction.tasks.client.ModelResponse.input_tokens` already
documents that these counts are never estimated. Settling at the bound
over-counts, which is the safe direction — an over-counted cap refuses calls
that would have fit, an under-counted one lets real money out.

Prices come from the catalog, or the governor refuses (I3)
-----------------------------------------------------------

:data:`CATALOG_PRICES` is the price source, and it is **empty**.
:mod:`backend.extraction.providers.catalog` ships no pricing on purpose — B4
leaves the provider decision and the negotiated rates with a human, and a figure
written here today would be an invented number that a reader would reasonably
treat as fact. So :meth:`PriceBook.price_for` raises
:class:`~backend.extraction.governor.errors.ModelPriceUnknownError` for every
model until an operator configures one. That refusal is the feature: a cap
enforced against a made-up price enforces nothing, and it looks exactly like a
cap that works.

Units and assumptions
---------------------

* :class:`Money` — an exact :class:`~decimal.Decimal` amount in the **major
  unit** of a named ISO-4217 currency (dollars, not cents), carried at
  :data:`MONEY_DECIMAL_PLACES` decimal places. Never a float: binary floating
  point cannot represent a tenth of a cent, and a cap is a comparison. Never a
  bare number: every amount states its currency (I4).
* :class:`TokenPrice` — currency per **1,000,000 tokens**, quoted separately for
  input and output, which is how every provider publishes them. Per-million
  rather than per-token so the configured figure is the figure on the vendor's
  price page, unscaled, and a transcription error is visible.
* Token counts are dimensionless counts of provider tokens.
* No exchange rates exist anywhere in this system. Combining two currencies
  raises (:class:`~backend.extraction.governor.errors.CurrencyMismatchError`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from backend.extraction.governor.errors import (
    CurrencyMismatchError,
    ModelPriceUnknownError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from backend.extraction.tasks.client import ModelRequest, ModelResponse

__all__ = [
    "CATALOG_PRICES",
    "MONEY_DECIMAL_PLACES",
    "REQUEST_FRAMING_TOKENS",
    "TOKENS_PER_PRICE_UNIT",
    "CallEstimate",
    "Money",
    "PriceBook",
    "TokenPrice",
    "actual_call_cost",
    "bound_input_tokens",
    "estimate_call_cost",
    "provider_of",
    "total",
]

MONEY_DECIMAL_PLACES: Final = 10
"""Decimal places every :class:`Money` amount is held at (dimensionless).

Ten, matching ``NUMERIC(20, 10)`` in migration 0013, so a value cannot change
by being written down and read back. Cost-tier models are priced in fractions
of a cent per call — a two-decimal money type would round a real call to zero
and a cap would then be enforced against nothing.
"""

_MONEY_QUANTUM: Final = Decimal(1).scaleb(-MONEY_DECIMAL_PLACES)
"""The smallest representable amount, ``1E-10`` of the currency's major unit."""

_CURRENCY_PATTERN: Final = re.compile(r"^[A-Z]{3}$")
"""ISO 4217 alphabetic code shape: exactly three uppercase letters.

A shape check, not a list of currencies. Enumerating the world's currencies here
would be a table nobody maintains; refusing ``"usd"``, ``"US$"`` and ``""``
catches the mistakes that actually happen, and two spellings of one currency
would otherwise become two budgets.
"""

TOKENS_PER_PRICE_UNIT: Final = 1_000_000
"""Tokens a :class:`TokenPrice` figure is quoted per: one million.

The unit every provider publishes in. Storing the vendor's own number unscaled
means a transcription error is visible by comparing against the price page,
rather than hidden inside a division somebody did once.
"""

REQUEST_FRAMING_TOKENS: Final = 1024
"""Allowance added to the input bound for the provider's message envelope (tokens).

A **margin, not a measurement.** :func:`bound_input_tokens` bounds the text this
system sends; a provider additionally spends tokens on role markers, tool
plumbing and whatever else its API wraps a message in, and that count is not
observable from here. So a fixed allowance is added, chosen to sit far above any
plausible envelope for a single system-plus-user message.

It is declared as a margin rather than dressed up as a per-provider constant
because an invented per-provider figure would be a fabricated fact (I3), and
because a margin can only be wrong in one direction that matters: too large
refuses a call that would have fit, too small lets a cap leak. It is a
:data:`typing.Final` constant rather than a parameter with a default so that
lowering it is an edit to this line, read next to this paragraph.
"""

_MODEL_QUALIFIER: Final = ":"
"""Separator in a qualified model identifier, matching
:func:`backend.extraction.tasks.pipeline.qualified_model`."""


@dataclass(frozen=True, slots=True)
class Money:
    """An exact amount of one currency. Never a bare number (I4).

    Attributes:
        amount: the amount in the currency's **major unit** (dollars, not
            cents), as an exact :class:`~decimal.Decimal` with at most
            :data:`MONEY_DECIMAL_PLACES` decimal places. May be negative: the
            spend ledger books a settlement as a signed delta against its
            reservation.
        currency: ISO-4217 alphabetic code, uppercase, e.g. ``"USD"``.

    Arithmetic between two currencies raises rather than converting. This system
    holds no exchange rate and has no business inventing one — a cap in one
    currency enforced against a price in another is a cap enforced against an
    unstated conversion.
    """

    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        """Reject an amount or currency that could not be compared or stored.

        Raises:
            ValueError: the currency is not three uppercase letters, the amount
                is not finite (``NaN``/``Infinity`` compare in ways no cap check
                should depend on), or the amount carries more than
                :data:`MONEY_DECIMAL_PLACES` decimal places. The last is
                refused rather than rounded here: rounding is a decision about
                which direction to err, and the caller is the only one who knows
                whether this amount is a cost (round up) or a limit (round
                down).
        """
        if not _CURRENCY_PATTERN.match(self.currency):
            msg = (
                f"currency must be a three-letter uppercase ISO 4217 code; got "
                f"{self.currency!r}. Two spellings of one currency would become two budgets"
            )
            raise ValueError(msg)
        if not self.amount.is_finite():
            msg = f"amount must be a finite decimal; got {self.amount!r}"
            raise ValueError(msg)
        exponent = self.amount.as_tuple().exponent
        if not isinstance(exponent, int) or -exponent > MONEY_DECIMAL_PLACES:
            msg = (
                f"amount {self.amount!r} carries more than {MONEY_DECIMAL_PLACES} decimal "
                "places, which cannot be stored exactly. Quantize it deliberately — upward "
                "for a cost, downward for a limit — rather than letting this module choose"
            )
            raise ValueError(msg)

    @classmethod
    def zero(cls, currency: str) -> Money:
        """Return zero in *currency*.

        Args:
            currency: ISO-4217 alphabetic code.

        Returns:
            ``Money(Decimal(0), currency)``.
        """
        return cls(Decimal(0), currency)

    def zero_like(self) -> Money:
        """Return zero in this amount's currency."""
        return Money.zero(self.currency)

    def __add__(self, other: Money) -> Money:
        """Return the sum, or raise if the currencies differ."""
        _require_same_currency(self, other)
        return Money(self.amount + other.amount, self.currency)

    def __sub__(self, other: Money) -> Money:
        """Return the difference, or raise if the currencies differ."""
        _require_same_currency(self, other)
        return Money(self.amount - other.amount, self.currency)

    def __neg__(self) -> Money:
        """Return the amount with its sign flipped."""
        return Money(-self.amount, self.currency)

    def __lt__(self, other: Money) -> bool:
        """Return whether this amount is strictly smaller (same currency only)."""
        _require_same_currency(self, other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        """Return whether this amount is smaller or equal (same currency only)."""
        _require_same_currency(self, other)
        return self.amount <= other.amount

    def __gt__(self, other: Money) -> bool:
        """Return whether this amount is strictly greater (same currency only)."""
        _require_same_currency(self, other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        """Return whether this amount is greater or equal (same currency only)."""
        _require_same_currency(self, other)
        return self.amount >= other.amount

    def __str__(self) -> str:
        """Render as ``"USD 0.0001234"`` — the currency is never omitted (I4)."""
        return f"{self.currency} {self.amount:f}"


def _require_same_currency(left: Money, right: Money) -> None:
    """Refuse a mixed-currency operation.

    Args:
        left: one amount.
        right: the other.

    Raises:
        backend.extraction.governor.errors.CurrencyMismatchError: the two
            amounts are in different currencies. Raised rather than converted —
            this system holds no exchange rate and has no business inventing
            one (I4).
    """
    if left.currency != right.currency:
        msg = (
            f"cannot combine or compare {left} with {right}: different currencies, and this "
            "system holds no exchange rate. Configure the cap and the price in the same currency"
        )
        raise CurrencyMismatchError(msg)


def total(amounts: Iterable[Money], *, currency: str) -> Money:
    """Return the sum of *amounts*, or zero in *currency* when there are none.

    Args:
        amounts: the amounts to add. Every one must be in *currency*.
        currency: the currency of the result, required because an empty sum has
            no amount to take a currency from — and a zero without a currency is
            the bare number I4 forbids.

    Returns:
        The sum, in *currency*.

    Raises:
        backend.extraction.governor.errors.CurrencyMismatchError: any amount is
            in a different currency.
    """
    running = Money.zero(currency)
    for amount in amounts:
        running = running + amount
    return running


def _cost_of_tokens(tokens: int, price_per_million: Money) -> Money:
    """Return the cost of *tokens* at *price_per_million*, **rounded up**.

    Args:
        tokens: token count (dimensionless, >= 0).
        price_per_million: currency per :data:`TOKENS_PER_PRICE_UNIT` tokens.

    Returns:
        ``tokens / 1_000_000 * price_per_million``, quantized to
        :data:`MONEY_DECIMAL_PLACES` places with ``ROUND_CEILING``. Upward,
        always: this figure is compared against a cap, and the only rounding
        error a cap can afford is one that refuses slightly too often.
    """
    exact = price_per_million.amount * Decimal(tokens) / Decimal(TOKENS_PER_PRICE_UNIT)
    return Money(exact.quantize(_MONEY_QUANTUM, rounding=ROUND_CEILING), price_per_million.currency)


@dataclass(frozen=True, slots=True)
class TokenPrice:
    """What one model costs, as the provider publishes it.

    Attributes:
        input_per_million_tokens: cost of :data:`TOKENS_PER_PRICE_UNIT` prompt
            tokens.
        output_per_million_tokens: cost of :data:`TOKENS_PER_PRICE_UNIT`
            response tokens.

    Both in the same currency; a price quoting its two halves in different
    currencies is not a price this system can add up.
    """

    input_per_million_tokens: Money
    output_per_million_tokens: Money

    def __post_init__(self) -> None:
        """Reject a price that is negative or mixes currencies.

        Raises:
            ValueError: either figure is negative. A negative price would make
                spending *increase* headroom.
            backend.extraction.governor.errors.CurrencyMismatchError: the two
                figures are in different currencies.
        """
        _require_same_currency(self.input_per_million_tokens, self.output_per_million_tokens)
        for label, figure in (
            ("input_per_million_tokens", self.input_per_million_tokens),
            ("output_per_million_tokens", self.output_per_million_tokens),
        ):
            if figure.amount < 0:
                msg = f"{label} must not be negative; got {figure}"
                raise ValueError(msg)

    @property
    def currency(self) -> str:
        """ISO-4217 code both figures are quoted in."""
        return self.input_per_million_tokens.currency


CATALOG_PRICES: Final[Mapping[str, TokenPrice]] = MappingProxyType({})
"""Model prices taken from the provider catalog, keyed by qualified model id.

**Empty, and that is the honest state of this repository.**
:mod:`backend.extraction.providers.catalog` documents why it ships no pricing:
§5-P7 requires cost-tier rather than frontier models, B4 leaves both the
provider choice and the rates with a human, and any figure written into a module
whose readers would treat it as fact is a fabricated one (I3).

It is spelled out as an empty mapping rather than left absent so that "where do
prices come from" has exactly one answer, and so a test can assert the answer is
still "nowhere" — which is what keeps
:class:`~backend.extraction.governor.errors.ModelPriceUnknownError` the expected
outcome rather than a surprise.
"""


@dataclass(frozen=True, slots=True)
class PriceBook:
    """The prices the governor is allowed to enforce against.

    Attributes:
        prices: qualified model identifier
            (:func:`~backend.extraction.tasks.pipeline.qualified_model`, e.g.
            ``"anthropic:some-cost-tier-model"``) to its
            :class:`TokenPrice`.

    A missing model is a refusal, never a default. There is no "average price",
    no "cheapest configured price", and no zero fallback: each would let a call
    be authorized against a number nobody chose, and the resulting cap would
    look like it was working.
    """

    prices: Mapping[str, TokenPrice]

    @classmethod
    def from_catalog(cls) -> PriceBook:
        """Return the price book the provider catalog supplies.

        Returns:
            A book over :data:`CATALOG_PRICES` — which is empty, so every
            :meth:`price_for` raises until an operator configures prices (B4).
        """
        return cls(CATALOG_PRICES)

    def price_for(self, model: str) -> TokenPrice:
        """Return the price of *model*, or refuse.

        Args:
            model: qualified model identifier, matched exactly.

        Returns:
            The configured :class:`TokenPrice`.

        Raises:
            backend.extraction.governor.errors.ModelPriceUnknownError: no price
                is configured for *model*. The message lists what is configured,
                because a caller that got this wrong needs to know what would
                have been right.
        """
        price = self.prices.get(model)
        if price is None:
            known = ", ".join(sorted(self.prices)) or "(nothing — the catalog ships no prices, B4)"
            msg = (
                f"no price is configured for model {model!r}, so the cost of a call to it "
                f"cannot be bounded and no cap can be enforced against it (I3). Configured: "
                f"{known}"
            )
            raise ModelPriceUnknownError(msg)
        return price

    def cheaper_of(self, first: str, second: str) -> str:
        """Return whichever of two models has the lower output price.

        Used only for reporting and configuration checks; the degrade decision
        compares whole-call estimates rather than headline rates, because which
        model is cheaper *for a given call* depends on the prompt-to-response
        ratio.

        Args:
            first: qualified model identifier.
            second: qualified model identifier.

        Returns:
            The identifier with the lower output price per million tokens; the
            first when they are equal.

        Raises:
            backend.extraction.governor.errors.ModelPriceUnknownError: either
                model is unpriced.
            backend.extraction.governor.errors.CurrencyMismatchError: the two
                prices are in different currencies.
        """
        first_price = self.price_for(first).output_per_million_tokens
        second_price = self.price_for(second).output_per_million_tokens
        return second if second_price < first_price else first


@dataclass(frozen=True, slots=True)
class CallEstimate:
    """The upper bound on one call's cost, and the counts it was derived from.

    Attributes:
        model: qualified model identifier the bound was computed for.
        input_tokens_bound: upper bound on prompt tokens (count) —
            :func:`bound_input_tokens`.
        output_tokens_bound: upper bound on response tokens (count), which is
            the request's ``max_tokens`` and is therefore exact.
        cost: the bound, in the price's currency. **Not an expectation.**

    Carried rather than recomputed so the ledger row, the refusal message and
    the operator's gauge all quote the same arithmetic, and so a reconciliation
    ratio has a denominator that is not a guess about how the bound was reached.
    """

    model: str
    input_tokens_bound: int
    output_tokens_bound: int
    cost: Money


def bound_input_tokens(text: str, *, framing_tokens: int = REQUEST_FRAMING_TOKENS) -> int:
    """Return an upper bound on the tokens *text* will cost as a prompt.

    The bound is ``len(text.encode("utf-8")) + framing_tokens``.

    **Why the byte length is a bound rather than an estimate.** The providers
    this platform can talk to (:class:`~backend.extraction.providers.catalog.Provider`)
    tokenize with byte-level BPE. Every token in such a vocabulary covers at
    least one byte of the input, so a string of *n* UTF-8 bytes cannot tokenize
    to more than *n* tokens. That is the whole argument, and it is an argument
    about the tokenizer family — not a ratio fitted to sample text.

    The bound is loose: real English runs about 3.5 to 4 bytes per token, so
    this over-states by roughly 4x. Loose in the safe direction, and the
    looseness is recovered by reconciliation
    (:func:`actual_call_cost`) rather than by tightening the bound with a
    fudge factor. **A ratio-based estimate would be the leak this module
    exists to prevent**: it would be right on average and wrong on exactly the
    inputs that are unusually token-dense, which are the expensive ones. The
    only honest way to tighten this is a real tokenizer for the model in
    question; until one is wired in, the byte count is what can be defended.

    Args:
        text: everything sent as prompt material — system instruction and
            user message concatenated by the caller.
        framing_tokens: allowance for the provider's message envelope
            (:data:`REQUEST_FRAMING_TOKENS`). Must not be negative.

    Returns:
        An upper bound on the prompt's token count (dimensionless).

    Raises:
        ValueError: *framing_tokens* is negative, which would turn an allowance
            into a discount.
    """
    if framing_tokens < 0:
        msg = f"framing_tokens must not be negative; got {framing_tokens}"
        raise ValueError(msg)
    return len(text.encode("utf-8")) + framing_tokens


def estimate_call_cost(
    request: ModelRequest,
    price: TokenPrice,
    *,
    framing_tokens: int = REQUEST_FRAMING_TOKENS,
    model: str | None = None,
) -> CallEstimate:
    """Return the **upper bound** on what *request* will cost at *price*.

    ``cost = bound_input_tokens(system + prompt) * input_rate
             + request.max_tokens * output_rate``

    with both terms rounded up (:func:`_cost_of_tokens`). Neither term is an
    expectation: the output term is the largest response the provider is
    permitted to return, and the input term is a bound on the prompt
    (:func:`bound_input_tokens`).

    Args:
        request: the call that has not been made yet.
        price: the model's published rates.
        framing_tokens: envelope allowance added to the input bound.
        model: qualified model identifier to record on the estimate, when it
            differs from ``request.model`` — the degrade path prices a
            substitute before it has built the substituted request. Defaults to
            ``request.model``.

    Returns:
        The :class:`CallEstimate`.

    Raises:
        ValueError: *framing_tokens* is negative.
    """
    input_bound = bound_input_tokens(request.system + request.prompt, framing_tokens=framing_tokens)
    output_bound = request.max_tokens
    cost = _cost_of_tokens(input_bound, price.input_per_million_tokens) + _cost_of_tokens(
        output_bound, price.output_per_million_tokens
    )
    return CallEstimate(
        model=model if model is not None else request.model,
        input_tokens_bound=input_bound,
        output_tokens_bound=output_bound,
        cost=cost,
    )


def actual_call_cost(response: ModelResponse, price: TokenPrice) -> Money | None:
    """Return what the call actually cost, or ``None`` when it cannot be known.

    Computed from the token counts **the provider reported**
    (:attr:`~backend.extraction.tasks.client.ModelResponse.input_tokens`), never
    from a re-estimate: an estimated actual is a fabricated measurement (I3),
    and it would silently replace the one number in this module that is not a
    bound.

    Args:
        response: what came back.
        price: the rates of the model that actually served the call — which is
            the substituted model when the governor degraded, not the one the
            caller asked for.

    Returns:
        The cost, rounded up, or ``None`` when the provider reported no input or
        no output token count. ``None`` means "unknown", and the caller settles
        at the reserved bound rather than inventing a figure.
    """
    if response.input_tokens is None or response.output_tokens is None:
        return None
    return _cost_of_tokens(response.input_tokens, price.input_per_million_tokens) + _cost_of_tokens(
        response.output_tokens, price.output_per_million_tokens
    )


def provider_of(model: str) -> str:
    """Return the provider-name half of a qualified model identifier.

    Args:
        model: ``"provider:model"`` as built by
            :func:`~backend.extraction.tasks.pipeline.qualified_model`.

    Returns:
        The text before the first ``":"``.

    Raises:
        ValueError: *model* carries no qualifier. Unqualified is refused rather
            than guessed: the qualifier names whose budget is about to be spent,
            and a governor that guesses that would charge one provider's cap for
            another provider's call.
    """
    head, separator, _ = model.partition(_MODEL_QUALIFIER)
    if not separator or not head:
        msg = (
            f"model {model!r} is not qualified with a provider; expected "
            f"'provider{_MODEL_QUALIFIER}model'. The qualifier names whose cap this call "
            "spends, and there is no safe way to guess it"
        )
        raise ValueError(msg)
    return head
