"""Shared test doubles and fixtures-in-function-form for the P7.7 governor tests.

**Nothing here pretends to be a provider.** Every model response below is canned
text supplied by the test that asks for it, and every price is a figure invented
*for a test* — chosen to make arithmetic checkable by hand, not transcribed from
anyone's price page. That distinction is the whole reason
:data:`~backend.extraction.governor.estimate.CATALOG_PRICES` is empty in the
shipped code: a number in a test file is scaffolding, and the same number in a
module a reader treats as fact is fabricated data (I3).

Currency is ``USD`` throughout, stated on every amount, never a bare number (I4).
"""

from __future__ import annotations

import asyncio
import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, Final

from backend.extraction.governor.caps import CapBook, CapPolicy, ProviderCap
from backend.extraction.governor.estimate import Money, PriceBook, TokenPrice
from backend.extraction.providers.catalog import Provider
from backend.extraction.tasks.client import ModelCallError, ModelRequest, ModelResponse

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

CURRENCY: Final = "USD"
"""The currency every amount in these tests is denominated in."""

PRIMARY_MODEL: Final = "anthropic:test-primary"
"""A qualified model identifier for the tests. Names no real deployment (B4)."""

CHEAP_MODEL: Final = "anthropic:test-cheap"
"""A second identifier, priced lower, used as the degrade target."""

OTHER_PROVIDER_MODEL: Final = "openai:test-other"
"""A model on a different provider, for the cross-provider refusals."""

AT = dt.datetime(2026, 8, 2, 12, 0, tzinfo=dt.UTC)
"""A fixed authorization instant, so window keys in assertions are literals."""


def usd(amount: str) -> Money:
    """Return *amount* as USD.

    Args:
        amount: decimal string, e.g. ``"0.50"``. A string rather than a float so
            the value is exactly what it reads as.

    Returns:
        The :class:`~backend.extraction.governor.estimate.Money`.
    """
    return Money(Decimal(amount), CURRENCY)


def price(*, input_rate: str, output_rate: str) -> TokenPrice:
    """Return a token price in USD per million tokens.

    Args:
        input_rate: USD per 1,000,000 prompt tokens.
        output_rate: USD per 1,000,000 response tokens.

    Returns:
        The :class:`~backend.extraction.governor.estimate.TokenPrice`.
    """
    return TokenPrice(
        input_per_million_tokens=usd(input_rate),
        output_per_million_tokens=usd(output_rate),
    )


def price_book(extra: Mapping[str, TokenPrice] | None = None) -> PriceBook:
    """Return a price book covering the primary and the cheap model.

    The rates are round numbers so a reader can verify an estimate by hand:
    the primary costs ten times the cheap model on both halves.

    Args:
        extra: further models to price, or ``None``.

    Returns:
        The :class:`~backend.extraction.governor.estimate.PriceBook`.
    """
    prices: dict[str, TokenPrice] = {
        PRIMARY_MODEL: price(input_rate="10.00", output_rate="100.00"),
        CHEAP_MODEL: price(input_rate="1.00", output_rate="10.00"),
    }
    if extra is not None:
        prices.update(extra)
    return PriceBook(prices)


def cap(
    *,
    daily: str = "1.00",
    monthly: str = "10.00",
    policy: CapPolicy = CapPolicy.HALT,
    degrade_to: str | None = None,
    provider: Provider = Provider.ANTHROPIC,
) -> ProviderCap:
    """Return one provider cap.

    Args:
        daily: daily limit as a USD decimal string.
        monthly: monthly limit as a USD decimal string.
        policy: halt or degrade.
        degrade_to: qualified fallback model, required when *policy* degrades.
        provider: whose cap this is.

    Returns:
        The :class:`~backend.extraction.governor.caps.ProviderCap`.
    """
    return ProviderCap(
        provider=provider,
        daily_limit=usd(daily),
        monthly_limit=usd(monthly),
        policy=policy,
        degrade_to=degrade_to,
    )


def cap_book(*caps: ProviderCap) -> CapBook:
    """Return a cap book, defaulting to a single halting anthropic cap.

    Args:
        *caps: the caps to configure. Empty means the default single cap —
            spelled out rather than left to
            :meth:`~backend.extraction.governor.caps.CapBook.of`, which refuses
            an empty book (B4).

    Returns:
        The :class:`~backend.extraction.governor.caps.CapBook`.
    """
    return CapBook.of(*(caps if caps else (cap(),)))


def request(
    *,
    model: str = PRIMARY_MODEL,
    prompt: str = "x" * 100,
    system: str = "s" * 10,
    max_tokens: int = 1000,
) -> ModelRequest:
    """Return a model request with hand-checkable sizes.

    Args:
        model: qualified model identifier.
        prompt: the anonymized user message. ASCII, so its UTF-8 byte length is
            its character count and the input bound is readable by inspection.
        system: the system instruction, ASCII for the same reason.
        max_tokens: response cap in tokens — the exact output-token bound.

    Returns:
        The :class:`~backend.extraction.tasks.client.ModelRequest`.
    """
    return ModelRequest(model=model, system=system, prompt=prompt, max_tokens=max_tokens)


def clock(at: dt.datetime = AT) -> Callable[[], dt.datetime]:
    """Return a clock function pinned to *at*.

    Args:
        at: the instant every call should report.

    Returns:
        A zero-argument callable returning *at*.
    """

    def _fixed() -> dt.datetime:
        return at

    return _fixed


class RecordingClient:
    """A model client that records every request and replays canned responses.

    The canned text is this test suite's input, not a provider's output. Its
    purpose is to make *what was sent* — and whether anything was sent at all —
    observable.
    """

    def __init__(
        self,
        *,
        text: str = "{}",
        input_tokens: int | None = 100,
        output_tokens: int | None = 50,
        delay: bool = False,
    ) -> None:
        """Build the double.

        Args:
            text: the response body replayed for every call.
            input_tokens: prompt tokens the double reports, or ``None`` to
                report none — which is how a provider that omits usage data is
                exercised.
            output_tokens: response tokens the double reports, or ``None``.
            delay: yield to the event loop before answering, so concurrent
                callers genuinely interleave around the call.
        """
        self.requests: list[ModelRequest] = []
        self._text = text
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._delay = delay

    @property
    def calls(self) -> int:
        """How many requests this client was handed (count)."""
        return len(self.requests)

    @property
    def models(self) -> tuple[str, ...]:
        """The model identifier of every request received, in order."""
        return tuple(request.model for request in self.requests)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request and return the canned response."""
        self.requests.append(request)
        if self._delay:
            await asyncio.sleep(0)
        return ModelResponse(
            text=self._text,
            model=request.model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=1.5,
        )


class NeverCalledClient:
    """A client that fails the test if it is ever called.

    Used wherever the governor must refuse **before** reaching the provider. A
    test that only checked the raised exception would pass just as happily if
    the request had been sent and the response thrown away, so the assertion has
    to live on this side of the seam.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail loudly."""
        msg = (
            f"the governor reached the model client, which it must not have: {request.model!r}. "
            "A cap that refuses after the request has left is an accounting system"
        )
        raise AssertionError(msg)


class FailingClient:
    """A client whose calls always raise, and which counts its attempts."""

    def __init__(self, message: str = "connection reset") -> None:
        """Build the double.

        Args:
            message: the failure text carried by the raised error.
        """
        self.calls = 0
        self._message = message

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the attempt and raise.

        Raises:
            backend.extraction.tasks.client.ModelCallError: always.
        """
        self.calls += 1
        raise ModelCallError(f"{self._message} calling {request.model!r}")
