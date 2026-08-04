"""The model call, as an interface the pipeline is given (P7.3, B4).

There are no provider keys (B4 is unresolved), so the extraction pipeline is
built against this interface and never against a vendor SDK. That is not a
workaround: the call is the one part of the pipeline that costs money, is
non-deterministic in wall-clock terms and can fail for reasons outside the
system, and injecting it means every other property — chunking, anonymization,
schema validation, cache addressing — is testable without spending anything.

**Nothing in this module calls a provider, and nothing in it fabricates a
response.** The default client (:class:`UnconfiguredModelClient`) raises. A test
double that returns canned text is legitimate and the tests use one; what would
not be legitimate is shipping such a double as the default and letting a run
produce plausible numbers with no model behind them (I3, §9.2). When the
provider adapters land (P7.1's registry supplies the credential), they
implement :class:`ModelClient` and the pipeline does not change.

Temperature 0
-------------

:data:`DEFAULT_TEMPERATURE` is ``0.0`` and it is the default of
:class:`ModelRequest`. §5-P7 requires temperature 0 everywhere for extraction,
for a reason that outlives the setting: a sampled extraction is not
reproducible, so it breaks I2, and it cannot be cached honestly — one draw would
be reused as though it were the answer
(:func:`backend.extraction.cache.assert_temperature_is_cacheable`).

The field is nevertheless *present* rather than hard-coded away, because §6.5
makes temperature a per-task operator setting stored by P7.1's registry. It is
present, defaulted to 0, validated, and refused by the pipeline when non-zero —
so the only way to run extraction at a sampling temperature is a deliberate code
change in a place where this docstring is sitting next to it.

Units: ``temperature`` is dimensionless; ``max_tokens`` is a count of provider
tokens; ``timeout_s`` is wall-clock **seconds**; ``latency_ms`` is wall-clock
**milliseconds**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "MAX_TEMPERATURE",
    "ModelCallError",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "ProviderNotConfiguredError",
    "UnconfiguredModelClient",
]

DEFAULT_TEMPERATURE: Final = 0.0
"""Sampling temperature for every extraction call. **Zero (§5-P7).**"""

MAX_TEMPERATURE: Final = 2.0
"""Largest temperature a request may carry (dimensionless).

The union of the providers' accepted ranges, matching
:mod:`backend.extraction.providers.assignments`: a value one provider rejects is
that provider's error to state at call time, and hard-coding one vendor's
ceiling here would silently misreport another's. The extraction pipeline refuses
anything but 0 regardless.
"""

DEFAULT_MAX_TOKENS: Final = 1024
"""Response cap in tokens when the caller does not choose one.

A bound, not a recommendation. Extraction returns a small schema-validated
object (:mod:`backend.extraction.tasks.schema`), not prose, so a low cap is the
right shape of default and a task needing more says so.
"""

DEFAULT_TIMEOUT_S: Final = 60.0
"""Per-request wall-clock timeout in seconds when the caller does not choose one."""


class ModelCallError(RuntimeError):
    """The provider call failed: transport, authentication, rate limit, timeout.

    Distinct from :class:`~backend.extraction.tasks.schema.SchemaValidationError`,
    which means the call *succeeded* and the answer was unusable. The two need
    telling apart: one is an infrastructure problem to retry, the other is a
    prompt problem to fix, and merging them would make the retry loop hammer a
    provider over a malformed response it will produce again identically at
    temperature 0.
    """


class ProviderNotConfiguredError(ModelCallError):
    """No usable model client is configured, so no call can be made.

    Raised by :class:`UnconfiguredModelClient`, which is what a pipeline holds
    when nobody supplied a real client. It raises rather than returning a
    plausible response: a missing connector is a blocker, not an excuse for a
    mock (I3).
    """


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One extraction call, fully specified.

    Attributes:
        model: Provider-side model identifier, verbatim (e.g.
            ``"anthropic:claude-3-5-haiku-20241022"``). Part of the cache
            address, so it must name the model exactly and stably.
        system: System instruction.
        prompt: The rendered user message — **anonymized text only**. The
            pipeline builds this from a masked payload; nothing in this module
            checks that, because a check here would be checking the caller's
            memory rather than the property. The property is asserted where it
            can be observed: the pipeline's tests inspect what an injected
            client actually receives.
        temperature: Dimensionless sampling temperature, 0 by default and 0 in
            all extraction (§5-P7).
        max_tokens: Response cap in tokens (count, ≥ 1).
        timeout_s: Per-request wall-clock timeout in seconds (> 0).
    """

    model: str
    system: str
    prompt: str
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        """Reject a request that could not be sent or could not be reproduced.

        Raises:
            ValueError: if ``model`` is empty or blank, if ``prompt`` is empty,
                if ``temperature`` is outside ``[0, MAX_TEMPERATURE]``, if
                ``max_tokens`` is below 1, or if ``timeout_s`` is not positive.
        """
        if not self.model.strip():
            msg = "model must be a non-empty identifier"
            raise ValueError(msg)
        if not self.prompt:
            msg = "prompt must be non-empty; there is nothing to extract from an empty message"
            raise ValueError(msg)
        if not 0.0 <= self.temperature <= MAX_TEMPERATURE:
            msg = (
                f"temperature must be in [0, {MAX_TEMPERATURE}] (dimensionless); "
                f"got {self.temperature!r}"
            )
            raise ValueError(msg)
        if self.max_tokens < 1:
            msg = f"max_tokens must be >= 1 tokens; got {self.max_tokens}"
            raise ValueError(msg)
        if self.timeout_s <= 0:
            msg = f"timeout_s must be > 0 seconds; got {self.timeout_s!r}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """What a provider returned.

    Attributes:
        text: The response body, **verbatim**. §5-P7 requires raw responses to
            be stored, so nothing may normalize, strip or re-encode this before
            it reaches the cache and the result row.
        model: Model identifier the provider reported serving the request, which
            may differ from the one requested (an alias resolving to a dated
            version). Recorded as returned.
        input_tokens: Prompt tokens **as reported by the provider** (count), or
            ``None`` when it reported none. Never estimated: an invented token
            count becomes an invented cost figure in the governor (I3).
        output_tokens: Response tokens as reported (count), or ``None``.
        latency_ms: Measured wall-clock duration in milliseconds, or ``None``
            when the client did not measure it.
    """

    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


@runtime_checkable
class ModelClient(Protocol):
    """Something that can execute a :class:`ModelRequest`.

    The single seam between the extraction pipeline and a paid external service.
    Implementations: the provider adapters (blocked on B4), and the test doubles
    the pipeline's tests inject.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute ``request`` and return the provider's response.

        Raises:
            ModelCallError: on any failure to obtain a response. Implementations
                must raise rather than return an empty or synthesized response
                (I3).
        """
        ...


class UnconfiguredModelClient:
    """The client a pipeline holds when nobody gave it one: it refuses.

    Every call raises :class:`ProviderNotConfiguredError`. This is the default
    precisely so that "no provider configured" surfaces as a loud failure at the
    call site instead of as a run that completes and produces numbers.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Refuse the call.

        Args:
            request: The call that will not be made; its model identifier is
                quoted in the error so the operator sees what was being asked
                for.

        Raises:
            ProviderNotConfiguredError: always.
        """
        msg = (
            f"no model client is configured, so the call to {request.model!r} cannot be made "
            "(B4: provider keys unresolved). Configure a credential in the provider registry "
            "and inject the corresponding client. Nothing here will synthesize a response"
        )
        raise ProviderNotConfiguredError(msg)
