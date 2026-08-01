"""Test-connection probe: one minimal live call, reporting latency and status (P7.1, §6.5).

What it does and what it deliberately does not
-----------------------------------------------

§6.5 asks for a "Test connection" button "issuing a minimal live call and
reporting latency and status". The call this module makes is the provider's
**model-listing endpoint** (``GET /v1/models`` at both providers), which:

* is authenticated, so it answers the question the operator is actually asking
  — *does this key work* — rather than merely whether the host resolves;
* consumes **no tokens**, so a probe costs nothing beyond a request. B4
  (provider keys and spend caps) is unresolved and P7.7's cost governor does
  not exist yet, so "minimal" has to mean *no billable units*, not *few*;
* returns quickly and is safe to repeat while an operator fixes a key.

What it therefore does **not** prove is that the specific model a task is
assigned to is available to this key. That is a real limitation and it is
stated rather than papered over: proving it would require a completion request,
which spends money, and spending money to answer a configuration question is
exactly what an unbudgeted system should not do. When the cost governor lands
(P7.7), a model-level probe becomes affordable to offer as a separate,
explicitly-budgeted action.

**Never automatic.** Nothing in this module runs on a schedule, on startup, on
render, or on a health check. It is invoked when an operator asks for it, and
the only route that reaches it is a POST (:mod:`backend.api.routes.providers`),
so it cannot be triggered by a page load or a link prefetch.

**No retries.** A failed probe is a report, not an operation to complete: a
retry would multiply the request count for a question whose answer is already
"it did not work", and would blur the latency figure the operator is reading.

Refusing without a key
----------------------

With no credential configured the probe raises
:class:`~backend.extraction.providers.registry.ProviderKeyNotConfiguredError`
**before constructing a request** — it does not return a hopeful result, does
not report a fabricated latency, and does not reach the network. A probe that
answered "ok" for an unconfigured provider would be a fabricated measurement
(I3), and one that answered "failed" would misattribute a local configuration
gap to the provider.

Secret isolation
----------------

The key appears in exactly one place: the outbound authorization header built
by :func:`build_probe_request`. It is never in a URL (so it cannot reach a log
through a request line), never in a :class:`ProbeResult`, and never in a log
line. :func:`_safe_detail` additionally refuses to quote any response text that
matches a credential shape, so a provider that echoed a key back could not
launder it into this module's output.

Testing status — read this before trusting the live path
---------------------------------------------------------

B4 is unresolved: there is no provider key in this repository or its CI, so the
**live path of this module has never been executed against a real provider.**
What is under test is the refusal path, the request construction (method, URL,
headers, and the absence of the key from the URL), and the outcome mapping
against a stubbed transport. Its behaviour against a real provider response is
therefore *unverified*, and no test here claims otherwise — stubbing a
provider's success reply and calling that coverage would be exactly the
fabricated verification I3 forbids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx

from backend.core.logging import contains_credential, get_logger
from backend.extraction.providers.catalog import Provider
from backend.extraction.providers.registry import load_provider_secret

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ANTHROPIC_VERSION_HEADER_VALUE",
    "PROBE_ENDPOINTS",
    "PROBE_TIMEOUT_S",
    "ProbeOutcome",
    "ProbeRequest",
    "ProbeResult",
    "build_probe_request",
    "probe_provider",
]

_logger = get_logger(__name__)

PROBE_TIMEOUT_S: Final = 10.0
"""Wall-clock timeout for the probe request, in seconds.

Short on purpose and unrelated to a task's configured ``timeout_s``: an
operator pressing "Test connection" is waiting on the answer, and a listing
endpoint that has not replied in ten seconds has already answered the useful
question.
"""

PROBE_ENDPOINTS: Final[Mapping[Provider, str]] = MappingProxyType(
    {
        Provider.ANTHROPIC: "https://api.anthropic.com/v1/models",
        Provider.OPENAI: "https://api.openai.com/v1/models",
    }
)
"""Authenticated, zero-token listing endpoint per provider.

Both are ``GET`` and both authenticate with the stored key, which is what makes
them usable as a credential check. Read-only mapping so a caller cannot
redirect a probe — and therefore a credential — at another host by mutating
module state.
"""

ANTHROPIC_VERSION_HEADER_VALUE: Final = "2023-06-01"
"""Value of Anthropic's required ``anthropic-version`` header.

A dated API version, not a client version: it pins the request/response
contract, and Anthropic requires it on every request. Named as a constant so
the pin is visible and greppable rather than buried in a header dict.
"""

_MAX_DETAIL_CHARS: Final = 200
"""Characters of a provider's error body quoted in a :class:`ProbeResult`."""

_HTTP_CLIENT_ERROR_FLOOR: Final = 400
_HTTP_SERVER_ERROR_FLOOR: Final = 500
_HTTP_UNAUTHORIZED: Final = 401
_HTTP_FORBIDDEN: Final = 403
_HTTP_TOO_MANY_REQUESTS: Final = 429


class ProbeOutcome(StrEnum):
    """What a probe found. Never a guess — every value maps to something observed."""

    OK = "ok"
    """The provider accepted the credential and answered successfully."""

    AUTHENTICATION_FAILED = "authentication_failed"
    """The provider rejected the credential (HTTP 401/403). The key is wrong or revoked."""

    RATE_LIMITED = "rate_limited"
    """The provider accepted the request but is throttling (HTTP 429). Not a key problem."""

    PROVIDER_ERROR = "provider_error"
    """The provider answered with some other non-2xx status; the detail carries it."""

    UNREACHABLE = "unreachable"
    """No HTTP response at all: DNS, TLS, connection refused, proxy failure."""

    TIMED_OUT = "timed_out"
    """No response within :data:`PROBE_TIMEOUT_S`."""


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    """The exact outbound request a probe would make.

    Split out from the sending so the construction is testable without a live
    call — which matters more than usual here, because B4 leaves the live call
    itself unexercised (module docstring).

    Attributes:
        method: HTTP method, always ``GET`` for the listing endpoints.
        url: absolute URL. **Carries no credential**, by construction: the key
            travels in :attr:`headers` only, so a URL that reaches a log,
            an error message or a proxy access log leaks nothing.
        headers: request headers, including the provider's authentication
            header. Read-only; the credential lives here and nowhere else.
    """

    method: str
    url: str
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The report §6.5 asks for: status and latency, with nothing secret in it.

    Attributes:
        provider: the provider probed.
        outcome: what was observed.
        latency_ms: round-trip wall-clock duration in milliseconds. For a
            failed probe this is the time to failure — approximately
            :data:`PROBE_TIMEOUT_S` when it timed out. It is always a real
            measurement of a real request: there is no path that produces a
            :class:`ProbeResult` without one having been sent.
        http_status: the provider's HTTP status code, or ``None`` when no
            response arrived.
        detail: a short human-readable explanation. Never contains the
            credential (see :func:`_safe_detail`).
    """

    provider: Provider
    outcome: ProbeOutcome
    latency_ms: float
    http_status: int | None
    detail: str


def build_probe_request(provider: Provider, api_key: str) -> ProbeRequest:
    """Build the probe request for *provider*, authenticated with *api_key*.

    Pure: it performs no I/O and reads no configuration, so a test can assert
    the exact wire shape — including that the key is in a header and not in the
    URL — without a provider, a key, or a network.

    Header shapes, as each provider documents them:

    * Anthropic — ``x-api-key: <key>`` plus ``anthropic-version:
      2023-06-01`` (required on every request).
    * OpenAI — ``Authorization: Bearer <key>``.

    Args:
        provider: provider to build the request for.
        api_key: the credential. Placed in the authentication header only.

    Returns:
        The request that would be sent.

    Raises:
        ValueError: *provider* has no probe endpoint. Unreachable through the
            enum today; it exists so that adding a member to
            :class:`~backend.extraction.providers.catalog.Provider` without
            adding its endpoint fails loudly here rather than producing a
            request to nowhere.
    """
    url = PROBE_ENDPOINTS.get(provider)
    if url is None:  # pragma: no cover - guarded by the enum/endpoint drift test
        msg = (
            f"provider {provider.value!r} has no probe endpoint configured; a provider the "
            "platform cannot construct a request for cannot be probed"
        )
        raise ValueError(msg)
    headers: dict[str, str] = {"accept": "application/json"}
    if provider is Provider.ANTHROPIC:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = ANTHROPIC_VERSION_HEADER_VALUE
    else:
        headers["authorization"] = f"Bearer {api_key}"
    return ProbeRequest(method="GET", url=url, headers=MappingProxyType(headers))


def _safe_detail(text: str, api_key: str) -> str:
    """Return a quotable excerpt of provider output, or a refusal to quote it.

    Two passes, in this order, because either alone is insufficient:

    1. the exact credential is removed, which handles a provider that echoes
       the key it was sent;
    2. what remains is checked against the credential *shapes*
       (:func:`backend.core.logging.contains_credential`), which handles a
       provider that echoes some *other* credential — an upstream token, an
       example key in an error message. If anything still looks like a
       credential, nothing is quoted at all.

    Args:
        text: raw provider output.
        api_key: the credential that was sent, so it can be excised.

    Returns:
        A truncated excerpt safe to store and display, or a fixed placeholder
        when the excerpt could not be made safe.
    """
    excerpt = text.strip().replace(api_key, "")[:_MAX_DETAIL_CHARS]
    if contains_credential(excerpt):
        return "<response withheld: it contains credential-shaped text>"
    return excerpt


def _outcome_for_status(status: int) -> ProbeOutcome:
    """Map an HTTP status code to what it tells the operator about their key."""
    if status < _HTTP_CLIENT_ERROR_FLOOR:
        return ProbeOutcome.OK
    if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
        return ProbeOutcome.AUTHENTICATION_FAILED
    if status == _HTTP_TOO_MANY_REQUESTS:
        return ProbeOutcome.RATE_LIMITED
    return ProbeOutcome.PROVIDER_ERROR


def _detail_for_status(provider: Provider, status: int, body: str) -> str:
    """Compose the human-readable explanation attached to a completed probe."""
    if status < _HTTP_CLIENT_ERROR_FLOOR:
        return f"{provider.value} accepted the credential (HTTP {status})."
    if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
        return (
            f"{provider.value} rejected the credential (HTTP {status}). "
            f"Rotate the stored key. Provider said: {body}"
        )
    if status == _HTTP_TOO_MANY_REQUESTS:
        return (
            f"{provider.value} is rate limiting (HTTP {status}); this says nothing about "
            f"whether the credential is valid. Provider said: {body}"
        )
    kind = "server" if status >= _HTTP_SERVER_ERROR_FLOOR else "client"
    return f"{provider.value} returned an unexpected {kind} error (HTTP {status}): {body}"


async def probe_provider(
    provider: Provider,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> ProbeResult:
    """Issue one minimal live call to *provider* and report status and latency.

    Spends a real request against a real provider. It is never called
    automatically; see the module docstring.

    Args:
        provider: provider to probe. Its credential must already be stored.
        transport: httpx transport to send through. Injected by tests to
            exercise the outcome mapping without a network; ``None`` uses
            httpx's default. **The live path this parameter bypasses is
            unexercised** — B4 leaves no key to exercise it with.
        timeout_s: wall-clock timeout in seconds for the whole request.

    Returns:
        The measured result. Every field describes something observed; there is
        no code path that returns a :class:`ProbeResult` without a request
        having been sent.

    Raises:
        backend.extraction.providers.registry.ProviderKeyNotConfiguredError:
            no credential is stored. Raised **before** a request is built, so
            an unconfigured provider is refused rather than reported on.
        backend.core.crypto.SecretsCryptoError: ``SECRETS_KEK`` is unset or
            malformed, or the stored ciphertext does not authenticate under it.
    """
    api_key = await load_provider_secret(provider)
    request = build_probe_request(provider, api_key)
    started = time.perf_counter()
    outcome: ProbeOutcome
    http_status: int | None = None
    async with httpx.AsyncClient(transport=transport, timeout=timeout_s) as client:
        try:
            response = await client.request(
                request.method, request.url, headers=dict(request.headers)
            )
        except httpx.TimeoutException:
            outcome = ProbeOutcome.TIMED_OUT
            detail = (
                f"{provider.value} did not respond within {timeout_s:g}s. The credential "
                "was neither accepted nor rejected."
            )
        except httpx.HTTPError as exc:
            outcome = ProbeOutcome.UNREACHABLE
            detail = (
                f"could not reach {provider.value}: {type(exc).__name__}. The credential "
                "was neither accepted nor rejected."
            )
        else:
            http_status = response.status_code
            outcome = _outcome_for_status(http_status)
            detail = _detail_for_status(provider, http_status, _safe_detail(response.text, api_key))
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    # provider, outcome, latency and status only: the request headers — the one
    # place the credential exists — are never logged.
    _logger.info(
        "provider_probe",
        provider=provider.value,
        outcome=outcome.value,
        http_status=http_status,
        latency_ms=latency_ms,
    )
    return ProbeResult(
        provider=provider,
        outcome=outcome,
        latency_ms=latency_ms,
        http_status=http_status,
        detail=detail,
    )
