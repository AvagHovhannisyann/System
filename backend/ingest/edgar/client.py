"""HTTP access to EDGAR, with SEC's fair-access policy enforced fail-closed (P3.2).

SEC publishes two conditions on automated access to EDGAR, both read from
``https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data``
on 2026-08-01:

- *"Please declare your user agent in request headers"*, with the sample
  ``User-Agent: Sample Company Name AdminContact@<sample company domain>.com``
  and ``Accept-Encoding: gzip, deflate``;
- *"Fair access — Current max request rate: 10 requests/second."*

The first is enforced here and the second by the connector's declared
:class:`~backend.ingest.ratelimit.RateLimit`, which the framework applies to
every call routed through :meth:`~backend.ingest.base.Connector.request`.

**Enforcement is fail-closed and there is no default contact string.** When
``settings.sec_user_agent`` is unset the client refuses to be constructed. The
alternative — sending something plausible — is worse than sending nothing: an
invented contact address is a false statement made to a regulator's
infrastructure on the operator's behalf, and it would defeat the only mechanism
SEC has for asking a specific requester to stop. Refusing costs one clear
configuration error; the alternative costs an access ban that no retry policy
can undo.

Status handling is delegated to
:func:`backend.ingest.errors.source_error_for_status`, which classifies 429 and
5xx as transient (worth retrying) and other 4xx as permanent. Two EDGAR
specifics are worth stating because they are not obvious from the status code
alone:

- a request for a daily-index file that does not exist answers **403**, not
  404 (verified live on a Saturday's ``master.idx``), and 403 is also what a
  fair-access block returns. The connector never learns which, so it never
  requests a date the quarter listing did not publish;
- EDGAR serves ``www.sec.gov`` and ``data.sec.gov`` from different
  infrastructure with different behaviour. This client talks only to
  ``www.sec.gov``; the reasoning for not using ``data.sec.gov`` is in
  :mod:`backend.ingest.edgar.parse`.
"""

from __future__ import annotations

from types import TracebackType
from typing import Final

import httpx

from backend.core.config import Settings, get_settings
from backend.ingest.errors import IngestError, TransientSourceError, source_error_for_status

__all__ = [
    "EDGAR_ACCEPT_ENCODING",
    "EdgarClient",
    "SecUserAgentNotConfiguredError",
    "resolve_sec_user_agent",
]

EDGAR_ACCEPT_ENCODING: Final = "gzip, deflate"
"""The ``Accept-Encoding`` SEC's published sample request headers ask for."""

_DEFAULT_TIMEOUT_S: Final = 30.0
"""Per-request timeout in seconds. Exceeded reads surface as transient."""

_MAX_RESPONSE_EXCERPT: Final = 200
"""Characters of a failed response quoted in an error message."""


class SecUserAgentNotConfiguredError(IngestError):
    """``settings.sec_user_agent`` is unset, so EDGAR access is refused.

    Raised before any request is made. See the module docstring for why the
    connector refuses rather than sending an anonymous or invented contact.
    """


def resolve_sec_user_agent(settings: Settings | None = None) -> str:
    """Return the configured SEC contact string, or refuse.

    Args:
        settings: configuration to read; defaults to the cached application
            settings.

    Returns:
        ``settings.sec_user_agent`` with surrounding whitespace removed.

    Raises:
        SecUserAgentNotConfiguredError: if the setting is unset or blank. The
            message names the environment variable to set and states why no
            default exists.
    """
    resolved = settings if settings is not None else get_settings()
    user_agent = (resolved.sec_user_agent or "").strip()
    if not user_agent:
        msg = (
            "SEC_USER_AGENT is not set, so the EDGAR connector refuses to run. SEC's "
            "fair-access policy requires every automated requester to declare contact "
            "information in the User-Agent header; there is deliberately no default, "
            "because sending an invented contact would be a false statement to SEC and "
            "would defeat the only channel it has for contacting this requester. Set "
            "SEC_USER_AGENT to a real name and contact address, e.g. "
            "'Example Research contact@example.com'"
        )
        raise SecUserAgentNotConfiguredError(msg)
    return user_agent


class EdgarClient:
    """Minimal EDGAR HTTP client: declared identity, classified failures.

    One instance per connector run. It owns an :class:`httpx.AsyncClient` and
    must be closed (it is an async context manager, and the connector uses it
    as one). It performs **no** pacing and **no** retrying: both belong to the
    framework, which applies them to every call routed through
    :meth:`~backend.ingest.base.Connector.request`. Putting them here as well
    would make the declared rate limit a suggestion enforced in two places.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Create a client that identifies itself on every request.

        Args:
            user_agent: the contact string to send. Obtain it from
                :func:`resolve_sec_user_agent`; it is required here rather than
                resolved internally so the refusal happens once, early, and is
                visible in the connector's constructor.
            transport: httpx transport override. Tests supply one serving
                captured EDGAR responses; production leaves it ``None``.
            timeout_s: per-request timeout, seconds.

        Raises:
            ValueError: if ``user_agent`` is blank — a blank header is
                indistinguishable from sending none.
        """
        if not user_agent.strip():
            msg = "user_agent must be a non-empty contact string (SEC fair-access policy)"
            raise ValueError(msg)
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": user_agent.strip(),
                "Accept-Encoding": EDGAR_ACCEPT_ENCODING,
            },
            timeout=timeout_s,
            transport=transport,
            follow_redirects=True,
        )

    async def __aenter__(self) -> EdgarClient:
        """Enter the context manager, returning this client."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client."""
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def get_text(self, url: str, *, source: str) -> str:
        """Fetch one URL and return its decoded body.

        Args:
            url: absolute EDGAR URL.
            source: connector source name, used in error messages.

        Returns:
            The response body decoded as text. EDGAR's archive files are
            latin-1 (company names carry non-ASCII bytes) and httpx's charset
            detection is not relied on: the body is decoded explicitly.

        Raises:
            TransientSourceError: on a connection failure, a timeout, or a
                status the error taxonomy classifies as retryable. The request
                never becomes a value — an unreachable source raises (I3).
            PermanentSourceError: on a status that the identical request cannot
                recover from.
        """
        try:
            response = await self._client.get(url)
        except httpx.HTTPError as exc:
            msg = f"{source}: {type(exc).__name__} requesting {url}: {exc}"
            raise TransientSourceError(msg) from exc
        if response.is_error:
            excerpt = response.text[:_MAX_RESPONSE_EXCERPT].replace("\n", " ")
            raise source_error_for_status(
                response.status_code, source=source, detail=f"{url} -> {excerpt}"
            )
        return response.content.decode("latin-1")
