"""HTTP access to the FRED/ALFRED API, key-gated and key-redacting (P3.8).

Everything asserted here was read from St. Louis Fed documentation on
2026-08-01; the URLs are named so a later reader can re-verify rather than
trust this docstring:

- ``https://fred.stlouisfed.org/docs/api/fred/series_observations.html`` —
  ``api_key`` is *"32 character alpha-numeric lowercase string, **required**"*;
- ``https://fred.stlouisfed.org/docs/api/fred/errors.html`` — *"All errors use
  standard HTTP status codes. Our API has rate limiting which returns a status
  code if exceeded."* The statuses it lists are **400, 404, 423, 429, 500**.
  Note what is absent: FRED documents *that* it rate-limits and **publishes no
  numeric ceiling**, which is why the connector's declared
  :class:`~backend.ingest.ratelimit.RateLimit` is an explicitly self-imposed
  conservative budget rather than a quoted figure (see the connector).

The key requirement was also confirmed against the live service on 2026-08-01:
``GET https://api.stlouisfed.org/fred/series/observations?series_id=GDPC1&file_type=json``
with no key answers **HTTP 400** with body ``{"error_code":400,
"error_message":"Bad Request.  Variable api_key is not set. ..."}``. There is
therefore no keyless tier for this endpoint, and no keyless path exists in this
module to fall back to.

Fail-closed, with no default
----------------------------

When ``FRED_API_KEY`` is unset the client refuses to be constructed. It does
not fall back to a keyless request (there is none), and it does not fall back
to cached, defaulted or otherwise invented observations: invariant I3 says an
unavailable source raises. The refusal happens at construction so a
misconfigured deployment fails where it is obvious rather than mid-run.

The key's *shape* is deliberately **not** validated. FRED documents a
32-character lowercase alphanumeric format, but rejecting anything else here
would turn a future format change into a total outage, and FRED's own answer to
a bad key is already precise (``400`` with *"The value for variable api_key is
not registered"*). Blank is refused; judging the rest is the source's job.

Secret isolation (invariant I5)
-------------------------------

The key travels in the **query string**, so any URL that reaches a log, an
error message or a stored ``source_url`` column would leak it. Every outward
string in this module is built by :func:`redacted_url`, which renders the
request without the ``api_key`` parameter at all — not masked, absent. The
key is added only at the moment httpx sends the request.
"""

from __future__ import annotations

import json
import os
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlencode

import httpx

from backend.ingest.errors import IngestError, PermanentSourceError, TransientSourceError
from backend.ingest.errors import source_error_for_status as _source_error_for_status

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "FRED_API_BASE",
    "FRED_API_KEY_ENV",
    "FredApiKeyNotConfiguredError",
    "FredClient",
    "redacted_url",
    "resolve_fred_api_key",
]

FRED_API_BASE: Final = "https://api.stlouisfed.org/fred"
"""Root of the FRED API version 1, the version documenting real-time periods."""

FRED_API_KEY_ENV: Final = "FRED_API_KEY"
"""Environment variable holding the FRED API key.

Read from :data:`os.environ` directly rather than from
:class:`backend.core.config.Settings`, because P3.8 does not own
``backend/core/config.py`` and adding a field there is an operator action. The
practical consequence is stated plainly rather than left to be discovered:
``Settings`` is what loads ``.env``, so a key placed **only** in ``.env`` will
*not* be visible here. It must be a real process environment variable (a
compose ``environment:`` entry, an exported shell variable) until a
``fred_api_key`` field is added to ``Settings`` and this resolver switched to
it.
"""

_API_KEY_PARAM: Final = "api_key"
"""Query parameter carrying the secret; never rendered into any outward string."""

_DEFAULT_TIMEOUT_S: Final = 30.0
"""Per-request timeout in seconds. Exceeded reads surface as transient."""

_MAX_RESPONSE_EXCERPT: Final = 200
"""Characters of a failed response quoted in an error message."""


class FredApiKeyNotConfiguredError(IngestError):
    """``FRED_API_KEY`` is unset, so FRED access is refused.

    Raised before any request is made. FRED requires a key on every endpoint
    (verified live: a keyless request answers HTTP 400), so there is nothing to
    degrade to — and degrading to invented values is exactly what invariant I3
    forbids.
    """


def resolve_fred_api_key(environ: Mapping[str, str] | None = None) -> str:
    """Return the configured FRED API key, or refuse to proceed.

    Args:
        environ: environment mapping to read; defaults to :data:`os.environ`.
            Injected by tests so the refusal path can be exercised without
            mutating the process environment.

    Returns:
        The key with surrounding whitespace removed. Never logged, never
        included in an error message, never stored.

    Raises:
        FredApiKeyNotConfiguredError: if the variable is unset or blank. The
            message names the variable and where to get a key, and states that
            no keyless path exists.
    """
    source = environ if environ is not None else os.environ
    api_key = source.get(FRED_API_KEY_ENV, "").strip()
    if not api_key:
        msg = (
            f"{FRED_API_KEY_ENV} is not set, so the FRED connector refuses to run. Every "
            "FRED API endpoint requires a key (verified 2026-08-01: a keyless request to "
            "fred/series/observations answers HTTP 400 'Variable api_key is not set'), so "
            "there is no keyless tier to fall back to and no cached or default series "
            "values will be substituted (invariant I3). Request a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html and set "
            f"{FRED_API_KEY_ENV} in the process environment"
        )
        raise FredApiKeyNotConfiguredError(msg)
    return api_key


def redacted_url(path: str, params: Mapping[str, str]) -> str:
    """Render a FRED request URL with the API key omitted entirely.

    Args:
        path: endpoint path relative to :data:`FRED_API_BASE`, e.g.
            ``"series/observations"``.
        params: query parameters. Any ``api_key`` entry is **dropped**, not
            masked — a masked secret in a log is still a secret's shape, and
            the value is of no diagnostic use (invariant I5).

    Returns:
        An absolute URL safe to log, to put in an exception message, and to
        store in a ``source_url`` column for reproducibility (I2). Parameters
        are sorted so the same request always renders identically, which is
        what makes the stored URL a stable provenance record.
    """
    safe = {key: value for key, value in params.items() if key != _API_KEY_PARAM}
    return f"{FRED_API_BASE}/{path.lstrip('/')}?{urlencode(sorted(safe.items()))}"


class FredClient:
    """Minimal FRED HTTP client: key attached late, key never emitted.

    One instance per connector run. It owns an :class:`httpx.AsyncClient` and
    must be closed (it is an async context manager, and the connector uses it
    as one). It performs **no** pacing and **no** retrying: both belong to the
    framework, which applies them to every call routed through
    :meth:`~backend.ingest.base.Connector.request`. Duplicating them here would
    make the declared rate limit a suggestion enforced in two places.
    """

    def __init__(
        self,
        *,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        """Create a client that authenticates every request.

        Args:
            api_key: the FRED API key. Required here rather than resolved
                internally, so the refusal happens once, early, and is visible
                in the connector's constructor. Held in memory only; it is
                never written to a log, an error or a database column.
            transport: httpx transport override. Tests supply one serving
                captured or constructed responses; production leaves it
                ``None``.
            timeout_s: per-request timeout, seconds.

        Raises:
            ValueError: if ``api_key`` is blank — an empty key produces the
                same HTTP 400 as sending none, so it is refused here where the
                message can say why.
        """
        if not api_key.strip():
            msg = (
                "api_key must be a non-empty FRED API key; obtain it from "
                "resolve_fred_api_key(), which refuses when the environment does not "
                "provide one"
            )
            raise ValueError(msg)
        self._api_key = api_key.strip()
        self._client = httpx.AsyncClient(timeout=timeout_s, transport=transport)

    async def __aenter__(self) -> FredClient:
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

    async def get_json(
        self, path: str, params: Mapping[str, str], *, source: str
    ) -> dict[str, Any]:
        """Fetch one FRED endpoint and return its decoded JSON body.

        Args:
            path: endpoint path relative to :data:`FRED_API_BASE`, e.g.
                ``"series/observations"``.
            params: query parameters **without** the API key, which this
                method adds. ``file_type=json`` is added too, because FRED's
                documented default is ``xml``.
            source: connector source name, used in error messages.

        Returns:
            The parsed JSON object. FRED's documented success bodies are JSON
            objects at the top level for every endpoint this client calls.

        Raises:
            TransientSourceError: on a connection failure, a timeout, or a
                status the shared taxonomy classifies as retryable (429, 5xx).
                The request never becomes a value — an unreachable source
                raises (I3).
            PermanentSourceError: on a status the identical request cannot
                recover from (400, 404, 423 — FRED's documented non-timing
                errors), or on a body that is not a JSON object. Every message
                names the **redacted** URL, never the key.
        """
        safe_url = redacted_url(path, params)
        request_params = {**params, "file_type": "json", _API_KEY_PARAM: self._api_key}
        try:
            response = await self._client.get(
                f"{FRED_API_BASE}/{path.lstrip('/')}", params=request_params
            )
        except httpx.HTTPError as exc:
            # str(exc) can embed the request URL, which carries the key; only
            # the exception *type* and the redacted URL are reported (I5).
            msg = f"{source}: {type(exc).__name__} requesting {safe_url}"
            raise TransientSourceError(msg) from exc
        if response.is_error:
            excerpt = response.text[:_MAX_RESPONSE_EXCERPT].replace("\n", " ")
            raise _source_error_for_status(
                response.status_code, source=source, detail=f"{safe_url} -> {excerpt}"
            )
        try:
            body: object = json.loads(response.content)
        except ValueError as exc:
            msg = f"{source}: {safe_url} returned a body that is not JSON: {exc}"
            raise PermanentSourceError(msg) from exc
        if not isinstance(body, dict):
            msg = (
                f"{source}: {safe_url} returned a JSON {type(body).__name__}, not the "
                "documented JSON object"
            )
            raise PermanentSourceError(msg)
        return body
