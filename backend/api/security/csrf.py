"""CSRF protection for state-changing requests (DIRECTIVE.md §7, CC.2).

Scheme: **signed double-submit token**
--------------------------------------

A token is a random 32-byte nonce sealed with
:class:`itsdangerous.URLSafeTimedSerializer` under a server-side key. It is
delivered to the browser in the ``csrf_token`` cookie and must be echoed back
in the ``X-CSRF-Token`` header on every unsafe request. Acceptance requires
all three of:

1. both the cookie and the header are present;
2. they are equal, compared with :func:`hmac.compare_digest` (constant time —
   a byte-at-a-time comparison leaks a valid token one position at a time to
   an attacker who can observe timing and control the header);
3. the value carries a valid, unexpired signature under the server key.

Step 3 is what the signature buys, and it is why plain double-submit is not
enough: cookies are not origin-isolated the way headers are, so an attacker
holding *any* sibling subdomain (or a network position on plain HTTP) can
plant a ``csrf_token`` cookie of their choosing on this host and then echo
the same known value in the header, satisfying steps 1 and 2. A token they
did not obtain from this server cannot carry our signature, so step 3 fails
and the forgery is rejected.

Design choice: **enforcement is default-on, exemption is explicit and named**
----------------------------------------------------------------------------

:class:`CsrfMiddleware` runs before routing and rejects *every* request whose
method is not in :data:`SAFE_METHODS` unless its path appears in
:data:`CSRF_EXEMPT_PATHS`. That is a deliberate inversion of the usual
FastAPI idiom, in which each mutating route remembers to add a
``Depends(csrf)``:

- A route that forgets an opt-in dependency is *silently unprotected*, and
  nothing in the codebase says so. A route that needs an exemption it does
  not have fails loudly, in every test that exercises it, on the first run.
- The complete list of unprotected mutating paths is therefore one named
  constant in one file, auditable at a glance, instead of the absence of a
  decorator spread across every router.
- This module ships *before* the first mutating endpoint exists (P3.10's
  re-sync trigger is the first consumer). Protection that must be remembered
  by code not yet written is protection that will eventually be forgotten;
  protection that is inherited by construction is not.

:data:`CSRF_EXEMPT_PATHS` is empty and every entry added to it must carry a
written justification, on the same principle as the dependency-audit
allowlist in D-010: an exemption nobody had to argue for is an exemption
nobody reviewed.

Cookie properties and their assumptions
---------------------------------------

The cookie is deliberately **not** ``HttpOnly``: the browser client has to
read it to populate the header, which is inherent to double-submit. This is
not a weakening — the token is not a credential. It authenticates nothing on
its own; it only proves the request came from script running on this origin,
which is exactly the property a cross-site forgery lacks. ``SameSite=Lax``
and (in production) ``Secure`` are set as defense in depth.

Tokens are seeded automatically: any safe-method response to a client with no
``csrf_token`` cookie carries a fresh one, so a client bootstraps simply by
loading the dashboard. There is no token endpoint to call and forget.
"""

from __future__ import annotations

import hmac
import secrets
from http.cookies import SimpleCookie
from typing import TYPE_CHECKING, Final

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.core.logging import get_logger

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

    from backend.api.security.settings import ApiSecuritySettings

__all__ = [
    "CSRF_COOKIE_NAME",
    "CSRF_EXEMPT_PATHS",
    "CSRF_HEADER_NAME",
    "SAFE_METHODS",
    "CsrfConfigurationError",
    "CsrfError",
    "CsrfMiddleware",
    "CsrfProtect",
    "build_csrf_cookie",
    "resolve_csrf_secret",
]

SAFE_METHODS: Final[frozenset[str]] = frozenset({"GET", "HEAD", "OPTIONS"})
"""HTTP methods exempt from CSRF verification because they must not change state.

``TRACE`` is *not* listed. It is nominally safe but is disabled at the server
and has its own cross-site history (XST); leaving it out means it is treated
as unsafe and rejected rather than waved through.
"""

CSRF_COOKIE_NAME: Final = "csrf_token"
"""Name of the cookie carrying the signed token."""

CSRF_HEADER_NAME: Final = "X-CSRF-Token"
"""Name of the request header the client echoes the cookie value in."""

CSRF_EXEMPT_PATHS: Final[frozenset[str]] = frozenset()
"""Exact request paths exempt from CSRF verification on unsafe methods.

Empty by design. Every entry added here must be accompanied by a comment
naming the endpoint, why it cannot carry a token, and what protects it
instead. Prefix matching is deliberately *not* supported: a prefix silently
exempts routes added later underneath it.
"""

_CSRF_SALT: Final = "quant-research-platform/csrf/v1"
"""Signer salt — domain-separates CSRF tokens from any other signed value."""

_NONCE_BYTES: Final = 32
"""Entropy per token, in bytes."""

_logger = get_logger(__name__)


class CsrfError(Exception):
    """A CSRF check failed; ``reason`` is a stable machine-readable code.

    The reason is safe to return to the client: it distinguishes "you sent no
    cookie" from "your token does not verify", which is diagnostic for an
    honest operator and useless to an attacker, who already knows which parts
    of the request they omitted.
    """

    def __init__(self, reason: str) -> None:
        """Record the failure ``reason`` code and use it as the message."""
        super().__init__(reason)
        self.reason = reason


class CsrfConfigurationError(RuntimeError):
    """CSRF protection cannot be configured safely; raised at app startup."""


class CsrfProtect:
    """Issues and verifies signed double-submit CSRF tokens.

    Stateless: no server-side token store, so nothing has to be replicated or
    expired. Validity is entirely a property of the signature and the embedded
    timestamp, which is what makes the scheme survive a process restart when a
    persistent ``CSRF_SECRET`` is configured.
    """

    def __init__(self, secret: str, *, max_age_s: int) -> None:
        """Build a token issuer/verifier.

        Args:
            secret: signing key. Any non-empty string; entropy is the
                caller's responsibility (see :func:`resolve_csrf_secret`).
            max_age_s: maximum accepted token age in seconds. Tokens older
                than this are rejected with reason ``csrf_token_expired``.

        Raises:
            CsrfConfigurationError: if ``secret`` is empty or ``max_age_s``
                is not positive — a zero-entropy or never-expiring
                configuration must fail loudly, not silently degrade.
        """
        if not secret:
            msg = "CSRF signing secret must be a non-empty string"
            raise CsrfConfigurationError(msg)
        if max_age_s <= 0:
            msg = f"CSRF token max age must be > 0 seconds; got {max_age_s}"
            raise CsrfConfigurationError(msg)
        self._serializer: URLSafeTimedSerializer = URLSafeTimedSerializer(secret, salt=_CSRF_SALT)
        self._max_age_s = max_age_s

    @property
    def max_age_s(self) -> int:
        """Maximum accepted token age in seconds."""
        return self._max_age_s

    def issue(self) -> str:
        """Return a fresh signed token carrying a new random nonce.

        The value is URL-safe ASCII and therefore usable as a cookie value
        without further quoting.
        """
        nonce = secrets.token_urlsafe(_NONCE_BYTES)
        token: str = self._serializer.dumps(nonce)
        return token

    def verify(self, cookie_value: str | None, header_value: str | None) -> None:
        """Validate a double-submit pair; return ``None`` on success.

        Args:
            cookie_value: value of the ``csrf_token`` cookie, or ``None``.
            header_value: value of the ``X-CSRF-Token`` header, or ``None``.

        Raises:
            CsrfError: with reason ``csrf_cookie_missing``,
                ``csrf_header_missing``, ``csrf_token_mismatch``,
                ``csrf_token_expired`` or ``csrf_token_invalid``.
        """
        if not cookie_value:
            raise CsrfError("csrf_cookie_missing")
        if not header_value:
            raise CsrfError("csrf_header_missing")
        if not hmac.compare_digest(cookie_value.encode(), header_value.encode()):
            raise CsrfError("csrf_token_mismatch")
        # The two are byte-identical here, so verifying one verifies both.
        try:
            self._serializer.loads(cookie_value, max_age=self._max_age_s)
        except SignatureExpired as exc:
            raise CsrfError("csrf_token_expired") from exc
        except BadSignature as exc:
            raise CsrfError("csrf_token_invalid") from exc


def resolve_csrf_secret(security: ApiSecuritySettings, environment: str) -> str:
    """Return the CSRF signing key, failing closed in production.

    Args:
        security: the API security settings holding the optional
            ``CSRF_SECRET``.
        environment: the deployment environment from
            :class:`backend.core.config.Settings` (``dev``/``test``/``prod``).

    Returns:
        The configured secret, or — outside production only — a fresh
        per-process key.

    Raises:
        CsrfConfigurationError: when ``environment`` is ``prod`` and
            ``CSRF_SECRET`` is unset.

    An ephemeral key is cryptographically fine but *operationally* wrong: it
    invalidates every outstanding token on restart and does not agree across
    worker processes, so a multi-worker production deployment would reject
    valid tokens roughly ``1 - 1/workers`` of the time. That failure is
    intermittent and looks like a client bug, which is exactly the kind of
    thing that gets "fixed" by disabling the check. Refusing to start is the
    cheaper failure. Outside production the convenience is worth it, and the
    warning says so out loud.
    """
    configured = security.csrf_secret
    if configured is not None and configured.get_secret_value():
        return configured.get_secret_value()
    if environment == "prod":
        msg = (
            "CSRF_SECRET is unset. It is required in production: an ephemeral "
            "per-process key silently breaks CSRF validation across worker "
            "processes and restarts. Generate one with "
            "'python -c \"import secrets; print(secrets.token_urlsafe(32))\"'."
        )
        raise CsrfConfigurationError(msg)
    _logger.warning(
        "csrf_secret_generated_ephemeral",
        environment=environment,
        detail=(
            "CSRF_SECRET unset; generated a per-process key. Tokens do not "
            "survive a restart and are not valid across worker processes."
        ),
    )
    return secrets.token_urlsafe(_NONCE_BYTES)


def build_csrf_cookie(
    name: str,
    value: str,
    *,
    max_age_s: int,
    secure: bool,
) -> str:
    """Render a ``Set-Cookie`` header value for a CSRF token.

    Args:
        name: cookie name.
        value: signed token (URL-safe ASCII).
        max_age_s: cookie lifetime in seconds; matches the token's own
            maximum age so the cookie does not outlive its validity.
        secure: whether to set the ``Secure`` attribute (production, where
            the dashboard is served over TLS).

    Returns:
        The header value, without the ``Set-Cookie:`` prefix.

    Not ``HttpOnly`` — see the module docstring; the client must read it.
    """
    cookie: SimpleCookie = SimpleCookie()
    cookie[name] = value
    morsel = cookie[name]
    morsel["path"] = "/"
    morsel["samesite"] = "Lax"
    morsel["max-age"] = str(max_age_s)
    if secure:
        morsel["secure"] = True
    return morsel.OutputString()


class CsrfMiddleware:
    """Enforce CSRF on unsafe methods and seed tokens on safe ones.

    Pure-ASGI, matching :class:`backend.api.middleware.CorrelationIdMiddleware`:
    no ``BaseHTTPMiddleware`` wrapper, so no response buffering and streaming
    responses still stream.

    Assumptions: non-HTTP scopes (lifespan, websocket) pass through untouched;
    path matching against the exemption list is exact and case-sensitive, on
    ``scope["path"]`` as routed (i.e. including any application root path).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        protect: CsrfProtect,
        exempt_paths: frozenset[str] = CSRF_EXEMPT_PATHS,
        secure_cookie: bool = False,
        cookie_name: str = CSRF_COOKIE_NAME,
        header_name: str = CSRF_HEADER_NAME,
    ) -> None:
        """Wrap ``app`` with CSRF enforcement.

        Args:
            app: the next ASGI application in the stack.
            protect: the token issuer/verifier.
            exempt_paths: exact paths exempt from verification. Defaults to
                :data:`CSRF_EXEMPT_PATHS`; overridden only by tests.
            secure_cookie: set the ``Secure`` cookie attribute.
            cookie_name: cookie carrying the token.
            header_name: request header echoing the token.
        """
        self._app = app
        self._protect = protect
        self._exempt_paths = exempt_paths
        self._secure_cookie = secure_cookie
        self._cookie_name = cookie_name
        self._header_name = header_name

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event cycle; see the class docstring for semantics."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        if scope["method"] in SAFE_METHODS:
            await self._handle_safe(scope, receive, send)
            return

        if scope["path"] not in self._exempt_paths:
            request = Request(scope)
            try:
                self._protect.verify(
                    request.cookies.get(self._cookie_name),
                    request.headers.get(self._header_name),
                )
            except CsrfError as exc:
                await self._reject(exc, scope, receive, send)
                return

        await self._app(scope, receive, send)

    async def _handle_safe(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass a safe-method request through, seeding a token when absent."""
        if Request(scope).cookies.get(self._cookie_name):
            await self._app(scope, receive, send)
            return

        cookie_header = build_csrf_cookie(
            self._cookie_name,
            self._protect.issue(),
            max_age_s=self._protect.max_age_s,
            secure=self._secure_cookie,
        )

        async def send_with_cookie(message: Message) -> None:
            """Attach the seeded ``Set-Cookie`` to the response start message."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("set-cookie", cookie_header)
            await send(message)

        await self._app(scope, receive, send_with_cookie)

    async def _reject(self, error: CsrfError, scope: Scope, receive: Receive, send: Send) -> None:
        """Log and answer a failed check with 403 and the machine-readable reason."""
        _logger.warning(
            "csrf_rejected",
            reason=error.reason,
            method=scope["method"],
            path=scope["path"],
        )
        response = JSONResponse(
            {"detail": "CSRF validation failed", "reason": error.reason},
            status_code=403,
        )
        await response(scope, receive, send)
