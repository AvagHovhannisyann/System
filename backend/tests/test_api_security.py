"""CSRF protection tests (CC.2, DIRECTIVE.md §7).

Covers the token primitive (:class:`CsrfProtect`), the secret-resolution
policy, and end-to-end enforcement through the real application built by
:func:`backend.api.app.create_app`, driven with the httpx ``AsyncClient`` +
``ASGITransport`` pattern established in ``test_health.py``.

The rate-limit middleware is switched off in the app-level CSRF tests
(``rate_limit_enabled=False``): it sits *outside* CSRF in the stack and fails
closed with no Redis, so leaving it on would answer every mutating request
with 503 before CSRF ever ran. The two together are exercised against a real
Redis in ``test_api_rate_limit.py``.
"""

from __future__ import annotations

import hmac
import time
from typing import TYPE_CHECKING, cast

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from itsdangerous import URLSafeTimedSerializer
from itsdangerous.timed import TimestampSigner
from pydantic import SecretStr
from starlette.responses import JSONResponse

from backend.api.app import create_app
from backend.api.routes import health as health_module
from backend.api.security.csrf import (
    CSRF_COOKIE_NAME,
    CSRF_EXEMPT_PATHS,
    CSRF_HEADER_NAME,
    SAFE_METHODS,
    CsrfConfigurationError,
    CsrfError,
    CsrfMiddleware,
    CsrfProtect,
    build_csrf_cookie,
    resolve_csrf_secret,
)
from backend.api.security.settings import DEFAULT_CSRF_TOKEN_MAX_AGE_S, ApiSecuritySettings
from backend.core.config import Settings

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send

_TEST_SIGNING_KEY = "csrf-unit-test-signing-key"
"""Throwaway signing key for unit-level token tests."""

_OTHER_SIGNING_KEY = "a-different-signing-key"
"""A second key, used to prove tokens do not verify across keys."""

_CSRF_SALT = "quant-research-platform/csrf/v1"
"""The shipped signer salt, restated here so a silent change to it fails a test."""

ECHO_PATH = "/api/_test/echo"
"""Mutating route added only inside these tests."""

READ_PATH = "/api/_test/read"
"""Safe route added only inside these tests."""

UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")
"""Methods that must be CSRF-protected."""


def _protect(max_age_s: int = 3600) -> CsrfProtect:
    """Build a token issuer/verifier on the throwaway test key."""
    return CsrfProtect(_TEST_SIGNING_KEY, max_age_s=max_age_s)


async def _healthy() -> None:
    """Fake health probe reporting a live component."""


def build_test_app(
    *,
    rate_limit_enabled: bool = False,
    rate_limit_requests: int = 3,
    rate_limit_window_s: float = 60.0,
    csrf_token_max_age_s: int = DEFAULT_CSRF_TOKEN_MAX_AGE_S,
) -> FastAPI:
    """Build the real application plus test routes added with zero ceremony.

    The mutating route deliberately declares no security dependency of any
    kind. That is the point of CC.2: a mutating route inherits CSRF
    protection by construction, so the test route is written exactly as a
    forgetful future author would write it.

    The health checkers are overridden with healthy fakes because these tests
    run without the application lifespan (``ASGITransport`` does not run it),
    so ``app.state.db_engine`` does not exist. Rate limiting is off by default
    here (see the module docstring); ``test_api_rate_limit.py`` turns it on
    with a small budget against a real Redis.
    """
    app = create_app(
        settings=Settings(environment="test"),
        security=ApiSecuritySettings(
            rate_limit_enabled=rate_limit_enabled,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window_s=rate_limit_window_s,
            csrf_token_max_age_s=csrf_token_max_age_s,
        ),
    )

    async def echo() -> dict[str, str]:
        """Trivial handler standing in for a future mutating endpoint."""
        return {"mutated": "yes"}

    async def read() -> dict[str, str]:
        """Trivial safe handler."""
        return {"read": "yes"}

    app.router.add_api_route(ECHO_PATH, echo, methods=list(UNSAFE_METHODS))
    app.router.add_api_route(READ_PATH, read, methods=["GET"])
    app.dependency_overrides[health_module.get_db_checker] = lambda: _healthy
    app.dependency_overrides[health_module.get_redis_checker] = lambda: _healthy
    return app


def installed_middleware(app: FastAPI) -> list[type]:
    """Return the middleware classes wrapping ``app``, outermost first.

    ``Middleware.cls`` is typed as a generic factory protocol, so it is cast
    back to a plain class object for identity comparisons.
    """
    return [cast("type", layer.cls) for layer in app.user_middleware]


def client_for(app: object) -> AsyncClient:
    """Create an httpx client speaking ASGI directly to ``app``."""
    return AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://testserver",
    )


# --------------------------------------------------------------------------
# CsrfProtect: the token primitive
# --------------------------------------------------------------------------


def test_issued_token_verifies_against_itself() -> None:
    """A freshly issued token, double-submitted, is accepted."""
    protect = _protect()
    token = protect.issue()
    protect.verify(token, token)  # must not raise


def test_issued_tokens_are_unique() -> None:
    """Each issue() call mints fresh entropy rather than a constant."""
    protect = _protect()
    assert len({protect.issue() for _ in range(20)}) == 20


def test_missing_cookie_is_rejected() -> None:
    """No cookie: rejected with the cookie-missing reason."""
    protect = _protect()
    with pytest.raises(CsrfError) as excinfo:
        protect.verify(None, protect.issue())
    assert excinfo.value.reason == "csrf_cookie_missing"


def test_empty_cookie_is_rejected() -> None:
    """An empty-string cookie counts as absent, not as a value to compare."""
    protect = _protect()
    with pytest.raises(CsrfError) as excinfo:
        protect.verify("", protect.issue())
    assert excinfo.value.reason == "csrf_cookie_missing"


def test_missing_header_is_rejected() -> None:
    """Cookie present but header absent: rejected with the header-missing reason."""
    protect = _protect()
    with pytest.raises(CsrfError) as excinfo:
        protect.verify(protect.issue(), None)
    assert excinfo.value.reason == "csrf_header_missing"


def test_two_different_valid_tokens_do_not_satisfy_double_submit() -> None:
    """Both halves individually valid but unequal: rejected as a mismatch."""
    protect = _protect()
    with pytest.raises(CsrfError) as excinfo:
        protect.verify(protect.issue(), protect.issue())
    assert excinfo.value.reason == "csrf_token_mismatch"


def test_tampered_token_fails_the_signature_check() -> None:
    """A token with one flipped character is rejected as invalid, not accepted."""
    protect = _protect()
    token = protect.issue()
    flipped = "A" if token[5] != "A" else "B"
    tampered = token[:5] + flipped + token[6:]
    with pytest.raises(CsrfError) as excinfo:
        protect.verify(tampered, tampered)
    assert excinfo.value.reason == "csrf_token_invalid"


def test_token_signed_with_another_key_is_rejected() -> None:
    """A structurally perfect token from a foreign key does not verify.

    This is the property that makes the scheme *signed* double-submit: an
    attacker who can plant a cookie (sibling subdomain, plain HTTP) and echo
    the same value in the header satisfies presence and equality, and still
    fails.
    """
    foreign = CsrfProtect(_OTHER_SIGNING_KEY, max_age_s=3600)
    forged = foreign.issue()
    with pytest.raises(CsrfError) as excinfo:
        _protect().verify(forged, forged)
    assert excinfo.value.reason == "csrf_token_invalid"


def test_expired_token_is_rejected_with_its_own_reason() -> None:
    """A correctly signed but stale token is rejected as expired."""
    protect = _protect(max_age_s=60)
    stale_at = int(time.time()) - 3600
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TimestampSigner, "get_timestamp", lambda _self: stale_at)
        stale = protect.issue()
    with pytest.raises(CsrfError) as excinfo:
        protect.verify(stale, stale)
    assert excinfo.value.reason == "csrf_token_expired"


def test_token_inside_max_age_is_still_accepted() -> None:
    """A token stamped just inside the window verifies — the expiry test is not vacuous."""
    protect = _protect(max_age_s=3600)
    recent_at = int(time.time()) - 60
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TimestampSigner, "get_timestamp", lambda _self: recent_at)
        recent = protect.issue()
    protect.verify(recent, recent)  # must not raise


def test_comparison_uses_constant_time_compare_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cookie/header comparison goes through :func:`hmac.compare_digest`.

    Timing cannot be asserted directly in a unit test, so the mechanism is
    asserted instead: a plain ``==`` would leak a valid token one byte at a
    time to an attacker who controls the header and can observe timing.
    """
    calls: list[tuple[bytes, bytes]] = []
    real_compare = hmac.compare_digest

    def recording_compare(left: bytes, right: bytes) -> bool:
        """Record the operands, then delegate to the real constant-time compare."""
        calls.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(hmac, "compare_digest", recording_compare)
    protect = _protect()
    token = protect.issue()
    protect.verify(token, token)
    # The *first* comparison is the double-submit check on the raw values; a
    # later one is itsdangerous comparing signature digests. If verify() used
    # ``==`` for the double-submit check, calls[0] would be the signature
    # bytes instead and this assertion would fail.
    assert calls[0] == (token.encode(), token.encode())


@pytest.mark.parametrize("bad_max_age", [0, -1])
def test_nonpositive_max_age_is_a_configuration_error(bad_max_age: int) -> None:
    """A never-expiring token configuration fails loudly at construction."""
    with pytest.raises(CsrfConfigurationError):
        CsrfProtect(_TEST_SIGNING_KEY, max_age_s=bad_max_age)


def test_empty_secret_is_a_configuration_error() -> None:
    """An empty signing key fails loudly instead of signing with nothing."""
    with pytest.raises(CsrfConfigurationError):
        CsrfProtect("", max_age_s=60)


def test_signed_token_payload_is_not_the_plain_nonce() -> None:
    """The cookie value is a sealed token, not a bare random string.

    Guards against a future 'simplification' to unsigned double-submit: the
    value must round-trip through the signer under the shipped salt.
    """
    token = _protect().issue()
    serializer: URLSafeTimedSerializer = URLSafeTimedSerializer(_TEST_SIGNING_KEY, salt=_CSRF_SALT)
    nonce = serializer.loads(token)
    assert isinstance(nonce, str)
    assert nonce != token


# --------------------------------------------------------------------------
# Secret resolution policy
# --------------------------------------------------------------------------


def test_configured_secret_is_used_verbatim() -> None:
    """A configured CSRF_SECRET is returned unchanged."""
    security = ApiSecuritySettings(csrf_secret=SecretStr(_TEST_SIGNING_KEY))
    assert resolve_csrf_secret(security, "prod") == _TEST_SIGNING_KEY


def test_missing_secret_in_production_refuses_to_start() -> None:
    """Production with no CSRF_SECRET is a startup error, not an ephemeral key."""
    with pytest.raises(CsrfConfigurationError, match="CSRF_SECRET"):
        resolve_csrf_secret(ApiSecuritySettings(csrf_secret=None), "prod")


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_missing_secret_outside_production_generates_an_ephemeral_key(environment: str) -> None:
    """Outside production an unset secret yields a fresh per-process key."""
    settings = ApiSecuritySettings(csrf_secret=None)
    first = resolve_csrf_secret(settings, environment)
    second = resolve_csrf_secret(settings, environment)
    assert first
    assert second
    assert first != second


def test_create_app_refuses_production_without_a_csrf_secret() -> None:
    """The failure surfaces at app construction, not on the first mutating request."""
    with pytest.raises(CsrfConfigurationError):
        create_app(
            settings=Settings(environment="prod"),
            security=ApiSecuritySettings(csrf_secret=None),
        )


# --------------------------------------------------------------------------
# Cookie rendering
# --------------------------------------------------------------------------


def test_cookie_attributes_are_lax_scoped_and_bounded() -> None:
    """The seeded cookie is path-wide, SameSite=Lax and carries a Max-Age."""
    header = build_csrf_cookie(CSRF_COOKIE_NAME, "value-123", max_age_s=99, secure=False)
    assert header.startswith(f"{CSRF_COOKIE_NAME}=value-123")
    assert "Path=/" in header
    assert "SameSite=Lax" in header
    assert "Max-Age=99" in header
    assert "Secure" not in header
    # Double-submit requires the client to read the token, so HttpOnly is
    # deliberately absent; assert it rather than leave it to a future edit.
    assert "HttpOnly" not in header


def test_cookie_is_marked_secure_when_requested() -> None:
    """Production (TLS) sets the Secure attribute."""
    header = build_csrf_cookie(CSRF_COOKIE_NAME, "value-123", max_age_s=99, secure=True)
    assert "Secure" in header


def test_production_app_marks_the_cookie_secure() -> None:
    """A production app wires ``secure_cookie=True`` through to the seeded cookie."""
    app = create_app(
        settings=Settings(environment="prod"),
        security=ApiSecuritySettings(
            csrf_secret=SecretStr(_TEST_SIGNING_KEY), rate_limit_enabled=False
        ),
    )
    csrf_layer = next(
        layer for layer in app.user_middleware if cast("type", layer.cls) is CsrfMiddleware
    )
    assert csrf_layer.kwargs["secure_cookie"] is True


# --------------------------------------------------------------------------
# End-to-end enforcement through the real application
# --------------------------------------------------------------------------


async def test_safe_request_seeds_a_token_cookie() -> None:
    """A cookie-less GET is answered with a fresh, valid token cookie."""
    app = build_test_app()
    async with client_for(app) as client:
        response = await client.get(READ_PATH)

    assert CSRF_COOKIE_NAME in response.cookies
    seeded = response.cookies[CSRF_COOKIE_NAME]
    app.state.csrf.verify(seeded, seeded)  # the seeded token is genuinely valid


async def test_health_semantics_survive_the_security_middleware() -> None:
    """D-005 is unchanged: /api/health still answers 200 with its documented body."""
    app = build_test_app()
    async with client_for(app) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["components"]) == {"db", "redis"}
    assert "x-request-id" in response.headers


async def test_existing_cookie_is_not_reseeded() -> None:
    """A client that already holds a token keeps it across safe requests."""
    app = build_test_app()
    async with client_for(app) as client:
        first = await client.get(READ_PATH)
        seeded = first.cookies[CSRF_COOKIE_NAME]
        second = await client.get(READ_PATH)

        assert "set-cookie" not in second.headers
        assert client.cookies.get(CSRF_COOKIE_NAME) == seeded


@pytest.mark.parametrize("method", UNSAFE_METHODS)
async def test_valid_double_submit_is_accepted(method: str) -> None:
    """Cookie and header carrying the same valid token: the request goes through."""
    app = build_test_app()
    token = app.state.csrf.issue()
    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, token)
        response = await client.request(method, ECHO_PATH, headers={CSRF_HEADER_NAME: token})

    assert response.status_code == 200
    assert response.json() == {"mutated": "yes"}


@pytest.mark.parametrize("method", UNSAFE_METHODS)
async def test_unsafe_request_without_a_token_is_refused(method: str) -> None:
    """Every unsafe method is protected — not just POST."""
    app = build_test_app()
    async with client_for(app) as client:
        response = await client.request(method, ECHO_PATH)

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_cookie_missing"


async def test_cookie_without_header_is_refused() -> None:
    """Holding the cookie is not enough; the header is the origin proof."""
    app = build_test_app()
    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, app.state.csrf.issue())
        response = await client.post(ECHO_PATH)

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_header_missing"


async def test_header_without_cookie_is_refused() -> None:
    """A header alone does not satisfy double-submit."""
    app = build_test_app()
    async with client_for(app) as client:
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: app.state.csrf.issue()})

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_cookie_missing"


async def test_mismatched_halves_are_refused() -> None:
    """Two individually valid tokens that differ are refused as a mismatch."""
    app = build_test_app()
    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, app.state.csrf.issue())
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: app.state.csrf.issue()})

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_token_mismatch"


async def test_attacker_planted_cookie_echoed_in_the_header_is_refused() -> None:
    """The cookie-injection attack that defeats *unsigned* double-submit.

    The attacker plants a cookie value of their choosing and echoes the same
    value in the header. Presence and equality both hold; only the signature
    check stands between this request and the endpoint.
    """
    app = build_test_app()
    planted = "attacker-chosen-value"
    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, planted)
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: planted})

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_token_invalid"


async def test_token_from_a_foreign_signing_key_is_refused() -> None:
    """A well-formed token minted under another key does not verify here."""
    app = build_test_app()
    forged = CsrfProtect(_OTHER_SIGNING_KEY, max_age_s=3600).issue()
    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, forged)
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: forged})

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_token_invalid"


_B64URL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
"""The URL-safe base64 alphabet itsdangerous encodes tokens with."""


def _materially_tamper(token: str, position: int) -> str:
    """Return ``token`` with ``position`` changed so the *decoded bytes* differ.

    Substituting the next character in the alphabet is enough everywhere except
    the final position, where only the top four bits are significant (see
    :func:`test_signature_tail_has_an_encoding_equivalence_class`); there the
    substitution has to step a whole equivalence class, hence ``+ 4``.
    """
    original = token[position]
    if original not in _B64URL_ALPHABET:  # a '.' section separator
        return token[:position] + "A" + token[position + 1 :]
    step = 4 if position == len(token) - 1 else 1
    index = (_B64URL_ALPHABET.index(original) + step) % len(_B64URL_ALPHABET)
    return token[:position] + _B64URL_ALPHABET[index] + token[position + 1 :]


def test_tampering_at_every_position_is_refused() -> None:
    """No single-character edit anywhere in a token survives verification.

    Exhaustive over positions rather than sampling one, because the position
    that used to be sampled — the last — is the single position where a naive
    edit does *not* always change the signature.
    """
    protect = CsrfProtect("k" * 32, max_age_s=3600)
    token = protect.issue()
    for position in range(len(token)):
        tampered = _materially_tamper(token, position)
        assert tampered != token, position
        with pytest.raises(CsrfError) as caught:
            protect.verify(tampered, tampered)
        assert caught.value.args[0] == "csrf_token_invalid", position


def test_signature_tail_has_an_encoding_equivalence_class() -> None:
    """The final character carries four significant bits, not six — documented, not a hole.

    An HMAC-SHA1 signature is 20 bytes = 160 bits, base64-encoded into 27
    characters = 162 bits. The two surplus bits are padding, and Python's
    decoder ignores them, so the four characters sharing the final character's
    top four bits all decode to the *same* signature and all verify.

    This is an encoding artefact, not a forgery route: producing any of the
    four requires already holding a valid token, which for double-submit CSRF
    means already having won. It is asserted here so the property is recorded
    where someone can find it — a previous version of the tampering test above
    flipped exactly this character to ``"A"``, which silently landed inside the
    equivalence class whenever the token ended in ``"A"`` and failed CI at a
    measured 5.9% of runs.
    """
    protect = CsrfProtect("k" * 32, max_age_s=3600)
    token = protect.issue()
    base = _B64URL_ALPHABET.index(token[-1]) & ~0b11
    equivalents = [token[:-1] + _B64URL_ALPHABET[base + offset] for offset in range(4)]

    assert token in equivalents
    for variant in equivalents:
        protect.verify(variant, variant)  # no raise: same decoded signature

    outside = token[:-1] + _B64URL_ALPHABET[(base + 4) % len(_B64URL_ALPHABET)]
    with pytest.raises(CsrfError):
        protect.verify(outside, outside)


async def test_tampered_cookie_and_header_are_refused() -> None:
    """A tampered token is refused end-to-end, on both halves of the double submit."""
    app = build_test_app()
    token: str = app.state.csrf.issue()
    tampered = _materially_tamper(token, 0)
    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, tampered)
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: tampered})

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_token_invalid"


async def test_expired_token_is_refused_end_to_end() -> None:
    """A stale token is refused by the middleware, with the expiry reason."""
    app = build_test_app(csrf_token_max_age_s=60)
    stale_at = int(time.time()) - 3600
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TimestampSigner, "get_timestamp", lambda _self: stale_at)
        stale = app.state.csrf.issue()

    async with client_for(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, stale)
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: stale})

    assert response.status_code == 403
    assert response.json()["reason"] == "csrf_token_expired"


@pytest.mark.parametrize("method", sorted(SAFE_METHODS))
async def test_safe_methods_are_exempt(method: str) -> None:
    """GET/HEAD/OPTIONS reach routing without a token.

    OPTIONS has no handler on the test route, so routing answers 405; the
    assertion is that CSRF did not intercept, i.e. the status is not 403.
    """
    app = build_test_app()
    async with client_for(app) as client:
        response = await client.request(method, READ_PATH)

    assert response.status_code != 403


async def test_rejection_carries_the_correlation_id() -> None:
    """A 403 is still a correlated response: the ID middleware wraps CSRF."""
    app = build_test_app()
    async with client_for(app) as client:
        response = await client.post(ECHO_PATH, headers={"X-Request-ID": "fixed-id-123"})

    assert response.status_code == 403
    assert response.headers["x-request-id"] == "fixed-id-123"


async def _bare_ok(scope: Scope, receive: Receive, send: Send) -> None:
    """Minimal ASGI application answering 200 to anything."""
    await JSONResponse({"mutated": "yes"})(scope, receive, send)


async def test_exempt_path_is_not_checked() -> None:
    """A path in the exemption list bypasses verification; others do not.

    Exercised on a directly constructed middleware because the shipped
    :data:`CSRF_EXEMPT_PATHS` is empty — and a test proving the mechanism
    works must not be the reason it stops being empty.
    """
    protect = _protect()
    enforcing = CsrfMiddleware(_bare_ok, protect=protect)
    exempting = CsrfMiddleware(_bare_ok, protect=protect, exempt_paths=frozenset({ECHO_PATH}))

    async with client_for(enforcing) as client:
        assert (await client.post(ECHO_PATH)).status_code == 403

    async with client_for(exempting) as client:
        assert (await client.post(ECHO_PATH)).status_code == 200
        # The exemption is exact-match, so a sibling path stays protected.
        assert (await client.post(f"{ECHO_PATH}/child")).status_code == 403


def test_shipped_exemption_list_is_empty() -> None:
    """No mutating path ships unprotected.

    If this fails, an exemption was added: check that it carries the written
    justification the module docstring requires, then update this test
    deliberately rather than reflexively.
    """
    assert not CSRF_EXEMPT_PATHS


def test_csrf_middleware_is_installed_by_the_app_factory() -> None:
    """CSRF is part of the default stack — not something a router opts into."""
    assert CsrfMiddleware in installed_middleware(build_test_app())
