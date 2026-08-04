"""Rate-limiting tests (CC.2, DIRECTIVE.md §7).

**Redis coverage — stated plainly, per I6.** Both levels are exercised and
neither claims the other's coverage:

- The *decision logic* (window arithmetic, per-key isolation, headers,
  fail-closed / fail-open branching) is driven through the middleware with
  :class:`~backend.api.security.ratelimit.InMemoryFixedWindowStore` and an
  injected clock, because reproducing a window boundary in real time would
  mean sleeping.
- The *Redis implementation itself* — the Lua script, the ``INCR`` +
  ``PEXPIRE`` + ``PTTL`` contract, real TTL expiry, and an end-to-end 429
  through the real application — runs against a **real Redis 7.4 container**
  started with testcontainers, the same mechanism ``backend/tests/integration``
  already uses for Postgres. There is no fake Redis anywhere in this file: no
  test asserts Redis behavior against a stand-in.
- The two are tied together by ``test_store_contract_*``, which runs the same
  assertions against **both** stores. That is what licenses using the
  in-memory one at all: its equivalence is tested, not assumed.

The one stand-in here is ``_HangingRedis``, which exists solely to hold an
``asyncio.timeout`` open. It asserts nothing about Redis semantics.

The container is started with host networking when the docker daemon offers no
``bridge`` network, mirroring ``backend/tests/integration/conftest.py``.
"""

from __future__ import annotations

import asyncio
import socket
from typing import TYPE_CHECKING, Any, cast

import pytest
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from starlette.responses import JSONResponse
from testcontainers.core.config import testcontainers_config
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from backend.api.app import create_app
from backend.api.middleware import CorrelationIdMiddleware
from backend.api.security.csrf import CSRF_COOKIE_NAME, CSRF_HEADER_NAME
from backend.api.security.ratelimit import (
    RATE_LIMIT_EXEMPT_PATHS,
    InMemoryFixedWindowStore,
    RateLimiterUnavailableError,
    RateLimitMiddleware,
    RateLimitPolicy,
    RateLimitStore,
    RedisFixedWindowStore,
    client_key,
    decide,
)
from backend.api.security.settings import ApiSecuritySettings
from backend.core.config import Settings
from backend.tests.test_api_security import (
    ECHO_PATH,
    READ_PATH,
    build_test_app,
    installed_middleware,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from starlette.types import ASGIApp, Receive, Scope, Send

_REDIS_IMAGE = "redis:7.4-alpine"
"""Pinned to the tag ``docker-compose.yml`` runs, so tests exercise the deployed engine."""

_REDIS_PORT = 6379
"""Redis' default port."""

_TEST_PATH = "/mutate"
"""Path used by the middleware-level tests (no routing involved)."""


# --------------------------------------------------------------------------
# Minimal ASGI targets
# --------------------------------------------------------------------------


async def _ok(scope: Scope, receive: Receive, send: Send) -> None:
    """Minimal ASGI application answering 200 to anything."""
    await JSONResponse({"ok": True})(scope, receive, send)


def _client(app: ASGIApp, *, client_address: tuple[str, int] = ("127.0.0.1", 9999)) -> AsyncClient:
    """Create an httpx client speaking ASGI directly to ``app`` from ``client_address``."""
    return AsyncClient(
        transport=ASGITransport(app=app, client=client_address),
        base_url="http://testserver",
    )


class _UnavailableStore:
    """A store that is always down; drives the fail-closed / fail-open branches."""

    async def hit(self, key: str, window_ms: int) -> tuple[int, int]:
        """Always raise, naming the request it refused."""
        msg = f"simulated outage for {key} over a {window_ms} ms window"
        raise RateLimiterUnavailableError(msg)


class _HangingRedis:
    """A Redis stand-in whose ``eval`` never returns; drives the timeout branch.

    This is *not* a fake Redis: it asserts nothing about Redis semantics. It
    exists only to hold the ``asyncio.timeout`` open, which a real server
    cannot be made to do reliably.
    """

    def __init__(self) -> None:
        """Record nothing; start with no observed call."""
        self.seen: tuple[object, ...] = ()

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> list[int]:
        """Record the call and then never return."""
        self.seen = (script, numkeys, keys_and_args)
        await asyncio.sleep(3600)
        return [0, 0]  # pragma: no cover - unreachable; the timeout always fires


# --------------------------------------------------------------------------
# Policy and decision arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("limit", "window_s"), [(0, 60.0), (-1, 60.0), (5, 0.0), (5, -1.0)])
def test_nonsensical_policy_is_rejected(limit: int, window_s: float) -> None:
    """A policy that would allow everything (or nothing) fails at construction."""
    with pytest.raises(ValueError, match="must be"):
        RateLimitPolicy(limit=limit, window_s=window_s)


@pytest.mark.parametrize(("window_s", "expected_ms"), [(60.0, 60_000), (0.25, 250), (0.0001, 1)])
def test_window_converts_to_whole_milliseconds(window_s: float, expected_ms: int) -> None:
    """Sub-millisecond windows floor at 1 ms rather than becoming a zero TTL."""
    assert RateLimitPolicy(limit=1, window_s=window_s).window_ms == expected_ms


def test_decision_under_the_limit_allows_and_reports_remaining() -> None:
    """The third of five requests is allowed with two left."""
    decision = decide(3, 30_000, RateLimitPolicy(limit=5, window_s=60.0))
    assert decision.allowed is True
    assert decision.remaining == 2


def test_decision_at_the_limit_is_the_last_allowed_request() -> None:
    """Request number ``limit`` is allowed and leaves nothing."""
    decision = decide(5, 30_000, RateLimitPolicy(limit=5, window_s=60.0))
    assert decision.allowed is True
    assert decision.remaining == 0


def test_decision_over_the_limit_refuses_with_a_usable_retry_after() -> None:
    """Request ``limit + 1`` is refused; Retry-After rounds the TTL up."""
    decision = decide(6, 30_001, RateLimitPolicy(limit=5, window_s=60.0))
    assert decision.allowed is False
    assert decision.remaining == 0
    assert decision.reset_after_s == 31


def test_retry_after_never_advises_an_immediate_retry() -> None:
    """A sub-second remaining window still reports at least one second."""
    assert decide(6, 1, RateLimitPolicy(limit=5, window_s=60.0)).reset_after_s == 1
    assert decide(6, 0, RateLimitPolicy(limit=5, window_s=60.0)).reset_after_s == 1


# --------------------------------------------------------------------------
# client_key
# --------------------------------------------------------------------------


def test_client_key_is_the_peer_address() -> None:
    """The bucket identity comes from the transport peer, not from a header."""
    assert client_key({"client": ("10.0.0.7", 5555)}) == "10.0.0.7"


@pytest.mark.parametrize("scope", [{}, {"client": None}, {"client": ("", 0)}])
def test_client_key_falls_back_to_a_single_shared_bucket(scope: dict[str, Any]) -> None:
    """A scope with no usable peer address shares one bucket (stricter, not looser)."""
    assert client_key(scope) == "unknown"


async def test_forwarded_for_header_cannot_mint_a_fresh_bucket() -> None:
    """X-Forwarded-For is not consulted, so spoofing it does not reset the budget."""
    store = InMemoryFixedWindowStore()
    app = RateLimitMiddleware(
        _ok, policy=RateLimitPolicy(limit=1, window_s=60.0), timeout_s=1.0, store=store
    )
    async with _client(app) as client:
        first = await client.post(_TEST_PATH)
        spoofed = await client.post(_TEST_PATH, headers={"X-Forwarded-For": "203.0.113.9"})

    assert first.status_code == 200
    assert spoofed.status_code == 429


# --------------------------------------------------------------------------
# Middleware behavior (in-memory store, injected clock)
# --------------------------------------------------------------------------


def _limited(
    policy: RateLimitPolicy,
    *,
    store: RateLimitStore | None = None,
    fail_open: bool = False,
    exempt_paths: frozenset[str] = RATE_LIMIT_EXEMPT_PATHS,
) -> RateLimitMiddleware:
    """Wrap the minimal ASGI app in a rate limiter with the given policy."""
    return RateLimitMiddleware(
        _ok,
        policy=policy,
        timeout_s=1.0,
        store=store if store is not None else InMemoryFixedWindowStore(),
        fail_open=fail_open,
        exempt_paths=exempt_paths,
    )


async def test_requests_under_the_limit_are_allowed_with_budget_headers() -> None:
    """Every request up to the limit succeeds and advertises what is left."""
    app = _limited(RateLimitPolicy(limit=3, window_s=60.0))
    async with _client(app) as client:
        responses = [await client.post(_TEST_PATH) for _ in range(3)]

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert [r.headers["X-RateLimit-Remaining"] for r in responses] == ["2", "1", "0"]
    assert responses[0].headers["X-RateLimit-Limit"] == "3"


async def test_request_over_the_limit_is_refused_with_429_and_retry_after() -> None:
    """The first over-budget request gets 429, Retry-After, and zero remaining."""
    app = _limited(RateLimitPolicy(limit=2, window_s=60.0))
    async with _client(app) as client:
        for _ in range(2):
            assert (await client.post(_TEST_PATH)).status_code == 200
        refused = await client.post(_TEST_PATH)

    assert refused.status_code == 429
    assert refused.json()["reason"] == "rate_limit_exceeded"
    assert int(refused.headers["Retry-After"]) >= 1
    assert refused.headers["X-RateLimit-Remaining"] == "0"
    assert refused.headers["X-RateLimit-Limit"] == "2"


async def test_window_resets_and_the_budget_returns() -> None:
    """Once the window elapses the same client is allowed again."""
    now = [1000.0]
    store = InMemoryFixedWindowStore(clock=lambda: now[0])
    app = _limited(RateLimitPolicy(limit=1, window_s=10.0), store=store)

    async with _client(app) as client:
        assert (await client.post(_TEST_PATH)).status_code == 200
        assert (await client.post(_TEST_PATH)).status_code == 429
        now[0] += 9.0  # still inside the window
        assert (await client.post(_TEST_PATH)).status_code == 429
        now[0] += 2.0  # window has now elapsed
        allowed = await client.post(_TEST_PATH)

    assert allowed.status_code == 200
    assert allowed.headers["X-RateLimit-Remaining"] == "0"


async def test_budgets_are_isolated_per_client() -> None:
    """One client exhausting its budget does not throttle another."""
    store = InMemoryFixedWindowStore()
    app = _limited(RateLimitPolicy(limit=1, window_s=60.0), store=store)

    async with _client(app, client_address=("10.0.0.1", 1111)) as first:
        assert (await first.post(_TEST_PATH)).status_code == 200
        assert (await first.post(_TEST_PATH)).status_code == 429

    async with _client(app, client_address=("10.0.0.2", 2222)) as second:
        assert (await second.post(_TEST_PATH)).status_code == 200


async def test_budgets_are_isolated_per_route() -> None:
    """Exhausting one endpoint's budget does not close another endpoint."""
    app = _limited(RateLimitPolicy(limit=1, window_s=60.0))
    async with _client(app) as client:
        assert (await client.post(_TEST_PATH)).status_code == 200
        assert (await client.post(_TEST_PATH)).status_code == 429
        assert (await client.post("/other")).status_code == 200


async def test_safe_methods_are_never_rate_limited() -> None:
    """Read traffic does not consume the budget and never touches the store."""
    app = _limited(RateLimitPolicy(limit=1, window_s=60.0), store=_UnavailableStore())
    async with _client(app) as client:
        for _ in range(5):
            response = await client.get(_TEST_PATH)
            assert response.status_code == 200
            assert "X-RateLimit-Limit" not in response.headers


async def test_exempt_path_is_not_limited() -> None:
    """A path in the exemption list bypasses the limiter."""
    app = _limited(RateLimitPolicy(limit=1, window_s=60.0), exempt_paths=frozenset({_TEST_PATH}))
    async with _client(app) as client:
        assert [(await client.post(_TEST_PATH)).status_code for _ in range(4)] == [200] * 4


def test_shipped_rate_limit_exemption_list_is_empty() -> None:
    """No mutating path ships unthrottled."""
    assert not RATE_LIMIT_EXEMPT_PATHS


# --------------------------------------------------------------------------
# The documented Redis-unavailable behavior
# --------------------------------------------------------------------------


async def test_unreachable_store_fails_closed_with_503() -> None:
    """The documented default: a mutating request is refused, not waved through."""
    app = _limited(RateLimitPolicy(limit=5, window_s=60.0), store=_UnavailableStore())
    async with _client(app) as client:
        response = await client.post(_TEST_PATH)

    assert response.status_code == 503
    assert response.json()["reason"] == "rate_limiter_unavailable"
    assert int(response.headers["Retry-After"]) >= 1


async def test_unreachable_store_can_be_made_to_fail_open_explicitly() -> None:
    """``fail_open=True`` passes the request through.

    The rejected alternative, proven reachable — so the fail-closed test is a
    real choice, not the only thing the code can do.
    """
    app = _limited(
        RateLimitPolicy(limit=5, window_s=60.0), store=_UnavailableStore(), fail_open=True
    )
    async with _client(app) as client:
        response = await client.post(_TEST_PATH)

    assert response.status_code == 200


async def test_unconfigured_redis_client_fails_closed() -> None:
    """An app with no ``state.redis`` is a misconfiguration, and it fails closed."""
    app = create_app(
        settings=Settings(environment="test"),
        security=ApiSecuritySettings(rate_limit_requests=5, rate_limit_window_s=60.0),
    )

    async def echo() -> dict[str, str]:
        """Trivial mutating handler."""
        return {"mutated": "yes"}

    app.router.add_api_route(ECHO_PATH, echo, methods=["POST"])
    token = app.state.csrf.issue()

    async with _client(app) as client:
        client.cookies.set(CSRF_COOKIE_NAME, token)
        response = await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: token})

    assert response.status_code == 503
    assert response.json()["reason"] == "rate_limiter_unavailable"


async def test_hanging_store_times_out_and_is_treated_as_unavailable() -> None:
    """A Redis that never answers is unavailability, not an indefinite hang."""
    hanging = _HangingRedis()
    store = RedisFixedWindowStore(cast("Redis", hanging), timeout_s=0.02)
    with pytest.raises(RateLimiterUnavailableError):
        await store.hit("some-key", 1000)
    assert hanging.seen  # the call really was attempted


async def test_refused_redis_connection_is_treated_as_unavailable() -> None:
    """A closed port surfaces as unavailability rather than an unhandled RedisError."""
    dead = Redis.from_url("redis://127.0.0.1:6390/0")
    store = RedisFixedWindowStore(dead, timeout_s=2.0)
    try:
        with pytest.raises(RateLimiterUnavailableError):
            await store.hit("some-key", 1000)
    finally:
        await dead.aclose()


# --------------------------------------------------------------------------
# Real Redis
# --------------------------------------------------------------------------


def _bridge_network_available() -> bool:
    """Return True when the docker daemon offers the default ``bridge`` network.

    Same probe as ``backend/tests/integration/conftest.py``: sandboxed daemons
    that expose only ``host``/``none`` cannot publish ports, so the container
    must run with host networking (and Ryuk, which needs a published port of
    its own, is disabled — the fixture stops the container deterministically
    via its context manager).
    """
    client = DockerClient()
    try:
        return bool(client.client.networks.list(names=["bridge"]))
    finally:
        client.client.close()


def _free_tcp_port() -> int:
    """Return a TCP port that is free right now on the loopback interface.

    Only meaningful for the host-network path below. Binding to port 0 lets the
    kernel choose, and the socket is closed before the port is handed over —
    a small race window, but the alternative (a fixed port) fails outright
    whenever anything else already holds it.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """Start a real Redis and yield its connection URL for the whole session.

    On a daemon with bridge networking (CI) the container publishes a mapped
    port normally. On a sandboxed daemon offering only ``host``/``none`` the
    container shares the host's network stack — and then Redis's default 6379
    may already be taken by something else on the host, which makes the
    container exit immediately with "Address in use". So the host-network path
    picks a free port and tells Redis to listen on it, rather than assuming
    6379 is available.
    """
    container = DockerContainer(_REDIS_IMAGE)
    container.waiting_for(LogMessageWaitStrategy("Ready to accept connections"))
    host_network = not _bridge_network_available()
    port = _REDIS_PORT
    if host_network:
        testcontainers_config.ryuk_disabled = True
        port = _free_tcp_port()
        container.with_kwargs(network_mode="host")
        container.with_command(f"redis-server --port {port}")
        container.ports = {}
    else:
        container.with_exposed_ports(_REDIS_PORT)

    with container as running:
        if host_network:
            yield f"redis://127.0.0.1:{port}/0"
        else:
            host = running.get_container_host_ip()
            mapped = running.get_exposed_port(_REDIS_PORT)
            yield f"redis://{host}:{mapped}/0"


@pytest.fixture
async def redis_client(redis_url: str) -> AsyncIterator[Redis]:
    """Yield a live Redis client against an empty database."""
    client: Redis = Redis.from_url(redis_url)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture(params=["in-memory", "redis"])
def any_store(request: pytest.FixtureRequest, redis_client: Redis) -> RateLimitStore:
    """Yield each :class:`RateLimitStore` implementation in turn.

    The tests below are the *shared contract*: both implementations must
    satisfy them identically. Without this, ``InMemoryFixedWindowStore``'s
    claim to have "the same semantics as Redis" would be an assertion nobody
    checks, and every middleware test driven through it would be measuring
    something other than production behavior.
    """
    if request.param == "in-memory":
        return InMemoryFixedWindowStore()
    return RedisFixedWindowStore(redis_client, timeout_s=2.0)


async def test_store_contract_counts_within_a_window(any_store: RateLimitStore) -> None:
    """Successive hits increment and the TTL is stamped once, then counts down."""
    first_count, first_ttl = await any_store.hit("window-key", 5_000)
    second_count, second_ttl = await any_store.hit("window-key", 5_000)

    assert (first_count, second_count) == (1, 2)
    assert 0 < first_ttl <= 5_000
    assert second_ttl <= first_ttl


async def test_store_contract_window_expires_and_the_counter_restarts(
    any_store: RateLimitStore,
) -> None:
    """After the TTL elapses the counter starts over.

    Deliberately timed against the real clock rather than an injected one:
    Redis expiry cannot be faked, and the point of the contract is that both
    implementations behave the same way under the same conditions.
    """
    assert (await any_store.hit("expiring-key", 300))[0] == 1
    assert (await any_store.hit("expiring-key", 300))[0] == 2
    await asyncio.sleep(0.45)
    count, ttl_ms = await any_store.hit("expiring-key", 300)

    assert count == 1
    assert 0 < ttl_ms <= 300


async def test_store_contract_keys_are_independent(any_store: RateLimitStore) -> None:
    """Two keys keep separate counters."""
    await any_store.hit("key-a", 5_000)
    await any_store.hit("key-a", 5_000)
    count_b, _ = await any_store.hit("key-b", 5_000)

    assert count_b == 1
    assert (await any_store.hit("key-a", 5_000))[0] == 3


async def test_store_contract_ttl_is_not_extended_by_later_hits(
    any_store: RateLimitStore,
) -> None:
    """The window is anchored: a later hit must not push the reset further out."""
    _, first_ttl = await any_store.hit("anchored-key", 1_000)
    await asyncio.sleep(0.2)
    _, later_ttl = await any_store.hit("anchored-key", 1_000)

    assert later_ttl < first_ttl


async def test_end_to_end_429_against_real_redis(redis_client: Redis) -> None:
    """The whole stack — correlation ID, rate limit, CSRF, routing — over real Redis."""
    app = build_test_app(rate_limit_enabled=True)
    app.state.redis = redis_client
    token = app.state.csrf.issue()

    async with _client(app, client_address=("10.1.2.3", 4444)) as client:
        client.cookies.set(CSRF_COOKIE_NAME, token)
        allowed = [
            await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: token}) for _ in range(3)
        ]
        refused = await client.post(
            ECHO_PATH, headers={CSRF_HEADER_NAME: token, "X-Request-ID": "rl-correlation"}
        )

    assert [r.status_code for r in allowed] == [200, 200, 200]
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1
    # The correlation middleware is outermost, so even a rejection is correlated.
    assert refused.headers["x-request-id"] == "rl-correlation"


async def test_reads_stay_available_while_a_client_is_throttled(redis_client: Redis) -> None:
    """The fail-closed argument's premise: throttling never touches the read surface."""
    app = build_test_app(rate_limit_enabled=True)
    app.state.redis = redis_client
    token = app.state.csrf.issue()

    async with _client(app, client_address=("10.1.2.4", 4444)) as client:
        client.cookies.set(CSRF_COOKIE_NAME, token)
        for _ in range(3):
            await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: token})
        assert (await client.post(ECHO_PATH, headers={CSRF_HEADER_NAME: token})).status_code == 429
        assert (await client.get(READ_PATH)).status_code == 200
        assert (await client.get("/api/health")).status_code == 200


async def test_middleware_order_puts_the_limiter_outside_csrf(redis_client: Redis) -> None:
    """A CSRF-less flood is throttled, so failed forgery attempts are not free."""
    app = build_test_app(rate_limit_enabled=True)
    app.state.redis = redis_client

    async with _client(app, client_address=("10.1.2.5", 4444)) as client:
        statuses = [(await client.post(ECHO_PATH)).status_code for _ in range(4)]

    assert statuses == [403, 403, 403, 429]


def test_app_factory_installs_the_limiter_by_default() -> None:
    """Rate limiting is part of the default stack, and correlation IDs wrap it."""
    app = create_app(settings=Settings(environment="test"), security=ApiSecuritySettings())
    installed = installed_middleware(app)

    assert RateLimitMiddleware in installed
    # user_middleware is ordered outermost-first, so the correlation middleware
    # must come before the limiter for a 429 to carry X-Request-ID.
    assert installed.index(CorrelationIdMiddleware) < installed.index(RateLimitMiddleware)


def test_disabling_rate_limiting_removes_the_middleware() -> None:
    """RATE_LIMIT_ENABLED=false is honored (and logged) rather than silently ignored."""
    app = create_app(
        settings=Settings(environment="test"),
        security=ApiSecuritySettings(rate_limit_enabled=False),
    )
    assert RateLimitMiddleware not in installed_middleware(app)
