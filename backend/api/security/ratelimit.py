"""Redis-backed rate limiting on mutating requests (DIRECTIVE.md §7, CC.2).

Algorithm: **anchored fixed window**
------------------------------------

One Redis counter per ``(client, route)`` pair. The first request of a window
sets the counter to 1 and stamps a TTL of the window length; every later
request increments it. When the TTL expires the key vanishes and the next
request starts a fresh window. The counter and the TTL stamp are applied in a
single Lua script, so a process that dies mid-request cannot leave an
immortal counter behind — the failure mode of the naive ``INCR`` then
``EXPIRE`` pair, which locks a client out permanently.

The window is anchored at the client's first request rather than aligned to
wall-clock boundaries. Aligned windows admit up to ``2 * limit`` requests
across a boundary; anchored ones bound any window-length interval that starts
at a request to ``limit``. A token bucket would smooth bursts further, but it
needs two values (level and last-refill timestamp) updated together, i.e. a
longer script and a stored floating-point clock, to protect an endpoint whose
realistic abuse is a stuck retry loop on an operator dashboard button. The
extra machinery buys nothing here.

Design choice: **fail closed when Redis is unavailable**
--------------------------------------------------------

If the store cannot be reached (down, timing out, or never configured), a
mutating request is refused with ``503`` — it is *not* waved through.

The usual argument for failing open is availability: a limiter outage should
not take down the product. That argument does not survive contact with this
system's shape.

- **Nothing readable is affected.** Safe methods are exempt, so the entire
  read surface of the research dashboard — every chart, every table, the
  health page an operator would be staring at during an outage — is
  untouched. Failing closed costs the ability to *mutate*, not the ability to
  look.
- **The mutations could not have worked anyway.** Redis is this platform's
  Celery broker, and it is a critical component of ``/api/health`` (D-005:
  Redis down means the API already reports itself degraded). The first
  consumer of this limiter is the Data Health manual re-sync trigger, which
  does nothing but enqueue a Celery job. Failing open would accept the
  request, return success, and drop the work — which is a *worse* outage than
  a 503, because it is a silent one.
- **What is unthrottled is expensive.** Mutating endpoints here start
  ingestion runs and, from Phase 7, LLM extraction under a hard daily spend
  cap (§6.5, §7). An unthrottled retry storm against those costs money and
  burns third-party quota — including SEC EDGAR fair-access budget, where the
  penalty for exceeding it is a ban a retry loop cannot fix. The moment the
  infrastructure is unhealthy is precisely when retry storms happen.
- **Single operator.** There is no multi-tenant availability SLA to weigh
  this against. The cost of a false refusal is one person retrying after
  Redis comes back.

The refusal is ``503``, not ``429``: the client did nothing wrong, and
conflating "you are over your budget" with "the limiter is broken" would hide
an infrastructure outage inside a normal throttling signal. A ``Retry-After``
is still sent.

``fail_open`` exists as a constructor argument so the behavior is testable and
so an operator who genuinely needs the other trade-off can make it — but it is
**deliberately not exposed as an environment variable**. Turning off a §7
control should require a code change and a ``DECISIONS.md`` entry, not a
line in ``.env`` (same reasoning as D-008's rejection of
``continue-on-error`` on the dependency scanner).

Client identity
---------------

The bucket key is the transport peer address from the ASGI scope. The
``X-Forwarded-For`` header is **deliberately not consulted**: no trusted-proxy
allowlist is configured, and an unvalidated forwarding header is a
rate-limit bypass — any client can spoof it and mint themselves a fresh
bucket per request. If a reverse proxy is ever put in front of this API, that
allowlist must be added here first.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast

from redis.exceptions import RedisError
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

from backend.api.security.csrf import SAFE_METHODS
from backend.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from redis.asyncio import Redis
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = [
    "RATE_LIMIT_EXEMPT_PATHS",
    "InMemoryFixedWindowStore",
    "RateLimitDecision",
    "RateLimitMiddleware",
    "RateLimitPolicy",
    "RateLimitStore",
    "RateLimiterUnavailableError",
    "RedisFixedWindowStore",
    "client_key",
    "decide",
]

RATE_LIMIT_EXEMPT_PATHS: Final[frozenset[str]] = frozenset()
"""Exact paths exempt from rate limiting on unsafe methods.

Empty by design, and exact-match only, for the same reason as
:data:`backend.api.security.csrf.CSRF_EXEMPT_PATHS`: a prefix would silently
exempt routes added underneath it later.
"""

UNSAFE_METHOD_HEADER_LIMIT: Final = "X-RateLimit-Limit"
"""Response header carrying the budget for the window."""

UNSAFE_METHOD_HEADER_REMAINING: Final = "X-RateLimit-Remaining"
"""Response header carrying the requests left in the current window."""

UNSAFE_METHOD_HEADER_RESET: Final = "X-RateLimit-Reset"
"""Response header carrying seconds until the current window resets."""

_KEY_PREFIX: Final = "ratelimit:v1"
"""Namespace for limiter keys in Redis; versioned so a semantics change can
start from a clean keyspace instead of inheriting counters written under
different rules."""

_HIT_SCRIPT: Final = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('PTTL', KEYS[1])}
"""
"""Atomic increment-and-stamp. Returns ``{count, ttl_ms}``.

``PTTL`` is read inside the script rather than by a second round trip so the
TTL reported to the client belongs to the same window as the count.
"""

_logger = get_logger(__name__)


class RateLimiterUnavailableError(RuntimeError):
    """The rate-limit store could not be consulted for this request."""


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """A request budget.

    Attributes:
        limit: requests allowed per key per window (>= 1).
        window_s: window length in seconds (> 0).
    """

    limit: int
    window_s: float

    def __post_init__(self) -> None:
        """Reject a nonsensical policy rather than silently allowing everything."""
        if self.limit < 1:
            msg = f"limit must be >= 1 request; got {self.limit}"
            raise ValueError(msg)
        if self.window_s <= 0:
            msg = f"window_s must be > 0 seconds; got {self.window_s}"
            raise ValueError(msg)

    @property
    def window_ms(self) -> int:
        """Window length in whole milliseconds (Redis TTL granularity), at least 1."""
        return max(1, round(self.window_s * 1000))


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Outcome of one limiter consultation.

    Attributes:
        allowed: whether the request may proceed.
        limit: the budget in force, requests per window.
        remaining: requests left in this window (never negative).
        reset_after_s: whole seconds until the window resets, at least 1.
            Doubles as the ``Retry-After`` value when ``allowed`` is false.
    """

    allowed: bool
    limit: int
    remaining: int
    reset_after_s: int


class RateLimitStore(Protocol):
    """A counter store supporting atomic increment-with-expiry."""

    async def hit(self, key: str, window_ms: int) -> tuple[int, int]:
        """Increment ``key``, stamping ``window_ms`` TTL on first use.

        Args:
            key: the counter key.
            window_ms: window length in milliseconds, applied as the TTL when
                this call creates the key.

        Returns:
            ``(count, ttl_ms)`` — the post-increment count and the remaining
            TTL of the key in milliseconds.

        Raises:
            RateLimiterUnavailableError: the store could not be reached.
        """
        ...


class RedisFixedWindowStore:
    """:class:`RateLimitStore` backed by Redis via :data:`_HIT_SCRIPT`.

    Every store failure — connection error, protocol error, timeout — is
    converted into :class:`RateLimiterUnavailableError` so the middleware has
    exactly one condition to reason about.
    """

    def __init__(self, client: Redis, *, timeout_s: float) -> None:
        """Wrap a Redis client.

        Args:
            client: an async Redis client (typically ``app.state.redis``).
            timeout_s: per-call timeout in seconds; a hung Redis must not hang
                the API, so exceeding it is treated as unavailability.
        """
        self._client = client
        self._timeout_s = timeout_s

    async def hit(self, key: str, window_ms: int) -> tuple[int, int]:
        """Run the atomic hit script; see :meth:`RateLimitStore.hit`."""
        try:
            async with asyncio.timeout(self._timeout_s):
                values = await cast(
                    "Awaitable[list[int]]",
                    self._client.eval(_HIT_SCRIPT, 1, key, str(window_ms)),
                )
        except (RedisError, OSError, TimeoutError) as exc:
            msg = f"rate-limit store unavailable: {exc!r}"
            raise RateLimiterUnavailableError(msg) from exc
        return int(values[0]), int(values[1])


class InMemoryFixedWindowStore:
    """Process-local :class:`RateLimitStore` with the same semantics as Redis.

    **Not for production.** It exists so the middleware's decision logic can be
    exercised against an injectable clock, which Redis expiry cannot provide.
    Its equivalence to :class:`RedisFixedWindowStore` is not assumed: both are
    run against the same contract tests (``test_store_contract_*`` in
    ``backend/tests/test_api_rate_limit.py``), so a divergence fails the suite
    rather than quietly invalidating every test that uses this class.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Create an empty store.

        Args:
            clock: monotonic time source in seconds; injectable so window
                expiry can be tested without wall-clock sleeping.
        """
        self._clock = clock
        self._counters: dict[str, tuple[int, float]] = {}

    async def hit(self, key: str, window_ms: int) -> tuple[int, int]:
        """Increment ``key``, expiring it lazily; see :meth:`RateLimitStore.hit`."""
        now = self._clock()
        count, expires_at = self._counters.get(key, (0, 0.0))
        if count == 0 or now >= expires_at:
            count, expires_at = 0, now + window_ms / 1000.0
        count += 1
        self._counters[key] = (count, expires_at)
        return count, max(0, round((expires_at - now) * 1000))


def client_key(scope: Scope) -> str:
    """Return the bucket identity for the client issuing this request.

    The ASGI transport peer address only — see the module docstring on why
    ``X-Forwarded-For`` is not consulted. A scope with no client address (some
    transports omit it) falls into a single shared ``unknown`` bucket, which
    errs toward throttling more rather than less.
    """
    client = scope.get("client")
    if not client:
        return "unknown"
    host = client[0]
    return str(host) if host else "unknown"


def decide(count: int, ttl_ms: int, policy: RateLimitPolicy) -> RateLimitDecision:
    """Turn a raw ``(count, ttl_ms)`` store result into a decision.

    Args:
        count: post-increment request count in the current window.
        ttl_ms: remaining window in milliseconds.
        policy: the budget in force.

    Returns:
        The decision, with ``reset_after_s`` rounded *up* to whole seconds and
        floored at 1 — a ``Retry-After: 0`` invites an immediate retry that is
        certain to be refused again.
    """
    reset_after_s = max(1, math.ceil(max(ttl_ms, 0) / 1000))
    return RateLimitDecision(
        allowed=count <= policy.limit,
        limit=policy.limit,
        remaining=max(0, policy.limit - count),
        reset_after_s=reset_after_s,
    )


class RateLimitMiddleware:
    """Rate-limit unsafe-method requests per client and route.

    Pure-ASGI, like the other middleware in this application. Safe methods
    (:data:`~backend.api.security.csrf.SAFE_METHODS`) pass through without
    touching the store at all, so the read-only dashboard costs no Redis
    round trips and keeps working when Redis is down.

    Assumptions: the store is resolved once per request; when none was
    injected it is built from ``scope["app"].state.redis``, which the
    application lifespan populates. A missing client address shares one
    bucket (see :func:`client_key`).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        policy: RateLimitPolicy,
        timeout_s: float,
        store: RateLimitStore | None = None,
        exempt_paths: frozenset[str] = RATE_LIMIT_EXEMPT_PATHS,
        fail_open: bool = False,
    ) -> None:
        """Wrap ``app`` with per-client rate limiting on mutating requests.

        Args:
            app: the next ASGI application in the stack.
            policy: the request budget.
            timeout_s: per-call store timeout in seconds.
            store: explicit store; when ``None`` a
                :class:`RedisFixedWindowStore` is built per request from
                ``app.state.redis``.
            exempt_paths: exact paths never rate limited. Defaults to
                :data:`RATE_LIMIT_EXEMPT_PATHS`.
            fail_open: allow requests through when the store is unreachable.
                Defaults to ``False``; see the module docstring for why, and
                why this is not an environment variable.
        """
        self._app = app
        self._policy = policy
        self._timeout_s = timeout_s
        self._store = store
        self._exempt_paths = exempt_paths
        self._fail_open = fail_open

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event cycle; see the class docstring for semantics."""
        if scope["type"] != "http" or scope["method"] in SAFE_METHODS:
            await self._app(scope, receive, send)
            return

        path = scope["path"]
        if path in self._exempt_paths:
            await self._app(scope, receive, send)
            return

        identity = client_key(scope)
        try:
            store = self._resolve_store(scope)
            count, ttl_ms = await store.hit(
                f"{_KEY_PREFIX}:{identity}:{path}", self._policy.window_ms
            )
        except RateLimiterUnavailableError as exc:
            await self._handle_unavailable(exc, scope, receive, send)
            return

        decision = decide(count, ttl_ms, self._policy)
        if not decision.allowed:
            _logger.warning(
                "rate_limit_exceeded",
                client=identity,
                path=path,
                method=scope["method"],
                limit=decision.limit,
                retry_after_s=decision.reset_after_s,
            )
            await self._refuse(decision, scope, receive, send)
            return

        await self._app(scope, receive, _with_budget_headers(send, decision))

    def _resolve_store(self, scope: Scope) -> RateLimitStore:
        """Return the injected store, or build one from ``app.state.redis``.

        Raises:
            RateLimiterUnavailableError: when no Redis client is configured on
                the application — a misconfiguration is an outage, and under
                the fail-closed policy it is treated as one.
        """
        if self._store is not None:
            return self._store
        app = scope.get("app")
        client = getattr(app.state, "redis", None) if app is not None else None
        if client is None:
            msg = "no Redis client on app.state.redis; rate-limit store unconfigured"
            raise RateLimiterUnavailableError(msg)
        return RedisFixedWindowStore(client, timeout_s=self._timeout_s)

    async def _handle_unavailable(
        self,
        error: RateLimiterUnavailableError,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Apply the fail-closed (or configured fail-open) policy."""
        if self._fail_open:
            _logger.error(
                "rate_limit_store_unavailable_failing_open",
                path=scope["path"],
                method=scope["method"],
                error=repr(error),
            )
            await self._app(scope, receive, send)
            return

        _logger.error(
            "rate_limit_store_unavailable_failing_closed",
            path=scope["path"],
            method=scope["method"],
            error=repr(error),
        )
        retry_after = max(1, math.ceil(self._policy.window_s))
        response = JSONResponse(
            {
                "detail": "Rate limiting is unavailable; mutating requests are refused.",
                "reason": "rate_limiter_unavailable",
            },
            status_code=503,
            headers={"Retry-After": str(retry_after)},
        )
        await response(scope, receive, send)

    async def _refuse(
        self, decision: RateLimitDecision, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Answer an over-budget request with 429 and the retry metadata."""
        response = JSONResponse(
            {"detail": "Rate limit exceeded.", "reason": "rate_limit_exceeded"},
            status_code=429,
            headers={
                "Retry-After": str(decision.reset_after_s),
                UNSAFE_METHOD_HEADER_LIMIT: str(decision.limit),
                UNSAFE_METHOD_HEADER_REMAINING: "0",
                UNSAFE_METHOD_HEADER_RESET: str(decision.reset_after_s),
            },
        )
        await response(scope, receive, send)


def _with_budget_headers(send: Send, decision: RateLimitDecision) -> Send:
    """Wrap ``send`` so an allowed response advertises the remaining budget."""

    async def send_with_headers(message: Message) -> None:
        """Attach the budget headers to the response start message."""
        if message["type"] == "http.response.start":
            headers = MutableHeaders(scope=message)
            headers[UNSAFE_METHOD_HEADER_LIMIT] = str(decision.limit)
            headers[UNSAFE_METHOD_HEADER_REMAINING] = str(decision.remaining)
            headers[UNSAFE_METHOD_HEADER_RESET] = str(decision.reset_after_s)
        await send(message)

    return send_with_headers
