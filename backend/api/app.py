"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from redis.asyncio import Redis

from backend.api.middleware import CorrelationIdMiddleware
from backend.api.routes.health import router as health_router
from backend.api.security import (
    ApiSecuritySettings,
    CsrfMiddleware,
    CsrfProtect,
    RateLimitMiddleware,
    RateLimitPolicy,
    resolve_csrf_secret,
)
from backend.core.config import Settings, get_settings
from backend.core.logging import configure_logging, get_logger
from backend.db import create_admin_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients on startup and dispose of them on shutdown.

    Stores an *admin* async database engine at ``app.state.db_engine`` (used
    only by the health probe — never a fact-table read path, per D-011) and
    the Redis client at ``app.state.redis``. Both are lazy: no network I/O
    happens until first use, so startup succeeds even when the backing
    services are down (the health endpoint then reports them as down).
    """
    settings = get_settings()
    engine = create_admin_engine(settings)
    redis: Redis = Redis.from_url(settings.redis_url)
    app.state.db_engine = engine
    app.state.redis = redis
    try:
        yield
    finally:
        await redis.aclose()
        await engine.dispose()


def create_app(
    settings: Settings | None = None,
    security: ApiSecuritySettings | None = None,
) -> FastAPI:
    """Build the FastAPI application: logging, middleware, routers.

    Configures structured logging (idempotent), installs the middleware
    stack, and mounts all routers under ``/api``. When ``settings`` is
    omitted the cached application settings are used; when ``security`` is
    omitted the §7 security settings are read fresh from the environment
    (deliberately uncached — see :mod:`backend.api.security.settings`).

    Middleware order — the intended request path, outermost first:

    1. :class:`~backend.api.middleware.CorrelationIdMiddleware` — outermost so
       that *every* response, including a 403 from CSRF or a 429 from the rate
       limiter, carries ``X-Request-ID`` and every rejection is logged under a
       correlation ID.
    2. :class:`~backend.api.security.ratelimit.RateLimitMiddleware` — outside
       CSRF, so a flood is shed at the cheaper point and so failed CSRF
       attempts still consume the attacker's budget rather than being free.
    3. :class:`~backend.api.security.csrf.CsrfMiddleware` — innermost of the
       three, immediately in front of routing.

    **The ``add_middleware`` calls below are therefore in the reverse of that
    order.** Starlette's ``add_middleware`` *inserts at the front* of
    ``user_middleware``, so the **last** one added ends up outermost. Reading
    the calls top-to-bottom as the request path is the natural mistake here;
    ``test_api_security.py`` and ``test_api_rate_limit.py`` both assert that a
    rejection response still carries ``X-Request-ID``, which fails the moment
    this order is "corrected".

    Both security middlewares are installed unconditionally (rate limiting
    subject to ``RATE_LIMIT_ENABLED``) and default to *enforcing*, so a
    router added later inherits protection without opting in. See
    :mod:`backend.api.security` for that design choice.

    ``app.state.csrf`` holds the :class:`~backend.api.security.csrf.CsrfProtect`
    instance so an in-process client (tests, and any future server-rendered
    view) can mint a valid token without reaching into the middleware stack.
    """
    resolved = settings if settings is not None else get_settings()
    resolved_security = security if security is not None else ApiSecuritySettings()
    configure_logging(resolved)

    app = FastAPI(
        title="quant-research-platform",
        version=resolved.app_version,
        lifespan=_lifespan,
    )

    csrf = CsrfProtect(
        resolve_csrf_secret(resolved_security, resolved.environment),
        max_age_s=resolved_security.csrf_token_max_age_s,
    )
    app.state.csrf = csrf

    # Added innermost-first; see the docstring — add_middleware inserts at the
    # front, so the last call below is the outermost middleware.
    app.add_middleware(
        CsrfMiddleware,
        protect=csrf,
        secure_cookie=resolved.environment == "prod",
    )
    if resolved_security.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            policy=RateLimitPolicy(
                limit=resolved_security.rate_limit_requests,
                window_s=resolved_security.rate_limit_window_s,
            ),
            timeout_s=resolved_security.rate_limit_timeout_s,
        )
    else:
        _logger.warning(
            "rate_limit_disabled",
            detail=(
                "RATE_LIMIT_ENABLED is false; mutating endpoints are unthrottled "
                "(DIRECTIVE.md section 7 requires rate limiting on all mutating endpoints)"
            ),
        )
    app.add_middleware(CorrelationIdMiddleware)

    app.include_router(health_router, prefix="/api")
    return app
