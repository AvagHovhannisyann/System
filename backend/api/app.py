"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from redis.asyncio import Redis

from backend.api.middleware import CorrelationIdMiddleware
from backend.api.routes.health import router as health_router
from backend.core.config import Settings, get_settings
from backend.core.logging import configure_logging
from backend.db.engine import create_db_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients on startup and dispose of them on shutdown.

    Stores the async database engine at ``app.state.db_engine`` and the
    Redis client at ``app.state.redis``. Both are lazy: no network I/O
    happens until first use, so startup succeeds even when the backing
    services are down (the health endpoint then reports them as down).
    """
    settings = get_settings()
    engine = create_db_engine(settings)
    redis: Redis = Redis.from_url(settings.redis_url)
    app.state.db_engine = engine
    app.state.redis = redis
    try:
        yield
    finally:
        await redis.aclose()
        await engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application: logging, correlation IDs, routers.

    Configures structured logging (idempotent), installs the
    correlation-ID middleware, and mounts all routers under ``/api``.
    When ``settings`` is omitted the cached application settings are used.
    """
    resolved = settings if settings is not None else get_settings()
    configure_logging(resolved)
    app = FastAPI(
        title="quant-research-platform",
        version=resolved.app_version,
        lifespan=_lifespan,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(health_router, prefix="/api")
    return app
