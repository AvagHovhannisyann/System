"""Health endpoint with per-component checks (DECISIONS.md D-005).

``GET /api/health`` probes the database (``SELECT 1``) and Redis (``PING``),
each bounded by :data:`CHECK_TIMEOUT_S`, and reports per-component status
with latency. HTTP 200 when every component is up, 503 (same body shape)
otherwise. The checkers are injected as FastAPI dependencies so tests can
substitute fakes without a live database or Redis.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger

HealthCheck = Callable[[], Awaitable[None]]
"""A health probe: returns normally when the component is up, raises otherwise."""

CHECK_TIMEOUT_S = 1.0
"""Per-component probe timeout in seconds; exceeding it marks the component down."""

router = APIRouter()
_logger = get_logger(__name__)


class ComponentStatus(BaseModel):
    """Status of a single dependency: up/down plus observed latency.

    ``latency_ms`` is the probe duration in milliseconds; for a failed
    probe it is the time-to-failure (approximately the timeout when the
    probe timed out).
    """

    up: bool
    latency_ms: float


class HealthResponse(BaseModel):
    """Overall health report: ``ok`` only when every component is up."""

    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, ComponentStatus]


def get_db_checker(request: Request) -> HealthCheck:
    """Return the default database probe (``SELECT 1`` on the app engine).

    Assumes the application lifespan stored an ``AsyncEngine`` at
    ``app.state.db_engine``. Tests override this dependency with fakes.
    """
    engine: AsyncEngine = request.app.state.db_engine

    async def check() -> None:
        """Open a connection and execute ``SELECT 1``; raises when the DB is down."""
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    return check


def get_redis_checker(request: Request) -> HealthCheck:
    """Return the default Redis probe (``PING`` on the app client).

    Assumes the application lifespan stored a ``redis.asyncio.Redis`` client
    at ``app.state.redis``. Tests override this dependency with fakes.
    """
    client = request.app.state.redis

    async def check() -> None:
        """Issue a Redis ``PING``; raises when Redis is unreachable."""
        await client.ping()

    return check


async def _probe(name: str, check: HealthCheck, timeout_s: float) -> ComponentStatus:
    """Run one health check with a timeout and report status plus latency.

    Any exception (including timeout) marks the component down; the failure
    is logged, never propagated, so one dead dependency cannot break the
    health endpoint itself.
    """
    start = time.perf_counter()
    up = True
    try:
        async with asyncio.timeout(timeout_s):
            await check()
    except Exception as exc:  # a health probe converts any failure into "down"
        up = False
        _logger.warning("health_check_failed", component=name, error=repr(exc))
    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return ComponentStatus(up=up, latency_ms=latency_ms)


@router.get("/health", response_model=HealthResponse)
async def get_health(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    db_check: Annotated[HealthCheck, Depends(get_db_checker)],
    redis_check: Annotated[HealthCheck, Depends(get_redis_checker)],
) -> HealthResponse:
    """Report per-component health; 200 when all up, 503 otherwise (D-005).

    Probes run concurrently, each bounded by :data:`CHECK_TIMEOUT_S`
    (module attribute, read at call time so tests may patch it).
    """
    db_status, redis_status = await asyncio.gather(
        _probe("db", db_check, CHECK_TIMEOUT_S),
        _probe("redis", redis_check, CHECK_TIMEOUT_S),
    )
    all_up = db_status.up and redis_status.up
    response.status_code = status.HTTP_200_OK if all_up else status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if all_up else "degraded",
        version=settings.app_version,
        components={"db": db_status, "redis": redis_status},
    )
