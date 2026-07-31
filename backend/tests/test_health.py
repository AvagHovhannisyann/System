"""Tests for /api/health with injected fake checkers (no live DB/Redis)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.app import create_app
from backend.api.routes import health as health_module
from backend.core.config import get_settings


async def _ok() -> None:
    """Fake checker that reports a healthy component."""


async def _boom() -> None:
    """Fake checker that reports a dead component."""
    raise RuntimeError("component down")


def _app_with_checkers(
    db_check: health_module.HealthCheck,
    redis_check: health_module.HealthCheck,
) -> FastAPI:
    """Build the real app with both health checkers overridden."""
    app = create_app()
    app.dependency_overrides[health_module.get_db_checker] = lambda: db_check
    app.dependency_overrides[health_module.get_redis_checker] = lambda: redis_check
    return app


def _client(app: FastAPI) -> AsyncClient:
    """Create an httpx client speaking ASGI directly to ``app``."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def test_all_healthy_returns_200_with_full_shape() -> None:
    """Healthy fakes: 200, status ok, version, and per-component up/latency."""
    async with _client(_app_with_checkers(_ok, _ok)) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == get_settings().app_version
    assert set(body["components"]) == {"db", "redis"}
    for component in body["components"].values():
        assert component["up"] is True
        assert isinstance(component["latency_ms"], float | int)
        assert component["latency_ms"] >= 0
    # The correlation middleware is wired into the real app.
    assert "x-request-id" in response.headers


async def test_failing_db_returns_503_degraded() -> None:
    """A failing DB checker yields 503 with db down and redis still up."""
    async with _client(_app_with_checkers(_boom, _ok)) as client:
        response = await client.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["db"]["up"] is False
    assert body["components"]["redis"]["up"] is True


async def test_failing_redis_returns_503_degraded() -> None:
    """A failing Redis checker yields 503 with redis down and db still up."""
    async with _client(_app_with_checkers(_ok, _boom)) as client:
        response = await client.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["components"]["db"]["up"] is True
    assert body["components"]["redis"]["up"] is False


async def test_slow_checker_times_out_as_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checker exceeding the probe timeout marks its component down."""
    monkeypatch.setattr(health_module, "CHECK_TIMEOUT_S", 0.02)

    async def slow() -> None:
        """Fake checker that outlives the (patched) probe timeout."""
        await asyncio.sleep(0.5)

    async with _client(_app_with_checkers(slow, _ok)) as client:
        response = await client.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["components"]["db"]["up"] is False
    assert body["components"]["redis"]["up"] is True
