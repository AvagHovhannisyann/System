"""Tests for the correlation-ID middleware: header generation, propagation, cleanup."""

from __future__ import annotations

import uuid

import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.api.middleware import REQUEST_ID_HEADER, CorrelationIdMiddleware


def _build_app() -> FastAPI:
    """Build a minimal app whose single route reports the bound request_id."""
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ctx")
    async def read_context() -> dict[str, str]:
        """Return the request_id currently bound in structlog contextvars."""
        bound = structlog.contextvars.get_contextvars()
        return {"request_id": str(bound.get("request_id"))}

    return app


def _client(app: FastAPI) -> AsyncClient:
    """Create an httpx client speaking ASGI directly to ``app``."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver")


async def test_generates_uuid4_when_header_absent() -> None:
    """Without an incoming header, a uuid4 is generated, bound, and returned."""
    app = _build_app()
    async with _client(app) as client:
        response = await client.get("/ctx")

    header_value = response.headers[REQUEST_ID_HEADER]
    assert uuid.UUID(header_value).version == 4
    # The same ID was bound in contextvars during request handling.
    assert response.json()["request_id"] == header_value


async def test_propagates_incoming_header() -> None:
    """An incoming X-Request-ID is reused verbatim for binding and response."""
    app = _build_app()
    async with _client(app) as client:
        response = await client.get("/ctx", headers={REQUEST_ID_HEADER: "abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "abc-123"
    assert response.json()["request_id"] == "abc-123"


async def test_binding_reset_after_request() -> None:
    """The request_id binding does not leak past the end of the request."""
    app = _build_app()
    async with _client(app) as client:
        await client.get("/ctx", headers={REQUEST_ID_HEADER: "leak-check"})

    assert "request_id" not in structlog.contextvars.get_contextvars()
