"""Correlation-ID middleware (DECISIONS.md D-003).

Pure-ASGI middleware: no Starlette ``BaseHTTPMiddleware`` wrapper, so it adds
no response buffering and works with streaming responses.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import structlog
from starlette.datastructures import Headers, MutableHeaders

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"


class CorrelationIdMiddleware:
    """Bind a per-request correlation ID and echo it on the response.

    For every HTTP request: take the incoming ``X-Request-ID`` header (an
    empty value counts as absent) or generate a ``uuid4``, bind it as
    ``request_id`` via ``structlog.contextvars`` for the lifetime of the
    request, and set ``X-Request-ID`` on the response. The binding is
    reset when the request finishes, even on error.

    Assumptions: the incoming header value is trusted as-is (no format
    validation); non-HTTP scopes (lifespan, websocket) pass through
    untouched.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``, the next ASGI application in the stack."""
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event cycle; see the class docstring for semantics."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming or str(uuid.uuid4())
        tokens = structlog.contextvars.bind_contextvars(request_id=request_id)

        async def send_with_request_id(message: Message) -> None:
            """Inject the correlation header into the response start message."""
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self._app(scope, receive, send_with_request_id)
        finally:
            structlog.contextvars.reset_contextvars(**tokens)
