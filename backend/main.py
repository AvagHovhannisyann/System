"""ASGI entrypoint: ``uvicorn backend.main:app``.

Kept intentionally minimal — all wiring lives in
:func:`backend.api.app.create_app` so tests build their own instances.
"""

from backend.api.app import create_app

app = create_app()
"""The ASGI application instance served by uvicorn."""
