"""Async engine and session-factory construction.

Factory functions only — importing this module performs no I/O and builds
no global engine. Callers own the returned objects' lifecycles.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import Settings, get_settings


def create_db_engine(settings: Settings | None = None) -> AsyncEngine:
    """Create a new async engine from settings.

    Assumes ``settings.database_url`` uses an async driver
    (``postgresql+asyncpg``). ``pool_pre_ping`` guards against stale pooled
    connections. The caller is responsible for ``await engine.dispose()``.
    When ``settings`` is omitted the cached application settings are used.
    """
    resolved = settings if settings is not None else get_settings()
    return create_async_engine(resolved.database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an ``async_sessionmaker`` bound to ``engine``.

    ``expire_on_commit=False`` so ORM objects stay usable after commit in
    async code paths without implicit lazy-load I/O.
    """
    return async_sessionmaker(engine, expire_on_commit=False)
