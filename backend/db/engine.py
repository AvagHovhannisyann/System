"""Private engine and session-factory state for the database layer.

**Import contract (D-011 bypass-prevention layer 3):** this module is
private to ``backend/db``. Ruff rule TID251 bans importing it anywhere else
in the codebase; the only public handles are re-exported by
:mod:`backend.db` — ``as_of``/``ingest_writer_session`` for data access and
the explicitly-named admin helpers below.

The raw sessionmaker is module-private (layer 1): the only sanctioned ways
to obtain a session are :func:`backend.db.asof.as_of` (versioned reads) and
:func:`backend.db.asof.ingest_writer_session` (append-only writes). Even a
session built in defiance of this contract is still caught at runtime by
the class-level ``do_orm_execute`` hook (layer 2).

Every engine this module creates — the process-wide application engine and
every admin engine — carries the **Core-level bitemporal read guard**
(:func:`backend.db._guard.install_core_guard`): compiled Core selects and
textual SQL referencing a bitemporal fact table raise
:class:`~backend.db._guard.BitemporalBypassError` at the engine, so raw
``Connection`` access (``session.connection()``, the admin engine,
``exec_driver_sql``) cannot read fact tables unversioned. The single
deliberate exception is :func:`_create_migration_engine` — module-private,
for alembic runs and the sanctioned test/db reset, where DDL like
``create_hypertable('price_bar', ...)`` and ``TRUNCATE`` must execute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import get_settings
from backend.db._guard import install_core_guard

if TYPE_CHECKING:
    from backend.core.config import Settings

__all__ = ["create_admin_engine", "dispose_database"]

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_writer_session_factory: async_sessionmaker[AsyncSession] | None = None


def _build_guarded_engine(url: str) -> AsyncEngine:
    """Create an async engine with the Core-level bitemporal read guard installed.

    The single construction path for every guarded engine this module hands
    out, so no engine can be created here without the guard by accident.
    ``pool_pre_ping`` guards against stale pooled connections.
    """
    engine = create_async_engine(url, pool_pre_ping=True)
    install_core_guard(engine.sync_engine)
    return engine


def _get_engine() -> AsyncEngine:
    """Return the lazily-created process-wide async engine (Core guard installed).

    Built from ``settings.database_url`` (``postgresql+asyncpg``) on first
    use. Reset via :func:`dispose_database`.
    """
    global _engine
    if _engine is None:
        _engine = _build_guarded_engine(get_settings().database_url)
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the module-private read sessionmaker (D-011 layer 1).

    Only :func:`backend.db.asof.as_of` may call this. ``expire_on_commit=False``
    so ORM objects stay usable after commit without implicit lazy-load I/O.
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _session_factory


def _get_writer_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the module-private *writer* sessionmaker (D-011 layer 1).

    Distinct from the read factory so the ingestion write path is a separate,
    auditable session source. Only :func:`backend.db.asof.ingest_writer_session`
    may call this. Sessions from it permit INSERT/flush; the class-level
    ``do_orm_execute`` hook still rejects unversioned SELECTs on bitemporal
    tables (D-011 layer 2). Shares the process-wide engine with the read
    factory — the isolation that matters is semantic, not connection-level.
    """
    global _writer_session_factory
    if _writer_session_factory is None:
        _writer_session_factory = async_sessionmaker(_get_engine(), expire_on_commit=False)
    return _writer_session_factory


async def dispose_database() -> None:
    """Admin helper: dispose the process-wide engine and reset lazy state.

    For application shutdown and test isolation (each pytest event loop must
    own its connection pool). Safe to call when nothing was created; the
    next session request rebuilds engine and factory from current settings.
    """
    global _engine, _session_factory, _writer_session_factory
    engine = _engine
    _engine = None
    _session_factory = None
    _writer_session_factory = None
    if engine is not None:
        await engine.dispose()


def create_admin_engine(settings: Settings | None = None) -> AsyncEngine:
    """Admin helper: create a **new** engine for infrastructure concerns.

    Sanctioned uses: health-check probes and catalog/monitoring queries —
    code that must talk to the server but never reads fact rows. This is
    *not* a data read path, and the restriction is **enforced**, not merely
    documented: the Core-level guard is installed on the returned engine, so
    any compiled or textual statement referencing a bitemporal fact table in
    a read capacity (including a fail-closed name-scan of raw SQL, and
    ``TRUNCATE`` of fact tables) raises
    :class:`~backend.db._guard.BitemporalBypassError` before any I/O. The
    caller owns the engine lifecycle (``await engine.dispose()``). When
    ``settings`` is omitted the cached application settings are used.
    """
    resolved = settings if settings is not None else get_settings()
    return _build_guarded_engine(resolved.database_url)


def _create_migration_engine(
    settings: Settings | None = None,
    # ANN401: create_async_engine's own kwargs are heterogeneous and untyped
    # upstream (poolclass, echo, connect_args, ...); narrowing here would only
    # be a guess that drifts from SQLAlchemy's signature.
    **engine_kwargs: Any,  # noqa: ANN401
) -> AsyncEngine:
    """Module-private: create an engine **without** the Core-level read guard.

    The only unguarded engine factory. Sanctioned uses, both inside
    ``backend/db``'s trust boundary:

    - the alembic migration environment (``migrations/env.py``), whose DDL
      necessarily names fact tables (``create_hypertable('price_bar', ...)``,
      triggers, indices); and
    - the integration-test database reset (``TRUNCATE`` of fact tables),
      reached via dynamic import in the test fixtures.

    Not importable outside ``backend/db`` (ruff TID251 bans this module
    elsewhere); it is deliberately absent from ``__all__``. ``engine_kwargs``
    pass through to ``create_async_engine`` (e.g. ``poolclass``). When
    ``settings`` is omitted the cached application settings are used.
    """
    resolved = settings if settings is not None else get_settings()
    return create_async_engine(resolved.database_url, **engine_kwargs)
