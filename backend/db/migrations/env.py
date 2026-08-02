"""Alembic migration environment (async-compatible).

Executed by the alembic CLI. The database URL always comes from backend
settings; ``alembic.ini`` stores no URL. Importing this module outside an
alembic run fails fast on ``context.config`` — that is intentional.

**Migrations connect as the schema owner, never as the application role**
(CC.9, D-017). The URL is ``settings.migration_url``
(``MIGRATION_DATABASE_URL``, falling back to ``DATABASE_URL``), not
``DATABASE_URL`` directly. The application's own credential names a role that
holds ``SELECT``/``INSERT`` and owns nothing, so it cannot create a table — and,
which is the point, cannot drop or disable the append-only triggers that make
the audit log immutable. Pointing this environment at ``DATABASE_URL`` in a
separated deployment does not silently weaken anything: PostgreSQL refuses the
first DDL statement.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig
from typing import TYPE_CHECKING

from alembic import context
from sqlalchemy import pool

from backend.core.config import get_settings
from backend.db.base import Base
from backend.db.engine import _create_migration_engine

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the migration database URL from application settings.

    Honors ``MIGRATION_DATABASE_URL`` (the schema-owner credential), falling
    back to ``DATABASE_URL``, via :func:`backend.core.config.get_settings` —
    see :attr:`backend.core.config.Settings.migration_url` and the module
    docstring for why migrations must not use the application role.
    """
    return get_settings().migration_url


def run_migrations_offline() -> None:
    """Emit migration SQL without connecting to a database (``--sql`` mode)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations on an established (sync-facade) connection."""
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine, run migrations through it, and dispose of it.

    The engine comes from the module-private *migration* factory — the one
    engine deliberately created **without** the Core-level bitemporal read
    guard, because migration DDL necessarily names fact tables (e.g.
    revision 0003's ``create_hypertable('price_bar', ...)``), and the one engine
    that connects on ``settings.migration_url`` as the schema owner (CC.9).
    ``NullPool`` because a migration run needs exactly one short-lived
    connection.
    """
    connectable = _create_migration_engine(poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entrypoint for online migrations; drives the async engine to completion."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
