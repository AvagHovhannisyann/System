"""Fixtures: session-scoped TimescaleDB container, migrated schema, per-test isolation.

The container (image pinned per D-004) lives for the whole test session;
``alembic upgrade head`` runs against it once, programmatically. Each test
then sees a clean, fully-migrated database: an autouse fixture truncates the
fact tables afterwards (TRUNCATE is the sanctioned admin/test reset — the
append-only triggers deliberately block only UPDATE/DELETE) and disposes the
process-wide engine so every test's event loop builds a fresh connection
pool.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.config import testcontainers_config
from testcontainers.core.docker_client import DockerClient

import backend.db.models  # noqa: F401 — imported for its side effect: populating the metadata
from backend.core.config import get_settings
from backend.db import dispose_database
from backend.db.base import Base

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TIMESCALE_IMAGE = "timescale/timescaledb:2.17.2-pg16"
_PG_CREDENTIAL = "quant"  # throwaway credential for an ephemeral test container


def _bridge_network_available() -> bool:
    """Return True when the docker daemon offers the default ``bridge`` network.

    Some sandboxed daemons ship only ``host``/``none`` networks; there, port
    publishing cannot allocate a mapping, so the container must run with host
    networking instead (and Ryuk — which itself needs a published port — is
    disabled; the fixture stops the database container deterministically via
    its context manager, so no reaper is required).
    """
    client = DockerClient()
    try:
        return bool(client.client.networks.list(names=["bridge"]))
    finally:
        client.client.close()


@pytest.fixture(scope="session")
def migrated_database_url() -> Iterator[str]:
    """Start TimescaleDB, point ``DATABASE_URL`` at it, run ``alembic upgrade head``.

    Yields the asyncpg connection URL. The environment override is undone at
    session end. The settings cache is cleared so the migration environment
    (and every later engine build) resolves the container URL. Readiness is
    probed by ``psql`` *inside* the container, so it works under both
    networking modes.
    """
    monkeypatch = pytest.MonkeyPatch()
    container = PostgresContainer(
        _TIMESCALE_IMAGE,
        username=_PG_CREDENTIAL,
        password=_PG_CREDENTIAL,
        dbname=_PG_CREDENTIAL,
        driver="asyncpg",
    )
    host_network = not _bridge_network_available()
    if host_network:
        testcontainers_config.ryuk_disabled = True
        container.with_kwargs(network_mode="host")
        container.ports = {}  # ports cannot be published without a bridge network
    try:
        with container as running:
            url = (
                f"postgresql+asyncpg://{_PG_CREDENTIAL}:{_PG_CREDENTIAL}"
                f"@127.0.0.1:5432/{_PG_CREDENTIAL}"
                if host_network
                else running.get_connection_url()
            )
            monkeypatch.setenv("DATABASE_URL", url)
            get_settings.cache_clear()
            config = Config(str(_REPO_ROOT / "alembic.ini"))
            config.set_main_option(
                "script_location", str(_REPO_ROOT / "backend" / "db" / "migrations")
            )
            command.upgrade(config, "head")
            yield url
    finally:
        monkeypatch.undo()


def _truncate_every_table() -> str:
    """Return a TRUNCATE naming every mapped table, derived from the metadata.

    **Derived rather than listed, because the list went stale and nobody
    noticed.** This reset used to name four tables — ``price_bar``,
    ``security_master``, ``security``, ``ingestion_run`` — which were all the
    tables that existed when it was written. Fourteen more arrived since
    (EDGAR, macro, extraction, universe, execution, monitoring), and none of
    them was ever reset between tests.

    That is invisible until a table carries state a later test reads back
    unconditionally. Order rows hid it by construction: every test mints a
    fresh idempotency key, so leftovers are unreachable. Halts did not — they
    are *global* by design, since the question a halt answers is "is trading
    stopped", not "is this order stopped". Eleven integration tests failed the
    first time the halt log ran against a real database, each seeing every halt
    its predecessors had engaged: ``assert 2 == 1``, then ``3 == 1``, then
    ``4 == 1``, climbing to ``22 == 5``.

    ``Base.metadata`` cannot go stale the way a literal can: a table that is not
    in it is not mapped, and a mapped table is truncated here the moment it is
    declared. ``alembic_version`` is safe by the same token — it belongs to
    Alembic, not to the metadata, so it is never named and the schema survives
    the reset.
    """
    names = ", ".join(sorted(Base.metadata.tables))
    return f"TRUNCATE TABLE {names} RESTART IDENTITY CASCADE"


@pytest.fixture(autouse=True)
async def _clean_database(
    migrated_database_url: str,  # noqa: ARG001 — fixture dependency: container + schema first
) -> AsyncIterator[None]:
    """Give every integration test a migrated, empty database and a fresh engine.

    Setup disposes the process-wide engine *before* handing over, and teardown
    truncates all fact tables (restarting identities) before disposing it again.

    **The setup-side disposal is not symmetry, it is the fix for a real bug.**
    ``_get_engine`` caches one engine per process, built from
    ``get_settings().database_url`` the first time anything asks for it.
    Disposing only on teardown leaves the *first* integration test of a session
    inheriting whatever engine an earlier, non-integration test happened to
    build — and that one was built before ``migrated_database_url`` pointed
    ``DATABASE_URL`` at the container, so it addresses the default
    ``localhost:5432`` where nothing is listening.

    This is not hypothetical. It surfaced the moment P5.1 added
    ``backend/tests/features/test_compute_lag.py``, which calls ``as_of()`` and
    sorts before ``backend/tests/integration/``: CI began erroring in exactly
    one place, ``test_asof_bind_integrity.py`` — the alphabetically first
    integration module, and so the only one not already handed a fresh engine
    by a previous test's teardown. Every later test passed, which is precisely
    what made it read like flaky infrastructure rather than an ordering
    dependency.

    The reset runs on the module-private *migration* engine, not the admin
    engine: the Core guard blocks textual SQL naming a fact table on every
    guarded engine, TRUNCATE included (fail-closed by design, D-011). Test
    reset is one of the two sanctioned unguarded uses named in
    ``backend.db.engine._create_migration_engine``, reached here by the same
    dynamic import the migration environment uses — importing it statically
    would (correctly) trip the TID251 import contract.
    """
    await dispose_database()
    yield
    from backend.db.engine import _create_migration_engine  # noqa: TID251 — see docstring

    reset_engine = _create_migration_engine()
    try:
        async with reset_engine.begin() as connection:
            await connection.execute(text(_truncate_every_table()))
    finally:
        await reset_engine.dispose()
        await dispose_database()
