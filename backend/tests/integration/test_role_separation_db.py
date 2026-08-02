"""CC.9 integration: the application role cannot revoke what enforces append-only.

The claim being proved is a **negative**, so every test here attempts the thing
and asserts the refusal. Attempts are issued on a plain SQLAlchemy engine built
directly on the application role's credentials — deliberately *beneath* the
application's own Core guard and ORM hook, because the question is not whether
``backend.db`` declines to emit the statement but whether PostgreSQL refuses it
when it arrives. A psql session, a future service, or a mistaken admin script
holding the same credential gets exactly these answers.

Four routes existed under a single role (D-012's residual weakness), and a
trigger stops none of them, because none is a row mutation:

1. ``ALTER TABLE ... DISABLE TRIGGER`` — an ownership check.
2. ``DROP TRIGGER`` / ``DROP FUNCTION`` — an ownership check.
3. ``SET session_replication_role = 'replica'`` — silences every ``ORIGIN``
   trigger for the session and needs superuser. This is the one that does not
   look like an attack.
4. ``TRUNCATE`` — fires no row trigger at all, and revisions 0003/0004 left it
   unblocked on purpose as the admin and test reset path.

Each has a test below. So does the other half, which matters just as much: the
app role must still be able to insert facts and read them through ``as_of()``,
or role separation has shipped a database that is secure and useless.

The negative tests reach Postgres directly; the positive ones go through the
real application paths (``ingest_writer_session``, ``as_of``) with
``DATABASE_URL`` pointed at the app role, so what they exercise is the code the
server actually runs.

This module needs a Docker daemon, so CI is where it first runs. Every assertion
in it was executed once against PostgreSQL 16.13 outside CI, with the whole
``0001..0015`` chain applied through this repository's alembic environment; each
refusal came back as SQLSTATE ``42501``. TimescaleDB was **not** present in that
run, so one thing here remains unexercised anywhere: that grants on ``price_bar``
and ``edgar_filing`` propagate to their chunks. Revision 0015 re-issues a grant
per hypertable by name for exactly that reason, and this suite's insert-and-read
test is what will show whether it worked.
"""

from __future__ import annotations

import datetime as dt
import importlib
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Final

import pytest
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from backend.core.config import Settings, get_settings
from backend.db import as_of, dispose_database, ingest_writer_session
from backend.db.models import PriceBar
from backend.tests.integration.factories import bar_version, create_security, insert_rows

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_MIGRATION = importlib.import_module("backend.db.migrations.versions.0015_role_separation")
"""Revision 0015 itself, so the credentials tested are the ones it installs.

Imported rather than re-stated: a copy of the role name and the development
default here would let the migration and the test drift apart, and a test that
authenticates as a role the migration never created fails for the wrong reason.
"""

_INSUFFICIENT_PRIVILEGE: Final = "42501"
"""SQLSTATE for both ``permission denied for ...`` and ``must be owner of ...``."""

_DAY: Final = dt.date(2024, 3, 7)
_KNOWN_AT: Final = dt.datetime(2024, 3, 7, 21, 0, tzinfo=dt.UTC)

_APPEND_ONLY_TRIGGER_SUFFIX: Final = "append_only"
_VERSION_TABLE: Final = "alembic_version"


def _sqlstate(error: DBAPIError) -> str | None:
    """Return the five-character SQLSTATE a driver exception carries, if any.

    SQLAlchemy exposes no portable accessor; asyncpg names the attribute
    ``sqlstate``. Deciding on the code rather than the exception class matters
    here because ``permission denied for table`` and ``must be owner of table``
    are different messages, different phrasing across major versions, and the
    same SQLSTATE.

    Args:
        error: the wrapped database error.

    Returns:
        The SQLSTATE, or ``None`` when the driver exposes none.
    """
    code: object = getattr(error.orig, "sqlstate", None)
    return code if isinstance(code, str) and len(code) == 5 else None


def _app_credentials() -> tuple[str, str]:
    """Return the application role name and password the migration installed.

    Resolved from settings exactly the way revision 0015 resolves them, so the
    tests authenticate as the role the migration actually created rather than as
    one spelled the same way by coincidence.

    Returns:
        ``(role, password)``.
    """
    settings = Settings()
    password: str = _MIGRATION._resolve_app_password(settings)
    return settings.app_db_role, password


def _app_url(owner_url: str) -> str:
    """Return ``owner_url`` re-pointed at the application role.

    Args:
        owner_url: the container's connection URL, owned-role credentials.

    Returns:
        The same host, port and database with the app role's credentials.
    """
    role, password = _app_credentials()
    url = make_url(owner_url).set(username=role, password=password)
    return url.render_as_string(hide_password=False)


@pytest.fixture
def app_url(migrated_database_url: str) -> str:
    """The container URL, authenticating as the least-privileged application role."""
    return _app_url(migrated_database_url)


async def _as_app(app_url: str, statement: str) -> None:
    """Execute one statement as the application role, straight at PostgreSQL.

    Args:
        app_url: connection URL carrying the app role's credentials.
        statement: raw SQL to attempt.
    """
    engine = create_async_engine(app_url, poolclass=NullPool)
    try:
        async with engine.begin() as connection:
            await connection.execute(sa.text(statement))
    finally:
        await engine.dispose()


async def _refused(app_url: str, statement: str) -> DBAPIError:
    """Assert PostgreSQL refuses ``statement`` for the app role; return the error.

    Args:
        app_url: connection URL carrying the app role's credentials.
        statement: raw SQL that must not be permitted.

    Returns:
        The raised error, so callers can assert on its SQLSTATE and text.
    """
    with pytest.raises(DBAPIError) as raised:
        await _as_app(app_url, statement)
    assert _sqlstate(raised.value) == _INSUFFICIENT_PRIVILEGE, (
        f"{statement!r} failed for the wrong reason: {raised.value}"
    )
    return raised.value


async def _owner_rows(
    owner_url: str, query: str, parameters: dict[str, object] | None = None
) -> list[sa.Row[Any]]:
    """Run a catalog query as the schema owner and return every row.

    Built from the container URL directly rather than through
    ``backend.db.engine``'s private migration factory. Nothing here needs that
    abstraction — these are catalog reads on a throwaway connection — and going
    around it keeps this module out of the private-engine allowlist that
    ``test_bypass_prevention.py`` maintains, which exists so that reaching for
    the private engine is always a deliberate, argued act.

    Args:
        owner_url: the container's connection URL, schema-owner credentials.
        query: SQL to execute.
        parameters: bind parameters, or ``None``.

    Returns:
        The result rows.
    """
    engine = create_async_engine(owner_url, poolclass=NullPool)
    try:
        async with engine.connect() as connection:
            result = await connection.execute(sa.text(query), parameters or {})
            return list(result.all())
    finally:
        await engine.dispose()


@pytest.fixture
async def connected_as_app_role(
    migrated_database_url: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Point the application's own engine at the app role for the duration of a test.

    ``MIGRATION_DATABASE_URL`` is pinned to the owner at the same time. The
    integration conftest truncates fact tables on the module-private *migration*
    engine after every test, and that engine resolves ``migration_url``; without
    the pin it would inherit the app credential and the reset would be refused —
    correctly, since the app role has no ``TRUNCATE``. Restored before the
    conftest's teardown runs, because this fixture is requested explicitly and
    the conftest's is autouse, so this one is finalized first.
    """
    monkeypatch.setenv("DATABASE_URL", _app_url(migrated_database_url))
    monkeypatch.setenv("MIGRATION_DATABASE_URL", migrated_database_url)
    get_settings.cache_clear()
    await dispose_database()
    try:
        yield
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
        await dispose_database()


# --------------------------------------------------------------------------
# The four routes that a trigger does not close
# --------------------------------------------------------------------------


async def test_app_role_cannot_disable_an_append_only_trigger(app_url: str) -> None:
    """Route 1. ``DISABLE TRIGGER`` is an ownership check, and the app owns nothing."""
    await _refused(
        app_url,
        "ALTER TABLE price_bar DISABLE TRIGGER trg_price_bar_append_only",
    )


async def test_app_role_cannot_drop_an_append_only_trigger(app_url: str) -> None:
    """Route 2, on the trigger."""
    await _refused(app_url, "DROP TRIGGER trg_price_bar_append_only ON price_bar")


async def test_app_role_cannot_drop_the_trigger_function(app_url: str) -> None:
    """Route 2, on the function every fact-table trigger calls.

    Dropping the function would disarm ``security_master`` and ``price_bar`` in
    one statement, so it is worth its own test rather than being folded into the
    trigger case.
    """
    await _refused(app_url, "DROP FUNCTION bitemporal_append_only_guard()")


async def test_app_role_cannot_silence_triggers_with_session_replication_role(
    app_url: str,
) -> None:
    """Route 3 — the quiet one.

    ``SET session_replication_role = 'replica'`` stops every ``ORIGIN`` trigger
    firing for the session and leaves ``DELETE`` working normally. It needs
    superuser, which is precisely why the app role is created ``NOSUPERUSER``. A
    role that could do this would defeat all four append-only revisions at once
    without touching a single object definition.
    """
    await _refused(app_url, "SET session_replication_role = 'replica'")


async def test_app_role_cannot_truncate_an_append_only_table(app_url: str) -> None:
    """Route 4 — the gap the triggers never covered.

    Revisions 0003/0004 left ``TRUNCATE`` unblocked deliberately: it is the
    sanctioned admin and test reset and never masquerades as a correction. Under
    one role that meant a single ``TRUNCATE config_change_event`` erased the
    immutable audit log with no error and no trace. Role separation is what
    actually closes it, because ``TRUNCATE`` is a privilege the app role is not
    granted anywhere.
    """
    await _refused(app_url, "TRUNCATE TABLE config_change_event")
    await _refused(app_url, "TRUNCATE TABLE price_bar")


# --------------------------------------------------------------------------
# The rest of the perimeter
# --------------------------------------------------------------------------


async def test_app_role_cannot_alter_a_table(app_url: str) -> None:
    """A fact table and the audit log, because the audit log is the claim at stake."""
    await _refused(app_url, "ALTER TABLE price_bar ADD COLUMN smuggled text")
    await _refused(app_url, "ALTER TABLE config_change_event ADD COLUMN smuggled text")


async def test_app_role_cannot_delete_a_fact_row(app_url: str) -> None:
    """No ``DELETE`` privilege, so the statement never reaches the trigger.

    Two layers now stand where one did: the privilege check refuses first, and
    the append-only trigger still refuses anything that somehow gets past it.
    """
    error = await _refused(app_url, "DELETE FROM price_bar")
    assert "append-only" not in str(error), (
        "the trigger message means the statement was permitted and then refused; the "
        "privilege check should have stopped it before it ran"
    )
    await _refused(app_url, "DELETE FROM config_change_event")
    await _refused(app_url, "DELETE FROM execution_order_transition")


async def test_app_role_cannot_update_a_fact_row(app_url: str) -> None:
    await _refused(app_url, "UPDATE price_bar SET close_usd = 0")
    await _refused(app_url, "UPDATE security_master SET ticker = 'EVIL'")


async def test_app_role_cannot_upsert_its_way_into_an_update(app_url: str) -> None:
    """``ON CONFLICT DO UPDATE`` is an UPDATE, and PostgreSQL checks it as one.

    Worth asserting because it is the one write shape that looks like an insert
    in application code and would otherwise be an unremarked hole in an
    insert-only grant.
    """
    await _refused(
        app_url,
        "INSERT INTO security (security_id) VALUES (1) "
        "ON CONFLICT (security_id) DO UPDATE SET security_id = 1",
    )


async def test_app_role_cannot_create_objects_it_would_own(app_url: str) -> None:
    """Ownership is the privilege that drops triggers, so the app may create nothing."""
    await _refused(app_url, "CREATE TABLE smuggled (id bigint)")
    await _refused(
        app_url,
        "CREATE FUNCTION smuggled_guard() RETURNS trigger LANGUAGE plpgsql AS $$ "
        "BEGIN RETURN NEW; END $$",
    )


async def test_app_role_cannot_stamp_a_migration(app_url: str) -> None:
    """A role that can write ``alembic_version`` can hide a migration that never ran."""
    await _refused(app_url, "INSERT INTO alembic_version (version_num) VALUES ('9999')")
    await _refused(app_url, "DELETE FROM alembic_version")


async def test_app_role_cannot_become_the_owner(app_url: str, migrated_database_url: str) -> None:
    """No role membership, so ``SET ROLE`` offers no way back to ownership."""
    owner = make_url(migrated_database_url).username
    assert owner is not None
    await _refused(app_url, f"SET ROLE {owner}")


async def test_app_role_cannot_escalate_its_own_attributes(app_url: str) -> None:
    role, _ = _app_credentials()
    await _refused(app_url, f"ALTER ROLE {role} SUPERUSER")
    await _refused(app_url, f"ALTER ROLE {role} CREATEROLE")
    await _refused(app_url, "CREATE ROLE smuggled LOGIN")


async def test_refusals_never_echo_the_role_password(app_url: str) -> None:
    """I5: a refusal is logged, so it must not carry the credential that produced it.

    The statement is chosen so the assertion means something. Under the
    development default the password happens to equal the role name, so a
    refusal of ``ALTER ROLE quant_app ...`` would echo that string for a reason
    that has nothing to do with credentials and the check would report a leak
    that is not one. ``DELETE FROM price_bar`` names neither the role nor the
    connection.
    """
    _, password = _app_credentials()
    error = await _refused(app_url, "DELETE FROM price_bar")
    assert password not in str(error)
    assert password not in repr(error)
    assert password not in str(error.statement)


# --------------------------------------------------------------------------
# Catalog invariants — these cover tables no test above names
# --------------------------------------------------------------------------


async def test_no_append_only_table_grants_the_app_role_a_way_to_mutate_it(
    migrated_database_url: str,
) -> None:
    """The grant matrix must mirror the trigger matrix, checked against the live schema.

    The statement tests above name a handful of tables. This one covers every
    table in the schema, including any a later migration adds: if it carries an
    append-only trigger, the app role must hold no ``UPDATE``, ``DELETE`` or
    ``TRUNCATE`` on it. Two enforcement layers that disagree are how one of them
    quietly stops being maintained.
    """
    role, _ = _app_credentials()
    rows = await _owner_rows(
        migrated_database_url,
        """
        SELECT c.relname AS table_name,
               has_table_privilege(:role, c.oid, 'UPDATE')   AS can_update,
               has_table_privilege(:role, c.oid, 'DELETE')   AS can_delete,
               has_table_privilege(:role, c.oid, 'TRUNCATE') AS can_truncate,
               EXISTS (
                   SELECT 1 FROM pg_trigger t
                    WHERE t.tgrelid = c.oid
                      AND NOT t.tgisinternal
                      AND t.tgname LIKE '%' || :suffix
               ) AS append_only
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relkind = 'r'
         ORDER BY c.relname
        """,
        {"role": role, "suffix": _APPEND_ONLY_TRIGGER_SUFFIX},
    )
    assert rows, "no tables found; the schema query is wrong and this test proves nothing"
    append_only = [row for row in rows if row.append_only]
    assert len(append_only) >= 10, (
        f"only {len(append_only)} append-only tables found; revisions 0003-0014 install "
        "far more, so the trigger detection is broken and this test is vacuous"
    )
    for row in append_only:
        assert not row.can_update, f"{row.table_name} is append-only but grants UPDATE"
        assert not row.can_delete, f"{row.table_name} is append-only but grants DELETE"
        assert not row.can_truncate, f"{row.table_name} is append-only but grants TRUNCATE"
    for row in rows:
        assert not row.can_truncate, (
            f"{row.table_name} grants TRUNCATE to the application role; TRUNCATE fires no "
            "row trigger, so a single statement would erase the table without a trace"
        )


async def test_every_table_the_app_reads_is_readable_and_insertable(
    migrated_database_url: str,
) -> None:
    """The other half of the claim: a locked-down database that cannot be written is useless.

    ``alembic_version`` is the deliberate exception — readable, never writable.
    """
    role, _ = _app_credentials()
    rows = await _owner_rows(
        migrated_database_url,
        """
        SELECT c.relname AS table_name,
               has_table_privilege(:role, c.oid, 'SELECT') AS can_select,
               has_table_privilege(:role, c.oid, 'INSERT') AS can_insert
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relkind = 'r'
         ORDER BY c.relname
        """,
        {"role": role},
    )
    assert rows
    for row in rows:
        assert row.can_select, f"the application cannot read {row.table_name}"
        if row.table_name == _VERSION_TABLE:
            assert not row.can_insert, "the application must not be able to stamp a revision"
        else:
            assert row.can_insert, f"the application cannot write {row.table_name}"


async def test_the_documented_mutable_tables_really_did_receive_their_grants(
    migrated_database_url: str,
) -> None:
    """Non-vacuity for the matrix above: the two exceptions exist and are exactly two.

    Deliberately asserts containment rather than equality against a frozen pair.
    A later migration may legitimately add a mutable table, and it will declare
    its own grant; the invariant that must never bend is the one asserted above —
    an append-only table never holds a mutation grant — and it is re-derived here
    for whatever the current set happens to be, so a careless addition still
    fails rather than passing under a widened allowance.
    """
    role, _ = _app_credentials()
    rows = await _owner_rows(
        migrated_database_url,
        """
        SELECT c.relname AS table_name,
               EXISTS (
                   SELECT 1 FROM pg_trigger t
                    WHERE t.tgrelid = c.oid
                      AND NOT t.tgisinternal
                      AND t.tgname LIKE '%' || :suffix
               ) AS append_only
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relkind = 'r'
           AND (has_table_privilege(:role, c.oid, 'UPDATE')
                OR has_table_privilege(:role, c.oid, 'DELETE'))
         ORDER BY c.relname
        """,
        {"role": role, "suffix": _APPEND_ONLY_TRIGGER_SUFFIX},
    )
    mutable = {row.table_name for row in rows}
    assert {"ingestion_run", "llm_provider_credential"} <= mutable, (
        f"revision 0015's mutable-table grants did not land: found {sorted(mutable)}. "
        "The ingestion run lifecycle and provider-key rotation both need them"
    )
    assert set(_MIGRATION.MUTABLE_TABLE_GRANTS) <= mutable
    for row in rows:
        assert not row.append_only, (
            f"{row.table_name} is append-only and yet the application role may mutate it"
        )


async def test_the_app_role_owns_nothing_and_can_escalate_to_nothing(
    migrated_database_url: str,
) -> None:
    """Ownership, not privilege, is what ``DISABLE TRIGGER`` and ``DROP`` check."""
    role, _ = _app_credentials()
    owned = await _owner_rows(
        migrated_database_url,
        """
        SELECT c.relname AS name FROM pg_class c
          JOIN pg_roles r ON r.oid = c.relowner
         WHERE r.rolname = :role
        UNION ALL
        SELECT p.proname FROM pg_proc p
          JOIN pg_roles r ON r.oid = p.proowner
         WHERE r.rolname = :role
        UNION ALL
        SELECT n.nspname FROM pg_namespace n
          JOIN pg_roles r ON r.oid = n.nspowner
         WHERE r.rolname = :role
        """,
        {"role": role},
    )
    assert owned == [], f"the application role owns {[row.name for row in owned]}"

    attributes = await _owner_rows(
        migrated_database_url,
        """
        SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication,
               rolbypassrls, rolcanlogin,
               (SELECT count(*) FROM pg_auth_members m WHERE m.member = pg_roles.oid)
                   AS memberships
          FROM pg_roles WHERE rolname = :role
        """,
        {"role": role},
    )
    assert len(attributes) == 1, "the application role does not exist; migration 0015 did not run"
    row = attributes[0]
    assert row.rolcanlogin, "the application role cannot log in"
    assert not row.rolsuper, "a superuser can SET session_replication_role and silence triggers"
    assert not row.rolcreaterole, "a role that can create roles can grant itself the owner"
    assert not row.rolcreatedb
    assert not row.rolreplication
    assert not row.rolbypassrls
    assert row.memberships == 0, "a role membership restores ownership through SET ROLE"


# --------------------------------------------------------------------------
# The app role can still do its job
# --------------------------------------------------------------------------


@pytest.mark.usefixtures("connected_as_app_role")
async def test_app_role_inserts_facts_and_reads_them_through_as_of() -> None:
    """Through the real application paths, with ``DATABASE_URL`` on the app role.

    This is the test that makes the negative ones meaningful. Role separation
    that also broke ingestion or the as-of layer would be a regression dressed
    as a hardening, and D-011 forbids regressing either.
    """
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _KNOWN_AT, "123.45"))

    async with as_of(_KNOWN_AT) as session:
        bars = list(await session.scalars(select(PriceBar)))
    assert len(bars) == 1
    assert bars[0].close_usd == Decimal("123.45")
    assert bars[0].security_id == security_id

    async with as_of(_KNOWN_AT - dt.timedelta(microseconds=1)) as session:
        assert list(await session.scalars(select(PriceBar))) == []


@pytest.mark.usefixtures("connected_as_app_role")
async def test_the_orm_write_path_is_refused_by_the_privilege_before_the_trigger() -> None:
    """Which of the two layers answers, stated rather than assumed.

    Before CC.9 an ORM ``UPDATE`` on a fact table was refused by the append-only
    trigger — that is what ``test_hypertable_append_only.py`` asserts, and it
    still holds for any role that *has* the privilege, including the owner. Under
    the application role the refusal now arrives one layer earlier, from the
    privilege check (``42501``), and the statement never reaches the trigger at
    all.

    Worth pinning down, because the two layers are not interchangeable: the
    privilege stops writes this credential issues, and the trigger stops writes
    from anything holding a credential that could otherwise do it. A change that
    silently swapped which one fires would be the first sign that one of them had
    been removed.
    """
    security_id = await create_security()
    await insert_rows(bar_version(security_id, _DAY, _KNOWN_AT, "10"))
    async with ingest_writer_session() as session:
        with pytest.raises(DBAPIError) as raised:
            await session.execute(sa.update(PriceBar).values(close_usd=Decimal("0")))
    assert _sqlstate(raised.value) == _INSUFFICIENT_PRIVILEGE, (
        f"the ORM update was refused for a different reason: {raised.value}"
    )


async def test_the_append_only_triggers_are_all_still_installed(
    migrated_database_url: str,
) -> None:
    """Defence in depth, not defence instead — the layer beneath is still there.

    Role separation makes the triggers unreachable for the application role, and
    an unreachable check is one nobody notices the absence of. This asserts every
    one of them is still attached, so a future revision cannot quietly drop them
    on the grounds that grants now cover it. Grants bind roles; the triggers bind
    the owner too, and the owner is a superuser in this stack.
    """
    rows = await _owner_rows(
        migrated_database_url,
        """
        SELECT c.relname AS table_name, t.tgname AS trigger_name, t.tgenabled AS enabled
          FROM pg_trigger t
          JOIN pg_class c ON c.oid = t.tgrelid
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND NOT t.tgisinternal
           AND t.tgname LIKE '%' || :suffix
         ORDER BY c.relname
        """,
        {"suffix": _APPEND_ONLY_TRIGGER_SUFFIX},
    )
    assert len(rows) >= 10, f"only {len(rows)} append-only triggers are installed"
    for row in rows:
        assert row.enabled == "O", (
            f"{row.trigger_name} on {row.table_name} has tgenabled={row.enabled!r}; "
            "'O' is the only value that fires on ordinary (origin) sessions"
        )
