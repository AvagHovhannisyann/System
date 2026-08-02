"""Database role separation: the app role cannot disable what enforces append-only (CC.9).

Revision ID: 0015
Revises: 0014
Create Date: 2026-08-02

The weakness this closes, stated exactly
----------------------------------------

D-012 records it plainly: append-only is enforced by ``BEFORE UPDATE OR DELETE``
triggers owned by the *same* role the application connects as, and that role can
``ALTER TABLE ... DISABLE TRIGGER`` or ``DROP TRIGGER`` on itself. So
"immutable audit log" was a **false claim** while the dashboard stated it as
fact (§6.11), and ``TESTING_LEDGER.md`` integrity — which is what makes the
Deflated Sharpe honest, because DSR is only as good as its trial count — rested
on the same unenforced assumption. D-017 set the deadline: before Phase 11.

Three routes were open to the single owning role, and a trigger stops none of
them, because none of them is a row mutation:

1. ``ALTER TABLE price_bar DISABLE TRIGGER trg_price_bar_append_only`` — needs
   table **ownership**, not a privilege.
2. ``DROP TRIGGER`` / ``DROP FUNCTION bitemporal_append_only_guard()`` — same.
3. ``SET session_replication_role = 'replica'`` — silently stops every ``ORIGIN``
   trigger firing for the session, needs superuser (or ``GRANT SET`` on the
   parameter), and leaves ``DELETE`` working normally. This is the one that does
   not look like an attack in a diff.

And a fourth the triggers never covered at all: ``TRUNCATE`` fires no row
trigger, and revisions 0003/0004 deliberately left it unblocked as the admin and
test reset path. Under one role, a single ``TRUNCATE config_change_event``
erased the immutable audit log with no error and no trace.

The two roles
-------------

**Schema owner** — whichever role runs alembic (``MIGRATION_DATABASE_URL``;
``POSTGRES_USER`` under compose). Owns every table, function and trigger. This
revision does not create it: it is the role executing this statement. It is the
only role that can change the schema, and the container entrypoint drops its
credential from the environment after ``alembic upgrade head`` and before
``exec uvicorn``, so the serving process does not hold it.

**Application role** — created here, named by ``APP_DB_ROLE`` (default
``quant_app``), authenticating with ``APP_DB_PASSWORD``. Attributes:
``LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS``. It is
granted no role membership, so it cannot ``SET ROLE`` to the owner, and it
**owns nothing**, which is what makes routes 1 and 2 above impossible for it:
those checks are ownership checks, and no grant substitutes for ownership.
``NOSUPERUSER`` closes route 3. Absence of the ``TRUNCATE`` privilege closes the
fourth.

The grant matrix
----------------

Default-deny, then exactly what the application genuinely needs::

    REVOKE ALL           ON SCHEMA public                  FROM app
    GRANT  USAGE         ON SCHEMA public                  TO   app   -- not CREATE
    GRANT  CONNECT       ON DATABASE <current>             TO   app
    REVOKE ALL           ON ALL TABLES IN SCHEMA public    FROM app
    GRANT  SELECT,INSERT ON ALL TABLES IN SCHEMA public    TO   app
    GRANT  USAGE         ON ALL SEQUENCES IN SCHEMA public TO   app
    GRANT  UPDATE        ON ingestion_run                  TO   app
    GRANT  UPDATE,DELETE ON llm_provider_credential        TO   app
    REVOKE ALL           ON alembic_version                FROM app
    GRANT  SELECT        ON alembic_version                TO   app
    ALTER DEFAULT PRIVILEGES FOR ROLE <owner> IN SCHEMA public
        GRANT SELECT, INSERT ON TABLES    TO app
        GRANT USAGE           ON SEQUENCES TO app

``UPDATE``/``DELETE``/``TRUNCATE``/``TRIGGER``/``REFERENCES`` are granted
**nowhere else**. The two exceptions are the two tables that revisions
0003-0014 deliberately left *without* an append-only trigger, and they are
exceptions for the reasons those revisions already record:

- ``ingestion_run`` (0005) — a run legitimately transitions ``running`` ->
  ``succeeded``/``failed``. It is operational metadata about our own pipeline,
  not a historical belief, and append-only would make a run impossible to close.
- ``llm_provider_credential`` (0009) — a rotation must *overwrite* the
  ciphertext and a deletion must remove it, so that a retired key stops being
  decryptable; keeping superseded ciphertext would widen a KEK compromise from
  "every key in use" to "every key ever used" (§7, I5). The history §6.5 asks
  for lives in the append-only ``config_change_event`` table as masked values.

So the grant matrix mirrors the trigger matrix exactly, and that correspondence
is asserted against the live catalog in
``backend/tests/integration/test_role_separation_db.py``: no table carrying an
append-only trigger may hold an ``UPDATE``, ``DELETE`` or ``TRUNCATE`` grant for
the app role.

``alembic_version`` is read-only for the app: a role that can stamp a revision
can convince the next deployment that a migration it never ran is already
applied.

``ALTER DEFAULT PRIVILEGES`` is what keeps this from rotting. Every table a
future migration creates is automatically ``SELECT``/``INSERT`` for the app
role and nothing more, so the *safe* case needs no maintenance and a future
table that genuinely needs ``UPDATE`` must say so in its own migration. The
failure direction is loud (``permission denied``), never silent.

TimescaleDB
-----------

``price_bar`` is a hypertable, so reads and writes land on chunks in
``_timescaledb_internal``. Schema ``USAGE`` is granted there, and a targeted
``GRANT`` is re-issued per hypertable: TimescaleDB propagates a grant named
directly on a hypertable to its existing chunks, and new chunks inherit the
hypertable's ACL. The blanket ``ALL TABLES IN SCHEMA public`` form is not relied
on for that propagation.

The password never appears in a statement
-----------------------------------------

``APP_DB_PASSWORD`` reaches PostgreSQL as a **bind parameter** to
``set_config(..., is_local => true)``, and the ``CREATE ROLE`` text is assembled
server-side by ``format(%L)`` from that transaction-local GUC. It is therefore
absent from every statement string this process holds, so it cannot surface in
an echoed statement, a ``DBAPIError`` message, or an alembic log line (I5). The
``EXCEPTION`` handler around role creation re-raises with the SQLSTATE only and
withholds the driver message, because PostgreSQL puts the failing ``EXECUTE``
text — which does carry the password — into the error context. For the same
reason this revision **refuses to run in offline (``--sql``) mode**: its whole
job cannot be expressed as a script without writing the credential into it.

Downgrade revokes everything, ``DROP OWNED BY`` the role (it owns nothing, so
this only clears privileges), and drops the role. It is a no-op when the role
does not exist.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa
from alembic import context, op

from backend.core.config import get_settings
from backend.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.core.config import Settings

# revision identifiers, used by Alembic.
revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_logger = get_logger(__name__)

ROLE_GUC: Final = "quant.app_db_role"
"""Transaction-local GUC carrying the application role name into the DO blocks."""

PASSWORD_GUC: Final = "quant.app_db_password"  # noqa: S105 — a GUC name, not a password
"""Transaction-local GUC carrying the application role password (never in SQL text)."""

DEV_DEFAULT_APP_PASSWORD: Final = "quant_app"  # noqa: S105 — see module docstring
"""Development-only password, mirroring ``${POSTGRES_APP_PASSWORD:-quant_app}`` in compose.

The same shape the stack already uses for ``POSTGRES_PASSWORD:-quant``: a known
throwaway credential for a local database that holds nothing. Refused at
``ENVIRONMENT=prod`` by :func:`_resolve_app_password`, so it cannot be shipped.
A test asserts this constant equals the compose default, because two defaults
that drift apart produce a role the application cannot authenticate as.
"""

MUTABLE_TABLE_GRANTS: Final[dict[str, str]] = {
    "ingestion_run": "UPDATE",
    "llm_provider_credential": "UPDATE, DELETE",
}
"""The complete set of tables the app role may mutate, and how (module docstring).

Declared here as data *and* written out as literal statements in
:data:`_GRANT_SQL` below. The duplication is deliberate and matches how
revisions 0003-0014 spell out their triggers: "what may this credential change"
is a question a reviewer answers by grepping for ``GRANT UPDATE``, and a loop
that assembles the statement makes that grep return nothing. A structural test
asserts the two agree, and that no table named here carries an append-only
trigger.
"""

_ROLE_NAME: Final = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
"""Bare lower-case SQL identifiers only — no quoting, no injection surface."""

_SET_ROLE_GUC: Final = sa.text("SELECT set_config('quant.app_db_role', :role, true)")
_READ_ROLE_GUC: Final = sa.text("SELECT current_setting('quant.app_db_role')")
_SET_PASSWORD_GUC: Final = sa.text("SELECT set_config('quant.app_db_password', :password, true)")

_CREATE_ROLE_SQL: Final = """
DO $$
DECLARE
    app_role text := current_setting('quant.app_db_role');
    app_password text := current_setting('quant.app_db_password');
    -- No CREATEROLE (it could grant itself the owner), no SUPERUSER (it could
    -- SET session_replication_role and silence every trigger), no BYPASSRLS.
    attributes CONSTANT text :=
        'LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS INHERIT';
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = app_role) THEN
        EXECUTE format('ALTER ROLE %I WITH %s PASSWORD %L',
                       app_role, attributes, app_password);
    ELSE
        EXECUTE format('CREATE ROLE %I WITH %s PASSWORD %L',
                       app_role, attributes, app_password);
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE EXCEPTION
        'could not create or re-point the application role (SQLSTATE %). The driver '
        'message is withheld deliberately: the statement that failed carries the role '
        'password, and PostgreSQL puts the failing statement into the error context '
        '(DIRECTIVE I5)', SQLSTATE;
END
$$
"""

_GRANT_SQL: Final = """
DO $$
DECLARE
    app_role text := current_setting('quant.app_db_role');
    hypertable text;
BEGIN
    -- Default-deny first, so a re-run cannot leave a stale grant standing.
    EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', app_role);
    EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', app_role);
    EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', app_role);

    EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), app_role);
    -- USAGE, never CREATE: the app role may not create objects it would own,
    -- and ownership is the privilege that drops triggers.
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', app_role);
    EXECUTE format('GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO %I', app_role);
    EXECUTE format('GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO %I', app_role);

    -- The only two mutable tables in the schema, and the only two without an
    -- append-only trigger. See MUTABLE_TABLE_GRANTS and the module docstring.
    EXECUTE format('GRANT UPDATE ON ingestion_run TO %I', app_role);
    EXECUTE format('GRANT UPDATE, DELETE ON llm_provider_credential TO %I', app_role);

    -- A role that can stamp a revision can tell the next deployment that a
    -- migration it never ran is already applied.
    IF to_regclass('public.alembic_version') IS NOT NULL THEN
        EXECUTE format('REVOKE ALL ON alembic_version FROM %I', app_role);
        EXECUTE format('GRANT SELECT ON alembic_version TO %I', app_role);
    END IF;

    -- Every table a later migration creates is SELECT/INSERT for the app role
    -- and nothing more, with no action required of that migration.
    EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
                   'GRANT SELECT, INSERT ON TABLES TO %I', current_user, app_role);
    EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
                   'GRANT USAGE ON SEQUENCES TO %I', current_user, app_role);

    -- Hypertable rows live in chunks under _timescaledb_internal.
    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = '_timescaledb_internal') THEN
        EXECUTE format('GRANT USAGE ON SCHEMA _timescaledb_internal TO %I', app_role);
    END IF;
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        -- Re-issued per hypertable by name: TimescaleDB propagates a grant
        -- named directly on a hypertable to its chunks, and new chunks inherit
        -- the hypertable's ACL. Not relying on the ALL TABLES form for that.
        FOR hypertable IN
            SELECT format('%I.%I', hypertable_schema, hypertable_name)
              FROM timescaledb_information.hypertables
        LOOP
            EXECUTE format('GRANT SELECT, INSERT ON %s TO %I', hypertable, app_role);
        END LOOP;
    END IF;
END
$$
"""

_REVOKE_SQL: Final = """
DO $$
DECLARE
    app_role text := current_setting('quant.app_db_role');
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = app_role) THEN
        RETURN;
    END IF;
    EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
                   'REVOKE ALL ON TABLES FROM %I', current_user, app_role);
    EXECUTE format('ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
                   'REVOKE ALL ON SEQUENCES FROM %I', current_user, app_role);
    EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', app_role);
    EXECUTE format('REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', app_role);
    EXECUTE format('REVOKE ALL ON SCHEMA public FROM %I', app_role);
    IF EXISTS (SELECT 1 FROM pg_namespace WHERE nspname = '_timescaledb_internal') THEN
        EXECUTE format('REVOKE ALL ON SCHEMA _timescaledb_internal FROM %I', app_role);
    END IF;
    EXECUTE format('REVOKE ALL ON DATABASE %I FROM %I', current_database(), app_role);
    -- The role owns nothing by construction, so this clears privileges rather
    -- than dropping objects; it is what makes DROP ROLE succeed.
    EXECUTE format('DROP OWNED BY %I', app_role);
    EXECUTE format('DROP ROLE %I', app_role);
END
$$
"""


def _validated_role(name: str) -> str:
    """Return ``name`` if it is a bare lower-case SQL identifier, else raise.

    The role name reaches PostgreSQL through a bind parameter and is quoted
    server-side by ``format(%I)``, so this is a second line rather than the
    only one. It exists because a role name that needs quoting is far more
    likely to be a misconfiguration than an intention, and a misconfigured
    role name produces a database the application silently cannot log into.

    Args:
        name: candidate role name from ``APP_DB_ROLE``.

    Returns:
        The name unchanged.

    Raises:
        ValueError: the name is not ``[a-z_][a-z0-9_]{0,62}``.
    """
    if _ROLE_NAME.fullmatch(name) is None:
        message = (
            f"APP_DB_ROLE {name!r} is not a bare lower-case SQL identifier "
            f"([a-z_][a-z0-9_]{{0,62}}). Role separation refuses to interpolate a "
            f"name it cannot vouch for into DDL."
        )
        raise ValueError(message)
    return name


def _resolve_app_password(settings: Settings) -> str:
    """Return the application role's password, refusing to invent one in production.

    Mirrors :func:`backend.api.security.csrf.resolve_csrf_secret`: configured
    wins; outside production an absent value falls back to the committed
    development default with a warning; in production an absent value is a hard
    failure rather than a known password shipped quietly.

    The returned value is passed to PostgreSQL as a bind parameter and never
    interpolated into a statement, logged, or echoed (I5).

    Args:
        settings: resolved application settings.

    Returns:
        The password to give the application role.

    Raises:
        RuntimeError: ``APP_DB_PASSWORD`` is unset or empty at
            ``ENVIRONMENT=prod``.
    """
    configured = settings.app_db_password
    if configured is not None and configured.get_secret_value():
        return configured.get_secret_value()
    if settings.environment == "prod":
        message = (
            "APP_DB_PASSWORD is unset. It is required in production: the application "
            "role's credential must come from the environment, and falling back to the "
            "committed development default would ship a password that is public in this "
            "repository (DIRECTIVE I5, §7)."
        )
        raise RuntimeError(message)
    _logger.warning(
        "app_db_password_default_used",
        role=settings.app_db_role,
        environment=settings.environment,
        detail=(
            "APP_DB_PASSWORD is unset; the application role was given the development "
            "default. Set APP_DB_PASSWORD outside local development."
        ),
    )
    return DEV_DEFAULT_APP_PASSWORD


def _bind_role_name() -> str:
    """Push the validated role name into a transaction-local GUC; return it.

    The name travels as a bind parameter and is quoted server-side by
    ``format(%I)``. ``is_local => true`` scopes the setting to the migration's
    transaction, so it is discarded at commit rather than lingering on a pooled
    connection.

    Returns:
        The validated application role name.

    Raises:
        RuntimeError: the revision is running in offline (``--sql``) mode, where
            the role parameters could only be delivered by writing them into the
            emitted script.
        ValueError: ``APP_DB_ROLE`` is not a bare SQL identifier.
    """
    if context.is_offline_mode():
        message = (
            "revision 0015 cannot run in offline (--sql) mode: delivering the application "
            "role's password would mean writing it into the generated script, and that "
            "credential must never be persisted or committed (DIRECTIVE I5). Run the "
            "migration online against the target database."
        )
        raise RuntimeError(message)
    role = _validated_role(get_settings().app_db_role)
    bind = op.get_bind()
    bind.execute(_SET_ROLE_GUC, {"role": role})
    # Read it back on a *separate* statement. A transaction-local setting that
    # does not survive to the next statement means this revision is not running
    # inside a transaction, and the DO blocks below would fail on a missing
    # parameter with no hint as to why. Better to say so here.
    if bind.execute(_READ_ROLE_GUC).scalar_one() != role:
        message = (
            "the transaction-local role parameter did not survive to the next statement, so "
            "this revision is not running inside a transaction. It delivers the role name and "
            "password through transaction-scoped settings precisely so the password never "
            "enters a statement string (DIRECTIVE I5); without a transaction that is not "
            "possible. Run alembic online against a database with transactional DDL."
        )
        raise RuntimeError(message)
    return role


def upgrade() -> None:
    """Create the least-privileged application role and grant it exactly its job."""
    role = _bind_role_name()
    # The password reaches the server only here, and only as a bind parameter.
    op.get_bind().execute(_SET_PASSWORD_GUC, {"password": _resolve_app_password(get_settings())})
    op.execute(_CREATE_ROLE_SQL)
    op.execute(_GRANT_SQL)
    _logger.info(
        "app_db_role_configured",
        role=role,
        mutable_tables=sorted(MUTABLE_TABLE_GRANTS),
    )


def downgrade() -> None:
    """Revoke every privilege from the application role and drop it.

    Deliberately does not resolve ``APP_DB_PASSWORD``: dropping a role needs no
    credential, and requiring one would make a rollback impossible in exactly
    the environment where the password was the thing misconfigured.
    """
    role = _bind_role_name()
    op.execute(_REVOKE_SQL)
    _logger.info("app_db_role_dropped", role=role)
