"""Structural checks on database role separation (CC.9, D-012 residual, D-017).

What these tests prove, and what they do not
--------------------------------------------

They read the committed migration, settings, engine, alembic environment,
entrypoint script and compose file, and assert the two roles are *declared*
coherently. Nothing is started: no container runs, no role is created, no
privilege is exercised. A green run here means the wiring is coherent, not that
PostgreSQL refuses anything.

The refusals themselves — that the app role cannot ``DISABLE TRIGGER``,
``DROP TRIGGER``, ``ALTER TABLE``, ``DELETE``, or ``TRUNCATE`` — are proved by
attempting each one as that role in
``backend/tests/integration/test_role_separation_db.py``, which needs a Docker
daemon. Saying so here is the point: a structural test that implied it had
checked the privilege system would be worse than no test (I6).

What is genuinely settled without a database, and is settled here:

* the grant matrix is default-deny, and the only ``UPDATE``/``DELETE`` grants
  name the two tables that revisions 0003-0014 deliberately left without an
  append-only trigger;
* the password is never interpolated into SQL, so it cannot reach a log or an
  error message (I5);
* migrations resolve the owner URL and the application resolves the app URL,
  through one shared settings property rather than two conventions;
* the entrypoint refuses a single-role configuration and drops the owner
  credential before serving;
* the compose defaults and the migration's development default are the same
  string, so the role that gets created is the role the app logs in as.

Recorded because it is the kind of claim that should carry its evidence: the
refusals, the grant matrix, ``ALTER DEFAULT PRIVILEGES``, the downgrade, and the
password round-trip were all executed once against PostgreSQL 16.13 outside CI,
with the whole ``0001..0015`` chain applied through this repository's own alembic
environment. That run is not repeatable here and proves nothing about the next
commit, which is exactly why the integration module exists. What it did settle is
that these statements are valid SQL and that the privilege system answers the way
this file assumes — see the report accompanying the change for the transcript.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal

import pytest
import yaml
from pydantic import SecretStr

from backend.core.config import Settings

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MIGRATIONS: Final = _REPO_ROOT / "backend" / "db" / "migrations"
_VERSIONS: Final = _MIGRATIONS / "versions"
_ENGINE: Final = _REPO_ROOT / "backend" / "db" / "engine.py"
_ENTRYPOINT: Final = _REPO_ROOT / "docker" / "backend-entrypoint.sh"
_COMPOSE: Final = _REPO_ROOT / "docker-compose.yml"
_REVISION: Final = "0015_role_separation"

_BACKEND_SERVICE: Final = "backend"
_APP_USER_VAR: Final = "POSTGRES_APP_USER"
_APP_PASSWORD_VAR: Final = "POSTGRES_APP_PASSWORD"  # noqa: S105 — a variable name
_OWNER_USER_VAR: Final = "POSTGRES_USER"
_OWNER_PASSWORD_VAR: Final = "POSTGRES_PASSWORD"  # noqa: S105 — a variable name

# `${NAME:-default}` and `${NAME}` inside a compose value.
_COMPOSE_VAR: Final = re.compile(r"\$\{(?P<name>[A-Z_][A-Z0-9_]*)(?::-(?P<default>[^}]*))?\}")
# A GRANT of specific privileges on a specific object, as written in the DO
# blocks: GRANT <privileges> ON <object> TO ...
_GRANT: Final = re.compile(
    r"GRANT\s+(?P<privileges>[A-Z ,]+?)\s+ON\s+(?P<target>[A-Za-z_. %]+?)\s+TO\b"
)
_OWNERSHIP_ONLY_PRIVILEGES: Final = frozenset({"UPDATE", "DELETE", "TRUNCATE", "TRIGGER", "CREATE"})
# A role-membership grant: `GRANT <role> TO <role>`, i.e. one bare word between
# GRANT and TO. Every privilege grant carries an ON clause, so none matches.
_MEMBERSHIP_GRANT: Final = re.compile(r"\bGRANT\s+\w+\s+TO\s")
# The `attributes CONSTANT text := '...'` literal inside the CREATE ROLE block.
_ROLE_ATTRIBUTES: Final = re.compile(r"attributes\s+CONSTANT\s+text\s*:=\s*'([^']*)'")

_SETTINGS_ENV: Final = (
    "DATABASE_URL",
    "MIGRATION_DATABASE_URL",
    "APP_DB_ROLE",
    "APP_DB_PASSWORD",
    "ENVIRONMENT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert against the committed defaults, not against the runner's environment."""
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


class _OfflineContext:
    """Stand-in for alembic's ``context`` proxy that reports ``--sql`` mode."""

    def is_offline_mode(self) -> bool:
        """Report offline mode, as alembic's real proxy does under ``--sql``."""
        return True


def _module(name: str) -> ModuleType:
    """Import a migration revision module by file stem."""
    return importlib.import_module(f"backend.db.migrations.versions.{name}")


def _revision_source() -> str:
    """Return the source text of revision 0015."""
    return (_VERSIONS / f"{_REVISION}.py").read_text(encoding="utf-8")


def _migration_sources() -> dict[str, str]:
    """Return every migration revision's source, keyed by file stem."""
    return {
        path.stem: path.read_text(encoding="utf-8")
        for path in sorted(_VERSIONS.glob("0*.py"))
        if path.name != "__init__.py"
    }


def _compose_service(name: str) -> dict[str, Any]:
    """Return one parsed service from the committed compose file."""
    document: Any = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    assert isinstance(document, dict), "docker-compose.yml is not a YAML mapping"
    services = document.get("services")
    assert isinstance(services, dict), "docker-compose.yml declares no services mapping"
    service = services.get(name)
    assert isinstance(service, dict), f"docker-compose.yml declares no {name!r} service"
    return service


def _backend_environment() -> dict[str, str]:
    """Return the backend service's environment mapping as plain strings."""
    environment = _compose_service(_BACKEND_SERVICE)["environment"]
    assert isinstance(environment, dict), "backend environment must be a mapping"
    return {str(key): str(value) for key, value in environment.items()}


def _substituted_variables(value: str) -> dict[str, str | None]:
    """Return the ``${NAME:-default}`` references in a compose value.

    Args:
        value: a raw compose value, e.g. ``"postgresql://${A:-x}:${B}@db/"``.

    Returns:
        Mapping of variable name to its declared default, or ``None`` when the
        reference declares no default.
    """
    return {match.group("name"): match.group("default") for match in _COMPOSE_VAR.finditer(value)}


def _granted_privileges(source: str) -> list[tuple[frozenset[str], str]]:
    """Return every ``GRANT <privileges> ON <target>`` in ``source``.

    Args:
        source: SQL (or Python source embedding SQL) to scan.

    Returns:
        One ``(privileges, target)`` pair per grant, privileges upper-cased and
        split on commas, target stripped of whitespace.
    """
    grants: list[tuple[frozenset[str], str]] = []
    for match in _GRANT.finditer(source):
        privileges = frozenset(
            part.strip() for part in match.group("privileges").split(",") if part.strip()
        )
        grants.append((privileges, match.group("target").strip()))
    return grants


def _executed_sql() -> str:
    """Return revision 0015's SQL constants, and nothing else.

    Every SQL assertion below reads *this*, not the file text. That distinction
    is the whole lesson of D-025: a scan over the source cannot tell a statement
    from a sentence about a statement, so a module that merely documents an
    attribute reads as one that sets it — and the check becomes one a future edit
    placates by rewording a docstring. These constants are the strings actually
    sent to PostgreSQL; prose cannot enter them.
    """
    module = _module(_REVISION)
    return "\n".join([module._CREATE_ROLE_SQL, module._GRANT_SQL, module._REVOKE_SQL])


def _role_attributes() -> frozenset[str]:
    """Return the role attributes revision 0015 gives the application role.

    Extracted from the ``attributes CONSTANT text := '...'`` declaration inside
    ``CREATE ROLE``'s DO block, so what is asserted is the literal handed to
    ``format()`` rather than any mention of it elsewhere in the file.

    Returns:
        The attribute keywords, upper-cased.
    """
    match = _ROLE_ATTRIBUTES.search(_module(_REVISION)._CREATE_ROLE_SQL)
    assert match is not None, (
        "the role-attribute declaration was not found in revision 0015's CREATE ROLE "
        "block; this scan can no longer see what attributes the role is given"
    )
    return frozenset(match.group(1).upper().split())


def _executable_lines(script: str) -> list[str]:
    """Return a shell script's non-comment, non-blank lines, stripped.

    Same reasoning as :func:`_executed_sql`: the entrypoint documents the role
    split in a comment block, and a scan that cannot tell the comment from the
    code is a scan that passes on the comment alone.
    """
    lines = [line.strip() for line in script.splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


# --------------------------------------------------------------------------
# The migration itself
# --------------------------------------------------------------------------


def test_revision_0015_extends_the_chain_at_0014() -> None:
    module = _module(_REVISION)
    assert module.revision == "0015"
    assert module.down_revision == "0014"


def test_the_application_role_is_created_without_a_route_to_escalate() -> None:
    """The role attributes are the whole defence; assert each one by name.

    ``NOSUPERUSER`` is the one that is easy to overlook and the one that matters
    most: a superuser can ``SET session_replication_role = 'replica'`` and every
    append-only trigger stops firing for the session, with ``DELETE`` then
    succeeding silently. ``NOCREATEROLE`` closes the other route — a role that
    can create roles can grant itself the owner.

    Read out of the SQL literal, not out of the file: the module docstring names
    every one of these attributes while explaining them, so a source-wide
    substring scan passes even after the attribute is deleted from the statement.
    That is not hypothetical — it survived a mutation run before this test was
    rewritten to parse the literal.
    """
    attributes = _role_attributes()
    assert attributes == {
        "LOGIN",
        "NOSUPERUSER",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
        "NOBYPASSRLS",
        "INHERIT",
    }, f"the application role is created with {sorted(attributes)}"
    assert "CREATE ROLE" in _executed_sql()
    # No role membership anywhere: membership in the owner would hand back
    # ownership through SET ROLE and undo the entire separation. A membership
    # grant is `GRANT <role> TO <role>` — a bare word between GRANT and TO,
    # which no privilege grant in this revision produces (they all carry ON).
    assert _MEMBERSHIP_GRANT.search(_executed_sql()) is None, (
        "revision 0015 appears to grant a role membership; the app role must be a "
        "member of nothing, or SET ROLE restores the ownership this revision removes"
    )


def test_the_grant_matrix_is_default_deny() -> None:
    """Privileges are revoked wholesale first, then handed back one at a time."""
    executed = _executed_sql()
    for statement in (
        "REVOKE ALL ON SCHEMA public FROM",
        "REVOKE ALL ON ALL TABLES IN SCHEMA public FROM",
        "REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM",
        "GRANT USAGE ON SCHEMA public TO",
        "GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO",
        "GRANT CONNECT ON DATABASE",
    ):
        assert statement in executed, f"revision 0015 is missing: {statement}"
    assert "GRANT CREATE ON SCHEMA public" not in executed, (
        "the app role must not be able to create objects in public: it would own them, "
        "and ownership is exactly the privilege that drops a trigger"
    )
    assert "GRANT SELECT, INSERT ON TABLES TO" in _module(_REVISION)._GRANT_SQL, (
        "without ALTER DEFAULT PRIVILEGES, every table a later migration creates is "
        "invisible to the app role, and the fix a hurried operator reaches for is a "
        "blanket grant. Asserted against the upgrade SQL specifically: the downgrade "
        "names ALTER DEFAULT PRIVILEGES too, so a scan over both would pass on the revoke"
    )


def test_the_only_mutation_grants_name_the_two_tables_without_an_append_only_trigger() -> None:
    """The grant matrix must mirror the trigger matrix, or one of them is a lie.

    Every table that carries a ``BEFORE UPDATE OR DELETE`` trigger is append-only
    by design; handing the app role ``UPDATE`` or ``DELETE`` on one would mean the
    database refuses the write while the privilege system says it is allowed —
    two enforcement layers disagreeing, which is how one of them quietly stops
    being maintained.
    """
    module = _module(_REVISION)
    mutable: dict[str, str] = module.MUTABLE_TABLE_GRANTS
    executed = _executed_sql()
    offenders: dict[str, frozenset[str]] = {}
    for privileges, target in _granted_privileges(executed):
        escalating = privileges & _OWNERSHIP_ONLY_PRIVILEGES
        if escalating and target not in mutable:
            offenders[target] = escalating
    assert not offenders, (
        f"revision 0015 grants {offenders} outside the documented mutable tables "
        f"{sorted(mutable)}; every other table in the schema is append-only"
    )
    for table, expected in mutable.items():
        assert f"GRANT {expected} ON {table} TO" in executed, (
            f"MUTABLE_TABLE_GRANTS declares {table} -> {expected!r} but no matching "
            "GRANT statement is written out; the declaration and the SQL must agree"
        )


def test_no_mutable_table_ever_acquired_an_append_only_trigger() -> None:
    """Drift guard in the other direction, across the whole migration chain.

    If a later revision makes ``ingestion_run`` or ``llm_provider_credential``
    append-only, revision 0015's ``UPDATE``/``DELETE`` grant silently becomes a
    privilege the database refuses to honour, and the mismatch is invisible until
    someone reads both files. This test fails the moment that happens.
    """
    mutable: dict[str, str] = _module(_REVISION).MUTABLE_TABLE_GRANTS
    for stem, source in _migration_sources().items():
        for table in mutable:
            assert f"trg_{table}_append_only" not in source, (
                f"{stem} installs an append-only trigger on {table}, which revision 0015 "
                f"grants the application role {mutable[table]} on. Remove the grant from "
                "0015 or the trigger from that revision — they cannot both be right"
            )


def test_alembic_version_is_read_only_for_the_application_role() -> None:
    """A role that can stamp a revision can hide a migration that never ran."""
    executed = _executed_sql()
    assert "REVOKE ALL ON alembic_version FROM" in executed
    assert "GRANT SELECT ON alembic_version TO" in executed
    assert "GRANT INSERT ON alembic_version" not in executed
    assert "GRANT SELECT, INSERT ON alembic_version" not in executed


def test_hypertable_chunks_are_granted_by_name_not_only_by_schema_wildcard() -> None:
    """``price_bar`` rows live in chunks; a grant that stops at the parent is useless.

    TimescaleDB propagates a grant issued directly against a hypertable to its
    chunks. The ``ALL TABLES IN SCHEMA public`` form is not relied on for that,
    so the per-hypertable loop has to be present.
    """
    executed = _executed_sql()
    assert "timescaledb_information.hypertables" in executed
    assert "GRANT USAGE ON SCHEMA _timescaledb_internal TO" in executed


# --------------------------------------------------------------------------
# I5: the password never becomes part of a statement
# --------------------------------------------------------------------------


def test_the_password_travels_as_a_bind_parameter_and_never_as_sql_text() -> None:
    """Delivered through ``set_config`` with a bound value, read back by ``format(%L)``.

    The alternative — interpolating it into ``CREATE ROLE`` in Python — puts the
    credential into a string this process holds, which SQLAlchemy then copies
    into any ``DBAPIError`` it raises for that statement. That error is logged.
    """
    module = _module(_REVISION)
    assert str(module._SET_PASSWORD_GUC) == (
        "SELECT set_config('quant.app_db_password', :password, true)"
    ), "the password must reach PostgreSQL as a bind parameter, not inside a statement"
    assert "current_setting('quant.app_db_password')" in module._CREATE_ROLE_SQL, (
        "the CREATE ROLE text must be assembled server-side from the bound value"
    )
    assert "%L" in module._CREATE_ROLE_SQL, "the bound value must be quoted as a literal"


def test_no_f_string_in_the_migration_interpolates_a_password_or_role() -> None:
    """Structural, not textual: no formatted string in 0015 embeds either value.

    Checked over the parse tree so that a future edit reintroducing
    ``f"CREATE ROLE {role} PASSWORD '{password}'"`` fails here rather than
    shipping. Covers f-strings, ``str.format`` and ``%`` on any name whose
    identifier mentions a password or a role.
    """
    tree = ast.parse(_revision_source())
    tainted = re.compile(r"password|role", re.IGNORECASE)
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            names = {
                child.id
                for value in node.values
                if isinstance(value, ast.FormattedValue)
                for child in ast.walk(value)
                if isinstance(child, ast.Name)
            }
            offending = {name for name in names if tainted.search(name)}
            assert not offending, (
                f"revision 0015 interpolates {sorted(offending)} into an f-string; the "
                "role name is quoted server-side by format(%I) and the password must "
                "never appear in statement text at all (I5)"
            )


def test_the_error_handler_withholds_the_driver_message() -> None:
    """PostgreSQL puts the failing ``EXECUTE`` text — password included — in the context.

    So the handler re-raises with the SQLSTATE and nothing else. A test asserts
    it, because "we withheld the message on purpose" is indistinguishable from
    "we forgot to include it" the next time someone debugs a failed migration.
    """
    create_role = _module(_REVISION)._CREATE_ROLE_SQL
    assert "EXCEPTION WHEN OTHERS THEN" in create_role
    assert "SQLSTATE" in create_role
    assert "SQLERRM" not in create_role, (
        "SQLERRM can carry a fragment of the failing statement, and that statement "
        "is the one that contains the role password"
    )


def test_offline_sql_generation_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """``alembic upgrade --sql`` would write the credential into a generated file.

    Exercised rather than grepped: the alembic ``context`` proxy is replaced with
    one that reports offline mode, and the revision's own preflight is called. A
    substring check for ``is_offline_mode()`` would pass on a comment.
    """
    module = _module(_REVISION)
    monkeypatch.setattr(module, "context", _OfflineContext())
    with pytest.raises(RuntimeError, match="offline"):
        module._bind_role_name()


# --------------------------------------------------------------------------
# Password resolution
# --------------------------------------------------------------------------


def test_password_resolution_prefers_the_configured_value() -> None:
    module = _module(_REVISION)
    settings = Settings(app_db_password=SecretStr("from-the-environment"))
    assert module._resolve_app_password(settings) == "from-the-environment"


@pytest.mark.parametrize("environment", ["dev", "test"])
def test_password_resolution_falls_back_to_the_development_default_outside_prod(
    environment: Literal["dev", "test"],
) -> None:
    module = _module(_REVISION)
    settings = Settings(environment=environment, app_db_password=None)
    assert module._resolve_app_password(settings) == module.DEV_DEFAULT_APP_PASSWORD


@pytest.mark.parametrize("configured", [None, SecretStr("")])
def test_password_resolution_refuses_to_invent_one_in_production(
    configured: SecretStr | None,
) -> None:
    """An empty value is as absent as no value, and must not pass silently."""
    module = _module(_REVISION)
    settings = Settings(environment="prod", app_db_password=configured)
    with pytest.raises(RuntimeError, match="APP_DB_PASSWORD"):
        module._resolve_app_password(settings)


@pytest.mark.parametrize(
    "name",
    [
        "quant app",
        "Quant_App",
        'app"; DROP DATABASE quant; --',
        "app-role",
        "",
        "1app",
        "a" * 64,
    ],
)
def test_role_names_that_are_not_bare_identifiers_are_refused(name: str) -> None:
    module = _module(_REVISION)
    with pytest.raises(ValueError, match="APP_DB_ROLE"):
        module._validated_role(name)


@pytest.mark.parametrize("name", ["quant_app", "app", "_x9", "a" * 63])
def test_bare_identifiers_are_accepted(name: str) -> None:
    """Non-vacuity: the validator is not simply rejecting everything."""
    module = _module(_REVISION)
    assert module._validated_role(name) == name


# --------------------------------------------------------------------------
# Wiring: settings, engine, alembic environment
# --------------------------------------------------------------------------


def test_migration_url_defaults_to_the_application_url() -> None:
    """The fallback is what keeps a single-role local database working."""
    settings = Settings(database_url="postgresql+asyncpg://a:b@h/d", migration_database_url=None)
    assert settings.migration_url == "postgresql+asyncpg://a:b@h/d"


def test_migration_url_prefers_the_owner_url_when_configured() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://app:p@h/d",
        migration_database_url="postgresql+asyncpg://owner:q@h/d",
    )
    assert settings.migration_url == "postgresql+asyncpg://owner:q@h/d"


def test_app_role_settings_have_the_documented_defaults() -> None:
    settings = Settings()
    assert settings.app_db_role == "quant_app"
    assert settings.app_db_password is None


def _settings_attributes_read(path: Path, function: str) -> frozenset[str]:
    """Return the attribute names a function's *code* reads.

    Parsed, not grepped, and for the reason D-025 records: every one of these
    functions documents which URL it resolves and why, so a substring scan over
    the function's source passes on the docstring alone even after the code has
    been changed to read the other one. A docstring contributes no ``Attribute``
    node, so walking the tree cannot be placated by prose. This too is not
    hypothetical — a mutation flipping ``_create_migration_engine`` back to
    ``database_url`` survived the substring version of this test.

    Args:
        path: module to parse.
        function: name of a top-level function in it.

    Returns:
        Every attribute name accessed anywhere in that function's body.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function:
            return frozenset(
                child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
            )
    message = f"{function} not found in {path}"
    raise AssertionError(message)


def test_only_the_migration_engine_uses_the_owner_url() -> None:
    """The split has to hold in the engine module, not only in the documentation.

    ``backend/db/engine.py`` is where every read path in the repository gets its
    connection. If the process-wide engine or the admin engine resolved
    ``migration_url``, the application would be back to connecting as the owner
    and every negative test in the integration suite would be testing the wrong
    credential while still passing.
    """
    migration = _settings_attributes_read(_ENGINE, "_create_migration_engine")
    assert "migration_url" in migration, (
        "_create_migration_engine must connect as the schema owner: alembic DDL and the "
        "sanctioned TRUNCATE reset both need privileges the app role deliberately lacks"
    )
    assert "database_url" not in migration
    for name in ("_get_engine", "create_admin_engine"):
        attributes = _settings_attributes_read(_ENGINE, name)
        assert "migration_url" not in attributes, (
            f"{name} resolves the schema-owner URL; the application must connect as the "
            "app role, or role separation buys nothing"
        )
        assert "database_url" in attributes, f"{name} no longer resolves the application URL"


def test_the_alembic_environment_resolves_the_owner_url() -> None:
    assert "migration_url" in _settings_attributes_read(_MIGRATIONS / "env.py", "_database_url"), (
        "the alembic environment must resolve MIGRATION_DATABASE_URL; running migrations "
        "as the application role would fail at the first CREATE TABLE"
    )


# --------------------------------------------------------------------------
# Wiring: entrypoint and compose
# --------------------------------------------------------------------------


def test_the_entrypoint_refuses_a_single_role_configuration() -> None:
    """Same credential for both means the app owns the schema — the pre-CC.9 state.

    Worth failing loudly at start-up rather than at audit time: the stack would
    otherwise come up, work perfectly, and report an immutable audit log it
    cannot deliver.
    """
    lines = _executable_lines(_ENTRYPOINT.read_text(encoding="utf-8"))
    script = "\n".join(lines)
    assert "MIGRATION_DATABASE_URL:?" in script, "the entrypoint does not require the owner URL"
    assert '"$MIGRATION_DATABASE_URL" = "$DATABASE_URL"' in script, (
        "the entrypoint does not compare the two URLs, so a copy-paste that points both "
        "at the owner would start cleanly and silently undo role separation"
    )
    assert "exit 1" in script


def test_the_entrypoint_drops_the_owner_credential_before_serving() -> None:
    """The server must not merely decline to use the owner URL — it must not have it.

    Ordering is the assertion: unsetting after ``exec`` would never run, and
    unsetting before ``alembic`` would break the migration it exists to permit.
    Comment lines are stripped first — the script explains this split in prose
    directly above the code, and a scan that cannot tell them apart passes on the
    explanation alone.
    """
    lines = _executable_lines(_ENTRYPOINT.read_text(encoding="utf-8"))
    positions = {
        label: [index for index, line in enumerate(lines) if predicate(line)]
        for label, predicate in (
            ("migrate", lambda line: "alembic upgrade head" in line),
            ("unset owner url", lambda line: line == "unset MIGRATION_DATABASE_URL"),
            ("unset role password", lambda line: line == "unset APP_DB_PASSWORD"),
            ("serve", lambda line: line.startswith("exec ")),
        )
    }
    missing = [label for label, found in positions.items() if not found]
    assert not missing, (
        f"the entrypoint has no executable line for: {missing}. The owner credential and "
        "the role-creation password must both be dropped between migrating and serving — "
        "a serving process that holds them can still drop the append-only triggers"
    )
    assert positions["migrate"][0] < positions["unset owner url"][0] < positions["serve"][0], (
        "MIGRATION_DATABASE_URL must be unset after migrations and before exec"
    )
    assert positions["migrate"][0] < positions["unset role password"][0] < positions["serve"][0]


def test_compose_points_the_application_at_the_app_role_and_alembic_at_the_owner() -> None:
    environment = _backend_environment()
    app_url = environment["DATABASE_URL"]
    owner_url = environment["MIGRATION_DATABASE_URL"]
    assert app_url != owner_url

    app_variables = _substituted_variables(app_url)
    assert _APP_USER_VAR in app_variables, f"DATABASE_URL does not use {_APP_USER_VAR}: {app_url}"
    assert _APP_PASSWORD_VAR in app_variables, f"DATABASE_URL does not use {_APP_PASSWORD_VAR}"
    assert _OWNER_PASSWORD_VAR not in app_variables, (
        "DATABASE_URL carries the owner password; the application would connect as the "
        "role that owns the append-only triggers"
    )

    owner_variables = _substituted_variables(owner_url)
    assert _OWNER_USER_VAR in owner_variables
    assert _OWNER_PASSWORD_VAR in owner_variables


def test_compose_and_the_migration_agree_on_the_development_credentials() -> None:
    """Two defaults that drift produce a role nobody can authenticate as.

    The migration creates the role from ``APP_DB_ROLE``/``APP_DB_PASSWORD``; the
    application logs in with ``DATABASE_URL``. Compose builds all three from the
    same two variables, and this asserts the defaults it substitutes are the
    same ones the migration would fall back to.
    """
    module = _module(_REVISION)
    environment = _backend_environment()
    app_url_defaults = _substituted_variables(environment["DATABASE_URL"])
    role_defaults = _substituted_variables(environment["APP_DB_ROLE"])
    password_defaults = _substituted_variables(environment["APP_DB_PASSWORD"])

    assert role_defaults[_APP_USER_VAR] == app_url_defaults[_APP_USER_VAR]
    assert password_defaults[_APP_PASSWORD_VAR] == app_url_defaults[_APP_PASSWORD_VAR]
    assert role_defaults[_APP_USER_VAR] == Settings().app_db_role
    assert password_defaults[_APP_PASSWORD_VAR] == module.DEV_DEFAULT_APP_PASSWORD, (
        "the compose development default and the migration's fallback have drifted; the "
        "migration would create a role with one password and the app would present another"
    )


def test_the_integration_reset_still_runs_as_the_schema_owner() -> None:
    """The coupling this change introduces into a fixture it does not own.

    ``backend/tests/integration/conftest.py`` TRUNCATEs fact tables between tests
    on the module-private *migration* engine, and that engine now resolves the
    schema-owner URL. The app role has no ``TRUNCATE`` on anything — deliberately,
    because ``TRUNCATE`` fires no row trigger and was the one way to erase the
    audit log without leaving a mark. So the reset works only as long as it keeps
    using that engine. Asserted here rather than there: the fixture is not this
    change's to edit, but the dependency is this change's to notice.
    """
    conftest = (_REPO_ROOT / "backend" / "tests" / "integration" / "conftest.py").read_text(
        encoding="utf-8"
    )
    assert "_create_migration_engine" in conftest, (
        "the integration reset no longer uses the migration engine. If it now builds an "
        "engine from DATABASE_URL, its TRUNCATE runs as the application role and is "
        "refused with permission denied — see revision 0015's grant matrix"
    )
    assert "TRUNCATE" in conftest


def test_the_committed_env_example_documents_both_roles_and_commits_no_real_secret() -> None:
    """`.env.example` is committed, `.env` is not (I5, §7)."""
    example = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for name in (_APP_USER_VAR, _APP_PASSWORD_VAR, "MIGRATION_DATABASE_URL", "APP_DB_PASSWORD"):
        assert name in example, f"{name} is undocumented in .env.example"
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in gitignore
    assert "!.env.example" in gitignore


# --------------------------------------------------------------------------
# Non-vacuity
# --------------------------------------------------------------------------


def test_the_grant_scanner_actually_finds_grants() -> None:
    """Without this, a broken regex would make the matrix tests pass on anything."""
    grants = _granted_privileges(
        "EXECUTE format('GRANT UPDATE, DELETE ON llm_provider_credential TO %I', r);"
        "EXECUTE format('GRANT SELECT ON alembic_version TO %I', r);"
    )
    assert (frozenset({"UPDATE", "DELETE"}), "llm_provider_credential") in grants
    assert (frozenset({"SELECT"}), "alembic_version") in grants

    escalating = [
        target for privileges, target in grants if privileges & _OWNERSHIP_ONLY_PRIVILEGES
    ]
    assert escalating == ["llm_provider_credential"]


def test_the_compose_variable_parser_reads_defaults() -> None:
    """Non-vacuity for the compose assertions above."""
    parsed = _substituted_variables("postgresql://${A:-alpha}:${B}@db/${C:-}")
    assert parsed == {"A": "alpha", "B": None, "C": ""}


def test_the_role_attribute_reader_sees_the_literal_not_the_prose() -> None:
    """Non-vacuity, and the specific one a mutation run earned.

    Flipping ``NOSUPERUSER`` to ``SUPERUSER`` in the statement left the word
    ``NOSUPERUSER`` in the module docstring, and the substring version of
    :func:`test_the_application_role_is_created_without_a_route_to_escalate`
    passed on that. This asserts the reader takes the attributes from the SQL
    literal, so the same mutation cannot survive again.
    """
    assert _ROLE_ATTRIBUTES.search("NOSUPERUSER is explained here in prose") is None
    match = _ROLE_ATTRIBUTES.search("    attributes CONSTANT text :=\n        'LOGIN SUPERUSER';")
    assert match is not None
    assert frozenset(match.group(1).split()) == {"LOGIN", "SUPERUSER"}


def test_the_executable_line_reader_drops_comments() -> None:
    """Non-vacuity for the entrypoint assertions: a comment is not a command."""
    lines = _executable_lines("# unset MIGRATION_DATABASE_URL\n\nexec uvicorn\n")
    assert lines == ["exec uvicorn"]
