"""Application configuration via pydantic-settings.

All configuration is environment-driven. Field names map to upper-case
environment variables (``database_url`` <- ``DATABASE_URL``); a local
``.env`` file is read when present. Assumes the process is launched from
the repository root (the ``.env`` path is resolved relative to the CWD).
"""

from __future__ import annotations

from functools import lru_cache
from importlib import metadata
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DISTRIBUTION_NAME = "quant-research-platform"


def _package_version() -> str:
    """Return the installed version of this distribution.

    Assumes the project was installed (``uv sync`` installs it into the
    virtualenv). Falls back to ``"0.0.0+unknown"`` when the distribution
    metadata is unavailable (e.g. sources used without installation) so
    that configuration loading never fails on a metadata lookup.
    """
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


class Settings(BaseSettings):
    """Runtime settings for the backend.

    Values come from (highest precedence first): constructor arguments,
    environment variables, a ``.env`` file, then the defaults below.
    Unknown environment entries are ignored. ``app_version`` defaults to
    the installed package version but may be overridden via ``APP_VERSION``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://quant:quant@localhost:5432/quant"
    """Connection URL the **application** uses, as the least-privileged app role.

    Since CC.9 (D-017) this credential names a role that owns nothing: it holds
    ``SELECT``/``INSERT`` and cannot ``ALTER``, ``DROP``, ``TRUNCATE`` or
    disable a trigger. Schema changes travel through
    :attr:`migration_database_url` instead. The default above is the local
    single-role development URL and stays that way so a bare ``pytest`` run
    against a throwaway database still works; the compose stack points this at
    the separated role.
    """

    migration_database_url: str | None = None
    """Connection URL used **only** by alembic and the sanctioned admin reset.

    Names the schema-owner role — the one role that may create, alter or drop
    anything, including the append-only triggers. Resolved through
    :attr:`migration_url`, which falls back to :attr:`database_url` when this is
    unset.

    The fallback exists so a single-role local database (and the integration
    test container, which owns no compose file) keeps working. Its failure mode
    in a separated deployment is loud rather than silent: migrations attempted
    as the app role are refused by PostgreSQL at the first ``CREATE TABLE``.
    """

    app_db_role: str = "quant_app"
    """Name of the application role that migration 0015 creates and grants.

    Must be a bare lower-case SQL identifier; revision 0015 validates it and
    refuses anything else rather than interpolating an unchecked name into DDL.
    """

    app_db_password: SecretStr | None = None
    """Password for :attr:`app_db_role`, from the environment only (I5).

    Consumed by revision 0015 when it creates or re-points the role, and never
    read anywhere else — the running application authenticates with
    :attr:`database_url`, which carries its own copy. ``SecretStr`` so the value
    cannot leak through ``repr`` or a settings dump. Unset is an error at
    ``ENVIRONMENT=prod``; outside prod the committed development default is used
    with a warning (see ``backend/db/migrations/versions/0015_role_separation.py``).
    """

    redis_url: str = "redis://localhost:6379/0"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_version: str = Field(default_factory=_package_version)

    sec_user_agent: str | None = None
    """Contact string sent as ``User-Agent`` to SEC EDGAR.

    SEC's fair-access policy requires every automated requester to identify
    itself with contact information. There is deliberately **no default**:
    the EDGAR connector refuses to run when this is unset rather than
    sending an anonymous or invented identifier (policy compliance is
    fail-closed, and a fabricated contact would be worse than none).
    """

    secrets_kek: SecretStr | None = None
    """Key-encryption key (Fernet, urlsafe-base64 32 bytes) from the environment.

    Wraps provider API keys at rest (§7, I5). Sourced **only** from the
    environment — never from the database, never committed — so a database
    dump alone cannot decrypt stored credentials. ``SecretStr`` so the value
    cannot leak through ``repr``/logs. Components needing it raise when it is
    unset rather than falling back to an unencrypted path.
    """

    @property
    def migration_url(self) -> str:
        """Return the URL that schema-owning work connects with (CC.9).

        The single resolution point for "which role runs DDL": alembic's
        migration environment and the module-private migration engine both read
        this, so they cannot disagree about it. Returns
        :attr:`migration_database_url` when set, otherwise
        :attr:`database_url` — see that field for why the fallback is safe.

        Returns:
            A SQLAlchemy connection URL string (``postgresql+asyncpg://...``).
        """
        return self.migration_database_url or self.database_url

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        """Upper-case string log levels so ``LOG_LEVEL=debug`` is accepted."""
        if isinstance(value, str):
            return value.upper()
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    Assumes the environment does not change mid-process; tests that mutate
    the environment must call ``get_settings.cache_clear()`` first.
    """
    return Settings()
