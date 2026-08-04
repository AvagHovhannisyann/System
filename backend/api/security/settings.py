"""Environment-driven configuration for the §7 API security mechanisms (CC.2).

A separate :class:`pydantic_settings.BaseSettings` model rather than fields on
:class:`backend.core.config.Settings`: these knobs are HTTP-boundary concerns
consumed only by :func:`backend.api.app.create_app`, and keeping them out of
the process-wide settings object avoids a cache-invalidation coupling: the
core ``Settings`` is wrapped in ``lru_cache``, whereas this model is built
fresh on every ``create_app`` call, so a test that mutates the environment and
rebuilds the app sees its own values with no cache-clearing ceremony.

Both models read the same ``.env`` file and ignore unknown entries, so there
is one configuration surface for the operator and two typed views of it.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CSRF_TOKEN_MAX_AGE_S = 8 * 60 * 60
"""Default CSRF token lifetime in seconds (8 h — one operator working day)."""

DEFAULT_RATE_LIMIT_REQUESTS = 30
"""Default mutating-request budget per client per route per window."""

DEFAULT_RATE_LIMIT_WINDOW_S = 60.0
"""Default rate-limit window length in seconds."""

DEFAULT_RATE_LIMIT_TIMEOUT_S = 0.5
"""Default per-call timeout in seconds for the rate-limit store round trip."""


class ApiSecuritySettings(BaseSettings):
    """CSRF and rate-limit configuration, sourced from the environment.

    Field names map to upper-case environment variables exactly as in
    :class:`backend.core.config.Settings` (``csrf_secret`` <- ``CSRF_SECRET``).
    Precedence: constructor arguments, environment, ``.env``, defaults.

    Units are explicit in the field names: ``*_s`` values are seconds,
    ``rate_limit_requests`` is a count of requests per window.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    csrf_secret: SecretStr | None = None
    """Signing key for CSRF tokens; see :func:`~backend.api.security.csrf.resolve_csrf_secret`.

    Deliberately distinct from ``SECRETS_KEK``: the KEK wraps provider API
    keys at rest (I5) and must never double as a general-purpose signing key,
    because compromising a short-lived web token then compromises stored
    credentials. Unset is tolerated outside production (an ephemeral
    per-process key is generated, with a warning); in production it is a
    startup error.
    """

    csrf_token_max_age_s: Annotated[int, Field(gt=0)] = DEFAULT_CSRF_TOKEN_MAX_AGE_S
    """Maximum age in seconds of an accepted CSRF token (also the cookie's ``Max-Age``)."""

    rate_limit_enabled: bool = True
    """Whether the rate-limit middleware is installed at all.

    Escape hatch for local debugging only. It is *not* a way to disable
    protection in production: a deployment that sets this to false has
    silently dropped a §7 control, which is why the default is on and the
    flag is logged at startup when it is off.
    """

    rate_limit_requests: Annotated[int, Field(ge=1)] = DEFAULT_RATE_LIMIT_REQUESTS
    """Requests allowed per client, per route, per window (>= 1)."""

    rate_limit_window_s: Annotated[float, Field(gt=0)] = DEFAULT_RATE_LIMIT_WINDOW_S
    """Window length in seconds (> 0)."""

    rate_limit_timeout_s: Annotated[float, Field(gt=0)] = DEFAULT_RATE_LIMIT_TIMEOUT_S
    """Timeout in seconds for one rate-limit store round trip.

    A hung Redis must not hang the API; exceeding this is treated exactly like
    Redis being unreachable, i.e. fail-closed (see
    :mod:`backend.api.security.ratelimit`).
    """
