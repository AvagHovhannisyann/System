"""API-layer security mechanisms required by DIRECTIVE.md §7 (CC.2).

Three §7 obligations land at the HTTP boundary and live here:

- :mod:`backend.api.security.csrf` — signed double-submit CSRF protection,
  **enforced by default** on every unsafe method;
- :mod:`backend.api.security.ratelimit` — Redis-backed per-client rate
  limiting on mutating requests, **fail-closed** when Redis is unreachable;
- :mod:`backend.api.security.settings` — the environment-driven knobs for
  both.

The fourth §7 obligation, "parameterized queries only", is a static
property rather than a runtime mechanism; it is enforced by the AST check in
``backend/tests/test_api_sql_parameterization.py``, which runs as part of the
ordinary test suite (and therefore in CI).

**Timing note.** These mechanisms were built *before* the first mutating
endpoint exists (the Data Health re-sync trigger, P3.10, is the first
consumer). That ordering is deliberate and it shapes the design: both
mechanisms are default-on middleware rather than per-route opt-ins, so the
first mutating route — and every one after it — is protected without its
author having to remember anything. See the module docstrings for the
argument.
"""

from backend.api.security.csrf import (
    CSRF_COOKIE_NAME,
    CSRF_EXEMPT_PATHS,
    CSRF_HEADER_NAME,
    SAFE_METHODS,
    CsrfConfigurationError,
    CsrfError,
    CsrfMiddleware,
    CsrfProtect,
    resolve_csrf_secret,
)
from backend.api.security.ratelimit import (
    RATE_LIMIT_EXEMPT_PATHS,
    InMemoryFixedWindowStore,
    RateLimitDecision,
    RateLimiterUnavailableError,
    RateLimitMiddleware,
    RateLimitPolicy,
    RateLimitStore,
    RedisFixedWindowStore,
)
from backend.api.security.settings import ApiSecuritySettings

__all__ = [
    "CSRF_COOKIE_NAME",
    "CSRF_EXEMPT_PATHS",
    "CSRF_HEADER_NAME",
    "RATE_LIMIT_EXEMPT_PATHS",
    "SAFE_METHODS",
    "ApiSecuritySettings",
    "CsrfConfigurationError",
    "CsrfError",
    "CsrfMiddleware",
    "CsrfProtect",
    "InMemoryFixedWindowStore",
    "RateLimitDecision",
    "RateLimitMiddleware",
    "RateLimitPolicy",
    "RateLimitStore",
    "RateLimiterUnavailableError",
    "RedisFixedWindowStore",
    "resolve_csrf_secret",
]
