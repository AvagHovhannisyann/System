"""LLM provider registry: credentials, per-task model assignment, connection probe (P7.1).

§6.5 calls the Agents & Extraction page "the page the operator lives in". This
package is the backend half of its first two sections, and the three rules that
shape it come from §6.5, §7 and I5 rather than from convenience:

- **Keys are encrypted at rest and permanently masked.**
  :mod:`backend.extraction.providers.registry` stores Fernet tokens produced
  under the environment KEK (:mod:`backend.core.crypto`) and hands out a view
  that has no field capable of holding a key. There is no reveal function, no
  reveal endpoint, and a test that asserts the API surface contains neither —
  not merely that nothing currently calls one.
- **Changing an assignment creates a version, it does not mutate one.**
  :mod:`backend.extraction.providers.assignments` appends a new row per change
  and the database refuses UPDATE and DELETE against past versions, so the
  configuration an extraction run used stays readable exactly as written (I2).
- **Every change is an event.** Both modules record through
  :mod:`backend.db.audit` with actor and correlation id, because §6.11's rule —
  config changes are events, not mutations — applies to provider credentials
  and model assignments like anything else.

:mod:`backend.extraction.providers.probe` is the "Test connection" action. It
issues one authenticated, zero-token listing request, reports latency and
status, and refuses outright when no key is configured rather than reporting a
result it did not measure. It is invoked explicitly and never on a timer:
B4 is unresolved, so no spend cap exists to run automatic calls under, and its
live path is consequently unexercised — the module docstring says so plainly.
"""

from __future__ import annotations

from backend.extraction.providers.assignments import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    MAX_TEMPERATURE,
    AssignmentNotConfiguredError,
    ModelAssignment,
    assign_task_model,
    assignment_versions,
    current_assignment,
    list_current_assignments,
)
from backend.extraction.providers.catalog import Provider, provider_from_name
from backend.extraction.providers.probe import (
    PROBE_ENDPOINTS,
    PROBE_TIMEOUT_S,
    ProbeOutcome,
    ProbeRequest,
    ProbeResult,
    build_probe_request,
    probe_provider,
)
from backend.extraction.providers.registry import (
    ProviderKeyAlreadyConfiguredError,
    ProviderKeyError,
    ProviderKeyNotConfiguredError,
    ProviderKeyView,
    add_provider_key,
    configured_providers,
    delete_provider_key,
    get_provider_key,
    list_provider_keys,
    load_provider_secret,
    rotate_provider_key,
)

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "MAX_TEMPERATURE",
    "PROBE_ENDPOINTS",
    "PROBE_TIMEOUT_S",
    "AssignmentNotConfiguredError",
    "ModelAssignment",
    "ProbeOutcome",
    "ProbeRequest",
    "ProbeResult",
    "Provider",
    "ProviderKeyAlreadyConfiguredError",
    "ProviderKeyError",
    "ProviderKeyNotConfiguredError",
    "ProviderKeyView",
    "add_provider_key",
    "assign_task_model",
    "assignment_versions",
    "build_probe_request",
    "configured_providers",
    "current_assignment",
    "delete_provider_key",
    "get_provider_key",
    "list_current_assignments",
    "list_provider_keys",
    "load_provider_secret",
    "probe_provider",
    "provider_from_name",
    "rotate_provider_key",
]
