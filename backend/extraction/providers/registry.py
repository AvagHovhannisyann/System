"""Provider API keys: encrypted at rest, masked in every reading (P7.1, §6.5, §7, I5).

What this module is allowed to hand out
---------------------------------------

Two shapes leave here, and only two:

* :class:`ProviderKeyView` — provider, **masked** rendering, rotation counter,
  timestamps. This is what every reader gets, including the API layer. It has
  no field that can carry a full key and no method that produces one.
* the plaintext, from :func:`load_provider_secret` — the single decryption path
  in the codebase, existing so an outbound provider call can be authenticated.
  Its result must never reach a response body, a template, a log line, or an
  exception message, and the API route module
  (:mod:`backend.api.routes.providers`) never imports it. A test asserts that
  statically, so the property survives a future author who does not read this
  paragraph.

There is no reveal function here, no ``unmask``, no ``reveal=True`` switch, and
no variant that widens the mask (§6.5: *display permanently masked with no
reveal endpoint*; §7: *no endpoint returns a full key under any
circumstance*).

Storage shape, and why the credential row is mutable
----------------------------------------------------

``llm_provider_credential`` holds **one row per provider: the key currently in
force.** Rotation UPDATEs it; deletion removes it. Every other table in this
schema is append-only, so the departure is worth the sentence: an append-only
credential table would keep every superseded ciphertext decryptable for as long
as the KEK lives, turning a KEK compromise from "every key in use" into "every
key ever used" — while rotation is normally performed *because* the old key
should stop existing.

The history §6.5 asks for is not lost, it is relocated to where it is safe:
every add, rotation and deletion is recorded in the append-only
``config_change_event`` log (:mod:`backend.db.audit`) with actor, correlation
id, and the **masked** before/after renderings. "What was the key before this
rotation?" is answerable as ``sk-...4f2a`` — which is precisely as much as §6.5
permits anyone to see — and that record is safe to keep forever because nothing
in it is recoverable.

Ordering, and a limitation this module cannot fix on its own
-------------------------------------------------------------

A credential change is two writes in two transactions: the state row, then the
audit event. :mod:`backend.db.audit` opens and commits its own transaction and
accepts no caller session — its module docstring names this limitation and says
the honest fix is a session-accepting variant *there*, not a caller reaching
around it to INSERT its own event. P7.1 does not own that file, so the
limitation stands, and the ordering is chosen deliberately rather than by
accident:

**state first, audit second.** If the audit write then fails, the exception
propagates unswallowed, so the operator learns the log may be missing an entry
instead of the call reporting success. The reverse order would let the log
assert a rotation that never took effect, and a log that over-reports is harder
to detect than one that fails loudly at the moment of the gap.

This is a known, recorded gap, not a resolved problem: it needs a
session-accepting recorder in :mod:`backend.db.audit` to close properly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa

from backend.core.crypto import encrypt_secret, mask_secret
from backend.core.logging import get_logger
from backend.db import ingest_writer_session
from backend.db.audit import record_config_changes
from backend.db.models import LlmProviderCredential
from backend.extraction.providers.catalog import Provider

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "AUDIT_SCOPE",
    "FIELD_KEY_VERSION",
    "FIELD_MASKED_KEY",
    "ProviderKeyAlreadyConfiguredError",
    "ProviderKeyError",
    "ProviderKeyNotConfiguredError",
    "ProviderKeyView",
    "add_provider_key",
    "configured_providers",
    "delete_provider_key",
    "get_provider_key",
    "list_provider_keys",
    "load_provider_secret",
    "rotate_provider_key",
]

_logger = get_logger(__name__)

AUDIT_SCOPE: Final = "llm_provider"
"""``config_change_event.scope`` under which credential changes are recorded."""

FIELD_MASKED_KEY: Final = "api_key_masked"
"""Audit field carrying the **masked** rendering — never the key, never the ciphertext.

Named ``..._masked`` rather than ``api_key`` so that no reader of the audit log,
and no future consumer joining on the field name, can mistake the recorded value
for the credential itself.
"""

FIELD_KEY_VERSION: Final = "key_version"
"""Audit field carrying the rotation counter, so a rotation is visible as such."""

_MIN_KEY_CHARS: Final = 8
"""Shortest credential accepted.

Not a format check — provider key formats change, and rejecting an unfamiliar
shape would turn a vendor's format change into a total outage (the same
reasoning :mod:`backend.ingest.fred.client` records for FRED). This only refuses
input too short to be any provider's key, which is almost always a truncated
paste or an empty form field.
"""


class ProviderKeyError(RuntimeError):
    """Base class for credential-registry failures raised here."""


class ProviderKeyNotConfiguredError(ProviderKeyError):
    """No credential is stored for the requested provider.

    Raised rather than returning ``None`` so that a caller about to spend money
    on a provider call cannot proceed on a falsy value it forgot to check
    (I3: an unavailable source raises).
    """


class ProviderKeyAlreadyConfiguredError(ProviderKeyError):
    """A credential already exists for this provider, so *adding* one is refused.

    Adding and rotating are separate operations on purpose: they mean different
    things to an operator and they must be distinguishable in the audit log.
    Replacing an existing key is :func:`rotate_provider_key`.
    """


@dataclass(frozen=True, slots=True)
class ProviderKeyView:
    """Everything a reader may know about a stored credential (§6.5).

    Deliberately has no field capable of carrying a full key or a ciphertext,
    so no serialization of this object — response body, log line, template —
    can leak one regardless of how carelessly it is written.

    Attributes:
        provider: which provider this credential authenticates against.
        masked_key: the permanent rendering, e.g. ``sk-...4f2a``. Always
            strictly shorter than the key it stands for.
        key_version: dimensionless rotation counter; 1 on first configuration.
        created_at: when the provider was first configured, UTC.
        rotated_at: when the key in force was last replaced, UTC; equals
            ``created_at`` until the first rotation.
    """

    provider: Provider
    masked_key: str
    key_version: int
    created_at: dt.datetime
    rotated_at: dt.datetime


def _view(row: LlmProviderCredential) -> ProviderKeyView:
    """Build the maskable-only view of a credential row."""
    return ProviderKeyView(
        provider=Provider(row.provider),
        masked_key=row.masked_display,
        key_version=row.key_version,
        created_at=row.created_at,
        rotated_at=row.rotated_at,
    )


def _validated_key(api_key: str) -> str:
    """Return *api_key* stripped of surrounding whitespace, or refuse it.

    Args:
        api_key: the operator-supplied credential.

    Returns:
        The key with surrounding whitespace removed — a near-universal artifact
        of pasting from a provider console, and removing it is safe because no
        provider's key format is whitespace-significant.

    Raises:
        ValueError: the key is empty, whitespace only, or shorter than
            :data:`_MIN_KEY_CHARS`. The message never quotes the input.
    """
    candidate = api_key.strip()
    if not candidate:
        msg = "api key is empty; a credential must be supplied"
        raise ValueError(msg)
    if len(candidate) < _MIN_KEY_CHARS:
        msg = (
            f"api key is shorter than {_MIN_KEY_CHARS} characters, which no provider issues; "
            "this is almost always a truncated paste. The key itself is not echoed here"
        )
        raise ValueError(msg)
    return candidate


async def _fetch(session: AsyncSession, provider: Provider) -> LlmProviderCredential | None:
    """Return the credential row for *provider* on an open session, or ``None``."""
    statement = sa.select(LlmProviderCredential).where(
        LlmProviderCredential.provider == provider.value
    )
    return (await session.scalars(statement)).one_or_none()


async def _record(
    provider: Provider,
    *,
    masked_key: str | None,
    key_version: int | None,
    actor: str,
    correlation_id: str | None,
) -> None:
    """Append the audit events for one credential change (module docstring: state first).

    Args:
        provider: the provider whose credential changed.
        masked_key: the masked rendering now in force, or ``None`` when the
            credential was deleted. Never a key and never a ciphertext.
        key_version: the rotation counter now in force, or ``None`` on deletion.
        actor: who made the change; a caller assertion, not an authenticated
            identity (:mod:`backend.db.audit`).
        correlation_id: request id, or ``None`` to resolve it from the request
            in flight.
    """
    await record_config_changes(
        AUDIT_SCOPE,
        provider.value,
        {FIELD_MASKED_KEY: masked_key, FIELD_KEY_VERSION: key_version},
        actor=actor,
        correlation_id=correlation_id,
    )


async def add_provider_key(
    provider: Provider,
    api_key: str,
    *,
    actor: str,
    correlation_id: str | None = None,
) -> ProviderKeyView:
    """Store a provider's first API key, encrypted, and record the change.

    The plaintext is encrypted before the row is built and is never written,
    logged, or returned; the caller's copy is the only one that ever existed
    outside this call.

    Args:
        provider: provider to configure.
        api_key: the credential. Surrounding whitespace is removed.
        actor: who is configuring it (recorded verbatim in the audit log; not
            authenticated).
        correlation_id: request id to record, or ``None`` to resolve it from
            the request in flight.

    Returns:
        The masked view of the stored credential.

    Raises:
        ProviderKeyAlreadyConfiguredError: a credential already exists for this
            provider; use :func:`rotate_provider_key`.
        ValueError: the key is empty or implausibly short.
        backend.core.crypto.SecretsCryptoError: ``SECRETS_KEK`` is unset or
            malformed. There is no unencrypted fallback (I5).
    """
    candidate = _validated_key(api_key)
    ciphertext = encrypt_secret(candidate)
    masked = mask_secret(candidate)
    async with ingest_writer_session() as session:
        if await _fetch(session, provider) is not None:
            msg = (
                f"provider {provider.value!r} already has a credential configured; "
                "replacing it is a rotation, which is recorded as such"
            )
            raise ProviderKeyAlreadyConfiguredError(msg)
        row = LlmProviderCredential(
            provider=provider.value,
            ciphertext=ciphertext,
            masked_display=masked,
            key_version=1,
        )
        session.add(row)
        await session.flush()
        await session.refresh(row)
        view = _view(row)
        await session.commit()
    await _record(
        provider,
        masked_key=view.masked_key,
        key_version=view.key_version,
        actor=actor,
        correlation_id=correlation_id,
    )
    _logger.info(
        "provider_key_added", provider=provider.value, key_version=view.key_version, actor=actor
    )
    return view


async def rotate_provider_key(
    provider: Provider,
    api_key: str,
    *,
    actor: str,
    correlation_id: str | None = None,
) -> ProviderKeyView:
    """Replace a provider's stored key with a new one and record the rotation.

    The superseded ciphertext is overwritten, not archived — see the module
    docstring for why keeping it would widen a KEK compromise. The *history* of
    the rotation survives in the audit log, which records the previous and new
    masked renderings and the version counter either side of the change.

    Args:
        provider: provider whose credential is being replaced.
        api_key: the new credential. Surrounding whitespace is removed.
        actor: who is rotating it (recorded verbatim; not authenticated).
        correlation_id: request id to record, or ``None`` to resolve it from
            the request in flight.

    Returns:
        The masked view of the newly stored credential, with ``key_version``
        one higher than before.

    Raises:
        ProviderKeyNotConfiguredError: no credential exists to rotate. Adding
            one is :func:`add_provider_key`; the two are kept distinct so the
            audit log distinguishes them.
        ValueError: the key is empty or implausibly short.
        backend.core.crypto.SecretsCryptoError: ``SECRETS_KEK`` is unset or
            malformed.
    """
    candidate = _validated_key(api_key)
    ciphertext = encrypt_secret(candidate)
    masked = mask_secret(candidate)
    async with ingest_writer_session() as session:
        row = await _fetch(session, provider)
        if row is None:
            raise ProviderKeyNotConfiguredError(_not_configured_message(provider))
        row.ciphertext = ciphertext
        row.masked_display = masked
        row.key_version += 1
        row.rotated_at = dt.datetime.now(dt.UTC)
        await session.flush()
        view = _view(row)
        await session.commit()
    await _record(
        provider,
        masked_key=view.masked_key,
        key_version=view.key_version,
        actor=actor,
        correlation_id=correlation_id,
    )
    _logger.info(
        "provider_key_rotated", provider=provider.value, key_version=view.key_version, actor=actor
    )
    return view


async def delete_provider_key(
    provider: Provider,
    *,
    actor: str,
    correlation_id: str | None = None,
) -> None:
    """Remove a provider's stored credential and record the deletion.

    The row is deleted, so the ciphertext stops existing. The audit log keeps
    the record of the act: a deletion writes events whose new value is JSON
    ``null`` for both the masked rendering and the version counter, with the
    previous masked rendering derived from the log itself.

    Task assignments naming this provider are deliberately left alone: an
    assignment is a statement of intent, and cascading a credential deletion
    into it would silently rewrite the operator's pipeline configuration. Calls
    for that provider refuse until a key is configured again.

    Args:
        provider: provider whose credential is being removed.
        actor: who is removing it (recorded verbatim; not authenticated).
        correlation_id: request id to record, or ``None`` to resolve it from
            the request in flight.

    Raises:
        ProviderKeyNotConfiguredError: there is no credential to delete.
    """
    async with ingest_writer_session() as session:
        row = await _fetch(session, provider)
        if row is None:
            raise ProviderKeyNotConfiguredError(_not_configured_message(provider))
        await session.delete(row)
        await session.commit()
    await _record(
        provider, masked_key=None, key_version=None, actor=actor, correlation_id=correlation_id
    )
    _logger.info("provider_key_deleted", provider=provider.value, actor=actor)


async def get_provider_key(provider: Provider) -> ProviderKeyView:
    """Return the masked view of one provider's stored credential.

    Args:
        provider: provider to read.

    Returns:
        The masked view — never a key, never a ciphertext.

    Raises:
        ProviderKeyNotConfiguredError: no credential is stored for *provider*.
    """
    async with ingest_writer_session() as session:
        row = await _fetch(session, provider)
        if row is None:
            raise ProviderKeyNotConfiguredError(_not_configured_message(provider))
        return _view(row)


async def list_provider_keys() -> tuple[ProviderKeyView, ...]:
    """Return the masked view of every stored credential, ordered by provider name.

    Providers with no credential are simply absent; the caller decides how to
    render "not configured" (the API layer lists every known provider and marks
    the unconfigured ones, so the operator can see what is *missing*).
    """
    statement = sa.select(LlmProviderCredential).order_by(LlmProviderCredential.provider)
    async with ingest_writer_session() as session:
        rows = (await session.scalars(statement)).all()
    return tuple(_view(row) for row in rows)


async def configured_providers() -> frozenset[Provider]:
    """Return the providers that currently have a credential stored."""
    return frozenset(view.provider for view in await list_provider_keys())


async def load_provider_secret(provider: Provider) -> str:
    """Return the decrypted API key for an **outbound provider call** only.

    The single decryption path in the codebase. Its result authenticates a
    request and nothing else: it must never reach a response body, a log line,
    an exception message, or a template. :mod:`backend.api.routes.providers`
    does not import this function, and a test asserts that the module's import
    graph still does not, so the API surface cannot acquire a reveal path by
    accident.

    Args:
        provider: provider whose credential is needed.

    Returns:
        The plaintext key.

    Raises:
        ProviderKeyNotConfiguredError: no credential is stored — the caller
            must refuse rather than proceed unauthenticated (I3).
        backend.core.crypto.SecretDecryptionError: the stored token is not
            authentic under the configured KEK (wrong ``SECRETS_KEK``, or the
            row was tampered with). Never falls back to returning the stored
            value.
        backend.core.crypto.SecretsCryptoError: ``SECRETS_KEK`` is unset or
            malformed.
    """
    async with ingest_writer_session() as session:
        row = await _fetch(session, provider)
        if row is None:
            raise ProviderKeyNotConfiguredError(_not_configured_message(provider))
        ciphertext = row.ciphertext
    # Imported inside the function so that "which modules can decrypt" stays a
    # question about call sites rather than about import lines: a static scan
    # for a reveal path in the API layer looks for this call, and a
    # module-level import here would make the answer less obvious, not more.
    from backend.core.crypto import decrypt_secret

    return decrypt_secret(ciphertext)


def _not_configured_message(provider: Provider) -> str:
    """Return the refusal message used wherever a missing credential is fatal."""
    return (
        f"no API key is configured for provider {provider.value!r}. Configure one in the "
        "provider registry first; there is no keyless or degraded path, and inventing a "
        "result would be fabricated data (I3)"
    )
