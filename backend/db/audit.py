"""Configuration audit log: config changes recorded as events (CC.1, DIRECTIVE §6.11).

§6.11 asks for an audit log of every configuration change — who, when, what
changed, previous value, new value — and states the design rule this module
implements: **config changes are events, not mutations**. There is therefore no
"current configuration" table anywhere. The current value of a configuration
key *is* a derived quantity: the ``new_value`` of the most recent event for
that key (:func:`current_value`). Nothing can change a setting without leaving
the event that changed it, because the event is the only place the setting has
ever lived.

Addressing a configuration key
------------------------------

A key is addressed by three text components, all required and all non-empty:

- ``scope`` — the owning subsystem, e.g. ``"feature_toggle"`` (P5.5),
  ``"extraction_task"`` / ``"prompt"`` (P7.x), ``"settings"`` (CC.4);
- ``target`` — the object within that subsystem, e.g. a feature name, an
  extraction-task name, a scheduler job name;
- ``field`` — the setting on that object, e.g. ``"enabled"``, ``"model"``.

The vocabulary is deliberately open text rather than an enum: every later
consumer would otherwise have to edit this module to add its own scope, which
turns a shared audit primitive into a merge point. What *is* enforced is that
all three are non-empty and carry no leading/trailing whitespace, so
``"enabled"`` and ``" enabled"`` cannot become two keys that look like one.

Values
------

Values are JSON (:data:`JsonValue`) and are validated *before* the write
(:func:`_require_json_value`), for the same reason
:func:`backend.ingest.checkpoint.normalize_checkpoint` validates checkpoints: a
value that does not survive a JSONB round-trip unchanged — a ``datetime``, a
``Decimal``, a ``set``, a NaN — is a silently-changing setting, and a setting
that changes type between reads is indistinguishable from one an operator
changed.

``previous_value`` is **not supplied by the caller**. It is read from the event
history inside the same transaction as the write, under a per-key advisory
lock, so what the log says the value used to be is what the log itself says it
used to be. A caller-supplied "previous" would be an unverified claim in the
one field whose whole purpose is to be checkable against the record.

``previous_value`` is ``NOT NULL`` and holds JSON ``null`` when a key's prior
value was ``None`` **and** on a key's very first event. Those two cases are
told apart by :attr:`ConfigChangeEvent.is_initial`, not by SQL ``NULL``,
because SQL ``NULL`` and JSON ``null`` both decode to Python ``None`` and a
distinction invisible in Python is a distinction that will be got wrong.

When — and why nothing here is ordered by the clock
---------------------------------------------------

``recorded_at`` comes from the database, never from the caller: its server
default is ``now()``, which PostgreSQL evaluates as the **transaction start**
instant. Two consequences, both deliberate:

- every event written by one :func:`record_config_changes` call carries the
  *same* ``recorded_at``. One operator action is one instant, not three
  timestamps microseconds apart that a reader would have to guess were the
  same act.
- across concurrent callers the clock does **not** order events. A
  transaction that began earlier but reached a given key later (it was
  waiting on the key's lock, or changed another field first) records a
  ``recorded_at`` *earlier* than the event it supersedes. The clock can also
  simply tie.

So ``recorded_at`` answers "when did this happen" — which is what §6.11 asks
for — and ``event_id`` answers "what happened after what". Every ordering and
every reconstruction in this module keys on ``event_id``; none keys on the
clock. Sorting an audit view by ``recorded_at`` is a display choice, not a
statement about sequence, and CC.4 should not present it as one.

Why this table is **not** bitemporal
------------------------------------

It carries no :class:`~backend.db.bitemporal.BitemporalMixin`, deliberately.
The bitemporal columns describe when a fact *was true in the world*
(``valid_from``/``valid_to``) and when it *became knowable to the market*
(``knowledge_time``, D-011). A configuration change is not a fact about the
world: it is a thing **we** did to our own system. It has no market
knowability, so any ``knowledge_time`` written here would be a fabricated
number in the one column whose entire meaning is that it is not fabricated
(invariant I3) — the same reasoning that keeps ``ingestion_run`` out of the
bitemporal store (:mod:`backend.ingest.runs`).

Nor does it need the bitemporal machinery: bitemporality exists to model
*corrections to beliefs about the past*, and this log has none. A configuration
event is never corrected; a later change is a later event. "What was the
configuration on 2026-03-01?" is answered by replaying events up to that
instant, which is what the event stream already is. Consequently the table is
not in the bitemporal registry, the Core-level read guard does not scope it,
and it is read without an ``as_of``.

Append-only, **not immutable** — read this before repeating §6.11's word
---------------------------------------------------------------------------

UPDATE and DELETE on this table are rejected by a ``BEFORE UPDATE OR DELETE``
row trigger installed by migration 0007, the same mechanism migrations
0003/0004 use on the fact tables. That trigger is real enforcement and it is
role-independent: no session, ORM or raw, edits a past event.

It is **not** immutability, and this module does not claim it is. Per D-012 the
compose stack runs a single database role which owns this table and its
trigger, so that role can ``ALTER TABLE ... DISABLE TRIGGER``, ``DROP TRIGGER``
or ``TRUNCATE`` the log. The application runs as that role. Until CC.9
(database role separation) lands, "immutable audit log" — the phrase §6.11 uses
— would be a **false claim** about this table, and D-017 makes role separation
an explicit prerequisite for presenting this log to the operator as
trustworthy. Nothing built on this module (CC.4's audit browser above all) may
describe the log as tamper-proof before CC.9 ships. The honest present-tense
statement is: *append-only, enforced by a database trigger that the owning role
can remove.*

Actor
-----

``actor`` is required and recorded verbatim. The platform has no
authentication yet (§1.1: single operator, local deployment), so an actor is a
**caller assertion, not a verified identity**. It is recorded anyway — an
unverified actor is more useful than no actor, and the column has to exist
before the identity does — but no consumer may present it as authenticated
until there is an authentication system to authenticate it.

Correlation id
--------------

``correlation_id`` ties a change to the request that made it. It defaults to
the ``request_id`` bound by :class:`backend.api.middleware.CorrelationIdMiddleware`
(D-003) for the request in flight, and is ``NULL`` when a change is made
outside a request (startup, a task, a shell). It is left ``NULL`` rather than
invented in that case.

Sessions
--------

Writes and reads use :func:`backend.db.asof.ingest_writer_session`, the
process's sanctioned append-only write session. Its name is historical — it was
introduced for the ingestion writer — but what it provides is exactly what this
module needs: a session that may INSERT and that cannot read bitemporal fact
tables unversioned. Adding a second sessionmaker to
:mod:`backend.db.engine` would buy nothing semantic.

Each recording call opens and commits **its own** transaction, and no public
function accepts a caller's session. A consumer therefore cannot make a
configuration change atomic with other database work. That is a deliberate
limitation and it costs nothing today: the event *is* the setting, so there is
no second write for it to be atomic with. It stops being free the day some
consumer needs a config change and a non-config row to land together — at
which point the honest change is a session-accepting variant here, not a
caller reaching around this module to INSERT its own event.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from typing import TYPE_CHECKING, Final

import sqlalchemy as sa
import structlog
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

# Sibling module inside backend/db's own trust boundary: ``backend.db`` re-exports
# this same function as the sanctioned append-only write path, and importing the
# package from one of its own submodules would only add an import cycle risk.
from backend.db.asof import ingest_writer_session
from backend.db.base import Base

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "SYSTEM_ACTOR",
    "ConfigChangeEvent",
    "ConfigNotSetError",
    "JsonValue",
    "config_history",
    "current_scope_values",
    "current_value",
    "current_values",
    "record_config_change",
    "record_config_changes",
]

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None
"""A configuration value: anything that survives a JSONB round-trip unchanged."""

SYSTEM_ACTOR: Final = "system"
"""Actor recorded for changes the platform makes to itself, with no human behind them.

A named constant so "the system did it" is one spelling across every consumer,
and so it is greppable when authentication finally distinguishes real actors
from this one.
"""

_MAX_JSON_DEPTH: Final = 16
"""Container nesting accepted in a configuration value.

Bounds recursion and, incidentally, self-referential structures: a list
containing itself is rejected at the limit instead of exhausting the stack.
Configuration values that need seventeen levels of nesting are not
configuration values.
"""

_DEFAULT_HISTORY_LIMIT: Final = 100
"""Rows :func:`config_history` returns when the caller does not say."""

_LOCK_KEY_BYTES: Final = 8
"""Digest width of the advisory-lock key: 8 bytes == one PG ``bigint``."""


class ConfigNotSetError(LookupError):
    """No configuration event has ever been recorded for the requested key.

    Distinct from "the key's value is ``None``", which is a recorded event with
    ``new_value`` JSON ``null``. Raised by :func:`current_value`; callers that
    want a fallback should use :func:`current_values` and ``.get(field,
    default)``, which keeps the fallback visible at the call site instead of
    hidden in this module.
    """


class ConfigChangeEvent(Base):
    """One recorded configuration change: who changed what, from what, to what.

    Append-only (migration 0007's trigger) and **not bitemporal** — see the
    module docstring for both, including why "immutable" is not yet a true
    statement about this table (D-012, D-017).

    All timestamps are ``TIMESTAMPTZ`` in UTC. ``event_id`` is a dimensionless
    database-generated surrogate whose ascending order is the recording order
    of events *for a given key* (writes to one key are serialized by an
    advisory lock, so a later event for that key always has a larger id);
    across unrelated keys it is an insertion order, not a clock.
    """

    __tablename__ = "config_change_event"
    __table_args__ = (
        # CHECK names are given unprefixed: the metadata naming convention
        # (ck_%(table_name)s_%(constraint_name)s) expands them, so an
        # already-prefixed name would double the prefix and diverge from the
        # name migration 0007 creates.
        sa.CheckConstraint("actor <> ''", name="actor_not_empty"),
        sa.CheckConstraint("scope <> ''", name="scope_not_empty"),
        sa.CheckConstraint("target <> ''", name="target_not_empty"),
        sa.CheckConstraint("field <> ''", name="field_not_empty"),
        # An initial event has, by definition, no previous value; JSON null is
        # the encoding for that (module docstring). Without this the two
        # columns could disagree about whether a key existed before.
        sa.CheckConstraint(
            "NOT is_initial OR previous_value = 'null'::jsonb",
            name="initial_event_has_no_previous_value",
        ),
        # Exactly the DISTINCT ON / ORDER BY shape of the reconstruction query
        # in _latest_per_key: the current value of a key is a single index seek.
        sa.Index(
            "ix_config_change_event_key",
            "scope",
            "target",
            "field",
            sa.text("event_id DESC"),
        ),
        # CC.4's audit browser reads newest-first across all scopes.
        sa.Index("ix_config_change_event_recorded_at", sa.text("recorded_at DESC")),
        # "Which config changes did this request make?" Partial, because changes
        # made outside a request carry no correlation id and are never the answer.
        sa.Index(
            "ix_config_change_event_correlation_id",
            "correlation_id",
            postgresql_where=sa.text("correlation_id IS NOT NULL"),
        ),
    )

    event_id: Mapped[int] = mapped_column(
        sa.BigInteger,
        sa.Identity(),
        primary_key=True,
        doc="Surrogate event key, database-generated. Dimensionless; ascending per key.",
    )
    recorded_at: Mapped[dt.datetime] = mapped_column(
        sa.TIMESTAMP(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
        doc=(
            "When the change was recorded, UTC, from the database clock at transaction "
            "start (never the caller's). Not an ordering key — see the module docstring."
        ),
    )
    actor: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="Who made the change, as asserted by the caller (not authenticated — see module doc).",
    )
    scope: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="Owning subsystem of the key, e.g. 'feature_toggle', 'extraction_task', 'settings'.",
    )
    target: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="Configured object within the scope, e.g. a feature name or scheduler job name.",
    )
    field: Mapped[str] = mapped_column(
        sa.Text,
        nullable=False,
        doc="Setting on the target that changed, e.g. 'enabled', 'model', 'prompt_version'.",
    )
    previous_value: Mapped[JsonValue] = mapped_column(
        JSONB,
        nullable=False,
        doc="Value before this event; JSON null when it was None or when is_initial is true.",
    )
    new_value: Mapped[JsonValue] = mapped_column(
        JSONB,
        nullable=False,
        doc="Value after this event; JSON null when the value was set to None.",
    )
    is_initial: Mapped[bool] = mapped_column(
        sa.Boolean,
        nullable=False,
        doc="True when this is the key's first event, so previous_value means 'nothing', not None.",
    )
    correlation_id: Mapped[str | None] = mapped_column(
        sa.Text,
        nullable=True,
        doc="Request id (D-003) this change was made under; NULL outside a request.",
    )


def _require_identifier(value: str, label: str) -> str:
    """Return one address component, or raise if it is unusable as a key part.

    Args:
        value: the candidate ``scope``/``target``/``field``/``actor`` string.
        label: the component's name, used in the error message.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: if ``value`` is not a ``str``.
        ValueError: if ``value`` is empty or carries leading/trailing
            whitespace. Whitespace is rejected rather than stripped because
            silently stripping would make ``"enabled"`` and ``" enabled"`` the
            same key at write time and different keys in any log line, dashboard
            or SQL query a human writes about them.
    """
    candidate: object = value
    if not isinstance(candidate, str):
        msg = f"{label} must be a str; got {type(candidate).__name__}"
        raise TypeError(msg)
    if not candidate:
        msg = f"{label} must be a non-empty string"
        raise ValueError(msg)
    if candidate != candidate.strip():
        msg = (
            f"{label} must not have leading or trailing whitespace; got {candidate!r}. "
            "Whitespace is rejected rather than trimmed so two spellings of one key "
            "cannot both be written"
        )
        raise ValueError(msg)
    return candidate


def _require_json_value(value: JsonValue, label: str, depth: int = 0) -> JsonValue:
    """Return a configuration value, or raise if it would not round-trip through JSONB.

    Args:
        value: the candidate value.
        label: a path-ish description of where the value sits, for the error
            message (e.g. ``"new_value['limits'][0]"``).
        depth: current nesting depth; callers pass the default.

    Returns:
        ``value`` unchanged.

    Raises:
        TypeError: if ``value`` (or anything nested in it) is not a JSON type,
            or if a mapping key is not a ``str``. ``datetime``, ``Decimal``,
            ``set`` and ORM objects are all rejected here: they either fail to
            serialize or come back as a different type than they went in, and a
            configuration value that changes type between reads is
            indistinguishable from one somebody changed.
        ValueError: if a float is NaN or infinite (JSON has no spelling for
            either, and PostgreSQL rejects them in ``jsonb``), or if nesting
            exceeds :data:`_MAX_JSON_DEPTH` — which also catches self-referential
            containers before they exhaust the stack.
    """
    if depth > _MAX_JSON_DEPTH:
        msg = (
            f"{label} nests deeper than {_MAX_JSON_DEPTH} levels; refusing to store it "
            "(a self-referential container looks exactly like this)"
        )
        raise ValueError(msg)
    candidate: object = value
    if candidate is None or isinstance(candidate, bool | str | int):
        # bool before int deliberately: bool is a subclass of int, and both are
        # valid JSON, so the order only matters for the error message never
        # reached here.
        return value
    if isinstance(candidate, float):
        if not math.isfinite(candidate):
            msg = (
                f"{label} is {candidate!r}, which JSON cannot express and PostgreSQL "
                "rejects in jsonb; store a string or null instead"
            )
            raise ValueError(msg)
        return value
    if isinstance(candidate, list):
        for index, item in enumerate(candidate):
            _require_json_value(item, f"{label}[{index}]", depth + 1)
        return value
    if isinstance(candidate, dict):
        for key, item in candidate.items():
            if not isinstance(key, str):
                msg = (
                    f"{label} has a non-str key {key!r} ({type(key).__name__}); JSON object "
                    "keys are strings, and a non-str key would come back as one"
                )
                raise TypeError(msg)
            _require_json_value(item, f"{label}[{key!r}]", depth + 1)
        return value
    msg = (
        f"{label} is a {type(candidate).__name__}, which is not a JSON value "
        "(str, int, float, bool, None, list, dict). Serialize it explicitly — a value "
        "that changes type across a JSONB round-trip is a silently-changing setting"
    )
    raise TypeError(msg)


def _current_correlation_id() -> str | None:
    """Return the request id bound to this context, or ``None`` outside a request.

    Reads the ``request_id`` bound by
    :class:`backend.api.middleware.CorrelationIdMiddleware` through
    ``structlog.contextvars`` — the same value every log line for the request
    carries (D-003), which is what makes a config event joinable to the request
    that produced it. Returns ``None`` rather than inventing an id when nothing
    is bound (startup, Celery task, shell) or when the bound value is not a
    non-empty string.
    """
    bound = structlog.contextvars.get_contextvars().get("request_id")
    if isinstance(bound, str) and bound:
        return bound
    return None


def _advisory_lock_key(scope: str, target: str, field: str) -> int:
    """Return the PostgreSQL advisory-lock key for one configuration key.

    Args:
        scope: owning subsystem.
        target: configured object.
        field: setting name.

    Returns:
        A signed 64-bit integer suitable for ``pg_advisory_xact_lock(bigint)``,
        derived from a BLAKE2b digest of the three components joined by a NUL
        byte (which cannot occur inside them, so no two distinct keys collide by
        concatenation). Hashed in Python rather than with PostgreSQL's
        ``hashtext`` so the mapping is deterministic, documented and testable
        instead of resting on an internal server function.
    """
    material = "\x00".join((scope, target, field)).encode()
    digest = hashlib.blake2b(material, digest_size=_LOCK_KEY_BYTES).digest()
    return int.from_bytes(digest, "big", signed=True)


def _latest_per_key(
    *,
    scope: str | None = None,
    target: str | None = None,
    field: str | None = None,
) -> Select[tuple[ConfigChangeEvent]]:
    """Build the "most recent event per configuration key" select.

    ``DISTINCT ON (scope, target, field) ... ORDER BY scope, target, field,
    event_id DESC`` — the current value of every matched key in one pass,
    matching ``ix_config_change_event_key`` exactly.

    The tiebreaker is ``event_id``, not ``recorded_at``. ``recorded_at`` is a
    transaction-start reading: two events can share one, and a transaction that
    started earlier but reached this key later carries one that is *older* than
    the event it supersedes (module docstring). Identity does neither. Per key
    that ordering is also the true recording order, because writes to one key
    are serialized by an advisory lock (:func:`_advisory_lock_key`), so the
    largest ``event_id`` is the value in force.

    Args:
        scope: restrict to one subsystem, or ``None`` for all.
        target: restrict to one configured object, or ``None`` for all.
        field: restrict to one setting, or ``None`` for all.

    Returns:
        The ORM select; the caller executes it on a session.
    """
    statement = (
        sa.select(ConfigChangeEvent)
        .distinct(ConfigChangeEvent.scope, ConfigChangeEvent.target, ConfigChangeEvent.field)
        .order_by(
            ConfigChangeEvent.scope,
            ConfigChangeEvent.target,
            ConfigChangeEvent.field,
            ConfigChangeEvent.event_id.desc(),
        )
    )
    if scope is not None:
        statement = statement.where(ConfigChangeEvent.scope == scope)
    if target is not None:
        statement = statement.where(ConfigChangeEvent.target == target)
    if field is not None:
        statement = statement.where(ConfigChangeEvent.field == field)
    return statement


async def _record(
    session: AsyncSession,
    *,
    scope: str,
    target: str,
    field: str,
    new_value: JsonValue,
    actor: str,
    correlation_id: str | None,
) -> ConfigChangeEvent:
    """Append one event for one key on an open session, deriving its previous value.

    Takes a transaction-scoped advisory lock on the key first, so the
    read-then-append below cannot interleave with a concurrent change to the
    same key and record a ``previous_value`` that was never the value. The lock
    is released by commit or rollback, whichever happens.

    Args:
        session: an open writer session; the caller commits.
        scope: owning subsystem (already validated).
        target: configured object (already validated).
        field: setting name (already validated).
        new_value: the value after this change (already validated).
        actor: who made the change (already validated).
        correlation_id: request id, or ``None``.

    Returns:
        The pending :class:`ConfigChangeEvent`, flushed and refreshed so its
        database-generated ``event_id`` and ``recorded_at`` are populated.
    """
    lock_key = sa.literal(_advisory_lock_key(scope, target, field), sa.BigInteger)
    await session.execute(sa.select(sa.func.pg_advisory_xact_lock(lock_key)))
    previous = (
        await session.scalars(_latest_per_key(scope=scope, target=target, field=field))
    ).one_or_none()
    event = ConfigChangeEvent(
        actor=actor,
        scope=scope,
        target=target,
        field=field,
        previous_value=None if previous is None else previous.new_value,
        new_value=new_value,
        is_initial=previous is None,
        correlation_id=correlation_id,
    )
    session.add(event)
    await session.flush()
    # Explicit refresh so recorded_at (a server default) is loaded before the
    # object detaches, rather than relying on the mapper's eager-default
    # heuristics. An ORM column load, so the read guard exempts it by design.
    await session.refresh(event)
    return event


async def record_config_change(
    scope: str,
    target: str,
    field: str,
    new_value: JsonValue,
    *,
    actor: str,
    correlation_id: str | None = None,
) -> ConfigChangeEvent:
    """Record a configuration change as an event and return it.

    This is the only way configuration changes in this system: there is no
    mutable row to update, so a change that is not recorded here did not happen
    (§6.11). ``previous_value`` is derived from the log, not accepted from the
    caller, under a per-key lock.

    A no-op set — writing the value a key already has — **is** recorded. The log
    answers "what did the operator do", and setting something to the value it
    already had is a thing the operator did; suppressing it would mean the log
    disagreed with the operator's own memory of having pressed the button.

    Args:
        scope: owning subsystem, e.g. ``"feature_toggle"``.
        target: configured object within the scope, e.g. a feature name.
        field: setting on the target, e.g. ``"enabled"``.
        new_value: the value being set. Must be JSON (:data:`JsonValue`).
        actor: who is making the change. Required, and recorded verbatim; it is
            a caller assertion, not an authenticated identity (module
            docstring). Use :data:`SYSTEM_ACTOR` for changes with no human
            behind them.
        correlation_id: request id to record. Defaults to the ``request_id``
            bound for the request in flight, and to ``NULL`` when there is no
            request — never to an invented value.

    Returns:
        The recorded :class:`ConfigChangeEvent`, detached and fully loaded.

    Raises:
        TypeError: if any address component is not a ``str``, or if
            ``new_value`` is not JSON.
        ValueError: if any address component is empty or padded with
            whitespace, or if ``new_value`` contains a non-finite float or
            nests too deeply.
    """
    events = await record_config_changes(
        scope,
        target,
        {field: new_value},
        actor=actor,
        correlation_id=correlation_id,
    )
    return events[0]


async def record_config_changes(
    scope: str,
    target: str,
    values: Mapping[str, JsonValue],
    *,
    actor: str,
    correlation_id: str | None = None,
) -> tuple[ConfigChangeEvent, ...]:
    """Record several field changes on one target as events, in one transaction.

    The batch form for a consumer that changes a whole settings form or a whole
    extraction-task assignment at once: either every field's event lands or none
    does, so the log never shows half of a change the operator made as one
    action. One event per field — the unit of an audit entry is a field, because
    "previous value / new value" is only meaningful per field.

    Keys are locked in a deterministic order (by advisory-lock key), so two
    concurrent batches overlapping on the same target cannot deadlock against
    each other.

    Args:
        scope: owning subsystem.
        target: configured object within the scope.
        values: ``{field: new_value}``; must be non-empty.
        actor: who is making the change (see :func:`record_config_change`).
        correlation_id: request id to record, or ``None`` to resolve it from
            the request in flight.

    Returns:
        The recorded events in the order they were written — the deterministic
        lock-acquisition order described above, not the caller's mapping order
        — detached and fully loaded.

    Raises:
        TypeError: if any address component is not a ``str``, or if any value
            is not JSON.
        ValueError: if ``values`` is empty, or if any address component or
            value is invalid (see :func:`record_config_change`).
    """
    checked_scope = _require_identifier(scope, "scope")
    checked_target = _require_identifier(target, "target")
    checked_actor = _require_identifier(actor, "actor")
    if not values:
        msg = "values must contain at least one field; an empty change is not a change"
        raise ValueError(msg)
    checked: list[tuple[str, JsonValue]] = []
    for field, value in values.items():
        checked_field = _require_identifier(field, "field")
        checked.append((checked_field, _require_json_value(value, f"values[{checked_field!r}]")))
    # Sorted by lock key so any two batches acquire shared keys in the same
    # order; the field name breaks ties (impossible for distinct fields, but
    # cheap to make the order total rather than merely usually total).
    checked.sort(
        key=lambda item: (_advisory_lock_key(checked_scope, checked_target, item[0]), item[0])
    )
    resolved_correlation_id = (
        correlation_id if correlation_id is not None else _current_correlation_id()
    )
    recorded: list[ConfigChangeEvent] = []
    async with ingest_writer_session() as session:
        for field, value in checked:
            recorded.append(
                await _record(
                    session,
                    scope=checked_scope,
                    target=checked_target,
                    field=field,
                    new_value=value,
                    actor=checked_actor,
                    correlation_id=resolved_correlation_id,
                )
            )
        await session.commit()
    return tuple(recorded)


async def current_value(scope: str, target: str, field: str) -> JsonValue:
    """Return a configuration key's current value, reconstructed from its events.

    The value is *derived*, never stored: it is the ``new_value`` of the most
    recent event for the key. Nothing else is consulted, so a value that no
    event produced cannot be returned.

    Args:
        scope: owning subsystem.
        target: configured object within the scope.
        field: setting on the target.

    Returns:
        The current value, which may legitimately be ``None`` if the key was
        explicitly set to null.

    Raises:
        ConfigNotSetError: if the key has no events at all. This is *not* the
            same as a value of ``None``; callers wanting a default should use
            :func:`current_values` and ``.get(field, default)`` so the default
            is visible where it is chosen.
        TypeError: if any address component is not a ``str``.
        ValueError: if any address component is empty or padded with whitespace.
    """
    checked_scope = _require_identifier(scope, "scope")
    checked_target = _require_identifier(target, "target")
    checked_field = _require_identifier(field, "field")
    async with ingest_writer_session() as session:
        event = (
            await session.scalars(
                _latest_per_key(scope=checked_scope, target=checked_target, field=checked_field)
            )
        ).one_or_none()
    if event is None:
        msg = (
            f"no configuration event has ever been recorded for "
            f"({checked_scope!r}, {checked_target!r}, {checked_field!r}); the key has no "
            "value, which is different from a value of None"
        )
        raise ConfigNotSetError(msg)
    return event.new_value


async def current_values(scope: str, target: str) -> dict[str, JsonValue]:
    """Return every currently-set field of one configured object.

    Args:
        scope: owning subsystem.
        target: configured object within the scope.

    Returns:
        ``{field: current value}`` for every field with at least one event;
        empty when the target has never been configured. Fields absent from the
        mapping have never been set, which is how a caller applies its own
        default (``result.get("enabled", True)``).

    Raises:
        TypeError: if ``scope`` or ``target`` is not a ``str``.
        ValueError: if ``scope`` or ``target`` is empty or padded with whitespace.
    """
    checked_scope = _require_identifier(scope, "scope")
    checked_target = _require_identifier(target, "target")
    async with ingest_writer_session() as session:
        events = (
            await session.scalars(_latest_per_key(scope=checked_scope, target=checked_target))
        ).all()
    return {event.field: event.new_value for event in events}


async def current_scope_values(scope: str) -> dict[str, dict[str, JsonValue]]:
    """Return the current configuration of every object in one subsystem.

    The shape a catalog page wants: P5.5 renders every feature's toggles, CC.4
    every settings group, in one round trip rather than one per target.

    Args:
        scope: owning subsystem.

    Returns:
        ``{target: {field: current value}}``; empty when nothing in the scope
        has ever been configured.

    Raises:
        TypeError: if ``scope`` is not a ``str``.
        ValueError: if ``scope`` is empty or padded with whitespace.
    """
    checked_scope = _require_identifier(scope, "scope")
    async with ingest_writer_session() as session:
        events = (await session.scalars(_latest_per_key(scope=checked_scope))).all()
    grouped: dict[str, dict[str, JsonValue]] = {}
    for event in events:
        grouped.setdefault(event.target, {})[event.field] = event.new_value
    return grouped


async def config_history(
    *,
    scope: str | None = None,
    target: str | None = None,
    field: str | None = None,
    correlation_id: str | None = None,
    before_event_id: int | None = None,
    limit: int = _DEFAULT_HISTORY_LIMIT,
) -> tuple[ConfigChangeEvent, ...]:
    """Return recorded configuration events, newest first.

    The read side of the audit trail (CC.4's browser). Filters are ANDed and
    each is optional, so this answers both "everything that ever happened to
    this one toggle" and "every configuration change request ``X`` made".

    Ordering is by ``event_id`` descending rather than ``recorded_at``: the
    transaction clock can tie, and under concurrency can even run backwards
    relative to the sequence of events (module docstring), while the identity
    can do neither. A stable newest-first order needs the id.

    Paging is by **keyset**, not ``OFFSET``: pass the last id of the previous
    page as ``before_event_id``. Offset paging over a table that is being
    appended to shifts rows between pages, so a reader scrolling an audit log
    while a change lands would see one entry twice and never see another —
    which in an audit trail reads as tampering. The cursor is also a bound on
    the trailing column of the index each filter already uses, rather than rows
    the server must count through and discard as ``OFFSET`` grows.

    Args:
        scope: restrict to one subsystem, or ``None`` for all.
        target: restrict to one configured object, or ``None`` for all.
        field: restrict to one setting, or ``None`` for all.
        correlation_id: restrict to the changes made under one request id, or
            ``None`` for all.
        before_event_id: return only events older than this id (exclusive), or
            ``None`` to start at the newest. The id need not exist; the cursor
            is a bound, not a lookup.
        limit: maximum events to return (count).

    Returns:
        Up to ``limit`` events, newest first, detached and fully loaded. Fewer
        than ``limit`` means the last page; the caller's next cursor is the
        ``event_id`` of the final element.

    Raises:
        ValueError: if ``limit`` is less than 1.
    """
    if limit < 1:
        msg = f"limit must be >= 1; got {limit}"
        raise ValueError(msg)
    statement = (
        sa.select(ConfigChangeEvent).order_by(ConfigChangeEvent.event_id.desc()).limit(limit)
    )
    if before_event_id is not None:
        statement = statement.where(ConfigChangeEvent.event_id < before_event_id)
    if scope is not None:
        statement = statement.where(ConfigChangeEvent.scope == scope)
    if target is not None:
        statement = statement.where(ConfigChangeEvent.target == target)
    if field is not None:
        statement = statement.where(ConfigChangeEvent.field == field)
    if correlation_id is not None:
        statement = statement.where(ConfigChangeEvent.correlation_id == correlation_id)
    async with ingest_writer_session() as session:
        events: Sequence[ConfigChangeEvent] = (await session.scalars(statement)).all()
    return tuple(events)
