"""Append-only version writing: closing intervals, and retracting facts.

Two things a connector cannot do by hand in an append-only store, both of
which produce a silent data defect when improvised: closing an open-ended
interval (:func:`close_open_interval`, :func:`supersede_open_interval`) and
withdrawing a fact it no longer believes (:func:`retract_fact`).

The problem
-----------

A bitemporal fact may be **open-ended**: ``valid_to = 'infinity'`` means "this
holds from ``valid_from`` until something supersedes it". A securities-master
identity is the canonical case — a ticker applies from the day it was assigned
until the day it changes.

The store is append-only: migration 0003's ``BEFORE UPDATE OR DELETE`` triggers
refuse to modify a written row, on every role and every chunk (D-012). So when
the ticker *does* change, the open interval **cannot be closed by updating
it**. There is no UPDATE available, and none should be: rewriting a row would
destroy the record of what we believed before.

The failure this prevents
-------------------------

Until this helper existed, the obvious thing for a connector to do — insert the
new identity as another open-ended row — produced **overlapping intervals**.
Row A says ``[2020-01-01, ∞)`` ticker ``ABC``; row B says ``[2024-06-01, ∞)``
ticker ``XYZ``. Both carry distinct ``valid_from`` values, so both are distinct
logical facts under the D-011 read semantics (latest knowledge wins *per*
(key, ``valid_from``)), and both are visible. Any point-in-time query at event
time 2024-07-01 gets **two** identities for one security. Downstream that is
either a silent duplicate row in a universe, or a join fan-out that
double-counts a position — and nothing raises, because every individual row is
valid.

The mechanism
-------------

The only append-only way to close an open interval is a **later-knowledge
correction row**: same logical key, **same** ``valid_from``, a strictly later
``knowledge_time``, and a bounded ``valid_to``. Under D-011's read semantics
the correction wins for every ``as_of`` at or after its ``knowledge_time``, so
the interval is bounded from then on; and for any earlier ``as_of`` the
original open row still wins, which is exactly right — before we learned about
the ticker change, "open-ended" was what we honestly believed.

:func:`supersede_open_interval` writes that correction *and* the successor
version in one step, because doing only half of it is the bug above. Use it
whenever a connector learns that an open-ended fact has ended.

Retractions
-----------

A **retraction** is the other half of the same idea: a later-knowledge version
saying "we no longer believe this fact", written when a source withdraws a
statement rather than restates it. Under D-011 it wins the latest-knowledge
read from its ``knowledge_time`` on and hides the fact.

It carries **no payload**. Every payload column of a retraction row is NULL,
and the database refuses any other shape
(``ck_<table>_retraction_payload_absent``, P2.10): a retraction that filled
``close_usd`` with a number would be fabricated data sitting in a fact table,
indistinguishable at the storage layer from a real quote (I3, directive §9.1).
:func:`retract_fact` is the sanctioned constructor and cannot produce one —
it copies the logical key and the valid interval, sets ``is_retraction``, and
sets every payload column to SQL ``NULL`` explicitly.

All three helpers return **new, unsaved ORM instances**; nothing here touches
the database. The caller adds them in a single writer-session transaction so
a close and its successor land atomically.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import inspect
from sqlalchemy.sql import null

from backend.db.bitemporal import INFINITY, BitemporalMixin
from backend.ingest.errors import SupersessionError
from backend.ingest.write import validate_knowledge_time

if TYPE_CHECKING:
    from sqlalchemy.orm import Mapper

__all__ = ["close_open_interval", "retract_fact", "supersede_open_interval"]

_DERIVED_COLUMNS = frozenset({"valid_from", "valid_to", "knowledge_time", "ingested_at"})
"""Columns the helpers set themselves; never copied verbatim from the source row.

``ingested_at`` is excluded because it is an audit column recording when *this*
row was written — copying the original's value would misreport the new row's
provenance. The other three are the temporal coordinates being changed.
"""


def _payload_values(row: BitemporalMixin) -> dict[str, Any]:
    """Return every mapped column value of ``row`` except the derived temporal ones."""
    mapper = cast("Mapper[Any]", inspect(type(row)))
    return {
        attribute.key: getattr(row, attribute.key)
        for attribute in mapper.column_attrs
        if attribute.key not in _DERIVED_COLUMNS
    }


def _require_aware(value: dt.datetime, name: str) -> dt.datetime:
    """Return ``value`` if timezone-aware, else raise :class:`TypeError`."""
    if value.tzinfo is None or value.utcoffset() is None:
        msg = (
            f"{name} must be a timezone-aware UTC datetime; got naive {value!r} "
            "(asyncpg reinterprets naive datetimes in host-local time; D-012)"
        )
        raise TypeError(msg)
    return value


def close_open_interval[RowT: BitemporalMixin](
    open_row: RowT,
    *,
    valid_to: dt.datetime,
    knowledge_time: dt.datetime,
    now: dt.datetime | None = None,
) -> RowT:
    """Build the correction row that closes an open-ended fact.

    The returned row repeats ``open_row``'s logical key, payload columns and
    ``valid_from`` exactly, and differs only in carrying a bounded ``valid_to``
    and a later ``knowledge_time``. It is a *new version of the same fact*, not
    a new fact — which is why ``valid_from`` must not move.

    Args:
        open_row: the open-ended row to close (``valid_to`` is
            :data:`~backend.db.bitemporal.INFINITY`). Not modified.
        valid_to: the event-time instant the fact stopped applying,
            timezone-aware UTC. Half-open semantics: the fact holds up to but
            not including this instant.
        knowledge_time: when we learned the fact had ended, timezone-aware
            UTC. Must be strictly later than ``open_row.knowledge_time``.
        now: reference instant for the future-knowledge-time check; defaults
            to the current UTC time. Injected by tests.

    Returns:
        A new unsaved instance of ``type(open_row)``. The caller writes it
        through the ingestion writer session.

    Raises:
        SupersessionError: if ``open_row`` is not open-ended (already bounded
            — closing it twice would create two conflicting versions of one
            belief), if it is a retraction (a retracted fact has nothing to
            close; re-assert it first), if ``valid_to`` does not lie strictly
            inside ``(valid_from, infinity)``, or if ``knowledge_time`` is not
            strictly later than the row being corrected (equal knowledge times
            collide on the primary key, and earlier ones would be invisible to
            every ``as_of`` that already sees the open row).
        TypeError: if either datetime argument is naive.
        FutureKnowledgeTimeError: if ``knowledge_time`` is in the future.
    """
    _require_aware(valid_to, "valid_to")
    _require_aware(knowledge_time, "knowledge_time")
    validate_knowledge_time(
        knowledge_time, context=f"close_open_interval({type(open_row).__name__})", now=now
    )
    if open_row.valid_to != INFINITY:
        msg = (
            f"{type(open_row).__name__} is not open-ended (valid_to="
            f"{open_row.valid_to!r}); only an interval ending at infinity can be closed"
        )
        raise SupersessionError(msg)
    if open_row.is_retraction:
        msg = (
            f"{type(open_row).__name__} version being closed is a retraction; a "
            "retracted fact is already invisible and has no interval to bound. "
            "Re-assert the fact with a later knowledge_time first"
        )
        raise SupersessionError(msg)
    if valid_to <= open_row.valid_from:
        msg = (
            f"valid_to {valid_to.isoformat()} must be strictly after valid_from "
            f"{open_row.valid_from.isoformat()}: a closed interval must be non-empty"
        )
        raise SupersessionError(msg)
    if valid_to >= INFINITY:
        msg = "valid_to must be a bounded instant; closing an interval at infinity is a no-op"
        raise SupersessionError(msg)
    if knowledge_time <= open_row.knowledge_time:
        msg = (
            f"knowledge_time {knowledge_time.isoformat()} must be strictly later than the "
            f"corrected row's {open_row.knowledge_time.isoformat()}: an equal value collides "
            "on the primary key, an earlier one would never win the latest-knowledge read"
        )
        raise SupersessionError(msg)
    values = _payload_values(open_row)
    values.update(
        valid_from=open_row.valid_from,
        valid_to=valid_to,
        knowledge_time=knowledge_time,
    )
    return type(open_row)(**values)


def retract_fact[RowT: BitemporalMixin](
    row: RowT,
    *,
    knowledge_time: dt.datetime,
    now: dt.datetime | None = None,
) -> RowT:
    """Build the retraction version that withdraws a fact we no longer believe.

    The returned row repeats ``row``'s logical key and its event-time interval
    exactly — that is how it addresses the fact — carries ``is_retraction =
    True`` and a strictly later ``knowledge_time``, and holds **no payload**:
    every payload column is set to SQL ``NULL``.

    The payload is null rather than copied, zeroed, or left to a default
    because a retraction states nothing about value. A number in those columns
    would be fabricated data inside a fact table, unable to be told apart from
    an observation at the storage layer (I3, directive §9.1); a database CHECK
    refuses it on every write path, and this constructor cannot express it.
    SQL ``NULL`` is used explicitly rather than Python ``None`` so a column
    with a default (``macro_observation.is_missing`` defaults to ``false``)
    cannot fill itself in when the row is inserted.

    Under D-011's read semantics the retraction wins for every ``as_of`` at or
    after ``knowledge_time`` and the fact is invisible from then on; earlier
    as-of queries still see the fact, which is exactly right — before the
    source withdrew it, we believed it. To re-assert the fact later, write a
    normal version with a still-later ``knowledge_time``.

    Args:
        row: the version being withdrawn — typically the currently winning
            one. Not modified; only its logical key, ``valid_from``,
            ``valid_to`` and ``knowledge_time`` are read.
        knowledge_time: when we learned the fact was withdrawn, timezone-aware
            UTC. Must be strictly later than ``row.knowledge_time``.
        now: reference instant for the future-knowledge-time check; defaults
            to the current UTC time. Injected by tests.

    Returns:
        A new unsaved instance of ``type(row)``. The caller writes it through
        the ingestion writer session.

    Raises:
        SupersessionError: if ``row`` is itself a retraction (the fact is
            already withdrawn; retracting it twice would record a belief
            change that did not happen), or if ``knowledge_time`` is not
            strictly later than ``row``'s (an equal value collides on the
            primary key, an earlier one would never win the latest-knowledge
            read).
        TypeError: if ``knowledge_time`` is naive.
        FutureKnowledgeTimeError: if ``knowledge_time`` is in the future.
    """
    _require_aware(knowledge_time, "knowledge_time")
    validate_knowledge_time(knowledge_time, context=f"retract_fact({type(row).__name__})", now=now)
    if row.is_retraction:
        msg = (
            f"{type(row).__name__} version is already a retraction (knowledge_time "
            f"{row.knowledge_time.isoformat()}); the fact is not currently believed and "
            "there is nothing to withdraw. Re-assert it with a later knowledge_time first"
        )
        raise SupersessionError(msg)
    if knowledge_time <= row.knowledge_time:
        msg = (
            f"knowledge_time {knowledge_time.isoformat()} must be strictly later than the "
            f"retracted row's {row.knowledge_time.isoformat()}: an equal value collides on "
            "the primary key, an earlier one would never win the latest-knowledge read"
        )
        raise SupersessionError(msg)
    values: dict[str, Any] = {name: getattr(row, name) for name in row.__bitemporal_key__}
    values.update({name: null() for name in type(row).__bitemporal_payload__})
    values.update(
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        knowledge_time=knowledge_time,
        is_retraction=True,
    )
    return type(row)(**values)


def supersede_open_interval[RowT: BitemporalMixin](
    open_row: RowT,
    *,
    boundary: dt.datetime,
    knowledge_time: dt.datetime,
    changes: dict[str, Any],
    now: dt.datetime | None = None,
) -> tuple[RowT, RowT]:
    """Close an open-ended fact at ``boundary`` and open its successor there.

    The complete append-only supersession: one correction bounding the old
    interval at ``boundary`` and one new open-ended version starting at
    ``boundary``. Writing both in one transaction is what keeps consecutive
    versions non-overlapping — writing only the successor is the defect this
    module exists to prevent (module docstring).

    Args:
        open_row: the currently open-ended version. Not modified.
        boundary: the event-time instant at which the old version stops and
            the new one starts, timezone-aware UTC. Half-open on both sides:
            the old holds up to but not including it, the new from and
            including it, so the two intervals abut without overlapping.
        knowledge_time: when we learned of the change, timezone-aware UTC.
            Shared by both rows — we learned both halves at the same instant.
        changes: payload column values that differ in the successor (e.g.
            ``{"ticker": "XYZ"}``). Keys must be mapped payload columns; the
            temporal columns are set by this function and the logical key
            columns must not move (a changed key is a different entity, not a
            supersession).
        now: reference instant for the future-knowledge-time check; defaults
            to the current UTC time. Injected by tests.

    Returns:
        ``(correction, successor)`` — two new unsaved instances of
        ``type(open_row)``. Add both in a single writer-session transaction.

    Raises:
        SupersessionError: for every condition :func:`close_open_interval`
            rejects, and additionally if ``changes`` names a column that is
            not a mapped payload column, names a temporal column, names a
            logical-key column, or is empty (a supersession that changes
            nothing is a duplicate, not a new version).
        TypeError: if either datetime argument is naive.
        FutureKnowledgeTimeError: if ``knowledge_time`` is in the future.
    """
    if not changes:
        msg = (
            "supersede_open_interval requires at least one changed payload column; "
            "a successor identical to its predecessor is a duplicate row, not a new version"
        )
        raise SupersessionError(msg)
    allowed = set(_payload_values(open_row)) - set(open_row.__bitemporal_key__) - {"is_retraction"}
    unknown = sorted(set(changes) - allowed)
    if unknown:
        msg = (
            f"changes names column(s) {unknown} that cannot be superseded on "
            f"{type(open_row).__name__}; allowed payload columns are {sorted(allowed)}. "
            "Temporal columns are set by this function; changing a logical-key column "
            "would describe a different entity, not a new version of this one"
        )
        raise SupersessionError(msg)
    correction = close_open_interval(
        open_row, valid_to=boundary, knowledge_time=knowledge_time, now=now
    )
    values = _payload_values(open_row)
    values.update(changes)
    values.update(
        valid_from=boundary,
        valid_to=INFINITY,
        knowledge_time=knowledge_time,
    )
    successor = type(open_row)(**values)
    return correction, successor
