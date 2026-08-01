"""Closing open-ended facts without an UPDATE (P3.1 — audit finding).

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

Both helpers return **new, unsaved ORM instances**; nothing here touches the
database. The caller adds them in a single writer-session transaction so the
close and the successor land atomically.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import inspect

from backend.db.bitemporal import INFINITY, BitemporalMixin
from backend.ingest.errors import SupersessionError
from backend.ingest.write import validate_knowledge_time

if TYPE_CHECKING:
    from sqlalchemy.orm import Mapper

__all__ = ["close_open_interval", "supersede_open_interval"]

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
