"""Knowledge-time hygiene enforced in the write path (P3.1, D-011 control).

D-011 defines ``knowledge_time`` as *when the information became knowable to
the market*. A row whose ``knowledge_time`` lies in the future therefore
claims we knew something before it was knowable — the exact shape of the
lookahead bias invariant I1 exists to make impossible. Phase 2 built no
defence against it: the column has no default and no CHECK (both deliberate,
so backfills can carry honest historical values), which leaves the write path
as the only place the check can live.

**Reject, not flag** — and the choice is deliberate. Flagging is right for the
live-lag check in :mod:`backend.ingest.quality`, where the data is real and
only the source's timeliness is in question. It is wrong here: there is no
such thing as an honest fact whose knowability is in the future, so a row that
claims one is not late data, it is wrong data. Writing it and flagging it
would put a row into an append-only store that can never be deleted, only
retracted — and until someone acts on the flag, every ``as_of`` query between
now and that future instant is at risk of returning it. The cost of rejecting
is a failed run with a precise error; the cost of flagging is a silent I1
violation with a note attached.

Enforcement point
-----------------

A ``before_flush`` listener on the ORM :class:`~sqlalchemy.orm.Session`
**class**, registered as an import side effect of this module (which
``backend.ingest/__init__.py`` imports), mirroring how :mod:`backend.db.asof`
installs its read enforcement. Class-level rather than session-level so no
writer can dodge it by constructing its own session, and ``before_flush``
rather than an ``__init__`` validator so it fires on the values actually about
to be sent — including any mutated after construction.

Scope, stated exactly rather than overclaimed: the listener covers every flush
in a process that has imported ``backend.ingest`` — which is every process
using the connector framework, since importing any ``backend.ingest`` module
executes the package ``__init__``. It is *not* installed by importing
``backend.db`` alone. That is the honest boundary: the guard belongs to the
ingestion write path, and the ingestion write path is the only thing that
writes fact rows.

The check compares against the writing host's UTC clock with **zero
tolerance**. No skew allowance is granted: a source whose clock runs ahead of
ours produces a loud, specific failure, which is a finding to record and
resolve (a documented per-source clock offset, or a corrected derivation),
not something to absorb behind a fudge constant that would then apply to every
source forever.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Final

from sqlalchemy import event
from sqlalchemy.orm import Session

from backend.db.bitemporal import BitemporalMixin
from backend.ingest.errors import FutureKnowledgeTimeError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.orm import UOWTransaction

__all__ = ["FUTURE_KNOWLEDGE_TIME_TOLERANCE", "validate_knowledge_time"]

FUTURE_KNOWLEDGE_TIME_TOLERANCE: Final = dt.timedelta(0)
"""Allowance for a ``knowledge_time`` ahead of the writing host's clock: none.

Zero by decision, not by omission (module docstring). It is a named constant so
the value is visible and so any future change to it is an explicit, reviewable
edit rather than a scattered comparison.
"""


def validate_knowledge_time(
    knowledge_time: dt.datetime,
    *,
    context: str,
    now: dt.datetime | None = None,
) -> dt.datetime:
    """Return ``knowledge_time`` unchanged, or raise if it is not writable.

    Args:
        knowledge_time: the value about to be written, timezone-aware UTC.
        context: short description of what is being written (model and key),
            used in the error message.
        now: the instant to compare against; defaults to the current UTC time.
            Injected by tests so the boundary can be exercised exactly.

    Returns:
        ``knowledge_time``, unchanged, when it is at or before ``now`` plus
        :data:`FUTURE_KNOWLEDGE_TIME_TOLERANCE`.

    Raises:
        TypeError: if ``knowledge_time`` is naive. asyncpg would reinterpret a
            naive value in host-local time (D-012), so a naive value is not
            merely unvalidatable, it is wrong by hours.
        FutureKnowledgeTimeError: if it is later than the allowance. The
            message states both instants and the excess.
    """
    if knowledge_time.tzinfo is None or knowledge_time.utcoffset() is None:
        msg = (
            f"{context}: knowledge_time must be timezone-aware UTC; got naive "
            f"{knowledge_time!r} (D-011/D-012)"
        )
        raise TypeError(msg)
    reference = now if now is not None else dt.datetime.now(dt.UTC)
    limit = reference + FUTURE_KNOWLEDGE_TIME_TOLERANCE
    if knowledge_time > limit:
        msg = (
            f"{context}: knowledge_time {knowledge_time.isoformat()} is in the future "
            f"(now {reference.isoformat()}, excess {knowledge_time - reference}). "
            "knowledge_time is when a fact became knowable to the market; a future "
            "value asserts knowledge that does not yet exist and would leak into "
            "as_of queries once that instant passes. Refusing the write (D-011/I1)"
        )
        raise FutureKnowledgeTimeError(msg)
    return knowledge_time


def _describe(instance: BitemporalMixin) -> str:
    """Return a short identifying string for a row, for error messages."""
    key_values = ", ".join(
        f"{name}={getattr(instance, name, None)!r}" for name in instance.__bitemporal_key__
    )
    valid_from = getattr(instance, "valid_from", None)
    return f"{type(instance).__name__}({key_values}, valid_from={valid_from!r})"


def _check_instances(instances: Iterable[object], now: dt.datetime) -> None:
    """Validate ``knowledge_time`` on every bitemporal instance in ``instances``."""
    for instance in instances:
        if not isinstance(instance, BitemporalMixin):
            continue
        knowledge_time = getattr(instance, "knowledge_time", None)
        if isinstance(knowledge_time, dt.datetime):
            validate_knowledge_time(knowledge_time, context=_describe(instance), now=now)


@event.listens_for(Session, "before_flush")
def _reject_future_knowledge_time(
    session: Session,
    flush_context: UOWTransaction,  # noqa: ARG001 — SQLAlchemy dispatches positionally
    instances: object,  # noqa: ARG001 — SQLAlchemy dispatches positionally
) -> None:
    """Refuse any flush carrying a bitemporal row with a future ``knowledge_time``.

    Registered on the ORM ``Session`` *class* as an import side effect of this
    module, so every flush in the process is covered — including sessions
    built outside the sanctioned ingestion factory. Both pending inserts
    (``session.new``) and modified instances (``session.dirty``) are checked;
    fact tables reject UPDATE at the database triggers anyway, so the second
    is defence in depth rather than a supported path.

    Rows with an unset or non-``datetime`` ``knowledge_time`` pass through
    untouched: the column is ``NOT NULL`` with no default, so the database
    refuses them a moment later with its own precise error, and duplicating
    that check here would only produce a worse message.
    """
    now = dt.datetime.now(dt.UTC)
    _check_instances(session.new, now)
    _check_instances(session.dirty, now)
