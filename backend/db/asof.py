"""As-of query layer for the bitemporal store (P2.3, DECISIONS.md D-011).

Read semantics implemented here, for each (logical key, ``valid_from``) group
of a bitemporal fact table at query timestamp ``as_of``:

1. only versions with ``knowledge_time <= as_of`` exist (boundary *inclusive*:
   a fact whose ``knowledge_time`` equals ``as_of`` is visible);
2. the version with the greatest ``knowledge_time`` wins (``DISTINCT ON``
   ordered by ``knowledge_time DESC``; the winner is deterministic because the
   primary key ``(*key, valid_from, knowledge_time)`` makes equal
   ``knowledge_time`` duplicates impossible — the PK itself is the total
   order, no further tiebreaker column is required);
3. if the winning version has ``is_retraction = true`` the fact is invisible.

Mechanism — statement rewrite, not ``with_loader_criteria``: latest-wins and
retraction masking depend on *other rows* of the same fact, so a row-local
loader criterion cannot express them. Instead a class-level
``do_orm_execute`` hook substitutes, for every bitemporal entity a SELECT
touches, a *versioned subquery* (the ``DISTINCT ON`` form above) via
``aliased(entity, subquery)`` plus SQLAlchemy's clause-adaption machinery
(``AliasedInsp._adapter``), which keeps ORM entity mapping intact — an
``as_of`` session running ``select(PriceBar)`` still yields ``PriceBar``
objects, now loaded from the versioned form. The rewrite is fail-closed: any
statement shape it cannot rewrite (textual SQL or text/literal fragments
naming a bitemporal table, user-``aliased()`` bitemporal entities, compound
selects, DML embedding a bitemporal read) raises instead of executing
unversioned. The rewriter's own subqueries are recognized by a
module-private token (:func:`backend.db._guard.mark_rewritten_subquery`),
never by name — a user subquery named ``_bitemporal_anything`` gets normal
enforcement.

Sessions:

- :func:`as_of` — the only sanctioned read path. Yields an ``AsyncSession``
  with the validated as-of timestamp bound in ``session.info``; the hook
  rewrites its bitemporal SELECTs. "Current view" is ``as_of(now)``; there is
  no unversioned read path.
- :func:`ingest_writer_session` — the ingestion write path. Permits
  INSERT/flush (``knowledge_time`` always supplied explicitly by the writer,
  never defaulted); its bitemporal SELECTs raise
  :class:`BitemporalBypassError` because no as-of is bound.

Documented exemption: ORM *column loads* (``is_column_load``, e.g.
``session.refresh`` after an INSERT) pass through unversioned. They re-read a
specific physical row addressed by the full primary key — which includes
``knowledge_time`` — so they cannot time-travel, and writers need them to
read server-generated audit columns. The hook marks them sanctioned so the
Core-level engine guard admits them too.

Below the ORM sits the Core-level engine guard
(:mod:`backend.db._guard`, installed by :mod:`backend.db.engine` on every
engine it creates), which is **default-deny at the SQL boundary**: the final
compiled SQL of every execution is name-scanned, and SQL naming a fact table
runs only if something explicitly sanctioned it. Executions this hook has
rewritten (or exempted as column loads) are marked with a module-private
token in their execution options, so exactly those pass; raw ``Connection``
access and textual SQL that never reach ORM events are refused there.

That inversion is what keeps this module's structural analysis off the
critical path for invariant I1: a statement shape
:func:`backend.db._guard.collect_references` fails to see is never
sanctioned, so it raises at the cursor boundary instead of executing
unversioned. Rewrite coverage is a usability property here; enforcement is
the guard's.
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa
from sqlalchemy import Select, TextClause, event, inspect, select
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.dml import UpdateBase

from backend.db._guard import (
    BitemporalBypassError,
    collect_dml_read_references,
    collect_references,
    mark_rewritten_subquery,
    sanctioned_execution_options,
    scan_sql_text,
)
from backend.db.bitemporal import bitemporal_classes
from backend.db.engine import _get_session_factory, _get_writer_session_factory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import ORMExecuteState
    from sqlalchemy.orm.util import AliasedClass, AliasedInsp
    from sqlalchemy.sql import ClauseElement

    from backend.db.bitemporal import BitemporalMixin

__all__ = [
    "AS_OF_INFO_KEY",
    "WRITER_INFO_KEY",
    "AsOfTimestampError",
    "BitemporalBypassError",
    "BitemporalRewriteError",
    "as_of",
    "ingest_writer_session",
]

AS_OF_INFO_KEY = "bitemporal_as_of"
"""``Session.info`` key holding the bound as-of timestamp (tz-aware UTC datetime)."""

WRITER_INFO_KEY = "bitemporal_writer"
"""``Session.info`` key marking a session produced by :func:`ingest_writer_session`."""

_REWRITE_NAME_PREFIX = "_bitemporal_"
"""Cosmetic name prefix for the rewrite's subqueries, for SQL readability only.

Trust is carried exclusively by the module-private token
(:func:`backend.db._guard.mark_rewritten_subquery`); the prefix grants no
exemption anywhere.
"""


class AsOfTimestampError(ValueError):
    """An ``as_of`` timestamp is naive, non-UTC, or in the future.

    Raised by :func:`as_of` before any session is created. The message states
    exactly which of the three requirements was violated and echoes the
    offending value.
    """


class BitemporalRewriteError(RuntimeError):
    """A bitemporal SELECT on a valid as-of session could not be rewritten.

    Fail-closed guard: rather than execute a statement shape the rewriter
    does not support (user-``aliased()`` bitemporal entities, compound
    selects, or any residual raw table reference after rewriting), the hook
    raises. Nothing unversioned ever reaches the database (invariant I1).
    """


def _validated_as_of(as_of_ts: dt.datetime) -> dt.datetime:
    """Validate and return an as-of timestamp (D-011 guard).

    Requirements: timezone-aware, UTC (zero utcoffset), and not later than
    the current wall clock. Raises :class:`AsOfTimestampError` naming the
    violated requirement otherwise. Units: an absolute point in time; the
    comparison ``knowledge_time <= as_of`` is inclusive at the boundary.
    """
    if as_of_ts.tzinfo is None or as_of_ts.utcoffset() is None:
        msg = f"as_of timestamp must be timezone-aware UTC; got naive datetime {as_of_ts!r}"
        raise AsOfTimestampError(msg)
    if as_of_ts.utcoffset() != dt.timedelta(0):
        msg = (
            f"as_of timestamp must be UTC (offset 0); got offset "
            f"{as_of_ts.utcoffset()} in {as_of_ts!r}"
        )
        raise AsOfTimestampError(msg)
    now = dt.datetime.now(dt.UTC)
    if as_of_ts > now:
        msg = (
            f"as_of timestamp may not be in the future: "
            f"{as_of_ts.isoformat()} > now {now.isoformat()}"
        )
        raise AsOfTimestampError(msg)
    return as_of_ts


@asynccontextmanager
async def as_of(as_of_ts: dt.datetime) -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` whose bitemporal SELECTs are pinned to ``as_of_ts``.

    The only sanctioned read path for bitemporal fact tables (D-011).
    ``as_of_ts`` must be timezone-aware UTC and not in the future
    (:class:`AsOfTimestampError` otherwise, raised before any I/O). Every
    SELECT touching a bitemporal mapper on the yielded session is rewritten to
    the versioned form: ``knowledge_time <= as_of_ts`` (boundary inclusive),
    latest knowledge wins per (logical key, ``valid_from``), winning
    retractions hide the fact. Non-bitemporal statements pass through
    unchanged. The session is closed on context exit; the caller neither
    commits nor writes through it.
    """
    ts = _validated_as_of(as_of_ts)
    session = _get_session_factory()(info={AS_OF_INFO_KEY: ts})
    try:
        yield session
    finally:
        await session.close()


@asynccontextmanager
async def ingest_writer_session() -> AsyncIterator[AsyncSession]:
    """Yield an ``AsyncSession`` for the append-only ingestion write path.

    Permits INSERT (``session.add`` + flush/commit and plain ORM
    ``insert().values()`` statements); the writer supplies ``knowledge_time``
    explicitly on every row under its connector's documented policy — there
    is no default to fall back on (D-011). Unversioned SELECTs on bitemporal
    tables raise :class:`BitemporalBypassError` — including reads embedded in
    DML (``INSERT ... FROM SELECT``, subqueries in SET/WHERE); reads go
    through :func:`as_of`. UPDATE and DELETE on fact tables are rejected by
    database triggers (migration 0003) regardless of session or role. The
    caller commits; the session is closed (rolling back anything uncommitted)
    on context exit.
    """
    session = _get_writer_session_factory()(info={WRITER_INFO_KEY: True})
    try:
        yield session
    finally:
        await session.close()


def _bitemporal_class_by_table_name() -> dict[str, type[BitemporalMixin]]:
    """Map table name -> registered bitemporal ORM class, from the registry.

    Rebuilt per call so models registered after this module's import (the
    registry fills when model modules are imported) are always seen.
    """
    mapping: dict[str, type[BitemporalMixin]] = {}
    for cls in bitemporal_classes():
        table = cast("sa.Table", cast("Any", cls).__table__)
        mapping[table.name] = cls
    return mapping


def _versioned_entity(cls: type[BitemporalMixin], as_of_ts: dt.datetime) -> AliasedClass[Any]:
    """Build the D-011 versioned form of one bitemporal entity at ``as_of_ts``.

    Two nested subqueries over the entity's table:

    1. ``DISTINCT ON (*logical key, valid_from) ... WHERE knowledge_time <=
       :as_of ORDER BY *key, valid_from, knowledge_time DESC`` — the latest
       version knowable at ``as_of_ts`` per fact (unique winner: the PK
       forbids equal ``knowledge_time`` within a group);
    2. ``WHERE NOT is_retraction`` — a winning retraction hides the fact.

    Returned as ``aliased(cls, <subquery>)`` so ORM entity mapping survives
    the substitution. Both subqueries carry the module-private rewriter
    token (unforgeable; names are cosmetic).
    """
    table = cast("sa.Table", cast("Any", cls).__table__)
    key_columns = [table.c[name] for name in cls.__bitemporal_key__]
    as_of_param = sa.literal(as_of_ts, sa.DateTime(timezone=True))
    versions = mark_rewritten_subquery(
        select(table)
        .where(table.c.knowledge_time <= as_of_param)
        .distinct(*key_columns, table.c.valid_from)
        .order_by(*key_columns, table.c.valid_from, table.c.knowledge_time.desc())
        .subquery(f"{_REWRITE_NAME_PREFIX}versions__{table.name}")
    )
    visible = mark_rewritten_subquery(
        select(versions)
        .where(~versions.c.is_retraction)
        .subquery(f"{_REWRITE_NAME_PREFIX}visible__{table.name}")
    )
    return cast(
        "AliasedClass[Any]",
        aliased(cast("Any", cls), visible, name=f"{table.name}_asof"),
    )


def _rewrite_select(
    statement: Select[Any],
    touched_tables: frozenset[str],
    as_of_ts: dt.datetime,
) -> Select[Any]:
    """Rewrite ``statement`` so every bitemporal reference is versioned at ``as_of_ts``.

    Two passes per touched entity: whole-entity elements in the columns
    clause are swapped for the versioned ``aliased()`` entity (preserving ORM
    entity results), then the clause adapter of that aliased entity rewrites
    every remaining reference — WHERE, ORDER BY, joins, subqueries — onto the
    versioned subquery. Assumes ``touched_tables`` came from
    :func:`backend.db._guard.collect_references` on this statement.
    """
    class_by_table = _bitemporal_class_by_table_name()
    replacements: dict[Any, AliasedClass[Any]] = {
        class_by_table[name]: _versioned_entity(class_by_table[name], as_of_ts)
        for name in sorted(touched_tables)
    }
    column_exprs = [description["expr"] for description in statement.column_descriptions]
    if any(expr in replacements for expr in column_exprs):
        statement = statement.with_only_columns(
            *[replacements.get(expr, expr) for expr in column_exprs]
        )
    for replacement in replacements.values():
        adapter = cast("AliasedInsp[Any]", inspect(replacement))._adapter
        statement = adapter.traverse(statement)
    return statement


def _reject_textual_bitemporal(statement: TextClause, table_names: frozenset[str]) -> None:
    """Raise if textual SQL names a bitemporal table (fail-closed name scan).

    Textual SQL cannot be rewritten to the versioned form, so any mention of
    a bitemporal table name (word-boundary, case-insensitive — over-matching
    is acceptable because the failure mode is a loud error, never a leak)
    raises :class:`BitemporalBypassError` on every session, as-of bound or
    not.
    """
    mentioned = scan_sql_text(statement.text, table_names)
    if mentioned:
        msg = (
            f"textual SQL references bitemporal table(s) {', '.join(sorted(mentioned))}; "
            "textual SQL cannot be rewritten to the versioned form — use ORM "
            "selects through as_of() (D-011)"
        )
        raise BitemporalBypassError(msg)


@event.listens_for(Session, "do_orm_execute")
def _enforce_bitemporal_reads(execute_state: ORMExecuteState) -> None:
    """Class-level hook enforcing and rewriting bitemporal reads (D-011 layer 2).

    Registered on the ORM :class:`Session` *class* as an import side effect
    of ``backend.db``, so every session in the process is covered — including
    hand-built ones and the sync session inside every ``AsyncSession`` — and
    the layer cannot be skipped by constructing sessions outside the
    sanctioned factories. Behavior:

    - column loads (``is_column_load``) pass through, marked sanctioned for
      the Core guard — PK-addressed re-reads of a specific physical version,
      incapable of time travel (module docstring);
    - textual SQL naming a bitemporal table raises on any session;
    - DML (INSERT/UPDATE/DELETE) that *embeds* a bitemporal read — ``INSERT
      ... FROM SELECT``, scalar subqueries in SET/VALUES/WHERE, references to
      a fact table other than the DML target — raises
      :class:`BitemporalBypassError` on any session. Plain ``INSERT ...
      VALUES`` into a fact table (the writer path) passes; UPDATE/DELETE of a
      fact table itself passes here and dies at the database triggers;
    - a bitemporal SELECT with no bound as-of raises
      :class:`BitemporalBypassError`; text/literal fragments naming a fact
      table inside a SELECT raise it on *any* session (they can never be
      versioned);
    - a bitemporal SELECT with a bound as-of is rewritten to the versioned
      form and its execution marked sanctioned for the Core guard; shapes the
      rewriter cannot handle raise :class:`BitemporalRewriteError`
      (fail-closed), verified by re-walking the rewritten statement for
      residual raw references.

    Statements this hook finds clean are returned **unsanctioned** on
    purpose: the Core guard then judges them on their compiled SQL, so a
    reference this hook's walker missed is refused rather than executed
    (:mod:`backend.db._guard`, default-deny).
    """
    if execute_state.is_column_load:
        execute_state.update_execution_options(**sanctioned_execution_options())
        return
    statement = execute_state.statement
    table_names = frozenset(_bitemporal_class_by_table_name())
    if isinstance(statement, TextClause):
        _reject_textual_bitemporal(statement, table_names)
        return
    if isinstance(statement, UpdateBase):
        dml_refs = collect_dml_read_references(statement, table_names)
        if dml_refs:
            # S608 suppressed below: this is a human-readable rejection
            # *message* quoting SQL keywords. No query is constructed here —
            # the offending statement is being refused, not executed.
            msg = (
                "DML statement embeds an unversioned read of bitemporal table(s) "  # noqa: S608
                f"{', '.join(sorted(dml_refs.all_tables()))}: reads inside "
                "INSERT ... FROM SELECT / subqueries cannot be versioned and are "
                "rejected on every session. Read through backend.db.as_of() and "
                "write plain INSERT ... VALUES rows instead (D-011/I1)"
            )
            raise BitemporalBypassError(msg)
        return
    if not execute_state.is_select:
        return
    refs = collect_references(cast("ClauseElement", statement), table_names)
    if refs.textual:
        msg = (
            f"textual SQL fragment references bitemporal table(s) "
            f"{', '.join(sorted(refs.textual))}; text()/literal fragments cannot "
            "be rewritten to the versioned form — use ORM selects through "
            "as_of() (D-011)"
        )
        raise BitemporalBypassError(msg)
    touched = sorted(refs.plain | refs.aliased)
    if not touched:
        return
    bound_as_of = execute_state.session.info.get(AS_OF_INFO_KEY)
    if bound_as_of is None:
        msg = (
            f"unversioned SELECT on bitemporal table(s) {', '.join(touched)}: "
            "this session has no bound as-of timestamp. Reads of bitemporal "
            "fact tables must go through backend.db.as_of(...) (D-011/I1)"
        )
        raise BitemporalBypassError(msg)
    if refs.aliased:
        msg = (
            f"aliased() references to bitemporal table(s) {', '.join(sorted(refs.aliased))} "
            "are not supported by the as-of rewriter; query the base entity "
            "instead (fail-closed per I1)"
        )
        raise BitemporalRewriteError(msg)
    if not isinstance(statement, Select):
        msg = (
            f"statement of type {type(statement).__name__} touching bitemporal "
            f"table(s) {', '.join(touched)} cannot be rewritten to the versioned "
            "form; use a plain ORM Select (fail-closed per I1)"
        )
        raise BitemporalRewriteError(msg)
    rewritten = _rewrite_select(statement, refs.plain, cast("dt.datetime", bound_as_of))
    residual = collect_references(rewritten, table_names)
    if residual:
        msg = (
            f"as-of rewrite left raw reference(s) to bitemporal table(s) "
            f"{', '.join(sorted(residual.all_tables()))}; refusing to execute "
            "(fail-closed per I1)"
        )
        raise BitemporalRewriteError(msg)
    execute_state.statement = rewritten
    execute_state.update_execution_options(**sanctioned_execution_options())
