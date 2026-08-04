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

The same inversion now covers the as-of **value**, not just the set of
tables. A statement can be perfectly versioned in shape and still be executed
with the wrong instant, because SQLAlchemy's compiled-statement cache makes
two rewrites of one shape at different as-of values share a cache key: if the
rewrite scatters the as-of across several differently-keyed bind parameters,
a cache hit resolves some of them from the value frozen into the cached
compiled object — the earlier execution's as-of — and the query silently
reads the store at that instant instead. Two mechanisms close it, and both
are required:

- :func:`_rewrite_select` builds every versioned subquery of one statement
  against a **single explicitly-keyed bind**
  (:func:`backend.db._guard.as_of_bind`), so exactly one parameter carries
  the value and the cache can only resolve it from the current execution;
- the rewritten statement is checked for that invariant
  (:func:`_assert_single_as_of_bind`), and the session's as-of travels with
  the execution's sanction so the Core guard re-verifies, at the cursor
  boundary, that the values *actually being sent* equal it
  (``backend.db._guard._verify_as_of_binds``). Both raise
  :class:`AsOfBindIntegrityError` and neither trusts the rewrite's output.

Retractions on the read path (P2.10)
------------------------------------

A retraction row carries **no payload** — every payload column is NULL and
the database refuses any other shape (:mod:`backend.db.bitemporal`). That
makes handing one to a caller a type error as well as a semantic one, so it
is refused twice more here rather than left to the masking predicate alone:

- :func:`_assert_retraction_mask` re-derives, from the *rewritten statement*,
  that every fact table the statement touches sits under a scope filtering
  ``NOT is_retraction``, and raises :class:`RetractionMaskError` otherwise.
  Same discipline as :func:`_assert_single_as_of_bind`: the rewriter's output
  is verified, never assumed, so a future edit that drops or weakens the
  predicate fails closed instead of quietly widening every read;
- :func:`_refuse_loaded_retraction` is a row-level backstop on the ORM
  ``load`` event: any bitemporal instance that arrives from the database
  marked ``is_retraction`` raises :class:`RetractedFactError` before the
  caller can touch it. It is path-independent — it does not care which query
  produced the row — which is what makes it a genuine second line rather than
  a restatement of the first.

The one place a retraction instance is legitimately reachable is the writer
that just built it: ``session.refresh`` after inserting one re-reads that
physical row (the documented column-load exemption above) and fires
``refresh``, not ``load``. Its payload attributes read as ``None`` there,
which is the single case where the non-optional ``Mapped[...]`` annotations
on payload columns state the *read path's* guarantee rather than a universal
one. Stated here rather than left to be discovered.
"""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import sqlalchemy as sa
from sqlalchemy import Select, TextClause, event, inspect, select
from sqlalchemy.orm import Mapper, Session, aliased
from sqlalchemy.sql import operators
from sqlalchemy.sql.dml import UpdateBase
from sqlalchemy.sql.elements import ColumnClause, UnaryExpression
from sqlalchemy.sql.selectable import Subquery, TableClause

from backend.db._guard import (
    AS_OF_BIND_KEY,
    AsOfBindIntegrityError,
    BitemporalBypassError,
    as_of_bind,
    collect_as_of_binds,
    collect_dml_read_references,
    collect_references,
    mark_rewritten_subquery,
    sanctioned_execution_options,
    scan_sql_text,
)
from backend.db.bitemporal import BitemporalMixin, bitemporal_classes
from backend.db.engine import _get_session_factory, _get_writer_session_factory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import ORMExecuteState, QueryContext
    from sqlalchemy.orm.util import AliasedClass, AliasedInsp
    from sqlalchemy.sql import ClauseElement
    from sqlalchemy.sql.elements import BindParameter

__all__ = [
    "AS_OF_INFO_KEY",
    "WRITER_INFO_KEY",
    "AsOfBindIntegrityError",
    "AsOfTimestampError",
    "BitemporalBypassError",
    "BitemporalRewriteError",
    "RetractedFactError",
    "RetractionMaskError",
    "as_of",
    "ingest_writer_session",
]

AS_OF_INFO_KEY = "bitemporal_as_of"
"""``Session.info`` key holding the bound as-of timestamp (tz-aware UTC datetime)."""

WRITER_INFO_KEY = "bitemporal_writer"
"""``Session.info`` key marking a session produced by :func:`ingest_writer_session`."""

_RETRACTION_COLUMN = "is_retraction"
"""The mixin column whose truth hides a fact from every later as-of read."""

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


class RetractionMaskError(BitemporalRewriteError):
    """A rewritten statement reaches a fact table without masking retractions.

    Raised by :func:`_assert_retraction_mask` before execution. A retraction
    carries no payload (P2.10), so a statement that can return one would hand
    a caller a row of NULLs shaped like an observation — the read-path half of
    the fabrication problem the storage constraints close. Fail-closed: the
    statement is refused rather than executed with a partial mask.
    """


class RetractedFactError(RuntimeError):
    """A retraction row reached application code as though it were a fact.

    Raised by :func:`_refuse_loaded_retraction` at ORM load time, whatever
    query produced the row. Reaching this is a defect in the read path, not a
    condition callers handle: the as-of layer masks retractions and verifies
    the mask, so a retraction arriving here means one of those failed.
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


def _versioned_entity(
    cls: type[BitemporalMixin], as_of_param: BindParameter[dt.datetime]
) -> AliasedClass[Any]:
    """Build the D-011 versioned form of one bitemporal entity at ``as_of_param``.

    Two nested subqueries over the entity's table:

    1. ``DISTINCT ON (*logical key, valid_from) ... WHERE knowledge_time <=
       :as_of ORDER BY *key, valid_from, knowledge_time DESC`` — the latest
       version knowable at the as-of instant per fact (unique winner: the PK
       forbids equal ``knowledge_time`` within a group);
    2. ``WHERE NOT is_retraction`` — a winning retraction hides the fact.

    Returned as ``aliased(cls, <subquery>)`` so ORM entity mapping survives
    the substitution. Both subqueries carry the module-private rewriter
    token (unforgeable; names are cosmetic).

    ``as_of_param`` is the **shared** bind object of the whole rewrite
    (:func:`backend.db._guard.as_of_bind`), not a timestamp: every versioned
    subquery in one statement must carry one bind identity, or the compiled
    cache can resolve some of them from a previous execution's value (see
    :func:`_rewrite_select`).
    """
    table = cast("sa.Table", cast("Any", cls).__table__)
    key_columns = [table.c[name] for name in cls.__bitemporal_key__]
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

    **One bind identity for the whole statement.** All versioned subqueries
    are built against a single explicitly-keyed
    :func:`~backend.db._guard.as_of_bind`, shared across every touched table.
    This is not a tidiness preference; it is what keeps the as-of value
    correct under SQLAlchemy's compiled-statement cache. With one adapter pass
    per table, a later pass clones a tree that already contains an earlier
    table's versioned subquery — so an *anonymous* as-of bind (``sa.literal``,
    or any ``unique=True`` bindparam) becomes several parameters with
    different keys, because ``BindParameter._clone`` re-keys anonymous binds.
    Two rewrites of one shape at different as-of values then compile to equal
    cache keys while the compiler's cache-key-to-bind matching (which pairs by
    ``.key`` across the clone lineage) cannot relate every parameter, and a
    cache hit resolves the unmatched ones from the value baked into the
    *cached* compiled object — the first execution's as-of. An explicit key
    survives cloning unchanged, so however many copies adaption makes, the
    compiler sees one parameter and resolves it from the current execution.
    Verified empirically against the installed SQLAlchemy (2.0.51) rather than
    assumed, and re-verified on every execution at the cursor boundary
    (:mod:`backend.db._guard`) — the cache is left enabled, because disabling
    it would tax every as-of query forever while treating a symptom.
    """
    class_by_table = _bitemporal_class_by_table_name()
    as_of_param = as_of_bind(as_of_ts)
    replacements: dict[Any, AliasedClass[Any]] = {
        class_by_table[name]: _versioned_entity(class_by_table[name], as_of_param)
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


def _assert_single_as_of_bind(statement: Select[Any], as_of_ts: dt.datetime) -> None:
    """Fail closed unless the rewritten statement carries exactly one as-of value.

    The post-rewrite invariant, checked on the statement itself before it is
    handed back to the ORM. Two claims, and the first is the one the
    compiled-cache defect violated:

    1. **exactly one distinct as-of bind key.** Several keys means several
       parameters carrying the same logical value, which is the precondition
       for the compiled cache to resolve some of them from an earlier
       execution (:func:`_rewrite_select`). The count of bind *objects* is
       deliberately not checked — clause adaption legitimately clones them,
       and cloning is harmless exactly as long as the key is preserved, which
       is what makes the key the identity that matters.
    2. **every as-of bind holds this session's as-of**, so a foreign bind
       colliding with :data:`~backend.db._guard.AS_OF_BIND_KEY` cannot
       silently override the instant (SQLAlchemy resolves same-key binds to
       one parameter without complaint).

    At least one as-of bind must exist: this runs only for statements the
    rewriter versioned, and every versioned subquery carries the predicate.
    Raises :class:`~backend.db._guard.AsOfBindIntegrityError`; the independent
    check on the values actually sent happens at the cursor boundary.
    """
    binds = collect_as_of_binds(statement)
    keys = {bind.key for bind in binds}
    if keys != {AS_OF_BIND_KEY}:
        msg = (
            f"as-of rewrite produced {len(keys)} distinct as-of bind key(s) "
            f"{sorted(keys)} over {len(binds)} bind(s); exactly one key "
            f"({AS_OF_BIND_KEY!r}) is required, because several keys let "
            "SQLAlchemy's compiled-statement cache resolve one of them from a "
            "previous execution's as_of. Refusing to execute (fail-closed per I1)"
        )
        raise AsOfBindIntegrityError(msg)
    wrong = sorted({str(bind.value) for bind in binds if bind.value != as_of_ts})
    if wrong:
        msg = (
            f"as-of rewrite produced bind value(s) {wrong} but this session bound "
            f"as_of {as_of_ts.isoformat()}; refusing to execute (fail-closed per I1)"
        )
        raise AsOfBindIntegrityError(msg)


def _is_retraction_mask(element: object) -> bool:
    """True when ``element`` is exactly the ``NOT is_retraction`` predicate.

    Structural, not textual: ``~column`` on a ``Boolean`` compiles to a
    :class:`~sqlalchemy.sql.elements.UnaryExpression` carrying the
    ``is_false`` operator over the column, which is what
    :func:`_versioned_entity` builds and therefore what this recognizes. A
    hand-written ``is_retraction == False`` is deliberately *not* accepted:
    this verifies the rewriter's own output against the one shape the
    rewriter emits, so a change to that shape has to be seen rather than
    absorbed.
    """
    return (
        isinstance(element, UnaryExpression)
        and element.operator is operators.is_false
        and isinstance(element.element, ColumnClause)
        and element.element.name == _RETRACTION_COLUMN
    )


def _masks_retractions(statement: Select[Any]) -> bool:
    """True when this select's **own** WHERE clause carries the retraction mask.

    The walk stops at nested ``Select``/``Subquery`` boundaries so an inner
    scope's mask is never credited to an outer one — a correlated subquery
    that happens to filter retractions says nothing about what its enclosing
    select can return.
    """
    where = statement.whereclause
    if where is None:
        return False
    stack: list[ClauseElement] = [where]
    seen: set[int] = set()
    while stack:
        element = stack.pop()
        if id(element) in seen:
            continue
        seen.add(id(element))
        if _is_retraction_mask(element):
            return True
        if isinstance(element, Select | Subquery):
            continue
        stack.extend(element.get_children())
    return False


def _assert_retraction_mask(statement: Select[Any], touched_tables: frozenset[str]) -> None:
    """Fail closed unless **every** path to a fact table passes under a mask.

    The post-rewrite invariant for P2.10, checked on the statement itself
    before it goes back to the ORM. The tree is walked carrying a "currently
    inside a scope that filters ``NOT is_retraction``" flag, which turns on
    when a ``Select`` whose own WHERE carries the mask is entered; a fact
    table reached with the flag still off is a violation.

    Per *reference*, not per table, because one masked reference does not
    redeem an unmasked one: a statement joining the versioned form of
    ``price_bar`` to the raw table would satisfy a per-table check and still
    return retractions. Nodes are memoized on (node, flag) rather than on
    node, so a subtree reachable both ways is judged both ways.

    An unmasked reference means the statement can return retraction rows —
    rows whose payload columns are all NULL by constraint — so a caller would
    receive a hollow object shaped like an observation. Raises
    :class:`RetractionMaskError`. Like :func:`_assert_single_as_of_bind` this
    re-derives the property from the finished statement rather than trusting
    that :func:`_versioned_entity` put the predicate there.
    """
    unmasked: set[str] = set()
    stack: list[tuple[ClauseElement, bool]] = [(statement, False)]
    seen: set[tuple[int, bool]] = set()
    while stack:
        element, masked = stack.pop()
        marker = (id(element), masked)
        if marker in seen:
            continue
        seen.add(marker)
        if isinstance(element, Select) and _masks_retractions(element):
            masked = True
        if isinstance(element, TableClause):
            if element.name in touched_tables and not masked:
                unmasked.add(element.name)
            continue
        if isinstance(element, ColumnClause) and element.table is not None:
            stack.append((element.table, masked))
        stack.extend((child, masked) for child in element.get_children())
    if unmasked:
        msg = (
            f"as-of rewrite left bitemporal table(s) {', '.join(sorted(unmasked))} reachable "
            "without a NOT is_retraction mask; a retraction carries no payload (P2.10), "
            "so such a row would reach the caller as an observation with NULL values. "
            "Refusing to execute (fail-closed per I1/I3)"
        )
        raise RetractionMaskError(msg)


@event.listens_for(Mapper, "load")
def _refuse_loaded_retraction(
    target: object,
    context: QueryContext,  # noqa: ARG001 — MapperEvents API signature
) -> None:
    """Refuse to hand a freshly loaded retraction row to application code.

    Row-level backstop, registered on the :class:`~sqlalchemy.orm.Mapper`
    *class* so it covers every query on every session in the process,
    including read paths written by later phases. A retraction that reaches
    here has already defeated the masking predicate and
    :func:`_assert_retraction_mask`, so :class:`RetractedFactError` is a
    defect report, not a condition to handle.

    ``is_retraction`` is read out of the loaded instance's attribute dict
    rather than through the attribute, so a column-restricted load
    (``load_only``) cannot make this guard emit a lazy SELECT from inside a
    load event. Absent from the dict means the column was not selected, which
    means the row cannot be shown to be a retraction here; enforcement for
    that shape stays with the statement-level mask, which applies to
    column-restricted loads exactly as it does to entity loads.

    Fires on the initial load only. ``session.refresh`` of a row already in
    the session dispatches ``refresh``, so the writer re-reading the
    retraction it just inserted is unaffected (module docstring).
    """
    if not isinstance(target, BitemporalMixin):
        return
    state = cast("Any", inspect(target))
    if state.dict.get(_RETRACTION_COLUMN) is not True:
        return
    table = cast("sa.Table", cast("Any", type(target)).__table__)
    msg = (
        f"a retraction row of {table.name} was loaded into application code: "
        "retractions record that a fact is no longer believed and carry no payload "
        "(every payload column is NULL by CHECK constraint, P2.10), so they must "
        "never be returned as observations. Reads go through backend.db.as_of(), "
        "which masks them (D-011/I1)"
    )
    raise RetractedFactError(msg)


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
      residual raw references. The rewritten statement must also carry
      exactly one as-of bind key holding exactly this session's as-of
      (:func:`_assert_single_as_of_bind`) and must mask retractions over
      every fact table it touches (:func:`_assert_retraction_mask`), and the
      session's as-of travels with the sanction as **ground truth** so the
      Core guard can re-check the values actually sent at the cursor boundary
      — neither the shape nor the value of the rewrite is taken on trust.

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
    as_of_ts = cast("dt.datetime", bound_as_of)
    rewritten = _rewrite_select(statement, refs.plain, as_of_ts)
    residual = collect_references(rewritten, table_names)
    if residual:
        msg = (
            f"as-of rewrite left raw reference(s) to bitemporal table(s) "
            f"{', '.join(sorted(residual.all_tables()))}; refusing to execute "
            "(fail-closed per I1)"
        )
        raise BitemporalRewriteError(msg)
    _assert_single_as_of_bind(rewritten, as_of_ts)
    _assert_retraction_mask(rewritten, refs.plain)
    execute_state.statement = rewritten
    execute_state.update_execution_options(**sanctioned_execution_options(as_of_ts))
