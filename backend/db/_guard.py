"""Shared bitemporal read-reference analysis and the Core-level engine guard (D-011).

**Import contract (D-011 bypass-prevention layer 3):** this module is private
to ``backend/db``. Ruff rule TID251 bans importing the private engine module
anywhere else in the codebase; the only public handle on this module's
behavior is the :class:`BitemporalBypassError` re-exported by
:mod:`backend.db`.

Two enforcement surfaces consume this module:

- the ORM ``do_orm_execute`` hook in :mod:`backend.db.asof` (layer 2), which
  uses :func:`collect_references` / :func:`collect_dml_read_references` to
  find bitemporal fact-table reads in ORM statements before rewriting or
  rejecting them; and
- the **Core-level engine guard** (:func:`install_core_guard`), registered on
  every engine :mod:`backend.db.engine` creates (application and admin).

Enforcement model: **default-deny at the SQL boundary (allowlist).**
------------------------------------------------------------------

``before_cursor_execute`` — which sees the *final compiled SQL string* of
every execution, whatever its origin — is the arbiter, and it denies by
default: a statement whose SQL names a bitemporal fact table executes only
if something explicitly vetted it and **stamped it sanctioned**. There is no
statement-type skip: compiled clause elements, ``text()``, ``exec_driver_sql``
and driver-level paths are all scanned identically.

This inversion is deliberate and load-bearing. The structural walker
(:func:`collect_references`) is a heuristic over SQLAlchemy clause trees; a
shape it cannot see would previously have produced a *silent unversioned
read*, because the string scan was skipped for anything compiled from a
clause element. Under default-deny a walker blind spot produces a loud
:class:`BitemporalBypassError` instead: the walker's completeness is no
longer the property I1 rests on.

Two independent channels carry a sanction, both keyed on module-private
token objects compared by identity — knowing a name or a key is useless,
forging admission requires the token itself:

1. **Execution options** (:data:`SANCTION_EXECUTION_OPTION` /
   :func:`sanctioned_execution_options`), set by the ORM hook in
   :mod:`backend.db.asof` on statements it has itself rewritten to the
   versioned form, and on ORM column loads it exempts. These reach
   ``context.execution_options`` because the ORM passes them as the
   caller-supplied execution options of the execution.
2. **A per-connection pre-compilation grant** (:func:`_grant_core_sanction`),
   issued by ``before_execute`` for the one Core-level shape that must reach
   Postgres naming a fact table without being a read: **plain DML whose
   target is a fact table and which embeds no fact-table read** — the
   ingestion writer's ``INSERT ... VALUES``, and the ``UPDATE``/``DELETE``
   that must reach the append-only database triggers to be refused there.
   The grant names the exact table(s) it covers and is matched against the
   execution by object identity, so it admits nothing else.

   *Why a connection grant and not ``before_execute(..., retval=True)``:* on
   the installed SQLAlchemy (2.0.x) ``Connection._execute_clauseelement``
   merges the execution options **before** dispatching ``before_execute``
   and passes that pre-computed mapping to the execution context, so an
   element returned from the event with new ``execution_options`` never
   reaches ``context.execution_options``. Verified empirically against the
   installed version rather than assumed. The grant is written to
   ``conn.info`` immediately before the statement compiles, revoked at the
   start of every ``before_execute``, and admits an execution only when
   ``context.invoked_statement`` (or ``context.compiled``) *is* the exact
   object that was vetted — so it cannot outlive or widen beyond its
   statement.

Deliberately **not** stamped, so that the SQL scan still judges them:

- **Selects the walker found clean.** Stamping those would reintroduce the
  very blind spot this design removes — a walker that sees nothing would
  hand out a pass. Their SQL simply must not name a fact table, which is
  what "clean" means, verified on the string rather than on the walker.
- **DDL.** ``ExecutableDDLElement`` includes :class:`sqlalchemy.schema.DDL`,
  which carries arbitrary SQL text; blanket-stamping DDL would reopen an
  arbitrary-read path through the public ``create_admin_engine``. DDL that
  names a fact table therefore fails closed on guarded engines — which costs
  nothing, because every sanctioned fact-table DDL (alembic migrations) runs
  on the unguarded module-private migration engine (D-012).

Textual SQL cannot be structurally analyzed, so it is name-scanned: a
conservative case-insensitive word-boundary regex over the SQL string
(:func:`scan_sql_text`). This is fail-closed by design — a false positive
(e.g. a string literal or column literal spelling a fact-table name, or
``TRUNCATE price_bar``) raises :class:`BitemporalBypassError` even though no
versioned read occurs. That is acceptable: the failure mode is a loud error,
never a leak, and sanctioned infrastructure work (migrations, test resets)
runs on the unguarded module-private migration engine instead. Under
default-deny that same conservatism now applies to compiled statements too:
an inline literal spelling a fact-table name inside a Core select is
refused.

The rewriter's own versioned subqueries are marked with a second private
token via :func:`mark_rewritten_subquery` (an instance attribute, which
survives SQLAlchemy's clause-adaption cloning) so the reference walker can
prune them; a user naming a subquery ``_bitemporal_anything`` gets no
exemption.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast

from sqlalchemy import event
from sqlalchemy.sql.ddl import ExecutableDDLElement
from sqlalchemy.sql.dml import UpdateBase
from sqlalchemy.sql.elements import ClauseElement, ColumnClause, TextClause
from sqlalchemy.sql.selectable import Alias, SelectBase, Subquery, TableClause

from backend.db.bitemporal import bitemporal_classes

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.engine import Connection, Engine
    from sqlalchemy.engine.interfaces import DBAPICursor, ExecutionContext
    from sqlalchemy.sql import Executable

__all__ = [
    "SANCTION_EXECUTION_OPTION",
    "BitemporalBypassError",
    "References",
    "collect_dml_read_references",
    "collect_references",
    "fact_table_names",
    "install_core_guard",
    "mark_rewritten_subquery",
    "sanctioned_execution_options",
    "scan_sql_text",
]

SANCTION_EXECUTION_OPTION: Final = "_bitemporal_sanctioned_execution"
"""Execution-option key carrying the sanction token for rewritten executions.

The key name is not a secret; admission requires the *value* to be the
module-private token object (checked with ``is``), which cannot be obtained
without importing this banned module.
"""

_EXECUTION_SANCTION_TOKEN: Final[object] = object()
"""Unforgeable marker: only executions carrying exactly this object pass the guard."""

_SUBQUERY_MARKER_ATTR: Final = "_bitemporal_rewriter_subquery_token"
"""Instance-attribute name under which rewriter-built subqueries carry their token.

An instance attribute (not a name prefix, not an annotation) because it
survives both clause-adaption cloning (``_clone`` copies ``__dict__``) and
``_deannotate()``. Setting the attribute without the private token object
grants nothing — membership is checked with ``is``.
"""

_SUBQUERY_SANCTION_TOKEN: Final[object] = object()
"""Unforgeable marker for subqueries built by the as-of rewriter itself."""

_CORE_SANCTION_INFO_KEY: Final = "_bitemporal_core_sanction"
"""``Connection.info`` key carrying the pre-compilation grant for one statement.

The carrier for Core-level sanctions, used because SQLAlchemy merges an
execution's options *before* dispatching ``before_execute`` (module
docstring). Like the option key above, the name is not a secret: the grant is
a :class:`_CoreSanction` whose token is checked with ``is``.
"""


class BitemporalBypassError(RuntimeError):
    """A statement would have reached a bitemporal fact table unsanctioned.

    D-011 bypass prevention. Raised before any row leaves the database, by
    either enforcement surface:

    - the class-level ORM ``do_orm_execute`` hook (:mod:`backend.db.asof`) —
      for any :class:`sqlalchemy.orm.Session` in the process whose SELECT (or
      DML-embedded read) touches a fact table with no bound as-of; and
    - the Core-level engine guard (:func:`install_core_guard`) on every engine
      ``backend.db`` creates, catching ``session.connection()``, the admin
      engine and ``exec_driver_sql``. Its ``before_execute`` half raises on a
      *detected* unversioned reference; its ``before_cursor_execute`` half
      raises on the **final SQL string** of any execution that names a fact
      table without carrying a sanction — the default-deny rule that makes a
      structural blind spot loud rather than silent (module docstring).

    The message names the offending table(s) and the sanctioned read path
    (``backend.db.as_of``).
    """


def fact_table_names() -> frozenset[str]:
    """Return the table names of every registered bitemporal fact model.

    Rebuilt per call so models registered after this module's import (the
    registry fills as model modules are imported) are always seen.
    """
    return frozenset(str(cast("Any", cls).__table__.name) for cls in bitemporal_classes())


class References(NamedTuple):
    """Bitemporal fact-table references found in one clause tree.

    - ``plain``: tables referenced directly (rewritable by the as-of layer);
    - ``aliased``: tables referenced through an :class:`Alias` (the rewriter
      refuses these, fail-closed);
    - ``textual``: table names found by scanning opaque textual fragments
      (``text()``, ``literal_column``) — never rewritable, always rejected.
    """

    plain: frozenset[str]
    aliased: frozenset[str]
    textual: frozenset[str]

    def all_tables(self) -> frozenset[str]:
        """Return every referenced table name across all three capacities."""
        return self.plain | self.aliased | self.textual

    def __bool__(self) -> bool:
        """True when any reference of any capacity was found."""
        return bool(self.plain or self.aliased or self.textual)


def sanctioned_execution_options() -> dict[str, object]:
    """Return execution options marking one execution as sanctioned by the rewriter.

    Only :mod:`backend.db.asof` calls this, for statements it has itself
    rewritten to the versioned form (or column loads it exempts). The Core
    guard admits exactly these executions.
    """
    return {SANCTION_EXECUTION_OPTION: _EXECUTION_SANCTION_TOKEN}


def _is_sanctioned_execution(execution_options: Mapping[str, Any]) -> bool:
    """True when the options carry the exact module-private sanction token."""
    return execution_options.get(SANCTION_EXECUTION_OPTION) is _EXECUTION_SANCTION_TOKEN


class _CoreSanction(NamedTuple):
    """One pre-compilation grant issued by ``before_execute`` for one statement.

    - ``token``: the module-private sanction object, compared by identity, so
      a grant written into ``conn.info`` by anything but this module is inert;
    - ``statement``: the exact object handed to ``before_execute``; the grant
      applies only to an execution whose ``context.invoked_statement`` (or
      ``context.compiled``, for the pre-built-``Compiled`` path) *is* it;
    - ``tables``: the fact-table names the grant covers. The cursor guard
      admits the execution only when every fact-table name found in the final
      SQL is in this set, so a grant for ``INSERT INTO price_bar`` cannot
      launder a statement that also names another fact table.
    """

    token: object
    statement: object
    tables: frozenset[str]


def _grant_core_sanction(conn: Connection, statement: object, tables: frozenset[str]) -> None:
    """Record that ``statement`` was vetted and may name ``tables`` in its SQL.

    Written to ``conn.info`` (per DBAPI connection; executions on a connection
    are strictly sequential) immediately before the statement is compiled and
    executed, because SQLAlchemy computes an execution's options before
    dispatching ``before_execute`` — see the module docstring.
    """
    conn.info[_CORE_SANCTION_INFO_KEY] = _CoreSanction(_EXECUTION_SANCTION_TOKEN, statement, tables)


def _revoke_core_sanction(conn: Connection) -> None:
    """Drop any outstanding grant on ``conn``.

    Called at the top of every ``before_execute`` so a grant can never
    outlive the statement it was issued for, even if compilation raises
    between the two events.
    """
    conn.info.pop(_CORE_SANCTION_INFO_KEY, None)


def _core_sanctioned_tables(conn: Connection, context: ExecutionContext | None) -> frozenset[str]:
    """Return the fact tables ``context``'s execution was granted, else empty.

    Identity-matched against the grant's statement object, so a stale or
    forged entry admits nothing: ``exec_driver_sql`` (no ``compiled``, no
    ``invoked_statement``) and any other statement never match.
    """
    granted = conn.info.get(_CORE_SANCTION_INFO_KEY)
    if (
        context is None
        or not isinstance(granted, _CoreSanction)
        or granted.token is not _EXECUTION_SANCTION_TOKEN
    ):
        return frozenset()
    if getattr(context, "invoked_statement", None) is granted.statement:
        return granted.tables
    if getattr(context, "compiled", None) is granted.statement:
        return granted.tables
    return frozenset()


def mark_rewritten_subquery(subquery: Subquery) -> Subquery:
    """Mark a rewriter-built subquery so the reference walker prunes it.

    The marker is a module-private token stored as an instance attribute; it
    survives clause-adaption cloning. Only :mod:`backend.db.asof` calls this,
    on the versioned subqueries it constructs itself.
    """
    subquery.__dict__[_SUBQUERY_MARKER_ATTR] = _SUBQUERY_SANCTION_TOKEN
    return subquery


def _is_rewriter_subquery(element: object) -> bool:
    """True when ``element`` (or its clone lineage) was built by the as-of rewriter."""
    return getattr(element, _SUBQUERY_MARKER_ATTR, None) is _SUBQUERY_SANCTION_TOKEN


def scan_sql_text(sql: str, table_names: frozenset[str]) -> frozenset[str]:
    """Return every fact-table name appearing as an identifier-like word in ``sql``.

    Conservative, fail-closed: a case-insensitive word-boundary match, so a
    string/column literal spelling a fact-table name is a (documented,
    acceptable) false positive. Over-matching produces a loud error, never an
    unversioned read.
    """
    return frozenset(
        name for name in table_names if re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE)
    )


def collect_references(element: ClauseElement, table_names: frozenset[str]) -> References:
    """Collect bitemporal fact-table references anywhere in a clause tree.

    Traversal follows ``get_children()`` and each column's owning table (so
    column-only selects are seen), descends into every subquery — a subquery's
    *name* buys no exemption — and prunes only subqueries carrying the
    module-private rewriter token (:func:`mark_rewritten_subquery`). Textual
    fragments (``text()``, ``literal_column``) are opaque-and-dangerous: their
    text is name-scanned and matches reported as ``textual`` (fail-closed).
    """
    plain: set[str] = set()
    aliased_refs: set[str] = set()
    textual: set[str] = set()
    seen: set[int] = set()

    def visit(elem: ClauseElement) -> None:
        if id(elem) in seen:
            return
        seen.add(id(elem))
        if isinstance(elem, Subquery) and _is_rewriter_subquery(elem):
            return
        if isinstance(elem, TextClause):
            textual.update(scan_sql_text(elem.text, table_names))
            return
        if isinstance(elem, Alias):
            inner = elem.element
            if isinstance(inner, TableClause) and inner.name in table_names:
                aliased_refs.add(inner.name)
                return
        if isinstance(elem, TableClause):
            if elem.name in table_names:
                plain.add(elem.name)
            return
        if isinstance(elem, ColumnClause):
            if elem.is_literal:
                textual.update(scan_sql_text(elem.name, table_names))
            if elem.table is not None:
                visit(elem.table)
        for child in elem.get_children():
            visit(child)

    visit(element)
    return References(frozenset(plain), frozenset(aliased_refs), frozenset(textual))


def collect_dml_read_references(statement: UpdateBase, table_names: frozenset[str]) -> References:
    """Collect fact-table references appearing in *read* capacity inside DML.

    The DML **target table itself is exempt** — writing a fact table is the
    writer path's job (INSERT) or dies at the append-only database triggers
    (UPDATE/DELETE); neither returns versioned data. Everything else is a
    read and is collected fail-closed:

    - any embedded select (``INSERT ... FROM SELECT``, scalar subqueries in
      SET/VALUES/WHERE, ``EXISTS``) is walked in full with
      :func:`collect_references` — inside a select even the target table's
      name counts, because ``INSERT INTO fact ... SELECT ... FROM fact`` is
      an unversioned read of ``fact``;
    - direct references to a *different* fact table (e.g. an implicit
      UPDATE..FROM join) are collected;
    - textual fragments are name-scanned as ``textual`` references.
    """
    target = statement.table
    target_name = target.name if isinstance(target, TableClause) else None
    plain: set[str] = set()
    aliased_refs: set[str] = set()
    textual: set[str] = set()
    seen: set[int] = set()

    def visit(elem: ClauseElement) -> None:
        if id(elem) in seen:
            return
        seen.add(id(elem))
        if isinstance(elem, SelectBase | Subquery):
            refs = collect_references(elem, table_names)
            plain.update(refs.plain)
            aliased_refs.update(refs.aliased)
            textual.update(refs.textual)
            return
        if isinstance(elem, TextClause):
            textual.update(scan_sql_text(elem.text, table_names))
            return
        if isinstance(elem, Alias):
            inner = elem.element
            if isinstance(inner, TableClause) and inner.name in table_names:
                aliased_refs.add(inner.name)
                return
        if isinstance(elem, TableClause):
            if elem.name in table_names and elem.name != target_name:
                plain.add(elem.name)
            return
        if isinstance(elem, ColumnClause):
            if elem.is_literal:
                textual.update(scan_sql_text(elem.name, table_names))
            if elem.table is not None:
                visit(elem.table)
        for child in elem.get_children():
            visit(child)

    visit(statement)
    return References(frozenset(plain), frozenset(aliased_refs), frozenset(textual))


def _raise_textual(mentioned: frozenset[str]) -> None:
    """Raise the fail-closed textual-SQL rejection naming the matched tables."""
    msg = (
        f"SQL text references bitemporal fact table(s) {', '.join(sorted(mentioned))}: "
        "textual/raw SQL against fact tables is blocked at the engine, fail-closed "
        "(conservative word-boundary name scan — false positives on e.g. string or "
        "column literals spelling a fact-table name are accepted by design). Reads "
        "go through backend.db.as_of() (D-011/I1); sanctioned infrastructure work "
        "(migrations, test reset) uses the private migration engine inside backend/db"
    )
    raise BitemporalBypassError(msg)


def _raise_unsanctioned_sql(tables: frozenset[str]) -> None:
    """Raise the default-deny rejection for unsanctioned SQL naming fact tables."""
    msg = (
        f"unsanctioned SQL naming bitemporal fact table(s) {', '.join(sorted(tables))} "
        "reached the driver: executions are denied at the SQL boundary unless "
        "something explicitly vetted and sanctioned them (the as_of() rewrite, an "
        "ORM column load, or a plain fact-table write vetted at before_execute). "
        "A structural blind spot therefore fails loudly instead of reading "
        "unversioned. This is a conservative word-boundary name scan over the "
        "final compiled SQL, so false positives (a string or column literal "
        "spelling a fact-table name, TRUNCATE of a fact table, DDL naming one) "
        "are accepted by design. Reads go through backend.db.as_of() (D-011/I1); "
        "sanctioned infrastructure work (migrations, test reset) uses the private "
        "migration engine inside backend/db"
    )
    raise BitemporalBypassError(msg)


def _raise_structural(tables: frozenset[str], capacity: str) -> None:
    """Raise the Core-guard rejection for a compiled statement reading fact tables."""
    msg = (
        f"unversioned Core-level {capacity} of bitemporal fact table(s) "
        f"{', '.join(sorted(tables))}: this execution did not come from the "
        "as_of() rewrite. Reads of bitemporal fact tables must go through "
        "backend.db.as_of() (D-011/I1)"
    )
    raise BitemporalBypassError(msg)


# ARG001 below: SQLAlchemy dispatches event listeners positionally against a
# fixed signature — every parameter must be accepted whether or not the guard
# reads it. Renaming or dropping one silently breaks listener registration.
def _core_before_execute(
    conn: Connection,
    clauseelement: Executable,
    multiparams: Sequence[Mapping[str, Any]],  # noqa: ARG001
    params: Mapping[str, Any],  # noqa: ARG001
    execution_options: Mapping[str, Any],
) -> None:
    """``before_execute`` guard: vet each statement, and sanction the writes.

    Runs before compilation, so detected violations raise at the most
    informative point (no I/O, full clause tree in hand). It is *not* the
    arbiter: :func:`_core_before_cursor_execute` denies by default on the
    final SQL, and this hook's job is to hand out the narrow positive
    sanction that lets legitimate writes through it.

    Per statement kind:

    - already sanctioned by the as-of rewrite / ORM column load: nothing to
      do (the token travels in ``execution_options``);
    - ``text()``: name-scanned and rejected on match, fail-closed;
    - DDL: vetted no further and **not** sanctioned — see the module
      docstring (``sqlalchemy.schema.DDL`` carries arbitrary SQL);
    - DML: an embedded fact-table read raises; otherwise, if the target is
      itself a fact table, the execution is **granted** a sanction naming
      exactly that table, which is what lets the ingestion writer's
      ``INSERT ... VALUES`` reach Postgres and lets ``UPDATE``/``DELETE``
      reach the append-only triggers that refuse them;
    - anything else (selects): a detected fact-table reference raises, and a
      clean statement is deliberately **not** sanctioned, so its final SQL is
      still scanned.
    """
    _revoke_core_sanction(conn)
    if _is_sanctioned_execution(execution_options):
        return
    table_names = fact_table_names()
    if not table_names:
        return
    element: object = clauseelement
    if not isinstance(element, ClauseElement):
        # e.g. a pre-built Compiled object: vet its source statement.
        element = getattr(clauseelement, "statement", None)
        if element is None or not isinstance(element, ClauseElement):
            return
    if isinstance(element, TextClause):
        mentioned = scan_sql_text(element.text, table_names)
        if mentioned:
            _raise_textual(mentioned)
        return
    if isinstance(element, ExecutableDDLElement):
        return
    if isinstance(element, UpdateBase):
        refs = collect_dml_read_references(element, table_names)
        if refs:
            _raise_structural(refs.all_tables(), "DML-embedded read")
        target = element.table
        if isinstance(target, TableClause) and target.name in table_names:
            _grant_core_sanction(conn, clauseelement, frozenset({target.name}))
        return
    refs = collect_references(element, table_names)
    if refs:
        _raise_structural(refs.all_tables(), "read")


def _core_before_cursor_execute(
    conn: Connection,
    cursor: DBAPICursor,  # noqa: ARG001
    statement: str,
    parameters: object,  # noqa: ARG001
    context: ExecutionContext | None,
    executemany: bool,  # noqa: ARG001
) -> None:
    """``before_cursor_execute`` guard: the default-deny arbiter on the final SQL.

    Every execution reaching the driver is name-scanned, with **no
    statement-type skip**: compiled clause elements, ``text()``,
    ``exec_driver_sql`` and driver-level paths alike. SQL naming a bitemporal
    fact table executes only when the execution carries a module-private
    sanction — the as-of rewrite / ORM column-load token in its execution
    options, or a ``before_execute`` grant that names the very tables found.
    Everything else raises :class:`BitemporalBypassError`.

    This is what makes the structural walker non-load-bearing: a clause shape
    :func:`collect_references` cannot see is never sanctioned, so it is
    refused here on the strength of the SQL text alone instead of executing
    unversioned.
    """
    if context is not None and _is_sanctioned_execution(context.execution_options):
        return
    mentioned = scan_sql_text(statement, fact_table_names())
    if not mentioned:
        return
    if not mentioned - _core_sanctioned_tables(conn, context):
        return
    _raise_unsanctioned_sql(mentioned)


def install_core_guard(engine: Engine) -> None:
    """Register the Core-level bitemporal read guard on a (sync) engine.

    Called by :mod:`backend.db.engine` for every engine it creates — the
    process-wide application engine and every admin engine. Both halves are
    required: ``before_execute`` vets and sanctions, ``before_cursor_execute``
    denies by default on the final SQL. The only engine deliberately created
    *without* the guard is the module-private migration engine (alembic runs,
    sanctioned test/db reset), which is not importable outside ``backend/db``.
    """
    event.listen(engine, "before_execute", _core_before_execute)
    event.listen(engine, "before_cursor_execute", _core_before_cursor_execute)
