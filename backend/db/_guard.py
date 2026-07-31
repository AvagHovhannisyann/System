"""Shared bitemporal read-reference analysis and the Core-level engine guard (D-011).

**Import contract (D-011 bypass-prevention layer 3):** this module is private
to ``backend/db``. Ruff rule TID251 bans importing it anywhere else in the
codebase; the only public handle on its behavior is the
:class:`BitemporalBypassError` re-exported by :mod:`backend.db`.

Two enforcement surfaces consume this module:

- the ORM ``do_orm_execute`` hook in :mod:`backend.db.asof` (layer 2), which
  uses :func:`collect_references` / :func:`collect_dml_read_references` to
  find bitemporal fact-table reads in ORM statements before rewriting or
  rejecting them; and
- the **Core-level engine guard** (:func:`install_core_guard`), registered on
  every engine :mod:`backend.db.engine` creates (application and admin). It
  fires on ``before_execute`` (compiled clause elements and ``text()``) and
  ``before_cursor_execute`` (the final SQL string, catching
  ``exec_driver_sql`` and any other textual path) and raises
  :class:`BitemporalBypassError` when the outgoing statement references a
  bitemporal fact table in a read capacity. Raw ``Connection`` access —
  ``session.connection()``, the admin engine, hand-built engines are not
  covered but sessions are — therefore cannot read fact tables unversioned.

Sanctioned executions: the as-of rewrite in :mod:`backend.db.asof` produces
statements that legitimately reference fact tables (inside the versioned
subquery form). The rewrite hook marks those executions with a module-private
**token object** under :data:`SANCTION_EXECUTION_OPTION`
(:func:`sanctioned_execution_options`); the Core guard admits an execution
only when the option value ``is`` that exact object. Knowing the option
*name* is useless — forging admission requires the token object itself, which
only this banned-outside-``backend/db`` module holds. Similarly, the
rewriter's own versioned subqueries are marked with a second private token
via :func:`mark_rewritten_subquery` (an instance attribute, which survives
SQLAlchemy's clause-adaption cloning) so the reference walker can prune them;
a user naming a subquery ``_bitemporal_anything`` gets no exemption.

Textual SQL cannot be structurally analyzed, so it is name-scanned: a
conservative case-insensitive word-boundary regex over the SQL string
(:func:`scan_sql_text`). This is fail-closed by design — a false positive
(e.g. a string literal or column literal spelling a fact-table name, or
``TRUNCATE price_bar``) raises :class:`BitemporalBypassError` even though no
versioned read occurs. That is acceptable: the failure mode is a loud error,
never a leak, and sanctioned infrastructure work (migrations, test resets)
runs on the unguarded module-private migration engine instead.
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


class BitemporalBypassError(RuntimeError):
    """A statement referenced a bitemporal fact table in a read capacity, unversioned.

    D-011 bypass prevention. Raised before any row leaves the database, by
    either enforcement surface:

    - the class-level ORM ``do_orm_execute`` hook (:mod:`backend.db.asof`) —
      for any :class:`sqlalchemy.orm.Session` in the process whose SELECT (or
      DML-embedded read) touches a fact table with no bound as-of; and
    - the Core-level engine guard (:func:`install_core_guard`) — for compiled
      Core statements and textual SQL on every engine ``backend.db`` creates,
      catching ``session.connection()``, the admin engine, and
      ``exec_driver_sql``.

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
    conn: Connection,  # noqa: ARG001
    clauseelement: Executable,
    multiparams: Sequence[Mapping[str, Any]],  # noqa: ARG001
    params: Mapping[str, Any],  # noqa: ARG001
    execution_options: Mapping[str, Any],
) -> None:
    """``before_execute`` guard: vet every compiled/textual statement pre-compilation.

    Admits executions sanctioned by the as-of rewrite (module-private token in
    ``execution_options``), plain DDL (not a read; sanctioned DDL runs on the
    unguarded migration engine anyway), and DML whose only fact-table
    reference is its own write target. Everything else touching a fact table
    raises :class:`BitemporalBypassError` before any I/O.
    """
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
        return
    refs = collect_references(element, table_names)
    if refs:
        _raise_structural(refs.all_tables(), "read")


def _core_before_cursor_execute(
    conn: Connection,  # noqa: ARG001
    cursor: DBAPICursor,  # noqa: ARG001
    statement: str,
    parameters: object,  # noqa: ARG001
    context: ExecutionContext | None,
    executemany: bool,  # noqa: ARG001
) -> None:
    """``before_cursor_execute`` guard: name-scan the final SQL string.

    Second net for textual paths: catches ``exec_driver_sql`` (which never
    fires ``before_execute``) and re-checks ``text()`` statements. Statements
    compiled from clause elements were already structurally vetted (or
    sanctioned) in ``before_execute`` and are skipped — the structural check
    is strictly stronger than a string scan.
    """
    if context is not None and _is_sanctioned_execution(context.execution_options):
        return
    compiled = getattr(context, "compiled", None)
    source = getattr(compiled, "statement", None)
    if source is not None and not isinstance(source, TextClause):
        return
    mentioned = scan_sql_text(statement, fact_table_names())
    if mentioned:
        _raise_textual(mentioned)


def install_core_guard(engine: Engine) -> None:
    """Register the Core-level bitemporal read guard on a (sync) engine.

    Called by :mod:`backend.db.engine` for every engine it creates — the
    process-wide application engine and every admin engine. The only engine
    deliberately created *without* the guard is the module-private migration
    engine (alembic runs, sanctioned test/db reset), which is not importable
    outside ``backend/db``.
    """
    event.listen(engine, "before_execute", _core_before_execute)
    event.listen(engine, "before_cursor_execute", _core_before_cursor_execute)
