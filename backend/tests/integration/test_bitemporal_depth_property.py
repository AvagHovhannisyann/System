"""P2.6-depth gate: DEPTH as an explicit Hypothesis parameter for the as-of layer.

The P2.6 suite in ``test_bitemporal_property.py`` randomizes *data*
(securities, valid intervals, knowledge times, retractions, as-of instants)
but holds the query *shape* fixed and shallow: its ~81k cases measure breadth
at roughly one nesting level. The as-of rewriter
(:mod:`backend.db.asof`) substitutes every bitemporal entity with a two-level
versioned subquery; nothing proved that the substitution still applies when
the fact reference is buried several levels down.

This module supplies that missing axis. A recursive query-shape strategy
composes shapes level by level — CTEs (single, chained, multiple), subqueries
in FROM, scalar subqueries in WHERE and in the columns clause, EXISTS/IN
subqueries, joins of derived tables against the non-bitemporal ``security``
anchor and against other derived tables, UNION/INTERSECT set operations, and
aggregates over derived tables — over both ORM-entity selects and
column selects, including mixed shapes joining a bitemporal entity to the
``security`` anchor and to the second bitemporal table
(``security_master``). ``DEPTH`` is an explicit parameter of the strategy,
and the depth actually reached is *measured* structurally (not assumed) and
published.

Every generated shape is executed through :func:`backend.db.as_of` against
the real TimescaleDB container (the session-scoped testcontainers fixture in
``conftest.py``), using only the sanctioned public API, and asserted on three
axes:

(a) **I1** — every returned row satisfies ``knowledge_time <= as_of``.  Every
    composable shape family here is *identity-preserving* on the projection
    ``(security_id, valid_from, knowledge_time, close)`` by construction, so
    the check is not merely "no future knowledge" but full set equality
    against an independent in-memory oracle: the exact visible set at that
    as-of instant, no row more and no row less.
(b) **The rewrite applied at every level** — a ``before_cursor_execute``
    listener records the final compiled SQL actually sent to the driver, and
    every fact-table reference in it must sit inside the versioned subquery
    form (a ``DISTINCT ON`` select over the fact table governed by a
    ``knowledge_time <=`` predicate). A raw ``FROM price_bar`` at any nesting
    level with no knowledge-time predicate governing it is a failure, and so
    is a fact table appearing in a JOIN/comma position.
(c) **Refusals fail closed** — shapes the rewriter declines (user
    ``aliased()`` entities, ``text()`` fragments, ``literal_column``
    fragments, top-level compound selects) must raise
    ``BitemporalRewriteError``/``BitemporalBypassError`` *and* must not have
    sent any SQL naming a fact table to the driver. A raise is an acceptable
    outcome; a silent unversioned read is not.

Accounting: a session-scoped accumulator records per-depth case counts, the
achieved maximum depth, and the rewritten-vs-refused split per shape family;
it publishes them as junit ``p26_depth_*`` testsuite properties (matching the
existing ``p26_*`` counters) and **fails at teardown** if the achieved depth
or the case counts fall below the floors, so the evidence cannot silently
regress.

**Why this module exists — the leak it caught (P2.11, now fixed).** The depth
axis was not a formality: on its first run it found a real I1 leak in the
rewriter that the ~76k-case breadth suite was structurally incapable of
seeing. When a statement touched *two* bitemporal tables and referenced the
first in two places, ``_rewrite_select`` ran a second clause adapter over an
already substituted tree; the clone left several distinct ``BindParameter``
objects carrying the as-of value, and SQLAlchemy's compiled-statement cache
then resolved only some of them from the current execution. The outermost
versioned subquery kept the as-of of whichever execution first compiled that
shape, so re-running one query across a rebalance calendar read the store as
of the wrong instant — returning future knowledge whenever the earlier
compilation used a later as-of. That is the exact shape Phases 4-6 write, and
it survived every other defence because the emitted SQL was the *correct*
versioned form; only the bind value was wrong.

The fix (one shared, explicitly-keyed as-of bind) plus its two fail-closed
backstops are covered by ``test_asof_bind_integrity.py``;
:func:`test_repeated_as_of_execution_does_not_reuse_a_stale_bind` below is the
minimal deterministic reproduction and is kept as a permanent regression test.
Do not weaken it, and do not let the depth floors drift down: the defect lived
in shapes that only appear once the generator is allowed to nest.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import re
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import event, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
from sqlalchemy.sql.elements import ColumnClause
from sqlalchemy.sql.selectable import CTE, SelectBase, Subquery, TableClause

from backend.db import (
    BitemporalBypassError,
    BitemporalRewriteError,
    as_of,
    dispose_database,
    ingest_writer_session,
)
from backend.db.models import PriceBar, Security, SecurityMaster
from backend.tests.integration.factories import create_security

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from sqlalchemy.engine.interfaces import DBAPICursor, ExecutionContext
    from sqlalchemy.sql import ClauseElement, Select

_Relation = Subquery | CTE
"""A derived relation exposing the four probe columns ``sid/vf/kt/close``."""

_FACT_TABLES = frozenset({PriceBar.__tablename__, SecurityMaster.__tablename__})
"""Bitemporal fact tables whose every SQL reference must be versioned."""

# --- population ------------------------------------------------------------

_KT_BASE = dt.datetime(2016, 1, 4, tzinfo=dt.UTC)
"""Origin for knowledge-time offsets; a decade in the past (backfill-style)."""

_VF_BASE = dt.datetime(2016, 1, 4, tzinfo=dt.UTC)
"""Origin for the event-time (``valid_from``) grid."""

_ONE_DAY = dt.timedelta(days=1)
_ONE_US = dt.timedelta(microseconds=1)

_N_SECURITIES = 3

_MASTER_KNOWLEDGE = _KT_BASE - 30 * _ONE_DAY
"""Identity rows are knowable before every as-of probe, so the master join is
identity-preserving on the price-bar projection at every probe."""


@dataclasses.dataclass(frozen=True)
class _Version:
    """One version of one fact, in index space."""

    security_index: int
    day: int
    knowledge_index: int
    close_cents: int
    is_retraction: bool = False


_POPULATION: tuple[_Version, ...] = (
    # security 0
    _Version(0, 0, 0, 10_000),
    _Version(0, 0, 3, 10_100),  # correction
    _Version(0, 1, 1, 20_000),
    _Version(0, 2, 2, 30_000),
    _Version(0, 2, 5, 30_000, is_retraction=True),  # retraction wins from kt5
    # security 1
    _Version(1, 0, 0, 40_000),
    _Version(1, 0, 4, 40_000, is_retraction=True),
    _Version(1, 0, 7, 40_100),  # retract-then-reassert
    _Version(1, 1, 6, 50_000),
    _Version(1, 2, 8, 60_000),
    # security 2
    _Version(2, 0, 9, 70_000),
    _Version(2, 1, 2, 80_000),
    _Version(2, 1, 10, 80_100),
    _Version(2, 2, 11, 90_000, is_retraction=True),  # never visible
)
"""Fixed population: corrections, retractions, retract-then-reassert, a
retraction-only fact, and facts that first become knowable late. Data
randomization is the existing P2.6 suite's axis; this module's axis is shape
depth, so the population is deterministic and the *shape* is drawn."""

_KNOWLEDGE_INDICES = tuple(sorted({version.knowledge_index for version in _POPULATION}))


def _knowledge_time(index: int) -> dt.datetime:
    """Knowledge time for grid index ``index`` (one day apart, UTC)."""
    return _KT_BASE + index * _ONE_DAY


def _valid_from(day: int) -> dt.datetime:
    """Event-time interval start for grid day ``day`` (UTC)."""
    return _VF_BASE + day * _ONE_DAY


def _as_of_probes() -> tuple[dt.datetime, ...]:
    """As-of instants: every knowledge boundary at -1us / exact / +1us, plus ends."""
    probes: set[dt.datetime] = {
        _KT_BASE - 2 * _ONE_DAY,
        _knowledge_time(max(_KNOWLEDGE_INDICES)) + 2 * _ONE_DAY,
    }
    for index in _KNOWLEDGE_INDICES:
        boundary = _knowledge_time(index)
        probes |= {boundary - _ONE_US, boundary, boundary + _ONE_US}
    return tuple(sorted(probes))


_AS_OF_PROBES = _as_of_probes()

_ProbeRow = tuple[int, dt.datetime, dt.datetime, int]
"""One visible fact as ``(security_id, valid_from, knowledge_time, close_cents)``."""


def _oracle(security_ids: Sequence[int], as_of_ts: dt.datetime) -> frozenset[_ProbeRow]:
    """Independent in-memory oracle: the visible set at ``as_of_ts``.

    Latest ``knowledge_time <= as_of`` wins per (logical key, ``valid_from``)
    — unique by construction, mirroring the composite primary key — and a
    winning retraction hides the fact (D-011).
    """
    groups: dict[tuple[int, int], list[_Version]] = {}
    for version in _POPULATION:
        groups.setdefault((version.security_index, version.day), []).append(version)
    visible: set[_ProbeRow] = set()
    for (security_index, day), versions in groups.items():
        knowable = [
            version for version in versions if _knowledge_time(version.knowledge_index) <= as_of_ts
        ]
        if not knowable:
            continue
        winner = max(knowable, key=lambda version: version.knowledge_index)
        if winner.is_retraction:
            continue
        visible.add(
            (
                security_ids[security_index],
                _valid_from(day),
                _knowledge_time(winner.knowledge_index),
                winner.close_cents,
            )
        )
    return frozenset(visible)


def _build_bar(version: _Version, security_ids: Sequence[int]) -> PriceBar:
    """Materialize one version as an ORM row (cents -> USD Decimal)."""
    valid_from = _valid_from(version.day)
    price = Decimal(version.close_cents) / Decimal(100)
    return PriceBar(
        security_id=security_ids[version.security_index],
        valid_from=valid_from,
        valid_to=valid_from + _ONE_DAY,
        knowledge_time=_knowledge_time(version.knowledge_index),
        is_retraction=version.is_retraction,
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal(1),
        volume_shares=1000,
    )


async def _seed() -> list[int]:
    """Create the identity anchors, one master version each, and the bars."""
    security_ids = [await create_security() for _ in range(_N_SECURITIES)]
    async with ingest_writer_session() as session:
        session.add_all(
            [
                SecurityMaster(
                    security_id=security_id,
                    ticker=f"T{index}",
                    name=f"T{index} Corp.",
                    exchange="XNAS",
                    valid_from=_VF_BASE - 365 * _ONE_DAY,
                    knowledge_time=_MASTER_KNOWLEDGE,
                )
                for index, security_id in enumerate(security_ids)
            ]
        )
        session.add_all([_build_bar(version, security_ids) for version in _POPULATION])
        await session.commit()
    return security_ids


# --- final-SQL capture and versioned-form verification ---------------------


class _SQLRecorder:
    """Records the final compiled SQL — and bind values — of every execution.

    Bind values are kept because the SQL text alone cannot show whether the
    ``knowledge_time <= :as_of`` predicate was given *this* query's as-of
    instant: a statement whose text is perfectly versioned can still be
    executed with a stale bound value.
    """

    def __init__(self) -> None:
        self.executions: list[tuple[str, object]] = []

    def clear(self) -> None:
        """Drop everything recorded so far."""
        self.executions.clear()

    def record(self, statement: str, parameters: object) -> None:
        """Append one execution exactly as handed to the DBAPI cursor."""
        self.executions.append((statement, parameters))

    @property
    def statements(self) -> list[str]:
        """The recorded statement strings, in execution order."""
        return [statement for statement, _ in self.executions]

    def fact_executions(self) -> list[tuple[str, object]]:
        """Return the recorded executions whose SQL names a bitemporal fact table."""
        return [
            execution
            for execution in self.executions
            if any(_word_matches(execution[0], name) for name in _FACT_TABLES)
        ]

    def fact_statements(self) -> list[str]:
        """Return the recorded statements whose SQL names a bitemporal fact table."""
        return [statement for statement, _ in self.fact_executions()]

    def digest(self) -> str:
        """A compact rendering of the recorded fact-table executions."""
        return "\n".join(
            f"--- SQL ---\n{statement}\n--- BINDS ---\n{parameters!r}"
            for statement, parameters in self.fact_executions()
        )


def _word_matches(sql: str, name: str) -> bool:
    """True when ``name`` appears in ``sql`` as a whole identifier-like word."""
    return re.search(rf"\b{re.escape(name)}\b", sql, re.IGNORECASE) is not None


def _enclosing_groups(sql: str, index: int) -> Iterator[str]:
    """Yield the parenthesized groups enclosing ``index``, innermost first.

    Assumes no parentheses inside string literals, which holds for the
    statements this module inspects (the as-of instant travels as a bind
    parameter, and identifiers never contain parentheses). A literal
    containing an unbalanced parenthesis would only ever make the scan
    *stricter*, never laxer.
    """
    cursor = index
    while True:
        depth = 0
        start = -1
        for position in range(cursor - 1, -1, -1):
            char = sql[position]
            if char == ")":
                depth += 1
            elif char == "(":
                if depth == 0:
                    start = position
                    break
                depth -= 1
        if start < 0:
            return
        depth = 0
        end = len(sql)
        for position in range(start, len(sql)):
            char = sql[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        yield sql[start:end]
        cursor = start


def _versioned_form_violations(sql: str, table: str) -> list[str]:
    """Return a description of every unversioned reference to ``table`` in ``sql``.

    Two independent rules, both fail-closed:

    1. every *bare* occurrence of the table name (a table reference rather
       than a ``table.column`` qualifier) must be in ``FROM`` position — a
       fact table reached through ``JOIN`` or a comma-join was not
       substituted by the rewriter;
    2. every occurrence at all must be enclosed by a subquery that *is* the
       versioned form: it selects ``DISTINCT ON`` from the fact table under a
       ``knowledge_time <=`` predicate. An occurrence with no such enclosing
       group is an unversioned read at whatever nesting level it sits.
    """
    violations: list[str] = []
    quoted = re.escape(table)
    from_reference = re.compile(rf"\bFROM\s+{quoted}\b", re.IGNORECASE)
    knowledge_predicate = re.compile(rf"\b{quoted}\.knowledge_time\s*<=", re.IGNORECASE)
    distinct_on = re.compile(rf"\bDISTINCT\s+ON\s*\(\s*{quoted}\.", re.IGNORECASE)

    for match in re.finditer(rf"\b{quoted}\b(?!\s*\.)", sql):
        preceding = sql[: match.start()]
        if not re.search(r"\bFROM\s+$", preceding, re.IGNORECASE):
            context = sql[max(0, match.start() - 60) : match.end() + 20]
            violations.append(
                f"{table} referenced outside FROM position at offset {match.start()} "
                f"(JOIN/comma-join of the raw fact table): ...{context}..."
            )

    for match in re.finditer(rf"\b{quoted}\b", sql):
        for group in _enclosing_groups(sql, match.start()):
            if not from_reference.search(group):
                continue
            if knowledge_predicate.search(group) and distinct_on.search(group):
                break
            violations.append(
                f"{table} read at offset {match.start()} sits in a subquery with no "
                f"governing knowledge_time predicate: {group[:400]}"
            )
            break
        else:
            violations.append(
                f"{table} referenced at offset {match.start()} with no enclosing "
                f"versioned subquery (unversioned read at nesting level 0)"
            )
    return violations


def _assert_every_fact_read_is_versioned(recorder: _SQLRecorder) -> int:
    """Assert the versioned form on every recorded statement; return how many.

    Returns the number of statements that named a fact table, so callers can
    assert the check was not vacuous.
    """
    fact_statements = recorder.fact_statements()
    for statement in fact_statements:
        for table in sorted(_FACT_TABLES):
            if not _word_matches(statement, table):
                continue
            violations = _versioned_form_violations(statement, table)
            assert not violations, (
                "as-of rewrite did not reach every level — unversioned fact-table "
                "read in the SQL actually sent to the database:\n"
                + "\n".join(violations)
                + f"\n\nfull statement:\n{statement}"
            )
    return len(fact_statements)


# --- structural depth measurement ------------------------------------------


def _fact_reference_levels(element: ClauseElement) -> list[int]:
    """Nesting level of every raw fact-table reference in a clause tree.

    Level 1 is the top-level SELECT; descending into a nested ``SelectBase``
    (a subquery, a CTE body, an EXISTS/IN/scalar subquery, or one leg of a
    compound select) increments it. Traversal mirrors
    ``backend.db._guard.collect_references``: ``get_children()`` plus each
    column's owning table, so column-only selects are seen. Measured on the
    statement *before* rewriting, so it reports the depth at which the author
    buried the fact reference.
    """
    levels: list[int] = []
    seen: set[tuple[int, int]] = set()

    def visit(elem: ClauseElement, level: int) -> None:
        key = (id(elem), level)
        if key in seen:
            return
        seen.add(key)
        if isinstance(elem, SelectBase):
            level += 1
        if isinstance(elem, TableClause):
            if elem.name in _FACT_TABLES:
                levels.append(level)
            return
        if isinstance(elem, ColumnClause) and elem.table is not None:
            visit(elem.table, level)
        for child in elem.get_children():
            visit(child, level)

    visit(element, 0)
    return levels


# --- shape algebra ---------------------------------------------------------

_BASE_FAMILIES = (
    "base_columns",
    "base_join_anchor",
    "base_join_master",
    "base_join_derived_self",
)
"""Depth-1 relations reading ``price_bar``, including mixed shapes joining
the non-bitemporal ``security`` anchor and the second bitemporal table."""

_WRAPPER_FAMILIES = (
    "cte",
    "chained_cte",
    "multi_cte",
    "subquery_from",
    "scalar_where_correlated",
    "scalar_select",
    "exists_filter",
    "in_filter",
    "join_anchor",
    "join_derived_and_anchor",
    "union_distinct",
    "intersect",
    "aggregate_group",
)
"""Composable wrappers. Every one is identity-preserving on the
``(sid, vf, kt, close)`` projection *by construction*, which is what lets the
oracle be exact set equality at every depth rather than a weaker property."""

_TOP_FAMILIES = ("top_columns", "top_orm_entity")
"""Top-level statement flavors: a column select over the final derived
relation, and an ORM-entity select over ``PriceBar`` filtered by it."""

_REFUSAL_FAMILIES = (
    "poison_aliased_entity",
    "poison_text_fragment",
    "poison_literal_column",
    "poison_compound_top",
)
"""Shapes the rewriter documents as unsupported; each must fail closed."""


@dataclasses.dataclass(frozen=True)
class _ShapeSpec:
    """One generated query shape, independent of the data it runs against."""

    base: str
    wrappers: tuple[str, ...]
    top: str
    refusal: str | None
    refusal_at: int
    as_of_indices: tuple[int, ...]

    def composed_families(self) -> tuple[str, ...]:
        """The rewritable families this spec composes (base, wrappers, top).

        Deliberately excludes :attr:`refusal`: a refused shape is refused
        *because of* its poison element, not because of the CTE it happens to
        sit inside, and merging the two would make the reported buckets read
        as though ordinary families were sometimes rejected.
        """
        return (self.base, *self.wrappers, self.top)


class _ShapeBuilder:
    """Builds a SQLAlchemy statement from a :class:`_ShapeSpec`.

    Names are generated per build so a shape may compose the same wrapper
    family several times without colliding aliases.
    """

    def __init__(self, spec: _ShapeSpec) -> None:
        self._spec = spec
        self._counter = 0

    def _name(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def _probe_columns(self, relation: _Relation) -> list[Any]:
        return [
            relation.c.sid.label("sid"),
            relation.c.vf.label("vf"),
            relation.c.kt.label("kt"),
            relation.c.close.label("close"),
        ]

    def _entity_columns(self) -> list[Any]:
        return [
            PriceBar.security_id.label("sid"),
            PriceBar.valid_from.label("vf"),
            PriceBar.knowledge_time.label("kt"),
            PriceBar.close_usd.label("close"),
        ]

    def base_relation(self, kind: str) -> _Relation:
        """Build a depth-1 derived relation over ``price_bar``."""
        if kind == "base_join_anchor":
            statement = select(*self._entity_columns()).join_from(
                PriceBar, Security, Security.security_id == PriceBar.security_id
            )
        elif kind == "base_join_master":
            statement = select(*self._entity_columns()).join_from(
                PriceBar,
                SecurityMaster,
                SecurityMaster.security_id == PriceBar.security_id,
            )
        elif kind == "base_join_derived_self":
            inner = (
                select(PriceBar.security_id.label("sid")).distinct().subquery(self._name("selfsq"))
            )
            statement = select(*self._entity_columns()).join_from(
                PriceBar, inner, inner.c.sid == PriceBar.security_id
            )
        elif kind == "poison_aliased_entity":
            alias = aliased(PriceBar)
            statement = select(
                alias.security_id.label("sid"),
                alias.valid_from.label("vf"),
                alias.knowledge_time.label("kt"),
                alias.close_usd.label("close"),
            )
        else:
            statement = select(*self._entity_columns())
        return statement.subquery(self._name("base"))

    def _fresh_base(self) -> _Relation:
        """A second, independent depth-1 relation for inner correlated reads."""
        return select(*self._entity_columns()).subquery(self._name("aux"))

    def wrap(self, kind: str, relation: _Relation) -> _Relation:
        """Wrap ``relation`` in one more level of the named shape family."""
        columns = self._probe_columns(relation)
        if kind == "cte":
            return select(*columns).cte(self._name("cte"))
        if kind == "chained_cte":
            first = select(*columns).cte(self._name("cte"))
            return select(*self._probe_columns(first)).cte(self._name("cte"))
        if kind == "multi_cte":
            rows = select(*columns).cte(self._name("cte"))
            latest = (
                select(rows.c.sid.label("sid"), func.max(rows.c.kt).label("mkt"))
                .group_by(rows.c.sid)
                .cte(self._name("cte"))
            )
            return (
                select(*self._probe_columns(rows))
                .join_from(rows, latest, latest.c.sid == rows.c.sid)
                .subquery(self._name("sq"))
            )
        if kind == "subquery_from":
            return select(*columns).subquery(self._name("sq"))
        if kind == "scalar_where_correlated":
            aux = self._fresh_base()
            newest = (
                select(func.max(aux.c.kt))
                .where(aux.c.sid == relation.c.sid, aux.c.vf == relation.c.vf)
                .correlate(relation)
                .scalar_subquery()
            )
            return select(*columns).where(relation.c.kt == newest).subquery(self._name("sq"))
        if kind == "scalar_select":
            aux = self._fresh_base()
            total = select(func.count()).select_from(aux).correlate(None).scalar_subquery()
            return select(*columns, total.label("extra")).subquery(self._name("sq"))
        if kind == "exists_filter":
            aux = self._fresh_base()
            present = (
                select(aux.c.sid).where(aux.c.sid == relation.c.sid).correlate(relation).exists()
            )
            return select(*columns).where(present).subquery(self._name("sq"))
        if kind == "in_filter":
            aux = self._fresh_base()
            known = select(aux.c.sid).correlate(None)
            return select(*columns).where(relation.c.sid.in_(known)).subquery(self._name("sq"))
        if kind == "join_anchor":
            return (
                select(*columns)
                .join_from(relation, Security, Security.security_id == relation.c.sid)
                .subquery(self._name("sq"))
            )
        if kind == "join_derived_and_anchor":
            aux = self._fresh_base()
            counted = (
                select(aux.c.sid.label("sid"), func.count().label("n"))
                .group_by(aux.c.sid)
                .subquery(self._name("agg"))
            )
            return (
                select(*columns)
                .join_from(relation, counted, counted.c.sid == relation.c.sid)
                .join(Security, Security.security_id == relation.c.sid)
                .subquery(self._name("sq"))
            )
        if kind == "union_distinct":
            aux = self._fresh_base()
            return (
                select(*columns).union(select(*self._probe_columns(aux))).subquery(self._name("sq"))
            )
        if kind == "intersect":
            aux = self._fresh_base()
            return (
                select(*columns)
                .intersect(select(*self._probe_columns(aux)))
                .subquery(self._name("sq"))
            )
        if kind == "aggregate_group":
            return (
                select(
                    relation.c.sid.label("sid"),
                    relation.c.vf.label("vf"),
                    relation.c.kt.label("kt"),
                    func.max(relation.c.close).label("close"),
                )
                .group_by(relation.c.sid, relation.c.vf, relation.c.kt)
                .subquery(self._name("sq"))
            )
        if kind == "poison_text_fragment":
            return (
                select(*columns)
                .where(sa.text(f"{PriceBar.__tablename__}.close_usd >= 0"))
                .subquery(self._name("sq"))
            )
        if kind == "poison_literal_column":
            fragment: Any = sa.literal_column(f"{PriceBar.__tablename__}.close_usd")
            return select(*columns, fragment.label("extra")).subquery(self._name("sq"))
        msg = f"unknown shape family {kind!r}"
        raise AssertionError(msg)

    def build(self) -> Select[Any]:
        """Compose the whole shape into one executable statement."""
        spec = self._spec
        base_kind = (
            "poison_aliased_entity" if spec.refusal == "poison_aliased_entity" else spec.base
        )
        relation = self.base_relation(base_kind)
        injectable = spec.refusal in {"poison_text_fragment", "poison_literal_column"}
        for index, wrapper in enumerate(spec.wrappers):
            relation = self.wrap(wrapper, relation)
            if injectable and index == spec.refusal_at:
                relation = self.wrap(str(spec.refusal), relation)
        if injectable and spec.refusal_at >= len(spec.wrappers):
            relation = self.wrap(str(spec.refusal), relation)

        if spec.top == "top_orm_entity":
            statement = select(PriceBar).where(
                sa.tuple_(PriceBar.security_id, PriceBar.valid_from).in_(
                    select(relation.c.sid, relation.c.vf)
                )
            )
        else:
            statement = select(*self._probe_columns(relation))
        if spec.refusal == "poison_compound_top":
            return statement.union_all(statement)  # type: ignore[return-value]
        return statement


def _build(spec: _ShapeSpec) -> Select[Any]:
    """Build the statement for ``spec`` with a fresh name generator."""
    return _ShapeBuilder(spec).build()


# --- strategies ------------------------------------------------------------

MAX_WRAPPERS = 5
"""Default upper bound on the ``depth`` parameter of :func:`_rewritable_specs`.

``depth`` here counts composable wrapper levels; the *measured* nesting depth
of the resulting statement is higher, because the base relation and the
top-level statement are levels of their own, ``chained_cte``/``multi_cte`` add
two levels each, and a wrapper's correlated inner read sits two levels below
its host. Five wrappers therefore reach a measured depth in the low teens —
comfortably past the floor of 4 this module enforces.
"""

MAX_REFUSAL_WRAPPERS = 3
"""Depth bound for refusal shapes: shallower, because a refusal is a single
binary outcome per shape rather than a row-by-row comparison, so breadth over
insertion points buys more than extra depth."""


_AS_OF_SCHEDULES = st.lists(
    st.integers(min_value=0, max_value=len(_AS_OF_PROBES) - 1),
    min_size=2,
    max_size=3,
    unique=True,
).map(tuple)
"""At least two *distinct* as-of instants per shape, so every example
re-executes one statement across the calendar — the pattern that exposes an
as-of value frozen into a cached compiled form."""


@st.composite
def _wrapper_chains(draw: st.DrawFn, max_depth: int) -> tuple[str, ...]:
    """Draw a wrapper chain of an explicitly drawn ``depth`` in ``[0, max_depth]``.

    The chain length is drawn first, and uniformly, so depth is a first-class
    dimension of the search rather than a by-product of a list strategy's
    size distribution (which is heavily biased towards short lists — exactly
    the shallow coverage this module exists to fix).
    """
    depth = draw(st.integers(min_value=0, max_value=max_depth))
    return tuple(draw(st.sampled_from(_WRAPPER_FAMILIES)) for _ in range(depth))


@st.composite
def _rewritable_specs(draw: st.DrawFn, max_depth: int = MAX_WRAPPERS) -> _ShapeSpec:
    """Draw a shape the rewriter is expected to handle, at a drawn depth."""
    return _ShapeSpec(
        base=draw(st.sampled_from(_BASE_FAMILIES)),
        wrappers=draw(_wrapper_chains(max_depth)),
        top=draw(st.sampled_from(_TOP_FAMILIES)),
        refusal=None,
        refusal_at=0,
        as_of_indices=draw(_AS_OF_SCHEDULES),
    )


@st.composite
def _refused_specs(draw: st.DrawFn, max_depth: int = MAX_REFUSAL_WRAPPERS) -> _ShapeSpec:
    """Draw a shape carrying an unsupported element at a drawn depth."""
    wrappers = draw(_wrapper_chains(max_depth))
    return _ShapeSpec(
        base=draw(st.sampled_from(_BASE_FAMILIES)),
        wrappers=wrappers,
        top=draw(st.sampled_from(_TOP_FAMILIES)),
        refusal=draw(st.sampled_from(_REFUSAL_FAMILIES)),
        refusal_at=draw(st.integers(min_value=0, max_value=len(wrappers))),
        as_of_indices=draw(_AS_OF_SCHEDULES),
    )


# --- accounting ------------------------------------------------------------

_DEPTH_FLOOR = 4
_CASE_FLOOR = 2_000
_REFUSAL_FLOOR = 60
_PER_DEPTH_FLOOR = 20
"""Floors asserted at teardown so the depth evidence cannot silently regress."""


class _DepthLedger:
    """Session-wide accounting of depth reached and shape-family outcomes."""

    def __init__(self) -> None:
        self.cases = 0
        self.cases_by_depth: dict[int, int] = {}
        self.shapes_by_depth: dict[int, int] = {}
        self.rewritten_families: dict[str, int] = {}
        self.families_in_refused_shapes: dict[str, int] = {}
        self.refusal_causes: dict[str, int] = {}
        self.refused_shapes = 0
        self.rewritten_shapes = 0
        self.max_depth = 0

    def record_shape(self, spec: _ShapeSpec, depth: int, *, refused: bool, cases: int) -> None:
        """Record one executed shape: its depth, its bucket, its case count.

        Three separate tallies, because collapsing them would misreport the
        result: ``rewritten_families`` counts families that were rewritten and
        verified; ``refusal_causes`` counts the *reason* a shape was refused;
        ``families_in_refused_shapes`` counts the ordinary families that were
        merely composed around a refused element (they are not themselves
        rejected — the same families appear in the rewritten bucket).
        """
        self.max_depth = max(self.max_depth, depth)
        self.cases += cases
        self.cases_by_depth[depth] = self.cases_by_depth.get(depth, 0) + cases
        self.shapes_by_depth[depth] = self.shapes_by_depth.get(depth, 0) + 1
        bucket = self.families_in_refused_shapes if refused else self.rewritten_families
        for family in spec.composed_families():
            bucket[family] = bucket.get(family, 0) + 1
        if refused:
            self.refused_shapes += 1
            cause = spec.refusal or "unspecified"
            self.refusal_causes[cause] = self.refusal_causes.get(cause, 0) + 1
        else:
            self.rewritten_shapes += 1


@pytest.fixture(scope="session")
def depth_ledger(
    record_testsuite_property: Callable[[str, object], None],
) -> Iterator[_DepthLedger]:
    """Accumulate depth evidence; publish it and enforce the floors at teardown."""
    ledger = _DepthLedger()
    started = time.monotonic()
    yield ledger
    elapsed = time.monotonic() - started

    record_testsuite_property("p26_depth_max_depth_reached", ledger.max_depth)
    record_testsuite_property("p26_depth_cases", ledger.cases)
    record_testsuite_property("p26_depth_shapes_rewritten", ledger.rewritten_shapes)
    record_testsuite_property("p26_depth_shapes_refused", ledger.refused_shapes)
    record_testsuite_property("p26_depth_seconds", round(elapsed, 1))
    for depth in sorted(ledger.cases_by_depth):
        record_testsuite_property(f"p26_depth_cases_at_depth_{depth}", ledger.cases_by_depth[depth])
        record_testsuite_property(
            f"p26_depth_shapes_at_depth_{depth}", ledger.shapes_by_depth[depth]
        )
    for family, count in sorted(ledger.rewritten_families.items()):
        record_testsuite_property(f"p26_depth_rewritten_family_{family}", count)
    for family, count in sorted(ledger.refusal_causes.items()):
        record_testsuite_property(f"p26_depth_refusal_cause_{family}", count)
    for family, count in sorted(ledger.families_in_refused_shapes.items()):
        record_testsuite_property(f"p26_depth_family_in_refused_shape_{family}", count)

    print(
        f"\nP2.6-depth: max measured nesting depth {ledger.max_depth}; "
        f"{ledger.cases} asserted cases across "
        f"{ledger.rewritten_shapes} rewritten + {ledger.refused_shapes} refused shapes "
        f"in {elapsed:.1f}s"
    )
    print(f"  cases per depth:   {dict(sorted(ledger.cases_by_depth.items()))}")
    print(f"  shapes per depth:  {dict(sorted(ledger.shapes_by_depth.items()))}")
    print(f"  rewritten families:{dict(sorted(ledger.rewritten_families.items()))}")
    print(f"  refusal causes:    {dict(sorted(ledger.refusal_causes.items()))}")
    print(
        f"  families composed around a refused element: "
        f"{dict(sorted(ledger.families_in_refused_shapes.items()))}"
    )

    assert ledger.max_depth >= _DEPTH_FLOOR, (
        f"depth evidence regressed: deepest fact reference exercised was nesting level "
        f"{ledger.max_depth}, floor is {_DEPTH_FLOOR}. Run the full module, not a subset."
    )
    assert ledger.cases >= _CASE_FLOOR, (
        f"depth evidence regressed: {ledger.cases} asserted cases, floor is {_CASE_FLOOR}."
    )
    assert ledger.refused_shapes >= _REFUSAL_FLOOR, (
        f"fail-closed evidence regressed: {ledger.refused_shapes} refused shapes asserted, "
        f"floor is {_REFUSAL_FLOOR}."
    )
    shallow = {
        depth: ledger.cases_by_depth.get(depth, 0)
        for depth in range(1, _DEPTH_FLOOR + 1)
        if ledger.cases_by_depth.get(depth, 0) < _PER_DEPTH_FLOOR
    }
    assert not shallow, (
        f"depth coverage is not uniform: depths with fewer than {_PER_DEPTH_FLOOR} "
        f"asserted cases: {shallow}"
    )


@pytest.fixture
def sql_recorder() -> Iterator[_SQLRecorder]:
    """Record the final SQL of every execution, on every engine in the process.

    Registered on the :class:`sqlalchemy.engine.Engine` *class*, so it sees
    the process-wide application engine whenever it is (re)built, and sees the
    statement string exactly as handed to the DBAPI cursor — after compilation
    and after the as-of rewrite.
    """
    recorder = _SQLRecorder()

    def listener(
        conn: object,  # noqa: ARG001 — SQLAlchemy dispatches positionally
        cursor: DBAPICursor,  # noqa: ARG001
        statement: str,
        parameters: object,
        context: ExecutionContext | None,  # noqa: ARG001
        executemany: bool,  # noqa: ARG001
    ) -> None:
        recorder.record(statement, parameters)

    event.listen(Engine, "before_cursor_execute", listener)
    try:
        yield recorder
    finally:
        event.remove(Engine, "before_cursor_execute", listener)


@pytest.fixture
def db_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """One event loop shared by every Hypothesis example of a test.

    The process-wide engine binds its pool to the loop that first uses it, so
    all examples must run on a single loop; teardown disposes the engine on
    that same loop before closing it.
    """
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        try:
            loop.run_until_complete(dispose_database())
        finally:
            loop.close()


@pytest.fixture
def hypothesis_population(db_loop: asyncio.AbstractEventLoop) -> list[int]:
    """Seed the fixed population once for all Hypothesis examples of a test.

    Sync counterpart of :func:`seeded_population`: a Hypothesis test is a
    plain function, so its examples run on ``db_loop`` and the population must
    be created on that same loop (the engine pool binds to whichever loop
    first uses it).
    """
    return db_loop.run_until_complete(_seed())


@pytest.fixture
async def seeded_population() -> list[int]:
    """Seed the fixed population on the running test's own event loop."""
    return await _seed()


# --- execution -------------------------------------------------------------


def _observed_rows(spec: _ShapeSpec, rows: Sequence[Any]) -> set[_ProbeRow]:
    """Normalize a result set to the ``(sid, vf, kt, close_cents)`` projection."""
    if spec.top == "top_orm_entity":
        return {
            (bar.security_id, bar.valid_from, bar.knowledge_time, int(bar.close_usd * 100))
            for (bar,) in rows
        }
    return {
        (int(sid), vf, kt, int(close * 100)) for sid, vf, kt, close in (row[:4] for row in rows)
    }


async def _run_rewritable(
    spec: _ShapeSpec,
    security_ids: Sequence[int],
    recorder: _SQLRecorder,
    ledger: _DepthLedger,
) -> None:
    """Execute one expected-rewritable shape at each of its as-of instants.

    The *same statement* is re-executed across as-of instants on purpose: that
    is how a backtest walks a rebalance calendar, and it is the only way to
    catch an as-of value that gets frozen into a cached compiled form instead
    of travelling with the execution. Each execution is judged independently
    against the oracle, so a stale bind shows up as a divergence on the second
    and later instants rather than as an untested assumption.
    """
    statement = _build(spec)
    depth = max(_fact_reference_levels(statement))
    cases = 0

    for as_of_index in spec.as_of_indices:
        as_of_ts = _AS_OF_PROBES[as_of_index]
        expected = _oracle(security_ids, as_of_ts)

        recorder.clear()
        async with as_of(as_of_ts) as session:
            rows = list((await session.execute(statement)).all())
        actual = _observed_rows(spec, rows)

        for _sid, _vf, knowledge_time, _close in sorted(actual):
            # (a) The I1 core property, asserted independently per returned row.
            assert knowledge_time <= as_of_ts, (
                f"I1 violation at nesting depth {depth}: returned knowledge_time "
                f"{knowledge_time.isoformat()} > as_of {as_of_ts.isoformat()}\n"
                f"spec: {spec}\nas-of schedule: "
                f"{[_AS_OF_PROBES[i].isoformat() for i in spec.as_of_indices]}\n"
                f"{recorder.digest()}"
            )
            cases += 1
        # Every wrapper is identity-preserving on this projection by
        # construction, so the oracle comparison is exact set equality rather
        # than a one-sided "no future knowledge" check.
        assert actual == expected, (
            f"as-of visible set diverges from the oracle at nesting depth {depth}.\n"
            f"spec: {spec}\nas_of: {as_of_ts.isoformat()}\n"
            f"as-of schedule: {[_AS_OF_PROBES[i].isoformat() for i in spec.as_of_indices]}\n"
            f"missing: {sorted(expected - actual)}\nunexpected: {sorted(actual - expected)}\n"
            f"{recorder.digest()}"
        )
        cases += len(expected) + 1

        # (b) The rewrite reached every level of the SQL actually sent.
        inspected = _assert_every_fact_read_is_versioned(recorder)
        assert inspected >= 1, (
            f"no statement naming a fact table was recorded for {spec}; the "
            "versioned-form check would have been vacuous"
        )
        cases += inspected

    ledger.record_shape(spec, depth, refused=False, cases=cases)


async def _run_refused(
    spec: _ShapeSpec,
    recorder: _SQLRecorder,
    ledger: _DepthLedger,
) -> None:
    """Execute one expected-refused shape and assert (c): fail closed, no SQL sent."""
    statement = _build(spec)
    depth = max(_fact_reference_levels(statement))
    cases = 0

    for as_of_index in spec.as_of_indices:
        as_of_ts = _AS_OF_PROBES[as_of_index]
        recorder.clear()
        async with as_of(as_of_ts) as session:
            with pytest.raises((BitemporalRewriteError, BitemporalBypassError)):
                await session.execute(statement)

        leaked = recorder.fact_statements()
        assert not leaked, (
            f"refused shape at nesting depth {depth} still sent SQL naming a fact table "
            f"to the database — a refusal must be fail-closed, never a partial read.\n"
            f"spec: {spec}\nstatements: {leaked}"
        )
        cases += 2

    ledger.record_shape(spec, depth, refused=True, cases=cases)


# --- tests -----------------------------------------------------------------


@given(spec=_rewritable_specs())
@settings(
    max_examples=220,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    print_blob=True,
)
def test_recursive_shapes_stay_versioned_at_every_depth(
    db_loop: asyncio.AbstractEventLoop,
    hypothesis_population: list[int],
    sql_recorder: _SQLRecorder,
    depth_ledger: _DepthLedger,
    spec: _ShapeSpec,
) -> None:
    """Randomly composed nested shapes never read a fact table unversioned.

    Sharing ``db_loop`` (and the engine pool bound to it) across examples is
    deliberate; the population is seeded once and every example queries it at
    a drawn as-of instant with a drawn shape, so isolation comes from the
    statement, not from the data. Hence the suppressed function-scoped-fixture
    health check.
    """
    db_loop.run_until_complete(
        _run_rewritable(spec, hypothesis_population, sql_recorder, depth_ledger)
    )


@given(spec=_refused_specs())
@settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    print_blob=True,
)
def test_unsupported_shapes_fail_closed_at_every_depth(
    db_loop: asyncio.AbstractEventLoop,
    hypothesis_population: list[int],  # noqa: ARG001 — data must exist for a leak to be observable
    sql_recorder: _SQLRecorder,
    depth_ledger: _DepthLedger,
    spec: _ShapeSpec,
) -> None:
    """Shapes the rewriter refuses raise, and send no fact-table SQL at all."""
    db_loop.run_until_complete(_run_refused(spec, sql_recorder, depth_ledger))


_DETERMINISTIC_CHAINS: tuple[tuple[str, ...], ...] = (
    (),
    ("cte",),
    ("cte", "subquery_from"),
    ("cte", "subquery_from", "exists_filter"),
    ("chained_cte", "multi_cte", "join_derived_and_anchor"),
    ("subquery_from", "union_distinct", "aggregate_group", "scalar_where_correlated"),
    ("join_anchor", "in_filter", "cte", "aggregate_group"),
    ("cte", "cte", "cte", "cte", "cte"),
    ("multi_cte", "chained_cte", "intersect", "in_filter", "scalar_select"),
)
"""Fixed ladders guaranteeing depth coverage 1..N regardless of Hypothesis draws.

Between them these chains use **every** wrapper family, which
:func:`test_every_shape_family_is_executed_by_a_deterministic_ladder` asserts:
a family that is declared but only ever reached by a Hypothesis draw would
silently stop being covered the moment the property test aborts early.
"""


_LADDER_SCHEDULE = (0, len(_AS_OF_PROBES) // 3, 2 * len(_AS_OF_PROBES) // 3, len(_AS_OF_PROBES) - 1)
"""Four as-of instants walked in order by every deterministic shape."""


@pytest.mark.parametrize(
    "chain", _DETERMINISTIC_CHAINS, ids=lambda chain: f"w{len(chain)}_{'_'.join(chain) or 'none'}"
)
@pytest.mark.parametrize("top", _TOP_FAMILIES)
@pytest.mark.parametrize("base", _BASE_FAMILIES)
async def test_deterministic_depth_ladder_is_versioned(
    seeded_population: list[int],
    sql_recorder: _SQLRecorder,
    depth_ledger: _DepthLedger,
    base: str,
    chain: tuple[str, ...],
    top: str,
) -> None:
    """A fixed ladder of nesting levels, each walked across four as-of instants.

    Guarantees the per-depth floors are met by construction — the Hypothesis
    test above widens the coverage, it does not carry it alone.
    """
    spec = _ShapeSpec(
        base=base,
        wrappers=chain,
        top=top,
        refusal=None,
        refusal_at=0,
        as_of_indices=_LADDER_SCHEDULE,
    )
    await _run_rewritable(spec, seeded_population, sql_recorder, depth_ledger)


async def test_depth_one_orm_entity_select_is_versioned(
    seeded_population: list[int],
    sql_recorder: _SQLRecorder,
    depth_ledger: _DepthLedger,
) -> None:
    """The shallow control: a bare ``select(PriceBar)`` at every as-of probe.

    Depth 1 is what the existing P2.6 suite already covers; carrying it here
    too makes the per-depth ladder start at the bottom rung, so a regression
    that only affects depth 1 is still visible in this module's counters.
    """
    statement = select(PriceBar)
    for as_of_ts in _AS_OF_PROBES:
        expected = _oracle(seeded_population, as_of_ts)
        sql_recorder.clear()
        async with as_of(as_of_ts) as session:
            bars = list((await session.scalars(statement)).all())
        actual = {
            (bar.security_id, bar.valid_from, bar.knowledge_time, int(bar.close_usd * 100))
            for bar in bars
        }
        for _sid, _vf, knowledge_time, _close in sorted(actual):
            assert knowledge_time <= as_of_ts, (
                f"I1 violation at depth 1: {knowledge_time.isoformat()} > "
                f"{as_of_ts.isoformat()}\n{sql_recorder.digest()}"
            )
        assert actual == expected, (
            f"depth-1 visible set diverges from the oracle at {as_of_ts.isoformat()}\n"
            f"missing: {sorted(expected - actual)}\nunexpected: {sorted(actual - expected)}"
        )
        inspected = _assert_every_fact_read_is_versioned(sql_recorder)
        assert inspected >= 1
        spec = _ShapeSpec(
            base="base_columns",
            wrappers=(),
            top="top_orm_plain",
            refusal=None,
            refusal_at=0,
            as_of_indices=(),
        )
        depth_ledger.record_shape(
            spec, 1, refused=False, cases=len(actual) + len(expected) + 1 + inspected
        )


@pytest.mark.parametrize("refusal", _REFUSAL_FAMILIES)
@pytest.mark.parametrize(
    "chain",
    _DETERMINISTIC_CHAINS[:5],
    ids=lambda chain: f"w{len(chain)}_{'_'.join(chain) or 'none'}",
)
async def test_deterministic_refusals_fail_closed(
    seeded_population: list[int],  # noqa: ARG001 — data must exist for a leak to be observable
    sql_recorder: _SQLRecorder,
    depth_ledger: _DepthLedger,
    refusal: str,
    chain: tuple[str, ...],
) -> None:
    """Every documented-unsupported element refuses at every rung of the ladder."""
    spec = _ShapeSpec(
        base="base_columns",
        wrappers=chain,
        top="top_columns",
        refusal=refusal,
        refusal_at=len(chain),
        as_of_indices=(0, len(_AS_OF_PROBES) - 1),
    )
    await _run_refused(spec, sql_recorder, depth_ledger)


async def test_repeated_as_of_execution_does_not_reuse_a_stale_bind(
    seeded_population: list[int],
    sql_recorder: _SQLRecorder,
) -> None:
    """Minimal deterministic repro: one statement, two as-of instants, one leak.

    A backtest walks a rebalance calendar by re-executing the *same* query at
    many as-of instants. The as-of value must travel with each execution. This
    shape — an ORM-entity select over ``price_bar`` filtered by a subquery that
    joins ``price_bar`` to ``security_master`` — is the smallest one where it
    does not: the first execution's as-of gets frozen into the outermost
    versioned subquery's bind, and every later execution reads the store as of
    that first instant. When the first instant is the later one, the second
    execution returns rows whose ``knowledge_time`` is in its own future
    (invariant I1). Reached through nothing but ``backend.db.as_of`` and public
    ORM constructs.
    """
    joined = (
        select(PriceBar.security_id.label("sid"), PriceBar.valid_from.label("vf"))
        .join_from(SecurityMaster, PriceBar, PriceBar.security_id == SecurityMaster.security_id)
        .subquery("stale_probe")
    )
    statement = select(PriceBar).where(
        sa.tuple_(PriceBar.security_id, PriceBar.valid_from).in_(select(joined.c.sid, joined.c.vf))
    )

    later = _knowledge_time(max(_KNOWLEDGE_INDICES)) + _ONE_DAY
    earlier = _knowledge_time(1)
    leaks: list[str] = []
    for as_of_ts in (later, earlier):
        sql_recorder.clear()
        async with as_of(as_of_ts) as session:
            bars = list((await session.scalars(statement)).all())
        expected = _oracle(seeded_population, as_of_ts)
        actual = {
            (bar.security_id, bar.valid_from, bar.knowledge_time, int(bar.close_usd * 100))
            for bar in bars
        }
        future = sorted(
            knowledge_time for _s, _v, knowledge_time, _c in actual if knowledge_time > as_of_ts
        )
        if future or actual != expected:
            leaks.append(
                f"as_of={as_of_ts.isoformat()}: "
                f"future-knowledge rows {[k.isoformat() for k in future]}, "
                f"missing {sorted(expected - actual)}, unexpected {sorted(actual - expected)}\n"
                f"{sql_recorder.digest()}"
            )
    assert not leaks, (
        "I1 violation: re-executing one statement at a second as-of instant reused "
        "the first instant's bound as-of value in the outermost versioned subquery.\n"
        + "\n\n".join(leaks)
    )


async def test_depth_probe_detects_an_unversioned_read() -> None:
    """The versioned-form verifier is not vacuous: it fails on a raw read.

    Guards the whole (b) axis. If ``_versioned_form_violations`` could not
    tell a raw fact-table read from the versioned form, every assertion above
    would pass regardless of what the rewriter did.
    """
    versioned = (
        "SELECT anon.close_usd FROM (SELECT DISTINCT ON (price_bar.security_id) "
        "price_bar.close_usd AS close_usd FROM price_bar "
        "WHERE price_bar.knowledge_time <= %(k)s) AS anon"
    )
    assert _versioned_form_violations(versioned, "price_bar") == []

    raw_top_level = "SELECT price_bar.close_usd FROM price_bar"
    assert _versioned_form_violations(raw_top_level, "price_bar")

    raw_nested = (
        "SELECT anon.close_usd FROM "
        "(SELECT price_bar.close_usd AS close_usd FROM price_bar) AS anon"
    )
    assert _versioned_form_violations(raw_nested, "price_bar")

    joined_raw = (
        "SELECT anon.c FROM (SELECT DISTINCT ON (price_bar.security_id) price_bar.close_usd AS c "
        "FROM price_bar JOIN price_bar ON true WHERE price_bar.knowledge_time <= %(k)s) AS anon"
    )
    assert _versioned_form_violations(joined_raw, "price_bar")


def test_depth_measurement_matches_hand_built_shapes() -> None:
    """The depth metric is measured, not assumed: check it on known shapes."""
    assert max(_fact_reference_levels(select(PriceBar))) == 1

    one = select(PriceBar.security_id.label("sid")).subquery("one")
    assert max(_fact_reference_levels(select(one.c.sid))) == 2

    first = select(PriceBar.security_id.label("sid")).cte("k1")
    second = select(first.c.sid.label("sid")).cte("k2")
    third = select(second.c.sid.label("sid")).cte("k3")
    assert max(_fact_reference_levels(select(third.c.sid))) == 4

    spec = _ShapeSpec(
        base="base_columns",
        wrappers=("chained_cte", "multi_cte", "join_derived_and_anchor"),
        top="top_orm_entity",
        refusal=None,
        refusal_at=0,
        as_of_indices=(0,),
    )
    assert max(_fact_reference_levels(_build(spec))) >= _DEPTH_FLOOR


def test_every_shape_family_is_executed_by_a_deterministic_ladder() -> None:
    """No shape family may be declared and then never actually executed.

    Two separate claims. First, every wrapper family appears in
    ``_DETERMINISTIC_CHAINS``, so it is executed against the database by the
    parametrized ladder rather than only when a Hypothesis draw happens to
    pick it — a property test that aborts early (as it does while the
    stale-bind defect stands) must not silently take a family's coverage down
    with it. Second, every family combination actually builds and buries its
    fact reference at least one level deep.
    """
    laddered = {wrapper for chain in _DETERMINISTIC_CHAINS for wrapper in chain}
    assert laddered == set(_WRAPPER_FAMILIES), (
        "wrapper families never executed by a deterministic ladder: "
        f"{sorted(set(_WRAPPER_FAMILIES) - laddered)}; families in a ladder but not "
        f"declared: {sorted(laddered - set(_WRAPPER_FAMILIES))}"
    )

    covered: set[str] = set()
    for wrapper in _WRAPPER_FAMILIES:
        for base in _BASE_FAMILIES:
            for top in _TOP_FAMILIES:
                spec = _ShapeSpec(
                    base=base,
                    wrappers=(wrapper,),
                    top=top,
                    refusal=None,
                    refusal_at=0,
                    as_of_indices=(0,),
                )
                statement = _build(spec)
                assert max(_fact_reference_levels(statement)) >= 2
                covered |= set(spec.composed_families())
    assert covered >= set(_WRAPPER_FAMILIES) | set(_BASE_FAMILIES) | set(_TOP_FAMILIES)

    for refusal in _REFUSAL_FAMILIES:
        spec = _ShapeSpec(
            base="base_columns",
            wrappers=("cte",),
            top="top_columns",
            refusal=refusal,
            refusal_at=1,
            as_of_indices=(0,),
        )
        assert _build(spec) is not None
