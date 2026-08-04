"""P2.3 backstop: the as-of *value* is verified, never trusted (D-011/I1).

Companion to ``test_bitemporal_depth_property.py``, which proves the rewriter
produces the versioned *shape* at every nesting level. Shape is only half of
invariant I1: a statement can be perfectly versioned and still be executed
with the wrong as-of instant, which returns rows whose ``knowledge_time`` is
in the query's own future just as surely as an unversioned read does.

That is not hypothetical. It shipped. ``_rewrite_select`` used to give every
touched fact table its own ``sa.literal()`` as-of parameter; with a second
fact table present, the second clause-adapter pass cloned a tree that already
contained the first table's versioned subquery, and because
``BindParameter._clone`` re-keys *anonymous* binds, one as-of value became
several parameters with unrelated keys. Two rewrites of one shape at
different as-of instants compile to equal cache keys, so the second execution
hit SQLAlchemy's compiled cache — where the compiler's cache-key-to-bind
matching (which pairs by ``.key`` across the clone lineage) could not relate
one parameter, and ``construct_params`` fell back to the value baked into the
cached ``Compiled`` object: the *first* execution's as-of.

The root cause is fixed (one shared, explicitly-keyed bind per rewrite), but
a fix alone would leave the class of defect open — any future rewriter change
could reintroduce a wrong value in a new way. So the value is checked twice,
and this module tests the checks rather than the fix:

- **post-rewrite** (``backend.db.asof._assert_single_as_of_bind``): the
  rewritten statement must carry exactly one distinct as-of bind key, holding
  exactly this session's as-of;
- **at the cursor boundary** (``backend.db._guard._verify_as_of_binds``): the
  values *actually resolved for this execution* must equal the as-of that
  sanctioned it — read from ``context.compiled_parameters``, which
  ``construct_params`` recomputes per execution, so a stale value served by a
  cache hit is visible to it.

Every test here breaks the rewriter on purpose — monkeypatching only
``backend.db`` internals, and neutering a check only where a test must reach
past one layer to prove the next is not dead code — then asserts the same two
things: a fail-closed raise, and **nothing executed**. Fail-closure is
observed on ``after_cursor_execute``, which fires only if the driver actually
ran the statement, so the evidence does not depend on listener ordering.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING, Any, cast

import pytest
import sqlalchemy as sa
from sqlalchemy import event, inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased

from backend.db import AsOfBindIntegrityError, as_of
from backend.db import _guard as guard_module
from backend.db import asof as asof_module
from backend.db._guard import AS_OF_BIND_KEY, mark_as_of_bind, mark_rewritten_subquery
from backend.db.models import PriceBar, SecurityMaster
from backend.tests.integration.factories import (
    bar_version,
    create_security,
    insert_rows,
    master_version,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from sqlalchemy.engine.interfaces import DBAPICursor, ExecutionContext
    from sqlalchemy.orm.util import AliasedClass, AliasedInsp
    from sqlalchemy.sql import Select
    from sqlalchemy.sql.elements import ColumnElement

    from backend.db.bitemporal import BitemporalMixin

_DAY = dt.date(2024, 1, 5)

_EARLY = dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC)
"""Knowledge time of the original bar."""

_LATE = dt.datetime(2024, 1, 9, 9, 0, tzinfo=dt.UTC)
"""Knowledge time of the correction — invisible at :data:`_EARLY_PROBE`."""

_LATE_PROBE = dt.datetime(2024, 1, 10, tzinfo=dt.UTC)
"""An as-of after both versions: the correction is visible."""

_EARLY_PROBE = dt.datetime(2024, 1, 6, tzinfo=dt.UTC)
"""An as-of between them: only the original is visible. Reading this instant
with :data:`_LATE_PROBE`'s bind is the I1 violation under test."""

_MASTER_KNOWLEDGE = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
"""Identity rows are knowable before every probe, so the master join never
changes which price bars are visible."""


@pytest.fixture
async def two_table_population() -> int:
    """Seed one security with a corrected bar and one identity version.

    The correction is what makes a stale as-of *observable*: reading
    :data:`_EARLY_PROBE` with :data:`_LATE_PROBE`'s bind returns the corrected
    close, whose ``knowledge_time`` is in that query's future.
    """
    security_id = await create_security()
    await insert_rows(
        master_version(security_id, "AAA", _MASTER_KNOWLEDGE),
        bar_version(security_id, _DAY, _EARLY, "100"),
        bar_version(security_id, _DAY, _LATE, "101"),
    )
    return security_id


def _two_table_statement() -> Select[Any]:
    """The minimal shape that used to leak: two fact tables, one referenced twice.

    An ORM-entity select over ``price_bar`` filtered by a subquery joining
    ``price_bar`` to ``security_master`` — identity-preserving on the price
    bars (every bar's security has a master version knowable before every
    probe), so any divergence is the as-of value, never the join.
    """
    joined = (
        select(PriceBar.security_id.label("sid"), PriceBar.valid_from.label("vf"))
        .join_from(SecurityMaster, PriceBar, PriceBar.security_id == SecurityMaster.security_id)
        .subquery("bind_integrity_probe")
    )
    return select(PriceBar).where(
        sa.tuple_(PriceBar.security_id, PriceBar.valid_from).in_(select(joined.c.sid, joined.c.vf))
    )


class _ExecutedSQL:
    """Statements naming a fact table that the driver actually ran."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def clear(self) -> None:
        """Drop everything recorded so far."""
        self.statements.clear()


@pytest.fixture
def executed_sql() -> Iterator[_ExecutedSQL]:
    """Record fact-table SQL on ``after_cursor_execute``, i.e. only if it ran.

    Deliberately not ``before_cursor_execute``: the boundary check raises
    *from* that event, and whether another listener on the same event already
    fired is an ordering detail. ``after_cursor_execute`` fires only once the
    driver has executed the statement, so an empty recording is direct
    evidence that nothing reached the database.
    """
    recorder = _ExecutedSQL()

    def listener(
        conn: object,  # noqa: ARG001 — SQLAlchemy dispatches listeners positionally
        cursor: DBAPICursor,  # noqa: ARG001
        statement: str,
        parameters: object,  # noqa: ARG001
        context: ExecutionContext | None,  # noqa: ARG001
        executemany: bool,  # noqa: ARG001
    ) -> None:
        if "price_bar" in statement or "security_master" in statement:
            recorder.statements.append(statement)

    event.listen(Engine, "after_cursor_execute", listener)
    try:
        yield recorder
    finally:
        event.remove(Engine, "after_cursor_execute", listener)


def _versioned_entity_with(
    cls: type[BitemporalMixin],
    predicate_bind: ColumnElement[Any],
    *,
    with_predicate: bool,
) -> AliasedClass[Any]:
    """Build a versioned entity the way the rewriter does, but breakably.

    Mirrors ``backend.db.asof._versioned_entity`` closely enough that the Core
    guard's structural analysis and its default-deny SQL scan behave exactly
    as they do for a real rewrite; the only differences are the ones a test
    injects — which bind object carries the as-of, or whether the
    ``knowledge_time`` predicate is present at all.
    """
    table = cast("sa.Table", cast("Any", cls).__table__)
    key_columns = [table.c[name] for name in cls.__bitemporal_key__]
    versions = select(table).distinct(*key_columns, table.c.valid_from)
    if with_predicate:
        versions = versions.where(table.c.knowledge_time <= predicate_bind)
    marked = mark_rewritten_subquery(
        versions.order_by(*key_columns, table.c.valid_from, table.c.knowledge_time.desc()).subquery(
            f"_bitemporal_versions__{table.name}"
        )
    )
    visible = mark_rewritten_subquery(
        select(marked).where(~marked.c.is_retraction).subquery(f"_bitemporal_visible__{table.name}")
    )
    return cast("AliasedClass[Any]", aliased(cast("Any", cls), visible, name=f"{table.name}_asof"))


def _install_broken_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    bind_factory: Callable[[dt.datetime, str], ColumnElement[Any]],
    *,
    with_predicate: bool = True,
) -> None:
    """Replace ``_rewrite_select`` with one built from ``bind_factory``.

    ``bind_factory(as_of_ts, table_name)`` returns the bind that table's
    ``knowledge_time`` predicate will use, so a test can hand out one shared
    bind, a fresh anonymous bind per table (the original defect), or a bind
    carrying the wrong value. Everything else — the two-level versioned
    subquery, the rewriter tokens, the sequential adapter passes — is the
    production shape, so only the injected fault is under test.
    """

    def broken_rewrite(
        statement: Select[Any],
        touched_tables: frozenset[str],
        as_of_ts: dt.datetime,
    ) -> Select[Any]:
        class_by_table = asof_module._bitemporal_class_by_table_name()
        replacements: dict[Any, AliasedClass[Any]] = {
            class_by_table[name]: _versioned_entity_with(
                class_by_table[name],
                bind_factory(as_of_ts, name),
                with_predicate=with_predicate,
            )
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

    monkeypatch.setattr(asof_module, "_rewrite_select", broken_rewrite)


def _anonymous_bind_per_table(as_of_ts: dt.datetime, table: str) -> ColumnElement[Any]:
    """The pre-fix bind factory, verbatim: a fresh anonymous literal per table.

    Unmarked, exactly as the shipped defect was — so the guard cannot even
    recognize these as as-of binds, which is itself a fail-closed condition
    (see :func:`test_unrecognizable_as_of_binds_are_refused_on_the_first_execution`).
    """
    del table
    return sa.literal(as_of_ts, sa.DateTime(timezone=True))


def _anonymous_marked_bind_per_table(as_of_ts: dt.datetime, table: str) -> ColumnElement[Any]:
    """The pre-fix mechanism, made *recognizable* so the value check is what fires.

    Identical to :func:`_anonymous_bind_per_table` — a fresh anonymous
    (``unique``) bind per table, which ``BindParameter._clone`` re-keys on
    every adapter pass, producing the several-parameters-one-value shape that
    lets the compiled cache resolve a stale value — except that the bind
    carries the rewriter's token. Without the token the guard refuses the very
    first execution for a different (also correct) reason, which would leave
    the stale-*value* direction untested.
    """
    del table
    return mark_as_of_bind(sa.literal(as_of_ts, sa.DateTime(timezone=True)))


def _disable_post_rewrite_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neuter the post-rewrite invariant so the *boundary* check is what is tested.

    Used only by tests whose subject is the second layer. Reaching past the
    first layer deliberately is the only way to prove the second one is not
    dead code: a defect the post-rewrite check happens to catch first would
    otherwise leave the boundary check untested, and its own failure silent.
    """

    def accept(statement: Select[Any], as_of_ts: dt.datetime) -> None:
        del statement, as_of_ts

    monkeypatch.setattr(asof_module, "_assert_single_as_of_bind", accept)


async def _rows_at(statement: Select[Any], as_of_ts: dt.datetime) -> list[PriceBar]:
    """Execute ``statement`` through the sanctioned read path at ``as_of_ts``."""
    async with as_of(as_of_ts) as session:
        return list((await session.scalars(statement)).all())


# --- the class of defect, end to end ---------------------------------------


async def test_stale_cached_as_of_bind_raises_instead_of_returning_future_knowledge(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The original defect, reintroduced on purpose, is refused at the boundary.

    Restores the pre-fix mechanism exactly — a fresh *anonymous* as-of bind
    per touched fact table — and neuters the post-rewrite invariant, so the
    only thing between SQLAlchemy's compiled cache and an I1 violation is
    ``_verify_as_of_binds``. The first execution (at the later instant)
    succeeds and populates the cache; the second (at the earlier instant)
    would otherwise return the corrected bar, whose ``knowledge_time`` is in
    its own future. It must raise instead, and nothing must execute.
    """
    _disable_post_rewrite_check(monkeypatch)
    _install_broken_rewrite(monkeypatch, _anonymous_marked_bind_per_table)
    statement = _two_table_statement()

    first = await _rows_at(statement, _LATE_PROBE)
    assert [bar.knowledge_time for bar in first] == [_LATE], (
        "the control execution must succeed and see the correction, otherwise the "
        "second execution proves nothing about a stale cached bind"
    )

    executed_sql.clear()
    with pytest.raises(AsOfBindIntegrityError) as raised:
        await _rows_at(statement, _EARLY_PROBE)

    assert "as-of bind" in str(raised.value)
    assert not executed_sql.statements, (
        "the refusal must be fail-closed: no SQL naming a fact table may reach the "
        f"driver once the mismatch is detected, got {executed_sql.statements}"
    )


async def test_stale_bind_leak_is_observable_when_the_backstop_is_removed(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backstop test above is not vacuous: without it, the leak returns rows.

    Same broken rewrite, with the boundary check disabled as well. This
    *demonstrates* the I1 violation — the second execution asks for
    ``_EARLY_PROBE`` and receives the bar knowable only at ``_LATE`` — which
    is what makes the previous test's raise meaningful rather than a
    coincidence of some unrelated error. It asserts the leak, so if
    SQLAlchemy's caching ever stops producing it, this test fails loudly and
    says the previous test needs a new way to reach the boundary check.
    """
    _disable_post_rewrite_check(monkeypatch)

    def accept(context: ExecutionContext | None) -> None:
        del context

    monkeypatch.setattr(guard_module, "_verify_as_of_binds", accept)
    _install_broken_rewrite(monkeypatch, _anonymous_bind_per_table)
    statement = _two_table_statement()

    await _rows_at(statement, _LATE_PROBE)
    leaked = await _rows_at(statement, _EARLY_PROBE)

    assert [bar.knowledge_time for bar in leaked] == [_LATE], (
        "expected the unguarded stale-bind leak to surface future knowledge at "
        f"{_EARLY_PROBE.isoformat()}, got {[bar.knowledge_time for bar in leaked]}"
    )
    assert leaked[0].knowledge_time > _EARLY_PROBE


# --- post-rewrite invariant -------------------------------------------------


async def test_rewrite_minting_one_bind_per_table_is_refused_post_rewrite(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several distinct as-of bind keys — the defect itself — never executes.

    This is the invariant the shipped defect violated. The *number of distinct
    bind keys* is what decides whether the compiled cache can resolve one of
    them from a previous execution: one key is safe however many times
    adaption clones it, two keys are not, whatever their values happen to be
    on this particular execution.
    """

    def keyed_per_table(as_of_ts: dt.datetime, table: str) -> ColumnElement[Any]:
        return mark_as_of_bind(
            sa.bindparam(f"as_of_{table}", value=as_of_ts, type_=sa.DateTime(timezone=True))
        )

    _install_broken_rewrite(monkeypatch, keyed_per_table)
    with pytest.raises(AsOfBindIntegrityError, match="distinct as-of bind key"):
        await _rows_at(_two_table_statement(), _EARLY_PROBE)
    assert not executed_sql.statements


async def test_rewrite_with_a_wrong_as_of_value_is_refused_post_rewrite(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single well-formed bind carrying the wrong instant is still refused.

    Shape right, key count right, value wrong — the case a shape-only check
    waves through.
    """

    def wrong_value(as_of_ts: dt.datetime, table: str) -> ColumnElement[Any]:
        del as_of_ts, table
        return mark_as_of_bind(
            sa.bindparam(AS_OF_BIND_KEY, value=_LATE_PROBE, type_=sa.DateTime(timezone=True))
        )

    _install_broken_rewrite(monkeypatch, wrong_value)
    with pytest.raises(AsOfBindIntegrityError, match="but this session bound as_of"):
        await _rows_at(_two_table_statement(), _EARLY_PROBE)
    assert not executed_sql.statements


async def test_rewrite_dropping_the_knowledge_time_predicate_is_refused_post_rewrite(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A versioned form with no ``knowledge_time`` predicate at all is refused.

    The regression a shape check based on subquery *nesting* would miss and a
    bind check catches: ``DISTINCT ON ... WHERE NOT is_retraction`` with the
    temporal filter deleted still looks like the versioned form, and reads the
    whole store.
    """
    _install_broken_rewrite(monkeypatch, _anonymous_bind_per_table, with_predicate=False)
    with pytest.raises(AsOfBindIntegrityError, match="distinct as-of bind key"):
        await _rows_at(_two_table_statement(), _EARLY_PROBE)
    assert not executed_sql.statements


# --- boundary check, each fail-closed direction -----------------------------


async def test_execution_declaring_an_as_of_but_sending_no_bind_is_refused(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sanctioned execution that sends no as-of parameter fails closed.

    The direction that keeps the backstop from depending on recognizing our
    own binds: a rewrite that loses the predicate — or renders it in some form
    this module cannot see — is refused because the execution *claimed* an
    as-of and then sent none, not because anything was recognized.
    """
    _disable_post_rewrite_check(monkeypatch)
    _install_broken_rewrite(monkeypatch, _anonymous_bind_per_table, with_predicate=False)
    with pytest.raises(AsOfBindIntegrityError, match="sends no as-of bind parameter"):
        await _rows_at(_two_table_statement(), _EARLY_PROBE)
    assert not executed_sql.statements


async def test_unrecognizable_as_of_binds_are_refused_on_the_first_execution(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped defect verbatim — unmarked anonymous binds — never runs at all.

    The pre-fix rewrite as it actually was: ``sa.literal()`` per table,
    carrying no rewriter token. The guard cannot tell those binds are the
    as-of, so the execution declares an as-of and then, as far as anything can
    verify, sends none. That is refused on the *first* execution, before a
    compiled cache even exists — which is why the backstop closes the class
    rather than only the cache-hit instance: a rewrite whose as-of parameters
    the guard cannot identify is not a rewrite it will let through.
    """
    _disable_post_rewrite_check(monkeypatch)
    _install_broken_rewrite(monkeypatch, _anonymous_bind_per_table)
    with pytest.raises(AsOfBindIntegrityError, match="sends no as-of bind parameter"):
        await _rows_at(_two_table_statement(), _LATE_PROBE)
    assert not executed_sql.statements


async def test_as_of_bind_without_a_sanctioning_expectation_is_refused(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An as-of bind nothing vouches for is refused, not assumed correct.

    Simulates a future refactor that sanctions a rewritten read but forgets to
    pass the session's as-of as ground truth. Unverifiable must not mean fine:
    with no expectation there is nothing to compare the value against, so the
    execution is denied exactly as an unsanctioned one would be.
    """

    def sanction_without_ground_truth(as_of: dt.datetime | None = None) -> dict[str, object]:
        del as_of
        return guard_module.sanctioned_execution_options()

    monkeypatch.setattr(asof_module, "sanctioned_execution_options", sanction_without_ground_truth)
    with pytest.raises(AsOfBindIntegrityError, match="no as-of expectation"):
        await _rows_at(_two_table_statement(), _EARLY_PROBE)
    assert not executed_sql.statements


async def test_foreign_bind_colliding_with_the_as_of_key_is_refused(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
    executed_sql: _ExecutedSQL,
) -> None:
    """A caller's own bind on the as-of key cannot silently move the as-of.

    SQLAlchemy resolves two binds sharing a key to a single parameter, last
    one compiled winning, with no error — so an application query carrying a
    bind named like ours would otherwise redirect the whole statement's as-of.
    Reached through nothing but public ORM constructs and
    :func:`backend.db.as_of`; no monkeypatching at all.
    """
    statement = select(PriceBar).where(
        PriceBar.valid_from
        >= sa.bindparam(AS_OF_BIND_KEY, value=_LATE_PROBE, type_=sa.DateTime(timezone=True))
    )
    with pytest.raises(AsOfBindIntegrityError):
        await _rows_at(statement, _EARLY_PROBE)
    assert not executed_sql.statements


# --- the fix itself, stated as a property -----------------------------------


async def test_repeated_execution_sends_exactly_one_as_of_parameter_per_execution(
    two_table_population: int,  # noqa: ARG001 — data must exist for a leak to be observable
) -> None:
    """One shared bind: one parameter in the SQL, this execution's value in it.

    The positive statement of the fix, asserted on what the driver is actually
    handed rather than on the clause tree — including across the compiled-cache
    hit the second execution takes.
    """
    sent: list[tuple[str, object]] = []

    def listener(
        conn: object,  # noqa: ARG001 — SQLAlchemy dispatches listeners positionally
        cursor: DBAPICursor,  # noqa: ARG001
        statement: str,
        parameters: object,
        context: ExecutionContext | None,  # noqa: ARG001
        executemany: bool,  # noqa: ARG001
    ) -> None:
        if "price_bar" in statement:
            sent.append((statement, parameters))

    event.listen(Engine, "before_cursor_execute", listener)
    try:
        statement = _two_table_statement()
        for probe, expected_knowledge in ((_LATE_PROBE, _LATE), (_EARLY_PROBE, _EARLY)):
            sent.clear()
            rows = await _rows_at(statement, probe)
            assert [bar.knowledge_time for bar in rows] == [expected_knowledge]
            assert len(sent) == 1
            sql, parameters = sent[0]
            placeholders = set(re.findall(r"\$\d+", sql))
            assert placeholders == {"$1"}, (
                f"the rewritten SQL must carry exactly one bind placeholder, got "
                f"{sorted(placeholders)}"
            )
            assert parameters == (probe,), (
                f"at as_of {probe.isoformat()} the driver received {parameters!r}; "
                "exactly one as-of parameter carrying this execution's instant is "
                "the whole point of the shared bind"
            )
    finally:
        event.remove(Engine, "before_cursor_execute", listener)
