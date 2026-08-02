"""P5.1: the availability lag as an enforced control, not a documented intention.

This is invariant I1 at the feature layer. A feature declares how long after an
event its value is knowable; the compute path must make that declaration the
only as-of instant the computation is ever given. A lag that is merely written
down produces backtests that look excellent and are worthless, and no
distribution check, type check or unit test reveals it.

The claim is attacked at three depths, weakest first.

**The instant.** :func:`resolve_as_of` is pure, so the rule can be checked
exhaustively without a database: a requested as-of may move backwards (a
historical replay legitimately reconstructs what was knowable earlier) and never
forwards, the boundary is inclusive, and a naive or non-UTC instant is refused
rather than compared.

**The binding.** ``compute_feature`` is asked for a real vector and the
computation records the as-of actually bound to the session it was handed,
under the key the database guard itself reads
(:data:`backend.db.asof.AS_OF_INFO_KEY`, imported rather than transcribed, so
this test cannot pass against a stale copy of the constant).

**The SQL.** The deepest check, and the reason this file is worth its runtime:
the computation issues a genuine ORM read of a fact table through the session it
was given, and the statement is intercepted *after* the D-011 rewrite has run.
The assertion is on the predicate that will reach the database —
``knowledge_time <= :as_of`` with the bound value — not on anything this package
says about itself. A lag enforced in every layer above but dropped at the
rewrite would fail here and nowhere else. No connection is opened: the spy
raises at the ORM boundary, before a cursor exists.

**Verdict on the enforcement, recorded because it is not absolute.** Along the
sanctioned path the instant is *derived*, never supplied: there is exactly one
``as_of(...)`` call site in this package, its argument is the value
:func:`resolve_as_of` returned, and no parameter of ``compute_feature`` reaches
that call except through the check — asserted structurally over the module
source below, so a future edit that threads a caller-supplied instant into the
session fails these tests. What is *not* structurally impossible is a
computation that ignores the session it was handed and calls the public
:func:`backend.db.as_of` itself: ``backend.db`` exports it to the whole
application, and ``@feature`` returns the decorated function unchanged, so its
defining module holds a direct reference. That path is covered by the D-011
layers and by review, not by this package, and the module docstring of
:mod:`backend.features.compute` says so rather than claiming more.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from sqlalchemy import event, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from backend.db import AsOfTimestampError
from backend.db.asof import AS_OF_INFO_KEY
from backend.db.models import PriceBar
from backend.features import compute as compute_module
from backend.features import errors as errors_module
from backend.features import registry as registry_module
from backend.features import spec as spec_module
from backend.features.compute import (
    FeatureComputeRequest,
    FeatureVector,
    compute_feature,
    resolve_as_of,
)
from backend.features.errors import (
    AvailabilityLagViolationError,
    FeatureComputeError,
    MalformedFeatureVectorError,
    UnknownFeatureError,
)
from backend.features.registry import FeatureRegistry
from backend.features.spec import MAX_AVAILABILITY_LAG, FeatureSpec, compute_instant

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from backend.features.compute import FloatArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# SQLAlchemy's dialect constructors carry no annotations, hence the ignore.
_PG_DIALECT = postgresql.dialect()  # type: ignore[no-untyped-call]
"""Compile captured statements as PostgreSQL, the dialect this platform runs on."""

_LAG_45 = dt.timedelta(days=45)
_COMPUTE_DATE = dt.date(2026, 3, 1)
_PERMITTED_45 = dt.datetime(2026, 1, 15, tzinfo=dt.UTC)
"""The worked example: a 45-day lag computing for 2026-03-01 may read to 2026-01-15."""


def make_spec(
    name: str = "probe_feature",
    *,
    availability_lag: dt.timedelta = _LAG_45,
    units: str = "dimensionless ratio",
) -> FeatureSpec:
    """Build a well-formed declaration with a chosen lag."""
    return FeatureSpec(
        name=name,
        definition=f"Declaration {name}, registered to exercise the compute path.",
        units=units,
        availability_lag=availability_lag,
        source_tables=frozenset({"price_bar"}),
    )


class Recorder:
    """A registered computation that records what it was handed and returns NaN.

    It computes nothing. Values are NaN rather than numbers precisely because a
    plausible-looking factor value invented here would be the fabrication I3
    forbids — the arithmetic is P5.3's, and this file is about the instant the
    computation is pinned to, not about what it computes.
    """

    def __init__(self) -> None:
        self.calls: list[FeatureComputeRequest] = []
        self.bound_as_of: list[dt.datetime] = []

    async def __call__(self, session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        """Record the session's bound as-of and the request; return NaN per security."""
        self.calls.append(request)
        self.bound_as_of.append(session.sync_session.info[AS_OF_INFO_KEY])
        return np.full(len(request.security_ids), np.nan, dtype=np.float64)


def registry_with(spec: FeatureSpec, computation: Any) -> FeatureRegistry:  # noqa: ANN401
    """Return a fresh registry holding exactly ``spec``."""
    registry = FeatureRegistry()
    registry.register(spec, computation)
    return registry


class _InterceptedError(Exception):
    """Raised by the SQL spy to abort a statement at the ORM boundary, before any I/O."""


class SqlSpy:
    """Captures statements *after* the D-011 as-of rewrite, then aborts them.

    Registered on the ORM ``Session`` class after
    ``backend.db.asof._enforce_bitemporal_reads``, so the statement it sees is
    the rewritten, versioned one that would have gone to the database.
    """

    def __init__(self) -> None:
        self.statements: list[Any] = []

    def __call__(self, execute_state: Any) -> None:  # noqa: ANN401 — SQLAlchemy ORMExecuteState
        """Record the rewritten statement and abort before a cursor is reached."""
        self.statements.append(execute_state.statement)
        raise _InterceptedError

    def datetime_binds(self) -> list[dt.datetime]:
        """Return every datetime bind parameter across the captured statements."""
        found: list[dt.datetime] = []
        for statement in self.statements:
            compiled = statement.compile(dialect=_PG_DIALECT)
            found.extend(
                value for value in compiled.params.values() if isinstance(value, dt.datetime)
            )
        return found

    def sql(self) -> str:
        """Return the captured statements as PostgreSQL text."""
        return "\n".join(str(s.compile(dialect=_PG_DIALECT)) for s in self.statements)


@pytest.fixture
def sql_spy() -> Iterator[SqlSpy]:
    """Install a ``do_orm_execute`` spy for the duration of one test."""
    spy = SqlSpy()
    event.listen(Session, "do_orm_execute", spy)
    try:
        yield spy
    finally:
        event.remove(Session, "do_orm_execute", spy)


class Reader:
    """A computation that issues a real fact-table read through the session it is given."""

    def __init__(self) -> None:
        self.ran = False

    async def __call__(self, session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        """Select from a bitemporal table, letting the spy intercept the rewritten statement."""
        self.ran = True
        with contextlib.suppress(_InterceptedError):
            await session.execute(select(PriceBar).where(PriceBar.security_id == 1))
        return np.full(len(request.security_ids), np.nan, dtype=np.float64)


async def run(
    spec: FeatureSpec,
    computation: Any,  # noqa: ANN401
    *,
    compute_date: dt.date = _COMPUTE_DATE,
    security_ids: Sequence[int] = (1, 2, 3),
    requested_as_of: dt.datetime | None = None,
) -> FeatureVector:
    """Register ``spec`` in a fresh registry and compute it."""
    return await compute_feature(
        spec.name,
        compute_date=compute_date,
        security_ids=security_ids,
        requested_as_of=requested_as_of,
        registry=registry_with(spec, computation),
    )


# A window that keeps every derived as-of comfortably in the past, so
# backend.db.as_of()'s own "not in the future" guard never fires and the
# property under test is the only thing that can fail an example.
_PAST_DATES = st.dates(min_value=dt.date(1990, 1, 1), max_value=dt.date(2020, 1, 1))
_LAGS = st.timedeltas(min_value=dt.timedelta(0), max_value=MAX_AVAILABILITY_LAG)
_OFFSETS = st.timedeltas(min_value=-dt.timedelta(days=4000), max_value=dt.timedelta(days=4000))


# ---------------------------------------------------------------------------
# resolve_as_of: the rule, without a database
# ---------------------------------------------------------------------------


def test_the_permitted_instant_is_the_declared_lag_before_the_compute_date() -> None:
    assert resolve_as_of(make_spec(), _COMPUTE_DATE) == _PERMITTED_45


def test_a_request_for_the_permitted_instant_exactly_is_allowed() -> None:
    """The boundary is inclusive, matching backend.db.as_of()."""
    assert resolve_as_of(make_spec(), _COMPUTE_DATE, requested_as_of=_PERMITTED_45) == _PERMITTED_45


def test_a_request_one_microsecond_fresher_is_refused() -> None:
    """The smallest representable step past the declaration is still lookahead."""
    with pytest.raises(AvailabilityLagViolationError):
        resolve_as_of(
            make_spec(),
            _COMPUTE_DATE,
            requested_as_of=_PERMITTED_45 + dt.timedelta(microseconds=1),
        )


def test_a_request_for_an_older_instant_is_honoured() -> None:
    """Reading older data is always I1-safe; it is what a historical replay does."""
    older = _PERMITTED_45 - dt.timedelta(days=900)
    assert resolve_as_of(make_spec(), _COMPUTE_DATE, requested_as_of=older) == older


def test_the_violation_error_carries_the_whole_arithmetic() -> None:
    """Diagnosable from the message alone: declaration, date, both instants, the excess."""
    requested = _PERMITTED_45 + dt.timedelta(days=10)
    with pytest.raises(AvailabilityLagViolationError) as excinfo:
        resolve_as_of(make_spec("book_to_price"), _COMPUTE_DATE, requested_as_of=requested)

    error = excinfo.value
    assert error.feature == "book_to_price"
    assert error.compute_date == _COMPUTE_DATE
    assert error.availability_lag == _LAG_45
    assert error.permitted_as_of == _PERMITTED_45
    assert error.requested_as_of == requested

    message = str(error)
    assert "book_to_price" in message
    assert "2026-01-15" in message
    assert "2026-01-25" in message
    assert "10 days" in message
    assert "45 days" in message


def test_a_naive_requested_instant_is_refused_rather_than_compared() -> None:
    """A naive value cannot be bounds-checked at all: the comparison raises TypeError."""
    naive = dt.datetime(2026, 1, 1)  # noqa: DTZ001 — passing a naive value is the point
    with pytest.raises(FeatureComputeError, match="naive"):
        resolve_as_of(make_spec(), _COMPUTE_DATE, requested_as_of=naive)


def test_a_non_utc_requested_instant_is_refused() -> None:
    """Even one that denotes a permitted instant: the store's column is UTC."""
    cet = dt.timezone(dt.timedelta(hours=2))
    with pytest.raises(FeatureComputeError, match="UTC"):
        resolve_as_of(make_spec(), _COMPUTE_DATE, requested_as_of=_PERMITTED_45.astimezone(cet))


def test_a_datetime_compute_date_is_refused_on_the_compute_path() -> None:
    with pytest.raises(FeatureComputeError, match="must be a date, not a datetime"):
        resolve_as_of(make_spec(), dt.datetime(2026, 3, 1, 16, tzinfo=dt.UTC))


def test_a_zero_lag_feature_may_not_read_anything_published_during_the_compute_date() -> None:
    """Midnight opening the date, not its close: a fact published at 16:05 cannot inform a trade."""
    permitted = resolve_as_of(make_spec(availability_lag=dt.timedelta(0)), _COMPUTE_DATE)
    assert permitted == dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    with pytest.raises(AvailabilityLagViolationError):
        resolve_as_of(
            make_spec(availability_lag=dt.timedelta(0)),
            _COMPUTE_DATE,
            requested_as_of=dt.datetime(2026, 3, 1, 16, 5, tzinfo=dt.UTC),
        )


@given(lag=_LAGS, compute_date=_PAST_DATES, offset=_OFFSETS)
@hypothesis_settings(max_examples=600, deadline=None)
def test_resolve_never_returns_an_instant_fresher_than_the_declaration_permits(
    lag: dt.timedelta, compute_date: dt.date, offset: dt.timedelta
) -> None:
    """Exhaustive over the rule: whatever is asked for, what comes back is permitted.

    Both branches are asserted. A resolver that raised on everything would pass
    a test that only checked the returned value, so the refusal is pinned to
    exactly the requests that are too fresh.
    """
    spec = make_spec(availability_lag=lag)
    permitted = spec.knowledge_cutoff(compute_date)
    requested = permitted + offset

    if offset > dt.timedelta(0):
        with pytest.raises(AvailabilityLagViolationError):
            resolve_as_of(spec, compute_date, requested_as_of=requested)
        return

    resolved = resolve_as_of(spec, compute_date, requested_as_of=requested)
    assert resolved == requested
    assert resolved <= permitted


@given(lag=_LAGS, compute_date=_PAST_DATES, extra=_LAGS)
@hypothesis_settings(max_examples=400, deadline=None)
def test_a_longer_declared_lag_never_permits_a_fresher_instant(
    lag: dt.timedelta, compute_date: dt.date, extra: dt.timedelta
) -> None:
    """Over-declaring a lag may only cost staleness, never buy freshness."""
    longer = min(lag + extra, MAX_AVAILABILITY_LAG)
    assert resolve_as_of(make_spec(availability_lag=longer), compute_date) <= resolve_as_of(
        make_spec(availability_lag=lag), compute_date
    )


# ---------------------------------------------------------------------------
# compute_feature: the instant that is actually bound to the session
# ---------------------------------------------------------------------------


async def test_the_computation_is_handed_a_session_pinned_to_its_declared_lag() -> None:
    recorder = Recorder()
    vector = await run(make_spec("book_to_price"), recorder)

    assert recorder.bound_as_of == [_PERMITTED_45]
    assert recorder.calls[0].as_of == _PERMITTED_45
    assert recorder.calls[0].feature == "book_to_price"
    assert recorder.calls[0].compute_date == _COMPUTE_DATE
    assert recorder.calls[0].security_ids == (1, 2, 3)
    assert vector.as_of == _PERMITTED_45


async def test_the_returned_vector_carries_the_units_and_the_instant_it_was_read_at() -> None:
    """Units travel with the numbers (§8); the instant is what proves which knowledge set (I2)."""
    vector = await run(make_spec("earnings_yield", units="log return, fraction"), Recorder())
    assert vector.feature == "earnings_yield"
    assert vector.units == "log return, fraction"
    assert vector.compute_date == _COMPUTE_DATE
    assert vector.as_of == _PERMITTED_45
    assert vector.security_ids == (1, 2, 3)
    assert vector.values.shape == (3,)


async def test_a_fresher_requested_instant_is_refused_before_the_computation_runs() -> None:
    """Refusing after the session opened would leave a read half-done; it must cost no I/O."""
    recorder = Recorder()
    with pytest.raises(AvailabilityLagViolationError):
        await run(
            make_spec(),
            recorder,
            requested_as_of=_PERMITTED_45 + dt.timedelta(microseconds=1),
        )
    assert recorder.calls == []
    assert recorder.bound_as_of == []


async def test_an_older_requested_instant_pins_the_session_to_it() -> None:
    recorder = Recorder()
    older = _PERMITTED_45 - dt.timedelta(days=30)
    vector = await run(make_spec(), recorder, requested_as_of=older)
    assert recorder.bound_as_of == [older]
    assert vector.as_of == older


async def test_a_naive_requested_instant_never_reaches_the_computation() -> None:
    recorder = Recorder()
    naive = dt.datetime(2026, 1, 1)  # noqa: DTZ001 — passing a naive value is the point
    with pytest.raises(FeatureComputeError, match="naive"):
        await run(make_spec(), recorder, requested_as_of=naive)
    assert recorder.calls == []


async def test_an_unknown_feature_is_refused_before_any_session_is_opened() -> None:
    with pytest.raises(UnknownFeatureError, match="momentum_12_1"):
        await compute_feature(
            "momentum_12_1",
            compute_date=_COMPUTE_DATE,
            security_ids=[1],
            registry=FeatureRegistry(),
        )


async def test_duplicate_security_ids_are_refused_and_named() -> None:
    """A repeated identifier is double-counted by every cross-sectional statistic downstream."""
    recorder = Recorder()
    with pytest.raises(FeatureComputeError, match=r"duplicate identifier\(s\) \[2\]"):
        await run(make_spec(), recorder, security_ids=[1, 2, 2, 3])
    assert recorder.calls == []


async def test_an_empty_universe_is_a_fact_not_an_error() -> None:
    recorder = Recorder()
    vector = await run(make_spec(), recorder, security_ids=[])
    assert vector.values.shape == (0,)
    assert recorder.calls[0].security_ids == ()


async def test_the_request_handed_to_a_computation_cannot_be_edited() -> None:
    """Re-pinning is not available from the request object either."""
    recorder = Recorder()
    await run(make_spec(), recorder)
    request = recorder.calls[0]
    with pytest.raises(AttributeError):
        request.as_of = dt.datetime.now(dt.UTC)  # type: ignore[misc]


async def test_a_future_compute_window_is_refused_by_the_as_of_layer() -> None:
    """A lag window that has not elapsed yet is a real condition, named where it belongs."""
    tomorrow = dt.datetime.now(dt.UTC).date() + dt.timedelta(days=1)
    recorder = Recorder()
    with pytest.raises(AsOfTimestampError, match="future"):
        await run(make_spec(availability_lag=dt.timedelta(0)), recorder, compute_date=tomorrow)
    assert recorder.calls == []


async def test_a_date_inside_an_elapsed_lag_window_computes_normally() -> None:
    """The mirror image: a 45-day lag may be computed for a date up to 45 days ahead.

    Nothing fresher than 45 days ago is read, so this is not lookahead — it is
    the schedule the declaration makes possible, and refusing it would be wrong.
    """
    recorder = Recorder()
    ahead = dt.datetime.now(dt.UTC).date() + dt.timedelta(days=10)
    vector = await run(make_spec(), recorder, compute_date=ahead)
    assert vector.as_of == compute_instant(ahead) - _LAG_45
    assert recorder.bound_as_of == [vector.as_of]


# ---------------------------------------------------------------------------
# The SQL that actually reaches the database
# ---------------------------------------------------------------------------


async def test_a_read_through_the_pinned_session_is_versioned_at_the_declared_cutoff(
    sql_spy: SqlSpy,
) -> None:
    """The deepest form of the claim: the emitted predicate carries the declared instant.

    Not "the framework recorded the right number" but "the statement heading for
    the database filters ``knowledge_time`` at 2026-01-15T00:00Z" — the 45-day
    lag, applied to the 2026-03-01 compute date, in the SQL itself.
    """
    reader = Reader()
    await run(make_spec(), reader)

    assert reader.ran
    assert sql_spy.statements, "the computation's read never reached the ORM boundary"
    assert "knowledge_time <=" in sql_spy.sql()
    assert sql_spy.datetime_binds() == [_PERMITTED_45]


async def test_an_older_requested_instant_shows_up_in_the_sql_too(sql_spy: SqlSpy) -> None:
    older = _PERMITTED_45 - dt.timedelta(days=365)
    await run(make_spec(), Reader(), requested_as_of=older)
    assert sql_spy.datetime_binds() == [older]


@given(lag=_LAGS, compute_date=_PAST_DATES)
@hypothesis_settings(max_examples=150, deadline=None)
def test_no_compute_call_emits_sql_fresher_than_the_declaration_permits(
    lag: dt.timedelta, compute_date: dt.date
) -> None:
    """The property at SQL depth, over random declarations and dates.

    The bound ``knowledge_time`` ceiling in the emitted statement is never later
    than ``midnight(compute_date) - lag``.

    Hypothesis drives a sync test that runs the coroutine itself, because the
    spy has to be installed and removed around each example rather than by a
    function-scoped fixture.
    """
    spec = make_spec(availability_lag=lag)
    permitted = spec.knowledge_cutoff(compute_date)
    spy = SqlSpy()
    event.listen(Session, "do_orm_execute", spy)
    try:
        asyncio.run(run(spec, Reader(), compute_date=compute_date))
    finally:
        event.remove(Session, "do_orm_execute", spy)

    binds = spy.datetime_binds()
    assert binds, "no versioned read was emitted"
    assert all(bound <= permitted for bound in binds), (binds, permitted)
    assert binds == [permitted]


@given(lag=_LAGS, compute_date=_PAST_DATES, offset=_OFFSETS)
@hypothesis_settings(max_examples=400, deadline=None)
def test_no_compute_call_pins_a_session_fresher_than_the_declaration_permits(
    lag: dt.timedelta, compute_date: dt.date, offset: dt.timedelta
) -> None:
    """The whole claim, stated over the sessions a computation was ever handed.

    Whatever the declaration, the date and the instant asked for, either the
    computation ran against a session pinned no later than
    ``midnight(compute_date) - lag``, or it never ran at all.
    """
    spec = make_spec(availability_lag=lag)
    permitted = spec.knowledge_cutoff(compute_date)
    recorder = Recorder()
    requested = permitted + offset

    try:
        asyncio.run(run(spec, recorder, compute_date=compute_date, requested_as_of=requested))
    except AvailabilityLagViolationError:
        assert offset > dt.timedelta(0)
        assert recorder.bound_as_of == []
        return

    assert offset <= dt.timedelta(0)
    assert recorder.bound_as_of == [requested]
    assert all(bound <= permitted for bound in recorder.bound_as_of)


# ---------------------------------------------------------------------------
# The shape of what a computation returns
# ---------------------------------------------------------------------------


async def test_a_vector_of_the_wrong_length_is_refused() -> None:
    """Misalignment attaches one company's numbers to another while looking healthy."""

    async def short(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        del session, request
        return np.zeros(2, dtype=np.float64)

    with pytest.raises(MalformedFeatureVectorError, match="expected 3 value"):
        await run(make_spec(), short)


async def test_a_two_dimensional_result_is_refused() -> None:
    async def matrix(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        del session, request
        return np.zeros((3, 2), dtype=np.float64)

    with pytest.raises(MalformedFeatureVectorError, match="one-dimensional"):
        await run(make_spec(), matrix)


async def test_a_non_numeric_result_is_refused() -> None:
    async def text(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        del session, request
        return np.array(["cheap", "dear", "fair"])

    with pytest.raises(MalformedFeatureVectorError, match="not numeric"):
        await run(make_spec(), text)


async def test_the_refusal_names_the_feature() -> None:
    async def short(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        del session, request
        return np.zeros(1, dtype=np.float64)

    with pytest.raises(MalformedFeatureVectorError) as excinfo:
        await run(make_spec("gross_profitability"), short)
    assert excinfo.value.feature == "gross_profitability"
    assert "gross_profitability" in str(excinfo.value)


async def test_nan_is_carried_through_rather_than_filled() -> None:
    """A missing value stays missing: NaN is never swapped for a plausible number (I3)."""

    async def partly_missing(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        del session
        return np.array([1.5, np.nan, -0.25][: len(request.security_ids)], dtype=np.float64)

    vector = await run(make_spec(), partly_missing)
    assert np.isnan(vector.values[1])
    assert vector.values[0] == pytest.approx(1.5)


async def test_the_returned_values_are_a_copy_of_the_computation_s_array() -> None:
    """A computation handing back its working array cannot edit a vector already returned."""
    working = np.array([1.0, 2.0, 3.0])

    async def hands_back_its_own_buffer(
        session: AsyncSession, request: FeatureComputeRequest
    ) -> FloatArray:
        del session, request
        return working

    vector = await run(make_spec(), hands_back_its_own_buffer)
    working[0] = 999.0
    assert vector.values[0] == pytest.approx(1.0)


async def test_integer_results_are_widened_to_float64() -> None:
    async def integers(session: AsyncSession, request: FeatureComputeRequest) -> FloatArray:
        del session
        return np.arange(len(request.security_ids))  # type: ignore[return-value]

    vector = await run(make_spec(), integers)
    assert vector.values.dtype == np.float64


# ---------------------------------------------------------------------------
# Structural: the instant is derived, never supplied
# ---------------------------------------------------------------------------

_OWNED_MODULES = (spec_module, registry_module, compute_module, errors_module)


def _tree(module: Any) -> ast.Module:  # noqa: ANN401 — a module object
    """Parse one of this package's modules from its source on disk."""
    assert module.__file__ is not None
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Return the top-level function named ``name``."""
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _called_names(node: ast.AST) -> list[ast.Call]:
    """Return every call in ``node`` whose callee is a bare name."""
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    ]


def test_the_package_opens_exactly_one_as_of_session_and_it_is_in_compute_feature() -> None:
    """One call site is what makes "the instant is derived" checkable at all."""
    sites = [
        (module.__name__, call)
        for module in _OWNED_MODULES
        for call in _called_names(_tree(module))
        if isinstance(call.func, ast.Name) and call.func.id == "as_of"
    ]
    assert len(sites) == 1, sites
    assert sites[0][0] == "backend.features.compute"

    in_compute_feature = _called_names(_function(_tree(compute_module), "compute_feature"))
    assert [c for c in in_compute_feature if isinstance(c.func, ast.Name) and c.func.id == "as_of"]


def test_the_as_of_call_is_passed_the_value_resolve_as_of_returned() -> None:
    """Not a caller's argument, and not recomputed: the checked value, verbatim."""
    body = _function(_tree(compute_module), "compute_feature")
    call = next(
        c for c in _called_names(body) if isinstance(c.func, ast.Name) and c.func.id == "as_of"
    )
    assert len(call.args) == 1
    assert not call.keywords
    assert isinstance(call.args[0], ast.Name)
    bound_name = call.args[0].id

    assignments = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == bound_name for t in node.targets)
    ]
    assert len(assignments) == 1, assignments
    source = assignments[0].value
    assert isinstance(source, ast.Call)
    assert isinstance(source.func, ast.Name)
    assert source.func.id == "resolve_as_of"


def test_a_caller_supplied_instant_reaches_nothing_but_the_check() -> None:
    """``requested_as_of`` is consumed by ``resolve_as_of`` and by nothing else.

    This is the assertion that would fail if a later edit threaded the caller's
    instant past the bound — for instance by falling back to it when the
    declaration's own arithmetic was inconvenient.
    """
    body = _function(_tree(compute_module), "compute_feature")
    uses = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Name) and node.id == "requested_as_of"
    ]
    assert uses, "requested_as_of is not referenced at all"

    passed_to_the_check = [
        keyword.value
        for call in _called_names(body)
        for keyword in call.keywords
        if isinstance(call.func, ast.Name)
        and call.func.id == "resolve_as_of"
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "requested_as_of"
    ]
    # Every reference in the function body is one of those keyword arguments:
    # the caller's instant is handed to the bounds check and to nothing else.
    assert len(passed_to_the_check) == len(uses)


def test_the_package_holds_no_way_to_mint_a_session_of_its_own() -> None:
    """No engine, no sessionmaker, no private db internals: ``as_of`` is the only handle."""
    banned = {
        "create_engine",
        "create_async_engine",
        "sessionmaker",
        "async_sessionmaker",
        "ingest_writer_session",
    }
    for module in _OWNED_MODULES:
        for node in ast.walk(_tree(module)):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "backend.db.engine", module.__name__
                assert not (banned & {alias.name for alias in node.names}), module.__name__
            elif isinstance(node, ast.Import):
                assert all(alias.name != "backend.db.engine" for alias in node.names)
