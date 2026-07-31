"""P2.6 gate: Hypothesis property suite for the as-of layer against real TimescaleDB.

Directive Phase 2 gate: random facts with random knowledge times and random
as-of queries; >= 10,000 asserted cases with **zero** failures; no returned
row may have ``knowledge_time > as_of`` (invariant I1).

Every example builds a random fact population — random logical keys, random
valid intervals (open-ended / adjacent / overlapping / instant), random
knowledge times (corrections, retractions, shuffled out-of-order inserts;
all knowledge times are far in the past relative to ingestion, i.e. every
insert is backfill-style) — through the sanctioned writer path, then runs
random ``as_of`` queries (before / after / exactly at knowledge times, with
boundary equality ``knowledge_time == as_of`` exercised explicitly) and
checks the database result against an independent in-memory oracle:

(a) no returned row has ``knowledge_time > as_of``;
(b) per (logical key, ``valid_from``) exactly the max-knowledge-time
    knowable version is returned. The implementation's documented tiebreaker
    is the composite primary key: equal knowledge times within one fact are
    impossible, which the strategy mirrors by drawing unique knowledge
    times per fact — so "latest wins" always has a unique winner;
(c) a fact whose winning version is a retraction is absent, and no returned
    row is a retraction;
(d) immutability of the past: after inserting a second batch whose
    knowledge times lie strictly after every phase-one as-of, re-querying
    those same as-of timestamps returns byte-identical results.

Case accounting: one case per (fact, query) expected-vs-actual comparison
(covers presence, absence, and version identity) plus one case per returned
row's I1 check. A session-scoped accumulator records the total, publishes it
as a junit testsuite property, and **fails at teardown if the total is below
10,000** — the count floor is enforced, not assumed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import itertools
import time
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select

from backend.db import as_of, dispose_database, ingest_writer_session
from backend.db.models import PriceBar
from backend.tests.integration.factories import create_security

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_ONE_US = dt.timedelta(microseconds=1)
_DAY_US = 86_400_000_000

_KT_BASE = dt.datetime(2016, 1, 4, tzinfo=dt.UTC)
"""Origin for knowledge-time offsets. A decade in the past: every insert the
suite performs is backfill-style (knowledge_time far before ingestion)."""

_VF_BASE = dt.datetime(2016, 1, 4, tzinfo=dt.UTC)
"""Origin for valid-interval offsets (event time)."""

_A_SPAN_US = 400 * _DAY_US
"""Phase-A knowledge times live in [0, _A_SPAN_US] microseconds after _KT_BASE."""

_B_OFFSET_US = _A_SPAN_US + 30 * _DAY_US
"""Phase-B knowledge times start here — strictly after every phase-A as-of
(phase-A as-ofs are capped at _A_SPAN_US + 2 days), so phase-B inserts must
be invisible to every re-queried phase-A as-of (immutability of the past)."""

_B_SPAN_US = 200 * _DAY_US

_VF_CHOICES_US = tuple(day * _DAY_US + hour * 3_600_000_000 for day in range(6) for hour in (0, 14))
"""Valid-from pool: six event days at two intraday anchors, so neighboring
facts can be adjacent or overlapping depending on the drawn valid_to."""

_VT_CHOICES_US = (_DAY_US, 3 * _DAY_US, 1, None)
"""valid_to minus valid_from: one day (adjacent daily bars), three days
(overlapping neighbors), one microsecond (minimal interval), or None for the
open-ended ``'infinity'`` server default."""

_A_AS_OF_FILLERS_US = tuple(-2 * _DAY_US + i * (_A_SPAN_US + 4 * _DAY_US) // 9 for i in range(10))
"""Deterministic phase-A as-of fillers guaranteeing >= 10 queries per example
even when Hypothesis draws colliding values (the >= 10,000-case floor must
hold for every possible draw, not just typical ones)."""

_B_AS_OF_FILLERS_US = tuple(_B_OFFSET_US + i * (_B_SPAN_US + 2 * _DAY_US) // 3 for i in range(4))


@dataclasses.dataclass(frozen=True)
class _VersionSpec:
    """One generated version of one fact, in offset space (microseconds)."""

    security_index: int
    valid_from_us: int
    valid_to_us: int | None  # offset from valid_from; None = open-ended ('infinity')
    knowledge_us: int  # offset from _KT_BASE (phase B specs are pre-shifted)
    close_cents: int  # unique payload so wrong-version returns are detectable
    is_retraction: bool


@dataclasses.dataclass(frozen=True)
class _Scenario:
    """One generated population plus its as-of query schedule."""

    n_securities: int
    rows_a: tuple[_VersionSpec, ...]  # phase A, already in shuffled insertion order
    rows_b: tuple[_VersionSpec, ...]  # later-knowledge batch, shuffled insertion order
    as_of_a_us: tuple[int, ...]
    as_of_b_us: tuple[int, ...]


@st.composite
def _scenarios(draw: st.DrawFn) -> _Scenario:
    """Draw a full random scenario (population + query schedule)."""
    n_securities = draw(st.integers(min_value=2, max_value=3))
    close_counter = 0
    rows_a: list[_VersionSpec] = []
    for security_index in range(n_securities):
        vf_offsets = draw(
            st.lists(st.sampled_from(_VF_CHOICES_US), min_size=2, max_size=4, unique=True)
        )
        for valid_from_us in vf_offsets:
            knowledge_offsets = draw(
                st.lists(
                    st.integers(min_value=0, max_value=_A_SPAN_US),
                    min_size=2,
                    max_size=4,
                    unique=True,
                )
            )
            for knowledge_us in knowledge_offsets:
                close_counter += 1
                rows_a.append(
                    _VersionSpec(
                        security_index=security_index,
                        valid_from_us=valid_from_us,
                        valid_to_us=draw(st.sampled_from(_VT_CHOICES_US)),
                        knowledge_us=knowledge_us,
                        close_cents=10_000 + close_counter,
                        is_retraction=draw(st.integers(min_value=0, max_value=3)) == 0,
                    )
                )
    if not any(row.is_retraction for row in rows_a):
        # Retraction masking must be exercised in every example, not just most.
        rows_a[-1] = dataclasses.replace(rows_a[-1], is_retraction=True)

    b_triples = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=n_securities - 1),
                st.sampled_from(_VF_CHOICES_US),
                st.integers(min_value=0, max_value=_B_SPAN_US),
            ),
            min_size=2,
            max_size=6,
            unique=True,
        )
    )
    rows_b: list[_VersionSpec] = []
    for security_index, valid_from_us, b_knowledge_us in b_triples:
        close_counter += 1
        rows_b.append(
            _VersionSpec(
                security_index=security_index,
                valid_from_us=valid_from_us,
                valid_to_us=draw(st.sampled_from(_VT_CHOICES_US)),
                knowledge_us=_B_OFFSET_US + b_knowledge_us,
                close_cents=10_000 + close_counter,
                is_retraction=draw(st.integers(min_value=0, max_value=3)) == 0,
            )
        )

    a_kts = sorted({row.knowledge_us for row in rows_a})
    exact = draw(
        st.lists(st.sampled_from(a_kts), min_size=1, max_size=min(8, len(a_kts)), unique=True)
    )
    eps = draw(
        st.lists(st.sampled_from(a_kts), min_size=1, max_size=min(4, len(a_kts)), unique=True)
    )
    randoms = draw(
        st.lists(
            st.integers(min_value=-_DAY_US, max_value=_A_SPAN_US + _DAY_US),
            min_size=2,
            max_size=8,
            unique=True,
        )
    )
    as_of_a = {
        *exact,  # boundary equality: knowledge_time == as_of IS visible
        *(kt - 1 for kt in eps),
        *(kt + 1 for kt in eps),
        *randoms,
        -2 * _DAY_US,  # before every knowledge time
        _A_SPAN_US + 2 * _DAY_US,  # after every phase-A knowledge time
    }
    if len(as_of_a) < 10:
        as_of_a |= set(_A_AS_OF_FILLERS_US)

    b_kts = sorted({row.knowledge_us for row in rows_b})
    exact_b = draw(
        st.lists(st.sampled_from(b_kts), min_size=1, max_size=min(3, len(b_kts)), unique=True)
    )
    randoms_b = draw(
        st.lists(
            st.integers(min_value=_B_OFFSET_US, max_value=_B_OFFSET_US + _B_SPAN_US + _DAY_US),
            min_size=2,
            max_size=4,
            unique=True,
        )
    )
    as_of_b = {
        *exact_b,
        *(kt - 1 for kt in exact_b),
        *randoms_b,
        _B_OFFSET_US + _B_SPAN_US + 2 * _DAY_US,
    }
    if len(as_of_b) < 4:
        as_of_b |= set(_B_AS_OF_FILLERS_US)

    return _Scenario(
        n_securities=n_securities,
        rows_a=tuple(draw(st.permutations(rows_a))),
        rows_b=tuple(draw(st.permutations(rows_b))),
        as_of_a_us=tuple(sorted(as_of_a)),
        as_of_b_us=tuple(sorted(as_of_b)),
    )


class _CaseCounter:
    """Session-wide accumulator of asserted (fact, query) cases."""

    def __init__(self) -> None:
        self.total = 0

    def add(self, n: int) -> None:
        self.total += n


@pytest.fixture(scope="session")
def property_case_counter(
    record_testsuite_property: Callable[[str, object], None],
) -> Iterator[_CaseCounter]:
    """Accumulate asserted cases across the whole suite; enforce the gate floor.

    Teardown publishes the achieved count and wall-clock as junit testsuite
    properties, prints them, and **fails** if fewer than 10,000 cases were
    asserted — the directive's case count is verified, never assumed.
    """
    counter = _CaseCounter()
    started = time.monotonic()
    yield counter
    elapsed = time.monotonic() - started
    record_testsuite_property("p26_bitemporal_property_cases", counter.total)
    record_testsuite_property("p26_bitemporal_property_seconds", round(elapsed, 1))
    print(f"\nP2.6 property suite: {counter.total} asserted cases in {elapsed:.1f}s")
    assert counter.total >= 10_000, (
        f"P2.6 gate requires >= 10,000 asserted property cases; got {counter.total}. "
        "Run the full P2.6 module, not a subset."
    )


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


def _to_knowledge_dt(offset_us: int) -> dt.datetime:
    return _KT_BASE + dt.timedelta(microseconds=offset_us)


def _to_valid_from_dt(offset_us: int) -> dt.datetime:
    return _VF_BASE + dt.timedelta(microseconds=offset_us)


def _build_bar(spec: _VersionSpec, security_ids: list[int]) -> PriceBar:
    """Materialize one version spec as an ORM row (USD cents -> USD Decimal)."""
    valid_from = _to_valid_from_dt(spec.valid_from_us)
    price = Decimal(spec.close_cents) / Decimal(100)
    bar = PriceBar(
        security_id=security_ids[spec.security_index],
        valid_from=valid_from,
        knowledge_time=_to_knowledge_dt(spec.knowledge_us),
        is_retraction=spec.is_retraction,
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal(1),
        volume_shares=1,
    )
    if spec.valid_to_us is not None:
        bar.valid_to = valid_from + dt.timedelta(microseconds=spec.valid_to_us)
    return bar  # valid_to omitted -> server default 'infinity' (open-ended)


async def _insert_specs(specs: tuple[_VersionSpec, ...], security_ids: list[int]) -> None:
    """Insert specs in their (shuffled) order through the sanctioned writer path."""
    if not specs:
        return
    async with ingest_writer_session() as session:
        session.add_all([_build_bar(spec, security_ids) for spec in specs])
        await session.commit()


def _expected_visible(
    rows: tuple[_VersionSpec, ...], as_of_us: int
) -> dict[tuple[int, int], tuple[int, int]]:
    """Independent oracle: (security_index, valid_from_us) -> (knowledge_us, close_cents).

    Latest ``knowledge_us <= as_of_us`` wins per fact (unique by
    construction, mirroring the PK); a winning retraction hides the fact.
    """
    groups: dict[tuple[int, int], list[_VersionSpec]] = {}
    for row in rows:
        groups.setdefault((row.security_index, row.valid_from_us), []).append(row)
    visible: dict[tuple[int, int], tuple[int, int]] = {}
    for key, versions in groups.items():
        knowable = [version for version in versions if version.knowledge_us <= as_of_us]
        if not knowable:
            continue
        winner = max(knowable, key=lambda version: version.knowledge_us)
        if not winner.is_retraction:
            visible[key] = (winner.knowledge_us, winner.close_cents)
    return visible


async def _query_and_check(
    rows: tuple[_VersionSpec, ...],
    as_of_us: int,
    security_ids: list[int],
    counter: _CaseCounter,
) -> dict[tuple[int, int], tuple[int, int]]:
    """Run one as-of query, assert it against the oracle, and count the cases."""
    as_of_ts = _to_knowledge_dt(as_of_us)
    async with as_of(as_of_ts) as session:
        result = await session.scalars(
            select(PriceBar).where(PriceBar.security_id.in_(security_ids))
        )
        bars = list(result.all())
    id_to_index = {security_id: index for index, security_id in enumerate(security_ids)}
    actual: dict[tuple[int, int], tuple[int, int]] = {}
    for bar in bars:
        # (a) The I1 core property, asserted independently per returned row.
        assert bar.knowledge_time <= as_of_ts, (
            f"I1 violation: returned knowledge_time {bar.knowledge_time.isoformat()} "
            f"> as_of {as_of_ts.isoformat()}"
        )
        # (c) A retraction version must never be returned.
        assert bar.is_retraction is False, "retraction version returned by as_of query"
        key = (
            id_to_index[bar.security_id],
            (bar.valid_from - _VF_BASE) // _ONE_US,
        )
        assert key not in actual, f"duplicate visible version returned for fact {key}"
        actual[key] = (
            (bar.knowledge_time - _KT_BASE) // _ONE_US,
            int(bar.close_usd * 100),
        )
    # (b) Exactly the max-knowledge knowable version per fact, no more, no less.
    expected = _expected_visible(rows, as_of_us)
    assert actual == expected, (
        f"as_of {as_of_ts.isoformat()}: visible set diverges from oracle.\n"
        f"expected: {expected}\nactual:   {actual}"
    )
    n_groups = len({(row.security_index, row.valid_from_us) for row in rows})
    counter.add(len(bars) + n_groups)
    return actual


async def _run_scenario(scenario: _Scenario, counter: _CaseCounter) -> None:
    """Execute one full scenario: insert A, query, insert B, re-query, query B."""
    security_ids = [await create_security() for _ in range(scenario.n_securities)]
    await _insert_specs(scenario.rows_a, security_ids)
    snapshots: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
    for as_of_us in scenario.as_of_a_us:
        snapshots[as_of_us] = await _query_and_check(
            scenario.rows_a, as_of_us, security_ids, counter
        )
    await _insert_specs(scenario.rows_b, security_ids)
    all_rows = (*scenario.rows_a, *scenario.rows_b)
    for as_of_us in scenario.as_of_a_us:
        # (d) Immutability of the past: the later-knowledge batch must not
        # change any previously observed as-of result — checked both against
        # the recorded snapshot and against the oracle over the full population.
        actual = await _query_and_check(all_rows, as_of_us, security_ids, counter)
        assert actual == snapshots[as_of_us], (
            f"immutability violated at as_of offset {as_of_us}: inserting rows with "
            f"knowledge_time > as_of changed the result.\n"
            f"before: {snapshots[as_of_us]}\nafter:  {actual}"
        )
    for as_of_us in scenario.as_of_b_us:
        await _query_and_check(all_rows, as_of_us, security_ids, counter)


@given(scenario=_scenarios())
@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    print_blob=True,
)
def test_asof_visibility_property(
    db_loop: asyncio.AbstractEventLoop,
    property_case_counter: _CaseCounter,
    scenario: _Scenario,
) -> None:
    """P2.6: randomized population/query schedules never leak future knowledge.

    Sharing ``db_loop`` (and the engine pool bound to it) across examples is
    deliberate; example isolation comes from fresh security ids per example,
    which scope every query. Hence the suppressed function-scoped-fixture
    health check.
    """
    db_loop.run_until_complete(_run_scenario(scenario, property_case_counter))


async def test_asof_exhaustive_boundary_grid(property_case_counter: _CaseCounter) -> None:
    """Deterministic sweep: every knowledge time probed at -1us / exact / +1us.

    Fixed population covering the qualitative shapes — single version,
    correction, retraction-final, retraction-then-reassertion, open-ended
    interval — swept densely around every knowledge-time boundary, then
    swept again after a later-knowledge batch (correction + retraction) to
    re-verify immutability of the past deterministically.
    """
    security_ids = [await create_security() for _ in range(2)]

    def kt(index: int) -> int:
        return index * 7 * _DAY_US + index * 3_600_000_000

    def spec(
        sec: int,
        day: int,
        kt_index: int,
        cents: int,
        *,
        retract: bool = False,
        open_ended: bool = False,
    ) -> _VersionSpec:
        return _VersionSpec(
            security_index=sec,
            valid_from_us=day * _DAY_US,
            valid_to_us=None if open_ended else _DAY_US,
            knowledge_us=kt(kt_index),
            close_cents=cents,
            is_retraction=retract,
        )

    rows_a = (
        spec(0, 0, 0, 501),  # single version
        spec(0, 1, 1, 502),  # correction pair ...
        spec(0, 1, 2, 503),
        spec(0, 2, 3, 504),  # retraction-final: invisible from kt(4) on
        spec(0, 2, 4, 505, retract=True),
        spec(1, 0, 5, 506),  # retract-then-reassert: visible, hidden, visible again
        spec(1, 0, 6, 507, retract=True),
        spec(1, 0, 7, 508),
        spec(1, 3, 8, 509, open_ended=True),  # open-ended valid interval
    )
    await _insert_specs(rows_a, security_ids)

    a_kts = sorted({row.knowledge_us for row in rows_a})
    sweep = {probe for boundary in a_kts for probe in (boundary - 1, boundary, boundary + 1)}
    sweep |= {(low + high) // 2 for low, high in itertools.pairwise(a_kts)}
    sweep |= {-_DAY_US, a_kts[-1] + 2 * _DAY_US}
    as_of_schedule = sorted(sweep)

    snapshots: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
    for as_of_us in as_of_schedule:
        snapshots[as_of_us] = await _query_and_check(
            rows_a, as_of_us, security_ids, property_case_counter
        )

    later_base = a_kts[-1] + 30 * _DAY_US  # strictly after every swept as-of
    rows_b = (
        dataclasses.replace(spec(0, 0, 0, 601), knowledge_us=later_base),
        dataclasses.replace(
            spec(1, 3, 0, 602, retract=True, open_ended=True),
            knowledge_us=later_base + _DAY_US,
        ),
    )
    await _insert_specs(rows_b, security_ids)
    all_rows = (*rows_a, *rows_b)
    for as_of_us in as_of_schedule:
        actual = await _query_and_check(all_rows, as_of_us, security_ids, property_case_counter)
        assert actual == snapshots[as_of_us], (
            f"immutability violated at as_of offset {as_of_us} after later-knowledge batch"
        )
    b_kts = sorted({row.knowledge_us for row in rows_b})
    b_schedule = sorted(
        {probe for boundary in b_kts for probe in (boundary - 1, boundary, boundary + 1)}
        | {b_kts[-1] + 2 * _DAY_US}
    )
    for as_of_us in b_schedule:
        await _query_and_check(all_rows, as_of_us, security_ids, property_case_counter)
