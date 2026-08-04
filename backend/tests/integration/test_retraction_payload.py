"""P2.10 integration: retraction rows carry no payload, against real TimescaleDB.

The unit suite (``backend/tests/test_retraction_payload.py``) proves the
constraint expressions and the write/read hooks in isolation and against an
in-memory SQLite copy of the fact tables. This file proves the same property
on the schema revision 0012 actually produces — hypertables, append-only
triggers, PostgreSQL's own three-valued evaluation of the CHECKs — and covers
the two things SQLite cannot show at all:

- that ``macro_observation.is_missing``, which carries a ``false`` server
  default, really stores NULL on a retraction rather than quietly defaulting
  (the reason the write path sends SQL ``NULL`` and not Python ``None``);
- that the pre-existing CHECK constraints of revisions 0006/0008 stay
  satisfiable once payload columns are NULL — three-valued logic makes them
  evaluate to NULL, which passes, but that is a claim about PostgreSQL and is
  asserted here rather than reasoned about.

Everything below runs on the sanctioned public surface, with no unguarded
engine. Three paths are used and each is the documented one for its job:

- ``ingest_writer_session()`` writes rows, and ``session.refresh()`` reads a
  just-written one back. A refresh is a **column load**, the documented
  exemption in :mod:`backend.db.asof`: it re-reads one physical row addressed
  by the full primary key, so it cannot time-travel, and it is the only way to
  see a retraction's stored columns — every versioned read masks them, which
  is the point;
- ``create_admin_engine()`` runs Core ``insert()`` statements that must be
  refused by a CHECK. The Core guard grants a DML sanction naming the target
  fact table (``backend.db._guard``), so these reach Postgres exactly as the
  writer's own inserts do, and are refused by the database rather than by the
  application;
- ``as_of()`` for every read a caller would actually make.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.db import BitemporalBypassError, as_of, create_admin_engine, ingest_writer_session
from backend.db.models import MacroObservation, PriceBar, SecurityMaster
from backend.ingest.supersession import retract_fact
from backend.tests.integration.factories import create_security, insert_rows

_VALID_FROM = dt.datetime(2024, 1, 5, tzinfo=dt.UTC)
_K1 = dt.datetime(2024, 1, 5, 21, 0, tzinfo=dt.UTC)  # observation knowable
_K2 = dt.datetime(2024, 1, 8, 12, 0, tzinfo=dt.UTC)  # retraction knowable
_K3 = dt.datetime(2024, 1, 9, 12, 0, tzinfo=dt.UTC)  # re-assertion knowable

_FACT_TABLES = (
    "security_master",
    "price_bar",
    "edgar_filing",
    "edgar_filing_document",
    "macro_series",
    "macro_observation",
)

_PAYLOAD_CONSTRAINTS = tuple(
    f"ck_{table}_{rule}"
    for table in _FACT_TABLES
    for rule in ("retraction_payload_absent", "observation_payload_present")
)


def _observation(security_id: int, knowledge_time: dt.datetime, close: str) -> PriceBar:
    """One complete daily bar: prices USD/share, volume in shares, factor dimensionless."""
    price = Decimal(close)
    return PriceBar(
        security_id=security_id,
        valid_from=_VALID_FROM,
        valid_to=_VALID_FROM + dt.timedelta(days=1),
        knowledge_time=knowledge_time,
        is_retraction=False,
        open_usd=price,
        high_usd=price,
        low_usd=price,
        close_usd=price,
        close_raw_usd=price,
        adjustment_factor=Decimal("1"),
        volume_shares=1_000,
    )


async def _catalog_rows(query: str, **parameters: object) -> list[sa.Row[Any]]:
    """Run a catalog query on an admin engine and return all rows.

    Every fact-table name travels as a **bound parameter**, never as SQL text,
    so the Core guard's conservative name scan sees nothing to refuse — which
    is what makes catalog introspection a sanctioned admin-engine use rather
    than a reason to reach for an unguarded engine.
    """
    engine = create_admin_engine()
    try:
        async with engine.connect() as connection:
            return list((await connection.execute(sa.text(query), parameters)).all())
    finally:
        await engine.dispose()


async def _expect_check_violation(statement: sa.Insert, constraint: str) -> None:
    """Assert Postgres refuses ``statement`` with the named CHECK constraint.

    Issued as a Core ``INSERT`` on an admin engine: the Core guard sanctions
    DML whose target is a fact table (that is how the writer's own inserts
    reach the database), so what refuses this is the constraint installed by
    revision 0012 and nothing in the application.
    """
    engine = create_admin_engine()
    try:
        with pytest.raises(IntegrityError, match=constraint):
            async with engine.begin() as connection:
                await connection.execute(statement)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# The schema revision 0012 produces
# ---------------------------------------------------------------------------


async def test_every_fact_table_carries_both_payload_constraints() -> None:
    """Revision 0012 installed all twelve CHECKs; none of them is optional."""
    rows = await _catalog_rows(
        "SELECT conname FROM pg_constraint WHERE contype = 'c' AND conname = ANY(:names)",
        names=list(_PAYLOAD_CONSTRAINTS),
    )
    assert {row[0] for row in rows} == set(_PAYLOAD_CONSTRAINTS)


@pytest.mark.parametrize("model", [SecurityMaster, PriceBar, MacroObservation])
async def test_database_nullability_matches_the_models(model: type) -> None:
    """Metadata and database agree column for column, payload columns included.

    The models drop NOT NULL on payload columns in Python; revision 0012 drops
    it in Postgres. A divergence would mean every metadata-level test in
    ``test_bitemporal_schema.py`` asserts a schema the database does not have.
    """
    table = model.__table__  # type: ignore[attr-defined]
    rows = await _catalog_rows(
        "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = :table",
        table=table.name,
    )
    in_database = {name: nullable == "YES" for name, nullable in rows}
    assert in_database == {column.name: column.nullable for column in table.columns}


# ---------------------------------------------------------------------------
# Storage: what the database refuses
# ---------------------------------------------------------------------------


async def test_database_refuses_a_retraction_carrying_a_payload() -> None:
    """No write path — ORM, Core, COPY or psql — can put a number on a retraction."""
    security_id = await create_security()
    await _expect_check_violation(
        sa.insert(PriceBar).values(
            security_id=security_id,
            valid_from=_VALID_FROM,
            valid_to=_VALID_FROM + dt.timedelta(days=1),
            knowledge_time=_K2,
            is_retraction=True,
            close_usd=Decimal("101"),
        ),
        "ck_price_bar_retraction_payload_absent",
    )


async def test_database_refuses_an_observation_missing_a_required_value() -> None:
    """The guarantee NOT NULL used to give observations is intact, just relocated."""
    security_id = await create_security()
    await _expect_check_violation(
        sa.insert(PriceBar).values(
            security_id=security_id,
            valid_from=_VALID_FROM,
            valid_to=_VALID_FROM + dt.timedelta(days=1),
            knowledge_time=_K1,
            is_retraction=False,
            close_usd=Decimal("101"),
        ),
        "ck_price_bar_observation_payload_present",
    )


async def test_database_accepts_an_observation_omitting_only_optional_values() -> None:
    """Columns the *source* may not state stay optional (the listing dates)."""
    security_id = await create_security()
    master = SecurityMaster(
        security_id=security_id,
        ticker="AAPL",
        name="Apple Inc.",
        exchange="XNAS",
        valid_from=_VALID_FROM,
        knowledge_time=_K1,
        is_retraction=False,
    )
    async with ingest_writer_session() as session:
        session.add(master)
        await session.commit()
        await session.refresh(master)
        assert master.ticker == "AAPL"
        assert master.first_listed_on is None
        assert master.delisted_on is None


# ---------------------------------------------------------------------------
# Write path: what the writer actually stores
# ---------------------------------------------------------------------------


async def test_retract_fact_stores_a_row_whose_every_payload_column_is_null() -> None:
    """The property, stated as the database holds it.

    ``session.refresh`` is the only way to look at a retraction's own columns
    — every versioned read masks the row — and it is a column load, so it is
    exempt from the as-of rewrite by design (module docstring).
    """
    security_id = await create_security()
    observation = _observation(security_id, _K1, "100")
    retraction = retract_fact(observation, knowledge_time=_K2)
    async with ingest_writer_session() as session:
        session.add_all([observation, retraction])
        await session.commit()
        await session.refresh(retraction)
        await session.refresh(observation)

        assert retraction.is_retraction is True
        assert retraction.security_id == security_id
        assert retraction.valid_from == _VALID_FROM
        assert retraction.knowledge_time == _K2
        for name in PriceBar.__bitemporal_payload__:
            assert getattr(retraction, name) is None, f"{name} must be NULL on a retraction"

        assert observation.is_retraction is False
        assert observation.close_usd == Decimal("100")
        assert observation.volume_shares == 1_000


async def test_a_writer_that_still_supplies_a_payload_stores_nothing_fabricated() -> None:
    """The pre-P2.10 shape lands as NULLs: the numbers it invented are not written.

    This is why the ORM hook clears instead of refusing — an older connector
    keeps working, and the store stops holding values nobody stated.
    """
    security_id = await create_security()
    legacy = _observation(security_id, _K2, "999")
    legacy.is_retraction = True
    async with ingest_writer_session() as session:
        session.add(legacy)
        await session.commit()
        await session.refresh(legacy)
        assert legacy.is_retraction is True
        for name in PriceBar.__bitemporal_payload__:
            assert getattr(legacy, name) is None


async def test_a_retraction_defeats_a_server_default_on_a_payload_column() -> None:
    """``macro_observation.is_missing`` defaults to ``false`` and must still store NULL.

    The one case Python ``None`` would silently get wrong: SQLAlchemy omits an
    unset column from the INSERT and Postgres fills in the default, which would
    put ``false`` — a claim about what FRED reported — on a row that reports
    nothing. The write path sends SQL ``NULL`` explicitly instead.
    """
    observation = MacroObservation(
        series_id="UNRATE",
        observation_date=dt.date(2024, 1, 1),
        value=Decimal("3.7"),
        is_missing=False,
        vintage_start_date=dt.date(2024, 2, 2),
        valid_from=dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        knowledge_time=_K1,
        is_retraction=False,
    )
    retraction = retract_fact(observation, knowledge_time=_K2)
    async with ingest_writer_session() as session:
        session.add_all([observation, retraction])
        await session.commit()
        await session.refresh(retraction)
        # Read through getattr: the payload annotations are non-optional because
        # they state the *read path's* guarantee, and this is the one place a
        # retraction's own columns are legitimately visible (module docstring).
        for name in MacroObservation.__bitemporal_payload__:
            assert getattr(retraction, name) is None, f"{name} must be NULL on a retraction"


async def test_a_null_payload_satisfies_the_pre_existing_check_constraints() -> None:
    """Revision 0008's ``missing_value`` CHECK must stay satisfiable on a retraction.

    ``(is_missing AND value IS NULL) OR (NOT is_missing AND value IS NOT NULL)``
    evaluates to NULL when ``is_missing`` is NULL, and a CHECK that evaluates to
    NULL passes. That is PostgreSQL's three-valued logic rather than something
    this task arranged, so it is asserted rather than assumed — a retraction
    that could not be written at all would be a worse outcome than the problem
    being fixed. The observation here is itself the ``is_missing = true`` case,
    so both branches of that constraint are exercised.
    """
    observation = MacroObservation(
        series_id="PAYEMS",
        observation_date=dt.date(2024, 2, 1),
        value=None,
        is_missing=True,
        vintage_start_date=dt.date(2024, 3, 8),
        valid_from=dt.datetime(2024, 2, 1, tzinfo=dt.UTC),
        knowledge_time=_K1,
        is_retraction=False,
    )
    retraction = retract_fact(observation, knowledge_time=_K2)
    async with ingest_writer_session() as session:
        session.add_all([observation, retraction])
        await session.commit()
        await session.refresh(observation)
        await session.refresh(retraction)
        assert observation.is_missing is True
        assert observation.value is None
        assert getattr(retraction, "is_missing") is None  # noqa: B009 — see above
        assert getattr(retraction, "value") is None  # noqa: B009 — see above


# ---------------------------------------------------------------------------
# Read path: the caller never sees a retraction, before or after
# ---------------------------------------------------------------------------


async def _bars_at(as_of_ts: dt.datetime, security_id: int) -> list[PriceBar]:
    async with as_of(as_of_ts) as session:
        result = await session.scalars(select(PriceBar).where(PriceBar.security_id == security_id))
        return list(result.all())


async def test_as_of_before_the_retraction_returns_the_observation_intact() -> None:
    """The other direction of the property: a genuine observation is unaffected."""
    security_id = await create_security()
    observation = _observation(security_id, _K1, "100")
    await insert_rows(observation, retract_fact(observation, knowledge_time=_K2))

    (bar,) = await _bars_at(_K2 - dt.timedelta(microseconds=1), security_id)
    assert bar.is_retraction is False
    assert bar.close_usd == Decimal("100")
    assert bar.volume_shares == 1_000
    assert bar.adjustment_factor == Decimal("1")


async def test_as_of_at_and_after_the_retraction_returns_nothing() -> None:
    """A winning retraction hides the fact; its NULL payload never reaches a caller."""
    security_id = await create_security()
    observation = _observation(security_id, _K1, "100")
    await insert_rows(observation, retract_fact(observation, knowledge_time=_K2))

    assert await _bars_at(_K2, security_id) == []
    assert await _bars_at(dt.datetime(2024, 6, 1, tzinfo=dt.UTC), security_id) == []


async def test_reasserting_after_a_retraction_restores_a_complete_payload() -> None:
    """Retraction is not deletion: a later version brings the fact back, whole."""
    security_id = await create_security()
    observation = _observation(security_id, _K1, "100")
    await insert_rows(
        observation,
        retract_fact(observation, knowledge_time=_K2),
        _observation(security_id, _K3, "102"),
    )

    assert [bar.close_usd for bar in await _bars_at(_K1, security_id)] == [Decimal("100")]
    assert await _bars_at(_K2, security_id) == []
    (restored,) = await _bars_at(_K3, security_id)
    assert restored.close_usd == Decimal("102")
    assert restored.volume_shares == 1_000


async def test_column_only_reads_never_surface_a_retraction_null() -> None:
    """A select of bare payload columns is masked too, so no NULL row appears.

    The shape that would hurt most if the mask were incomplete: the caller gets
    scalars, not entities, so a retraction would arrive as a ``None`` where a
    price belongs rather than as an obviously wrong object.
    """
    security_id = await create_security()
    observation = _observation(security_id, _K1, "100")
    await insert_rows(observation, retract_fact(observation, knowledge_time=_K2))

    async with as_of(dt.datetime(2024, 6, 1, tzinfo=dt.UTC)) as session:
        result = await session.execute(
            select(PriceBar.close_usd, PriceBar.volume_shares).where(
                PriceBar.security_id == security_id
            )
        )
        assert list(result.all()) == []


async def test_the_writer_session_still_cannot_query_a_retraction_back() -> None:
    """Reads stay on the as-of path even for the writer that just wrote the row.

    The column-load exemption used above is exactly that — a refresh of one
    row the writer already holds. An actual SELECT on the same session is
    still refused, so "the writer can see its own retraction" does not widen
    into a general unversioned read.
    """
    security_id = await create_security()
    observation = _observation(security_id, _K1, "100")
    await insert_rows(observation, retract_fact(observation, knowledge_time=_K2))

    async with ingest_writer_session() as session:
        with pytest.raises(BitemporalBypassError, match="no bound as-of"):
            await session.scalars(select(PriceBar).where(PriceBar.security_id == security_id))
