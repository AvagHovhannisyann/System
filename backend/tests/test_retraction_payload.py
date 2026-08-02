"""P2.10 unit tests: a retraction's payload is structurally not-a-value.

Four enforcement points, tested independently because each has to hold on its
own — no one of them is allowed to be the only thing standing between a
retraction and a fabricated number in a fact table (I3):

1. **storage** — ``ck_<table>_retraction_payload_absent`` refuses a retraction
   carrying any payload value, and ``ck_<table>_observation_payload_present``
   refuses an observation missing a required one;
2. **write path** — :func:`backend.ingest.supersession.retract_fact` cannot
   express a retraction with a payload, and the ORM ``before_insert`` hook
   clears one supplied by an older writer;
3. **read path, statement level** —
   :func:`backend.db.asof._assert_retraction_mask` refuses to execute a
   rewritten statement that can reach a fact table without masking
   retractions;
4. **read path, row level** —
   :func:`backend.db.asof._refuse_loaded_retraction` raises if a retraction
   row is nevertheless loaded into application code.

(1) and (2) are exercised **against a real SQL engine** — an in-memory SQLite
holding copies of the real fact tables, constraints and all, so the ORM flush
path and the CHECK expressions are executed rather than inspected. SQLite is
not the deployment target, and the copies exist only because two of the fact
tables carry PostgreSQL-specific DDL that SQLite cannot parse (documented at
:func:`_sqlite_fact_tables`); what it proves is that the constraint
expressions are valid SQL with the intended three-valued semantics and that
the ORM sends what the tests claim it sends. The same properties against
TimescaleDB, on the schema revision 0012 actually produces, are in
``backend/tests/integration/test_retraction_payload.py``.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapper, Session
from sqlalchemy.sql.elements import Null

from backend.db.asof import (
    RetractedFactError,
    RetractionMaskError,
    _assert_retraction_mask,
    _refuse_loaded_retraction,
    _rewrite_select,
)
from backend.db.bitemporal import _clear_retraction_payload
from backend.db.models import EdgarFiling, PriceBar, Security, SecurityMaster
from backend.ingest.errors import FutureKnowledgeTimeError, SupersessionError
from backend.ingest.supersession import retract_fact

if TYPE_CHECKING:
    from collections.abc import Iterator

_VALID_FROM = dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
_DAY = dt.timedelta(days=1)
_KNOWN_AT = dt.datetime(2024, 3, 2, tzinfo=dt.UTC)

_SQLITE_UNSUPPORTED = ("macro_observation",)
"""Fact tables excluded from the SQLite harness, and why.

``macro_observation`` carries ``CHECK (valid_from = timezone('UTC',
observation_date::timestamp))`` from revision 0008 — a PostgreSQL function and
a PostgreSQL cast, neither of which SQLite can parse, so the table cannot be
created there at all. Its P2.10 constraints are covered by the metadata tests
in ``test_bitemporal_schema.py`` (which are parameterized over every
registered model, this one included) and by the integration suite. Nothing
about the payload rule is special-cased for it.
"""


def _observation(security_id: int, knowledge_time: dt.datetime) -> PriceBar:
    """Build one complete PriceBar observation (USD/share, volume in shares)."""
    return PriceBar(
        security_id=security_id,
        valid_from=_VALID_FROM,
        valid_to=_VALID_FROM + _DAY,
        knowledge_time=knowledge_time,
        is_retraction=False,
        open_usd=Decimal("10.00"),
        high_usd=Decimal("12.50"),
        low_usd=Decimal("9.75"),
        close_usd=Decimal("11.25"),
        close_raw_usd=Decimal("11.25"),
        adjustment_factor=Decimal("1"),
        volume_shares=4_200,
    )


# ---------------------------------------------------------------------------
# Write path: retract_fact
# ---------------------------------------------------------------------------


def test_retract_fact_repeats_the_fact_and_carries_no_payload() -> None:
    """The retraction addresses the same fact and states nothing about value."""
    original = _observation(7, _KNOWN_AT)
    retraction = retract_fact(original, knowledge_time=_KNOWN_AT + _DAY)

    assert retraction.is_retraction is True
    assert retraction.security_id == original.security_id
    assert retraction.valid_from == original.valid_from
    assert retraction.valid_to == original.valid_to
    assert retraction.knowledge_time == _KNOWN_AT + _DAY
    for name in PriceBar.__bitemporal_payload__:
        assert isinstance(getattr(retraction, name), Null), (
            f"{name} must be SQL NULL on a retraction, not a value and not Python None "
            "(a column default would otherwise fill it in)"
        )


def test_retract_fact_does_not_mutate_the_row_it_retracts() -> None:
    original = _observation(7, _KNOWN_AT)
    retract_fact(original, knowledge_time=_KNOWN_AT + _DAY)
    assert original.close_usd == Decimal("11.25")
    assert original.is_retraction is False


def test_retract_fact_works_on_a_multi_column_logical_key() -> None:
    """The key is copied whole, whatever its arity (EdgarFiling: accession + CIK)."""
    filing = EdgarFiling(
        accession_number="0001104659-24-032038",
        cik=320193,
        company_name="APPLE INC",
        form_type="8-K",
        filing_date=dt.date(2024, 3, 11),
        index_date=dt.date(2024, 3, 11),
        document_count=3,
        source_url="https://www.sec.gov/Archives/edgar/data/320193/index-headers.html",
        valid_from=_VALID_FROM,
        valid_to=_VALID_FROM + _DAY,
        knowledge_time=_KNOWN_AT,
    )
    retraction = retract_fact(filing, knowledge_time=_KNOWN_AT + _DAY)
    assert retraction.accession_number == "0001104659-24-032038"
    assert retraction.cik == 320193
    for name in EdgarFiling.__bitemporal_payload__:
        assert isinstance(getattr(retraction, name), Null)


def test_retracting_a_retraction_is_refused() -> None:
    """Nothing to withdraw: the fact is already not believed."""
    retraction = retract_fact(_observation(7, _KNOWN_AT), knowledge_time=_KNOWN_AT + _DAY)
    with pytest.raises(SupersessionError, match="already a retraction"):
        retract_fact(retraction, knowledge_time=_KNOWN_AT + 2 * _DAY)


@pytest.mark.parametrize("offset", [dt.timedelta(0), -_DAY])
def test_retraction_knowledge_time_must_be_strictly_later(offset: dt.timedelta) -> None:
    """Equal collides on the primary key; earlier could never win the read."""
    with pytest.raises(SupersessionError, match="strictly later"):
        retract_fact(_observation(7, _KNOWN_AT), knowledge_time=_KNOWN_AT + offset)


def test_retraction_knowledge_time_must_be_aware() -> None:
    naive = dt.datetime(2024, 3, 5)  # noqa: DTZ001 — the point is to pass a naive value
    with pytest.raises(TypeError, match="timezone-aware"):
        retract_fact(_observation(7, _KNOWN_AT), knowledge_time=naive)


def test_retraction_knowledge_time_may_not_be_in_the_future() -> None:
    future = dt.datetime.now(dt.UTC) + dt.timedelta(days=30)
    with pytest.raises(FutureKnowledgeTimeError):
        retract_fact(_observation(7, _KNOWN_AT), knowledge_time=future)


# ---------------------------------------------------------------------------
# Write path: the ORM before_insert hook
# ---------------------------------------------------------------------------


def test_before_insert_hook_is_registered_on_every_mapper() -> None:
    """Dead code cannot enforce anything: the listener must actually be attached."""
    assert sa.event.contains(Mapper, "before_insert", _clear_retraction_payload)


def test_hook_clears_a_payload_supplied_on_a_retraction() -> None:
    """A writer predating the rule stops fabricating rather than starts failing."""
    row = _observation(7, _KNOWN_AT)
    row.is_retraction = True
    _clear_retraction_payload(cast("Any", None), cast("Any", None), row)
    for name in PriceBar.__bitemporal_payload__:
        assert isinstance(getattr(row, name), Null)


def test_hook_leaves_an_observation_untouched() -> None:
    row = _observation(7, _KNOWN_AT)
    _clear_retraction_payload(cast("Any", None), cast("Any", None), row)
    assert row.close_usd == Decimal("11.25")
    assert row.volume_shares == 4_200


def test_hook_ignores_non_bitemporal_rows() -> None:
    anchor = Security()
    _clear_retraction_payload(cast("Any", None), cast("Any", None), anchor)


def test_hook_names_the_discarded_columns(capsys: pytest.CaptureFixture[str]) -> None:
    """Clearing is not silent: a mistaken is_retraction shows up in the log."""
    row = _observation(7, _KNOWN_AT)
    row.is_retraction = True
    _clear_retraction_payload(cast("Any", None), cast("Any", None), row)
    logged = capsys.readouterr().out
    assert "retraction_payload_cleared" in logged
    assert "close_usd" in logged
    assert "volume_shares" in logged


def test_hook_stays_quiet_for_a_correctly_built_retraction(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A row from retract_fact already carries SQL NULL; nothing was discarded."""
    retraction = retract_fact(_observation(7, _KNOWN_AT), knowledge_time=_KNOWN_AT + _DAY)
    _clear_retraction_payload(cast("Any", None), cast("Any", None), retraction)
    assert "retraction_payload_cleared" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Storage: the CHECK constraints, executed against a real SQL engine
# ---------------------------------------------------------------------------


def _sqlite_fact_tables() -> sa.MetaData:
    """Copy the fact tables into fresh metadata SQLite can create.

    One difference from the deployed schema, and only one: ``valid_to``'s
    server default is PostgreSQL's ``'infinity'::timestamptz``, which SQLite
    cannot parse, so the copies drop it and every row below states ``valid_to``
    explicitly. The P2.10 CHECK constraints are copied verbatim — they are what
    is under test. ``macro_observation`` is excluded for the reason given at
    :data:`_SQLITE_UNSUPPORTED`. Copying rather than mutating keeps the real
    metadata every other test inspects untouched.
    """
    metadata = sa.MetaData()
    cast("sa.Table", cast("Any", Security).__table__).to_metadata(metadata)
    for model in (SecurityMaster, PriceBar, EdgarFiling):
        copy = cast("sa.Table", cast("Any", model).__table__).to_metadata(metadata)
        copy.c.valid_to.server_default = None
    return metadata


@pytest.fixture
def fact_store() -> Iterator[sa.Engine]:
    """An in-memory SQLite engine holding the copied fact tables."""
    engine = sa.create_engine("sqlite://")
    _sqlite_fact_tables().create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def _stored(engine: sa.Engine) -> list[sa.Row[Any]]:
    """Read price_bar back through a raw connection (no ORM, no as-of rewrite)."""
    with engine.connect() as connection:
        return list(
            connection.exec_driver_sql(
                "SELECT is_retraction, close_usd, volume_shares, adjustment_factor "
                "FROM price_bar ORDER BY knowledge_time"
            )
        )


def test_orm_stores_an_observation_with_its_payload_intact(fact_store: sa.Engine) -> None:
    """The control: nothing about a genuine observation changed."""
    with Session(fact_store) as session:
        session.add(_observation(1, _KNOWN_AT))
        session.commit()
    (row,) = _stored(fact_store)
    assert row == (0, 11.25, 4_200, 1)


def test_orm_stores_a_retraction_with_a_null_payload(fact_store: sa.Engine) -> None:
    """End to end: retract_fact -> flush -> NULLs on disk, CHECK satisfied."""
    with Session(fact_store) as session:
        session.add(_observation(1, _KNOWN_AT))
        session.add(retract_fact(_observation(1, _KNOWN_AT), knowledge_time=_KNOWN_AT + _DAY))
        session.commit()
    observation, retraction = _stored(fact_store)
    assert observation == (0, 11.25, 4_200, 1)
    assert retraction == (1, None, None, None)


def test_orm_flush_of_a_legacy_retraction_writes_no_fabricated_values(
    fact_store: sa.Engine,
) -> None:
    """A writer that still fills the payload gets NULLs written, not its numbers.

    This is the shape every retraction had before P2.10 and the reason the
    ``before_insert`` hook clears rather than refuses: the row still lands,
    but the numbers it invented do not reach the store.
    """
    legacy = _observation(1, _KNOWN_AT)
    legacy.is_retraction = True
    with Session(fact_store) as session:
        session.add(legacy)
        session.commit()
    (row,) = _stored(fact_store)
    assert row == (1, None, None, None)


def _core_insert(engine: sa.Engine, *, is_retraction: bool, close_usd: float | None) -> None:
    """Insert one price_bar row through raw SQL, bypassing every ORM-level hook.

    Deliberately a driver-level statement: it is the write path the ORM hooks
    cannot see (Core ``insert()``, ``COPY``, ``psql``), which is exactly what
    the CHECK constraints have to hold against on their own. ``close_usd`` is
    a float because SQLite's driver has no ``Decimal`` binding; the value is
    the test's own, not data.
    """
    columns = "security_id, valid_from, valid_to, knowledge_time, is_retraction"
    values: dict[str, object] = {
        "security_id": 1,
        "valid_from": _VALID_FROM.isoformat(),
        "valid_to": (_VALID_FROM + _DAY).isoformat(),
        "knowledge_time": _KNOWN_AT.isoformat(),
        "is_retraction": is_retraction,
    }
    placeholders = ":security_id, :valid_from, :valid_to, :knowledge_time, :is_retraction"
    if close_usd is not None:
        columns += ", close_usd"
        placeholders += ", :close_usd"
        values["close_usd"] = close_usd
    with engine.begin() as connection:
        connection.execute(
            sa.text(f"INSERT INTO price_bar ({columns}) VALUES ({placeholders})"),  # noqa: S608
            values,
        )


def test_check_refuses_a_retraction_carrying_a_payload(fact_store: sa.Engine) -> None:
    """The storage-level guarantee: no write path can put a number on a retraction."""
    with pytest.raises(IntegrityError, match="retraction_payload_absent"):
        _core_insert(fact_store, is_retraction=True, close_usd=11.25)


def test_check_refuses_an_observation_missing_a_required_value(fact_store: sa.Engine) -> None:
    """Dropping NOT NULL did not weaken observations: the CHECK carries it now."""
    with pytest.raises(IntegrityError, match="observation_payload_present"):
        _core_insert(fact_store, is_retraction=False, close_usd=11.25)


def test_check_accepts_a_bare_retraction_written_through_raw_sql(
    fact_store: sa.Engine,
) -> None:
    """The complement: a retraction that states nothing is exactly what is allowed."""
    _core_insert(fact_store, is_retraction=True, close_usd=None)
    assert _stored(fact_store) == [(1, None, None, None)]


def test_check_accepts_an_observation_omitting_only_optional_values(
    fact_store: sa.Engine,
) -> None:
    """A column the *source* may not state stays optional (SecurityMaster listing dates)."""
    with Session(fact_store) as session:
        session.add(
            SecurityMaster(
                security_id=1,
                ticker="AAPL",
                name="Apple Inc.",
                exchange="XNAS",
                valid_from=_VALID_FROM,
                valid_to=_VALID_FROM + _DAY,
                knowledge_time=_KNOWN_AT,
            )
        )
        session.commit()
    with fact_store.connect() as connection:
        (row,) = connection.exec_driver_sql("SELECT ticker, first_listed_on FROM security_master")
    assert row == ("AAPL", None)


# ---------------------------------------------------------------------------
# Read path: the statement-level mask backstop
# ---------------------------------------------------------------------------


def test_rewritten_statement_masks_every_touched_fact_table() -> None:
    """The rewriter's output satisfies the invariant the hook asserts on it."""
    statement = sa.select(PriceBar, SecurityMaster).join(
        SecurityMaster, SecurityMaster.security_id == PriceBar.security_id
    )
    touched = frozenset({"price_bar", "security_master"})
    _assert_retraction_mask(_rewrite_select(statement, touched, _KNOWN_AT), touched)


def test_column_only_select_is_masked_too() -> None:
    """A select of bare columns is versioned and masked exactly like an entity select."""
    statement = sa.select(PriceBar.security_id, PriceBar.close_usd)
    touched = frozenset({"price_bar"})
    _assert_retraction_mask(_rewrite_select(statement, touched, _KNOWN_AT), touched)


def test_unmasked_fact_table_is_refused() -> None:
    """An unversioned statement reaches price_bar with no mask and must not execute."""
    with pytest.raises(RetractionMaskError, match="price_bar"):
        _assert_retraction_mask(sa.select(PriceBar), frozenset({"price_bar"}))


def test_partial_mask_is_refused() -> None:
    """One masked table out of two is still a leak, and is named in the message."""
    touched = frozenset({"price_bar"})
    rewritten = _rewrite_select(sa.select(PriceBar), touched, _KNOWN_AT)
    joined = rewritten.join(SecurityMaster, SecurityMaster.security_id == PriceBar.security_id)
    with pytest.raises(RetractionMaskError, match="security_master"):
        _assert_retraction_mask(joined, frozenset({"price_bar", "security_master"}))


def test_a_mask_in_a_nested_scope_is_not_credited_to_its_parent() -> None:
    """An inner select's mask says nothing about what the outer one can return."""
    inner = sa.select(PriceBar.security_id).where(~PriceBar.is_retraction).subquery()
    outer = sa.select(PriceBar).where(PriceBar.security_id.in_(sa.select(inner.c.security_id)))
    with pytest.raises(RetractionMaskError, match="price_bar"):
        _assert_retraction_mask(outer, frozenset({"price_bar"}))


# ---------------------------------------------------------------------------
# Read path: the row-level load backstop
# ---------------------------------------------------------------------------


def test_load_guard_is_registered_on_every_mapper() -> None:
    assert sa.event.contains(Mapper, "load", _refuse_loaded_retraction)


def test_loading_a_retraction_raises() -> None:
    """Whatever query produced it, a retraction never reaches application code."""
    retraction = retract_fact(_observation(7, _KNOWN_AT), knowledge_time=_KNOWN_AT + _DAY)
    with pytest.raises(RetractedFactError, match="price_bar"):
        _refuse_loaded_retraction(retraction, cast("Any", None))


def test_loading_an_observation_is_allowed() -> None:
    _refuse_loaded_retraction(_observation(7, _KNOWN_AT), cast("Any", None))


def test_load_guard_ignores_non_bitemporal_rows() -> None:
    _refuse_loaded_retraction(Security(), cast("Any", None))


def test_load_guard_does_not_fire_when_is_retraction_was_not_loaded() -> None:
    """A column-restricted load cannot show the row to be a retraction.

    Reading the attribute here would emit a lazy SELECT from inside a load
    event; the guard reads the instance dict instead, so an unloaded column
    means "cannot tell" and enforcement stays with the statement-level mask.
    """
    partial = PriceBar(security_id=7)
    assert "is_retraction" not in sa.inspect(partial).dict
    _refuse_loaded_retraction(partial, cast("Any", None))
