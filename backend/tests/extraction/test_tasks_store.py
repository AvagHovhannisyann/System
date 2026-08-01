"""P7.3 storage tests: the record kept, and the schema that refuses to let it be edited.

Two halves, both without a database:

* the in-memory result store's behaviour, and the value conversions the
  PostgreSQL one performs before binding — the ``NUMERIC`` columns are bound as
  :class:`~decimal.Decimal` on purpose, and a float slipping through is the kind
  of bug that only shows up as ``312.69999999999999`` on somebody's latency
  chart;
* migration 0010's own text: the revision chain, the append-only triggers, and
  the two schema decisions worth pinning against a future author who would
  "tidy them up".

The durable round-trips need real PostgreSQL. Migration 0010 runs as part of
``alembic upgrade head`` in the integration fixture, so a broken revision fails
that entire suite rather than passing quietly here.
"""

from __future__ import annotations

import datetime as dt
import importlib
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import pytest
from sqlalchemy.dialects.postgresql import JSONB

from backend.db.models import (
    ExtractionGoldenScore,
    ExtractionPromptActivation,
    ExtractionPromptVersion,
    ExtractionResult,
)
from backend.extraction.tasks.pipeline import ChunkExtraction
from backend.extraction.tasks.schema import ExtractionOutput
from backend.extraction.tasks.store import (
    InMemoryResultStore,
    PostgresResultStore,
    _latency,
    _output_json,
)

_MIGRATION = "0010_extraction_framework"


def _load(module_name: str) -> ModuleType:
    """Import one migration revision module."""
    return importlib.import_module(f"backend.db.migrations.versions.{module_name}")


class _Score(ExtractionOutput):
    """A minimal output model with a tuple field, to check JSON serialization."""

    tone_shift: float
    evidence: tuple[str, ...]


def _extraction(*, valid: bool = True) -> ChunkExtraction:
    """Build one extraction record."""
    output = _Score(tone_shift=0.25, evidence=("a quoted line",)) if valid else None
    return ChunkExtraction(
        task="probe_task",
        document_id="acc-1->acc-2",
        chunk_index=0,
        chunk_count=1,
        prompt_version_hash="0" * 32,
        payload_digest="f" * 64,
        model="anthropic:a-cost-tier-model",
        raw_response='{"tone_shift": 0.25, "evidence": ["a quoted line"]}',
        output=output,
        validation_errors=() if valid else ("tone_shift: Input should be a valid number",),
        cache_hit=False,
        input_tokens=11,
        output_tokens=7,
        latency_ms=312.7,
        extracted_at=dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC),
    )


# --------------------------------------------------------------------------
# The in-memory store
# --------------------------------------------------------------------------


async def test_the_in_memory_store_appends_and_never_replaces() -> None:
    """Re-running is a new record: two rows disagreeing is itself the observation."""
    store = InMemoryResultStore()
    await store.record(_extraction(), correlation_id="c1")
    await store.record(_extraction(), correlation_id="c1")

    assert len(store) == 2
    assert store.records[0] == store.records[1]


async def test_a_rejected_response_is_recorded_with_its_errors_and_no_output() -> None:
    """A schema failure is data about the prompt, so the record must carry both facts."""
    store = InMemoryResultStore()
    await store.record(_extraction(valid=False), correlation_id=None)

    record = store.records[0]
    assert record.output is None
    assert record.validation_errors
    assert not record.schema_valid
    assert record.raw_response


def test_both_result_stores_answer_the_same_protocol() -> None:
    """A caller must not be able to depend on which implementation it holds."""
    from backend.extraction.tasks.pipeline import ResultSink

    assert isinstance(InMemoryResultStore(), ResultSink)
    assert isinstance(PostgresResultStore(), ResultSink)


# --------------------------------------------------------------------------
# Value conversion before binding
# --------------------------------------------------------------------------


def test_a_validated_output_is_serialized_as_json_scalars() -> None:
    """A tuple must become a list, or a JSONB round trip would not return what was accepted."""
    payload = _output_json(_extraction())
    assert payload == {"tone_shift": 0.25, "evidence": ["a quoted line"]}
    assert isinstance(payload["evidence"], list)


def test_a_rejected_response_serializes_to_null_rather_than_an_empty_object() -> None:
    """``{}`` would be an output that validated to nothing; NULL is the recorded outcome."""
    assert _output_json(_extraction(valid=False)) is None


def test_a_missing_output_binds_as_sql_null_and_not_as_json_null() -> None:
    """Regression: without ``none_as_null`` a rejected response cannot be stored at all.

    SQLAlchemy's default is to bind Python ``None`` into a JSONB column as JSON
    ``null``, which is **not** SQL NULL — so ``output IS NOT NULL`` is true, and
    the ``output_xor_validation_errors`` CHECK rejects every rejected-response
    row. This surfaced only against a real PostgreSQL; the flag is asserted here
    so a future edit to the column cannot quietly reintroduce it.

    It is the same SQL-NULL-versus-JSON-null trap :mod:`backend.db.audit`
    documents, arriving from the write side. Here SQL NULL is the right storage:
    "there is no output" is an absence, not a JSON value.
    """
    column = ExtractionResult.__table__.c["output"]
    assert isinstance(column.type, JSONB)
    assert column.type.none_as_null is True
    assert column.nullable is True


def test_latency_is_bound_as_a_decimal_at_the_columns_own_scale() -> None:
    """Binding a float to NUMERIC is what turns 312.7 into 312.69999999999999 on a chart."""
    value = _latency(312.7)
    assert isinstance(value, Decimal)
    assert value == Decimal("312.700")
    assert str(value) == "312.700"


def test_an_unmeasured_latency_stays_null_rather_than_becoming_zero() -> None:
    """Zero milliseconds is a measurement; a cache hit made no call to measure (I3)."""
    assert _latency(None) is None


# --------------------------------------------------------------------------
# Migration 0010
# --------------------------------------------------------------------------


def test_migration_0010_follows_0009() -> None:
    """The chain is linear, and 0009 exists to follow."""
    migration = _load(_MIGRATION)
    assert migration.revision == "0010"
    assert migration.down_revision == "0009"
    assert _load("0009_provider_registry").revision == "0009"


@pytest.mark.parametrize(
    "table",
    [
        "extraction_prompt_version",
        "extraction_prompt_activation",
        "extraction_golden_score",
        "extraction_result",
    ],
)
def test_every_extraction_table_carries_the_append_only_trigger(table: str) -> None:
    """An observation that can be edited afterwards is not evidence.

    Checked against the migration's own text: the trigger is what makes this a
    database guarantee rather than a convention the next author can forget.
    """
    source = Path(str(_load(_MIGRATION).__file__)).read_text()
    assert f"CREATE TRIGGER trg_{table}_append_only " in source
    assert f"BEFORE UPDATE OR DELETE ON {table} " in source


def test_the_extraction_tables_are_declared_by_both_the_orm_and_the_migration() -> None:
    """This repository hand-writes migrations, so agreement is asserted, not assumed."""
    source = Path(str(_load(_MIGRATION).__file__)).read_text()
    for model in (
        ExtractionPromptVersion,
        ExtractionPromptActivation,
        ExtractionGoldenScore,
        ExtractionResult,
    ):
        assert f'op.create_table(\n        "{model.__tablename__}"' in source


def test_a_result_row_must_say_whether_it_validated_or_not() -> None:
    """Two independently nullable columns would permit records nobody could act on.

    An output *and* errors, or neither, are both states a reader cannot
    interpret, so the schema forbids them outright.
    """
    source = Path(str(_load(_MIGRATION).__file__)).read_text()
    assert "(output IS NOT NULL) <> (jsonb_array_length(validation_errors) > 0)" in source


def test_agreement_is_constrained_to_the_unit_interval() -> None:
    """§8: a fraction, never a percentage. A caller passing 85 must fail at the database too."""
    source = Path(str(_load(_MIGRATION).__file__)).read_text()
    assert "agreement >= 0 AND agreement <= 1" in source


def test_an_extraction_result_is_not_foreign_keyed_to_the_prompt_library() -> None:
    """Refusing to record an observation to protect a join would discard the observation.

    A prompt is addressable whether or not anyone chose to save it to the
    library, so ``extraction_result`` records the hash and nothing constrains it.
    """
    source = Path(str(_load(_MIGRATION).__file__)).read_text()
    result_block = source.split('op.create_table(\n        "extraction_result"')[1]
    assert "ForeignKeyConstraint" not in result_block


def test_the_downgrade_drops_everything_the_upgrade_created() -> None:
    """A revision that cannot be undone is a one-way door nobody agreed to walk through."""
    source = Path(str(_load(_MIGRATION).__file__)).read_text()
    for table in (
        "extraction_prompt_version",
        "extraction_prompt_activation",
        "extraction_golden_score",
        "extraction_result",
    ):
        assert f'op.drop_table("{table}")' in source
    assert "DROP FUNCTION extraction_append_only_guard()" in source
