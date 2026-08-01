"""Where an extraction record is kept (P7.3, §5-P7 step 5, §6.5).

§5-P7 ends its pipeline sentence with *"store with the prompt version hash"*,
and requires that raw responses are stored. This module is that store: one
append-only row per model call, holding the response verbatim, the validated
output when validation accepted it, the validation errors when it did not, and
the prompt version hash and model that produced both.

Why an extraction is append-only
--------------------------------

An extraction is an **observation** — this prompt, this model, this text, this
answer. An observation that can be edited afterwards is not evidence, and every
downstream consumer treats it as evidence: the golden set scores it (P7.8), the
contamination probe differences it (P7.9), the document inspector shows it to
an operator (§6.5), and the feature library eventually trains on it. Migration
0010 installs the same ``BEFORE UPDATE OR DELETE`` row trigger revisions
0003/0004/0007/0009 use, so this is enforced by the database rather than by
convention.

Re-running the same call is therefore a **new row**, not an overwrite. At
temperature 0 with the same payload, prompt and model the answer should be
identical — and if it is not, two rows saying different things is precisely the
observation worth keeping, because it means an assumption this whole design
rests on is false.

What is deliberately *not* stored
----------------------------------

The anonymization mapping. It is the one artifact that re-identifies a
document, and a durable copy of it beside a durable copy of the anonymized text
would make the anonymization decorative. The stored row carries
``payload_digest`` — enough to prove two extractions read the same text, and
not enough to reconstruct it.

The document text itself, in either form. The source lives in the ingestion
tables; the payload is reproducible from it, the declared entities and the
anonymizer configuration, all of which are deterministic
(:mod:`backend.extraction.anonymize`).

Units: token counts are counts **as reported by the provider**, never estimated
(I3); ``latency_ms`` is wall-clock milliseconds; timestamps are UTC.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

from backend.db import ingest_writer_session
from backend.db.models import ExtractionResult

if TYPE_CHECKING:
    from backend.extraction.tasks.pipeline import ChunkExtraction

__all__ = [
    "InMemoryResultStore",
    "PostgresResultStore",
]


def _output_json(extraction: ChunkExtraction) -> dict[str, Any] | None:
    """Return the validated output as JSON-able data, or ``None`` when rejected.

    Serialized through Pydantic's own JSON mode rather than ``dict()`` so that
    every value in the column is a JSON scalar — a tuple becomes a list, a float
    stays a float — and a round trip through JSONB returns what validation
    accepted rather than something that merely resembles it.
    """
    if extraction.output is None:
        return None
    dumped: dict[str, Any] = extraction.output.model_dump(mode="json")
    return dumped


def _latency(latency_ms: float | None) -> Decimal | None:
    """Return a measured latency as the column's ``NUMERIC``, or ``None``.

    Bound as :class:`~decimal.Decimal` rather than as a float: the column is
    ``NUMERIC(12, 3)``, and handing the driver a float there is what turns
    ``312.7`` into ``312.69999999999999`` in a latency-distribution chart
    (§6.5). Rounded to the column's own scale so the stored value is the value
    the caller can read back, not one the database quietly truncated.
    """
    if latency_ms is None:
        return None
    return Decimal(str(latency_ms)).quantize(Decimal("0.001"))


class InMemoryResultStore:
    """A :class:`~backend.extraction.tasks.pipeline.ResultSink` held in memory.

    Append-only like the durable one, and the reference for what the protocol
    means, so the pipeline's tests exercise the same semantics without a
    database. Not a cache of the durable store and never reads from one:
    anything recorded here lives until the process ends.

    Not thread-safe and not safe across event loops; the async method never
    awaits.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._records: list[ChunkExtraction] = []

    def __len__(self) -> int:
        """Records held (count)."""
        return len(self._records)

    @property
    def records(self) -> tuple[ChunkExtraction, ...]:
        """Every record, in the order it was written."""
        return tuple(self._records)

    async def record(self, extraction: ChunkExtraction, *, correlation_id: str | None) -> None:
        """Append one extraction record.

        Args:
            extraction: The record. Stored as given — it is a frozen dataclass,
                so nothing here can alter it afterwards.
            correlation_id: Request id (D-003), accepted for protocol
                compatibility and not retained: an in-memory store has no
                cross-request reader to serve it to.
        """
        del correlation_id
        self._records.append(extraction)


class PostgresResultStore:
    """A :class:`~backend.extraction.tasks.pipeline.ResultSink` backed by PostgreSQL.

    Writes through the sanctioned append-only write path
    (:func:`backend.db.ingest_writer_session`). The table is not bitemporal:
    an extraction has no market knowability — it is something *we* did to a
    document, like an ingestion run or a configuration change — and inventing a
    ``knowledge_time`` for it would be a fabricated value in the one column
    whose meaning is that it is not fabricated (I3). Same reasoning as
    revisions 0005, 0007 and 0009.
    """

    async def record(self, extraction: ChunkExtraction, *, correlation_id: str | None) -> None:
        """Insert one extraction record.

        Args:
            extraction: The record to persist.
            correlation_id: Request id (D-003) to record, or ``None`` outside a
                request.

        Raises:
            sqlalchemy.exc.IntegrityError: a CHECK or NOT NULL was violated —
                which for this table means the caller built a record the schema
                says cannot exist, and is a bug rather than a data condition.
        """
        row = ExtractionResult(
            task=extraction.task,
            document_id=extraction.document_id,
            chunk_index=extraction.chunk_index,
            chunk_count=extraction.chunk_count,
            prompt_version_hash=extraction.prompt_version_hash,
            payload_digest=extraction.payload_digest,
            model=extraction.model,
            raw_response=extraction.raw_response,
            output=_output_json(extraction),
            validation_errors=list(extraction.validation_errors),
            cache_hit=extraction.cache_hit,
            input_tokens=extraction.input_tokens,
            output_tokens=extraction.output_tokens,
            latency_ms=_latency(extraction.latency_ms),
            correlation_id=correlation_id,
        )
        async with ingest_writer_session() as session:
            session.add(row)
            await session.commit()

    async def for_document(self, task: str, document_id: str) -> tuple[ExtractionResult, ...]:
        """Return every stored record for one task and document, newest first.

        The document inspector's query (§6.5): every chunk, every raw response,
        and the prompt version each was produced under.

        Args:
            task: The extraction task's name.
            document_id: The source document's identifier.

        Returns:
            Rows ordered by ``result_id`` descending — insertion order, not the
            clock, for the reason :mod:`backend.db.audit` gives.
        """
        statement = (
            sa.select(ExtractionResult)
            .where(ExtractionResult.task == task, ExtractionResult.document_id == document_id)
            .order_by(ExtractionResult.result_id.desc())
        )
        async with ingest_writer_session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(rows)

    async def for_prompt_version(self, prompt_version_hash: str) -> tuple[ExtractionResult, ...]:
        """Return every stored record produced under one prompt version, newest first.

        What "re-run the golden set on this prompt version" reads, and what a
        quality-trend chart (§6.5) is built from.

        Args:
            prompt_version_hash: The prompt's content address.

        Returns:
            Rows ordered by ``result_id`` descending.
        """
        statement = (
            sa.select(ExtractionResult)
            .where(ExtractionResult.prompt_version_hash == prompt_version_hash)
            .order_by(ExtractionResult.result_id.desc())
        )
        async with ingest_writer_session() as session:
            rows = (await session.scalars(statement)).all()
        return tuple(rows)
