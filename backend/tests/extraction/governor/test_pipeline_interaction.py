"""P7.7 against the real P7.3 pipeline: governance without touching the pipeline.

The governed client is a
:class:`~backend.extraction.tasks.client.ModelClient`, so
:class:`~backend.extraction.tasks.pipeline.ExtractionPipeline` takes it as-is.
That is the point of the wrapper shape, and it is proved here against the real
pipeline rather than against a hand-rolled caller: every chunk of every document
is authorized, a document that would breach mid-run stops mid-run, and nothing
in ``backend/extraction/tasks/`` was edited to make it so.

**A known cross-module gap is pinned here, deliberately.**
``test_a_degraded_run_files_the_answer_under_the_requested_model_known_gap``
asserts behaviour that is *wrong* and says so in its name. The pipeline computes
``ChunkExtraction.model`` and the cache address from the model it was *asked
for*, both before the client is reached, so a degrading governor's substitute
answer is filed under the primary's name. P7.7 does not own
``backend/extraction/tasks/pipeline.py`` and will not silently work around it;
the assertion exists so the gap cannot be forgotten, so the ledger's role as the
record of which model answered is visible, and so the fix flips a documented
assertion rather than surprising someone.
"""

from __future__ import annotations

import pytest

from backend.extraction.entities import company
from backend.extraction.governor.caps import CapPolicy
from backend.extraction.governor.errors import SpendCapExceededError
from backend.extraction.governor.governor import CostGovernor
from backend.extraction.governor.guard import GovernedModelClient
from backend.extraction.governor.ledger import InMemorySpendLedger, SpendEvent
from backend.extraction.tasks import (
    ChunkingConfig,
    ExtractionOutput,
    ExtractionPipeline,
    ExtractionTask,
    SourceDocument,
)
from backend.tests.extraction.governor.doubles import (
    CHEAP_MODEL,
    PRIMARY_MODEL,
    RecordingClient,
    cap,
    cap_book,
    clock,
    price_book,
)

_VALID = '{"tone_shift": 0.25, "note": "hedging increased"}'
"""A response satisfying :class:`_Score`. Written here; not produced by a model."""


class _Score(ExtractionOutput):
    """A minimal delta output, for exercising the pipeline rather than a task."""

    tone_shift: float
    note: str


def _task() -> ExtractionTask:
    """Build a small per-chunk task."""
    return ExtractionTask(
        name="governed_probe",
        output_model=_Score,
        system="Compare CURRENT against PRIOR and report the change.",
        template="$document\n",
        max_tokens=256,
    )


def _document(paragraphs: int = 4) -> SourceDocument:
    """Build a document long enough to need several chunks."""
    text = "\n\n".join(
        f"Northwind Trading Company reported a change in paragraph {index}."
        for index in range(paragraphs)
    )
    return SourceDocument(
        document_id="governed-doc-1",
        text=text,
        entities=(company("Northwind Trading Company"),),
    )


def _pipeline(
    inner: RecordingClient,
    *,
    daily: str,
    monthly: str = "100.00",
    policy: CapPolicy = CapPolicy.HALT,
    degrade_to: str | None = None,
) -> tuple[ExtractionPipeline, InMemorySpendLedger]:
    """Wire a real pipeline whose client is governed."""
    ledger = InMemorySpendLedger()
    governor = CostGovernor(
        caps=cap_book(cap(daily=daily, monthly=monthly, policy=policy, degrade_to=degrade_to)),
        prices=price_book(),
        ledger=ledger,
        clock=clock(),
    )
    pipeline = ExtractionPipeline(
        client=GovernedModelClient(inner=inner, governor=governor),
        chunking=ChunkingConfig(max_chars=80),
    )
    return pipeline, ledger


async def test_every_chunk_of_a_run_is_authorized_individually() -> None:
    """Per call, not per run: a document does not get one permission for N calls."""
    inner = RecordingClient(text=_VALID, input_tokens=40, output_tokens=20)
    pipeline, ledger = _pipeline(inner, daily="10.00")
    document = _document()

    run = await pipeline.run(_task(), document, model=PRIMARY_MODEL)

    assert len(run.chunks) > 1
    reserved = [row for row in await ledger.records() if row.event is SpendEvent.RESERVED]
    assert len(reserved) == len(run.chunks) == inner.calls


async def test_a_run_stops_at_the_cap_and_the_remaining_chunks_are_never_sent() -> None:
    """The refusal propagates out of the pipeline, which is the intended behaviour.

    A backfill that hits its cap should stop loudly, not quietly continue on the
    documents that happened to fit.
    """
    inner = RecordingClient(text=_VALID, input_tokens=None, output_tokens=None)
    # One chunk's bound is roughly USD 0.037; two fit under USD 0.09, three do not.
    pipeline, ledger = _pipeline(inner, daily="0.09")
    document = _document()

    with pytest.raises(SpendCapExceededError):
        await pipeline.run(_task(), document, model=PRIMARY_MODEL)

    assert inner.calls == 2
    reserved = [row for row in await ledger.records() if row.event is SpendEvent.RESERVED]
    assert len(reserved) == 2


async def test_a_cache_hit_costs_nothing_because_the_client_is_never_reached() -> None:
    """Governance sits behind the cache, so a hit spends no headroom.

    Not an accident of ordering — the pipeline consults the cache before the
    client, and the governed client is the client. A governor placed in front of
    the cache would charge for answers it already had.
    """
    inner = RecordingClient(text=_VALID, input_tokens=40, output_tokens=20)
    pipeline, ledger = _pipeline(inner, daily="10.00")
    document = _document()
    task = _task()

    first = await pipeline.run(task, document, model=PRIMARY_MODEL)
    calls_after_first = inner.calls
    rows_after_first = len(await ledger.records())

    second = await pipeline.run(task, document, model=PRIMARY_MODEL)

    assert second.stats.hits == len(second.chunks)
    assert first.stats.hits == 0
    assert inner.calls == calls_after_first
    assert len(await ledger.records()) == rows_after_first


async def test_a_degraded_run_files_the_answer_under_the_requested_model_known_gap() -> None:
    """The substitute answers, the ledger records it, and the extraction row does not.

    **This pins a defect, not a design.** ``ChunkExtraction.model`` and the cache
    address are both computed in
    :mod:`backend.extraction.tasks.pipeline` from the model the caller asked for,
    before the client is reached, so a degraded answer is attributed to a model
    that never read the document (I2). P7.7 does not own that file. Until it
    records ``ModelResponse.model`` instead, the spend ledger is the record of
    which model answered — and this assertion is what stops that from being
    forgotten.
    """
    inner = RecordingClient(text=_VALID, input_tokens=40, output_tokens=20)
    pipeline, ledger = _pipeline(
        inner, daily="0.02", policy=CapPolicy.DEGRADE, degrade_to=CHEAP_MODEL
    )
    document = _document(paragraphs=1)

    run = await pipeline.run(_task(), document, model=PRIMARY_MODEL)

    # What actually happened: the cheap model was asked, and the ledger says so.
    assert inner.models == (CHEAP_MODEL,)
    rows = await ledger.records()
    assert all(row.served_model == CHEAP_MODEL for row in rows)
    assert all(row.requested_model == PRIMARY_MODEL for row in rows)
    assert all(row.degraded is True for row in rows)
    # The gap: the extraction record names the model that was asked for.
    assert run.model == PRIMARY_MODEL
    assert run.chunks[0].model == PRIMARY_MODEL
