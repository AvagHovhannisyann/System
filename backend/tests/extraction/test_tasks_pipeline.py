"""P7.3 pipeline tests: what the model client actually receives, and what is stored.

The one property worth more than all the others here is that **anonymization
happened before anything left the process**, and it is asserted the only way it
can honestly be asserted: by injecting a recording model client, running real
EDGAR text through the pipeline, and inspecting the requests the client was
handed. Asserting it inside the pipeline would only check the code against
itself; a caller can be lied to by a masker that ran and did nothing, and the
request is the last place the truth is visible.

The documents are the captured EDGAR responses in ``backend/tests/fixtures/edgar/``
(shared with P3.2 and P7.2), and the entity declarations are transcribed from
their own headers — see ``edgar_text.py``. Nothing here paraphrases a filing.

**The live path is unexercised, and none of these tests claims otherwise.** B4
is unresolved: there is no provider key in this repository or its CI, so no code
under test has ever spoken to a real model. Every model response below comes
from a deliberate test double whose canned text is *this test's* input, not a
provider's output. A double is legitimate scaffolding for testing the pipeline
around the call; presenting it as evidence about a provider's behaviour would be
the fabricated verification I3 forbids.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.extraction.entities import company, identifier
from backend.extraction.leak import detect_leaks
from backend.extraction.tasks import (
    ChunkingConfig,
    ChunkingPolicy,
    DocumentTooLargeError,
    EmptyDocumentError,
    ExtractionOutput,
    ExtractionPipeline,
    ExtractionTask,
    ModelRequest,
    ModelResponse,
    ProviderNotConfiguredError,
    SourceDocument,
    TaskNotRegisteredError,
    TaskRegistry,
    UnconfiguredModelClient,
    builtin_tasks,
    chunk_document,
    paired_document,
    qualified_model,
)
from backend.extraction.tasks.store import InMemoryResultStore
from backend.extraction.temporal import year_exemption_reason
from backend.tests.extraction.edgar_text import (
    ADAPTHEALTH_FORM4,
    SAP_6K,
    adapthealth_entities,
    read,
    sap_entities,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

_MODEL = "anthropic:a-cost-tier-model"
"""A model identifier for the tests. Names no real deployment: B4 leaves none."""


# --------------------------------------------------------------------------
# Test doubles. Neither pretends to be a provider.
# --------------------------------------------------------------------------


class RecordingClient:
    """A model client that records every request and replays canned responses.

    The canned text is supplied by the test that constructs it, so it is this
    file's input rather than any provider's output. Its purpose is to make the
    request *observable*, which is what the anonymization assertions need.
    """

    def __init__(self, responses: Sequence[str] | str = "{}") -> None:
        """Build the double.

        Args:
            responses: One response per call, in order, or a single string
                replayed for every call.
        """
        self.requests: list[ModelRequest] = []
        self._responses = [responses] if isinstance(responses, str) else list(responses)

    @property
    def calls(self) -> int:
        """How many requests this client was handed (count)."""
        return len(self.requests)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request and return the next canned response."""
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self._responses) - 1)
        return ModelResponse(
            text=self._responses[index],
            model=request.model,
            input_tokens=11,
            output_tokens=7,
            latency_ms=1.5,
        )


class ExplodingClient:
    """A client that fails the test if it is ever called.

    Used where the pipeline must refuse *before* reaching the network. A test
    that only checked the raised exception would pass just as happily if the
    call had been made and then discarded.
    """

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Fail loudly."""
        msg = f"the pipeline reached the model client, which it must not have: {request.model!r}"
        raise AssertionError(msg)


class _Score(ExtractionOutput):
    """A minimal delta output, for exercising the pipeline rather than a task."""

    tone_shift: float
    note: str


def _task(
    *,
    name: str = "probe_task",
    system: str = "Compare CURRENT against PRIOR and report the change.",
    policy: ChunkingPolicy = ChunkingPolicy.PER_CHUNK,
) -> ExtractionTask:
    """Build a small task for pipeline tests."""
    return ExtractionTask(
        name=name,
        output_model=_Score,
        system=system,
        template="$document\n",
        chunking_policy=policy,
    )


_VALID = '{"tone_shift": 0.25, "note": "hedging increased"}'
"""A response satisfying :class:`_Score`. Written here; not produced by a model."""


# --------------------------------------------------------------------------
# The property that matters: what left the process was masked
# --------------------------------------------------------------------------


def _adapthealth_document() -> SourceDocument:
    """A real captured EDGAR filing with the entities its own header declares."""
    return SourceDocument(
        document_id="0001193805-24-000360",
        text=read(ADAPTHEALTH_FORM4),
        entities=tuple(adapthealth_entities()),
    )


async def test_the_client_receives_masked_text_and_never_the_declared_names() -> None:
    """Every declared company, person and identifier is absent from what was sent.

    Run against a real filing, and checked against the *original* first: an
    assertion that a name is missing from the payload proves nothing if the name
    was never in the document. ``occurrences_in_original`` makes that visible,
    so this test cannot pass vacuously.
    """
    document = _adapthealth_document()
    client = RecordingClient(_VALID)
    pipeline = ExtractionPipeline(client=client, chunking=ChunkingConfig(max_chars=1_000_000))

    run = await pipeline.run(_task(), document, model=_MODEL)

    assert client.calls == 1
    sent = client.requests[0].prompt
    report = detect_leaks(original=document.text, anonymized=sent, entities=list(document.entities))
    assert report.leaks == (), report.summary()
    # Not vacuous: the scanners found these names in the source before masking.
    assert report.occurrences_in_original["AdaptHealth Corp."] > 0
    assert report.occurrences_in_original["0001725255"] > 0
    # And the payload is the masked text, not something that merely omitted them.
    assert "[COMPANY_1]" in sent
    assert sent == f"{run.payload}\n"


async def test_the_client_never_receives_the_document_id_or_the_entity_mapping() -> None:
    """Provenance stays on this side of the seam.

    The document id is how the platform names a filing; sending it would hand
    the model the identifier anonymization exists to remove. The mapping is the
    one artifact that reverses masking, so it must not be reachable from any
    request either.
    """
    document = _adapthealth_document()
    client = RecordingClient(_VALID)
    pipeline = ExtractionPipeline(client=client, chunking=ChunkingConfig(max_chars=1_000_000))

    run = await pipeline.run(_task(), document, model=_MODEL, correlation_id="corr-1")

    request = client.requests[0]
    whole_request = f"{request.system}\n{request.prompt}\n{request.model}"
    assert document.document_id not in whole_request
    assert "corr-1" not in whole_request
    # The mapping exists — masking really happened — and it stayed in process.
    assert run.anonymized.mapping.entries
    for entry in run.anonymized.mapping.entries:
        assert entry.canonical not in whole_request


async def test_every_declared_date_writing_is_gone_from_what_was_sent() -> None:
    """§5-P7 strips *all* dates, and the sent payload is where that is checkable.

    The temporal scanners in :mod:`backend.extraction.leak` are written
    independently of the masker precisely so they can disagree with it; running
    them over the request body is what turns "we called the anonymizer" into
    "the anonymizer worked on this text".
    """
    document = SourceDocument(
        document_id="0001104659-24-082105",
        text=read(SAP_6K),
        entities=tuple(sap_entities()),
    )
    client = RecordingClient(_VALID)
    pipeline = ExtractionPipeline(client=client, chunking=ChunkingConfig(max_chars=1_000_000))

    await pipeline.run(_task(), document, model=_MODEL)

    sent = client.requests[0].prompt
    report = detect_leaks(original=document.text, anonymized=sent, entities=list(document.entities))
    assert report.leaks == (), report.summary()
    # Independently of the detector: every four-digit number still in the
    # payload must carry a *stated* exemption. "Securities Exchange Act of
    # 1934" is a statute, not a period, and masking it would damage the text
    # for nothing; anything without a reason would be a date that survived.
    survivors = [
        (match.group(), year_exemption_reason(sent, match.start(), match.end()))
        for match in re.finditer(r"\d{4}", sent)
    ]
    unexplained = [value for value, reason in survivors if reason is None]
    assert not unexplained, f"unmasked calendar years reached the model: {unexplained}"


async def test_the_request_carries_temperature_zero_and_the_tasks_own_limits() -> None:
    """§5-P7: temperature 0 everywhere. Asserted on the request, not on a constant."""
    task = _task()
    client = RecordingClient(_VALID)
    pipeline = ExtractionPipeline(client=client)

    await pipeline.run(task, SourceDocument("d1", "some text"), model=_MODEL, timeout_s=12.5)

    request = client.requests[0]
    assert request.temperature == 0.0
    assert request.max_tokens == task.max_tokens
    assert request.timeout_s == 12.5
    assert request.system == task.system
    assert request.model == _MODEL


# --------------------------------------------------------------------------
# Refusals, all of them before a call is made
# --------------------------------------------------------------------------


async def test_no_client_means_a_refusal_and_never_a_synthesized_response() -> None:
    """B4 is unresolved, so the default client raises rather than inventing a value (I3)."""
    pipeline = ExtractionPipeline()
    assert isinstance(pipeline._client, UnconfiguredModelClient)
    with pytest.raises(ProviderNotConfiguredError, match="no model client is configured"):
        await pipeline.run(_task(), SourceDocument("d1", "text"), model=_MODEL)


@pytest.mark.parametrize("temperature", [0.1, 0.7, 1.0])
async def test_a_non_zero_temperature_is_refused_before_the_client_is_touched(
    temperature: float,
) -> None:
    """A sampled extraction is neither reproducible (I2) nor honestly cacheable."""
    pipeline = ExtractionPipeline(client=ExplodingClient())
    with pytest.raises(ValueError, match="refusing to cache a call at temperature"):
        await pipeline.run(
            _task(), SourceDocument("d1", "text"), model=_MODEL, temperature=temperature
        )


async def test_a_whole_document_task_refuses_a_payload_that_needs_two_calls() -> None:
    """Half of a comparison is a different question, and its answer would look real."""
    pipeline = ExtractionPipeline(client=ExplodingClient(), chunking=ChunkingConfig(max_chars=100))
    document = SourceDocument("d1", "paragraph one\n\n" + ("word " * 400))
    with pytest.raises(DocumentTooLargeError, match="requires the whole document in one call"):
        await pipeline.run(_task(policy=ChunkingPolicy.WHOLE_DOCUMENT), document, model=_MODEL)


async def test_an_empty_document_is_refused_rather_than_sent() -> None:
    """A response to an empty payload is not an extraction of anything."""
    pipeline = ExtractionPipeline(client=ExplodingClient())
    with pytest.raises(EmptyDocumentError, match="produced no chunks"):
        await pipeline.run(_task(), SourceDocument("d1", ""), model=_MODEL)


async def test_a_document_that_cannot_name_itself_is_refused_at_construction() -> None:
    """An extraction that cannot say which document it came from is not reproducible (I2)."""
    with pytest.raises(ValueError, match="document_id must be non-empty"):
        SourceDocument("   ", "text")


# --------------------------------------------------------------------------
# Validation: rejection is recorded, never swallowed and never repaired
# --------------------------------------------------------------------------


async def test_a_valid_response_is_parsed_and_stored_with_the_prompt_version_hash() -> None:
    """§5-P7's last step, asserted on the stored record rather than the return value."""
    task = _task()
    results = InMemoryResultStore()
    pipeline = ExtractionPipeline(client=RecordingClient(_VALID), results=results)

    run = await pipeline.run(task, SourceDocument("d1", "text"), model=_MODEL, correlation_id="c1")

    assert len(results) == 1
    record = results.records[0]
    assert record.prompt_version_hash == task.prompt.version_hash
    assert record.raw_response == _VALID
    assert record.model == _MODEL
    assert record.schema_valid
    assert isinstance(record.output, _Score)
    assert record.output.tone_shift == 0.25
    assert run.outputs == (record.output,)
    assert run.schema_failures == 0


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ('```json\n{"tone_shift": 0.1, "note": "x"}\n```', "a fence is a rejection, not a fixup"),
        ('{"tone_shift": "0.1", "note": "x"}', "strict mode does not coerce a quoted number"),
        ('{"tone_shift": 0.1, "note": "x", "company": "Acme"}', "an extra key is a rejection"),
        ("not json at all", "prose is not an extraction"),
        ("", "an empty response satisfies nothing"),
    ],
)
async def test_a_rejected_response_is_still_stored_raw_with_its_errors(raw: str, why: str) -> None:
    """A schema failure is data about the prompt (Gate G7 counts them), not an exception.

    The raw response is kept because re-asking at temperature 0 reproduces the
    same malformed answer and costs money to rediscover.
    """
    results = InMemoryResultStore()
    pipeline = ExtractionPipeline(client=RecordingClient(raw), results=results)

    run = await pipeline.run(_task(), SourceDocument("d1", "text"), model=_MODEL)

    record = results.records[0]
    assert not record.schema_valid, why
    assert record.output is None
    assert record.validation_errors
    assert record.raw_response == raw
    assert run.schema_failures == 1
    assert run.outputs == ()


async def test_a_rejected_response_is_cached_so_it_is_not_paid_for_twice() -> None:
    """The failure is recorded by storing the response, never by omitting the entry."""
    client = RecordingClient("not json at all")
    pipeline = ExtractionPipeline(client=client)
    document = SourceDocument("d1", "text")

    await pipeline.run(_task(), document, model=_MODEL)
    second = await pipeline.run(_task(), document, model=_MODEL)

    assert client.calls == 1
    assert second.chunks[0].cache_hit
    assert not second.chunks[0].schema_valid


# --------------------------------------------------------------------------
# Deltas, not states (§5-P7)
# --------------------------------------------------------------------------


def test_pairing_a_document_with_itself_is_refused() -> None:
    """A delta of zero by construction, recorded as a measurement, is a fabrication."""
    doc = SourceDocument("acc-1", "text")
    with pytest.raises(ValueError, match="cannot pair document"):
        paired_document(doc, doc)


def test_a_paired_document_marks_both_halves_and_unions_their_entities() -> None:
    """The comparison happens inside one call, so both halves must be in one payload."""
    previous = SourceDocument("acc-1", "older text", (company("Acme Corp."),))
    current = SourceDocument("acc-2", "newer text", (company("Acme Corp."), identifier("0001")))

    pair = paired_document(previous, current)

    assert pair.document_id == "acc-1->acc-2"
    assert pair.text.index("older text") < pair.text.index("newer text")
    assert "PRIOR DOCUMENT" in pair.text
    assert "CURRENT DOCUMENT" in pair.text
    # De-duplicated on the whole entity: the shared company is declared once.
    assert len(pair.entities) == 2


async def test_the_two_halves_of_a_pair_share_one_placeholder_namespace() -> None:
    """The same subject must be recognisable across both halves or a delta means nothing.

    This is why the pipeline masks the whole document before chunking rather
    than the other way round: per-chunk masking would give the two halves
    independent placeholder numbering.
    """
    previous = SourceDocument("acc-1", "Acme Corp. faces supply risk.", (company("Acme Corp."),))
    current = SourceDocument("acc-2", "Acme Corp. still faces it.", (company("Acme Corp."),))
    client = RecordingClient(_VALID)
    pipeline = ExtractionPipeline(client=client)

    await pipeline.run(
        _task(policy=ChunkingPolicy.WHOLE_DOCUMENT),
        paired_document(previous, current),
        model=_MODEL,
    )

    sent = client.requests[0].prompt
    assert "Acme" not in sent
    assert sent.count("[COMPANY_1]") == 2


def test_every_builtin_task_measures_a_change_rather_than_a_level() -> None:
    """§5-P7: extract deltas, not states. Pinned against the output schemas themselves.

    A predictor fed levels learns the roster — tone level is largely a property
    of the industry and the drafting firm. So every field of every built-in task
    is either an explicit change, a count reported for *both* documents (which
    is a change expressed as its two terms), or one of the two documented
    non-signal fields. A future author adding a level-only task fails here.
    """
    non_signal = {"evidence", "confidence"}
    paired_counts = {"declined_questions_current", "declined_questions_prior"}
    set_delta = {"added", "removed", "retained_count"}
    permitted_levels = {"magnitude_direction", "magnitude_size", "policy_change_disclosed"}
    for task in builtin_tasks():
        fields = set(task.output_model.model_fields)
        delta_like = {
            name
            for name in fields
            if name.endswith(("_shift", "_gap")) or name in paired_counts or name in set_delta
        }
        assert delta_like, f"{task.name} reports no change at all"
        leftover = fields - delta_like - non_signal - permitted_levels
        assert not leftover, f"{task.name} carries level-only fields: {sorted(leftover)}"


def test_every_builtin_task_reads_a_whole_document_and_forbids_identification() -> None:
    """A delta task cannot answer from half a comparison, and must not be invited to guess."""
    for task in builtin_tasks():
        assert task.chunking_policy is ChunkingPolicy.WHOLE_DOCUMENT
        assert "Do not guess" in task.system
        assert "single JSON object" in task.system
        # The schema is embedded, so instruction and validator cannot drift.
        assert '"properties"' in task.system


# --------------------------------------------------------------------------
# Task construction and the registry
# --------------------------------------------------------------------------


def test_a_template_asking_for_anything_but_the_payload_is_refused() -> None:
    """A template naming ``$ticker`` is asking to undo the anonymization."""
    with pytest.raises(ValueError, match=r"must take exactly \$document"):
        ExtractionTask(
            name="bad",
            output_model=_Score,
            system="s",
            template="Company $ticker said: $document",
        )


def test_a_tasks_prompt_address_follows_its_own_text_and_schema() -> None:
    """The address is derived on every read, so it cannot be stale (cache invalidation)."""
    original = _task()
    reworded = _task(system=original.system + " Be brief.")
    assert original.prompt.version_hash != reworded.prompt.version_hash
    assert _task().prompt.version_hash == original.prompt.version_hash


def test_the_registry_refuses_two_tasks_under_one_name() -> None:
    """Two questions sharing a name would share a prompt history and a golden score."""
    with pytest.raises(ValueError, match="duplicate extraction task name"):
        TaskRegistry((_task(name="same"), _task(name="same", system="different")))


def test_the_registry_raises_for_an_unknown_name_rather_than_returning_none() -> None:
    """A run must not proceed against a silently absent task."""
    with pytest.raises(TaskNotRegisteredError, match="no extraction task named"):
        builtin_tasks().get("no_such_task")


@pytest.mark.parametrize(
    ("provider", "model"),
    [("", "m"), ("anthropic", ""), (" anthropic", "m"), ("anthropic", "a:b")],
)
def test_a_model_identifier_that_would_be_ambiguous_is_refused(provider: str, model: str) -> None:
    """The qualified identifier is part of a cache address and is read by humans."""
    with pytest.raises(ValueError, match=r"must be non-empty|may contain"):
        qualified_model(provider, model)


def test_the_qualified_model_identifier_names_provider_and_model() -> None:
    """Two providers serving a model of the same name are two different answers."""
    assert qualified_model("anthropic", "some-model") == "anthropic:some-model"


# --------------------------------------------------------------------------
# Chunking: the two properties the cache and the probe depend on
# --------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    text=st.text(min_size=0, max_size=2000),
    max_chars=st.integers(min_value=1, max_value=200),
)
def test_chunks_tile_the_document_exactly(text: str, max_chars: int) -> None:
    """No character is dropped, duplicated or reordered.

    A dropped span is invisible: the extraction simply never sees that sentence
    and reports confidently about the rest.
    """
    chunks = chunk_document(text, ChunkingConfig(max_chars=max_chars))
    assert "".join(chunk.text for chunk in chunks) == text
    if not text:
        assert chunks == ()
        return
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for earlier, later in itertools.pairwise(chunks):
        assert earlier.end == later.start
    for index, chunk in enumerate(chunks):
        assert chunk.index == index
        assert chunk.text == text[chunk.start : chunk.end]
        assert 0 < len(chunk.text) <= max_chars


@settings(max_examples=100, deadline=None)
@given(text=st.text(min_size=0, max_size=2000), max_chars=st.integers(1, 200))
def test_chunking_is_deterministic(text: str, max_chars: int) -> None:
    """The cache address is a digest of the payload, so a drifting chunker misses every entry."""
    config = ChunkingConfig(max_chars=max_chars)
    assert chunk_document(text, config) == chunk_document(text, config)


def test_a_chunking_config_that_cannot_produce_chunks_is_refused() -> None:
    """Zero characters per chunk is not a small chunk, it is no chunk."""
    with pytest.raises(ValueError, match="max_chars must be >= 1"):
        ChunkingConfig(max_chars=0)


async def test_a_per_chunk_task_extracts_each_chunk_independently() -> None:
    """One *distinct* chunk is one call, one cache address and one stored response.

    Distinct, not "one call per chunk": the address is a digest of the payload,
    so two byte-identical chunks of one document share an entry and the second
    is a hit. That is the cache working, and pinning it here keeps a future
    author from reading a call count as a chunk count.
    """
    client = RecordingClient(_VALID)
    pipeline = ExtractionPipeline(client=client, chunking=ChunkingConfig(max_chars=60))
    text = "\n\n".join(
        f"paragraph {index} discusses {'supply chains' if index % 2 else 'litigation'} "
        f"and mentions figure {index * 7}"
        for index in range(5)
    )

    run = await pipeline.run(_task(), SourceDocument("d1", text), model=_MODEL)

    assert len(run.chunks) > 1
    assert client.calls == len({chunk.payload_digest for chunk in run.chunks})
    assert [chunk.chunk_index for chunk in run.chunks] == list(range(len(run.chunks)))
    assert {chunk.chunk_count for chunk in run.chunks} == {len(run.chunks)}
