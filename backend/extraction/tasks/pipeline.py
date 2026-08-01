"""The extraction pipeline: document to schema-validated value (P7.3, §5-P7).

§5-P7 spells the pipeline out: *document → chunk → anonymize → extraction →
schema validation → store with the prompt version hash*. This module is that
sentence, executed once per chunk, with the model call injected
(:mod:`backend.extraction.tasks.client`) because B4 leaves the platform with no
provider key.

It is a function of its inputs and a small object holding four collaborators —
a model client, a cache, an optional result store, and the chunking and
anonymizer configuration. There is no subclassing seam and no per-task
override: a task is data (:mod:`backend.extraction.tasks.base`), so a new task
cannot opt out of anonymization, of temperature 0, or of schema validation,
because there is no method for it to override.

Anonymize first, then chunk — and why that is not the directive's order
-----------------------------------------------------------------------

§5-P7 lists chunking before anonymization. This module reverses them, and the
reversal is deliberate rather than careless:

* **Placeholder numbering is document-scoped.** Entity placeholders are fixed
  per declared entity, but temporal placeholders are *allocated in order of
  appearance* (:mod:`backend.extraction.rules`). Anonymizing each chunk
  separately would make ``[DATE_1]`` a different date in every chunk, and for
  a paired document (:func:`~backend.extraction.tasks.base.paired_document`)
  it would break the one property a delta task depends on — that the same
  subject carries the same placeholder in both halves.
* **The safety property is strictly stronger this way.** Masking the whole
  document once means no code path exists in which an unmasked substring is
  the unit of work: what gets chunked is already the payload. Chunk-then-mask
  leaves a window — a chunk in hand, not yet masked — and a window is
  something a future edit can widen.

The observable consequence is that :class:`~backend.extraction.tasks.chunking.Chunk`
offsets index the *anonymized* text, not the source. That is stated on
:attr:`ChunkExtraction.chunk_index` rather than left for a reader to discover.

What actually leaves the process
--------------------------------

Exactly one string per call: :attr:`~backend.extraction.tasks.client.ModelRequest.prompt`,
rendered from the anonymized chunk, plus the task's system instruction. No
document id, no entity list, no mapping, no correlation id. The mapping that
would reverse the masking is never given to the client and is never stored on a
result — it stays on the in-process :class:`ExtractionRun` for the operator's
document inspector (§6.5) and goes no further.

That is a property of *what the client receives*, so it is asserted where it
can be observed: ``backend/tests/extraction/test_tasks_pipeline.py`` injects a
recording client and checks the captured requests against the original names,
tickers and dates. Asserting it here — inside the code that builds the request
— would only be checking this module against itself.

Failures are recorded, not swallowed
-------------------------------------

A response that fails schema validation does not raise out of :meth:`run`. It
produces a :class:`ChunkExtraction` with ``output=None`` and the validation
errors attached, and the raw response is stored anyway — in the cache and on
the result. §5-P7 requires raw responses to be kept, and a rejected response is
the one worth keeping: it is the evidence a prompt needs work, and Gate G7 has
to be able to count them. Re-asking at temperature 0 would produce the same
malformed answer and cost money to rediscover.

A *call* failure (:class:`~backend.extraction.tasks.client.ModelCallError`) does
propagate. The two are different: one means the prompt is wrong, the other
means the infrastructure is. Merging them would have a retry loop hammering a
provider over a response it will reproduce exactly.

Units: ``latency_ms`` is wall-clock milliseconds; token counts are counts as
*reported by the provider* and are never estimated (I3); ``hit_rate`` is a
fraction in [0, 1], never a percentage (§8).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from backend.core.logging import get_logger
from backend.extraction.anonymize import anonymize
from backend.extraction.cache import (
    CacheKey,
    CacheStats,
    ExtractionCache,
    InMemoryCacheStore,
    assert_temperature_is_cacheable,
)
from backend.extraction.tasks.base import DOCUMENT_VARIABLE, ChunkingPolicy
from backend.extraction.tasks.chunking import ChunkingConfig, chunk_document
from backend.extraction.tasks.client import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    ModelClient,
    ModelRequest,
    UnconfiguredModelClient,
)
from backend.extraction.tasks.schema import (
    ExtractionOutput,
    SchemaValidationError,
    parse_extraction_output,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.extraction.anonymize import AnonymizerConfig
    from backend.extraction.entities import AnonymizedDocument
    from backend.extraction.prompts.versioning import PromptVersion
    from backend.extraction.tasks.base import ExtractionTask, SourceDocument

__all__ = [
    "ChunkExtraction",
    "DocumentTooLargeError",
    "EmptyDocumentError",
    "ExtractionError",
    "ExtractionPipeline",
    "ExtractionRun",
    "ResultSink",
    "qualified_model",
]

_logger = get_logger(__name__)

_MODEL_QUALIFIER: Final = ":"
"""Separator between a provider name and a provider-side model identifier.

``anthropic:claude-3-5-haiku-20241022``. Qualifying is not cosmetic: the model
is part of the cache address, and two providers serving a model of the same
name are two different answers to the same question.
"""


class ExtractionError(RuntimeError):
    """The pipeline refused to run a document. Never raised for a bad *response*."""


class EmptyDocumentError(ExtractionError):
    """The document produced no chunks, so there is nothing to extract from.

    An empty payload would become an empty model call, and whatever came back
    would be a response about nothing — recorded, indistinguishable from a real
    extraction, and wrong (I3).
    """


class DocumentTooLargeError(ExtractionError):
    """A ``WHOLE_DOCUMENT`` task's payload did not fit in one call.

    Refusing is the honest behaviour. Half of a two-document comparison is not
    a weaker comparison, it is a different and unstated question, and its answer
    would sit in the store looking exactly like a real one.
    """


def qualified_model(provider: str, model: str) -> str:
    """Return the cache-addressable model identifier for a provider and model.

    Args:
        provider: Provider name, e.g. ``"anthropic"``. Non-empty, unpadded.
        model: Provider-side model identifier, verbatim. Non-empty, unpadded.

    Returns:
        ``f"{provider}:{model}"`` — the string used as the ``model`` component
        of a cache key and recorded on every result.

    Raises:
        ValueError: if either part is empty, padded with whitespace, or already
            contains the separator. A model identifier that itself contains
            ``":"`` would make the qualified string ambiguous to read, and this
            string is read by humans in the document inspector.
    """
    for label, value in (("provider", provider), ("model", model)):
        if not value or value != value.strip():
            msg = f"{label} must be non-empty and unpadded; got {value!r}"
            raise ValueError(msg)
    if _MODEL_QUALIFIER in provider or _MODEL_QUALIFIER in model:
        msg = (
            f"neither provider nor model may contain {_MODEL_QUALIFIER!r}; "
            f"got provider={provider!r} model={model!r}"
        )
        raise ValueError(msg)
    return f"{provider}{_MODEL_QUALIFIER}{model}"


@dataclass(frozen=True, slots=True)
class ChunkExtraction:
    """One model call's worth of extraction: what was asked, and what came back.

    Attributes:
        task: The extraction task's name.
        document_id: The source document's identifier. Provenance only — it is
            never sent to a model and never part of a cache address.
        chunk_index: Position of this chunk, 0-based. Indexes the **anonymized**
            document, for the reason in the module docstring.
        chunk_count: How many chunks the document produced (count).
        prompt_version_hash: Content address of the prompt used
            (:attr:`~backend.extraction.prompts.versioning.PromptVersion.version_hash`).
            §5-P7 requires an extraction to be stored with this.
        payload_digest: Digest of the anonymized text actually sent.
        model: Qualified model identifier (:func:`qualified_model`).
        raw_response: The provider's response, **verbatim** (§5-P7).
        output: The validated output, or ``None`` when validation rejected the
            response. ``None`` is a recorded outcome, not an omission.
        validation_errors: One line per schema problem, empty when ``output``
            is present.
        cache_hit: Whether the response came from the cache rather than a call.
            **This is the field a spend calculation filters on.** A hit spent
            nothing, so summing token counts across every record would count
            the same call once per reuse.
        input_tokens: Prompt tokens **as reported by the provider** (count), or
            ``None``. Never estimated (I3). On a cache hit these describe the
            call that originally produced the response — this run spent none.
        output_tokens: Response tokens as reported (count), or ``None``. Same
            caveat on a hit.
        latency_ms: Wall-clock duration of **this run's** call in milliseconds;
            ``None`` on a cache hit (no call was made to time) or when the
            client did not measure it. Deliberately not carried over from the
            cached entry: a latency distribution (§6.5) built from reused
            measurements would describe a provider that was never asked.
        extracted_at: When this record was produced, UTC.
    """

    task: str
    document_id: str
    chunk_index: int
    chunk_count: int
    prompt_version_hash: str
    payload_digest: str
    model: str
    raw_response: str
    output: ExtractionOutput | None
    validation_errors: tuple[str, ...]
    cache_hit: bool
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    extracted_at: dt.datetime

    @property
    def schema_valid(self) -> bool:
        """Whether the response satisfied the task's output schema."""
        return self.output is not None


@dataclass(frozen=True, slots=True)
class ExtractionRun:
    """Everything one document produced, and the anonymized text it produced it from.

    Attributes:
        task: The extraction task's name.
        document_id: The source document's identifier.
        model: Qualified model identifier used for every chunk.
        prompt_version_hash: The prompt version used for every chunk.
        anonymized: The masking result — the payload actually chunked, plus the
            mapping that reverses it. Held **in process only**: the mapping is
            the one artifact capable of re-identifying a document, so it is
            never sent to a model, never written to the result store, and never
            logged. The operator's document inspector (§6.5) is its only
            intended reader.
        chunks: One record per chunk, in document order.
    """

    task: str
    document_id: str
    model: str
    prompt_version_hash: str
    anonymized: AnonymizedDocument
    chunks: tuple[ChunkExtraction, ...]

    @property
    def payload(self) -> str:
        """The anonymized text the chunks were cut from — what was actually read."""
        return self.anonymized.text

    @property
    def outputs(self) -> tuple[ExtractionOutput, ...]:
        """The validated outputs, in chunk order, skipping rejected responses.

        Deliberately *not* padded with ``None`` placeholders: a caller wanting
        to know which chunks failed reads :attr:`chunks`, where the failure sits
        next to the raw response that caused it.
        """
        return tuple(chunk.output for chunk in self.chunks if chunk.output is not None)

    @property
    def stats(self) -> CacheStats:
        """This run's own hit/miss counts, computed from its records.

        Derived from :attr:`chunks` rather than read off the shared
        :class:`~backend.extraction.cache.ExtractionCache` counter, so a run's
        hit rate is a property of the run and not of everything else the process
        happened to do first — which is the number Gate G7 asks for.
        """
        hits = sum(1 for chunk in self.chunks if chunk.cache_hit)
        return CacheStats(hits=hits, misses=len(self.chunks) - hits)

    @property
    def schema_failures(self) -> int:
        """Chunks whose response failed schema validation (count)."""
        return sum(1 for chunk in self.chunks if not chunk.schema_valid)


@runtime_checkable
class ResultSink(Protocol):
    """Somewhere an extraction record is durably kept.

    Append-only by contract: an extraction is an observation, and an observation
    that can be edited afterwards is not evidence. The durable implementation is
    :class:`backend.extraction.tasks.store.PostgresResultStore`; the in-memory
    one is :class:`backend.extraction.tasks.store.InMemoryResultStore`.
    """

    async def record(self, extraction: ChunkExtraction, *, correlation_id: str | None) -> None:
        """Persist one chunk's extraction record."""
        ...


class ExtractionPipeline:
    """Runs one task over one document: anonymize, chunk, call, validate, store.

    Holds the collaborators rather than constructing them, so a caller decides
    what a run costs: which model client (a real provider adapter, or a test
    double), which cache store (process-local or Redis), and whether results are
    persisted at all.

    The default client is
    :class:`~backend.extraction.tasks.client.UnconfiguredModelClient`, which
    raises. That is deliberate: with B4 unresolved, "no provider configured"
    must surface as a loud failure at the call site, never as a run that
    completes and produces plausible numbers (I3, §9.2).
    """

    def __init__(
        self,
        *,
        client: ModelClient | None = None,
        cache: ExtractionCache | None = None,
        results: ResultSink | None = None,
        chunking: ChunkingConfig | None = None,
        anonymizer: AnonymizerConfig | None = None,
    ) -> None:
        """Build a pipeline.

        Args:
            client: What executes a model request. ``None`` installs the
                refusing client, so a caller who forgot to supply one finds out
                at the first call instead of at the first suspicious number.
            cache: Where raw responses live and where hits are counted.
                ``None`` creates a fresh process-local cache.
            results: Where extraction records are persisted, or ``None`` to run
                without persistence (a dry run, or a test).
            chunking: Chunk sizing. ``None`` means the defaults.
            anonymizer: Which masking rules are active. ``None`` means the
                strict defaults, which is what extraction should run with —
                every switch that turns one off widens a re-identification
                channel (:mod:`backend.extraction.anonymize`).
        """
        self._client: ModelClient = client if client is not None else UnconfiguredModelClient()
        self._cache = cache if cache is not None else ExtractionCache(InMemoryCacheStore())
        self._results = results
        self._chunking = chunking if chunking is not None else ChunkingConfig()
        self._anonymizer = anonymizer

    @property
    def cache(self) -> ExtractionCache:
        """The cache this pipeline reads and writes, for inspection and stats resets."""
        return self._cache

    async def run(
        self,
        task: ExtractionTask,
        document: SourceDocument,
        *,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        correlation_id: str | None = None,
    ) -> ExtractionRun:
        """Extract ``task`` from ``document`` and return every chunk's record.

        Args:
            task: What to extract.
            document: What to extract it from, with its declared entities.
                Anonymization is metadata-driven, so an entity that is not
                declared is an entity that is not masked — see
                :mod:`backend.extraction.anonymize` for the limits.
            model: Qualified model identifier (:func:`qualified_model`). Part of
                the cache address, so it must name the model exactly and stably.
            temperature: Dimensionless sampling temperature. **Must be 0**
                (§5-P7); anything else is refused before a call is made, because
                a sampled response cannot be cached honestly and cannot be
                reproduced (I2).
            timeout_s: Per-request wall-clock timeout in seconds.
            correlation_id: Request id (D-003) recorded against persisted
                results, or ``None``. Never sent to the model.

        Returns:
            The :class:`ExtractionRun`.

        Raises:
            EmptyDocumentError: the document produced no chunks.
            DocumentTooLargeError: the task declares ``WHOLE_DOCUMENT`` and the
                payload needed more than one chunk.
            ValueError: ``temperature`` is not 0.
            backend.extraction.tasks.client.ModelCallError: the provider call
                failed. Propagated rather than recorded: an infrastructure
                failure is not an extraction result.
        """
        assert_temperature_is_cacheable(temperature)
        anonymized = anonymize(document.text, document.entities, self._anonymizer)
        chunks = chunk_document(anonymized.text, self._chunking)
        if not chunks:
            msg = (
                f"document {document.document_id!r} produced no chunks after anonymization; "
                "there is nothing to extract from an empty payload"
            )
            raise EmptyDocumentError(msg)
        if task.chunking_policy is ChunkingPolicy.WHOLE_DOCUMENT and len(chunks) > 1:
            msg = (
                f"task {task.name!r} requires the whole document in one call but "
                f"{document.document_id!r} needed {len(chunks)} chunks of at most "
                f"{self._chunking.max_chars} characters. Answering from one chunk would be a "
                "fabricated comparison (I3); raise max_chars or split the source document "
                "deliberately"
            )
            raise DocumentTooLargeError(msg)

        # Resolved once per run and passed down rather than re-read per chunk.
        # ``ExtractionTask.prompt`` rebuilds the version — and re-derives the
        # output model's JSON schema — on every access, which is the right
        # default (the address can never be stale) and the wrong thing to do
        # once per chunk across a backfill. The task is frozen, so one read per
        # run is the same answer as one read per chunk.
        prompt = task.prompt
        records: list[ChunkExtraction] = []
        for chunk in chunks:
            record = await self._extract_chunk(
                task,
                document,
                prompt=prompt,
                payload=chunk.text,
                chunk_index=chunk.index,
                chunk_count=len(chunks),
                model=model,
                temperature=temperature,
                timeout_s=timeout_s,
                correlation_id=correlation_id,
            )
            records.append(record)
        return ExtractionRun(
            task=task.name,
            document_id=document.document_id,
            model=model,
            prompt_version_hash=prompt.version_hash,
            anonymized=anonymized,
            chunks=tuple(records),
        )

    async def _extract_chunk(
        self,
        task: ExtractionTask,
        document: SourceDocument,
        *,
        prompt: PromptVersion,
        payload: str,
        chunk_index: int,
        chunk_count: int,
        model: str,
        temperature: float,
        timeout_s: float,
        correlation_id: str | None,
    ) -> ChunkExtraction:
        """Run one chunk: look up, call if missed, validate, record.

        The cache is consulted before the client is touched, so a hit costs
        nothing and the client never sees a request it did not need to answer —
        the property Gate G7's hit rate is a measurement of.

        ``temperature`` arrives already checked by :meth:`run` and is therefore
        known to be 0. It is threaded through rather than re-substituted with
        the constant so that the value on the wire is provably the value the
        caller asked for and the check passed, not a second constant that a
        future edit could let drift away from the first.
        """
        key = CacheKey.build(payload=payload, prompt=prompt, model=model)
        entry = await self._cache.lookup(key)
        cache_hit = entry is not None
        if entry is None:
            request = ModelRequest(
                model=model,
                system=task.system,
                prompt=prompt.render({DOCUMENT_VARIABLE: payload}),
                temperature=temperature,
                max_tokens=task.max_tokens,
                timeout_s=timeout_s,
            )
            response = await self._client.complete(request)
            entry = await self._cache.store_response(
                key,
                response.text,
                task=task.name,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
            )
        output: ExtractionOutput | None
        errors: tuple[str, ...]
        try:
            output = parse_extraction_output(entry.raw_response, task.output_model)
            errors = ()
        except SchemaValidationError as exc:
            output = None
            errors = exc.errors
            _logger.warning(
                "extraction_schema_rejected",
                task=task.name,
                document_id=document.document_id,
                chunk_index=chunk_index,
                model=model,
                prompt_version_hash=prompt.version_hash,
                errors=list(errors),
            )
        record = ChunkExtraction(
            task=task.name,
            document_id=document.document_id,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            prompt_version_hash=prompt.version_hash,
            payload_digest=key.payload_digest,
            model=model,
            raw_response=entry.raw_response,
            output=output,
            validation_errors=errors,
            cache_hit=cache_hit,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            latency_ms=entry.latency_ms if not cache_hit else None,
            extracted_at=dt.datetime.now(tz=dt.UTC),
        )
        if self._results is not None:
            await self._results.record(record, correlation_id=correlation_id)
        return record

    async def run_batch(
        self,
        task: ExtractionTask,
        documents: Sequence[SourceDocument],
        *,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        correlation_id: str | None = None,
    ) -> tuple[ExtractionRun, ...]:
        """Run one task over several documents, in order, and return every run.

        Sequential on purpose. Parallelism across *models* is P7.5's ensemble
        and parallelism across documents needs the cost governor (P7.7) to bound
        what a burst can spend; adding either here would spend money on a
        schedule nobody chose while B4 leaves no cap configured.

        Args:
            task: What to extract.
            documents: What to extract it from, in a stable order.
            model: Qualified model identifier.
            temperature: Must be 0 (§5-P7).
            timeout_s: Per-request timeout in seconds.
            correlation_id: Request id recorded against persisted results.

        Returns:
            One :class:`ExtractionRun` per document, in the order given.
        """
        return tuple(
            [
                await self.run(
                    task,
                    document,
                    model=model,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    correlation_id=correlation_id,
                )
                for document in documents
            ]
        )
