"""Running one delta task over one document (P7.4).

Four steps, in this order, with no way to skip one:

1. **Resolve the baseline at the anchor.**
   :func:`~backend.extraction.tasks.delta.resolve.resolve_prior_filing` opens
   the store at the current document's own knowledge time. No argument of
   :meth:`DeltaExtractor.extract` can point it elsewhere — see below.
2. **Load the baseline's text**, from a caller-supplied
   :class:`PriorTextSource`. ``edgar_filing`` stores a manifest and a URL, never
   a document body, so the body has to come from somewhere else and this module
   will not pretend otherwise (I3).
3. **Compose the pair and hand it to the pipeline.** Anonymization, chunking,
   the model call and schema validation all happen inside
   :class:`~backend.extraction.tasks.pipeline.ExtractionPipeline`, which is the
   only thing here that talks to a client.
4. **Return one of three outcomes**
   (:mod:`backend.extraction.tasks.delta.outcome`), never a number that could be
   mistaken for another.

Why there is no ``as_of`` parameter
------------------------------------

:meth:`DeltaExtractor.extract` takes a task spec, a document and a model. It
takes no ``as_of``, ``anchor``, ``instant``, ``now``, ``session``, ``asof_ts``
or ``knowledge_time``, and neither does anything it calls. The anchor is
:attr:`CurrentDocument.knowledge_time`, reached through
:meth:`~backend.extraction.tasks.delta.anchor.KnowledgeAnchor.of`, and there is
nothing to configure it with.

That is a structural claim rather than a convention, so it is asserted
structurally — from the AST of every module in this package and again through
``inspect.signature`` on the runtime objects — in
``backend/tests/extraction/delta/test_runner.py``. The same discipline D-033
applied to the absence of a ``venue`` parameter on the execution package, and
for the same reason: the failure it prevents is a later edit adding a
"convenience" override that reads the store at a instant nobody audited.

Why this object never holds a model client
-------------------------------------------

It holds an :class:`~backend.extraction.tasks.pipeline.ExtractionPipeline` and
calls :meth:`~backend.extraction.tasks.pipeline.ExtractionPipeline.run`. It has
no client attribute, so there is no path from a delta task to a provider that
bypasses anonymization, the cache, temperature 0 or schema validation. Under B4
the pipeline's default client raises
(:class:`~backend.extraction.tasks.client.ProviderNotConfiguredError`), and a
governed client (P7.7) refuses without configured caps — both of which surface
here as an exception rather than as an outcome, because "no model answered" is
not an observation about a filing.

What leaves the process
------------------------

One string per call: the rendered prompt built from the **anonymized** payload.
Both halves of the pair are masked together, in one pass, so the same entity
carries the same placeholder in both — the property that makes "the same risk
factor" identifiable across two filings. The entity declarations are the union
of a floor derived from each document's own ``edgar_filing`` row (filer name,
CIK) and whatever the caller and the text source add; the floor is never
dropped, so a source that declares nothing still gets the filer masked.

Units: instants are timezone-aware UTC; ``gap`` on a stamp is wall clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from backend.core.logging import get_logger
from backend.extraction.tasks.base import SourceDocument, paired_document
from backend.extraction.tasks.client import DEFAULT_TEMPERATURE, DEFAULT_TIMEOUT_S
from backend.extraction.tasks.delta.anchor import BaselineIdentityError, KnowledgeAnchor
from backend.extraction.tasks.delta.outcome import (
    DeltaMeasured,
    DeltaRejected,
    DeltaStamp,
    NoComparisonPossible,
    NoComparisonReason,
)
from backend.extraction.tasks.delta.resolve import resolve_prior_filing

if TYPE_CHECKING:
    from backend.extraction.entities import Entity
    from backend.extraction.tasks.delta.anchor import CurrentDocument, PriorDocumentRef
    from backend.extraction.tasks.delta.outcome import DeltaOutcome
    from backend.extraction.tasks.delta.spec import DeltaTaskSpec
    from backend.extraction.tasks.pipeline import ChunkExtraction, ExtractionPipeline

__all__ = ["DeltaExtractor", "PriorDocumentText", "PriorTextSource"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PriorDocumentText:
    """The body of a baseline document, plus anything extra known about it.

    Attributes:
        text: the document text, before anonymization. Must be non-blank — an
            empty half would make the comparison a comparison with nothing.
        entities: entity declarations **in addition to** the floor derived from
            the filing row (filer name, CIK). Former conformed names, tickers
            and named officers belong here: ``edgar_filing`` stores none of
            them, so a text source that has them should pass them. They are
            unioned with the floor, never substituted for it.
    """

    text: str
    entities: tuple[Entity, ...] = ()

    def __post_init__(self) -> None:
        """Reject an empty body.

        Raises:
            ValueError: if ``text`` is blank. A source that has no text must
                return ``None`` so the run records
                :attr:`~backend.extraction.tasks.delta.outcome.NoComparisonReason.PRIOR_TEXT_UNAVAILABLE`,
                rather than hand back an empty string that would be compared
                against a real document.
        """
        if not self.text.strip():
            msg = (
                "a prior document's text must be non-empty; return None to record "
                "PRIOR_TEXT_UNAVAILABLE instead of comparing a document against nothing (I3)"
            )
            raise ValueError(msg)


@runtime_checkable
class PriorTextSource(Protocol):
    """Something that can supply a baseline document's body.

    A seam for the same reason
    :class:`~backend.extraction.tasks.client.ModelClient` is one: the body is
    not in this platform's store. ``edgar_filing`` and
    ``edgar_filing_document`` carry a manifest and a URL (P3.2), so fetching a
    body means a network call to EDGAR, and inventing a table to hold one would
    be inventing a data source (I3).

    There is deliberately **no default implementation and no default
    argument**. A default that returned ``None`` would make every run in the
    platform record "prior text unavailable" forever, which reads as a fact
    about issuers rather than as an unwired collaborator.
    """

    async def load(self, ref: PriorDocumentRef) -> PriorDocumentText | None:
        """Return the baseline's body, or ``None`` when this source does not have it.

        Implementations must not synthesize, summarize or substitute a document;
        ``None`` is the honest answer and it becomes a recorded outcome.
        """
        ...


def _merged_entities(*groups: tuple[Entity, ...]) -> tuple[Entity, ...]:
    """Return the union of several entity declarations, order preserved."""
    merged: dict[Entity, None] = {}
    for group in groups:
        for entity in group:
            merged.setdefault(entity, None)
    return tuple(merged)


class DeltaExtractor:
    """Runs one delta task over one document, at that document's own anchor.

    Holds two collaborators and constructs neither, so a caller decides what a
    run costs and what it reads:

    * an :class:`~backend.extraction.tasks.pipeline.ExtractionPipeline`, which
      owns the model client, the cache, the anonymizer configuration and the
      result store. This object never touches a client directly;
    * a :class:`PriorTextSource`, because the baseline's body is not in the
      store.

    It holds no clock, no session and no anchor.
    """

    def __init__(self, *, pipeline: ExtractionPipeline, text_source: PriorTextSource) -> None:
        """Build the extractor.

        Args:
            pipeline: the extraction pipeline. Under B4 its default client
                raises, so a caller who supplied no client finds out at the
                first call rather than at the first suspicious number.
            text_source: where a baseline document's body comes from. Required:
                there is no default, because a default would answer "no text"
                for every document in the platform and look like data.
        """
        self._pipeline = pipeline
        self._text_source = text_source

    async def extract(
        self,
        spec: DeltaTaskSpec,
        current: CurrentDocument,
        *,
        model: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        correlation_id: str | None = None,
    ) -> DeltaOutcome:
        """Run ``spec`` over ``current`` and the document that preceded it.

        Note the parameters that are **absent**: there is no as-of, anchor,
        instant, clock or session argument. The baseline query is answered at
        ``current.knowledge_time`` and there is nothing to point it elsewhere
        (module docstring).

        Args:
            spec: which delta task to run, and which family of documents it
                compares.
            current: the document the delta is extracted for. Its
                ``knowledge_time`` is the anchor.
            model: qualified model identifier
                (:func:`~backend.extraction.tasks.pipeline.qualified_model`).
            timeout_s: per-request wall-clock timeout in seconds.
            correlation_id: request id (D-003) recorded against persisted
                results, or ``None``. Never sent to the model.

        Returns:
            A :class:`~backend.extraction.tasks.delta.outcome.DeltaMeasured`,
            :class:`~backend.extraction.tasks.delta.outcome.DeltaRejected` or
            :class:`~backend.extraction.tasks.delta.outcome.NoComparisonPossible`.
            The last is not a delta of zero and carries no number at all.

        Raises:
            backend.extraction.tasks.delta.anchor.BaselineSourceUnavailableError:
                the task's baseline store does not exist (earnings-call
                transcripts, P3.6/B1). A blocker, not a data condition.
            backend.extraction.tasks.delta.anchor.DocumentClassMismatchError:
                ``current``'s form is not read by this task.
            backend.extraction.tasks.delta.anchor.BaselineTemporalIntegrityError:
                a row arrived that was not knowable at the anchor — a defect in
                the read path.
            backend.extraction.tasks.client.ProviderNotConfiguredError: no model
                client is configured (B4). "No model answered" is not an
                observation about a filing, so it propagates.
            backend.extraction.tasks.pipeline.DocumentTooLargeError: the
                composed pair did not fit in one call. Answering from one half
                would be a fabricated comparison (I3).
        """
        anchor = KnowledgeAnchor.of(current)
        prior = await resolve_prior_filing(current, document_class=spec.document_class)
        if prior is None:
            return self._no_comparison(
                spec,
                current,
                anchor,
                reason=NoComparisonReason.NO_PRIOR_DOCUMENT,
                detail=(
                    f"no {spec.document_class.name} of form {current.form_type!r} by CIK "
                    f"{current.cik} was knowable at {anchor.instant.isoformat()}. This is "
                    "the absence of a baseline, not a delta of zero"
                ),
            )
        loaded = await self._text_source.load(prior)
        if loaded is None:
            return self._no_comparison(
                spec,
                current,
                anchor,
                reason=NoComparisonReason.PRIOR_TEXT_UNAVAILABLE,
                detail=(
                    f"baseline {prior.accession_number!r} was knowable at "
                    f"{anchor.instant.isoformat()} but its text could not be loaded. This "
                    "is a gap in this platform, not a fact about the issuer"
                ),
            )

        _verify_pair(prior, current, anchor)
        pair = paired_document(
            SourceDocument(
                document_id=prior.accession_number,
                text=loaded.text,
                entities=_merged_entities(prior.entities, loaded.entities),
            ),
            SourceDocument(
                document_id=current.accession_number,
                text=current.text,
                entities=current.declared_entities,
            ),
        )
        run = await self._pipeline.run(
            spec.task,
            pair,
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            timeout_s=timeout_s,
            correlation_id=correlation_id,
        )
        chunk = _sole_chunk(run.chunks, task=spec.name, document_id=pair.document_id)
        stamp = DeltaStamp(
            task=spec.name,
            prompt_version_hash=chunk.prompt_version_hash,
            model=chunk.model,
            payload_digest=chunk.payload_digest,
            anchor=anchor.instant,
            prior_document_id=prior.accession_number,
            current_document_id=current.accession_number,
            prior_knowledge_time=prior.knowledge_time,
            current_knowledge_time=current.knowledge_time,
            gap=prior.gap_to(current),
        )
        if chunk.output is None:
            _logger.warning(
                "delta_extraction_rejected",
                task=spec.name,
                current_document_id=current.accession_number,
                prior_document_id=prior.accession_number,
                prompt_version_hash=chunk.prompt_version_hash,
                errors=list(chunk.validation_errors),
            )
            return DeltaRejected(
                stamp=stamp,
                raw_response=chunk.raw_response,
                validation_errors=chunk.validation_errors,
                cache_hit=chunk.cache_hit,
            )
        return DeltaMeasured(
            stamp=stamp,
            output=chunk.output,
            raw_response=chunk.raw_response,
            cache_hit=chunk.cache_hit,
        )

    @staticmethod
    def _no_comparison(
        spec: DeltaTaskSpec,
        current: CurrentDocument,
        anchor: KnowledgeAnchor,
        *,
        reason: NoComparisonReason,
        detail: str,
    ) -> NoComparisonPossible:
        """Build and log a no-comparison record.

        Logged at info because it is an ordinary outcome — first-time
        registrants exist — but logged nonetheless, because a *rate* of these
        rising is how an ingestion gap announces itself.
        """
        _logger.info(
            "delta_no_comparison",
            task=spec.name,
            reason=reason.value,
            current_document_id=current.accession_number,
            anchor=anchor.instant.isoformat(),
        )
        return NoComparisonPossible(
            task=spec.name,
            reason=reason,
            anchor=anchor,
            current_document_id=current.accession_number,
            detail=detail,
        )


def _verify_pair(
    prior: PriorDocumentRef, current: CurrentDocument, anchor: KnowledgeAnchor
) -> None:
    """Re-check the pair's direction and anchor **before** anything is spent.

    :func:`~backend.extraction.tasks.delta.anchor.select_baseline` guarantees
    both properties, and
    :class:`~backend.extraction.tasks.delta.outcome.DeltaStamp` refuses a pair
    that violates either — but the stamp is built *after* the model has answered.
    Checking here means a backwards pair costs nothing and, more importantly,
    never reaches a provider: a comparison run in the wrong direction returns a
    perfectly well-formed answer with every sign inverted, which is the one
    failure mode that survives schema validation untouched.

    Args:
        prior: the chosen baseline.
        current: the document the delta is for.
        anchor: the instant the baseline query was answered at.

    Raises:
        BaselineIdentityError: the baseline was not accepted strictly before the
            current document.
        ValueError: the baseline was chosen at a different instant from the one
            this run will stamp the result with.
    """
    if prior.acceptance >= current.acceptance:
        msg = (
            f"baseline {prior.accession_number!r} was accepted at "
            f"{prior.acceptance.isoformat()}, at or after the current document's "
            f"{current.acceptance.isoformat()}. Sending this pair would produce a "
            "well-formed answer with every sign inverted (I1)"
        )
        raise BaselineIdentityError(msg)
    if prior.chosen_at != anchor:
        msg = (
            f"baseline {prior.accession_number!r} was chosen at "
            f"{prior.chosen_at.instant.isoformat()} but this run is anchored at "
            f"{anchor.instant.isoformat()}; the result would be stamped with an instant "
            "that did not choose it (I1, I2)"
        )
        raise ValueError(msg)


def _sole_chunk(
    chunks: tuple[ChunkExtraction, ...], *, task: str, document_id: str
) -> ChunkExtraction:
    """Return the single chunk a ``WHOLE_DOCUMENT`` delta task must have produced.

    Args:
        chunks: the run's chunk records.
        task: the task name, for the message.
        document_id: the composed pair's identifier, for the message.

    Returns:
        The one record.

    Raises:
        RuntimeError: if there is not exactly one. Every delta spec declares
            ``WHOLE_DOCUMENT`` chunking and the pipeline raises
            :class:`~backend.extraction.tasks.pipeline.DocumentTooLargeError`
            before splitting one, so reaching here means those two guarantees
            disagree — and picking a chunk would silently report a comparison
            computed from part of the pair.
    """
    if len(chunks) != 1:
        msg = (
            f"delta task {task!r} produced {len(chunks)} chunk records for {document_id!r}; "
            "a paired comparison is exactly one call. Selecting one would report an answer "
            "computed from part of the pair as though it were the whole comparison (I3)"
        )
        raise RuntimeError(msg)
    return chunks[0]
