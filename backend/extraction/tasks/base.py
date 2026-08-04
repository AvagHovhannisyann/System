"""What an extraction task is, and what it is given (P7.3, §5-P7).

A task is *data*, not a class hierarchy: a name, an output model, a system
instruction, a template, and a chunking policy. Everything an implementation
would otherwise override — how to chunk, how to anonymize, how to build the
request, how to cache, how to validate — is the same for every task and lives in
:mod:`backend.extraction.tasks.pipeline`. Making tasks data rather than
subclasses means a new task cannot accidentally opt out of anonymization or of
temperature 0, because there is no method for it to override.

The one variable
----------------

Every extraction prompt takes exactly one variable, ``$document``, and it is
always the anonymized payload. That uniformity is what lets the pipeline treat
every task identically, and it is checked at construction rather than trusted:
a template referring to ``$ticker`` would be a prompt asking to be told the
thing anonymization exists to remove.

Deltas, not states
------------------

§5-P7 is explicit that extraction produces *changes*, not levels: the signal is
in how this filing differs from the last one, not in how gloomy this one sounds.
A tone level is largely a property of the industry and the drafting firm, and a
predictor fed levels learns the roster. So most tasks read a **pair** of
documents, composed into one payload by :func:`paired_document`, which puts the
prior document and the current one in one call under explicit section markers.

Composing rather than making two calls is deliberate: the comparison is the
task. Two independent scorings differenced afterwards measure the difference of
two noisy absolute judgements, which is noisier than the judgement of a
difference and is not the question §5-P7 asks.

The pair is anonymized as one document, so the same entity gets the same
placeholder in both halves — which matters, because "the risk factor about
``COMPANY_1``'s supplier" must be recognisable as the same subject across the
two filings for a delta to mean anything.

Chunking policy
---------------

A paired document is roughly twice the size of one filing section, and a delta
task cannot answer from half of it. :class:`ChunkingPolicy` therefore lets a
task say so: ``WHOLE_DOCUMENT`` means "if this does not fit in one call, raise".
Refusing is the honest behaviour — an answer computed from the first chunk of a
two-chunk comparison is a fabricated comparison (I3), and it would look exactly
like a real one in the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.extraction.prompts.versioning import PromptVersion
from backend.extraction.tasks.client import DEFAULT_MAX_TOKENS
from backend.extraction.tasks.schema import schema_digest

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from backend.extraction.entities import Entity
    from backend.extraction.tasks.schema import ExtractionOutput

__all__ = [
    "DOCUMENT_VARIABLE",
    "ChunkingPolicy",
    "ExtractionTask",
    "SourceDocument",
    "TaskNotRegisteredError",
    "TaskRegistry",
    "paired_document",
]

DOCUMENT_VARIABLE: Final = "document"
"""The single variable every extraction template substitutes: ``$document``."""

_PRIOR_MARKER: Final = "=== PRIOR DOCUMENT ==="
"""Section marker introducing the earlier document of a pair.

Deliberately free of names, dates and ordinal words that could hint at a period
(no "2023 10-K", no "last year"). The marker survives anonymization because it
contains nothing to mask, and it must: a model that cannot tell the two halves
apart cannot report a direction of change.
"""

_CURRENT_MARKER: Final = "=== CURRENT DOCUMENT ==="
"""Section marker introducing the later document of a pair."""


class ChunkingPolicy(StrEnum):
    """How a task tolerates a document being split across calls."""

    PER_CHUNK = "PER_CHUNK"
    """Each chunk is extracted independently; results are returned per chunk.

    For tasks whose unit of evidence is local — a passage of Q&A, a paragraph of
    accounting language. The pipeline does not aggregate across chunks: median
    aggregation across *models* is P7.5's job, and inventing a
    cross-chunk aggregation here would produce a document-level number nobody
    specified.
    """

    WHOLE_DOCUMENT = "WHOLE_DOCUMENT"
    """The document must fit in one call, or the pipeline raises.

    For tasks that compare two halves of a composed document: half a comparison
    is not a weaker comparison, it is a different and unstated question.
    """


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A document to extract from, with what is known to identify it.

    Attributes:
        document_id: Stable identifier — an EDGAR accession, a transcript id, or
            for a composed pair, a derived identifier naming both sides. Used
            for provenance on the stored result; **never** part of the cache
            address and never sent to a model.
        text: The document text, before anonymization.
        entities: What the ingestion metadata knows identifies it — filer and
            issuer names with former names, reporting owners, tickers, CIKs.
            Anonymization is metadata-driven, so an entity that is not declared
            is an entity that is not masked
            (:mod:`backend.extraction.anonymize` states the limits).
    """

    document_id: str
    text: str
    entities: tuple[Entity, ...] = ()

    def __post_init__(self) -> None:
        """Reject a document that cannot be identified in the store.

        Raises:
            ValueError: if ``document_id`` is empty or blank. An extraction that
                cannot say which document it came from is not reproducible (I2).
        """
        if not self.document_id.strip():
            msg = "document_id must be non-empty: an extraction must name its source (I2)"
            raise ValueError(msg)


def _merged_entities(*groups: Sequence[Entity]) -> tuple[Entity, ...]:
    """Return the union of several entity declarations, order preserved.

    De-duplicated on the whole entity, so the same company declared for both
    halves of a pair is masked once, under one placeholder.
    """
    merged: dict[Entity, None] = {}
    for group in groups:
        for entity in group:
            merged.setdefault(entity, None)
    return tuple(merged)


def paired_document(
    previous: SourceDocument,
    current: SourceDocument,
    *,
    document_id: str | None = None,
) -> SourceDocument:
    """Compose two documents into the single payload a delta task reads.

    The result is one document with two marked sections, so the comparison
    happens inside one model call and the two halves share one anonymization —
    and therefore one set of placeholders, which is what makes "the same risk
    factor" identifiable across them.

    Args:
        previous: The earlier document.
        current: The later document.
        document_id: Identifier for the composed document. Defaults to
            ``f"{previous.document_id}->{current.document_id}"``, which names
            both sides and their order.

    Returns:
        A :class:`SourceDocument` whose text is the prior section followed by
        the current one, and whose entities are the union of both declarations.

    Raises:
        ValueError: if the two documents have the same ``document_id`` — a
            document compared with itself has a delta of zero by construction,
            and storing that as a measurement would be a fabricated observation.
    """
    if previous.document_id == current.document_id:
        msg = (
            f"cannot pair document {previous.document_id!r} with itself: the delta is zero by "
            "construction, and recording it as an extraction would be a fabricated observation"
        )
        raise ValueError(msg)
    text = f"{_PRIOR_MARKER}\n{previous.text}\n\n{_CURRENT_MARKER}\n{current.text}\n"
    return SourceDocument(
        document_id=document_id or f"{previous.document_id}->{current.document_id}",
        text=text,
        entities=_merged_entities(previous.entities, current.entities),
    )


@dataclass(frozen=True, slots=True)
class ExtractionTask:
    """One extraction task: what to ask, of what shape, over how much text.

    Attributes:
        name: Task identity, e.g. ``"risk_factor_language_delta"``. Also the
            prompt's name, so a task's prompt history is addressed by it.
        output_model: The Pydantic model every response must satisfy. Its schema
            is embedded in the prompt *and* included in the prompt's content
            address, so instruction and validator cannot drift apart.
        system: System instruction.
        template: User message template, whose only variable is ``$document``.
        chunking_policy: Whether the task tolerates being split across calls.
        max_tokens: Response cap for this task in tokens (count). Extraction
            responses are small objects; a task returning lists of risk factors
            says so by raising this.
        description: One line for the operator's task list (§6.5).
    """

    name: str
    output_model: type[ExtractionOutput]
    system: str
    template: str
    chunking_policy: ChunkingPolicy = ChunkingPolicy.PER_CHUNK
    max_tokens: int = DEFAULT_MAX_TOKENS
    description: str = ""

    def __post_init__(self) -> None:
        """Reject a task whose prompt does not take exactly the payload.

        Raises:
            ValueError: if the template's variables are not exactly
                ``{"document"}``, or if ``max_tokens`` is below 1. A template
                asking for anything else is asking for something the pipeline
                will not give it — and a template asking for an identifier is
                asking to undo the anonymization.
        """
        if self.max_tokens < 1:
            msg = f"max_tokens must be >= 1 tokens; got {self.max_tokens}"
            raise ValueError(msg)
        variables = self.prompt.variables
        if variables != frozenset({DOCUMENT_VARIABLE}):
            msg = (
                f"task {self.name!r} template must take exactly ${DOCUMENT_VARIABLE} and "
                f"nothing else; got {sorted(variables)}. The payload is the only thing the "
                "pipeline supplies, and it is anonymized"
            )
            raise ValueError(msg)

    @property
    def prompt(self) -> PromptVersion:
        """This task's prompt version, content-addressed.

        Rebuilt on each access from the task's own fields, so the address always
        describes the text that will actually be sent — the property the cache's
        automatic invalidation rests on
        (:mod:`backend.extraction.cache`).
        """
        return PromptVersion(
            name=self.name,
            system=self.system,
            template=self.template,
            schema_digest=schema_digest(self.output_model),
        )

    def render(self, payload: str) -> str:
        """Render the user message for one anonymized payload.

        Args:
            payload: The anonymized text to extract from.

        Returns:
            The rendered message.
        """
        return self.prompt.render({DOCUMENT_VARIABLE: payload})


class TaskNotRegisteredError(LookupError):
    """A task was requested by a name the registry does not hold.

    Raised rather than returning ``None`` so a run cannot proceed against a
    silently absent task and store results under a name nothing defines.
    """


class TaskRegistry:
    """A named collection of extraction tasks.

    A plain object rather than a module-level mutable dict: import-time
    registration makes the set of tasks depend on which modules happened to be
    imported, which is exactly the kind of implicit configuration that makes a
    run irreproducible (I2). Callers construct a registry from an explicit list
    (:func:`backend.extraction.tasks.library.builtin_tasks`) and pass it around.
    """

    def __init__(self, tasks: Iterable[ExtractionTask] = ()) -> None:
        """Build a registry.

        Args:
            tasks: The tasks to hold.

        Raises:
            ValueError: if two tasks share a name. Two tasks under one name
                would share a prompt history and a golden-set score while
                asking different questions.
        """
        self._tasks: dict[str, ExtractionTask] = {}
        for task in tasks:
            if task.name in self._tasks:
                msg = f"duplicate extraction task name {task.name!r}"
                raise ValueError(msg)
            self._tasks[task.name] = task

    def __len__(self) -> int:
        """Number of registered tasks (count)."""
        return len(self._tasks)

    def __iter__(self) -> Iterator[ExtractionTask]:
        """Iterate the tasks in registration order."""
        return iter(self._tasks.values())

    def __contains__(self, name: object) -> bool:
        """Whether a task name is registered."""
        return name in self._tasks

    @property
    def names(self) -> tuple[str, ...]:
        """Registered task names, in registration order."""
        return tuple(self._tasks)

    def get(self, name: str) -> ExtractionTask:
        """Return one task by name.

        Args:
            name: The task name.

        Returns:
            The task.

        Raises:
            TaskNotRegisteredError: if no task has that name.
        """
        try:
            return self._tasks[name]
        except KeyError as exc:
            msg = f"no extraction task named {name!r}; registered: {sorted(self._tasks)}"
            raise TaskNotRegisteredError(msg) from exc
