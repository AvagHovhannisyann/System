"""What a delta run produced — three outcomes that cannot be confused (P7.4, I2, I3).

A delta task can end in exactly three states, and the whole point of this module
is that they are three *types* rather than three values of one:

:class:`DeltaMeasured`
    Two documents were compared and a model answered in schema. The answer may
    be all zeros — **that is "no change", and it is a measurement.** Consecutive
    filings usually differ very little, so zero is the modal honest answer and
    the prompts say so.

:class:`DeltaRejected`
    Two documents were compared and the answer did not satisfy the schema. The
    raw response is kept, because §5-P7 requires it and because a rejected
    response is the evidence a prompt needs work.

:class:`NoComparisonPossible`
    **No comparison was made.** There was no baseline knowable at the anchor, or
    there was one and its text could not be loaded. No question reached a model.

Why "no change" and "no comparison possible" must not share a representation
----------------------------------------------------------------------------

Because the cheapest way to collapse them is a float, and a float of ``0.0``
means "measured, and unchanged" everywhere else in the feature library. A
missing baseline rendered as zero enters a cross-sectional z-score, gets ranked
against real measurements, and describes a company that filed for the first time
as one whose language did not move. Nothing downstream can tell the two apart
afterwards, and nothing about the number looks wrong.

This is the same error D-030 removed from retraction payloads (a retraction
carries no payload, enforced by CHECK, rather than an invented one) and the same
one D-031 removed from drift reporting (an unmeasurable feature becomes a row
with **no renderable numeric attribute**, never a zero on a chart). So
:class:`NoComparisonPossible` carries no number at all: no score, no count, no
``__float__``. There is nothing on it to plot, average, or z-score by accident,
and a consumer that wants to count them has to reach for
:attr:`NoComparisonPossible.reason`, which says which of the two things happened.

The two knowledge times, and which one is the feature's
--------------------------------------------------------

A delta reads a document from period *t-1* and a document from period *t*. It is
knowable at *t*, not at *t-1*: nobody could compute the comparison before the
later document existed. :class:`DeltaStamp` therefore carries both and derives
:attr:`~DeltaStamp.feature_knowledge_time` as the **later** of the two, with the
ordering enforced at construction so the derivation cannot pick the earlier one.
Getting this backwards would date every LLM feature a full reporting period
early — a lookahead of months, applied uniformly, which no distribution check
would notice.

Units: every instant is timezone-aware UTC. ``gap`` is a wall-clock duration
between the two documents' acceptance instants. Score fields live on the task
output models (:mod:`backend.extraction.tasks.library`) and are dimensionless.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.extraction.tasks.delta.anchor import KnowledgeAnchor
    from backend.extraction.tasks.schema import ExtractionOutput

__all__ = [
    "DeltaMeasured",
    "DeltaOutcome",
    "DeltaRejected",
    "DeltaStamp",
    "NoComparisonPossible",
    "NoComparisonReason",
]


class NoComparisonReason(StrEnum):
    """Why no comparison was made. Two reasons, and they are not the same fact."""

    NO_PRIOR_DOCUMENT = "NO_PRIOR_DOCUMENT"
    """The store held no eligible earlier document at the anchor instant.

    A statement about the issuer's filing history *as it was knowable then*: a
    first-time registrant, a filer that changed reporting regime, or a history
    that simply does not reach back that far. Nothing is wrong with the platform
    when this happens.
    """

    PRIOR_TEXT_UNAVAILABLE = "PRIOR_TEXT_UNAVAILABLE"
    """An eligible earlier document exists, but its text could not be loaded.

    A statement about the platform, not the issuer: ``edgar_filing`` stores a
    manifest and a URL, never the document body, so the text comes from a source
    the caller supplies and that source can be missing the document. Kept apart
    from :attr:`NO_PRIOR_DOCUMENT` because merging them would report an
    ingestion gap as a fact about a company — and the population of documents
    that never got fetched is not a random sample.
    """


@dataclass(frozen=True, slots=True)
class DeltaStamp:
    """Everything needed to regenerate one delta extraction (I2, §5-P7).

    Attributes:
        task: the extraction task's name.
        prompt_version_hash: content address of the prompt that produced the
            answer (:attr:`~backend.extraction.prompts.versioning.PromptVersion.version_hash`).
            §5-P7 requires an extraction to be stored with this, and a delta
            needs it more than a single-document task does: the *comparison
            instructions* live in the prompt, so two prompt versions can compare
            the same two documents and mean different things.
        model: qualified model identifier
            (:func:`~backend.extraction.tasks.pipeline.qualified_model`).
        payload_digest: digest of the anonymized text actually sent. This is
            what attests the anonymizer configuration: the payload is a
            deterministic function of both documents' text, the declared
            entities and the masking rules, so two runs agreeing here read
            byte-identical text.
        anchor: the instant the baseline query was answered at, UTC. Equal to
            ``current_knowledge_time`` by construction; recorded separately
            because "which document was the baseline" is only reproducible once
            the instant that chose it is written down.
        prior_document_id: the baseline's accession number.
        current_document_id: the current document's accession number.
        prior_knowledge_time: when the baseline became knowable, UTC.
        current_knowledge_time: when the current document became knowable, UTC.
        gap: ``current.acceptance - prior.acceptance``, wall clock. Reported so a
            consumer can see a pair that spans an implausible interval; never
            used here as a filter.
    """

    task: str
    prompt_version_hash: str
    model: str
    payload_digest: str
    anchor: dt.datetime
    prior_document_id: str
    current_document_id: str
    prior_knowledge_time: dt.datetime
    current_knowledge_time: dt.datetime
    gap: dt.timedelta

    def __post_init__(self) -> None:
        """Enforce the ordering of the two knowledge times and the anchor's identity.

        Raises:
            ValueError: if any required field is blank; if the two documents are
                the same; if ``prior_knowledge_time`` is later than
                ``current_knowledge_time`` (the pair was composed backwards, so
                every signed score would carry the wrong sign); if ``anchor`` is
                not exactly ``current_knowledge_time``; or if ``gap`` is not
                positive.
        """
        for label, value in (
            ("task", self.task),
            ("prompt_version_hash", self.prompt_version_hash),
            ("model", self.model),
            ("payload_digest", self.payload_digest),
            ("prior_document_id", self.prior_document_id),
            ("current_document_id", self.current_document_id),
        ):
            if not value.strip():
                msg = f"{label} must be non-empty on a delta stamp (I2)"
                raise ValueError(msg)
        if self.prior_document_id == self.current_document_id:
            msg = (
                f"delta stamp names {self.current_document_id!r} as both halves of the "
                "pair; a document compared with itself is zero by construction (I3)"
            )
            raise ValueError(msg)
        if self.prior_knowledge_time > self.current_knowledge_time:
            msg = (
                f"prior document {self.prior_document_id!r} became knowable at "
                f"{self.prior_knowledge_time.isoformat()}, after the current document "
                f"{self.current_document_id!r} at "
                f"{self.current_knowledge_time.isoformat()}. The pair is the wrong way "
                "round, and every signed score would carry the opposite sign (I1)"
            )
            raise ValueError(msg)
        if self.anchor != self.current_knowledge_time:
            msg = (
                f"delta stamp anchors at {self.anchor.isoformat()} while the current "
                f"document became knowable at {self.current_knowledge_time.isoformat()}. "
                "The baseline query must be answered at the current document's own "
                "knowledge time and at no other instant (I1)"
            )
            raise ValueError(msg)
        if self.gap <= dt.timedelta(0):
            msg = (
                f"gap between {self.prior_document_id!r} and {self.current_document_id!r} "
                f"is {self.gap}; a baseline is accepted strictly before the document it "
                "is compared against"
            )
            raise ValueError(msg)

    @property
    def feature_knowledge_time(self) -> dt.datetime:
        """When this delta became knowable: the **later** of the two, UTC.

        Derived as a maximum rather than returned from
        :attr:`current_knowledge_time` directly, so it is structurally incapable
        of yielding the earlier instant even if the ordering check above were
        weakened. Both mechanisms are deliberate; see the module docstring for
        what a period of foresight applied uniformly would do to a backtest.
        """
        return max(self.prior_knowledge_time, self.current_knowledge_time)


@dataclass(frozen=True, slots=True)
class DeltaMeasured:
    """A comparison happened and the answer satisfied the task's schema.

    Attributes:
        stamp: how to regenerate this result (I2).
        output: the validated, frozen output. All-zero shift fields mean **no
            change** — a measurement, and the modal one on consecutive filings.
        raw_response: the provider's response verbatim (§5-P7).
        cache_hit: whether the response was served from the cache rather than a
            call. A spend calculation filters on this.
    """

    stamp: DeltaStamp
    output: ExtractionOutput
    raw_response: str
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class DeltaRejected:
    """A comparison happened and the answer did not satisfy the task's schema.

    Not an exception: a malformed response is data about the prompt, and Gate G7
    has to be able to count them. It is also not a measurement — there is no
    output field to read, so nothing downstream can treat a rejection as a
    number.

    Attributes:
        stamp: how to regenerate this attempt (I2).
        raw_response: the provider's response verbatim, kept precisely because
            it was rejected — re-asking at temperature 0 reproduces it exactly
            and costs money to rediscover.
        validation_errors: one line per schema problem, never empty.
        cache_hit: whether the response was served from the cache.
    """

    stamp: DeltaStamp
    raw_response: str
    validation_errors: tuple[str, ...]
    cache_hit: bool

    def __post_init__(self) -> None:
        """Require at least one error.

        Raises:
            ValueError: if ``validation_errors`` is empty. A rejection with no
                stated reason is indistinguishable from a result that was
                dropped, and it would make the schema-failure count meaningless.
        """
        if not self.validation_errors:
            msg = (
                "a rejected response must carry at least one validation error; a rejection "
                "with no reason cannot be told apart from a lost result"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class NoComparisonPossible:
    """No comparison was made, and this is **not** a delta of zero.

    Deliberately carries no number of any kind — no score, no count, no
    ``__float__``, no ``__index__``. There is nothing here to average, rank or
    plot, which is the property that keeps a missing baseline out of a
    cross-sectional feature (see the module docstring, and D-031's
    ``UnmeasurableFeature`` for the same shape applied to drift reporting).

    Attributes:
        task: the extraction task that would have run.
        reason: which of the two things happened —
            :attr:`NoComparisonReason.NO_PRIOR_DOCUMENT` (a fact about the
            issuer) or :attr:`NoComparisonReason.PRIOR_TEXT_UNAVAILABLE` (a fact
            about the platform).
        anchor: the instant the baseline query was answered at. Recorded because
            "there was no previous filing" is only meaningful with the instant
            attached: at a later anchor the answer may differ, and that is not a
            contradiction.
        current_document_id: the document the delta was being extracted for.
        detail: one human-readable line for the operator's document inspector.
    """

    task: str
    reason: NoComparisonReason
    anchor: KnowledgeAnchor
    current_document_id: str
    detail: str

    def __post_init__(self) -> None:
        """Require the identifying fields.

        Raises:
            ValueError: if the task name, document id or detail is blank.
        """
        for label, value in (
            ("task", self.task),
            ("current_document_id", self.current_document_id),
            ("detail", self.detail),
        ):
            if not value.strip():
                msg = f"{label} must be non-empty on a no-comparison record"
                raise ValueError(msg)


DeltaOutcome = DeltaMeasured | DeltaRejected | NoComparisonPossible
"""Everything a delta run can produce.

Three members, and a consumer that pattern-matches on this union has to name
:class:`NoComparisonPossible` explicitly. That is the point: there is no default
branch in which it quietly becomes a number.
"""
