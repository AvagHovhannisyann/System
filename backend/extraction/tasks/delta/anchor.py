"""Temporal anchoring for a two-document delta (P7.4, I1).

Every task in this package compares **two documents from different points in
time**, which makes I1 easier to violate here than anywhere else in the
extraction pipeline. A single-document task reads what it was handed; a delta
task has to *go and find* the other half, and that search is a query — so it is
a query that must be answered at a knowledge instant, not at the wall clock and
not by filing order.

The three claims this module makes, and where each is enforced
--------------------------------------------------------------

**1. The baseline is whatever was knowable at the current document's knowledge
time — nothing later.** The anchor is not a parameter anybody chooses: it is
:attr:`CurrentDocument.knowledge_time`, and :meth:`KnowledgeAnchor.of` is the
only way one is built inside this package. :mod:`backend.extraction.tasks.delta.resolve`
opens the store at exactly that instant and at no other, and
:class:`~backend.extraction.tasks.delta.runner.DeltaExtractor` exposes no
parameter that could carry a different one.

*Why this is the whole game.* "The previous 10-K" reads like a fact about a
filer. It is not; it is the answer to a question, and the answer changes with
when you ask. A restatement filed eighteen months later is, today, the most
recent version of that history — and a backtest that silently adopts it as the
baseline is comparing a filing against a document nobody could read at the time.
That is a lookahead which improves results and leaves no trace.

**2. A row that should not have been visible is refused, never used.**
:func:`select_baseline` re-derives every eligibility condition from the rows it
was handed — including ``knowledge_time <= anchor``, which the as-of layer has
already enforced. Duplicated on purpose, in the same spirit as
:class:`backend.features.factors._prices.PriceTemporalIntegrityError`: if a row
arrives that the layer should have hidden, that is a defect in the read path,
and the honest response is to stop rather than to filter it out quietly. Every
rejection raises. The *only* empty answer this module produces is "the store
held no candidate at all", which is a fact about the issuer rather than about
the plumbing.

**3. A delta has two knowledge times and the feature's is the later one.**
Stated on :class:`~backend.extraction.tasks.delta.outcome.DeltaStamp` and
enforced there — a delta computed from a prior filing is not knowable when the
prior filing was; it is knowable when the *current* one became knowable.

Matching is by exact form type, and amendments are not eligible
---------------------------------------------------------------

A baseline must carry the **same** ``form_type`` as the current document, not
merely a related one. Three problems die at once:

* a 10-K's risk-factor set is not comparable with a 10-Q's update to it;
* a filer that moved between reporting regimes (10-K to 20-F) would otherwise
  produce a delta that measures the regime change;
* **an amendment can never become a baseline.** ``10-K/A`` is not ``10-K``, so
  the restated document is excluded by the same predicate that excludes a 10-Q,
  without anyone having to remember that amendments are special. It is excluded
  again by :data:`DocumentClass.form_types`, which contains no amendment form,
  and a third time by the event-time rule below.

Units: every instant in this module is a timezone-aware UTC ``datetime``.
``cik`` is a dimensionless EDGAR Central Index Key. There are no durations here
except :attr:`PriorDocumentRef.gap_to`, which is wall-clock.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from backend.extraction.entities import company, identifier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.extraction.entities import Entity

__all__ = [
    "ANNUAL_REPORT",
    "EARNINGS_CALL",
    "PERIODIC_REPORT",
    "BaselineIdentityError",
    "BaselineSource",
    "BaselineSourceUnavailableError",
    "BaselineTemporalIntegrityError",
    "CurrentDocument",
    "DeltaAnchorError",
    "DocumentClass",
    "DocumentClassMismatchError",
    "FilingCandidate",
    "KnowledgeAnchor",
    "PriorDocumentRef",
    "select_baseline",
]


class DeltaAnchorError(RuntimeError):
    """Base class for every refusal this module makes. Never raised directly."""


class DocumentClassMismatchError(DeltaAnchorError):
    """The current document's form is not one this delta task reads.

    A caller error, raised rather than answered with "no comparison possible":
    running the risk-factor task over an 8-K is a mistake in the run
    configuration, not a fact about the issuer, and recording it as a data
    condition would put it in the same bucket as a genuinely absent baseline.
    """


class BaselineTemporalIntegrityError(DeltaAnchorError):
    """A candidate baseline was not knowable at the anchor instant (I1).

    A lookahead detector, not a filter. The as-of layer answers the baseline
    query at :attr:`KnowledgeAnchor.instant`, so no row it returns can carry a
    later ``knowledge_time``; one that does means the read path failed, and the
    document that would have been chosen is precisely the later-published
    restatement this package exists to keep out. Dropping it silently would
    leave a working system that reads the store at the wrong instant and never
    says so.

    Attributes:
        accession_number: the offending candidate.
        knowledge_time: when that candidate became knowable (UTC).
        anchor: the instant the query was supposed to be answered at (UTC).
    """

    def __init__(
        self, *, accession_number: str, knowledge_time: dt.datetime, anchor: dt.datetime
    ) -> None:
        """Build the error from the offending candidate and the anchor.

        Args:
            accession_number: the candidate that should not have been visible.
            knowledge_time: that candidate's knowledge time, UTC.
            anchor: the anchor instant the query was answered at, UTC.
        """
        self.accession_number = accession_number
        self.knowledge_time = knowledge_time
        self.anchor = anchor
        super().__init__(
            f"filing {accession_number!r} became knowable at "
            f"{knowledge_time.isoformat()}, after the anchor "
            f"{anchor.isoformat()} this baseline query is answered at. It cannot "
            f"be the baseline for a document that predates it, and it must not "
            f"have been visible at all: the as-of layer bounds every read by the "
            f"anchor (D-011). Refusing to choose a baseline from a knowledge set "
            f"that is not the anchor's (I1)."
        )


class BaselineIdentityError(DeltaAnchorError):
    """A candidate baseline is not a comparable earlier document of the same kind.

    Covers a different filer, a different form type, the current document
    itself, a document accepted at or after the current one, and two candidates
    tied on acceptance instant. Every one of them is refused rather than
    skipped: the baseline query already restricts to same filer, same form and
    strictly earlier acceptance, so a candidate failing one of these conditions
    means the query and this check disagree about what was asked for.
    """


class BaselineSourceUnavailableError(DeltaAnchorError):
    """The store this task's baseline would come from does not exist yet.

    Distinct from "no prior document" and deliberately not a
    :class:`~backend.extraction.tasks.delta.outcome.NoComparisonPossible`
    value. An absent baseline is an observation about an issuer; an unbuilt
    connector is a blocker, and turning a blocker into a data value is how a
    missing data source stops being visible. Earnings-call transcripts are the
    live case: P3.6 is blocked on B1, so no transcript store exists and nothing
    here invents one (I3, §9.4).
    """


class BaselineSource(StrEnum):
    """Which store a delta task's baseline is looked up in."""

    EDGAR_FILING = "EDGAR_FILING"
    """``edgar_filing`` — SEC submissions keyed by accession and CIK (P3.2)."""

    EARNINGS_CALL_TRANSCRIPT = "EARNINGS_CALL_TRANSCRIPT"
    """Earnings-call transcripts. **No such store exists** (P3.6, blocked on B1).

    Present because the task that needs it is defined and its prompt, schema and
    comparison rules are real; absent from the resolver, which raises
    :class:`BaselineSourceUnavailableError` rather than reading a table that
    would have to be invented to exist.
    """


@dataclass(frozen=True, slots=True)
class DocumentClass:
    """A family of documents that may be compared with one another.

    Attributes:
        name: identity, for messages and for the operator's task list (§6.5).
        source: which store the baseline is looked up in.
        form_types: the form types this class admits. **Contains no amendment
            form**: an amendment restates an earlier document, so admitting one
            would make a restatement eligible as a baseline through the front
            door. Matching is by exact form type anyway (see the module
            docstring), so this set is the second of three barriers.
    """

    name: str
    source: BaselineSource
    form_types: frozenset[str]

    def __post_init__(self) -> None:
        """Reject a class that admits nothing, or that admits an amendment.

        Raises:
            ValueError: if ``form_types`` is empty for an EDGAR-sourced class,
                or if any member names an amendment (``/A``). A class admitting
                nothing would make every document a mismatch; a class admitting
                an amendment would defeat the barrier this field exists to be.
        """
        if not self.name.strip():
            msg = "a document class must be named"
            raise ValueError(msg)
        if self.source is BaselineSource.EDGAR_FILING and not self.form_types:
            msg = f"document class {self.name!r} admits no form type, so no document is eligible"
            raise ValueError(msg)
        amendments = sorted(form for form in self.form_types if form.endswith("/A"))
        if amendments:
            msg = (
                f"document class {self.name!r} admits amendment form(s) {amendments}: an "
                "amendment restates an earlier document, and admitting one would let a "
                "later-published restatement become a baseline (I1)"
            )
            raise ValueError(msg)

    def admits(self, form_type: str) -> bool:
        """Whether a document of this form type may be read by this class.

        Args:
            form_type: EDGAR form type, verbatim from the submission header.

        Returns:
            ``True`` when the form is admitted.
        """
        return form_type in self.form_types


ANNUAL_REPORT: Final = DocumentClass(
    name="annual_report",
    source=BaselineSource.EDGAR_FILING,
    form_types=frozenset({"10-K", "20-F", "40-F"}),
)
"""Annual reports: the documents that carry a complete risk-factor section.

10-K for a domestic registrant, 20-F and 40-F for foreign private issuers.
Quarterly reports are excluded because Item 1A in a 10-Q is an *update* to the
annual set, so comparing one against a 10-K would report every unchanged risk as
removed.
"""

PERIODIC_REPORT: Final = DocumentClass(
    name="periodic_report",
    source=BaselineSource.EDGAR_FILING,
    form_types=frozenset({"10-K", "10-Q", "20-F", "40-F"}),
)
"""Periodic reports: where MD&A, outlook language and accounting policy live.

Wider than :data:`ANNUAL_REPORT` because these constructs are restated every
quarter. Exact-form matching still applies, so a 10-Q's baseline is the previous
10-Q and never the annual report — the two differ in scope and in audit status,
and a delta across them would measure that difference.
"""

EARNINGS_CALL: Final = DocumentClass(
    name="earnings_call",
    source=BaselineSource.EARNINGS_CALL_TRANSCRIPT,
    form_types=frozenset(),
)
"""Earnings-call transcripts — analyst Q&A.

Empty ``form_types``: a transcript is not an EDGAR submission and carries no
form type. **Nothing resolves this class today** (P3.6 is blocked on B1); the
resolver raises rather than guessing at a schema for a table that does not
exist.
"""


def _validated_instant(value: dt.datetime, *, label: str) -> dt.datetime:
    """Return ``value`` if it is a timezone-aware UTC instant, else raise.

    Args:
        value: the instant to check.
        label: what it is, for the message.

    Returns:
        The instant, unchanged.

    Raises:
        ValueError: if it is naive or carries a non-zero UTC offset. A naive
            instant cannot be compared with a knowledge time without guessing a
            zone, and a guess here moves a document across the anchor.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        msg = f"{label} must be timezone-aware UTC; got naive {value!r}"
        raise ValueError(msg)
    if value.utcoffset() != dt.timedelta(0):
        msg = f"{label} must be UTC (offset 0); got offset {value.utcoffset()} in {value!r}"
        raise ValueError(msg)
    return value


def _filer_entities(company_name: str, cik: int) -> tuple[Entity, ...]:
    """Return the entity declarations derivable from a filing row alone.

    This is the anonymization **floor** for a document read out of
    ``edgar_filing``: the filer name as the index spells it, and the CIK in both
    the zero-padded and unpadded writings
    (:func:`backend.extraction.entities.identifier` generates the pair).

    It is a floor and not a complete declaration, and the gap is stated rather
    than hidden: ``edgar_filing`` does not store former conformed names or
    tickers, so a document that refers to the filer by a former name or a symbol
    is masked only if a caller supplies those separately
    (:class:`~backend.extraction.tasks.delta.runner.PriorDocumentText` carries
    additional declarations, and they are unioned with these, never substituted
    for them).

    Args:
        company_name: filer name exactly as ``edgar_filing`` stores it.
        cik: EDGAR Central Index Key (dimensionless).

    Returns:
        The company and identifier declarations, in a stable order.
    """
    return (company(company_name), identifier(f"{cik:010d}"))


@dataclass(frozen=True, slots=True)
class KnowledgeAnchor:
    """The instant a baseline query is answered at, and what fixed it.

    Attributes:
        instant: the as-of timestamp, timezone-aware UTC.
        anchored_to: the accession number of the document whose knowledge time
            this is. Carried so a stored delta can state *why* it was answered
            at this instant — an anchor with no provenance is indistinguishable
            from a timestamp somebody chose.
    """

    instant: dt.datetime
    anchored_to: str

    def __post_init__(self) -> None:
        """Validate the instant and require a document to be anchored to.

        Raises:
            ValueError: if the instant is naive or non-UTC, or if
                ``anchored_to`` is blank. An anchor that cannot name its
                document is a free parameter wearing an anchor's type.
        """
        _validated_instant(self.instant, label="anchor instant")
        if not self.anchored_to.strip():
            msg = (
                "an anchor must name the document whose knowledge time it is; a nameless "
                "anchor is an arbitrary instant with a reassuring type (I1)"
            )
            raise ValueError(msg)

    @classmethod
    def of(cls, current: CurrentDocument) -> KnowledgeAnchor:
        """Return the anchor for a delta on ``current``: its own knowledge time.

        The only constructor used anywhere in this package, and the reason the
        anchor cannot drift: there is no arithmetic here, no rounding, no
        clock read and no caller-supplied alternative.

        Args:
            current: the document the delta is being extracted for.

        Returns:
            The anchor.
        """
        return cls(instant=current.knowledge_time, anchored_to=current.accession_number)


@dataclass(frozen=True, slots=True)
class CurrentDocument:
    """The later half of a delta pair: the document being extracted *for*.

    Attributes:
        accession_number: EDGAR accession in dashed form. Provenance only; never
            sent to a model.
        cik: the filer's Central Index Key (dimensionless).
        company_name: filer name as ``edgar_filing`` stores it. Used to build
            the anonymization floor, never sent to a model.
        form_type: EDGAR form type, verbatim.
        acceptance: when EDGAR accepted the submission — its event time, UTC.
        knowledge_time: when it became knowable, UTC. **This is the anchor.**
            Equal to ``acceptance`` on a filing's first version and later than it
            when a header correction re-versioned the row (D-011); the two are
            separate fields here for exactly that reason.
        text: the document text, before anonymization.
        entities: entity declarations the caller knows beyond the floor derived
            from ``company_name`` and ``cik`` — former names, tickers, officers.
    """

    accession_number: str
    cik: int
    company_name: str
    form_type: str
    acceptance: dt.datetime
    knowledge_time: dt.datetime
    text: str
    entities: tuple[Entity, ...] = ()

    def __post_init__(self) -> None:
        """Validate identity, the two instants and their order.

        Raises:
            ValueError: if the accession, company name or form type is blank, if
                the text is empty, if either instant is naive or non-UTC, or if
                ``knowledge_time`` precedes ``acceptance``. A document knowable
                before it was accepted is the lookahead B5 describes, and it must
                not be able to anchor anything.
        """
        for label, value in (
            ("accession_number", self.accession_number),
            ("company_name", self.company_name),
            ("form_type", self.form_type),
        ):
            if not value.strip():
                msg = f"{label} must be non-empty: a delta must name the document it read (I2)"
                raise ValueError(msg)
        if not self.text.strip():
            msg = (
                f"document {self.accession_number!r} has no text; there is nothing to "
                "compare and an empty half would make the delta a fabricated comparison (I3)"
            )
            raise ValueError(msg)
        _validated_instant(self.acceptance, label="acceptance")
        _validated_instant(self.knowledge_time, label="knowledge_time")
        if self.knowledge_time < self.acceptance:
            msg = (
                f"document {self.accession_number!r} claims to have been knowable at "
                f"{self.knowledge_time.isoformat()}, before EDGAR accepted it at "
                f"{self.acceptance.isoformat()}. Anchoring a baseline query at an instant "
                "earlier than the document existed would admit filings nobody could read (I1)"
            )
            raise ValueError(msg)

    @property
    def anchor(self) -> KnowledgeAnchor:
        """The anchor for this document's delta — its own knowledge time."""
        return KnowledgeAnchor.of(self)

    @property
    def declared_entities(self) -> tuple[Entity, ...]:
        """The floor from the filing row, unioned with whatever the caller added.

        The floor is never dropped, so a caller who declares nothing still gets
        the filer name and CIK masked.
        """
        merged: dict[Entity, None] = {}
        for entity in (*_filer_entities(self.company_name, self.cik), *self.entities):
            merged.setdefault(entity, None)
        return tuple(merged)


@dataclass(frozen=True, slots=True)
class FilingCandidate:
    """One row the baseline query returned, exactly as the store stated it.

    A separate type from :class:`PriorDocumentRef` on purpose: a candidate is
    something the store offered, a reference is something
    :func:`select_baseline` accepted. Collapsing them would remove the place
    where the difference between the two is checked.

    Attributes:
        accession_number: EDGAR accession in dashed form.
        cik: the filer's Central Index Key (dimensionless).
        company_name: filer name as stored.
        form_type: EDGAR form type, verbatim.
        acceptance: the submission's acceptance instant (event time), UTC.
        knowledge_time: when the row became knowable, UTC.
    """

    accession_number: str
    cik: int
    company_name: str
    form_type: str
    acceptance: dt.datetime
    knowledge_time: dt.datetime

    def __post_init__(self) -> None:
        """Validate the two instants.

        Raises:
            ValueError: if either is naive or carries a non-zero UTC offset.
        """
        _validated_instant(self.acceptance, label="candidate acceptance")
        _validated_instant(self.knowledge_time, label="candidate knowledge_time")


@dataclass(frozen=True, slots=True)
class PriorDocumentRef:
    """The earlier half of a delta pair, as chosen at a stated anchor.

    Attributes:
        accession_number: EDGAR accession in dashed form.
        cik: the filer's Central Index Key (dimensionless).
        company_name: filer name as stored at the anchor instant — the
            point-in-time name, not today's.
        form_type: EDGAR form type, equal to the current document's by
            construction.
        acceptance: the submission's acceptance instant, UTC. Strictly earlier
            than the current document's.
        knowledge_time: when this document became knowable, UTC. Never later
            than the anchor.
        chosen_at: the anchor the choice was made at. Carried so a stored delta
            can be re-derived: "the previous 10-K" is only a well-formed
            statement once this instant is attached to it.
    """

    accession_number: str
    cik: int
    company_name: str
    form_type: str
    acceptance: dt.datetime
    knowledge_time: dt.datetime
    chosen_at: KnowledgeAnchor

    @property
    def entities(self) -> tuple[Entity, ...]:
        """The anonymization floor derivable from this row (see :func:`_filer_entities`)."""
        return _filer_entities(self.company_name, self.cik)

    def gap_to(self, current: CurrentDocument) -> dt.timedelta:
        """Return the wall-clock gap between this document and ``current``.

        Reported, never used as a filter. A twelve-year gap between "consecutive"
        annual reports means the ingestion history has a hole, and a consumer
        that wants to exclude such a pair can — but this module will not decide
        that on its behalf, because the threshold would be a number nobody chose.

        Args:
            current: the later document of the pair.

        Returns:
            ``current.acceptance - self.acceptance``, a positive duration.
        """
        return current.acceptance - self.acceptance


def _refuse_ineligible(
    candidate: FilingCandidate, *, current: CurrentDocument, anchor: KnowledgeAnchor
) -> None:
    """Raise unless ``candidate`` could be ``current``'s baseline.

    Ordered so the temporal claim is tested first: a row that was not knowable
    at the anchor is a read-path defect and must be reported as one even if it
    would also have failed an identity check.

    Args:
        candidate: the row to judge.
        current: the document the delta is for.
        anchor: the instant the query was answered at.

    Raises:
        BaselineTemporalIntegrityError: the candidate was not knowable at the
            anchor.
        BaselineIdentityError: the candidate is a different filer, a different
            form, the current document itself, or was accepted at or after it.
    """
    if candidate.knowledge_time > anchor.instant:
        raise BaselineTemporalIntegrityError(
            accession_number=candidate.accession_number,
            knowledge_time=candidate.knowledge_time,
            anchor=anchor.instant,
        )
    if candidate.cik != current.cik:
        msg = (
            f"candidate {candidate.accession_number!r} is filed under CIK {candidate.cik} "
            f"but the current document is CIK {current.cik}; a delta across filers "
            "measures the difference between two companies"
        )
        raise BaselineIdentityError(msg)
    if candidate.form_type != current.form_type:
        msg = (
            f"candidate {candidate.accession_number!r} is form {candidate.form_type!r} but "
            f"the current document is {current.form_type!r}; baselines match by exact form, "
            "which is also what keeps an amendment from becoming one"
        )
        raise BaselineIdentityError(msg)
    if candidate.accession_number == current.accession_number:
        msg = (
            f"candidate {candidate.accession_number!r} is the current document; a document "
            "compared with itself has a delta of zero by construction, and recording that "
            "as a measurement would be a fabricated observation (I3)"
        )
        raise BaselineIdentityError(msg)
    if candidate.acceptance >= current.acceptance:
        msg = (
            f"candidate {candidate.accession_number!r} was accepted at "
            f"{candidate.acceptance.isoformat()}, at or after the current document's "
            f"{current.acceptance.isoformat()}. A document filed later cannot be the "
            "baseline it is compared against — this is the barrier a restatement hits "
            "on the event-time axis (I1)"
        )
        raise BaselineIdentityError(msg)


def select_baseline(
    candidates: Sequence[FilingCandidate],
    *,
    current: CurrentDocument,
    anchor: KnowledgeAnchor,
    document_class: DocumentClass,
) -> PriorDocumentRef | None:
    """Choose the baseline for ``current`` from rows visible at ``anchor``.

    Pure: no clock, no I/O, no store. Everything it decides is decidable from
    its arguments, which is what makes the eligibility rules testable on
    constructed pairs.

    Every rejection **raises**. The rules restate what the baseline query
    already filtered on, so under a correct read path none of them can fire —
    exactly like
    :class:`backend.features.factors._prices.PriceTemporalIntegrityError`, and
    for the same reason: the day one does fire, the read path is wrong and a
    silent skip would replace a loud failure with a plausible number.

    Args:
        candidates: rows returned by the baseline query, in any order. Expected
            to be already restricted to the same filer, the same form and
            strictly earlier acceptance; this function does not trust that.
        current: the document the delta is for.
        anchor: the instant the query was answered at. Must be ``current``'s own
            anchor — a mismatch means the caller answered the query at an
            instant that is not the one the result will be stamped with.
        document_class: which family of documents this task compares.

    Returns:
        The chosen :class:`PriorDocumentRef`, or ``None`` when no candidate
        survived. ``None`` means *no comparison is possible*, which the caller
        must keep distinct from a measured delta of zero
        (:mod:`backend.extraction.tasks.delta.outcome`).

    Raises:
        DocumentClassMismatchError: ``current``'s form is not admitted by
            ``document_class``.
        BaselineTemporalIntegrityError: a candidate was not knowable at the
            anchor.
        BaselineIdentityError: a candidate is not a comparable earlier document,
            or two candidates tie on acceptance instant.
        ValueError: ``anchor`` is not ``current``'s own anchor.
    """
    if anchor != current.anchor:
        msg = (
            f"baseline query for {current.accession_number!r} was answered at "
            f"{anchor.instant.isoformat()} (anchored to {anchor.anchored_to!r}) but the "
            f"document's own knowledge time is {current.knowledge_time.isoformat()}. The "
            "anchor is the current document's knowledge time and nothing else (I1)"
        )
        raise ValueError(msg)
    if not document_class.admits(current.form_type):
        msg = (
            f"form {current.form_type!r} is not read by document class "
            f"{document_class.name!r} (admits {sorted(document_class.form_types)}). "
            "Amendments are deliberately absent: a restatement is not a baseline"
        )
        raise DocumentClassMismatchError(msg)

    for candidate in candidates:
        _refuse_ineligible(candidate, current=current, anchor=anchor)
    if not candidates:
        return None

    latest = max(candidate.acceptance for candidate in candidates)
    tied = [candidate for candidate in candidates if candidate.acceptance == latest]
    if len(tied) > 1:
        msg = (
            f"{len(tied)} candidate baselines for {current.accession_number!r} share the "
            f"acceptance instant {latest.isoformat()}: "
            f"{sorted(candidate.accession_number for candidate in tied)}. Picking one would "
            "choose a baseline by result order, which is not a choice this module is "
            "entitled to make"
        )
        raise BaselineIdentityError(msg)
    chosen = tied[0]
    return PriorDocumentRef(
        accession_number=chosen.accession_number,
        cik=chosen.cik,
        company_name=chosen.company_name,
        form_type=chosen.form_type,
        acceptance=chosen.acceptance,
        knowledge_time=chosen.knowledge_time,
        chosen_at=anchor,
    )
