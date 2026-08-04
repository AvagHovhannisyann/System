"""P7.4: the temporal contract, on constructed document pairs.

Everything here is pure — no store, no clock, no model. The instants are
constructed, which is what a test about ordering in time has to do; the rules
they exercise are the ones that decide which document a delta is compared
against, and getting one of them wrong is a lookahead that improves results and
leaves no trace (I1).
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.extraction.entities import EntityKind, company, ticker
from backend.extraction.tasks.delta.anchor import (
    ANNUAL_REPORT,
    EARNINGS_CALL,
    PERIODIC_REPORT,
    BaselineIdentityError,
    BaselineSource,
    BaselineTemporalIntegrityError,
    CurrentDocument,
    DocumentClass,
    DocumentClassMismatchError,
    FilingCandidate,
    KnowledgeAnchor,
    select_baseline,
)
from backend.tests.extraction.delta.doubles import utc

# Constructed instants. A filer's 2024 annual report, the 2023 one before it,
# and a header correction to the 2023 one published four months *after* the 2024
# report was filed — the restatement shape this package exists to keep out.
PRIOR_ACCEPTED = utc(2023, 2, 15, 21, 4)
CURRENT_ACCEPTED = utc(2024, 2, 20, 22, 11)
RESTATED_KNOWN = utc(2024, 6, 1, 13, 30)

CIK = 1725255
"""The AdaptHealth CIK, taken from the captured fixture header (P7.2)."""


def _current(
    *,
    form_type: str = "10-K",
    accepted: dt.datetime = CURRENT_ACCEPTED,
    known: dt.datetime | None = None,
    accession: str = "0001725255-24-000010",
) -> CurrentDocument:
    """Build the later half of a pair."""
    return CurrentDocument(
        accession_number=accession,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type=form_type,
        acceptance=accepted,
        knowledge_time=known if known is not None else accepted,
        text="Item 1A. Risk Factors. Our results depend on reimbursement rates.",
    )


def _candidate(
    *,
    accession: str = "0001725255-23-000004",
    cik: int = CIK,
    form_type: str = "10-K",
    accepted: dt.datetime = PRIOR_ACCEPTED,
    known: dt.datetime | None = None,
    company_name: str = "AdaptHealth Corp.",
) -> FilingCandidate:
    """Build a row the baseline query might have returned."""
    return FilingCandidate(
        accession_number=accession,
        cik=cik,
        company_name=company_name,
        form_type=form_type,
        acceptance=accepted,
        knowledge_time=known if known is not None else accepted,
    )


# ---------------------------------------------------------------------------
# The anchor is the current document's knowledge time, and only that
# ---------------------------------------------------------------------------


def test_the_anchor_is_the_current_documents_knowledge_time() -> None:
    """Not its acceptance instant, and not the clock."""
    current = _current(accepted=CURRENT_ACCEPTED, known=CURRENT_ACCEPTED + dt.timedelta(days=3))
    anchor = KnowledgeAnchor.of(current)
    assert anchor.instant == current.knowledge_time
    assert anchor.instant != current.acceptance
    assert anchor.anchored_to == current.accession_number


def test_an_anchor_must_name_the_document_it_came_from() -> None:
    """A nameless anchor is an arbitrary instant with a reassuring type."""
    with pytest.raises(ValueError, match="must name the document"):
        KnowledgeAnchor(instant=CURRENT_ACCEPTED, anchored_to="   ")


@pytest.mark.parametrize(
    "instant",
    [
        dt.datetime(2024, 2, 20, 22, 11),  # noqa: DTZ001 — naive, which is the point
        dt.datetime(2024, 2, 20, 22, 11, tzinfo=dt.timezone(dt.timedelta(hours=-5))),
    ],
)
def test_a_non_utc_anchor_is_refused(instant: dt.datetime) -> None:
    """A guessed zone moves a document across the anchor by hours."""
    with pytest.raises(ValueError, match="anchor instant must be"):
        KnowledgeAnchor(instant=instant, anchored_to="0001725255-24-000010")


def test_a_document_knowable_before_it_was_accepted_cannot_anchor_anything() -> None:
    """The B5 shape: knowledge earlier than existence would admit unreadable filings."""
    with pytest.raises(ValueError, match="before EDGAR accepted it"):
        _current(accepted=CURRENT_ACCEPTED, known=CURRENT_ACCEPTED - dt.timedelta(hours=1))


def test_select_baseline_refuses_an_anchor_that_is_not_the_documents_own() -> None:
    """Answering the query at one instant and stamping the result with another."""
    current = _current()
    foreign = KnowledgeAnchor(instant=RESTATED_KNOWN, anchored_to=current.accession_number)
    with pytest.raises(ValueError, match="anchor is the current document's knowledge time"):
        select_baseline(
            [_candidate()], current=current, anchor=foreign, document_class=ANNUAL_REPORT
        )


# ---------------------------------------------------------------------------
# The restatement barriers
# ---------------------------------------------------------------------------


def test_a_candidate_knowable_after_the_anchor_is_refused_not_skipped() -> None:
    """The read path should have hidden it; a silent skip would hide the defect.

    This is the later-published restatement arriving through a broken as-of
    layer. Refusing is what turns a wrong number into a stopped run.
    """
    current = _current()
    restated = _candidate(known=RESTATED_KNOWN)
    with pytest.raises(BaselineTemporalIntegrityError) as caught:
        select_baseline(
            [restated],
            current=current,
            anchor=current.anchor,
            document_class=ANNUAL_REPORT,
        )
    assert caught.value.accession_number == restated.accession_number
    assert caught.value.anchor == current.knowledge_time
    assert caught.value.knowledge_time == RESTATED_KNOWN


def test_the_temporal_check_runs_before_the_identity_checks() -> None:
    """A row that fails both must be reported as the read-path defect it is."""
    current = _current()
    both_wrong = _candidate(cik=CIK + 1, known=RESTATED_KNOWN)
    with pytest.raises(BaselineTemporalIntegrityError):
        select_baseline(
            [both_wrong], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_a_filing_accepted_after_the_current_one_cannot_be_its_baseline() -> None:
    """The event-time barrier, independent of the knowledge-time one."""
    current = _current()
    later = _candidate(
        accession="0001725255-24-000090",
        accepted=CURRENT_ACCEPTED + dt.timedelta(days=1),
        known=CURRENT_ACCEPTED - dt.timedelta(days=1),
    )
    with pytest.raises(BaselineIdentityError, match="at or after the current document"):
        select_baseline(
            [later], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_an_amendment_is_not_the_same_form_and_so_is_not_a_baseline() -> None:
    """``10-K/A`` restates a 10-K; exact-form matching is what keeps it out."""
    current = _current()
    amendment = _candidate(accession="0001725255-23-000044", form_type="10-K/A")
    with pytest.raises(BaselineIdentityError, match="baselines match by exact form"):
        select_baseline(
            [amendment], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_no_document_class_admits_an_amendment_form() -> None:
    """The second barrier, checked over every shipped class."""
    for document_class in (ANNUAL_REPORT, PERIODIC_REPORT, EARNINGS_CALL):
        assert not [form for form in document_class.form_types if form.endswith("/A")]


def test_a_document_class_that_admits_an_amendment_cannot_be_constructed() -> None:
    """So a later edit cannot widen the set by accident."""
    with pytest.raises(ValueError, match="admits amendment form"):
        DocumentClass(
            name="lax",
            source=BaselineSource.EDGAR_FILING,
            form_types=frozenset({"10-K", "10-K/A"}),
        )


def test_a_different_filers_history_is_refused() -> None:
    """A delta across filers measures the difference between two companies."""
    current = _current()
    other = _candidate(cik=CIK + 1)
    with pytest.raises(BaselineIdentityError, match="filed under CIK"):
        select_baseline(
            [other], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_a_document_cannot_be_its_own_baseline() -> None:
    """Zero by construction, and recording it would be a fabricated observation."""
    current = _current()
    itself = _candidate(
        accession=current.accession_number,
        accepted=CURRENT_ACCEPTED - dt.timedelta(seconds=1),
    )
    with pytest.raises(BaselineIdentityError, match="is the current document"):
        select_baseline(
            [itself], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_a_quarterly_report_is_not_a_baseline_for_an_annual_one() -> None:
    """Item 1A in a 10-Q updates the annual set; comparing them fakes a purge."""
    current = _current()
    quarterly = _candidate(accession="0001725255-23-000031", form_type="10-Q")
    with pytest.raises(BaselineIdentityError, match="baselines match by exact form"):
        select_baseline(
            [quarterly], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_an_eight_k_is_not_read_by_the_annual_report_class() -> None:
    """A run-configuration error, raised rather than recorded as a data condition."""
    current = _current(form_type="8-K")
    with pytest.raises(DocumentClassMismatchError, match="not read by document class"):
        select_baseline([], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT)


# ---------------------------------------------------------------------------
# Choosing among eligible candidates
# ---------------------------------------------------------------------------


def test_the_newest_eligible_filing_wins() -> None:
    """Newest by acceptance instant, among rows that survived every check."""
    current = _current()
    older = _candidate(accession="0001725255-22-000002", accepted=utc(2022, 2, 16, 20, 0))
    newer = _candidate(accession="0001725255-23-000004", accepted=PRIOR_ACCEPTED)
    chosen = select_baseline(
        [older, newer], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
    )
    assert chosen is not None
    assert chosen.accession_number == newer.accession_number
    assert chosen.chosen_at == current.anchor
    assert chosen.gap_to(current) == CURRENT_ACCEPTED - PRIOR_ACCEPTED


def test_a_tie_on_acceptance_instant_is_refused_rather_than_broken_by_order() -> None:
    """Picking one would choose a baseline by result order."""
    current = _current()
    first = _candidate(accession="0001725255-23-000004")
    second = _candidate(accession="0001725255-23-000005")
    with pytest.raises(BaselineIdentityError, match="share the acceptance instant"):
        select_baseline(
            [first, second], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
        )


def test_no_candidates_yields_none_and_nothing_numeric() -> None:
    """The one legitimate empty answer: nothing eligible was knowable."""
    current = _current()
    assert (
        select_baseline([], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT)
        is None
    )


def test_the_baseline_carries_the_name_the_store_held_at_the_anchor() -> None:
    """Point-in-time identity: the filer's name then, not the one it uses now."""
    current = _current()
    chosen = select_baseline(
        [_candidate(company_name="DFB Healthcare Acquisitions Corp.")],
        current=current,
        anchor=current.anchor,
        document_class=ANNUAL_REPORT,
    )
    assert chosen is not None
    assert chosen.company_name == "DFB Healthcare Acquisitions Corp."


# ---------------------------------------------------------------------------
# The anonymization floor derived from a filing row
# ---------------------------------------------------------------------------


def test_the_baseline_declares_its_filer_and_cik_without_being_told() -> None:
    """A text source that declares nothing still gets the filer masked."""
    current = _current()
    chosen = select_baseline(
        [_candidate()], current=current, anchor=current.anchor, document_class=ANNUAL_REPORT
    )
    assert chosen is not None
    kinds = {entity.kind for entity in chosen.entities}
    assert kinds == {EntityKind.COMPANY, EntityKind.IDENTIFIER}
    names = {entity.name for entity in chosen.entities}
    assert names == {"AdaptHealth Corp.", "0001725255"}


def test_the_current_documents_floor_is_never_dropped_for_caller_declarations() -> None:
    """Caller entities are added to the floor, never substituted for it."""
    current = CurrentDocument(
        accession_number="0001725255-24-000010",
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=CURRENT_ACCEPTED,
        text="Item 1A. Risk Factors.",
        entities=(ticker("AHCO"), company("DFB Healthcare Acquisitions Corp.")),
    )
    names = {entity.name for entity in current.declared_entities}
    assert {"AdaptHealth Corp.", "0001725255"} <= names
    assert {"AHCO", "DFB Healthcare Acquisitions Corp."} <= names


def test_declared_entities_are_de_duplicated() -> None:
    """Declaring the filer again must not double the rule set."""
    current = CurrentDocument(
        accession_number="0001725255-24-000010",
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=CURRENT_ACCEPTED,
        text="Item 1A. Risk Factors.",
        entities=(company("AdaptHealth Corp."),),
    )
    assert len(current.declared_entities) == 2


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_document_that_cannot_name_itself_is_refused(blank: str) -> None:
    """An extraction that cannot say which document it read is not reproducible."""
    with pytest.raises(ValueError, match="must be non-empty"):
        _current(accession=blank)


def test_an_empty_document_is_refused() -> None:
    """An empty half would make the comparison a comparison with nothing."""
    with pytest.raises(ValueError, match="has no text"):
        CurrentDocument(
            accession_number="0001725255-24-000010",
            cik=CIK,
            company_name="AdaptHealth Corp.",
            form_type="10-K",
            acceptance=CURRENT_ACCEPTED,
            knowledge_time=CURRENT_ACCEPTED,
            text="   \n ",
        )


def test_a_candidate_with_a_naive_instant_is_refused() -> None:
    """A naive instant cannot be compared with an anchor without guessing a zone."""
    with pytest.raises(ValueError, match="candidate acceptance must be"):
        FilingCandidate(
            accession_number="0001725255-23-000004",
            cik=CIK,
            company_name="AdaptHealth Corp.",
            form_type="10-K",
            acceptance=dt.datetime(2023, 2, 15, 21, 4),  # noqa: DTZ001 — naive, the point
            knowledge_time=PRIOR_ACCEPTED,
        )


def test_the_earnings_call_class_declares_no_form_types() -> None:
    """A transcript is not an EDGAR submission; giving it a form would be invention."""
    assert EARNINGS_CALL.form_types == frozenset()
    assert EARNINGS_CALL.source is BaselineSource.EARNINGS_CALL_TRANSCRIPT
    assert not EARNINGS_CALL.admits("8-K")


def test_an_edgar_class_admitting_nothing_cannot_be_constructed() -> None:
    """It would make every document a mismatch and no baseline reachable."""
    with pytest.raises(ValueError, match="admits no form type"):
        DocumentClass(name="void", source=BaselineSource.EDGAR_FILING, form_types=frozenset())
