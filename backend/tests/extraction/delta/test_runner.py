"""P7.4: running a delta end to end, and what leaves the process when it does.

The properties worth more than the rest, in order:

1. **What reached the model was anonymized.** Asserted the only way it can be
   asserted honestly — by injecting a recording client, running real captured
   EDGAR text through the extractor, and inspecting the request the client was
   handed. Asserting it inside the runner would check the code against itself,
   and a masker that ran and did nothing would satisfy that check.
2. **A missing baseline is not a zero.** Two different absences produce two
   different reasons, and neither produces a number.
3. **A malformed response is refused, not coerced.** A quoted number is not a
   number, and a fenced block is not bare JSON.
4. **No model was ever called.** B4 leaves no provider key, so the default
   client raises. Every response below is canned text written in this file.

The document pairings here are **constructed**: two independently captured
filings by different issuers are presented to the extractor as one filer's
history, so that a single run masks two real documents at once. What is under
test is the payload that leaves the process and the outcome that comes back —
not the plausibility of the pair, whose eligibility rules are exercised on
constructed instants in ``test_anchor.py`` and ``test_resolve.py``.
"""

from __future__ import annotations

import datetime as dt
import inspect
import re

import pytest

from backend.extraction.governor.caps import CapBook
from backend.extraction.governor.errors import CapsNotConfiguredError
from backend.extraction.leak import detect_leaks
from backend.extraction.tasks.chunking import ChunkingConfig
from backend.extraction.tasks.client import ProviderNotConfiguredError
from backend.extraction.tasks.delta import resolve as resolve_module
from backend.extraction.tasks.delta import runner as runner_module
from backend.extraction.tasks.delta.anchor import (
    BaselineIdentityError,
    BaselineSourceUnavailableError,
    CurrentDocument,
    DocumentClassMismatchError,
    PriorDocumentRef,
)
from backend.extraction.tasks.delta.outcome import (
    DeltaMeasured,
    DeltaRejected,
    NoComparisonPossible,
    NoComparisonReason,
)
from backend.extraction.tasks.delta.qa_evasiveness import QA_EVASIVENESS_SHIFT_SPEC
from backend.extraction.tasks.delta.risk_factor_language import RISK_FACTOR_LANGUAGE_DELTA_SPEC
from backend.extraction.tasks.delta.runner import (
    DeltaExtractor,
    PriorDocumentText,
    PriorTextSource,
)
from backend.extraction.tasks.library import RiskFactorLanguageDelta
from backend.extraction.tasks.pipeline import ExtractionPipeline
from backend.extraction.temporal import year_exemption_reason
from backend.tests.extraction.delta.doubles import (
    MODEL,
    EmptyTextSource,
    FilingRow,
    RecordingClient,
    RecordingStore,
    StubTextSource,
    UnaskedTextSource,
    utc,
)
from backend.tests.extraction.edgar_text import (
    ADAPTHEALTH_FORM4,
    SAP_6K,
    adapthealth_entities,
    read,
    sap_entities,
)

PRIOR_ACCEPTED = utc(2023, 2, 15, 21, 4)
CURRENT_ACCEPTED = utc(2024, 2, 20, 22, 11)

CIK = 1725255
PRIOR_ACCESSION = "0001104659-24-082105"
CURRENT_ACCESSION = "0001193805-24-000360"

_VALID = (
    '{"severity_shift": 0.25, "hedging_shift": -0.1, "specificity_shift": 0.0, '
    '"evidence": ["[COMPANY_1] now quantifies the exposure"], "confidence": 0.7}'
)
"""A response satisfying ``RiskFactorLanguageDelta``. Written here; not a model's."""

_UNCHANGED = (
    '{"severity_shift": 0.0, "hedging_shift": 0.0, "specificity_shift": 0.0, '
    '"evidence": [], "confidence": 0.9}'
)
"""A response saying nothing moved — the modal honest answer on consecutive filings."""


def _current_document() -> CurrentDocument:
    """The later half: a real captured filing with the entities its header declares."""
    return CurrentDocument(
        accession_number=CURRENT_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=CURRENT_ACCEPTED,
        text=read(ADAPTHEALTH_FORM4),
        entities=tuple(adapthealth_entities()),
    )


def _prior_row(*, company_name: str = "AdaptHealth Corp.") -> FilingRow:
    """The earlier half's ``edgar_filing`` row."""
    return FilingRow(
        accession_number=PRIOR_ACCESSION,
        cik=CIK,
        company_name=company_name,
        form_type="10-K",
        valid_from=PRIOR_ACCEPTED,
        knowledge_time=PRIOR_ACCEPTED,
    )


def _sap_body(*, declare_entities: bool = True) -> PriorDocumentText:
    """The earlier half's body: a second real captured filing."""
    return PriorDocumentText(
        text=read(SAP_6K),
        entities=tuple(sap_entities()) if declare_entities else (),
    )


def _extractor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    client: RecordingClient,
    rows: list[FilingRow] | None = None,
    text_source: PriorTextSource | None = None,
) -> tuple[DeltaExtractor, RecordingStore]:
    """Wire an extractor over a recording store, a recording client and a body source."""
    store = RecordingStore(rows if rows is not None else [_prior_row()])
    monkeypatch.setattr(resolve_module, "as_of", store.as_of)
    pipeline = ExtractionPipeline(client=client, chunking=ChunkingConfig(max_chars=1_000_000))
    source = (
        text_source if text_source is not None else StubTextSource({PRIOR_ACCESSION: _sap_body()})
    )
    return DeltaExtractor(pipeline=pipeline, text_source=source), store


# ---------------------------------------------------------------------------
# 1. What left the process was masked
# ---------------------------------------------------------------------------


async def test_both_halves_of_the_pair_reach_the_model_anonymized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every declared name and identifier from *either* filing is gone from the request.

    Checked against the originals first: an assertion that a name is missing
    proves nothing if the name was never there. ``occurrences_in_original``
    makes that visible, so this test cannot pass vacuously.
    """
    current = _current_document()
    prior_body = _sap_body()
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(monkeypatch, client=client)

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, DeltaMeasured)
    assert client.calls == 1
    sent = client.requests[0].prompt
    declared = [*adapthealth_entities(), *sap_entities()]
    report = detect_leaks(
        original=f"{prior_body.text}\n{current.text}", anonymized=sent, entities=declared
    )
    assert report.leaks == (), report.summary()
    for name in ("AdaptHealth Corp.", "SAP SE", "0001725255", "0001000184"):
        assert report.occurrences_in_original[name] > 0
    # And the payload is masked text, not something that merely omitted them.
    assert "[COMPANY_1]" in sent


async def test_every_date_in_both_halves_is_gone_from_what_was_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§5-P7 strips *all* dates, and a delta pair is where two documents' worth arrive.

    Checked independently of the leak detector as well: every standalone
    four-digit number still in the payload must carry a *stated* exemption
    (:func:`~backend.extraction.temporal.year_exemption_reason`). Anything
    without one would be a calendar year that reached the model, which is half
    of the contamination control the P7.9 probe rests on.

    "Standalone" matches the masker's own boundary semantics: ``19462`` is the
    ZIP code of an address in the AdaptHealth header, not the year 1946 with a
    stray digit after it, and scanning without the digit boundaries would report
    the first four characters of every long number as a surviving year.
    """
    current = _current_document()
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(monkeypatch, client=client)

    await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    sent = client.requests[0].prompt
    unexplained = [
        match.group()
        for match in re.finditer(r"(?<!\d)\d{4}(?!\d)", sent)
        if year_exemption_reason(sent, match.start(), match.end()) is None
    ]
    assert not unexplained, f"unmasked calendar years reached the model: {unexplained}"


async def test_the_baselines_filer_is_masked_even_when_the_text_source_declares_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The floor comes from the **baseline's own** ``edgar_filing`` row.

    The filer is under its former name at the anchor — ``DFB Healthcare
    Acquisitions Corp.`` — and under its current one in the later document. Only
    the baseline row carries the former name, so masking it proves the *prior
    half's* declarations reached the masker rather than the current half's
    covering for them. That is the point-in-time identity D-011 stores and the
    reason a delta reads the name the store held then.

    The test also **measures** the documented gap rather than claiming it away:
    ``edgar_filing`` carries one filer name and one CIK, so the other reporting
    owners named in this joint filing are not masked by the floor. That residue
    is real, it is stated in :mod:`backend.extraction.tasks.delta.anchor`, and
    it is asserted here so a change in it has to be seen rather than discovered.
    """
    current = CurrentDocument(
        accession_number=CURRENT_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="10-K",
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=CURRENT_ACCEPTED,
        text="Item 1A. Risk Factors. Reimbursement rates may decline further.",
        entities=(),
    )
    body = read(ADAPTHEALTH_FORM4)
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(
        monkeypatch,
        client=client,
        rows=[_prior_row(company_name="DFB Healthcare Acquisitions Corp.")],
        text_source=StubTextSource({PRIOR_ACCESSION: PriorDocumentText(text=body)}),
    )

    await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    sent = client.requests[0].prompt
    # Declared only by the baseline row, and masked.
    assert "DFB Healthcare Acquisitions Corp." in body
    assert "DFB Healthcare" not in sent
    # Declared only by the current document's row, and masked.
    assert "AdaptHealth Corp." in body
    assert "AdaptHealth" not in sent
    # Declared by both rows' CIK, and masked in both writings.
    assert "1725255" in body
    assert "1725255" not in sent
    # The stated limit, measured: a co-filer the row does not name survives.
    assert "DEERFIELD PARTNERS, L.P." in body
    assert "DEERFIELD PARTNERS, L.P." in sent


async def test_the_request_carries_neither_document_id_nor_the_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provenance stays on this side of the seam — for both halves of the pair."""
    current = _current_document()
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(monkeypatch, client=client)

    await extractor.extract(
        RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL, correlation_id="corr-7"
    )

    request = client.requests[0]
    whole = f"{request.system}\n{request.prompt}\n{request.model}"
    assert CURRENT_ACCESSION not in whole
    assert PRIOR_ACCESSION not in whole
    assert "corr-7" not in whole


async def test_the_pair_reaches_the_model_in_one_call_with_the_prior_half_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direction is the whole sign convention; one call is what makes it a comparison."""
    current = _current_document()
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(monkeypatch, client=client)

    await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert client.calls == 1
    sent = client.requests[0].prompt
    assert sent.index("=== PRIOR DOCUMENT ===") < sent.index("=== CURRENT DOCUMENT ===")
    assert client.requests[0].temperature == 0.0


# ---------------------------------------------------------------------------
# 2. The stamp
# ---------------------------------------------------------------------------


async def test_a_measured_delta_carries_the_prompt_version_and_both_knowledge_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I2, and the rule that the feature's knowledge time is the later one."""
    current = _current_document()
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(monkeypatch, client=client)

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, DeltaMeasured)
    stamp = outcome.stamp
    assert stamp.task == "risk_factor_language_delta"
    assert stamp.prompt_version_hash == RISK_FACTOR_LANGUAGE_DELTA_SPEC.task.prompt.version_hash
    assert stamp.model == MODEL
    assert stamp.payload_digest
    assert stamp.anchor == current.knowledge_time
    assert stamp.prior_knowledge_time == PRIOR_ACCEPTED
    assert stamp.current_knowledge_time == CURRENT_ACCEPTED
    assert stamp.feature_knowledge_time == CURRENT_ACCEPTED
    assert stamp.gap == CURRENT_ACCEPTED - PRIOR_ACCEPTED
    assert stamp.prior_document_id == PRIOR_ACCESSION
    assert stamp.current_document_id == CURRENT_ACCESSION


async def test_an_all_zero_answer_is_a_measurement_not_an_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged answer comes back measured, distinguishable by type."""
    current = _current_document()
    extractor, _ = _extractor(monkeypatch, client=RecordingClient(_UNCHANGED))

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, DeltaMeasured)
    assert isinstance(outcome.output, RiskFactorLanguageDelta)
    assert outcome.output.severity_shift == 0.0


# ---------------------------------------------------------------------------
# 3. No comparison possible — two reasons, no numbers
# ---------------------------------------------------------------------------


async def test_no_prior_document_yields_a_reason_and_never_a_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first-time registrant. The model is never called."""
    current = _current_document()
    client = RecordingClient(_VALID)
    extractor, store = _extractor(
        monkeypatch, client=client, rows=[], text_source=UnaskedTextSource()
    )

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, NoComparisonPossible)
    assert outcome.reason is NoComparisonReason.NO_PRIOR_DOCUMENT
    assert outcome.anchor.instant == current.knowledge_time
    assert outcome.current_document_id == CURRENT_ACCESSION
    assert client.calls == 0
    assert store.opened_at == [current.knowledge_time]
    numeric = [
        name
        for name in dir(outcome)
        if not name.startswith("_") and isinstance(getattr(outcome, name), int | float | complex)
    ]
    assert numeric == []


async def test_an_unfetchable_baseline_is_a_different_fact_from_an_absent_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One is about the issuer, the other about this platform.

    Merging them would report an ingestion gap as a property of a company.
    """
    current = _current_document()
    client = RecordingClient(_VALID)
    source = EmptyTextSource()
    extractor, _ = _extractor(monkeypatch, client=client, text_source=source)

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, NoComparisonPossible)
    assert outcome.reason is NoComparisonReason.PRIOR_TEXT_UNAVAILABLE
    # The baseline really was resolved — the source was asked for it by name.
    assert source.asked == [PRIOR_ACCESSION]
    assert client.calls == 0


async def test_the_two_absences_are_reported_differently_for_the_same_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction, shown side by side rather than asserted twice apart."""
    current = _current_document()
    absent, _ = _extractor(
        monkeypatch, client=RecordingClient(_VALID), rows=[], text_source=UnaskedTextSource()
    )
    first = await absent.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    unfetched, _ = _extractor(
        monkeypatch, client=RecordingClient(_VALID), text_source=EmptyTextSource()
    )
    second = await unfetched.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(first, NoComparisonPossible)
    assert isinstance(second, NoComparisonPossible)
    assert first.reason is not second.reason
    assert first != second


async def test_an_empty_prior_body_cannot_be_offered_at_all() -> None:
    """A source with nothing must return ``None``, not an empty string."""
    with pytest.raises(ValueError, match="must be non-empty"):
        PriorDocumentText(text="   \n")


# ---------------------------------------------------------------------------
# 4. Schema validation refuses; it never repairs
# ---------------------------------------------------------------------------


_QUOTED_NUMBER = (
    '{"severity_shift": "0.25", "hedging_shift": 0.0, "specificity_shift": 0.0, '
    '"evidence": [], "confidence": 0.5}'
)
_OUT_OF_RANGE = (
    '{"severity_shift": 4.0, "hedging_shift": 0.0, "specificity_shift": 0.0, '
    '"evidence": [], "confidence": 0.5}'
)
_EXTRA_FIELD = (
    '{"severity_shift": 0.1, "hedging_shift": 0.0, "specificity_shift": 0.0, '
    '"evidence": [], "confidence": 0.5, "ticker": "AHCO"}'
)
_FENCED = (
    '```json\n{"severity_shift": 0.1, "hedging_shift": 0.0, "specificity_shift": 0.0, '
    '"evidence": [], "confidence": 0.5}\n```'
)


@pytest.mark.parametrize(
    ("response", "why"),
    [
        (_QUOTED_NUMBER, "a quoted number is not a number"),
        (_OUT_OF_RANGE, "out of the declared [-1, 1] range"),
        (_EXTRA_FIELD, "an extra field means a different question was answered"),
        (_FENCED, "a fenced block is not bare JSON"),
        ('{"severity_shift": 0.1}', "missing required fields"),
        ("not json at all", "not JSON"),
    ],
)
async def test_a_malformed_response_is_rejected_rather_than_coerced(
    monkeypatch: pytest.MonkeyPatch, response: str, why: str
) -> None:
    """Rejected, recorded, and never turned into a number by a helpful conversion."""
    current = _current_document()
    extractor, _ = _extractor(monkeypatch, client=RecordingClient(response))

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, DeltaRejected), why
    assert outcome.validation_errors
    assert outcome.raw_response == response
    assert not hasattr(outcome, "output")


async def test_a_rejection_still_carries_the_full_stamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected response is evidence about a prompt version, so it names one."""
    current = _current_document()
    extractor, _ = _extractor(monkeypatch, client=RecordingClient('{"severity_shift": 0.1}'))

    outcome = await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert isinstance(outcome, DeltaRejected)
    assert outcome.stamp.prompt_version_hash
    assert outcome.stamp.anchor == current.knowledge_time


# ---------------------------------------------------------------------------
# 5. Refusals that are blockers, not data
# ---------------------------------------------------------------------------


async def test_no_configured_client_means_a_refusal_not_a_plausible_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B4: the default client raises. "No model answered" is not an observation."""
    current = _current_document()
    store = RecordingStore([_prior_row()])
    monkeypatch.setattr(resolve_module, "as_of", store.as_of)
    extractor = DeltaExtractor(
        pipeline=ExtractionPipeline(chunking=ChunkingConfig(max_chars=1_000_000)),
        text_source=StubTextSource({PRIOR_ACCESSION: _sap_body()}),
    )

    with pytest.raises(ProviderNotConfiguredError, match="no model client is configured"):
        await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)


async def test_a_backwards_pair_is_refused_before_anything_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner re-checks the direction the resolver promised.

    A comparison run the wrong way round returns a perfectly well-formed answer
    with every sign inverted — the one failure that survives schema validation
    untouched — so it must not reach a provider. Exercised by handing the runner
    a baseline its own resolver would never produce, which is what a
    re-verification is for.
    """
    current = _current_document()
    client = RecordingClient(_VALID)
    extractor, _ = _extractor(monkeypatch, client=client)

    async def _backwards(document: CurrentDocument, *, document_class: object) -> PriorDocumentRef:
        del document_class
        return PriorDocumentRef(
            accession_number=PRIOR_ACCESSION,
            cik=document.cik,
            company_name=document.company_name,
            form_type=document.form_type,
            acceptance=document.acceptance + dt.timedelta(days=30),
            knowledge_time=document.knowledge_time,
            chosen_at=document.anchor,
        )

    monkeypatch.setattr(runner_module, "resolve_prior_filing", _backwards)

    with pytest.raises(BaselineIdentityError, match="every sign inverted"):
        await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)
    assert client.calls == 0


def test_no_governed_client_can_be_built_either_because_no_cap_is_configured() -> None:
    """The other half of B4: the governor refuses to run without configured caps.

    So the two live paths to a provider are both closed — an unconfigured
    client raises, and a governed client cannot be constructed at all.
    """
    with pytest.raises(CapsNotConfiguredError, match="no spend cap is configured"):
        CapBook.of()


async def test_a_transcript_task_refuses_and_is_not_recorded_as_a_data_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P3.6 is blocked on B1. An unbuilt connector is a blocker, not a value."""
    current = _current_document()
    extractor, _ = _extractor(
        monkeypatch, client=RecordingClient(_VALID), text_source=UnaskedTextSource()
    )

    with pytest.raises(BaselineSourceUnavailableError, match="blocked on B1"):
        await extractor.extract(QA_EVASIVENESS_SHIFT_SPEC, current, model=MODEL)


async def test_running_a_task_over_the_wrong_form_is_a_caller_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run-configuration mistake, raised rather than filed as "no comparison"."""
    current = CurrentDocument(
        accession_number=CURRENT_ACCESSION,
        cik=CIK,
        company_name="AdaptHealth Corp.",
        form_type="8-K",
        acceptance=CURRENT_ACCEPTED,
        knowledge_time=CURRENT_ACCEPTED,
        text=read(ADAPTHEALTH_FORM4),
    )
    extractor, _ = _extractor(
        monkeypatch, client=RecordingClient(_VALID), rows=[], text_source=UnaskedTextSource()
    )

    with pytest.raises(DocumentClassMismatchError, match="not read by document class"):
        await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)


# ---------------------------------------------------------------------------
# 6. Structure: nothing here can be pointed at another instant
# ---------------------------------------------------------------------------


def test_extract_takes_no_as_of_anchor_session_or_clock() -> None:
    """Verified on the runtime object, not only in the source (D-033's discipline)."""
    parameters = set(inspect.signature(DeltaExtractor.extract).parameters)
    assert parameters == {"self", "spec", "current", "model", "timeout_s", "correlation_id"}


def test_the_extractor_holds_no_model_client_of_its_own() -> None:
    """Every call goes through the pipeline, so nothing can skip anonymization."""
    pipeline = ExtractionPipeline(client=RecordingClient(_VALID))
    extractor = DeltaExtractor(pipeline=pipeline, text_source=EmptyTextSource())
    held = {name: getattr(extractor, name) for name in dir(extractor) if not name.startswith("__")}
    assert not [value for value in held.values() if hasattr(value, "complete")]


def test_a_text_source_is_required_and_has_no_default() -> None:
    """A default returning ``None`` would file every document as "text unavailable"."""
    parameters = inspect.signature(DeltaExtractor.__init__).parameters
    assert parameters["text_source"].default is inspect.Parameter.empty
    assert parameters["pipeline"].default is inspect.Parameter.empty


async def test_the_store_is_opened_exactly_once_per_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One anchor, one read. A second read could be answered at a second instant."""
    current = _current_document()
    extractor, store = _extractor(monkeypatch, client=RecordingClient(_VALID))

    await extractor.extract(RISK_FACTOR_LANGUAGE_DELTA_SPEC, current, model=MODEL)

    assert store.opened_at == [current.knowledge_time]
    assert store.opened_at[0] < dt.datetime.now(dt.UTC)
