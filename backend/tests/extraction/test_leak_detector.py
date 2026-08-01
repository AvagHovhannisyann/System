"""P7.2: proving the leak detector fires (and on what).

A detector that never fires is worthless, and worse than worthless here: the
whole point of :mod:`backend.extraction.leak` is that P7.9's contamination probe
should measure the *model's* recall rather than our masking misses, and a
detector stuck at "clean" would let every miss through under a green light.
``assert report.leaks == ()`` on well-masked text cannot distinguish a working
detector from ``return LeakReport((), {})``.

So the tests here are built the other way round. Each one produces text that is
deliberately under-masked — by switching a real masking rule off, or by
withholding an entity from the masker while still declaring it to the detector —
and asserts the detector *reports* what survived, with the right kind and the
right severity. The under-masked text is captured EDGAR text wherever the
fixtures contain the writing in question; constructed strings are used only for
the two classes the captured headers do not contain (a ticker, a bare year in
prose) and are labelled where they appear.

The negative control is here too, immediately beside the positive ones, so the
pair is read together: on strictly-masked text the same detector is silent.
"""

from __future__ import annotations

import pytest

from backend.extraction.anonymize import AnonymizerConfig, anonymize
from backend.extraction.entities import company, identifier, person, ticker
from backend.extraction.leak import LeakKind, Severity, detect_leaks
from backend.tests.extraction.edgar_text import (
    ADAPTHEALTH_FORM4,
    DAILY_INDEX,
    SAP_6K,
    adapthealth_entities,
    read,
    sap_entities,
)

# (switch turned off, the kind that must then be reported). Every case masks the
# captured AdaptHealth joint Form 4 with all other rules still in force, so what
# the detector finds is real filing text that a real rule would have removed.
_UNDER_MASKING_CASES = [
    pytest.param(AnonymizerConfig(mask_dates=False), LeakKind.DATE, id="dates-off"),
    pytest.param(
        AnonymizerConfig(mask_accession_numbers=False), LeakKind.ACCESSION, id="accessions-off"
    ),
    pytest.param(
        AnonymizerConfig(mask_identifiers=False), LeakKind.IDENTIFIER, id="identifiers-off"
    ),
]


def test_the_detector_is_silent_on_strictly_masked_text() -> None:
    """The negative control. Read it with the positive cases below, not alone."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    assert report.leaks == (), report.summary()
    assert report.clean


@pytest.mark.parametrize(("config", "expected"), _UNDER_MASKING_CASES)
def test_the_detector_fires_when_a_masking_rule_is_switched_off(
    config: AnonymizerConfig, expected: LeakKind
) -> None:
    """Switch one rule off, and the detector reports exactly that class of leak."""
    raw = read(ADAPTHEALTH_FORM4)
    entities = adapthealth_entities()
    document = anonymize(raw, entities, config)
    report = detect_leaks(original=raw, anonymized=document.text, entities=entities)

    assert report.leaks, "under-masked text must produce at least one leak"
    assert not report.clean
    assert {finding.kind for finding in report.leaks} == {expected}
    assert all(finding.severity is Severity.LEAK for finding in report.leaks)


def test_the_detector_fires_when_entities_are_withheld_from_the_masker() -> None:
    """The masker is told nothing; the detector is told everything.

    This is the failure the metadata-driven design is most exposed to — an
    ingestion path that forgets to pass the reporting owners — and it is the one
    case where the detector is genuinely checking the masker rather than
    checking a switch.
    """
    raw = read(ADAPTHEALTH_FORM4)
    entities = adapthealth_entities()
    document = anonymize(raw, [])
    report = detect_leaks(original=raw, anonymized=document.text, entities=entities)

    kinds = {finding.kind for finding in report.leaks}
    assert LeakKind.COMPANY_NAME in kinds
    assert LeakKind.PERSON_NAME in kinds
    assert LeakKind.IDENTIFIER in kinds


def test_the_detector_fires_on_a_second_filing_too() -> None:
    """Not a fixture-specific accident: the same withholding on the SAP 6-K."""
    raw = read(SAP_6K)
    document = anonymize(raw, [])
    report = detect_leaks(original=raw, anonymized=document.text, entities=sap_entities())
    assert {finding.matched for finding in report.leaks} >= {"SAP SE", "SAP AG"}


def test_the_detector_finds_a_name_split_across_a_line_break() -> None:
    """The reason the scan runs over a whitespace-collapsed copy.

    Constructed: the captured headers write the name on one line. A masker that
    matched only literal text would miss the wrapped writing, and a detector
    that also matched only literal text would agree with it and report nothing —
    two independent-looking checks with one shared blind spot.
    """
    original = "AdaptHealth Corp. filed."
    under_masked = "AdaptHealth\n   Corp. filed."
    report = detect_leaks(
        original=original, anonymized=under_masked, entities=[company("AdaptHealth Corp.")]
    )
    assert [finding.kind for finding in report.leaks] == [LeakKind.COMPANY_NAME]
    finding = report.leaks[0]
    assert under_masked[finding.start : finding.end] == finding.matched
    assert "\n" in finding.matched


def test_the_detector_finds_an_unpadded_form_of_a_declared_identifier() -> None:
    """A CIK is ``0001725255`` in a header and ``1725255`` in a URL."""
    original = "CIK 0001725255 also written 1725255"
    under_masked = "CIK [ID_1] also written 1725255"
    report = detect_leaks(
        original=original, anonymized=under_masked, entities=[identifier("0001725255")]
    )
    assert [finding.matched for finding in report.leaks] == ["1725255"]


def test_the_detector_finds_a_ticker() -> None:
    """Constructed: the captured headers carry no ticker (see ``edgar_text``)."""
    report = detect_leaks(
        original="(NASDAQ: ZZZQ)", anonymized="(NASDAQ: ZZZQ)", entities=[ticker("ZZZQ")]
    )
    assert [finding.kind for finding in report.leaks] == [LeakKind.TICKER]


def test_the_detector_finds_a_bare_year_the_masker_was_told_to_leave() -> None:
    """A daily index dated only by a four-digit year, with the year rule off."""
    raw = read(DAILY_INDEX)
    document = anonymize(raw, [], AnonymizerConfig(mask_bare_years=False))
    report = detect_leaks(original=raw, anonymized=document.text)
    assert [finding.kind for finding in report.leaks] == [LeakKind.YEAR]
    assert report.leaks[0].matched == "2024"


def test_the_detector_reports_an_unmasked_numeric_date() -> None:
    """One date can produce two findings, and that is the intended direction.

    The numeric scanner reports the whole ``11-03-2024``; the year scanner
    independently reports the ``2024`` inside it, because its boundary
    deliberately permits a leading ``-`` (a year in a URL path would otherwise
    be invisible to it). Two overlapping reports of one leak is noise; one
    missed leak is a contaminated feature.
    """
    report = detect_leaks(original="dated 11-03-2024", anonymized="dated 11-03-2024")
    matched = {(finding.kind, finding.matched) for finding in report.leaks}
    assert (LeakKind.DATE, "11-03-2024") in matched
    assert (LeakKind.YEAR, "2024") in matched


def test_a_dotted_european_date_is_masked_and_its_absence_is_checked() -> None:
    """``11.03.2024`` is how the captured 6-K's German filer writes a date.

    It needed its own rule: the dot before ``2024`` triggers the bare-year
    exemption, so before this rule existed the whole date survived and the only
    report was a ``RESIDUAL`` on the year. That is precisely the shape of miss
    the detector exists to surface — a full date, silently intact.
    """
    assert anonymize("dated 11.03.2024").text == "dated [DATE_1]"
    report = detect_leaks(original="dated 11.03.2024", anonymized="dated 11.03.2024")
    assert (LeakKind.DATE, "11.03.2024") in {(f.kind, f.matched) for f in report.leaks}


def test_outline_numbering_is_not_mistaken_for_a_dotted_date() -> None:
    """The precision half: ``2.1.3`` and ``Item 1.01`` are not calendar days."""
    assert anonymize("see section 2.1.3").text == "see section 2.1.3"
    assert anonymize("Item 1.01").text == "Item 1.01"


def test_switching_dates_off_leaves_the_entity_scan_working() -> None:
    """``scan_dates=False`` silences the temporal scanners and nothing else."""
    raw = read(SAP_6K)
    document = anonymize(raw, [])
    report = detect_leaks(
        original=raw, anonymized=document.text, entities=sap_entities(), scan_dates=False
    )
    assert report.leaks, "entity leaks must still be reported"
    assert all(
        finding.kind is not LeakKind.DATE and finding.kind is not LeakKind.YEAR
        for finding in report.findings
    )


def test_a_residual_is_reported_but_does_not_fail_the_gate() -> None:
    """The three stated year exemptions are counted, not treated as failures."""
    for text in ("a charge of $2024", "2000 shares were sold", "under the 1934 Act"):
        masked = anonymize(text).text
        report = detect_leaks(original=text, anonymized=masked)
        assert report.leaks == (), text
        assert report.clean, text
        assert [finding.kind for finding in report.residuals] == [LeakKind.YEAR], text
        assert "stated exemption" in report.residuals[0].detail


def test_a_common_word_surname_is_a_residual_not_a_leak() -> None:
    """``May`` is exempt from the surname-alone rule, so it cannot be a failure.

    Reporting the masker's own stated exemption at ``LEAK`` would make the gate
    unreachable for anyone called May, March or Will, and the real leaks would
    drown in it. The finding is still made, at the severity that says "known
    limit" rather than "broken".
    """
    entities = [person("Jane May")]
    original = "Ms. May met Jane May."
    masked = anonymize(original, entities).text
    assert masked == "Ms. May met [PERSON_1]."
    report = detect_leaks(original=original, anonymized=masked, entities=entities)
    assert report.leaks == ()
    surname_findings = [f for f in report.residuals if f.kind is LeakKind.PERSON_NAME]
    assert surname_findings
    assert all(f.severity is Severity.RESIDUAL for f in surname_findings)


def test_a_surname_after_an_article_is_masked_rather_than_left_to_the_detector() -> None:
    """``the Flynn family`` is masked, and the readability cost is the point.

    An earlier draft guarded the surname-alone rule against ``the``/``a``/``an``
    so that ``the Cook`` read as a noun. That left a hole the detector — which
    has no such guard — reported as a ``LEAK``, so the masker was deliberately
    leaving behind something its own checker called a failure. The guard is
    gone: over-masking a capitalised common noun is visible in the replacement
    list and switchable off, and under-masking a declared officer is neither.
    """
    entities = [person("James E. Flynn")]
    original = "the Flynn family and a Flynn holding"
    document = anonymize(original, entities)
    assert document.text == "the [PERSON_1] family and a [PERSON_1] holding"
    report = detect_leaks(original=original, anonymized=document.text, entities=entities)
    assert report.leaks == ()


def test_the_cost_of_removing_the_article_guard_is_recorded() -> None:
    """A declared officer named Cook makes the capitalised word unreadable.

    This is the over-masking side of the trade above, written down so it is a
    decision rather than a surprise. ``mask_surname_alone=False`` turns it off
    for a run that needs the precision, at the cost of the surname-alone class.
    """
    entities = [person("Alan Cook")]
    original = "The Cook prepared it. Alan Cook signed."
    assert anonymize(original, entities).text == "The [PERSON_1] prepared it. [PERSON_1] signed."
    relaxed = anonymize(original, entities, AnonymizerConfig(mask_surname_alone=False))
    assert relaxed.text == "The Cook prepared it. [PERSON_1] signed."
    # And with it off, the detector says so rather than staying quiet.
    report = detect_leaks(original=original, anonymized=relaxed.text, entities=entities)
    assert [finding.kind for finding in report.leaks] == [LeakKind.PERSON_NAME]


def test_a_surname_that_is_not_a_common_word_is_a_leak() -> None:
    """The other half of the exemption: an ordinary surname left standing fails."""
    entities = [person("James E. Flynn")]
    report = detect_leaks(
        original="Mr. Flynn resigned", anonymized="Mr. Flynn resigned", entities=entities
    )
    assert [finding.kind for finding in report.leaks] == [LeakKind.PERSON_NAME]


def test_occurrences_in_original_exposes_a_vacuous_pass() -> None:
    """A clean report proves nothing when the name was never in the document."""
    report = detect_leaks(
        original="a document about nothing",
        anonymized="a document about nothing",
        entities=[company("Nonexistent Holdings Inc.")],
    )
    assert report.leaks == ()
    assert report.occurrences_in_original["Nonexistent Holdings Inc."] == 0


def test_occurrences_in_original_is_non_zero_when_the_name_was_really_there() -> None:
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    assert report.occurrences_in_original["AdaptHealth Corp."] > 0
    assert report.occurrences_in_original["Flynn James E"] > 0
    assert report.occurrences_in_original["0001725255"] > 0


def test_a_pre_existing_placeholder_is_info_not_a_leak() -> None:
    """A source that already contains ``[COMPANY_1]`` makes restore ambiguous."""
    report = detect_leaks(original="see [COMPANY_1] below", anonymized="see [COMPANY_1] below")
    assert report.leaks == ()
    assert [finding.kind for finding in report.findings] == [LeakKind.PLACEHOLDER_COLLISION]
    assert report.findings[0].severity is Severity.INFO


def test_the_summary_line_carries_counts_and_never_values() -> None:
    """``summary()`` is the one part of a report meant to be logged (§7, I5)."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, [])
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    summary = report.summary()
    assert summary == (
        f"leaks={len(report.leaks)} residuals={len(report.residuals)} total={len(report.findings)}"
    )
    for secret in ("AdaptHealth", "Deerfield", "Flynn", "0001725255"):
        assert secret not in summary


def test_every_finding_points_at_the_text_it_matched() -> None:
    """Offsets are into the anonymized text; the operator view relies on it."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, [], AnonymizerConfig(mask_dates=False))
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    assert report.findings
    for finding in report.findings:
        if finding.kind is LeakKind.PLACEHOLDER_COLLISION:
            continue  # offsets into the original, by definition of the finding
        assert document.text[finding.start : finding.end] == finding.matched
        assert finding.matched in finding.context.replace("\n", " ") or finding.matched.strip()


def test_findings_are_ordered_by_position() -> None:
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, [], AnonymizerConfig(mask_dates=False))
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    starts = [finding.start for finding in report.findings]
    assert starts == sorted(starts)


def test_leaks_and_residuals_partition_by_severity() -> None:
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, [], AnonymizerConfig(mask_dates=False))
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    assert all(finding.severity is Severity.LEAK for finding in report.leaks)
    assert all(finding.severity is Severity.RESIDUAL for finding in report.residuals)
    assert set(report.leaks).isdisjoint(report.residuals)
    assert report.clean is (len(report.leaks) == 0)
