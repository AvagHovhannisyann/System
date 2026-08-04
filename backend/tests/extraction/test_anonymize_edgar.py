"""P7.2: masking real EDGAR text, and what survives it.

Every document masked here is a captured SEC response (see
``backend/tests/extraction/edgar_text.py``). The three assertions that matter:

1. a company written ten different ways in one real header survives in none of
   them;
2. two different declared entities never share a placeholder, and one entity
   never has two;
3. the leak detector, run on the result, reports nothing at ``LEAK`` severity.

The last of those is the one that generalises. The first two check the cases we
thought of; the detector checks with patterns written independently of the
masker, so it can disagree with it — and the residual findings it *does* report
are recorded here as the honest statement of what this masker leaves behind.
"""

from __future__ import annotations

import re

import pytest

from backend.extraction.anonymize import AnonymizerConfig, anonymize
from backend.extraction.entities import company, person, ticker
from backend.extraction.leak import LeakKind, Severity, detect_leaks
from backend.extraction.rules import MaskKind
from backend.tests.extraction.edgar_text import (
    ADAPTHEALTH_FORM4,
    DAILY_INDEX,
    SAP_6K,
    adapthealth_entities,
    read,
    sap_entities,
)

# The ten writings of the Deerfield filer family that the AdaptHealth joint
# Form 4 header actually contains, plus the issuer's two. Transcribed from the
# fixture; `test_every_declared_writing_is_present_in_the_fixture` proves each
# one is really there, so a passing masking assertion cannot be vacuous.
_ADAPTHEALTH_WRITINGS = (
    "AdaptHealth Corp.",
    "DFB Healthcare Acquisitions Corp.",
    "DEERFIELD MANAGEMENT COMPANY, L.P. (SERIES C)",
    "DEERFIELD MANAGEMENT CO",
    "DEERFIELD MANAGEMENT CO /NY",
    "Deerfield Mgmt L.P.",
    "DEERFIELD CAPITAL LP",
    "DEERFIELD CAPITAL LP ET AL",
    "DEERFIELD PARTNERS, L.P.",
    "DEERFIELD PARTNERS, LP",
    "Deerfield Private Design Fund IV, L.P.",
    "Deerfield Mgmt IV, L.P.",
    "Flynn James E",
)


def _bounded(value: str) -> re.Pattern[str]:
    """Word-bounded, case-insensitive, whitespace-flexible search for ``value``."""
    body = r"\s+".join(re.escape(p) for p in value.split())
    return re.compile(r"(?<![0-9A-Za-z])" + body + r"(?![0-9A-Za-z])", re.IGNORECASE)


@pytest.mark.parametrize("writing", _ADAPTHEALTH_WRITINGS)
def test_every_declared_writing_is_present_in_the_fixture(writing: str) -> None:
    assert _bounded(writing).search(read(ADAPTHEALTH_FORM4)) is not None


@pytest.mark.parametrize("writing", _ADAPTHEALTH_WRITINGS)
def test_no_declared_writing_survives_anonymization(writing: str) -> None:
    raw = read(ADAPTHEALTH_FORM4)
    masked = anonymize(raw, adapthealth_entities()).text
    assert _bounded(writing).search(masked) is None


def test_abbreviated_writings_are_caught_without_the_leading_token_rule() -> None:
    """``Mgmt``/``CO``/``L.P.`` reach the same entity as ``Management Company, L.P.``.

    Run with ``mask_leading_token=False`` on purpose. With it on, every one of
    these would be masked by the bare token ``Deerfield`` and the test would
    prove nothing about abbreviation handling — which is the part most likely
    to rot silently.
    """
    raw = read(ADAPTHEALTH_FORM4)
    entities = [
        company(
            "DEERFIELD MANAGEMENT COMPANY, L.P. (SERIES C)",
            "DEERFIELD MANAGEMENT CO",
            "DEERFIELD MANAGEMENT CO /NY",
        )
    ]
    masked = anonymize(raw, entities, AnonymizerConfig(mask_leading_token=False)).text
    for writing in (
        "DEERFIELD MANAGEMENT COMPANY, L.P. (SERIES C)",
        "DEERFIELD MANAGEMENT CO",
        "Deerfield Mgmt",
    ):
        assert _bounded(writing).search(masked) is None
    # Precision, the other half: a *different* declared-name family is not
    # collapsed into this one when the leading-token rule is off.
    assert _bounded("DEERFIELD PARTNERS, L.P.").search(masked) is not None


def test_placeholders_are_consistent_within_a_document() -> None:
    """One entity, one placeholder — no matter which writing was on the page."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    by_placeholder: dict[str, set[str]] = {}
    for replacement in document.replacements:
        by_placeholder.setdefault(replacement.placeholder, set()).add(replacement.rule)

    issuer = [r for r in document.replacements if r.original == "AdaptHealth Corp."]
    former = [r for r in document.replacements if r.original == "DFB Healthcare Acquisitions Corp."]
    assert issuer, "fixture must contain the issuer's conformed name"
    assert former, "fixture must contain the issuer's former conformed name"
    assert {r.placeholder for r in issuer} == {r.placeholder for r in former}
    assert len({r.placeholder for r in issuer}) == 1


def test_different_entities_get_different_placeholders() -> None:
    """Six declared filer entities, six distinct company placeholders."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    companies = {r.placeholder for r in document.replacements if r.kind is MaskKind.COMPANY}
    assert len(companies) == 6
    persons = {r.placeholder for r in document.replacements if r.kind is MaskKind.PERSON}
    assert persons == {"[PERSON_1]"}
    assert companies.isdisjoint(persons)


def test_the_same_date_written_two_ways_shares_one_placeholder() -> None:
    """The daily index writes 2024-03-11 as ``Mar 11, 2024`` and as ``20240311``."""
    raw = read(DAILY_INDEX)
    document = anonymize(raw, [])
    written = {r.original: r.placeholder for r in document.replacements if r.kind is MaskKind.DATE}
    assert "Mar 11, 2024" in written, "fixture must contain the prose writing"
    assert "20240311" in written, "fixture must contain the compact writing"
    assert written["Mar 11, 2024"] == written["20240311"]


def test_acceptance_timestamp_and_filing_dates_are_gone() -> None:
    """The 14-digit acceptance stamp is the timestamp the whole P3 layer turns on."""
    raw = read(ADAPTHEALTH_FORM4)
    assert "20240312190139" in raw
    masked = anonymize(raw, adapthealth_entities()).text
    assert "20240312190139" not in masked
    assert "20240312" not in masked
    assert "2024-03-12 19:01:39" not in masked


def test_declared_ciks_do_not_survive() -> None:
    raw = read(ADAPTHEALTH_FORM4)
    masked = anonymize(raw, adapthealth_entities()).text
    for cik in ("0001725255", "1725255", "0001352546"):
        assert _bounded(cik).search(masked) is None


def test_leak_detector_reports_no_leaks_on_the_adapthealth_filing() -> None:
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    assert report.leaks == (), report.summary()
    assert report.clean
    # Not vacuous: the detector found the names in the original.
    assert report.occurrences_in_original["AdaptHealth Corp."] > 0
    assert report.occurrences_in_original["Flynn James E"] > 0


def test_leak_detector_reports_no_leaks_on_the_sap_filing() -> None:
    raw = read(SAP_6K)
    document = anonymize(raw, sap_entities())
    report = detect_leaks(original=raw, anonymized=document.text, entities=sap_entities())
    assert report.leaks == (), report.summary()
    assert report.occurrences_in_original["SAP SE"] > 0


def test_leak_detector_reports_no_leaks_on_the_daily_index() -> None:
    raw = read(DAILY_INDEX)
    entities = [company("UNIFI INC"), company("SEPARATE ACCOUNT NO. 49")]
    document = anonymize(raw, entities)
    report = detect_leaks(original=raw, anonymized=document.text, entities=entities)
    assert report.leaks == (), report.summary()


def test_statute_years_are_the_only_residual_on_the_adapthealth_filing() -> None:
    """What this masker leaves behind on a real filing, recorded rather than claimed.

    ``SEC ACT: 1934 Act`` is a statute reference, not a document date, and the
    unit/statute exemption in the bare-year rule spares it. The detector still
    reports it — at ``RESIDUAL`` — because the exemption is a hole in the date
    guarantee and holes get counted. If this assertion ever starts failing with
    a *new* kind of residual, that is the signal it exists for.
    """
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    report = detect_leaks(original=raw, anonymized=document.text, entities=adapthealth_entities())
    assert {finding.kind for finding in report.residuals} == {LeakKind.YEAR}
    assert {finding.matched for finding in report.residuals} == {"1934"}


def test_a_company_name_inside_a_filename_token_is_a_known_miss() -> None:
    """``e619363_4-ahcorp.xml`` survives, and this records that it does.

    The AdaptHealth Form 4 manifest names its primary document
    ``e619363_4-ahcorp.xml``. ``ahcorp`` is a fragment inside a token, and every
    rule here matches on token boundaries. Reaching it would need substring
    matching, which redacts fragments of ordinary words and would do more damage
    than this leaves. Stated, not fixed — a chunker that strips the document
    manifest before extraction (P7.3) removes it more cleanly than a regex can.
    """
    raw = read(ADAPTHEALTH_FORM4)
    assert "ahcorp" in raw
    masked = anonymize(raw, adapthealth_entities()).text
    assert "ahcorp" in masked
    report = detect_leaks(original=raw, anonymized=masked, entities=adapthealth_entities())
    assert all(finding.matched != "ahcorp" for finding in report.findings)


def test_undeclared_addresses_and_registry_numbers_are_a_known_miss() -> None:
    """Street addresses, phone numbers, IRS and film numbers are not detected.

    All four identify the filer about as well as its name does. None is masked
    unless declared, and none is reported. A caller that cares must pass them as
    ``identifier`` entities; this test exists so nobody discovers that by
    accident in P7.9.
    """
    raw = read(ADAPTHEALTH_FORM4)
    masked = anonymize(raw, adapthealth_entities()).text
    for survivor in ("220 WEST GERMANTOWN PIKE", "610-630-6357", "823677704", "24743961"):
        assert survivor in masked


def test_ticker_masking_on_constructed_probes() -> None:
    """Constructed strings, not filing text: the captured headers carry no ticker.

    Exercises the ticker rule at inputs the fixtures do not contain, the same
    way ``test_edgar_parse`` exercises the timezone conversion at local times
    EDGAR cannot emit. ``ZZZQ`` is a placeholder symbol chosen so the test
    asserts nothing about any real security.
    """
    entities = [ticker("ZZZQ")]
    assert anonymize("(NASDAQ: ZZZQ) rose", entities).text == "(NASDAQ: [TICKER_1]) rose"
    assert anonymize("NYSE:ZZZQ", entities).text == "NYSE:[TICKER_1]"
    assert anonymize("Zzzq gained", entities).text == "[TICKER_1] gained"
    # Known miss, recorded: a lower-case writing is not matched, because a
    # case-insensitive ticker rule would redact ordinary words for the many
    # symbols that are ones (ALL, KEY, ON).
    assert anonymize("zzzq gained", entities).text == "zzzq gained"


def test_possessive_and_line_broken_writings_on_constructed_probes() -> None:
    """Constructed strings: neither writing occurs in the captured headers."""
    entities = [company("AdaptHealth Corp."), person("James E. Flynn")]
    assert anonymize("AdaptHealth's revenue", entities).text == "[COMPANY_1]'s revenue"
    assert anonymize("AdaptHealth Corp.'s board", entities).text == "[COMPANY_1]'s board"
    assert anonymize("AdaptHealth,\nInc. said", entities).text == "[COMPANY_1] said"
    assert anonymize("Mr. Flynn resigned", entities).text == "Mr. [PERSON_1] resigned"
    assert anonymize("Flynn's shares", entities).text == "[PERSON_1]'s shares"


def test_a_spaced_variant_of_a_closed_compound_name_is_a_known_miss() -> None:
    """``Adapt Health`` for ``AdaptHealth`` is not matched, and is recorded as such.

    Token-level matching cannot see inside a closed compound. Catching this
    would need per-name segmentation, which invents word boundaries that are
    not in the declared name.
    """
    assert anonymize("Adapt Health filed", [company("AdaptHealth Corp.")]).text == (
        "Adapt Health filed"
    )


def test_masking_is_deterministic() -> None:
    """Two runs, byte-identical output — P7.9 compares scorings of one document."""
    raw = read(ADAPTHEALTH_FORM4)
    first = anonymize(raw, adapthealth_entities())
    second = anonymize(raw, adapthealth_entities())
    assert first.text == second.text
    assert first.replacements == second.replacements


def test_severity_ordering_of_a_residual_is_not_a_leak() -> None:
    raw = read(SAP_6K)
    document = anonymize(raw, sap_entities())
    report = detect_leaks(original=raw, anonymized=document.text, entities=sap_entities())
    assert all(finding.severity is not Severity.LEAK for finding in report.residuals)
