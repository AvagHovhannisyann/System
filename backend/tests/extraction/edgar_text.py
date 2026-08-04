"""Real EDGAR text and the entity metadata that goes with it (P7.2 test support).

The anonymizer is only worth testing on documents nobody wrote for it. Every
string these tests mask comes out of ``backend/tests/fixtures/edgar/`` — the
responses the P3.2 connector captured off ``www.sec.gov``, each carrying its own
provenance block. Nothing here paraphrases, trims or "cleans up" a filing.

The entity declarations below are transcribed from the ``COMPANY CONFORMED
NAME``, ``FORMER CONFORMED NAME`` and ``CENTRAL INDEX KEY`` fields of the
fixture they name, which is exactly what the connector would hand the
anonymizer at run time. They are metadata read out of the document, not
invented facts about it.

Constructed strings appear in these tests only where a *pure function* is being
exercised at inputs the captured fixtures happen not to contain (a possessive,
a line-broken date). Those are labelled at their use site, following the
precedent set in ``backend/tests/ingest/test_edgar_parse.py``: exercising a
conversion at a value is not the same as inventing a source payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from backend.extraction.entities import (
    Entity,
    company,
    identifier,
    person_from_edgar_conformed_name,
)

FIXTURE_DIR: Final = Path(__file__).resolve().parents[1] / "fixtures" / "edgar"
"""Directory holding the captured EDGAR responses (shared with the P3.2 tests)."""

ADAPTHEALTH_FORM4: Final = "0001193805-24-000360-index-headers.html"
"""Joint Form 4: one issuer, six reporting-owner entities, one individual."""

SAP_6K: Final = "0001104659-24-082105-index-headers.html"
"""6-K by a foreign private issuer, with two former conformed names."""

UNIFI_8K: Final = "0000950170-24-029225-index-headers.html"
"""8-K header naming two companies."""

DAILY_INDEX: Final = "master.20240311.sample.idx"
"""A daily index: pipe-delimited rows, dates in two writings, real company names."""


def read(name: str) -> str:
    """Return a captured fixture verbatim.

    Args:
        name: File name inside ``backend/tests/fixtures/edgar/``.

    Returns:
        The file's text, including its provenance block. The provenance block is
        left in deliberately: it is prose containing dates and names, which is
        more of the sort of text the anonymizer will meet than SGML alone.
    """
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


def adapthealth_entities() -> list[Entity]:
    """Every entity named in the header of the AdaptHealth joint Form 4.

    Transcribed from that fixture's ``COMPANY CONFORMED NAME``, ``FORMER
    CONFORMED NAME`` and ``CENTRAL INDEX KEY`` fields, and from nothing else.
    No ticker is declared: the captured header does not carry one, and asserting
    a symbol from memory would be exactly the fabricated fact I3 forbids.
    Ticker masking is covered separately, on constructed inputs.
    """
    return [
        company("AdaptHealth Corp.", "DFB Healthcare Acquisitions Corp."),
        company(
            "DEERFIELD MANAGEMENT COMPANY, L.P. (SERIES C)",
            "DEERFIELD MANAGEMENT CO",
            "DEERFIELD MANAGEMENT CO /NY",
        ),
        company("Deerfield Mgmt L.P.", "DEERFIELD CAPITAL LP", "DEERFIELD CAPITAL LP ET AL"),
        company("DEERFIELD PARTNERS, L.P.", "DEERFIELD PARTNERS, LP"),
        company("Deerfield Private Design Fund IV, L.P."),
        company("Deerfield Mgmt IV, L.P."),
        person_from_edgar_conformed_name("Flynn James E"),
        identifier("0001725255"),
        identifier("0001009258"),
        identifier("0001010823"),
        identifier("0001301041"),
        identifier("0001352546"),
        identifier("0001680307"),
        identifier("0001713467"),
    ]


def sap_entities() -> list[Entity]:
    """The filer and its former names, from the SAP 6-K header."""
    return [
        company(
            "SAP SE",
            "SAP AG",
            "SAP AKTIENGESELLSCHAFT SYSTEMS APPLICATIONS PRODUCTS IN DATA",
        ),
        identifier("0001000184"),
    ]
