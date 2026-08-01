"""P7.2: what a caller may declare, and what is refused.

An :class:`~backend.extraction.entities.Entity` is the whole input to the
name-masking half of this package: what the caller declares is exactly what gets
masked and exactly what the leak detector then checks for. That makes the
constructors' edges worth pinning down — a declaration that is silently accepted
and means nothing produces a document that looks masked and is not.

The surname convention gets the most attention here because it is the one place
a plausible guess is wrong half the time: EDGAR writes an individual
surname-first (``Flynn James E``) and prose writes them forename-first, and
picking the wrong token leaves the real surname standing in the text.
"""

from __future__ import annotations

import pytest

from backend.extraction.anonymize import anonymize
from backend.extraction.entities import (
    Entity,
    EntityKind,
    company,
    identifier,
    person,
    person_from_edgar_conformed_name,
    ticker,
)
from backend.extraction.rules import MaskKind
from backend.extraction.surface import core_tokens


def test_an_empty_name_is_refused() -> None:
    """A declaration that masks nothing must not look like one that masks."""
    with pytest.raises(ValueError, match="non-empty"):
        company("   ")


def test_person_only_fields_are_refused_on_other_kinds() -> None:
    """A surname on a company would build a surface rule nothing else expects."""
    with pytest.raises(ValueError, match="PERSON-only"):
        Entity(kind=EntityKind.COMPANY, name="AdaptHealth Corp.", surname="Corp")
    with pytest.raises(ValueError, match="PERSON-only"):
        Entity(kind=EntityKind.TICKER, name="ZZZQ", given_names=("Z",))


def test_writings_are_the_name_then_aliases_deduplicated_in_order() -> None:
    entity = company("SAP SE", "SAP AG", "SAP SE", "  ")
    assert entity.writings == ("SAP SE", "SAP AG")


def test_each_kind_maps_to_its_placeholder_category() -> None:
    assert company("A B").mask_kind is MaskKind.COMPANY
    assert person("A B").mask_kind is MaskKind.PERSON
    assert ticker("ZZZQ").mask_kind is MaskKind.TICKER
    assert identifier("0001725255").mask_kind is MaskKind.IDENTIFIER


def test_person_reads_a_natural_order_name() -> None:
    entity = person("James E. Flynn")
    assert entity.surname == "Flynn"
    assert entity.given_names == ("James", "E")


def test_person_ignores_a_generational_suffix_when_finding_the_surname() -> None:
    """``John Smith Jr.`` is a Smith, not a Jr."""
    entity = person("John Smith Jr.")
    assert entity.surname == "Smith"


def test_person_accepts_an_explicit_surname_and_given_names() -> None:
    """The escape hatch for a name no convention parses."""
    entity = person("Jean-Luc de la Fontaine", surname="de la Fontaine", given_names=("Jean-Luc",))
    assert entity.surname == "de la Fontaine"
    assert entity.given_names == ("Jean-Luc",)


def test_person_from_edgar_conformed_name_reads_surname_first() -> None:
    """``Flynn James E`` is James E. Flynn, and guessing the other way is a leak.

    Reading this as natural order would put the surname-alone rule on ``E`` and
    leave every bare ``Flynn`` in the document.
    """
    entity = person_from_edgar_conformed_name("Flynn James E")
    assert entity.surname == "Flynn"
    assert entity.given_names == ("James", "E")


def test_a_single_token_person_name_has_no_given_names() -> None:
    entity = person("Prince")
    assert entity.surname == "Prince"
    assert entity.given_names == ()


def test_core_tokens_drops_a_leading_article_and_the_legal_form() -> None:
    assert core_tokens("The AdaptHealth Corporation") == ("AdaptHealth",)
    assert core_tokens("DEERFIELD MANAGEMENT COMPANY, L.P.") == ("DEERFIELD", "MANAGEMENT")


def test_core_tokens_never_returns_nothing() -> None:
    """A rule built from no tokens would match everywhere.

    ``The Company`` is all article and legal form. Returning an empty tuple
    would produce a pattern of nothing, which matches at every position.
    """
    assert core_tokens("The Company") == ("Company",)
    assert core_tokens("Inc.") == ("Inc",)


def test_core_tokens_merges_runs_of_single_letters() -> None:
    """``L.P.`` is one token, so the legal-form check can recognise it."""
    assert core_tokens("J. P. Morgan Co") == ("JP", "Morgan")


def test_a_name_with_no_alphanumeric_tokens_yields_no_core_tokens() -> None:
    """``Entity`` accepts it (it is non-empty), so the token layer must not crash.

    A name of pure punctuation has nothing to build a pattern from. Returning no
    tokens is what makes the core rule decline to exist for it; the verbatim
    rule still matches the string literally.
    """
    assert core_tokens("---") == ()
    assert anonymize("--- filed ---", [company("---")]).text == "[COMPANY_1] filed [COMPANY_1]"
