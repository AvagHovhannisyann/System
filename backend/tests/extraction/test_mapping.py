"""P7.2: the reversible mapping, and keeping it away from the model.

The mapping is the exact inverse of the masking. If it ever reaches a prompt the
control is not merely weakened, it is cancelled — the model is handed both the
anonymized text and the key. It cannot be dropped either: without it the stored
extraction cannot be traced back to the filing it came from, which §6.5's
document inspector and any audit of a bad feature both need.

So it is kept, and kept apart. These tests check the two structural defences
that make "kept apart" more than a comment:

1. the payload and the key are *different objects*, so sending the payload is
   the natural thing to write and sending the key requires reaching for it;
2. both render redacted, so the accident — a traceback, a ``structlog`` event,
   a debugger frame, an f-string in a log line — cannot spill them.

Neither stops a caller determined to concatenate ``mapping.entries`` into a
prompt. Nothing in a library can. They stop the accident, which is the failure
that actually happens.
"""

from __future__ import annotations

from backend.extraction.anonymize import anonymize
from backend.extraction.entities import (
    AnonymizedDocument,
    EntityMapping,
    company,
    identifier,
    merge_writings,
    person,
)
from backend.extraction.rules import MaskKind
from backend.tests.extraction.edgar_text import (
    ADAPTHEALTH_FORM4,
    adapthealth_entities,
    read,
)


def test_the_mapping_repr_shows_placeholders_and_counts_but_no_values() -> None:
    """A traceback that prints the mapping must not print the filing's names."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    rendered = repr(document.mapping)

    assert "redacted" in rendered
    assert "[COMPANY_1]" in rendered
    for secret in ("AdaptHealth", "Deerfield", "Flynn", "0001725255", "DFB"):
        assert secret not in rendered, secret


def test_the_document_repr_shows_no_text_and_no_replaced_original() -> None:
    """``AnonymizedDocument`` is the object most likely to reach a log line.

    Redacting :class:`EntityMapping` alone would not be enough: every
    :class:`~backend.extraction.rules.Replacement` carries the original text of
    its span, so the default dataclass ``repr`` would print the key twice over
    and the whole document besides.
    """
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    rendered = repr(document)

    assert "redacted" in rendered
    for secret in ("AdaptHealth", "Deerfield", "Flynn", "0001725255", "<SEC-HEADER>"):
        assert secret not in rendered, secret
    assert str(len(document.replacements)) in rendered


def test_the_payload_and_the_key_are_separate_objects() -> None:
    """The text is a plain ``str``; nothing reachable from it re-identifies it."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    assert isinstance(document.text, str)
    assert isinstance(document.mapping, EntityMapping)
    assert isinstance(document, AnonymizedDocument)
    # Sending `document.text` sends no canonical value with it.
    for entry in document.mapping.entries:
        if entry.kind in {MaskKind.COMPANY, MaskKind.PERSON, MaskKind.IDENTIFIER}:
            assert entry.canonical not in document.text, entry.placeholder


def test_every_placeholder_in_the_text_is_in_the_mapping() -> None:
    """An un-mappable placeholder would be an un-auditable redaction."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    index = document.mapping.by_placeholder
    assert document.placeholders()
    for placeholder in document.placeholders():
        assert placeholder in index
        assert index[placeholder].occurrences >= 1


def test_restore_recovers_the_declared_names_on_a_real_filing() -> None:
    """The mapping really is the inverse, up to surface variation.

    Restoration is to the *canonical* writing, so it is semantically the
    original rather than byte-identical to it — a document that wrote
    ``DEERFIELD MANAGEMENT CO`` comes back as the declared conformed name.
    """
    raw = read(ADAPTHEALTH_FORM4)
    entities = adapthealth_entities()
    document = anonymize(raw, entities)
    restored = document.mapping.restore(document.text)

    assert "[COMPANY_1]" not in restored
    assert "AdaptHealth Corp." in restored
    assert "Flynn James E" in restored


def test_restore_puts_back_exactly_what_the_placeholder_replaced() -> None:
    """On text with no surface variation, restore is the identity."""
    original = "AdaptHealth Corp. reported on 2024-03-11."
    entities = [company("AdaptHealth Corp.")]
    document = anonymize(original, entities)
    assert document.text == "[COMPANY_1] reported on [DATE_1]."
    assert document.mapping.restore(document.text) == original


def test_restore_leaves_an_unknown_placeholder_shaped_token_alone() -> None:
    """A token this mapping never emitted is not this mapping's to interpret."""
    document = anonymize("AdaptHealth Corp.", [company("AdaptHealth Corp.")])
    assert document.mapping.restore("[COMPANY_1] and [COMPANY_9]") == (
        "AdaptHealth Corp. and [COMPANY_9]"
    )


def test_restore_on_an_empty_mapping_is_the_identity() -> None:
    assert EntityMapping(entries=()).restore("nothing to restore") == "nothing to restore"


def test_canonical_for_returns_none_for_an_unknown_placeholder() -> None:
    document = anonymize("AdaptHealth Corp.", [company("AdaptHealth Corp.")])
    assert document.mapping.canonical_for("[COMPANY_1]") == "AdaptHealth Corp."
    assert document.mapping.canonical_for("[COMPANY_42]") is None


def test_a_pre_existing_placeholder_is_surfaced_rather_than_silently_tolerated() -> None:
    """It makes restore ambiguous for that token, so the caller is told."""
    document = anonymize("see [COMPANY_1] and AdaptHealth Corp.", [company("AdaptHealth Corp.")])
    assert document.pre_existing_placeholders == ("[COMPANY_1]",)


def test_counts_by_kind_counts_spans_not_characters() -> None:
    document = anonymize(
        "AdaptHealth Corp. and AdaptHealth Corp. on 2024-03-11",
        [company("AdaptHealth Corp.")],
    )
    assert document.counts_by_kind() == {MaskKind.COMPANY: 2, MaskKind.DATE: 1}


def test_placeholders_are_listed_in_order_of_first_appearance() -> None:
    document = anonymize(
        "2024-03-11: AdaptHealth Corp. and 2024-03-11 again",
        [company("AdaptHealth Corp.")],
    )
    assert document.placeholders() == ("[DATE_1]", "[COMPANY_1]")


def test_surface_forms_record_every_writing_a_placeholder_stood_for() -> None:
    """The over-masking measurement (§6.5) reads this."""
    raw = read(ADAPTHEALTH_FORM4)
    document = anonymize(raw, adapthealth_entities())
    entry = document.mapping.by_placeholder["[COMPANY_1]"]
    assert "AdaptHealth Corp." in entry.surface_forms
    assert "DFB Healthcare Acquisitions Corp." in entry.surface_forms
    assert entry.occurrences >= len(entry.surface_forms)


def test_the_same_entity_keeps_its_placeholder_across_documents() -> None:
    """Declaration order fixes the number, so cached and re-run scores compare.

    A placeholder numbered by order of *appearance* would give one company two
    different names in two documents, and P7.9's paired comparison would be
    measuring the numbering.
    """
    entities = [company("AdaptHealth Corp."), person("James E. Flynn")]
    first = anonymize("AdaptHealth Corp. hired James E. Flynn, we hear", entities)
    second = anonymize("James E. Flynn left AdaptHealth Corp., we hear", entities)
    assert first.text == "[COMPANY_1] hired [PERSON_1], we hear"
    assert second.text == "[PERSON_1] left [COMPANY_1], we hear"


def test_a_trailing_abbreviation_absorbs_the_sentence_full_stop() -> None:
    """``... left AdaptHealth Corp.`` loses the sentence's period. Recorded, not fixed.

    The declared writing ends in ``Corp.`` and the sentence ends in the same
    character, so the match takes both. Telling them apart needs a sentence
    model, and the cost of getting it wrong the other way — leaving ``Corp.``
    standing beside the placeholder — is a name fragment in the payload. This
    direction costs one punctuation mark.
    """
    document = anonymize("It left AdaptHealth Corp.", [company("AdaptHealth Corp.")])
    assert document.text == "It left [COMPANY_1]"


def test_merge_writings_unions_declared_writings_in_order() -> None:
    entities = [company("SAP SE", "SAP AG"), identifier("0001000184")]
    assert merge_writings(entities) == ("SAP SE", "SAP AG", "0001000184")
