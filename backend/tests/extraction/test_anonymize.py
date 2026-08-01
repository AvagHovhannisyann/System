"""Regression tests for placeholder opacity (P7.2).

Masking must be idempotent. These cover a defect Hypothesis found after the
package landed: emitted placeholders were themselves maskable text, so a second
pass corrupted them.
"""

from __future__ import annotations

from backend.extraction.anonymize import anonymize
from backend.extraction.entities import company


def test_a_placeholder_is_opaque_to_a_second_masking_pass() -> None:
    """Masking twice leaves placeholders byte-identical.

    An entity whose name repeats a word the placeholder itself uses re-matched
    *inside* the emitted placeholder: ``Company Company`` masked to
    ``[COMPANY_1]``, whose interior matched again and became
    ``[[COMPANY_1]_1]``. That corrupts the token the model reads and breaks the
    reverse mapping, so the leak detector could no longer tie the placeholder
    back to its entity — a silent failure in the component whose whole purpose
    is making masking checkable.
    """
    entities = [company("Company Company")]
    once = anonymize("Company Company reported.", entities)
    twice = anonymize(once.text, entities)
    assert once.text == twice.text
    assert "[[" not in twice.text


def test_text_arriving_with_placeholders_from_an_earlier_stage_is_untouched() -> None:
    """Partly-masked input keeps existing tokens while new entities still mask.

    The chunker may hand over text that already carries placeholders. Those must
    survive verbatim, and masking must still do its job on everything else —
    protecting placeholders must not become an excuse to stop masking.
    """
    masked = anonymize("[COMPANY_7] and [DATE_2] and Acme Corp", [company("Acme Corp")])
    assert "[COMPANY_7]" in masked.text
    assert "[DATE_2]" in masked.text
    assert "Acme Corp" not in masked.text


def test_one_entity_cannot_swallow_the_start_of_another() -> None:
    """A declared entity must never go unmasked because a neighbour absorbed it.

    Found by Hypothesis. Company patterns carry an *optional legal tail*, and
    legal forms include ordinary words like ``Company``, so a declaration of
    ``Aaab Aaaaa`` also matches ``Aaab Aaaaa Company`` — consuming the token
    that begins a second declared entity ``Company Aaaa``, which was then never
    masked at all.

    That is the worst failure this package can have and the hardest to notice:
    the output *looks* masked, and the leak detector's own idea of what should
    have been masked comes from the same arbitration, so it agrees. The entity
    simply leaves the building with the document.
    """
    masked = anonymize(
        "Aaab Aaaaa Company Aaaa",
        [company("Aaab Aaaaa"), company("Company Aaaa")],
    )
    assert masked.text == "[COMPANY_1] [COMPANY_2]"


def test_a_shorter_competing_description_of_the_same_span_still_loses() -> None:
    """Coverage must not defeat leftmost-longest where the short match is contained.

    ``March`` inside ``March 31, 2024`` is a competing description of one span,
    not a second entity, so the long match must still win. Without this
    distinction the coverage pass would dismantle every legitimate long match to
    place the short alternative.
    """
    masked = anonymize("filed March 31, 2024 by Acme Corp", [company("Acme Corp")])
    assert "[DATE_1]" in masked.text
    assert "March" not in masked.text
    assert "31" not in masked.text
