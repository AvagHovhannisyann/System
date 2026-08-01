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
