"""P7.2: property tests for the anonymization pass (§8).

The example-based tests check the cases someone thought of. These check the
statements the rest of Phase 7 is entitled to rely on, over inputs nobody chose:

* **Determinism.** The same text, entities and config give byte-identical
  output. P7.9's contamination probe scores one document twice and compares; a
  masker that drifted would make that comparison measure itself.
* **Faithfulness of the audit trail.** Every recorded replacement really is the
  span it says it is, spans do not overlap, and splicing them back into the
  original reproduces the payload exactly. §6.5's inspector and every
  over-masking measurement read those offsets.
* **Removal.** A declared name dropped anywhere into arbitrary surrounding text
  does not survive, and the independently-written detector agrees.
* **Reversibility.** Restoring the payload recovers the canonical writings.
* **Stability.** One entity gets one placeholder; two entities never share one;
  and the number does not depend on which document is being masked.
* **Idempotence.** Masking already-masked text changes nothing — a placeholder
  is never itself mistaken for something to mask.

The generated text is filler, not filings: it is drawn from a small alphabet and
carries no claim to be a document. What it stands in for is "whatever else was
on the page around the name", which is exactly the part the rules must not
depend on. The real filing text lives in the example-based tests, from the
captured fixtures.
"""

from __future__ import annotations

import re

from hypothesis import HealthCheck, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.extraction.anonymize import AnonymizerConfig, anonymize
from backend.extraction.entities import Entity, company, identifier, person, ticker
from backend.extraction.leak import LeakKind, detect_leaks
from backend.extraction.rules import PLACEHOLDER_SHAPE, MaskKind

# Filler characters only: letters, spaces, newlines and a little punctuation.
# Digits are excluded so the filler cannot accidentally spell a date and turn a
# statement about names into a statement about the temporal rules.
_FILLER = st.text(alphabet=" \n\t.,;:()abcdefghijklmnopqrstuvwxyz", max_size=60)

# A narrower alphabet for the one property that asserts text comes back
# *unchanged*. The wide filler above can spell things that are genuinely
# maskable — Hypothesis found "mar" — and a test asserting "nothing changed"
# would then be asserting that the temporal rules do not work. These letters
# cannot spell any month name or abbreviation, nor any relative-period phrase.
_INERT_FILLER = st.text(alphabet=" \n.,;:()abfgklqvwxz", max_size=60)

# Declared names, built from tokens that are not English words, so the filler
# cannot collide with them by chance. Two tokens minimum: a one-token name would
# exercise a different (and separately tested) path.
_NAME_TOKEN = st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=4, max_size=8).map(
    lambda token: token.title()
)
_COMPANY_NAME = st.builds(lambda a, b: f"{a} {b}", _NAME_TOKEN, _NAME_TOKEN)

_SETTINGS = hypothesis_settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(text=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_masking_is_deterministic(text: str, name: str) -> None:
    entities = [company(name)]
    first = anonymize(text, entities)
    second = anonymize(text, entities)
    assert first.text == second.text
    assert first.replacements == second.replacements
    assert first.mapping.entries == second.mapping.entries


@given(text=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_replacements_are_a_faithful_non_overlapping_record(text: str, name: str) -> None:
    """Each span is what it claims, spans advance, and they rebuild the payload."""
    source = f"{text} {name} {text}"
    document = anonymize(source, [company(name)])

    pieces: list[str] = []
    cursor = 0
    for replacement in document.replacements:
        assert source[replacement.start : replacement.end] == replacement.original
        assert replacement.start >= cursor
        assert replacement.end > replacement.start
        pieces.append(source[cursor : replacement.start])
        pieces.append(replacement.placeholder)
        cursor = replacement.end
    pieces.append(source[cursor:])
    assert "".join(pieces) == document.text


@given(before=_FILLER, after=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_a_declared_name_does_not_survive_whatever_surrounds_it(
    before: str, after: str, name: str
) -> None:
    """Removal must not depend on the rest of the page.

    The name is separated from the filler by spaces: a name glued to a
    neighbouring word is a *different* string, and whether that should be masked
    is the separate, deliberately-answered question in
    ``test_a_spaced_variant_of_a_closed_compound_name_is_a_known_miss``.
    """
    source = f"{before} {name} {after}"
    entities = [company(name)]
    document = anonymize(source, entities)

    bounded = re.compile(r"(?<![0-9A-Za-z])" + re.escape(name) + r"(?![0-9A-Za-z])")
    assert bounded.search(document.text) is None
    report = detect_leaks(original=source, anonymized=document.text, entities=entities)
    assert not [f for f in report.leaks if f.kind is LeakKind.COMPANY_NAME]
    assert report.occurrences_in_original[name] >= 1


@given(before=_FILLER, after=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_a_name_split_across_a_line_break_does_not_survive_either(
    before: str, after: str, name: str
) -> None:
    """Filing text wraps. A newline inside a name does not make it two names."""
    first_token, second_token = name.split(" ")
    wrapped = f"{first_token}\n   {second_token}"
    source = f"{before} {wrapped} {after}"
    document = anonymize(source, [company(name)])
    assert wrapped not in document.text
    assert "[COMPANY_1]" in document.text


@given(text=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_restoring_the_payload_recovers_the_declared_writing(text: str, name: str) -> None:
    source = f"{text} {name}"
    document = anonymize(source, [company(name)])
    restored = document.mapping.restore(document.text)
    assert name in restored
    assert "[COMPANY_1]" not in restored


@given(text=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_masking_is_idempotent(text: str, name: str) -> None:
    """A placeholder is never itself something to mask.

    Chunking (P7.3) and caching both re-handle already-masked text, and a second
    pass that renumbered or re-redacted would silently break the cache key.
    """
    entities = [company(name)]
    once = anonymize(f"{text} {name} {text}", entities)
    twice = anonymize(once.text, entities)
    assert twice.text == once.text
    assert twice.replacements == ()


@given(
    tokens=st.lists(_NAME_TOKEN, min_size=4, max_size=8, unique=True),
    text=_FILLER,
)
@_SETTINGS
def test_entities_with_no_shared_tokens_never_share_a_placeholder(
    tokens: list[str], text: str
) -> None:
    """Names built from disjoint tokens keep disjoint placeholders.

    The disjointness is required, not incidental. Two declared names that share
    a token — ``Deerfield Management`` and ``Deerfield Partners`` — are
    deliberately allowed to collapse onto the first one's placeholder by the
    leading-token rule, which trades precision for recall on purpose. That case
    is covered by ``test_abbreviated_writings_are_caught_without_the_leading_token_rule``
    on real filing text; what is asserted here is the part that must hold
    unconditionally once the names are genuinely distinct.
    """
    names = [f"{a} {b}" for a, b in zip(tokens[::2], tokens[1::2], strict=False)]
    entities = [company(name) for name in names]
    source = text + " " + " ".join(names)
    document = anonymize(source, entities)

    by_placeholder: dict[str, set[str]] = {}
    for replacement in document.replacements:
        if replacement.kind is MaskKind.COMPANY:
            by_placeholder.setdefault(replacement.placeholder, set()).add(replacement.original)
    assert len(by_placeholder) == len(names)
    for originals in by_placeholder.values():
        assert len(originals) == 1


@given(names=st.lists(_COMPANY_NAME, min_size=1, max_size=4, unique=True), text=_FILLER)
@_SETTINGS
def test_a_placeholder_is_never_reused_for_two_entities(names: list[str], text: str) -> None:
    """Holds even when declared names overlap, which the previous test excludes.

    Overlapping names may collapse onto *fewer* placeholders — that is the
    leading-token rule's stated trade — but never onto a shared one, because the
    number comes from declaration order rather than from what matched.
    """
    entities = [company(name) for name in names]
    document = anonymize(text + " " + " ".join(names), entities)

    owners: dict[str, set[str]] = {}
    for replacement in document.replacements:
        if replacement.kind is MaskKind.COMPANY:
            owners.setdefault(replacement.placeholder, set()).add(replacement.rule)
    assert len(owners) <= len(names)
    for placeholder, rules in owners.items():
        index = int(placeholder.removeprefix("[COMPANY_").removesuffix("]"))
        assert 1 <= index <= len(names)
        canonical = document.mapping.canonical_for(placeholder)
        assert canonical == names[index - 1]
        assert rules


@given(first_text=_FILLER, second_text=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_an_entity_keeps_its_placeholder_across_documents(
    first_text: str, second_text: str, name: str
) -> None:
    """Numbering comes from declaration order, never from order of appearance."""
    entities = [company("Zzzq Placeholderco"), company(name)]
    first = anonymize(f"{first_text} {name}", entities)
    second = anonymize(f"{second_text} {name} {second_text}", entities)

    used = {r.placeholder for r in first.replacements if r.kind is MaskKind.COMPANY}
    also_used = {r.placeholder for r in second.replacements if r.kind is MaskKind.COMPANY}
    assert used == also_used == {"[COMPANY_2]"}


@given(text=_FILLER)
@_SETTINGS
def test_every_placeholder_in_the_payload_is_reversible(text: str) -> None:
    """No placeholder may appear that the mapping cannot explain."""
    source = f"{text} Zzzq Placeholderco filed on 2024-03-11 for Q1 2024"
    document = anonymize(source, [company("Zzzq Placeholderco")])
    index = document.mapping.by_placeholder
    for placeholder in PLACEHOLDER_SHAPE.findall(document.text):
        assert placeholder in index


@given(text=_INERT_FILLER)
@_SETTINGS
def test_text_with_nothing_to_mask_is_returned_unchanged(text: str) -> None:
    """Over-masking is the safe direction, but not at any price."""
    document = anonymize(text, [])
    assert document.text == text
    assert document.replacements == ()


@given(
    year=st.integers(min_value=1900, max_value=2099),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
)
@_SETTINGS
def test_a_calendar_day_is_removed_in_every_writing_of_it(year: int, month: int, day: int) -> None:
    """The guarantee that has to hold for every date, not the sampled ones."""
    writings = [
        f"{year:04d}-{month:02d}-{day:02d}",
        f"{year:04d}{month:02d}{day:02d}",
        f"{month}/{day}/{year}",
        f"{day}.{month}.{year}",
    ]
    for writing in writings:
        masked = anonymize(f"as of {writing} the").text
        assert str(year) not in masked, writing
        assert masked.startswith("as of ["), writing


@given(
    year=st.integers(min_value=1900, max_value=2099),
    month=st.integers(min_value=1, max_value=12),
    day=st.integers(min_value=1, max_value=28),
)
@_SETTINGS
def test_two_writings_of_one_iso_day_share_a_placeholder(year: int, month: int, day: int) -> None:
    """Coherence: the reader must be able to see that two mentions are one day."""
    iso = f"{year:04d}-{month:02d}-{day:02d}"
    compact = f"{year:04d}{month:02d}{day:02d}"
    document = anonymize(f"{iso} and {compact}")
    assert document.text == "[DATE_1] and [DATE_1]"


@given(text=_FILLER, symbol=st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=3, max_size=5))
@_SETTINGS
def test_a_declared_ticker_does_not_survive(text: str, symbol: str) -> None:
    source = f"{text} ({symbol}) {text}"
    document = anonymize(source, [ticker(symbol)])
    assert re.search(r"(?<![0-9A-Za-z])" + symbol + r"(?![0-9A-Za-z])", document.text) is None


@given(digits=st.integers(min_value=100_000, max_value=9_999_999_999))
@_SETTINGS
def test_a_declared_cik_does_not_survive_in_either_padding(digits: int) -> None:
    """A CIK is zero-padded in a header and unpadded in a URL; both must go.

    The floor keeps the generated number longer than the digits inside a
    placeholder: a one-digit "CIK" would be "found" in ``[ID_1]`` by the check
    below, which would be the test failing rather than the masker.
    """
    padded = f"{digits:010d}"
    unpadded = str(digits)
    document = anonymize(f"CIK {padded} also {unpadded}", [identifier(padded)])
    assert padded not in document.text
    assert re.search(r"(?<![0-9A-Za-z])" + unpadded + r"(?![0-9A-Za-z])", document.text) is None


@given(text=_FILLER, surname=_NAME_TOKEN, given_name=_NAME_TOKEN)
@_SETTINGS
def test_a_declared_person_does_not_survive_in_any_writing_of_the_name(
    text: str, surname: str, given_name: str
) -> None:
    """EDGAR writes an individual surname-first; prose writes them the other way.

    The bare surname is included on purpose: it is the writing most of a filing
    actually uses after the first mention, and it is the one an article in front
    of it used to hide (see the note in :mod:`backend.extraction.surface`).
    """
    entities: list[Entity] = [person(f"{given_name} {surname}")]
    writings = (
        f"{given_name} {surname}",
        f"{surname}, {given_name}",
        f"{given_name[0]}. {surname}",
        f"Mr. {surname}",
        f"the {surname}",
        surname,
    )
    for writing in writings:
        document = anonymize(f"{text} {writing} {text}", entities)
        assert surname not in document.text, writing


@given(text=_FILLER, name=_COMPANY_NAME)
@_SETTINGS
def test_switching_every_rule_off_leaves_the_text_alone(text: str, name: str) -> None:
    """The config switches are real, which is what makes the leak tests honest."""
    source = f"{text} {name} on 2024-03-11 in Q1 2024"
    permissive = AnonymizerConfig(
        mask_leading_token=False,
        mask_surname_alone=False,
        mask_dates=False,
        mask_periods=False,
        mask_bare_years=False,
        mask_bare_months=False,
        mask_relative_periods=False,
        mask_times=False,
        mask_accession_numbers=False,
        mask_tickers=False,
        mask_identifiers=False,
    )
    document = anonymize(source, [], permissive)
    assert document.text == source
