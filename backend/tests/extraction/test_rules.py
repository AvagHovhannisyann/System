"""P7.2: the overlap policy, tested on its own.

:func:`~backend.extraction.rules.apply_rules` knows nothing about companies or
dates — it takes patterns and arbitrates between their matches. That separation
is worth having only if the arbitration is checked here rather than inferred
from masking results, because the failure it prevents is invisible downstream: a
rule rewriting text that another rule was about to match produces a plausible
document with a name half-redacted inside it.

The policy is leftmost, then longest, then rule priority. Every test below fixes
one clause of that sentence.
"""

from __future__ import annotations

import re

import pytest

from backend.extraction.rules import (
    PLACEHOLDER_SHAPE,
    MaskKind,
    MaskRule,
    apply_rules,
)


def _rule(
    name: str,
    body: str,
    *,
    priority: int = 1_000,
    placeholder: str | None = None,
    canonical: str | None = None,
    kind: MaskKind = MaskKind.COMPANY,
) -> MaskRule:
    """Build a simple rule for these tests."""
    return MaskRule(
        name=name,
        kind=kind,
        pattern=re.compile(body),
        priority=priority,
        placeholder=placeholder,
        canonical=canonical,
    )


def test_the_longest_match_at_a_position_wins() -> None:
    """``quarter ended March 31, 2024`` masks whole, not as a date beside a word."""
    rules = [
        _rule("short", r"March", kind=MaskKind.DATE, placeholder="[SHORT]"),
        _rule("long", r"March 31, 2024", kind=MaskKind.DATE, placeholder="[LONG]"),
    ]
    assert apply_rules("on March 31, 2024 we", rules).text == "on [LONG] we"


def test_the_leftmost_match_wins_when_two_overlap() -> None:
    """A later rule cannot claim a span that starts inside an earlier match."""
    rules = [
        _rule("left", r"abc", placeholder="[L]"),
        _rule("right", r"bcd", placeholder="[R]"),
    ]
    assert apply_rules("abcd", rules).text == "[L]d"


def test_priority_breaks_a_tie_between_equal_spans() -> None:
    """Same start, same length: the lower priority number wins, deterministically."""
    rules = [
        _rule("loser", r"Deerfield", priority=200, placeholder="[LOSER]"),
        _rule("winner", r"Deerfield", priority=100, placeholder="[WINNER]"),
    ]
    assert apply_rules("Deerfield", rules).text == "[WINNER]"
    # Rule order in the sequence must not matter — only priority.
    assert apply_rules("Deerfield", list(reversed(rules))).text == "[WINNER]"


def test_a_non_overlapping_later_match_is_still_applied() -> None:
    rules = [_rule("a", r"one", placeholder="[A]"), _rule("b", r"two", placeholder="[B]")]
    assert apply_rules("one and two", rules).text == "[A] and [B]"


def test_an_allocated_placeholder_is_numbered_by_first_appearance() -> None:
    """Not by value, not by chronology — by the order the document mentions it."""
    rules = [_rule("word", r"\b[a-z]+\b", kind=MaskKind.DATE)]
    application = apply_rules("beta alpha beta", rules)
    assert application.text == "[DATE_1] [DATE_2] [DATE_1]"


def test_equal_keys_share_a_placeholder_across_different_writings() -> None:
    """``key_fn`` is what makes two spellings of one date one placeholder."""
    rule = MaskRule(
        name="upper_or_lower",
        kind=MaskKind.DATE,
        pattern=re.compile(r"[A-Za-z]+"),
        key_fn=lambda matched: matched.casefold(),
    )
    assert apply_rules("Alpha alpha ALPHA beta", [rule]).text == (
        "[DATE_1] [DATE_1] [DATE_1] [DATE_2]"
    )


def test_the_default_key_folds_case_and_collapses_whitespace() -> None:
    rule = MaskRule(
        name="two_words", kind=MaskKind.DATE, pattern=re.compile(r"[A-Za-z]+\s+[A-Za-z]+")
    )
    assert apply_rules("Alpha  Beta;alpha beta", [rule]).text == "[DATE_1];[DATE_1]"


def test_an_accept_predicate_rejects_a_match_before_it_becomes_a_candidate() -> None:
    """The eight-digit rule needs this: a film number is not a date."""
    rule = MaskRule(
        name="only_even",
        kind=MaskKind.DATE,
        pattern=re.compile(r"\d+"),
        accept=lambda match: int(match.group(0)) % 2 == 0,
    )
    assert apply_rules("1 2 3 4", [rule]).text == "1 [DATE_1] 3 [DATE_2]"


def test_a_rejected_match_does_not_block_a_shorter_accepted_one() -> None:
    """Rejection happens before arbitration, so the span is genuinely free."""
    rules = [
        MaskRule(
            name="rejecting",
            kind=MaskKind.DATE,
            pattern=re.compile(r"abcd"),
            priority=1,
            accept=lambda _match: False,
            placeholder="[NO]",
        ),
        _rule("accepting", r"abc", priority=2, placeholder="[YES]"),
    ]
    assert apply_rules("abcd", rules).text == "[YES]d"


def test_a_zero_width_match_is_ignored() -> None:
    """A pattern that can match nothing must not emit an infinity of placeholders."""
    rule = _rule("empty", r"x*", placeholder="[E]")
    assert apply_rules("yyy", [rule]).text == "yyy"


def test_a_fixed_placeholder_records_its_canonical_not_the_matched_text() -> None:
    rules = [_rule("named", r"Deerfield\w*", placeholder="[COMPANY_1]", canonical="Deerfield L.P.")]
    application = apply_rules("Deerfields and Deerfield", rules)
    assert application.text == "[COMPANY_1] and [COMPANY_1]"
    (entry,) = application.allocated
    assert entry.canonical == "Deerfield L.P."
    assert entry.surface_forms == ("Deerfields", "Deerfield")
    assert entry.occurrences == 2


def test_replacements_reconstruct_the_output_exactly() -> None:
    """Offsets are into the original and are the operator inspector's contract."""
    text = "Deerfield met Flynn on 2024-03-11"
    rules = [
        _rule("company", r"Deerfield", placeholder="[COMPANY_1]"),
        _rule("person", r"Flynn", kind=MaskKind.PERSON, placeholder="[PERSON_1]"),
        _rule("date", r"\d{4}-\d{2}-\d{2}", kind=MaskKind.DATE, placeholder="[DATE_1]"),
    ]
    application = apply_rules(text, rules)

    pieces: list[str] = []
    cursor = 0
    for replacement in application.replacements:
        assert text[replacement.start : replacement.end] == replacement.original
        assert replacement.start >= cursor, "spans must not overlap"
        pieces.append(text[cursor : replacement.start])
        pieces.append(replacement.placeholder)
        cursor = replacement.end
    pieces.append(text[cursor:])
    assert "".join(pieces) == application.text


def test_applying_no_rules_returns_the_text_unchanged() -> None:
    application = apply_rules("untouched", [])
    assert application.text == "untouched"
    assert application.replacements == ()
    assert application.allocated == ()


def test_application_is_a_pure_function_of_text_and_rules() -> None:
    """P7.9 compares two scorings of one document; drift would measure itself."""
    text = "alpha 2024-03-11 beta 2024-03-11 alpha"
    rules = [
        _rule("word", r"al\w+", placeholder="[COMPANY_1]"),
        _rule("date", r"\d{4}-\d{2}-\d{2}", kind=MaskKind.DATE),
    ]
    first = apply_rules(text, rules)
    second = apply_rules(text, rules)
    assert first == second


@pytest.mark.parametrize(
    "candidate",
    ["[COMPANY_1]", "[DATE_12]", "[ID_3]", "[RELATIVE_PERIOD_1]", "[PERSON]"],
)
def test_placeholder_shape_matches_what_the_module_emits(candidate: str) -> None:
    assert PLACEHOLDER_SHAPE.fullmatch(candidate) is not None


@pytest.mark.parametrize("candidate", ["[lowercase_1]", "[Mixed_1]", "COMPANY_1", "[]"])
def test_placeholder_shape_rejects_other_bracketed_text(candidate: str) -> None:
    assert PLACEHOLDER_SHAPE.fullmatch(candidate) is None
