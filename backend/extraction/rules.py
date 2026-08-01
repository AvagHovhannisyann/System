"""Mask rules and the deterministic rule-application pass (P7.2).

A *rule* is a compiled regular expression plus the placeholder policy for what
it matches. :func:`apply_rules` runs every rule over one document, resolves the
overlaps, and rewrites the text. Nothing in this module knows what a company or
a date is; that lives in :mod:`backend.extraction.surface` and
:mod:`backend.extraction.temporal`. Keeping the arbitration here means the
overlap policy is stated once and testable on its own.

Overlap policy — leftmost, then longest, then declaration order
---------------------------------------------------------------

Rules are matched independently and then arbitrated, because the alternative
(rewriting the text rule by rule) makes the result depend on rule order in ways
that are invisible until a placeholder lands inside another rule's match. Every
match from every rule becomes a candidate span; candidates are sorted by start
ascending, then by span length descending, then by rule priority ascending; a
single left-to-right sweep keeps a candidate when it starts at or after the end
of the last one kept.

Leftmost-then-longest is what makes ``quarter ended March 31, 2024`` mask as one
period rather than leaving ``quarter ended`` beside a masked date, and what
makes the declared name ``Deerfield Partners, L.P.`` win over the bare leading
token ``Deerfield`` of a *different* declared entity that starts at the same
offset. Ties beyond that are broken by priority, which is assigned at rule
construction from declaration order, so the output is a pure function of
(text, entities, config) — a requirement, not a nicety, because the
contamination probe in P7.9 compares two scorings of the same document and a
non-deterministic masker would make that comparison meaningless.

Placeholders
------------

A rule either carries a fixed ``placeholder`` (declared entities: the number
comes from declaration order, so the same entity keeps the same placeholder in
every document it appears in) or leaves it to the allocator (temporal
expressions: the number comes from order of first appearance within the
document, and equal *values* share a number even when written differently —
``2024-03-11`` and ``March 11, 2024`` both become ``[DATE_1]``).

Units: all offsets are character indices into the original ``str``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

__all__ = [
    "LEFT_BOUNDARY",
    "PLACEHOLDER_SHAPE",
    "RIGHT_BOUNDARY",
    "AllocatedPlaceholder",
    "MaskKind",
    "MaskRule",
    "Replacement",
    "RuleApplication",
    "apply_rules",
]


class MaskKind(StrEnum):
    """What a placeholder stands for.

    The value is embedded verbatim in the placeholder text, so these names are
    part of what the extraction model sees: they must describe the *category*
    and never the entity.
    """

    COMPANY = "COMPANY"
    PERSON = "PERSON"
    TICKER = "TICKER"
    IDENTIFIER = "ID"
    ACCESSION = "ACCESSION"
    DATE = "DATE"
    PERIOD = "PERIOD"
    YEAR = "YEAR"
    MONTH = "MONTH"
    TIME = "TIME"
    RELATIVE_PERIOD = "RELATIVE_PERIOD"


LEFT_BOUNDARY: Final = r"(?<![0-9A-Za-z])"
"""Left word boundary that treats digits and letters as word characters.

``\\b`` is unusable here: a company name may legitimately end in ``.`` (``Corp.``)
and a date may end in a digit, so ``\\b``'s definition flips depending on the
last character of the pattern. These explicit lookarounds do not.
"""

RIGHT_BOUNDARY: Final = r"(?![0-9A-Za-z])"
"""Right word boundary; see :data:`LEFT_BOUNDARY`.

Deliberately excludes the apostrophe, so ``AdaptHealth's`` masks to
``[COMPANY_1]'s`` — the possessive clitic is not identifying and keeping it
preserves the sentence.
"""

_MAX_COVERAGE_PASSES: Final = 4
"""Bound on shadow-resolution sweeps.

A rule that still cannot be placed after this many passes keeps the ordinary
leftmost-longest result rather than looping.
"""

PLACEHOLDER_SHAPE: Final = re.compile(r"\[[A-Z][A-Z_]*(?:_\d+)?\]")
"""Shape of a placeholder this module emits, used to detect prior collisions."""


@dataclass(frozen=True, slots=True)
class MaskRule:
    """One masking rule: a pattern, what it stands for, and how it is numbered.

    Attributes:
        name: Stable identifier for audit and for the document inspector
            (§6.5). Appears in every :class:`Replacement` the rule produces.
        kind: Placeholder category.
        pattern: Compiled pattern. Its whole match (group 0) is replaced.
        priority: Tie-break for equal start and equal length; lower wins.
        placeholder: Fixed placeholder text, or ``None`` to let the allocator
            number the match by order of first appearance.
        canonical: Value recorded in the reversible mapping for a fixed
            placeholder. Ignored when ``placeholder`` is ``None``.
        key_fn: Collapses a matched string to the key that decides whether two
            matches share a placeholder. ``None`` means "the matched text,
            case-folded and whitespace-collapsed".
        accept: Optional predicate re-checking a match against context the
            pattern cannot express (for example: eight digits are a date only
            if they parse as one). A rejected match is not a candidate.
    """

    name: str
    kind: MaskKind
    pattern: re.Pattern[str]
    priority: int = 1_000
    placeholder: str | None = None
    canonical: str | None = None
    key_fn: Callable[[str], str] | None = None
    accept: Callable[[re.Match[str]], bool] | None = None


@dataclass(frozen=True, slots=True)
class Replacement:
    """One span of the original document that was replaced.

    Attributes:
        start: Character offset of the span's first character in the original.
        end: Character offset one past the span's last character.
        original: The exact text that was replaced.
        placeholder: The text written in its place.
        kind: Placeholder category.
        rule: :attr:`MaskRule.name` of the rule that matched.
    """

    start: int
    end: int
    original: str
    placeholder: str
    kind: MaskKind
    rule: str


@dataclass(frozen=True, slots=True)
class AllocatedPlaceholder:
    """A placeholder and everything needed to reverse it.

    Attributes:
        placeholder: The placeholder text, e.g. ``[COMPANY_1]``.
        kind: Placeholder category.
        canonical: The value the placeholder stands for — the declared entity
            name, or for an allocated placeholder the first surface form seen.
        surface_forms: Every distinct string this placeholder replaced, in
            order of first appearance.
        occurrences: How many spans this placeholder replaced (count).
    """

    placeholder: str
    kind: MaskKind
    canonical: str
    surface_forms: tuple[str, ...]
    occurrences: int


@dataclass(frozen=True, slots=True)
class RuleApplication:
    """Result of running a rule set over one document.

    Attributes:
        text: The rewritten document.
        replacements: Every span replaced, in document order.
        allocated: One entry per distinct placeholder used, in the order the
            placeholders were first emitted.
    """

    text: str
    replacements: tuple[Replacement, ...]
    allocated: tuple[AllocatedPlaceholder, ...]


_WHITESPACE = re.compile(r"\s+")


def _default_key(matched: str) -> str:
    """Case-fold and collapse whitespace so equal writings share a placeholder."""
    return _WHITESPACE.sub(" ", matched).strip().casefold()


@dataclass(slots=True)
class _Bucket:
    """Mutable accumulator behind one :class:`AllocatedPlaceholder`."""

    placeholder: str
    kind: MaskKind
    canonical: str
    surface_forms: list[str]
    occurrences: int


def _candidates(text: str, rules: Sequence[MaskRule]) -> list[tuple[int, int, MaskRule]]:
    """Collect every accepted match of every rule as a (start, end, rule) span."""
    found: list[tuple[int, int, MaskRule]] = []
    for rule in rules:
        for match in rule.pattern.finditer(text):
            if match.start() == match.end():
                continue
            if rule.accept is not None and not rule.accept(match):
                continue
            found.append((match.start(), match.end(), rule))
    return found


def _coverage_key(rule: MaskRule) -> object:
    """Return what a rule *covers*, for the shadow check in selection.

    Declared entities carry a fixed placeholder and contribute several
    alternative surface rules (verbatim, core tokens, leading token) — all of
    them cover the same entity, and exactly one is expected to win. Allocator
    rules have no fixed placeholder, so each stands for itself.
    """
    return rule.placeholder if rule.placeholder is not None else id(rule)


def _placeholder_spans(text: str) -> list[tuple[int, int]]:
    """Return the spans of every already-emitted placeholder in ``text``.

    Masking must be **idempotent**: running it twice, or running it over text
    that already carries placeholders from an earlier stage, must leave those
    placeholders exactly as they are. Without this, an entity whose name
    contains a word the placeholder itself uses re-matches inside the
    placeholder — an entity named ``Company Company`` masks to
    ``[COMPANY_1]``, whose interior then matches again and becomes
    ``[[COMPANY_1]_1]``. That corrupts the token the model is meant to read,
    and it corrupts the reverse mapping, so the leak detector can no longer
    tie a placeholder back to its entity.
    """
    return [match.span() for match in PLACEHOLDER_SHAPE.finditer(text)]


def _select(
    candidates: Iterable[tuple[int, int, MaskRule]],
    protected: Sequence[tuple[int, int]] = (),
) -> list[tuple[int, int, MaskRule]]:
    """Resolve overlaps: leftmost, then longest, then lowest rule priority.

    Candidates intersecting a ``protected`` span (an existing placeholder) are
    dropped, so placeholders are opaque to further masking.
    """
    ordered = sorted(candidates, key=lambda c: (c[0], -(c[1] - c[0]), c[2].priority, c[2].name))
    kept: list[tuple[int, int, MaskRule]] = []
    cursor = 0
    for start, end, rule in ordered:
        if start < cursor:
            continue
        # Any overlap at all disqualifies the candidate — a match that merely
        # clips a placeholder's edge would still split the token.
        if any(
            start < guarded_end and guarded_start < end for guarded_start, guarded_end in protected
        ):
            continue
        kept.append((start, end, rule))
        cursor = end
    return kept


def _select_covering_every_entity(
    candidates: Iterable[tuple[int, int, MaskRule]],
    protected: Sequence[tuple[int, int]] = (),
) -> list[tuple[int, int, MaskRule]]:
    """Select spans, refusing to let one rule's match swallow another rule's only match.

    Leftmost-longest alone can silently drop a declared entity. Company patterns
    carry an *optional legal tail*, and legal forms include ordinary words like
    ``Company``, so a declaration of ``Aaab Aaaaa`` matches ``Aaab Aaaaa
    Company`` — consuming the token that begins a second declared entity
    ``Company Aaaa``, which is then never masked at all. An unmasked declared
    entity is precisely the failure this package exists to prevent, and it is
    invisible: the output looks masked, and the leak detector's own view of what
    should have been masked came from the same arbitration.

    So: run the ordinary sweep, then look for rules that had candidates but won
    nothing. For each, drop the *longer* rival spans that shadowed it — only
    those, and only from other rules — and sweep again. Iteration is bounded;
    if a rule still cannot be placed, the ordinary result stands rather than
    looping. Shrinking a rival's span can never manufacture a match that the
    text does not contain, so this cannot over-mask.
    """
    pool = list(candidates)
    for _ in range(_MAX_COVERAGE_PASSES):
        selected = _select(pool, protected)
        # Coverage is per *entity*, not per rule. One entity contributes several
        # alternative surface rules (verbatim, core tokens, leading token); they
        # are alternatives, and exactly one is expected to win. Counting them
        # individually would treat the losing variants as shadowed and keep
        # dismantling the winner until only the shortest variant survived —
        # masking "Aaab" and leaving "Aaaaa" of "Aaab Aaaaa" exposed.
        chosen = {_coverage_key(rule) for _, _, rule in selected}
        # Only *declared entities* are owed coverage. An allocator rule (dates,
        # periods) that matches nowhere is an ordinary non-event — it describes a
        # writing the document simply does not contain — whereas a declared
        # entity that matches nowhere is an unmasked name. Treating allocator
        # rules as shadowed would have the loop dismantle legitimate winners to
        # place patterns nobody claimed were present.
        shadowed = [
            c for c in pool if c[2].placeholder is not None and _coverage_key(c[2]) not in chosen
        ]
        if not shadowed:
            return selected
        # Blockers are identified by (span, rule identity), not by object
        # identity: _select builds fresh tuples, so a winner is never the same
        # object as the pool entry it came from.
        blockers = {
            (winner[0], winner[1], id(winner[2]))
            for winner in selected
            for start, end, rule in shadowed
            if winner[0] < end
            and start < winner[1]
            and id(winner[2]) != id(rule)
            # Only a strictly longer rival is worth removing; dropping an equal
            # or shorter one trades one unmasked entity for another.
            and (winner[1] - winner[0]) > (end - start)
            # And only when the shadowed span *extends past* the winner. A span
            # contained within the winner is a competing description of the same
            # text — "March" inside "March 31, 2024" — and losing is correct.
            # One that runs past the winner's end is a different entity being
            # swallowed, which is the case this pass exists for.
            and end > winner[1]
        }
        if not blockers:
            return selected
        pool = [c for c in pool if (c[0], c[1], id(c[2])) not in blockers]
    return _select(pool, protected)


def apply_rules(text: str, rules: Sequence[MaskRule]) -> RuleApplication:
    """Apply every rule to ``text`` and return the rewritten document.

    The result is a pure function of its arguments: the same text and the same
    rule sequence always produce the same placeholders, in the same order.

    Args:
        text: The document to rewrite. Not modified.
        rules: Rules to apply. Order affects only tie-breaking, via
            :attr:`MaskRule.priority`.

    Returns:
        A :class:`RuleApplication` holding the rewritten text, every
        replacement in document order, and one entry per distinct placeholder.
    """
    selected = _select_covering_every_entity(_candidates(text, rules), _placeholder_spans(text))

    buckets: dict[str, _Bucket] = {}
    counters: dict[MaskKind, int] = {}
    keys: dict[tuple[MaskKind, str], str] = {}

    pieces: list[str] = []
    replacements: list[Replacement] = []
    cursor = 0

    for start, end, rule in selected:
        matched = text[start:end]
        if rule.placeholder is not None:
            placeholder = rule.placeholder
            canonical = rule.canonical if rule.canonical is not None else matched
        else:
            key_fn = rule.key_fn if rule.key_fn is not None else _default_key
            identity = (rule.kind, key_fn(matched))
            existing = keys.get(identity)
            if existing is None:
                counters[rule.kind] = counters.get(rule.kind, 0) + 1
                existing = f"[{rule.kind}_{counters[rule.kind]}]"
                keys[identity] = existing
            placeholder = existing
            canonical = matched

        bucket = buckets.get(placeholder)
        if bucket is None:
            bucket = _Bucket(placeholder, rule.kind, canonical, [], 0)
            buckets[placeholder] = bucket
        if matched not in bucket.surface_forms:
            bucket.surface_forms.append(matched)
        bucket.occurrences += 1

        pieces.append(text[cursor:start])
        pieces.append(placeholder)
        cursor = end
        replacements.append(
            Replacement(
                start=start,
                end=end,
                original=matched,
                placeholder=placeholder,
                kind=rule.kind,
                rule=rule.name,
            )
        )

    pieces.append(text[cursor:])
    allocated = tuple(
        AllocatedPlaceholder(
            placeholder=bucket.placeholder,
            kind=bucket.kind,
            canonical=bucket.canonical,
            surface_forms=tuple(bucket.surface_forms),
            occurrences=bucket.occurrences,
        )
        for bucket in buckets.values()
    )
    return RuleApplication(
        text="".join(pieces),
        replacements=tuple(replacements),
        allocated=allocated,
    )
