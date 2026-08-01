"""Surface forms: turning a declared entity into the patterns a filing writes it in (P7.2).

A filing does not repeat its conformed name. One captured Form 4 header names
the same family of filers ten different ways — ``DEERFIELD MANAGEMENT COMPANY,
L.P. (SERIES C)``, ``Deerfield Mgmt L.P.``, ``DEERFIELD MANAGEMENT CO``,
``DEERFIELD MANAGEMENT CO /NY``, ``DEERFIELD PARTNERS, L.P.``, ``DEERFIELD
PARTNERS, LP``, ``Deerfield Private Design Fund IV, L.P.``, ``Deerfield Mgmt
IV, L.P.``, ``DEERFIELD CAPITAL LP``, ``DEERFIELD CAPITAL LP ET AL`` — differing
in case, in legal suffix, in whether ``Management`` is abbreviated to ``Mgmt``
and ``Company`` to ``CO``. Matching the declared string literally would leave
most of that standing.

So each declared writing produces several rules, in decreasing specificity:

1. **verbatim** — the declared string with whitespace made flexible, so a name
   broken across a line break still matches;
2. **core tokens** — the name with its legal-form suffix dropped and every
   remaining token allowed to appear in any of its known abbreviations, with an
   optional legal-form tail. This is what catches ``Deerfield Mgmt L.P.`` from
   a declaration of ``Deerfield Management Company, L.P.``;
3. **leading token** — the first core token alone, for the extremely common
   short reference (``Deerfield`` for ``Deerfield Management Company, L.P.``).

Rule 3 is the one that trades precision for recall, and it does so
deliberately. A generic leading token (``United``, ``General``, ``National``)
is excluded by :data:`GENERIC_NAME_TOKENS`, but a distinctive one is masked
everywhere it appears, including where it is not the company. That is the safe
direction of error for this control: over-masking costs readability and is
visible in :attr:`~backend.extraction.entities.AnonymizedDocument.replacements`,
while under-masking is a silent contamination channel that the P7.9 probe would
then attribute to the model. It can be turned off
(``AnonymizerConfig.mask_leading_token``) when precision matters more.

Person names are handled the same way, from an explicit surname and forenames
rather than a guess: EDGAR writes individuals surname-first (``Flynn James E``)
and prose writes them forename-first, and picking the wrong token as the
surname leaves the real surname in the text.

Units: everything here returns compiled patterns; no offsets, no counts.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

from backend.extraction.entities import Entity, EntityKind
from backend.extraction.rules import LEFT_BOUNDARY, RIGHT_BOUNDARY, MaskRule

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "COMMON_WORD_SURNAMES",
    "GENERIC_NAME_TOKENS",
    "core_tokens",
    "entity_rules",
]

_TOKEN_SEPARATOR: Final = r"\s*[,&/\u2010-\u2015-]?\s*"  # noqa: S105
r"""Separator allowed between two tokens of a name.

Permits whitespace including a newline (``\s``, which in Python's ``str``
patterns already covers U+00A0 and the other Unicode spaces filings use), a
comma, an ampersand and the dash family, and permits nothing at all. It
deliberately does **not** permit a full stop followed by whitespace: allowing
that would let ``...acquired Deerfield.  Management believes...`` match as one
company name across a sentence boundary.

The ``noqa: S105`` above is a false positive: flake8-bandit flags any string
constant whose *name* contains "token", and this one is a regular-expression
fragment about name tokens, not a credential. No secret is defined in this
package.
"""

_LEGAL_FORMS: Final = (
    "corporation",
    "corp",
    "incorporated",
    "inc",
    "company",
    "co",
    "cos",
    "limited",
    "ltd",
    "llc",
    "lp",
    "llp",
    "plc",
    "sa",
    "se",
    "ag",
    "nv",
    "bv",
    "gmbh",
    "ab",
    "oyj",
    "spa",
    "pte",
    "pty",
    "kk",
    "srl",
    "bhd",
    "nl",
    "kgaa",
    "aktiengesellschaft",
)
"""Legal-form tokens dropped from the core and allowed as an optional tail.

Only true legal forms. ``Partners``, ``Group``, ``Holdings`` and ``Trust`` are
*not* here even though filings drop them casually, because they distinguish one
declared entity from another (``Deerfield Management`` from ``Deerfield
Partners``) and collapsing them would give two entities the same pattern.
"""

_TOKEN_ABBREVIATIONS: Final[dict[str, tuple[str, ...]]] = {
    "company": ("co",),
    "co": ("company",),
    "corporation": ("corp",),
    "corp": ("corporation",),
    "incorporated": ("inc",),
    "inc": ("incorporated",),
    "limited": ("ltd",),
    "ltd": ("limited",),
    "management": ("mgmt", "mgt"),
    "mgmt": ("management", "mgt"),
    "mgt": ("management", "mgmt"),
    "international": ("intl", "int'l"),
    "intl": ("international", "int'l"),
    "technologies": ("tech", "technology"),
    "technology": ("tech", "technologies"),
    "holdings": ("hldgs", "holding"),
    "holding": ("holdings", "hldgs"),
    "industries": ("inds", "industrial"),
    "manufacturing": ("mfg",),
    "mfg": ("manufacturing",),
    "associates": ("assoc", "assocs"),
    "brothers": ("bros",),
    "bros": ("brothers",),
    "laboratories": ("labs", "laboratory"),
    "labs": ("laboratories", "laboratory"),
    "systems": ("sys",),
    "services": ("svcs", "svc"),
    "service": ("svc",),
    "national": ("natl", "nat'l"),
    "american": ("amer",),
    "and": ("&",),
    "&": ("and",),
    "partners": ("ptrs",),
    "resources": ("res",),
    "enterprises": ("ent",),
    "development": ("dev",),
    "financial": ("finl", "fin'l"),
    "communications": ("comms", "comm"),
    "properties": ("props",),
    "pharmaceuticals": ("pharma", "pharmaceutical"),
    "pharmaceutical": ("pharma", "pharmaceuticals"),
}
"""Bidirectional abbreviation equivalences applied token by token."""

GENERIC_NAME_TOKENS: Final = frozenset(
    {
        "advanced",
        "allied",
        "american",
        "america",
        "applied",
        "atlantic",
        "capital",
        "central",
        "century",
        "commercial",
        "consumer",
        "continental",
        "digital",
        "east",
        "eastern",
        "energy",
        "enterprise",
        "federal",
        "financial",
        "first",
        "general",
        "global",
        "great",
        "imperial",
        "independence",
        "industrial",
        "insurance",
        "integrated",
        "international",
        "liberty",
        "media",
        "medical",
        "national",
        "new",
        "north",
        "northern",
        "pacific",
        "premier",
        "prime",
        "quality",
        "republic",
        "resources",
        "royal",
        "security",
        "select",
        "service",
        "services",
        "solutions",
        "south",
        "southern",
        "standard",
        "state",
        "sterling",
        "strategic",
        "summit",
        "superior",
        "union",
        "united",
        "universal",
        "value",
        "west",
        "western",
    }
)
"""Leading tokens too generic to mask on their own.

Masking ``United`` for ``United Technologies`` would also mask ``United
States``, which mangles the document without improving the control. The full
name and its abbreviations are still matched by the verbatim and core rules.
"""

COMMON_WORD_SURNAMES: Final = frozenset({"may", "march", "august", "april", "june", "will"})
"""Surnames that are also month names or modal verbs.

For these the surname-alone rule is skipped: masking every ``May`` in a filing
would destroy more than it protects. The full name is still matched. This is a
stated limitation — a person referred to only as "Ms. May" survives.
"""

# An earlier draft guarded the surname-alone rule with
# `(?<![Tt]he )(?<![Aa] )(?<![Aa]n )`, so that `the Cook`, `a Baker` and `an
# Archer` were read as nouns rather than as the officer. It is gone, for two
# reasons.
#
# First, it disagreed with the leak detector, which has no such guard and
# reports a surviving declared surname as a LEAK. A masker that deliberately
# leaves something its own checker calls a failure makes `report.leaks == ()`
# unreachable, and a gate nobody can pass is a gate everybody switches off.
#
# Second, it traded the wrong way round. The rule is case-sensitive, so it only
# ever fires on a capitalised surname; `a baker's dozen` was never at risk. What
# the guard actually protected was `The Cook` at the start of a sentence, and it
# paid for that with a silent hole for `the Flynn family` — under-masking, which
# this package treats as the unacceptable direction because it is invisible
# downstream and P7.9 would attribute it to the model.
#
# The cost of removing it is that a declared officer named Cook or Price makes
# those words unreadable wherever they are capitalised in their own filing.
# That is over-masking: visible in `AnonymizedDocument.replacements`, and
# switchable off per-run with `AnonymizerConfig.mask_surname_alone`.

_NAME_SUFFIXES: Final = ("jr", "sr", "ii", "iii", "iv", "md", "phd", "cpa", "esq")


def _clean_for_tokens(name: str) -> str:
    """Drop parenthetical qualifiers before tokenizing.

    ``DEERFIELD MANAGEMENT COMPANY, L.P. (SERIES C)`` tokenizes to the same core
    as its siblings; the qualifier is matched by the verbatim rule instead.
    """
    return re.sub(r"\([^)]*\)", " ", name)


def _raw_tokens(name: str) -> list[str]:
    """Split a name into alphanumeric tokens, merging runs of single letters.

    ``L.P.`` becomes one token ``LP`` and ``J. P. Morgan`` becomes ``JP``,
    ``Morgan`` — otherwise the legal-form check below would see ``l`` and ``p``
    and keep them as core tokens.
    """
    tokens = re.findall(r"[A-Za-z0-9&'\u2019]+", _clean_for_tokens(name))
    merged: list[str] = []
    run: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if run:
            merged.append("".join(run))
            run = []
        merged.append(token)
    if run:
        merged.append("".join(run))
    return merged


def core_tokens(name: str) -> tuple[str, ...]:
    """Return ``name``'s distinguishing tokens: no leading article, no legal form.

    Args:
        name: A company name as written.

    Returns:
        The tokens that identify the company. Never empty: if every token is a
        legal form (``"The Company"``), the original tokens are returned rather
        than nothing, because a rule built from nothing matches everything.
    """
    tokens = _raw_tokens(name)
    if tokens and tokens[0].casefold() == "the":
        tokens = tokens[1:]
    core = list(tokens)
    while len(core) > 1 and core[-1].casefold() in _LEGAL_FORMS:
        core.pop()
    if not core:
        return tuple(tokens)
    return tuple(core)


def _token_alternatives(token: str) -> str:
    """Regex alternation for one token and its known abbreviations."""
    folded = token.casefold()
    forms = [token, *_TOKEN_ABBREVIATIONS.get(folded, ())]
    seen: list[str] = []
    for form in forms:
        if form.casefold() not in {s.casefold() for s in seen}:
            seen.append(form)
    seen.sort(key=len, reverse=True)
    return "(?:" + "|".join(re.escape(form) + r"\.?" for form in seen) + ")"


_LEGAL_TAIL: Final = (
    r"(?:"
    + _TOKEN_SEPARATOR
    + r"(?:"
    + "|".join(sorted((re.escape(form) for form in _LEGAL_FORMS), key=len, reverse=True))
    + r"|(?:[A-Za-z]\.){2,4}"
    + r")\.?"
    + r"){0,4}"
)
"""Optional trailing legal form, repeatable — ``Corp.``, ``, L.P.``, ``Co., Ltd.``.

``(?:[A-Za-z]\\.){2,4}`` covers dotted initialisms generically (``L.P.``,
``S.p.A.``, ``N.V.``) so the list above does not need every punctuation variant.
"""


def _verbatim_pattern(writing: str) -> str:
    r"""Escape ``writing`` but let any whitespace run match any whitespace run.

    The flexibility is what lets a name split across a line break match: EDGAR
    text wraps, and ``AdaptHealth\n  Corp.`` is the same name.
    """
    parts = [re.escape(part) for part in re.split(r"\s+", writing.strip()) if part]
    return r"\s+".join(parts)


def _core_pattern(writing: str) -> str | None:
    """Pattern matching ``writing``'s core tokens with abbreviations and a legal tail."""
    tokens = core_tokens(writing)
    if not tokens:
        return None
    body = _TOKEN_SEPARATOR.join(_token_alternatives(token) for token in tokens)
    return r"(?:[Tt]he\s+)?" + body + _LEGAL_TAIL


def _compile(pattern: str, *, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    """Compile ``pattern`` wrapped in the alphanumeric word boundaries."""
    return re.compile(LEFT_BOUNDARY + pattern + RIGHT_BOUNDARY, flags)


def _company_rules(entity: Entity, placeholder: str, base: int, *, leading: bool) -> list[MaskRule]:
    """Build verbatim, core and leading-token rules for a company."""
    rules: list[MaskRule] = []
    for order, writing in enumerate(entity.writings):
        rules.append(
            MaskRule(
                name=f"company.verbatim[{writing}]",
                kind=entity.mask_kind,
                pattern=_compile(_verbatim_pattern(writing)),
                priority=base + order,
                placeholder=placeholder,
                canonical=entity.name,
            )
        )
        core = _core_pattern(writing)
        if core is not None:
            rules.append(
                MaskRule(
                    name=f"company.core[{writing}]",
                    kind=entity.mask_kind,
                    pattern=_compile(core),
                    priority=base + 100 + order,
                    placeholder=placeholder,
                    canonical=entity.name,
                )
            )
    if not leading:
        return rules
    seen: set[str] = set()
    for order, writing in enumerate(entity.writings):
        tokens = core_tokens(writing)
        if len(tokens) < 2:
            continue
        lead = tokens[0]
        folded = lead.casefold()
        if folded in seen or folded in GENERIC_NAME_TOKENS or folded in _LEGAL_FORMS:
            continue
        if len(lead) < 4 or not lead[0].isalpha():
            continue
        seen.add(folded)
        rules.append(
            MaskRule(
                name=f"company.leading_token[{lead}]",
                kind=entity.mask_kind,
                pattern=_compile(_token_alternatives(lead) + _LEGAL_TAIL),
                priority=base + 200 + order,
                placeholder=placeholder,
                canonical=entity.name,
            )
        )
    return rules


def _person_writings(entity: Entity) -> list[str]:
    """Every multi-token writing of a person's name worth searching for."""
    surname = (entity.surname or "").strip()
    given = [g.strip(".") for g in entity.given_names if g.strip(".")]
    writings: list[str] = list(entity.writings)
    if surname and given:
        initials = [g[0] for g in given]
        writings.extend(
            [
                " ".join([*given, surname]),
                f"{given[0]} {surname}",
                f"{surname}, {' '.join(given)}",
                f"{surname} {' '.join(given)}",
                f"{'. '.join(initials)}. {surname}",
                f"{initials[0]}. {surname}",
            ]
        )
    unique: list[str] = []
    for writing in writings:
        if writing and writing not in unique:
            unique.append(writing)
    return unique


def _person_rules(
    entity: Entity, placeholder: str, base: int, *, surname_alone: bool
) -> list[MaskRule]:
    """Build full-name rules for a person, and optionally a surname-alone rule."""
    suffix_tail = r"(?:\s*,?\s*(?:" + "|".join(_NAME_SUFFIXES) + r")\.?)?"
    rules = [
        MaskRule(
            name=f"person.name[{writing}]",
            kind=entity.mask_kind,
            pattern=_compile(_verbatim_pattern(writing) + suffix_tail),
            priority=base + order,
            placeholder=placeholder,
            canonical=entity.name,
        )
        for order, writing in enumerate(_person_writings(entity))
    ]
    surname = (entity.surname or "").strip()
    if surname_alone and surname and surname.casefold() not in COMMON_WORD_SURNAMES:
        rules.append(
            MaskRule(
                name=f"person.surname[{surname}]",
                kind=entity.mask_kind,
                # Case-sensitive: `Cook` the officer, never `cook` the verb.
                pattern=re.compile(LEFT_BOUNDARY + re.escape(surname) + RIGHT_BOUNDARY),
                priority=base + 300,
                placeholder=placeholder,
                canonical=entity.name,
            )
        )
    return rules


def _ticker_rules(entity: Entity, placeholder: str, base: int) -> list[MaskRule]:
    """Build case-sensitive rules for a ticker in its listed and title casing.

    Case-sensitive on purpose: many symbols are ordinary words (``ALL``,
    ``CAR``, ``KEY``, ``ON``) and a case-insensitive match would redact the
    English. Upper and title case are matched, lower case is not — so a
    filing writing ``ahco`` in a URL slug is a known miss.
    """
    rules: list[MaskRule] = []
    for order, symbol in enumerate(entity.writings):
        variants = {symbol, symbol.upper(), symbol.title()}
        alternation = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
        rules.append(
            MaskRule(
                name=f"ticker[{symbol}]",
                kind=entity.mask_kind,
                pattern=re.compile(LEFT_BOUNDARY + "(?:" + alternation + ")" + RIGHT_BOUNDARY),
                priority=base + order,
                placeholder=placeholder,
                canonical=entity.name,
            )
        )
    return rules


def _identifier_writings(entity: Entity) -> list[str]:
    """Declared identifier writings plus padded/unpadded forms of digit strings.

    A CIK is written ``0001725255`` in an SGML header and ``1725255`` in a URL;
    declaring one has to cover the other, or half the occurrences survive.
    """
    writings: list[str] = []
    for writing in entity.writings:
        candidates = [writing]
        if writing.isdigit():
            candidates.extend((writing.lstrip("0") or writing, writing.zfill(10)))
        for candidate in candidates:
            if candidate and candidate not in writings:
                writings.append(candidate)
    return writings


def _identifier_rules(entity: Entity, placeholder: str, base: int) -> list[MaskRule]:
    """Build exact rules for a registry identifier and its zero-padding variants."""
    return [
        MaskRule(
            name=f"identifier[{writing}]",
            kind=entity.mask_kind,
            pattern=re.compile(LEFT_BOUNDARY + re.escape(writing) + RIGHT_BOUNDARY),
            priority=base + order,
            placeholder=placeholder,
            canonical=entity.name,
        )
        for order, writing in enumerate(_identifier_writings(entity))
    ]


def entity_rules(
    entities: Sequence[Entity],
    *,
    mask_leading_token: bool = True,
    mask_surname_alone: bool = True,
) -> tuple[MaskRule, ...]:
    """Build every surface rule for ``entities``, with stable placeholders.

    Placeholder numbers come from declaration order within a kind, not from
    order of appearance, so the same declared entity keeps the same placeholder
    across every document in a run — which is what makes cached and re-run
    extractions comparable.

    Args:
        entities: Declared entities, in a stable order (the caller's ingestion
            metadata order is fine; it just must not vary between runs).
        mask_leading_token: Mask a company's distinctive first token on its
            own. See the module docstring for the precision/recall trade.
        mask_surname_alone: Mask a person's surname on its own.

    Returns:
        Rules for every entity, ready for
        :func:`~backend.extraction.rules.apply_rules`.
    """
    counters: dict[EntityKind, int] = {}
    rules: list[MaskRule] = []
    for index, entity in enumerate(entities):
        counters[entity.kind] = counters.get(entity.kind, 0) + 1
        placeholder = f"[{entity.mask_kind}_{counters[entity.kind]}]"
        base = index * 1_000
        if entity.kind is EntityKind.COMPANY:
            rules.extend(_company_rules(entity, placeholder, base, leading=mask_leading_token))
        elif entity.kind is EntityKind.PERSON:
            rules.extend(_person_rules(entity, placeholder, base, surname_alone=mask_surname_alone))
        elif entity.kind is EntityKind.TICKER:
            rules.extend(_ticker_rules(entity, placeholder, base))
        else:
            rules.extend(_identifier_rules(entity, placeholder, base))
    return tuple(rules)
