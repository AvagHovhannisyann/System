"""Declared entities and the reversible mapping that must not reach a model (P7.2).

An :class:`Entity` is a fact the pipeline already knows about a document from
its ingestion metadata — the filer's conformed name, its former names, the
reporting owners, the ticker, the CIK. Anonymization is metadata-driven rather
than a named-entity model: what we can prove we masked is exactly what we were
told to mask, which is the only claim the leak detector can check.

The mapping is the part that has to be handled carefully. It is the inverse of
the masking, so it re-identifies the document completely; the whole control
fails if it is ever concatenated into a prompt or written to a log line beside
the anonymized text. Two structural defences here:

* :class:`AnonymizedDocument` separates the two — :attr:`~AnonymizedDocument.text`
  is the only field intended to leave the process, and the mapping is a
  different object rather than a field of the payload;
* :class:`EntityMapping` renders redacted. ``repr`` of a mapping shows the
  placeholders and the counts and never the values, so an exception traceback,
  a ``structlog`` event or a debugger frame cannot spill it (§7 requires the
  same of API keys, for the same reason).

Neither defence stops determined misuse. They stop the accident.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from backend.extraction.rules import AllocatedPlaceholder, MaskKind, Replacement

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "AnonymizedDocument",
    "Entity",
    "EntityKind",
    "EntityMapping",
    "company",
    "identifier",
    "merge_writings",
    "person",
    "person_from_edgar_conformed_name",
    "ticker",
]


class EntityKind(StrEnum):
    """Kinds of entity a caller may declare.

    A strict subset of :class:`~backend.extraction.rules.MaskKind`: temporal
    placeholders are derived from the text, never declared.
    """

    COMPANY = "COMPANY"
    PERSON = "PERSON"
    TICKER = "TICKER"
    IDENTIFIER = "IDENTIFIER"


_MASK_KIND: dict[EntityKind, MaskKind] = {
    EntityKind.COMPANY: MaskKind.COMPANY,
    EntityKind.PERSON: MaskKind.PERSON,
    EntityKind.TICKER: MaskKind.TICKER,
    EntityKind.IDENTIFIER: MaskKind.IDENTIFIER,
}

_NAME_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v", "md", "phd", "cpa", "esq"})


@dataclass(frozen=True, slots=True)
class Entity:
    """One identifiable thing that must not survive anonymization.

    Attributes:
        kind: What sort of entity this is.
        name: The canonical name. Recorded in the mapping and used to build the
            surface forms searched for.
        aliases: Further known writings — former conformed names, trade names,
            an unpadded CIK. Each alias is masked in its own right.
        surname: ``PERSON`` only. The family name, used for the surname-alone
            surface form. ``None`` means "derive from :attr:`name`".
        given_names: ``PERSON`` only. Forenames and initials, used to build
            reordered and initialised writings.
    """

    kind: EntityKind
    name: str
    aliases: tuple[str, ...] = ()
    surname: str | None = None
    given_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject empty names and person-only fields set on non-person kinds."""
        if not self.name.strip():
            msg = "entity name must be non-empty; an empty name masks nothing"
            raise ValueError(msg)
        if self.kind is not EntityKind.PERSON and (self.surname or self.given_names):
            msg = f"surname/given_names are PERSON-only; got kind={self.kind}"
            raise ValueError(msg)

    @property
    def mask_kind(self) -> MaskKind:
        """The placeholder category this entity is masked under."""
        return _MASK_KIND[self.kind]

    @property
    def writings(self) -> tuple[str, ...]:
        """Canonical name followed by aliases, de-duplicated, order preserved."""
        seen: list[str] = []
        for value in (self.name, *self.aliases):
            cleaned = value.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return tuple(seen)


def company(name: str, *aliases: str) -> Entity:
    """Declare an issuer or filer.

    Args:
        name: Conformed or legal name, e.g. ``"AdaptHealth Corp."``.
        *aliases: Former conformed names and any other known writing.

    Returns:
        A ``COMPANY`` entity.
    """
    return Entity(kind=EntityKind.COMPANY, name=name, aliases=tuple(aliases))


def person(
    name: str,
    *aliases: str,
    surname: str | None = None,
    given_names: Sequence[str] = (),
) -> Entity:
    """Declare a named individual (an officer, director or reporting owner).

    Args:
        name: The name as written, e.g. ``"James E. Flynn"``.
        *aliases: Other known writings.
        surname: Family name. When omitted it is taken to be the last
            non-suffix token of ``name`` — the natural-order convention. EDGAR
            conformed names are *not* in natural order; use
            :func:`person_from_edgar_conformed_name` for those rather than
            letting this guess.
        given_names: Forenames and initials. When omitted they are taken to be
            every non-suffix token before the surname.

    Returns:
        A ``PERSON`` entity.
    """
    tokens = [t for t in re.split(r"[^\w'\u2019]+", name) if t]
    core = [t for t in tokens if t.strip(".").casefold() not in _NAME_SUFFIXES]
    resolved_surname = surname if surname else (core[-1] if core else name)
    if given_names:
        resolved_given = tuple(given_names)
    else:
        resolved_given = tuple(core[:-1]) if len(core) > 1 else ()
    return Entity(
        kind=EntityKind.PERSON,
        name=name,
        aliases=tuple(aliases),
        surname=resolved_surname,
        given_names=resolved_given,
    )


def person_from_edgar_conformed_name(conformed: str, *aliases: str) -> Entity:
    """Declare a person from EDGAR's ``COMPANY CONFORMED NAME`` for an individual.

    EDGAR writes an individual reporting owner surname-first and unpunctuated —
    ``"Flynn James E"`` is James E. Flynn. Reading that as natural order would
    put the surname-alone rule on the wrong token and leave the real surname in
    the text, so the convention is applied explicitly here instead of guessed
    in :func:`person`.

    Args:
        conformed: The conformed name, surname first.
        *aliases: Other known writings.

    Returns:
        A ``PERSON`` entity whose surname is the first token.
    """
    tokens = [t for t in re.split(r"[^\w'\u2019]+", conformed) if t]
    core = [t for t in tokens if t.strip(".").casefold() not in _NAME_SUFFIXES]
    surname = core[0] if core else conformed
    return Entity(
        kind=EntityKind.PERSON,
        name=conformed,
        aliases=tuple(aliases),
        surname=surname,
        given_names=tuple(core[1:]),
    )


def ticker(symbol: str, *aliases: str) -> Entity:
    """Declare an exchange ticker, e.g. ``"AHCO"``.

    Args:
        symbol: The symbol as listed.
        *aliases: Other symbols for the same security (dual listings, class
            suffixes such as ``"BRK.B"``).

    Returns:
        A ``TICKER`` entity.
    """
    return Entity(kind=EntityKind.TICKER, name=symbol, aliases=tuple(aliases))


def identifier(value: str, *aliases: str) -> Entity:
    """Declare a registry identifier — a CIK, IRS number, LEI, file number.

    Identifiers are as re-identifying as a name and a model may well have
    memorised the common ones, so they are masked on the same footing. Note
    that only *declared* identifiers are masked: this module does not scan for
    identifier-shaped strings it was not told about (see the package docstring's
    statement of limits).

    Args:
        value: The identifier as written.
        *aliases: Equivalent writings. For a CIK, both the zero-padded
            ten-digit and unpadded forms are generated automatically, so they
            need not be passed here.

    Returns:
        An ``IDENTIFIER`` entity.
    """
    return Entity(kind=EntityKind.IDENTIFIER, name=value, aliases=tuple(aliases))


@dataclass(frozen=True, slots=True, repr=False)
class EntityMapping:
    """Placeholder-to-original mapping, which must **never** be sent to a model.

    Holding it beside the anonymized text in one object would make it one
    attribute access away from a prompt; it is a separate object for that
    reason, and its ``repr`` is redacted so a traceback or log line cannot
    spill it.

    Attributes:
        entries: One entry per distinct placeholder emitted, in emission order.
    """

    entries: tuple[AllocatedPlaceholder, ...]

    def __repr__(self) -> str:
        """Render placeholders and counts only — never the values behind them."""
        summary = ", ".join(f"{e.placeholder}x{e.occurrences}" for e in self.entries)
        return f"EntityMapping(<redacted: {len(self.entries)} placeholders> {summary})"

    @property
    def by_placeholder(self) -> Mapping[str, AllocatedPlaceholder]:
        """Index of the entries keyed by placeholder text."""
        return {entry.placeholder: entry for entry in self.entries}

    def canonical_for(self, placeholder: str) -> str | None:
        """Return the value a placeholder stands for, or ``None`` if unknown."""
        entry = self.by_placeholder.get(placeholder)
        return None if entry is None else entry.canonical

    def restore(self, text: str) -> str:
        """Re-identify ``text`` by substituting canonical values for placeholders.

        The inverse of masking, up to surface variation: a placeholder that
        replaced several writings of one entity restores to the canonical one,
        so ``restore(anonymize(doc).text)`` is semantically the original, not
        byte-identical to it.

        Args:
            text: Anonymized text, or any text containing these placeholders.

        Returns:
            The text with every known placeholder replaced by its canonical
            value. Unknown placeholder-shaped tokens are left untouched.
        """
        index = self.by_placeholder
        if not index:
            return text
        pattern = re.compile("|".join(re.escape(p) for p in sorted(index, key=len, reverse=True)))
        return pattern.sub(lambda m: index[m.group(0)].canonical, text)


@dataclass(frozen=True, slots=True, repr=False)
class AnonymizedDocument:
    """The payload safe to send, plus everything needed to audit or reverse it.

    Attributes:
        text: The anonymized document. **This is the only field intended to
            reach an extraction model.**
        mapping: The reversible mapping. Keep it in the store, not in a prompt.
        replacements: Every replaced span in document order, for the operator
            document inspector (§6.5) and for measuring over-masking. Each one
            carries the original text of the span, so this field
            re-identifies the document as completely as :attr:`mapping` does
            and gets the same handling: store it, never log it.
        pre_existing_placeholders: Placeholder-shaped tokens already present in
            the source document. Each one makes :meth:`EntityMapping.restore`
            ambiguous for that token, so they are surfaced rather than
            silently tolerated.
    """

    text: str
    mapping: EntityMapping
    replacements: tuple[Replacement, ...] = ()
    pre_existing_placeholders: tuple[str, ...] = ()

    def __repr__(self) -> str:
        """Render counts only — never the text, and never a replaced original.

        This is the object most likely to end up in a traceback or a log line,
        and the default dataclass ``repr`` would print the whole document
        alongside every original span — the two things this package exists to
        keep apart. Redacting :class:`EntityMapping` alone would not have been
        enough.
        """
        return (
            f"AnonymizedDocument(<redacted: {len(self.text)} chars, "
            f"{len(self.replacements)} replacements, "
            f"{len(self.mapping.entries)} placeholders>)"
        )

    def counts_by_kind(self) -> dict[MaskKind, int]:
        """Number of replaced spans per placeholder category (count, not chars)."""
        counts: dict[MaskKind, int] = {}
        for replacement in self.replacements:
            counts[replacement.kind] = counts.get(replacement.kind, 0) + 1
        return counts

    def placeholders(self) -> tuple[str, ...]:
        """Distinct placeholders present, in order of first appearance."""
        seen: list[str] = []
        for replacement in self.replacements:
            if replacement.placeholder not in seen:
                seen.append(replacement.placeholder)
        return tuple(seen)


def merge_writings(entities: Iterable[Entity]) -> tuple[str, ...]:
    """Every declared writing across ``entities``, de-duplicated, order preserved.

    Args:
        entities: Declared entities.

    Returns:
        The union of each entity's :attr:`Entity.writings`.
    """
    seen: list[str] = []
    for entity in entities:
        for writing in entity.writings:
            if writing not in seen:
                seen.append(writing)
    return tuple(seen)
