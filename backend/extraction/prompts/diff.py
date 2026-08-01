"""Structured difference between two prompt versions (P7.10, §6.5).

§6.5 asks for a "side-by-side diff between versions" on the operator's
Agents & Extraction page. What that page needs is not a rendered patch: it needs
*alignment* — which lines of the old version correspond to which lines of the
new one, and which have no counterpart — so it can lay two columns beside each
other and colour them. So this module emits data
(:class:`PromptDiff`), and :func:`unified_lines` exists only for the times a
human wants to read the same information in a terminal or a commit message.

Diffing is per section — ``system``, ``template``, ``schema`` — rather than over
one concatenated blob, because the three fail differently. A changed system
instruction is a change of task; a changed template is usually a change of
wording; a changed schema digest means the *output model* moved and the prompt
text may be untouched. Merging them would hide the third case, which is the one
an operator is least likely to be expecting (see
:mod:`backend.extraction.prompts.versioning` on why the schema is part of a
prompt's address at all).

The schema section is a digest, not text, so its diff is necessarily "equal" or
"replaced" — there is nothing finer to show, and pretending otherwise by
diffing hex characters would produce a coloured mess that means nothing.

Units: line numbers are 0-based indices into the section's line list, and every
range is half-open ``[start, end)``. Line lists are produced by
:meth:`str.splitlines`, so trailing newlines do not appear as empty final lines.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from backend.extraction.prompts.versioning import PromptVersion

__all__ = [
    "DiffOp",
    "DiffSegment",
    "PromptDiff",
    "SectionDiff",
    "diff_versions",
    "unified_lines",
]


class DiffOp(StrEnum):
    """What happened to one aligned run of lines."""

    EQUAL = "EQUAL"
    REPLACE = "REPLACE"
    INSERT = "INSERT"
    DELETE = "DELETE"


@dataclass(frozen=True, slots=True)
class DiffSegment:
    """One aligned run of lines, common to both sides or present on only one.

    Attributes:
        op: What happened to this run.
        old_start: First line index in the old section (0-based, inclusive).
        old_end: One past the last line index in the old section.
        new_start: First line index in the new section (0-based, inclusive).
        new_end: One past the last line index in the new section.
        old_lines: The old side's lines for this run; empty for ``INSERT``.
        new_lines: The new side's lines for this run; empty for ``DELETE``.
    """

    op: DiffOp
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    old_lines: tuple[str, ...]
    new_lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SectionDiff:
    """The difference in one section of a prompt.

    Attributes:
        section: ``"system"``, ``"template"`` or ``"schema"``.
        changed: True when the two sides differ at all.
        segments: The full alignment, in order, covering both sides
            completely — including unchanged runs, which a side-by-side view
            needs in order to keep the two columns in step.
    """

    section: str
    changed: bool
    segments: tuple[DiffSegment, ...]

    @property
    def added_lines(self) -> int:
        """Lines present on the new side and not the old (count)."""
        return sum(
            len(s.new_lines) for s in self.segments if s.op in (DiffOp.INSERT, DiffOp.REPLACE)
        )

    @property
    def removed_lines(self) -> int:
        """Lines present on the old side and not the new (count)."""
        return sum(
            len(s.old_lines) for s in self.segments if s.op in (DiffOp.DELETE, DiffOp.REPLACE)
        )


@dataclass(frozen=True, slots=True)
class PromptDiff:
    """The difference between two versions of a prompt.

    Attributes:
        name: The prompt's name. A diff is only ever built between two versions
            of the same prompt; :func:`diff_versions` refuses otherwise.
        old_version_hash: Content address of the left-hand version.
        new_version_hash: Content address of the right-hand version.
        sections: One :class:`SectionDiff` per section, in the fixed order
            ``system``, ``template``, ``schema``.
    """

    name: str
    old_version_hash: str
    new_version_hash: str
    sections: tuple[SectionDiff, ...]

    @property
    def changed(self) -> bool:
        """True when the two versions differ in any section.

        Equivalent to ``old_version_hash != new_version_hash``: the address
        covers exactly the sections diffed here, so the two can never disagree.
        That redundancy is deliberate — a mismatch between them would mean the
        address had stopped describing the content, and a test pins it.
        """
        return any(section.changed for section in self.sections)


def _segments(old: str, new: str) -> tuple[DiffSegment, ...]:
    """Align two texts line-wise and return every run, changed or not."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    return tuple(
        DiffSegment(
            op=DiffOp(tag.upper()),
            old_start=i1,
            old_end=i2,
            new_start=j1,
            new_end=j2,
            old_lines=tuple(old_lines[i1:i2]),
            new_lines=tuple(new_lines[j1:j2]),
        )
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
    )


def diff_versions(old: PromptVersion, new: PromptVersion) -> PromptDiff:
    """Return the section-by-section difference between two prompt versions.

    Args:
        old: The left-hand version (typically the one in force).
        new: The right-hand version (typically the candidate, or — during a
            rollback review — the older version being returned to).

    Returns:
        A :class:`PromptDiff` covering both sides completely.

    Raises:
        ValueError: if the two versions are versions of different prompts.
            Diffing across names would render a full replacement of every line
            and read as a catastrophic edit rather than as the mistake it is.
    """
    if old.name != new.name:
        msg = (
            f"cannot diff versions of different prompts: {old.name!r} vs {new.name!r}. "
            "A cross-prompt diff renders as a total rewrite, which hides the mistake"
        )
        raise ValueError(msg)
    sections = (
        SectionDiff(
            section="system",
            changed=old.system != new.system,
            segments=_segments(old.system, new.system),
        ),
        SectionDiff(
            section="template",
            changed=old.template != new.template,
            segments=_segments(old.template, new.template),
        ),
        SectionDiff(
            section="schema",
            changed=old.schema_digest != new.schema_digest,
            segments=_segments(old.schema_digest, new.schema_digest),
        ),
    )
    return PromptDiff(
        name=old.name,
        old_version_hash=old.version_hash,
        new_version_hash=new.version_hash,
        sections=sections,
    )


def unified_lines(diff: PromptDiff, *, context: int = 3) -> Iterator[str]:
    """Yield a unified-diff rendering of ``diff``, one line at a time.

    A convenience for logs, reports and terminals. The operator page renders
    :class:`PromptDiff` directly; nothing in the system parses this text back.

    Args:
        diff: The structured difference.
        context: Unchanged lines of context around each hunk (count).

    Yields:
        Lines without trailing newlines, in unified-diff format, with a
        ``--- section`` / ``+++ section`` header per changed section. Unchanged
        sections are skipped entirely.
    """
    for section in diff.sections:
        if not section.changed:
            continue
        old_lines = [line for segment in section.segments for line in segment.old_lines]
        new_lines = [line for segment in section.segments for line in segment.new_lines]
        yield from (
            line.rstrip("\n")
            for line in difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"{diff.name}:{section.section}@{diff.old_version_hash}",
                tofile=f"{diff.name}:{section.section}@{diff.new_version_hash}",
                n=context,
                lineterm="",
            )
        )
