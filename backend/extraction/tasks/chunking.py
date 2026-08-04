"""Deterministic document chunking (P7.3, §5-P7 step 1).

A chunk is a unit of work for the extraction pipeline: one chunk becomes one
model call, one cache key and one stored raw response. Two properties are
therefore load-bearing rather than nice to have, and both are pinned by property
tests:

**Determinism.** The same text and the same configuration always produce the
same chunks, byte for byte. The cache key is a digest of the payload
(:mod:`backend.extraction.cache`), so a chunker that drifted — with a hash
seed, a locale, a dict ordering — would silently miss every entry it wrote on
the previous run, and Gate G7's hit rate would fall for a reason nobody could
find. The P7.9 contamination probe compares two scorings of one document, which
also requires that "one document" chunks the same way twice.

**Exact coverage.** ``"".join(chunk.text for chunk in chunks)`` reconstructs the
input exactly, and the chunks' offsets tile ``[0, len(text))`` without gaps or
overlap. Not "approximately covers": a chunker that drops the whitespace it
splits on loses table alignment and paragraph structure, and a document that
reads differently after chunking is a document the golden set was not labelled
on. A dropped span is also invisible — the extraction simply never sees that
sentence, and reports a confident answer about the rest.

There is deliberately **no overlap between chunks.** Overlap is the usual fix
for a fact that straddles a boundary, and it costs a duplicated call and, worse,
duplicated evidence: the same sentence extracted twice, aggregated as two
independent observations. The costs are real and the benefit is speculative
until measured, so the pipeline splits at paragraph boundaries where it can and
the tasks that cannot tolerate a split declare
:attr:`~backend.extraction.tasks.base.ChunkingPolicy.WHOLE_DOCUMENT` and refuse
oversized documents rather than answering from a fragment.

How boundaries are chosen
-------------------------

1. The text is cut into segments at blank-line (paragraph) boundaries. The
   separator stays attached to the segment before it, so no character is
   orphaned.
2. A segment longer than ``max_chars`` is hard-split — at the last sentence
   terminator inside the window, else at the last whitespace, else exactly at
   the limit. A table with no whitespace still splits, at a worse place, rather
   than producing an oversized chunk that a provider would reject.
3. Segments are packed greedily into chunks up to ``max_chars``.

Units: every size and offset in this module is a count of **characters** (Python
``str`` indices), not tokens and not bytes. Tokens are the provider's unit and
vary by tokenizer, so a token budget cannot be enforced here honestly; the
character budget is a conservative proxy and the caller sets it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DEFAULT_MAX_CHARS",
    "Chunk",
    "ChunkingConfig",
    "chunk_document",
]

DEFAULT_MAX_CHARS: Final = 12_000
"""Default chunk ceiling in characters.

Roughly 3k tokens of English prose at the usual four-characters-per-token rule
of thumb — comfortably inside every cost-tier model's context window with room
for the instruction and the response, and small enough that one malformed
response costs little to re-request. It is a default, not a finding: no
measurement of extraction quality against chunk size exists yet, and pretending
this number was tuned would be inventing a result.
"""

_PARAGRAPH_BREAK: Final = re.compile(r"\n[ \t]*\n[ \t\n]*")
"""A blank-line paragraph boundary, including any run of further blank lines.

Matched so the break can be kept with the text preceding it, which is what makes
the partition exact.
"""

_SENTENCE_END: Final = re.compile(r"[.!?][\"')\]]*[ \t\n]")
"""A sentence terminator followed by its closing punctuation and whitespace.

Used only to choose a less-bad hard-split point inside an oversized paragraph.
It is not a sentence segmenter and does not need to be: getting "Inc." wrong
moves a split by a few characters, it does not lose any.
"""


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """How a document is cut up.

    Attributes:
        max_chars: Maximum characters in one chunk (count, ≥ 1). Every chunk
            satisfies ``len(text) <= max_chars``.
    """

    max_chars: int = DEFAULT_MAX_CHARS

    def __post_init__(self) -> None:
        """Reject a configuration that cannot produce chunks.

        Raises:
            ValueError: if ``max_chars`` is below 1.
        """
        if self.max_chars < 1:
            msg = f"max_chars must be >= 1 characters; got {self.max_chars}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One contiguous piece of a document.

    Attributes:
        index: Position in the chunk sequence, 0-based (dimensionless).
        start: Character offset of the first character in the source document.
        end: Character offset one past the last, so ``text ==
            document[start:end]`` and consecutive chunks satisfy
            ``previous.end == next.start``.
        text: The characters themselves, verbatim — including the whitespace
            the split happened at.
    """

    index: int
    start: int
    end: int
    text: str

    @property
    def length(self) -> int:
        """Characters in this chunk (count)."""
        return self.end - self.start


def _paragraph_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` spans cut at paragraph breaks, covering ``text`` exactly.

    Each break is kept at the end of the span it follows, so concatenating every
    span reproduces the input.
    """
    cursor = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if match.end() > cursor:
            yield cursor, match.end()
            cursor = match.end()
    if cursor < len(text):
        yield cursor, len(text)


def _split_point(text: str, start: int, end: int, max_chars: int) -> int:
    """Return the offset to cut an oversized span at, in ``(start, start + max_chars]``.

    Prefers a sentence terminator, then any whitespace, then the hard limit.
    Always returns strictly more than ``start``, so the caller always makes
    progress and cannot loop.
    """
    limit = start + max_chars
    window = text[start:limit]
    sentence_cuts = [m.end() for m in _SENTENCE_END.finditer(window)]
    if sentence_cuts:
        return start + sentence_cuts[-1]
    whitespace_cuts = [m.end() for m in re.finditer(r"\s+", window)]
    if whitespace_cuts and start + whitespace_cuts[-1] > start:
        return start + whitespace_cuts[-1]
    return min(limit, end)


def _atomic_spans(text: str, max_chars: int) -> Iterator[tuple[int, int]]:
    """Yield spans no longer than ``max_chars`` that tile ``text`` exactly."""
    for start, end in _paragraph_spans(text):
        cursor = start
        while end - cursor > max_chars:
            cut = _split_point(text, cursor, end, max_chars)
            yield cursor, cut
            cursor = cut
        if cursor < end:
            yield cursor, end


def chunk_document(text: str, config: ChunkingConfig | None = None) -> tuple[Chunk, ...]:
    """Cut ``text`` into chunks that tile it exactly.

    Deterministic: the same arguments always produce the same result, byte for
    byte. Nothing here consults a clock, a hash seed, a locale or the
    environment.

    Args:
        text: The document. Not modified.
        config: Chunk sizing. ``None`` means :class:`ChunkingConfig` defaults.

    Returns:
        Chunks in document order. Guarantees, all property-tested:

        * ``"".join(c.text for c in result) == text``;
        * ``result[0].start == 0`` and ``result[-1].end == len(text)``, with
          ``result[i].end == result[i + 1].start``;
        * every chunk is non-empty and no longer than ``config.max_chars``;
        * an empty document produces no chunks — an empty chunk would become an
          empty model call, and a response to an empty document is not an
          extraction of anything.
    """
    settings = config if config is not None else ChunkingConfig()
    chunks: list[Chunk] = []
    pending_start: int | None = None
    pending_end = 0
    for start, end in _atomic_spans(text, settings.max_chars):
        if pending_start is None:
            pending_start, pending_end = start, end
            continue
        if end - pending_start <= settings.max_chars:
            pending_end = end
            continue
        chunks.append(
            Chunk(
                index=len(chunks),
                start=pending_start,
                end=pending_end,
                text=text[pending_start:pending_end],
            )
        )
        pending_start, pending_end = start, end
    if pending_start is not None:
        chunks.append(
            Chunk(
                index=len(chunks),
                start=pending_start,
                end=pending_end,
                text=text[pending_start:pending_end],
            )
        )
    return tuple(chunks)
