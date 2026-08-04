"""Content-addressed prompt versions (P7.10, §6.5).

A prompt version has no version *number* and no mutable "current text" row. Its
identity **is** a digest of its content, so:

* two writings of the same prompt are the same version, everywhere, without
  anyone reconciling them;
* editing a prompt cannot edit a version — it produces a different address, and
  the old address keeps resolving to exactly the text it always did;
* **rollback is selection, not mutation.** Going back to last week's prompt
  means pointing the task at an earlier hash. Nothing is rewritten, so nothing
  can be lost, and the "rolled-back-to" version is byte-identical to the one
  that was measured (:mod:`backend.extraction.prompts.store` attaches
  golden-set scores to the hash, and a score attached to a mutable prompt would
  silently start describing text it never saw);
* **cache invalidation falls out of the address.** The extraction cache is
  keyed on ``hash(document + prompt_version + model)``
  (:mod:`backend.extraction.cache`), and the ``prompt_version`` component is
  :attr:`PromptVersion.version_hash` — a *property*, derived from the content
  on every read, never a stored string a caller could pass stale. A changed
  prompt therefore addresses different cache entries by construction. There is
  no invalidation step to forget, and no window in which a new prompt reads an
  old response.

What is inside the address, and why
-----------------------------------

Four fields, all of them things that change what the model is asked:

``name``
    The prompt's identity. Two prompts that happen to share text are still two
    prompts with two histories, two golden-set scores and two rollback
    timelines, so the name is part of the content rather than a label on it.

``system`` and ``template``
    The instruction and the user message, verbatim. The template's variables
    are ``$``-style (:class:`string.Template`) rather than ``str.format``
    braces, because extraction prompts contain JSON examples and a brace-based
    template would treat every one of them as a field reference.

``schema_digest``
    A digest of the Pydantic model the response must validate against
    (:func:`backend.extraction.tasks.schema.schema_digest`). This is the field
    that is easy to leave out and expensive to leave out: if the output schema
    gains a field while the prompt text is untouched, a cached response written
    under the old schema is still addressable and would be re-validated — and
    possibly re-*accepted* — under the new one. Including the schema makes a
    schema change a new prompt version, which is what it actually is: the model
    is being asked for something different.

What is deliberately *not* in the address: who wrote it, when, and any note
about why. Those describe the act of saving a version, not the version, and
they live on the store's record (:class:`~backend.extraction.prompts.store.PromptRecord`).
Putting them in the digest would give one prompt text two addresses depending
on who typed it.

Units: :attr:`PromptVersion.version_hash` is 32 lowercase hex characters — a
128-bit BLAKE2b digest, domain-separated and length-prefixed so that no two
distinct field tuples can serialize to the same byte string.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from string import Template
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "PROMPT_HASH_BYTES",
    "PromptRenderError",
    "PromptVersion",
    "content_digest",
]

PROMPT_HASH_BYTES: Final = 16
"""Digest width of a prompt address in bytes (16 bytes == 32 hex characters).

128 bits. A prompt library is a human-sized collection — hundreds of versions,
not billions — so this is a display-friendly width with collision resistance
several orders of magnitude beyond what the population needs.
"""

_DOMAIN: Final = b"quant-research-platform/prompt-version/1"
"""Domain-separation prefix.

Mixed into every prompt digest so that a prompt digest and any other BLAKE2b
digest in this repository over the same bytes are different values. Bumping the
trailing number would re-address every prompt in the store, which is a
migration, not an edit — hence it is a constant with a version in it rather
than a bare string.
"""


class PromptRenderError(ValueError):
    """A prompt was rendered with the wrong set of variables.

    Raised for both directions of mismatch — a variable the template needs and
    the caller did not supply, and a variable the caller supplied that the
    template does not use. The second is an error rather than a shrug because
    a silently-ignored variable is how a prompt ends up describing a document
    it was never given.
    """


def content_digest(*parts: str) -> str:
    """Return the length-prefixed BLAKE2b digest of ``parts``, as hex.

    Length-prefixed rather than delimiter-joined: ``("ab", "c")`` and
    ``("a", "bc")`` must not produce one digest, and there is no delimiter that
    cannot occur inside a prompt. Each part contributes its UTF-8 byte length as
    a decimal, a colon, then its bytes.

    Args:
        parts: The content fields, in a fixed order chosen by the caller.
            Order is significant.

    Returns:
        ``2 * PROMPT_HASH_BYTES`` lowercase hex characters (dimensionless).
    """
    digest = hashlib.blake2b(digest_size=PROMPT_HASH_BYTES)
    digest.update(_DOMAIN)
    for part in parts:
        encoded = part.encode()
        digest.update(f"{len(encoded)}:".encode())
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """One immutable version of one prompt, addressed by its own content.

    Attributes:
        name: The prompt's identity, conventionally the extraction task it
            serves (e.g. ``"risk_factor_language_delta"``). Non-empty, no
            leading or trailing whitespace.
        system: The system instruction sent with every call.
        template: The user message, a :class:`string.Template` source. Its
            variables are written ``$document`` or ``${document}``; a literal
            dollar sign is ``$$``.
        schema_digest: Digest of the Pydantic output model the response must
            satisfy. See the module docstring for why this is part of the
            address.
    """

    name: str
    system: str
    template: str
    schema_digest: str

    def __post_init__(self) -> None:
        """Reject a version that could not be addressed or rendered.

        Raises:
            ValueError: if ``name`` is empty or padded with whitespace (padding
                is rejected rather than trimmed, matching
                :mod:`backend.db.audit`: trimming would make ``"tone"`` and
                ``" tone"`` one key at write time and two in every log line and
                hand-written query about them), if ``template`` is empty, if
                ``schema_digest`` is empty, or if ``template`` is not a valid
                :class:`string.Template` source.
        """
        if not self.name or self.name != self.name.strip():
            msg = (
                f"prompt name must be non-empty and unpadded; got {self.name!r}. "
                "Whitespace is rejected rather than trimmed so two spellings of one "
                "prompt cannot both be addressed"
            )
            raise ValueError(msg)
        if not self.template:
            msg = f"prompt {self.name!r} has an empty template; there is nothing to send"
            raise ValueError(msg)
        if not self.schema_digest:
            msg = (
                f"prompt {self.name!r} has an empty schema_digest; the response schema is "
                "part of the prompt's address (see the module docstring)"
            )
            raise ValueError(msg)
        if not Template(self.template).is_valid():
            msg = (
                f"prompt {self.name!r} has an invalid template: a bare '$' that starts "
                "neither '$name' nor '${name}' nor '$$'. Write a literal dollar sign as '$$'"
            )
            raise ValueError(msg)

    @property
    def version_hash(self) -> str:
        """This version's content address: 32 lowercase hex characters.

        Derived on every read rather than stored, so no code path can carry a
        hash that no longer matches the text beside it. This is the
        ``prompt_version`` component of the extraction cache key, which is what
        makes a prompt edit invalidate exactly its own cache entries and
        nothing else.
        """
        return content_digest(self.name, self.system, self.template, self.schema_digest)

    @property
    def variables(self) -> frozenset[str]:
        """Names the template substitutes, e.g. ``{"document"}`` (dimensionless set)."""
        return frozenset(Template(self.template).get_identifiers())

    def render(self, variables: Mapping[str, str]) -> str:
        """Return the user message with ``variables`` substituted.

        Args:
            variables: ``{name: value}``. Must match :attr:`variables` exactly
                — no missing names, no extra ones.

        Returns:
            The rendered message. The system instruction is *not* included; it
            is sent separately (:class:`backend.extraction.tasks.client.ModelRequest`).

        Raises:
            PromptRenderError: if the supplied names differ from
                :attr:`variables` in either direction.
        """
        expected = self.variables
        supplied = frozenset(variables)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            msg = (
                f"prompt {self.name!r} takes variables {sorted(expected)}; "
                f"missing={missing} unexpected={unexpected}"
            )
            raise PromptRenderError(msg)
        return Template(self.template).substitute(variables)
