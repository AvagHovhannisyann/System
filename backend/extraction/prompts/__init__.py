"""Prompt management: content-addressed versions, history, diff, rollback (P7.10, §6.5).

§6.5 asks for full version history per prompt, a side-by-side diff between
versions, a golden-set score attached to each one, and one-click rollback.
Every one of those follows from a single decision, which is why it is the
package's organising idea rather than an implementation detail:

**A prompt version's identity is a digest of its own content.** So —

- *history* is a list of addresses, and no two writings of the same text are
  two versions;
- *rollback is selection*, not restoration: pointing a prompt at an earlier
  hash returns text that is byte-identical to the text that was measured, so
  the golden-set score attached to that hash still describes what is now in
  force. A rollback implemented as "copy the old text back into the current
  row" could not make that claim;
- *cache invalidation is structural*: the extraction cache is keyed on
  ``hash(document + prompt_version + model)``
  (:mod:`backend.extraction.cache`), so a changed prompt addresses entries
  that do not exist. There is no invalidation step to forget and no window in
  which a new prompt reads an old response.

Modules:

- :mod:`backend.extraction.prompts.versioning` — the version and its address.
  The address is a *property* derived from the text on every read, never a
  stored string a caller could pass stale.
- :mod:`backend.extraction.prompts.diff` — the alignment §6.5's two-column
  view needs, per section (``system``, ``template``, ``schema``) because the
  three fail differently and merging them hides the schema case.
- :mod:`backend.extraction.prompts.store` — the record around a version: who
  saved it, which one is in force, and the golden-set scores. Also
  :func:`~backend.extraction.prompts.store.golden_verdict`, which refuses to
  judge a score without both a threshold and the statement of where the
  threshold came from (D-014: the directive's 85% is the operator's prior, and
  the human noise floor is unmeasured — B3).
- :mod:`backend.extraction.prompts.postgres` — the durable store, append-only
  by trigger, recording every activation as a configuration event (§6.11).

:class:`~backend.extraction.prompts.postgres.PostgresPromptStore` is
deliberately **not** re-exported here, and neither is
:mod:`backend.extraction.tasks.store`'s durable counterpart. Importing it pulls
in :mod:`backend.db`, and the point of keeping it one import away is that the
addressing, diffing and verdict logic above stays usable — and testable — in a
process that has no database. A caller that wants durability names the module.
"""

from __future__ import annotations

from backend.extraction.prompts.diff import (
    DiffOp,
    DiffSegment,
    PromptDiff,
    SectionDiff,
    diff_versions,
    unified_lines,
)
from backend.extraction.prompts.store import (
    AUDIT_FIELD,
    AUDIT_SCOPE,
    Activation,
    GoldenSetScore,
    GoldenVerdict,
    InMemoryPromptStore,
    NoActivePromptError,
    PromptRecord,
    PromptStore,
    PromptVersionNotFoundError,
    golden_verdict,
)
from backend.extraction.prompts.versioning import (
    PROMPT_HASH_BYTES,
    PromptRenderError,
    PromptVersion,
    content_digest,
)

__all__ = [
    "AUDIT_FIELD",
    "AUDIT_SCOPE",
    "PROMPT_HASH_BYTES",
    "Activation",
    "DiffOp",
    "DiffSegment",
    "GoldenSetScore",
    "GoldenVerdict",
    "InMemoryPromptStore",
    "NoActivePromptError",
    "PromptDiff",
    "PromptRecord",
    "PromptRenderError",
    "PromptStore",
    "PromptVersion",
    "PromptVersionNotFoundError",
    "SectionDiff",
    "content_digest",
    "diff_versions",
    "golden_verdict",
    "unified_lines",
]
