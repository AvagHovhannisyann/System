"""The extraction task framework: document to schema-validated value (P7.3, P7.4).

§5-P7's pipeline sentence, made executable: *document → chunk → anonymize →
extraction → schema validation → store with the prompt version hash*. The
package is deliberately shaped so that no task can opt out of any step.

One ordering differs from the directive's wording and does so knowingly:
**masking happens before chunking, not after.** Temporal placeholders are
allocated in order of appearance, so masking each chunk separately would make
``[DATE_1]`` a different date in every chunk and would break the one property a
paired-document delta task depends on. It is also the strictly stronger safety
posture — there is no window in which an unmasked substring is the unit of
work. :mod:`backend.extraction.tasks.pipeline` carries the full reasoning.

The modules:

- :mod:`backend.extraction.tasks.base` — a task is **data**, not a subclass. A
  name, an output model, a system instruction, a template and a chunking
  policy. There is no method to override, so there is no way for a new task to
  skip anonymization or temperature 0. It also holds
  :func:`~backend.extraction.tasks.base.paired_document`, which composes two
  filings into one payload, because §5-P7 wants **deltas, not states** and a
  difference judged in one call is not the difference of two noisy absolute
  judgements.
- :mod:`backend.extraction.tasks.chunking` — deterministic, exactly-covering
  chunking. Determinism because the cache address is a digest of the payload;
  exact coverage because a dropped span is invisible and the extraction simply
  reports confidently about the rest.
- :mod:`backend.extraction.tasks.schema` — Pydantic output models that
  **reject and never repair**: strict typing, no extra fields, no code-fence
  unwrapping. Laundering a non-compliant response destroys the only signal
  available for whether a prompt is working.
- :mod:`backend.extraction.tasks.client` — the model call as an **injected
  interface**. B4 is unresolved, so nothing here calls a provider and nothing
  fabricates a response: the default client raises (I3).
- :mod:`backend.extraction.tasks.pipeline` — the flow itself, plus the
  refusals (an oversized whole-document task, a non-zero temperature).
- :mod:`backend.extraction.tasks.library` — the five delta tasks §5-P7 names.
- :mod:`backend.extraction.tasks.store` — where a record is kept, append-only.
  Deliberately **not** re-exported below: importing it pulls in
  :mod:`backend.db`, and keeping it one import away is what lets everything
  above stay usable — and testable — in a process with no database. A caller
  that wants durability names the module.

**The live path is unexercised.** There is no provider key in this repository
(B4), so no code here has ever executed against a real model. What the tests
cover is everything on either side of that seam — chunking, masking, request
construction, validation, cache addressing, storage — driven through a
deliberate test double. A double is legitimate scaffolding; calling it evidence
about a provider's behaviour would not be (I3).
"""

from __future__ import annotations

from backend.extraction.tasks.base import (
    DOCUMENT_VARIABLE,
    ChunkingPolicy,
    ExtractionTask,
    SourceDocument,
    TaskNotRegisteredError,
    TaskRegistry,
    paired_document,
)
from backend.extraction.tasks.chunking import (
    DEFAULT_MAX_CHARS,
    Chunk,
    ChunkingConfig,
    chunk_document,
)
from backend.extraction.tasks.client import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_S,
    MAX_TEMPERATURE,
    ModelCallError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    ProviderNotConfiguredError,
    UnconfiguredModelClient,
)
from backend.extraction.tasks.library import builtin_tasks
from backend.extraction.tasks.pipeline import (
    ChunkExtraction,
    DocumentTooLargeError,
    EmptyDocumentError,
    ExtractionError,
    ExtractionPipeline,
    ExtractionRun,
    ResultSink,
    qualified_model,
)
from backend.extraction.tasks.schema import (
    ExtractionOutput,
    SchemaValidationError,
    parse_extraction_output,
    schema_digest,
    schema_text,
)

__all__ = [
    "DEFAULT_MAX_CHARS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TIMEOUT_S",
    "DOCUMENT_VARIABLE",
    "MAX_TEMPERATURE",
    "Chunk",
    "ChunkExtraction",
    "ChunkingConfig",
    "ChunkingPolicy",
    "DocumentTooLargeError",
    "EmptyDocumentError",
    "ExtractionError",
    "ExtractionOutput",
    "ExtractionPipeline",
    "ExtractionRun",
    "ExtractionTask",
    "ModelCallError",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "ProviderNotConfiguredError",
    "ResultSink",
    "SchemaValidationError",
    "SourceDocument",
    "TaskNotRegisteredError",
    "TaskRegistry",
    "UnconfiguredModelClient",
    "builtin_tasks",
    "chunk_document",
    "paired_document",
    "parse_extraction_output",
    "qualified_model",
    "schema_digest",
    "schema_text",
]
