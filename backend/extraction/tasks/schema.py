"""Schema-validated extraction outputs (P7.3, §5-P7 step 4).

An extraction produces a *number the predictor will train on*. Everything in
this module exists to keep a model's prose from becoming that number by
accident.

Validation rejects; it never repairs
------------------------------------

:func:`parse_extraction_output` is strict in every direction Pydantic offers,
and each strictness is a specific failure it refuses to launder:

* ``strict=True`` — **no coercion.** ``"0.4"`` is not ``0.4`` and ``1`` is not
  ``True``. A model that returned a quoted number was not following the schema,
  and a pipeline that quietly converts it has stopped being able to tell a
  compliant response from a non-compliant one — which is the only signal
  available for whether a prompt is working;
* ``extra="forbid"`` — a field the schema does not define is a rejection, not a
  discard. An extra key usually means the model answered a *different* question
  (it invented ``"company"``, or ``"score"`` beside ``"tone"``), and silently
  dropping it keeps the wrong answer's remaining fields;
* ``frozen=True`` — a validated output cannot be edited afterwards. What is
  stored beside the raw response is what validation accepted;
* bare JSON only. A fenced ```` ```json ```` block is a rejection. Unwrapping
  fences is a small, tempting repair, and it is the first step of a ratchet that
  ends with regex-extracting numbers out of prose. If a provider needs fences,
  the fix belongs in the prompt, where it is visible and versioned, not in a
  silent transport-layer fixup.

A rejection is *not* an exception the pipeline swallows: the raw response is
still cached and still stored on the result, with the validation error beside
it (:mod:`backend.extraction.tasks.pipeline`). A schema failure is data about
the prompt, and Gate G7 needs to be able to count them.

Confidence is not certainty
---------------------------

Every output model here carries ``confidence`` and ``evidence``. Neither is a
probability the platform computed — they are the *model's own* self-report, and
a self-reported confidence is not calibrated. They are recorded because a
disagreement score (P7.5) and a document inspector (§6.5) need something to show
an operator, and because an extraction with no quoted evidence is unauditable.
Nothing downstream may weight by ``confidence`` until somebody measures whether
it correlates with correctness on the golden set.

Units: all scores are **dimensionless fractions**. Shift-style fields run
``[-1, 1]``, where 0 means no change between the two documents and the sign is
documented per field. Level-style fields run ``[0, 1]``. There are no
percentages anywhere in this module (§8).
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ExtractionOutput",
    "SchemaValidationError",
    "parse_extraction_output",
    "schema_digest",
    "schema_text",
]

_SCHEMA_DOMAIN: Final = b"quant-research-platform/extraction-schema/1"
"""Domain-separation prefix for output-schema digests."""

_SCHEMA_HASH_BYTES: Final = 16
"""Digest width of a schema fingerprint in bytes (32 hex characters)."""


class ExtractionOutput(BaseModel):
    """Base class for every extraction task's response model.

    The configuration is the contract; see the module docstring for why each
    setting is what it is. Subclasses add fields and must document the units and
    the sign convention of every one of them (§8) — a signed score whose
    direction is undocumented is a coin flip in the feature library.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SchemaValidationError(ValueError):
    """A model response did not satisfy its task's output schema.

    Carries the raw response so the caller can store it: §5-P7 requires raw
    responses to be kept, and a rejected response is exactly the one worth
    keeping — it is the evidence that a prompt needs work.

    Attributes:
        raw: The provider's response text, verbatim.
        model_name: The output model it was validated against.
        errors: One human-readable line per problem, in Pydantic's order.
    """

    def __init__(self, message: str, *, raw: str, model_name: str, errors: Sequence[str]) -> None:
        """Build the error.

        Args:
            message: Summary line.
            raw: The provider's response text, verbatim.
            model_name: Name of the output model.
            errors: Per-problem descriptions.
        """
        super().__init__(message)
        self.raw = raw
        self.model_name = model_name
        self.errors = tuple(errors)


def schema_digest(model: type[ExtractionOutput]) -> str:
    """Return a stable fingerprint of an output model's JSON schema.

    This is part of a prompt's content address
    (:mod:`backend.extraction.prompts.versioning`), which is what makes a schema
    change invalidate the cache: a response cached under the old schema is not
    addressable by a prompt that now demands a different shape.

    The digest is taken over the JSON schema serialized with sorted keys, so it
    depends on field names, types, constraints, ordering-independent structure
    and docstrings — everything the model is told to produce — and not on
    Python-level details like the class's module.

    Args:
        model: The output model.

    Returns:
        32 lowercase hex characters (dimensionless).
    """
    canonical = json.dumps(model.model_json_schema(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(digest_size=_SCHEMA_HASH_BYTES)
    digest.update(_SCHEMA_DOMAIN)
    digest.update(canonical.encode())
    return digest.hexdigest()


def schema_text(model: type[ExtractionOutput]) -> str:
    """Return the model's JSON schema as indented text, for embedding in a prompt.

    The prompt shows the model the schema it must satisfy rather than describing
    it in prose, so the instruction and the validator cannot drift apart: both
    are generated from the same class, and a field added to the class changes
    both the prompt text and the prompt's address in the same commit.

    Args:
        model: The output model.

    Returns:
        Indented JSON with sorted keys — deterministic, because it is part of a
        content-addressed prompt.
    """
    return json.dumps(model.model_json_schema(), sort_keys=True, indent=2)


def parse_extraction_output[OutputT: ExtractionOutput](raw: str, model: type[OutputT]) -> OutputT:
    """Validate a raw model response against a task's output schema.

    Args:
        raw: The provider's response text, verbatim. Leading and trailing
            whitespace is ignored — that is transport noise, not content. No
            other repair is attempted; see the module docstring on fences.
        model: The output model to validate against.

    Returns:
        The validated, frozen output.

    Raises:
        SchemaValidationError: if the response is not JSON, is not a JSON
            object, or does not satisfy the schema — including any coercion the
            schema would have needed. The exception carries ``raw`` so the
            caller can store the response it rejected.
    """
    text = raw.strip()
    if not text:
        msg = f"empty response cannot satisfy {model.__name__}"
        raise SchemaValidationError(msg, raw=raw, model_name=model.__name__, errors=(msg,))
    try:
        decoded: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = (
            f"response is not JSON ({exc.msg} at line {exc.lineno} column {exc.colno}); "
            f"{model.__name__} expects a bare JSON object with no code fence and no prose"
        )
        raise SchemaValidationError(msg, raw=raw, model_name=model.__name__, errors=(msg,)) from exc
    if not isinstance(decoded, dict):
        msg = (
            f"response decoded to a {type(decoded).__name__}; {model.__name__} expects a "
            "JSON object"
        )
        raise SchemaValidationError(msg, raw=raw, model_name=model.__name__, errors=(msg,))
    try:
        return model.model_validate_json(text, strict=True)
    except ValidationError as exc:
        details = tuple(
            f"{'.'.join(str(part) for part in error['loc']) or '<root>'}: {error['msg']}"
            for error in exc.errors()
        )
        msg = f"response does not satisfy {model.__name__}: {'; '.join(details)}"
        raise SchemaValidationError(
            msg, raw=raw, model_name=model.__name__, errors=details
        ) from exc
