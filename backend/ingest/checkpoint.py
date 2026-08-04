"""Incremental-sync checkpoints (P3.1).

A checkpoint is a connector-owned, JSON-serializable description of *how far
the source has been consumed* — for example ``{"last_index_date":
"2024-01-05", "last_accession": "0000320193-24-000006"}``. The framework never
interprets it; it only stores it on the ingestion-run record and hands the
most recent one back to the connector at the start of the next run
(:func:`backend.ingest.runs.latest_checkpoint`).

**There is no separate checkpoint table.** The effective checkpoint of a
source *is* the ``checkpoint_after`` of its most recent run that recorded one,
which makes it impossible for checkpoint state and run history to disagree —
a second table would let a checkpoint claim progress no run ever made.

Values are restricted to JSON scalars (:data:`JsonScalar`) inside a flat
mapping. The restriction is deliberate: the checkpoint round-trips through a
``JSONB`` column, and nested/derived Python objects would either fail to
serialize or come back as a different type than they went in — which is how a
resume position silently shifts. :func:`normalize_checkpoint` rejects anything
that would not survive the round-trip unchanged, at the point the connector
produces it rather than at the point a later run misreads it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

__all__ = ["Checkpoint", "JsonScalar", "normalize_checkpoint"]

type JsonScalar = str | int | float | bool | None
"""The value types a checkpoint entry may hold, all round-tripping through JSONB."""

type Checkpoint = Mapping[str, JsonScalar]
"""A flat, JSON-serializable mapping describing consumed-so-far position."""

_ALLOWED_VALUE_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))


def normalize_checkpoint(checkpoint: Checkpoint) -> dict[str, JsonScalar]:
    """Validate a checkpoint and return it as a plain ``dict`` for storage.

    Args:
        checkpoint: mapping produced by a connector.

    Returns:
        A shallow ``dict`` copy, safe to hand to the ``JSONB`` column.

    Raises:
        TypeError: if ``checkpoint`` is not a mapping, if any key is not a
            ``str``, or if any value is not a JSON scalar (``str``, ``int``,
            ``float``, ``bool``, ``None``). Notably ``datetime`` is rejected:
            it does not survive a JSONB round-trip as a ``datetime``, and a
            checkpoint that changes type between runs is a silent resume bug.
            Serialize instants to ISO-8601 strings explicitly instead.
    """
    # Widened to ``object`` on purpose. The annotation *promises* str keys and
    # JSON scalar values; this function exists to verify that promise against
    # what a connector actually produced at runtime, and a static type is not
    # evidence about a value parsed out of a vendor response.
    candidate: object = checkpoint
    if not isinstance(candidate, Mapping):
        msg = f"checkpoint must be a mapping of str -> JSON scalar; got {type(candidate).__name__}"
        raise TypeError(msg)
    normalized: dict[str, JsonScalar] = {}
    for key, value in candidate.items():
        if not isinstance(key, str):
            msg = f"checkpoint keys must be str; got {type(key).__name__} key {key!r}"
            raise TypeError(msg)
        if not isinstance(value, _ALLOWED_VALUE_TYPES):
            msg = (
                f"checkpoint value for {key!r} must be a JSON scalar "
                f"(str, int, float, bool, None); got {type(value).__name__} {value!r}. "
                "Serialize instants as ISO-8601 strings — a value that changes type "
                "across a JSONB round-trip silently moves the resume position"
            )
            raise TypeError(msg)
        normalized[key] = cast("JsonScalar", value)
    return normalized
