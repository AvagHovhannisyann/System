"""Checkpoint validation (P3.1).

The JSONB round-trip itself is exercised against real Postgres in
``backend/tests/integration/test_ingestion_runs.py``; here the concern is the
validation that keeps a checkpoint round-trippable in the first place.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest

from backend.ingest.checkpoint import JsonScalar, normalize_checkpoint


def test_scalar_values_round_trip_unchanged() -> None:
    checkpoint: dict[str, JsonScalar] = {
        "last_index_date": "2024-01-05",
        "documents_seen": 412,
        "coverage_fraction": 0.98,
        "complete": True,
        "next_cursor": None,
    }
    assert normalize_checkpoint(checkpoint) == checkpoint


def test_result_is_a_plain_dict_copy() -> None:
    """The stored value must not alias the connector's own mutable state."""
    source: dict[str, Any] = {"page": 1}
    normalized = normalize_checkpoint(source)
    source["page"] = 2
    assert normalized == {"page": 1}
    assert isinstance(normalized, dict)


def test_empty_checkpoint_is_valid() -> None:
    """A source may legitimately checkpoint 'nothing consumed but I ran'."""
    assert normalize_checkpoint({}) == {}


@pytest.mark.parametrize(
    "value",
    [
        dt.datetime(2024, 1, 5, tzinfo=dt.UTC),
        dt.date(2024, 1, 5),
        Decimal("1.5"),
        ["a", "b"],
        {"nested": 1},
        object(),
    ],
)
def test_non_scalar_values_are_rejected(value: object) -> None:
    """A value that changes type across JSONB silently moves the resume position."""
    with pytest.raises(TypeError, match="must be a JSON scalar"):
        normalize_checkpoint({"position": value})  # type: ignore[dict-item]


def test_non_string_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="keys must be str"):
        normalize_checkpoint({1: "x"})  # type: ignore[dict-item]


@pytest.mark.parametrize("value", ["not-a-mapping", 42, None, ["a"]])
def test_non_mapping_checkpoint_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        normalize_checkpoint(value)  # type: ignore[arg-type]
