"""The idempotency key: derived from content, stable across processes, sensitive to every field."""

from __future__ import annotations

import datetime as dt
import json
import re
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from backend.execution.idempotency import (
    IDEMPOTENCY_SCHEMA,
    idempotency_key,
    idempotency_preimage,
    key_of_preimage,
)
from backend.execution.orders import OrderType, Side, TimeInForce
from backend.tests.execution.fixtures import CONFIG_HASH_B, make_intent, make_stamp

HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def test_the_key_is_a_sha256_of_the_preimage() -> None:
    intent = make_intent()
    preimage = idempotency_preimage(intent)
    key = idempotency_key(intent)
    assert HEX64.fullmatch(key)
    assert key_of_preimage(preimage) == key


def test_the_key_is_stable_across_repeated_computation() -> None:
    # The retry's whole safety rests on this: the second attempt recomputes the
    # same key from the same instruction, with nothing remembered.
    intent = make_intent()
    assert len({idempotency_key(intent) for _ in range(50)}) == 1
    assert idempotency_key(intent) == idempotency_key(make_intent())


def test_the_preimage_names_the_schema_version_first_field() -> None:
    body = json.loads(idempotency_preimage(make_intent()))
    assert body["schema"] == IDEMPOTENCY_SCHEMA
    assert IDEMPOTENCY_SCHEMA.strip() != ""


def test_the_preimage_carries_no_clock_counter_or_identifier() -> None:
    # The failure mode this test exists for: someone adds `submitted_at` or an
    # attempt counter "for traceability", and every retry becomes a new order.
    body = json.loads(idempotency_preimage(make_intent()))
    forbidden = (
        "time",
        "timestamp",
        "at",
        "now",
        "clock",
        "counter",
        "sequence",
        "attempt",
        "nonce",
        "uuid",
        "random",
        "order_id",
        "pid",
        "host",
    )
    for field in body:
        assert field == "rebalance_date" or not any(
            token == field or field.endswith(f"_{token}") for token in forbidden
        ), field


def test_the_preimage_is_canonical_json_with_sorted_keys() -> None:
    preimage = idempotency_preimage(make_intent())
    body = json.loads(preimage)
    assert preimage == json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert list(body) == sorted(body)


def test_the_preimage_holds_no_floats() -> None:
    # A binary float has no exact decimal rendering, so a price round-tripped
    # through one would key differently on two machines.
    body = json.loads(idempotency_preimage(make_intent()))
    for name, value in body.items():
        assert not isinstance(value, float), name


def test_the_venue_is_recorded_and_is_the_paper_venue() -> None:
    body = json.loads(idempotency_preimage(make_intent()))
    assert body["venue"] == "paper"


def test_the_stamp_travels_into_the_key() -> None:
    # I2: a fill traces to the config and commit that produced it, and two runs
    # under different configs must not share an order identity.
    body = json.loads(idempotency_preimage(make_intent()))
    stamp = make_stamp()
    assert body["git_commit"] == stamp.git_commit
    assert body["git_dirty"] == stamp.git_dirty
    assert body["data_version"] == stamp.data_version
    assert body["config_hash"] == stamp.config_hash
    assert body["seed"] == stamp.seed


def test_equal_prices_written_differently_produce_one_key() -> None:
    # Decimal("123.45") and Decimal("123.4500") are the same price. Two keys for
    # one price is a retry that double-sends.
    assert idempotency_key(make_intent(limit_price_usd=Decimal("123.45"))) == idempotency_key(
        make_intent(limit_price_usd=Decimal("123.450000"))
    )
    assert idempotency_key(make_intent(limit_price_usd=Decimal("1E+2"))) == idempotency_key(
        make_intent(limit_price_usd=Decimal("100"))
    )


CHANGES = [
    ("security_id", {"security_id": 43}),
    ("side", {"side": Side.SELL}),
    ("quantity_shares", {"quantity_shares": 101}),
    ("order_type", {"order_type": OrderType.MARKET, "limit_price_usd": None}),
    ("time_in_force", {"time_in_force": TimeInForce.GTC}),
    ("limit_price_usd", {"limit_price_usd": Decimal("123.46")}),
    ("rebalance_date", {"rebalance_date": dt.date(2026, 8, 4)}),
    ("slice_index", {"slice_index": 1, "slice_count": 2}),
    ("slice_count", {"slice_count": 3}),
    ("git_commit", {"stamp": make_stamp(git_commit="c" * 40)}),
    ("git_dirty", {"stamp": make_stamp(git_dirty=True)}),
    ("data_version", {"stamp": make_stamp(data_version="sharadar-2026-08-02")}),
    ("config_hash", {"stamp": make_stamp(config_hash=CONFIG_HASH_B)}),
    ("seed", {"stamp": make_stamp(seed=8)}),
]


@pytest.mark.parametrize(("field", "override"), CHANGES, ids=[name for name, _ in CHANGES])
def test_changing_any_content_field_changes_the_key(
    field: str, override: dict[str, object]
) -> None:
    # Non-vacuity of the key: if a field can change without moving the key, two
    # different orders share an identity and one silently replaces the other.
    assert idempotency_key(make_intent()) != idempotency_key(make_intent(**override)), field


def test_every_key_produced_by_the_change_matrix_is_distinct() -> None:
    keys = {idempotency_key(make_intent())}
    for _, override in CHANGES:
        keys.add(idempotency_key(make_intent(**override)))
    # The market/limit change collapses two fields at once, so the matrix
    # produces one key per row plus the baseline.
    assert len(keys) == len(CHANGES) + 1


def test_slices_of_one_parent_are_distinct_orders() -> None:
    # P11.4 will submit N slices of one target. If they keyed identically the
    # store would absorb slices 2..N and the book would be short.
    keys = {idempotency_key(make_intent(slice_index=index, slice_count=5)) for index in range(5)}
    assert len(keys) == 5


def test_an_identical_replan_produces_identical_keys() -> None:
    # Same commit, same config, same data, same seed, same date: re-running the
    # plan is the same decision, and absorbing it is correct behaviour rather
    # than a limitation.
    first = [idempotency_key(make_intent(security_id=n)) for n in range(1, 20)]
    second = [idempotency_key(make_intent(security_id=n)) for n in range(1, 20)]
    assert first == second


@given(
    security_id=st.integers(min_value=1, max_value=10_000),
    quantity=st.integers(min_value=1, max_value=1_000_000),
    slice_count=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_distinct_orders_get_distinct_keys(
    security_id: int, quantity: int, slice_count: int, seed: int
) -> None:
    intent = make_intent(
        security_id=security_id,
        quantity_shares=quantity,
        slice_index=0,
        slice_count=slice_count,
        stamp=make_stamp(seed=seed),
    )
    other = make_intent(
        security_id=security_id,
        quantity_shares=quantity + 1,
        slice_index=0,
        slice_count=slice_count,
        stamp=make_stamp(seed=seed),
    )
    assert idempotency_key(intent) != idempotency_key(other)
    assert idempotency_key(intent) == idempotency_key(intent)
