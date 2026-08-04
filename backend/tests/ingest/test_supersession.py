"""Open-interval supersession: shape and validation of the correction rows (P3.1).

The *semantics* — that an as-of before the correction still sees the open
interval and one after sees the bounded pair — are proven against real
TimescaleDB in ``backend/tests/integration/test_ingest_write_path.py``. What
is checked here is the construction: which columns are copied, which are set,
and every way the helper must refuse rather than build a row that silently
corrupts the interval structure.
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.db.bitemporal import INFINITY
from backend.db.models import SecurityMaster
from backend.ingest.errors import FutureKnowledgeTimeError, SupersessionError
from backend.ingest.supersession import close_open_interval, supersede_open_interval

_LISTED = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
_RENAMED = dt.datetime(2024, 6, 1, tzinfo=dt.UTC)
_KNOWN_FIRST = dt.datetime(2020, 1, 2, tzinfo=dt.UTC)
_KNOWN_LATER = dt.datetime(2024, 6, 2, tzinfo=dt.UTC)
_NOW = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)


def _open_master(*, is_retraction: bool = False) -> SecurityMaster:
    """An open-ended securities-master identity: ``[2020-01-01, infinity)``."""
    return SecurityMaster(
        security_id=42,
        ticker="ABC",
        name="ABC Corp.",
        exchange="XNAS",
        first_listed_on=dt.date(2020, 1, 1),
        valid_from=_LISTED,
        valid_to=INFINITY,
        knowledge_time=_KNOWN_FIRST,
        is_retraction=is_retraction,
    )


# --- close_open_interval ----------------------------------------------------


def test_correction_repeats_the_key_and_valid_from_and_bounds_valid_to() -> None:
    """Same fact, later belief: only valid_to and knowledge_time may move."""
    original = _open_master()
    correction = close_open_interval(
        original, valid_to=_RENAMED, knowledge_time=_KNOWN_LATER, now=_NOW
    )
    assert correction is not original
    assert correction.security_id == original.security_id
    assert correction.valid_from == original.valid_from
    assert correction.valid_to == _RENAMED
    assert correction.knowledge_time == _KNOWN_LATER
    assert correction.ticker == original.ticker
    assert correction.name == original.name
    assert correction.exchange == original.exchange
    assert correction.first_listed_on == original.first_listed_on
    assert correction.is_retraction is False


def test_the_original_row_is_left_untouched() -> None:
    """Append-only: closing an interval must not mutate the row being corrected."""
    original = _open_master()
    close_open_interval(original, valid_to=_RENAMED, knowledge_time=_KNOWN_LATER, now=_NOW)
    assert original.valid_to == INFINITY
    assert original.knowledge_time == _KNOWN_FIRST


def test_ingested_at_is_not_copied_from_the_corrected_row() -> None:
    """ingested_at records when *this* row was written; copying it would misreport."""
    original = _open_master()
    original.ingested_at = dt.datetime(2020, 1, 3, tzinfo=dt.UTC)
    correction = close_open_interval(
        original, valid_to=_RENAMED, knowledge_time=_KNOWN_LATER, now=_NOW
    )
    assert correction.ingested_at is None


def test_closing_an_already_bounded_interval_is_refused() -> None:
    original = _open_master()
    original.valid_to = _RENAMED
    with pytest.raises(SupersessionError, match="not open-ended"):
        close_open_interval(
            original,
            valid_to=_RENAMED + dt.timedelta(days=1),
            knowledge_time=_KNOWN_LATER,
            now=_NOW,
        )


def test_closing_a_retraction_is_refused() -> None:
    with pytest.raises(SupersessionError, match="retraction"):
        close_open_interval(
            _open_master(is_retraction=True),
            valid_to=_RENAMED,
            knowledge_time=_KNOWN_LATER,
            now=_NOW,
        )


@pytest.mark.parametrize("valid_to", [_LISTED, _LISTED - dt.timedelta(days=1)])
def test_valid_to_must_be_strictly_after_valid_from(valid_to: dt.datetime) -> None:
    with pytest.raises(SupersessionError, match="strictly after valid_from"):
        close_open_interval(
            _open_master(), valid_to=valid_to, knowledge_time=_KNOWN_LATER, now=_NOW
        )


def test_closing_at_infinity_is_refused_as_a_no_op() -> None:
    with pytest.raises(SupersessionError, match="bounded instant"):
        close_open_interval(
            _open_master(), valid_to=INFINITY, knowledge_time=_KNOWN_LATER, now=_NOW
        )


@pytest.mark.parametrize(
    "knowledge_time", [_KNOWN_FIRST, _KNOWN_FIRST - dt.timedelta(microseconds=1)]
)
def test_knowledge_time_must_be_strictly_later(knowledge_time: dt.datetime) -> None:
    """Equal collides on the primary key; earlier could never win latest-knowledge."""
    with pytest.raises(SupersessionError, match="strictly later"):
        close_open_interval(
            _open_master(), valid_to=_RENAMED, knowledge_time=knowledge_time, now=_NOW
        )


def test_future_knowledge_time_is_refused() -> None:
    with pytest.raises(FutureKnowledgeTimeError):
        close_open_interval(
            _open_master(),
            valid_to=_RENAMED,
            knowledge_time=_NOW + dt.timedelta(seconds=1),
            now=_NOW,
        )


@pytest.mark.parametrize("argument", ["valid_to", "knowledge_time"])
def test_naive_datetimes_are_refused(argument: str) -> None:
    naive = dt.datetime(2024, 6, 1)  # noqa: DTZ001 — the point of the test
    kwargs: dict[str, dt.datetime] = {"valid_to": _RENAMED, "knowledge_time": _KNOWN_LATER}
    kwargs[argument] = naive
    with pytest.raises(TypeError, match="timezone-aware"):
        close_open_interval(_open_master(), now=_NOW, **kwargs)


# --- supersede_open_interval ------------------------------------------------


def test_supersession_produces_abutting_non_overlapping_intervals() -> None:
    """The defect this exists to prevent: two open intervals covering one instant."""
    original = _open_master()
    correction, successor = supersede_open_interval(
        original,
        boundary=_RENAMED,
        knowledge_time=_KNOWN_LATER,
        changes={"ticker": "XYZ", "name": "XYZ Corp."},
        now=_NOW,
    )
    assert (correction.valid_from, correction.valid_to) == (_LISTED, _RENAMED)
    assert (successor.valid_from, successor.valid_to) == (_RENAMED, INFINITY)
    assert correction.valid_to == successor.valid_from  # abut, half-open: no overlap
    assert correction.ticker == "ABC"
    assert successor.ticker == "XYZ"
    assert successor.name == "XYZ Corp."
    assert successor.exchange == original.exchange
    assert correction.knowledge_time == successor.knowledge_time == _KNOWN_LATER


def test_unchanged_payload_columns_carry_over_to_the_successor() -> None:
    correction, successor = supersede_open_interval(
        _open_master(),
        boundary=_RENAMED,
        knowledge_time=_KNOWN_LATER,
        changes={"ticker": "XYZ"},
        now=_NOW,
    )
    assert successor.name == correction.name == "ABC Corp."
    assert successor.first_listed_on == dt.date(2020, 1, 1)


def test_a_supersession_that_changes_nothing_is_refused() -> None:
    with pytest.raises(SupersessionError, match="at least one changed payload column"):
        supersede_open_interval(
            _open_master(), boundary=_RENAMED, knowledge_time=_KNOWN_LATER, changes={}, now=_NOW
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"security_id": 99},
        {"valid_from": _RENAMED},
        {"valid_to": _RENAMED},
        {"knowledge_time": _KNOWN_LATER},
        {"ingested_at": _KNOWN_LATER},
        {"is_retraction": True},
        {"no_such_column": 1},
    ],
)
def test_changes_may_only_name_payload_columns(changes: dict[str, object]) -> None:
    """Key columns describe a different entity; temporal columns are ours to set."""
    with pytest.raises(SupersessionError, match="cannot be superseded"):
        supersede_open_interval(
            _open_master(),
            boundary=_RENAMED,
            knowledge_time=_KNOWN_LATER,
            changes=changes,
            now=_NOW,
        )


def test_supersession_inherits_every_close_validation() -> None:
    """The close half is not bypassed by going through the combined helper."""
    with pytest.raises(SupersessionError, match="strictly later"):
        supersede_open_interval(
            _open_master(),
            boundary=_RENAMED,
            knowledge_time=_KNOWN_FIRST,
            changes={"ticker": "XYZ"},
            now=_NOW,
        )
