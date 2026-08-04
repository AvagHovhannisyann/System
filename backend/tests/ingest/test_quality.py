"""Knowledge-time policy declaration and the live-run lag check (P3.1, D-011).

D-011 declines to enforce a ``knowledge_time`` / ``ingested_at`` relation in
the database because backfills legitimately break it. The lag check is the
compensating control it promises instead: it exists to make a live feed that
has quietly fallen behind *visible*, and these tests pin down the boundary at
which it fires and the cases where it must report "not measured" rather than
"zero".
"""

from __future__ import annotations

import datetime as dt

import pytest

from backend.ingest.quality import (
    DataQualityMetric,
    KnowledgeTimePolicy,
    knowledge_time_lag_report,
    metrics_as_json,
)

_NOW = dt.datetime(2026, 8, 1, 12, 0, tzinfo=dt.UTC)
_POLICY = KnowledgeTimePolicy(
    description="probe: source-provided availability timestamp, converted to UTC",
    max_live_lag=dt.timedelta(hours=1),
)


def test_policy_requires_a_description() -> None:
    """A connector must state how it derives knowledge_time, not merely have one."""
    with pytest.raises(ValueError, match="must state how knowledge_time is derived"):
        KnowledgeTimePolicy(description="   ", max_live_lag=dt.timedelta(hours=1))


def test_policy_rejects_a_negative_lag() -> None:
    with pytest.raises(ValueError, match="max_live_lag"):
        KnowledgeTimePolicy(description="probe", max_live_lag=dt.timedelta(seconds=-1))


def test_lag_within_the_declared_bound_is_not_flagged() -> None:
    report = knowledge_time_lag_report(
        [_NOW - dt.timedelta(minutes=30), _NOW - dt.timedelta(minutes=5)],
        observed_at=_NOW,
        policy=_POLICY,
    )
    assert report.rows_considered == 2
    assert report.observed_max_lag == dt.timedelta(minutes=30)
    assert report.declared_max_lag == dt.timedelta(hours=1)
    assert not report.breached


def test_lag_exactly_at_the_bound_is_not_a_breach() -> None:
    """The declared lag is an inclusive allowance; the boundary is pinned here."""
    report = knowledge_time_lag_report(
        [_NOW - dt.timedelta(hours=1)], observed_at=_NOW, policy=_POLICY
    )
    assert report.observed_max_lag == dt.timedelta(hours=1)
    assert not report.breached


def test_lag_beyond_the_bound_is_flagged() -> None:
    report = knowledge_time_lag_report(
        [_NOW - dt.timedelta(minutes=1), _NOW - dt.timedelta(hours=26)],
        observed_at=_NOW,
        policy=_POLICY,
    )
    assert report.observed_max_lag == dt.timedelta(hours=26)
    assert report.breached


def test_no_rows_reports_no_measurement_rather_than_zero_lag() -> None:
    """Absence of evidence must not be recorded as evidence of timeliness."""
    report = knowledge_time_lag_report([], observed_at=_NOW, policy=_POLICY)
    assert report.rows_considered == 0
    assert report.observed_max_lag is None
    assert not report.breached


def test_naive_datetimes_are_rejected() -> None:
    naive = dt.datetime(2026, 8, 1, 12, 0)  # noqa: DTZ001 — the point of the test
    with pytest.raises(TypeError, match="timezone-aware"):
        knowledge_time_lag_report([naive], observed_at=_NOW, policy=_POLICY)
    with pytest.raises(TypeError, match="observed_at must be timezone-aware"):
        knowledge_time_lag_report([_NOW], observed_at=naive, policy=_POLICY)


def test_metrics_render_value_and_unit_together() -> None:
    """Every stored number carries its unit — directive section 8."""
    rendered = metrics_as_json(
        [
            DataQualityMetric("rows_written", 12, "rows"),
            DataQualityMetric("knowledge_time_max_lag", 3.5, "seconds"),
        ]
    )
    assert rendered == {
        "rows_written": {"value": 12, "unit": "rows"},
        "knowledge_time_max_lag": {"value": 3.5, "unit": "seconds"},
    }


def test_duplicate_metric_names_are_rejected() -> None:
    """Last-one-wins would silently drop a measurement."""
    with pytest.raises(ValueError, match="duplicate data-quality metric"):
        metrics_as_json(
            [
                DataQualityMetric("rows_written", 1, "rows"),
                DataQualityMetric("rows_written", 2, "rows"),
            ]
        )
