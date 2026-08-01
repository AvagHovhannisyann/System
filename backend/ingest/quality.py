"""Data-quality metrics and knowledge-time hygiene checks (P3.1, D-011).

Two things live here.

**1. The knowledge-time policy a connector must declare.** D-011 requires every
connector to state, in code, how it derives ``knowledge_time`` — EDGAR
acceptance timestamp, vendor availability timestamp, or a documented
conservative lag applied to a date-only field. :class:`KnowledgeTimePolicy`
carries that statement plus the connector's declared upper bound on how far
behind ingestion a *live* run's knowledge times may legitimately fall.

**2. The compensating control D-011 promises and Phase 2 did not build.**
D-011 explicitly declines to impose a database CHECK relating
``knowledge_time`` to ``ingested_at``, because backfills legitimately violate
any such relation (a 2016 filing ingested today has an honest 2016 knowledge
time). Instead it promises two things at the run level, both implemented here
and in :mod:`backend.ingest.write`:

- every run records whether it is a ``backfill`` or a ``live`` run
  (:mod:`backend.ingest.runs`); and
- **live** runs are checked: if the rows they wrote carry knowledge times
  trailing the write instant by more than the source's declared
  ``max_live_lag``, the run is *flagged* — the data is kept and the run still
  succeeds, because a lagging source is a source problem, not a corrupt row.
  What must never happen is that it goes unnoticed: a live feed quietly
  falling hours behind is how "point in time" silently becomes "point in
  time, minus a day", and every backtest built on it is optimistic.

The flag is a data-quality metric, not an exception. Rejecting would throw
away real data over a source's tardiness; ignoring would hide a drifting feed.
Flagging keeps both the data and the evidence, and P3.9's data-quality report
is its consumer.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DataQualityMetric",
    "KnowledgeTimeLagReport",
    "KnowledgeTimePolicy",
    "knowledge_time_lag_report",
    "metrics_as_json",
]


@dataclass(frozen=True, slots=True)
class KnowledgeTimePolicy:
    """How one connector derives ``knowledge_time``, and its expected live lag.

    Attributes:
        description: prose statement of the derivation, displayed by the
            data-quality report (D-011 requires the policy to be visible, not
            merely implemented). Example: "SEC EDGAR acceptance timestamp from
            the daily index, converted from US/Eastern to UTC".
        max_live_lag: declared upper bound on ``write_instant -
            knowledge_time`` for rows written by a **live** run. A run whose
            observed maximum lag exceeds this is flagged (never rejected — see
            the module docstring). Backfill runs are exempt by construction:
            their knowledge times are historical on purpose.
    """

    description: str
    max_live_lag: dt.timedelta

    def __post_init__(self) -> None:
        """Validate the declaration; an empty or negative policy is a defect."""
        if not self.description.strip():
            msg = "KnowledgeTimePolicy.description must state how knowledge_time is derived"
            raise ValueError(msg)
        if self.max_live_lag < dt.timedelta(0):
            msg = f"max_live_lag must be >= 0; got {self.max_live_lag}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DataQualityMetric:
    """One measurement emitted by an ingestion run.

    Attributes:
        name: stable identifier, dot-free snake_case (e.g.
            ``rows_written``), used as the JSON key on the run record.
        value: the measured value. Units are carried separately and are never
            implied by the name.
        unit: the unit of ``value`` — e.g. ``"rows"``, ``"seconds"``,
            ``"count"``, ``"boolean"``. Explicit because the directive's §8
            unit rule applies to every number this system reports, not only
            to financial ones.
    """

    name: str
    value: float | int | bool | str | None
    unit: str


@dataclass(frozen=True, slots=True)
class KnowledgeTimeLagReport:
    """Result of the live-run knowledge-time lag check.

    Attributes:
        rows_considered: number of rows whose ``knowledge_time`` was examined
            (dimensionless count).
        observed_max_lag: the largest ``observed_at - knowledge_time`` across
            those rows, or ``None`` when no rows were written (no evidence is
            not the same as zero lag, and must not be reported as such).
        declared_max_lag: the connector's declared bound, copied here so the
            report is self-contained.
        breached: ``True`` when ``observed_max_lag`` exceeds
            ``declared_max_lag``. ``False`` when it does not, and ``False``
            when there were no rows — with ``observed_max_lag`` ``None``
            making the absence of evidence visible rather than implied.
    """

    rows_considered: int
    observed_max_lag: dt.timedelta | None
    declared_max_lag: dt.timedelta
    breached: bool


def knowledge_time_lag_report(
    knowledge_times: Iterable[dt.datetime],
    *,
    observed_at: dt.datetime,
    policy: KnowledgeTimePolicy,
) -> KnowledgeTimeLagReport:
    """Measure how far written knowledge times trail the write instant.

    Args:
        knowledge_times: the ``knowledge_time`` of every row written by the
            run. All must be timezone-aware (the write path guarantees it).
        observed_at: the instant the rows were durably written — the practical
            stand-in for ``ingested_at``, which the database stamps a few
            milliseconds earlier. Must be timezone-aware.
        policy: the connector's declared knowledge-time policy.

    Returns:
        A :class:`KnowledgeTimeLagReport`. Lag is measured in whole
        :class:`datetime.timedelta` units (not seconds) so no precision is
        lost on the way to the report.

    Raises:
        TypeError: if ``observed_at`` or any knowledge time is naive.
            Comparing a naive against an aware datetime raises anyway; failing
            here says *why*.
    """
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        msg = f"observed_at must be timezone-aware; got naive {observed_at!r}"
        raise TypeError(msg)
    max_lag: dt.timedelta | None = None
    considered = 0
    for knowledge_time in knowledge_times:
        if knowledge_time.tzinfo is None or knowledge_time.utcoffset() is None:
            msg = f"knowledge_time must be timezone-aware; got naive {knowledge_time!r}"
            raise TypeError(msg)
        considered += 1
        lag = observed_at - knowledge_time
        if max_lag is None or lag > max_lag:
            max_lag = lag
    return KnowledgeTimeLagReport(
        rows_considered=considered,
        observed_max_lag=max_lag,
        declared_max_lag=policy.max_live_lag,
        breached=max_lag is not None and max_lag > policy.max_live_lag,
    )


def metrics_as_json(metrics: Sequence[DataQualityMetric]) -> dict[str, dict[str, object]]:
    """Render metrics for the run record's ``JSONB`` column.

    Args:
        metrics: the metrics emitted by one run.

    Returns:
        ``{metric name: {"value": ..., "unit": ...}}``. Keeping the unit
        beside every value means a later reader (P3.9's report, the Data
        Health page) cannot misread seconds as milliseconds.

    Raises:
        ValueError: on a duplicate metric name — silently keeping the last
            one would drop a measurement.
    """
    rendered: dict[str, dict[str, object]] = {}
    for metric in metrics:
        if metric.name in rendered:
            msg = f"duplicate data-quality metric name {metric.name!r}"
            raise ValueError(msg)
        rendered[metric.name] = {"value": metric.value, "unit": metric.unit}
    return rendered
