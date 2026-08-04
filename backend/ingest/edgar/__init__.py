"""SEC EDGAR connector and its data-quality report (P3.2, P3.9).

The reference implementation of the P3.1 connector contract, chosen first
because acceptance timestamps are the hardest temporal-correctness problem in
Phase 3 and because a mistake in them is invisible in every test that does not
specifically look for it.

- :mod:`backend.ingest.edgar.parse` — pure parsing of EDGAR's published
  artifacts, and the US/Eastern-to-UTC conversion that produces
  ``knowledge_time``. Read its docstring first: it records how the timezone was
  verified against the live service, and why ``data.sec.gov`` is not used.
- :mod:`backend.ingest.edgar.client` — HTTP access with SEC's fair-access
  policy enforced fail-closed (no contact string configured, no requests made).
- :mod:`backend.ingest.edgar.connector` — the connector itself: incremental
  sync over the daily index, resumable by checkpoint, idempotent on re-write.
- :mod:`backend.ingest.edgar.report` — P3.9's per-source coverage, gap and
  staleness report, in which every figure is a measurement or ``None``.

Importing this package registers the connector under ``sec_edgar``.
"""

from __future__ import annotations

from backend.ingest.edgar.client import (
    EdgarClient,
    SecUserAgentNotConfiguredError,
    resolve_sec_user_agent,
)
from backend.ingest.edgar.connector import (
    CHECKPOINT_LAST_INDEX_DATE,
    DEFAULT_FORM_TYPES,
    SEC_EDGAR_SOURCE,
    SecEdgarConnector,
)
from backend.ingest.edgar.parse import (
    EARLIEST_ISO_NAMED_DAILY_INDEX,
    EDGAR_TIMEZONE,
    DailyIndexEntry,
    DailyIndexFile,
    DailyIndexParse,
    FilingDocument,
    FilingHeader,
    QuarterListing,
    acceptance_datetime_to_utc,
    parse_daily_index,
    parse_filing_header,
    parse_quarter_listing,
)
from backend.ingest.edgar.report import (
    CoverageWindow,
    Gap,
    KnowledgeTimeLagObservation,
    RunSummary,
    SourceQualityReport,
    Staleness,
    build_edgar_report,
    detect_gaps,
    edgar_coverage,
    run_history,
)

__all__ = [
    "CHECKPOINT_LAST_INDEX_DATE",
    "DEFAULT_FORM_TYPES",
    "EARLIEST_ISO_NAMED_DAILY_INDEX",
    "EDGAR_TIMEZONE",
    "SEC_EDGAR_SOURCE",
    "CoverageWindow",
    "DailyIndexEntry",
    "DailyIndexFile",
    "DailyIndexParse",
    "EdgarClient",
    "FilingDocument",
    "FilingHeader",
    "Gap",
    "KnowledgeTimeLagObservation",
    "QuarterListing",
    "RunSummary",
    "SecEdgarConnector",
    "SecUserAgentNotConfiguredError",
    "SourceQualityReport",
    "Staleness",
    "acceptance_datetime_to_utc",
    "build_edgar_report",
    "detect_gaps",
    "edgar_coverage",
    "parse_daily_index",
    "parse_filing_header",
    "parse_quarter_listing",
    "resolve_sec_user_agent",
    "run_history",
]
