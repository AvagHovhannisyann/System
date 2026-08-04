"""FRED/ALFRED macro-series connector (P3.8).

Public surface of ``backend.ingest.fred``. Three modules:

- :mod:`backend.ingest.fred.client` — key-gated HTTP access. FRED requires an
  API key on every endpoint (verified live), so the client refuses to be
  constructed without one and never falls back to a keyless or invented path.
  Every outward string is rendered with the key **removed**, not masked
  (invariant I5).
- :mod:`backend.ingest.fred.parse` — response parsing and, most importantly,
  the derivation of ``knowledge_time`` from an ALFRED **vintage date** under a
  documented conservative lag (D-011's rule for a date-only source).
- :mod:`backend.ingest.fred.connector` — the connector itself, which ingests
  the **complete revision history** rather than current values, so a revision
  becomes a later-knowledge row and an ``as_of`` before it returns the number
  that was believed then.

The single idea worth carrying out of this package: macro data is the easiest
place in this platform to introduce lookahead, because macro series are
revised and the revised series still looks like a perfectly ordinary time
series. Storing only the latest values would make every backtest quietly
better than it should be. Vintages are what prevent that, and ALFRED real-time
periods are how FRED exposes them.
"""

from __future__ import annotations

from backend.ingest.fred.client import (
    FRED_API_BASE,
    FRED_API_KEY_ENV,
    FredApiKeyNotConfiguredError,
    FredClient,
    redacted_url,
    resolve_fred_api_key,
)
from backend.ingest.fred.connector import (
    CHECKPOINT_VINTAGE_PREFIX,
    DEFAULT_SERIES_IDS,
    FRED_SOURCE,
    FredConnector,
)
from backend.ingest.fred.parse import (
    EARLIEST_REALTIME_DATE,
    LATEST_REALTIME_DATE,
    MISSING_VALUE_MARKER,
    OBSERVATIONS_OUTPUT_TYPE,
    SERIES_DEFINITION_VALID_FROM,
    VINTAGE_TIMEZONE,
    Observation,
    ObservationPage,
    SeriesVersion,
    observation_date_to_valid_from,
    observations_page_limit,
    parse_observations,
    parse_series_versions,
    vintage_date_to_knowledge_time,
)

__all__ = [
    "CHECKPOINT_VINTAGE_PREFIX",
    "DEFAULT_SERIES_IDS",
    "EARLIEST_REALTIME_DATE",
    "FRED_API_BASE",
    "FRED_API_KEY_ENV",
    "FRED_SOURCE",
    "LATEST_REALTIME_DATE",
    "MISSING_VALUE_MARKER",
    "OBSERVATIONS_OUTPUT_TYPE",
    "SERIES_DEFINITION_VALID_FROM",
    "VINTAGE_TIMEZONE",
    "FredApiKeyNotConfiguredError",
    "FredClient",
    "FredConnector",
    "Observation",
    "ObservationPage",
    "SeriesVersion",
    "observation_date_to_valid_from",
    "observations_page_limit",
    "parse_observations",
    "parse_series_versions",
    "redacted_url",
    "resolve_fred_api_key",
    "vintage_date_to_knowledge_time",
]
