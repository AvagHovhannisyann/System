"""Experiment tracking and data versioning (CC.3) — the machinery behind I2.

I2 requires that any result be regenerable from *git commit + data version +
config hash + seed*, with all four stored alongside every artifact. This package
is where those four values are produced and attached:

- :mod:`backend.tracking.stamp` — the four components, validated together as a
  :class:`~backend.tracking.stamp.ReproducibilityStamp`, with a canonical config
  hash and a dirty-working-tree marker.
- :mod:`backend.tracking.data_version` — recording and retrieving the data
  version identifier, plus the DVC workflow and an honest report of whether DVC
  is installed here.
- :mod:`backend.tracking.mlflow_run` — MLflow runs that refuse to start without
  a stamp. No server required.
- :mod:`backend.tracking.ledger` — append-only writer for ``TESTING_LEDGER.md``,
  whose row count is the trial count the Deflated Sharpe Ratio deflates by.

Reading the ledger is not this package's job: :mod:`backend.backtest.ledger`
owns that and has no write path. This package appends through it, so the number
written in a row's ``#`` column is the number the DSR will see.
"""

from backend.tracking.data_version import (
    DATA_VERSION_ENV_VAR,
    DataVersionError,
    DataVersionUnavailableError,
    DvcAvailability,
    dvc_availability,
    read_data_version,
    resolve_data_version,
    write_data_version,
)
from backend.tracking.ledger import LEDGER_COLUMNS, LedgerAppendError, TrialRecord, append_trial
from backend.tracking.mlflow_run import (
    ConfigMismatchError,
    LocalTrackingStore,
    TrackedRun,
    TrackingError,
    UnstampedRunError,
    local_tracking_store,
    tracked_run,
)
from backend.tracking.stamp import (
    ConfigHashError,
    GitState,
    GitStateUnavailableError,
    IncompleteStampError,
    ReproducibilityStamp,
    StampError,
    canonical_config_hash,
    canonical_config_json,
    git_state,
)

__all__ = [
    "DATA_VERSION_ENV_VAR",
    "LEDGER_COLUMNS",
    "ConfigHashError",
    "ConfigMismatchError",
    "DataVersionError",
    "DataVersionUnavailableError",
    "DvcAvailability",
    "GitState",
    "GitStateUnavailableError",
    "IncompleteStampError",
    "LedgerAppendError",
    "LocalTrackingStore",
    "ReproducibilityStamp",
    "StampError",
    "TrackedRun",
    "TrackingError",
    "TrialRecord",
    "UnstampedRunError",
    "append_trial",
    "canonical_config_hash",
    "canonical_config_json",
    "dvc_availability",
    "git_state",
    "local_tracking_store",
    "read_data_version",
    "resolve_data_version",
    "tracked_run",
    "write_data_version",
]
