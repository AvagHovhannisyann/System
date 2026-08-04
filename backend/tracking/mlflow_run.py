"""MLflow run wrapper — every run carries a reproducibility stamp (CC.3, I2).

What this adds over calling MLflow directly
-------------------------------------------

MLflow will happily record a run with no provenance at all. This wrapper will
not: :func:`tracked_run` takes a :class:`~backend.tracking.stamp.ReproducibilityStamp`
and refuses **before creating the run** if it cannot get one. A refused run
leaves nothing behind, so the store never contains a result whose origin is
unknown. Recording first and hoping to backfill the provenance later is how
unreproducible numbers end up in a comparison table.

The stamp is written as ``repro.*`` tags, which makes the four I2 components
searchable in the MLflow UI (``tags."repro.config_hash" = '…'`` finds every run
of one configuration).

No server required
------------------

:func:`local_tracking_store` builds a serverless store under a directory —
suitable for tests and local work, and used by the test-suite in a ``tmp_path``.
Nothing here starts, requires, or contacts an MLflow server; the tracking URI is
always an explicit argument, so a run can never silently land in a default
``./mlruns`` nobody looks at.

Backend choice: **SQLite**, not the file store. MLflow 3.15 put the filesystem
backend in maintenance mode — constructing it raises unless
``MLFLOW_ALLOW_FILE_STORE=true``. ``sqlite:///…`` is the supported serverless
backend, needs no server, and is what the tests exercise. Artifacts still land
on the local filesystem, under an artifact root that is always passed
explicitly (MLflow's default with a SQL backend is ``./mlartifacts`` relative to
the current directory, which would scatter artifacts wherever a job happened to
be launched from).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from mlflow.tracking import MlflowClient

from backend.tracking.stamp import (
    ReproducibilityStamp,
    canonical_config_hash,
    canonical_config_json,
)

__all__ = [
    "CONFIG_ARTIFACT_NAME",
    "ConfigMismatchError",
    "LocalTrackingStore",
    "TrackedRun",
    "TrackingError",
    "UnstampedRunError",
    "local_tracking_store",
    "tracked_run",
]

CONFIG_ARTIFACT_NAME = "config.json"
"""Artifact name for the canonical JSON that the config hash was taken over."""

_FINISHED = "FINISHED"
_FAILED = "FAILED"


class TrackingError(RuntimeError):
    """Base class for tracking failures."""


class UnstampedRunError(TrackingError):
    """Raised when a run cannot be stamped with all four I2 components.

    Raised before the run is created, so no partial record exists.
    """


class ConfigMismatchError(TrackingError):
    """Raised when the config passed to a run does not hash to the stamp's hash.

    That mismatch means the stamp describes a different configuration from the
    one being run — the artifact would name a config it was not produced with,
    which is worse than not recording the config at all.
    """


@dataclass(frozen=True, slots=True)
class LocalTrackingStore:
    """A serverless MLflow store rooted at a local directory.

    Attributes:
        tracking_uri: SQLite URI for run metadata (``sqlite:///…/mlflow.db``).
        artifact_root: ``file://`` URI of the artifact directory.
    """

    tracking_uri: str
    artifact_root: str


def local_tracking_store(directory: Path) -> LocalTrackingStore:
    """Create (or point at) a serverless MLflow store under ``directory``.

    Creates ``directory`` and ``directory/artifacts`` if absent. The SQLite file
    is created by MLflow on first use, which also runs its schema migrations —
    the first call against a fresh database takes a couple of seconds.

    Assumes a POSIX path (the SQLite URI is built as ``sqlite:///<abs path>``).

    Args:
        directory: directory to hold ``mlflow.db`` and ``artifacts/``.

    Returns:
        A :class:`LocalTrackingStore`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = directory / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return LocalTrackingStore(
        tracking_uri=f"sqlite:///{(directory / 'mlflow.db').resolve()}",
        artifact_root=artifacts.resolve().as_uri(),
    )


@dataclass(frozen=True, slots=True)
class TrackedRun:
    """An open MLflow run that is known to carry a reproducibility stamp.

    Attributes:
        client: the MLflow client the run was created with.
        run_id: MLflow run id — a good value for the ledger's ``experiment id``
            column, since it links the row back to the recorded run.
        experiment_id: MLflow experiment id.
        stamp: the stamp recorded on the run.
    """

    client: MlflowClient
    run_id: str
    experiment_id: str
    stamp: ReproducibilityStamp

    @property
    def artifact_uri(self) -> str:
        """Return the run's artifact root URI."""
        uri: str | None = self.client.get_run(self.run_id).info.artifact_uri
        if uri is None:  # pragma: no cover - MLflow always sets this for a live run
            msg = f"run {self.run_id} has no artifact URI"
            raise TrackingError(msg)
        return uri

    def log_params(self, params: Mapping[str, object]) -> None:
        """Record run parameters, stringified.

        MLflow parameters are immutable: logging the same key twice with
        different values raises rather than overwriting.

        Args:
            params: parameter names to values. Values are converted with
                ``str``; ``None`` is refused, because the string ``"None"`` is
                indistinguishable from a parameter deliberately set to the
                literal text "None".

        Raises:
            TrackingError: if any value is ``None``.
        """
        for key, value in params.items():
            if value is None:
                msg = f"parameter {key!r} is None; record an explicit value or omit the parameter"
                raise TrackingError(msg)
        for key, value in params.items():
            self.client.log_param(self.run_id, key, str(value))

    def log_metrics(self, metrics: Mapping[str, float], *, step: int = 0) -> None:
        """Record run metrics.

        Args:
            metrics: metric names to values. Units are the caller's; state them
                in the metric name (``ic_mean``, ``sharpe_net_annualised``).
            step: optional step index for time-series metrics.

        Raises:
            TrackingError: if a value is not a finite real number. NaN and
                infinity are refused because MLflow stores them and the UI shows
                them as values, which reads as a measurement rather than as the
                failed computation it actually is.
        """
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                msg = f"metric {key!r} is {type(value).__name__}, not a real number"
                raise TrackingError(msg)
            if not math.isfinite(value):
                msg = f"metric {key!r} is {value}; a non-finite metric is a failed computation"
                raise TrackingError(msg)
        for key, value in metrics.items():
            self.client.log_metric(self.run_id, key, float(value), step=step)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        """Copy a local file or directory into the run's artifact store.

        Args:
            path: existing local file or directory.
            artifact_path: destination directory within the run's artifacts.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
        """
        if not path.exists():
            msg = f"artifact {path} does not exist"
            raise FileNotFoundError(msg)
        if path.is_dir():
            self.client.log_artifacts(self.run_id, str(path), artifact_path=artifact_path)
        else:
            self.client.log_artifact(self.run_id, str(path), artifact_path=artifact_path)

    def log_text(self, text: str, *, artifact_file: str) -> None:
        """Write ``text`` into the run's artifacts as ``artifact_file``.

        Args:
            text: the content to store.
            artifact_file: path within the run's artifacts, e.g. ``notes.md``.
        """
        self.client.log_text(self.run_id, text, artifact_file)


def _ensure_experiment(
    client: MlflowClient,
    name: str,
    artifact_location: str | None,
) -> str:
    """Return the id of experiment ``name``, creating it if needed.

    Args:
        client: MLflow client.
        name: experiment name.
        artifact_location: artifact root for a newly created experiment. Ignored
            when the experiment already exists (MLflow fixes it at creation).

    Returns:
        The experiment id.

    Raises:
        TrackingError: if the experiment exists but has been deleted. Logging
            into a deleted experiment hides the run from the UI's default view,
            which is indistinguishable from the run never having happened.
    """
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        if existing.lifecycle_stage != "active":
            msg = (
                f"experiment {name!r} is {existing.lifecycle_stage}; restore it or use "
                "another name rather than recording into a deleted experiment"
            )
            raise TrackingError(msg)
        return str(existing.experiment_id)
    return str(client.create_experiment(name, artifact_location=artifact_location))


@contextmanager
def tracked_run(
    *,
    tracking_uri: str,
    experiment: str,
    stamp: ReproducibilityStamp,
    run_name: str | None = None,
    artifact_location: str | None = None,
    config: Mapping[str, object] | None = None,
    require_clean: bool = False,
) -> Iterator[TrackedRun]:
    """Open an MLflow run carrying ``stamp``, or refuse to open one at all.

    The run is terminated ``FINISHED`` on a clean exit and ``FAILED`` if the body
    raises — the exception is re-raised, never swallowed. A crashed run that
    reads as finished would put a partial result next to complete ones.

    Args:
        tracking_uri: MLflow tracking URI. Always explicit; use
            :func:`local_tracking_store` for a serverless local store.
        experiment: experiment name, created if it does not exist.
        stamp: the four I2 components for this run.
        run_name: optional human-readable run name.
        artifact_location: artifact root used only when creating the experiment.
        config: the configuration ``stamp.config_hash`` was taken over. When
            given it is verified against the hash and stored as
            ``config.json``, so the hash has a preimage in the run rather than
            being an opaque identifier nobody can resolve.
        require_clean: when true, refuse to record a run from a dirty working
            tree. Off by default — during development most runs are dirty, and
            the honest record of that is the ``repro.git_dirty`` tag. Turn it on
            for runs whose results will be published.

    Yields:
        A :class:`TrackedRun`.

    Raises:
        UnstampedRunError: if ``stamp`` is not a
            :class:`~backend.tracking.stamp.ReproducibilityStamp`, or if
            ``require_clean`` is set and the tree was dirty.
        ConfigMismatchError: if ``config`` does not hash to ``stamp.config_hash``.
        TrackingError: if the experiment cannot be used.
    """
    # Checked at runtime, not only by the annotation: the refusal has to hold
    # for callers that reach this from untyped code, which is where an
    # unstamped run would otherwise slip through.
    supplied: object = stamp
    if not isinstance(supplied, ReproducibilityStamp):
        msg = (
            f"tracked_run requires a ReproducibilityStamp, got {type(supplied).__name__}; "
            "a run without git commit, data version, config hash and seed records a "
            "result nobody can regenerate (I2)"
        )
        raise UnstampedRunError(msg)
    if require_clean and stamp.git_dirty:
        msg = (
            f"working tree was dirty at {stamp.git_commit}; this run was configured to "
            "refuse results that cannot be regenerated from a commit"
        )
        raise UnstampedRunError(msg)
    if config is not None:
        actual = canonical_config_hash(config)
        if actual != stamp.config_hash:
            msg = (
                f"config hashes to {actual} but the stamp says {stamp.config_hash}; "
                "the stamp describes a different configuration from the one being run"
            )
            raise ConfigMismatchError(msg)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _ensure_experiment(client, experiment, artifact_location)
    run = client.create_run(
        experiment_id=experiment_id,
        tags=stamp.as_tags(),
        run_name=run_name,
    )
    tracked = TrackedRun(
        client=client,
        run_id=run.info.run_id,
        experiment_id=experiment_id,
        stamp=stamp,
    )
    if config is not None:
        tracked.log_text(canonical_config_json(config), artifact_file=CONFIG_ARTIFACT_NAME)
    try:
        yield tracked
    except BaseException:
        client.set_terminated(tracked.run_id, status=_FAILED)
        raise
    client.set_terminated(tracked.run_id, status=_FINISHED)
