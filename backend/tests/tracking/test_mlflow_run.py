"""Tests for the MLflow run wrapper (CC.3) against a local, serverless store.

No MLflow server is started or contacted: the store is a SQLite file plus an
artifact directory under ``tmp_path``. Two things are being pinned. Params,
metrics and artifacts must survive a round trip — otherwise the tracking layer
is decoration. And a run that cannot be stamped with all four I2 components must
leave *nothing* behind: refusing after creating the run would still put an
unattributable result in the store.
"""

from __future__ import annotations

from pathlib import Path

import mlflow.artifacts
import pytest
from mlflow.tracking import MlflowClient

from backend.tracking.mlflow_run import (
    CONFIG_ARTIFACT_NAME,
    ConfigMismatchError,
    LocalTrackingStore,
    TrackingError,
    UnstampedRunError,
    local_tracking_store,
    tracked_run,
)
from backend.tracking.stamp import (
    IncompleteStampError,
    ReproducibilityStamp,
    canonical_config_hash,
)

CONFIG: dict[str, object] = {"model": "lightgbm", "params": {"max_depth": 4}}


def _stamp(*, dirty: bool = False, config: dict[str, object] | None = None) -> ReproducibilityStamp:
    return ReproducibilityStamp(
        git_commit="0" * 40,
        git_dirty=dirty,
        data_version="dvc:abc123",
        config_hash=canonical_config_hash(CONFIG if config is None else config),
        seed=42,
    )


@pytest.fixture(scope="session")
def store(tmp_path_factory: pytest.TempPathFactory) -> LocalTrackingStore:
    """A serverless MLflow store shared by the module.

    Session-scoped because creating the SQLite backend runs MLflow's schema
    migrations (a couple of seconds); each test uses its own experiment name for
    isolation.
    """
    return local_tracking_store(tmp_path_factory.mktemp("mlflow-store"))


@pytest.fixture
def client(store: LocalTrackingStore) -> MlflowClient:
    return MlflowClient(tracking_uri=store.tracking_uri)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_params_metrics_and_artifacts_round_trip(
    store: LocalTrackingStore,
    client: MlflowClient,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "feature_importance.csv"
    artifact.write_text("feature,gain\nmom_12_1,0.31\n", encoding="utf-8")
    stamp = _stamp()

    with tracked_run(
        tracking_uri=store.tracking_uri,
        experiment="round-trip",
        artifact_location=store.artifact_root,
        stamp=stamp,
        run_name="fold-0",
        config=CONFIG,
    ) as run:
        run.log_params({"max_depth": 4, "feature_fraction": 0.6, "objective": "rank"})
        run.log_metrics({"ic_mean": 0.031, "ic_t_stat": 2.4})
        run.log_artifact(artifact)
        run.log_text("free-form note", artifact_file="notes.txt")
        run_id = run.run_id

    recorded = client.get_run(run_id)

    assert recorded.info.status == "FINISHED"
    assert recorded.info.run_name == "fold-0"
    assert recorded.data.params == {
        "max_depth": "4",
        "feature_fraction": "0.6",
        "objective": "rank",
    }
    assert recorded.data.metrics == pytest.approx({"ic_mean": 0.031, "ic_t_stat": 2.4})

    stored = {item.path for item in client.list_artifacts(run_id)}
    assert {"feature_importance.csv", "notes.txt", CONFIG_ARTIFACT_NAME} <= stored

    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="feature_importance.csv",
            tracking_uri=store.tracking_uri,
            dst_path=str(tmp_path / "downloaded"),
        )
    )
    assert downloaded.read_text(encoding="utf-8") == artifact.read_text(encoding="utf-8")
    assert recorded.info.artifact_uri is not None


def test_a_directory_of_artifacts_round_trips(
    store: LocalTrackingStore,
    client: MlflowClient,
    tmp_path: Path,
) -> None:
    folds = tmp_path / "folds"
    folds.mkdir()
    (folds / "fold_0.csv").write_text("0.1\n", encoding="utf-8")
    (folds / "fold_1.csv").write_text("0.2\n", encoding="utf-8")

    with tracked_run(
        tracking_uri=store.tracking_uri,
        experiment="artifact-directory",
        artifact_location=store.artifact_root,
        stamp=_stamp(),
    ) as run:
        run.log_artifact(folds, artifact_path="folds")
        assert run.artifact_uri.startswith("file://")
        run_id = run.run_id

    stored = {item.path for item in client.list_artifacts(run_id, path="folds")}
    assert stored == {"folds/fold_0.csv", "folds/fold_1.csv"}


def test_recording_into_a_deleted_experiment_is_refused(
    store: LocalTrackingStore,
    client: MlflowClient,
) -> None:
    # A run hidden in a deleted experiment is indistinguishable, in the UI's
    # default view, from a run that never happened.
    experiment_id = client.create_experiment(
        "deleted-experiment",
        artifact_location=store.artifact_root,
    )
    client.delete_experiment(experiment_id)

    with (
        pytest.raises(TrackingError),
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="deleted-experiment",
            artifact_location=store.artifact_root,
            stamp=_stamp(),
        ),
    ):
        pass  # pragma: no cover - refused before yielding


def test_every_run_carries_the_four_i2_components_as_tags(
    store: LocalTrackingStore,
    client: MlflowClient,
) -> None:
    stamp = _stamp()

    with tracked_run(
        tracking_uri=store.tracking_uri,
        experiment="stamped",
        artifact_location=store.artifact_root,
        stamp=stamp,
    ) as run:
        run_id = run.run_id

    tags = client.get_run(run_id).data.tags

    assert tags["repro.git_commit"] == stamp.git_commit
    assert tags["repro.data_version"] == stamp.data_version
    assert tags["repro.config_hash"] == stamp.config_hash
    assert tags["repro.seed"] == "42"
    assert tags["repro.reproducible"] == "true"


def test_the_config_hash_has_a_preimage_stored_with_the_run(
    store: LocalTrackingStore,
    tmp_path: Path,
) -> None:
    with tracked_run(
        tracking_uri=store.tracking_uri,
        experiment="config-artifact",
        artifact_location=store.artifact_root,
        stamp=_stamp(),
        config=CONFIG,
    ) as run:
        run_id = run.run_id

    stored = Path(
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=CONFIG_ARTIFACT_NAME,
            tracking_uri=store.tracking_uri,
            dst_path=str(tmp_path / "cfg"),
        )
    ).read_text(encoding="utf-8")

    assert stored == '{"model":"lightgbm","params":{"max_depth":4}}'


def test_a_dirty_tree_is_recorded_as_not_reproducible(
    store: LocalTrackingStore,
    client: MlflowClient,
) -> None:
    with tracked_run(
        tracking_uri=store.tracking_uri,
        experiment="dirty-tree",
        artifact_location=store.artifact_root,
        stamp=_stamp(dirty=True),
    ) as run:
        run_id = run.run_id

    tags = client.get_run(run_id).data.tags

    assert tags["repro.git_dirty"] == "true"
    assert tags["repro.reproducible"] == "false"
    assert tags["repro.git_reference"].endswith("-dirty")


# ---------------------------------------------------------------------------
# Refusals — nothing is recorded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stamp",
    [
        pytest.param(None, id="no-stamp"),
        pytest.param("0" * 40, id="bare-commit-string"),
        pytest.param({"git_commit": "0" * 40, "seed": 1}, id="dict-that-looks-like-a-stamp"),
    ],
)
def test_a_run_without_a_stamp_is_refused_and_records_nothing(
    store: LocalTrackingStore,
    client: MlflowClient,
    stamp: object,
) -> None:
    experiment = f"unstamped-{type(stamp).__name__}"

    with (
        pytest.raises(UnstampedRunError),
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment=experiment,
            artifact_location=store.artifact_root,
            stamp=stamp,  # type: ignore[arg-type]
        ),
    ):
        pass  # pragma: no cover - the context manager must refuse before yielding

    assert client.get_experiment_by_name(experiment) is None


def test_an_incomplete_stamp_cannot_even_be_constructed() -> None:
    # The refusal happens one layer earlier than the run: there is no way to
    # hand tracked_run a stamp that is missing one of the four components.
    with pytest.raises(IncompleteStampError):
        ReproducibilityStamp(
            git_commit="0" * 40,
            git_dirty=False,
            data_version="",
            config_hash="a" * 64,
            seed=42,
        )


def test_require_clean_refuses_a_dirty_tree_and_records_nothing(
    store: LocalTrackingStore,
    client: MlflowClient,
) -> None:
    with (
        pytest.raises(UnstampedRunError),
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="require-clean",
            artifact_location=store.artifact_root,
            stamp=_stamp(dirty=True),
            require_clean=True,
        ),
    ):
        pass  # pragma: no cover - refused before yielding

    assert client.get_experiment_by_name("require-clean") is None


def test_a_config_that_disagrees_with_the_stamp_is_refused(
    store: LocalTrackingStore,
    client: MlflowClient,
) -> None:
    with (
        pytest.raises(ConfigMismatchError),
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="config-mismatch",
            artifact_location=store.artifact_root,
            stamp=_stamp(),
            config={"model": "something-else"},
        ),
    ):
        pass  # pragma: no cover - refused before yielding

    assert client.get_experiment_by_name("config-mismatch") is None


def test_a_run_whose_body_raises_is_recorded_as_failed(
    store: LocalTrackingStore,
    client: MlflowClient,
) -> None:
    captured: dict[str, str] = {}

    def crash_inside_a_run() -> None:
        with tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="failing",
            artifact_location=store.artifact_root,
            stamp=_stamp(),
        ) as run:
            captured["run_id"] = run.run_id
            raise ZeroDivisionError

    with pytest.raises(ZeroDivisionError):
        crash_inside_a_run()

    # A crashed run that reads as FINISHED would sit next to complete results.
    assert client.get_run(captured["run_id"]).info.status == "FAILED"


@pytest.mark.parametrize(
    "metrics",
    [
        pytest.param({"sharpe": float("nan")}, id="nan"),
        pytest.param({"sharpe": float("inf")}, id="infinity"),
        pytest.param({"sharpe": "0.5"}, id="string"),
        pytest.param({"sharpe": True}, id="bool"),
    ],
)
def test_a_metric_that_is_not_a_finite_number_is_refused(
    store: LocalTrackingStore,
    metrics: dict[str, object],
) -> None:
    with (
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="bad-metrics",
            artifact_location=store.artifact_root,
            stamp=_stamp(),
        ) as run,
        pytest.raises(TrackingError),
    ):
        run.log_metrics(metrics)  # type: ignore[arg-type]


def test_a_none_parameter_is_refused(store: LocalTrackingStore) -> None:
    with (
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="none-param",
            artifact_location=store.artifact_root,
            stamp=_stamp(),
        ) as run,
        pytest.raises(TrackingError),
    ):
        run.log_params({"max_depth": None})


def test_a_missing_artifact_file_is_refused(store: LocalTrackingStore, tmp_path: Path) -> None:
    with (
        tracked_run(
            tracking_uri=store.tracking_uri,
            experiment="missing-artifact",
            artifact_location=store.artifact_root,
            stamp=_stamp(),
        ) as run,
        pytest.raises(FileNotFoundError),
    ):
        run.log_artifact(tmp_path / "absent.csv")


def test_the_store_needs_no_server_and_lives_entirely_under_the_given_directory(
    tmp_path: Path,
) -> None:
    store = local_tracking_store(tmp_path / "nested" / "store")

    assert store.tracking_uri.startswith("sqlite:///")
    assert store.artifact_root.startswith("file://")
    assert (tmp_path / "nested" / "store" / "artifacts").is_dir()

    with tracked_run(
        tracking_uri=store.tracking_uri,
        experiment="isolated",
        artifact_location=store.artifact_root,
        stamp=_stamp(),
    ) as run:
        run.log_metrics({"ic_mean": 0.01})

    assert (tmp_path / "nested" / "store" / "mlflow.db").is_file()
