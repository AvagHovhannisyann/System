"""Tests for the data-version component of I2, and the honest DVC report (CC.3).

The failure this module guards against is a *default* data version. If a run
that does not know which data it read can still produce a stamp, every artifact
looks reproducible and none of them are. So the only outcomes here are "a
recorded identifier" and "an exception".
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from backend.tracking.data_version import (
    DATA_VERSION_ENV_VAR,
    DataVersionError,
    DataVersionUnavailableError,
    default_pointer_path,
    dvc_availability,
    read_data_version,
    resolve_data_version,
    write_data_version,
)


def test_a_recorded_data_version_round_trips(tmp_path: Path) -> None:
    pointer = tmp_path / ".data-version"

    written = write_data_version(pointer, "dvc:9c1185a5c5e9fc54612808977ee8f548b2258d31")

    assert written == "dvc:9c1185a5c5e9fc54612808977ee8f548b2258d31"
    assert read_data_version(pointer) == written


def test_the_pointer_file_documents_itself(tmp_path: Path) -> None:
    pointer = tmp_path / ".data-version"
    write_data_version(pointer, "v1")

    text = pointer.read_text(encoding="utf-8")

    assert text.startswith("#")
    assert "I2" in text
    assert read_data_version(pointer) == "v1"


def test_recording_a_new_version_replaces_the_previous_one(tmp_path: Path) -> None:
    pointer = tmp_path / ".data-version"
    write_data_version(pointer, "v1")

    write_data_version(pointer, "v2")

    assert read_data_version(pointer) == "v2"


def test_a_missing_pointer_refuses_rather_than_returning_a_placeholder(tmp_path: Path) -> None:
    with pytest.raises(DataVersionUnavailableError):
        read_data_version(tmp_path / "absent")


def test_a_pointer_with_only_comments_refuses(tmp_path: Path) -> None:
    pointer = tmp_path / ".data-version"
    pointer.write_text("# nothing recorded yet\n\n", encoding="utf-8")

    with pytest.raises(DataVersionUnavailableError):
        read_data_version(pointer)


def test_an_ambiguous_pointer_refuses(tmp_path: Path) -> None:
    # Two identifiers means nobody knows which snapshot a result came from;
    # first-wins would pick one and look confident about it.
    pointer = tmp_path / ".data-version"
    pointer.write_text("v1\nv2\n", encoding="utf-8")

    with pytest.raises(DataVersionError):
        read_data_version(pointer)


@pytest.mark.parametrize(
    "version",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param("a|b", id="pipe-would-corrupt-a-ledger-cell"),
        pytest.param("a#b", id="hash-would-read-as-a-comment"),
        pytest.param("x" * 201, id="absurdly-long"),
    ],
)
def test_an_invalid_data_version_is_refused(tmp_path: Path, version: str) -> None:
    with pytest.raises(DataVersionError):
        write_data_version(tmp_path / ".data-version", version)


def test_the_environment_variable_wins_over_the_pointer(tmp_path: Path) -> None:
    pointer = tmp_path / ".data-version"
    write_data_version(pointer, "from-file")

    resolved = resolve_data_version(
        repo_root=tmp_path,
        env={DATA_VERSION_ENV_VAR: "from-env"},
        pointer_path=pointer,
    )

    assert resolved == "from-env"


def test_resolution_falls_back_to_the_pointer_file(tmp_path: Path) -> None:
    write_data_version(default_pointer_path(tmp_path), "from-file")

    assert resolve_data_version(repo_root=tmp_path, env={}) == "from-file"


def test_an_empty_environment_variable_does_not_count_as_a_version(tmp_path: Path) -> None:
    write_data_version(default_pointer_path(tmp_path), "from-file")

    resolved = resolve_data_version(repo_root=tmp_path, env={DATA_VERSION_ENV_VAR: "  "})

    assert resolved == "from-file"


def test_with_no_recorded_version_resolution_refuses_and_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(DataVersionUnavailableError) as excinfo:
        resolve_data_version(repo_root=tmp_path, env={})

    message = str(excinfo.value)
    assert DATA_VERSION_ENV_VAR in message
    assert "dvc" in message.lower()


# ---------------------------------------------------------------------------
# DVC availability — reported, never assumed
# ---------------------------------------------------------------------------


def test_dvc_availability_reports_the_real_state_of_the_tooling(tmp_path: Path) -> None:
    # Deliberately not asserting "DVC is absent": this test must keep telling
    # the truth after DVC is added to the dependencies.
    availability = dvc_availability(tmp_path)

    assert availability.package_importable is (importlib.util.find_spec("dvc") is not None)
    assert availability.cli_on_path is (shutil.which("dvc") is not None)
    assert availability.repository_initialised is False


def test_an_initialised_dvc_directory_is_detected(tmp_path: Path) -> None:
    (tmp_path / ".dvc").mkdir()

    assert dvc_availability(tmp_path).repository_initialised is True


def test_usable_requires_both_the_cli_and_an_initialised_repository(tmp_path: Path) -> None:
    (tmp_path / ".dvc").mkdir()
    availability = dvc_availability(tmp_path)

    assert availability.usable is (shutil.which("dvc") is not None)


def test_the_description_names_the_next_action_when_dvc_is_unusable(tmp_path: Path) -> None:
    description = dvc_availability(tmp_path).describe()

    assert "dvc init" in description or "dvc add" in description
