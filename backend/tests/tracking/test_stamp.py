"""Tests for the reproducibility stamp — I2's four components (CC.3).

Two properties carry the weight here. First, the config hash must be an
*identity*: the same configuration written in any key order must hash to the
same digest, or it cannot be used to recognise that two runs were the same
experiment. Second, a stamp taken from a dirty working tree must say so, because
a result produced from uncommitted code is not regenerable from its commit and
the stamp is the only place that fact can be recorded.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from backend.tracking.stamp import (
    ConfigHashError,
    GitStateUnavailableError,
    IncompleteStampError,
    ReproducibilityStamp,
    canonical_config_hash,
    canonical_config_json,
    git_state,
    repository_root,
)

VALID_COMMIT = "0" * 40
VALID_HASH = "a" * 64


def _git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` with identity flags, so no global config is needed."""
    executable = shutil.which("git")
    assert executable is not None, "git is required: the stamp's commit component depends on it"
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell; path from shutil.which
        [
            executable,
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=test",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout


@pytest.fixture
def committed_repo(tmp_path: Path) -> Path:
    """Return a fresh git repository with one commit and a clean tree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    (repo / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "initial")
    return repo


# ---------------------------------------------------------------------------
# Canonical config hashing
# ---------------------------------------------------------------------------


def test_same_config_in_different_key_orders_hashes_identically() -> None:
    first = {
        "model": "lightgbm",
        "params": {"max_depth": 4, "feature_fraction": 0.6, "lambda_l2": 1.0},
        "features": ["mom_12_1", "btp", "gp"],
    }
    second = {
        "features": ["mom_12_1", "btp", "gp"],
        "params": {"lambda_l2": 1.0, "feature_fraction": 0.6, "max_depth": 4},
        "model": "lightgbm",
    }

    assert first == second
    assert list(first) != list(second)  # the dicts really are ordered differently
    assert canonical_config_hash(first) == canonical_config_hash(second)
    assert canonical_config_json(first) == canonical_config_json(second)


def test_key_order_is_normalised_at_every_nesting_level() -> None:
    deep_a = {"a": {"b": {"c": 1, "d": 2}, "e": [{"f": 1, "g": 2}]}}
    deep_b = {"a": {"e": [{"g": 2, "f": 1}], "b": {"d": 2, "c": 1}}}

    assert canonical_config_hash(deep_a) == canonical_config_hash(deep_b)


@pytest.mark.parametrize(
    "other",
    [
        pytest.param({"model": "lightgbm", "seed_note": "changed value"}, id="value-changed"),
        pytest.param({"model": "lightgbm", "extra": 1}, id="key-added"),
        pytest.param({"model": "xgboost"}, id="model-changed"),
        pytest.param({"model": ["lightgbm"]}, id="type-changed"),
    ],
)
def test_a_different_config_hashes_differently(other: dict[str, object]) -> None:
    baseline = {"model": "lightgbm"}

    assert canonical_config_hash(other) != canonical_config_hash(baseline)


def test_list_order_changes_the_hash_because_it_can_change_results() -> None:
    assert canonical_config_hash({"features": ["a", "b"]}) != canonical_config_hash(
        {"features": ["b", "a"]}
    )


def test_numeric_type_changes_the_hash() -> None:
    # 1 and 1.0 are equal in Python but render differently in JSON, and the
    # int/float distinction routinely changes downstream arithmetic.
    assert canonical_config_hash({"n": 1}) != canonical_config_hash({"n": 1.0})


def test_the_hash_is_a_64_character_lowercase_sha256() -> None:
    digest = canonical_config_hash({"model": "lightgbm"})

    assert len(digest) == 64
    assert digest == digest.lower()
    assert set(digest) <= set("0123456789abcdef")


def test_canonical_json_sorts_keys_and_omits_insignificant_whitespace() -> None:
    assert canonical_config_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


@pytest.mark.parametrize(
    "config",
    [
        pytest.param({"n": float("nan")}, id="nan"),
        pytest.param({"n": float("inf")}, id="infinity"),
        pytest.param({1: "int-key"}, id="non-string-key"),
        pytest.param({"nested": {2: "int-key"}}, id="nested-non-string-key"),
        pytest.param({"s": {1, 2}}, id="set-not-serialisable"),
        pytest.param({"p": Path("data/prices")}, id="path-not-serialisable"),
    ],
)
def test_a_config_without_a_canonical_form_is_refused(config: object) -> None:
    with pytest.raises(ConfigHashError):
        canonical_config_hash(config)  # type: ignore[arg-type]


def test_a_non_mapping_config_is_refused() -> None:
    with pytest.raises(ConfigHashError):
        canonical_config_hash(["not", "a", "mapping"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Git state — the dirty-tree marker
# ---------------------------------------------------------------------------


def test_a_clean_tree_is_not_marked_dirty(committed_repo: Path) -> None:
    state = git_state(committed_repo)

    assert len(state.commit) == 40
    assert state.dirty is False


def test_a_modified_file_marks_the_tree_dirty(committed_repo: Path) -> None:
    (committed_repo / "tracked.txt").write_text("edited\n", encoding="utf-8")

    assert git_state(committed_repo).dirty is True


def test_an_untracked_file_marks_the_tree_dirty(committed_repo: Path) -> None:
    # Code that exists only in the working tree is as absent from the commit as
    # an uncommitted edit, so it must count.
    (committed_repo / "new_module.py").write_text("x = 1\n", encoding="utf-8")

    assert git_state(committed_repo).dirty is True


def test_a_staged_but_uncommitted_change_marks_the_tree_dirty(committed_repo: Path) -> None:
    (committed_repo / "tracked.txt").write_text("staged\n", encoding="utf-8")
    _git(committed_repo, "add", "tracked.txt")

    assert git_state(committed_repo).dirty is True


def test_the_dirty_marker_travels_in_the_git_reference(committed_repo: Path) -> None:
    clean = ReproducibilityStamp(
        git_commit=git_state(committed_repo).commit,
        git_dirty=False,
        data_version="v1",
        config_hash=VALID_HASH,
        seed=7,
    )
    (committed_repo / "tracked.txt").write_text("edited\n", encoding="utf-8")
    dirty_state = git_state(committed_repo)
    dirty = ReproducibilityStamp(
        git_commit=dirty_state.commit,
        git_dirty=dirty_state.dirty,
        data_version="v1",
        config_hash=VALID_HASH,
        seed=7,
    )

    assert clean.git_reference == clean.git_commit
    assert clean.reproducible is True
    assert dirty.git_reference == f"{dirty.git_commit}-dirty"
    assert dirty.reproducible is False
    assert dirty.as_tags()["repro.git_dirty"] == "true"
    assert dirty.as_tags()["repro.reproducible"] == "false"


def test_a_directory_that_is_not_a_repository_refuses_rather_than_guessing(tmp_path: Path) -> None:
    with pytest.raises(GitStateUnavailableError):
        git_state(tmp_path)


def test_without_git_on_the_path_the_stamp_refuses_rather_than_omitting_the_commit(
    monkeypatch: pytest.MonkeyPatch,
    committed_repo: Path,
) -> None:
    monkeypatch.setattr("backend.tracking.stamp.shutil.which", lambda _name: None)

    with pytest.raises(GitStateUnavailableError):
        git_state(committed_repo)


def test_a_repository_with_no_commits_refuses(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    repo.mkdir()
    _git(repo, "init", "--quiet")

    with pytest.raises(GitStateUnavailableError):
        git_state(repo)


# ---------------------------------------------------------------------------
# The stamp itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("git_commit", "", id="empty-commit"),
        pytest.param("git_commit", "abc1234", id="abbreviated-commit"),
        pytest.param("git_commit", "HEAD", id="symbolic-commit"),
        pytest.param("data_version", "", id="empty-data-version"),
        pytest.param("data_version", "   ", id="blank-data-version"),
        pytest.param("data_version", "a|b", id="pipe-in-data-version"),
        pytest.param("config_hash", "", id="empty-config-hash"),
        pytest.param("config_hash", "deadbeef", id="truncated-config-hash"),
        pytest.param("seed", None, id="missing-seed"),
        pytest.param("seed", True, id="bool-seed"),
        pytest.param("seed", -1, id="negative-seed"),
        pytest.param("seed", "7", id="string-seed"),
        pytest.param("git_dirty", "false", id="string-dirty-flag"),
    ],
)
def test_a_stamp_missing_an_i2_component_is_refused(field: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "git_commit": VALID_COMMIT,
        "git_dirty": False,
        "data_version": "dvc:abc123",
        "config_hash": VALID_HASH,
        "seed": 42,
    }
    kwargs[field] = value

    with pytest.raises(IncompleteStampError):
        ReproducibilityStamp(**kwargs)  # type: ignore[arg-type]


def test_a_complete_stamp_exposes_all_four_components_as_tags() -> None:
    stamp = ReproducibilityStamp(
        git_commit=VALID_COMMIT,
        git_dirty=False,
        data_version="dvc:abc123",
        config_hash=VALID_HASH,
        seed=42,
    )
    tags = stamp.as_tags()

    assert tags["repro.git_commit"] == VALID_COMMIT
    assert tags["repro.data_version"] == "dvc:abc123"
    assert tags["repro.config_hash"] == VALID_HASH
    assert tags["repro.seed"] == "42"
    # MLflow tag values are strings; a bool or int would be coerced silently.
    assert all(isinstance(value, str) for value in tags.values())


def test_create_builds_a_stamp_from_a_real_repository(committed_repo: Path) -> None:
    config = {"model": "lightgbm", "params": {"max_depth": 4}}
    stamp = ReproducibilityStamp.create(
        config=config,
        seed=1234,
        data_version="dvc:abc123",
        repo_root=committed_repo,
    )

    assert stamp.git_commit == git_state(committed_repo).commit
    assert stamp.git_dirty is False
    assert stamp.config_hash == canonical_config_hash(config)
    assert stamp.seed == 1234
    assert stamp.data_version == "dvc:abc123"


def test_create_marks_this_repository_honestly() -> None:
    # Against the real repository: whatever the tree state is, the stamp must
    # agree with git rather than assume a clean checkout.
    stamp = ReproducibilityStamp.create(config={"a": 1}, seed=0, data_version="local")

    assert stamp.git_dirty is git_state().dirty
    assert stamp.git_commit == git_state().commit


def test_repository_root_is_the_directory_holding_the_directive() -> None:
    assert (repository_root() / "DIRECTIVE.md").is_file()
