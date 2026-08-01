"""Reproducibility stamp — the four components invariant I2 demands (CC.3).

I2 (directive §2): *"Any result must be regenerable from: git commit + data
version + config hash + seed. Store all four with every artifact."*

A :class:`ReproducibilityStamp` is those four values together, and it is the
only way to get them into an MLflow run or a ``TESTING_LEDGER.md`` row. It
cannot be constructed partially: every field is validated on construction, so a
caller that cannot supply one of the four gets an exception instead of a
half-stamped artifact. That is deliberate — an artifact carrying three of the
four components looks reproducible in a listing and is not.

The dirty-tree marker
---------------------

``git_dirty`` records whether the working tree differed from ``HEAD`` when the
stamp was taken, **including untracked files**. A result produced from
uncommitted code is not regenerable from its commit, so the stamp says so rather
than quietly recording the commit it was *nearly* produced from. Untracked files
count because a module that exists only in the working tree is exactly as absent
from the commit as an uncommitted edit.

Canonical config hashing
------------------------

``config_hash`` is the SHA-256 of the config serialised as canonical JSON: keys
sorted at every level, no insignificant whitespace, non-finite floats and
non-string keys refused. Sorting is what makes the hash an *identity* — two runs
configured identically but assembled in a different order must land on the same
hash, or the hash cannot be used to recognise a repeated experiment.

Two things deliberately **do** change the hash, because they can change results:
list order (``[1, 2] != [2, 1]``) and numeric type (``1`` hashes differently from
``1.0``, since JSON renders them differently and the distinction is often the
difference between integer and float arithmetic downstream).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

__all__ = [
    "STAMP_TAG_PREFIX",
    "ConfigHashError",
    "GitState",
    "GitStateUnavailableError",
    "IncompleteStampError",
    "ReproducibilityStamp",
    "StampError",
    "canonical_config_hash",
    "canonical_config_json",
    "git_state",
    "repository_root",
]

STAMP_TAG_PREFIX = "repro."
"""Prefix for the MLflow tag keys carrying the stamp (``repro.git_commit`` …)."""

_GIT_TIMEOUT_S = 30.0
_COMMIT_PATTERN = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")
_FORBIDDEN_IN_CELL = ("|", "\n", "\r")


class StampError(ValueError):
    """Base class for every refusal raised by this module."""


class ConfigHashError(StampError):
    """Raised when a config cannot be canonicalised into a stable identity.

    Fatal on purpose. The alternative — hashing something whose serialisation
    depends on insertion order or float repr — produces a hash that fails to
    match itself on a rerun, which is worse than no hash at all because it looks
    like a real identity.
    """


class IncompleteStampError(StampError):
    """Raised when one of I2's four components is missing or malformed.

    An artifact stamped with three of four components is not reproducible, so
    construction fails rather than recording a stamp with a gap in it.
    """


class GitStateUnavailableError(StampError):
    """Raised when the git commit and dirty state cannot be determined.

    Not downgraded to a placeholder commit: "unknown" in the git column of a
    ledger row is indistinguishable, later, from a commit nobody wrote down.
    """


def repository_root() -> Path:
    """Return the repository root inferred from this file's location.

    Assumes the package layout committed in the directive's §4 tree
    (``<root>/backend/tracking/stamp.py``). Callers working outside that layout
    should pass an explicit ``repo_root``.

    Returns:
        Absolute path to the repository root.
    """
    return Path(__file__).resolve().parents[2]


def _reject_unhashable_shapes(value: object, path: str) -> None:
    """Recursively refuse config shapes whose canonical form is ambiguous.

    Args:
        value: the config fragment being checked.
        path: dotted path to ``value``, used in the error message.

    Raises:
        ConfigHashError: if a mapping key is not a string. Non-string keys are
            coerced to strings by JSON, so ``{1: "a"}`` and ``{"1": "a"}`` would
            collide onto one hash while meaning different things.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                msg = (
                    f"config key at {path}[{key!r}] is {type(key).__name__}, not str; "
                    "JSON coerces non-string keys and two different configs would "
                    "collide onto one hash"
                )
                raise ConfigHashError(msg)
            _reject_unhashable_shapes(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _reject_unhashable_shapes(item, f"{path}[{index}]")


def canonical_config_json(config: Mapping[str, object]) -> str:
    """Serialise ``config`` to canonical JSON — the preimage of the config hash.

    Canonical means: keys sorted at every nesting level, minimal separators, no
    NaN/Infinity, UTF-8 text kept as-is. Key order in the input is therefore
    irrelevant; list order is not, because list order can change results.

    Args:
        config: a JSON-serialisable mapping of configuration values.

    Returns:
        The canonical JSON text (unicode, no trailing newline).

    Raises:
        ConfigHashError: if ``config`` is not a mapping, contains a non-string
            key, holds a value JSON cannot represent, or contains NaN/Infinity
            (which have no canonical JSON form and are never a deliberate
            configuration value).
    """
    # Checked at runtime, not only by the annotation: this hash is the identity
    # of an experiment, and callers reach it from untyped places (a YAML load, a
    # request body). `object` first so the check is not optimised away as
    # unreachable by the type checker.
    supplied: object = config
    if not isinstance(supplied, Mapping):
        msg = f"config must be a mapping, got {type(supplied).__name__}"
        raise ConfigHashError(msg)
    _reject_unhashable_shapes(config, "config")
    try:
        return json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        msg = f"config is not canonically JSON-serialisable: {exc}"
        raise ConfigHashError(msg) from exc


def canonical_config_hash(config: Mapping[str, object]) -> str:
    """Return the SHA-256 hex digest of the canonical JSON form of ``config``.

    Units: 64 lowercase hex characters. The same configuration written with its
    keys in any order yields the same digest; any change to a value, a key, a
    list's order, or a value's JSON type changes it.

    Args:
        config: a JSON-serialisable mapping of configuration values.

    Returns:
        Lowercase 64-character hex digest.

    Raises:
        ConfigHashError: propagated from :func:`canonical_config_json`.
    """
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GitState:
    """The commit a result was produced at, and whether the tree was dirty.

    Attributes:
        commit: full commit SHA of ``HEAD``.
        dirty: whether the working tree differed from ``HEAD``, counting
            untracked files.
    """

    commit: str
    dirty: bool


def _run_git(git: str, repo_root: Path, *args: str) -> str:
    """Run a read-only git command and return its stdout.

    Args:
        git: absolute path to the git executable.
        repo_root: repository to run in (passed via ``-C``).
        *args: git arguments, e.g. ``("rev-parse", "HEAD")``.

    Returns:
        The command's stdout.

    Raises:
        GitStateUnavailableError: if git exits non-zero or does not finish
            within 30 seconds.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell; git path from shutil.which
            [git, "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"git {' '.join(args)} failed in {repo_root}: {exc}"
        raise GitStateUnavailableError(msg) from exc
    if completed.returncode != 0:
        msg = (
            f"git {' '.join(args)} exited {completed.returncode} in {repo_root}: "
            f"{completed.stderr.strip()}"
        )
        raise GitStateUnavailableError(msg)
    return completed.stdout


def git_state(repo_root: Path | None = None) -> GitState:
    """Return the current commit and dirty flag for ``repo_root``.

    ``dirty`` is true when ``git status --porcelain`` reports anything at all:
    staged changes, unstaged changes, or untracked files. Untracked files count
    because code present only in the working tree is absent from the commit, so
    a result that used it is not regenerable from that commit.

    Args:
        repo_root: repository to inspect. Defaults to :func:`repository_root`.

    Returns:
        A :class:`GitState`.

    Raises:
        GitStateUnavailableError: if git is not on ``PATH``, ``repo_root`` is
            not a repository, or ``HEAD`` does not resolve (an empty repository
            with no commits).
    """
    root = repository_root() if repo_root is None else repo_root
    git = shutil.which("git")
    if git is None:
        msg = (
            "git is not on PATH, so the commit component of the reproducibility "
            "stamp (I2) cannot be determined"
        )
        raise GitStateUnavailableError(msg)
    commit = _run_git(git, root, "rev-parse", "HEAD").strip()
    if not _COMMIT_PATTERN.fullmatch(commit):
        msg = f"git rev-parse HEAD returned {commit!r}, which is not a commit SHA"
        raise GitStateUnavailableError(msg)
    status = _run_git(git, root, "status", "--porcelain")
    return GitState(commit=commit, dirty=status.strip() != "")


def _require_ledger_safe(field: str, value: str) -> None:
    """Refuse stamp values that cannot survive a markdown ledger cell.

    Args:
        field: field name, for the error message.
        value: the value to check.

    Raises:
        IncompleteStampError: if the value is blank, or contains a pipe or a
            newline. Those characters would split or truncate a
            ``TESTING_LEDGER.md`` row, and a corrupted row makes the reader
            refuse the whole file (§9.7 keeps that count trustworthy).
    """
    if value.strip() == "":
        msg = f"{field} is empty; I2 requires all four components to be recorded"
        raise IncompleteStampError(msg)
    for character in _FORBIDDEN_IN_CELL:
        if character in value:
            msg = f"{field}={value!r} contains {character!r}, which would corrupt a ledger row"
            raise IncompleteStampError(msg)


@dataclass(frozen=True, slots=True)
class ReproducibilityStamp:
    """The four components of I2, validated together.

    Attributes:
        git_commit: full commit SHA (40 or 64 lowercase hex characters).
        git_dirty: whether the working tree was dirty when the stamp was taken.
        data_version: identifier of the data snapshot used — see
            :mod:`backend.tracking.data_version`.
        config_hash: SHA-256 hex digest from :func:`canonical_config_hash`.
        seed: the random seed used, recorded verbatim. Non-negative int; ``bool``
            is refused because ``True`` would silently record as seed 1.
    """

    git_commit: str
    git_dirty: bool
    data_version: str
    config_hash: str
    seed: int

    def __post_init__(self) -> None:
        """Validate all four components; refuse an incomplete stamp.

        Raises:
            IncompleteStampError: if any component is missing or malformed.
        """
        dirty: object = self.git_dirty
        if not isinstance(dirty, bool):
            msg = f"git_dirty must be a bool, got {type(dirty).__name__}"
            raise IncompleteStampError(msg)
        _require_ledger_safe("git_commit", self.git_commit)
        if not _COMMIT_PATTERN.fullmatch(self.git_commit):
            msg = (
                f"git_commit={self.git_commit!r} is not a full commit SHA; an abbreviated "
                "or symbolic reference does not identify a tree unambiguously"
            )
            raise IncompleteStampError(msg)
        _require_ledger_safe("data_version", self.data_version)
        _require_ledger_safe("config_hash", self.config_hash)
        if not _SHA256_PATTERN.fullmatch(self.config_hash):
            msg = f"config_hash={self.config_hash!r} is not a 64-character SHA-256 hex digest"
            raise IncompleteStampError(msg)
        seed: object = self.seed
        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = f"seed must be an int (bool is refused), got {type(seed).__name__}"
            raise IncompleteStampError(msg)
        if self.seed < 0:
            msg = f"seed={self.seed} is negative; seeds are recorded as given and must be usable"
            raise IncompleteStampError(msg)

    @property
    def git_reference(self) -> str:
        """Return the commit, suffixed ``-dirty`` when the tree was not clean.

        This is the string written to the ledger's ``git commit`` column, so the
        dirty marker travels with the row rather than living in a separate flag
        a reader might not look at.
        """
        return f"{self.git_commit}-dirty" if self.git_dirty else self.git_commit

    @property
    def reproducible(self) -> bool:
        """Return whether the result can be regenerated from the commit alone.

        False when the tree was dirty: the code that produced the result is not
        the code at ``git_commit``.
        """
        return not self.git_dirty

    def as_tags(self) -> dict[str, str]:
        """Return the stamp as MLflow tags, all values stringified.

        Returns:
            Mapping of ``repro.*`` tag keys to string values. Booleans render as
            ``"true"``/``"false"`` so they are greppable in the MLflow UI.
        """
        return {
            f"{STAMP_TAG_PREFIX}git_commit": self.git_commit,
            f"{STAMP_TAG_PREFIX}git_dirty": "true" if self.git_dirty else "false",
            f"{STAMP_TAG_PREFIX}git_reference": self.git_reference,
            f"{STAMP_TAG_PREFIX}data_version": self.data_version,
            f"{STAMP_TAG_PREFIX}config_hash": self.config_hash,
            f"{STAMP_TAG_PREFIX}seed": str(self.seed),
            f"{STAMP_TAG_PREFIX}reproducible": "true" if self.reproducible else "false",
        }

    @classmethod
    def create(
        cls,
        *,
        config: Mapping[str, object],
        seed: int,
        data_version: str,
        repo_root: Path | None = None,
    ) -> Self:
        """Build a stamp by hashing ``config`` and reading the git state.

        Args:
            config: the run's configuration; hashed canonically.
            seed: the random seed the run will use.
            data_version: identifier of the data snapshot — obtain it from
                :func:`backend.tracking.data_version.resolve_data_version`
                rather than typing one in, so it cannot drift from the data.
            repo_root: repository to read git state from. Defaults to
                :func:`repository_root`.

        Returns:
            A fully populated :class:`ReproducibilityStamp`.

        Raises:
            ConfigHashError: if ``config`` cannot be canonicalised.
            GitStateUnavailableError: if git state cannot be read.
            IncompleteStampError: if any resulting component is malformed.
        """
        state = git_state(repo_root)
        return cls(
            git_commit=state.commit,
            git_dirty=state.dirty,
            data_version=data_version.strip(),
            config_hash=canonical_config_hash(config),
            seed=seed,
        )
