"""Data versioning — I2's *data version* component, and the DVC workflow (CC.3).

Status of DVC in this repository
--------------------------------

**DVC is not installed.** Checked rather than assumed (§9.4): ``import dvc``
raises ``ModuleNotFoundError``, no ``dvc`` executable is on ``PATH``, and there
is no ``.dvc/`` directory. ``dvc`` is not in ``pyproject.toml`` either, and
``pyproject.toml`` is not this task's to edit — so ``dvc init`` has not been run
and nothing here calls a DVC API. Writing a wrapper against an uninstalled
package would be inventing usage that no test could exercise.

What exists instead is the part that is testable today and that DVC would feed:
a recorded, retrievable **data version identifier**, and a refusal when there
isn't one. :func:`dvc_availability` reports the real state of the tooling at
call time, so this module tells the truth before and after DVC lands.

The workflow, for when DVC is installed
---------------------------------------

``data/`` is gitignored (it holds vendor data and derived artifacts, which §0.4
forbids committing). DVC's model fits that exactly: the *data* stays out of git,
a small *pointer* is committed, and the pointer is what makes a result
regenerable.

1. Add the dependency (``dvc`` — the owner of ``pyproject.toml`` does this) and
   run ``dvc init`` at the repository root. That creates ``.dvc/`` and
   ``.dvcignore``, both committed.
2. ``dvc add data`` — writes ``data.dvc`` (a small YAML pointer holding the
   directory's content hash) and appends ``/data`` to ``.gitignore``. Commit
   ``data.dvc``; never commit ``data/`` itself.
3. Configure a remote (``dvc remote add -d storage <url>``) and ``dvc push`` so
   another machine, or this machine after a wipe, can ``dvc pull`` the exact
   bytes the pointer names.
4. After any change to ``data/``, re-run ``dvc add data`` and commit the updated
   ``data.dvc``. The data version for a run is then the pointer's content hash.
5. Record that hash with :func:`write_data_version` (or export ``DATA_VERSION``)
   so every run stamps the snapshot it actually read.

Until step 1 happens, steps 2-4 are unavailable and :func:`resolve_data_version`
depends on the identifier being recorded explicitly. It never invents one: an
unknown data version means the result is not reproducible, and saying so is the
whole point of I2.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from backend.tracking.stamp import repository_root

__all__ = [
    "DATA_VERSION_ENV_VAR",
    "DATA_VERSION_POINTER_NAME",
    "DataVersionError",
    "DataVersionUnavailableError",
    "DvcAvailability",
    "default_pointer_path",
    "dvc_availability",
    "read_data_version",
    "resolve_data_version",
    "write_data_version",
]

DATA_VERSION_ENV_VAR = "DATA_VERSION"
"""Environment variable that pins the data version for a process."""

DATA_VERSION_POINTER_NAME = ".data-version"
"""Name of the pointer file at the repository root.

At the root rather than inside ``data/`` on purpose: ``data/`` is gitignored, so
a pointer living there could never be committed, and a data version that is not
committed alongside the code cannot reconstruct anything.
"""

_POINTER_HEADER = (
    "# Data version for reproducibility invariant I2 — see backend/tracking/data_version.py.\n"
    "# One identifier, one line. With DVC this is the content hash from data.dvc.\n"
)
_MAX_VERSION_LENGTH = 200
_FORBIDDEN_IN_VERSION = ("|", "\n", "\r", "#")


class DataVersionError(RuntimeError):
    """Base class for data-version failures."""


class DataVersionUnavailableError(DataVersionError):
    """Raised when no data version can be determined.

    Fatal rather than defaulted. A placeholder such as ``"unknown"`` would flow
    into a stamp and a ledger row and make an unreproducible result look
    reproducible — the exact failure I2 exists to prevent.
    """


def default_pointer_path(repo_root: Path | None = None) -> Path:
    """Return the conventional pointer-file path for a repository.

    Args:
        repo_root: repository root. Defaults to the inferred root.

    Returns:
        ``<repo_root>/.data-version``.
    """
    root = repository_root() if repo_root is None else repo_root
    return root / DATA_VERSION_POINTER_NAME


@dataclass(frozen=True, slots=True)
class DvcAvailability:
    """What DVC tooling is actually present, measured at call time.

    Attributes:
        package_importable: whether ``import dvc`` would succeed.
        cli_on_path: whether a ``dvc`` executable is on ``PATH``.
        repository_initialised: whether ``<repo_root>/.dvc`` exists.
    """

    package_importable: bool
    cli_on_path: bool
    repository_initialised: bool

    @property
    def usable(self) -> bool:
        """Return whether DVC could version data here right now.

        Requires both the tooling (CLI) and an initialised repository. The
        Python package alone is not enough to run ``dvc add``.
        """
        return self.cli_on_path and self.repository_initialised

    def describe(self) -> str:
        """Return a one-line human-readable summary, with the next action.

        Returns:
            A sentence stating what is present and what is missing. Used in
            error messages and in reports to a human, so it names the concrete
            next step rather than only the failure.
        """
        if self.usable:
            return (
                "DVC is available (CLI on PATH, repository initialised); "
                "`dvc add data` produces the data version to record"
            )
        missing = []
        if not self.package_importable:
            missing.append("the dvc python package is not installed")
        if not self.cli_on_path:
            missing.append("no dvc executable on PATH")
        if not self.repository_initialised:
            missing.append("no .dvc/ directory (dvc init has not been run)")
        return (
            f"DVC is not usable here: {'; '.join(missing)}. "
            "Add `dvc` to the project dependencies and run `dvc init`, then "
            "`dvc add data`; until then record the data version explicitly."
        )


def dvc_availability(repo_root: Path | None = None) -> DvcAvailability:
    """Report whether DVC is installed and initialised, without importing it.

    Uses :func:`importlib.util.find_spec` rather than a real import so that a
    broken DVC installation reports as present-but-broken at use time instead of
    raising here.

    Args:
        repo_root: repository root to check for ``.dvc/``. Defaults to the
            inferred root.

    Returns:
        A :class:`DvcAvailability` snapshot.
    """
    root = repository_root() if repo_root is None else repo_root
    return DvcAvailability(
        package_importable=importlib.util.find_spec("dvc") is not None,
        cli_on_path=shutil.which("dvc") is not None,
        repository_initialised=(root / ".dvc").is_dir(),
    )


def _validate_version(version: str) -> str:
    """Validate and normalise a data version identifier.

    Args:
        version: the raw identifier.

    Returns:
        The stripped identifier.

    Raises:
        DataVersionError: if it is blank, over 200 characters, or contains a
            character that would break a pointer file or a ledger cell
            (``|``, ``#``, or a newline).
    """
    stripped = version.strip()
    if stripped == "":
        msg = "data version is empty"
        raise DataVersionError(msg)
    if len(stripped) > _MAX_VERSION_LENGTH:
        msg = f"data version is {len(stripped)} characters, over the {_MAX_VERSION_LENGTH} limit"
        raise DataVersionError(msg)
    for character in _FORBIDDEN_IN_VERSION:
        if character in stripped:
            msg = f"data version {stripped!r} contains {character!r}, which is not allowed"
            raise DataVersionError(msg)
    return stripped


def write_data_version(pointer_path: Path, version: str) -> str:
    """Record ``version`` in the pointer file, replacing any previous value.

    Overwriting is correct here: the pointer names the *current* data snapshot
    and moves when the data does. History of which run used which snapshot lives
    in the run's stamp and in ``TESTING_LEDGER.md``, both of which are
    append-only — this file is not a log.

    Args:
        pointer_path: file to write. Parent directories are created.
        version: the identifier (e.g. a DVC content hash).

    Returns:
        The normalised identifier that was written.

    Raises:
        DataVersionError: if ``version`` is invalid.
        OSError: if the file cannot be written.
    """
    normalised = _validate_version(version)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(f"{_POINTER_HEADER}{normalised}\n", encoding="utf-8")
    return normalised


def read_data_version(pointer_path: Path) -> str:
    """Read the data version from a pointer file.

    The format is one identifier on its own line; ``#`` comment lines and blank
    lines are ignored. More than one identifier is an error rather than a
    first-wins guess — an ambiguous pointer means nobody knows which snapshot a
    result used.

    Args:
        pointer_path: file to read.

    Returns:
        The identifier.

    Raises:
        DataVersionUnavailableError: if the file does not exist or holds no
            identifier.
        DataVersionError: if the file holds more than one identifier, or one
            that fails validation.
    """
    try:
        text = pointer_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"no data-version pointer at {pointer_path}"
        raise DataVersionUnavailableError(msg) from exc
    candidates = [
        line.strip()
        for line in text.splitlines()
        if line.strip() != "" and not line.lstrip().startswith("#")
    ]
    if not candidates:
        msg = f"{pointer_path} contains no data version (only comments or blank lines)"
        raise DataVersionUnavailableError(msg)
    if len(candidates) > 1:
        msg = (
            f"{pointer_path} contains {len(candidates)} identifiers; exactly one is "
            "required, because a result can only have been produced from one snapshot"
        )
        raise DataVersionError(msg)
    return _validate_version(candidates[0])


def resolve_data_version(
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    pointer_path: Path | None = None,
) -> str:
    """Return the data version for this process, or refuse.

    Precedence:

    1. the ``DATA_VERSION`` environment variable, so a scheduled or CI run can
       pin the snapshot it was handed;
    2. the pointer file at ``<repo_root>/.data-version``.

    There is no third step. If neither source has a value the call raises, and
    the run that needed it does not start — an artifact with no data version is
    not regenerable, and recording a guess would hide that.

    Args:
        repo_root: repository root. Defaults to the inferred root.
        env: environment mapping. Defaults to ``os.environ``.
        pointer_path: override for the pointer file location.

    Returns:
        The data version identifier.

    Raises:
        DataVersionUnavailableError: if neither source supplies one. The message
            includes :meth:`DvcAvailability.describe` so the reader learns
            whether DVC could have supplied it.
        DataVersionError: if a supplied value is invalid.
    """
    environment: Mapping[str, str] = os.environ if env is None else env
    from_env = environment.get(DATA_VERSION_ENV_VAR)
    if from_env is not None and from_env.strip() != "":
        return _validate_version(from_env)

    path = default_pointer_path(repo_root) if pointer_path is None else pointer_path
    try:
        return read_data_version(path)
    except DataVersionUnavailableError as exc:
        msg = (
            f"no data version available: {DATA_VERSION_ENV_VAR} is unset and {exc}. "
            f"{dvc_availability(repo_root).describe()}"
        )
        raise DataVersionUnavailableError(msg) from exc
