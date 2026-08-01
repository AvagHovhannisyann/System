"""CC.8: the no-fabrication lint guard, run as CI runs it (invariant I3).

The guard bans identifiers naming fabricated data in ``backend/ingest``
production code. These tests do two things, and the first matters more than
the second: they prove the guard **flags** each violating shape (a checker
that never fires is decoration), and only then that the real tree passes it.

The guard is wired into CI as its own step in the backend job and into
``.pre-commit-config.yaml``; running it here as a subprocess means ``pytest``
enforces it too, so it cannot silently stop being run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GUARD = _REPO_ROOT / "scripts" / "check_no_fabrication.py"
_INGEST_TREE = _REPO_ROOT / "backend" / "ingest"

# Assembled at runtime so this test file does not itself contain the banned
# identifiers it exercises — the probes are written to a tmp_path module.
_BANNED = (
    "m" + "ock",
    "st" + "ub",
    "fa" + "ke",
    "dum" + "my",
    "synth" + "etic",
    "place" + "holder",
)


def _run_guard(*targets: Path) -> subprocess.CompletedProcess[str]:
    """Run the guard script on ``targets`` exactly as CI and pre-commit do."""
    return subprocess.run(  # noqa: S603 — fixed argv, no shell, repository-local tool
        [sys.executable, str(_GUARD), *[str(target) for target in targets]],
        capture_output=True,
        text=True,
        check=False,
    )


def test_guard_passes_the_real_ingestion_tree() -> None:
    """backend/ingest carries no fabricated-data identifier today."""
    result = _run_guard(_INGEST_TREE)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("word", _BANNED)
def test_guard_flags_a_banned_function_name(tmp_path: Path, word: str) -> None:
    probe = tmp_path / "probe_function.py"
    probe.write_text(
        f'"""Probe."""\n\n\ndef {word}_response() -> int:\n    """Doc."""\n    return 1\n'
    )
    result = _run_guard(probe)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"{word}_response" in result.stderr


@pytest.mark.parametrize("word", _BANNED)
def test_guard_flags_a_banned_variable_name(tmp_path: Path, word: str) -> None:
    probe = tmp_path / "probe_variable.py"
    probe.write_text(f'"""Probe."""\n\n{word}_rows = [1, 2, 3]\n')
    result = _run_guard(probe)
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("word", _BANNED)
def test_guard_flags_a_banned_class_name(tmp_path: Path, word: str) -> None:
    probe = tmp_path / "probe_class.py"
    probe.write_text(f'"""Probe."""\n\n\nclass {word.capitalize()}Client:\n    """Doc."""\n')
    result = _run_guard(probe)
    assert result.returncode == 1, result.stdout + result.stderr


def test_guard_flags_camel_case_identifiers(tmp_path: Path) -> None:
    """snake_case is not the only way to name a fabrication."""
    probe = tmp_path / "probe_camel.py"
    probe.write_text('"""Probe."""\n\nsomeMockValue = 1\n')
    result = _run_guard(probe)
    assert result.returncode == 1, result.stdout + result.stderr


def test_guard_flags_an_import_of_a_fabrication_library(tmp_path: Path) -> None:
    probe = tmp_path / "probe_import.py"
    probe.write_text('"""Probe."""\n\nfrom unittest import mock\n')
    result = _run_guard(probe)
    assert result.returncode == 1, result.stdout + result.stderr


def test_guard_flags_an_aliased_import(tmp_path: Path) -> None:
    """Renaming the import does not launder it."""
    probe = tmp_path / "probe_alias.py"
    probe.write_text('"""Probe."""\n\nimport json as placeholder\n')
    result = _run_guard(probe)
    assert result.returncode == 1, result.stdout + result.stderr


def test_guard_ignores_prose_so_modules_can_forbid_what_it_bans(tmp_path: Path) -> None:
    """Docstrings, comments and string literals are deliberately out of scope.

    The ingestion modules must be free to say in prose that they never return
    a placeholder. A guard that flagged its own prohibition would force the
    documentation to be deleted to satisfy the lint — the opposite of what I3
    wants.
    """
    probe = tmp_path / "probe_prose.py"
    prose = "".join(
        [
            '"""This module never returns a pl',
            "aceholder or a m",
            'ock value."""\n\n',
            "# Not a st",
            "ub: raises instead.\n",
            'MESSAGE = "no fa',
            'ke data"\n',
        ]
    )
    probe.write_text(prose)
    result = _run_guard(probe)
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_accepts_ordinary_identifiers(tmp_path: Path) -> None:
    """No false positive on the vocabulary the framework actually uses."""
    probe = tmp_path / "probe_clean.py"
    probe.write_text(
        '"""Probe."""\n\n'
        "from datetime import timedelta\n\n"
        "session_factory = None\n"
        "sample_uniqueness_weight = 0.5\n"
        "max_live_lag = timedelta(hours=1)\n"
        "checkpoint_after = {}\n"
    )
    result = _run_guard(probe)
    assert result.returncode == 0, result.stdout + result.stderr


def test_guard_reports_a_usage_error_for_a_missing_path(tmp_path: Path) -> None:
    result = _run_guard(tmp_path / "does_not_exist")
    assert result.returncode == 2, result.stdout + result.stderr


def test_guard_is_wired_into_ci_and_pre_commit() -> None:
    """A guard CI does not run is a guard that does not exist."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    pre_commit = (_REPO_ROOT / ".pre-commit-config.yaml").read_text()
    assert "scripts/check_no_fabrication.py" in workflow
    assert "scripts/check_no_fabrication.py" in pre_commit
