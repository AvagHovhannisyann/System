"""The CI workflow file must be valid YAML (guards a silent failure mode).

A malformed workflow does not fail loudly — GitHub reports the run as
"failure" with **zero jobs**, which looks similar enough to a normal red
build that it went unnoticed for fifteen commits here. The specific trap: a
step ``name`` containing a colon-space (``I3: no mock...``) parses as a
nested mapping, so the whole file is rejected before any job starts.

This test exists because the failure is invisible in the place people look
(the commit's check list appears empty rather than red).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOW_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _workflows() -> list[Path]:
    """Return every workflow file shipped in the repository."""
    return sorted(_WORKFLOW_DIR.glob("*.yml")) + sorted(_WORKFLOW_DIR.glob("*.yaml"))


def test_at_least_one_workflow_exists() -> None:
    """Guard the guard: an empty glob would make every check below vacuous."""
    assert _workflows(), f"no workflow files found under {_WORKFLOW_DIR}"


@pytest.mark.parametrize("workflow", _workflows(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml_with_jobs(workflow: Path) -> None:
    """Each workflow parses and declares at least one job.

    Parsing is necessary but not sufficient: a file can parse and still
    declare nothing runnable, which produces the same silent no-jobs result.
    """
    document: Any = yaml.safe_load(workflow.read_text())
    assert isinstance(document, dict), f"{workflow.name} is not a YAML mapping"
    jobs = document.get("jobs")
    assert isinstance(jobs, dict), f"{workflow.name}: jobs is not a mapping"
    assert jobs, f"{workflow.name} declares no jobs"
    for job_name, job in jobs.items():
        assert isinstance(job, dict), f"{workflow.name}: job {job_name} is not a mapping"
        assert job.get("steps"), f"{workflow.name}: job {job_name} has no steps"
