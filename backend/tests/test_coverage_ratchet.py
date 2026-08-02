"""CC.6 — the coverage floors are committed, enforced, and cannot drift down.

DIRECTIVE.md section 8 sets them: ">= 85% backend, >= 70% frontend". This
module asserts that both are *configured and enforced*, not merely reported.

It deliberately does not assert a coverage percentage. The run that measures
coverage is the run this file is part of, and a test that inspected its own
run's total would be asserting on the weakest signal in the build. What it
checks instead is the machinery around the number, because every way this
kind of gate is known to fail is a machinery failure:

* a floor that is configured but never evaluated — no ``--cov`` in CI, so
  ``fail_under`` has nothing to compare against and the config reads as
  correct while enforcing nothing;
* a floor that rounds away — coverage compares ``round(total, precision)``
  against ``fail_under``, so at the default precision of 0 a real 84.6%
  clears an 85 floor;
* a floor that auto-updates to the last run, which ratifies whatever a lucky
  run reached and by construction cannot detect decay;
* an exclusion list that grows one file at a time until the percentage
  describes a subset nobody chose.

**What a coverage gate proves.** That the counted lines ran. That is all. It
is satisfied by a test that imports a module and asserts nothing, and it is
satisfied faster by removing an awkward file from the denominator than by
testing it. That is why the exclusion lists for both stacks are pinned here,
why ``# pragma: no cover`` in shipped code is budgeted here, and why the
prose explaining the limits of the gate is anchored here so it cannot be
quietly deleted from the config it describes.

Passing this file is not evidence that the suite is honest (I6). It is
evidence that one specific dishonest move — quietly lowering the bar — is
not available.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_FRONTEND = _REPO_ROOT / "frontend"
_VITEST_CONFIG = _FRONTEND / "vitest.config.mts"
_PACKAGE_JSON = _FRONTEND / "package.json"

# --------------------------------------------------------------------------
# The committed numbers. Each is duplicated in the config it governs, on
# purpose: raising a floor, widening an exclusion list or spending another
# `# pragma: no cover` then costs a two-file edit that shows up in review,
# and lowering one quietly is not possible.
# --------------------------------------------------------------------------
BACKEND_FLOOR = 85
FRONTEND_FLOOR = 70

# Exactly the entries in [tool.coverage.run].omit, in order. Each carries its
# justification beside it in pyproject.toml; that the justification exists is
# asserted below, because an unexplained omit is indistinguishable from
# hiding a file that was hard to test.
COMMITTED_BACKEND_OMIT = [
    "backend/db/migrations/*",
    "backend/tests/*",
]

# Ceiling on `# pragma: no cover` in *shipped* backend code (the test package
# is not in the coverage denominator, so a pragma there moves no number).
# Every pragma is a hand-carved hole in the measurement, so the count is
# capped rather than left to grow: raising this ceiling is a deliberate edit
# with a diff, which is the whole point. It is a ceiling, not a target —
# lower it whenever the real count drops.
SHIPPED_PRAGMA_BUDGET = 9

# A pragma has to say *why*. Eight characters is a low bar deliberately: it
# rejects a bare `# pragma: no cover` without pretending this test can judge
# whether a stated reason is a good one. A human reviewer does that.
_MIN_PRAGMA_REASON_CHARS = 8

# Punctuation that separates the pragma from its reason and carries no
# meaning of its own, stripped before the reason is measured.
_PRAGMA_REASON_SEPARATORS = " \t:#" + "-" + "\u2013" + "\u2014"

# The whole first-party frontend surface, with nothing carved out.
COMMITTED_FRONTEND_INCLUDE = {
    "app/**/*.{ts,tsx}",
    "components/**/*.{ts,tsx}",
    "lib/**/*.{ts,tsx}",
}
COMMITTED_FRONTEND_EXCLUDE = {"**/*.d.ts"}

_PRAGMA_RE = re.compile(r"#\s*pragma:\s*no\s+cover(?P<reason>.*)$")


def _pyproject() -> dict[str, Any]:
    return tomllib.loads(_PYPROJECT.read_text())


def _coverage_run() -> dict[str, Any]:
    section: dict[str, Any] = _pyproject()["tool"]["coverage"]["run"]
    return section


def _coverage_report() -> dict[str, Any]:
    section: dict[str, Any] = _pyproject()["tool"]["coverage"]["report"]
    return section


def _workflow() -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text())
    return document


def _job(name: str) -> dict[str, Any]:
    jobs = _workflow()["jobs"]
    assert name in jobs, f"CI workflow declares no {name!r} job"
    job: dict[str, Any] = jobs[name]
    return job


def _run_commands(job_name: str) -> list[str]:
    return [str(step["run"]) for step in _job(job_name)["steps"] if "run" in step]


def _shipped_python_files() -> list[Path]:
    """Every backend .py file that the coverage floor is actually about."""
    return [
        path
        for path in sorted((_REPO_ROOT / "backend").rglob("*.py"))
        if path.relative_to(_REPO_ROOT).parts[1] != "tests"
    ]


def _shipped_pragmas() -> list[tuple[str, int, str]]:
    """Return (relative path, line number, stated reason) per shipped pragma."""
    found: list[tuple[str, int, str]] = []
    for path in _shipped_python_files():
        relative = str(path.relative_to(_REPO_ROOT))
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = _PRAGMA_RE.search(line)
            if match is not None:
                found.append((relative, number, match.group("reason")))
    return found


def _omit_entries_with_comments() -> list[tuple[str, str]]:
    """Pair each omit entry in pyproject.toml with the comment block above it."""
    lines = _PYPROJECT.read_text().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("omit = [")]
    assert len(starts) == 1, "expected exactly one omit list in pyproject.toml"
    start = starts[0]
    end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "]")
    entries: list[tuple[str, str]] = []
    for index in range(start + 1, end):
        stripped = lines[index].strip()
        if not stripped.startswith('"'):
            continue
        comment: list[str] = []
        cursor = index - 1
        while cursor > start and lines[cursor].strip().startswith("#"):
            comment.insert(0, lines[cursor].strip().lstrip("#").strip())
            cursor -= 1
        entries.append((stripped.strip('",'), " ".join(comment)))
    return entries


def _vitest_coverage_block() -> str:
    """The text of the `coverage: {...}` block, so `test.include` cannot match."""
    text = _VITEST_CONFIG.read_text()
    index = text.find("coverage: {")
    assert index != -1, f"no coverage block found in {_VITEST_CONFIG.name}"
    return text[index:]


def _vitest_string_array(name: str) -> set[str]:
    block = re.search(rf"{name}:\s*\[(?P<body>[^\]]*)\]", _vitest_coverage_block())
    assert block is not None, f"no {name!r} array in the vitest coverage block"
    return set(re.findall(r'"([^"]+)"', block.group("body")))


def _package_scripts() -> dict[str, str]:
    scripts: dict[str, str] = json.loads(_PACKAGE_JSON.read_text())["scripts"]
    return scripts


# --------------------------------------------------------------------------
# The backend floor exists and is the committed number
# --------------------------------------------------------------------------


def test_the_backend_floor_is_the_committed_number() -> None:
    """`fail_under` is set, and set to the number this file pins."""
    report = _coverage_report()
    assert "fail_under" in report, (
        "[tool.coverage.report] has no fail_under — without it pytest-cov prints a "
        "coverage table and exits 0 no matter how far the number falls (CC.6)"
    )
    assert report["fail_under"] == BACKEND_FLOOR, (
        f"backend floor is {report['fail_under']}, this test pins {BACKEND_FLOOR}. "
        "Raising the floor is meant to be a deliberate edit in both places; "
        "if this failed because someone lowered it, that is the case this "
        "test exists to catch (DIRECTIVE.md section 8)."
    )


def test_the_backend_floor_cannot_be_rounded_away() -> None:
    """A sub-floor total must fail, not round up into a pass.

    coverage evaluates ``round(total, precision) < fail_under``. At the
    default precision of 0 a genuine 84.6% rounds to 85 and clears an 85
    floor, so the floor quietly means "84.5" instead of what it says.
    """
    precision = _coverage_report().get("precision", 0)
    assert precision >= 1, (
        f"[tool.coverage.report].precision is {precision}; at precision 0 coverage "
        f"rounds the total before comparing, so anything above "
        f"{BACKEND_FLOOR - 0.5} passes a floor of {BACKEND_FLOOR}"
    )


def test_the_backend_gate_counts_branches_not_only_lines() -> None:
    """Branch coverage, or an untested `else` arm reads as fully covered."""
    assert _coverage_run().get("branch") is True, (
        "[tool.coverage.run].branch must stay true — with it off, an `if` whose "
        "false arm never runs still counts as 100% covered and the same "
        "percentage means materially less"
    )


def test_the_backend_gate_measures_the_whole_backend() -> None:
    """Narrowing `source` would shrink what the percentage is even about."""
    assert _coverage_run().get("source") == ["backend"], (
        "[tool.coverage.run].source must stay ['backend']: a narrower source is a "
        "silent way to make the floor describe a hand-picked subset"
    )


def test_no_competing_coverage_config_shadows_pyproject() -> None:
    """Coverage reads the first config it finds, and pyproject is last.

    A ``.coveragerc`` (or a ``[coverage:*]`` section in setup.cfg / tox.ini)
    takes precedence over pyproject.toml wholesale, so the committed floor
    above would be ignored while still reading as configured.
    """
    assert not (_REPO_ROOT / ".coveragerc").exists(), (
        ".coveragerc overrides [tool.coverage.*] in pyproject.toml entirely — "
        "the committed floor would silently stop applying"
    )
    for name in ("setup.cfg", "tox.ini"):
        path = _REPO_ROOT / name
        if path.exists():
            assert "[coverage:" not in path.read_text(), (
                f"{name} carries coverage config, which takes precedence over "
                "pyproject.toml and would shadow the committed floor"
            )


# --------------------------------------------------------------------------
# CI enforces the backend floor rather than reporting it
# --------------------------------------------------------------------------


def test_ci_collects_backend_coverage_so_the_floor_is_evaluated() -> None:
    """`fail_under` only fires on a run that measured something."""
    pytest_commands = [command for command in _run_commands("backend") if "pytest" in command]
    assert len(pytest_commands) == 1, (
        f"expected exactly one pytest step in the CI backend job, found {len(pytest_commands)}"
    )
    assert "--cov=backend" in pytest_commands[0], (
        "the CI pytest step does not pass --cov=backend, so pytest-cov computes no "
        "total and [tool.coverage.report].fail_under is never applied. The floor "
        "would be configured and unenforced — the exact failure CC.6 is about."
    )


def test_ci_neither_disables_nor_overrides_the_backend_floor() -> None:
    """A CLI flag beats the config; only a flag equal to the floor is allowed."""
    for command in _run_commands("backend"):
        assert "--no-cov" not in command, f"--no-cov disables the gate entirely: {command}"
        for value in re.findall(r"--cov-fail-under[=\s]+(\S+)", command):
            assert float(value) == BACKEND_FLOOR, (
                f"CI passes --cov-fail-under={value}, which overrides the committed "
                f"floor of {BACKEND_FLOOR} in pyproject.toml"
            )


def test_pytest_addopts_do_not_neutralise_the_floor() -> None:
    """Addopts apply to every run, including CI's."""
    addopts = str(_pyproject()["tool"]["pytest"]["ini_options"].get("addopts", ""))
    assert "--no-cov" not in addopts, "addopts disables coverage for every run"
    assert "--cov-fail-under" not in addopts, (
        "addopts overrides the committed floor for every run, including CI's"
    )


def test_the_coverage_steps_are_not_advisory() -> None:
    """`continue-on-error` turns a red gate into a warning nobody reads."""
    for name in ("backend", "frontend"):
        job = _job(name)
        assert not job.get("continue-on-error"), f"the {name} job is continue-on-error"
        for step in job["steps"]:
            assert not step.get("continue-on-error"), (
                f"a step in the {name} job is continue-on-error, so its failure "
                f"does not fail the build: {step.get('name', step.get('run'))!r}"
            )


# --------------------------------------------------------------------------
# The exclusion lists are themselves checked
# --------------------------------------------------------------------------


def test_the_backend_exclusion_list_is_the_committed_allowlist() -> None:
    """Excluding a file is the cheapest way to raise a percentage."""
    assert _coverage_run().get("omit") == COMMITTED_BACKEND_OMIT, (
        f"[tool.coverage.run].omit is {_coverage_run().get('omit')!r}, this test "
        f"pins {COMMITTED_BACKEND_OMIT!r}. Every entry removes code from the "
        "denominator, so the list is committed in two places rather than one."
    )


def test_every_backend_exclusion_states_why_beside_itself() -> None:
    """An unexplained omit is indistinguishable from hiding a hard file."""
    for entry, justification in _omit_entries_with_comments():
        assert len(justification) >= 60, (
            f"omit entry {entry!r} has no substantive justification comment above "
            f"it (found {len(justification)} characters). I6: the reason has to be "
            "about what the code *is*, not about how awkward it is to test."
        )


def test_no_extra_exclusion_patterns_are_configured() -> None:
    """`exclude_lines` / `exclude_also` can delete whole categories of code."""
    report = _coverage_report()
    for key in ("exclude_lines", "exclude_also"):
        assert key not in report, (
            f"[tool.coverage.report].{key} is set. It replaces or extends the "
            "default exclusion regex, which can drop arbitrary code from the "
            "measurement far more quietly than an omit entry. If a new pattern is "
            "genuinely needed, add it here too so the diff is visible."
        )


def test_pragma_no_cover_cannot_sprawl_through_shipped_code() -> None:
    """Every pragma is a hole in the measurement, so they are budgeted."""
    pragmas = _shipped_pragmas()
    listing = "\n".join(f"  {path}:{number}" for path, number, _ in pragmas)
    assert len(pragmas) <= SHIPPED_PRAGMA_BUDGET, (
        f"{len(pragmas)} `# pragma: no cover` in shipped backend code, budget is "
        f"{SHIPPED_PRAGMA_BUDGET}:\n{listing}\n"
        "A pragma removes a line from the denominator, so adding one raises the "
        "coverage percentage without testing anything (DIRECTIVE.md section 2, "
        "I6). Delete the pragma and test the line, or raise the budget here "
        "deliberately with the reason in the commit message."
    )


def test_every_shipped_pragma_states_why() -> None:
    """A bare pragma is an unexplained hole; a justified one can be reviewed."""
    unjustified = [
        f"{path}:{number}"
        for path, number, reason in _shipped_pragmas()
        if len(reason.strip(_PRAGMA_REASON_SEPARATORS)) < _MIN_PRAGMA_REASON_CHARS
    ]
    assert not unjustified, (
        "`# pragma: no cover` with no stated reason:\n"
        + "\n".join(f"  {location}" for location in unjustified)
        + "\nWrite why the line is unreachable beside it, e.g. "
        "`# pragma: no cover - the FK makes this unreachable`."
    )


# --------------------------------------------------------------------------
# The frontend floor exists, is committed, and CI evaluates it
# --------------------------------------------------------------------------


def test_the_frontend_floor_is_the_committed_number_on_all_four_metrics() -> None:
    """Statements, branches, functions and lines, or the gate has a soft side."""
    block = re.search(r"thresholds:\s*\{(?P<body>[^}]*)\}", _vitest_coverage_block())
    assert block is not None, "no coverage thresholds in vitest.config.mts"
    thresholds = dict(re.findall(r"(\w+)\s*:\s*([0-9.]+)", block.group("body")))
    assert set(thresholds) == {"statements", "branches", "functions", "lines"}, (
        f"frontend thresholds cover {sorted(thresholds)}; all four metrics must be "
        "held, or the gate can be cleared by covering statements while leaving "
        "the branches — the failure paths — untested"
    )
    for metric, value in thresholds.items():
        assert float(value) == FRONTEND_FLOOR, (
            f"frontend {metric} threshold is {value}, this test pins {FRONTEND_FLOOR} "
            "(DIRECTIVE.md section 8). Changing it is meant to be a deliberate edit "
            "in both places."
        )


def test_the_frontend_floor_does_not_auto_ratchet() -> None:
    """`autoUpdate` rewrites the thresholds to whatever the last run reached.

    That is not a ratchet: it ratifies a lucky run, and it can never report a
    decay because it moves with it.
    """
    assert re.search(r"autoUpdate:\s*false", _vitest_coverage_block()), (
        "vitest.config.mts must set `autoUpdate: false` explicitly in the "
        "coverage thresholds. It defaults to false, but stating it is what stops "
        "someone reaching for it when the floor next fails."
    )


def test_the_frontend_coverage_denominator_carves_nothing_out() -> None:
    """The whole first-party surface stays in, awkward files included."""
    assert _vitest_string_array("include") == COMMITTED_FRONTEND_INCLUDE, (
        "the vitest coverage `include` globs no longer match the committed set "
        f"{sorted(COMMITTED_FRONTEND_INCLUDE)}. Narrowing them removes files from "
        "the denominator, which raises the percentage without testing anything."
    )
    assert _vitest_string_array("exclude") == COMMITTED_FRONTEND_EXCLUDE, (
        "the vitest coverage `exclude` list no longer matches the committed set "
        f"{sorted(COMMITTED_FRONTEND_EXCLUDE)} (type declarations only, which "
        "contain no executable statements)"
    )
    for directory in ("app", "components", "lib"):
        assert (_FRONTEND / directory).is_dir(), (
            f"frontend/{directory} is in the coverage denominator but does not "
            "exist; the include globs and the tree have drifted apart"
        )


def test_ci_collects_frontend_coverage_so_the_thresholds_are_evaluated() -> None:
    """Without `--coverage` the thresholds are configured and never applied."""
    scripts = _package_scripts()
    assert "--coverage" in scripts.get("test:ci", ""), (
        "frontend `test:ci` does not pass --coverage, so vitest runs the suite and "
        "never evaluates the committed thresholds — configured, unenforced"
    )
    frontend_commands = _run_commands("frontend")
    assert any("npm run test:ci" in command for command in frontend_commands), (
        "the CI frontend job does not run `npm run test:ci`, which is the only "
        f"step that applies the {FRONTEND_FLOOR}% floor"
    )


# --------------------------------------------------------------------------
# The limits of the gate are documented where the gate is configured
# --------------------------------------------------------------------------


def test_what_the_gate_does_not_prove_is_documented_beside_the_numbers() -> None:
    """A gate this weak has to say so where someone will read it.

    97% line coverage with no assertions passes. The note next to each floor
    is the only thing standing between the number and a false reading of it,
    so it is load-bearing and asserted rather than left to survive edits on
    goodwill.
    """
    for path in (_PYPROJECT, _VITEST_CONFIG):
        text = path.read_text().upper()
        assert "WHAT THIS GATE DOES NOT PROVE" in text, (
            f"{path.name} configures a coverage floor without stating what it does "
            "not prove. DIRECTIVE.md section 2 I6 is about test honesty, and a "
            "coverage percentage is trivially satisfiable dishonestly — the "
            "caveat belongs next to the number, not in a commit message."
        )
        assert "HOW THE FLOOR MOVES" in text, (
            f"{path.name} does not say how its floor is meant to be changed; "
            "without that, the obvious response to a failing gate is to lower it"
        )
