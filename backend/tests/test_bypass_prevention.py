"""P2.4 tests: the three D-011 bypass-prevention layers.

Layer 1 (curated public surface / private sessionmaker), layer 2 (runtime
Session-class enforcement, incl. sync sessions and textual SQL), and layer 3
(import contract: ruff TID251 banned-api, checked both as configuration and
functionally by running ruff on probe files). End-to-end enforcement against
a real database lives in ``backend/tests/integration/``.
"""

from __future__ import annotations

import ast
import datetime as dt
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, event, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import Session, aliased

import backend.db
from backend.db.asof import (
    BitemporalBypassError,
    BitemporalRewriteError,
    _enforce_bitemporal_reads,
)
from backend.db.models import PriceBar

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Built by concatenation so this file itself passes the source scan below.
_BANNED_MODULE = "backend.db." + "engine"
_BANNED_FROM_IMPORT = "from backend.db import " + "engine"

_EXPECTED_PUBLIC_SURFACE = {
    "AsOfBindIntegrityError",
    "AsOfTimestampError",
    "BitemporalBypassError",
    "BitemporalRewriteError",
    "as_of",
    "create_admin_engine",
    "dispose_database",
    "ingest_writer_session",
}


# --- Layer 1: curated surface, module-private sessionmaker -----------------


def test_backend_db_public_surface_is_exactly_the_curated_set() -> None:
    """backend.db exports as_of, the writer factory, admin helpers, error types — nothing else."""
    assert set(backend.db.__all__) == _EXPECTED_PUBLIC_SURFACE


def test_no_public_name_exposes_a_sessionmaker_or_engine() -> None:
    """No exported object is a raw sessionmaker/engine handle (D-011 layer 1)."""
    for name in backend.db.__all__:
        exported = getattr(backend.db, name)
        assert not isinstance(exported, async_sessionmaker), name
        assert not isinstance(exported, sa.Engine), name


def test_engine_module_exports_only_named_admin_helpers() -> None:
    """The private engine module's own __all__ is the two named admin helpers."""
    engine_module = sys.modules[_BANNED_MODULE]
    assert set(engine_module.__all__) == {"create_admin_engine", "dispose_database"}


# --- Layer 2: runtime enforcement at the Session class level ---------------


def test_hook_is_registered_on_the_session_class() -> None:
    """The enforcement hook listens on the ORM Session *class*, covering all sessions."""
    assert event.contains(Session, "do_orm_execute", _enforce_bitemporal_reads)


def test_plain_sync_session_cannot_read_bitemporal_tables() -> None:
    """Even a hand-built *sync* Session on an unrelated engine is enforced."""
    engine = create_engine("sqlite://")
    try:
        with Session(engine) as session, pytest.raises(BitemporalBypassError, match="price_bar"):
            session.execute(select(PriceBar))
    finally:
        engine.dispose()


def test_non_bitemporal_statements_pass_without_as_of() -> None:
    """Enforcement is scoped to bitemporal tables; ordinary statements run untouched."""
    engine = create_engine("sqlite://")
    try:
        with Session(engine) as session:
            assert session.execute(select(sa.literal(1))).scalar_one() == 1
    finally:
        engine.dispose()


def test_textual_sql_naming_a_bitemporal_table_raises() -> None:
    """Textual SQL cannot be rewritten, so naming a bitemporal table fails closed."""
    engine = create_engine("sqlite://")
    try:
        with Session(engine) as session, pytest.raises(BitemporalBypassError, match="price_bar"):
            session.execute(sa.text("SELECT close_usd FROM price_bar"))
    finally:
        engine.dispose()


def test_aliased_bitemporal_entity_fails_closed_even_with_as_of() -> None:
    """Shapes the rewriter does not support raise rather than run unversioned (I1)."""
    bound_as_of = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    engine = create_engine("sqlite://")
    try:
        with Session(engine, info={"bitemporal_as_of": bound_as_of}) as session:
            probe = aliased(PriceBar)
            with pytest.raises(BitemporalRewriteError, match="aliased"):
                session.execute(select(probe.close_usd))
    finally:
        engine.dispose()


# --- Layer 3: import contract (ruff TID251 banned-api) ---------------------


def test_import_contract_is_configured_in_pyproject() -> None:
    """Pyproject selects TID and bans the private engine module outside backend/db."""
    config = tomllib.loads(_PYPROJECT.read_text())
    lint = config["tool"]["ruff"]["lint"]
    assert "TID" in lint["select"]
    banned = lint["flake8-tidy-imports"]["banned-api"]
    assert _BANNED_MODULE in banned
    assert banned[_BANNED_MODULE]["msg"]
    assert "TID251" in lint["per-file-ignores"]["backend/db/**"]


def _run_ruff_tid251(probe: Path) -> subprocess.CompletedProcess[str]:
    """Run ruff (TID251 only, repo config) on ``probe`` and return the result."""
    return subprocess.run(  # noqa: S603 — fixed argv, no shell, test-only tooling call
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--select",
            "TID251",
            "--config",
            str(_PYPROJECT),
            str(probe),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_ruff_flags_private_engine_imports(tmp_path: Path) -> None:
    """Both import forms of the private engine module are rejected by TID251."""
    probe = tmp_path / "probe_banned.py"
    probe.write_text(f"import {_BANNED_MODULE}\n{_BANNED_FROM_IMPORT}\n")
    result = _run_ruff_tid251(probe)
    assert result.returncode != 0, result.stdout + result.stderr
    assert result.stdout.count("TID251") >= 2, result.stdout


def test_ruff_accepts_the_public_surface(tmp_path: Path) -> None:
    """Importing the curated backend.db surface passes the import contract."""
    probe = tmp_path / "probe_public.py"
    probe.write_text("from backend.db import as_of, ingest_writer_session\n")
    result = _run_ruff_tid251(probe)
    assert result.returncode == 0, result.stdout + result.stderr


_SANCTIONED_PRIVATE_ENGINE_USERS = frozenset(
    {
        # Per-test database reset: TRUNCATE of fact tables is textual SQL naming
        # those tables, which the Core guard refuses on every guarded engine
        # (fail-closed, D-011). Test reset is one of the two sanctioned uses
        # documented on the private migration-engine factory itself.
        # (Note: this file's own scan constants are split across a concatenation
        # so the module never matches itself — keep the literal out of comments.)
        "backend/tests/integration/conftest.py",
        # Append-only trigger tests: the triggers are the defense-in-depth layer
        # *beneath* the application guard, so exercising them requires reaching
        # Postgres directly. Routed through a guarded engine, these tests would
        # only re-assert the app guard and leave the triggers unverified.
        "backend/tests/integration/test_hypertable_append_only.py",
        # P3.8 FRED vintage tests: same two sanctioned reasons. The per-test
        # reset TRUNCATEs fact tables (textual SQL naming them, which the Core
        # guard refuses by design), and the vintage assertions must read the
        # RAW stored rows — knowledge_time and vintage_start_date as persisted.
        # Reading those through as_of() would show the versioned view, which is
        # the very thing under test: whether a revision was stored as a
        # later-knowledge row rather than an overwrite.
        "backend/tests/integration/test_fred_ingestion.py",
        # P3.2 EDGAR fact tables: same two reasons as the two entries above, for
        # the tables migration 0006 adds. Its append-only triggers and hypertable
        # layout must be checked beneath the guard, and the module resets its own
        # tables — the shared conftest reset cannot reach them, because they hold
        # no foreign key for its TRUNCATE ... CASCADE to follow.
        "backend/tests/integration/test_edgar_ingestion.py",
    }
)
"""Test-infrastructure files allowed to touch the private engine, each justified.

Deliberately an explicit allowlist rather than a blanket ``backend/tests``
exemption: application code must never import the private engine, and a *new*
test reaching for it should have to justify itself here rather than inherit a
silent exception.
"""


def _docstring_constants(tree: ast.Module) -> frozenset[int]:
    """Identify the string nodes that are docstrings rather than executable values."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return frozenset(docstrings)


def _private_engine_references(source: str) -> list[str]:
    """Return every way ``source`` reaches the private engine module, in code.

    Covers the three routes that actually grant a handle — ``import x``,
    ``from x import y``, and a string literal fed to a dynamic importer — and
    deliberately excludes docstrings and comments, which are prose *about* the
    ban rather than a use of it.
    """
    tree = ast.parse(source)
    docstrings = _docstring_constants(tree)
    references: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references += [
                f"import {alias.name}"
                for alias in node.names
                if alias.name == _BANNED_MODULE or alias.name.startswith(f"{_BANNED_MODULE}.")
            ]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == _BANNED_MODULE or module.startswith(f"{_BANNED_MODULE}."):
                references.append(f"from {module} import ...")
            elif module == "backend.db":
                references += [
                    f"from backend.db import {alias.name}"
                    for alias in node.names
                    if alias.name == _BANNED_MODULE.rsplit(".", 1)[1]
                ]
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and _BANNED_MODULE in node.value
        ):
            references.append(f"string literal {node.value!r}")
    return references


def test_no_source_outside_db_layer_imports_engine_internals() -> None:
    """Belt-and-braces source scan: nothing unsanctioned *reaches* the private engine.

    Parsed rather than grepped. The substring scan this replaces could not tell
    an import from a sentence, so a module that merely *documented* the ban —
    ``backend/features/compute.py`` explaining why it accepts a pinned session
    and constructs no engine — was reported as a violation. The failure mode
    that matters is the inverse: a scan people learn to placate by rewording
    comments is one they will eventually placate by rewording the wrong thing.

    Parsing loses nothing. Dynamic access still needs the module's name as a
    *string value*, and string constants are checked; only docstrings and
    comments, which cannot import anything, are exempt.
    """
    offenders: dict[str, list[str]] = {}
    for path in sorted((_REPO_ROOT / "backend").rglob("*.py")):
        relative = path.relative_to(_REPO_ROOT)
        if relative.parts[:2] == ("backend", "db"):
            continue
        if relative.as_posix() in _SANCTIONED_PRIVATE_ENGINE_USERS:
            continue
        found = _private_engine_references(path.read_text())
        if found:
            offenders[relative.as_posix()] = found
    assert offenders == {}


def test_the_source_scan_catches_every_route_to_the_private_engine() -> None:
    """Non-vacuity: each reachable form is detected, and prose about it is not.

    Without this, replacing the grep with a parser could quietly have replaced a
    noisy check with a silent one.
    """
    assert _private_engine_references(f"import {_BANNED_MODULE}")
    assert _private_engine_references(f"import {_BANNED_MODULE} as e")
    assert _private_engine_references(f"from {_BANNED_MODULE} import create_admin_engine")
    assert _private_engine_references(_BANNED_FROM_IMPORT)
    assert _private_engine_references(f'importlib.import_module("{_BANNED_MODULE}")')
    assert _private_engine_references(f'x = "{_BANNED_MODULE}"')

    assert not _private_engine_references(f'"""Prose naming {_BANNED_MODULE} in a docstring."""')
    assert not _private_engine_references(f"# a comment naming {_BANNED_MODULE}\nx = 1")
    assert not _private_engine_references("from backend.db import as_of, ingest_writer_session")


def test_sanctioned_private_engine_allowlist_has_no_stale_entries() -> None:
    """Every allowlisted file must exist and actually use the private engine.

    Keeps the exception list honest: a file that stops needing the private
    engine (or is renamed away) must be removed from the allowlist rather than
    leaving a dormant hole that a future file could silently occupy.
    """
    for entry in sorted(_SANCTIONED_PRIVATE_ENGINE_USERS):
        path = _REPO_ROOT / entry
        assert path.is_file(), f"allowlisted file no longer exists: {entry}"
        content = path.read_text()
        assert _BANNED_MODULE in content or _BANNED_FROM_IMPORT in content, (
            f"{entry} no longer uses the private engine — remove it from the allowlist"
        )
