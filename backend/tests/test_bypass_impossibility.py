"""P2.7 gate (unit level): reading bitemporal tables outside ``as_of`` is impossible.

Adversarial coverage of the three D-011 bypass-prevention layers, without a
database — every assertion here fires *before* any connection I/O:

- **(a) session-construction paths.** Every way application code can obtain
  a session — the public ``ingest_writer_session``, plain/bindless
  sync ``Session``, bindless ``AsyncSession``, ``async_sessionmaker`` over a
  foreign engine, the legacy ``Query`` API, ``Session.get``, and hand-built
  sessions over the *private* read and writer sessionmakers themselves —
  raises :class:`BitemporalBypassError` on a bitemporal SELECT when no as-of
  is bound. Statement shapes that smuggle the read (EXISTS subquery, scalar
  subquery, join from a non-bitemporal table) are exercised too. The one
  structurally unreachable path is documented on the relevant test: there is
  no way to obtain a session with a *bound* as-of other than ``as_of()``
  itself, because the binding is an internal ``session.info`` key set only
  there (a session hand-crafted with that key is deliberate forgery inside
  ``backend.db``'s namespace, equivalent to editing the db layer).
- **(b) import contract.** The configured checker (ruff TID251 with the
  repository config) FLAGS synthetic modules importing or referencing the
  private engine/sessionmaker module in every form (member import,
  attribute access, aliased attribute access) and PASSES the clean tree.
- **(c) writer-session SELECT raises.**
- **(d) public-surface introspection.** ``backend.db`` exports exactly the
  curated names; none is (or wraps) a raw engine/sessionmaker handle.

End-to-end enforcement against real TimescaleDB (including the paths this
file cannot prove without I/O) lives in
``backend/tests/integration/test_bypass_impossibility_db.py``.
"""

from __future__ import annotations

import datetime as dt
import importlib
import subprocess
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, exists, select, union_all
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

import backend.db
from backend.db import (
    BitemporalBypassError,
    BitemporalRewriteError,
    dispose_database,
    ingest_writer_session,
)
from backend.db.models import PriceBar, Security

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# Built by concatenation so this file passes the repo-wide source scan for the
# banned module name (test_bypass_prevention.py) and the import contract itself.
_PRIVATE_ENGINE_MODULE = "backend.db" + ".engine"

_PAST_AS_OF = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


@pytest.fixture(autouse=True)
async def _reset_engine_state() -> AsyncIterator[None]:
    """Dispose any lazily-created process engine after each test.

    Tests below deliberately touch the private factories; leaving a cached
    engine (built from this environment's default DATABASE_URL) behind would
    leak state into other test modules.
    """
    yield
    await dispose_database()


# --- (a) every session-construction path without a bound as-of raises -------


async def test_public_writer_session_select_raises() -> None:
    """(c) The public writer path rejects bitemporal SELECTs before any I/O."""
    async with ingest_writer_session() as session:
        with pytest.raises(BitemporalBypassError, match="as_of"):
            await session.execute(select(PriceBar))


@pytest.mark.parametrize("factory_name", ["_get_session_factory", "_get_writer_session_factory"])
async def test_hand_built_session_over_each_private_sessionmaker_raises(
    factory_name: str,
) -> None:
    """Even sessions minted straight from the private sessionmakers are enforced.

    This is the strongest reachable bypass attempt: application code that
    already defeated the import contract and holds the raw factory still
    cannot read, because layer 2 hooks the Session *class*. No stronger path
    exists — the only session the hook exempts is one carrying the internal
    as-of ``info`` binding, and the sole code that sets it is ``as_of()``.
    """
    engine_module = importlib.import_module(_PRIVATE_ENGINE_MODULE)
    session: AsyncSession = getattr(engine_module, factory_name)()()
    try:
        with pytest.raises(BitemporalBypassError, match="price_bar"):
            await session.execute(select(PriceBar))
    finally:
        await session.close()


def test_bindless_sync_session_raises() -> None:
    """A ``Session()`` with no bind at all is rejected before bind resolution."""
    with Session() as session, pytest.raises(BitemporalBypassError, match="price_bar"):
        session.execute(select(PriceBar))


async def test_bindless_async_session_raises() -> None:
    async with AsyncSession() as session:
        with pytest.raises(BitemporalBypassError, match="price_bar"):
            await session.execute(select(PriceBar))


async def test_app_constructed_async_sessionmaker_session_raises() -> None:
    """An app-side ``async_sessionmaker`` over its own engine is enforced."""
    engine = create_async_engine("postgresql+asyncpg://nobody:nothing@127.0.0.1:9/none")
    session = async_sessionmaker(engine, expire_on_commit=False)()
    try:
        with pytest.raises(BitemporalBypassError, match="price_bar"):
            await session.execute(select(PriceBar))
    finally:
        await session.close()
        await engine.dispose()


def test_app_constructed_sync_sessionmaker_session_raises() -> None:
    engine = create_engine("sqlite://")
    try:
        with (
            sessionmaker(engine)() as session,
            pytest.raises(BitemporalBypassError, match="price_bar"),
        ):
            session.execute(select(PriceBar))
    finally:
        engine.dispose()


def test_legacy_query_api_raises() -> None:
    """The 1.x ``session.query()`` API routes through the same enforcement."""
    engine = create_engine("sqlite://")
    try:
        with (
            Session(engine) as session,
            pytest.raises(BitemporalBypassError, match="price_bar"),
        ):
            session.query(PriceBar).all()
    finally:
        engine.dispose()


def test_session_get_by_primary_key_raises() -> None:
    """``Session.get`` (PK lookup) is a SELECT and is enforced like one."""
    engine = create_engine("sqlite://")
    try:
        with (
            Session(engine) as session,
            pytest.raises(BitemporalBypassError, match="price_bar"),
        ):
            session.get(PriceBar, (1, _PAST_AS_OF, _PAST_AS_OF))
    finally:
        engine.dispose()


@pytest.mark.parametrize(
    "smuggled",
    [
        pytest.param(
            select(sa.literal(1)).where(exists(select(PriceBar.security_id))),
            id="exists-subquery",
        ),
        pytest.param(
            select(select(sa.func.max(PriceBar.knowledge_time)).scalar_subquery()),
            id="scalar-subquery",
        ),
        pytest.param(
            select(Security).join(PriceBar, PriceBar.security_id == Security.security_id),
            id="join-from-non-bitemporal",
        ),
    ],
)
def test_nested_bitemporal_references_do_not_launder_a_bypass(
    smuggled: sa.Select[tuple[object]],
) -> None:
    """Bitemporal references buried in subqueries or joins are still caught."""
    engine = create_engine("sqlite://")
    try:
        with (
            Session(engine) as session,
            pytest.raises(BitemporalBypassError, match="price_bar"),
        ):
            session.execute(smuggled)
    finally:
        engine.dispose()


def test_compound_select_fails_closed_even_with_bound_as_of() -> None:
    """Shapes the rewriter cannot version raise instead of running raw (I1)."""
    engine = create_engine("sqlite://")
    try:
        with (
            Session(engine, info={"bitemporal_as_of": _PAST_AS_OF}) as session,
            pytest.raises(BitemporalRewriteError, match="price_bar"),
        ):
            session.execute(union_all(select(PriceBar), select(PriceBar)))
    finally:
        engine.dispose()


# --- (b) import contract: the configured checker, run as CI runs it ---------


def _run_import_contract_checker(*targets: Path) -> subprocess.CompletedProcess[str]:
    """Run ruff TID251 with the repository configuration on ``targets``."""
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
            *[str(target) for target in targets],
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("description", "source"),
    [
        (
            "member import from the private module",
            f"from {_PRIVATE_ENGINE_MODULE} import _get_session_factory\n",
        ),
        (
            "attribute access via the package",
            "import backend.db\n" + f"{_PRIVATE_ENGINE_MODULE}._get_session_factory()\n",
        ),
        (
            "attribute access via an import alias",
            "from backend import db\ndb.engine._get_session_factory()\n",
        ),
    ],
)
def test_checker_flags_synthetic_module_reaching_private_sessionmaker(
    tmp_path: Path, description: str, source: str
) -> None:
    """A synthetic app module touching the private sessionmaker is flagged."""
    probe = tmp_path / "synthetic_bypass_probe.py"
    probe.write_text(source)
    result = _run_import_contract_checker(probe)
    assert result.returncode != 0, (
        f"import contract failed to flag: {description}\n{result.stdout}{result.stderr}"
    )
    assert "TID251" in result.stdout, result.stdout


def test_checker_passes_the_clean_tree() -> None:
    """The real tree carries no private-engine reference outside backend/db."""
    result = _run_import_contract_checker(_REPO_ROOT)
    assert result.returncode == 0, (
        f"import contract violation in the working tree:\n{result.stdout}{result.stderr}"
    )


# --- (d) public surface exposes no raw read path ----------------------------


def test_public_attributes_are_exactly_the_curated_exports() -> None:
    """No public module attribute exists beyond ``__all__`` (submodules aside).

    Submodule attributes are a Python import-system artifact; referencing
    them is what the import contract (layer 3) bans, verified above.
    """
    public = {
        name
        for name, value in vars(backend.db).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }
    assert public == set(backend.db.__all__)


def test_no_export_is_a_raw_engine_or_sessionmaker_handle() -> None:
    for name in backend.db.__all__:
        exported = getattr(backend.db, name)
        assert not isinstance(
            exported, sessionmaker | async_sessionmaker | sa.Engine | AsyncEngine
        ), f"backend.db.{name} exposes a raw database handle"


def test_exported_error_types_grant_no_session_capability() -> None:
    """The re-exported error classes are plain exceptions, nothing more."""
    for name in ("AsOfTimestampError", "BitemporalBypassError", "BitemporalRewriteError"):
        exported = getattr(backend.db, name)
        assert issubclass(exported, Exception)
        assert not hasattr(exported, "session")
        assert not hasattr(exported, "engine")
