"""Paper-only and I3 for the P11.3/P11.5 modules, and for the fixtures themselves.

``test_paper_only.py`` parametrises its scans over every module in
``backend/execution``, so ``reconciliation.py``, ``halt.py`` and ``killswitch.py``
are already covered by the token, import, URL, port and AST-parameter checks the
moment they exist. Two gaps are left, and this file closes them:

1. **The runtime signature scan** in that file iterates a fixed tuple of modules
   (``orders``, ``lifecycle``, ``idempotency``, ``store``). The new modules are
   scanned here through the same constant, so a re-export or a decorator cannot
   smuggle a venue parameter past the AST.
2. **Fixtures are obviously fixtures (I3).** No adapter exists — P11.1 is blocked
   on B2 — so a test snapshot claiming to be a paper-account statement would be a
   fabricated broker response presented as real. That is asserted over the test
   sources themselves rather than left to convention.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Final

import pytest

from backend.execution import halt, killswitch, reconciliation
from backend.execution.reconciliation import REPORTABLE_ORIGINS, SnapshotOrigin
from backend.tests.execution import control_fixtures
from backend.tests.execution.control_fixtures import reported_snapshot
from backend.tests.execution.test_paper_only import FORBIDDEN_PARAMETERS

CONTROL_MODULES: Final = (reconciliation, halt, killswitch)

TEST_ROOT: Final = Path(__file__).parents[1]
TEST_SOURCES: Final = (
    *sorted(TEST_ROOT.rglob("test_*.py")),
    Path(control_fixtures.__file__),
)

_BROKER_ORIGIN = re.compile(r"""origin\s*=\s*(SnapshotOrigin\.PAPER_BROKER|["']paper_broker["'])""")
"""A test constructing a snapshot that claims to have come from a paper account."""


def test_the_modules_and_sources_under_scan_all_exist() -> None:
    # Guards this file against becoming vacuous the way an empty glob would.
    assert {module.__name__.rsplit(".", 1)[-1] for module in CONTROL_MODULES} == {
        "reconciliation",
        "halt",
        "killswitch",
    }
    names = {path.name for path in TEST_SOURCES}
    for expected in (
        "control_fixtures.py",
        "test_reconciliation.py",
        "test_killswitch.py",
        "test_halt.py",
        "test_reconciliation_store.py",
        "test_control_migration.py",
    ):
        assert expected in names, expected
    assert (TEST_ROOT / "integration" / "test_reconciliation.py").is_file()


@pytest.mark.parametrize("module", CONTROL_MODULES, ids=lambda module: module.__name__)
def test_no_public_callable_admits_a_venue_client_or_endpoint(module: object) -> None:
    exported = getattr(module, "__all__", ())
    assert exported, module
    for name in exported:
        candidate = getattr(module, name)
        if not callable(candidate) or isinstance(candidate, type):
            continue
        parameters = set(inspect.signature(candidate).parameters)
        assert parameters & FORBIDDEN_PARAMETERS == set(), (module, name, parameters)


@pytest.mark.parametrize("module", CONTROL_MODULES, ids=lambda module: module.__name__)
def test_no_method_on_an_exported_type_admits_a_venue(module: object) -> None:
    # The AST scan covers `def` statements; this covers the same claim through the
    # constructed classes, where a generated __init__ (a dataclass's) lives.
    for name in getattr(module, "__all__", ()):
        candidate = getattr(module, name)
        if not isinstance(candidate, type):
            continue
        for attribute, member in vars(candidate).items():
            if not callable(member):
                continue
            try:
                parameters = set(inspect.signature(member).parameters)
            except (TypeError, ValueError):
                continue
            assert parameters & FORBIDDEN_PARAMETERS == set(), (module, name, attribute)


def test_no_snapshot_origin_could_denote_a_real_money_account() -> None:
    for member in SnapshotOrigin:
        assert not re.search(r"live|real_money|production", member.value, re.IGNORECASE)
        assert not re.search(r"LIVE|REAL_MONEY|PRODUCTION", member.name)


def test_our_own_ledger_cannot_stand_in_for_the_statement() -> None:
    assert SnapshotOrigin.INTERNAL_LEDGER not in REPORTABLE_ORIGINS
    assert {SnapshotOrigin.PAPER_BROKER, SnapshotOrigin.SIMULATED} == REPORTABLE_ORIGINS


def test_the_fixture_statement_is_labelled_simulated_not_paper_broker() -> None:
    # I3: no adapter exists (P11.1 is blocked on B2), so a fixture claiming to be
    # a paper-account statement would be a fabricated broker response presented
    # as real.
    assert reported_snapshot().origin is SnapshotOrigin.SIMULATED
    assert reported_snapshot().as_json()["origin"] == "simulated"


def test_no_test_anywhere_constructs_a_paper_account_snapshot() -> None:
    # Scanned over every test source rather than over the objects, so a snapshot
    # built in a branch no test currently reaches is caught too. Naming the enum
    # member is fine — assigning it as a snapshot's origin is not, because that is
    # a fabricated broker response presented as real (I3).
    offenders = [
        (path.name, match.group(0))
        for path in TEST_SOURCES
        for match in _BROKER_ORIGIN.finditer(path.read_text(encoding="utf-8"))
    ]
    assert not offenders


def test_the_control_modules_hold_no_transport_of_their_own() -> None:
    # Restated here as an explicit claim rather than only as an absence in a
    # parametrised scan elsewhere: these modules are handed a snapshot and a
    # session, and can reach nothing.
    for module in CONTROL_MODULES:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        assert "://" not in source
        for forbidden in ("import socket", "import httpx", "import requests", "ib_insync"):
            assert forbidden not in source, (module.__name__, forbidden)
