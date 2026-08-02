"""The integration conftest must hand every test a freshly-built engine.

This file exists because of a defect that presented as flaky infrastructure.
``backend.db.engine`` caches one engine per process, built from
``get_settings().database_url`` the first time anything asks for it. The
integration conftest points ``DATABASE_URL`` at a throwaway container, but a
test module that sorts *earlier* than ``backend/tests/integration/`` can build
the engine first, against the default ``localhost:5432``. Disposing the engine
only on teardown leaves the session's first integration test holding that stale
handle.

The symptom is maximally misleading: exactly one test errors, with a connection
refusal, while every other integration test passes — because each of those was
handed a fresh engine by the previous test's teardown. It reads like a database
that was not ready. It is an ordering dependency.

Asserted structurally rather than by running the fixture, because reproducing it
needs a Docker daemon *and* a specific collection order. A structural test costs
nothing and runs everywhere, including the environments where the bug is
invisible.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_CONFTEST: Final = (
    Path(__file__).resolve().parents[2] / "backend" / "tests" / "integration" / "conftest.py"
)
_FIXTURE_NAME: Final = "_clean_database"
_DISPOSE: Final = "dispose_database"


def _fixture_body() -> list[ast.stmt]:
    """Return the statements of the autouse per-test database fixture."""
    tree = ast.parse(_CONFTEST.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == _FIXTURE_NAME:
            return node.body
    message = f"{_FIXTURE_NAME} not found in {_CONFTEST}"
    raise AssertionError(message)


def _calls_dispose(statements: list[ast.stmt]) -> bool:
    """Whether any statement in ``statements`` calls ``dispose_database``."""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == _DISPOSE
        for statement in statements
        for node in ast.walk(statement)
    )


def _yield_index(statements: list[ast.stmt]) -> int:
    """Index of the statement containing the fixture's ``yield``."""
    for index, statement in enumerate(statements):
        if any(isinstance(node, ast.Yield) for node in ast.walk(statement)):
            return index
    message = f"{_FIXTURE_NAME} has no yield; it is not a setup/teardown fixture"
    raise AssertionError(message)


def test_the_engine_is_disposed_before_each_integration_test_not_only_after() -> None:
    """Setup must dispose too, or the first integration test inherits a stale engine.

    A teardown-only disposal is correct for every test *except* the first, and
    the first is the one nobody looks at because the failure names a database
    connection rather than an ordering rule.
    """
    body = _fixture_body()
    split = _yield_index(body)

    assert _calls_dispose(body[:split]), (
        f"{_FIXTURE_NAME} must call {_DISPOSE}() before yielding: without it the "
        "session's first integration test uses whatever engine an earlier module "
        "built, which points at the default database rather than the container"
    )
    assert _calls_dispose(body[split + 1 :]), (
        f"{_FIXTURE_NAME} must still call {_DISPOSE}() after yielding, so the next "
        "test's event loop cannot inherit pooled connections bound to a dead loop"
    )


def test_this_contract_test_can_actually_fail() -> None:
    """Non-vacuity: the helpers detect absence, not merely presence.

    Without this, a refactor that broke ``_calls_dispose`` would turn the test
    above into one that passes for any fixture at all.
    """
    teardown_only = ast.parse("async def f():\n    yield\n    await dispose_database()\n").body[0]
    assert isinstance(teardown_only, ast.AsyncFunctionDef)
    split = _yield_index(teardown_only.body)
    assert not _calls_dispose(teardown_only.body[:split])
    assert _calls_dispose(teardown_only.body[split + 1 :])
