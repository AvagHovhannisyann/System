"""The alert-store contract against the in-memory double (P12.4).

Runs every body in ``store_contract.py``. What passing here does *not* establish
is written down in ``doubles.py``: the append-only triggers, the remaining CHECK
constraints and concurrency are properties of Postgres, exercised in
``backend/tests/integration/test_monitoring_alerts_db.py`` and nowhere else.
"""

from __future__ import annotations

import pytest

from backend.tests.monitoring import store_contract
from backend.tests.monitoring.doubles import InMemoryAlertStore


@pytest.mark.parametrize("contract", store_contract.CONTRACT, ids=lambda body: body.__name__)
async def test_contract_holds_for_the_in_memory_store(
    contract: object,
) -> None:
    assert callable(contract)
    await contract(InMemoryAlertStore())


def test_every_contract_body_is_wired_into_this_runner() -> None:
    """A property added to the contract must reach both backends, not just one."""
    marks = getattr(test_contract_holds_for_the_in_memory_store, "pytestmark", [])
    parametrized = [mark for mark in marks if mark.name == "parametrize"]
    assert len(parametrized) == 1
    assert len(parametrized[0].args[1]) == len(store_contract.CONTRACT)
