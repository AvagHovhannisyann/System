"""Unit checks on the migration chain (no database).

Revision linkage and the presence of the P2.5 physical-layout statements in
revision 0003. Real behavior (hypertable conversion, triggers, indices) is
exercised against TimescaleDB in ``backend/tests/integration/``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType


def _load(module_name: str) -> ModuleType:
    return importlib.import_module(f"backend.db.migrations.versions.{module_name}")


def test_revision_chain_is_linear() -> None:
    baseline = _load("0001_baseline")
    fact_tables = _load("0002_bitemporal_fact_tables")
    physical = _load("0003_hypertable_append_only")
    assert baseline.revision == "0001"
    assert baseline.down_revision is None
    assert fact_tables.revision == "0002"
    assert fact_tables.down_revision == "0001"
    assert physical.revision == "0003"
    assert physical.down_revision == "0002"


def test_0003_declares_hypertable_indices_and_append_only_triggers() -> None:
    """Presence check on the raw-SQL migration; semantics are integration-tested."""
    module = _load("0003_hypertable_append_only")
    source = Path(str(module.__file__)).read_text()
    assert "create_hypertable" in source
    assert "'valid_from'" in source
    assert "INTERVAL '1 month'" in source
    assert "knowledge_time DESC" in source
    assert "BEFORE UPDATE OR DELETE" in source
