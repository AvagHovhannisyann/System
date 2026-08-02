"""P11.3 persistence: the row is the comparison's inputs and output, and it re-runs.

The claim under test is narrow and load-bearing: a verdict recorded today can be
re-derived tomorrow, from the row alone, at a different commit, and must produce
the same digest. Everything else here supports that — the payloads being stored
in full, the digests being taken over exactly those payloads, the tolerance being
copied onto the row rather than read from code at query time.

Against an in-memory double, so what is proved is the store's own behaviour. That
Postgres refuses a row whose ``matched`` disagrees with its ``break_count``, or a
``reported_origin`` naming a real account, is in
``backend/tests/integration/test_reconciliation.py``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from backend.execution.errors import ReconciliationReplayError
from backend.execution.reconciliation import (
    load_reconciliation,
    reconcile,
    record_reconciliation,
    rerun,
)
from backend.tests.execution.control_fixtures import (
    CYCLE_ID,
    OBSERVED_AT,
    ControlRows,
    ControlSessionDouble,
    as_session,
    broken_result,
    clean_result,
    internal_snapshot,
    reported_snapshot,
    restart,
)
from backend.tests.execution.fixtures import COMMIT_B, CONFIG_HASH_B, make_stamp


async def test_recording_a_verdict_stores_both_snapshots_and_their_digests() -> None:
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    result = broken_result()
    reconciliation_id = await record_reconciliation(session, result)
    row = rows.reconciliations[0]
    assert row["reconciliation_id"] == reconciliation_id
    assert row["cycle_id"] == CYCLE_ID
    assert row["internal_snapshot"] == result.internal.as_json()
    assert row["reported_snapshot"] == result.reported.as_json()
    assert row["internal_digest"] == result.internal.digest
    assert row["reported_digest"] == result.reported.digest
    assert row["result_digest"] == result.result_digest


async def test_the_row_records_the_verdict_and_the_tolerance_that_produced_it() -> None:
    # The tolerance is copied rather than joined at read time: one widened next
    # month must not rewrite last month's verdict.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(cash_usd=Decimal("100.00")),
        reported=reported_snapshot(cash_usd=Decimal("99.00")),
        stamp=make_stamp(),
        cash_tolerance_usd=Decimal("0.02"),
    )
    await record_reconciliation(session, result)
    row = rows.reconciliations[0]
    assert row["cash_tolerance_usd"] == Decimal("0.02")
    assert row["matched"] is False
    assert row["break_count"] == 1
    assert row["finding_count"] == 1


async def test_a_clean_verdict_records_as_matched_with_no_breaks() -> None:
    rows = ControlRows()
    await record_reconciliation(as_session(ControlSessionDouble(rows)), clean_result())
    row = rows.reconciliations[0]
    assert row["matched"] is True
    assert row["break_count"] == 0
    assert row["finding_count"] == 0


async def test_an_observation_only_verdict_still_matches_but_is_counted() -> None:
    # A flat position one side did not name is recorded and does not halt: the
    # two counts differ, which is exactly the distinction being preserved.
    rows = ControlRows()
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={5: 0}),
        reported=reported_snapshot(positions={}),
        stamp=make_stamp(),
    )
    await record_reconciliation(as_session(ControlSessionDouble(rows)), result)
    row = rows.reconciliations[0]
    assert row["matched"] is True
    assert row["break_count"] == 0
    assert row["finding_count"] == 1


async def test_the_row_carries_all_four_i2_stamp_components() -> None:
    rows = ControlRows()
    stamp = make_stamp()
    await record_reconciliation(as_session(ControlSessionDouble(rows)), clean_result())
    row = rows.reconciliations[0]
    assert row["git_commit"] == stamp.git_commit
    assert row["git_dirty"] == stamp.git_dirty
    assert row["data_version"] == stamp.data_version
    assert row["config_hash"] == stamp.config_hash
    assert row["seed"] == stamp.seed


async def test_the_row_records_both_observation_instants() -> None:
    rows = ControlRows()
    await record_reconciliation(as_session(ControlSessionDouble(rows)), clean_result())
    row = rows.reconciliations[0]
    assert row["internal_observed_at"] == OBSERVED_AT
    assert row["reported_observed_at"] == OBSERVED_AT


async def test_the_reported_origin_recorded_is_the_fixture_origin_not_a_broker_one() -> None:
    # I3: nothing constructed in a test may be recorded as something a venue said.
    rows = ControlRows()
    await record_reconciliation(as_session(ControlSessionDouble(rows)), clean_result())
    assert rows.reconciliations[0]["reported_origin"] == "simulated"
    assert rows.reconciliations[0]["internal_origin"] == "internal_ledger"


async def test_recording_does_not_commit_the_callers_transaction() -> None:
    # The caller owns the transaction, matching backend.execution.store. The
    # double has no commit at all, so a store that called one would fail here.
    session: Any = ControlSessionDouble(ControlRows())
    assert not hasattr(session, "commit")
    await record_reconciliation(as_session(session), clean_result())


async def test_a_recorded_verdict_reruns_to_the_same_digest_after_a_restart() -> None:
    # The whole re-runnability chain, end to end: record it, throw the process
    # away, read the row back, re-derive the verdict at a different commit.
    rows = ControlRows()
    result = broken_result()
    reconciliation_id = await record_reconciliation(as_session(ControlSessionDouble(rows)), result)
    reborn = as_session(restart(rows))
    stored = await load_reconciliation(reborn, reconciliation_id)
    assert stored.cycle_id == result.cycle_id
    assert stored.result_digest == result.result_digest
    assert stored.break_count == len(result.breaks)
    assert stored.matched == result.matched
    replayed = rerun(stored, stamp=make_stamp(git_commit=COMMIT_B, config_hash=CONFIG_HASH_B))
    assert replayed.result_digest == result.result_digest
    assert [finding.kind for finding in replayed.findings] == [
        finding.kind for finding in result.findings
    ]
    assert [finding.detail for finding in replayed.findings] == [
        finding.detail for finding in result.findings
    ]


async def test_loading_a_verdict_that_was_never_recorded_is_an_error() -> None:
    with pytest.raises(ReconciliationReplayError, match="cannot be investigated"):
        await load_reconciliation(as_session(ControlSessionDouble(ControlRows())), 999)


async def test_two_verdicts_in_one_cycle_are_both_kept() -> None:
    # Append-only: a correction is a new reconciliation, never an edit of the one
    # that was wrong.
    rows = ControlRows()
    session = as_session(ControlSessionDouble(rows))
    first = await record_reconciliation(session, broken_result())
    second = await record_reconciliation(session, clean_result())
    assert first != second
    assert len(rows.reconciliations) == 2
    assert (await load_reconciliation(session, first)).matched is False
    assert (await load_reconciliation(session, second)).matched is True
