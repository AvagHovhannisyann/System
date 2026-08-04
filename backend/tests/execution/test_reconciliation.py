"""P11.3: the comparison itself — tolerance, the asymmetric cases, and re-runnability.

Both directions of the gate are here, and both matter equally. An injected
mismatch must be caught (``test_*_is_a_break``), and a clean pair of snapshots
must **not** produce a finding (``test_agreeing_snapshots_reconcile_cleanly`` and
the property tests): a reconciliation that fires on a healthy book gets ignored
within a week, which is the same outcome as one that never fires.

The boundary tests are deliberate pairs. A tolerance is only meaningful if both
"exactly at it" and "one representable unit past it" are pinned, because a
mutation from ``<=`` to ``<`` changes only one of them.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.execution.errors import (
    ReconciliationMismatchError,
    ReconciliationReplayError,
    SnapshotValidationError,
)
from backend.execution.reconciliation import (
    BREAKING_KINDS,
    CASH_TOLERANCE_USD,
    MAX_CASH_TOLERANCE_USD,
    MAX_SNAPSHOT_SKEW_SECONDS,
    SNAPSHOT_SCHEMA,
    Finding,
    MismatchKind,
    PositionSnapshot,
    Severity,
    SnapshotOrigin,
    StoredReconciliation,
    reconcile,
    require_matched,
    rerun,
    snapshot_of_json,
)
from backend.tests.execution.control_fixtures import (
    CYCLE_ID,
    OBSERVED_AT,
    internal_snapshot,
    reported_snapshot,
)
from backend.tests.execution.fixtures import COMMIT_B, CONFIG_HASH_B, make_stamp

CENT = Decimal("0.01")
ULP = Decimal("0.000001")
"""One unit at the schema's six-decimal scale — the smallest step past a boundary."""


def compare(
    *,
    internal_positions: dict[int, int] | None = None,
    reported_positions: dict[int, int] | None = None,
    internal_cash: Decimal = Decimal("100000.00"),
    reported_cash: Decimal = Decimal("100000.00"),
    tolerance: Decimal = CASH_TOLERANCE_USD,
) -> tuple[Finding, ...]:
    """Reconcile two constructed snapshots and return the findings."""
    return reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions=internal_positions, cash_usd=internal_cash),
        reported=reported_snapshot(positions=reported_positions, cash_usd=reported_cash),
        stamp=make_stamp(),
        cash_tolerance_usd=tolerance,
    ).findings


def only(findings: tuple[Finding, ...]) -> Finding:
    """Return the single finding, asserting there is exactly one."""
    assert len(findings) == 1, [finding.kind.value for finding in findings]
    return findings[0]


# ---------------------------------------------------------------------------
# The clean direction: no false positives.
# ---------------------------------------------------------------------------


def test_agreeing_snapshots_reconcile_cleanly() -> None:
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 100, 2: -50, 3: 0}),
        reported=reported_snapshot(positions={1: 100, 2: -50, 3: 0}),
        stamp=make_stamp(),
    )
    assert result.findings == ()
    assert result.matched
    assert result.breaks == ()


def test_two_explicit_zeros_produce_no_finding_at_all() -> None:
    # Both sides looked and both said flat: nothing is asymmetric, so there is
    # nothing to record.
    assert compare(internal_positions={5: 0}, reported_positions={5: 0}) == ()


def test_identical_cash_produces_no_finding() -> None:
    assert compare(internal_cash=Decimal("1234.56"), reported_cash=Decimal("1234.56")) == ()


# ---------------------------------------------------------------------------
# Cash tolerance, pinned on both sides of the boundary.
# ---------------------------------------------------------------------------


def test_a_difference_exactly_at_the_tolerance_is_within_it() -> None:
    assert (
        compare(internal_cash=Decimal("100000.00"), reported_cash=Decimal("100000.00") - CENT) == ()
    )


def test_one_representable_unit_past_the_tolerance_is_a_break() -> None:
    finding = only(
        compare(
            internal_cash=Decimal("100000.00"),
            reported_cash=Decimal("100000.00") - CENT - ULP,
        )
    )
    assert finding.kind is MismatchKind.CASH_DISAGREES
    assert finding.severity is Severity.BREAK
    assert finding.difference_usd == CENT + ULP
    assert finding.tolerance_usd == CASH_TOLERANCE_USD


def test_the_cash_difference_is_signed_so_the_direction_is_on_the_record() -> None:
    # Our books above the statement, and below it, are different situations: the
    # first is cash we think we have and do not.
    over = only(compare(internal_cash=Decimal("100.00"), reported_cash=Decimal("90.00")))
    under = only(compare(internal_cash=Decimal("90.00"), reported_cash=Decimal("100.00")))
    assert over.difference_usd == Decimal("10.00")
    assert under.difference_usd == Decimal("-10.00")


def test_a_tightened_tolerance_catches_what_the_default_absorbs() -> None:
    # Zero tolerance is allowed: tightening is always permitted.
    half_cent = Decimal("0.005")
    assert compare(internal_cash=Decimal("10.00"), reported_cash=Decimal("10.00") - half_cent) == ()
    finding = only(
        compare(
            internal_cash=Decimal("10.00"),
            reported_cash=Decimal("10.00") - half_cent,
            tolerance=Decimal("0"),
        )
    )
    assert finding.kind is MismatchKind.CASH_DISAGREES


def test_a_tolerance_above_the_ceiling_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="exceeds the ceiling"):
        compare(tolerance=MAX_CASH_TOLERANCE_USD + ULP)


def test_the_ceiling_itself_is_accepted() -> None:
    assert compare(tolerance=MAX_CASH_TOLERANCE_USD) == ()


def test_a_negative_tolerance_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="negative"):
        compare(tolerance=Decimal("-0.01"))


def test_the_default_tolerance_bounds_one_quantisation_step_and_nothing_more() -> None:
    # The derivation, asserted rather than only argued: half a cent is the most a
    # two-decimal rendering can move a six-decimal value, and the tolerance is
    # twice that. A whole cent of real activity is not absorbed.
    assert 2 * Decimal("0.005") == CASH_TOLERANCE_USD
    assert compare(internal_cash=Decimal("100.004999"), reported_cash=Decimal("100.00")) == ()
    assert len(compare(internal_cash=Decimal("100.02"), reported_cash=Decimal("100.00"))) == 1


def test_the_three_thresholds_are_pinned_to_their_stated_magnitudes() -> None:
    # Absolute, not relative to the constants themselves.
    #
    # This exists because a mutation run caught the gap: every other threshold
    # test is written as `MAX_CASH_TOLERANCE_USD + ULP` or
    # `MAX_SNAPSHOT_SKEW_SECONDS + 1`, which follows the constant wherever it
    # goes. Widening the skew limit to a billion seconds left the whole suite
    # green, because the tests measured the limit against itself. A threshold's
    # *magnitude* is the claim; it has to be asserted as a literal.
    assert Decimal("0.01") == CASH_TOLERANCE_USD
    assert Decimal("0.05") == MAX_CASH_TOLERANCE_USD
    assert MAX_SNAPSHOT_SKEW_SECONDS == 300


def test_a_five_minute_skew_is_refused_at_a_literal_magnitude() -> None:
    # The same claim as the relative test below, stated so that widening the
    # constant fails here rather than silently moving the boundary.
    with pytest.raises(SnapshotValidationError, match="apart"):
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(),
            reported=reported_snapshot(observed_at=OBSERVED_AT + dt.timedelta(seconds=301)),
            stamp=make_stamp(),
        )


def test_a_tolerance_of_ten_cents_is_refused_at_a_literal_magnitude() -> None:
    with pytest.raises(SnapshotValidationError, match="exceeds the ceiling"):
        compare(tolerance=Decimal("0.10"))


# ---------------------------------------------------------------------------
# The four asymmetric position cases.
# ---------------------------------------------------------------------------


def test_a_position_the_statement_holds_and_we_do_not_is_the_most_serious_break() -> None:
    finding = only(compare(internal_positions={}, reported_positions={42: 300}))
    assert finding.kind is MismatchKind.POSITION_UNKNOWN_TO_US
    assert finding.severity is Severity.BREAK
    assert finding.security_id == 42
    assert finding.internal_shares is None
    assert finding.reported_shares == 300
    assert "did not name the instrument" in finding.detail


def test_an_explicit_flat_on_our_side_is_a_different_record_from_an_absence() -> None:
    # Same economic break, different fact: we *looked* and said flat, which is a
    # contradiction rather than a gap. The record keeps them apart.
    silent = only(compare(internal_positions={}, reported_positions={42: 300}))
    explicit = only(compare(internal_positions={42: 0}, reported_positions={42: 300}))
    assert silent.kind is explicit.kind is MismatchKind.POSITION_UNKNOWN_TO_US
    assert silent.internal_shares is None
    assert explicit.internal_shares == 0
    assert "did not name the instrument" in silent.detail
    assert "says flat" in explicit.detail


def test_a_position_we_hold_and_the_statement_does_not_is_a_break_of_its_own_kind() -> None:
    finding = only(compare(internal_positions={42: 300}, reported_positions={}))
    assert finding.kind is MismatchKind.POSITION_UNKNOWN_TO_STATEMENT
    assert finding.severity is Severity.BREAK
    assert finding.internal_shares == 300
    assert finding.reported_shares is None
    assert "sale of shares that are not there" in finding.detail


def test_the_two_directions_are_never_conflated() -> None:
    # The distinction the brief calls out: broker-has-what-we-do-not is not the
    # same finding as we-have-what-broker-does-not, and a taxonomy that merged
    # them would make the more serious one invisible.
    theirs = only(compare(reported_positions={42: 300}))
    ours = only(compare(internal_positions={42: 300}))
    assert theirs.kind is not ours.kind
    assert theirs.kind in BREAKING_KINDS
    assert ours.kind in BREAKING_KINDS


def test_differing_quantities_on_both_sides_are_a_quantity_disagreement() -> None:
    finding = only(compare(internal_positions={42: 300}, reported_positions={42: 250}))
    assert finding.kind is MismatchKind.POSITION_QUANTITY_DISAGREES
    assert finding.severity is Severity.BREAK
    assert finding.internal_shares == 300
    assert finding.reported_shares == 250
    assert "difference of 50" in finding.detail


def test_a_sign_reversal_is_flagged_inside_the_quantity_disagreement() -> None:
    finding = only(compare(internal_positions={42: 300}, reported_positions={42: -300}))
    assert finding.kind is MismatchKind.POSITION_QUANTITY_DISAGREES
    assert "sign reversed" in finding.detail


def test_a_single_share_of_difference_is_a_break_because_positions_have_no_tolerance() -> None:
    finding = only(compare(internal_positions={42: 300}, reported_positions={42: 299}))
    assert finding.severity is Severity.BREAK


def test_flat_on_both_sides_with_one_silent_is_recorded_but_is_not_a_break() -> None:
    # The distinction the whole taxonomy exists for: an explicit zero is a
    # statement, an absence is silence, and collapsing them makes a truncated feed
    # look like a confirmed flat book.
    for internal, reported in (({}, {5: 0}), ({5: 0}, {})):
        result = reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(positions=internal),
            reported=reported_snapshot(positions=reported),
            stamp=make_stamp(),
        )
        finding = only(result.findings)
        assert finding.kind is MismatchKind.POSITION_FLAT_BUT_ONE_SIDE_SILENT
        assert finding.severity is Severity.OBSERVATION
        assert not finding.is_break
        assert result.matched, "an observation must not halt a cycle"
        assert result.breaks == ()


def test_an_observation_is_still_persisted_in_the_findings() -> None:
    # It must not be discarded merely because it does not halt: the point of
    # recording it is that a truncated statement is detectable afterwards.
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={5: 0}),
        reported=reported_snapshot(positions={}),
        stamp=make_stamp(),
    )
    payload = result.as_json()
    findings = payload["findings"]
    assert isinstance(findings, list)
    assert len(findings) == 1


def test_every_mismatch_kind_has_a_declared_severity() -> None:
    # The import-time guard, restated as a test so the intent is visible: a kind
    # added later must be classified, never defaulted.
    for kind in MismatchKind:
        assert (kind in BREAKING_KINDS) == (
            kind is not MismatchKind.POSITION_FLAT_BUT_ONE_SIDE_SILENT
        )
    assert BREAKING_KINDS, "a reconciliation with no breaking kind could never halt anything"


# ---------------------------------------------------------------------------
# Ordering and determinism.
# ---------------------------------------------------------------------------


def test_findings_are_ordered_cash_first_then_by_security_id() -> None:
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={9: 1, 3: 1, 11: 1}, cash_usd=Decimal("1.00")),
        reported=reported_snapshot(positions={9: 2, 3: 2, 11: 2}, cash_usd=Decimal("99.00")),
        stamp=make_stamp(),
    )
    assert [finding.security_id for finding in result.findings] == [None, 3, 9, 11]


def test_the_verdict_does_not_depend_on_the_order_positions_were_supplied_in() -> None:
    forward = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 5, 2: 6, 3: 7}),
        reported=reported_snapshot(positions={1: 9, 2: 6, 3: 7}),
        stamp=make_stamp(),
    )
    backward = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={3: 7, 2: 6, 1: 5}),
        reported=reported_snapshot(positions={3: 7, 2: 6, 1: 9}),
        stamp=make_stamp(),
    )
    assert forward.result_digest == backward.result_digest
    assert forward.internal.digest == backward.internal.digest


def test_the_same_comparison_run_twice_produces_the_same_digest() -> None:
    first = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 5}),
        reported=reported_snapshot(positions={1: 6}),
        stamp=make_stamp(),
    )
    second = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 5}),
        reported=reported_snapshot(positions={1: 6}),
        stamp=make_stamp(),
    )
    assert first.result_digest == second.result_digest


def test_the_result_digest_excludes_the_stamp_so_a_later_rerun_can_match() -> None:
    # The property re-runnability rests on. If the stamp were in the digest, a
    # re-run at a later commit would differ by construction and nothing could be
    # verified.
    today = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 5}),
        reported=reported_snapshot(positions={1: 6}),
        stamp=make_stamp(),
    )
    later = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 5}),
        reported=reported_snapshot(positions={1: 6}),
        stamp=make_stamp(git_commit=COMMIT_B, config_hash=CONFIG_HASH_B, seed=999),
    )
    assert today.stamp != later.stamp
    assert today.result_digest == later.result_digest


def test_the_result_digest_changes_when_any_input_changes() -> None:
    base = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 5}),
        reported=reported_snapshot(positions={1: 6}),
        stamp=make_stamp(),
    )
    variants = (
        reconcile(
            cycle_id="another-cycle",
            internal=internal_snapshot(positions={1: 5}),
            reported=reported_snapshot(positions={1: 6}),
            stamp=make_stamp(),
        ),
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(positions={1: 4}),
            reported=reported_snapshot(positions={1: 6}),
            stamp=make_stamp(),
        ),
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(positions={1: 5}, cash_usd=Decimal("1.00")),
            reported=reported_snapshot(positions={1: 6}),
            stamp=make_stamp(),
        ),
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(positions={1: 5}),
            reported=reported_snapshot(positions={1: 6}),
            stamp=make_stamp(),
            cash_tolerance_usd=Decimal("0"),
        ),
    )
    for variant in variants:
        assert variant.result_digest != base.result_digest


def test_a_snapshot_digest_is_the_digest_of_the_payload_stored_beside_it() -> None:
    snapshot = internal_snapshot(positions={2: 7, 1: 3})
    payload = snapshot.as_json()
    assert payload["schema"] == SNAPSHOT_SCHEMA
    assert payload["positions"] == [[1, 3], [2, 7]]
    assert snapshot_of_json(payload).digest == snapshot.digest


def test_equal_instants_written_in_different_zones_digest_identically() -> None:
    utc = internal_snapshot(observed_at=OBSERVED_AT)
    shifted = internal_snapshot(
        observed_at=OBSERVED_AT.astimezone(dt.timezone(dt.timedelta(hours=5)))
    )
    assert utc.digest == shifted.digest


def test_equal_cash_written_at_different_scales_digests_identically() -> None:
    assert (
        internal_snapshot(cash_usd=Decimal("1.5")).digest
        == internal_snapshot(cash_usd=Decimal("1.500000")).digest
    )


# ---------------------------------------------------------------------------
# Snapshot validation: failing towards a halt.
# ---------------------------------------------------------------------------


def test_a_non_finite_cash_balance_is_refused_at_construction() -> None:
    # The fail-open this prevents: NaN compares false against every tolerance, so
    # a reconciliation against a NaN balance would report a clean book.
    for value in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
        with pytest.raises(SnapshotValidationError, match="not finite"):
            internal_snapshot(cash_usd=value)


def test_cash_beyond_the_schema_scale_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="decimal places"):
        internal_snapshot(cash_usd=Decimal("1.0000001"))


def test_a_naive_observation_instant_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="timezone-naive"):
        internal_snapshot(observed_at=dt.datetime(2026, 8, 3, 15, 0))  # noqa: DTZ001


def test_a_boolean_share_count_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="bool is refused"):
        PositionSnapshot(
            origin=SnapshotOrigin.INTERNAL_LEDGER,
            observed_at=OBSERVED_AT,
            cash_usd=Decimal("0"),
            positions={1: True},
        )


def test_a_non_positive_security_id_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="security_id must be positive"):
        internal_snapshot(positions={0: 10})


def test_snapshots_taken_too_far_apart_are_refused_rather_than_compared() -> None:
    # The mirror of a too-wide tolerance: a comparison that manufactures breaks.
    late = OBSERVED_AT + dt.timedelta(seconds=MAX_SNAPSHOT_SKEW_SECONDS + 1)
    with pytest.raises(SnapshotValidationError, match="apart"):
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(),
            reported=reported_snapshot(observed_at=late),
            stamp=make_stamp(),
        )


def test_snapshots_exactly_at_the_skew_limit_are_compared() -> None:
    edge = OBSERVED_AT + dt.timedelta(seconds=MAX_SNAPSHOT_SKEW_SECONDS)
    assert (
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(),
            reported=reported_snapshot(observed_at=edge),
            stamp=make_stamp(),
        ).findings
        == ()
    )


def test_reconciling_our_ledger_against_itself_is_refused() -> None:
    # It would always pass and prove nothing — the most comfortable way for this
    # control to become decorative.
    with pytest.raises(SnapshotValidationError, match="proves nothing"):
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(),
            reported=internal_snapshot(),
            stamp=make_stamp(),
        )


def test_a_statement_supplied_as_the_internal_side_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="internal snapshot's origin"):
        reconcile(
            cycle_id=CYCLE_ID,
            internal=reported_snapshot(),
            reported=reported_snapshot(),
            stamp=make_stamp(),
        )


def test_a_blank_cycle_id_is_refused() -> None:
    with pytest.raises(SnapshotValidationError, match="cycle_id is blank"):
        reconcile(
            cycle_id="   ",
            internal=internal_snapshot(),
            reported=reported_snapshot(),
            stamp=make_stamp(),
        )


def test_no_snapshot_origin_denotes_a_real_money_account() -> None:
    # I3 at the type level, in the same shape as FillSource.
    assert {member.value for member in SnapshotOrigin} == {
        "internal_ledger",
        "paper_broker",
        "simulated",
    }


# ---------------------------------------------------------------------------
# require_matched, and re-running a stored verdict.
# ---------------------------------------------------------------------------


def test_require_matched_is_silent_on_a_clean_verdict() -> None:
    require_matched(
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(positions={1: 1}),
            reported=reported_snapshot(positions={1: 1}),
            stamp=make_stamp(),
        )
    )


def test_require_matched_raises_on_a_break_and_names_the_cycle() -> None:
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={1: 1}),
        reported=reported_snapshot(positions={1: 2}),
        stamp=make_stamp(),
    )
    with pytest.raises(ReconciliationMismatchError) as caught:
        require_matched(result)
    assert caught.value.cycle_id == CYCLE_ID
    assert caught.value.break_count == 1
    assert caught.value.result_digest == result.result_digest


def test_require_matched_does_not_raise_on_an_observation_only_verdict() -> None:
    require_matched(
        reconcile(
            cycle_id=CYCLE_ID,
            internal=internal_snapshot(positions={1: 0}),
            reported=reported_snapshot(positions={}),
            stamp=make_stamp(),
        )
    )


def stored_from(result_cycle: str = CYCLE_ID) -> tuple[StoredReconciliation, str]:
    """Build a stored row from a broken verdict, as ``record_reconciliation`` would."""
    result = reconcile(
        cycle_id=result_cycle,
        internal=internal_snapshot(positions={1: 100, 2: 0}),
        reported=reported_snapshot(positions={1: 90, 3: 5}, cash_usd=Decimal("99999.00")),
        stamp=make_stamp(),
    )
    return (
        StoredReconciliation(
            reconciliation_id=1,
            cycle_id=result.cycle_id,
            internal_payload=result.internal.as_json(),
            reported_payload=result.reported.as_json(),
            cash_tolerance_usd=result.cash_tolerance_usd,
            result_digest=result.result_digest,
            matched=result.matched,
            break_count=len(result.breaks),
        ),
        result.result_digest,
    )


def test_a_stored_verdict_reruns_to_the_same_digest_under_a_different_stamp() -> None:
    # The claim "deterministic and re-runnable on a stored snapshot", executed: a
    # later commit, a later data version, a different seed — same verdict.
    stored, digest = stored_from()
    replayed = rerun(stored, stamp=make_stamp(git_commit=COMMIT_B, config_hash=CONFIG_HASH_B))
    assert replayed.result_digest == digest
    assert [finding.kind for finding in replayed.findings] == [
        MismatchKind.CASH_DISAGREES,
        MismatchKind.POSITION_QUANTITY_DISAGREES,
        MismatchKind.POSITION_FLAT_BUT_ONE_SIDE_SILENT,
        MismatchKind.POSITION_UNKNOWN_TO_US,
    ]


def test_a_rerun_over_a_tampered_snapshot_is_refused_rather_than_reported() -> None:
    stored, _ = stored_from()
    tampered = dict(stored.reported_payload)
    tampered["positions"] = [[1, 100], [3, 5]]
    with pytest.raises(ReconciliationReplayError, match="cannot be presented"):
        rerun(
            StoredReconciliation(
                reconciliation_id=stored.reconciliation_id,
                cycle_id=stored.cycle_id,
                internal_payload=stored.internal_payload,
                reported_payload=tampered,
                cash_tolerance_usd=stored.cash_tolerance_usd,
                result_digest=stored.result_digest,
                matched=stored.matched,
                break_count=stored.break_count,
            ),
            stamp=make_stamp(),
        )


def test_a_payload_from_another_schema_version_is_refused() -> None:
    payload = dict(internal_snapshot().as_json())
    payload["schema"] = "execution.reconciliation.snapshot.v0"
    with pytest.raises(SnapshotValidationError, match="declares schema"):
        snapshot_of_json(payload)


def test_a_payload_missing_a_field_is_refused_rather_than_half_read() -> None:
    payload = dict(internal_snapshot().as_json())
    del payload["cash_usd"]
    with pytest.raises(SnapshotValidationError, match="missing"):
        snapshot_of_json(payload)


def test_a_payload_whose_positions_are_malformed_is_refused() -> None:
    payload = dict(internal_snapshot(positions={1: 2}).as_json())
    payload["positions"] = [[1, 2, 3]]
    with pytest.raises(SnapshotValidationError, match="pair"):
        snapshot_of_json(payload)


# ---------------------------------------------------------------------------
# Properties.
# ---------------------------------------------------------------------------

_POSITIONS = st.dictionaries(
    st.integers(min_value=1, max_value=40),
    st.integers(min_value=-1000, max_value=1000),
    max_size=8,
)
_CASH = st.decimals(
    min_value=Decimal("-1000000"),
    max_value=Decimal("1000000"),
    allow_nan=False,
    allow_infinity=False,
    places=6,
)


@given(positions=_POSITIONS, cash=_CASH)
@settings(max_examples=200, deadline=None)
def test_a_snapshot_never_disagrees_with_itself(positions: dict[int, int], cash: Decimal) -> None:
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions=positions, cash_usd=cash),
        reported=reported_snapshot(positions=positions, cash_usd=cash),
        stamp=make_stamp(),
    )
    assert result.findings == ()
    assert result.matched


@given(positions=_POSITIONS, cash=_CASH)
@settings(max_examples=200, deadline=None)
def test_matched_is_exactly_the_absence_of_breaks(positions: dict[int, int], cash: Decimal) -> None:
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions=positions, cash_usd=cash),
        reported=reported_snapshot(positions={}, cash_usd=Decimal("0")),
        stamp=make_stamp(),
    )
    assert result.matched == (not result.breaks)
    assert all(
        finding.is_break == (finding.severity is Severity.BREAK) for finding in result.findings
    )


@given(
    security_id=st.integers(min_value=1, max_value=40),
    ours=st.integers(min_value=-500, max_value=500),
    delta=st.integers(min_value=1, max_value=500),
)
@settings(max_examples=200, deadline=None)
def test_any_share_difference_at_all_is_a_break(security_id: int, ours: int, delta: int) -> None:
    # Positions carry no tolerance, and this is the property that says so: there
    # is no pair of quantities whose difference is absorbed.
    result = reconcile(
        cycle_id=CYCLE_ID,
        internal=internal_snapshot(positions={security_id: ours}),
        reported=reported_snapshot(positions={security_id: ours + delta}),
        stamp=make_stamp(),
    )
    assert not result.matched
    assert len(result.breaks) == 1


@given(positions=_POSITIONS, cash=_CASH)
@settings(max_examples=100, deadline=None)
def test_a_snapshot_round_trips_through_its_stored_payload(
    positions: dict[int, int], cash: Decimal
) -> None:
    snapshot = internal_snapshot(positions=positions, cash_usd=cash)
    rebuilt = snapshot_of_json(snapshot.as_json())
    assert rebuilt.digest == snapshot.digest
    assert dict(rebuilt.positions) == dict(snapshot.positions)
    assert rebuilt.cash_usd == snapshot.cash_usd
