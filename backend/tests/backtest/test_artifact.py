"""Tests for the run artifact: I2 provenance, I4 cost flags, and intervals.

The load-bearing tests here are the negative ones. An artifact that *can* be
built without a git commit, without a data version, or without cost information
is an artifact that eventually will be, and the resulting number will look
exactly like an honest one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from backend.backtest.artifact import (
    NET_OF_COST_DISCLOSURE,
    ArtifactError,
    BenchmarkComparison,
    CostProvenance,
    EquityCurve,
    Estimate,
    IntervalConfig,
    JsonValue,
    PerformanceSummary,
    RealizedAccounting,
    ReproducibilityError,
    ReproducibilityStamp,
    RunArtifact,
    TrackRecord,
    canonical_json,
    compare_to_benchmark,
    config_hash,
    current_git_commit,
    max_drawdown_of_equity,
    summarize_track_record,
)
from backend.costs.model import UNCALIBRATED_DEFAULTS, CostModelParams

if TYPE_CHECKING:
    from collections.abc import Sequence

COMMIT = "0" * 39 + "a"
DIGEST = "a" * 64


def _dates(count: int) -> tuple[dt.datetime, ...]:
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    return tuple(base + dt.timedelta(days=index) for index in range(count))


def _curve(values: Sequence[float]) -> EquityCurve:
    return EquityCurve(dates=_dates(len(values)), equity_usd=np.asarray(values, dtype=np.float64))


def _accounting(final: float = 1_050_000.0) -> RealizedAccounting:
    return RealizedAccounting(
        initial_capital_usd=1_000_000.0,
        final_equity_usd=final,
        half_spread_usd=500.0,
        commission_usd=100.0,
        impact_usd=632.0,
        borrow_usd=0.0,
        total_cost_usd=1_232.0,
        traded_notional_usd=1_000_000.0,
        n_orders=1,
        n_rebalances=5,
    )


def _stamp(seed: int = 11) -> ReproducibilityStamp:
    return ReproducibilityStamp(
        git_commit=COMMIT, data_version="fixture-v1", config_hash=DIGEST, seed=seed
    )


def _estimate(value: float = 1.0) -> Estimate:
    return Estimate(
        value=value,
        lower=value - 1.0,
        upper=value + 1.0,
        level=0.95,
        method="fixture",
        units="dimensionless",
    )


def _summary() -> PerformanceSummary:
    return PerformanceSummary(
        metrics={
            "total_return": _estimate(),
            "annualized_return": _estimate(),
            "annualized_volatility": _estimate(),
            "sharpe_ratio": _estimate(),
            "max_drawdown": _estimate(0.1),
        },
        periods_per_year=252.0,
        n_periods=5,
        accounting=_accounting(),
    )


def _track(values: Sequence[float]) -> TrackRecord:
    return TrackRecord(equity=_curve(values), summary=_summary())


def _comparison() -> BenchmarkComparison:
    return BenchmarkComparison(
        active_return=_estimate(0.001),
        information_ratio=_estimate(0.4),
        benchmark_description="buy-and-hold BMK 100%",
    )


def _artifact(**overrides: object) -> RunArtifact:
    kwargs: dict[str, object] = {
        "stamp": _stamp(),
        "costs": CostProvenance.from_params(UNCALIBRATED_DEFAULTS),
        "strategy": _track([1_000_000.0, 1_010_000.0, 1_005_000.0, 1_050_000.0]),
        "benchmark": _track([1_000_000.0, 1_002_000.0, 1_001_000.0, 1_004_000.0]),
        "comparison": _comparison(),
    }
    kwargs.update(overrides)
    return RunArtifact(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# I2: configuration hashing
# ---------------------------------------------------------------------------


def test_canonical_json_is_insertion_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_config_hash_is_the_sha256_of_the_canonical_encoding() -> None:
    config: dict[str, JsonValue] = {"alpha": 1, "beta": [1, 2, {"c": True}]}
    expected = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    assert config_hash(config) == expected


def test_config_hash_distinguishes_configurations() -> None:
    assert config_hash({"seed": 1}) != config_hash({"seed": 2})


def test_config_hash_refuses_a_nan_rather_than_hashing_an_unreproducible_run() -> None:
    with pytest.raises(ReproducibilityError, match="not canonically JSON-encodable"):
        config_hash({"threshold": math.nan})


def test_config_hash_refuses_a_value_json_cannot_represent() -> None:
    with pytest.raises(ReproducibilityError, match="not canonically JSON-encodable"):
        config_hash({"when": dt.datetime(2024, 1, 1, tzinfo=dt.UTC)})  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# I2: git commit resolution
# ---------------------------------------------------------------------------


def _make_git_dir(root: Path, *, head: str, refs: dict[str, str] | None = None) -> Path:
    git_dir = root / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text(head, encoding="utf-8")
    for ref, value in (refs or {}).items():
        path = git_dir / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    return git_dir


def test_current_git_commit_resolves_a_loose_ref(tmp_path: Path) -> None:
    _make_git_dir(tmp_path, head="ref: refs/heads/main\n", refs={"refs/heads/main": f"{COMMIT}\n"})
    work = tmp_path / "backend" / "backtest"
    work.mkdir(parents=True)
    assert current_git_commit(work) == COMMIT


def test_current_git_commit_resolves_a_packed_ref(tmp_path: Path) -> None:
    git_dir = _make_git_dir(tmp_path, head="ref: refs/heads/main\n")
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{COMMIT} refs/heads/main\n",
        encoding="utf-8",
    )
    assert current_git_commit(tmp_path) == COMMIT


def test_current_git_commit_reads_a_detached_head(tmp_path: Path) -> None:
    _make_git_dir(tmp_path, head=f"{COMMIT}\n")
    assert current_git_commit(tmp_path) == COMMIT


def test_current_git_commit_follows_a_worktree_gitdir_pointer(tmp_path: Path) -> None:
    real = tmp_path / "real.git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (real / "refs" / "heads" / "main").write_text(f"{COMMIT}\n", encoding="utf-8")
    tree = tmp_path / "worktree"
    tree.mkdir()
    (tree / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
    assert current_git_commit(tree) == COMMIT


def test_current_git_commit_raises_when_there_is_no_checkout(tmp_path: Path) -> None:
    with pytest.raises(ReproducibilityError, match=r"no \.git directory"):
        current_git_commit(tmp_path)


def test_current_git_commit_raises_rather_than_returning_a_placeholder(tmp_path: Path) -> None:
    _make_git_dir(tmp_path, head="ref: refs/heads/main\n", refs={"refs/heads/main": "not-a-sha\n"})
    with pytest.raises(ReproducibilityError, match="lowercase hex object"):
        current_git_commit(tmp_path)


def test_current_git_commit_resolves_in_this_repository() -> None:
    commit = current_git_commit()
    assert len(commit) in {40, 64}
    assert set(commit) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# I2: the stamp itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field_name",
    ["git_commit", "data_version", "config_hash", "seed"],
)
def test_a_stamp_cannot_be_built_without_any_of_its_four_fields(field_name: str) -> None:
    kwargs: dict[str, object] = {
        "git_commit": COMMIT,
        "data_version": "v1",
        "config_hash": DIGEST,
        "seed": 0,
    }
    del kwargs[field_name]
    with pytest.raises(TypeError, match=field_name):
        ReproducibilityStamp(**kwargs)  # type: ignore[arg-type]


def test_a_stamp_rejects_a_blank_data_version() -> None:
    with pytest.raises(ReproducibilityError, match="data_version must name"):
        ReproducibilityStamp(git_commit=COMMIT, data_version="   ", config_hash=DIGEST, seed=0)


def test_a_stamp_rejects_a_malformed_commit() -> None:
    with pytest.raises(ReproducibilityError, match="lowercase hex object"):
        ReproducibilityStamp(git_commit="HEAD", data_version="v1", config_hash=DIGEST, seed=0)


def test_a_stamp_rejects_a_malformed_config_hash() -> None:
    with pytest.raises(ReproducibilityError, match="SHA-256 hex digest"):
        ReproducibilityStamp(git_commit=COMMIT, data_version="v1", config_hash="abc", seed=0)


def test_a_stamp_rejects_a_negative_seed() -> None:
    with pytest.raises(ReproducibilityError, match=r"\[0, 2\*\*64\)"):
        ReproducibilityStamp(git_commit=COMMIT, data_version="v1", config_hash=DIGEST, seed=-1)


def test_a_stamp_rejects_a_bool_masquerading_as_a_seed() -> None:
    with pytest.raises(ReproducibilityError, match="seed must be an int"):
        ReproducibilityStamp(
            git_commit=COMMIT,
            data_version="v1",
            config_hash=DIGEST,
            seed=True,
        )


# ---------------------------------------------------------------------------
# I4: cost provenance reaches the artifact
# ---------------------------------------------------------------------------


def test_cost_provenance_carries_the_flag_and_basis_from_the_cost_model() -> None:
    provenance = CostProvenance.from_params(UNCALIBRATED_DEFAULTS)
    assert provenance.uncalibrated is True
    assert provenance.calibration_basis == UNCALIBRATED_DEFAULTS.calibration_basis
    assert provenance.parameters["half_spread_bps"] == UNCALIBRATED_DEFAULTS.half_spread_bps


def test_cost_provenance_rejects_a_blank_basis() -> None:
    with pytest.raises(ArtifactError, match="calibration_basis must be non-empty"):
        CostProvenance(uncalibrated=True, calibration_basis="  ", parameters={})


def test_cost_provenance_records_a_zeroed_cost_model_rather_than_hiding_it() -> None:
    # A caller can always zero the cost parameters. What they cannot do is make
    # that invisible: the artifact carries the numbers that were charged.
    free = UNCALIBRATED_DEFAULTS.with_parameters(
        half_spread_bps=0.0,
        commission_bps=0.0,
        impact_coefficient=0.0,
        default_daily_volatility_bps=0.0,
        borrow_rate_bps_per_year=0.0,
    )
    provenance = CostProvenance.from_params(free)
    assert provenance.parameters == {
        "half_spread_bps": 0.0,
        "commission_bps": 0.0,
        "impact_coefficient": 0.0,
        "default_daily_volatility_bps": 0.0,
        "borrow_rate_bps_per_year": 0.0,
    }


def test_an_artifact_cannot_be_constructed_without_cost_information() -> None:
    kwargs: dict[str, object] = {
        "stamp": _stamp(),
        "strategy": _track([1_000_000.0, 1_010_000.0, 1_005_000.0, 1_050_000.0]),
        "benchmark": _track([1_000_000.0, 1_002_000.0, 1_001_000.0, 1_004_000.0]),
        "comparison": _comparison(),
    }
    with pytest.raises(TypeError, match="costs"):
        RunArtifact(**kwargs)  # type: ignore[arg-type]


def test_an_artifact_cannot_be_constructed_without_a_reproducibility_stamp() -> None:
    kwargs: dict[str, object] = {
        "costs": CostProvenance.from_params(UNCALIBRATED_DEFAULTS),
        "strategy": _track([1_000_000.0, 1_010_000.0, 1_005_000.0, 1_050_000.0]),
        "benchmark": _track([1_000_000.0, 1_002_000.0, 1_001_000.0, 1_004_000.0]),
        "comparison": _comparison(),
    }
    with pytest.raises(TypeError, match="stamp"):
        RunArtifact(**kwargs)  # type: ignore[arg-type]


def test_an_artifact_cannot_be_constructed_without_a_benchmark() -> None:
    kwargs: dict[str, object] = {
        "stamp": _stamp(),
        "costs": CostProvenance.from_params(UNCALIBRATED_DEFAULTS),
        "strategy": _track([1_000_000.0, 1_010_000.0, 1_005_000.0, 1_050_000.0]),
        "comparison": _comparison(),
    }
    with pytest.raises(TypeError, match="benchmark"):
        RunArtifact(**kwargs)  # type: ignore[arg-type]


def test_an_artifact_rejects_a_benchmark_measured_over_different_dates() -> None:
    other_dates = tuple(moment + dt.timedelta(days=100) for moment in _dates(4))
    benchmark = TrackRecord(
        equity=EquityCurve(
            dates=other_dates,
            equity_usd=np.asarray([1e6, 1.002e6, 1.001e6, 1.004e6], dtype=np.float64),
        ),
        summary=_summary(),
    )
    with pytest.raises(ArtifactError, match="benchmark dates differ"):
        _artifact(benchmark=benchmark)


def test_the_disclosures_always_state_that_figures_are_net_of_costs() -> None:
    assert NET_OF_COST_DISCLOSURE in _artifact().disclosures


def test_the_uncalibrated_flag_reaches_the_disclosures_and_the_payload() -> None:
    artifact = _artifact()
    joined = " ".join(artifact.disclosures)
    assert "COSTS ARE UNCALIBRATED" in joined
    assert UNCALIBRATED_DEFAULTS.calibration_basis in joined
    payload = artifact.to_dict()
    assert payload["uncalibrated"] is True
    assert payload["calibration_basis"] == UNCALIBRATED_DEFAULTS.calibration_basis


def test_a_calibrated_model_drops_the_uncalibrated_warning_but_keeps_the_net_statement() -> None:
    calibrated = CostModelParams(
        half_spread_bps=7.0,
        commission_bps=1.5,
        impact_coefficient=1.2,
        default_daily_volatility_bps=250.0,
        borrow_rate_bps_per_year=150.0,
        uncalibrated=False,
        calibration_basis="fitted to P11 paper fills with the D-013 haircut applied",
    )
    artifact = _artifact(costs=CostProvenance.from_params(calibrated))
    assert artifact.disclosures == (NET_OF_COST_DISCLOSURE,)
    assert artifact.to_dict()["uncalibrated"] is False


# ---------------------------------------------------------------------------
# §5-P10: no point estimate alone
# ---------------------------------------------------------------------------


def test_an_estimate_cannot_be_coerced_to_a_bare_float() -> None:
    with pytest.raises(TypeError, match="float"):
        float(_estimate())  # type: ignore[arg-type]


def test_an_estimate_cannot_be_formatted_as_a_bare_number() -> None:
    # The failure mode this closes: `f"Sharpe {summary.sharpe:.2f}"` in a
    # template, which would render a point estimate alone and look correct.
    with pytest.raises(TypeError, match="format string"):
        format(_estimate(), ".2f")


def test_an_estimate_does_not_support_arithmetic() -> None:
    with pytest.raises(TypeError):
        _ = _estimate() + 1.0  # type: ignore[operator]


def test_an_estimates_string_form_always_shows_the_interval() -> None:
    rendered = f"{_estimate(1.25)}"
    assert "0.25" in rendered
    assert "2.25" in rendered
    assert "95%" in rendered


def test_an_estimate_renders_with_its_interval() -> None:
    rendered = str(Estimate(value=1.5, lower=0.5, upper=2.5, level=0.9, method="m", units="u"))
    assert "1.5" in rendered
    assert "0.5" in rendered
    assert "2.5" in rendered


def test_an_estimate_rejects_an_inverted_interval() -> None:
    with pytest.raises(ValueError, match="inverted"):
        Estimate(value=1.0, lower=2.0, upper=0.0, level=0.95, method="m", units="u")


def test_an_estimate_rejects_an_unlabelled_method() -> None:
    with pytest.raises(ValueError, match="method must state"):
        Estimate(value=1.0, lower=0.0, upper=2.0, level=0.95, method="  ", units="u")


def test_an_estimate_rejects_an_unlabelled_unit() -> None:
    with pytest.raises(ValueError, match="units must state"):
        Estimate(value=1.0, lower=0.0, upper=2.0, level=0.95, method="m", units="")


def test_an_estimate_rejects_a_level_outside_the_open_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"level must lie strictly in \(0, 1\)"):
        Estimate(value=1.0, lower=0.0, upper=2.0, level=1.0, method="m", units="u")


@pytest.mark.parametrize(
    "missing",
    ["total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"],
)
def test_a_summary_missing_any_required_metric_is_rejected(missing: str) -> None:
    metrics = {name: _estimate() for name in _summary().metrics}
    del metrics[missing]
    with pytest.raises(ArtifactError, match=missing):
        PerformanceSummary(
            metrics=metrics, periods_per_year=252.0, n_periods=5, accounting=_accounting()
        )


# ---------------------------------------------------------------------------
# Equity curves and drawdown
# ---------------------------------------------------------------------------


def test_net_returns_are_derived_from_equity_so_they_cannot_disagree() -> None:
    curve = _curve([100.0, 110.0, 99.0])
    assert curve.net_returns == pytest.approx([0.1, -0.1], rel=1e-12)
    assert curve.n_periods == 2


def test_an_equity_curve_rejects_a_non_positive_value() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _curve([100.0, 0.0, 50.0])


def test_an_equity_curve_rejects_naive_dates() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        EquityCurve(
            dates=(dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)),  # noqa: DTZ001
            equity_usd=np.asarray([1.0, 2.0, 3.0]),
        )


def test_an_equity_curve_rejects_dates_that_do_not_increase() -> None:
    base = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    with pytest.raises(ValueError, match="strictly increase"):
        EquityCurve(
            dates=(base, base, base + dt.timedelta(days=1)),
            equity_usd=np.asarray([1.0, 2.0, 3.0]),
        )


def test_max_drawdown_matches_hand_computation() -> None:
    # peak 120 at index 1, trough 90 at index 3 -> (120 - 90) / 120 = 0.25
    assert max_drawdown_of_equity([100.0, 120.0, 110.0, 90.0, 130.0]) == pytest.approx(0.25)


def test_max_drawdown_of_a_monotonically_rising_path_is_zero() -> None:
    assert max_drawdown_of_equity([100.0, 101.0, 102.0]) == 0.0


# ---------------------------------------------------------------------------
# Summaries, seeds and digests
# ---------------------------------------------------------------------------


def _varied_curve() -> EquityCurve:
    rng = np.random.default_rng(20240101)
    returns = rng.normal(0.0004, 0.01, 200)
    equity = 1_000_000.0 * np.cumprod(1.0 + returns)
    return EquityCurve(
        dates=_dates(201),
        equity_usd=np.concatenate([[1_000_000.0], equity]),
    )


def test_the_same_seed_produces_identical_intervals() -> None:
    curve = _varied_curve()
    first = summarize_track_record(
        equity=curve,
        accounting=_accounting(),
        periods_per_year=252.0,
        intervals=IntervalConfig(seed=5),
    )
    second = summarize_track_record(
        equity=curve,
        accounting=_accounting(),
        periods_per_year=252.0,
        intervals=IntervalConfig(seed=5),
    )
    assert first.to_dict() == second.to_dict()


def test_a_different_seed_moves_the_intervals_but_not_the_point_estimates() -> None:
    curve = _varied_curve()
    first = summarize_track_record(
        equity=curve,
        accounting=_accounting(),
        periods_per_year=252.0,
        intervals=IntervalConfig(seed=5),
    )
    second = summarize_track_record(
        equity=curve,
        accounting=_accounting(),
        periods_per_year=252.0,
        intervals=IntervalConfig(seed=6),
    )
    for name, estimate in first.metrics.items():
        other = second.metrics[name]
        assert estimate.value == other.value, name
        assert (estimate.lower, estimate.upper) != (other.lower, other.upper), name


def test_every_summarized_metric_carries_a_finite_interval() -> None:
    summary = summarize_track_record(
        equity=_varied_curve(),
        accounting=_accounting(),
        periods_per_year=252.0,
        intervals=IntervalConfig(seed=1),
    )
    assert set(summary.metrics) == {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    }
    for estimate in summary.metrics.values():
        assert estimate.lower <= estimate.upper
        assert math.isfinite(estimate.value)


def test_the_results_digest_ignores_the_creation_timestamp() -> None:
    early = _artifact(created_at=dt.datetime(2024, 1, 1, tzinfo=dt.UTC))
    late = _artifact(created_at=dt.datetime(2025, 6, 1, tzinfo=dt.UTC))
    assert early.results_digest() == late.results_digest()


def test_the_results_digest_changes_with_the_stamp() -> None:
    assert _artifact().results_digest() != _artifact(stamp=_stamp(seed=12)).results_digest()


def test_comparing_paths_on_different_dates_is_refused() -> None:
    strategy = _curve([1.0, 1.1, 1.2, 1.15])
    other_dates = tuple(moment + dt.timedelta(days=50) for moment in strategy.dates)
    benchmark = EquityCurve(
        dates=other_dates, equity_usd=np.asarray([1.0, 1.05, 1.02, 1.09], dtype=np.float64)
    )
    with pytest.raises(ValueError, match="identical dates"):
        compare_to_benchmark(
            strategy=strategy,
            benchmark=benchmark,
            benchmark_description="fixture",
            periods_per_year=252.0,
            intervals=IntervalConfig(seed=1),
        )


def test_a_comparison_requires_the_benchmark_to_describe_itself() -> None:
    with pytest.raises(ArtifactError, match="benchmark_description"):
        BenchmarkComparison(
            active_return=_estimate(),
            information_ratio=_estimate(),
            benchmark_description="   ",
        )


# ---------------------------------------------------------------------------
# Interval configuration and the remaining guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, r"\[0, 2\*\*64\)"),
        ({"seed": 1, "level": 1.0}, r"level must lie strictly in \(0, 1\)"),
        ({"seed": 1, "n_resamples": 1}, "at least 2"),
        ({"seed": 1, "block_length": 0}, "positive number of observations"),
    ],
)
def test_an_interval_configuration_validates_its_parameters(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        IntervalConfig(**kwargs)  # type: ignore[arg-type]


def test_the_default_block_length_follows_the_cube_root_rule() -> None:
    config = IntervalConfig(seed=1)
    assert config.resolved_block_length(1000) == 10
    assert config.resolved_block_length(27) == 3
    assert config.resolved_block_length(1) == 1


def test_an_explicit_block_length_is_capped_at_the_sample_length() -> None:
    assert IntervalConfig(seed=1, block_length=50).resolved_block_length(10) == 10


def test_the_method_label_names_what_the_interval_does_not_account_for() -> None:
    label = IntervalConfig(seed=7, n_resamples=500).method_label(100)
    assert "circular block bootstrap" in label
    assert "seed=7" in label
    assert "trial count" in label


def test_an_artifact_rejects_a_naive_creation_timestamp() -> None:
    with pytest.raises(ArtifactError, match="timezone-aware UTC"):
        _artifact(created_at=dt.datetime(2024, 1, 1))  # noqa: DTZ001


def test_canonical_json_refuses_an_infinity() -> None:
    with pytest.raises(ReproducibilityError, match="not canonically JSON-encodable"):
        canonical_json({"limit": math.inf})


def test_an_equity_curve_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        EquityCurve(dates=_dates(3), equity_usd=np.asarray([1.0, 2.0], dtype=np.float64))


def test_an_equity_curve_rejects_a_sample_too_short_for_a_dispersion_statistic() -> None:
    with pytest.raises(ValueError, match="at least 3 points"):
        _curve([100.0, 101.0])


def test_realized_accounting_reports_execution_cost_in_basis_points_of_notional() -> None:
    # (500 + 100 + 632) / 1_000_000 * 10_000 = 12.32 bps; borrow is excluded
    # because it accrues on held exposure, not on traded notional.
    assert _accounting().cost_bps_of_traded_notional == pytest.approx(12.32)


def test_a_summary_refuses_a_non_positive_annualization_factor() -> None:
    with pytest.raises(ValueError, match="periods_per_year must be positive"):
        summarize_track_record(
            equity=_varied_curve(),
            accounting=_accounting(),
            periods_per_year=0.0,
            intervals=IntervalConfig(seed=1),
        )


def test_a_flat_return_series_refuses_a_sharpe_rather_than_reporting_infinity() -> None:
    flat = EquityCurve(
        dates=_dates(5),
        equity_usd=np.asarray([100.0] * 5, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="zero standard deviation"):
        summarize_track_record(
            equity=flat,
            accounting=_accounting(),
            periods_per_year=252.0,
            intervals=IntervalConfig(seed=1),
        )


# ---------------------------------------------------------------------------
# The remaining I2 failure paths, and the summary's named accessors
# ---------------------------------------------------------------------------


def test_current_git_commit_refuses_a_gitdir_file_without_a_pointer(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("not a gitdir pointer\n", encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="gitdir:"):
        current_git_commit(tmp_path)


def test_current_git_commit_refuses_an_unresolvable_ref(tmp_path: Path) -> None:
    _make_git_dir(tmp_path, head="ref: refs/heads/missing\n")
    with pytest.raises(ReproducibilityError, match="could not be resolved"):
        current_git_commit(tmp_path)


def test_current_git_commit_refuses_a_git_directory_without_a_head(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    with pytest.raises(ReproducibilityError, match="does not exist"):
        current_git_commit(tmp_path)


def test_current_git_commit_resolves_a_relative_worktree_pointer(tmp_path: Path) -> None:
    real = tmp_path / "real.git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (real / "refs" / "heads" / "main").write_text(f"{COMMIT}\n", encoding="utf-8")
    tree = tmp_path / "worktree"
    tree.mkdir()
    (tree / ".git").write_text("gitdir: ../real.git\n", encoding="utf-8")
    assert current_git_commit(tree) == COMMIT


def test_max_drawdown_refuses_a_path_that_touches_zero() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        max_drawdown_of_equity([100.0, 0.0, 50.0])


def test_the_summary_exposes_every_required_metric_by_name() -> None:
    summary = summarize_track_record(
        equity=_varied_curve(),
        accounting=_accounting(),
        periods_per_year=252.0,
        intervals=IntervalConfig(seed=3),
    )
    named = (
        summary.total_return,
        summary.annualized_return,
        summary.annualized_volatility,
        summary.sharpe,
        summary.max_drawdown,
    )
    assert all(isinstance(item, Estimate) for item in named)
    assert [item.value for item in named] == [
        summary.metrics[name].value
        for name in (
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        )
    ]


def test_accounting_ratios_are_zero_rather_than_undefined_when_nothing_traded() -> None:
    idle = RealizedAccounting(
        initial_capital_usd=1_000_000.0,
        final_equity_usd=1_000_000.0,
        half_spread_usd=0.0,
        commission_usd=0.0,
        impact_usd=0.0,
        borrow_usd=0.0,
        total_cost_usd=0.0,
        traded_notional_usd=0.0,
        n_orders=0,
        n_rebalances=0,
    )
    assert idle.cost_bps_of_traded_notional == 0.0
    assert idle.turnover_per_rebalance == 0.0


def test_an_estimate_rejects_a_non_finite_bound() -> None:
    # A NaN interval renders as a blank cell rather than as an error, which is
    # the failure mode this whole module exists to prevent.
    with pytest.raises(ValueError, match="must be finite"):
        Estimate(value=1.0, lower=math.nan, upper=2.0, level=0.95, method="m", units="u")


def test_an_interval_configuration_rejects_a_bool_masquerading_as_a_seed() -> None:
    with pytest.raises(ValueError, match="seed must be an int"):
        IntervalConfig(seed=True)
