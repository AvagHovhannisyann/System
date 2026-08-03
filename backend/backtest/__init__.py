"""Validation framework — the statistics that decide whether a result is real.

Directive §5, Phase 10: *"Build this with maximum care. Every other phase
depends on it being correct."* A bug in this package does not fail loudly. It
produces a confident, well-formatted number that makes a worthless strategy look
tradeable, and every downstream decision inherits it.

The three pieces built here answer three different questions, and none of them
substitutes for another:

- :mod:`backend.backtest.cpcv` (P10.2) — **how variable is this result?**
  Combinatorial Purged Cross-Validation replaces the single walk-forward path
  with ``C(N-1, k-1)`` complete out-of-sample paths, so the output is a
  distribution rather than one lucky draw.
- :mod:`backend.backtest.dsr` (P10.3) — **is this number too good, given how
  hard I searched?** The Deflated Sharpe Ratio corrects an observed Sharpe for
  the number of trials, the spread of those trials, the skewness and kurtosis of
  the returns, and the sample length. The trial count is a required argument
  with no default anywhere in the module.
- :mod:`backend.backtest.pbo` (P10.4) — **does picking the in-sample winner
  generalize at all?** The Probability of Backtest Overfitting is the fraction
  of combinatorially symmetric splits in which the in-sample best landed at or
  below the out-of-sample median. Pure noise gives 0.5.

:mod:`backend.backtest.metrics` holds the shared statistics and pins the
estimator conventions (population skewness, **non-excess** kurtosis, per-period
Sharpe ratios) so the three cannot silently disagree.
:mod:`backend.backtest.ledger` is a **read-only** reader for
``TESTING_LEDGER.md``, which supplies the Deflated Sharpe Ratio's trial count.

:mod:`backend.backtest.synthetic` (P10.6) is the **synthetic-truth harness** —
the regression instrument for everything above. It generates seeded synthetic
markets with an injected signal of known strength (and a pure-noise mode), drives
the real engine over them, and reports what these statistics concluded. Re-run it
after any change to this package: gate G10 depends on it, and a validation stack
that has quietly stopped detecting its own failures looks exactly like one that
works.

The backtest engine itself is P10.1 and does not live here yet.

Standing contract for every function in this package: the returns you pass in
are **net of modelled costs** (invariant I4) and are simple per-period returns
expressed as fractions. Nothing here can verify either, which is why both are
restated at every entry point.
"""

from __future__ import annotations

from backend.backtest.cpcv import (
    CombinatorialPurgedCV,
    CPCVSplit,
    CPCVSplits,
    EmptyTrainingSetError,
    PathDistribution,
    path_sharpe_ratios,
)
from backend.backtest.dsr import (
    EULER_MASCHERONI,
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    deflated_sharpe_ratio_from_trials,
    expected_maximum_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from backend.backtest.ledger import (
    LedgerFormatError,
    LedgerRow,
    TrialLedger,
    read_testing_ledger,
)
from backend.backtest.metrics import (
    as_float_array,
    average_ranks,
    kurtosis,
    sharpe_ratio,
    skewness,
    standard_normal_cdf,
    standard_normal_ppf,
)
from backend.backtest.pbo import (
    PBOResult,
    PerformanceFunction,
    probability_of_backtest_overfitting,
)
from backend.backtest.synthetic import (
    SYNTHETIC_DATA_VERSION_PREFIX,
    SYNTHETIC_DISCLOSURE,
    ZERO_COST_PARAMS,
    ConfigurationSweep,
    CostMode,
    SharpeSweep,
    StrategyCandidate,
    SyntheticDataError,
    SyntheticMarket,
    SyntheticRun,
    SyntheticSpec,
    build_synthetic_market,
    disjoint_asset_blocks,
    format_report,
    require_synthetic_source,
    run_synthetic_backtest,
    sweep_configurations,
    sweep_seeds,
)

__all__ = [
    "EULER_MASCHERONI",
    "SYNTHETIC_DATA_VERSION_PREFIX",
    "SYNTHETIC_DISCLOSURE",
    "ZERO_COST_PARAMS",
    "CPCVSplit",
    "CPCVSplits",
    "CombinatorialPurgedCV",
    "ConfigurationSweep",
    "CostMode",
    "DeflatedSharpeResult",
    "EmptyTrainingSetError",
    "LedgerFormatError",
    "LedgerRow",
    "PBOResult",
    "PathDistribution",
    "PerformanceFunction",
    "SharpeSweep",
    "StrategyCandidate",
    "SyntheticDataError",
    "SyntheticMarket",
    "SyntheticRun",
    "SyntheticSpec",
    "TrialLedger",
    "as_float_array",
    "average_ranks",
    "build_synthetic_market",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "deflated_sharpe_ratio_from_trials",
    "disjoint_asset_blocks",
    "expected_maximum_sharpe_ratio",
    "format_report",
    "kurtosis",
    "path_sharpe_ratios",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "read_testing_ledger",
    "require_synthetic_source",
    "run_synthetic_backtest",
    "sharpe_ratio",
    "skewness",
    "standard_normal_cdf",
    "standard_normal_ppf",
    "sweep_configurations",
    "sweep_seeds",
]
