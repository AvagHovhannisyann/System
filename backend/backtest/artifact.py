"""Run artifacts: reproducibility (I2), cost provenance (I4), intervals (§5-P10).

This module holds the *result types* of the backtest engine and nothing that
simulates anything. It exists so that three directive requirements are carried
by the type system rather than by the discipline of whoever writes the next
consumer:

**I2 — reproducibility.** :class:`ReproducibilityStamp` carries the four fields
the directive names — git commit, data version, config hash, seed — with no
defaults and no ``"unknown"`` fallback. A :class:`RunArtifact` cannot be
constructed without one, so a result that cannot say how to regenerate itself
cannot exist. :func:`config_hash` produces the hash from canonical JSON, so the
same configuration hashes identically across processes and machines.

**I4 — cost realism.** :class:`CostProvenance` carries
:attr:`~backend.costs.model.CostModelParams.uncalibrated` and
``calibration_basis`` from the P9.3 cost model onto the artifact and into
:meth:`RunArtifact.to_dict`, which is what the dashboard renders. A
:class:`RunArtifact` cannot be constructed without cost information, and its
:attr:`~RunArtifact.disclosures` always states that every figure is net of
modelled costs — plus, while the model is uncalibrated, that the costs are
assumptions rather than measurements (directive §5-P9 Gate G9: "until then,
conservative defaults, clearly flagged").

**§5-P10 — no point estimate is ever displayed alone.** Every statistic on a
:class:`PerformanceSummary` is an :class:`Estimate`, which is a value *and* an
interval *and* the method that produced it. There is no representation of a
bare metric to display by accident: :class:`Estimate` has no ``__float__``, and
:class:`PerformanceSummary` refuses construction when a required metric is
missing.

**Benchmark coupling.** :class:`RunArtifact` requires a benchmark track record
and rejects one whose dates differ from the strategy's. Reporting a strategy
without its benchmark is not a matter of remembering to; it is unconstructible.

Units, stated once and repeated at every entry point:

* returns are simple per-period returns as **fractions** (``0.01`` is 1%), and
  are **net of modelled costs** — every return series in this module is
  produced by the engine after costs, never before;
* equity is in **US dollars**;
* ``periods_per_year`` is the annualization factor (252 for daily bars);
* drawdown is a **positive fraction** of the running peak (``0.12`` is a 12%
  drawdown).

Intervals are **circular block bootstrap percentile intervals** over the
realized net return series, seeded from the run's :class:`ReproducibilityStamp`.
They describe sampling uncertainty of one backtest path under the assumption
that the return series is stationary and that dependence dies out within a
block. They do **not** correct for the number of strategies tried — that is
what :mod:`backend.backtest.dsr` is for — and they do not turn one lucky path
into a distribution of paths, which is what :mod:`backend.backtest.cpcv` is for.
An interval here is the weakest of the three statements and must never be
presented as if it were one of the other two.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from backend.backtest.metrics import (
    FloatArray,
    IntArray,
    as_float_array,
    sharpe_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from backend.costs.model import CostModelParams

__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_CONFIDENCE_LEVEL",
    "NET_OF_COST_DISCLOSURE",
    "ArtifactError",
    "BenchmarkComparison",
    "CostProvenance",
    "EquityCurve",
    "Estimate",
    "IntervalConfig",
    "JsonValue",
    "PerformanceSummary",
    "ReproducibilityError",
    "ReproducibilityStamp",
    "RunArtifact",
    "TrackRecord",
    "canonical_json",
    "compare_to_benchmark",
    "config_hash",
    "current_git_commit",
    "summarize_track_record",
]

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None

DEFAULT_CONFIDENCE_LEVEL: Final = 0.95
"""Central mass of every reported interval unless a caller overrides it."""

DEFAULT_BOOTSTRAP_RESAMPLES: Final = 1_000
"""Bootstrap replicates per metric. 1000 resolves a 95% percentile interval to
about the 25th and 975th order statistics — enough for a displayed interval,
not enough for a tail probability. Raise it if a tail is what you need."""

NET_OF_COST_DISCLOSURE: Final = (
    "Every figure is net of modelled transaction costs (half-spread, commission, "
    "square-root impact) and of borrow accrued on short exposure. No gross figure "
    "is reported anywhere in this artifact (invariant I4)."
)
"""Disclosure attached to every artifact. Present whether or not costs are calibrated."""

_REQUIRED_METRICS: Final = (
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
)
"""Metrics a :class:`PerformanceSummary` must carry. Missing one is a construction error."""

_HEX_DIGITS: Final = frozenset("0123456789abcdef")
_GIT_SHA_LENGTHS: Final = frozenset({40, 64})
_MAX_SEED: Final = 2**64


class ArtifactError(Exception):
    """Base class for run-artifact construction failures."""


class ReproducibilityError(ArtifactError, ValueError):
    """Raised when a run cannot state how to regenerate itself (invariant I2).

    Covers a missing or malformed git commit, a blank data version, a
    malformed config hash, and a seed outside the representable range. It is
    fatal rather than a warning because the alternative — stamping ``unknown``
    and carrying on — produces an artifact that *looks* reproducible in every
    UI that renders it, which is worse than having no artifact at all.
    """


# ---------------------------------------------------------------------------
# Configuration hashing (I2: "config hash")
# ---------------------------------------------------------------------------


def canonical_json(config: Mapping[str, JsonValue]) -> str:
    """Render a configuration mapping as canonical JSON.

    Canonical means: keys sorted, no insignificant whitespace, ASCII-escaped,
    and NaN/Infinity rejected. Two processes that hold the same configuration
    therefore produce the same string regardless of insertion order, locale, or
    ``PYTHONHASHSEED``, which is what makes :func:`config_hash` a stable
    identity rather than a per-process one.

    Args:
        config: the configuration. Values must be JSON types
            (:data:`JsonValue`). Units and meaning are the caller's business;
            this function only fixes the encoding.

    Returns:
        The canonical JSON encoding of ``config``.

    Raises:
        ReproducibilityError: if the mapping contains a value JSON cannot
            represent, or a NaN or infinity. A NaN in a configuration hashes
            fine and compares unequal to itself forever after; refusing it here
            costs nothing and removes a class of unreproducible run.
    """
    try:
        return json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        msg = (
            f"configuration is not canonically JSON-encodable ({error}); a run whose "
            "configuration cannot be serialized cannot be reproduced from it (I2)"
        )
        raise ReproducibilityError(msg) from error


def config_hash(config: Mapping[str, JsonValue]) -> str:
    """Return the SHA-256 hex digest of a configuration's canonical JSON.

    Args:
        config: the configuration, as for :func:`canonical_json`.

    Returns:
        A 64-character lowercase hex digest. Stable across processes, machines
        and Python versions for the same configuration.

    Raises:
        ReproducibilityError: propagated from :func:`canonical_json`.
    """
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Git commit resolution (I2: "git commit")
# ---------------------------------------------------------------------------


def _resolve_git_dir(start: Path) -> Path:
    """Return the ``.git`` directory governing ``start``, following worktree files."""
    for candidate in (start, *start.parents):
        git_path = candidate / ".git"
        if git_path.is_dir():
            return git_path
        if git_path.is_file():
            # A linked worktree or submodule: the file holds "gitdir: <path>".
            content = git_path.read_text(encoding="utf-8").strip()
            prefix = "gitdir:"
            if not content.startswith(prefix):
                msg = f"{git_path} is a file but does not contain a 'gitdir:' pointer"
                raise ReproducibilityError(msg)
            target = Path(content.removeprefix(prefix).strip())
            return target if target.is_absolute() else (candidate / target).resolve()
    msg = (
        f"no .git directory found at or above {start}; the run cannot record the "
        "git commit invariant I2 requires. Pass the commit explicitly if this code "
        "is running from an export rather than a checkout."
    )
    raise ReproducibilityError(msg)


def _read_ref(git_dir: Path, ref: str) -> str:
    """Resolve a symbolic ref to a commit id, checking loose refs then packed-refs."""
    loose = git_dir / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip() == ref:
                return parts[0].strip()
    msg = f"git ref {ref!r} could not be resolved in {git_dir}"
    raise ReproducibilityError(msg)


def current_git_commit(start: Path | None = None) -> str:
    """Return the commit id of ``HEAD``, read from the ``.git`` directory.

    Implemented by reading ``.git`` directly rather than by running ``git``:
    the artifact is written inside library code, and a subprocess call there is
    both a portability and a sandboxing liability.

    **What this cannot tell you.** It reports what ``HEAD`` points at. It cannot
    detect uncommitted modifications, so an artifact produced from a dirty
    worktree carries a commit id that does not describe the code that ran. That
    gap is real and is not closed here; runs whose reproducibility claim must be
    airtight have to be produced from a clean checkout (CI is the natural place),
    and an operator-facing run entrypoint should assert cleanliness before
    calling this. Recorded plainly rather than papered over.

    Args:
        start: directory to begin the upward search from. Defaults to the
            directory containing this module, which resolves to the repository
            checkout in every normal deployment.

    Returns:
        The 40-character (SHA-1) or 64-character (SHA-256) lowercase hex commit id.

    Raises:
        ReproducibilityError: if no ``.git`` is found, if ``HEAD`` or its ref
            cannot be read, or if the resolved value is not a hex commit id.
            Never returns a placeholder — see :class:`ReproducibilityError`.
    """
    origin = (start or Path(__file__).resolve().parent).resolve()
    git_dir = _resolve_git_dir(origin)
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        msg = f"{head_path} does not exist; cannot determine the current commit"
        raise ReproducibilityError(msg)
    head = head_path.read_text(encoding="utf-8").strip()
    commit = (
        _read_ref(git_dir, head.removeprefix("ref:").strip()) if head.startswith("ref:") else head
    )
    return _validated_commit(commit)


def _validated_commit(commit: str) -> str:
    """Return ``commit`` if it is a lowercase hex git object id, else raise."""
    value = commit.strip()
    if len(value) not in _GIT_SHA_LENGTHS or not set(value) <= _HEX_DIGITS:
        msg = (
            f"git commit {commit!r} is not a 40- or 64-character lowercase hex object "
            "id; an artifact may not record a commit it cannot resolve (I2)"
        )
        raise ReproducibilityError(msg)
    return value


# ---------------------------------------------------------------------------
# The I2 stamp and the I4 provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReproducibilityStamp:
    """The four things a result needs to be regenerable (invariant I2).

    Every field is required. There is no default, no ``None``, and no
    placeholder string: directive §2 says "Store all four with every artifact",
    and a field that can default to ``"unknown"`` is a field that will.

    Attributes:
        git_commit: the commit the run executed from — 40 hex characters
            (SHA-1) or 64 (SHA-256), lowercase. See :func:`current_git_commit`
            for what it does and does not attest.
        data_version: identifier of the dataset the run read, supplied by the
            data source itself rather than by the caller, so a run cannot
            mislabel its own inputs. Non-empty.
        config_hash: 64-character SHA-256 hex digest of the run configuration,
            from :func:`config_hash`.
        seed: the pseudo-random seed the run used, in ``[0, 2**64)``. The
            simulation is deterministic; the seed drives interval estimation
            (see :class:`IntervalConfig`), so two runs differing only in seed
            agree on every point estimate and differ on every interval.
    """

    git_commit: str
    data_version: str
    config_hash: str
    seed: int

    def __post_init__(self) -> None:
        """Validate all four fields.

        Raises:
            ReproducibilityError: if the commit is not a hex object id, the
                data version is blank, the config hash is not a SHA-256 hex
                digest, or the seed is not an integer in ``[0, 2**64)``.
        """
        object.__setattr__(self, "git_commit", _validated_commit(self.git_commit))
        if not self.data_version.strip():
            msg = (
                "data_version must name the dataset the run read; a blank version "
                "makes the result unreproducible even with the commit and the config (I2)"
            )
            raise ReproducibilityError(msg)
        digest = self.config_hash.strip().lower()
        if len(digest) != 64 or not set(digest) <= _HEX_DIGITS:
            msg = f"config_hash must be a 64-character SHA-256 hex digest; got {self.config_hash!r}"
            raise ReproducibilityError(msg)
        object.__setattr__(self, "config_hash", digest)
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            msg = f"seed must be an int, not {type(self.seed).__name__}"
            raise ReproducibilityError(msg)
        if not 0 <= self.seed < _MAX_SEED:
            msg = f"seed must lie in [0, 2**64); got {self.seed!r}"
            raise ReproducibilityError(msg)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the stamp as a JSON-safe mapping for storage and display."""
        return {
            "git_commit": self.git_commit,
            "data_version": self.data_version,
            "config_hash": self.config_hash,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class CostProvenance:
    """Where an artifact's cost numbers came from (invariant I4, Gate G9).

    The P9.3 cost model marks every :class:`~backend.costs.model.TradeCost`
    with ``uncalibrated`` and a ``calibration_basis``. This type carries both
    onto the run artifact and into :meth:`RunArtifact.to_dict`, so a consumer
    can state that the numbers rest on assumed costs without reaching back into
    the cost package or reading prose.

    Attributes:
        uncalibrated: ``True`` while no cost parameter has been fitted to an
            observed fill. Per DECISIONS.md D-013 this stays ``True`` even
            after P11 paper fills exist unless a documented haircut is applied,
            because paper fills bound slippage from below.
        calibration_basis: the parameter set's own statement of provenance,
            displayed verbatim wherever a cost-derived number is shown.
        parameters: the numeric cost parameters actually used, in their
            documented units. Carried so that a run with the cost model turned
            down — or off — is visible in its own artifact rather than
            indistinguishable from a properly costed one.
    """

    uncalibrated: bool
    calibration_basis: str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        """Validate the provenance statement.

        Raises:
            ArtifactError: if ``calibration_basis`` is blank. An empty basis
                makes an assumed cost indistinguishable from a measured one,
                which is precisely what the flag exists to prevent.
        """
        if not self.calibration_basis.strip():
            msg = "calibration_basis must be non-empty: costs must state their provenance (I4)"
            raise ArtifactError(msg)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @classmethod
    def from_params(cls, params: CostModelParams) -> CostProvenance:
        """Build provenance from a :class:`~backend.costs.model.CostModelParams`.

        Args:
            params: the parameter set the run costed its trades with.

        Returns:
            A :class:`CostProvenance` carrying the flag, the basis string and a
            snapshot of every numeric parameter.
        """
        return cls(
            uncalibrated=params.uncalibrated,
            calibration_basis=params.calibration_basis,
            parameters={
                "half_spread_bps": float(params.half_spread_bps),
                "commission_bps": float(params.commission_bps),
                "impact_coefficient": float(params.impact_coefficient),
                "default_daily_volatility_bps": float(params.default_daily_volatility_bps),
                "borrow_rate_bps_per_year": float(params.borrow_rate_bps_per_year),
            },
        )

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the provenance as a JSON-safe mapping."""
        return {
            "uncalibrated": self.uncalibrated,
            "calibration_basis": self.calibration_basis,
            "parameters": dict(self.parameters),
        }


# ---------------------------------------------------------------------------
# Intervals (§5-P10: "No point estimate is ever displayed alone")
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Estimate:
    """A metric that travels with its interval.

    Directive §5-P10 states the requirement as a UI convention. This type turns
    as much of it as a type can into a representation guarantee, and the limit
    is worth stating exactly rather than overclaiming:

    * **What is guaranteed.** A metric is never *represented* as a bare number.
      Every metric on a :class:`PerformanceSummary` and every field of a
      :class:`BenchmarkComparison` is an :class:`Estimate`, so anything holding
      a metric holds its interval too; :meth:`to_dict` always emits the bounds
      alongside the value; :meth:`__str__` always renders them; and no implicit
      numeric coercion exists — the class defines no ``__float__``, no
      ``__index__`` and no ``__format__``, so ``float(estimate)``,
      ``f"{estimate:.2f}"`` and arithmetic on it all raise :class:`TypeError`
      rather than silently discarding the interval.
    * **What is not.** ``estimate.value`` is a public field and reading it
      yields a bare float. It has to: something eventually renders the number.
      The guarantee is that reaching for it is an explicit, greppable act by a
      consumer who had the interval in hand, not the path of least resistance.

    Attributes:
        value: the point estimate, in the units named by ``units``.
        lower: lower interval bound, same units.
        upper: upper interval bound, same units.
        level: central mass the interval covers, in ``(0, 1)`` — ``0.95`` is a
            95% interval.
        method: how the interval was produced, stated precisely enough that a
            reader knows what it does and does not account for.
        units: the units of ``value``, ``lower`` and ``upper``.

    Note:
        A percentile bootstrap interval is not guaranteed to contain the point
        estimate when the bootstrap distribution is strongly skewed, so
        containment is **not** validated. Only ``lower <= upper`` is. A value
        outside its own interval is information about the sample, not a defect
        to be clamped away.
    """

    value: float
    lower: float
    upper: float
    level: float
    method: str
    units: str

    def __post_init__(self) -> None:
        """Validate finiteness, ordering, level and labelling.

        Raises:
            ValueError: if any bound is non-finite, if ``lower > upper``, if
                ``level`` is not strictly inside ``(0, 1)``, or if ``method`` or
                ``units`` is blank. An unlabelled interval is not reportable:
                the reader cannot tell a bootstrap percentile interval from a
                normal-theory one, and they mean different things.
        """
        for name in ("value", "lower", "upper"):
            number = float(getattr(self, name))
            if not math.isfinite(number):
                msg = f"Estimate.{name} must be finite; got {number!r}"
                raise ValueError(msg)
            object.__setattr__(self, name, number)
        if self.lower > self.upper:
            msg = f"Estimate interval is inverted: lower={self.lower!r} > upper={self.upper!r}"
            raise ValueError(msg)
        if not 0.0 < self.level < 1.0:
            msg = f"Estimate.level must lie strictly in (0, 1); got {self.level!r}"
            raise ValueError(msg)
        if not self.method.strip():
            msg = "Estimate.method must state how the interval was produced"
            raise ValueError(msg)
        if not self.units.strip():
            msg = "Estimate.units must state the units of the value and its bounds"
            raise ValueError(msg)

    def __str__(self) -> str:
        """Render the value with its interval, never the value alone."""
        return f"{self.value:.6g} [{self.lower:.6g}, {self.upper:.6g}] ({self.level:.0%} interval)"

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the estimate as a JSON-safe mapping including the interval."""
        return {
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "level": self.level,
            "method": self.method,
            "units": self.units,
        }


@dataclass(frozen=True, slots=True)
class IntervalConfig:
    """How intervals are estimated, and the seed that makes them reproducible.

    Attributes:
        seed: seed for the resampler, in ``[0, 2**64)``. Recorded on the run's
            :class:`ReproducibilityStamp`.
        level: central mass of the reported interval, in ``(0, 1)``.
        n_resamples: bootstrap replicates per metric.
        block_length: length of the resampled blocks, in observations. ``None``
            selects ``max(1, round(n ** (1/3)))``, the usual rule of thumb.
            A block bootstrap is used rather than an i.i.d. one because daily
            strategy returns are serially dependent — turnover, momentum and
            volatility clustering all induce it — and an i.i.d. bootstrap would
            report intervals that are too narrow, which is the wrong direction
            to be wrong in.
    """

    seed: int
    level: float = DEFAULT_CONFIDENCE_LEVEL
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    block_length: int | None = None

    def __post_init__(self) -> None:
        """Validate the interval configuration.

        Raises:
            ValueError: if the seed is out of range, the level is not strictly
                inside ``(0, 1)``, the replicate count is below 2, or the block
                length is not a positive integer.
        """
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            msg = f"seed must be an int, not {type(self.seed).__name__}"
            raise ValueError(msg)
        if not 0 <= self.seed < _MAX_SEED:
            msg = f"seed must lie in [0, 2**64); got {self.seed!r}"
            raise ValueError(msg)
        if not 0.0 < self.level < 1.0:
            msg = f"level must lie strictly in (0, 1); got {self.level!r}"
            raise ValueError(msg)
        if self.n_resamples < 2:
            msg = f"n_resamples must be at least 2 to form an interval; got {self.n_resamples!r}"
            raise ValueError(msg)
        if self.block_length is not None and self.block_length < 1:
            msg = (
                f"block_length must be a positive number of observations; got {self.block_length!r}"
            )
            raise ValueError(msg)

    def resolved_block_length(self, n_observations: int) -> int:
        """Return the block length to use for a sample of ``n_observations``.

        Args:
            n_observations: length of the return series being resampled.

        Returns:
            ``block_length`` if set, else ``max(1, round(n ** (1/3)))`` capped
            at the sample length.
        """
        if self.block_length is not None:
            return min(self.block_length, n_observations)
        return max(1, min(n_observations, round(math.pow(n_observations, 1.0 / 3.0))))

    def method_label(self, n_observations: int) -> str:
        """Return the ``Estimate.method`` string describing this configuration.

        Args:
            n_observations: length of the return series being resampled.

        Returns:
            A string naming the resampling scheme, its parameters and the seed,
            and stating what the interval does not account for.
        """
        return (
            f"circular block bootstrap percentile interval "
            f"(B={self.n_resamples}, block={self.resolved_block_length(n_observations)} periods, "
            f"seed={self.seed}); sampling uncertainty of one net-of-cost path only — "
            f"not corrected for trial count (see backend.backtest.dsr) and not a "
            f"distribution over backtest paths (see backend.backtest.cpcv)"
        )


def _block_bootstrap_indices(
    *,
    rng: np.random.Generator,
    n_observations: int,
    block_length: int,
    n_resamples: int,
) -> IntArray:
    """Return ``(n_resamples, n_observations)`` circular block bootstrap indices."""
    n_blocks = -(-n_observations // block_length)
    starts = rng.integers(0, n_observations, size=(n_resamples, n_blocks), dtype=np.int64)
    offsets = np.arange(block_length, dtype=np.int64)
    drawn = (starts[:, :, None] + offsets[None, None, :]) % n_observations
    return np.asarray(drawn.reshape(n_resamples, -1)[:, :n_observations], dtype=np.int64)


def _equity_paths(returns: FloatArray) -> FloatArray:
    """Return ``(..., T+1)`` growth-of-one paths for ``(..., T)`` return rows."""
    growth = np.cumprod(1.0 + returns, axis=-1)
    ones = np.ones((*growth.shape[:-1], 1), dtype=np.float64)
    return np.concatenate([ones, growth], axis=-1)


def _total_return(returns: FloatArray) -> FloatArray:
    """Return compounded total return per row, as a fraction."""
    return np.asarray(np.prod(1.0 + returns, axis=-1) - 1.0, dtype=np.float64)


def _annualized_return(returns: FloatArray, periods_per_year: float) -> FloatArray:
    """Return the geometric annualized return per row, as a fraction."""
    growth = np.prod(1.0 + returns, axis=-1)
    exponent = periods_per_year / returns.shape[-1]
    with np.errstate(invalid="ignore"):
        annualized = np.where(growth > 0.0, np.power(np.abs(growth), exponent) - 1.0, np.nan)
    return np.asarray(annualized, dtype=np.float64)


def _annualized_volatility(returns: FloatArray, periods_per_year: float) -> FloatArray:
    """Return the annualized sample standard deviation per row, as a fraction."""
    return np.asarray(
        np.std(returns, axis=-1, ddof=1) * math.sqrt(periods_per_year), dtype=np.float64
    )


def _annualized_sharpe(returns: FloatArray, periods_per_year: float) -> FloatArray:
    """Return the annualized Sharpe ratio per row, matching :func:`sharpe_ratio`."""
    deviation = np.std(returns, axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(deviation > 0.0, np.mean(returns, axis=-1) / deviation, np.nan)
    return np.asarray(ratio * math.sqrt(periods_per_year), dtype=np.float64)


def _max_drawdown(returns: FloatArray) -> FloatArray:
    """Return the maximum drawdown per row, as a positive fraction of the peak."""
    paths = _equity_paths(returns)
    peaks = np.maximum.accumulate(paths, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peaks > 0.0, 1.0 - paths / peaks, np.nan)
    return np.asarray(np.max(drawdown, axis=-1), dtype=np.float64)


def max_drawdown_of_equity(equity_usd: Sequence[float] | FloatArray) -> float:
    """Return the maximum drawdown of an equity path, as a positive fraction.

    Args:
        equity_usd: portfolio value over time in **US dollars**, strictly
            positive and in chronological order.

    Returns:
        ``max_t (peak_t - equity_t) / peak_t``, a fraction in ``[0, 1)``.
        ``0.12`` means a 12% peak-to-trough decline. Zero for a path that never
        declines from its running peak.

    Raises:
        ValueError: if the path fails validation or contains a non-positive
            value, which would make the ratio meaningless.
    """
    values = as_float_array(equity_usd, name="equity_usd")
    if np.any(values <= 0.0):
        msg = "equity_usd must be strictly positive to express drawdown as a fraction of the peak"
        raise ValueError(msg)
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))


def _percentile_interval(
    replicates: FloatArray, level: float, *, metric: str
) -> tuple[float, float]:
    """Return the two-sided percentile interval of finite replicates.

    Non-finite replicates are dropped rather than propagated: a resample of a
    block bootstrap can be degenerate (every drawn block identical), which makes
    a ratio metric undefined for that replicate alone. Dropping a handful is
    honest; dropping nearly all of them means the metric is undefined for this
    series, and that raises.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        msg = (
            f"bootstrap produced fewer than two usable replicates for {metric!r}; the "
            "interval would be a point estimate wearing an interval's clothes. The usual "
            "cause is a return series with no variation to resample."
        )
        raise ValueError(msg)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class EquityCurve:
    """A dated, strictly positive, net-of-cost portfolio value path.

    Net returns are a **derived property**, never a stored field, so a curve
    whose returns disagree with its equity cannot be constructed.

    Attributes:
        dates: strictly increasing timezone-aware UTC instants, one per equity
            point. Length ``T + 1``.
        equity_usd: portfolio value in **US dollars** at each date, strictly
            positive. ``equity_usd[0]`` is the initial capital.
    """

    dates: tuple[dt.datetime, ...]
    equity_usd: FloatArray

    def __post_init__(self) -> None:
        """Validate the dates and the equity path.

        Raises:
            ValueError: if there are fewer than three points (two returns are
                the minimum for any dispersion statistic), if the lengths
                disagree, if a date is naive, non-UTC or out of order, or if any
                equity value is not strictly positive. A path that touches zero
                has no defined return thereafter, and reporting one would be
                fiction.
        """
        values = as_float_array(self.equity_usd, name="equity_usd")
        object.__setattr__(self, "equity_usd", values)
        object.__setattr__(self, "dates", tuple(self.dates))
        if len(self.dates) != values.size:
            msg = f"dates ({len(self.dates)}) and equity_usd ({values.size}) must have equal length"
            raise ValueError(msg)
        if values.size < 3:
            msg = (
                f"an equity curve needs at least 3 points (2 return periods) before any "
                f"dispersion statistic is defined; got {values.size}"
            )
            raise ValueError(msg)
        for index, moment in enumerate(self.dates):
            if moment.tzinfo is None or moment.utcoffset() != dt.timedelta(0):
                msg = f"dates[{index}] must be timezone-aware UTC; got {moment!r}"
                raise ValueError(msg)
            if index and moment <= self.dates[index - 1]:
                msg = f"dates must strictly increase; dates[{index}]={moment!r} does not"
                raise ValueError(msg)
        if np.any(values <= 0.0):
            msg = "equity_usd must be strictly positive at every point"
            raise ValueError(msg)

    @property
    def net_returns(self) -> FloatArray:
        """Return the per-period **net-of-cost** simple returns, as fractions.

        Length ``T``. ``net_returns[i]`` is the return realized over
        ``(dates[i], dates[i + 1]]``.
        """
        values = self.equity_usd
        return np.asarray(values[1:] / values[:-1] - 1.0, dtype=np.float64)

    @property
    def n_periods(self) -> int:
        """Return the number of return periods, ``len(dates) - 1``."""
        return int(self.equity_usd.size - 1)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the curve as a JSON-safe mapping of ISO dates and values."""
        return {
            "dates": [moment.isoformat() for moment in self.dates],
            "equity_usd": [float(value) for value in self.equity_usd],
            "net_returns": [float(value) for value in self.net_returns],
            "units": "equity_usd in USD; net_returns are per-period fractions, net of costs",
        }


@dataclass(frozen=True, slots=True)
class RealizedAccounting:
    """Exact realized quantities of a run — facts, not estimates.

    These are deliberately **not** :class:`Estimate` objects. An interval on a
    realized cost total would be false precision: the run traded what it traded
    and paid what the model charged. Only statistics of the return *process*
    carry intervals, and those live on :class:`PerformanceSummary`.

    Attributes:
        initial_capital_usd: starting portfolio value, US dollars.
        final_equity_usd: ending portfolio value, US dollars.
        half_spread_usd: total paid crossing the spread, US dollars.
        commission_usd: total commission and fees, US dollars.
        impact_usd: total modelled market impact, US dollars.
        borrow_usd: total borrow accrued on short exposure, US dollars.
        total_cost_usd: sum of the four components, US dollars.
        traded_notional_usd: total absolute notional traded, US dollars.
        n_orders: number of non-zero orders placed across the run.
        n_rebalances: number of decision points at which the strategy was asked
            for target weights.
    """

    initial_capital_usd: float
    final_equity_usd: float
    half_spread_usd: float
    commission_usd: float
    impact_usd: float
    borrow_usd: float
    total_cost_usd: float
    traded_notional_usd: float
    n_orders: int
    n_rebalances: int

    @property
    def cost_bps_of_traded_notional(self) -> float:
        """Return execution cost as basis points of traded notional, or 0 if nothing traded.

        Excludes borrow, which accrues on held exposure rather than on traded
        notional and would be meaningless expressed per unit traded.
        """
        if self.traded_notional_usd == 0.0:
            return 0.0
        execution = self.half_spread_usd + self.commission_usd + self.impact_usd
        return execution / self.traded_notional_usd * 10_000.0

    @property
    def turnover_per_rebalance(self) -> float:
        """Return mean traded notional per rebalance as a fraction of initial capital."""
        if self.n_rebalances == 0:
            return 0.0
        return self.traded_notional_usd / self.n_rebalances / self.initial_capital_usd

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the accounting as a JSON-safe mapping, with a cost waterfall."""
        return {
            "initial_capital_usd": self.initial_capital_usd,
            "final_equity_usd": self.final_equity_usd,
            "cost_waterfall_usd": {
                "half_spread": self.half_spread_usd,
                "commission": self.commission_usd,
                "impact": self.impact_usd,
                "borrow": self.borrow_usd,
                "total": self.total_cost_usd,
            },
            "traded_notional_usd": self.traded_notional_usd,
            "cost_bps_of_traded_notional": self.cost_bps_of_traded_notional,
            "turnover_per_rebalance": self.turnover_per_rebalance,
            "n_orders": self.n_orders,
            "n_rebalances": self.n_rebalances,
            "units": "all *_usd in USD; cost_bps_of_traded_notional in basis points",
        }


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    """Every displayed statistic of one net-of-cost track record, with intervals.

    Attributes:
        metrics: metric name to :class:`Estimate`. Must contain every name in
            :data:`_REQUIRED_METRICS`; a summary that is missing one cannot be
            constructed, so a UI cannot render a partial result and a reader
            cannot mistake an omission for a zero.
        periods_per_year: annualization factor used for the annualized metrics.
        n_periods: number of return periods the statistics were computed on.
        accounting: exact realized quantities (see :class:`RealizedAccounting`).
    """

    metrics: Mapping[str, Estimate]
    periods_per_year: float
    n_periods: int
    accounting: RealizedAccounting

    def __post_init__(self) -> None:
        """Freeze the metric mapping and require the full metric set.

        Raises:
            ArtifactError: if any required metric is absent.
        """
        object.__setattr__(self, "metrics", dict(self.metrics))
        missing = [name for name in _REQUIRED_METRICS if name not in self.metrics]
        if missing:
            msg = (
                f"PerformanceSummary is missing required metric(s) {missing}; every "
                "result must carry the full set so a display cannot omit one silently"
            )
            raise ArtifactError(msg)

    @property
    def total_return(self) -> Estimate:
        """Return the compounded net-of-cost total return, as a fraction."""
        return self.metrics["total_return"]

    @property
    def annualized_return(self) -> Estimate:
        """Return the geometric annualized net-of-cost return, as a fraction."""
        return self.metrics["annualized_return"]

    @property
    def annualized_volatility(self) -> Estimate:
        """Return the annualized standard deviation of net returns, as a fraction."""
        return self.metrics["annualized_volatility"]

    @property
    def sharpe(self) -> Estimate:
        """Return the annualized net-of-cost Sharpe ratio, dimensionless."""
        return self.metrics["sharpe_ratio"]

    @property
    def max_drawdown(self) -> Estimate:
        """Return the maximum drawdown as a positive fraction of the running peak."""
        return self.metrics["max_drawdown"]

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the summary as a JSON-safe mapping. Every metric carries its interval."""
        return {
            "metrics": {name: estimate.to_dict() for name, estimate in self.metrics.items()},
            "periods_per_year": self.periods_per_year,
            "n_periods": self.n_periods,
            "accounting": self.accounting.to_dict(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class TrackRecord:
    """One net-of-cost equity path together with its interval-carrying summary.

    Attributes:
        equity: the dated equity path, in US dollars.
        summary: the statistics of that path, every one an :class:`Estimate`.
    """

    equity: EquityCurve
    summary: PerformanceSummary

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the track record as a JSON-safe mapping."""
        return {"equity": self.equity.to_dict(), "summary": self.summary.to_dict()}


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Strategy against benchmark, both net of costs, both with intervals.

    Attributes:
        active_return: mean per-period strategy return minus benchmark return,
            as a fraction. Positive means the strategy beat buy-and-hold after
            costs on both sides.
        information_ratio: annualized mean active return divided by the
            annualized standard deviation of active returns, dimensionless.
        benchmark_description: what the benchmark actually is, in words, so a
            reader is never left guessing which index or weighting was used.
    """

    active_return: Estimate
    information_ratio: Estimate
    benchmark_description: str

    def __post_init__(self) -> None:
        """Require the benchmark to describe itself.

        Raises:
            ArtifactError: if ``benchmark_description`` is blank. "Outperformed
                the benchmark" is not a statement until the benchmark is named.
        """
        if not self.benchmark_description.strip():
            msg = "benchmark_description must state what the strategy is being compared against"
            raise ArtifactError(msg)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the comparison as a JSON-safe mapping."""
        return {
            "active_return": self.active_return.to_dict(),
            "information_ratio": self.information_ratio.to_dict(),
            "benchmark_description": self.benchmark_description,
        }


@dataclass(frozen=True, slots=True, eq=False)
class RunArtifact:
    """The complete, self-describing record of one backtest run.

    Four things are structurally impossible to omit, each for a directive
    reason:

    * **the reproducibility stamp** (I2) — no default, no placeholder;
    * **the cost provenance** (I4) — the artifact cannot be built without it,
      and its ``uncalibrated`` flag and basis string reach
      :meth:`to_dict` and therefore the UI;
    * **the benchmark** (§5-P10) — a required field, validated to share the
      strategy's dates, so "always shown alongside every strategy result" is
      enforced rather than remembered;
    * **intervals** — every metric on both summaries is an :class:`Estimate`.

    Attributes:
        stamp: the I2 reproducibility stamp.
        costs: the I4 cost provenance.
        strategy: the strategy's net-of-cost track record.
        benchmark: the benchmark's net-of-cost track record, on the same dates.
        comparison: strategy-versus-benchmark statistics, with intervals.
        created_at: when the artifact was built, timezone-aware UTC. Excluded
            from :meth:`results_digest` because it is metadata about the run,
            not a result of it.
    """

    stamp: ReproducibilityStamp
    costs: CostProvenance
    strategy: TrackRecord
    benchmark: TrackRecord
    comparison: BenchmarkComparison
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def __post_init__(self) -> None:
        """Validate that the benchmark is comparable to the strategy.

        Raises:
            ArtifactError: if the benchmark's dates differ from the strategy's
                in length or in value. A benchmark measured over different dates
                is not a comparison, it is a coincidence; and if ``created_at``
                is not timezone-aware UTC.
        """
        if self.created_at.tzinfo is None or self.created_at.utcoffset() != dt.timedelta(0):
            msg = f"created_at must be timezone-aware UTC; got {self.created_at!r}"
            raise ArtifactError(msg)
        strategy_dates = self.strategy.equity.dates
        benchmark_dates = self.benchmark.equity.dates
        if strategy_dates != benchmark_dates:
            msg = (
                "benchmark dates differ from strategy dates: a benchmark computed over "
                f"a different calendar is not a comparison "
                f"(strategy {len(strategy_dates)} points, benchmark {len(benchmark_dates)})"
            )
            raise ArtifactError(msg)

    @property
    def disclosures(self) -> tuple[str, ...]:
        """Return the statements that must accompany every displayed figure.

        Always contains :data:`NET_OF_COST_DISCLOSURE`. While the cost model is
        uncalibrated it additionally carries the model's own
        ``calibration_basis``, which is how Gate G9's "conservative defaults,
        clearly flagged" clause reaches the operator rather than stopping at the
        cost package.
        """
        statements = [NET_OF_COST_DISCLOSURE]
        if self.costs.uncalibrated:
            statements.append(
                "COSTS ARE UNCALIBRATED — every figure below rests on assumed cost "
                f"parameters, not measured ones. Basis: {self.costs.calibration_basis}"
            )
        return tuple(statements)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the whole artifact as a JSON-safe mapping.

        This is the payload the API and dashboard render. It carries the I2
        stamp, the I4 cost provenance including ``uncalibrated`` and
        ``calibration_basis``, both track records with every metric's interval,
        the comparison, and the disclosures.
        """
        return {
            "created_at": self.created_at.isoformat(),
            "reproducibility": self.stamp.to_dict(),
            "costs": self.costs.to_dict(),
            "uncalibrated": self.costs.uncalibrated,
            "calibration_basis": self.costs.calibration_basis,
            "strategy": self.strategy.to_dict(),
            "benchmark": self.benchmark.to_dict(),
            "comparison": self.comparison.to_dict(),
            "disclosures": list(self.disclosures),
        }

    def results_digest(self) -> str:
        """Return a SHA-256 digest over everything except ``created_at``.

        Two runs of the same code on the same data with the same configuration
        and seed produce the same digest; changing the seed changes it, because
        the seed drives interval estimation. This is the handle the
        reproducibility test compares, and the handle an operator can compare
        between two runs without diffing two large JSON documents.

        Returns:
            A 64-character lowercase hex digest.
        """
        payload = self.to_dict()
        payload.pop("created_at", None)
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def summarize_track_record(
    *,
    equity: EquityCurve,
    accounting: RealizedAccounting,
    periods_per_year: float,
    intervals: IntervalConfig,
) -> PerformanceSummary:
    """Summarize a net-of-cost equity path with an interval on every metric.

    The point estimates are computed directly from the realized path; the
    intervals come from a circular block bootstrap of the realized net return
    series, seeded from ``intervals``. The Sharpe ratio point estimate is
    delegated to :func:`backend.backtest.metrics.sharpe_ratio` so this module
    cannot drift from the estimator conventions the rest of the validation
    framework uses.

    Args:
        equity: the net-of-cost equity path, in US dollars.
        accounting: exact realized quantities for the same run.
        periods_per_year: annualization factor (252 for daily bars). Must be
            positive.
        intervals: bootstrap configuration and seed.

    Returns:
        A :class:`PerformanceSummary` carrying an :class:`Estimate` for every
        required metric.

    Raises:
        ValueError: if ``periods_per_year`` is not positive, if the net returns
            have zero variance (an undefined Sharpe ratio, which must never be
            reported as an infinity), or if the bootstrap cannot form an
            interval.
    """
    if periods_per_year <= 0.0:
        msg = f"periods_per_year must be positive; got {periods_per_year!r}"
        raise ValueError(msg)
    returns = equity.net_returns
    n_observations = int(returns.size)
    block_length = intervals.resolved_block_length(n_observations)
    method = intervals.method_label(n_observations)
    rng = np.random.default_rng(intervals.seed)
    indices = _block_bootstrap_indices(
        rng=rng,
        n_observations=n_observations,
        block_length=block_length,
        n_resamples=intervals.n_resamples,
    )
    resampled = returns[indices]

    point_sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    specification: tuple[tuple[str, float, FloatArray, str], ...] = (
        (
            "total_return",
            float(_total_return(returns)),
            _total_return(resampled),
            "fraction of initial capital, net of costs",
        ),
        (
            "annualized_return",
            float(_annualized_return(returns, periods_per_year)),
            _annualized_return(resampled, periods_per_year),
            "fraction per year, geometric, net of costs",
        ),
        (
            "annualized_volatility",
            float(_annualized_volatility(returns, periods_per_year)),
            _annualized_volatility(resampled, periods_per_year),
            "fraction per year, standard deviation of net returns",
        ),
        (
            "sharpe_ratio",
            point_sharpe,
            _annualized_sharpe(resampled, periods_per_year),
            "dimensionless, annualized, net of costs, zero risk-free rate",
        ),
        (
            "max_drawdown",
            max_drawdown_of_equity(equity.equity_usd),
            _max_drawdown(resampled),
            "positive fraction of the running peak",
        ),
    )

    metrics: dict[str, Estimate] = {}
    for name, value, replicates, units in specification:
        lower, upper = _percentile_interval(replicates, intervals.level, metric=name)
        metrics[name] = Estimate(
            value=value,
            lower=lower,
            upper=upper,
            level=intervals.level,
            method=method,
            units=units,
        )
    return PerformanceSummary(
        metrics=metrics,
        periods_per_year=periods_per_year,
        n_periods=n_observations,
        accounting=accounting,
    )


def compare_to_benchmark(
    *,
    strategy: EquityCurve,
    benchmark: EquityCurve,
    benchmark_description: str,
    periods_per_year: float,
    intervals: IntervalConfig,
) -> BenchmarkComparison:
    """Compare two net-of-cost paths measured over the same dates.

    Both series are already net of modelled costs, including the benchmark's
    own entry cost — a buy-and-hold that pays nothing to get invested is not a
    benchmark, it is a handicap awarded to the strategy.

    Args:
        strategy: the strategy's net-of-cost equity path.
        benchmark: the benchmark's net-of-cost equity path.
        benchmark_description: what the benchmark is, in words.
        periods_per_year: annualization factor.
        intervals: bootstrap configuration and seed.

    Returns:
        A :class:`BenchmarkComparison` with intervals on both statistics.

    Raises:
        ValueError: if the two paths do not share identical dates, or if the
            active return series has zero variance (an undefined information
            ratio).
    """
    if strategy.dates != benchmark.dates:
        msg = (
            "strategy and benchmark must be measured on identical dates; "
            f"got {len(strategy.dates)} and {len(benchmark.dates)} points"
        )
        raise ValueError(msg)
    active = np.asarray(strategy.net_returns - benchmark.net_returns, dtype=np.float64)
    if float(np.std(active, ddof=1)) == 0.0:
        # Raised rather than reported as an infinity or a zero. The usual cause
        # is a strategy that *is* its benchmark — an exact tracker, or a run
        # whose benchmark holds the only name the strategy trades. Its active
        # return is a constant, so there is no ratio to report; "0.0" would read
        # as "no edge" when the truth is "this number does not exist".
        msg = (
            "the active return series has zero variance, so the information ratio is "
            "undefined. The strategy tracks its benchmark exactly over these dates "
            "— compare it against a benchmark it does not replicate."
        )
        raise ValueError(msg)
    n_observations = int(active.size)
    method = intervals.method_label(n_observations)
    # Point estimates first: a strategy that tracks its benchmark exactly has no
    # information ratio at all, and `sharpe_ratio` says so far more clearly than
    # a bootstrap that finds nothing to resample.
    point_active = float(np.mean(active))
    point_ratio = sharpe_ratio(active, periods_per_year=periods_per_year)
    rng = np.random.default_rng(intervals.seed)
    indices = _block_bootstrap_indices(
        rng=rng,
        n_observations=n_observations,
        block_length=intervals.resolved_block_length(n_observations),
        n_resamples=intervals.n_resamples,
    )
    resampled = active[indices]

    mean_lower, mean_upper = _percentile_interval(
        np.asarray(np.mean(resampled, axis=-1), dtype=np.float64),
        intervals.level,
        metric="active_return",
    )
    ratio_lower, ratio_upper = _percentile_interval(
        _annualized_sharpe(resampled, periods_per_year),
        intervals.level,
        metric="information_ratio",
    )
    return BenchmarkComparison(
        active_return=Estimate(
            value=point_active,
            lower=mean_lower,
            upper=mean_upper,
            level=intervals.level,
            method=method,
            units="fraction per period, strategy net return minus benchmark net return",
        ),
        information_ratio=Estimate(
            value=point_ratio,
            lower=ratio_lower,
            upper=ratio_upper,
            level=intervals.level,
            method=method,
            units="dimensionless, annualized mean active return over its standard deviation",
        ),
        benchmark_description=benchmark_description,
    )
