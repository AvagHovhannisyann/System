"""Deflated Sharpe Ratio — P10.3.

What the number means
---------------------

Search long enough over enough configurations and one of them will show a high
Sharpe ratio on pure noise. The Deflated Sharpe Ratio (DSR) is the probability
that an observed Sharpe ratio exceeds what the *search itself* would be expected
to produce, given:

a. how many configurations were tried,
b. how much those configurations' Sharpe ratios varied,
c. the skewness and kurtosis of the return series, and
d. how long the sample is.

It is built in two pieces.

**Probabilistic Sharpe Ratio (PSR).** Bailey & López de Prado (2012), "The
Sharpe Ratio Efficient Frontier", *Journal of Risk* 15(2), 3-44 — the
probability that the true Sharpe ratio exceeds a benchmark ``SR*``::

    PSR(SR*) = Z[ (SR - SR*) * sqrt(n - 1)
                  / sqrt(1 - g3 * SR + (g4 - 1) / 4 * SR ** 2) ]

with ``Z`` the standard normal CDF, ``SR`` the observed **per-period** Sharpe
ratio, ``n`` the number of return observations, ``g3`` the skewness and ``g4``
the **non-excess** kurtosis of the returns. Negative skew and fat tails both
enlarge the denominator, which lowers the probability — that is the whole point
of using them: a Sharpe ratio earned by selling tail risk is worth less than the
same number earned from a symmetric, thin-tailed series.

**The deflated benchmark.** Bailey & López de Prado (2014), "The Deflated Sharpe
Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality",
*Journal of Portfolio Management* 40(5), 94-107. With ``N`` independent trials
whose Sharpe ratios have variance ``V``, the expected **maximum** Sharpe ratio
under a null of no skill is, by the extreme-value approximation for the maximum
of ``N`` iid Gaussians::

    SR*_0 = sqrt(V) * [ (1 - gamma) * Z^-1[1 - 1/N]
                        + gamma * Z^-1[1 - 1/(N * e)] ]

with ``gamma`` the Euler-Mascheroni constant. ``DSR = PSR(SR*_0)``.

Why the trial count is a required argument
------------------------------------------

A DSR computed with ``trials=1`` when 200 configurations were tried is the most
misleading number this system could emit: it is a rigorous-looking probability
that has skipped the only correction that matters. There is therefore **no
default** for ``trials`` or ``trial_sharpe_variance`` anywhere in this module —
omitting them is a ``TypeError``, not a quiet 1. The caller supplies them from
``TESTING_LEDGER.md`` (see :mod:`backend.backtest.ledger`), which directive §9.7
forbids deleting rows from for exactly this reason.

Caveat that needs a human: the formula assumes the ``N`` trials are
**independent**. A ledger of 200 rows that are mostly small perturbations of one
another represents far fewer than 200 independent trials, and using the raw row
count then *over*-deflates. Over-deflation is the conservative direction, so the
raw count is the right default behaviour, but a strategy rejected only narrowly
deserves a look at how correlated its trials really were.

Units
-----

``observed_sharpe``, ``benchmark_sharpe``, the trial Sharpe ratios and their
variance must all be **per observation period** and in the same periodicity as
``n_observations``. Passing an annualized Sharpe with a daily observation count
inflates the result by roughly ``sqrt(252)`` and produces a confident, wrong
answer. The ``*_from_returns`` and ``*_from_trials`` entry points exist to make
that mistake harder: they derive the Sharpe ratio and the moments from the
return series themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from backend.backtest import metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "EULER_MASCHERONI",
    "DeflatedSharpeResult",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "deflated_sharpe_ratio_from_trials",
    "expected_maximum_sharpe_ratio",
    "probabilistic_sharpe_ratio",
]

EULER_MASCHERONI = 0.5772156649015328606
"""The Euler-Mascheroni constant, the ``gamma`` of the expected-maximum formula."""

_KURTOSIS_FLOOR_TOLERANCE = 1e-9
"""How far below 1 a non-excess kurtosis may fall before it is refused.

Non-excess kurtosis is at least 1 for every distribution, with equality only for
the two-point symmetric one. That minimum is *attainable* — a returns series that
alternates between two values reaches it exactly — and evaluating ``m4 / m2 ** 2``
on such a sample in floating point lands a few ulps either side of 1. Refusing
``0.9999999999999999`` would reject a correctly computed moment.

The guard exists to catch an *excess* kurtosis passed where a non-excess one
belongs, and that mistake is off by about 3. A tolerance nine orders of magnitude
tighter than the error being guarded against keeps the guard useful while letting
the attainable minimum through.
"""


def probabilistic_sharpe_ratio(
    *,
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Compute the Probabilistic Sharpe Ratio ``PSR(SR*)``.

    ``PSR(SR*) = Z[(SR - SR*) * sqrt(n - 1) / sqrt(1 - g3*SR + (g4-1)/4 * SR**2)]``
    — Bailey & López de Prado (2012).

    Args:
        observed_sharpe: the estimated Sharpe ratio ``SR``, **per observation
            period** (not annualized), dimensionless.
        benchmark_sharpe: the threshold ``SR*`` the true Sharpe ratio must
            exceed, in the same per-period units. Use 0 for "better than
            nothing"; use :func:`expected_maximum_sharpe_ratio` for the deflated
            version.
        n_observations: number of return observations ``n`` behind
            ``observed_sharpe``. Must be at least 2.
        skewness: ``g3``, the population skewness of those returns
            (:func:`backend.backtest.metrics.skewness`).
        kurtosis: ``g4``, the population **non-excess** kurtosis of those
            returns (:func:`backend.backtest.metrics.kurtosis`); 3 for a
            Gaussian sample, never below 1. Passing an *excess* kurtosis here
            silently changes the answer. Values a few ulps below 1 are accepted
            (see :data:`_KURTOSIS_FLOOR_TOLERANCE`); an excess-kurtosis mix-up
            is off by about 3 and is still refused.

    Returns:
        A probability in ``[0, 1]``: the estimated probability that the true
        Sharpe ratio exceeds ``benchmark_sharpe``.

    Raises:
        ValueError: if ``n_observations < 2``, if ``kurtosis`` falls below 1 by
            more than :data:`_KURTOSIS_FLOOR_TOLERANCE` (no real distribution
            does, so it signals an excess-kurtosis mix-up), if any input is not
            finite, or if the variance term
            ``1 - g3*SR + (g4-1)/4 * SR**2`` is not positive — which means the
            moment estimates are mutually inconsistent and the formula has no
            meaning for them.
    """
    for name, value in (
        ("observed_sharpe", observed_sharpe),
        ("benchmark_sharpe", benchmark_sharpe),
        ("skewness", skewness),
        ("kurtosis", kurtosis),
    ):
        if not math.isfinite(value):
            msg = f"{name} must be finite; got {value!r}"
            raise ValueError(msg)
    if n_observations < 2:
        msg = f"n_observations must be at least 2; got {n_observations}"
        raise ValueError(msg)
    if kurtosis < 1.0 - _KURTOSIS_FLOOR_TOLERANCE:
        msg = (
            f"kurtosis must be at least 1 (non-excess kurtosis is >= 1 for every "
            f"distribution); got {kurtosis!r} — an excess kurtosis was probably passed"
        )
        raise ValueError(msg)
    variance_term = 1.0 - skewness * observed_sharpe + (kurtosis - 1.0) / 4.0 * observed_sharpe**2
    if variance_term <= 0.0:
        msg = (
            "the Sharpe ratio variance term 1 - g3*SR + (g4-1)/4*SR**2 is "
            f"{variance_term!r}, which is not positive; the supplied moments are "
            "mutually inconsistent and PSR is undefined for them"
        )
        raise ValueError(msg)
    z_score = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1)
    z_score /= math.sqrt(variance_term)
    return metrics.standard_normal_cdf(z_score)


def expected_maximum_sharpe_ratio(*, trials: int, trial_sharpe_variance: float) -> float:
    """Compute ``SR*_0``, the Sharpe ratio a search of this size produces by luck.

    ``SR*_0 = sqrt(V) * [(1 - gamma) * Z^-1[1 - 1/N] + gamma * Z^-1[1 - 1/(N*e)]]``
    — Bailey & López de Prado (2014), the extreme-value approximation to the
    expected maximum of ``N`` iid ``N(0, V)`` draws.

    Args:
        trials: ``N``, the number of **independent** configurations evaluated.
            Must be at least 1. Taken from ``TESTING_LEDGER.md``; see the module
            docstring on why correlated trials make this an upper bound on the
            effective count.
        trial_sharpe_variance: ``V``, the variance of those trials' per-period
            Sharpe ratios. Must be non-negative.

    Returns:
        The expected maximum Sharpe ratio under the null of no skill, in the
        same per-period units as the trial Sharpe ratios. Zero when
        ``trials == 1`` (the maximum of one draw from a zero-mean distribution
        has expectation zero) or when the variance is zero.

    Raises:
        ValueError: if ``trials < 1`` or ``trial_sharpe_variance`` is negative
            or not finite.
    """
    if trials < 1:
        msg = f"trials must be at least 1; got {trials}"
        raise ValueError(msg)
    if not math.isfinite(trial_sharpe_variance) or trial_sharpe_variance < 0.0:
        msg = (
            f"trial_sharpe_variance must be finite and non-negative; got {trial_sharpe_variance!r}"
        )
        raise ValueError(msg)
    if trials == 1 or trial_sharpe_variance == 0.0:
        # The extreme-value expression diverges at N = 1 (Z^-1[0] = -inf) while
        # the quantity it approximates is exactly 0 there: the expected maximum
        # of a single zero-mean draw. Return the exact value rather than the
        # approximation's singularity.
        return 0.0
    first = metrics.standard_normal_ppf(1.0 - 1.0 / trials)
    second = metrics.standard_normal_ppf(1.0 - 1.0 / (trials * math.e))
    return math.sqrt(trial_sharpe_variance) * (
        (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
    )


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    """A Deflated Sharpe Ratio together with every input that produced it.

    The inputs travel with the number on purpose: directive §6.7 requires the
    trial count to be displayed alongside the result, and a DSR without its
    trial count is not interpretable.

    Attributes:
        value: the DSR itself — a probability in ``[0, 1]`` that the strategy's
            true Sharpe ratio exceeds what a search of this size would produce
            by luck. Not a Sharpe ratio; it has no return units.
        observed_sharpe: the per-period Sharpe ratio that was deflated.
        expected_maximum_sharpe: ``SR*_0``, the deflated benchmark, per period.
        trials: ``N``, the trial count used.
        trial_sharpe_variance: ``V``, the variance across trial Sharpe ratios.
        n_observations: ``n``, the length of the return sample.
        skewness: ``g3`` of the return sample.
        kurtosis: ``g4`` (non-excess) of the return sample.
    """

    value: float
    observed_sharpe: float
    expected_maximum_sharpe: float
    trials: int
    trial_sharpe_variance: float
    n_observations: int
    skewness: float
    kurtosis: float


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    trials: int,
    trial_sharpe_variance: float,
) -> DeflatedSharpeResult:
    """Deflate a Sharpe ratio for selection bias and non-normality.

    ``DSR = PSR(SR*_0)`` where ``SR*_0`` is
    :func:`expected_maximum_sharpe_ratio` — Bailey & López de Prado (2014).

    Args:
        observed_sharpe: the estimated Sharpe ratio, **per observation period**,
            computed on **net-of-cost** returns (invariant I4).
        n_observations: number of return observations behind it. At least 2.
        skewness: ``g3`` of those returns.
        kurtosis: ``g4`` (non-excess) of those returns.
        trials: ``N``, the number of configurations evaluated. **Required — no
            default.** Passing 1 when many were tried removes the deflation
            entirely.
        trial_sharpe_variance: ``V``, the variance of the trials' per-period
            Sharpe ratios. **Required — no default.**

    Returns:
        A :class:`DeflatedSharpeResult`.

    Raises:
        ValueError: on any invalid input (see
            :func:`probabilistic_sharpe_ratio` and
            :func:`expected_maximum_sharpe_ratio`), and specifically when
            ``trials > 1`` with ``trial_sharpe_variance == 0``: a zero variance
            makes the deflation vanish no matter how many trials are declared,
            so the result would silently equal the *undeflated* PSR. That is
            almost always a missing-data error and is refused rather than
            reported.
    """
    if trials > 1 and trial_sharpe_variance == 0.0:
        msg = (
            f"trial_sharpe_variance is 0 with trials={trials}: the deflation term would "
            "vanish and the result would silently equal the undeflated Probabilistic "
            "Sharpe Ratio. Supply the variance of the trials' Sharpe ratios."
        )
        raise ValueError(msg)
    benchmark = expected_maximum_sharpe_ratio(
        trials=trials, trial_sharpe_variance=trial_sharpe_variance
    )
    value = probabilistic_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=benchmark,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    return DeflatedSharpeResult(
        value=value,
        observed_sharpe=observed_sharpe,
        expected_maximum_sharpe=benchmark,
        trials=trials,
        trial_sharpe_variance=trial_sharpe_variance,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def deflated_sharpe_ratio_from_returns(
    returns: Sequence[float] | npt.ArrayLike,
    *,
    trials: int,
    trial_sharpe_variance: float,
    risk_free_rate: float = 0.0,
    ddof: int = 1,
) -> DeflatedSharpeResult:
    """Deflate the Sharpe ratio of a return series, deriving its moments.

    Preferred over :func:`deflated_sharpe_ratio` because the Sharpe ratio,
    skewness, kurtosis and observation count are all read off the same series
    and therefore cannot disagree about periodicity.

    Args:
        returns: simple per-period returns as fractions, **net of modelled
            costs** (invariant I4). The Sharpe ratio derived here is per period
            and is never annualized, because the deflation formula requires per
            period.
        trials: ``N``, the number of configurations evaluated. Required.
        trial_sharpe_variance: ``V``, the variance of the trials' per-period
            Sharpe ratios. Required.
        risk_free_rate: per-period risk-free rate as a fraction. Default 0.
        ddof: delta degrees of freedom for the standard deviation. Default 1.

    Returns:
        A :class:`DeflatedSharpeResult`.

    Raises:
        ValueError: on any invalid input; see :func:`deflated_sharpe_ratio`.
    """
    values = metrics.as_float_array(returns, name="returns")
    return deflated_sharpe_ratio(
        observed_sharpe=metrics.sharpe_ratio(
            values, risk_free_rate=risk_free_rate, periods_per_year=None, ddof=ddof
        ),
        n_observations=int(values.size),
        skewness=metrics.skewness(values),
        kurtosis=metrics.kurtosis(values),
        trials=trials,
        trial_sharpe_variance=trial_sharpe_variance,
    )


def deflated_sharpe_ratio_from_trials(
    returns: Sequence[float] | npt.ArrayLike,
    *,
    trial_sharpes: Sequence[float] | npt.ArrayLike,
    risk_free_rate: float = 0.0,
    ddof: int = 1,
) -> DeflatedSharpeResult:
    """Deflate a Sharpe ratio using the full set of trial Sharpe ratios.

    This is the entry point for a caller holding the testing ledger: ``N`` is
    the number of recorded trials and ``V`` is their sample variance, so neither
    can be understated by accident.

    Args:
        returns: the selected strategy's per-period **net-of-cost** returns.
        trial_sharpes: the **per-period** Sharpe ratio of every configuration
            evaluated, including the selected one. Must contain at least one
            value.
        risk_free_rate: per-period risk-free rate as a fraction. Default 0.
        ddof: delta degrees of freedom for the return standard deviation.
            Default 1. The variance across trials always uses the sample
            estimator (``ddof=1``).

    Returns:
        A :class:`DeflatedSharpeResult`.

    Raises:
        ValueError: on any invalid input, and specifically when the selected
            strategy's Sharpe ratio exceeds every recorded trial's. That means
            the strategy being reported is not in the trial set, so the trial
            count understates the search — the exact condition directive §9.7
            exists to prevent — or the two are in different units (annualized
            versus per period). Either way the deflation would be wrong, so it
            is refused.
    """
    values = metrics.as_float_array(returns, name="returns")
    trials_array = metrics.as_float_array(trial_sharpes, name="trial_sharpes")
    observed = metrics.sharpe_ratio(
        values, risk_free_rate=risk_free_rate, periods_per_year=None, ddof=ddof
    )
    maximum_trial = float(np.max(trials_array))
    tolerance = 1e-9 * max(1.0, abs(maximum_trial))
    if observed > maximum_trial + tolerance:
        msg = (
            f"observed Sharpe ratio {observed!r} exceeds the best recorded trial "
            f"{maximum_trial!r}: the reported strategy is not among the trials, so the "
            "trial count understates the search. Either the ledger is incomplete or the "
            "trial Sharpe ratios are not in per-period units."
        )
        raise ValueError(msg)
    variance = float(np.var(trials_array, ddof=1)) if trials_array.size > 1 else 0.0
    return deflated_sharpe_ratio(
        observed_sharpe=observed,
        n_observations=int(values.size),
        skewness=metrics.skewness(values),
        kurtosis=metrics.kurtosis(values),
        trials=int(trials_array.size),
        trial_sharpe_variance=variance,
    )
