"""Property tests for the cross-sectional transform pipeline (P5.2).

`DECISIONS.md` **D-018** requires a property suite to randomize the *structure*
of what it exercises rather than the values fed through one fixed structure, so
the generators below draw a cross-section's shape first — the shape of the value
distribution, the absence pattern, the sector layout, the beta layout, the
magnitude — and its numbers second. A panel is then a list of independently
shaped cross-sections, which is what a real rebalance history looks like.

The properties, in the order they matter:

**1. No cross-date leakage.** The claim is that every transform sees exactly one
date. It is asserted two ways, because either alone is weak:

- *Structurally*, by reading the source rather than the behaviour. The two
  modules import nothing but ``numpy`` and each other; no public function takes
  a date, an index, a session or a configuration; no module-level binding is a
  mutable container a cache could live in; and repeated calls are pure. There is
  no expression in either file through which another date's values could arrive.
- *Behaviourally*, by perturbation. Replacing one date's cross-section with
  arbitrary other numbers leaves every other date's output **bit-for-bit**
  identical; a date transformed alone equals the same date transformed inside a
  panel; appending later dates never disturbs an earlier one.

The behavioural property is only worth asserting if it can fail, and for a
stateless function it trivially cannot. So the file also contains
:func:`_pooled_transform` — the leaking implementation, the one that computes a
percentile and a mean over a flattened panel — and asserts that it *does* fail
the same property. That is what makes the passing assertion evidence rather than
decoration, and it is the shape the bug takes in practice: a panel handed to a
vectorized `numpy` call, no error, no symptom, a better backtest.

**2. Idempotence, at the precision it actually holds.** Winsorization
bit-for-bit; the three statistical steps to a tolerance derived from the
cross-section's conditioning rather than picked to pass. The pipeline as a whole
is *not* idempotent and is shown so by construction in ``test_transforms.py``.

**3. NaN is excluded, never imputed.** Stated as the strongest form available:
an absent name must leave every other name's output exactly as it would be if
that name had been deleted from the array outright.

**4. Geometry.** Within-group residual means are zero, beta residuals are
orthogonal to beta, and each step's projection inequality holds over the names
*that step* retains. That last qualification is load-bearing and was found by
this suite: the inequality does **not** chain end-to-end, because a later step
can drop the very name that absorbed an earlier step's energy. The counterexample
is pinned rather than smoothed over.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st

from backend.features import _stats, transforms
from backend.features.transforms import (
    beta_neutralize,
    cross_sectional_zscore,
    neutralize,
    transform_cross_section,
    winsorize,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

EPS = float(np.finfo(np.float64).eps)

_VALUE_SHAPES = ("gaussian", "heavy_tailed", "two_clusters", "ramp", "near_constant", "constant")
_ABSENCE_SHAPES = ("none", "scattered", "mostly", "all")
_SECTOR_SHAPES = ("balanced", "one_singleton", "one_giant", "all_distinct", "all_same")
_BETA_SHAPES = ("spread", "constant", "some_missing", "bimodal")


@dataclass(frozen=True)
class _CrossSection:
    """One date's cross-section, plus the structural choices that produced it."""

    values: npt.NDArray[np.float64]
    groups: npt.NDArray[np.int64]
    betas: npt.NDArray[np.float64]
    shape: tuple[str, str, str, str]

    @property
    def size(self) -> int:
        return int(self.values.size)


def _shaped_values(shape: str, size: int, seed: int, magnitude: float) -> npt.NDArray[np.float64]:
    """Build a value cross-section of a chosen *shape*, in the caller's magnitude."""
    rng = np.random.default_rng(seed)
    if shape == "gaussian":
        raw = rng.standard_normal(size)
    elif shape == "heavy_tailed":
        raw = rng.standard_normal(size)
        if size:
            raw[rng.integers(0, size, max(size // 8, 1))] *= 500.0
    elif shape == "two_clusters":
        raw = rng.standard_normal(size) * 0.1 + np.where(rng.random(size) < 0.5, -3.0, 3.0)
    elif shape == "ramp":
        raw = np.linspace(-1.0, 1.0, size) if size else np.empty(0)
    elif shape == "near_constant":
        raw = 1.0 + rng.standard_normal(size) * 1e-9
    else:  # constant: the zero-dispersion degeneracy, reached on purpose
        raw = np.full(size, 2.5)
    return np.asarray(raw * magnitude, dtype=np.float64)


def _apply_absence(
    values: npt.NDArray[np.float64], shape: str, seed: int
) -> npt.NDArray[np.float64]:
    """Mark some names "not available" in a chosen pattern."""
    rng = np.random.default_rng(seed + 977)
    out = values.copy()
    if shape == "scattered":
        out[rng.random(out.size) < 0.3] = np.nan
    elif shape == "mostly":
        out[rng.random(out.size) < 0.9] = np.nan
    elif shape == "all":
        out[:] = np.nan
    return out


def _shaped_sectors(shape: str, size: int, seed: int) -> npt.NDArray[np.int64]:
    """Build a sector layout, including the ones that trigger the NaN rules."""
    rng = np.random.default_rng(seed + 31)
    if size == 0:
        return np.empty(0, dtype=np.int64)
    if shape == "balanced":
        labels = rng.integers(0, max(size // 4, 1), size)
    elif shape == "one_singleton":
        labels = np.zeros(size, dtype=np.int64)
        labels[0] = 1  # exactly one name alone in its sector
    elif shape == "one_giant":
        labels = np.zeros(size, dtype=np.int64)
        labels[: max(size // 10, 1)] = np.arange(1, max(size // 10, 1) + 1)
    elif shape == "all_distinct":
        labels = np.arange(size)  # every sector a singleton
    else:
        labels = np.zeros(size, dtype=np.int64)
    return np.asarray(labels, dtype=np.int64)


def _shaped_betas(shape: str, size: int, seed: int) -> npt.NDArray[np.float64]:
    """Build a beta cross-section (dimensionless)."""
    rng = np.random.default_rng(seed + 613)
    if shape == "spread":
        betas = 1.0 + rng.standard_normal(size) * 0.35
    elif shape == "constant":
        betas = np.full(size, 1.05)
    elif shape == "some_missing":
        betas = 1.0 + rng.standard_normal(size) * 0.35
        betas[rng.random(size) < 0.35] = np.nan
    else:
        betas = np.where(rng.random(size) < 0.5, 0.6, 1.7)
    return np.asarray(betas, dtype=np.float64)


@st.composite
def _cross_sections(draw: st.DrawFn, min_size: int = 0, max_size: int = 40) -> _CrossSection:
    """Draw one date's cross-section, structure first."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    value_shape = draw(st.sampled_from(_VALUE_SHAPES))
    absence_shape = draw(st.sampled_from(_ABSENCE_SHAPES))
    sector_shape = draw(st.sampled_from(_SECTOR_SHAPES))
    beta_shape = draw(st.sampled_from(_BETA_SHAPES))
    magnitude = draw(st.sampled_from([1e-6, 1e-2, 1.0, 1e3, 1e6]))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    values = _apply_absence(_shaped_values(value_shape, size, seed, magnitude), absence_shape, seed)
    return _CrossSection(
        values=values,
        groups=_shaped_sectors(sector_shape, size, seed),
        betas=_shaped_betas(beta_shape, size, seed),
        shape=(value_shape, absence_shape, sector_shape, beta_shape),
    )


@st.composite
def _panels(draw: st.DrawFn) -> list[_CrossSection]:
    """Draw a multi-date panel. Dates are independent and may differ in width."""
    n_dates = draw(st.integers(min_value=2, max_value=5))
    return [draw(_cross_sections(min_size=1, max_size=25)) for _ in range(n_dates)]


def _transform(date: _CrossSection, *, with_betas: bool = True) -> npt.NDArray[np.float64]:
    """Run the full pipeline on one date."""
    return transform_cross_section(
        date.values, groups=date.groups, betas=date.betas if with_betas else None
    )


def _identical(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> bool:
    """Bit-for-bit equality, with ``NaN`` counting as equal to ``NaN``."""
    return bool(np.array_equal(left, right, equal_nan=True))


def _observed(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(values[~np.isnan(values)], dtype=np.float64)


def _scale(values: npt.NDArray[np.float64]) -> float:
    """Largest present magnitude, in the input's units; ``1.0`` if nothing is present."""
    observed = _observed(values)
    return float(np.max(np.abs(observed))) if observed.size else 1.0


def _conditioning(values: npt.NDArray[np.float64]) -> float:
    """``max|x| / sigma`` over the present values — the amplification factor.

    Every claim in this module that is true "up to floating point" degrades with
    this number, and for one reason: subtracting a mean that is large relative
    to the spread cancels leading digits, so a mean known to ``eps * max|x|``
    yields deviations known only to ``eps * kappa`` of their own size. Tolerances
    below are written as multiples of ``eps * kappa`` rather than as constants,
    so they stay tight on the well-conditioned cross-sections that dominate and
    do not silently pass on the ill-conditioned ones.

    ``ZERO_DISPERSION_RELATIVE_TOLERANCE`` caps this at ``1e12``: above that the
    transforms return ``NaN`` instead of a number.
    """
    observed = _observed(values)
    if observed.size < 2:
        return 1.0
    dispersion = float(np.std(observed, ddof=1))
    if dispersion == 0.0:
        return 1.0
    return max(float(np.max(np.abs(observed))) / dispersion, 1.0)


# ==========================================================================
# 1a. No cross-date leakage — behavioural
# ==========================================================================


def _pooled_transform(panel: list[_CrossSection]) -> list[npt.NDArray[np.float64]]:
    """The leaking implementation, written out so the property can be seen to bite.

    Every step here is individually reasonable-looking and the result has no
    symptom: it is a well-behaved standardized feature. It is also computed with
    a mean and a standard deviation taken across *every* date in the panel, so
    the value assigned to a security in 2004 depends on what happened in 2019.
    A backtest built on it simply improves.

    Nothing in :mod:`backend.features.transforms` can produce this — the
    functions never receive more than one date — and the tests below assert that
    this reference does fail the leakage property that the real transforms pass.
    """
    flattened = np.concatenate([date.values for date in panel])
    scores = cross_sectional_zscore(flattened)
    out: list[npt.NDArray[np.float64]] = []
    start = 0
    for date in panel:
        out.append(scores[start : start + date.size])
        start += date.size
    return out


class TestNoCrossDateLeakageBehaviourally:
    @given(panel=_panels(), target=st.integers(min_value=0, max_value=4))
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_perturbing_one_date_leaves_every_other_date_bit_for_bit_identical(
        self, panel: list[_CrossSection], target: int
    ) -> None:
        index = target % len(panel)
        baseline = [_transform(date) for date in panel]

        perturbed = list(panel)
        original = panel[index]
        perturbed[index] = _CrossSection(
            values=original.values * -1_000.0 + 7.0,
            groups=original.groups,
            betas=original.betas,
            shape=original.shape,
        )
        after = [_transform(date) for date in perturbed]

        for position, (before_values, after_values) in enumerate(zip(baseline, after, strict=True)):
            if position == index:
                continue
            assert _identical(before_values, after_values)

    @given(panel=_panels())
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_a_date_transformed_alone_matches_the_same_date_inside_a_panel(
        self, panel: list[_CrossSection]
    ) -> None:
        in_panel = [_transform(date) for date in panel]
        for date, result in zip(panel, in_panel, strict=True):
            assert _identical(_transform(date), result)

    @given(panel=_panels())
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_appending_later_dates_never_changes_an_earlier_dates_output(
        self, panel: list[_CrossSection]
    ) -> None:
        """Look-ahead in its most direct form: the future must not be readable."""
        history = [_transform(date) for date in panel[:1]]
        for extra in range(1, len(panel)):
            grown = [_transform(date) for date in panel[: extra + 1]]
            assert _identical(grown[0], history[0])

    @given(panel=_panels())
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_reordering_the_dates_does_not_change_any_dates_output(
        self, panel: list[_CrossSection]
    ) -> None:
        forwards = [_transform(date) for date in panel]
        backwards = [_transform(date) for date in reversed(panel)]
        for result, reversed_result in zip(forwards, reversed(backwards), strict=True):
            assert _identical(result, reversed_result)

    def test_a_pooled_implementation_fails_the_very_same_property(self) -> None:
        """Non-vacuity, permanently in the record.

        A property no implementation can violate proves nothing. This runs the
        leaking reference through the identical perturbation and shows date 0's
        output moving because date 1 changed — which is exactly what the passing
        assertions above rule out for the real transforms.
        """
        rng = np.random.default_rng(97)
        first = _CrossSection(
            values=rng.standard_normal(12),
            groups=np.zeros(12, dtype=np.int64),
            betas=np.ones(12),
            shape=("gaussian", "none", "all_same", "constant"),
        )
        second = _CrossSection(
            values=rng.standard_normal(12),
            groups=np.zeros(12, dtype=np.int64),
            betas=np.ones(12),
            shape=("gaussian", "none", "all_same", "constant"),
        )
        louder = _CrossSection(
            values=second.values * 1_000.0,
            groups=second.groups,
            betas=second.betas,
            shape=second.shape,
        )

        leaked_before = _pooled_transform([first, second])[0]
        leaked_after = _pooled_transform([first, louder])[0]
        assert not _identical(leaked_before, leaked_after)
        # ...and the movement is large, not a rounding difference.
        assert float(np.max(np.abs(leaked_after - leaked_before))) > 0.1

        honest_before = cross_sectional_zscore(first.values)
        honest_after = cross_sectional_zscore(first.values)
        assert _identical(honest_before, honest_after)


# ==========================================================================
# 1b. No cross-date leakage — structural
# ==========================================================================

_ALLOWED_IMPORTS = frozenset(
    {"__future__", "typing", "math", "numpy", "numpy.typing", "backend.features._stats"}
)
"""Everything the transform modules are permitted to reach.

Not a style rule. A transform that can import :mod:`backend.db`, ``pandas``,
``datetime`` or the filesystem can be handed — or can fetch — a second date, and
the cross-sectional contract stops being enforceable by reading the code.
"""

_FORBIDDEN_PARAMETER_WORDS = (
    "date",
    "time",
    "stamp",
    "instant",
    "as_of",
    "asof",
    "session",
    "config",
    "registry",
    "panel",
    "history",
    "window",
    "lookback",
    "universe",
    "path",
)


def _imported_modules(source: Path) -> set[str]:
    """Every module named in an ``import`` statement anywhere in a file."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{source.name} uses a relative import, which hides its target"
            names.add(node.module or "")
    return names


_PUBLIC_TRANSFORMS: dict[str, Callable[..., npt.NDArray[np.float64]]] = {
    "winsorize": winsorize,
    "cross_sectional_zscore": cross_sectional_zscore,
    "neutralize": neutralize,
    "beta_neutralize": beta_neutralize,
    "transform_cross_section": transform_cross_section,
}


class TestNoCrossDateLeakageStructurally:
    """The defence that does not depend on anyone remembering to run a test.

    A behavioural leakage test can only catch a leak through a channel the test
    thought to perturb. These assertions instead pin the absence of any channel:
    nothing is imported that could reach a second date, nothing is accepted that
    could name one, and nothing is retained between calls that could remember
    one.
    """

    @pytest.mark.parametrize("module", [transforms, _stats], ids=["transforms", "_stats"])
    def test_the_module_imports_nothing_that_could_reach_another_date(self, module: object) -> None:
        source = Path(str(getattr(module, "__file__", "")))
        assert source.is_file()
        imported = _imported_modules(source)
        assert imported, "the scan found no imports at all, which means it is not working"
        assert imported <= _ALLOWED_IMPORTS, (
            f"unexpected imports: {sorted(imported - _ALLOWED_IMPORTS)}"
        )

    def test_the_import_allowlist_would_notice_a_new_dependency(self) -> None:
        """The scanner is checked against a file that does import the world."""
        this_file = _imported_modules(Path(__file__))
        assert "backend.features.transforms" in this_file
        assert not this_file <= _ALLOWED_IMPORTS

    @pytest.mark.parametrize("name", sorted(_PUBLIC_TRANSFORMS))
    def test_no_transform_accepts_a_date_a_session_or_a_configuration(self, name: str) -> None:
        parameters = inspect.signature(_PUBLIC_TRANSFORMS[name]).parameters
        assert parameters, f"{name} takes no arguments"
        for parameter_name, parameter in parameters.items():
            lowered = parameter_name.lower()
            assert not any(word in lowered for word in _FORBIDDEN_PARAMETER_WORDS), (
                f"{name}({parameter_name}) names something that is not a cross-section"
            )
            assert parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }, f"{name} accepts **kwargs, so its inputs are not enumerable by reading it"

    def test_the_declared_parameters_are_exactly_the_cross_sectional_ones(self) -> None:
        expected = {
            "winsorize": ["values", "lower_pct", "upper_pct"],
            "cross_sectional_zscore": ["values"],
            "neutralize": ["values", "groups"],
            "beta_neutralize": ["values", "betas"],
            "transform_cross_section": ["values", "groups", "betas", "lower_pct", "upper_pct"],
        }
        for name, parameter_names in expected.items():
            assert list(inspect.signature(_PUBLIC_TRANSFORMS[name]).parameters) == parameter_names

    @pytest.mark.parametrize("module", [transforms, _stats], ids=["transforms", "_stats"])
    def test_the_module_holds_no_mutable_state_a_cache_could_live_in(self, module: object) -> None:
        for name, value in vars(module).items():
            if name.startswith("__"):
                continue
            assert not isinstance(value, list | dict | set | bytearray | np.ndarray), (
                f"{name} is a mutable module-level container; one date's values could "
                f"survive in it into the next call"
            )

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=200, deadline=None)
    def test_transforms_are_pure_and_do_not_mutate_or_alias_their_inputs(
        self, date: _CrossSection
    ) -> None:
        values_before = date.values.copy()
        groups_before = date.groups.copy()
        betas_before = date.betas.copy()

        first = _transform(date)
        # An unrelated call in between: if anything were memoized on module
        # state, this is where the wrong date's statistics would be picked up.
        _transform(
            _CrossSection(
                values=date.values * 13.0 + 4.0,
                groups=date.groups,
                betas=date.betas,
                shape=date.shape,
            )
        )
        second = _transform(date)

        assert _identical(first, second)
        assert _identical(date.values, values_before)
        assert np.array_equal(date.groups, groups_before)
        assert _identical(date.betas, betas_before)
        assert not np.shares_memory(first, date.values)


# ==========================================================================
# 2. Idempotence
# ==========================================================================


class TestIdempotence:
    @given(date=_cross_sections(), lower=st.sampled_from([0.0, 1.0, 5.0, 25.0]))
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_winsorize_is_exactly_idempotent(self, date: _CrossSection, lower: float) -> None:
        once = winsorize(date.values, lower_pct=lower, upper_pct=100.0 - lower)
        twice = winsorize(once, lower_pct=lower, upper_pct=100.0 - lower)
        assert _identical(once, twice)

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_z_scoring_is_idempotent_to_its_conditioning(self, date: _CrossSection) -> None:
        once = cross_sectional_zscore(date.values)
        present = ~np.isnan(once)
        assume(bool(present.any()))
        twice = cross_sectional_zscore(once)
        assert np.array_equal(np.isnan(once), np.isnan(twice))

        observed = _observed(date.values)
        kappa = float(np.max(np.abs(observed))) / float(np.std(observed, ddof=1))
        bound = 32.0 * EPS * max(kappa, float(observed.size))
        assert float(np.max(np.abs(twice[present] - once[present]))) <= bound

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_neutralize_is_idempotent_and_its_nan_pattern_is_a_fixed_point(
        self, date: _CrossSection
    ) -> None:
        once = neutralize(date.values, groups=date.groups)
        twice = neutralize(once, groups=date.groups)
        assert np.array_equal(np.isnan(once), np.isnan(twice))

        present = ~np.isnan(once)
        assume(bool(present.any()))
        scale = _scale(date.values)
        bound = 32.0 * EPS * scale * max(date.size, 1)
        assert float(np.max(np.abs(twice[present] - once[present]))) <= bound

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_beta_neutralize_is_idempotent_and_its_nan_pattern_is_a_fixed_point(
        self, date: _CrossSection
    ) -> None:
        once = beta_neutralize(date.values, betas=date.betas)
        twice = beta_neutralize(once, betas=date.betas)
        assert np.array_equal(np.isnan(once), np.isnan(twice))

        present = ~np.isnan(once)
        assume(bool(present.any()))
        beta_kappa = _conditioning(date.betas)
        scale = _scale(date.values)
        bound = 64.0 * EPS * scale * max(date.size, 1) * max(beta_kappa, 1.0)
        assert float(np.max(np.abs(twice[present] - once[present]))) <= bound


# ==========================================================================
# 3. NaN policy
# ==========================================================================


class TestMissingValuePolicy:
    @given(date=_cross_sections(min_size=1))
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_an_absent_name_could_have_been_deleted_from_the_array(
        self, date: _CrossSection
    ) -> None:
        """The strongest statement of "excluded from every statistic".

        If a ``NaN`` were imputed — with the mean, with zero, with the sector
        average — it would join the sample and move the percentile, the mean,
        the standard deviation and the regression fit, and every *other* name's
        output would differ from the deleted-name answer. This holds it to
        exactly the deleted-name answer.
        """
        present = ~np.isnan(date.values)
        assume(bool(present.any()))
        assume(bool((~present).any()))

        kept = _CrossSection(
            values=date.values[present],
            groups=date.groups[present],
            betas=date.betas[present],
            shape=date.shape,
        )
        assert _identical(winsorize(date.values)[present], winsorize(kept.values))
        assert _identical(
            cross_sectional_zscore(date.values)[present], cross_sectional_zscore(kept.values)
        )
        assert _identical(
            neutralize(date.values, groups=date.groups)[present],
            neutralize(kept.values, groups=kept.groups),
        )
        assert _identical(
            beta_neutralize(date.values, betas=date.betas)[present],
            beta_neutralize(kept.values, betas=kept.betas),
        )

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_an_absent_input_can_never_acquire_a_value(self, date: _CrossSection) -> None:
        absent = np.isnan(date.values)
        for result in (
            winsorize(date.values),
            cross_sectional_zscore(date.values),
            neutralize(date.values, groups=date.groups),
            beta_neutralize(date.values, betas=date.betas),
            _transform(date),
        ):
            assert bool(np.all(np.isnan(result[absent])))

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_an_absent_beta_removes_its_security_from_the_output(self, date: _CrossSection) -> None:
        absent_beta = np.isnan(date.betas)
        residual = beta_neutralize(date.values, betas=date.betas)
        assert bool(np.all(np.isnan(residual[absent_beta])))

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_no_transform_ever_returns_an_infinity(self, date: _CrossSection) -> None:
        for result in (
            winsorize(date.values),
            cross_sectional_zscore(date.values),
            neutralize(date.values, groups=date.groups),
            beta_neutralize(date.values, betas=date.betas),
            _transform(date),
            _transform(date, with_betas=False),
        ):
            assert not bool(np.any(np.isinf(result)))

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_shape_dtype_and_element_order_are_preserved(self, date: _CrossSection) -> None:
        for result in (
            winsorize(date.values),
            cross_sectional_zscore(date.values),
            neutralize(date.values, groups=date.groups),
            beta_neutralize(date.values, betas=date.betas),
            _transform(date),
        ):
            assert result.shape == date.values.shape
            assert result.dtype == np.float64

    @given(date=_cross_sections(min_size=2), seed=st.integers(min_value=0, max_value=2**32 - 1))
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_every_transform_commutes_with_reordering_the_securities(
        self, date: _CrossSection, seed: int
    ) -> None:
        order = np.random.default_rng(seed).permutation(date.size)
        shuffled = _CrossSection(
            values=date.values[order],
            groups=date.groups[order],
            betas=date.betas[order],
            shape=date.shape,
        )
        # Winsorization is bit-for-bit equivariant: its cut points come from a
        # sort and an index, which do not depend on the input order at all.
        assert _identical(winsorize(date.values)[order], winsorize(shuffled.values))

        # The statistical steps agree to rounding rather than to the bit.
        # `numpy`'s pairwise summation adds the same values in a different order
        # once the securities are permuted, so the mean and the group totals can
        # differ in the last ulp. That is a fact about summation, not about the
        # transforms treating the caller's order as meaningful.
        scale = _scale(date.values)
        slack = 64.0 * EPS * _conditioning(date.values) * max(date.size, 1)
        for baseline, permuted in (
            (cross_sectional_zscore(date.values), cross_sectional_zscore(shuffled.values)),
            (
                neutralize(date.values, groups=date.groups),
                neutralize(shuffled.values, groups=shuffled.groups),
            ),
            (
                beta_neutralize(date.values, betas=date.betas),
                beta_neutralize(shuffled.values, betas=shuffled.betas),
            ),
        ):
            reordered = baseline[order]
            assert np.array_equal(np.isnan(reordered), np.isnan(permuted))
            present = ~np.isnan(reordered)
            if present.any():
                assert reordered[present] == pytest.approx(
                    permuted[present], rel=slack, abs=slack * max(scale, 1e-300)
                )


# ==========================================================================
# 4. Geometry: what neutralization actually removes
# ==========================================================================


class TestNeutralizationGeometry:
    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_every_usable_group_is_left_with_a_zero_mean(self, date: _CrossSection) -> None:
        residual = neutralize(date.values, groups=date.groups)
        assume(_observed(date.values).size > 0)
        scale = _scale(date.values)
        for label in np.unique(date.groups):
            members = residual[date.groups == label]
            present = members[~np.isnan(members)]
            if present.size >= 2:
                assert abs(float(np.mean(present))) <= 1e-9 * max(scale, 1e-300)

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_the_beta_residual_is_orthogonal_to_beta(self, date: _CrossSection) -> None:
        residual = beta_neutralize(date.values, betas=date.betas)
        present = ~np.isnan(residual)
        assume(bool(present.any()))
        scale = _scale(date.values)
        beta_scale = _scale(date.betas)

        assert abs(float(np.sum(residual[present]))) <= 1e-9 * scale * date.size
        centered = date.betas[present] - float(np.mean(date.betas[present]))
        assert abs(float(np.dot(residual[present], centered))) <= (
            1e-9 * scale * beta_scale * date.size
        )

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_neutralization_never_grows_the_energy_of_the_names_it_keeps(
        self, date: _CrossSection
    ) -> None:
        """The shrinkage claim in the form that survives names being dropped.

        Demeaning within groups is an orthogonal projection, and it is applied
        group by group, so on the names it retains the residual's sum of squares
        cannot exceed the input's. The familiar "the standard deviation falls
        below 1" is the same statement only when nothing is dropped — see
        ``test_transforms.py::TestUnits``.
        """
        residual = neutralize(date.values, groups=date.groups)
        retained = ~np.isnan(residual)
        assume(bool(retained.any()))
        energy_in = float(np.sum(date.values[retained] ** 2))
        energy_out = float(np.sum(residual[retained] ** 2))
        assert energy_out <= energy_in * (1.0 + 1e-9) + 1e-300

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_beta_neutralization_never_grows_the_energy_of_the_names_it_keeps(
        self, date: _CrossSection
    ) -> None:
        residual = beta_neutralize(date.values, betas=date.betas)
        retained = ~np.isnan(residual)
        assume(bool(retained.any()))
        energy_in = float(np.sum(date.values[retained] ** 2))
        energy_out = float(np.sum(residual[retained] ** 2))
        assert energy_out <= energy_in * (1.0 + 1e-9) + 1e-300

    @given(date=_cross_sections())
    @hypothesis_settings(max_examples=400, deadline=None)
    def test_the_pipeline_shrinks_at_every_step_relative_to_that_steps_own_input(
        self, date: _CrossSection
    ) -> None:
        """Each step shrinks. The chain does not, and the two are not the same claim.

        Every step is an orthogonal projection on the names *it* retains, so
        each individually cannot grow their energy. The composition can, because
        a later step may drop the very name that absorbed an earlier step's
        energy — see
        ``test_the_shrinkage_inequality_does_not_chain_across_a_dropped_name``,
        which exhibits it. So this asserts the true, per-step form.
        """
        winsorized = winsorize(date.values)
        scores = cross_sectional_zscore(winsorized)
        sector_neutral = neutralize(scores, groups=date.groups)
        final = beta_neutralize(sector_neutral, betas=date.betas)

        for step_input, step_output in (
            (scores, sector_neutral),
            (sector_neutral, final),
        ):
            retained = ~np.isnan(step_output)
            if not retained.any():
                continue
            energy_in = float(np.sum(step_input[retained] ** 2))
            energy_out = float(np.sum(step_output[retained] ** 2))
            assert energy_out <= energy_in * (1.0 + 1e-9) + 1e-300

    def test_the_shrinkage_inequality_does_not_chain_across_a_dropped_name(self) -> None:
        """The convenient end-to-end claim, exhibited as false.

        One name carries almost all the cross-sectional dispersion and shares a
        sector with three others. Sector demeaning parks most of that energy in
        *its* residual, which is fine — the inequality holds over the four of
        them together. Then its beta is missing, so beta neutralization drops
        it, and the three survivors are left holding more energy than their own
        z-scores had. Nothing is wrong; the inequality is simply per step and
        over that step's retained set, and stating it end-to-end would be a
        claim the pipeline does not make.
        """
        values = np.array(
            [1.7e-4, 8.2e-7, 3.3e-7, -1.3e-6, 9.1e-7, 4.5e-7, -5.4e-7, 5.8e-7, 3.6e-7]
        )
        sectors = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1], dtype=np.int64)
        betas = np.array(
            [np.nan, np.nan, 0.775, np.nan, 1.009, 1.472, 0.376, 0.943, 1.688], dtype=np.float64
        )

        scores = cross_sectional_zscore(winsorize(values))
        out = transform_cross_section(values, groups=sectors, betas=betas)
        retained = ~np.isnan(out)

        assert retained.sum() < values.size  # names really were dropped
        assert float(np.sum(out[retained] ** 2)) > float(np.sum(scores[retained] ** 2))


# ==========================================================================
# 5. Units, as equivariance
# ==========================================================================


class TestUnitEquivariance:
    # A change of units is a positive rescale plus a shift, and the shift is
    # drawn *relative to the cross-section's own magnitude*. An absolute offset
    # of -11.5 on a feature whose values are all near 1e-6 is not a change of
    # units; it is catastrophic cancellation, and it destroys the information in
    # the input before any transform sees it. Testing equivariance there would
    # measure float64, not this module.
    _FACTORS = st.sampled_from([2.0, 8.0, 1024.0, 0.25])
    _OFFSET_RATIOS = st.sampled_from([0.0, 3.0, -11.5])

    @given(date=_cross_sections(), factor=_FACTORS, offset_ratio=_OFFSET_RATIOS)
    @hypothesis_settings(
        max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_winsorize_carries_the_inputs_units_through_unchanged(
        self, date: _CrossSection, factor: float, offset_ratio: float
    ) -> None:
        assume(_observed(date.values).size > 0)
        offset = offset_ratio * _scale(date.values)

        rescaled = winsorize(date.values * factor + offset)
        expected = winsorize(date.values) * factor + offset
        present = ~np.isnan(rescaled)
        assert rescaled[present] == pytest.approx(expected[present], rel=1e-12, abs=1e-300)

    @given(date=_cross_sections(), factor=_FACTORS, offset_ratio=_OFFSET_RATIOS)
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_z_scoring_is_invariant_under_an_affine_change_of_units(
        self, date: _CrossSection, factor: float, offset_ratio: float
    ) -> None:
        baseline = cross_sectional_zscore(date.values)
        present = ~np.isnan(baseline)
        assume(bool(present.any()))
        offset = offset_ratio * _scale(date.values)

        rescaled = cross_sectional_zscore(date.values * factor + offset)
        slack = 256.0 * EPS * _conditioning(date.values) * (1.0 + abs(offset_ratio))
        assert rescaled[present] == pytest.approx(baseline[present], rel=slack, abs=slack)

    @given(date=_cross_sections(), factor=_FACTORS, offset_ratio=_OFFSET_RATIOS)
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_neutralization_returns_residuals_in_the_inputs_units(
        self, date: _CrossSection, factor: float, offset_ratio: float
    ) -> None:
        baseline = neutralize(date.values, groups=date.groups)
        present = ~np.isnan(baseline)
        assume(bool(present.any()))
        scale = _scale(date.values)
        offset = offset_ratio * scale

        rescaled = neutralize(date.values * factor + offset, groups=date.groups)
        assert rescaled[present] == pytest.approx(
            baseline[present] * factor, rel=1e-9, abs=1e-9 * max(scale, 1e-300) * factor
        )

    @given(date=_cross_sections(), factor=_FACTORS, offset_ratio=_OFFSET_RATIOS)
    @hypothesis_settings(max_examples=300, deadline=None)
    def test_the_beta_residual_is_in_the_values_units_and_ignores_the_betas_units(
        self, date: _CrossSection, factor: float, offset_ratio: float
    ) -> None:
        baseline = beta_neutralize(date.values, betas=date.betas)
        present = ~np.isnan(baseline)
        assume(bool(present.any()))
        scale = _scale(date.values)

        # Rescaling the feature rescales the residual: it is in the feature's units.
        in_values = beta_neutralize(date.values * factor + offset_ratio * scale, betas=date.betas)
        assert in_values[present] == pytest.approx(
            baseline[present] * factor, rel=1e-8, abs=1e-8 * max(scale, 1e-300) * factor
        )

        # Rescaling beta does not: only its spread enters the fit, and the
        # slope absorbs the change. Beta is dimensionless in both directions.
        beta_scale = _scale(date.betas)
        in_betas = beta_neutralize(
            date.values, betas=date.betas * factor + offset_ratio * beta_scale
        )
        assert in_betas[present] == pytest.approx(
            baseline[present], rel=1e-8, abs=1e-8 * max(scale, 1e-300)
        )
