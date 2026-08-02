"""Tests for the drift reporting layer (P12.2).

The detector's job is to produce a number. This layer's job is to make that
number readable without making it look like more than it is, and the tests here
are organised around the three ways that goes wrong:

1. **The ladder acquires the authority of a measurement.** The 0.10 / 0.25 cut
   points are credit-scorecard folklore. So the tests assert that the basis
   travels in the payload, that the thresholds are replaceable, and — the part
   that matters — that a ladder which *cannot classify* (rungs out of order,
   non-positive, non-finite) is refused rather than silently marking everything
   the same band.

2. **A refusal becomes a zero.** A feature the detector would not measure is
   recorded as unmeasurable, and the tests assert there is no attribute and no
   serialised key on that record a template could render as a PSI, and that the
   report says it is incomplete.

3. **A number outlives its reference (I2).** Every finding is asserted to carry
   the reference identifier, the reference fingerprint, and the measuring run's
   reproducibility stamp — through ``to_dict`` and into JSON, since that is the
   form the dashboard receives.

Detection itself is asserted here too, in both directions, because the band is
what an operator actually reads: a feature drawn from its own reference must
band ``STABLE`` and a shifted one must band ``MAJOR`` in the same report.

All fixtures are synthetic and named so (I3 — see
``backend/tests/monitoring/fixtures.py``).
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from backend.monitoring.drift import (
    CONVENTIONAL_BANDS_BASIS,
    DEFAULT_MAJOR_THRESHOLD,
    DEFAULT_MODERATE_THRESHOLD,
    DriftBand,
    DriftBands,
    DriftReport,
    FeatureDrift,
    UnmeasurableFeature,
    drift_report,
    measure_feature_drift,
)
from backend.monitoring.errors import (
    DriftBandError,
    InsufficientSampleError,
    MonitoringInputError,
)
from backend.tests.monitoring.fixtures import (
    OBSERVED_SIZE,
    equal_decile_reference,
    fixture_stamp,
    gaussian_reference,
    gaussian_sample,
    with_absent,
)

if TYPE_CHECKING:
    from backend.monitoring.psi import ReferenceDistribution

AS_OF: Final = dt.date(2021, 6, 30)
"""A fixture date. No cross-section in this repository belongs to any date (B1)."""


def _reference(name: str) -> ReferenceDistribution:
    return gaussian_reference(feature=f"FIXTURE_{name}", reference_id=f"FIXTURE_train_{name}")


# ---------------------------------------------------------------------------
# 1. The ladder is convention, and says so
# ---------------------------------------------------------------------------


def test_the_default_ladder_is_the_conventional_one_and_states_its_basis() -> None:
    bands = DriftBands()
    assert bands.moderate == DEFAULT_MODERATE_THRESHOLD == 0.10
    assert bands.major == DEFAULT_MAJOR_THRESHOLD == 0.25
    assert bands.basis == CONVENTIONAL_BANDS_BASIS
    payload = bands.to_dict()
    assert "convention" in str(payload["basis"])
    assert "Siddiqi" in str(payload["basis"])
    assert "not derived" in str(payload["basis"])
    # The bin-count caveat travels too: the ladder was calibrated on deciles.
    assert "ten bins" in str(payload["basis"])
    json.dumps(payload)


def test_the_thresholds_are_replaceable() -> None:
    bands = DriftBands(moderate=0.02, major=0.05, basis="derived for 4 bins on a 500-name universe")
    assert bands.classify(0.03) is DriftBand.MODERATE
    assert bands.classify(0.06) is DriftBand.MAJOR
    assert bands.to_dict()["basis"] == "derived for 4 bins on a 500-name universe"


@pytest.mark.parametrize(
    ("moderate", "major"),
    [
        (0.25, 0.10),  # rungs inverted
        (0.10, 0.10),  # rungs equal: the middle band can never be reported
        (0.0, 0.25),  # fires on an identical sample
        (-0.1, 0.25),
        (0.10, 0.0),
        (math.inf, math.inf),
        (0.10, math.inf),
        (math.nan, 0.25),
        (0.10, math.nan),
    ],
)
def test_a_ladder_that_cannot_classify_is_refused(moderate: float, major: float) -> None:
    with pytest.raises(DriftBandError):
        DriftBands(moderate=moderate, major=major)


def test_a_ladder_without_a_stated_basis_is_refused() -> None:
    with pytest.raises(DriftBandError, match="basis is blank"):
        DriftBands(basis="   ")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, DriftBand.STABLE),
        (0.0999999, DriftBand.STABLE),
        (0.10, DriftBand.MODERATE),  # inclusive on the upper band
        (0.2499999, DriftBand.MODERATE),
        (0.25, DriftBand.MAJOR),
        (12.4, DriftBand.MAJOR),
    ],
)
def test_the_boundaries_are_inclusive_on_the_upper_band(value: float, expected: DriftBand) -> None:
    assert DriftBands().classify(value) is expected


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.01])
def test_a_value_that_is_not_a_divergence_is_refused_rather_than_called_stable(
    bad: float,
) -> None:
    # NaN compares false against every threshold, so the naive implementation
    # returns STABLE for it — a blank cell reading as "nothing to report".
    with pytest.raises(DriftBandError):
        DriftBands().classify(bad)


def test_a_band_is_a_string_in_the_payload_without_a_second_mapping() -> None:
    assert str(DriftBand.MAJOR) == "major"
    assert json.dumps({"band": DriftBand.MODERATE}) == '{"band": "moderate"}'


# ---------------------------------------------------------------------------
# 2. Detection, in both directions, as the operator reads it
# ---------------------------------------------------------------------------


def test_a_stable_feature_bands_stable_and_a_shifted_one_bands_major() -> None:
    reference = _reference("momentum")
    stable = measure_feature_drift(reference=reference, observed=gaussian_sample(seed=41))
    shifted = measure_feature_drift(
        reference=reference, observed=gaussian_sample(seed=41, shift=1.0)
    )
    assert stable.band is DriftBand.STABLE
    assert stable.distribution_band is DriftBand.STABLE
    assert stable.availability_band is DriftBand.STABLE
    # Of the order of this sample's own noise floor, not merely below the rung.
    assert stable.psi.value < 5.0 * stable.psi.null_expected_value
    assert shifted.band is DriftBand.MAJOR
    assert shifted.exceeds_sampling_noise is True
    assert shifted.psi.value > 40.0 * stable.psi.value


def test_a_value_within_the_sampling_noise_is_marked_as_such() -> None:
    # `exceeds_sampling_noise` is the derived reading the conventional ladder
    # cannot give: whether the number is bigger than what a distribution that
    # did not move would have produced anyway.
    reference = equal_decile_reference()
    identical = measure_feature_drift(
        reference=reference, observed=np.arange(20_000, dtype=np.float64)
    )
    assert identical.psi.value == 0.0
    assert identical.psi.null_expected_value > 0.0
    assert identical.exceeds_sampling_noise is False
    assert identical.band is DriftBand.STABLE


def test_an_availability_collapse_bands_major_while_the_shape_bands_stable() -> None:
    reference = _reference("earnings_yield")
    observed = with_absent(gaussian_sample(seed=42), absent=800)
    finding = measure_feature_drift(reference=reference, observed=observed)
    assert finding.distribution_band is DriftBand.STABLE
    assert finding.availability_band is DriftBand.MAJOR
    # The worse of the two is what the report shows, so a source that broke
    # cannot be hidden by a shape that did not move.
    assert finding.band is DriftBand.MAJOR


def test_a_floor_driven_value_is_flagged_as_such() -> None:
    reference = equal_decile_reference()
    finding = measure_feature_drift(reference=reference, observed=np.full(OBSERVED_SIZE, 10_500.0))
    assert finding.band is DriftBand.MAJOR
    assert finding.floor_driven is True
    quiet = measure_feature_drift(
        reference=reference, observed=np.arange(OBSERVED_SIZE, dtype=np.float64) * 10.0
    )
    assert quiet.floor_driven is False


def test_measure_feature_drift_does_not_swallow_a_refusal() -> None:
    # A caller measuring one feature wants the refusal, not a row about it.
    with pytest.raises(InsufficientSampleError):
        measure_feature_drift(
            reference=_reference("size"), observed=gaussian_sample(seed=43, size=100)
        )


def test_the_finding_carries_the_ladder_it_was_classified_on() -> None:
    reference = _reference("accruals")
    strict = DriftBands(moderate=0.001, major=0.002, basis="fixture ladder, deliberately strict")
    finding = measure_feature_drift(
        reference=reference, observed=gaussian_sample(seed=44), bands=strict
    )
    # The same measurement bands differently on a different ladder, which is
    # exactly why the ladder is stored on the finding rather than looked up.
    assert finding.band is DriftBand.MAJOR
    assert measure_feature_drift(reference=reference, observed=gaussian_sample(seed=44)).band is (
        DriftBand.STABLE
    )
    assert finding.to_dict()["bands"] == strict.to_dict()


# ---------------------------------------------------------------------------
# 3. The report names its references and its run (I2)
# ---------------------------------------------------------------------------


def test_a_report_carries_the_stamp_and_every_reference_identity() -> None:
    stamp = fixture_stamp(seed=17)
    references = [_reference("momentum"), _reference("book_to_price")]
    report = drift_report(
        as_of=AS_OF,
        stamp=stamp,
        observations=[
            (references[0], gaussian_sample(seed=51)),
            (references[1], gaussian_sample(seed=52, shift=1.5)),
        ],
    )
    payload = report.to_dict()
    text = json.dumps(payload)

    assert payload["as_of"] == "2021-06-30"
    assert payload["git_reference"] == stamp.git_reference
    assert payload["data_version"] == stamp.data_version
    assert payload["config_hash"] == stamp.config_hash
    assert payload["seed"] == 17
    assert payload["reproducible"] is True
    for reference in references:
        assert reference.reference_id in text
        assert reference.fingerprint in text
    for finding in report.measured:
        assert finding.reference_id == finding.psi.reference.reference_id
        assert finding.reference_fingerprint == finding.psi.reference.fingerprint
    assert payload["complete"] is True
    assert payload["worst_band"] == "major"
    assert report.by_band(DriftBand.MAJOR)[0].feature == "FIXTURE_book_to_price"
    assert report.by_band(DriftBand.STABLE)[0].feature == "FIXTURE_momentum"


def test_a_report_cannot_be_built_without_a_stamp() -> None:
    with pytest.raises(DriftBandError, match="ReproducibilityStamp"):
        DriftReport(
            as_of=AS_OF,
            stamp="0" * 40,  # type: ignore[arg-type]
            bands=DriftBands(),
            measured=(),
            unmeasurable=(),
        )


def test_a_report_belongs_to_a_date_not_a_timestamp() -> None:
    with pytest.raises(DriftBandError, match=r"datetime\.date"):
        DriftReport(
            as_of=dt.datetime(2021, 6, 30, 12, 0, tzinfo=dt.UTC),
            stamp=fixture_stamp(),
            bands=DriftBands(),
            measured=(),
            unmeasurable=(),
        )


def test_one_feature_cannot_appear_twice_in_one_report() -> None:
    reference = _reference("momentum")
    with pytest.raises(DriftBandError, match="appear more than once"):
        drift_report(
            as_of=AS_OF,
            stamp=fixture_stamp(),
            observations=[
                (reference, gaussian_sample(seed=53)),
                (reference, gaussian_sample(seed=54)),
            ],
        )


# ---------------------------------------------------------------------------
# 4. A refusal is reported as a refusal, never as zero
# ---------------------------------------------------------------------------


def test_an_unmeasurable_feature_is_recorded_without_a_number() -> None:
    thin = _reference("short_interest")
    report = drift_report(
        as_of=AS_OF,
        stamp=fixture_stamp(),
        observations=[
            (_reference("momentum"), gaussian_sample(seed=61)),
            (thin, gaussian_sample(seed=62, size=100)),
        ],
    )
    assert len(report.measured) == 1
    assert len(report.unmeasurable) == 1
    assert report.complete is False

    entry = report.unmeasurable[0]
    assert entry.feature == thin.feature
    assert entry.reference_id == thin.reference_id
    assert entry.reference_fingerprint == thin.fingerprint
    assert entry.quantity == "present (non-NaN) observations"
    assert entry.n_observed == 100
    assert entry.minimum_required == 360
    assert "noise presented as a signal" in entry.reason

    # There is no attribute on the record a template could render as a PSI.
    assert not hasattr(entry, "value")
    assert not hasattr(entry, "psi")
    assert not hasattr(entry, "band")
    payload = entry.to_dict()
    assert payload["measured"] is False
    assert "value" not in payload
    assert "band" not in payload
    assert "psi" not in payload
    # No scalar on the record is a stand-in PSI: the only numbers are the two
    # counts that explain the refusal, and the nested availability shift, which
    # is a measurement in its own right and labelled as one.
    scalars = {key: item for key, item in payload.items() if not isinstance(item, dict)}
    assert set(scalars) == {
        "measured",
        "feature",
        "reference_id",
        "reference_fingerprint",
        "reason",
        "quantity",
        "n_observed",
        "minimum_required",
    }
    numbers = [
        item
        for item in scalars.values()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    assert numbers == [100, 360]
    json.dumps(payload)


def test_the_report_says_it_is_incomplete_next_to_its_worst_band() -> None:
    report = drift_report(
        as_of=AS_OF,
        stamp=fixture_stamp(),
        observations=[(_reference("roic"), gaussian_sample(seed=63, size=50))],
    )
    payload = report.to_dict()
    # "stable" here means "nothing was measured", and the payload must never
    # carry the one without the other.
    assert payload["worst_band"] == "stable"
    assert payload["complete"] is False
    assert payload["n_measured"] == 0
    assert payload["n_unmeasurable"] == 1
    assert report.measured == ()


def test_a_refusal_carries_the_availability_it_could_still_measure() -> None:
    reference = _reference("asset_growth")
    observed = np.full(1_000, np.nan)
    observed[:100] = gaussian_sample(seed=64, size=100)
    report = drift_report(as_of=AS_OF, stamp=fixture_stamp(), observations=[(reference, observed)])
    entry = report.unmeasurable[0]
    assert entry.availability is not None
    assert entry.availability.observed_absent_fraction == pytest.approx(0.9)
    # The upstream break is the finding, not a consolation for its absence.
    assert entry.availability.value > 1.0
    payload = entry.to_dict()
    assert isinstance(payload["availability"], dict)
    assert payload["availability"]["rate_change"] == pytest.approx(0.9)


def test_a_wholly_absent_cross_section_is_a_refusal_with_no_value() -> None:
    reference = _reference("amihud")
    report = drift_report(
        as_of=AS_OF,
        stamp=fixture_stamp(),
        observations=[(reference, np.full(OBSERVED_SIZE, np.nan))],
    )
    assert report.complete is False
    assert report.measured == ()
    entry = report.unmeasurable[0]
    assert entry.n_observed == 0
    assert entry.availability is not None
    assert entry.availability.observed_absent_fraction == 1.0


def test_a_malformed_array_propagates_rather_than_becoming_thirty_rows() -> None:
    # A two-dimensional panel is a fact about the calling code, wrong on every
    # date or none.
    with pytest.raises(MonitoringInputError, match="one-dimensional cross-section"):
        drift_report(
            as_of=AS_OF,
            stamp=fixture_stamp(),
            observations=[
                (_reference("momentum"), gaussian_sample(seed=65)),
                (_reference("size"), gaussian_sample(seed=66, size=2_000).reshape(100, 20)),
            ],
        )


def test_an_unmeasurable_feature_is_built_from_the_detectors_own_words() -> None:
    reference = _reference("borrow")
    with pytest.raises(InsufficientSampleError) as raised:
        measure_feature_drift(reference=reference, observed=gaussian_sample(seed=67, size=42))
    refusal = raised.value
    entry = UnmeasurableFeature.from_error(refusal, reference=reference)
    assert entry.reason == str(refusal)
    assert entry.quantity == refusal.quantity
    assert entry.n_observed == refusal.n_observations
    assert entry.minimum_required == refusal.minimum
    assert entry.feature == refusal.feature
    assert entry.reference_id == refusal.reference_id


# ---------------------------------------------------------------------------
# 5. The whole report survives serialisation
# ---------------------------------------------------------------------------


def test_a_mixed_report_round_trips_through_json_intact() -> None:
    stamp = fixture_stamp(seed=3)
    report = drift_report(
        as_of=AS_OF,
        stamp=stamp,
        observations=[
            (_reference("momentum"), gaussian_sample(seed=71)),
            (_reference("book_to_price"), gaussian_sample(seed=72, shift=0.4)),
            (_reference("gross_profitability"), gaussian_sample(seed=73, shift=2.0)),
            (_reference("short_interest"), gaussian_sample(seed=74, size=120)),
        ],
    )
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["n_measured"] == 3
    assert restored["n_unmeasurable"] == 1
    assert restored["complete"] is False
    assert restored["worst_band"] == "major"
    assert [entry["feature"] for entry in restored["measured"]] == [
        "FIXTURE_momentum",
        "FIXTURE_book_to_price",
        "FIXTURE_gross_profitability",
    ]
    assert [entry["band"] for entry in restored["measured"]] == ["stable", "moderate", "major"]
    for entry in restored["measured"]:
        assert entry["measured"] is True
        assert entry["psi"]["reference"]["reference_id"].startswith("FIXTURE_train_")
        assert entry["psi"]["reference"]["data_version"] == stamp.data_version
        assert entry["reference_fingerprint"] == entry["psi"]["reference"]["fingerprint"]
    assert restored["unmeasurable"][0]["measured"] is False


def test_an_empty_report_is_representable_and_says_nothing_was_measured() -> None:
    report = drift_report(as_of=AS_OF, stamp=fixture_stamp(), observations=[])
    assert report.complete is True
    assert report.worst_band is DriftBand.STABLE
    payload = report.to_dict()
    assert payload["n_measured"] == 0
    assert payload["measured"] == []


def test_the_finding_exposes_the_full_psi_payload_including_the_floor() -> None:
    reference = equal_decile_reference()
    finding: FeatureDrift = measure_feature_drift(
        reference=reference, observed=np.full(OBSERVED_SIZE, 10_500.0)
    )
    payload = finding.to_dict()
    assert isinstance(payload["psi"], dict)
    assert payload["psi"]["epsilon"] == reference.epsilon
    assert payload["psi"]["floored_bins"] == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert payload["floor_driven"] is True
    assert payload["distribution_band"] == "major"
    json.dumps(payload)
