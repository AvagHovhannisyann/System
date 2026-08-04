"""Tests for the append-only ledger writer (CC.3, directive §9.7).

Two guarantees are load-bearing and both are asserted here directly.

**Nothing already in the file changes.** §9.7 forbids deleting a row because a
missing failed experiment shrinks the trial count *N* and inflates every
Deflated Sharpe Ratio computed from it. The test compares raw bytes: the file
after an append must start with exactly the bytes it had before.

**The row is readable by the consumer that matters.** Rows are parsed back with
:mod:`backend.backtest.ledger` — the reader the DSR uses. A row this writer
produced but that reader refuses would break the trial count instead of
maintaining it, so the round trip is what makes the link trustworthy.
"""

from __future__ import annotations

import itertools
import shutil
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.backtest.ledger import LedgerFormatError, read_testing_ledger
from backend.tracking.ledger import LEDGER_COLUMNS, LedgerAppendError, TrialRecord, append_trial
from backend.tracking.stamp import ReproducibilityStamp

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COMMITTED_LEDGER = _REPOSITORY_ROOT / "TESTING_LEDGER.md"

STAMP = ReproducibilityStamp(
    git_commit="1" * 40,
    git_dirty=False,
    data_version="dvc:abc123",
    config_hash="b" * 64,
    seed=42,
)
DIRTY_STAMP = ReproducibilityStamp(
    git_commit="2" * 40,
    git_dirty=True,
    data_version="dvc:abc123",
    config_hash="c" * 64,
    seed=7,
)


def _record(**overrides: object) -> TrialRecord:
    fields: dict[str, object] = {
        "phase_task": "P8.4",
        "experiment_id": "mlflow:0123456789abcdef",
        "stamp": STAMP,
        "cv_scheme": "purged 5-fold, 21d embargo",
        "metrics": "IC 0.031, t=2.4",
        "result": "kept as baseline",
        "notes": "",
    }
    fields.update(overrides)
    return TrialRecord(**fields)  # type: ignore[arg-type]


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    """A copy of the committed ``TESTING_LEDGER.md``, so the real one is untouched."""
    destination = tmp_path / "TESTING_LEDGER.md"
    shutil.copyfile(_COMMITTED_LEDGER, destination)
    return destination


# ---------------------------------------------------------------------------
# Append-only: prior content is byte-identical
# ---------------------------------------------------------------------------


def test_an_append_leaves_every_prior_byte_untouched(ledger: Path) -> None:
    before = ledger.read_bytes()

    append_trial(ledger, _record())

    after = ledger.read_bytes()
    assert after.startswith(before)
    assert len(after) > len(before)


def test_repeated_appends_never_rewrite_an_earlier_row(ledger: Path) -> None:
    snapshots = [ledger.read_bytes()]
    for index in range(4):
        append_trial(ledger, _record(experiment_id=f"mlflow:{index}"))
        snapshots.append(ledger.read_bytes())

    for earlier, later in itertools.pairwise(snapshots):
        assert later.startswith(earlier)

    rows = read_testing_ledger(ledger).rows
    assert [row.cells[3] for row in rows] == [f"mlflow:{index}" for index in range(4)]


def test_a_ledger_that_already_has_rows_keeps_them_byte_for_byte(ledger: Path) -> None:
    append_trial(ledger, _record(experiment_id="mlflow:first"))
    with_one_row = ledger.read_bytes()
    first_row_line = [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if "mlflow:first" in line
    ]

    append_trial(ledger, _record(experiment_id="mlflow:second"))

    assert ledger.read_bytes().startswith(with_one_row)
    assert [
        line for line in ledger.read_text(encoding="utf-8").splitlines() if "mlflow:first" in line
    ] == first_row_line


def test_a_file_without_a_trailing_newline_is_extended_not_corrupted(ledger: Path) -> None:
    text = ledger.read_text(encoding="utf-8").rstrip("\n")
    ledger.write_text(text, encoding="utf-8")
    before = ledger.read_bytes()

    append_trial(ledger, _record())

    assert ledger.read_bytes().startswith(before)
    assert read_testing_ledger(ledger).trial_count == 1


# ---------------------------------------------------------------------------
# Round trip through the reader the DSR uses
# ---------------------------------------------------------------------------


def test_an_appended_row_is_parseable_by_the_backtest_reader(ledger: Path) -> None:
    before = read_testing_ledger(ledger)

    number = append_trial(
        ledger,
        _record(notes="first real trial"),
        timestamp=datetime(2026, 8, 1, 12, 30, 45, tzinfo=UTC),
    )

    after = read_testing_ledger(ledger)
    assert number == before.trial_count + 1
    assert after.trial_count == before.trial_count + 1
    assert after.header == before.header

    row = dict(zip(after.header, after.rows[-1].cells, strict=True))
    assert row["#"] == str(number)
    assert row["timestamp (UTC)"] == "2026-08-01T12:30:45Z"
    assert row["phase/task"] == "P8.4"
    assert row["experiment id"] == "mlflow:0123456789abcdef"
    assert row["config_hash"] == STAMP.config_hash
    assert row["data_version"] == STAMP.data_version
    assert row["git commit"] == STAMP.git_commit
    assert row["seed"] == "42"
    assert row["CV scheme"] == "purged 5-fold, 21d embargo"
    assert row["metric(s)"] == "IC 0.031, t=2.4"
    assert row["result"] == "kept as baseline"
    assert row["notes"] == "first real trial"


def test_the_row_carries_all_four_i2_components(ledger: Path) -> None:
    append_trial(ledger, _record())

    cells = read_testing_ledger(ledger).rows[-1].cells

    assert STAMP.git_commit in cells
    assert STAMP.data_version in cells
    assert STAMP.config_hash in cells
    assert str(STAMP.seed) in cells


def test_a_dirty_tree_is_visible_in_the_row(ledger: Path) -> None:
    append_trial(ledger, _record(stamp=DIRTY_STAMP))

    row = dict(zip(LEDGER_COLUMNS, read_testing_ledger(ledger).rows[-1].cells, strict=True))

    assert row["git commit"] == f"{DIRTY_STAMP.git_commit}-dirty"


def test_the_trial_number_matches_the_count_the_dsr_will_see(ledger: Path) -> None:
    numbers = [append_trial(ledger, _record(experiment_id=f"run-{i}")) for i in range(3)]

    assert numbers == [1, 2, 3]
    assert read_testing_ledger(ledger).trial_count == 3


def test_a_timestamp_in_another_timezone_is_normalised_to_utc(ledger: Path) -> None:
    moment = datetime(2026, 8, 1, 9, 0, 0, tzinfo=timezone(timedelta(hours=-4)))

    append_trial(ledger, _record(), timestamp=moment)

    row = dict(zip(LEDGER_COLUMNS, read_testing_ledger(ledger).rows[-1].cells, strict=True))
    assert row["timestamp (UTC)"] == "2026-08-01T13:00:00Z"


def test_the_committed_schema_is_the_one_this_writer_produces() -> None:
    # If the ledger's header is ever edited, this fails here rather than
    # producing rows that silently no longer line up with their columns.
    assert read_testing_ledger(_COMMITTED_LEDGER).header == LEDGER_COLUMNS


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("phase_task", "P8|4", id="pipe-splits-the-row"),
        pytest.param("result", "line one\nline two", id="newline-truncates-the-row"),
        pytest.param("notes", "a\rb", id="carriage-return"),
        pytest.param("phase_task", "", id="empty-required-cell"),
        pytest.param("experiment_id", "   ", id="blank-required-cell"),
        pytest.param("result", "", id="empty-result"),
    ],
)
def test_a_cell_that_would_corrupt_the_table_is_refused(
    ledger: Path,
    field: str,
    value: str,
) -> None:
    before = ledger.read_bytes()

    with pytest.raises(LedgerAppendError):
        append_trial(ledger, _record(**{field: value}))

    assert ledger.read_bytes() == before


def test_a_row_of_dashes_is_still_counted_and_not_read_as_a_separator(ledger: Path) -> None:
    # The reader skips ``|---|---|`` separator rows. A trial row whose free-text
    # cells happen to be dashes must not vanish into that case: the numbered
    # first cell and the timestamp keep it a data row, and append_trial verifies
    # the row was counted before returning.
    number = append_trial(
        ledger,
        _record(phase_task="---", cv_scheme="---", metrics="---", result="---"),
        timestamp=datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert number == 1
    assert read_testing_ledger(ledger).trial_count == 1


def test_a_naive_timestamp_is_refused(ledger: Path) -> None:
    before = ledger.read_bytes()

    with pytest.raises(LedgerAppendError):
        append_trial(ledger, _record(), timestamp=datetime(2026, 8, 1, 12, 0, 0))  # noqa: DTZ001

    assert ledger.read_bytes() == before


def test_a_ledger_with_a_different_schema_is_refused(tmp_path: Path) -> None:
    other = tmp_path / "TESTING_LEDGER.md"
    other.write_text("| a | b |\n|---|---|\n", encoding="utf-8")
    before = other.read_bytes()

    with pytest.raises(LedgerAppendError):
        append_trial(other, _record())

    assert other.read_bytes() == before


def test_an_unparseable_ledger_is_not_appended_to(tmp_path: Path) -> None:
    broken = tmp_path / "TESTING_LEDGER.md"
    broken.write_text("no table here at all\n", encoding="utf-8")
    before = broken.read_bytes()

    with pytest.raises(LedgerFormatError):
        append_trial(broken, _record())

    assert broken.read_bytes() == before


def test_a_missing_ledger_is_never_created(tmp_path: Path) -> None:
    # Creating one would restart the trial count at zero and silently discard
    # every experiment already recorded.
    absent = tmp_path / "TESTING_LEDGER.md"

    with pytest.raises(FileNotFoundError):
        append_trial(absent, _record())

    assert not absent.exists()


def test_there_is_no_write_path_in_the_reader_module() -> None:
    # The read/write split is the guarantee: the module the DSR imports cannot
    # modify the ledger, and this writer only ever appends.
    import backend.backtest.ledger as reader

    source = Path(reader.__file__).read_text(encoding="utf-8")
    assert "write_text" not in source
    assert "open(" not in source
