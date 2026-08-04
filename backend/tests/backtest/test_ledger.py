"""Tests for the ``TESTING_LEDGER.md`` reader (P10.3's trial count).

The ledger supplies ``N`` to the Deflated Sharpe Ratio. Directive §9.7 makes it
append-only because an under-counted ledger inflates every DSR computed from it,
so the failure this module must never have is *silently returning too few rows*.
Every test below is written around that: a malformed file raises, a missing file
raises, and an empty ledger returns zero rather than a comfortable 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.backtest.dsr import deflated_sharpe_ratio_from_returns
from backend.backtest.ledger import LedgerFormatError, read_testing_ledger

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_LEDGER_PATH = _REPOSITORY_ROOT / "TESTING_LEDGER.md"


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The real ledger in this repository
# ---------------------------------------------------------------------------


def test_the_committed_ledger_parses_and_its_row_count_is_the_trial_count() -> None:
    ledger = read_testing_ledger(_LEDGER_PATH)

    assert ledger.header[0] == "#"
    assert "config_hash" in ledger.header
    assert "seed" in ledger.header
    assert "git commit" in ledger.header
    assert ledger.trial_count == len(ledger.rows)

    # Counted independently of the parser: table rows are the pipe-delimited
    # lines that are neither the header nor a separator.
    lines = [
        line.strip()
        for line in _LEDGER_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    separators = [line for line in lines if set(line) <= set("|-: ")]
    assert ledger.trial_count == len(lines) - len(separators) - 1

    for row in ledger.rows:
        assert len(row.cells) == len(ledger.header)
        assert row.line_number >= 1


def test_an_empty_ledger_cannot_produce_a_deflated_sharpe_ratio(tmp_path: Path) -> None:
    # An empty ledger means "nothing has been evaluated", never "one thing was
    # evaluated". Wiring its trial count straight into the DSR must therefore
    # fail closed rather than quietly deflate against a single trial — which is
    # the same thing as not deflating at all.
    path = _write(tmp_path / "TESTING_LEDGER.md", "| # | experiment |\n|---|---|\n")
    ledger = read_testing_ledger(path)
    assert ledger.trial_count == 0
    with pytest.raises(ValueError, match="trials must be at least 1"):
        deflated_sharpe_ratio_from_returns(
            [0.01, -0.005, 0.02, 0.003, -0.01],
            trials=ledger.trial_count,
            trial_sharpe_variance=0.0,
        )


def test_a_populated_ledger_feeds_its_row_count_straight_into_the_deflation(
    tmp_path: Path,
) -> None:
    # The wiring the Deflated Sharpe Ratio depends on: N is the number of rows,
    # nothing else. Twelve rows must deflate strictly harder than three.
    header = "| # | experiment |\n|---|---|\n"
    small = read_testing_ledger(
        _write(tmp_path / "small.md", header + "".join(f"| {i} | e{i} |\n" for i in range(3)))
    )
    large = read_testing_ledger(
        _write(tmp_path / "large.md", header + "".join(f"| {i} | e{i} |\n" for i in range(12)))
    )
    assert (small.trial_count, large.trial_count) == (3, 12)

    returns = [0.01, -0.005, 0.02, 0.003, -0.01, 0.008, -0.002, 0.011]
    deflated_small = deflated_sharpe_ratio_from_returns(
        returns, trials=small.trial_count, trial_sharpe_variance=0.04
    )
    deflated_large = deflated_sharpe_ratio_from_returns(
        returns, trials=large.trial_count, trial_sharpe_variance=0.04
    )
    assert deflated_large.value < deflated_small.value
    assert deflated_small.trials == 3
    assert deflated_large.trials == 12


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_rows_are_counted_and_their_cells_preserved(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "TESTING_LEDGER.md",
        "# Ledger\n"
        "\n"
        "Some preamble prose that is not a table.\n"
        "\n"
        "| # | experiment | result |\n"
        "|---|---|---|\n"
        "| 1 | momentum-12-1 | IC 0.021 |\n"
        "| 2 | momentum-6-1 | IC 0.004 |\n"
        "| 3 | value-composite | abandoned |\n"
        "\n"
        "*a trailing note*\n",
    )
    ledger = read_testing_ledger(path)

    assert ledger.header == ("#", "experiment", "result")
    assert ledger.trial_count == 3
    assert [row.cells for row in ledger.rows] == [
        ("1", "momentum-12-1", "IC 0.021"),
        ("2", "momentum-6-1", "IC 0.004"),
        ("3", "value-composite", "abandoned"),
    ]
    # Line numbers are 1-based so a human can jump straight to a bad row.
    assert [row.line_number for row in ledger.rows] == [7, 8, 9]


def test_an_empty_table_yields_zero_trials_rather_than_a_comfortable_one(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "TESTING_LEDGER.md",
        "| # | experiment |\n|---|---|\n\n*(nothing evaluated yet)*\n",
    )
    ledger = read_testing_ledger(path)
    assert ledger.trial_count == 0
    assert ledger.rows == ()


def test_a_row_whose_width_disagrees_with_the_header_is_fatal(tmp_path: Path) -> None:
    # The alternative to raising is returning a trial count that might be too
    # small, and a too-small trial count inflates every DSR computed from it.
    path = _write(
        tmp_path / "TESTING_LEDGER.md",
        "| # | experiment | result |\n|---|---|---|\n| 1 | ok | fine |\n| 2 | missing-a-cell |\n",
    )
    with pytest.raises(LedgerFormatError, match="line 4 has 2 cells but the header declares 3"):
        read_testing_ledger(path)


def test_a_file_without_a_table_is_fatal(tmp_path: Path) -> None:
    path = _write(tmp_path / "TESTING_LEDGER.md", "# Ledger\n\nNo table here at all.\n")
    with pytest.raises(LedgerFormatError, match="no markdown table header"):
        read_testing_ledger(path)


def test_a_header_without_a_separator_is_not_treated_as_a_table(tmp_path: Path) -> None:
    path = _write(tmp_path / "TESTING_LEDGER.md", "| a | b |\n| 1 | 2 |\n")
    with pytest.raises(LedgerFormatError, match="no markdown table header"):
        read_testing_ledger(path)


def test_a_missing_ledger_is_never_an_empty_ledger(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_testing_ledger(tmp_path / "does-not-exist.md")


def test_alignment_markers_and_repeated_separators_are_tolerated(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "TESTING_LEDGER.md",
        "| # | experiment |\n|:--|--:|\n| 1 | a |\n| :-: | :-: |\n| 2 | b |\n",
    )
    ledger = read_testing_ledger(path)
    assert ledger.trial_count == 2
    assert [row.cells[0] for row in ledger.rows] == ["1", "2"]


def test_only_the_first_table_supplies_the_header(tmp_path: Path) -> None:
    # A second table cannot redefine the column count, so it cannot start
    # silently rejecting rows of the real one. Its rows — header included — are
    # counted as trials when their width matches, which over-counts. That is the
    # safe direction: over-counting over-deflates, while under-counting inflates
    # every Deflated Sharpe Ratio computed from the file.
    path = _write(
        tmp_path / "TESTING_LEDGER.md",
        "| # | experiment |\n|---|---|\n| 1 | a |\n\n| other | table |\n|---|---|\n| 2 | b |\n",
    )
    ledger = read_testing_ledger(path)
    assert ledger.header == ("#", "experiment")
    assert ledger.trial_count == 3
    assert [row.cells for row in ledger.rows] == [("1", "a"), ("other", "table"), ("2", "b")]


def test_surrounding_whitespace_in_cells_is_stripped(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "TESTING_LEDGER.md",
        "|  #  |  experiment  |\n|---|---|\n|   1   |   spaced out   |\n",
    )
    ledger = read_testing_ledger(path)
    assert ledger.header == ("#", "experiment")
    assert ledger.rows[0].cells == ("1", "spaced out")


def test_the_reader_has_no_write_path(tmp_path: Path) -> None:
    # The ledger is append-only state owned by the training entrypoint. A reader
    # that could edit it would defeat the guarantee it exists to provide, so the
    # module must not export anything that writes.
    from backend.backtest import ledger as ledger_module

    exported = [getattr(ledger_module, name) for name in ledger_module.__all__]
    callables = [item for item in exported if callable(item)]
    assert [item.__name__ for item in callables if not isinstance(item, type)] == [
        "read_testing_ledger"
    ]

    original = "| # |\n|---|\n| 1 |\n"
    path = _write(tmp_path / "TESTING_LEDGER.md", original)
    read_testing_ledger(path)
    assert path.read_text(encoding="utf-8") == original
