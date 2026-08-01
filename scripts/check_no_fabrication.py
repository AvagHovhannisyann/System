#!/usr/bin/env python
"""CC.8 — no-fabrication guard as code (invariant I3).

Scans the ingestion **production** tree for identifiers that name fabricated
data: ``mock``, ``stub``, ``fake``, ``dummy``, ``synthetic``, ``placeholder``,
``fabricated``, ``lorem``, ``faker``. Any such name in shipped code is either a
stub returning plausible values (directive §9.2) or synthetic data on a path
that will be mistaken for real (§9.1), and both are the failure mode invariant
I3 exists to prevent.

Scope and honest limitations
----------------------------

- **Identifiers and imports only.** Docstrings, comments and string literals
  are *not* scanned, deliberately: the modules under scrutiny must be free to
  explain in prose that they never return a placeholder, and a checker that
  flagged its own prohibition would be self-defeating.
- **Production paths only.** ``backend/tests`` is not scanned. A test double
  that simulates an unavailable source is legitimate scaffolding — it is how
  CC.8's contract test proves the base class raises. What is forbidden is
  production code returning fabricated values.
- **A name-level guard is not a proof.** Nothing here can detect a fabricated
  value that happens to be well named. This check makes the *careless* case
  impossible and leaves the deliberate case to the base-class contract test
  and to review. It is stated here rather than left to be discovered.

Usage::

    python scripts/check_no_fabrication.py [PATH ...]

Defaults to ``backend/ingest``. Exits 0 when clean, 1 when violations are
found (printing ``file:line:col: identifier``), 2 on a usage error.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple

REPO_ROOT: Final = Path(__file__).resolve().parents[1]

DEFAULT_TARGETS: Final = ("backend/ingest",)
"""Production trees scanned when no path is given on the command line."""

BANNED_SEGMENTS: Final = frozenset(
    {
        "mock",
        "mocks",
        "mocked",
        "mocking",
        "stub",
        "stubs",
        "stubbed",
        "fake",
        "fakes",
        "faked",
        "faker",
        "dummy",
        "dummies",
        "synthetic",
        "placeholder",
        "placeholders",
        "fabricate",
        "fabricated",
        "lorem",
        "ipsum",
    }
)
"""Word segments that may not appear in an identifier in the scanned tree."""

BANNED_SUBSTRINGS: Final = frozenset({"fabricat"})
"""Substrings caught even when an identifier carries no word separators."""

_SEGMENT_SPLIT: Final = re.compile(r"[^A-Za-z0-9]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
"""Splits an identifier on separators and camelCase/PascalCase boundaries."""


class Violation(NamedTuple):
    """One banned identifier found in the scanned tree.

    Attributes:
        path: file path, relative to the repository root.
        line: 1-based line number.
        column: 0-based column offset.
        identifier: the offending name, verbatim.
        reason: which rule matched (segment or substring), for the message.
    """

    path: Path
    line: int
    column: int
    identifier: str
    reason: str

    def render(self) -> str:
        """Return the violation as a ``file:line:col: message`` string."""
        return f"{self.path}:{self.line}:{self.column}: {self.identifier!r} — {self.reason}"


def segments(identifier: str) -> list[str]:
    """Split an identifier into lower-cased word segments.

    Handles ``snake_case``, ``SCREAMING_SNAKE_CASE``, ``camelCase``,
    ``PascalCase`` and dotted module paths alike, so ``_mockClient``,
    ``MOCK_ROWS`` and ``unittest.mock`` all yield a ``mock`` segment.
    """
    return [part.lower() for part in _SEGMENT_SPLIT.split(identifier) if part]


def banned_reason(identifier: str) -> str | None:
    """Return why ``identifier`` is banned, or ``None`` when it is acceptable."""
    for segment in segments(identifier):
        if segment in BANNED_SEGMENTS:
            return f"identifier segment {segment!r} names fabricated data (I3)"
    lowered = identifier.lower()
    for substring in BANNED_SUBSTRINGS:
        if substring in lowered:
            return f"identifier contains {substring!r}, which names fabricated data (I3)"
    return None


def _identifiers(tree: ast.AST) -> list[tuple[str, int, int]]:
    """Collect every declared or referenced identifier in an AST.

    Covers definitions (functions, classes, arguments), name loads and stores,
    attribute names, import module paths and aliases, and ``global``/
    ``nonlocal`` declarations. String literals and comments are excluded by
    construction — the AST carries no comments, and constants are not names.
    """
    found: list[tuple[str, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.append((node.name, node.lineno, node.col_offset))
        elif isinstance(node, ast.Name):
            found.append((node.id, node.lineno, node.col_offset))
        elif isinstance(node, ast.Attribute):
            found.append((node.attr, node.lineno, node.col_offset))
        elif isinstance(node, ast.arg):
            found.append((node.arg, node.lineno, node.col_offset))
        elif isinstance(node, ast.alias):
            found.append((node.name, node.lineno, node.col_offset))
            if node.asname:
                found.append((node.asname, node.lineno, node.col_offset))
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append((node.module, node.lineno, node.col_offset))
        elif isinstance(node, ast.Global | ast.Nonlocal):
            found.extend((name, node.lineno, node.col_offset) for name in node.names)
    return found


def scan_file(path: Path) -> list[Violation]:
    """Return every banned identifier in one Python file.

    Raises:
        SyntaxError: if the file does not parse. Deliberately not caught: a
            file the guard cannot read is a file the guard cannot vouch for.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Repository-relative when possible (readable CI output), absolute otherwise
    # — the guard accepts arbitrary paths, and a path outside the repository is
    # a legitimate target, not an error.
    relative = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    violations: list[Violation] = []
    for identifier, line, column in _identifiers(tree):
        reason = banned_reason(identifier)
        if reason is not None:
            violations.append(Violation(relative, line, column, identifier, reason))
    return sorted(violations)


def scan_paths(targets: list[Path]) -> list[Violation]:
    """Return every banned identifier under the given files/directories."""
    violations: list[Violation] = []
    for target in targets:
        if target.is_dir():
            for path in sorted(target.rglob("*.py")):
                violations.extend(scan_file(path))
        elif target.is_file():
            violations.extend(scan_file(target))
        else:
            msg = f"no such file or directory: {target}"
            raise FileNotFoundError(msg)
    return violations


def main(argv: list[str]) -> int:
    """Run the guard over ``argv`` paths (or the defaults) and report.

    Returns:
        Process exit status: 0 clean, 1 violations found, 2 usage error.
    """
    raw_targets = argv[1:] or [str(REPO_ROOT / target) for target in DEFAULT_TARGETS]
    targets = [Path(target) for target in raw_targets]
    try:
        violations = scan_paths(targets)
    except (FileNotFoundError, SyntaxError) as exc:
        print(f"check_no_fabrication: {exc}", file=sys.stderr)
        return 2
    if violations:
        print(
            "I3 violation — fabricated-data identifiers in ingestion production code:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation.render()}", file=sys.stderr)
        print(
            "\nProduction code must never stub, mimic or stand in for a data source. "
            "If a source is unavailable, raise (DIRECTIVE.md I3, sections 9.1-9.2). "
            "Test doubles belong under backend/tests.",
            file=sys.stderr,
        )
        return 1
    scanned = ", ".join(str(target) for target in targets)
    print(f"CC.8 no-fabrication guard: clean ({scanned})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
