"""Static check: no string-interpolated SQL in application code (DIRECTIVE.md §7, CC.2).

§7 requires "parameterized queries only". ``backend/db`` already makes it
impossible to *reach* a fact table outside the versioned query layer (D-011,
D-012, D-018), but that is a different property: a sanctioned statement can
still be assembled by string concatenation. This module closes that gap with
an AST scan of every shipped module under ``backend/``.

What it flags
-------------

Two independent rules, either of which is a finding:

``sink``
    A call to a SQL-executing sink — ``text``, ``execute``, ``executemany``,
    ``exec_driver_sql``, ``executescript`` — whose first positional argument
    is a *dynamically constructed* string: an f-string with a substitution, a
    ``%`` format, a ``+`` concatenation involving a non-literal, or
    ``str.format``.

``shape``
    Any dynamically constructed string, anywhere, whose literal skeleton
    begins with a SQL statement keyword (``SELECT``, ``INSERT INTO``,
    ``UPDATE``, ``DELETE FROM``, DDL verbs, ...). This catches the common
    two-step shape where the SQL is built into a local variable and handed to
    a sink further down.

Why it is scoped this way
-------------------------

Precision matters more than reach here, because a check that cries wolf gets
switched off. ``backend/db/_guard.py`` and ``backend/db/asof.py`` are full of
SQL-shaped strings — error messages naming statements, regexes built from
table names — and none of them are queries. Anchoring the ``shape`` rule to
the *start* of the literal skeleton, and the ``sink`` rule to an actual call
site, distinguishes those from real query construction: the clean tree
produces only the
schema-migration findings covered by :func:`is_tolerated` — DDL identifier
interpolation, which no database can express as bind parameters.

Limits, stated plainly
----------------------

This is a syntactic check, not dataflow analysis. A dynamic string that does
*not* start with a SQL keyword and reaches a sink through one or more
intermediate variables or a helper function is not detected. It is a tripwire
against the ordinary mistake, on top of the structural guarantees in
``backend/db`` — not a proof of absence.

``backend/tests`` is deliberately excluded: the bypass-prevention suites build
malformed SQL *on purpose* to prove the database guard refuses it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
"""Repository root (``backend/tests/x.py`` -> two levels up)."""

APPLICATION_ROOT: Final = REPO_ROOT / "backend"
"""Tree scanned by the check."""

EXCLUDED_DIRECTORIES: Final[frozenset[str]] = frozenset({"tests"})
"""Directories under ``backend/`` skipped by the scan; see the module docstring."""

SQL_SINKS: Final[frozenset[str]] = frozenset(
    {"text", "execute", "executemany", "exec_driver_sql", "executescript"}
)
"""Callables whose first positional argument is executed as SQL."""

DDL_PREFIXES: Final[tuple[str, ...]] = (
    "create ",
    "drop ",
    "alter ",
    "truncate",
    "grant ",
    "revoke ",
    "comment on",
)
"""Lower-cased openings of schema-definition statements.

DDL is the one place interpolation is unavoidable: no database accepts a bind
parameter in the position of an *identifier*, so ``CREATE INDEX ix_{table}``
has no parameterized form.
"""

DML_PREFIXES: Final[tuple[str, ...]] = (
    "select ",
    "insert into",
    "update ",
    "delete from",
    "merge into",
    "copy ",
)
"""Lower-cased openings of data-manipulation statements.

These always have a parameterized form, so interpolating one is never
necessary — anywhere, including in a migration.
"""

SQL_STATEMENT_PREFIXES: Final[tuple[str, ...]] = DDL_PREFIXES + DML_PREFIXES
"""Lower-cased openings that mark a literal skeleton as a SQL statement."""

PLACEHOLDER: Final = "?"
"""Stands in for an interpolated expression when a skeleton is rendered."""

MIGRATIONS_PREFIX: Final = "backend/db/migrations/versions/"
"""Alembic revision directory — the only place interpolation is tolerated."""


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    """One place where SQL appears to be assembled from a dynamic string.

    Attributes:
        path: repository-relative POSIX path of the module.
        qualname: dotted name of the enclosing class/function, or
            ``"<module>"``.
        skeleton: the literal parts of the string with every interpolated
            expression replaced by :data:`PLACEHOLDER` and whitespace
            collapsed. Stable across reformatting and line moves, which is
            what makes it usable as an allowlist key.
    """

    path: str
    qualname: str
    skeleton: str


def is_tolerated(finding: Finding) -> bool:
    """Return whether a finding falls inside the one documented exception.

    **The exception, and its boundary.** An Alembic revision under
    :data:`MIGRATIONS_PREFIX` may interpolate a *schema-definition* statement.
    Two things justify it, and both are needed:

    1. **It has no parameterized form.** Identifiers — table, index, trigger,
       column names — cannot be bind parameters in Postgres. A revision that
       creates the same trigger on each fact table has to interpolate the
       name; there is no correct alternative to point the author at.
    2. **Nothing untrusted is in scope.** Revisions are static code executed
       offline by ``alembic upgrade`` at deploy time. No request, no database
       value and no operator input reaches them; the interpolated values are
       module-level constants in the revision itself.

    Neither justification covers **DML**. ``SELECT``/``INSERT``/``UPDATE``/
    ``DELETE`` always have a parameterized form, so an interpolated one in a
    revision is reported like anywhere else — and so is *any* interpolation
    outside the revisions directory, which is where request data lives.

    This is a rule rather than a hand-maintained list of approved statements.
    A list would be more precise per statement, and would also fail every time
    a new revision reused the established DDL pattern, which makes it a tax on
    unrelated work and — in a repository several people commit migrations to —
    an invitation to widen it without reading it. The residual risk the rule
    accepts is a *new* interpolated DDL shape in a revision going unreviewed;
    the risk it removes is interpolated SQL anywhere a value can flow.
    """
    return finding.path.startswith(MIGRATIONS_PREFIX) and finding.skeleton.lower().startswith(
        DDL_PREFIXES
    )


def _literal_skeleton(node: ast.expr) -> tuple[str | None, bool]:
    """Render an expression's literal skeleton and say whether it is dynamic.

    Args:
        node: any expression node.

    Returns:
        ``(skeleton, dynamic)``. ``skeleton`` is the concatenation of the
        string literals in the expression with :data:`PLACEHOLDER` standing in
        for interpolated parts, or ``None`` when the expression contributes no
        literal text at all. ``dynamic`` is ``True`` when the value is not a
        compile-time constant string.
    """
    if isinstance(node, ast.Constant):
        return (node.value, False) if isinstance(node.value, str) else (None, False)

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        dynamic = False
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append(PLACEHOLDER)
                dynamic = True
        return "".join(parts), dynamic

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Mod):
        left, left_dynamic = _literal_skeleton(node.left)
        right, right_dynamic = _literal_skeleton(node.right)
        if left is None and right is None:
            return None, False
        dynamic = (
            left_dynamic
            or right_dynamic
            or left is None
            or right is None
            or isinstance(node.op, ast.Mod)
        )
        return (left or PLACEHOLDER) + (right or PLACEHOLDER), dynamic

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    ):
        template, _ = _literal_skeleton(node.func.value)
        return template, template is not None

    return None, False


def _normalize(skeleton: str) -> str:
    """Collapse runs of whitespace so formatting changes do not move a finding."""
    return " ".join(skeleton.split())


def _looks_like_sql(skeleton: str) -> bool:
    """Return whether a literal skeleton opens with a SQL statement keyword."""
    opening = skeleton.lstrip().lower()
    return opening.startswith(SQL_STATEMENT_PREFIXES)


def _called_name(node: ast.Call) -> str | None:
    """Return the bare name of the callable in a call node, if it has one."""
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return None


class _InterpolatedSqlVisitor(ast.NodeVisitor):
    """Collect :class:`Finding` objects for one module."""

    def __init__(self, path: str) -> None:
        """Start an empty collection for the module at ``path``."""
        self.path = path
        self.findings: list[Finding] = []
        self._scope: list[str] = []

    def _record(self, skeleton: str) -> None:
        """Add a finding for the current scope."""
        self.findings.append(
            Finding(
                path=self.path,
                qualname=".".join(self._scope) or "<module>",
                skeleton=_normalize(skeleton),
            )
        )

    def _visit_scope(self, node: ast.AST, name: str) -> None:
        """Walk ``node``'s children with ``name`` pushed onto the scope stack."""
        self._scope.append(name)
        self.generic_visit(node)
        self._scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track the enclosing function name."""
        self._visit_scope(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Track the enclosing coroutine name."""
        self._visit_scope(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track the enclosing class name."""
        self._visit_scope(node, node.name)

    def visit_Call(self, node: ast.Call) -> None:
        """Apply the ``sink`` rule, then continue walking."""
        if _called_name(node) in SQL_SINKS and node.args:
            skeleton, dynamic = _literal_skeleton(node.args[0])
            if dynamic and skeleton is not None:
                self._record(skeleton)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """Apply the ``shape`` rule to f-strings."""
        self._check_shape(node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Apply the ``shape`` rule to ``+`` and ``%`` string expressions."""
        self._check_shape(node)
        self.generic_visit(node)

    def _check_shape(self, node: ast.expr) -> None:
        """Record a finding when a dynamic string opens with a SQL keyword."""
        skeleton, dynamic = _literal_skeleton(node)
        if dynamic and skeleton is not None and _looks_like_sql(skeleton):
            self._record(skeleton)


def scan_source(source: str, path: str) -> set[Finding]:
    """Return the deduplicated findings in one module's source text.

    Args:
        source: Python source code.
        path: repository-relative path used to label findings.

    Returns:
        A set of findings. The ``sink`` and ``shape`` rules routinely report
        the same statement; deduplicating on ``(path, qualname, skeleton)``
        collapses those into one, so a finding corresponds to a statement
        rather than to a rule firing.
    """
    visitor = _InterpolatedSqlVisitor(path)
    visitor.visit(ast.parse(source))
    return set(visitor.findings)


def _label(module_path: Path, root: Path) -> str:
    """Return the path used to label findings: repo-relative when possible."""
    base = REPO_ROOT if module_path.is_relative_to(REPO_ROOT) else root
    return module_path.relative_to(base).as_posix()


def scan_tree(root: Path, *, excluded_directories: frozenset[str] = frozenset()) -> set[Finding]:
    """Scan every ``.py`` file under ``root``, skipping ``excluded_directories``.

    Args:
        root: directory to walk.
        excluded_directories: directory *names* (any depth) to skip.

    Returns:
        The union of findings across the tree, labelled with paths relative to
        :data:`REPO_ROOT` when possible.
    """
    findings: set[Finding] = set()
    for module_path in sorted(root.rglob("*.py")):
        if excluded_directories & set(module_path.relative_to(root).parts[:-1]):
            continue
        relative = _label(module_path, root)
        findings |= scan_source(module_path.read_text(encoding="utf-8"), relative)
    return findings


def application_findings() -> set[Finding]:
    """Scan the shipped backend tree with the project's exclusions applied."""
    return scan_tree(APPLICATION_ROOT, excluded_directories=EXCLUDED_DIRECTORIES)


# --------------------------------------------------------------------------
# The check itself
# --------------------------------------------------------------------------


def _report(findings: set[Finding]) -> str:
    """Render findings one per line for an assertion message."""
    return "\n".join(f"  {f.path}::{f.qualname}: {f.skeleton}" for f in sorted(findings))


def test_no_interpolated_sql_outside_schema_migrations() -> None:
    """No module on a data path builds SQL from a dynamic string.

    This is the §7 check proper. Everything that can ever see a request, a
    database value or an operator input lives here.
    """
    offenders = {f for f in application_findings() if not f.path.startswith(MIGRATIONS_PREFIX)}
    assert not offenders, (
        "String-interpolated SQL found in application code (DIRECTIVE.md section 7 "
        "requires parameterized queries only). Pass the value as a bind parameter "
        "instead of formatting it into the statement:\n" + _report(offenders)
    )


def test_schema_migrations_interpolate_only_ddl() -> None:
    """Revisions may interpolate identifiers into DDL, never into DML.

    Identifiers have no bind-parameter form, so DDL interpolation is
    unavoidable; ``SELECT``/``INSERT``/``UPDATE``/``DELETE`` always have one,
    so an interpolated statement of that shape is a finding even in a
    migration. See :func:`is_tolerated` for the full argument.
    """
    offenders = {
        f
        for f in application_findings()
        if f.path.startswith(MIGRATIONS_PREFIX) and not is_tolerated(f)
    }
    assert not offenders, (
        "A schema migration interpolates a non-DDL statement. Identifiers cannot "
        "be bound, but data can and must be:\n" + _report(offenders)
    )


def test_every_finding_in_the_tree_is_accounted_for() -> None:
    """The two checks above together cover every finding — no third category."""
    assert not {f for f in application_findings() if not is_tolerated(f)}


def test_tolerance_rule_does_not_cover_runtime_code() -> None:
    """The exception is anchored to the revisions directory, not to a statement shape.

    A ``CREATE INDEX`` built by interpolation inside, say, an ingest module
    would be reported — the DDL argument only holds where nothing untrusted
    can reach the statement.
    """
    ddl_in_runtime_code = Finding(
        path="backend/ingest/edgar.py",
        qualname="build",
        skeleton="CREATE INDEX ix_? ON ?",
    )
    ddl_in_a_revision = Finding(
        path=f"{MIGRATIONS_PREFIX}0003_hypertable_append_only.py",
        qualname="upgrade",
        skeleton="CREATE INDEX ix_? ON ?",
    )
    dml_in_a_revision = Finding(
        path=f"{MIGRATIONS_PREFIX}0003_hypertable_append_only.py",
        qualname="upgrade",
        skeleton="UPDATE ? SET x = 1",
    )

    assert not is_tolerated(ddl_in_runtime_code)
    assert not is_tolerated(dml_in_a_revision)
    assert is_tolerated(ddl_in_a_revision)


# --------------------------------------------------------------------------
# The check is itself checked: synthetic violations must be caught
# --------------------------------------------------------------------------

_FSTRING_INTO_TEXT = '''
from sqlalchemy import text

def load(conn, table):
    """Interpolate a table name into text()."""
    return conn.execute(text(f"SELECT * FROM {table}"))
'''

_PERCENT_INTO_EXECUTE = '''
def load(cursor, ticker):
    """Old-style percent formatting straight into execute()."""
    return cursor.execute("SELECT close FROM price_bar WHERE ticker = '%s'" % ticker)
'''

_CONCAT_INTO_DRIVER_SQL = '''
def load(conn, where):
    """String concatenation into the driver-level escape hatch."""
    return conn.exec_driver_sql("SELECT * FROM security WHERE " + where)
'''

_FORMAT_INTO_OP_EXECUTE = '''
from alembic import op

def upgrade():
    """str.format into a migration execute()."""
    op.execute("DELETE FROM {} WHERE id = 1".format(user_supplied))
'''

_ASSIGNED_THEN_EXECUTED = '''
def load(conn, column):
    """Built into a local first, executed later — the two-step shape."""
    statement = f"SELECT {column} FROM price_bar"
    return conn.execute(statement)
'''

_TRIPLE_QUOTED_FSTRING = '''
def load(conn, table):
    """A multi-line f-string is no different."""
    return conn.execute(f"""
        SELECT *
        FROM {table}
    """)
'''


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("f-string into text()", _FSTRING_INTO_TEXT),
        ("percent format into execute()", _PERCENT_INTO_EXECUTE),
        ("concatenation into exec_driver_sql()", _CONCAT_INTO_DRIVER_SQL),
        ("str.format into op.execute()", _FORMAT_INTO_OP_EXECUTE),
        ("assigned first, executed later", _ASSIGNED_THEN_EXECUTED),
        ("multi-line f-string", _TRIPLE_QUOTED_FSTRING),
    ],
)
def test_synthetic_violation_is_flagged(label: str, source: str) -> None:
    """Each interpolation shape a developer might reach for is caught."""
    assert scan_source(source, "synthetic.py"), label


def test_synthetic_violation_in_the_tree_is_flagged(tmp_path: Path) -> None:
    """A violation written into a scanned tree is reported by the tree scan.

    Proves the whole path — walk, parse, report — not just the parser, and
    that the clean-tree assertion above would actually fail if the tree were
    dirty.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "queries.py").write_text(_FSTRING_INTO_TEXT, encoding="utf-8")
    (package / "clean.py").write_text(_PARAMETERIZED_CLEAN, encoding="utf-8")

    findings = scan_tree(package)

    assert {f.path for f in findings} == {"queries.py"}


def test_excluded_directories_are_skipped(tmp_path: Path) -> None:
    """A violation under an excluded directory is not reported."""
    package = tmp_path / "pkg"
    (package / "tests").mkdir(parents=True)
    (package / "tests" / "probe.py").write_text(_FSTRING_INTO_TEXT, encoding="utf-8")

    assert not scan_tree(package, excluded_directories=frozenset({"tests"}))


# --------------------------------------------------------------------------
# ... and clean code must not be flagged
# --------------------------------------------------------------------------

_PARAMETERIZED_CLEAN = '''
from sqlalchemy import bindparam, text

def load(conn, table_id):
    """Bind parameters: the correct shape."""
    statement = text("SELECT * FROM price_bar WHERE security_id = :security_id")
    return conn.execute(statement, {"security_id": table_id})

def also_fine(conn, as_of):
    """A bindparam() built from a variable is not string interpolation."""
    return conn.execute(
        text("SELECT 1 WHERE knowledge_time <= :as_of").bindparams(bindparam("as_of", as_of))
    )
'''

_GUARD_LIKE_ERROR_MESSAGES = '''
import re

def explain(tables, sql):
    """Error text and regexes that merely *mention* SQL are not queries."""
    matched = [name for name in tables if re.search(rf"\\b{re.escape(name)}\\b", sql)]
    raise RuntimeError(
        f"unsanctioned SQL naming bitemporal fact table(s) {sorted(matched)}: this "
        f"execution did not come from the versioned query layer. INSERT INTO fact "
        f"... SELECT ... FROM fact is a read and is refused."
    )

def report(count):
    """Percent formatting outside SQL is not a finding."""
    return "%d rows" % count
'''


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("parameterized queries", _PARAMETERIZED_CLEAN),
        ("guard-style error messages and regexes", _GUARD_LIKE_ERROR_MESSAGES),
    ],
)
def test_clean_source_is_not_flagged(label: str, source: str) -> None:
    """Correct code and SQL-shaped prose produce no findings."""
    assert not scan_source(source, "clean.py"), label


def test_the_real_guard_modules_produce_no_findings() -> None:
    """The existing bypass-guard code is not a false positive.

    ``backend/db/_guard.py`` and ``backend/db/asof.py`` contain more
    SQL-shaped string building than anything else in the repository. If the
    check cannot tell their diagnostics apart from a query, it is too blunt
    to keep.
    """
    noisy = {"backend/db/_guard.py", "backend/db/asof.py"}
    assert not {f for f in application_findings() if f.path in noisy}
