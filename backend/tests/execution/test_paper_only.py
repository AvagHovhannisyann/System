"""Paper-only is structural: proved against this package's own source, not asserted in prose.

Directive §1.1 and §9.5: "No code path may place a real order. This is not
configurable." A test that merely checked a flag was off would be testing the
flag. These tests check that there is nothing to configure — no transport, no
routing seam, no venue parameter, and no representation for a live order or a
live fill — by tokenizing and parsing every module in ``backend/execution``.

The tokenizer drops strings, f-string text and comments, so a forbidden word in a
docstring cannot pass or fail anything: only names that are *code* count.
"""

from __future__ import annotations

import ast
import inspect
import io
import re
import token as token_module
import tokenize
from pathlib import Path
from typing import Final, cast

import pytest
import sqlalchemy as sa

import backend.execution
from backend.execution import idempotency, lifecycle, orders, store
from backend.execution.orders import ExecutionVenue, FillSource

PACKAGE_ROOT: Final = Path(backend.execution.__file__).parent
MODULE_PATHS: Final = sorted(PACKAGE_ROOT.glob("*.py"))

CODE_TOKENS: Final = frozenset({token_module.NAME, token_module.NUMBER, token_module.OP})

# Names that would constitute, or be needed to build, a connection to anything.
FORBIDDEN_NAMES: Final = frozenset(
    {
        "aiohttp",
        "connect",
        "connectAsync",
        "endpoint",
        "environ",
        "gateway",
        "getenv",
        "host",
        "hostname",
        "httpx",
        "ib_insync",
        "live",
        "os",
        "port",
        "create_connection",
        "open_connection",
        "requests",
        "socket",
        "start_server",
        "ssl",
        "subprocess",
        "tws",
        "url",
        "urllib",
        "urlopen",
        "websocket",
        "websockets",
        # A routing seam is as good as an endpoint: something a config could
        # select an implementation into.
        "ABC",
        "ABCMeta",
        "Callable",
        "Protocol",
        "abstractmethod",
        "get_settings",
        "import_module",
        "entry_points",
    }
)

# Ports that identify a broker gateway. None should appear: there is no
# transport at all, so neither the live pair nor the paper pair belongs here.
FORBIDDEN_NUMBERS: Final = frozenset({"7496", "7497", "4001", "4002"})

# Method names that would make this package a router rather than a ledger.
FORBIDDEN_CALLABLE_NAMES: Final = frozenset(
    {
        "submit",
        "submit_order",
        "send",
        "send_order",
        "place",
        "place_order",
        "route",
        "route_order",
        "transmit",
        "dispatch",
        "connect",
        "disconnect",
        "request",
    }
)

# Parameter names through which a venue, a client or an endpoint could be
# injected into an otherwise transport-free package.
FORBIDDEN_PARAMETERS: Final = frozenset(
    {
        "venue",
        "adapter",
        "broker",
        "client",
        "connection_url",
        "endpoint",
        "gateway",
        "host",
        "port",
        "transport",
        "url",
        "settings",
        "live",
    }
)

IMPORT_ALLOWLIST: Final = frozenset(
    {
        "__future__",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "hashlib",
        "types",
        "typing",
        # The database session type, and only that. `asyncio` is not on the
        # forbidden-name list solely because it is a component of this module
        # path; the test below pins it to exactly that use.
        "sqlalchemy",
        "sqlalchemy.exc",
        "sqlalchemy.ext.asyncio",
        "sqlalchemy.sql",
    }
)


def _code_tokens(path: Path) -> list[tokenize.TokenInfo]:
    """Return only the NAME/NUMBER/OP tokens of a module — never strings or comments."""
    source = path.read_text(encoding="utf-8")
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [item for item in tokens if item.type in CODE_TOKENS]


def _trees() -> list[tuple[Path, ast.Module]]:
    """Parse every module in the package."""
    return [(path, ast.parse(path.read_text(encoding="utf-8"))) for path in MODULE_PATHS]


def test_the_package_has_modules_to_check() -> None:
    # Guards the whole file against becoming vacuous by globbing nothing.
    names = {path.name for path in MODULE_PATHS}
    assert names == {
        "__init__.py",
        "errors.py",
        "halt.py",
        "idempotency.py",
        "killswitch.py",
        "lifecycle.py",
        "orders.py",
        "reconciliation.py",
        "store.py",
    }


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda path: path.name)
def test_no_module_names_anything_that_could_reach_a_venue(path: Path) -> None:
    used = {item.string for item in _code_tokens(path) if item.type == token_module.NAME}
    assert used & FORBIDDEN_NAMES == set(), sorted(used & FORBIDDEN_NAMES)


def test_asyncio_appears_only_inside_the_sqlalchemy_session_module_path() -> None:
    # `asyncio` can open connections, so its presence is checked rather than
    # allowed outright: the only occurrence permitted is the `sqlalchemy.ext.
    # asyncio` import path, which yields a database session and nothing else.
    for path, tree in _trees():
        source = path.read_text(encoding="utf-8")
        if "asyncio" not in source:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert "asyncio" not in " ".join(alias.name for alias in node.names), path.name
        occurrences = source.count("asyncio")
        assert occurrences == source.count("sqlalchemy.ext.asyncio"), path.name


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda path: path.name)
def test_no_module_contains_a_broker_gateway_port(path: Path) -> None:
    numbers = {item.string for item in _code_tokens(path) if item.type == token_module.NUMBER}
    assert numbers & FORBIDDEN_NUMBERS == set()


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda path: path.name)
def test_no_module_contains_a_url_anywhere_including_strings(path: Path) -> None:
    # Scanned over the raw text, not the token stream: a URL hidden in a string
    # would still be a configured destination if anything could use it.
    assert "://" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda path: path.name)
def test_every_import_is_stdlib_sqlalchemy_or_this_repository(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in IMPORT_ALLOWLIST, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module in IMPORT_ALLOWLIST or module.startswith("backend."), module


def test_no_module_imports_application_settings() -> None:
    # Nothing here reads configuration, so no configuration can point it
    # anywhere. This is the load-bearing half of "not configurable".
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "config" not in (node.module or ""), (path.name, node.module)


def test_the_package_defines_no_interface_an_adapter_could_be_selected_into() -> None:
    # No Protocol, no ABC, no abstract method: there is no seam where a live
    # implementation could be registered by configuration.
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {ast.unparse(base) for base in node.bases}
                assert not bases & {"Protocol", "ABC"}, (path.name, node.name)


def test_no_function_or_method_is_named_like_a_router() -> None:
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.name not in FORBIDDEN_CALLABLE_NAMES, (path.name, node.name)


def test_no_function_takes_a_venue_client_or_endpoint_parameter() -> None:
    # The other way an endpoint arrives: not as a name in this package, but as
    # something handed in. Every argument of every function is checked.
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            arguments = node.args
            names = {
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            }
            for extra in (arguments.vararg, arguments.kwarg):
                if extra is not None:
                    names.add(extra.arg)
            assert names & FORBIDDEN_PARAMETERS == set(), (path.name, node.name, names)


def test_no_public_callable_signature_admits_a_venue() -> None:
    # The same claim checked through the runtime objects rather than the source,
    # so a re-export or a decorator cannot smuggle one past the AST scan.
    for module in (orders, lifecycle, idempotency, store):
        for name in module.__all__:
            candidate = getattr(module, name)
            if not callable(candidate) or isinstance(candidate, type):
                continue
            parameters = set(inspect.signature(candidate).parameters)
            assert parameters & FORBIDDEN_PARAMETERS == set(), (module.__name__, name)


def test_the_venue_enum_has_exactly_one_member() -> None:
    # "Live" is not a disabled option, it is an absent one. A second member is a
    # code change and a migration, never a configuration value.
    assert len(ExecutionVenue) == 1
    assert list(ExecutionVenue) == [ExecutionVenue.PAPER]
    assert ExecutionVenue.PAPER.value == "paper"


def test_no_venue_member_could_denote_a_live_destination() -> None:
    for member in ExecutionVenue:
        assert not re.search(r"live|real|prod", member.value, re.IGNORECASE)
        assert not re.search(r"LIVE|REAL|PROD", member.name)


def test_the_fill_source_enum_admits_no_live_execution() -> None:
    # I3: a simulated fill must not be mistakable for a broker fill, and neither
    # may be mistakable for a real one. There is no member for the third case.
    assert {member.value for member in FillSource} == {"simulated", "paper_broker"}
    for member in FillSource:
        assert not re.search(r"live|real_money|production", member.value, re.IGNORECASE)


def test_the_schema_pins_the_venue_and_the_fill_source() -> None:
    # The database restates what the types make true, so a writer that skips
    # Python entirely is still bound.
    from backend.db import models

    order_table = cast("sa.Table", models.ExecutionOrder.__table__)
    order_checks = {
        str(constraint.name): str(constraint.sqltext)
        for constraint in order_table.constraints
        if isinstance(constraint, sa.CheckConstraint)
    }
    assert order_checks["ck_execution_order_venue_is_paper"] == "venue = 'paper'"
    transition_table = cast("sa.Table", models.ExecutionOrderTransition.__table__)
    transition_checks = {
        str(constraint.name): str(constraint.sqltext)
        for constraint in transition_table.constraints
        if isinstance(constraint, sa.CheckConstraint)
    }
    source_check = transition_checks["ck_execution_order_transition_fill_source_is_not_live"]
    assert "'simulated'" in source_check
    assert "'paper_broker'" in source_check
    basis_check = transition_checks["ck_execution_order_transition_fill_cost_basis_is_lower_bound"]
    assert "'lower_bound'" in basis_check


def test_the_order_table_has_no_writable_venue_path() -> None:
    # The column exists, carries a server default, and no code supplies it.
    from backend.db import models

    column = cast("sa.Table", models.ExecutionOrder.__table__).columns["venue"]
    default = column.server_default
    assert default is not None
    assert "paper" in str(getattr(default, "arg", default))
    # No writer names the column: `stored_venue` (the value read back and
    # refused) is deliberately a different identifier, so this scan cannot be
    # satisfied by a rename.
    source = (PACKAGE_ROOT / "store.py").read_text(encoding="utf-8")
    assert re.search(r"(?<![\w])venue\s*=", source) is None
