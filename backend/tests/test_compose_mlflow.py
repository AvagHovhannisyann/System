"""Structural checks on the MLflow service in ``docker-compose.yml`` (CC.3, I2).

What these tests prove, and what they do not
--------------------------------------------

They parse the committed compose file and assert that the tracking server is
declared the way the backend expects to reach it. Nothing is started: no image
is pulled, no container runs, no HTTP request is made. A green run here means
*the declaration is coherent*, not that the stack comes up. Specifically still
unproven by anything in this file: that the pinned image pulls, that the
container can write to the mounted volume, that compose's dependency ordering
resolves, and that the backend reaches the server across the compose network.
Those need a Docker daemon. Naming them is the point — a test that implies more
than it checks is worse than no test (I6).

Two assertions here exist because of failure modes that are silent rather than
loud. Both were found by reading MLflow 3.15's server code, and the host-header
behaviour was then confirmed against a server run directly from this
repository's virtualenv with the same flags:

* ``--allowed-hosts``. MLflow 3 validates the ``Host`` header against localhost
  and private IP ranges and answers 403 to anything else. ``mlflow:5000`` is a
  hostname, so every backend API call would be refused — while ``/health`` is
  exempt from that check, meaning the container would still report *healthy*.
  A healthcheck cannot catch this; a text assertion can.
* The store paths. A named volume proves nothing if the SQLite file and the
  artifact root are written somewhere else in the container: history would
  vanish on the next ``docker compose down``, which is the opposite of what a
  tracking server is for.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import yaml
from mlflow.environment_variables import MLFLOW_TRACKING_URI

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"
_LOCK_PATH = _REPO_ROOT / "uv.lock"

_SERVICE = "mlflow"
_BACKEND_SERVICE = "backend"
_PORT = 5000
_SQLITE_PREFIX = "sqlite:///"


def _compose() -> dict[str, Any]:
    """Return the parsed compose file."""
    document: Any = yaml.safe_load(_COMPOSE_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{_COMPOSE_PATH.name} is not a YAML mapping"
    return document


def _service(name: str) -> dict[str, Any]:
    """Return service ``name`` from the compose file."""
    services = _compose().get("services")
    assert isinstance(services, dict), f"{_COMPOSE_PATH.name} declares no services mapping"
    service = services.get(name)
    assert isinstance(service, dict), f"{_COMPOSE_PATH.name} declares no {name!r} service"
    return service


def _command_options(service: dict[str, Any]) -> dict[str, str]:
    """Return the ``--flag=value`` options of a service's command.

    Accepts both ``--flag=value`` and ``--flag value`` spellings so the check
    survives a reformatting of the command list.

    Args:
        service: a parsed compose service.

    Returns:
        Mapping of option name (without leading dashes) to value. Flags with no
        value map to the empty string.
    """
    command = service.get("command")
    assert isinstance(command, list), "the mlflow command must be a list, so it is not shell-split"
    tokens = [str(token) for token in command]
    options: dict[str, str] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token.startswith("--"):
            continue
        name, separator, value = token[2:].partition("=")
        if separator:
            options[name] = value
        elif index < len(tokens) and not tokens[index].startswith("--"):
            options[name] = tokens[index]
            index += 1
        else:
            options[name] = ""
    return options


def _volume_mounts(service: dict[str, Any]) -> dict[str, str]:
    """Return a service's volume mounts as ``{source: target}``.

    Args:
        service: a parsed compose service.

    Returns:
        Mapping of volume name (or host path) to container mount point.
    """
    mounts: dict[str, str] = {}
    for entry in service.get("volumes", []):
        if isinstance(entry, str):
            source, _, rest = entry.partition(":")
            target, _, _ = rest.partition(":")
            mounts[source] = target
        elif isinstance(entry, dict):
            mounts[str(entry.get("source"))] = str(entry.get("target"))
    return mounts


def _locked_version(package: str) -> str:
    """Return the version ``uv.lock`` resolves ``package`` to."""
    lock: dict[str, Any] = tomllib.loads(_LOCK_PATH.read_text(encoding="utf-8"))
    versions = [
        str(entry["version"]) for entry in lock.get("package", []) if entry.get("name") == package
    ]
    assert versions, f"{package} is not in uv.lock"
    return versions[0]


def test_compose_file_parses_and_declares_the_expected_services() -> None:
    """Guard the guard: every check below is vacuous if the file does not parse."""
    services = _compose()["services"]
    for name in (_SERVICE, _BACKEND_SERVICE):
        assert name in services, f"{name!r} missing from {_COMPOSE_PATH.name}"


def test_mlflow_image_is_pinned_to_the_locked_client_version() -> None:
    """The server image must pin the exact version the client lock resolves to.

    Client and server share the backing-store schema. A floating tag would let a
    ``docker compose pull`` migrate the store out from under the pinned client,
    and a drifted pin is the same hazard one release later.
    """
    image = str(_service(_SERVICE)["image"])
    # rpartition, not partition: a registry host may itself carry a port. A "/"
    # in the result means the colon belonged to that host, not to a tag.
    _, separator, tag = image.rpartition(":")
    untagged = f"mlflow image {image!r} has no tag; an unpinned server is not reproducible"
    assert separator, untagged
    assert "/" not in tag, untagged
    assert tag not in {"latest", "master"}, f"mlflow image tag {tag!r} is not a pin"
    assert tag == f"v{_locked_version('mlflow')}", (
        f"mlflow server image is {tag} but uv.lock resolves the client to "
        f"{_locked_version('mlflow')}; bump both together"
    )


def test_mlflow_has_a_healthcheck_against_its_own_health_endpoint() -> None:
    """Dependency ordering is only as good as the check it waits on."""
    healthcheck = _service(_SERVICE).get("healthcheck")
    assert isinstance(healthcheck, dict), "the mlflow service declares no healthcheck"
    probe = " ".join(str(part) for part in healthcheck["test"])
    assert "/health" in probe, f"healthcheck does not call /health: {probe}"
    assert f":{_PORT}" in probe, f"healthcheck does not target port {_PORT}: {probe}"
    for field in ("interval", "timeout", "retries"):
        assert field in healthcheck, f"healthcheck is missing {field}"


def test_mlflow_state_is_written_inside_a_persistent_named_volume() -> None:
    """Runs and artifacts must both land on a declared named volume.

    A tracking server whose history disappears with the container cannot support
    I2: a stamped result nobody can look up again is not a record of anything.
    """
    service = _service(_SERVICE)
    mounts = _volume_mounts(service)
    assert mounts, "the mlflow service mounts nothing, so its history dies with the container"

    declared = _compose().get("volumes") or {}
    for source, target in mounts.items():
        assert source in declared, f"volume {source!r} is not declared at the top level"
        assert target.startswith("/"), f"mount target {target!r} is not an absolute path"

    options = _command_options(service)
    backend_store = options["backend-store-uri"]
    assert backend_store.startswith(_SQLITE_PREFIX), (
        f"backend store {backend_store!r} is not the sqlite backend this stack pins"
    )
    store_path = backend_store[len(_SQLITE_PREFIX) :]
    artifacts_path = options["artifacts-destination"]
    targets = tuple(mounts.values())
    for label, path in (("backend store", store_path), ("artifacts destination", artifacts_path)):
        assert path.startswith("/"), f"{label} {path!r} is not an absolute path"
        assert any(path.startswith(f"{target}/") for target in targets), (
            f"{label} {path!r} is outside the mounted volumes {targets}; it would not survive "
            "the container being replaced"
        )


def test_backend_tracking_uri_addresses_the_mlflow_service() -> None:
    """The backend's tracking URI must name the service and the port it serves.

    The variable name is taken from the installed MLflow client rather than
    typed in, so this cannot pass by asserting a plausible-looking string the
    client never reads.
    """
    environment = _service(_BACKEND_SERVICE)["environment"]
    assert isinstance(environment, dict), "backend environment must be a mapping"
    raw = environment.get(MLFLOW_TRACKING_URI.name)
    assert raw is not None, (
        f"backend has no {MLFLOW_TRACKING_URI.name}; the MLflow client would fall back to a "
        "local ./mlruns store inside the container and every run would be lost"
    )
    parsed = urlparse(str(raw))
    assert parsed.scheme == "http", f"tracking URI {raw!r} is not http"
    assert parsed.hostname == _SERVICE, f"tracking URI {raw!r} does not name the {_SERVICE} service"
    assert parsed.port == _PORT, f"tracking URI {raw!r} does not use port {_PORT}"

    served = _command_options(_service(_SERVICE))
    assert served["port"] == str(_PORT), "the server listens on a port the client does not call"
    # S104 (bind-all-interfaces) is the intended configuration here: the server
    # has to accept connections from the compose network, and the published
    # port is the deliberate, documented exposure.
    assert served["host"] == "0.0.0.0", (  # noqa: S104
        "the server binds loopback only, so nothing on the compose network can reach it"
    )


def test_mlflow_accepts_the_host_header_the_backend_will_send() -> None:
    """``--allowed-hosts`` must list the address the backend calls it by.

    MLflow 3 answers 403 to an unlisted ``Host``, and passing this flag replaces
    the built-in defaults rather than extending them — so omitting either the
    service name or localhost silently breaks one of the two callers.
    """
    allowed = _command_options(_service(_SERVICE))["allowed-hosts"].split(",")
    entries = [entry.strip() for entry in allowed]
    assert f"{_SERVICE}:{_PORT}" in entries, (
        f"{_SERVICE}:{_PORT} is not in --allowed-hosts {entries}; the backend's calls would be "
        "rejected with 403 while /health kept reporting the container healthy"
    )
    assert any(entry.startswith("localhost") for entry in entries), (
        f"no localhost entry in --allowed-hosts {entries}; the published port would be unusable "
        "from a browser on the host"
    )


@pytest.mark.parametrize("dependency", ["db", "redis", _SERVICE])
def test_backend_waits_for_its_dependencies_to_be_healthy(dependency: str) -> None:
    """The backend must not start before the services it records into are up."""
    depends_on = _service(_BACKEND_SERVICE)["depends_on"]
    assert dependency in depends_on, f"backend does not depend on {dependency}"
    assert depends_on[dependency]["condition"] == "service_healthy", (
        f"backend waits for {dependency} to start, not to be healthy"
    )
