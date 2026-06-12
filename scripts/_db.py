"""Connection helpers shared by the data-layer scripts and integration tests.

The DSN is assembled from environment variables (see `.env.example`) so the same
code points at the local docker-compose Postgres (loopback `127.0.0.1:5436`) or
any other instance without edits. `psycopg` is imported lazily inside
`connect()` so the pure parts of this module — and the unit tests that exercise
`conn_params()` — do not require the `ingest` dependency group to be installed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from psycopg import Connection

# Defaults match docker-compose.yml: the pgvector service publishes 5432 inside
# the container on the host's loopback 5436, with dev/dev/dev credentials.
_DEFAULTS = {
    "host": "127.0.0.1",
    "port": "5436",
    "dbname": "dev",
    "user": "dev",
    "password": "dev",
}


def conn_params(*, user: str | None = None, password: str | None = None) -> dict[str, str]:
    """Build libpq connection keywords from the environment.

    `user` / `password` overrides let a caller connect as the SELECT-only
    `router_ro` role without a second set of environment variables.
    """
    params = {
        "host": os.environ.get("POSTGRES_HOST", _DEFAULTS["host"]),
        "port": os.environ.get("POSTGRES_PORT", _DEFAULTS["port"]),
        "dbname": os.environ.get("POSTGRES_DB", _DEFAULTS["dbname"]),
        "user": os.environ.get("POSTGRES_USER", _DEFAULTS["user"]),
        "password": os.environ.get("POSTGRES_PASSWORD", _DEFAULTS["password"]),
    }
    if user is not None:
        params["user"] = user
    if password is not None:
        params["password"] = password
    return params


def router_ro_credentials() -> tuple[str, str]:
    """Login name + password for the read-only `router_ro` role."""
    return (
        os.environ.get("ROUTER_RO_USER", "router_ro"),
        os.environ.get("ROUTER_RO_PASSWORD", "router_ro_dev"),
    )


def connect(*, user: str | None = None, password: str | None = None) -> Connection[Any]:
    """Open a psycopg connection using `conn_params()`.

    Imported lazily so importing this module stays free of the `ingest` deps.
    The params are folded into a conninfo string (rather than spread as kwargs)
    so they reach libpq as connection parameters, not psycopg's typed options.
    """
    import psycopg
    from psycopg.conninfo import make_conninfo

    conninfo = make_conninfo(**conn_params(user=user, password=password))
    return psycopg.connect(conninfo)
