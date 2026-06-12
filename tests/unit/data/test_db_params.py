"""Unit tests for the pure DSN/credential helpers (psycopg stays lazy)."""

from __future__ import annotations

import pytest

from scripts import _db


def test_conn_params_defaults_to_compose_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    pg_vars = ("HOST", "PORT", "DB", "USER", "PASSWORD")
    for key in pg_vars:
        monkeypatch.delenv(f"POSTGRES_{key}", raising=False)
    params = _db.conn_params()
    assert params == {
        "host": "127.0.0.1",
        "port": "5436",
        "dbname": "dev",
        "user": "dev",
        "password": "dev",
    }


def test_conn_params_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6000")
    params = _db.conn_params()
    assert params["host"] == "db.internal"
    assert params["port"] == "6000"


def test_conn_params_user_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_USER", "dev")
    params = _db.conn_params(user="router_ro", password="secret")
    assert params["user"] == "router_ro"
    assert params["password"] == "secret"


def test_router_ro_credentials_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROUTER_RO_USER", raising=False)
    monkeypatch.delenv("ROUTER_RO_PASSWORD", raising=False)
    assert _db.router_ro_credentials() == ("router_ro", "router_ro_dev")
