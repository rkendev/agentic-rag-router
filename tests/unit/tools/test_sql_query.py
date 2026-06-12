"""Unit tests for `sql_query`: the SELECT validator and the orchestration.

The validator is the security-critical surface, so its accept/reject table is
adversarial: stacked queries, CTE-wrapped writes, and comment-obfuscated
writes must all be rejected, while ordinary analytic SELECTs (including
`WITH ... SELECT` CTEs and parenthesised UNIONs) must pass.
"""

from __future__ import annotations

import pytest

from agentic_rag_router.tools.envelope import ERROR_BACKEND, ERROR_VALIDATION, TOOL_SQL_QUERY
from agentic_rag_router.tools.sql_query import (
    ROW_CAP,
    STATEMENT_TIMEOUT_MS,
    RouterRoExecutor,
    SqlValidationError,
    sql_query,
    validate_select,
)
from tests.unit.tools.fakes import FakeSqlExecutor, install_fake_psycopg

# ---------------------------------------------------------------------------
# Validator --- accepts
# ---------------------------------------------------------------------------

ACCEPTED = [
    "SELECT count(*) FROM taxi_trips",
    "select vendor_id, count(*) from taxi_trips group by vendor_id",
    "SELECT * FROM taxi_trips WHERE vendor_id = 1 LIMIT 10",
    "SELECT avg(total_amount) FROM taxi_trips WHERE passenger_count > 1",
    # trailing semicolon + whitespace is fine (single statement)
    "SELECT 1;   ",
    # WITH ... SELECT is a read-only CTE
    "WITH busy AS (SELECT vendor_id, count(*) c FROM taxi_trips GROUP BY vendor_id) "
    "SELECT * FROM busy ORDER BY c DESC",
    # parenthesised UNION of two SELECTs
    "(SELECT vendor_id FROM taxi_trips) UNION (SELECT rate_code_id FROM taxi_trips)",
    # identifiers that merely contain a keyword as a substring must not trip
    "SELECT tpep_pickup_datetime, created_at_proxy FROM taxi_trips",
    # a benign line comment is stripped, statement still valid
    "SELECT 1 -- just a count\n",
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_validate_select_accepts(sql: str) -> None:
    out = validate_select(sql)
    assert out  # returns the normalised single statement
    assert ";" not in out  # trailing semicolon stripped


# ---------------------------------------------------------------------------
# Validator --- rejects
# ---------------------------------------------------------------------------

REJECTED = [
    pytest.param("", id="empty"),
    pytest.param("   \n\t ", id="whitespace-only"),
    # non-empty input that is *only* a comment -> nothing left after stripping
    pytest.param("-- just a comment", id="line-comment-only"),
    pytest.param("/* nothing here */", id="block-comment-only"),
    pytest.param("INSERT INTO taxi_trips (vendor_id) VALUES (1)", id="insert"),
    pytest.param("UPDATE taxi_trips SET vendor_id = 2 WHERE false", id="update"),
    pytest.param("DELETE FROM taxi_trips WHERE false", id="delete"),
    pytest.param("DROP TABLE taxi_trips", id="drop"),
    pytest.param("TRUNCATE taxi_trips", id="truncate"),
    pytest.param("ALTER TABLE taxi_trips ADD COLUMN x int", id="alter"),
    pytest.param("CREATE TABLE evil (id int)", id="create"),
    pytest.param("GRANT SELECT ON taxi_trips TO public", id="grant"),
    # stacked / piggy-backed second statement
    pytest.param("SELECT 1; DROP TABLE taxi_trips", id="stacked-semicolon"),
    pytest.param("SELECT 1; SELECT 2", id="stacked-two-selects"),
    # CTE-wrapped write (data-modifying CTE)
    pytest.param(
        "WITH w AS (DELETE FROM taxi_trips RETURNING *) SELECT * FROM w",
        id="cte-delete",
    ),
    pytest.param(
        "WITH w AS (INSERT INTO taxi_trips DEFAULT VALUES RETURNING *) SELECT * FROM w",
        id="cte-insert",
    ),
    # comment-obfuscated: comment hides the stacked write, but stripping it
    # exposes the second statement
    pytest.param("SELECT 1 -- harmless\n; DROP TABLE taxi_trips", id="line-comment-stack"),
    pytest.param("SELECT/**/1;/**/DELETE FROM taxi_trips", id="block-comment-stack"),
    # split keyword via a block comment re-fuses after stripping -> caught
    pytest.param("SELECT 1; DR/**/OP TABLE taxi_trips", id="split-keyword"),
    # SELECT ... INTO creates a table
    pytest.param("SELECT * INTO evil FROM taxi_trips", id="select-into"),
    # does not start with SELECT/WITH
    pytest.param("VACUUM taxi_trips", id="vacuum"),
    pytest.param("SET statement_timeout = 0", id="set"),
    pytest.param("COPY taxi_trips TO STDOUT", id="copy"),
]


@pytest.mark.parametrize("sql", REJECTED)
def test_validate_select_rejects(sql: str) -> None:
    with pytest.raises(SqlValidationError):
        validate_select(sql)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_sql_query_rejected_returns_validation_envelope() -> None:
    executor = FakeSqlExecutor()
    result = sql_query("DROP TABLE taxi_trips", executor=executor)
    assert result.ok is False
    assert result.tool == TOOL_SQL_QUERY
    assert result.error_code == ERROR_VALIDATION
    assert result.data is None
    assert result.latency_ms >= 0
    assert executor.executed == []  # never reached the backend


def test_sql_query_success() -> None:
    rows: list[dict[str, object]] = [{"count": 2_964_624}]
    executor = FakeSqlExecutor(rows)
    result = sql_query("SELECT count(*) FROM taxi_trips", executor=executor)
    assert result.ok is True
    assert result.data == rows
    assert result.error_code is None
    assert executor.executed == ["SELECT count(*) FROM taxi_trips"]


def test_sql_query_backend_error_returns_backend_envelope() -> None:
    executor = FakeSqlExecutor(error=RuntimeError("statement timeout"))
    result = sql_query("SELECT count(*) FROM taxi_trips", executor=executor)
    assert result.ok is False
    assert result.error_code == ERROR_BACKEND
    assert result.error_message is not None
    assert "timeout" in result.error_message


# ---------------------------------------------------------------------------
# Real RouterRoExecutor over a fake psycopg
# ---------------------------------------------------------------------------


def test_router_ro_executor_runs_and_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, object]] = [{"vendor_id": 1}, {"vendor_id": 2}]
    record = install_fake_psycopg(monkeypatch, rows)
    executor = RouterRoExecutor()
    out = executor.execute("SELECT vendor_id FROM taxi_trips")

    assert out == rows
    # connected as router_ro with the default statement timeout in the conninfo
    assert record.conninfo is not None
    assert "user=router_ro" in record.conninfo
    assert f"statement_timeout={STATEMENT_TIMEOUT_MS}" in record.conninfo
    # the validated SQL was executed and the read-only txn rolled back
    assert record.cursor.executed[0][0] == "SELECT vendor_id FROM taxi_trips"
    assert record.connection.rolled_back is True


def test_router_ro_executor_custom_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    rows: list[dict[str, object]] = [{"n": i} for i in range(10)]
    record = install_fake_psycopg(monkeypatch, rows)
    executor = RouterRoExecutor(statement_timeout_ms=1234, row_cap=3)
    out = executor.execute("SELECT n FROM t")
    assert out == rows[:3]  # row cap applied via fetchmany
    assert record.conninfo is not None
    assert "statement_timeout=1234" in record.conninfo


def test_row_cap_default_is_200() -> None:
    assert ROW_CAP == 200
