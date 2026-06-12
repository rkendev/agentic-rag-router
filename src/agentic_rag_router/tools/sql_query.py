"""The `sql_query` tool --- validated SELECT over the NYC taxi-trips table.

In D4 the routing model authors the SQL itself (the table schema lives in the
tool description). This adapter does not call another model; it takes the SQL
as a string and applies two independent layers of defence:

1. **Front door --- `validate_select`.** A single statement, SELECT-only.
   Multiple statements, CTE-wrapped writes (`WITH ... INSERT`), stacked
   queries via `;`, and comment-obfuscated writes are all rejected *before*
   any connection is opened.
2. **Backstop --- the `router_ro` role.** Execution happens as a login that
   was granted `SELECT` and nothing else (see `scripts/init_db.py`). Even if
   a crafted statement slipped past the validator, the database itself
   refuses the write.

Belt and suspenders: the validator gives clear, fast rejection with a useful
message; the grant guarantees safety. Execution also carries a
``statement_timeout`` and caps the rows it returns, so a pathological but
valid SELECT (a cross join over 3M rows) can neither hang the caller nor
flood it with rows.

The validator is deliberately *conservative*: because it scans for SQL
keywords as bare words, a benign SELECT that mentions a write keyword as a
string literal or identifier (e.g. ``SELECT 'delete' AS action``) is
rejected. The routing model can rephrase, and the `router_ro` grant --- not
the keyword scan --- is what actually keeps the table safe.
"""

from __future__ import annotations

import os
import re
import time
from typing import Protocol

from agentic_rag_router.tools.envelope import (
    ERROR_BACKEND,
    ERROR_VALIDATION,
    TOOL_SQL_QUERY,
    ToolResult,
    error_result,
    ok_result,
)

# Execution guardrails. Both are caller-overridable on the real executor.
STATEMENT_TIMEOUT_MS = 5_000
ROW_CAP = 200

# Keywords that must never appear in a read-only query. Matched as whole words
# (case-insensitive) so identifiers like `updated_at` or `created_date` do not
# trip them, but `WITH x AS (INSERT ...)` and `... ; DROP TABLE ...` do. `INTO`
# is included to reject the `SELECT ... INTO new_table` create-as form.
_FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "MERGE",
    "COPY",
    "CALL",
    "DO",
    "VACUUM",
    "ANALYZE",
    "REINDEX",
    "REFRESH",
    "SET",
    "RESET",
    "LOCK",
    "COMMENT",
    "INTO",
)
_FORBIDDEN_RE = re.compile(r"\b(?:" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")
_LEADING_KEYWORD_RE = re.compile(r"[A-Za-z]+")
_ALLOWED_HEADS = frozenset({"SELECT", "WITH"})


class SqlValidationError(ValueError):
    """The SQL failed the read-only validator. Carries a human-readable reason."""


def _strip_comments(sql: str) -> str:
    """Remove block and line comments so they cannot hide a second statement.

    Replacing each comment with a space (not the empty string) keeps tokens
    that were only separated by a comment --- e.g. ``SELECT/**/1`` --- from
    fusing into one word and slipping a split keyword past the scan.
    """
    sql = _BLOCK_COMMENT_RE.sub(" ", sql)
    return _LINE_COMMENT_RE.sub(" ", sql)


def validate_select(sql: str) -> str:
    """Return the single validated SELECT statement, or raise.

    Raises
    ------
    SqlValidationError
        If `sql` is empty, contains more than one statement, does not begin
        with `SELECT`/`WITH`, or mentions a write / DDL / session keyword.
    """
    if not sql or not sql.strip():
        raise SqlValidationError("empty SQL")

    decommented = _strip_comments(sql)

    # Exactly one statement. A single trailing `;` yields one non-empty piece;
    # an interior `;` (stacked / piggy-backed query) yields two or more.
    statements = [piece for piece in decommented.split(";") if piece.strip()]
    if not statements:
        raise SqlValidationError("no executable statement after stripping comments")
    if len(statements) > 1:
        raise SqlValidationError("multiple statements are not allowed")

    statement = statements[0].strip()

    # Must read, not write. Allow a leading `(` for parenthesised SELECT/UNION.
    head_text = statement.lstrip("( \t\r\n")
    head_match = _LEADING_KEYWORD_RE.match(head_text)
    if head_match is None or head_match.group(0).upper() not in _ALLOWED_HEADS:
        raise SqlValidationError("only a single SELECT (optionally WITH ... SELECT) is allowed")

    # No write / DDL / session keyword anywhere --- catches CTE-wrapped writes.
    forbidden = _FORBIDDEN_RE.search(statement)
    if forbidden is not None:
        raise SqlValidationError(f"forbidden keyword: {forbidden.group(0).upper()}")

    return statement


class SqlExecutor(Protocol):
    """Runs a validated SELECT and returns its rows as dicts.

    Implementations own the connection, the statement timeout, and the row
    cap. The `sql_query` orchestration only ever hands them SQL that already
    passed `validate_select`.
    """

    def execute(self, sql: str) -> list[dict[str, object]]:
        """Execute `sql` and return up to the row cap as a list of row dicts."""
        ...


def sql_query(sql: str, *, executor: SqlExecutor) -> ToolResult:
    """Validate and run a read-only SQL query against the taxi-trips table.

    Returns a `ToolResult`: ``ok=False`` with `ERROR_VALIDATION` if the SQL is
    rejected at the front door, ``ok=False`` with `ERROR_BACKEND` if execution
    fails (DB error, timeout), ``ok=True`` with the rows otherwise. Never
    raises for an operational failure --- the router branches on the envelope.
    """
    start = time.perf_counter()
    try:
        validated = validate_select(sql)
    except SqlValidationError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return error_result(TOOL_SQL_QUERY, ERROR_VALIDATION, str(exc), latency_ms)

    try:
        rows = executor.execute(validated)
    except Exception as exc:  # any backend failure becomes a failed envelope, never a raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        return error_result(TOOL_SQL_QUERY, ERROR_BACKEND, str(exc), latency_ms)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ok_result(TOOL_SQL_QUERY, rows, latency_ms)


def _router_ro_conn_params() -> dict[str, str]:
    """libpq connection keywords for the read-only `router_ro` login.

    Read from the environment with the same defaults as `scripts/_db.py`. The
    tools package does not import `scripts` (it is not packaged into the
    wheel), so the handful of parameter names are repeated here rather than
    shared --- a deliberate decoupling of shippable `src/` from dev scripts.
    """
    return {
        "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
        "port": os.environ.get("POSTGRES_PORT", "5436"),
        "dbname": os.environ.get("POSTGRES_DB", "dev"),
        "user": os.environ.get("ROUTER_RO_USER", "router_ro"),
        "password": os.environ.get("ROUTER_RO_PASSWORD", "router_ro_dev"),
    }


class RouterRoExecutor:
    """`SqlExecutor` that runs as `router_ro` with a timeout and a row cap.

    `psycopg` is imported lazily so the tools package stays free of the
    `ingest` dependency group until a real query is run. The connection is
    opened per call with `statement_timeout` set as a libpq connection option
    (so no SQL is built from the timeout value), at most `row_cap` rows are
    fetched, and the read-only transaction is rolled back.
    """

    def __init__(
        self,
        *,
        statement_timeout_ms: int = STATEMENT_TIMEOUT_MS,
        row_cap: int = ROW_CAP,
    ) -> None:
        self._statement_timeout_ms = statement_timeout_ms
        self._row_cap = row_cap

    def execute(self, sql: str) -> list[dict[str, object]]:
        import psycopg
        from psycopg.conninfo import make_conninfo
        from psycopg.rows import dict_row

        # statement_timeout rides in the conninfo as a libpq option, so no SQL
        # is ever built from the timeout value (the int is guarded regardless).
        conninfo = make_conninfo(
            **_router_ro_conn_params(),
            options=f"-c statement_timeout={int(self._statement_timeout_ms)}",
        )
        with psycopg.connect(conninfo, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)  # validated single SELECT; router_ro is SELECT-only
                rows = cur.fetchmany(self._row_cap)
            conn.rollback()
        return [dict(row) for row in rows]
