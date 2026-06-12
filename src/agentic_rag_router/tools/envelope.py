"""The shared result envelope every tool adapter returns.

`ToolResult` is the single shape the router (D4) and evidence grading (D5)
consume, regardless of which substrate answered. Operational failures
(rejected SQL, a DB timeout, an HTTP error) are returned *as* a
`ToolResult` with ``ok=False`` and a machine-readable ``error_code`` --- not
raised --- so the router can branch on the envelope without a try/except
around every tool call. Programming errors (a bad ``k``, an empty query)
still raise ``ValueError``; those are bugs in the caller, not substrate
failures to be routed around.

The evidence-grade field is intentionally absent: grading is D5 and adding a
stub now would freeze a shape before its contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tool identifiers --- the `tool` field of every envelope. Kept as constants so
# the router and the tests reference one spelling, not three string literals.
TOOL_VECTOR_SEARCH = "vector_search"
TOOL_SQL_QUERY = "sql_query"
TOOL_WEB_SEARCH = "web_search"

# Machine-readable `error_code` values. The router branches on these; the
# human-readable `error_message` carries the detail for logs and debugging.
ERROR_VALIDATION = "validation_error"  # input rejected before any I/O (e.g. non-SELECT SQL)
ERROR_BACKEND = "backend_error"  # the substrate failed (DB error, query timeout)
ERROR_HTTP = "http_error"  # an HTTP call failed (non-2xx, connection, timeout)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Uniform outcome of a single tool invocation.

    Parameters
    ----------
    ok:
        ``True`` when ``data`` holds the answer; ``False`` when the call
        failed and ``error_code`` / ``error_message`` explain why.
    tool:
        Which adapter produced this result (one of the ``TOOL_*`` constants).
    data:
        On success, the rows the tool returned (each a plain ``dict``). On
        failure, ``None``.
    error_code:
        On failure, one of the ``ERROR_*`` constants. ``None`` on success.
    error_message:
        On failure, a human-readable description. ``None`` on success.
    latency_ms:
        Wall-clock duration of the call in milliseconds, measured by the
        adapter. Present on both success and failure --- a slow failure is
        as interesting as a slow success.
    """

    ok: bool
    tool: str
    data: list[dict[str, object]] | None
    error_code: str | None
    error_message: str | None
    latency_ms: int


def ok_result(tool: str, data: list[dict[str, object]], latency_ms: int) -> ToolResult:
    """Build a successful envelope."""
    return ToolResult(
        ok=True,
        tool=tool,
        data=data,
        error_code=None,
        error_message=None,
        latency_ms=latency_ms,
    )


def error_result(tool: str, error_code: str, error_message: str, latency_ms: int) -> ToolResult:
    """Build a failed envelope."""
    return ToolResult(
        ok=False,
        tool=tool,
        data=None,
        error_code=error_code,
        error_message=error_message,
        latency_ms=latency_ms,
    )
