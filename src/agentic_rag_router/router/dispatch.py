"""The dispatcher --- turns a model tool call into an executed adapter result.

`run_router` (`loop.py`) hands each ``tool_use`` block's name and input to
`Dispatcher.dispatch`, which routes it to the matching T003 adapter
(`tools/`), runs it, and packages the outcome as a `DispatchOutcome`: the JSON
string that becomes the model's ``tool_result`` content, plus the bookkeeping
the loop records in its trajectory and citations.

Failure handling has two layers:

- An adapter's *operational* failure (rejected SQL, DB error, HTTP error) comes
  back as a `ToolResult` with ``ok=False`` --- serialized as an error
  ``tool_result`` (``is_error=True``) so the model can react and the loop
  continues.
- A *malformed* tool call (the model omits ``query``/``sql``, or supplies an
  empty value the adapter rejects with ``ValueError``) is caught here and
  turned into the same error shape, so a bad model call can never crash the
  loop.

An unknown tool name returns an error ``tool_result`` too --- the loop keeps
going rather than aborting.

The adapters themselves are untouched by D4 (the descriptions are the tunable
surface). The JSON serialization uses ``default=str`` so non-JSON row values
from the real substrates (``date``/``Decimal``) degrade to strings instead of
raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentic_rag_router.tools.envelope import (
    ERROR_VALIDATION,
    TOOL_SQL_QUERY,
    TOOL_VECTOR_SEARCH,
    TOOL_WEB_SEARCH,
    ToolResult,
)
from agentic_rag_router.tools.sql_query import SqlExecutor, sql_query
from agentic_rag_router.tools.vector_search import (
    DEFAULT_K,
    EmbedderPort,
    VectorRepository,
    vector_search,
)
from agentic_rag_router.tools.web_search import (
    DEFAULT_MAX_RESULTS,
    WebSearchClient,
    web_search,
)

# Error code for a tool name the dispatcher does not recognise.
ERROR_UNKNOWN_TOOL = "unknown_tool"


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    """Everything the loop needs from one dispatched tool call.

    Parameters
    ----------
    content:
        The JSON string handed to the model as the ``tool_result`` content ---
        the rows on success, the error code/message on failure.
    is_error:
        ``True`` when ``content`` describes a failure (sets the ``tool_result``
        ``is_error`` flag so the model treats it as such).
    ok:
        Whether the underlying tool succeeded (drives the trajectory step).
    error_code:
        Machine-readable failure code, or ``None`` on success.
    latency_ms:
        Adapter-measured wall-clock duration (0 for failures detected before an
        adapter ran, e.g. an unknown tool or a malformed call).
    citations:
        Source identifiers for successful evidence; empty on failure.
    """

    content: str
    is_error: bool
    ok: bool
    error_code: str | None
    latency_ms: int
    citations: list[dict[str, object]]


class Dispatcher:
    """Routes tool-use calls to the T003 adapters and packages their results.

    Holds the substrate ports each adapter needs (an embedder + vector
    repository for `vector_search`, a SQL executor for `sql_query`, and an
    optional web-search client for `web_search` --- omitted, the adapter builds
    a real Tavily client from the environment). The dispatcher is constructed
    once at composition time and reused for every tool call in a run.
    """

    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        repository: VectorRepository,
        executor: SqlExecutor,
        web_client: WebSearchClient | None = None,
    ) -> None:
        self._embedder = embedder
        self._repository = repository
        self._executor = executor
        self._web_client = web_client

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> DispatchOutcome:
        """Execute the tool ``name`` with ``tool_input`` and package the outcome."""
        try:
            if name == TOOL_VECTOR_SEARCH:
                result = vector_search(
                    tool_input["query"],
                    int(tool_input.get("k", DEFAULT_K)),
                    embedder=self._embedder,
                    repository=self._repository,
                )
            elif name == TOOL_SQL_QUERY:
                result = sql_query(tool_input["sql"], executor=self._executor)
            elif name == TOOL_WEB_SEARCH:
                result = web_search(
                    tool_input["query"],
                    int(tool_input.get("max_results", DEFAULT_MAX_RESULTS)),
                    client=self._web_client,
                )
            else:
                return self._unknown_tool(name)
        except (ValueError, KeyError) as exc:
            # Malformed model call (missing key, empty query/sql). Treat as a
            # validation failure the model can recover from --- never a crash.
            return self._malformed(exc)

        return self._from_result(result)

    @staticmethod
    def _unknown_tool(name: str) -> DispatchOutcome:
        content = json.dumps({"ok": False, "error_code": ERROR_UNKNOWN_TOOL, "tool": name})
        return DispatchOutcome(
            content=content,
            is_error=True,
            ok=False,
            error_code=ERROR_UNKNOWN_TOOL,
            latency_ms=0,
            citations=[],
        )

    @staticmethod
    def _malformed(exc: Exception) -> DispatchOutcome:
        content = json.dumps(
            {"ok": False, "error_code": ERROR_VALIDATION, "error_message": str(exc)}
        )
        return DispatchOutcome(
            content=content,
            is_error=True,
            ok=False,
            error_code=ERROR_VALIDATION,
            latency_ms=0,
            citations=[],
        )

    def _from_result(self, result: ToolResult) -> DispatchOutcome:
        if result.ok:
            content = json.dumps(
                {"ok": True, "tool": result.tool, "data": result.data},
                default=str,
            )
            return DispatchOutcome(
                content=content,
                is_error=False,
                ok=True,
                error_code=None,
                latency_ms=result.latency_ms,
                citations=self._citations(result),
            )
        content = json.dumps(
            {
                "ok": False,
                "tool": result.tool,
                "error_code": result.error_code,
                "error_message": result.error_message,
            },
            default=str,
        )
        return DispatchOutcome(
            content=content,
            is_error=True,
            ok=False,
            error_code=result.error_code,
            latency_ms=result.latency_ms,
            citations=[],
        )

    @staticmethod
    def _citations(result: ToolResult) -> list[dict[str, object]]:
        """Source identifiers for a successful result.

        Simple version (D5 refines): vector and web evidence cite one source per
        row (the arXiv id / the URL); a SQL aggregate cites the table it ran
        against, once, when it produced any rows.
        """
        data = result.data or []
        if result.tool == TOOL_VECTOR_SEARCH:
            return [
                {
                    "tool": TOOL_VECTOR_SEARCH,
                    "source": row.get("arxiv_id"),
                    "title": row.get("title"),
                }
                for row in data
            ]
        if result.tool == TOOL_WEB_SEARCH:
            return [
                {"tool": TOOL_WEB_SEARCH, "source": row.get("url"), "title": row.get("title")}
                for row in data
            ]
        # sql_query: one citation for the table, only when there is a result.
        if data:
            return [{"tool": TOOL_SQL_QUERY, "source": "taxi_trips"}]
        return []
