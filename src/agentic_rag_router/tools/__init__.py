"""Tool adapters the router routes across (D3).

Each adapter wraps exactly one retrieval substrate and returns the shared
`ToolResult` envelope --- never a raw exception for an operational failure.
The router (D4) and evidence grading (D5) consume these envelopes; neither
lives here.

Three adapters ship in this package:

- `vector_search` --- cosine top-k over the `corpus_docs` pgvector table,
  embedding the query with the same pinned MiniLM model as ingestion.
- `sql_query` --- validates a single SELECT (front door) and executes it as
  the read-only `router_ro` role with a statement timeout and a row cap.
- `web_search` --- a Tavily REST call over `httpx`.

Heavy substrate dependencies (`psycopg`, `sentence-transformers`) are
lazy-imported inside the concrete adapters so importing this package --- and
the unit tests that exercise the orchestration logic with fakes --- stays
free of the `ingest` dependency group. See the package modules for the
port Protocols each tool depends on.
"""

from __future__ import annotations

from agentic_rag_router.tools.envelope import ToolResult
from agentic_rag_router.tools.sql_query import sql_query
from agentic_rag_router.tools.vector_search import vector_search
from agentic_rag_router.tools.web_search import web_search

__all__ = [
    "ToolResult",
    "sql_query",
    "vector_search",
    "web_search",
]
