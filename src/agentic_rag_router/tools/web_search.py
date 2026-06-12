"""The `web_search` tool --- a Tavily REST call over httpx.

Covers the `web_only` class: facts dated after the corpus cutoff, which the
vector and SQL substrates cannot answer. Tavily returns ranked web results;
this adapter maps each to a uniform title/url/snippet/published shape.

`httpx` is a first-class project dependency (no lazy import needed). The API
key is sent in the `Authorization: Bearer` header, never in the URL or body,
so recorded VCR cassettes scrub it with a single `filter_headers` rule. Replay
tests run with a dummy key --- nothing validates the key locally, the cassette
supplies the response.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import httpx

from agentic_rag_router.tools.envelope import (
    ERROR_HTTP,
    TOOL_WEB_SEARCH,
    ToolResult,
    error_result,
    ok_result,
)

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
DEFAULT_TIMEOUT_S = 30.0


class WebSearchClient(Protocol):
    """Performs a web search and returns ranked results as dicts."""

    def search(self, query: str, max_results: int) -> list[dict[str, object]]:
        """Return up to `max_results` results, each title/url/snippet/published."""
        ...


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    client: WebSearchClient | None = None,
) -> ToolResult:
    """Search the web via Tavily and return ranked results.

    `query` must be non-empty (a caller bug otherwise --- raises `ValueError`).
    An HTTP failure (non-2xx, connection refused, timeout) is an operational
    failure and comes back as a `ToolResult` with ``ok=False`` and
    `ERROR_HTTP`. When `client` is omitted a real `TavilyClient` is built from
    the environment.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty or whitespace-only")

    active_client = client if client is not None else TavilyClient()

    start = time.perf_counter()
    try:
        results = active_client.search(query, max_results)
    except httpx.HTTPError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return error_result(TOOL_WEB_SEARCH, ERROR_HTTP, str(exc), latency_ms)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ok_result(TOOL_WEB_SEARCH, results, latency_ms)


class TavilyClient:
    """`WebSearchClient` backed by the Tavily `/search` REST endpoint.

    The key is read from the `TAVILY_API_KEY` environment variable unless one
    is passed explicitly (the replay tests pass a dummy). Each Tavily result's
    ``content`` becomes ``snippet`` and ``published_date`` becomes
    ``published`` (absent for non-news results, so it may be ``None``).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        self._timeout_seconds = timeout_seconds

    def search(self, query: str, max_results: int) -> list[dict[str, object]]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = {"query": query, "max_results": max_results}
        response = httpx.post(TAVILY_URL, json=body, headers=headers, timeout=self._timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content"),
                "published": item.get("published_date"),
            }
            for item in payload.get("results", [])
        ]
