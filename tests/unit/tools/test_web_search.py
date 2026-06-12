"""Unit tests for `web_search`: field mapping, error handling, cassette replay.

Field mapping and the HTTP-error path are tested by monkeypatching `httpx.post`
with crafted responses (deterministic, no network). The recorded-cassette test
replays one real Tavily call with the key scrubbed, proving the request/response
parsing works against a real payload with only a dummy key present.
"""

from __future__ import annotations

import os

import httpx
import pytest

from agentic_rag_router.tools.envelope import ERROR_HTTP, TOOL_WEB_SEARCH
from agentic_rag_router.tools.web_search import TavilyClient, web_search

# Stable query used for the recorded cassette (the body is not matched on
# replay, but a fixed string keeps re-recordings reproducible).
_TAVILY_QUERY = "What are the latest developments in retrieval-augmented generation in 2026?"


def _fake_post(payload: dict[str, object], status: int = 200) -> object:
    def _post(url: str, *, json: object, headers: object, timeout: object) -> httpx.Response:
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))

    return _post


def test_web_search_maps_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    payload: dict[str, object] = {
        "results": [
            {
                "title": "RAG advances",
                "url": "https://example.com/a",
                "content": "snippet one",
                "published_date": "2026-06-10",
            },
            # second result omits published_date -> mapped to None
            {"title": "More RAG", "url": "https://example.com/b", "content": "snippet two"},
        ]
    }
    monkeypatch.setattr(httpx, "post", _fake_post(payload))

    result = web_search("rag in 2026", max_results=2)  # client=None -> real TavilyClient

    assert result.ok is True
    assert result.tool == TOOL_WEB_SEARCH
    assert result.data == [
        {
            "title": "RAG advances",
            "url": "https://example.com/a",
            "snippet": "snippet one",
            "published": "2026-06-10",
        },
        {
            "title": "More RAG",
            "url": "https://example.com/b",
            "snippet": "snippet two",
            "published": None,
        },
    ]


def test_web_search_http_error_returns_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "post", _fake_post({"error": "boom"}, status=500))
    result = web_search("rag in 2026")
    assert result.ok is False
    assert result.tool == TOOL_WEB_SEARCH
    assert result.error_code == ERROR_HTTP
    assert result.data is None


def test_web_search_empty_query_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        web_search("")


@pytest.mark.vcr
def test_web_search_replays_recorded_cassette() -> None:
    # Real key when recording; dummy on replay (the cassette supplies the
    # response and the recorded auth header is scrubbed).
    key = os.environ.get("TAVILY_API_KEY", "dummy-key")
    client = TavilyClient(api_key=key)

    result = web_search(_TAVILY_QUERY, max_results=3, client=client)

    assert result.ok is True
    assert result.tool == TOOL_WEB_SEARCH
    assert result.data is not None
    assert len(result.data) >= 1
    first = result.data[0]
    assert set(first) == {"title", "url", "snippet", "published"}
    assert isinstance(first["url"], str)
