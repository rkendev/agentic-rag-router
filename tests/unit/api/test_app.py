"""Unit tests for the ``POST /ask`` app, driven via FastAPI's TestClient.

The app's heavy collaborators are injected as the same honest doubles the loop
tests use (`tests/unit/router/fakes.py`), so no model, substrate, or ingest
group is touched. `create_app(client=..., dispatcher=...)` bypasses the
production lifespan builders entirely; the `TestClient` context manager runs the
lifespan so the fakes land on ``app.state``.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from agentic_rag_router.api.app import create_app
from agentic_rag_router.router.grading import GRADE_SUFFICIENT
from agentic_rag_router.router.loop import REFUSAL_SENTINEL
from agentic_rag_router.router.schema import TOOLS
from tests.unit.router.fakes import (
    FakeAnthropicClient,
    FakeDispatcher,
    make_text_response,
    make_tool_use_response,
    success_outcome,
)

_VS = "vector_search"


def _client(responses: list[Any], dispatcher: FakeDispatcher) -> TestClient:
    """A TestClient over an app wired with injected fakes."""
    app = create_app(
        client=FakeAnthropicClient(responses),
        dispatcher=dispatcher,
        tools=TOOLS,
    )
    return TestClient(app)


def test_ask_success_returns_answer_and_citations() -> None:
    responses = [
        make_tool_use_response([("tu_1", _VS, {"query": "attention"})]),
        make_text_response("Attention weights tokens by relevance."),
    ]
    dispatcher = FakeDispatcher(
        default=success_outcome(
            grade=GRADE_SUFFICIENT, citations=[{"tool": _VS, "source": "2401.1"}]
        )
    )

    with _client(responses, dispatcher) as client:
        response = client.post("/ask", json={"question": "what is attention?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Attention weights tokens by relevance."
    assert body["citations"] == [{"tool": _VS, "source": "2401.1"}]
    assert body["refusal_reason"] is None
    assert body["iterations"] == 2
    # Trajectory carries the per-step grade.
    assert body["trajectory"][0]["tool"] == _VS
    assert body["trajectory"][0]["grade"] == GRADE_SUFFICIENT


def test_ask_refusal_carries_zero_citations() -> None:
    responses = [
        make_tool_use_response([("tu_1", _VS, {"query": "hyperparameters"})]),
        make_text_response("REFUSE: the abstracts do not contain that detail."),
    ]
    # Even though the tool returned sufficient evidence, the model refused.
    dispatcher = FakeDispatcher(
        default=success_outcome(grade=GRADE_SUFFICIENT, citations=[{"tool": _VS, "source": "x"}])
    )

    with _client(responses, dispatcher) as client:
        response = client.post("/ask", json={"question": "exact learning rate?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["refusal_reason"] == REFUSAL_SENTINEL


def test_ask_empty_question_is_422() -> None:
    dispatcher = FakeDispatcher(default=success_outcome())
    with _client([make_text_response()], dispatcher) as client:
        empty = client.post("/ask", json={"question": ""})
        whitespace = client.post("/ask", json={"question": "   "})
        missing = client.post("/ask", json={})

    assert empty.status_code == 422
    assert whitespace.status_code == 422
    assert missing.status_code == 422


def test_ask_response_envelope_shape() -> None:
    responses = [
        make_tool_use_response([("tu_1", _VS, {"query": "x"})]),
        make_text_response("Grounded."),
    ]
    dispatcher = FakeDispatcher(default=success_outcome(grade=GRADE_SUFFICIENT))

    with _client(responses, dispatcher) as client:
        body = client.post("/ask", json={"question": "q"}).json()

    assert set(body) == {"answer", "citations", "trajectory", "refusal_reason", "iterations"}
    step = body["trajectory"][0]
    assert set(step) == {"tool", "input", "latency_ms", "ok", "error_code", "grade"}
