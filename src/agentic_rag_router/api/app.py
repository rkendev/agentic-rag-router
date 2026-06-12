"""The ``POST /ask`` FastAPI application (D5).

A thin HTTP envelope around `router.loop.run_router`. The heavy pieces --- the
Anthropic client and the tool `Dispatcher` (embedder + pgvector repository +
``router_ro`` executor + Tavily client) --- are expensive to build, so they are
constructed ONCE in the app lifespan and stashed on ``app.state``; the endpoint
reuses them for every request.

Two deliberate choices:

- **Sync endpoint.** ``ask`` is a plain ``def`` (not ``async def``). The router
  loop makes blocking model + substrate calls, so FastAPI runs the handler in
  its threadpool --- the house convention for blocking inference. An ``async``
  handler would block the event loop.
- **Importable without the ingest group.** `create_app` and module import do not
  touch ``psycopg`` / ``sentence-transformers``: the substrate adapters
  lazy-import those only when a real query runs, and the real-dependency builders
  (`_build_client` / `_build_dispatcher`) run only in the production lifespan.
  Tests inject fakes via `create_app(client=..., dispatcher=...)`, so the build
  helpers are excluded from coverage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from agentic_rag_router.router.dispatch import Dispatcher
from agentic_rag_router.router.loop import AnthropicClientPort, DispatcherPort, run_router
from agentic_rag_router.router.schema import TOOLS


class AskRequest(BaseModel):
    """The request body for ``POST /ask`` --- a single non-empty question."""

    question: str = Field(description="The natural-language question to route.")

    @field_validator("question")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        """Reject empty / whitespace-only questions (-> 422) and trim."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace-only")
        return stripped


class TrajectoryStepOut(BaseModel):
    """One tool invocation in the response trajectory, including its grade."""

    tool: str
    input: dict[str, Any]
    latency_ms: int
    ok: bool
    error_code: str | None
    grade: str


class AskResponse(BaseModel):
    """The full router envelope returned to the caller.

    Mirrors `router.loop.RouterResponse`: the answer (``None`` on refusal), the
    ``sufficient``-only citations (empty on refusal), the per-step trajectory
    with grades, the machine-readable ``refusal_reason`` (``None`` on a normal
    answer), and the model-turn count.
    """

    answer: str | None
    citations: list[dict[str, Any]]
    trajectory: list[TrajectoryStepOut]
    refusal_reason: str | None
    iterations: int


def _build_client() -> AnthropicClientPort:  # pragma: no cover - real credentials/network
    """Build the live Anthropic router client from `Settings` (production only)."""
    from agentic_rag_router.infrastructure.settings import Settings
    from agentic_rag_router.router.client import AnthropicRouterClient

    return AnthropicRouterClient.from_settings(Settings())


def _build_dispatcher() -> DispatcherPort:  # pragma: no cover - real substrates
    """Build the `Dispatcher` over the real substrates (production only)."""
    from agentic_rag_router.tools.sql_query import RouterRoExecutor
    from agentic_rag_router.tools.vector_search import (
        PgVectorRepository,
        SentenceTransformerEmbedder,
    )

    return Dispatcher(
        embedder=SentenceTransformerEmbedder(),
        repository=PgVectorRepository(),
        executor=RouterRoExecutor(),
    )


def create_app(
    *,
    client: AnthropicClientPort | None = None,
    dispatcher: DispatcherPort | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    The three router collaborators can be injected (tests pass fakes); when
    omitted they are built from the environment in the lifespan, so a missing
    API key or unreachable substrate fails at startup, not at import.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.client = client if client is not None else _build_client()
        app.state.dispatcher = dispatcher if dispatcher is not None else _build_dispatcher()
        app.state.tools = tools if tools is not None else TOOLS
        yield

    app = FastAPI(
        title="agentic-rag-router",
        summary="Route a question across vector/SQL/web; grade evidence; refuse when unsupported.",
        lifespan=lifespan,
    )

    @app.post("/ask")
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        """Route ``payload.question`` and return the full router envelope.

        Sync ``def`` on purpose --- FastAPI runs it in a threadpool so the
        blocking model + substrate calls do not stall the event loop.
        """
        result = run_router(
            payload.question,
            client=request.app.state.client,
            tools=request.app.state.tools,
            dispatcher=request.app.state.dispatcher,
        )
        return AskResponse.model_validate(asdict(result))

    return app


# Module-level instance for ``uvicorn agentic_rag_router.api.app:app``.
app = create_app()
