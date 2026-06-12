"""The routing layer (D4) --- tool schema, the loop, and the dispatcher.

This package turns the three T003 tool adapters into a router: `schema`
provides the Anthropic tool definitions whose descriptions ARE the routing
policy, `loop.run_router` drives Claude across them, `dispatch.Dispatcher`
executes the chosen tools, and `client.AnthropicRouterClient` is the concrete
Claude Sonnet call behind the loop's port.
"""

from __future__ import annotations

from agentic_rag_router.router.client import AnthropicRouterClient
from agentic_rag_router.router.dispatch import Dispatcher, DispatchOutcome
from agentic_rag_router.router.loop import (
    AnthropicClientPort,
    RouterResponse,
    TrajectoryStep,
    run_router,
)
from agentic_rag_router.router.schema import SYSTEM_PROMPT, TOOLS, build_tools

__all__ = [
    "SYSTEM_PROMPT",
    "TOOLS",
    "AnthropicClientPort",
    "AnthropicRouterClient",
    "DispatchOutcome",
    "Dispatcher",
    "RouterResponse",
    "TrajectoryStep",
    "build_tools",
    "run_router",
]
