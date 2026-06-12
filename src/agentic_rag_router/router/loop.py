"""The routing loop --- `run_router` drives Claude across the three tools.

`run_router` is a hand-written agentic loop (no framework). It hands the tool
schema (`schema.py`) to the model through an `AnthropicClientPort`, executes
whichever tools the model calls via a `Dispatcher` (`dispatch.py`), and feeds
the results back until the model answers or the iteration budget is spent.

Three empirical quirks from prior project history are wired in deliberately:

1. ``tool_choice={"type":"any"}`` (or a specific tool) RE-FORCES a tool call on
   *every* turn, so the loop would never reach ``end_turn``. We force only on
   iteration 0 to guarantee a routing decision, then relax to
   ``{"type":"auto"}`` so the model can stop.
2. A single response can carry MULTIPLE parallel ``tool_use`` blocks; every one
   must receive a matching ``tool_result`` block in the next user turn. We
   iterate over *all* blocks (the historical bug answered only the first).
3. A hard cap of 5 iterations: on exhaustion we return the envelope with
   ``refusal_reason="iteration_budget_exhausted"`` and zero citations.

Prompt caching is intentionally skipped (the 3-tool system prompt is well under
the caching floor and Sonnet caching is not worth the complexity here).

The return shape is `RouterResponse`. There is no evidence-grade field --- that
is D5; adding a stub now would freeze a shape before its contract exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag_router.router.dispatch import DispatchOutcome

# Tool-choice payloads. Constants so the loop and its tests reference one
# spelling, and so the "force on iter 0, then relax" rule reads at a glance.
TOOL_CHOICE_ANY: dict[str, str] = {"type": "any"}
TOOL_CHOICE_AUTO: dict[str, str] = {"type": "auto"}

# Hard iteration cap (quirk 3) and the refusal reason returned on exhaustion.
MAX_ITERATIONS = 5
REFUSAL_ITERATION_BUDGET = "iteration_budget_exhausted"

# Anthropic stop_reason / content-block type spellings the loop branches on.
_STOP_END_TURN = "end_turn"
_BLOCK_TOOL_USE = "tool_use"
_BLOCK_TEXT = "text"


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One tool invocation the router made, recorded for observability.

    Mirrors the per-call facts the `Dispatcher` surfaces: which ``tool`` ran,
    the ``input`` the model supplied, the ``latency_ms`` the adapter measured,
    whether it was ``ok``, and the machine-readable ``error_code`` on failure
    (``None`` on success).
    """

    tool: str
    input: dict[str, object]
    latency_ms: int
    ok: bool
    error_code: str | None


@dataclass(frozen=True, slots=True)
class RouterResponse:
    """The router's outcome for one question.

    Parameters
    ----------
    answer:
        The model's final text answer, or ``None`` when the loop ended without
        one (e.g. iteration budget exhausted, or a non-text final turn).
    citations:
        Source identifiers for the successful tool evidence behind the answer
        (simple version: every successful tool result's source ids; D5
        refines this to "evidence actually referenced"). Empty on refusal.
    trajectory:
        Every tool invocation made, in order.
    refusal_reason:
        ``None`` on a normal answer; a machine-readable reason when the router
        declined --- currently only ``"iteration_budget_exhausted"`` (D5 adds
        evidence-based refusals).
    iterations:
        How many model turns the loop took (1..``MAX_ITERATIONS``).
    """

    answer: str | None
    citations: list[dict[str, object]]
    trajectory: list[TrajectoryStep]
    refusal_reason: str | None
    iterations: int


class DispatcherPort(Protocol):
    """Maps a tool-use name + input to an executed `DispatchOutcome`.

    `dispatch.Dispatcher` satisfies this structurally. The port abstracts the
    dispatcher *behaviour*; the concrete `DispatchOutcome` it returns is just a
    data holder, so the loop depends on that shape directly rather than
    re-declaring it.
    """

    def dispatch(self, name: str, tool_input: dict[str, Any]) -> DispatchOutcome:
        """Execute the named tool with ``tool_input`` and return the outcome."""
        ...


class AnthropicClientPort(Protocol):
    """The one model call the loop needs --- the seam unit tests fake.

    Implementations own the model id, the system prompt, and ``max_tokens``;
    the loop supplies only the per-turn ``messages``, the ``tools`` schema, and
    the ``tool_choice`` for this turn, and gets back a Messages-API response
    object (anything exposing ``stop_reason`` and ``content``).
    """

    def create_message(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, str],
    ) -> Any:
        """Call the Messages API for one turn and return the raw response."""
        ...


def _extract_text(response: Any) -> str | None:
    """Return the first non-empty text block of ``response``, or ``None``.

    Uses ``getattr`` so a fabricated test response (a ``SimpleNamespace``) and a
    real SDK ``Message`` are handled identically --- mirrors the ``getattr``
    style already used in `infrastructure/anthropic_adapter.py`.
    """
    for block in response.content:
        if getattr(block, "type", None) == _BLOCK_TEXT:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text.strip():
                return text
    return None


def run_router(
    question: str,
    *,
    client: AnthropicClientPort,
    tools: list[dict[str, Any]],
    dispatcher: DispatcherPort,
) -> RouterResponse:
    """Route ``question`` across the tools and return a `RouterResponse`.

    Iteration 0 forces a tool call (``tool_choice`` any) so the model commits to
    a route; subsequent iterations relax to ``auto`` so it can stop (quirk 1).
    Every ``tool_use`` block in a response is dispatched and answered with a
    matching ``tool_result`` (quirk 2). If the model has not answered within
    ``MAX_ITERATIONS`` turns, the loop returns a refusal envelope with zero
    citations (quirk 3).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    trajectory: list[TrajectoryStep] = []
    citations: list[dict[str, object]] = []

    for index in range(MAX_ITERATIONS):
        tool_choice = TOOL_CHOICE_ANY if index == 0 else TOOL_CHOICE_AUTO
        response = client.create_message(messages=messages, tools=tools, tool_choice=tool_choice)

        tool_use_blocks = [
            b for b in response.content if getattr(b, "type", None) == _BLOCK_TOOL_USE
        ]

        # Terminal: the model answered (end_turn) or produced no tool call to
        # act on. Either way the loop is done --- return what we have.
        if response.stop_reason == _STOP_END_TURN or not tool_use_blocks:
            return RouterResponse(
                answer=_extract_text(response),
                citations=citations,
                trajectory=trajectory,
                refusal_reason=None,
                iterations=index + 1,
            )

        # Record the assistant turn verbatim (the SDK accepts its own content
        # blocks echoed back), then answer EVERY tool_use block (quirk 2).
        messages.append({"role": "assistant", "content": response.content})
        tool_results: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            tool_input = dict(block.input)
            outcome = dispatcher.dispatch(block.name, tool_input)
            trajectory.append(
                TrajectoryStep(
                    tool=block.name,
                    input=tool_input,
                    latency_ms=outcome.latency_ms,
                    ok=outcome.ok,
                    error_code=outcome.error_code,
                )
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": outcome.content,
                    "is_error": outcome.is_error,
                }
            )
            citations.extend(outcome.citations)
        messages.append({"role": "user", "content": tool_results})

    # Budget exhausted (quirk 3): refuse with zero citations.
    return RouterResponse(
        answer=None,
        citations=[],
        trajectory=trajectory,
        refusal_reason=REFUSAL_ITERATION_BUDGET,
        iterations=MAX_ITERATIONS,
    )
