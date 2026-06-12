"""Unit tests for `run_router` --- the routing loop mechanics.

Every test drives the loop with a `FakeAnthropicClient` (scripted responses,
records the per-turn ``tool_choice``) and a `FakeDispatcher` (preset outcomes,
records dispatched calls). No network, no substrates --- this isolates the loop's
control flow: force-then-relax, parallel tool blocks, the iteration cap, and
how failed/unknown tool outcomes flow back to the model.
"""

from __future__ import annotations

from types import SimpleNamespace

from agentic_rag_router.router.loop import (
    MAX_ITERATIONS,
    REFUSAL_ITERATION_BUDGET,
    TrajectoryStep,
    run_router,
)
from agentic_rag_router.router.schema import TOOLS
from tests.unit.router.fakes import (
    FakeAnthropicClient,
    FakeDispatcher,
    error_outcome,
    make_empty_response,
    make_text_response,
    make_tool_use_response,
    success_outcome,
)

_VS = "vector_search"
_SQL = "sql_query"


def test_single_tool_path_returns_answer_and_citations() -> None:
    client = FakeAnthropicClient(
        [
            make_tool_use_response([("tu_1", _VS, {"query": "what is attention"})]),
            make_text_response("Attention weights tokens by relevance."),
        ]
    )
    dispatcher = FakeDispatcher(
        default=success_outcome(citations=[{"tool": _VS, "source": "2401.00001"}])
    )

    result = run_router("what is attention?", client=client, tools=TOOLS, dispatcher=dispatcher)

    assert result.answer == "Attention weights tokens by relevance."
    assert result.refusal_reason is None
    assert result.iterations == 2
    assert result.citations == [{"tool": _VS, "source": "2401.00001"}]
    assert result.trajectory == [
        TrajectoryStep(
            tool=_VS,
            input={"query": "what is attention"},
            latency_ms=7,
            ok=True,
            error_code=None,
        )
    ]
    assert dispatcher.calls == [(_VS, {"query": "what is attention"})]


def test_iteration_zero_forces_any_then_relaxes_to_auto() -> None:
    client = FakeAnthropicClient(
        [
            make_tool_use_response([("tu_1", _VS, {"query": "x"})]),
            make_text_response(),
        ]
    )
    run_router("q", client=client, tools=TOOLS, dispatcher=FakeDispatcher())

    assert client.tool_choices == [{"type": "any"}, {"type": "auto"}]


def test_parallel_tool_blocks_are_all_answered() -> None:
    client = FakeAnthropicClient(
        [
            make_tool_use_response(
                [
                    ("tu_1", _VS, {"query": "concept"}),
                    ("tu_2", _SQL, {"sql": "SELECT count(*) FROM taxi_trips"}),
                ]
            ),
            make_text_response("Combined answer."),
        ]
    )
    dispatcher = FakeDispatcher(
        outcomes={
            _VS: success_outcome(citations=[{"tool": _VS, "source": "a"}]),
            _SQL: success_outcome(citations=[{"tool": _SQL, "source": "taxi_trips"}]),
        }
    )

    result = run_router("hybrid?", client=client, tools=TOOLS, dispatcher=dispatcher)

    # Both blocks dispatched, both recorded in the trajectory and citations.
    assert [name for name, _ in dispatcher.calls] == [_VS, _SQL]
    assert [step.tool for step in result.trajectory] == [_VS, _SQL]
    assert result.citations == [
        {"tool": _VS, "source": "a"},
        {"tool": _SQL, "source": "taxi_trips"},
    ]

    # Every tool_use block received a matching tool_result in the next user turn.
    second_turn_messages = client.messages_seen[1]
    tool_result_turn = second_turn_messages[-1]
    assert tool_result_turn["role"] == "user"
    returned_ids = [block["tool_use_id"] for block in tool_result_turn["content"]]
    assert returned_ids == ["tu_1", "tu_2"]


def test_iteration_cap_returns_refusal_with_zero_citations() -> None:
    # The client repeats its last response, so every turn is a tool_use and the
    # model never reaches end_turn --- the loop must hit the cap and refuse.
    client = FakeAnthropicClient([make_tool_use_response([("tu", _VS, {"query": "loop"})])])
    dispatcher = FakeDispatcher(default=success_outcome(citations=[{"tool": _VS, "source": "z"}]))

    result = run_router("never ends", client=client, tools=TOOLS, dispatcher=dispatcher)

    assert result.answer is None
    assert result.refusal_reason == REFUSAL_ITERATION_BUDGET
    assert result.citations == []
    assert result.iterations == MAX_ITERATIONS
    assert len(result.trajectory) == MAX_ITERATIONS
    # Forced once, then relaxed for the rest.
    assert client.tool_choices == [{"type": "any"}] + [{"type": "auto"}] * (MAX_ITERATIONS - 1)


def test_failed_tool_outcome_continues_and_carries_is_error() -> None:
    client = FakeAnthropicClient(
        [
            make_tool_use_response([("tu_1", _SQL, {"sql": "SELECT bogus"})]),
            make_text_response("Recovered without that tool."),
        ]
    )
    dispatcher = FakeDispatcher(default=error_outcome(error_code="backend_error"))

    result = run_router("q", client=client, tools=TOOLS, dispatcher=dispatcher)

    # Loop continued past the failure to the final answer.
    assert result.answer == "Recovered without that tool."
    assert result.refusal_reason is None
    # The failure is recorded but contributes no citation.
    assert result.trajectory[0].ok is False
    assert result.trajectory[0].error_code == "backend_error"
    assert result.citations == []
    # The tool_result fed back to the model is flagged as an error.
    tool_result = client.messages_seen[1][-1]["content"][0]
    assert tool_result["is_error"] is True


def test_unknown_tool_outcome_is_flagged_and_loop_continues() -> None:
    # The loop is tool-agnostic: an unknown name is the dispatcher's outcome to
    # decide. Here the dispatcher reports it as an error; the loop keeps going.
    client = FakeAnthropicClient(
        [
            make_tool_use_response([("tu_1", "frobnicate", {"x": 1})]),
            make_text_response("Done."),
        ]
    )
    dispatcher = FakeDispatcher(default=error_outcome(error_code="unknown_tool"))

    result = run_router("q", client=client, tools=TOOLS, dispatcher=dispatcher)

    assert result.answer == "Done."
    assert result.trajectory[0].tool == "frobnicate"
    assert result.trajectory[0].error_code == "unknown_tool"
    assert dispatcher.calls == [("frobnicate", {"x": 1})]


def test_terminal_without_tool_blocks_returns_no_answer() -> None:
    # A non-end_turn response with neither tool_use nor text (e.g. truncated):
    # the loop stops, with no answer and no refusal reason.
    client = FakeAnthropicClient([make_empty_response()])

    result = run_router("q", client=client, tools=TOOLS, dispatcher=FakeDispatcher())

    assert result.answer is None
    assert result.refusal_reason is None
    assert result.iterations == 1
    assert result.trajectory == []


def test_answer_extraction_skips_non_text_and_empty_blocks() -> None:
    # The final turn carries a non-text block and a whitespace-only text block
    # before the real answer; _extract_text must skip both and return the first
    # non-empty text.
    mixed = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text="   "),
            SimpleNamespace(type="text", text="The real answer."),
        ],
    )
    client = FakeAnthropicClient([mixed])

    result = run_router("q", client=client, tools=TOOLS, dispatcher=FakeDispatcher())

    assert result.answer == "The real answer."


def test_answer_extraction_returns_none_when_no_usable_text() -> None:
    # An end_turn whose only text block is empty yields no answer (loop runs to
    # the end of _extract_text without returning).
    empty_text = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="")],
    )
    client = FakeAnthropicClient([empty_text])

    result = run_router("q", client=client, tools=TOOLS, dispatcher=FakeDispatcher())

    assert result.answer is None
    assert result.refusal_reason is None


def test_model_answers_without_calling_a_tool() -> None:
    # If the model returns end_turn on the very first turn, the answer is taken
    # and no tools are dispatched.
    client = FakeAnthropicClient([make_text_response("Direct answer.")])
    dispatcher = FakeDispatcher()

    result = run_router("q", client=client, tools=TOOLS, dispatcher=dispatcher)

    assert result.answer == "Direct answer."
    assert result.iterations == 1
    assert result.trajectory == []
    assert dispatcher.calls == []
