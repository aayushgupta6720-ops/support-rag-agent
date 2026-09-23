import pytest

import app.agent.graph as graph
from app.agent.prompts import (
    DIRECT_ANSWER_PROMPT_VERSION,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    GROUNDED_ANSWER_PROMPT_VERSION,
    GROUNDED_ANSWER_SYSTEM_PROMPT,
    ROUTER_PROMPT_VERSION,
)
from app.agent.tools import SEARCH_DOCS_TOOL_NAME
from app.core.config import get_settings
from app.core.observability import start_trace
from app.core.pricing import generation_cost_usd
from tests.fakes import FakeGenerate, chunk, model_response


def _answer(text: str):
    return model_response(parsed=graph.AgentAnswer(answer=text))


@pytest.fixture
def retrieve_calls(monkeypatch):
    """Patch retrieval; tests set `.chunks` to control what comes back."""

    class FakeRetrieve:
        chunks = [chunk("password-reset", "Reset links last 30 minutes."), chunk("two-factor-auth")]
        queries: list[str] = []

        async def __call__(self, query):
            self.queries.append(query)
            return self.chunks

    fake = FakeRetrieve()
    monkeypatch.setattr(graph, "retrieve", fake)
    return fake


async def test_tool_call_routes_through_retrieval(monkeypatch, retrieve_calls):
    fake = FakeGenerate([
        model_response(function_call=(SEARCH_DOCS_TOOL_NAME, {"query": "reset link expiry"})),
        _answer("30 minutes."),
    ])
    monkeypatch.setattr(graph, "generate", fake)

    result = await graph.run_agent("how long is the reset link good for?")

    # the router's rewritten query is what gets searched, not the raw user text
    assert retrieve_calls.queries == ["reset link expiry"]
    assert result["answer"] == "30 minutes."
    assert result["sources"] == ["password-reset", "two-factor-auth"]
    assert result["router_prompt_version"] == ROUTER_PROMPT_VERSION
    assert result["answer_prompt_version"] == GROUNDED_ANSWER_PROMPT_VERSION

    generation = fake.calls[1]
    assert generation["system_instruction"] == GROUNDED_ANSWER_SYSTEM_PROMPT
    assert "Reset links last 30 minutes." in generation["prompt"]
    assert "how long is the reset link good for?" in generation["prompt"]
    assert generation["response_schema"] is graph.AgentAnswer


async def test_tool_call_without_query_arg_falls_back_to_user_query(monkeypatch, retrieve_calls):
    monkeypatch.setattr(graph, "generate", FakeGenerate([
        model_response(function_call=(SEARCH_DOCS_TOOL_NAME, {})),
        _answer("..."),
    ]))

    await graph.run_agent("refund policy?")

    assert retrieve_calls.queries == ["refund policy?"]


async def test_no_tool_call_answers_directly_without_retrieval(monkeypatch, retrieve_calls):
    fake = FakeGenerate([model_response(), _answer("Hi! How can I help?")])
    monkeypatch.setattr(graph, "generate", fake)

    result = await graph.run_agent("hello")

    assert retrieve_calls.queries == []
    assert "chunks" not in result
    assert result["sources"] == []
    assert result["answer_prompt_version"] == DIRECT_ANSWER_PROMPT_VERSION
    assert fake.calls[1]["system_instruction"] == DIRECT_ANSWER_SYSTEM_PROMPT
    assert fake.calls[1]["prompt"] == "hello"


async def test_empty_retrieval_tells_model_nothing_was_found(monkeypatch, retrieve_calls):
    retrieve_calls.chunks = []
    fake = FakeGenerate([
        model_response(function_call=(SEARCH_DOCS_TOOL_NAME, {"query": "dark mode"})),
        _answer("I don't have enough information."),
    ])
    monkeypatch.setattr(graph, "generate", fake)

    result = await graph.run_agent("do you have dark mode?")

    # still the grounded prompt (so the model is told to not guess), but with
    # an explicit "nothing found" instead of an empty context block
    assert fake.calls[1]["system_instruction"] == GROUNDED_ANSWER_SYSTEM_PROMPT
    assert fake.calls[1]["prompt"].startswith("No relevant support docs were found.")
    assert result["sources"] == []


async def test_trace_records_tokens_and_cost_per_llm_step(monkeypatch, retrieve_calls):
    monkeypatch.setattr(graph, "generate", FakeGenerate([
        model_response(input_tokens=100, output_tokens=20),
        model_response(parsed=graph.AgentAnswer(answer="hi"), input_tokens=50, output_tokens=10),
    ]))
    trace = start_trace()

    await graph.run_agent("hi")

    assert [s.name for s in trace.steps] == ["route", "generate"]
    route = trace.steps[0]
    assert (route.input_tokens, route.output_tokens) == (100, 20)
    assert route.cost_usd == generation_cost_usd(get_settings().generation_model, 100, 20)
    assert trace.total_tokens == 180
