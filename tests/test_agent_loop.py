"""The tool loop (AGENT_MODE=loop), with the model and the search scripted."""

import pytest
from google.genai import types

import app.agent.loop as loop
from app.agent.graph import run_agent
from app.agent.history import Turn
from app.core.config import get_settings
from app.core.gemini_client import ModelOutputError
from tests.fakes import chunk


def call(name: str, signature: bytes | None = None, **args) -> types.GenerateContentResponse:
    part = types.Part(function_call=types.FunctionCall(name=name, args=args, id=f"{name}-1"), thought_signature=signature)
    return reply(part)


def text(value: str) -> types.GenerateContentResponse:
    return reply(types.Part(text=value))


def reply(*parts: types.Part) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=100, candidates_token_count=20),
    )


@pytest.fixture
def model(monkeypatch):
    """Scripted replies, in order; records what each call was given."""
    script: list[types.GenerateContentResponse] = []
    seen: list[dict] = []

    async def fake(**kwargs):
        seen.append(kwargs)
        return script.pop(0)

    monkeypatch.setattr(loop, "generate_contents", fake)
    monkeypatch.setattr(get_settings(), "agent_mode", "loop")
    return script, seen


@pytest.fixture
def searched(monkeypatch):
    queries: list[str] = []

    async def fake_retrieve(query):
        queries.append(query)
        if "rate" in query:
            return [chunk("api-rate-limits", score=0.7)]
        return [chunk("billing-refunds", score=0.63)]

    monkeypatch.setattr(loop, "retrieve", fake_retrieve)
    return queries


async def test_it_searches_answers_and_cites_only_what_it_saw(model, searched):
    script, seen = model
    script += [
        call("search_support_docs", query="annual plan refund window"),
        call("give_answer", answer="Annual plans are refundable within 14 days.", sources=["billing-refunds", "made-up-doc"]),
    ]
    result = await run_agent("Can I get a refund on my annual plan?")

    assert searched == ["annual plan refund window"]
    assert result["answer"] == "Annual plans are refundable within 14 days."
    assert result["sources"] == ["billing-refunds"]  # an id it never saw is dropped
    assert result["retrieved_sources"] == ["billing-refunds"] and result["search_query"] == "annual plan refund window"
    assert "chunks" in result and result["router_prompt_version"] == result["answer_prompt_version"] == "agent_v1"
    assert result["tool_calls"] == ["search_support_docs", "give_answer"] and result["steps"] == 2
    # The second call carries the search results back to the model.
    answered = seen[1]["contents"][-1].parts[0].function_response
    assert answered.name == "search_support_docs" and answered.id == "search_support_docs-1"
    assert answered.response["results"][0]["article_id"] == "billing-refunds"


async def test_small_talk_is_answered_without_searching(model, searched):
    script, seen = model
    script.append(call("give_answer", answer="Hi! How can I help?", sources=[]))
    result = await run_agent("hello")
    assert searched == [] and len(seen) == 1
    assert result["answer"] == "Hi! How can I help?" and result["sources"] == []
    assert "chunks" not in result  # /chat logs used_tool: false


async def test_it_can_search_twice_and_read_a_whole_article(model, searched):
    script, seen = model
    script += [
        call("search_support_docs", query="payment fails downgrade"),
        call("search_support_docs", query="free tier API rate limit"),
        call("read_article", article_id="billing-refunds"),
        call("give_answer", answer="After three failed retries you drop to the free tier: 100 requests a minute.",
             sources=["billing-refunds", "api-rate-limits"]),
    ]
    result = await run_agent("If my payment fails and I'm downgraded, what happens to my rate limit?")

    assert searched == ["payment fails downgrade", "free tier API rate limit"]
    assert result["articles_read"] == ["billing-refunds"] and result["search_queries"] == searched
    assert result["sources"] == ["billing-refunds", "api-rate-limits"]
    # The article came back whole, including the failed-payment paragraph a chunk had missed.
    article = seen[3]["contents"][-1].parts[0].function_response.response
    assert "we retry the payment three times" in article["text"]
    assert result["chunks"][-1].text.startswith("# Billing and Refunds")


async def test_an_unknown_article_is_an_error_the_model_sees(model, searched):
    script, seen = model
    script += [call("read_article", article_id="nope"), call("give_answer", answer="I don't have enough information.", sources=[])]
    result = await run_agent("Tell me about nope")
    assert seen[1]["contents"][-1].parts[0].function_response.response == {"error": "No article has the id 'nope'."}
    assert result["sources"] == []


async def test_after_the_step_limit_it_may_only_answer(model, searched, monkeypatch):
    monkeypatch.setattr(get_settings(), "agent_max_steps", 2)
    script, seen = model
    script += [call("search_support_docs", query="a"), call("search_support_docs", query="b"),
               call("give_answer", answer="Here's what I found.", sources=[])]
    await run_agent("something hard")
    # Every turn must call a tool; the last may only call give_answer.
    assert [s["tool_config"].function_calling_config.mode for s in seen] == ["ANY", "ANY", "ANY"]
    assert [s["tool_config"].function_calling_config.allowed_function_names for s in seen] == [None, None, ["give_answer"]]


async def test_the_models_own_turns_go_back_to_it_with_their_thought_signatures(model, searched):
    script, seen = model
    script += [call("search_support_docs", signature=b"sig-123", query="refunds"),
               call("give_answer", answer="Refunds take 5-10 business days.", sources=["billing-refunds"])]
    await run_agent("How long do refunds take?")
    replayed = seen[1]["contents"][-2].parts[0]
    assert replayed.function_call.name == "search_support_docs" and replayed.thought_signature == b"sig-123"


async def test_earlier_turns_come_first_and_a_plain_text_reply_is_taken_as_the_answer(model, searched):
    script, seen = model
    script.append(text("You're welcome!"))
    history = [Turn("user", "How do I reset my password?"), Turn("model", "Use Forgot password? on the login page.")]
    result = await run_agent("thanks", history=history)
    assert [c.role for c in seen[0]["contents"]] == ["user", "model", "user"]
    assert result["answer"] == "You're welcome!" and result["sources"] == []


async def test_a_turn_with_nothing_usable_is_an_error(model, searched):
    script, _ = model
    script.append(types.GenerateContentResponse(candidates=[]))
    with pytest.raises(ModelOutputError):
        await run_agent("hello")
